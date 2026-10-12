# Context and memory

An agent is **stateless by default**: every `run()`, `run_stream()` and `complete()`
starts from an empty history, so unrelated calls never leak into each other and nothing
costs memory you did not ask for. Memory is one setting, in three levels:

```python
from agent_framework import Agent, MemoryConfig

Agent("claude-opus-5")                                   # stateless
Agent("claude-opus-5", memory=MemoryConfig())            # the conversation, in RAM
Agent(                                                   # + long-term memory in SQLite
    "claude-opus-5",
    memory=MemoryConfig(store="support.db", namespace="customer-42"),
    session_id="ticket-4412",
)
```

With `MemoryConfig()` the conversation carries over from run to run in this process, and
is kept within the context window by trimming (below) — no LLM call, no database.

**Long-term memory** (`store=`) uses the standard library's `sqlite3`, so there is nothing
extra to install and no database code to write. A namespace keeps three things:

- *Facts.* `await agent.memory.remember(key, value)` stores one (`forget(key)` removes
  it). Facts that share whole words with a new message are added to that run's history
  once, just before the message. SQLite searches them with its full-text index: every
  fact, in any script, with accents folded — "iletisim" finds "iletişim", "faturalar" finds
  "faturalar", "art" does not find "party" — best match first. Words of one or two letters
  are ignored unless written in capitals ("DB", "CI") or part of a key ("db_host"). A fact
  whose value changed since it was shown is shown again, marked `updated; replaces: …`,
  because the earlier line stays in the history.
- *Sessions.* Every run saves the conversation, and an agent with the same `session_id`
  continues it — in another process, after a restart. It is loaded before the first run
  (`await Agent.connect(...)` loads it at once); a new `session_id` starts fresh. Besides
  the current history, the store keeps a log of every message, including the ones a trim
  later removes. **One live agent per session:** loading takes ownership, and if a second
  agent opens the same session (the same ticket on another server), the first one's next
  save is refused and recorded, and its next `run()` raises `SessionTakenOverError`
  before any model call — build a new agent to continue from what the store holds. Two
  agents never interleave into one conversation, and none silently stops saving.
- *A summary.* Turns a trim removes are summarized (one LLM call per trim), and every new
  conversation in the namespace starts with that summary. A conversation that never
  trimmed is summarized once it has been idle for `idle_after_minutes` (30): a new
  conversation summarizes up to `idle_summaries_per_start` (2) of them before its first
  call, oldest first, and `await agent.memory.summarize_idle(limit)` works through the
  rest — run it from a background job to keep first calls fast. Each idle conversation is
  summarized under a short lease, so two new conversations never cover the same one and
  a process that dies mid-call only lets its lease expire. `await agent.memory.summarize()`
  summarizes the current conversation on demand. The summary is kept to a fifth of
  `max_context_tokens` (at most 8,000 tokens) by merging its oldest parts, since it heads
  every conversation and no trim removes it; `await agent.memory.rolling_summary()`
  returns it as text. Summary calls use `summary_model` if you set one (same provider as
  the agent), and count toward cost tracking and `max_budget_usd`.

`await agent.memory.close()` closes a store the agent opened.
`await agent.clear_history()` forgets the conversation and saves it empty, so the session
does not come back. The store only grows unless you let it forget: `retention_days`
deletes conversations idle for that long whose whole log is already in the summary, and
`await agent.memory.sessions()`, `delete_session(id)` and `prune_sessions(idle_for=...)`
do the same by hand. `namespace`, `summarize`, `summary_model`, `idle_after_minutes`,
`idle_summaries_per_start` and `retention_days` only mean something with a store (or a
`memory_backend`), and are rejected without one rather than silently ignored.

| Where | Setting | Default | Meaning |
|---|---|---|---|
| `AgentConfig` | `max_context_tokens` | min(70 % of the window, 150k, window − response share) | the prompt is trimmed when it reaches this size |
| `AgentConfig` | `max_tool_result_tokens` | min(16k, 10 % of the window, ⅕ of `max_context_tokens`) | a bigger tool result is cut (head + tail) when it arrives |
| `AgentConfig` / `Agent(...)` | `memory` | `None` = stateless | turn memory on with a `MemoryConfig` |
| `MemoryConfig` | `store` | `None` = RAM only | a SQLite file → long-term memory |
| `MemoryConfig` | `namespace` | `"default"` | whose memory: a user, a customer, a project |
| `MemoryConfig` | `summarize` | `True` | write summaries on their own (`summarize()` works either way) |
| `MemoryConfig` | `summary_model` | the agent's model | a cheaper model of the same provider for summaries |
| `MemoryConfig` | `idle_after_minutes` | `30` | a conversation inactive this long may be summarized by others |
| `MemoryConfig` | `idle_summaries_per_start` | `2` | idle conversations a new one summarizes before its first call (`0` = only `summarize_idle()`) |
| `MemoryConfig` | `retention_days` | `None` = keep all | delete fully summarized conversations idle this long |
| `Agent(...)` | `session_id` | a fresh id per agent | which conversation to save and resume |
| `Agent(...)` | `context_filter` | `None` | your function to change what is sent (advanced) |
| `Agent(...)` | `memory_backend` | `None` | your own long-term backend instead of SQLite (advanced) |
| `@tool(...)` | `max_result_tokens` | the agent's default | the cap for this tool |
| `@tool(...)` | `clear_old_results` | `True` | `False` = this tool's results are never cleared |
| `agent.run(...)` | `dry_run` | `False` | show what would be sent, without calling the model |

A `memory_backend` goes with `memory=MemoryConfig(...)` (without a `store`) and stays
yours to close. It implements `LongTermBackend` for facts; to keep sessions and summaries
too, it also implements `SessionStore` — otherwise the agent warns at construction and uses
its facts only.

**How the context is kept in bounds.** Stored messages are never edited: what the model
sees is rendered from them, and a trim only removes whole old turns from the start of the
history. Between trims nothing already sent changes, so the provider's prompt cache keeps
hitting. When the prompt (system + tools + messages) outgrows `max_context_tokens`, one
trim takes it down to 60 % of the limit, with no LLM call:

1. Old tool results become a one-line note. The newest five stay (fewer on small
   windows), as do errors, small results, `ask_user` answers, tools with
   `clear_old_results=False`, and any result the model has not seen yet.
2. If that is not enough, whole old turns go. The conversation's first user message is
   pinned, and the newest three turns always stay.
3. Reasoning from before the current turn is left out, since it was produced against the
   old prefix — except on a model that requires it (Kimi), where steps 1 and 2 do all
   the trimming.

On models that bind reasoning to the conversation prefix — Claude Fable 5.1, Opus 5.5,
Sonnet 5.5, and any Claude model the catalog does not know yet — editing what was already
sent in the middle of a run would get the next request rejected, so a trim happens only
at the start of a run. Inside a long run, Anthropic's server-side tool-result clearing
takes over (requested automatically against the vendor API; off for a custom `base_url`).
A prompt that still cannot fit is stopped before it is sent, as
`StopReason.CONTEXT_EXCEEDED`.

**When something fails.** With memory, a run that fails before the model replies — the
prompt would not fit, the provider rejected it — takes its message back out of the
history, so one oversized message cannot make every later run fail; a run you stop keeps
its message. A store that cannot be read or written costs the run its memory, never the
run itself: the error is recorded in the trace, the run continues, and a session that
could not be loaded is never saved over. A failed summary leaves the session's marker
where it was, so the next summary covers the same turns again from the log. The one
error that is raised rather than recorded is `SessionTakenOverError`, at the start of a
run, once another agent owns the session.

**See what is sent.** `result.context` reports the last call: messages sent and stored,
the token estimate, cleared results, dropped reasoning and the cache-hit ratio — the same
numbers every `llm_call` span carries. `await agent.run(prompt, dry_run=True)` returns the
view the next call would send (with `StopReason.DRY_RUN`), without calling the model,
summarizing, saving, or changing the history.

**Shape it yourself.** `context_filter(messages, info) -> messages` (sync or async) runs
before every call and changes only what is sent. Its output is repaired into a valid
conversation — every tool call keeps its result, the first message is the user's. A rule
such as "remember only the last N exchanges" belongs here. On models that bind reasoning
to the prefix, what was already sent must stay as it was sent: a filter that drops or
keeps the same messages on every call is fine, but one that re-appends a reminder at the
end of every call moves the earlier copy, and in the middle of a run that edit is
rejected and traced.

```python
cid = agent.checkpoint()          # full fidelity: messages, context state, cost
await agent.run("try something risky")
agent.restore(cid)                # exact rollback
```
