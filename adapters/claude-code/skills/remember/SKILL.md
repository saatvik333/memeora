---
name: remember
description: Proactively persist durable facts, preferences, and decisions to memeora so they survive across sessions. Use when the user states a lasting preference, a project decision or constraint is made, or a non-obvious fact is worth keeping.
---

When something worth remembering comes up, call the `remember` tool from the
**memeora** MCP server to store it (omit `scope` to use the current project).

Store, with the right `kind`:
- **preference** — how the user likes to work (tools, style, conventions).
- **fact** — durable, non-obvious facts about the user, project, or environment.
- **episode** — notable decisions or events worth recalling later.

Be selective: store what a future session would genuinely benefit from, not
transient details or anything the user marked private. One memory per call.
