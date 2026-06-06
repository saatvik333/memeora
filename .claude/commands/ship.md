---
description: Finish a build-order step the memeora way — gate, docs, memory, commit
---

Finish the current unit of work using the project's established ritual. Do these
in order; do not skip:

1. **Gate locally.** Run `scripts/check.sh` (mirrors CI: rustfmt, clippy
   `--all-features -D warnings`, test `--all-features`, cargo-deny). Fix every
   failure before continuing — do not commit red.
   - New native/ML dependency? It often drags in licenses the allowlist lacks
     (e.g. `NCSA`, `CDLA-Permissive-2.0`). Add them to `deny.toml` with a comment.
   - `fastembed` code is feature-gated — the `--all-features` pass is the only one
     that compiles it, so trust it over a plain `cargo check`.
2. **Self-review the diff.** Read what changed; look for the bug you'd catch in
   review (off-by-one, wrong index, silent `unwrap`, missing edge case).
3. **Keep docs current.** Update `README.md` status and any affected
   `docs/ARCHITECTURE.md` rationale. The architecture doc is the source of truth.
4. **Update native memory.** Bump the phase line in the `project-memeora` memory
   (absolute dates, what's done, what's next).
5. **Commit.** Conventional Commits, subject < 200 chars, scope the crate
   (`feat(core): …`). End with the `Co-Authored-By` trailer. The pre-commit hook
   re-runs the gate; use `--no-verify` only if you already ran `scripts/check.sh`.

Then stop and report what shipped + what's next. Don't start the next step
unless asked.
