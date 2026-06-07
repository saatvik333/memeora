// memeora — OpenCode plugin (the only non-Rust adapter).
//
// A thin shim: it shells out to the `memeora` CLI (a client over the daemon) so
// no IPC protocol is reimplemented in TypeScript. It adds three custom tools and
// injects the project profile into the compaction prompt so memory survives
// context compaction.
//
// The simplest integration is actually zero code: register `memeora-mcp` as an
// MCP server in opencode.jsonc (see README). This plugin adds the convenience
// tools + compaction injection on top of (or instead of) that.

import { type Plugin, tool } from "@opencode-ai/plugin"
import { execFile } from "node:child_process"
import { promisify } from "node:util"

const run = promisify(execFile)

/** Run the `memeora` CLI and return trimmed stdout (empty string on failure). */
async function memeora(args: string[]): Promise<string> {
  try {
    const { stdout } = await run("memeora", args)
    return stdout.trim()
  } catch {
    // Best-effort: if the daemon/CLI is unavailable, stay silent.
    return ""
  }
}

export const MemeoraPlugin: Plugin = async ({ directory }) => {
  // Resolve the same project scope the hooks/MCP use (daemon-free, hashed by the
  // CLI). Falls back to the raw directory if the CLI is missing.
  const scope = (await memeora(["scope", directory])) || directory

  return {
    tool: {
      memeora_recall: tool({
        description: "Search memeora's persistent memory for this project.",
        args: {
          query: tool.schema.string().describe("What to recall"),
        },
        async execute(args) {
          return (
            (await memeora(["recall", scope, args.query])) || "(no memories found)"
          )
        },
      }),

      memeora_remember: tool({
        description:
          "Store a durable fact, preference, or episode in memeora's memory.",
        args: {
          content: tool.schema.string().describe("The memory to store"),
          kind: tool.schema
            .string()
            .optional()
            .describe("fact | preference | episode (default: fact)"),
        },
        async execute(args) {
          const id = await memeora([
            "add",
            scope,
            args.content,
            "--kind",
            args.kind ?? "fact",
          ])
          return id ? `stored memory ${id}` : "(could not store memory)"
        },
      }),

      memeora_context: tool({
        description:
          "Load the stored profile (stable facts + recent context) for this project.",
        args: {},
        async execute() {
          return (await memeora(["context", scope])) || "(no profile yet)"
        },
      }),
    },

    // Preserve memory across context compaction by injecting the profile.
    "experimental.session.compacting": async (_input, output) => {
      const profile = await memeora(["context", scope])
      if (profile) {
        output.context.push(`## Persistent memory (memeora)\n${profile}`)
      }
    },
  }
}
