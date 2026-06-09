// Proto-parity: the TS SDK re-declares constants the Rust `memeora-proto` owns.
// This test reads the Rust source and fails if they ever drift, so a protocol
// change can't silently ship a mismatched client. Run with `bun test`.

import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import path from "node:path";
import { MAX_MESSAGE_BYTES, PROTOCOL_VERSION } from "../src/index.ts";

// sdk/ts/test → repo root
const ROOT = path.resolve(import.meta.dir, "../../..");

function rust(rel: string): string {
  return readFileSync(path.join(ROOT, rel), "utf8");
}

test("PROTOCOL_VERSION matches the Rust proto", () => {
  const lib = rust("crates/proto/src/lib.rs");
  const m = lib.match(/PROTOCOL_VERSION:\s*u32\s*=\s*(\d+)/);
  expect(m).not.toBeNull();
  expect(Number(m![1])).toBe(PROTOCOL_VERSION);
});

test("capability tokens match the Rust proto", () => {
  const lib = rust("crates/proto/src/lib.rs");
  const expected = ["ingest", "add", "recall", "context", "list", "forget"];
  for (const cap of expected) {
    expect(lib).toContain(`pub const ${cap.toUpperCase()}: &str = "${cap}";`);
  }
});

test("MAX_MESSAGE_BYTES matches the Rust framing", () => {
  const frame = rust("crates/proto/src/frame.rs");
  const m = frame.match(/MAX_MESSAGE_BYTES:\s*u32\s*=\s*([0-9*\s]+);/);
  expect(m).not.toBeNull();
  const value = m![1].split("*").reduce((acc, part) => acc * Number(part.trim()), 1);
  expect(value).toBe(MAX_MESSAGE_BYTES);
});
