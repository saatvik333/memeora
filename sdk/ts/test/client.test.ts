// Tests for the hand-rolled framing/handshake against an in-process stub daemon
// (no real memeora-daemon needed). Run with `bun test`.

import { afterEach, expect, test } from "bun:test";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { Client } from "../src/index.ts";

const MAX = 16 * 1024 * 1024;

function frame(obj: unknown): Buffer {
  const body = Buffer.from(JSON.stringify(obj), "utf8");
  const header = Buffer.alloc(4);
  header.writeUInt32BE(body.length, 0);
  return Buffer.concat([header, body]);
}

type Handler = (req: any, socket: net.Socket) => void;

let servers: net.Server[] = [];
afterEach(() => {
  for (const s of servers) s.close();
  servers = [];
});

function startStub(handler: Handler): Promise<string> {
  const sockPath = path.join(
    os.tmpdir(),
    `memeora-sdk-test-${Date.now()}-${servers.length}.sock`,
  );
  const server = net.createServer((socket) => {
    let buf = Buffer.alloc(0);
    socket.on("data", (chunk) => {
      buf = Buffer.concat([buf, chunk]);
      while (buf.length >= 4) {
        const len = buf.readUInt32BE(0);
        if (buf.length < 4 + len) break;
        const body = buf.subarray(4, 4 + len);
        buf = buf.subarray(4 + len);
        handler(JSON.parse(body.toString("utf8")), socket);
      }
    });
  });
  servers.push(server);
  return new Promise((resolve) => server.listen(sockPath, () => resolve(sockPath)));
}

// Auto-answer the handshake, delegate everything else.
function withHello(rest: Handler, hello?: unknown): Handler {
  return (req, s) => {
    if (req.op === "hello") {
      s.write(
        frame(hello ?? { type: "hello", protocol_version: 1, server_version: "1.0.0", capabilities: [] }),
      );
      return;
    }
    rest(req, s);
  };
}

test("handshake captures server version and capabilities", async () => {
  const p = await startStub(
    withHello(() => {}, {
      type: "hello",
      protocol_version: 1,
      server_version: "9.9.9",
      capabilities: ["recall", "add"],
    }),
  );
  const c = await Client.connect(p);
  expect(c.getServerVersion()).toBe("9.9.9");
  expect(c.supports("recall")).toBe(true);
  expect(c.supports("nope")).toBe(false);
  c.close();
});

test("hello without capabilities defaults to empty (back-compat)", async () => {
  const p = await startStub(
    withHello(() => {}, { type: "hello", protocol_version: 1, server_version: "1.0.0" }),
  );
  const c = await Client.connect(p);
  expect(c.getCapabilities()).toEqual([]);
  c.close();
});

test("hello with future fields still connects", async () => {
  const p = await startStub(
    withHello(() => {}, {
      type: "hello",
      protocol_version: 1,
      server_version: "1.0.0",
      capabilities: [],
      future: true,
    }),
  );
  const c = await Client.connect(p);
  expect(c.getCapabilities()).toEqual([]);
  c.close();
});

test("protocol version mismatch rejects", async () => {
  const p = await startStub(
    withHello(() => {}, { type: "hello", protocol_version: 2, server_version: "1.0.0" }),
  );
  await expect(Client.connect(p)).rejects.toThrow(/protocol version mismatch/);
});

test("response split across chunks is reassembled", async () => {
  const p = await startStub(
    withHello((req, s) => {
      if (req.op === "recall") {
        const f = frame({
          type: "memories",
          memories: [{ id: "m1", content: "hi", kind: "fact", strength: 1, created_at: 1, score: 0.5 }],
        });
        s.write(f.subarray(0, 3)); // header split mid-way
        setTimeout(() => s.write(f.subarray(3)), 5);
      }
    }),
  );
  const c = await Client.connect(p);
  const hits = await c.recall("s", "q", 5);
  expect(hits.length).toBe(1);
  expect(hits[0]!.id).toBe("m1");
  c.close();
});

test("concurrent calls resolve in FIFO order", async () => {
  const p = await startStub(
    withHello((req, s) => {
      if (req.op === "add") s.write(frame({ type: "added", id: req.content }));
    }),
  );
  const c = await Client.connect(p);
  const [a, b] = await Promise.all([c.add("s", "first"), c.add("s", "second")]);
  expect(a).toBe("first");
  expect(b).toBe("second");
  c.close();
});

test("oversize outgoing frame rejects without desyncing the queue", async () => {
  const p = await startStub(
    withHello((req, s) => {
      if (req.op === "add") s.write(frame({ type: "added", id: "ok" }));
    }),
  );
  const c = await Client.connect(p);
  const huge = "x".repeat(MAX + 1);
  await expect(c.add("s", huge)).rejects.toThrow(/MAX_MESSAGE_BYTES/);
  // The queue is intact: a normal call still works.
  expect(await c.add("s", "small")).toBe("ok");
  c.close();
});

test("daemon error response throws the message", async () => {
  const p = await startStub(
    withHello((req, s) => {
      if (req.op === "recall") s.write(frame({ type: "error", message: "boom" }));
    }),
  );
  const c = await Client.connect(p);
  await expect(c.recall("s", "q")).rejects.toThrow(/boom/);
  c.close();
});
