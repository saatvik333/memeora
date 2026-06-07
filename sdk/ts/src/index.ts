// TypeScript client for the memeora daemon IPC protocol.
//
// Mirrors the Rust `memeora-client`: a length-delimited JSON framing
// (u32 big-endian length prefix + UTF-8 JSON body) over a local socket, with one
// typed method per protocol verb. The contract is documented in docs/PROTOCOL.md.

import net from "node:net";

/** Wire protocol version this client speaks. */
export const PROTOCOL_VERSION = 1;
/** Default socket name the daemon listens on. */
export const DEFAULT_SOCKET = "memeora-daemon.sock";
/** Maximum frame size (matches the daemon's guard). */
const MAX_MESSAGE_BYTES = 16 * 1024 * 1024;

/** A memory projected onto the wire. */
export interface MemoryDto {
  id: string;
  content: string;
  kind: string;
  strength: number;
  created_at: number;
  /** Relevance score for search results; absent for plain listings. */
  score?: number | null;
}

interface HelloResponse {
  type: "hello";
  protocol_version: number;
  server_version: string;
  capabilities?: string[];
}
interface IngestedResponse { type: "ingested"; added: number; reinforced: number }
interface AddedResponse { type: "added"; id: string }
interface MemoriesResponse { type: "memories"; memories: MemoryDto[] }
interface ContextResponse { type: "context"; statics: MemoryDto[]; dynamics: MemoryDto[] }
interface ForgottenResponse { type: "forgotten" }
interface ErrorResponse { type: "error"; message: string }

type Response =
  | HelloResponse
  | IngestedResponse
  | AddedResponse
  | MemoriesResponse
  | ContextResponse
  | ForgottenResponse
  | ErrorResponse;

/**
 * Resolve a socket string to a Node connect target.
 *
 * A string containing a path separator is used verbatim (a filesystem socket
 * path on Unix, or a `\\.\pipe\...` path on Windows) — this is the most portable
 * choice and recommended for cross-language use (run the daemon with
 * `MEMEORA_SOCKET=/path/to.sock`). A bare name maps to the platform's namespaced
 * form: a Windows named pipe, or the Linux abstract namespace.
 */
function resolveSocket(socket: string): string {
  if (socket.includes("/") || socket.includes("\\")) return socket;
  if (process.platform === "win32") return `\\\\.\\pipe\\${socket}`;
  if (process.platform === "linux") return `\0${socket}`; // abstract namespace
  // Other Unix has no abstract namespace; recommend a filesystem path instead.
  return socket;
}

/** A connected client to a memeora daemon. */
export class Client {
  private socket: net.Socket;
  private buffer: Buffer = Buffer.alloc(0);
  private pending: Array<{
    resolve: (r: Response) => void;
    reject: (e: Error) => void;
  }> = [];
  private serverVersion = "";
  private capabilities: string[] = [];

  private constructor(socket: net.Socket) {
    this.socket = socket;
    socket.on("data", (chunk) => this.onData(chunk));
    socket.on("error", (err) => this.failAll(err));
    socket.on("close", () => this.failAll(new Error("daemon closed the connection")));
  }

  /**
   * Connect to a daemon and perform the version handshake. Rejects if the
   * daemon's protocol version differs from {@link PROTOCOL_VERSION}.
   */
  static async connect(socket: string = DEFAULT_SOCKET): Promise<Client> {
    const sock = net.connect(resolveSocket(socket));
    const client = new Client(sock);
    await new Promise<void>((resolve, reject) => {
      sock.once("connect", resolve);
      sock.once("error", reject);
    });
    const hello = await client.call({ op: "hello", protocol_version: PROTOCOL_VERSION });
    if (hello.type !== "hello") throw client.unexpected(hello);
    if (hello.protocol_version !== PROTOCOL_VERSION) {
      sock.destroy();
      throw new Error(
        `protocol version mismatch: client speaks v${PROTOCOL_VERSION}, daemon speaks v${hello.protocol_version}`,
      );
    }
    client.serverVersion = hello.server_version;
    client.capabilities = hello.capabilities ?? [];
    return client;
  }

  /** The daemon's crate version, captured at connect. */
  getServerVersion(): string {
    return this.serverVersion;
  }

  /** Capabilities the daemon advertised at connect. */
  getCapabilities(): string[] {
    return [...this.capabilities];
  }

  /** Whether the connected daemon advertised support for `capability`. */
  supports(capability: string): boolean {
    return this.capabilities.includes(capability);
  }

  /** Ingest raw text; returns the `(added, reinforced)` counts. */
  async ingest(scope: string, text: string): Promise<{ added: number; reinforced: number }> {
    const r = await this.call({ op: "ingest", scope, text });
    if (r.type === "ingested") return { added: r.added, reinforced: r.reinforced };
    throw this.unexpected(r);
  }

  /** Add a single explicit memory; returns its id. */
  async add(scope: string, content: string, kind = "fact"): Promise<string> {
    const r = await this.call({ op: "add", scope, content, kind });
    if (r.type === "added") return r.id;
    throw this.unexpected(r);
  }

  /** Hybrid search within a scope. */
  async recall(scope: string, query: string, k = 10): Promise<MemoryDto[]> {
    const r = await this.call({ op: "recall", scope, query, k });
    if (r.type === "memories") return r.memories;
    throw this.unexpected(r);
  }

  /** Fetch the profile (stable facts/prefs + recent episodes) for a scope. */
  async context(scope: string): Promise<{ statics: MemoryDto[]; dynamics: MemoryDto[] }> {
    const r = await this.call({ op: "context", scope });
    if (r.type === "context") return { statics: r.statics, dynamics: r.dynamics };
    throw this.unexpected(r);
  }

  /** List the latest memories in a scope. */
  async list(scope: string, limit = 20): Promise<MemoryDto[]> {
    const r = await this.call({ op: "list", scope, limit });
    if (r.type === "memories") return r.memories;
    throw this.unexpected(r);
  }

  /** Soft-forget a memory by id. */
  async forget(id: string): Promise<void> {
    const r = await this.call({ op: "forget", id });
    if (r.type === "forgotten") return;
    throw this.unexpected(r);
  }

  /** Close the connection. */
  close(): void {
    this.socket.end();
  }

  // --- framing internals ---

  private call(request: unknown): Promise<Response> {
    return new Promise<Response>((resolve, reject) => {
      this.pending.push({ resolve, reject });
      const body = Buffer.from(JSON.stringify(request), "utf8");
      const header = Buffer.alloc(4);
      header.writeUInt32BE(body.length, 0);
      this.socket.write(Buffer.concat([header, body]));
    });
  }

  private onData(chunk: Buffer): void {
    this.buffer = Buffer.concat([this.buffer, chunk]);
    // Responses arrive in request order, so resolve the FIFO queue per frame.
    while (this.buffer.length >= 4) {
      const len = this.buffer.readUInt32BE(0);
      if (len > MAX_MESSAGE_BYTES) {
        this.failAll(new Error(`frame too large: ${len} bytes`));
        return;
      }
      if (this.buffer.length < 4 + len) break;
      const body = this.buffer.subarray(4, 4 + len);
      this.buffer = this.buffer.subarray(4 + len);
      const waiter = this.pending.shift();
      if (!waiter) continue;
      try {
        waiter.resolve(JSON.parse(body.toString("utf8")) as Response);
      } catch (e) {
        waiter.reject(e instanceof Error ? e : new Error(String(e)));
      }
    }
  }

  private failAll(err: Error): void {
    const waiters = this.pending;
    this.pending = [];
    for (const w of waiters) w.reject(err);
  }

  private unexpected(r: Response): Error {
    if (r.type === "error") return new Error(r.message);
    return new Error(`unexpected daemon response: ${r.type}`);
  }
}
