// Typed client for the daemon's dashboard JSON API + SSE live stream.
// Shapes mirror the daemon's `dashboard` DTOs.

export interface Scope {
  tag: string;
  latest: number;
  total: number;
}

export interface GraphNode {
  id: string;
  content: string;
  kind: string;
  strength: number;
  is_latest: boolean;
  created_at: number;
  last_accessed_at: number;
  expires_at: number | null;
}

export interface GraphEdge {
  source: string;
  target: string;
  kind: string;
  created_at: number;
}

export interface GraphData {
  scope: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface Mem {
  id: string;
  content: string;
  kind: string;
  strength: number;
  created_at: number;
  score: number | null;
}

export interface ChangeEvent {
  scope: string;
  op: string;
}

async function getJSON<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
  return (await res.json()) as T;
}

export const api = {
  scopes: () => getJSON<Scope[]>("/api/scopes"),

  graph: (scope: string) =>
    getJSON<GraphData>(`/api/graph?scope=${encodeURIComponent(scope)}`),

  search: (scope: string, q: string, k = 10) =>
    getJSON<Mem[]>(
      `/api/search?scope=${encodeURIComponent(scope)}&q=${encodeURIComponent(q)}&k=${k}`,
    ),

  async forget(id: string): Promise<void> {
    const res = await fetch("/api/forget", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ id }),
    });
    if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
  },

  // Subscribe to the live change stream. Returns the EventSource so callers can
  // close it on teardown.
  events(onChange: (e: ChangeEvent) => void): EventSource {
    const es = new EventSource("/api/events");
    es.addEventListener("change", (ev) => {
      try {
        onChange(JSON.parse((ev as MessageEvent).data) as ChangeEvent);
      } catch {
        /* ignore malformed events */
      }
    });
    return es;
  },
};

const KIND_COLORS: Record<string, string> = {
  fact: "#4c6ef5",
  preference: "#f59f00",
  episode: "#37b24d",
};

/** Node color by memory kind (gray fallback for unknown kinds). */
export function kindColor(kind: string): string {
  return KIND_COLORS[kind] ?? "#868e96";
}

/** A compact relative time like "3h ago" from a Unix-seconds timestamp. */
export function relativeTime(unixSeconds: number): string {
  const secs = Math.max(0, Math.floor(Date.now() / 1000) - unixSeconds);
  if (secs < 60) return `${secs}s ago`;
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

/** Friendlier label for a container tag (drops the `memeora_` prefix, trims hashes). */
export function shortenScope(tag: string): string {
  const t = tag.replace(/^memeora_/, "");
  const m = t.match(/^(user|project)_([0-9a-f]{6,})/i);
  if (m) return `${m[1]}_${m[2].slice(0, 6)}…`;
  return t;
}
