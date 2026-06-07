<script lang="ts">
  import { onMount } from "svelte";
  import GraphView from "./Graph.svelte";
  import {
    api,
    kindColor,
    relativeTime,
    shortenScope,
    type Scope,
    type GraphData,
    type Mem,
  } from "./lib/api";

  let scopes = $state<Scope[]>([]);
  let current = $state<string | null>(null);
  let graph = $state<GraphData>({ scope: "", nodes: [], edges: [] });
  let selectedId = $state<string | null>(null);
  let query = $state("");
  let results = $state<Mem[]>([]);
  let live = $state(false);
  let error = $state<string | null>(null);

  // Theme: initialized from <html data-theme> (set pre-paint in index.html), then
  // toggled + persisted here. `dark` drives the graph's colors reactively.
  let theme = $state<"light" | "dark">(
    document.documentElement.dataset.theme === "dark" ? "dark" : "light",
  );
  let dark = $derived(theme === "dark");
  $effect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("memeora-theme", theme);
    } catch {
      /* storage may be unavailable (private mode) — non-fatal */
    }
  });
  function toggleTheme() {
    theme = theme === "dark" ? "light" : "dark";
  }

  // Resolve the selected node from the (capped) graph first, then fall back to a
  // search result — a hit can live outside the 2000-node graph window, and without
  // this fallback clicking it would silently open nothing. Search `Mem`s lack
  // is_latest/last_accessed_at, so synthesize sensible defaults for the inspector.
  let selected = $derived.by(() => {
    if (!selectedId) return null;
    const node = graph.nodes.find((n) => n.id === selectedId);
    if (node) return node;
    const hit = results.find((m) => m.id === selectedId);
    if (hit) return { ...hit, is_latest: true, last_accessed_at: hit.created_at };
    return null;
  });

  async function loadScopes() {
    try {
      scopes = await api.scopes();
      if (!current && scopes.length) selectScope(scopes[0].tag);
    } catch (e) {
      error = String(e);
    }
  }

  async function loadGraph(scope: string) {
    try {
      graph = await api.graph(scope);
      error = null;
    } catch (e) {
      error = String(e);
    }
  }

  function selectScope(tag: string) {
    current = tag;
    selectedId = null;
    results = [];
    query = "";
    loadGraph(tag);
  }

  async function runSearch() {
    if (!current || !query.trim()) {
      results = [];
      return;
    }
    try {
      results = await api.search(current, query);
    } catch (e) {
      error = String(e);
    }
  }

  async function forget(id: string) {
    try {
      await api.forget(id);
      selectedId = null;
      if (current) loadGraph(current);
    } catch (e) {
      error = String(e);
    }
  }

  onMount(() => {
    loadScopes();
    // Live mode: refetch the current scope (and the scope list) on any change.
    const es = api.events((ev) => {
      live = true;
      if (current && ev.scope === current) loadGraph(current);
      loadScopes();
    });
    return () => es.close();
  });
</script>

<div class="app">
  <aside class="sidebar">
    <h1>
      memeora
      <span class="live" class:on={live} title={live ? "live" : "idle"}>●</span>
      <button
        class="theme-toggle"
        onclick={toggleTheme}
        title={dark ? "Switch to light mode" : "Switch to dark mode"}
        aria-label="Toggle dark mode"
      >
        {dark ? "☀" : "☾"}
      </button>
    </h1>

    <input
      class="search"
      placeholder="Search this space…"
      bind:value={query}
      onkeydown={(e) => e.key === "Enter" && runSearch()}
    />

    {#if results.length}
      <div class="results">
        <h2>Results</h2>
        <ul>
          {#each results as r (r.id)}
            <li>
              <button onclick={() => (selectedId = r.id)}>
                <span class="dot" style="background:{kindColor(r.kind)}"></span>
                {r.content}
              </button>
            </li>
          {/each}
        </ul>
      </div>
    {/if}

    <h2>Spaces</h2>
    <ul class="scopes">
      {#each scopes as s (s.tag)}
        <li class:active={s.tag === current}>
          <button onclick={() => selectScope(s.tag)} title={s.tag}>
            <span>{shortenScope(s.tag)}</span>
            <em>{s.latest}</em>
          </button>
        </li>
      {:else}
        <li class="muted">No spaces yet.</li>
      {/each}
    </ul>

    {#if error}<p class="error">{error}</p>{/if}

    <footer>
      <span class="lk" style="background:{kindColor('fact')}"></span> fact
      <span class="lk" style="background:{kindColor('preference')}"></span> pref
      <span class="lk" style="background:{kindColor('episode')}"></span> episode
    </footer>
  </aside>

  <main class="canvas">
    {#if graph.nodes.length}
      {#key graph.scope}
        <GraphView data={graph} {dark} onselect={(id) => (selectedId = id)} />
      {/key}
    {:else}
      <div class="empty">No memories in this space yet.</div>
    {/if}
  </main>

  {#if selected}
    <aside class="inspector">
      <button class="close" onclick={() => (selectedId = null)}>×</button>
      <span class="kind" style="background:{kindColor(selected.kind)}">
        {selected.kind}
      </span>
      {#if !selected.is_latest}<span class="kind muted">superseded</span>{/if}
      <p class="content">{selected.content}</p>
      <dl>
        <dt>strength</dt>
        <dd>{selected.strength.toFixed(2)}</dd>
        <dt>created</dt>
        <dd>{relativeTime(selected.created_at)}</dd>
        <dt>last seen</dt>
        <dd>{relativeTime(selected.last_accessed_at)}</dd>
        <dt>id</dt>
        <dd class="mono">{selected.id.slice(0, 16)}…</dd>
      </dl>
      <button class="forget" onclick={() => forget(selected!.id)}>Forget</button>
    </aside>
  {/if}
</div>
