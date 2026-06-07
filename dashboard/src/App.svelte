<script lang="ts">
  import { onMount } from "svelte";
  import { AlertDialog, ScrollArea, Select, Separator, Tooltip } from "bits-ui";
  import GraphView from "./Graph.svelte";
  import {
    api,
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
  let searched = $state(false);
  let searching = $state(false);
  let live = $state(false);
  let error = $state<string | null>(null);
  let graphLoading = $state(false);
  let forgotten = $state<string[]>([]);

  // Theme is set on <html data-theme> pre-paint (index.html); toggle + persist here.
  let theme = $state<"light" | "dark">(
    document.documentElement.dataset.theme === "dark" ? "dark" : "light",
  );
  let dark = $derived(theme === "dark");
  $effect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("memeora-theme", theme);
    } catch {
      /* storage unavailable (private mode) — non-fatal */
    }
  });
  const toggleTheme = () => (theme = theme === "dark" ? "light" : "dark");

  let currentLabel = $derived(current ? shortenScope(current) : "Select a space");

  // Resolve the selected node from the graph first, then a search result.
  let selected = $derived.by(() => {
    if (!selectedId) return null;
    const node = graph.nodes.find((n) => n.id === selectedId);
    if (node) return node;
    const hit = results.find((m) => m.id === selectedId);
    if (hit) return { ...hit, is_latest: true, last_accessed_at: hit.created_at };
    return null;
  });
  // A memory already forgotten (this session) or superseded can't be forgotten again.
  let isForgotten = $derived(
    !!selected && (forgotten.includes(selected.id) || !selected.is_latest),
  );

  // Stable callbacks so passing them to <GraphView> never rebuilds its layout.
  const handleSelect = (id: string | null) => (selectedId = id);
  const setLoading = (v: boolean) => (graphLoading = v);

  async function loadScopes() {
    try {
      scopes = await api.scopes();
      if (!current && scopes.length) selectScope(scopes[0].tag);
    } catch (e) {
      error = String(e);
    }
  }

  async function loadGraph(scope: string) {
    graphLoading = true;
    try {
      graph = await api.graph(scope);
      error = null;
      if (!graph.nodes.length) graphLoading = false; // no graph to lay out
    } catch (e) {
      error = String(e);
      graphLoading = false;
    }
  }

  function selectScope(tag: string) {
    if (!tag || tag === current) return;
    current = tag;
    selectedId = null;
    results = [];
    searched = false;
    query = "";
    forgotten = [];
    loadGraph(tag);
  }

  async function runSearch() {
    if (!current || !query.trim()) {
      results = [];
      searched = false;
      return;
    }
    searching = true;
    try {
      results = await api.search(current, query);
      searched = true;
      selectedId = null; // show results, not a lingering inspector
    } catch (e) {
      error = String(e);
    } finally {
      searching = false;
    }
  }

  async function forget(id: string) {
    try {
      await api.forget(id);
      // Mark forgotten (dims it in the graph, hides the Forget button) without a
      // full graph reload — the soft-forget is persisted server-side.
      forgotten = [...forgotten, id];
      selectedId = null;
    } catch (e) {
      error = String(e);
    }
  }

  onMount(() => {
    loadScopes();
    const es = api.events((ev) => {
      live = true;
      if (current && ev.scope === current) loadGraph(current);
      loadScopes();
    });
    return () => es.close();
  });
</script>

{#snippet caret()}
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
    stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <path d="m6 9 6 6 6-6" /></svg>
{/snippet}
{#snippet check()}
  <svg class="check" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <path d="M20 6 9 17l-5-5" /></svg>
{/snippet}
{#snippet sun()}
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
    stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <circle cx="12" cy="12" r="4" />
    <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
  </svg>
{/snippet}
{#snippet moon()}
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
    stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z" /></svg>
{/snippet}

<Tooltip.Provider delayDuration={300}>
  <div class="app">
    <main class="canvas">
      {#if graph.nodes.length}
        {#key graph.scope}
          <GraphView
            data={graph}
            {dark}
            selected={selectedId}
            {forgotten}
            onselect={handleSelect}
            onloading={setLoading}
          />
        {/key}
      {:else if !graphLoading}
        <div class="empty">No memories in this space yet.</div>
      {/if}
      {#if graphLoading}
        <div class="loading"><span class="spinner"></span></div>
      {/if}
    </main>

    <aside class="sidebar">
      <header class="brand">
        <span class="brand-name">memeora</span>
        <span class="dot" class:on={live} title={live ? "live" : "idle"}></span>
        <Tooltip.Root>
          <Tooltip.Trigger
            class="icon-btn spacer"
            onclick={toggleTheme}
            aria-label="Toggle theme"
          >
            {#if dark}{@render sun()}{:else}{@render moon()}{/if}
          </Tooltip.Trigger>
          <Tooltip.Portal>
            <Tooltip.Content sideOffset={6}>
              {dark ? "Light mode" : "Dark mode"}
            </Tooltip.Content>
          </Tooltip.Portal>
        </Tooltip.Root>
      </header>

      <Separator.Root />

      <div class="section">
        <Select.Root
          type="single"
          value={current ?? ""}
          onValueChange={(v) => selectScope(v)}
        >
          <Select.Trigger aria-label="Select a space">
            <span class="truncate">{currentLabel}</span>
            {@render caret()}
          </Select.Trigger>
          <Select.Portal>
            <Select.Content sideOffset={6}>
              <Select.Viewport>
                {#each scopes as s (s.tag)}
                  <Select.Item value={s.tag} label={shortenScope(s.tag)}>
                    {#snippet children({ selected })}
                      {#if selected}{@render check()}{:else}<span class="check"></span>{/if}
                      <span class="truncate">{shortenScope(s.tag)}</span>
                      <span class="spacer muted">{s.latest}</span>
                    {/snippet}
                  </Select.Item>
                {/each}
              </Select.Viewport>
            </Select.Content>
          </Select.Portal>
        </Select.Root>
      </div>

      <div class="section">
        <input
          class="input"
          placeholder="Search this space…"
          bind:value={query}
          onkeydown={(e) => e.key === "Enter" && runSearch()}
        />
      </div>

      <Separator.Root />

      <div class="grow">
        <ScrollArea.Root>
          <ScrollArea.Viewport>
            {#if selected}
              <div class="section">
                <span class="kind">{selected.kind}</span>
                {#if isForgotten}
                  <span class="kind is-muted">forgotten</span>
                {/if}
                <p class="content">{selected.content}</p>
                <dl class="meta">
                  <dt>strength</dt>
                  <dd>{selected.strength.toFixed(2)}</dd>
                  <dt>created</dt>
                  <dd>{relativeTime(selected.created_at)}</dd>
                  <dt>last seen</dt>
                  <dd>{relativeTime(selected.last_accessed_at)}</dd>
                  <dt>id</dt>
                  <dd class="mono">{selected.id.slice(0, 16)}…</dd>
                </dl>

                {#if isForgotten}
                  <p class="muted">Soft-forgotten — hidden from recall.</p>
                {:else}
                  <AlertDialog.Root>
                    <AlertDialog.Trigger class="btn btn-danger">Forget</AlertDialog.Trigger>
                    <AlertDialog.Portal>
                      <AlertDialog.Overlay />
                      <AlertDialog.Content>
                        <AlertDialog.Title>Forget this memory?</AlertDialog.Title>
                        <AlertDialog.Description>
                          It's soft-forgotten (kept but hidden from recall), not deleted.
                        </AlertDialog.Description>
                        <div class="dialog-actions">
                          <AlertDialog.Cancel class="btn">Cancel</AlertDialog.Cancel>
                          <AlertDialog.Action
                            class="btn btn-danger"
                            onclick={() => selected && forget(selected.id)}
                          >
                            Forget
                          </AlertDialog.Action>
                        </div>
                      </AlertDialog.Content>
                    </AlertDialog.Portal>
                  </AlertDialog.Root>
                {/if}
              </div>
            {:else}
              <div class="section">
                {#if searching}
                  <div class="searching"><span class="spinner"></span></div>
                {:else if results.length}
                  <p class="label">Results</p>
                  {#each results as r (r.id)}
                    <button class="row" onclick={() => (selectedId = r.id)}>
                      <span>{r.content}</span>
                    </button>
                  {/each}
                {:else if searched}
                  <p class="muted">No matches.</p>
                {:else}
                  <p class="muted">Type to search this space.</p>
                {/if}
              </div>
            {/if}
          </ScrollArea.Viewport>
          <ScrollArea.Scrollbar orientation="vertical">
            <ScrollArea.Thumb />
          </ScrollArea.Scrollbar>
        </ScrollArea.Root>
      </div>

      {#if error}<p class="error">{error}</p>{/if}
    </aside>
  </div>
</Tooltip.Provider>
