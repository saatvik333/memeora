<script module lang="ts">
  // Per-scope layout + community cache (module-level so it survives the {#key}
  // remount). Reloads of the same space reuse positions/colours → no reshuffle.
  const layoutCache = new Map<
    string,
    { pos: Record<string, [number, number]>; comm: Record<string, number> }
  >();
</script>

<script lang="ts">
  import Graph from "graphology";
  import Sigma from "sigma";
  import forceAtlas2 from "graphology-layout-forceatlas2";
  import FA2Layout from "graphology-layout-forceatlas2/worker";
  import louvain from "graphology-communities-louvain";
  import { NodeBorderProgram } from "@sigma/node-border";
  import type { GraphData } from "./lib/api";

  let {
    data,
    onselect,
    selected = null,
    forgotten = [],
    onloading,
    dark = false,
  }: {
    data: GraphData;
    dark?: boolean;
    selected?: string | null;
    forgotten?: string[];
    onselect?: (id: string | null) => void;
    onloading?: (v: boolean) => void;
  } = $props();

  let container: HTMLDivElement;
  let renderer: Sigma | undefined;
  // Non-reactive mirrors read by the node reducer (so selection/forget changes
  // re-render via refresh() without rebuilding the whole layout).
  const selRef: { id: string | null } = { id: null };
  const goneRef = new Set<string>();

  // Distinguishable, lightly-muted categorical palette (Tableau-10 style),
  // cycled across detected communities.
  const PALETTE = [
    "#5b8bb5", "#e8915b", "#cf6a6b", "#7ec4be", "#6aa86a",
    "#e3c95c", "#b58bb0", "#e2a0aa", "#a8857a", "#b7b0ab",
    "#6f9bd1", "#d98c4a", "#9a7fc2", "#5fb39a", "#c9a13b",
  ];

  const zoomIn = () => renderer?.getCamera().animatedZoom();
  const zoomOut = () => renderer?.getCamera().animatedUnzoom();
  const resetView = () => renderer?.getCamera().animatedReset();

  $effect(() => {
    if (!container) return;
    onloading?.(true);

    const nodeDim = dark ? "#4a4e55" : "#c4c4c1";
    const selBorder = dark ? "#f4f4f5" : "#15171a";
    // Cross-cluster links: muted neutral, dark-on-light / light-on-dark so they
    // stay visible in both themes.
    const edgeCross = dark ? "#7d828c33" : "#33363b3d";
    // Within-cluster links take a dimmed cluster colour; light mode needs more
    // opacity than dark to read against white.
    const intraAlpha = dark ? "73" : "b3";

    const ids = new Set(data.nodes.map((n) => n.id));
    const deg = new Map<string, number>();
    const validEdges: [string, string][] = [];
    const seenE = new Set<string>();
    for (const e of data.edges) {
      if (!ids.has(e.source) || !ids.has(e.target) || e.source === e.target) continue;
      const k = e.source < e.target ? `${e.source}|${e.target}` : `${e.target}|${e.source}`;
      if (seenE.has(k)) continue;
      seenE.add(k);
      validEdges.push([e.source, e.target]);
      deg.set(e.source, (deg.get(e.source) ?? 0) + 1);
      deg.set(e.target, (deg.get(e.target) ?? 0) + 1);
    }

    // Undirected: links are undirected, and Louvain requires a non-mixed graph.
    const g = new Graph({ type: "undirected" });
    const count = Math.max(data.nodes.length, 1);
    const big = count > 12;
    // Small spaces have lots of room, so nodes are sized up to stay legible.
    const sizeOf = (deg: number) =>
      big ? Math.min(3 + Math.sqrt(deg) * 1.7, 16) : Math.min(10 + Math.sqrt(deg) * 3, 24);
    // Reuse a cached layout for this scope when it covers every node.
    const cache = layoutCache.get(data.scope);
    const haveLayout = !!cache && data.nodes.every((n) => cache.pos[n.id]);
    data.nodes.forEach((n, i) => {
      if (g.hasNode(n.id)) return;
      const p = cache?.pos[n.id];
      const a = (2 * Math.PI * i) / count;
      g.addNode(n.id, {
        x: p ? p[0] : Math.cos(a) * 300,
        y: p ? p[1] : Math.sin(a) * 300,
        // Size by importance (degree) — pronounced enough that hubs stand out.
        size: sizeOf(deg.get(n.id) ?? 0),
        type: "border",
        borderSize: 1.5,
      });
    });
    for (const [s, t] of validEdges) {
      if (!g.hasEdge(s, t)) g.addEdge(s, t, { size: 0.6 });
    }

    // Communities (Louvain) → per-cluster colour. Edges take a dimmed cluster
    // colour within a cluster, muted neutral across clusters.
    let community: Record<string, number> = haveLayout && cache ? cache.comm : {};
    if (!haveLayout) {
      try {
        if (g.size > 0) community = louvain(g);
      } catch {
        community = {}; // never let community detection break rendering
      }
    }
    const colorOf = (id: string) => PALETTE[(community[id] ?? 0) % PALETTE.length];
    const latest = new Map(data.nodes.map((n) => [n.id, n.is_latest]));
    g.forEachNode((id) => {
      const c = latest.get(id) ? colorOf(id) : nodeDim;
      g.setNodeAttribute(id, "color", c);
      g.setNodeAttribute(id, "borderColor", c);
    });
    g.forEachEdge((edge, _a, s, t) => {
      const sameCluster = (community[s] ?? -1) === (community[t] ?? -2);
      g.setEdgeAttribute(
        edge,
        "color",
        sameCluster && latest.get(s) ? colorOf(s) + intraAlpha : edgeCross,
      );
    });

    // Reducer: dim forgotten nodes, draw a contrasting border on the selected one.
    const reducer = (node: string, attrs: Record<string, unknown>) => {
      const out = { ...attrs };
      if (goneRef.has(node)) {
        out.color = nodeDim;
        out.borderColor = nodeDim;
      }
      if (node === selRef.id) {
        out.borderColor = selBorder;
        out.borderSize = 3.5;
        out.size = ((out.size as number) ?? 5) + 2;
        out.zIndex = 1;
      }
      return out;
    };

    let warmupTimer: ReturnType<typeof setTimeout> | undefined;
    let fa2: FA2Layout | undefined;

    const settings = {
      ...forceAtlas2.inferSettings(g),
      barnesHutOptimize: true,
      gravity: 1.2,
      scalingRatio: 14,
    };

    const finish = () => {
      // Cache the settled layout + communities for this scope.
      const pos: Record<string, [number, number]> = {};
      g.forEachNode((id, a) => (pos[id] = [a.x as number, a.y as number]));
      layoutCache.set(data.scope, { pos, comm: community });

      renderer = new Sigma(g, container, {
        renderLabels: false,
        hideEdgesOnMove: true,
        defaultNodeType: "border",
        nodeProgramClasses: { border: NodeBorderProgram },
        defaultEdgeColor: edgeCross,
        nodeReducer: reducer,
      });

      let dragged: string | null = null;
      renderer.on("downNode", ({ node }) => {
        dragged = node;
        if (!renderer!.getCustomBBox()) renderer!.setCustomBBox(renderer!.getBBox());
        renderer!.getCamera().disable();
        container.style.cursor = "grabbing";
      });
      renderer.getMouseCaptor().on("mousemovebody", (e) => {
        if (!dragged) return;
        const pos = renderer!.viewportToGraph(e);
        g.setNodeAttribute(dragged, "x", pos.x);
        g.setNodeAttribute(dragged, "y", pos.y);
        e.preventSigmaDefault();
      });
      renderer.getMouseCaptor().on("mouseup", () => {
        if (!dragged) return;
        dragged = null;
        renderer!.getCamera().enable();
        renderer!.setCustomBBox(null);
        container.style.cursor = "";
      });
      renderer.on("enterNode", () => {
        if (!dragged) container.style.cursor = "grab";
      });
      renderer.on("leaveNode", () => {
        if (!dragged) container.style.cursor = "";
      });
      renderer.on("clickNode", ({ node }) => onselect?.(node));
      renderer.on("clickStage", () => onselect?.(null));

      onloading?.(false);
    };

    if (big && !haveLayout) {
      // Larger graph, first open: settle in a worker so the UI (and the spinner)
      // stays responsive, then render the settled, static layout.
      fa2 = new FA2Layout(g, { settings });
      fa2.start();
      warmupTimer = setTimeout(() => {
        fa2?.stop();
        fa2?.kill();
        fa2 = undefined;
        finish();
      }, 1500);
    } else {
      // Cached layout (instant, no reshuffle) or a tiny graph (the ring seed is
      // already clean) — render straight away.
      finish();
    }

    return () => {
      clearTimeout(warmupTimer);
      fa2?.stop();
      fa2?.kill();
      renderer?.kill();
      renderer = undefined;
    };
  });

  // Apply selection / forget changes without rebuilding the layout.
  $effect(() => {
    selRef.id = selected;
    goneRef.clear();
    for (const id of forgotten) goneRef.add(id);
    renderer?.refresh({ skipIndexation: true });
  });
</script>

<div class="graph">
  <div class="sigma" bind:this={container}></div>
  <div class="zoom">
    <button class="zoom-btn" onclick={zoomIn} aria-label="Zoom in">
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 5v14M5 12h14" /></svg>
    </button>
    <button class="zoom-btn" onclick={zoomOut} aria-label="Zoom out">
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14" /></svg>
    </button>
    <button class="zoom-btn" onclick={resetView} aria-label="Reset view">
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <circle cx="12" cy="12" r="3" /><path d="M12 3v3M12 18v3M3 12h3M18 12h3" />
      </svg>
    </button>
  </div>
</div>

<style>
  .graph {
    position: relative;
    width: 100%;
    height: 100%;
  }
  .sigma {
    position: absolute;
    inset: 0;
  }
  .zoom {
    position: absolute;
    right: 14px;
    bottom: 14px;
    z-index: 3;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .zoom-btn {
    display: grid;
    place-items: center;
    width: 30px;
    height: 30px;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--panel);
    color: var(--muted);
    cursor: pointer;
  }
  .zoom-btn:hover {
    background: var(--elevated);
    color: var(--text);
    border-color: var(--border-strong);
  }
  .zoom-btn svg {
    width: 16px;
    height: 16px;
    fill: none;
    stroke: currentColor;
    stroke-width: 2;
    stroke-linecap: round;
    stroke-linejoin: round;
  }
</style>
