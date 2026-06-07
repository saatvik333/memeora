<script lang="ts">
  import Graph from "graphology";
  import Sigma from "sigma";
  import forceAtlas2 from "graphology-layout-forceatlas2";
  import type { GraphData } from "./lib/api";
  import { kindColor } from "./lib/api";

  let { data, onselect }: {
    data: GraphData;
    onselect?: (id: string | null) => void;
  } = $props();

  let container: HTMLDivElement;

  function truncate(text: string, n = 56): string {
    return text.length > n ? text.slice(0, n) + "…" : text;
  }

  function build(d: GraphData): Graph {
    const g = new Graph();
    const count = Math.max(d.nodes.length, 1);
    d.nodes.forEach((node, i) => {
      // Seed positions on a circle; ForceAtlas2 then spreads them out.
      const angle = (2 * Math.PI * i) / count;
      g.addNode(node.id, {
        label: truncate(node.content),
        x: Math.cos(angle),
        y: Math.sin(angle),
        size: Math.min(4 + node.strength * 2.5, 16),
        // Superseded / forgotten memories are dimmed rather than hidden.
        color: node.is_latest ? kindColor(node.kind) : "#d4d4d4",
      });
    });
    for (const e of d.edges) {
      if (
        g.hasNode(e.source) &&
        g.hasNode(e.target) &&
        !g.hasEdge(e.source, e.target)
      ) {
        g.addEdge(e.source, e.target, { size: 1, color: "#d0d0d0" });
      }
    }
    if (g.order > 2) {
      forceAtlas2.assign(g, {
        iterations: 200,
        settings: forceAtlas2.inferSettings(g),
      });
    }
    return g;
  }

  // Rebuild + render whenever the graph data changes; tear down on cleanup.
  $effect(() => {
    const renderer = new Sigma(build(data), container, {
      renderEdgeLabels: false,
    });
    renderer.on("clickNode", (e) => onselect?.(e.node));
    renderer.on("clickStage", () => onselect?.(null));
    return () => renderer.kill();
  });
</script>

<div class="graph" bind:this={container}></div>

<style>
  .graph {
    width: 100%;
    height: 100%;
  }
</style>
