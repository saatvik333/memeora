<script lang="ts">
  import Graph from "graphology";
  import Sigma from "sigma";
  import { drawDiscNodeLabel } from "sigma/rendering";
  import type { Settings } from "sigma/settings";
  import type { NodeDisplayData, PartialButFor } from "sigma/types";
  import forceAtlas2 from "graphology-layout-forceatlas2";
  import NodeOrbProgram from "./orb-program";
  import type { GraphData } from "./lib/api";
  import { kindColor } from "./lib/api";

  let { data, onselect, dark = false }: {
    data: GraphData;
    dark?: boolean;
    onselect?: (id: string | null) => void;
  } = $props();

  let container: HTMLDivElement;

  function truncate(text: string, n = 56): string {
    return text.length > n ? text.slice(0, n) + "…" : text;
  }

  // Sigma's default node-hover renderer hardcodes a white background + a heavy
  // black shadow, so it ignores the theme. Override the documented hook to mirror
  // its geometry with theme colors and a light shadow, then reuse Sigma's own
  // label drawer for the text (which already honors `labelColor`).
  function makeHover(isDark: boolean) {
    const bg = isDark ? "#1e1e21" : "#ffffff";
    const shadow = isDark ? "rgba(0,0,0,0.55)" : "rgba(0,0,0,0.12)";
    return (
      context: CanvasRenderingContext2D,
      data: PartialButFor<NodeDisplayData, "x" | "y" | "size" | "label" | "color">,
      settings: Settings,
    ) => {
      const { labelSize: size, labelFont: font, labelWeight: weight } = settings;
      context.font = `${weight} ${size}px ${font}`;
      context.fillStyle = bg;
      context.shadowOffsetX = 0;
      context.shadowOffsetY = 0;
      context.shadowBlur = 4;
      context.shadowColor = shadow;
      const PADDING = 2;
      if (typeof data.label === "string") {
        const textWidth = context.measureText(data.label).width;
        const boxWidth = Math.round(textWidth + 5);
        const boxHeight = Math.round(size + 2 * PADDING);
        const radius = Math.max(data.size, size / 2) + PADDING;
        const angle = Math.asin(boxHeight / 2 / radius);
        const dx = Math.sqrt(Math.abs(radius ** 2 - (boxHeight / 2) ** 2));
        context.beginPath();
        context.moveTo(data.x + dx, data.y + boxHeight / 2);
        context.lineTo(data.x + radius + boxWidth, data.y + boxHeight / 2);
        context.lineTo(data.x + radius + boxWidth, data.y - boxHeight / 2);
        context.lineTo(data.x + dx, data.y - boxHeight / 2);
        context.arc(data.x, data.y, radius, angle, -angle);
        context.closePath();
        context.fill();
      } else {
        context.beginPath();
        context.arc(data.x, data.y, data.size + PADDING, 0, Math.PI * 2);
        context.closePath();
        context.fill();
      }
      context.shadowBlur = 0;
      drawDiscNodeLabel(context, data, settings);
    };
  }

  function build(d: GraphData, isDark: boolean): Graph {
    // Dimmed (superseded/forgotten) nodes and edges differ per theme so they read
    // as "faded" against either a light or a dark canvas.
    const dimNode = isDark ? "#34343a" : "#d6d6d3";
    const edgeColor = isDark ? "#2a2a2e" : "#e2e2e0";
    const g = new Graph();
    const count = Math.max(d.nodes.length, 1);
    const strengths = new Map<string, number>();
    d.nodes.forEach((node, i) => {
      // Seed positions on a circle; ForceAtlas2 then spreads them out.
      const angle = (2 * Math.PI * i) / count;
      strengths.set(node.id, node.strength);
      g.addNode(node.id, {
        label: truncate(node.content),
        x: Math.cos(angle),
        y: Math.sin(angle),
        size: 6,
        // Superseded / forgotten memories are dimmed rather than hidden.
        color: node.is_latest ? kindColor(node.kind) : dimNode,
      });
    });
    for (const e of d.edges) {
      if (
        g.hasNode(e.source) &&
        g.hasNode(e.target) &&
        !g.hasEdge(e.source, e.target)
      ) {
        g.addEdge(e.source, e.target, { size: 1, color: edgeColor });
      }
    }
    // Obsidian-style sizing: a memory grows with how connected it is (degree),
    // nudged by its strength — so hubs read as larger orbs.
    g.forEachNode((id) => {
      const deg = g.degree(id);
      const strength = strengths.get(id) ?? 0;
      g.setNodeAttribute(id, "size", Math.min(6 + Math.sqrt(deg) * 4 + strength * 1.5, 26));
    });
    if (g.order > 2) {
      forceAtlas2.assign(g, {
        iterations: 200,
        settings: forceAtlas2.inferSettings(g),
      });
    }
    return g;
  }

  // Rebuild + render whenever the graph data or theme changes; tear down on
  // cleanup. The seed layout is deterministic, so re-theming keeps node positions.
  $effect(() => {
    const g = build(data, dark);

    // Obsidian's signature interaction: hovering a node highlights it and its
    // neighbors while fading everything else. Done with Sigma reducers driven by
    // `hovered`, recomputed each render (refresh()).
    let hovered: string | null = null;
    const fadedNode = dark ? "#2a2a2e" : "#e6e6e3";
    const litEdge = dark ? "#6b6b72" : "#b7b7b2";
    const fadedEdge = dark ? "#1b1b1e" : "#f0f0ed";

    const renderer = new Sigma(g, container, {
      renderEdgeLabels: false,
      labelColor: { color: dark ? "#ededed" : "#1c1c1c" },
      defaultDrawNodeHover: makeHover(dark),
      nodeProgramClasses: { orb: NodeOrbProgram },
      defaultNodeType: "orb",
      nodeReducer: (node, attrs) => {
        if (!hovered || node === hovered || g.areNeighbors(hovered, node)) {
          return attrs;
        }
        return { ...attrs, color: fadedNode, label: "" };
      },
      edgeReducer: (edge, attrs) => {
        if (!hovered) return attrs;
        if (g.hasExtremity(edge, hovered)) return { ...attrs, color: litEdge };
        return { ...attrs, color: fadedEdge };
      },
    });
    renderer.on("clickNode", (e) => onselect?.(e.node));
    renderer.on("clickStage", () => onselect?.(null));
    renderer.on("enterNode", (e) => {
      hovered = e.node;
      renderer.refresh();
    });
    renderer.on("leaveNode", () => {
      hovered = null;
      renderer.refresh();
    });
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
