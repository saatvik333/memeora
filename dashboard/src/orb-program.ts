// A glowing "orb" node renderer for Sigma — the Obsidian graph look.
//
// Sigma's stock `NodeCircleProgram` carves a flat, hard-edged disc. This subclass
// keeps all of its (proven) data plumbing and swaps only the GLSL: the carrier
// triangle is enlarged to leave room around the disc, and the fragment shader adds
// a soft radial falloff outside the core — a bright orb with a glow halo, which
// reads especially well on a dark canvas.
//
// Sigma blends with premultiplied alpha (`gl.blendFunc(ONE, ONE_MINUS_SRC_ALPHA)`),
// so the fragment outputs `v_color * coverage`, exactly like the stock program.

import { NodeCircleProgram } from "sigma/rendering";

const VERTEX_SHADER = /* glsl */ `
attribute vec4 a_id;
attribute vec4 a_color;
attribute vec2 a_position;
attribute float a_size;
attribute float a_angle;

uniform mat3 u_matrix;
uniform float u_sizeRatio;
uniform float u_correctionRatio;

varying vec4 v_color;
varying vec2 v_diffVector;
varying float v_radius;

const float bias = 255.0 / 254.0;
// Enlarge the carrier triangle past the disc so the glow has room to fall off.
const float GLOW = 1.8;

void main() {
  float size = a_size * u_correctionRatio / u_sizeRatio * 4.0;
  vec2 diffVector = size * GLOW * vec2(cos(a_angle), sin(a_angle));
  vec2 position = a_position + diffVector;
  gl_Position = vec4((u_matrix * vec3(position, 1)).xy, 0, 1);

  v_diffVector = diffVector;
  v_radius = size / 2.0;

  #ifdef PICKING_MODE
  v_color = a_id;
  #else
  v_color = a_color;
  #endif
  v_color.a *= bias;
}
`;

const FRAGMENT_SHADER = /* glsl */ `
precision highp float;

varying vec4 v_color;
varying vec2 v_diffVector;
varying float v_radius;

uniform float u_correctionRatio;

const vec4 transparent = vec4(0.0, 0.0, 0.0, 0.0);

void main(void) {
  float border = u_correctionRatio * 2.0;
  float d = length(v_diffVector);

  #ifdef PICKING_MODE
  // Only the solid core is clickable — not the glow halo.
  if (d > v_radius + border)
    gl_FragColor = transparent;
  else
    gl_FragColor = v_color;
  #else
  // Bright core with an antialiased edge.
  float core = 1.0 - smoothstep(v_radius - border, v_radius + border, d);
  // Soft halo falling off outside the core — the orb glow.
  float glow = 1.0 - smoothstep(v_radius, v_radius * 1.7, d);
  glow = pow(max(glow, 0.0), 2.0) * 0.5;
  float a = clamp(max(core, glow), 0.0, 1.0);
  // Premultiplied alpha (Sigma blends with ONE, ONE_MINUS_SRC_ALPHA).
  gl_FragColor = v_color * a;
  #endif
}
`;

export default class NodeOrbProgram extends NodeCircleProgram {
  getDefinition() {
    return {
      ...super.getDefinition(),
      VERTEX_SHADER_SOURCE: VERTEX_SHADER,
      FRAGMENT_SHADER_SOURCE: FRAGMENT_SHADER,
    };
  }
}
