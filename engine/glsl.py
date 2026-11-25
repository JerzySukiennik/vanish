"""Shared GLSL snippets: texture slots, noise, background and person sampling."""

HEADER = """#version 330
uniform sampler2D u_live;
uniform sampler2D u_plate;
uniform sampler2D u_hide;
uniform sampler2D u_hideTight;
uniform sampler2D u_liveMask;
uniform sampler2D u_snap;
uniform sampler2D u_snapMask;
uniform sampler2D u_snapHide;
uniform sampler2D u_plateFill;
uniform vec2 u_res;
uniform float u_t;
uniform float u_time;
uniform float u_dur;
uniform vec3 u_gain;

float hash11(float p) { p = fract(p * 0.1031); p *= p + 33.33; p *= p + p; return fract(p); }
float hash21(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
vec2 hash22(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973)); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.xx + p3.yz) * p3.zy); }
float vnoise(vec2 p) {
    vec2 i = floor(p); vec2 f = fract(p); vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash21(i), hash21(i + vec2(1.0, 0.0)), u.x), mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), u.x), u.y);
}
float fbm(vec2 p) { float s = 0.0; float a = 0.5; for (int i = 0; i < 4; i++) { s += a * vnoise(p); p *= 2.03; a *= 0.5; } return s / 0.9375; }
bool inside(vec2 uv) { return uv.x >= 0.0 && uv.y >= 0.0 && uv.x <= 1.0 && uv.y <= 1.0; }
vec3 plateLod(vec2 uv, float lod) {
    vec4 p = textureLod(u_plate, uv, lod);
    vec3 f = textureLod(u_plateFill, uv, 0.0).rgb;
    return min(mix(f, p.rgb, smoothstep(0.35, 0.65, p.a)) * u_gain, vec3(1.0));
}
vec3 plateAt(vec2 uv) { return plateLod(uv, 0.0); }
vec3 liveAt(vec2 uv) { return textureLod(u_live, uv, 0.0).rgb; }
vec3 snapAt(vec2 uv) { return textureLod(u_snap, uv, 0.0).rgb; }
vec3 bgLive(vec2 uv) { return mix(liveAt(uv), plateAt(uv), textureLod(u_hide, uv, 0.0).r); }
vec3 bgSnap(vec2 uv) { return mix(snapAt(uv), plateAt(uv), textureLod(u_snapHide, uv, 0.0).r); }
vec3 decon(vec3 c, float a, vec3 bg) { return clamp((c - (1.0 - a) * bg) / max(a, 0.02), 0.0, 1.0); }
vec4 person(vec2 uv) {
    if (!inside(uv)) return vec4(0.0);
    float a = textureLod(u_snapMask, uv, 0.0).r;
    return vec4(decon(snapAt(uv), a, plateAt(uv)), a);
}
"""

FULLSCREEN_VS = """#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() { v_uv = in_pos; gl_Position = vec4(in_pos * 2.0 - 1.0, 0.0, 1.0); }
"""

GONE_FS = HEADER + """
in vec2 v_uv; out vec4 f_color;
void main() { f_color = vec4(bgLive(v_uv), 1.0); }
"""

BLEND_FS = HEADER + """
uniform float u_k;
in vec2 v_uv; out vec4 f_color;
void main() {
    vec4 p = person(v_uv);
    vec3 still = mix(bgLive(v_uv), p.rgb, p.a);
    f_color = vec4(mix(still, liveAt(v_uv), u_k), 1.0);
}
"""

PLATE_FS = """#version 330
uniform sampler2D u_prev;
uniform sampler2D u_live;
uniform sampler2D u_hide;
uniform float u_rate;
uniform vec3 u_gain;
in vec2 v_uv; out vec4 f_color;
void main() {
    vec4 p = textureLod(u_prev, v_uv, 0.0);
    vec3 l = textureLod(u_live, v_uv, 0.0).rgb / max(u_gain, vec3(0.05));
    float h = textureLod(u_hide, v_uv, 0.0).r;
    float free = 1.0 - smoothstep(0.01, 0.06, h);
    float w = p.a < 0.5 ? free : u_rate * free;
    f_color = vec4(mix(p.rgb, l, w), max(p.a, step(0.5, free)));
}
"""

POINT_FS = """#version 330
in vec4 v_col; out vec4 f_color;
void main() { f_color = v_col; }
"""

SOFT_POINT_FS = """#version 330
in vec4 v_col; out vec4 f_color;
void main() {
    float d = length(gl_PointCoord * 2.0 - 1.0);
    f_color = vec4(v_col.rgb, v_col.a * (1.0 - smoothstep(0.25, 1.0, d)));
}
"""
