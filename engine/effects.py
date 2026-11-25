"""The vanish effects. Each renders a frame for progress t in [0, 1]: 0 = person present, 1 = person gone."""

import math

import moderngl
import numpy as np

from . import glsl
from .renderer import H, MH, MW, W


def clamp01(x):
    return max(0.0, min(1.0, x))


def smooth(a, b, x):
    t = clamp01((x - a) / (b - a))
    return t * t * (3.0 - 2.0 * t)


def ease_out_back(x, s=1.6):
    x = clamp01(x) - 1.0
    return x * x * ((s + 1.0) * x + s) + 1.0


def quat_axis(axis, ang):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    return np.concatenate([[math.cos(ang / 2)], a * math.sin(ang / 2)])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def slerp(a, b, t):
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        r = a + (b - a) * t
        return r / np.linalg.norm(r)
    th = math.acos(d)
    return (math.sin((1 - t) * th) * a + math.sin(t * th) * b) / math.sin(th)


def quat_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def mat3_uniform(m):
    return tuple(float(v) for v in np.asarray(m, "f4").T.flatten())


class Effect:
    key = ""
    name = ""
    gesture = ""
    duration = 2.0

    def __init__(self, renderer):
        self.r = renderer
        self.ctx = renderer.ctx
        self.info = None
        self.build()

    def build(self):
        pass

    def begin(self, info, mask8):
        self.info = info

    def draw(self, t, time):
        raise NotImplementedError


DUST_COMMON = """
uniform vec4 u_box;
uniform vec2 u_wind;
float releaseT(vec2 uv) {
    vec2 c = 0.5 * (u_box.xy + u_box.zw);
    vec2 hs = max(0.5 * (u_box.zw - u_box.xy), vec2(0.05));
    float along = dot((uv - c) / hs, u_wind);
    float r = clamp(0.5 - 0.5 * along, 0.0, 1.0);
    float n = fbm(uv * vec2(13.0, 7.5));
    return 0.04 + clamp(r * 0.8 + (n - 0.5) * 0.55, 0.0, 1.0) * 0.5;
}
"""

DUST_FS = glsl.HEADER + DUST_COMMON + """
in vec2 v_uv; out vec4 f_color;
void main() {
    vec4 p = person(v_uv);
    float tr = releaseT(v_uv);
    float ash = smoothstep(tr - 0.06, tr, u_t);
    vec3 c = mix(p.rgb, p.rgb * 0.32 + vec3(0.015), ash);
    float a = p.a * step(u_t, tr);
    f_color = vec4(mix(bgLive(v_uv), c, a), 1.0);
}
"""

DUST_VS = glsl.HEADER + DUST_COMMON + """
in vec2 in_uv; in float in_seed;
out vec4 v_col;
void main() {
    float tr = releaseT(in_uv);
    float age = (u_t - tr) * u_dur;
    vec4 p = person(in_uv);
    if (age <= 0.0 || p.a < 0.35) { gl_Position = vec4(-9.0, -9.0, 0.0, 1.0); gl_PointSize = 1.0; v_col = vec4(0.0); return; }
    float h1 = hash11(in_seed * 91.7);
    float h2 = hash11(in_seed * 13.3 + 4.1);
    float h3 = hash11(in_seed * 57.1 + 2.3);
    vec2 px = in_uv * u_res;
    vec2 w = u_wind;
    vec2 perp = vec2(-w.y, w.x);
    float sp = mix(70.0, 330.0, h1);
    vec2 d = w * (sp * age + 380.0 * age * age * (0.4 + h2));
    d.y -= (30.0 + 150.0 * h3) * age * (0.5 + age);
    vec2 np = px * 0.0045 + vec2(age * 0.9, h2 * 7.0);
    vec2 turb = vec2(vnoise(np) - 0.5, vnoise(np + 17.3) - 0.5) * 2.0;
    d += turb * 140.0 * age + perp * (h2 - 0.5) * 70.0 * age;
    px += d;
    float life = 1.0 - smoothstep(0.3 + 0.35 * h1, 0.95 + 0.4 * h3, age);
    float lum = dot(p.rgb, vec3(0.299, 0.587, 0.114));
    vec3 ash = mix(vec3(lum) * vec3(1.0, 0.97, 0.92), vec3(0.30, 0.28, 0.26), 0.4);
    v_col = vec4(mix(p.rgb, ash, smoothstep(0.0, 0.6, age)), p.a * life);
    gl_PointSize = mix(2.7, 1.1, smoothstep(0.0, 1.0, age)) * (0.75 + 0.7 * h2);
    gl_Position = vec4(px / u_res * 2.0 - 1.0, 0.0, 1.0);
}
"""


class Dust(Effect):
    key = "dust"
    name = "Pył na wiatr"
    gesture = "otwarta dłoń"
    duration = 3.4

    def build(self):
        self.fs = self.r.program(DUST_FS)
        self.ps = self.ctx.program(vertex_shader=DUST_VS, fragment_shader=glsl.POINT_FS)
        self.vao = None

    def begin(self, info, mask8):
        self.info = info
        ys, xs = np.nonzero(mask8 > 90)
        rng = np.random.default_rng()
        n = len(xs)
        uv = np.empty((n, 3), "f4")
        uv[:, 0] = (xs + 0.5 + rng.uniform(-0.45, 0.45, n)) / MW
        uv[:, 1] = (ys + 0.5 + rng.uniform(-0.45, 0.45, n)) / MH
        uv[:, 2] = rng.uniform(0, 1000, n)
        if self.vao is not None:
            self.vao.release()
            self.vbo.release()
        self.vbo = self.ctx.buffer(uv.tobytes())
        self.vao = self.ctx.vertex_array(self.ps, [(self.vbo, "2f 1f", "in_uv", "in_seed")], skip_errors=True)
        self.count = n
        d = info.get("dir", -info["exit"])
        w = np.array([d, -0.32])
        self.wind = tuple(float(v) for v in w / np.linalg.norm(w))
        self.box = info["box"]

    def draw(self, t, time):
        r = self.r
        r.begin_frame()
        kw = dict(t=t, time=time, dur=self.duration, u_box=self.box, u_wind=self.wind)
        r.full(self.fs, **kw)
        if self.count:
            self.ctx.enable(moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)
            r.bind(self.ps, **kw)
            self.vao.render(moderngl.POINTS)
            self.ctx.disable(moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)


GLITCH_FS = glsl.HEADER + """
in vec2 v_uv; out vec4 f_color;
void main() {
    vec2 uv = v_uv;
    float t = u_t;
    float q = floor(u_time * 15.0);
    float I = smoothstep(0.0, 0.45, t);
    float fadeOut = 1.0 - smoothstep(0.86, 0.98, t);
    float spike = step(0.62, hash11(q * 1.37 + 5.0)) * I * fadeOut;
    float bands = floor(mix(10.0, 46.0, hash11(q * 3.1)));
    float b = floor(uv.y * bands);
    float on = step(1.0 - (0.2 + 0.45 * I + 0.3 * spike), hash21(vec2(b, q * 1.9)));
    float dx = (hash21(vec2(b * 7.1, q)) - 0.5) * (0.035 + 0.17 * spike) * I * on;
    vec2 suv = uv + vec2(dx, 0.0);
    vec2 cell = floor(uv * vec2(48.0, 27.0));
    if (hash21(cell + vec2(q * 1.7, q * 0.3)) < 0.1 * I + 0.15 * spike) suv += (hash22(cell + q) - 0.5) * vec2(0.12, 0.04);
    float split = 0.003 + 0.012 * I + 0.02 * spike;
    vec4 pr = person(suv + vec2(split, 0.0));
    vec4 pg = person(suv);
    vec4 pb = person(suv - vec2(split, 0.0));
    vec3 pc = vec3(pr.r, pg.g, pb.b);
    float a = max(pg.a, max(pr.a, pb.a));
    float vanish = smoothstep(0.42, 0.9, t);
    float frameOff = step(hash11(q * 9.17 + 1.3), vanish);
    vec2 blk = floor(uv * vec2(24.0, 13.5));
    float blkOff = step(hash21(blk + 3.7), vanish * 1.15 - 0.08);
    a *= (1.0 - frameOff) * (1.0 - blkOff) * (1.0 - step(0.97, t));
    pc = mix(pc, floor(pc * 6.0 + 0.5) / 6.0, spike);
    pc *= 1.0 - 0.14 * I * step(0.5, fract(uv.y * u_res.y * 0.5));
    vec2 buv = uv + vec2(dx * 0.3 * spike, 0.0);
    vec3 bg = bgLive(buv);
    bg.r = mix(bg.r, bgLive(buv + vec2(split * 0.7, 0.0)).r, spike);
    f_color = vec4(mix(bg, pc, a), 1.0);
}
"""


class Glitch(Effect):
    key = "glitch"
    name = "Cyfrowy glitch"
    gesture = "rogi"
    duration = 1.8

    def build(self):
        self.fs = self.r.program(GLITCH_FS)

    def draw(self, t, time):
        self.r.begin_frame()
        self.r.full(self.fs, t=t, time=time, dur=self.duration)


FREEZE_FS = glsl.HEADER + """
in vec2 v_uv; out vec4 f_color;
void main() {
    float t = u_t;
    float degrade = smoothstep(0.24, 0.55, t) * (1.0 - smoothstep(0.88, 1.0, t));
    float drop = smoothstep(0.52, 0.86, t);
    float bs = exp2(mix(0.0, 5.0, degrade));
    vec2 px = v_uv * u_res;
    vec2 cellid = floor(px / 32.0);
    float smear = step(hash21(cellid + 11.0), 0.12 * degrade);
    vec2 bpx = (floor(px / bs) + 0.5) * bs;
    bpx.y -= smear * fract(hash21(cellid + 3.0) + u_time * 0.4) * 48.0;
    vec2 buv = bpx / u_res;
    float lod = log2(bs);
    vec3 snapc = textureLod(u_snap, buv, lod).rgb;
    vec3 platec = plateLod(buv, lod);
    float sh = textureLod(u_snapHide, buv, max(lod - 1.0, 0.0)).r;
    vec3 bgF = mix(snapc, platec, sh);
    float gone = step(hash21(cellid), drop * 1.02);
    vec3 frozen = mix(snapc, bgF, gone);
    float lv = mix(255.0, 14.0, degrade);
    frozen = floor(frozen * lv + 0.5) / lv;
    float rec = smoothstep(0.9, 1.0, t);
    f_color = vec4(mix(frozen, bgLive(v_uv), rec), 1.0);
}
"""


class Freeze(Effect):
    key = "freeze"
    name = "Zawieszony obraz"
    gesture = "pięść"
    duration = 3.4

    def build(self):
        self.fs = self.r.program(FREEZE_FS)

    def draw(self, t, time):
        self.r.begin_frame()
        self.r.full(self.fs, t=t, time=time, dur=self.duration)


GLASS_VS = glsl.HEADER + """
in vec2 in_pos; in vec2 in_cen; in float in_id; in float in_edge;
uniform vec2 u_impact; uniform float u_R;
out vec2 v_uv; out float v_edge; out float v_glint; out vec2 v_px0;
void main() {
    float h1 = hash11(in_id * 1.7 + 0.3);
    float h2 = hash11(in_id * 7.3 + 1.1);
    float h3 = hash11(in_id * 3.9 + 2.7);
    float h4 = hash11(in_id * 5.1 + 9.2);
    vec2 away = in_cen - u_impact;
    float al = length(away);
    float d = al / u_R;
    away = al > 0.001 ? away / al : vec2(h1 - 0.5, h2 - 0.5);
    float delay = 0.17 + d * 0.3 + h1 * 0.07;
    float lt = max(0.0, u_t - delay) * u_dur;
    vec2 vel = away * (90.0 + 300.0 * max(0.0, 1.0 - d) + 80.0 * h2) + vec2((h3 - 0.5) * 120.0, -(60.0 + 140.0 * h4));
    vec2 off = vel * lt + vec2(0.0, 1700.0) * 0.5 * lt * lt;
    float crackR = smoothstep(0.0, 0.16, u_t) * u_R * 1.6;
    off += away * 1.5 * step(al, crackR) * step(u_t, delay);
    float ang = (h2 - 0.5) * 7.0 * lt;
    float tilt = (h3 - 0.5) * 9.0 * lt;
    vec2 l = in_pos - in_cen;
    float c = cos(ang); float s = sin(ang);
    l = vec2(c * l.x - s * l.y, s * l.x + c * l.y);
    l.x *= cos(tilt);
    vec2 p = in_cen + l + off;
    v_uv = in_pos / u_res;
    v_px0 = in_pos;
    v_edge = in_edge;
    v_glint = pow(abs(sin(tilt + h4 * 6.28)), 12.0) * step(0.001, lt);
    gl_Position = vec4(p / u_res * 2.0 - 1.0, 0.0, 1.0);
}
"""

GLASS_FS = glsl.HEADER + """
in vec2 v_uv; in float v_edge; in float v_glint; in vec2 v_px0;
uniform vec2 u_impact; uniform float u_R;
out vec4 f_color;
void main() {
    vec4 p = person(v_uv);
    float crackR = smoothstep(0.0, 0.16, u_t) * u_R * 1.6;
    float reached = 1.0 - smoothstep(crackR - 8.0, crackR, length(v_px0 - u_impact));
    float w = fwidth(v_edge);
    float line = 1.0 - smoothstep(0.0, w * 1.5, v_edge);
    float shade = 1.0 - smoothstep(w * 1.5, w * 3.5, v_edge);
    vec3 c = p.rgb;
    c = mix(c, c * 0.55, shade * reached * 0.55);
    c = mix(c, vec3(1.0), line * reached * 0.85);
    c += v_glint * 0.45 + 0.05 * reached;
    c += exp(-length(v_px0 - u_impact) / 16.0) * (1.0 - smoothstep(0.0, 0.08, u_t));
    f_color = vec4(c, p.a * (1.0 - smoothstep(0.86, 1.0, u_t)));
}
"""

GLITTER_VS = glsl.HEADER + """
in float in_seed;
uniform vec2 u_impact;
out vec4 v_col;
void main() {
    float h1 = hash11(in_seed * 3.1);
    float h2 = hash11(in_seed * 7.7 + 1.0);
    float h3 = hash11(in_seed * 1.9 + 5.0);
    float age = (u_t - 0.15 - h3 * 0.05) * u_dur;
    if (age <= 0.0) { gl_Position = vec4(-9.0, -9.0, 0.0, 1.0); gl_PointSize = 1.0; v_col = vec4(0.0); return; }
    float a = h1 * 6.2832;
    vec2 dir = vec2(cos(a), sin(a));
    vec2 p = u_impact + dir * mix(200.0, 900.0, h2) * age + vec2(0.0, 1100.0) * age * age;
    float life = 1.0 - smoothstep(0.25, 0.7, age);
    v_col = vec4(vec3(0.95, 0.97, 1.0), 0.85 * life * step(0.5, fract(age * 18.0 + h3)));
    gl_PointSize = mix(1.5, 3.2, h3);
    gl_Position = vec4(p / u_res * 2.0 - 1.0, 0.0, 1.0);
}
"""


def _clip(poly, mid, n):
    out = []
    m = len(poly)
    for k in range(m):
        a = poly[k]
        b = poly[(k + 1) % m]
        da = float(np.dot(a - mid, n))
        db = float(np.dot(b - mid, n))
        if da <= 0:
            out.append(a)
        if (da <= 0) != (db <= 0):
            out.append(a + (b - a) * (da / (da - db)))
    return out


def voronoi(seeds, rect):
    x0, y0, x1, y1 = rect
    cells = []
    for i, s in enumerate(seeds):
        poly = [np.array(p, float) for p in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
        for j, o in enumerate(seeds):
            if i == j:
                continue
            poly = _clip(poly, (s + o) / 2.0, o - s)
            if len(poly) < 3:
                break
        if len(poly) >= 3:
            cells.append(poly)
    return cells


class Glass(Effect):
    key = "glass"
    name = "Rozbita szyba"
    gesture = "pistolet z palców"
    duration = 2.8

    def build(self):
        self.bg = self.r.program(glsl.GONE_FS)
        self.prog = self.ctx.program(vertex_shader=GLASS_VS, fragment_shader=GLASS_FS)
        self.gprog = self.ctx.program(vertex_shader=GLITTER_VS, fragment_shader=glsl.POINT_FS)
        seeds = np.arange(260, dtype="f4")
        self.gvbo = self.ctx.buffer(seeds.tobytes())
        self.gvao = self.ctx.vertex_array(self.gprog, [(self.gvbo, "1f", "in_seed")], skip_errors=True)
        self.vao = None

    def begin(self, info, mask8):
        self.info = info
        x0, y0, x1, y1 = info["box"]
        pad = 0.03
        rect = (max(0, x0 - pad) * W, max(0, y0 - pad) * H, min(1, x1 + pad) * W, min(1, y1 + pad) * H)
        impact = np.array([info["chest"][0] * W, info["chest"][1] * H])
        R = 0.6 * max(rect[2] - rect[0], rect[3] - rect[1])
        rng = np.random.default_rng()
        near = []
        for _ in range(28):
            ang = rng.uniform(0, 2 * math.pi)
            rad = R * rng.uniform(0.02, 1.0) ** 1.7
            near.append(impact + rad * np.array([math.cos(ang), math.sin(ang)]))
        far = [np.array([rng.uniform(rect[0], rect[2]), rng.uniform(rect[1], rect[3])]) for _ in range(30)]
        seeds = [p for p in near + far if rect[0] <= p[0] <= rect[2] and rect[1] <= p[1] <= rect[3]]
        cells = voronoi(seeds, rect)
        verts = []
        for cid, poly in enumerate(cells):
            c = np.mean(poly, axis=0)
            for k in range(len(poly)):
                a = poly[k]
                b = poly[(k + 1) % len(poly)]
                verts += [c[0], c[1], c[0], c[1], cid, 1.0]
                verts += [a[0], a[1], c[0], c[1], cid, 0.0]
                verts += [b[0], b[1], c[0], c[1], cid, 0.0]
        data = np.array(verts, "f4")
        if self.vao is not None:
            self.vao.release()
            self.vbo.release()
        self.vbo = self.ctx.buffer(data.tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, "2f 2f 1f 1f", "in_pos", "in_cen", "in_id", "in_edge")], skip_errors=True)
        self.impact = (float(impact[0]), float(impact[1]))
        self.R = float(R)

    def draw(self, t, time):
        r = self.r
        r.begin_frame()
        r.full(self.bg)
        kw = dict(t=t, time=time, dur=self.duration, u_impact=self.impact, u_R=self.R)
        self.ctx.enable(moderngl.BLEND)
        r.bind(self.prog, **kw)
        self.vao.render(moderngl.TRIANGLES)
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        r.bind(self.gprog, **kw)
        self.gvao.render(moderngl.POINTS)
        self.ctx.disable(moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)


MELT_FS = glsl.HEADER + """
uniform vec4 u_box;
in vec2 v_uv; out vec4 f_color;
void main() {
    vec2 uv = v_uv;
    float Tb = pow(u_t, 2.0) * 1.5;
    float Td = pow(u_t, 1.25) * 1.5;
    float n1 = fbm(vec2(uv.x * 6.0, 2.3));
    float n2 = vnoise(vec2(uv.x * 38.0, 7.7));
    float n3 = vnoise(vec2(uv.x * 95.0, 3.1));
    float drip = pow(clamp(n2 * 1.7 - 0.6, 0.0, 1.0), 2.0) + 0.6 * pow(clamp(n3 * 1.8 - 0.8, 0.0, 1.0), 2.0);
    float top = u_box.y;
    float h = clamp((uv.y - top) / max(1.0 - top, 0.1), 0.0, 1.0);
    float base = (1.0 - top) + 0.08;
    float D = base * (Tb * (0.8 + 0.3 * n1) + Td * drip * 0.9) * (0.85 + 0.45 * h);
    float wob = sin(uv.y * 38.0 - u_time * 4.0 + uv.x * 9.0) * 0.003 * Td;
    vec2 suv = uv - vec2(wob, D);
    vec4 acc = vec4(0.0);
    float sm = 0.03 * Td * (0.25 + drip);
    int n = u_t < 0.02 ? 1 : 5;
    for (int i = 0; i < n; i++) {
        vec4 p = person(suv - vec2(0.0, sm * float(i) / 4.0));
        acc += vec4(p.rgb * p.a, p.a);
    }
    acc /= float(n);
    vec3 pc = acc.a > 0.001 ? acc.rgb / acc.a : vec3(0.0);
    float slope = dFdx(D) * u_res.x;
    float spec = pow(clamp(abs(slope) * 0.02, 0.0, 1.0), 1.5) * min(Td, 1.0);
    float sheen = pow(vnoise(vec2(uv.x * 60.0, (uv.y - D) * 30.0)), 6.0) * min(Td, 1.0);
    float core = pow(clamp(drip, 0.0, 1.0), 0.5) * min(Td * 2.0, 1.0);
    pc = pc * (1.0 - 0.12 * Td - 0.12 * core) + vec3(1.0, 0.97, 0.92) * (0.3 * spec + 0.3 * sheen);
    pc = mix(pc, clamp(pc * pc * 1.3, 0.0, 1.0), 0.3 * min(Td, 1.0));
    float a = acc.a * (1.0 - smoothstep(0.94, 1.0, u_t));
    f_color = vec4(mix(bgLive(uv), pc, a), 1.0);
}
"""


class Melt(Effect):
    key = "melt"
    name = "Roztopienie"
    gesture = "kciuk w dół"
    duration = 3.6

    def build(self):
        self.fs = self.r.program(MELT_FS)

    def draw(self, t, time):
        self.r.begin_frame()
        self.r.full(self.fs, t=t, time=time, dur=self.duration, u_box=self.info["box"])


PLANE_VS = glsl.HEADER + """
in vec2 in_xy;
uniform vec2 u_size; uniform float u_L; uniform float u_k;
uniform float u_a1; uniform float u_a2; uniform float u_b; uniform float u_g; uniform float u_scale;
uniform mat3 u_rot; uniform vec3 u_pivotL; uniform vec3 u_pivotW; uniform vec3 u_trans;
uniform vec2 u_origin; uniform float u_F;
out vec2 v_uv; out vec3 v_n; out vec3 v_view; out vec2 v_local;
vec3 rotA(vec3 p, vec3 o, vec3 ax, float an) { vec3 v = p - o; float c = cos(an); float s = sin(an); return o + v * c + cross(ax, v) * s + ax * dot(ax, v) * (1.0 - c); }
vec3 rotV(vec3 v, vec3 ax, float an) { float c = cos(an); float s = sin(an); return v * c + cross(ax, v) * s + ax * dot(ax, v) * (1.0 - c); }
void main() {
    vec3 p = vec3(in_xy, 0.0);
    vec3 n = vec3(0.0, 0.0, -1.0);
    float hw = u_size.x * 0.5;
    vec3 axL = normalize(vec3(-hw, u_L, 0.0));
    vec3 axR = normalize(vec3(hw, u_L, 0.0));
    if (in_xy.x < 0.0 && in_xy.y < -in_xy.x * (u_L / hw)) { p = rotA(p, vec3(0.0), axL, -u_a1); n = rotV(n, axL, -u_a1); p.z -= 0.8 * smoothstep(2.6, 3.1, u_a1); }
    if (in_xy.x > 0.0 && in_xy.y < in_xy.x * (u_L / hw)) { p = rotA(p, vec3(0.0), axR, u_a2); n = rotV(n, axR, u_a2); p.z -= 0.8 * smoothstep(2.6, 3.1, u_a2); }
    float side = p.x < 0.0 ? -1.0 : 1.0;
    vec3 Y = vec3(0.0, 1.0, 0.0);
    if (abs(p.x) > u_k) { float ga = side < 0.0 ? -u_g : u_g; p = rotA(p, vec3(side * u_k, 0.0, 0.0), Y, ga); n = rotV(n, Y, ga); }
    float ba = side < 0.0 ? u_b : -u_b;
    p = rotA(p, vec3(0.0), Y, ba);
    n = rotV(n, Y, ba);
    vec3 q = u_rot * ((p - u_pivotL) * u_scale);
    vec3 w = u_pivotW + q + u_trans;
    vec2 c = u_res * 0.5;
    float persp = u_F / max(u_F + w.z, 1.0);
    vec2 sp = c + (w.xy - c) * persp;
    v_uv = (u_origin + in_xy) / u_res;
    v_n = u_rot * n;
    v_view = w - vec3(c, -u_F);
    v_local = in_xy;
    gl_Position = vec4(sp / u_res * 2.0 - 1.0, clamp(w.z / 6000.0, -0.99, 0.99), 1.0);
}
"""

PLANE_FS = glsl.HEADER + """
in vec2 v_uv; in vec3 v_n; in vec3 v_view; in vec2 v_local;
uniform float u_print; uniform vec2 u_size;
out vec4 f_color;
void main() {
    vec3 n = normalize(v_n);
    bool front = dot(n, v_view) < 0.0;
    vec3 nn = front ? n : -n;
    vec3 L = normalize(vec3(-0.35, -0.55, -0.75));
    float diff = 0.6 + 0.4 * max(dot(nn, L), 0.0);
    float fiber = (fbm(v_local * 0.09) - 0.5) * 0.04;
    vec3 paper = vec3(0.965, 0.955, 0.935) + fiber;
    vec3 col;
    if (front) {
        vec4 p = person(v_uv);
        vec3 printed = mix(paper, p.rgb * mix(vec3(1.0), paper, 0.08), p.a);
        col = mix(snapAt(v_uv), printed, u_print);
    } else {
        col = paper * 0.93;
    }
    float hw = u_size.x * 0.5;
    float e = min(min(v_local.x + hw, hw - v_local.x), min(v_local.y, u_size.y - v_local.y));
    col *= mix(1.0, 0.8, (1.0 - smoothstep(0.5, 2.0, e)) * u_print);
    col *= mix(1.0, diff, u_print);
    f_color = vec4(col, 1.0);
}
"""


class Plane(Effect):
    key = "plane"
    name = "Papierowy samolot"
    gesture = "V z palców"
    duration = 4.4
    GRID = 48

    def build(self):
        self.bg = self.r.program(glsl.GONE_FS)
        self.prog = self.ctx.program(vertex_shader=PLANE_VS, fragment_shader=PLANE_FS)
        g = self.GRID
        idx = []
        for j in range(g - 1):
            for i in range(g - 1):
                a = j * g + i
                idx += [a, a + 1, a + g, a + 1, a + g + 1, a + g]
        self.ibo = self.ctx.buffer(np.array(idx, "i4").tobytes())
        self.vbo = self.ctx.buffer(reserve=g * g * 8)
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, "2f", "in_xy")], self.ibo, skip_errors=True)

    def begin(self, info, mask8):
        self.info = info
        x0, y0, x1, y1 = info["box"]
        pad = 0.025
        sx0, sx1 = max(0.0, x0 - pad) * W, min(1.0, x1 + pad) * W
        sy0, sy1 = max(0.0, y0 - pad) * H, min(1.0, y1 + 0.0) * H
        w = sx1 - sx0
        h = sy1 - sy0
        g = self.GRID
        xs = np.linspace(-w / 2, w / 2, g)
        ys = np.linspace(0, h, g)
        gx, gy = np.meshgrid(xs, ys)
        self.vbo.write(np.stack([gx, gy], -1).astype("f4").tobytes())
        self.size = (float(w), float(h))
        self.L = float(min(w / 2, h * 0.48))
        self.k = float(w * 0.11)
        self.origin = (float((sx0 + sx1) / 2), float(sy0))
        self.pivotL = (0.0, float(h * 0.45), 0.0)
        self.pivotW = (self.origin[0], self.origin[1] + h * 0.45, 0.0)
        self.dir = float(info.get("dir", info["exit"]))
        q_flip = quat_axis((0, 1, -1), math.pi)
        q_pitch = quat_axis((1, 0, 0), 0.3)
        q_yaw = quat_axis((0, 1, 0), 0.75 * self.dir)
        self.q_final = quat_mul(q_yaw, quat_mul(q_pitch, q_flip))

    def draw(self, t, time):
        r = self.r
        r.begin_frame()
        r.full(self.bg)
        pr = smooth(0.0, 0.12, t)
        a1 = math.pi * smooth(0.12, 0.27, t)
        a2 = math.pi * smooth(0.17, 0.32, t)
        b = 1.3 * smooth(0.34, 0.5, t)
        g = 1.3 * 0.97 * smooth(0.47, 0.62, t)
        scale = 1.0 - 0.38 * smooth(0.1, 0.62, t)
        o = smooth(0.58, 0.78, t)
        lift = smooth(0.03, 0.14, t) * (1.0 - o)
        q = quat_mul(slerp(np.array([1.0, 0, 0, 0]), self.q_final, o), quat_axis((1, 0, 0), -0.25 * lift))
        tau = smooth(0.72, 1.0, t) ** 1.3
        bob = 24.0 * math.sin(tau * 9.0) * tau
        trans = (self.dir * 1300.0 * tau * tau, -280.0 * tau - 200.0 * tau * tau + bob, 2600.0 * tau * tau)
        if tau > 0:
            q = quat_mul(quat_axis((0, 0, 1), 0.25 * math.sin(tau * 7.0) * self.dir), q)
        self.ctx.enable(moderngl.DEPTH_TEST)
        r.bind(
            self.prog, t=t, time=time, dur=self.duration,
            u_size=self.size, u_L=self.L, u_k=self.k, u_a1=a1, u_a2=a2, u_b=b, u_g=g, u_scale=scale,
            u_rot=mat3_uniform(quat_mat(q)), u_pivotL=self.pivotL, u_pivotW=self.pivotW, u_trans=trans,
            u_origin=self.origin, u_F=1400.0, u_print=pr,
        )
        self.vao.render(moderngl.TRIANGLES)
        self.ctx.disable(moderngl.DEPTH_TEST)


HOOK_FS = glsl.HEADER + """
uniform vec2 u_neck; uniform float u_dir; uniform float u_hookX; uniform float u_R;
uniform float u_pull; uniform float u_lagK; uniform float u_vel; uniform float u_lines; uniform vec4 u_box;
in vec2 v_uv; out vec4 f_color;
float sdSeg(vec2 p, vec2 a, vec2 b) { vec2 pa = p - a; vec2 ba = b - a; float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0); return length(pa - ba * h); }
void main() {
    vec2 uv = v_uv;
    vec2 px = uv * u_res;
    float lag = exp(-abs(uv.y - u_neck.y) / max(u_lagK, 0.001));
    float dx = u_pull * lag;
    float blur = u_vel * 0.035 * lag;
    vec4 acc = vec4(0.0);
    int n = u_vel > 0.0 ? 6 : 1;
    for (int i = 0; i < n; i++) {
        float o = blur * float(i) / max(float(n - 1), 1.0);
        vec4 p = person(uv - vec2(u_dir * (dx - o), 0.0));
        acc += vec4(p.rgb * p.a, p.a);
    }
    acc /= float(n);
    vec3 pc = acc.a > 0.001 ? acc.rgb / acc.a : vec3(0.0);
    vec3 col = mix(bgLive(uv), pc, acc.a);
    float row = floor(px.y / 6.0);
    float on = step(0.8, hash11(row * 3.7 + 1.0)) * step(u_box.y, uv.y) * step(uv.y, u_box.w);
    float x0 = u_neck.x * u_res.x - u_dir * hash11(row * 1.3) * 160.0;
    float x1 = (u_neck.x + u_dir * u_pull) * u_res.x;
    float along = (px.x - x0) * u_dir;
    float span = max((x1 - x0) * u_dir, 1.0);
    float streak = on * step(0.0, along) * step(along, span) * smoothstep(0.0, span, along);
    col = mix(col, vec3(1.0), streak * u_lines * 0.55);
    vec2 c = vec2(u_hookX * u_res.x, u_neck.y * u_res.y);
    float R = u_R;
    vec2 top = c + vec2(0.0, -R);
    float dShaft = sdSeg(px, top, top + vec2(u_dir * 3000.0, 0.0));
    vec2 q = px - c;
    float dArc = q.x * u_dir <= 0.0 ? abs(length(q) - R) : min(length(px - top), length(px - (c + vec2(0.0, R))));
    float dTail = sdSeg(px, c + vec2(0.0, R), c + vec2(u_dir * R * 0.45, R * 0.7));
    float d = min(min(dShaft, dArc), dTail);
    float T = 0.19 * R;
    float fill = 1.0 - smoothstep(T - 1.0, T, d);
    float outline = 1.0 - smoothstep(T + 1.5, T + 2.6, d);
    float hi = 1.0 - smoothstep(0.0, T * 0.5, abs(d - T * 0.35));
    vec3 wood = vec3(0.47, 0.30, 0.16) * (0.78 + 0.3 * (1.0 - d / T)) + vec3(0.12, 0.08, 0.04) * hi;
    col = mix(col, vec3(0.10, 0.06, 0.03), outline);
    col = mix(col, wood, fill);
    f_color = vec4(col, 1.0);
}
"""


class Hook(Effect):
    key = "hook"
    name = "Hak zza kulis"
    gesture = "shaka"
    duration = 2.4

    def build(self):
        self.fs = self.r.program(HOOK_FS)

    def begin(self, info, mask8):
        self.info = info
        self.dir = float(info.get("dir", info["exit"]))
        self.neck = info["neck"]
        self.R = float(max(info["head_r"] * W * 0.75, 30.0))

    def draw(self, t, time):
        d = self.dir
        nx = self.neck[0]
        off = 1.0 + 0.15 if d > 0 else -0.15
        enter = ease_out_back(t / 0.28, 1.3)
        hook_x = off + (nx - off) * enter
        tug = 0.012 * math.sin(math.pi * clamp01((t - 0.28) / 0.1))
        u = clamp01((t - 0.38) / 0.34)
        pull = tug + 1.9 * u ** 2.3
        if t > 0.28:
            hook_x = nx + d * pull
        lag_k = 0.08 + 1.6 * u * u
        vel = 1.9 * 2.3 * u ** 1.3 / (0.34 * self.duration) if 0 < u < 1 else 0.0
        lines = smooth(0.38, 0.48, t) * (1.0 - smooth(0.66, 0.85, t))
        self.r.begin_frame()
        self.r.full(
            self.fs, t=t, time=time, dur=self.duration,
            u_neck=self.neck, u_dir=d, u_hookX=hook_x, u_R=self.R, u_pull=pull, u_lagK=lag_k,
            u_vel=vel, u_lines=lines, u_box=self.info["box"],
        )


DASH_FS = glsl.HEADER + """
uniform float u_dir; uniform vec4 u_box;
uniform float u_X[6]; uniform float u_A[6];
uniform float u_sx; uniform float u_lean; uniform float u_vel; uniform float u_lines; uniform vec2 u_shake;
in vec2 v_uv; out vec4 f_color;
vec4 sprite(vec2 uv, float X) {
    float cx = 0.5 * (u_box.x + u_box.z);
    float h = clamp((uv.y - u_box.y) / max(u_box.w - u_box.y, 0.1), 0.0, 1.0);
    float lean = u_lean * (1.0 - h);
    vec2 s = uv;
    s.x = cx + (uv.x - cx - u_dir * X - lean) / u_sx;
    return person(s);
}
void main() {
    vec2 uv = v_uv;
    vec2 px = uv * u_res;
    vec3 col = bgLive(uv + u_shake / u_res);
    for (int k = 5; k >= 1; k--) {
        if (u_A[k] <= 0.0) continue;
        vec4 p = sprite(uv, u_X[k]);
        float lum = dot(p.rgb, vec3(0.3, 0.59, 0.11));
        vec3 tint = mix(p.rgb, vec3(lum) * vec3(0.86, 0.93, 1.05), 0.55);
        col = mix(col, tint, p.a * u_A[k]);
    }
    vec4 acc = vec4(0.0);
    int n = u_vel > 0.0 ? 6 : 1;
    for (int i = 0; i < n; i++) {
        vec4 p = sprite(uv, u_X[0] - u_vel * 0.03 * float(i) / max(float(n - 1), 1.0));
        acc += vec4(p.rgb * p.a, p.a);
    }
    acc /= float(n);
    vec3 pc = acc.a > 0.001 ? acc.rgb / acc.a : vec3(0.0);
    col = mix(col, pc, acc.a * u_A[0]);
    float row = floor(px.y / 5.0);
    float on = step(0.82, hash11(row * 3.7 + 1.0)) * step(u_box.y, uv.y) * step(uv.y, u_box.w);
    float cx = 0.5 * (u_box.x + u_box.z) * u_res.x;
    float x0 = cx - u_dir * hash11(row * 1.3) * 220.0;
    float x1 = cx + u_dir * u_X[0] * u_res.x;
    float along = (px.x - x0) * u_dir;
    float span = max((x1 - x0) * u_dir, 1.0);
    float streak = on * step(0.0, along) * step(along, span) * smoothstep(0.0, span, along);
    col = mix(col, vec3(1.0), streak * u_lines * 0.5);
    f_color = vec4(col, 1.0);
}
"""

DASH_DUST_VS = glsl.HEADER + """
in vec2 in_p; in float in_seed;
uniform float u_dir; uniform float u_spawn;
out vec4 v_col;
void main() {
    float h1 = hash11(in_seed * 3.3);
    float h2 = hash11(in_seed * 9.1 + 2.0);
    float h3 = hash11(in_seed * 5.7 + 7.0);
    float age = (u_t - u_spawn - h3 * 0.06) * u_dur;
    if (age <= 0.0) { gl_Position = vec4(-9.0, -9.0, 0.0, 1.0); gl_PointSize = 1.0; v_col = vec4(0.0); return; }
    float drag = (1.0 - exp(-age * 2.2)) / 2.2;
    vec2 v = vec2(-u_dir * mix(120.0, 560.0, h1), -mix(30.0, 280.0, h2));
    vec2 p = in_p + v * drag + vec2(sin(age * 2.0 + in_seed) * 14.0, -age * 30.0);
    float life = 1.0 - smoothstep(0.2, 1.2 + 0.4 * h2, age);
    vec3 base = plateLod(in_p / u_res, 3.0);
    v_col = vec4(mix(base, vec3(0.86, 0.83, 0.78), 0.55), 0.3 * life * smoothstep(0.0, 0.08, age));
    gl_PointSize = mix(8.0, 46.0, smoothstep(0.0, 1.2, age)) * (0.7 + 0.6 * h1);
    gl_Position = vec4(p / u_res * 2.0 - 1.0, 0.0, 1.0);
}
"""


class Dash(Effect):
    key = "dash"
    name = "Ucieczka z kadru"
    gesture = "kciuk w bok"
    duration = 2.2
    T0 = 0.18
    TL = 0.2

    def build(self):
        self.fs = self.r.program(DASH_FS)
        self.dprog = self.ctx.program(vertex_shader=DASH_DUST_VS, fragment_shader=glsl.SOFT_POINT_FS)
        self.dvbo = self.ctx.buffer(reserve=420 * 12)
        self.dvao = self.ctx.vertex_array(self.dprog, [(self.dvbo, "2f 1f", "in_p", "in_seed")], skip_errors=True)
        self.rng = np.random.default_rng()

    def begin(self, info, mask8):
        self.info = info
        self.dir = float(info.get("dir", info["exit"]))
        x0, y0, x1, y1 = info["box"]
        n = 420
        pts = np.empty((n, 3), "f4")
        pts[:, 0] = self.rng.uniform(x0, x1, n) * W
        pts[:, 1] = self.rng.uniform(max(y0, y1 - 0.2), y1, n) * H
        pts[:, 2] = self.rng.uniform(0, 1000, n)
        self.dvbo.write(pts.tobytes())

    def X(self, t):
        u = clamp01((t - self.T0) / self.TL)
        return 1.8 * u ** 2.4

    def draw(self, t, time):
        u = clamp01((t - self.T0) / self.TL)
        a = smooth(0.0, self.T0, t) * (1.0 - smooth(self.T0, self.T0 + 0.05, t))
        lean = -self.dir * 0.03 * a + self.dir * 0.05 * u * (1.0 - u) * 4.0
        sx = 1.0 - 0.06 * a + 0.7 * math.sin(math.pi * u) * (u < 1)
        X = [self.X(t)]
        A = [1.0 if u < 1 else 0.0]
        fade = 1.0 - smooth(0.42, 0.62, t)
        for k in range(1, 6):
            tk = t - k * 0.016 * self.duration / 2.2
            uk = clamp01((tk - self.T0) / self.TL)
            X.append(self.X(tk))
            A.append(0.4 * (1.0 - k / 6.0) * fade if uk > 0 else 0.0)
        vel = 1.8 * 2.4 * u ** 1.4 / (self.TL * self.duration) if 0 < u < 1 else 0.0
        env = smooth(self.T0, self.T0 + 0.03, t) * (1.0 - smooth(self.T0 + 0.08, self.T0 + 0.3, t))
        q = math.floor(time * 30.0)
        rnd = np.random.default_rng(int(q) + 17)
        shake = tuple(float(v) for v in rnd.uniform(-5.0, 5.0, 2) * env)
        lines = smooth(self.T0, self.T0 + 0.05, t) * (1.0 - smooth(self.T0 + 0.25, self.T0 + 0.5, t))
        r = self.r
        r.begin_frame()
        kw = dict(t=t, time=time, dur=self.duration, u_dir=self.dir, u_box=self.info["box"])
        r.full(self.fs, u_X=X, u_A=A, u_sx=sx, u_lean=lean, u_vel=vel, u_lines=lines, u_shake=shake, **kw)
        self.ctx.enable(moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)
        r.bind(self.dprog, u_spawn=self.T0 + 0.02, **kw)
        self.dvao.render(moderngl.POINTS)
        self.ctx.disable(moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)


SHRINK_FS = glsl.HEADER + """
uniform vec2 u_anchor; uniform float u_scale; uniform float u_safe;
in vec2 v_uv; out vec4 f_color;
vec3 liveDown(vec2 s, float scale) {
    float r = max(0.0, 0.5 / scale - 0.5);
    if (r < 0.25) return textureLod(u_live, s, 0.0).rgb;
    vec2 px = r * 0.9 / u_res;
    vec3 acc = vec3(0.0);
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) acc += textureLod(u_live, s + vec2(float(i), float(j)) * px, 0.0).rgb;
    }
    return acc / 9.0;
}
void main() {
    vec2 uv = v_uv;
    float h = mix(texture(u_hideTight, uv).r, texture(u_hide, uv).r, u_safe);
    vec3 B = mix(liveAt(uv), plateAt(uv), h);
    vec2 s = u_anchor + (uv - u_anchor) / u_scale;
    float a = 0.0;
    vec3 F = vec3(0.0);
    if (inside(s)) {
        float lod = max(log2(1.0 / u_scale), 0.0);
        a = textureLod(u_liveMask, s, max(lod - 1.0, 0.0)).r;
        F = decon(liveDown(s, u_scale), a, plateLod(s, lod));
    }
    f_color = vec4(mix(B, F, a), 1.0);
}
"""


class Shrink(Effect):
    key = "shrink"
    name = "Kurczenie"
    gesture = "szczypta"
    duration = 1.1
    S_MIN = 0.12

    def build(self):
        self.fs = self.r.program(SHRINK_FS)
        self.anchor = (0.5, 1.0)

    def begin(self, info, mask8):
        self.info = info
        self.anchor = info["anchor"]

    def draw_scale(self, scale, safe):
        self.r.begin_frame()
        self.r.full(self.fs, u_anchor=self.anchor, u_scale=float(scale), u_safe=float(safe))

    def draw(self, t, time):
        self.draw_scale(1.0 - (1.0 - self.S_MIN) * t, 1.0)


EFFECTS = [Dust, Glitch, Freeze, Glass, Melt, Plane, Hook, Dash, Shrink]

GESTURE_TO_EFFECT = {
    "palm": "dust",
    "horns": "glitch",
    "fist": "freeze",
    "gun": "glass",
    "thumb_down": "melt",
    "peace": "plane",
    "shaka": "hook",
    "thumb_side": "dash",
    "pinch": "shrink",
}
