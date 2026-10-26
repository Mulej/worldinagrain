// Gathers per-cell emission into a half-resolution texture for the bloom.

struct GlowParams {
    size: vec2<u32>,
    lit: u32,   // 1 in the Lit view
    day: f32,   // time of day, as Params.day
}

@group(0) @binding(0) var<uniform> G: GlowParams;
@group(0) @binding(1) var<storage, read> elems: array<Elem>;
@group(0) @binding(2) var<storage, read> cells: array<vec4<u32>>;
@group(0) @binding(3) var outTex: texture_storage_2d<rgba16float, write>;
// The light reaching each light cell of 2 x 2 cells (visible rgb), and the
// direct sunlight in it (light.wgsl); and per cell the colour render.wgsl
// draws a shiny one in (see shine).
@group(0) @binding(4) var lightTex: texture_2d<f32>;
@group(0) @binding(5) var shineOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var sunTex: texture_2d<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(outTex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    var sum = vec3<f32>(0.0);
    for (var j = 0u; j < 2u; j++) {
        for (var i = 0u; i < 2u; i++) {
            let x = min(gid.x * 2u + i, G.size.x - 1u);
            let y = min(gid.y * 2u + j, G.size.y - 1u);
            sum += emission(unpackCell(cells[y * G.size.x + x]));
        }
    }
    textureStore(outTex, gid.xy, vec4<f32>(sum * 0.25, 1.0));
}

// ------------------------------------------------------------------ shine
//
// A shiny cell (Elem.shine: polished metal) shows the light around it as a
// mirror does, tinted by its colour, worked out here per cell for the view
// drawn, and drawn by render.wgsl. Seen face on, a block of it is a slab
// with its edges rounded off over BEVEL cells, so each edge faces its own
// way and catches the light from there: the top of a steel beam is bright
// with the sky, its underside dark with the ground, a side by lava orange.
// The eye looks straight in (+z, toward it); y is down.
const BEVEL: f32 = 1.75;
// In the Normal view, which has no light worked out: a bright sky above, a
// dark ground below, a key light from the upper left, and broad soft bands
// of a window's reflection across flat faces.
const STUDIO_SKY: vec3<f32> = vec3<f32>(0.92, 0.95, 1.0);
const STUDIO_GROUND: vec3<f32> = vec3<f32>(0.16, 0.15, 0.15);
const STUDIO_KEY: vec3<f32> = vec3<f32>(-0.45, -0.65, 0.61);
// In the Lit view, an edge shows the light this many cells off the way it
// faces.
const MIRROR_REACH: i32 = 5;

@compute @workgroup_size(8, 8)
fn shine(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid.xy >= G.size)) {
        return;
    }
    let c = unpackCell(cells[gid.y * G.size.x + gid.x]);
    if (elems[c.t].shine <= 0.0) {
        return;
    }
    let n = surfaceNormal(vec2<i32>(gid.xy), c.t);
    let g = vec2<f32>(gid.xy) + 0.5;
    var col: vec3<f32>;
    if (G.lit != 0u) {
        col = litShine(c, vec2<i32>(gid.xy), n);
    } else {
        col = studioShine(c, g, n);
    }
    textureStore(shineOut, gid.xy, vec4<f32>(col, 1.0));
}

// Which way the surface of a shiny cell at p of type t faces: face on in
// the middle of its block, tipping toward the nearest point of a cell of
// anything else within BEVEL of its middle. The world's edges aren't the
// block's.
fn surfaceNormal(p: vec2<i32>, t: u32) -> vec3<f32> {
    var near = BEVEL;
    var away = vec2<f32>(0.0);
    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            let q = p + vec2<i32>(dx, dy);
            if (any(q < vec2<i32>(0)) || any(q >= vec2<i32>(G.size)) || wordType(cells[u32(q.y) * G.size.x + u32(q.x)].x) == t) {
                continue;
            }
            let d = vec2<f32>(f32(dx), f32(dy));
            let off = max(abs(d) - 0.5, vec2<f32>(0.0)) * sign(d);
            let dist = length(off);
            if (dist < near) {
                near = dist;
                away = off / dist;
            }
        }
    }
    let tip = 0.9 * (1.0 - near / BEVEL);
    return vec3<f32>(away * tip, sqrt(1.0 - tip * tip));
}

// The mirror's tint: the metal's colour, as bright as it goes.
fn mirrorTint(base: vec3<f32>) -> vec3<f32> {
    return base / max(max(base.r, base.g), max(base.b, 1e-3));
}

// Broad soft bands across flat faces, as of a window's reflection.
fn windowBands(g: vec2<f32>) -> f32 {
    return smoothstep(0.25, 0.75, 0.5 + 0.5 * sin((g.x + 0.6 * g.y) * 0.21));
}

// A shiny cell in the Normal view (see the studio above).
fn studioShine(c: Cell, g: vec2<f32>, n: vec3<f32>) -> vec3<f32> {
    let base = baseColor(c);
    // The way the eye's line of sight leaves the mirror.
    let r = vec3<f32>(2.0 * n.z * n.xy, 2.0 * n.z * n.z - 1.0);
    let face = mix(0.38, 0.78, windowBands(g));
    let env = mix(STUDIO_GROUND, STUDIO_SKY, smoothstep(-0.35, 0.35, -r.y)) * (1.0 - r.z) + face * r.z;
    let key = normalize(STUDIO_KEY);
    let glint = pow(max(dot(n, normalize(key + vec3<f32>(0.0, 0.0, 1.0))), 0.0), 40.0);
    let matte = base * (0.55 + 0.45 * max(dot(n, key), 0.0));
    return mix(matte, mirrorTint(base) * env + glint, elems[c.t].shine);
}

// The light reaching cell p.
fn lightAtCell(p: vec2<i32>) -> vec3<f32> {
    return textureLoad(lightTex, clamp(p / 2, vec2<i32>(0), vec2<i32>(textureDimensions(lightTex)) - 1), 0).rgb;
}

// A shiny cell in the Lit view: what it mirrors of the light around, as the
// eye takes it in (render.wgsl's litColor), what of the light reaching it
// it scatters, and the sun glinting off the edges that face it. Its face
// mirrors the light around its block, its edges what lies the way they
// face.
fn litShine(c: Cell, p: vec2<i32>, n: vec3<f32>) -> vec3<f32> {
    let base = baseColor(c);
    let g = vec2<f32>(p) + 0.5;
    let side = length(n.xy);
    var mirrored = lightAround(g, c.t) * mix(0.55, 1.0, windowBands(g));
    if (side > 0.0) {
        let facing = lightAtCell(p + vec2<i32>(round(n.xy / side * f32(MIRROR_REACH))));
        mirrored = mix(mirrored, facing, min(side * 2.0, 1.0));
    }
    let sun = normalize(vec3<f32>(sunDirection(G.day), 0.2));
    let h = normalize(sun + vec3<f32>(0.0, 0.0, 1.0));
    let sunAt = textureLoad(sunTex, clamp(p / 2, vec2<i32>(0), vec2<i32>(textureDimensions(sunTex)) - 1), 0).x;
    let glint = pow(max(dot(n, h), 0.0), 60.0) * sunAt * sunlight(G.day);
    let matte = base * (1.0 - exp(-EXPOSURE * lightAtCell(p)) + LIT_FLOOR);
    return mix(matte, mirrorTint(base) * (1.0 - exp(-EXPOSURE * mirrored)) + glint, elems[c.t].shine);
}

// The light around the shiny block of type t that g (cells) is in: what
// reaches rings of points around g, from near to far, those out of the
// block only, the nearest ring with any. Deep in a block of metal no light
// gets in, but its face, turned to the eye, mirrors what is in front of the
// world, taken to be lit as what is around it.
fn lightAround(g: vec2<f32>, t: u32) -> vec3<f32> {
    let size = vec2<i32>(G.size);
    for (var reach = 4.0; reach < 70.0; reach *= 2.0) {
        var sum = vec3<f32>(0.0);
        var out = 0.0;
        for (var s = 0u; s < 8u; s++) {
            let a = f32(s) * 0.7853982 + 0.3927;
            let q = vec2<i32>(floor(g + vec2<f32>(cos(a), sin(a)) * reach));
            if (any(q < vec2<i32>(0)) || any(q >= size) || wordType(cells[u32(q.y) * G.size.x + u32(q.x)].x) != t) {
                sum += lightAtCell(q);
                out += 1.0;
            }
        }
        if (out > 0.0) {
            return sum / out;
        }
    }
    return vec3<f32>(0.0);
}
