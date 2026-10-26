// Draws the grid letterboxed into the canvas, plus bloom and the brush outline.

struct View {
    grid: vec2<f32>,       // cells
    origin: vec2<f32>,     // canvas pixels of the grid's top-left corner
    scale: f32,            // canvas pixels per cell
    time: f32,
    mouse: vec2<f32>,      // cells
    lineStart: vec2<f32>,  // cells
    radius: f32,           // cells
    cursor: u32,           // CURSOR_*, or 0 hidden; plus CURSOR_LINE for a line from lineStart
    mode: u32,             // VIEW_*
    electric: u32,         // 1 while currents are being solved (flow is current)
    day: f32,              // time of day, as Params.day
    pad2: u32,
    canvas: vec2<f32>,     // canvas pixels (for figures.wgsl's quads)
}

// The brush's outline (ui.js's cursor): round, square, a spray's (dashed),
// a box from lineStart to the pointer, a fill's cross; a round, square or
// spray brush along a line from lineStart.
const CURSOR_ROUND: u32 = 1u;
const CURSOR_SQUARE: u32 = 2u;
const CURSOR_SPRAY: u32 = 3u;
const CURSOR_BOX: u32 = 4u;
const CURSOR_FILL: u32 = 5u;
const CURSOR_LINE: u32 = 16u;

// VIEW_* come from js/shared.js.

@group(0) @binding(0) var<uniform> V: View;
@group(0) @binding(1) var<storage, read> elems: array<Elem>;
@group(0) @binding(2) var<storage, read> cells: array<vec4<u32>>;
@group(0) @binding(3) var glowTex: texture_2d<f32>;
@group(0) @binding(4) var glowSampler: sampler;
@group(0) @binding(5) var<storage, read> air: array<vec4<f32>>;
// Current density per cell, 2 x f16 (electric.wgsl); only conductors' is kept up.
@group(0) @binding(6) var<storage, read> flow: array<u32>;
// The light reaching each light cell of 2 x 2 cells (light.wgsl): visible
// rgb, sunlight included, in units of full sunlight.
@group(0) @binding(7) var lightTex: texture_2d<f32>;
// The direct sunlight in that, before its colour (light.wgsl's sunOut).
@group(0) @binding(8) var sunTex: texture_2d<f32>;
// The pull of the world's masses per light cell, cells/step² (gravity.wgsl).
@group(0) @binding(9) var gravityTex: texture_2d<f32>;
// Per shiny cell, its colour in the view drawn (glow.wgsl's shine).
@group(0) @binding(10) var shineTex: texture_2d<f32>;

const BORDER: vec3<f32> = vec3<f32>(0.02, 0.022, 0.03);
const NEUTRAL: vec3<f32> = vec3<f32>(0.07, 0.07, 0.1);
const AIR_GLOW: f32 = 0.7;  // how bright air shows the light scattered around it (the sky by day)
const HAZE: f32 = 0.1;      // and the sunbeams through it

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

fn heatColor(t: f32) -> vec3<f32> {
    if (t < AMBIENT_T) {
        let x = clamp((AMBIENT_T - t) / 120.0, 0.0, 1.0);
        return mix(NEUTRAL, vec3<f32>(0.35, 0.75, 1.0), sqrt(x));
    }
    let x = sqrt(clamp((t - AMBIENT_T) / 1600.0, 0.0, 1.0));
    if (x < 0.25) {
        return mix(NEUTRAL, vec3<f32>(0.6, 0.05, 0.12), x / 0.25);
    }
    if (x < 0.5) {
        return mix(vec3<f32>(0.6, 0.05, 0.12), vec3<f32>(0.95, 0.35, 0.05), (x - 0.25) / 0.25);
    }
    if (x < 0.75) {
        return mix(vec3<f32>(0.95, 0.35, 0.05), vec3<f32>(1.0, 0.8, 0.2), (x - 0.5) / 0.25);
    }
    return mix(vec3<f32>(1.0, 0.8, 0.2), vec3<f32>(1.0, 1.0, 0.95), (x - 0.75) / 0.25);
}

fn airAt(g: vec2<f32>) -> vec4<f32> {
    let dims = airDims(vec2<u32>(V.grid));
    let q = g / f32(AIR_CELL) - 0.5;
    let hi = vec2<i32>(dims) - 1;
    let i0 = clamp(vec2<i32>(floor(q)), vec2<i32>(0), hi);
    let i1 = min(i0 + 1, hi);
    let f = clamp(q - vec2<f32>(i0), vec2<f32>(0.0), vec2<f32>(1.0));
    let a00 = air[u32(i0.y) * dims.x + u32(i0.x)];
    let a10 = air[u32(i0.y) * dims.x + u32(i1.x)];
    let a01 = air[u32(i1.y) * dims.x + u32(i0.x)];
    let a11 = air[u32(i1.y) * dims.x + u32(i1.x)];
    return mix(mix(a00, a10, f.x), mix(a01, a11, f.x), f.y);
}

// Red for overpressure, blue for suction, brightened where wind blows.
fn pressureColor(g: vec2<f32>) -> vec3<f32> {
    let a = airAt(g);
    return NEUTRAL
        + vec3<f32>(1.0, 0.35, 0.15) * clamp(a.x / 40.0, 0.0, 1.0)
        + vec3<f32>(0.2, 0.5, 1.0) * clamp(-a.x / 40.0, 0.0, 1.0)
        + vec3<f32>(0.25) * clamp(length(a.yz) / 8.0, 0.0, 1.0);
}

// Air, flames and smoke by their oxygen (in g, the gas plane's): teal
// fresh, orange running low, dark red past what anything can burn in.
// Anything else is dark.
fn oxygenColor(c: Cell, g: Gas) -> vec3<f32> {
    if (!carriesOxygen(c.t)) {
        return NEUTRAL * 0.6;
    }
    let o = oxygenOf(c.t, g);
    let low = clamp((1.0 - o) / (1.0 - SMOTHER), 0.0, 1.0);
    let col = mix(vec3<f32>(0.08, 0.3, 0.34), vec3<f32>(0.95, 0.55, 0.1), low);
    return mix(col, vec3<f32>(0.4, 0.04, 0.18), clamp((SMOTHER - o) / SMOTHER * 2.0, 0.0, 1.0));
}

// Air (any gas that holds vapour) by its relative humidity, dry dark to
// saturated blue, mist white; water itself, damp powder darker.
fn humidityColor(c: Cell, g: Gas) -> vec3<f32> {
    if (holdsVapour(c.t)) {
        let rh = clamp(g.vapour / saturation(c.temp), 0.0, 1.0);
        let col = mix(NEUTRAL, vec3<f32>(0.12, 0.4, 0.9), rh);
        return mix(col, MIST, clamp(g.mist / MIST_THICK * 2.0, 0.0, 1.0));
    }
    let e = elems[c.t];
    // Water by how much it carries: dark when fresh, the solute's own colour
    // when saturated.
    if (holdsSolute(c.t)) {
        let s = elems[soluteOf(c)];
        let tint = select(vec3<f32>(0.0), vec3<f32>(s.solR, s.solG, s.solB) * 1.3, soluteOf(c) != 0u);
        return mix(vec3<f32>(0.04, 0.08, 0.2), tint, shareOf(strengthOf(c)));
    }
    if (e.kind == KIND_LIQUID) {
        return baseColor(c) * 0.8;
    }
    if (holdsSoilWater(c.t)) {
        return mix(NEUTRAL * 0.6, vec3<f32>(0.1, 0.3, 0.7), shareOf(soilWaterOf(c)));
    }
    return NEUTRAL * 0.6;
}

// The light reaching the world at g (cells), visible rgb.
fn lightAt(g: vec2<f32>) -> vec3<f32> {
    let uv = g / (2.0 * vec2<f32>(textureDimensions(lightTex)));
    return textureSampleLevel(lightTex, glowSampler, uv, 0.0).rgb;
}

// The Lit view: everything in the light reaching it, as the eye adapts to
// it, and what glows on top, so caves and nights are dark and flames and
// lava light up what is around them. Air shows the light scattered around
// it, faintly, and a little of the sunbeams through it: blue under the open
// sky by day, dark at night, warm around a fire; steam and mist, lit, as
// haze and fog (gas, what it holds of the air).
// (c is the cell at `at`, which a liquid cell's top drawn as the cell above
// it isn't g's.)
fn litColor(c: Cell, g: vec2<f32>, at: vec2<f32>, current: vec2<f32>, gas: Gas) -> vec3<f32> {
    let light = lightAt(g);
    let seen = 1.0 - exp(-EXPOSURE * light);
    if (c.t == AIR) {
        let direct = sunlight(V.day) * sunAt(g);
        let haze = 1.0 - exp(-EXPOSURE * (max(light - direct, vec3<f32>(0.0)) + HAZE * direct));
        let air = mix(BG * 0.3 + haze * AIR_GLOW, steamColor() * seen, steamHaze(gas));
        let fog = clamp(gas.mist / MIST_THICK, 0.0, 0.85);
        return mix(air, MIST * seen, fog) + emission(c);
    }
    // A shiny cell shows what it mirrors of the light around it, as lit
    // already (glow.wgsl's shine).
    var base = baseColor(c);
    var lit = seen + LIT_FLOOR;
    if (elems[c.t].shine > 0.0) {
        base = textureLoad(shineTex, vec2<i32>(at), 0).rgb;
        lit = vec3<f32>(1.0);
    }
    return cellColor(c, at, V.time, current, base, gas) * lit + emission(c);
}

// The pull of the world's masses at g (cells), Earth's left out: its
// direction as a hue round the colour wheel, its strength as brightness on a
// log scale, from a thousandth of Earth's g to ten g.
fn gravityColor(g: vec2<f32>) -> vec3<f32> {
    let size = vec2<i32>(textureDimensions(gravityTex));
    let pull = textureLoad(gravityTex, clamp(vec2<i32>(g * 0.5), vec2<i32>(0), size - 1), 0).xy;
    let strength = length(pull) / elems[WATER].weight;
    let level = clamp((log2(max(strength, 1e-9)) * 0.30103 + 3.0) / 4.0, 0.0, 1.0);
    let angle = atan2(pull.y, pull.x);
    let hue = 0.5 + 0.5 * cos(angle + vec3<f32>(0.0, 2.094, 4.189));
    return NEUTRAL + hue * level;
}

// The direct sunlight at g (cells), interpolated as lightTex is (sunTex
// isn't filterable).
fn sunAt(g: vec2<f32>) -> f32 {
    let size = vec2<i32>(textureDimensions(sunTex));
    let q = g * 0.5 - 0.5;
    let i = vec2<i32>(floor(q));
    let f = q - floor(q);
    let a = textureLoad(sunTex, clamp(i, vec2<i32>(0), size - 1), 0).x;
    let b = textureLoad(sunTex, clamp(i + vec2<i32>(1, 0), vec2<i32>(0), size - 1), 0).x;
    let c = textureLoad(sunTex, clamp(i + vec2<i32>(0, 1), vec2<i32>(0), size - 1), 0).x;
    let d = textureLoad(sunTex, clamp(i + vec2<i32>(1, 1), vec2<i32>(0), size - 1), 0).x;
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

fn segmentDistance(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let ab = b - a;
    let h = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0);
    return length(p - a - ab * h);
}

// The same measured square (the larger of across and down): the least over
// the segment, which is convex along it, found by ternary search.
fn squareDistance(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    var lo = 0.0;
    var hi = 1.0;
    for (var k = 0; k < 16; k++) {
        let t0 = mix(lo, hi, 0.382);
        let t1 = mix(lo, hi, 0.618);
        let d0 = abs(p - mix(a, b, t0));
        let d1 = abs(p - mix(a, b, t1));
        if (max(d0.x, d0.y) < max(d1.x, d1.y)) {
            hi = t1;
        } else {
            lo = t0;
        }
    }
    let d = abs(p - mix(a, b, (lo + hi) * 0.5));
    return max(d.x, d.y);
}

// How far, in cells, g is from the brush's outline (View.cursor).
fn cursorDistance(g: vec2<f32>) -> f32 {
    let kind = V.cursor & 15u;
    let start = select(V.mouse, V.lineStart, (V.cursor & CURSOR_LINE) != 0u);
    switch kind {
        case CURSOR_SQUARE: {
            return abs(squareDistance(g, start, V.mouse) - V.radius);
        }
        case CURSOR_BOX: {
            let lo = min(V.lineStart, V.mouse);
            let hi = max(V.lineStart, V.mouse);
            return abs(max(max(lo.x - g.x, g.x - hi.x), max(lo.y - g.y, g.y - hi.y)));
        }
        case CURSOR_FILL: {
            let d = abs(g - V.mouse);
            return select(1e9, min(d.x, d.y), max(d.x, d.y) < 3.0);
        }
        default: {
            let d = segmentDistance(g, start, V.mouse);
            // A spray's outline dashed.
            let around = atan2(g.y - V.mouse.y, g.x - V.mouse.x);
            if (kind == CURSOR_SPRAY && fract(around * 12.0 / 6.2831853) < 0.5) {
                return 1e9;
            }
            return abs(d - V.radius);
        }
    }
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let g = (pos.xy - V.origin) / V.scale;
    var col = BORDER;
    if (all(g >= vec2<f32>(0.0)) && all(g < V.grid)) {
        var ci = vec2<u32>(g);
        var i = ci.y * u32(V.grid.x) + ci.x;
        var c = unpackCell(cells[i]);
        // A cell of liquid holding less than a cell (liquidAmount: at its
        // surface) is drawn filled from the bottom that far, and over that
        // as the cell above it, so the surface stands at its height within
        // the cell. (ci that cell then: a steel lid over the water took its
        // shine from the water's texel, which the shine pass never writes.)
        if (elems[c.t].kind == KIND_LIQUID && ci.y > 0u && fract(g.y) < 1.0 - liquidAmount(c)) {
            i -= u32(V.grid.x);
            ci.y -= 1u;
            c = unpackCell(cells[i]);
        }
        // What it holds of the air (the gas plane, after the cells).
        var gas = noGas();
        if (holdsGas(c.t)) {
            gas = gasFrom(cells[gasIndex(u32(V.grid.x) * u32(V.grid.y), i)], i);
        }
        var current = vec2<f32>(0.0);
        if (V.electric != 0u && elems[c.t].sigma > 0.0) {
            current = unpack2x16float(flow[i]);
        }
        if (V.mode == VIEW_HEAT) {
            col = heatColor(tempOf(c));
        } else if (V.mode == VIEW_PRESSURE) {
            col = pressureColor(g);
        } else if (V.mode == VIEW_OXYGEN) {
            col = oxygenColor(c, gas);
        } else if (V.mode == VIEW_HUMIDITY) {
            col = humidityColor(c, gas);
        } else if (V.mode == VIEW_GRAVITY) {
            col = gravityColor(g);
        } else if (V.mode == VIEW_LIT) {
            col = litColor(c, g, vec2<f32>(ci), current, gas) + textureSampleLevel(glowTex, glowSampler, g / V.grid, 0.0).rgb * 0.9;
        } else {
            var base = baseColor(c);
            if (elems[c.t].shine > 0.0) {
                base = textureLoad(shineTex, vec2<i32>(ci), 0).rgb;
            }
            col = cellColor(c, vec2<f32>(ci), V.time, current, base, gas);
            col += textureSampleLevel(glowTex, glowSampler, g / V.grid, 0.0).rgb * 0.9;
        }
        if (V.mode != 0u && V.mode != VIEW_LIT && c.t != AIR) {
            col = mix(col, baseColor(c), 0.2);
        }
    }
    // (Only near the pointer, or along a line or box being drawn: as far as
    // a square brush's corners.)
    let reach = V.radius * 1.4143 + 4.0 + length(V.mouse - V.lineStart) * select(0.0, 1.0, V.cursor > CURSOR_FILL || (V.cursor & 15u) == CURSOR_BOX);
    if (V.cursor != 0u && length(g - V.mouse) <= reach) {
        let px = cursorDistance(g) * V.scale;
        col = mix(col, vec3<f32>(1.0), (1.0 - smoothstep(0.6, 1.6, px)) * 0.5);
    }
    return vec4<f32>(col, 1.0);
}
