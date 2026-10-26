// Light: visible light (sunlight and skylight, flames, glowing matter)
// worked out every few steps (sim.js LIGHT_EVERY) over light cells of 2 x 2
// cells. The Lit view shows it (render.wgsl), plants grow by it
// (update.wgsl, block.wgsl), and the sun's light heats what it falls on
// (update.wgsl). (Radiant heat, the thermal infrared, is radiate.wgsl's,
// carried conservatively cell by cell: REWRITE_PLAN.md stage 6.)
//
// Each cell lets through some share of the light (elements.js extinction;
// cellClarity in common.wgsl for a cell of the world's size) and gives off
// light where it doesn't: its glow and what it scatters of the light around
// it, in its colour. A ray crossing a cell picks up (1 - clear) of what the
// cell gives off and keeps clear of what it brought.
//
// Light from everywhere, with shadows: radiance cascades (A. Sannikov,
// "Radiance Cascades: A Novel Approach to Calculating Global Illumination",
// 2023). Light from far away comes through a hierarchy of probes: cascade k
// has a probe per 2^k x 2^k light cells, each casting 4 x 2^k rays evenly
// spread around it, over the distances from 2^k - 1/2 to 2^(k+1) - 1/2 light
// cells. A shadow softens with the distance to what casts it (its penumbra
// widens in proportion), so further away fewer probes, each looking in more
// directions, see enough: twice as far, half as many probes a side and twice
// the directions, which keeps every cascade's work the same (the classic
// scheme quadruples the directions and the distances instead, which takes
// more rays further out). Starting with the farthest, each cascade's rays
// carry on with what the next one up found beyond them along the two
// directions that split theirs, from its probes around that are out in the
// open (a probe inside a wall casts no rays: see openProbe), each reached
// by a ray of its own so nothing between is skipped (see castRay).
// The lowest cascade, from each light cell to the cells around it, gives
// the light reaching that cell from each of four quarters. Rays leaving the
// world through its top see the sky; through its bottom, the ground; and
// past its sides the world is taken to go on as its edge does.
//
// Direct sunlight is worked out apart, as a beam swept across the world: a
// sun that small casts sharp shadows, which the cascades would blur. So is
// the sunlight a mirror sends on (bounce), as a beam of its own.
//
// Passes, every LIGHT_EVERY steps:
//   build    per light cell what it lets through, sideways and up or down
//            (each crossing two of its cells), and what it gives off, from its
//            cells and the light last worked out
//   sun      the sunlight reaching each light cell: one workgroup sweeping
//            the world a row at a time (with the sun low, a column)
//   bounce   the sunlight mirrors throw on: a thread per light cell, those
//            of a sunlit mirror sending its beam on
//   cascade  each cascade from the farthest down, one dispatch each, a
//            thread per probe and direction; cascade 0 a thread per light
//            cell and direction of cascade 1, what lies past the cells
//            around it (see farther)
//   gather   the cells around each light cell and cascade 0 beyond them:
//            the light reaching it (for the Lit view, plants and what it
//            reflects), and its share of sunlight for update.wgsl to hand
//            out to its cells

struct Cascade {
    level: u32,  // k
    count: u32,  // cascades there are, 0 to count - 1
    pad: vec2<u32>,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> elems: array<Elem>;
@group(0) @binding(2) var<storage, read> cells: array<vec4<u32>>;
// Per light cell (build): what it gives off where it isn't clear (visible
// rgb, 3 x f16, and in the fourth whether it is a mirror, see Medium), the
// shares it stops crossing it sideways and up or down (2 x f16: what it
// lets through is 1 less them, kept so a nearly clear cell's is exact; in 8
// bits a light cell of water let through 0.996 for its 0.998 and took in
// twice what it should), and how much light its cells take in between them
// (f16): each cell 1 - its clarity.
@group(0) @binding(3) var sceneOut: texture_storage_2d<rgba32uint, write>;
@group(0) @binding(4) var scene: texture_2d<u32>;
// The light as last worked out (gather's lightOut).
@group(0) @binding(5) var lightIn: texture_2d<f32>;
// Direct sunlight reaching each light cell, through a face square to the
// sweep: from above, or with the sun low, from the side. 1 is full sunlight
// on a surface square to the sun.
@group(0) @binding(6) var sunOut: texture_storage_2d<r32float, write>;
@group(0) @binding(7) var sunIn: texture_2d<f32>;
// Cascades, each laid out direction by direction, a block of its probes per
// direction side by side: the visible rgb gathered along each ray.
@group(0) @binding(8) var upper: texture_2d<f32>;
@group(0) @binding(9) var cascadeOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(11) var<uniform> C: Cascade;
// The light reaching each light cell through its open sides (visible rgb,
// sunlight included; see openLight), and for update.wgsl the share of the
// sunlight it takes in each of its cells gets per unit it takes in.
@group(0) @binding(12) var lightOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(13) var heatOut: texture_storage_2d<r32float, write>;
// Mirrors (see specular). Per light cell (gather): the visible light
// reaching it from each quarter of directions (right, down, left, up; see
// quarter), the direct sunlight in the one it comes from. And per light
// cell (build): what its shiny cells show, as a mirror, a ray going each of
// those ways (as glow is: where it isn't clear). Packed (packLight).
@group(0) @binding(17) var quartersIn: texture_2d<u32>;
@group(0) @binding(18) var mirrorOut: texture_storage_2d<rgba32uint, write>;
@group(0) @binding(19) var mirrorIn: texture_2d<u32>;
@group(0) @binding(20) var quartersOut: texture_storage_2d<rgba32uint, write>;
// Per light cell, the sunlight mirrors throw on it (bounce), as direct
// sunlight is, in BOUNCE_SCALEths; gather takes it and clears it.
@group(0) @binding(21) var<storage, read_write> bounced: array<atomic<u32>>;
// Where people's bodies are (common.wgsl's occupantWord): opaque, so they
// cast shadows (REWRITE_PLAN.md 10-1). (Their sprites are drawn over the
// world, figures.wgsl.)
@group(0) @binding(22) var<storage, read> occupants: array<u32>;

// What a person's body gives back of the light on it, skin and clothes.
const PERSON_ALBEDO: vec3<f32> = vec3<f32>(0.55, 0.45, 0.38);

const PI: f32 = 3.1415927;
// Glow (shade.wgsl's emission, which also feeds the bloom) as a light source.
const GLOW: f32 = 1.5;
// A ray stops once less than this share of the light gets through.
const OPAQUE: f32 = 0.002;
// A light cell whose cells reflect as mirrors more than this share of the
// light they stop (its shine, by its luminance) is a mirror (Medium).
const MIRROR_SHINE: f32 = 0.3;

fn lightSize() -> vec2<u32> {
    return (P.size + 1u) / 2u;
}

fn probes(k: u32) -> vec2<u32> {
    return (lightSize() + (1u << k) - 1u) >> vec2<u32>(k);
}

fn directions(k: u32) -> u32 {
    return 4u << k;
}

// Direction a of cascade k, in the middle of its share of the circle, which
// the next cascade up splits in two: its directions 2a and 2a + 1.
fn direction(a: u32, k: u32) -> vec2<f32> {
    let angle = (f32(a) + 0.5) * 2.0 * PI / f32(directions(k));
    return vec2<f32>(cos(angle), sin(angle));
}

// Where cascade k's rays start, in light cells from their probe.
fn reach(k: u32) -> f32 {
    return f32(1u << k) - 0.5;
}

// The light arriving from the open sky, and from the ground under the world
// (none).
fn skyGlow() -> vec3<f32> {
    return skylight(P.day);
}

fn groundGlow() -> vec3<f32> {
    return vec3<f32>(0.0);
}

struct Medium {
    glow: vec3<f32>,  // given off where it isn't clear
    across: f32,      // share let through crossing it sideways
    down: f32,        // crossing it up or down
    absorb: f32,      // its cells' 1 - clarity, summed
    mirror: bool,     // shiny: it reflects as a mirror (specular, bounce)
}

fn medium(q: vec2<i32>) -> Medium {
    let v = textureLoad(scene, q, 0);
    var m: Medium;
    let blue = unpack2x16float(v.y);
    m.glow = vec3<f32>(unpack2x16float(v.x), blue.x);
    m.mirror = blue.y > 0.0;
    let stop = unpack2x16float(v.z);
    m.across = 1.0 - stop.x;
    m.down = 1.0 - stop.y;
    m.absorb = unpack2x16float(v.w).x;
    return m;
}

// Share let through along d (a unit vector): crossing two cells sideways, up
// or down, or in between.
fn clearAlong(m: Medium, d: vec2<f32>) -> f32 {
    return mix(m.down, m.across, abs(d.x) / (abs(d.x) + abs(d.y)));
}

// The share of the sun's beam (going along v) a light cell of m lets
// through from one line of the sweep to the next (sweepSun): its path there
// is 1 / flux of a light cell, flux the larger of v's components, and
// Beer-Lambert takes clearAlong to that power. (One light cell's a line,
// the sun at 45° crossed fog at 1 / √2 of its depth: 10 light cells of
// thick fog let through 1.2 % of the beam, not 0.18 %.)
fn sunClear(m: Medium, v: vec2<f32>) -> f32 {
    return pow(clearAlong(m, v), 1.0 / max(abs(v.x), abs(v.y)));
}

// Visible light as one word, 8 bits of each of rgb by its square root, up
// to LIGHT_TOP (fine steps in the dark, a cave's dim light, and coarser in
// the sun).
const LIGHT_TOP: f32 = 16.0;

fn packLight(c: vec3<f32>) -> u32 {
    return pack4x8unorm(vec4<f32>(sqrt(clamp(c / LIGHT_TOP, vec3<f32>(0.0), vec3<f32>(1.0))), 0.0));
}

fn unpackLight(v: u32) -> vec3<f32> {
    let s = unpack4x8unorm(v).rgb;
    return s * s * LIGHT_TOP;
}

// The quarter of directions d is in: 0 right, 1 down, 2 left, 3 up.
fn quarterOf(d: vec2<f32>) -> u32 {
    if (abs(d.x) >= abs(d.y)) {
        return select(2u, 0u, d.x > 0.0);
    }
    return select(3u, 1u, d.y > 0.0);
}

// What a shiny light cell at q shows a ray going along d, on top of the
// light it scatters every way (its glow): as a mirror, the light reaching
// it from the way the ray leaves it (see mirrorView). Polished metal turned
// to the sun throws its light on what lies the other way, into a shaded
// corner or a cave's mouth.
fn specular(q: vec2<i32>, d: vec2<f32>) -> vec3<f32> {
    return unpackLight(textureLoad(mirrorIn, q, 0)[quarterOf(d)]);
}

// What a light cell whose shiny cells reflect `shine` of the light (as glow
// is: where it isn't clear), its surface facing n, shows as a mirror a ray
// going each of the four ways (right, down, left, up; straight that way):
// the light reaching it from the quarter of directions around the way the
// ray leaves it (gather), none from behind.
fn mirrorView(q: vec2<u32>, shine: vec3<f32>, n: vec2<f32>) -> vec4<u32> {
    let around = textureLoad(quartersIn, q, 0);
    var out = vec4<u32>(0u);
    for (var b = 0u; b < 4u; b++) {
        let d = select(vec2<f32>(0.0, 2.0 - f32(b)), vec2<f32>(1.0 - f32(b), 0.0), (b & 1u) == 0u);
        let along = dot(d, n);
        if (along < 0.0) {
            out[b] = packLight(shine * unpackLight(around[quarterOf(d - 2.0 * along * n)]));
        }
    }
    return out;
}

// ------------------------------------------------------------------ build

// What one cell adds to its light cell, given the light reaching it (lit):
// the visible light it gives off (glowing and scattering) times how much it
// isn't clear, what it reflects as a mirror (shine; see specular), that,
// and how clear it is.
struct Part {
    visible: vec3<f32>,
    shine: vec3<f32>,
    absorb: f32,
    clear: f32,
}

fn part(q: vec2<u32>, lit: vec4<f32>) -> Part {
    let p = min(q, P.size - 1u);
    let i = p.y * P.size.x + p.x;
    if (isOccupant(occupants[i])) {
        return Part(PERSON_ALBEDO * lit.rgb, vec3<f32>(0.0), 1.0, 0.0);
    }
    let c = unpackCell(cells[i]);
    let e = elems[c.t];
    // A gas's mist and vapour (the gas plane, after the cells) dim what passes.
    var g = noGas();
    if (holdsVapour(c.t)) {
        g = gasFrom(cells[gasIndex(P.size.x * P.size.y, i)], i);
    }
    let clear = clarityOf(c, g, P.dx).x;
    var r: Part;
    r.absorb = 1.0 - clear;
    r.visible = (emission(c) * GLOW + albedo(c.t) * (1.0 - e.shine) * lit.rgb) * r.absorb;
    r.shine = albedo(c.t) * e.shine * r.absorb;
    r.clear = clear;
    return r;
}

@compute @workgroup_size(8, 8)
fn build(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid.xy >= lightSize())) {
        return;
    }
    let lit = textureLoad(lightIn, gid.xy, 0);
    let o = gid.xy * 2u;
    let a = part(o, lit);
    let b = part(o + vec2<u32>(1u, 0u), lit);
    let c = part(o + vec2<u32>(0u, 1u), lit);
    let d = part(o + vec2<u32>(1u, 1u), lit);
    // Crossing sideways goes through a row of two cells, either row; up or
    // down, a column.
    let across = (a.clear * b.clear + c.clear * d.clear) * 0.5;
    let down = (a.clear * c.clear + b.clear * d.clear) * 0.5;
    let absorb = a.absorb + b.absorb + c.absorb + d.absorb;
    let glow = (a.visible + b.visible + c.visible + d.visible) / max(absorb, 1e-6);
    let shine = a.shine + b.shine + c.shine + d.shine;
    // A mirror when shine is most of what its cells stop (metal, scrap,
    // molten metal: they reflect as mirrors, the sun's beam too).
    let shiny = dot(shine, vec3<f32>(0.2126, 0.7152, 0.0722)) > MIRROR_SHINE * absorb;
    textureStore(sceneOut, gid.xy, vec4<u32>(
        pack2x16float(glow.rg),
        pack2x16float(vec2<f32>(glow.b, select(0.0, 1.0, shiny))),
        pack2x16float(vec2<f32>(1.0 - across, 1.0 - down)),
        pack2x16float(vec2<f32>(absorb, 0.0)),
    ));
    var mirror = vec4<u32>(0u);
    if (any(shine > vec3<f32>(0.0))) {
        mirror = mirrorView(gid.xy, shine / max(absorb, 1e-6), facing(o));
    }
    textureStore(mirrorOut, gid.xy, mirror);
}

// Which way the surface in the light cell whose first cell is o faces: away
// from it, toward the cells around it that let light through.
fn facing(o: vec2<u32>) -> vec2<f32> {
    var out = vec2<f32>(0.0);
    for (var y = -1; y <= 2; y++) {
        for (var x = -1; x <= 2; x++) {
            let q = clamp(vec2<i32>(o) + vec2<i32>(x, y), vec2<i32>(0), vec2<i32>(P.size) - 1);
            let t = wordType(cells[u32(q.y) * P.size.x + u32(q.x)].x);
            out += select(elems[t].clarity, 1.0, t == AIR) * (vec2<f32>(f32(x), f32(y)) - 0.5);
        }
    }
    return out / max(length(out), 1e-6);
}

// -------------------------------------------------------------------- sun
//
// The beam is swept along lines across the world, rows from the top down or
// with the sun low, columns from its side, the light leaving each line
// falling on the next one shifted by up to a cell: each light cell takes it
// from the two cells of the line before nearest to where it came from,
// between them in proportion. What each lets through goes on, so all the
// light is accounted for and none is counted twice, and a flat floor under a
// low sun gets as little of it per cell as it should. (It spreads a little
// sideways as it goes, so shadows far from what casts them are soft.)
//
// One workgroup does it all, a line after another, so it takes a while
// however little work each line is; nothing after it but gather needs what
// it finds, so the GPU can work on the farthest cascade alongside. Each of
// its threads takes a run of the cells along each line, worked through in
// place from its far end, the cell before its run taken from the thread
// before it, as that cell was after the last line (edges, one set for each
// line by turns). How clear each cell is comes in for a band of lines at a
// time, all at once (band, two cells a word, each the share it stops as
// f16, as build keeps it), so the threads don't wait on memory line after
// line.

const SWEEP: u32 = 256u;        // threads
const SWEEP_MAX: u32 = 2048u;   // light cells along a line, at most
const BAND_WORDS: u32 = 1536u;
var<workgroup> beam: array<f32, 2048>;
var<workgroup> edges: array<f32, 512>;
var<workgroup> band: array<u32, 1536>;

// The light cell at step i along line `line` of the sweep.
fn sweepCell(line: u32, i: u32, steep: bool, mirror: bool) -> vec2<u32> {
    var q = select(vec2<u32>(line, i), vec2<u32>(i, line), steep);
    if (mirror) {
        q.x = lightSize().x - 1u - q.x;
    }
    return q;
}

fn sweepSun(t: u32) {
    // Uniform: there is no sunlight to sweep at night.
    if (all(sunlight(P.day) <= vec3<f32>(0.0))) {
        return;
    }
    let size = lightSize();
    let v = -sunDirection(P.day);  // the way the light goes
    let steep = abs(v.y) >= abs(v.x);
    // Lines are rows, or columns; along a line the light shifts by `shift`
    // cells per line, towards higher indices along it. With the sun on the
    // right, rows are read right to left, and columns are taken from the
    // right.
    let lines = select(size.x, size.y, steep);
    let long = min(select(size.y, size.x, steep), SWEEP_MAX);
    let mirror = v.x < 0.0;
    let shift = clamp(select(v.y / abs(v.x), abs(v.x) / v.y, steep), 0.0, 1.0);
    let flux = select(abs(v.x), v.y, steep);
    let run = (long + SWEEP - 1u) / SWEEP;
    let first = min(t * run, long);
    let last = min(first + run, long);
    let words = (long + 1u) / 2u;
    let bandLines = BAND_WORDS / words;

    if (steep) {
        for (var i = first; i < last; i++) {
            beam[i] = 1.0;
        }
    } else {
        // Light coming in from the side has crossed the world beyond its
        // edge, taken to go on as the edge column does: 1 / shift of its
        // cells in each row on the way down, the light at each row what is
        // left past all the rows above. Each thread multiplies through its
        // run, then takes in those of the threads before it.
        let x = select(0u, size.x - 1u, mirror);
        let crossed = 1.0 / max(shift, 1.0 / 64.0);
        var through = 1.0;
        for (var i = first; i < last; i++) {
            through *= pow(sunClear(medium(vec2<i32>(vec2<u32>(x, i))), v), crossed);
            beam[i] = through;
        }
        edges[t] = through;
        workgroupBarrier();
        var before = 1.0;
        for (var u = 0u; u < t; u++) {
            before *= edges[u];
        }
        for (var i = first; i < last; i++) {
            beam[i] *= before;
        }
    }
    workgroupBarrier();
    if (first > 0u) {
        edges[t] = beam[first - 1u];
    }

    for (var line0 = 0u; line0 < lines; line0 += bandLines) {
        let count = min(bandLines, lines - line0);
        for (var w = t; w < count * words; w += SWEEP) {
            let line = line0 + w / words;
            let i = (w % words) * 2u;
            var stop = vec2<f32>(0.0);
            for (var j = 0u; j < 2u; j++) {
                if (i + j < long) {
                    stop[j] = 1.0 - sunClear(medium(vec2<i32>(sweepCell(line, i + j, steep, mirror))), v);
                }
            }
            band[w] = pack2x16float(stop);
        }
        workgroupBarrier();
        for (var j = 0u; j < count; j++) {
            let line = line0 + j;
            let parity = (line & 1u) * SWEEP;
            // The cell before this thread's run, as it left the last line.
            // The first run's has none: past the side of the world in a row
            // it is the edge cell itself, above the world in a column it is
            // the sky.
            var before = select(1.0, beam[0], steep);
            if (first > 0u) {
                before = edges[parity + t];
            }
            for (var n = last; n > first; n--) {
                let i = n - 1u;
                let prev = select(beam[max(i, 1u) - 1u], before, i == first);
                let into = mix(beam[i], prev, shift);
                textureStore(sunOut, sweepCell(line, i, steep, mirror), vec4<f32>(into * flux, 0.0, 0.0, 0.0));
                beam[i] = into * (1.0 - unpack2x16float(band[j * words + i / 2u])[i & 1u]);
            }
            if (last > first && t + 1u < SWEEP) {
                edges[(SWEEP - parity) + t + 1u] = beam[last - 1u];
            }
            workgroupBarrier();
        }
    }
}

// ----------------------------------------------------------------- bounce
//
// A mirror in the sun sends its beam on, the way the sun's light leaves it
// (its surface facing as build's facing has it): each sunlit light cell of
// shiny cells, a beam as wide as the sunlight it catches, as strong as it
// reflects of it, on through the light cells along that way until something
// stops it, each of them taking what reaches it as direct sunlight (gather).
// So a steel plate turned to the sun lights a dark room through its door
// as the sun would, and a curved one brings the sun to a point. (Scattered
// light from mirrors goes with the cascades, see specular.)
const BOUNCE_SCALE: f32 = 65536.0;
const BOUNCE_REACH: u32 = 1024u;  // light cells a beam goes, at most

@compute @workgroup_size(8, 8)
fn bounce(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = lightSize();
    if (any(gid.xy >= size) || !medium(vec2<i32>(gid.xy)).mirror) {
        return;
    }
    let direct = textureLoad(sunIn, gid.xy, 0).x;
    if (direct <= 0.0) {
        return;
    }
    // How much of the light its cells stop they reflect as a mirror: of all
    // the light cell stops (build's absorb, the mist in its air's too: by
    // its cells' alone, a fogged mirror sent on the fog's share as well).
    let o = gid.xy * 2u;
    var shine = 0.0;
    for (var k = 0u; k < 4u; k++) {
        let q = min(o + vec2<u32>(k & 1u, k >> 1u), P.size - 1u);
        let t = wordType(cells[q.y * P.size.x + q.x].x);
        let absorb = 1.0 - clarityAt(elems[t].clarity, P.dx);
        shine += absorb * elems[t].shine * dot(albedo(t), vec3<f32>(0.2126, 0.7152, 0.0722));
    }
    let stop = max(medium(vec2<i32>(gid.xy)).absorb, 1e-6);
    let n = facing(o);
    let v = -sunDirection(P.day);  // the way the light goes
    let along = dot(v, n);
    // Only a surface that goes on to either side (a mirror light cell around
    // it each way along it) is flat enough to send a beam: the rounded tip
    // of a plate scatters what it catches every way.
    let c = vec2<i32>(gid.xy);
    let side = vec2<f32>(-n.y, n.x);
    var ahead = false;
    var behind = false;
    for (var k = 0; k < 9; k++) {
        let off = vec2<i32>(k % 3 - 1, k / 3 - 1);
        if (k != 4 && mirrorAt(c + off)) {
            let s = dot(vec2<f32>(off), side);
            ahead = ahead || s > 0.3;
            behind = behind || s < -0.3;
        }
    }
    if (shine <= 0.0 || along >= 0.0 || !ahead || !behind) {
        return;
    }
    // What of the sunlight reaching it the light cell stops (its mirror's
    // share of it), as much as its cells reflect.
    let strength = direct * (1.0 - sunClear(medium(c), v)) * shine / stop;
    let r = v - 2.0 * along * n;
    // Half a light cell at a time, half of it shared out over the four
    // light cells nearest, so beams side by side light all evenly.
    var p = vec2<f32>(gid.xy) + 0.5;
    var through = 1.0;
    for (var s = 0u; s < 2u * BOUNCE_REACH; s++) {
        p += 0.5 * r;
        let at = vec2<i32>(floor(p));
        if (any(at < vec2<i32>(0)) || any(at >= vec2<i32>(size))) {
            break;
        }
        // Not in the mirror's own light cell: what it stops is in strength
        // already (dimmed by it again, a light cell all of polished metal
        // sent no beam at all).
        if (all(at == vec2<i32>(gid.xy))) {
            continue;
        }
        let q = p - 0.5;
        let i = vec2<i32>(floor(q));
        let f = q - floor(q);
        let share = 0.5 * strength * through * BOUNCE_SCALE;
        for (var k = 0; k < 4; k++) {
            let j = i + vec2<i32>(k & 1, k >> 1);
            let w = select(1.0 - f, f, vec2<bool>((k & 1) != 0, (k >> 1) != 0));
            if (all(j >= vec2<i32>(0)) && all(j < vec2<i32>(size))) {
                atomicAdd(&bounced[u32(j.y) * size.x + u32(j.x)], u32(share * w.x * w.y));
            }
        }
        through *= sqrt(clearAlong(medium(at), r));
        if (through < 0.01) {
            break;
        }
    }
}

// Whether the light cell at q is (or holds) a mirror, in the world.
fn mirrorAt(q: vec2<i32>) -> bool {
    return all(q >= vec2<i32>(0)) && all(q < vec2<i32>(lightSize())) && medium(q).mirror;
}

// ---------------------------------------------------------------- cascade

// Where probe i of cascade k is, in light cells: in the middle of light cell
// i x 2^k, so every other probe of a cascade is where one of the next
// cascade's is, and the light cells' own (cascade 0) are where they are.
fn probeAt(i: vec2<u32>, k: u32) -> vec2<f32> {
    return vec2<f32>(i * (1u << k)) + 0.5;
}

// Whether a probe at p (light cells) is out in the open. One inside a wall
// casts no rays: they would start past the wall's near side and see what
// is behind it, the far side of a sealed room, say, and through the probes
// around it, the light inside would see that too. (Which would let a room's
// heat out, and the cold in.)
fn openProbe(p: vec2<f32>) -> bool {
    let m = medium(clamp(vec2<i32>(p), vec2<i32>(0), vec2<i32>(lightSize()) - 1));
    return m.across + m.down > 0.5;
}

// What a ray gathers over a stretch of the world, and the share of the light
// let through all of it.
struct Ray {
    light: vec3<f32>,
    through: f32,
}

// A ray from `start`, `steps` steps of `stride` (a light cell long at
// most, so it can't step over a wall that thick), each crossing that much
// of a light cell.
fn march(start: vec2<f32>, stride: vec2<f32>, steps: f32) -> Ray {
    let size = vec2<i32>(lightSize());
    let span = length(stride);
    let d = stride / span;
    var r = Ray(vec3<f32>(0.0), 1.0);
    var s = 0.0;
    loop {
        if (s >= steps) {
            break;
        }
        let q = start + stride * (s + 0.5);
        let at = vec2<i32>(floor(q));
        if (at.y < 0 || at.y >= size.y) {
            r.light += r.through * select(groundGlow(), skyGlow(), at.y < 0);
            r.through = 0.0;
            break;
        }
        // Past a side of the world, which goes on as its edge column, the
        // ray crosses the same cell over and over until it goes up or down
        // a row: all those steps at once.
        var crossed = 1.0;
        if (at.x < 0 || at.x >= size.x) {
            let rest = select(q.y - floor(q.y), floor(q.y) + 1.0 - q.y, stride.y > 0.0);
            crossed = clamp(ceil(rest / max(abs(stride.y), 1e-4)), 1.0, steps - s);
        }
        let cell = vec2<i32>(clamp(at.x, 0, size.x - 1), at.y);
        // Most of what rays cross is clear air, which neither gives off
        // anything nor dims them: a light cell build found stopping nothing
        // either way (its z, 1 - across and 1 - down, both 0). Taken as any
        // other, it would add r.through * 0 * glow and leave r.through as
        // it was, the same to the bit.
        if (textureLoad(scene, cell, 0).z == 0u) {
            s += crossed;
            continue;
        }
        let m = medium(cell);
        var clear = clearAlong(m, d);
        if (crossed * span != 1.0) {
            clear = pow(clear, crossed * span);
        }
        var glow = m.glow;
        if (m.mirror) {
            glow += specular(cell, d);
        }
        r.light += r.through * (1.0 - clear) * glow;
        r.through *= clear;
        s += crossed;
        if (r.through < OPAQUE) {
            r.through = 0.0;
            break;
        }
    }
    return r;
}

// What probes of cascade k1 found beyond a ray of the cascade below that
// ends at `start`, along d: what cascade k1 stores along its directions b0
// and b1 (which split d's share of the circle; for cascade 0, b0 = b1 = d's
// own), from its probes j0 + (0 or 1 along each side), f of the way to the
// second (none where f is 0), each reached by a ray of its own; summed by
// how much each counts (weight).
//
// The probes above aren't where the one below is, so their rays don't
// start where its ray ends: they start as far from them, the same way.
// Carrying on from the end of its ray with what they found, interpolated
// between them, would skip what lies between, and see through a wall thin
// enough to fit there: from inside a glasshouse, the sky past its roof. So
// the ray is carried on to where each of them starts its own (the "bilinear
// fix" of radiance cascades). A probe below with both indices even is where
// one of cascade k1's is: one ray. One odd: two, half each. Both: four.
struct Beyond {
    light: vec3<f32>,
    weight: f32,
}

// Probes of cascade k1 j0 + (0 or 1 along each side), f of the way to the
// second: which of the four count (f isn't 0 for them) and are out in the
// open (one inside a wall stores nothing, see openProbe), as bits.
fn openAround(j0: vec2<u32>, f: vec2<f32>, k1: u32) -> u32 {
    let n1 = probes(k1);
    var open = 0u;
    for (var c = 0u; c < 4u; c++) {
        let side = vec2<u32>(c & 1u, c >> 1u);
        let w = select(1.0 - f, f, side == vec2<u32>(1u));
        if (w.x * w.y > 0.0 && openProbe(probeAt(min(j0 + side, n1 - 1u), k1))) {
            open |= 1u << c;
        }
    }
    return open;
}

fn reachUpper(start: vec2<f32>, d: vec2<f32>, b0: u32, b1: u32, j0: vec2<u32>, f: vec2<f32>, open: u32, k1: u32) -> Beyond {
    let n1 = probes(k1);
    var out = Beyond(vec3<f32>(0.0), 0.0);
    for (var c = 0u; c < 4u; c++) {
        if ((open & (1u << c)) == 0u) {
            continue;
        }
        let side = vec2<u32>(c & 1u, c >> 1u);
        let w = select(1.0 - f, f, side == vec2<u32>(1u));
        let j = min(j0 + side, n1 - 1u);
        let span = probeAt(j, k1) + d * reach(k1) - start;
        let count = ceil(length(span) - 0.01);
        var r = Ray(vec3<f32>(0.0), 1.0);
        if (count >= 1.0) {
            r = march(start, span / count, count);
        }
        var far = textureLoad(upper, vec2<u32>(b0 * n1.x + j.x, j.y), 0).rgb;
        if (b1 != b0) {
            far = 0.5 * (far + textureLoad(upper, vec2<u32>(b1 * n1.x + j.x, j.y), 0).rgb);
        }
        out.light += w.x * w.y * (r.light + r.through * far);
        out.weight += w.x * w.y;
    }
    return out;
}

// The same from cascade k1's probes around probe i of the cascade below
// (see reachUpper), `open` of them as openAround finds; with none of those
// open (it is beside a wall they are inside of), from those around where
// its ray ends, further out.
fn reachFrom(start: vec2<f32>, d: vec2<f32>, b0: u32, b1: u32, i: vec2<u32>, open: u32, k1: u32) -> Beyond {
    if (open != 0u) {
        return reachUpper(start, d, b0, b1, i / 2u, 0.5 * vec2<f32>(i & vec2<u32>(1u)), open, k1);
    }
    let u = clamp((start - 0.5) / f32(1u << k1), vec2<f32>(0.0), vec2<f32>(probes(k1) - 1u));
    let j0 = vec2<u32>(u);
    let f = fract(u);
    return reachUpper(start, d, b0, b1, j0, f, openAround(j0, f, k1), k1);
}

// openAround for the probes of cascade k + 1 around probe i of cascade k.
fn openAbove(i: vec2<u32>, k: u32) -> u32 {
    return openAround(i / 2u, 0.5 * vec2<f32>(i & vec2<u32>(1u)), k + 1u);
}

// Probes of cascade k reach the next cascade's in four ways (see castRay):
// one where both their indices are even (it is where one of those is), two
// where one is odd, four where both are. A thread per probe and direction,
// laid out so a workgroup mostly takes probes of one kind, with the even
// indices first along each side, then the odd, so its threads take about as
// long. Which probe the thread at index t of n along a side takes.
fn probeIndex(t: u32, n: u32) -> u32 {
    let evens = (n + 1u) / 2u;
    return select(2u * (t - evens) + 1u, 2u * t, t < evens);
}

// Cascade 0 is gather's: a thread per light cell and direction of cascade 1
// (farther).
@compute @workgroup_size(8, 8)
fn cascade(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = probes(C.level);
    if (C.level == 0u) {
        if (gid.x < directions(1u) * n.x && gid.y < n.y) {
            farther(gid.x / n.x, vec2<u32>(probeIndex(gid.x % n.x, n.x), probeIndex(gid.y, n.y)));
        }
    } else if (gid.x < directions(C.level) * n.x && gid.y < n.y) {
        let a = gid.x / n.x;
        castRay(a, vec2<u32>(probeIndex(gid.x % n.x, n.x), probeIndex(gid.y, n.y)));
    }
}

@compute @workgroup_size(256)
fn sun(@builtin(local_invocation_index) t: u32) {
    sweepSun(t);
}

// The ray of cascade C.level from probe i along direction a: over the
// cascade's distances, and on with what the cascade above found beyond
// (reachUpper).
fn castRay(a: u32, i: vec2<u32>) {
    let k = C.level;
    let n = probes(k);
    let texel = vec2<u32>(a * n.x + i.x, i.y);
    let p = probeAt(i, k);
    if (!openProbe(p)) {
        textureStore(cascadeOut, texel, vec4<f32>(0.0));
        return;
    }
    let d = direction(a, k);
    let size = vec2<i32>(lightSize());
    let lit = textureLoad(lightIn, clamp(vec2<i32>(p), vec2<i32>(0), size - 1), 0).rgb;
    let start = p + d * reach(k);
    let steps = f32(1u << k);
    if (k + 1u >= C.count) {
        // Past the farthest cascade it has left the world, or runs on along
        // its edge to the horizon.
        let r = march(start, d, steps);
        textureStore(cascadeOut, texel, vec4<f32>(r.light + r.through * select(groundGlow(), skyGlow(), d.y <= 0.0), 0.0));
        return;
    }
    let up = reachFrom(start, d, 2u * a, 2u * a + 1u, i, openAbove(i, k), k + 1u);
    if (up.weight < 0.01) {
        // None of them is open: there is no telling what is beyond, and it
        // is taken to be what reached the probe last time.
        let r = march(start, d, steps);
        textureStore(cascadeOut, texel, vec4<f32>(r.light + r.through * lit, 0.0));
        return;
    }
    textureStore(cascadeOut, texel, vec4<f32>(up.light / up.weight, 0.0));
}

// ----------------------------------------------------------------- gather

// What a ray from a light cell sees in the cell beside it at q, along d:
// light given off there, the share let through, and how open that side is
// (as clear, but the world's outside is open). Above the world is the sky,
// below it the ground, and past its sides it goes on as its edge.
struct Seen {
    light: vec3<f32>,
    clear: f32,
    open: f32,
}

fn seen(q: vec2<i32>, d: vec2<f32>) -> Seen {
    let size = vec2<i32>(lightSize());
    if (q.y < 0 || q.y >= size.y) {
        return Seen(select(groundGlow(), skyGlow(), q.y < 0), 0.0, 1.0);
    }
    let m = medium(vec2<i32>(clamp(q.x, 0, size.x - 1), q.y));
    let clear = clearAlong(m, d);
    return Seen((1.0 - clear) * m.glow, clear, clear);
}

// The light reaching light cell c (centre p) from the quarter of directions
// around b (0 right, 1 down, 2 left, 3 up): from the cell beside it that way
// (side) and half as much each from the two on its corners, and what
// cascade 1 found beyond them, along the two directions of it in that
// quarter; and how open that side is.
struct Quarter {
    light: vec3<f32>,
    open: f32,
}

fn quarter(c: vec2<u32>, b: u32, side: Seen, l: Seen, r: Seen) -> Quarter {
    let light = 0.5 * side.light + 0.25 * (l.light + r.light);
    let through = 0.5 * side.clear + 0.25 * (l.clear + r.clear);
    let open = 0.5 * side.open + 0.25 * (l.open + r.open);
    let far = 0.5 * (textureLoad(upper, fartherTexel((2u * b + 7u) % 8u, c), 0).rgb
        + textureLoad(upper, fartherTexel(2u * b, c), 0).rgb);
    return Quarter(light + through * far, open);
}

// Cascade 0's dispatch: what cascade 1 found beyond the cells around light
// cell c, along its direction a, for gather. The cells beside it reach a
// light cell and a half out, and the rest is carried on from there
// (reachFrom). Not knowing what is beyond, it is taken to be dark.
fn farther(a: u32, c: vec2<u32>) {
    let p = vec2<f32>(c) + 0.5;
    let d = direction(a, 1u);
    let up = reachFrom(p + 1.5 * d, d, a, a, c, openAbove(c, 0u), 1u);
    let light = select(vec3<f32>(0.0), up.light / max(up.weight, 1e-6), up.weight >= 0.01);
    textureStore(cascadeOut, fartherTexel(a, c), vec4<f32>(light, 0.0));
}

// Where farther keeps direction a for light cell c: the eight directions in
// blocks of the world's light cells, four across and two down (so the
// texture is no wider than cascade 1's).
fn fartherTexel(a: u32, c: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a % 4u, a / 4u) * lightSize() + c;
}

// What reaches a light cell through its open sides: what lights it up and
// what it reflects. So a wall doesn't pass on what reaches one side of it to
// the other, as light seeping through it. Inside an opaque block no light
// gets in, so the inside of a thick wall, which rays can glimpse past what
// is in front of it (the probes of a far cascade aren't where the rays
// below them end), is never brighter than what it is made of.
fn openLight(a: Quarter, b: Quarter, c: Quarter, d: Quarter) -> vec3<f32> {
    let open = a.open + b.open + c.open + d.open;
    let sum = a.open * a.light + b.open * b.light + c.open * c.light + d.open * d.light;
    return select(vec3<f32>(0.0), sum / open, open > 0.01);
}

@compute @workgroup_size(8, 8)
fn gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (all(gid.xy < lightSize())) {
        gatherCell(gid.xy);
    }
}

fn gatherCell(gid: vec2<u32>) {
    let c = vec2<i32>(gid);
    let m = medium(c);
    let slant = vec2<f32>(1.0, 1.0);
    let r = seen(c + vec2<i32>(1, 0), vec2<f32>(1.0, 0.0));
    let b = seen(c + vec2<i32>(0, 1), vec2<f32>(0.0, 1.0));
    let l = seen(c + vec2<i32>(-1, 0), vec2<f32>(1.0, 0.0));
    let a = seen(c + vec2<i32>(0, -1), vec2<f32>(0.0, 1.0));
    let rb = seen(c + vec2<i32>(1, 1), slant);
    let lb = seen(c + vec2<i32>(-1, 1), slant);
    let la = seen(c + vec2<i32>(-1, -1), slant);
    let ra = seen(c + vec2<i32>(1, -1), slant);
    let right = quarter(gid, 0u, r, ra, rb);
    let below = quarter(gid, 1u, b, rb, lb);
    let left = quarter(gid, 2u, l, lb, la);
    let above = quarter(gid, 3u, a, la, ra);

    // Sunlight, from the sun and thrown on by mirrors (bounce, taken here for
    // the next time): what it doesn't let through on the way, its cells
    // sharing that out by how much each takes in (update.wgsl). (A mirror's
    // beam crosses it its own way, taken as the sun's.) The sunlight that
    // falls on it is two cells' widths of it (`direct` is per unit of
    // width), so its cells share two cells' worth (REWRITE_PLAN.md 6-4: one
    // cell's, half of it, until then).
    let w = lightSize().x;
    let direct = textureLoad(sunIn, gid, 0).x
        + f32(atomicExchange(&bounced[gid.y * w + gid.x], 0u)) / BOUNCE_SCALE;
    let colour = sunlight(P.day);
    let strength = dot(colour, vec3<f32>(0.2126, 0.7152, 0.0722));
    let into = 1.0 - sunClear(m, -sunDirection(P.day));
    let sunShare = select(0.0, 2.0 * strength * direct * into / m.absorb, m.absorb > 0.0);

    let open = openLight(right, below, left, above);
    textureStore(lightOut, gid, vec4<f32>(open + colour * direct, 0.0));
    // For mirrors (specular), the light from each quarter. (The sun they send
    // on as a beam, see bounce.)
    textureStore(quartersOut, gid, vec4<u32>(
        packLight(right.light.rgb), packLight(below.light.rgb), packLight(left.light.rgb), packLight(above.light.rgb),
    ));
    textureStore(heatOut, gid, vec4<f32>(sunShare, 0.0, 0.0, 0.0));
}
