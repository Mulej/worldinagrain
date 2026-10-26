// Heat's clock (REWRITE_PLAN.md, decision 1). A step stands for dt seconds
// of real time and dayDt of the day's (Params; js/shared.js): heat near the
// outdoor temperature goes on the day's clock, so the sun warms the ground
// and the night chills it in a day as long as the day is, and hot things
// and what is around them in real time, so lava stays molten for minutes
// and wood beside it warms as it would. How real-time heat is at a place,
// 0 (the day's clock) to 1 (real time), is the clock field, a value per
// light cell: as real-time as the hottest matter in it is (realness in
// common.wgsl), and as the hottest matter
// around it is, out to CLOCK_REACH cells, less and less over CLOCK_REACH
// more (common.wgsl's clockStep makes it seconds, evenly in the logarithm
// between the two clocks). A real-time event's heat only gets so far: in
// five minutes some 2 cm into rock and 6.5 cm into steel (the diffusion
// length √(αt)), so past the ramp the day's clock is what matters. Heat
// passing between two cells goes by the faster of their clocks, so both
// work it out alike and heat is conserved (local time stepping; Gear &
// Wells, BIT 24, 1984; Berger & Colella, J. Comput. Phys. 82, 1989).
// Heaters and coolers are held at their temperature by their power, as a
// stove's ring is, and stay on the day's clock: they don't make one.
//
// Worked out every step (COMPATIBILITY.md `clock-refresh`: with the light,
// every 4 steps, hot matter moving more than about 2 cells a step outran
// its field), in three passes over light cells: each one's hottest matter
// (source); from those, the most that reaches each along its row
// (across); then from those, down its column (down), so out to a square
// around each. A field that is gone fades over 1 / CLOCK_FADE steps rather
// than at once: a flame flickers in and out of a light cell, and its
// surroundings would otherwise switch between clocks.

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> elems: array<Elem>;
@group(0) @binding(2) var<storage, read> cells: array<vec4<u32>>;
// Per light cell: the realness of its hottest matter (source).
@group(0) @binding(3) var<storage, read_write> hot: array<f32>;
@group(0) @binding(4) var<storage, read_write> along: array<f32>;
// The field, per light cell (read by update.wgsl, heat.wgsl, electric.wgsl
// as a texture).
@group(0) @binding(5) var clock: texture_storage_2d<r32float, read_write>;
// Per light cell: how real-time the current's heat made it last step, f32
// bits (electric.wgsl's MAX_JOULE_RISE), taken and cleared by source.
@group(0) @binding(6) var<storage, read_write> heated: array<atomic<u32>>;

const CLOCK_FADE: f32 = 1.0 / 32.0;
// Light cells (2 cells across) looked across either way: as far as the ramp
// reaches, 2 CLOCK_REACH cells.
const SPAN: i32 = i32(CLOCK_REACH);

// The most of a light cell's realness that reaches k light cells away.
fn reaching(k: i32) -> f32 {
    return clamp(2.0 - f32(2 * abs(k)) / CLOCK_REACH, 0.0, 1.0);
}

fn lightCells() -> vec2<u32> {
    return (P.size + 1u) / 2u;
}

// How real-time the matter in the cell at p makes heat: by the temperature
// it acts at (its phase points at 1 atm, as the light's build takes them);
// none for a heater or cooler, held at its temperature on the day's clock;
// all for something burning, whose burning surface is hot however cold the
// cell is through (lit by a flame; stage 5).
fn cellRealness(p: vec2<u32>) -> f32 {
    let q = min(p, P.size - 1u);
    let c = unpackCell(cells[q.y * P.size.x + q.x]);
    if ((elems[c.t].flags & F_FIXED_TEMP) != 0u) {
        return 0.0;
    }
    if ((c.flags & FL_BURN) != 0u && elems[c.t].kind != KIND_GAS) {
        return 1.0;
    }
    return realness(effTempIn(c.t, c.temp, phaseOf(c.t, soluteOf(c), strengthOf(c), 0.0)));
}

@compute @workgroup_size(8, 8)
fn source(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = lightCells();
    if (any(gid.xy >= n)) {
        return;
    }
    let o = gid.xy * 2u;
    let k = gid.y * n.x + gid.x;
    let current = bitcast<f32>(atomicExchange(&heated[k], 0u));
    hot[k] = max(max(max(cellRealness(o), cellRealness(o + vec2<u32>(1u, 0u))),
        max(cellRealness(o + vec2<u32>(0u, 1u)), cellRealness(o + vec2<u32>(1u, 1u)))), current);
}

@compute @workgroup_size(8, 8)
fn across(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = lightCells();
    if (any(gid.xy >= n)) {
        return;
    }
    var most = 0.0;
    for (var k = -SPAN; k <= SPAN; k++) {
        let x = i32(gid.x) + k;
        if (x >= 0 && x < i32(n.x)) {
            most = max(most, min(hot[gid.y * n.x + u32(x)], reaching(k)));
        }
    }
    along[gid.y * n.x + gid.x] = most;
}

@compute @workgroup_size(8, 8)
fn down(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = lightCells();
    if (any(gid.xy >= n)) {
        return;
    }
    var most = 0.0;
    for (var k = -SPAN; k <= SPAN; k++) {
        let y = i32(gid.y) + k;
        if (y >= 0 && y < i32(n.y)) {
            most = max(most, min(along[u32(y) * n.x + gid.x], reaching(k)));
        }
    }
    let was = textureLoad(clock, gid.xy).x;
    textureStore(clock, gid.xy, vec4<f32>(max(most, was - CLOCK_FADE), 0.0, 0.0, 0.0));
}
