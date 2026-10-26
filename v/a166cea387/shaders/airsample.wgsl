// Sampling the air grid at a particle's position, for the passes that push
// things around with it. A module including this declares `P` and `air`.

// What the air's pressure does to matter of density rho (kg/m³, the
// element's rho) where it changes by grad (airsample's, P_ATM an
// atmosphere, per cell): -∇p / ρ, in cells a step² (a = -∇p / ρ, times dt² /
// dx). So the open air's fall in pressure with height lifts what is in it by
// the air's share of its weight, and a blast's front throws it by as much
// as its pressure over its density. (Until 4e a tuned 0.15 × 100 / density,
// some 1500 times less at a centimetre a cell.)
fn pressurePush(grad: vec2<f32>, rho: f32) -> vec2<f32> {
    let s = P.dt / P.dx;
    return -grad * (101325.0 / P_ATM) * s * s / max(rho, 1e-3);
}

// The air's pressure p (P_ATM an atmosphere, gauge, as its pass reports it)
// in the liquid's units, as a body's forces are summed: cells² a step² times
// water's density, p (dt / dx)² / 1000 kg/m³.
fn inLiquidUnits(p: f32) -> f32 {
    let s = P.dt / P.dx;
    return p * (101325.0 / P_ATM) * s * s / 1000.0;
}

struct AirSample {
    p: f32,
    vel: vec2<f32>,
    grad: vec2<f32>,  // pressure change per particle cell
}

fn airAt(i: vec2<i32>, dims: vec2<u32>) -> vec4<f32> {
    return air[u32(i.y) * dims.x + u32(i.x)];
}

// Bilinear air at a position given in air-cell index space.
fn airLerp(q: vec2<f32>) -> vec4<f32> {
    let dims = airDims(P.size);
    let hi = vec2<i32>(dims) - 1;
    let i0 = clamp(vec2<i32>(floor(q)), vec2<i32>(0), hi);
    let i1 = min(i0 + 1, hi);
    let f = clamp(q - vec2<f32>(i0), vec2<f32>(0.0), vec2<f32>(1.0));
    let top = mix(airAt(i0, dims), airAt(vec2<i32>(i1.x, i0.y), dims), f.x);
    let bottom = mix(airAt(vec2<i32>(i0.x, i1.y), dims), airAt(i1, dims), f.x);
    return mix(top, bottom, f.y);
}

// The air's pressure at cell c as atmLog, the cell's own (sampleAir at its
// middle reads the same): what its phase points go by.
fn cellAtmLog(c: vec2<i32>) -> f32 {
    return atmLog(airAt(c / i32(AIR_CELL), airDims(P.size)).x);
}

// The temperature the cell v at c acts at (effTempIn), its phase points at
// its own air pressure: what either side of a face works out for it, so
// both agree on what passes (each took its neighbour's at its own pressure,
// and across a face between two pressures heat was made or lost).
fn tempAtCell(v: vec4<u32>, c: vec2<i32>) -> f32 {
    let t = cellType(v);
    return effTempIn(t, cellTemp(v), phaseOf(t, wordSolute(v.x), wordStrength(v.x), cellAtmLog(c)));
}

// Just the air's pressure at a position (in cells).
fn airPressureAt(pos: vec2<f32>) -> f32 {
    return airLerp(pos / f32(AIR_CELL) - 0.5).x;
}

// The wind at a position (in cells): x velocity sits on the air cells'
// right faces and y velocity on their bottom faces, so each is sampled from
// its own offset grid.
fn windAt(pos: vec2<f32>) -> vec2<f32> {
    let q = pos / f32(AIR_CELL);
    return vec2<f32>(airLerp(q - vec2<f32>(1.0, 0.5)).y, airLerp(q - vec2<f32>(0.5, 1.0)).z);
}

// How fast the wind is at the middle of cell c, m/s (a gas's film grows with
// it: common.wgsl's gasFilm).
fn windSpeedAt(c: vec2<i32>) -> f32 {
    return length(windAt(vec2<f32>(c) + 0.5)) * P.dx / P.dt;
}

// The air's pressure at air cell i, and 1 where it is a fluid's (gas, or
// liquid or powder passing its background on), 0 for a solid's (common.wgsl's
// AIR_SOLID) or outside the world.
fn fluidPressureAt(i: vec2<i32>, dims: vec2<u32>) -> vec2<f32> {
    if (any(i < vec2<i32>(0)) || any(i >= vec2<i32>(dims))) {
        return vec2<f32>(0.0);
    }
    let a = airAt(i, dims);
    return vec2<f32>(a.x, select(1.0, 0.0, a.w <= AIR_SOLID));
}

// How much the air's pressure changes a cell across air cell c along d (a
// unit step): between its fluid neighbours either side, or from it to the
// one where the other is solid, none between two solids. A solid holds the
// pressure against it: the gas beyond a wall pushes nothing on this side of
// it. (Taken through a wall, a sealed tank at less than the outside air's
// pressure threw the water off its walls, at some fifty cells a step².)
fn pressureSlope(c: vec2<i32>, d: vec2<i32>, dims: vec2<u32>) -> f32 {
    let own = airAt(c, dims).x;
    let lo = fluidPressureAt(c - d, dims);
    let hi = fluidPressureAt(c + d, dims);
    if (lo.y > 0.0 && hi.y > 0.0) {
        return 0.5 * (hi.x - lo.x);
    }
    if (hi.y > 0.0) {
        return hi.x - own;
    }
    if (lo.y > 0.0) {
        return own - lo.x;
    }
    return 0.0;
}

// The velocity on air cell i's right (x) or bottom (y) face; none outside
// the world.
fn faceU(i: vec2<i32>, dims: vec2<u32>) -> f32 {
    if (any(i < vec2<i32>(0)) || any(i >= vec2<i32>(dims))) {
        return 0.0;
    }
    return airAt(i, dims).y;
}

fn faceV(i: vec2<i32>, dims: vec2<u32>) -> f32 {
    if (any(i < vec2<i32>(0)) || any(i >= vec2<i32>(dims))) {
        return 0.0;
    }
    return airAt(i, dims).z;
}

// Whether air cell i holds gas (common.wgsl's AIR_BLOCKED).
fn gasCellAt(i: vec2<i32>, dims: vec2<u32>) -> bool {
    return all(i >= vec2<i32>(0)) && all(i < vec2<i32>(dims)) && airAt(i, dims).w > AIR_BLOCKED;
}

// The four steps to the cells beside one: right, left, down, up.
fn besideStep(k: i32) -> vec2<i32> {
    return select(vec2<i32>(0, 1 - 2 * (k & 1)), vec2<i32>(1 - 2 * (k & 1), 0), k < 2);
}

// The wind of the cell of gas q met by matter beside it at q - d (d a unit
// step): the matter's own faces are shut (it is a wall to the flow), so
// what it meets is this cell's air: the velocity on its face across from
// the matter, along d, and the mean of its two faces across d
// (REWRITE_PLAN.md 4e-2; airsource.wgsl's pushIn pushes the same faces).
fn windIn(q: vec2<i32>, d: vec2<i32>, dims: vec2<u32>) -> vec2<f32> {
    if (d.x != 0) {
        let along = select(faceU(q - vec2<i32>(1, 0), dims), faceU(q, dims), d.x > 0);
        return vec2<f32>(along, 0.5 * (faceV(q, dims) + faceV(q - vec2<i32>(0, 1), dims)));
    }
    let along = select(faceV(q - vec2<i32>(0, 1), dims), faceV(q, dims), d.y > 0);
    return vec2<f32>(0.5 * (faceU(q, dims) + faceU(q - vec2<i32>(1, 0), dims)), along);
}

// The wind in the gas beside the cell c of matter (windIn of each cell of
// gas beside it, their mean), and how many cells of gas there are beside it.
fn windBeside(c: vec2<i32>) -> vec3<f32> {
    let dims = airDims(P.size);
    var sum = vec2<f32>(0.0);
    var n = 0.0;
    for (var k = 0; k < 4; k++) {
        let d = besideStep(k);
        if (gasCellAt(c + d, dims)) {
            sum += windIn(c + d, d, dims);
            n += 1.0;
        }
    }
    return vec3<f32>(sum / max(n, 1.0), n);
}

// Air at a particle position (in cells). Pressure sits at air-cell centers;
// its gradient is the cell's the position is in (pressureSlope).
fn sampleAir(pos: vec2<f32>) -> AirSample {
    let q = pos / f32(AIR_CELL);
    let dims = airDims(P.size);
    let c = clamp(vec2<i32>(floor(q)), vec2<i32>(0), vec2<i32>(dims) - 1);
    var s: AirSample;
    s.p = airLerp(q - 0.5).x;
    s.vel = windAt(pos);
    s.grad = vec2<f32>(pressureSlope(c, vec2<i32>(1, 0), dims), pressureSlope(c, vec2<i32>(0, 1), dims)) / f32(AIR_CELL);
    return s;
}
