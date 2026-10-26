// What other passes put into the air (REWRITE_PLAN.md stages C and 4c,
// COMPATIBILITY.md `source-buffer`): pushes on its faces, and gas pushed
// aside, whole, with what it holds. A pass adds to a cell's slot; the next
// air step takes in the pushes (momentum, air.wgsl), and the next carry the
// gas (carry.wgsl's load: into the cell if it holds gas, else into the first
// cell beside it that does), and each clears what it took. So what a gas
// held is never lost where something takes its place: in a sealed room the
// rest of its gas is squeezed by it, and the room's pressure rises.
//
// SOURCE_WORDS words a cell, exact sums of two words each (common.wgsl's
// exactParts, EXACT_FINE; heatin.wgsl does the same): the velocity added to
// its right face and to its bottom face (cells a step); then gas: its dry
// gas's amount, vapour and mist, the oxygen used up of its dry gas (the
// share times the amount), and its heat (J, its heat capacity times its
// temperature in °C, as carry.wgsl keeps it). A module including this binds
// `source: array<atomic<u32>>`.

const SOURCE_WORDS: u32 = 16u;
const SOURCE_RIGHT: u32 = 0u;
const SOURCE_DOWN: u32 = 1u;
const SOURCE_AMOUNT: u32 = 2u;
const SOURCE_VAPOUR: u32 = 3u;
const SOURCE_MIST: u32 = 4u;
const SOURCE_LOST: u32 = 5u;
const SOURCE_HEAT: u32 = 6u;

fn addSource(i: u32, k: u32, x: f32) {
    if (x == 0.0) {
        return;
    }
    let at = SOURCE_WORDS * i + 2u * k;
    let w = exactParts(x, EXACT_FINE);
    let old = atomicAdd(&source[at], w.x);
    atomicAdd(&source[at + 1u], w.y + exactCarry(old, w.x));
}

// Cell i's air pushed by dv (cells a step): its right face's velocity by
// dv.x, its bottom face's by dv.y.
fn pushAir(i: u32, dv: vec2<f32>) {
    addSource(i, SOURCE_RIGHT, dv.x);
    addSource(i, SOURCE_DOWN, dv.y);
}

// The air of the cell of gas q pushed by du (cells a step) as matter beside
// it at q - d (d a unit step) drags it (REWRITE_PLAN.md 4e-2): on the faces
// airsample.wgsl's windIn reads it from, so what it reads there next is du
// more. (Faces outside the world, dims the air's cells, are left out.)
fn pushIn(q: vec2<i32>, d: vec2<i32>, du: vec2<f32>, dims: vec2<u32>) {
    if (d.x != 0) {
        pushFace(select(q - vec2<i32>(1, 0), q, d.x > 0), vec2<f32>(du.x, 0.0), dims);
        pushFace(q, vec2<f32>(0.0, du.y), dims);
        pushFace(q - vec2<i32>(0, 1), vec2<f32>(0.0, du.y), dims);
    } else {
        pushFace(select(q - vec2<i32>(0, 1), q, d.y > 0), vec2<f32>(0.0, du.y), dims);
        pushFace(q, vec2<f32>(du.x, 0.0), dims);
        pushFace(q - vec2<i32>(1, 0), vec2<f32>(du.x, 0.0), dims);
    }
}

// pushAir on air cell i's right and bottom faces, where it is in the world.
fn pushFace(i: vec2<i32>, dv: vec2<f32>, dims: vec2<u32>) {
    if (all(i >= vec2<i32>(0)) && all(i < vec2<i32>(dims))) {
        pushAir(u32(i.y) * dims.x + u32(i.x), dv);
    }
}

// Gas holding g at temp (°C), pushed out of cell i by what takes its place
// (or let go of there), into the air there.
fn pushGas(i: u32, g: Gas, temp: f32) {
    addSource(i, SOURCE_AMOUNT, g.amount);
    addSource(i, SOURCE_VAPOUR, g.vapour);
    addSource(i, SOURCE_MIST, g.mist);
    addSource(i, SOURCE_LOST, g.lost * g.amount);
    addSource(i, SOURCE_HEAT, gasHeatCap(g) * temp);
}

// What cell i was given since it last took it in, and none left. (Read
// first, and taken only if there is anything: nothing adds to it in the
// pass that takes it, the adders' dispatches being others, and nearly all
// cells are given nothing; an exchange on every word wrote every word.
// Where two take the one cell's, as the carry's load can, D-2, the first
// to exchange has it, as before.)
fn takeSource(i: u32, k: u32) -> f32 {
    let at = SOURCE_WORDS * i + 2u * k;
    if (atomicLoad(&source[at]) == 0u && atomicLoad(&source[at + 1u]) == 0u) {
        return 0.0;
    }
    let lo = atomicExchange(&source[at], 0u);
    return exactValue(vec2<u32>(lo, atomicExchange(&source[at + 1u], 0u)), EXACT_FINE);
}
