// Gravity (GRAVITY_PLAN.md): the pull of every mass in the world on every
// other, by Newton's law, as a field the moving matter follows on top of
// Earth's own pull (which isn't in it).
//
// Each cell has the mass of a cube a cell on a side (the world is a cell
// deep) of its element's density, under the real G; a black hole's cells
// have what makes the hole pull with P.holePull's share of Earth's g at its
// edge. Both come in as "pull" μ: the field of a mass μ at r cells is
// μ / r² cells/step² (μ = G m Δt², Earth's g taken at a centimetre a cell;
// see sim.js).
//
// Summing every cell's pull on every cell would be far too slow, so it goes
// as Barnes-Hut does on a tree, here on a grid: blocks of BLOCK × BLOCK
// cells, each with its mass and centre of mass, and coarser levels of them
// 2 × 2 at a time (a pyramid). A block takes the pull of the blocks around
// it directly, and of each farther one at the coarsest level that is still
// well apart from it, at its centre of mass (the interaction lists of the
// fast multipole method, with monopoles only; Greengard & Rokhlin, J. Comput.
// Phys. 73, 1987). Every block is counted once, the world's edges are open
// (no wrapping, no boundary to solve against), and there is no iterative
// solve: a few hundred pulls a block.
//
//   mass    per block, its mass and centre of mass; levels 1-3 of the
//           pyramid in workgroup memory (8 × 8 blocks a workgroup)
//   upper   the levels above, in one workgroup
//   field   per block, the pull on its middle and the potential there, and
//           down (pull.wgsl)

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> elems: array<Elem>;
@group(0) @binding(2) var<storage, read> cells: array<vec4<u32>>;
// Per block of each level in turn: its mass, and mass × x and × y (cells).
// After them, per block of level 0, its down word as last worked out, plus
// one (none yet: 0), for downStep's hysteresis.
@group(0) @binding(3) var<storage, read_write> masses: array<vec4<f32>>;
// Per block of level 0: the masses' pull on its middle, cells/step² (x, y),
// their potential there, cells²/step² (z), and its down word (w; pull.wgsl).
@group(0) @binding(4) var pulls: texture_storage_2d<rgba32float, write>;

const BLOCK: u32 = GRAVITY_BLOCK;     // cells a side of a block of level 0 (the light's cells)
const TOP_BLOCKS: u32 = GRAVITY_TOP_BLOCKS;  // the top level has no more blocks than this
const SOFTENING: f32 = 1.0;      // cells: a block's pull on its own middle is finite
const UPPER_THREADS: u32 = 256u;

// Blocks a side at level l.
fn levelDims(l: u32) -> vec2<u32> {
    let n0 = (P.size + vec2<u32>(BLOCK - 1u)) / BLOCK;
    return (n0 + vec2<u32>((1u << l) - 1u)) >> vec2<u32>(l);
}

// Where level l starts in `masses`.
fn levelBase(l: u32) -> u32 {
    var base = 0u;
    for (var k = 0u; k < l; k++) {
        let d = levelDims(k);
        base += d.x * d.y;
    }
    return base;
}

// The coarsest level, with no more than TOP_BLOCKS blocks.
fn topLevel() -> u32 {
    for (var l = 0u; l < 31u; l++) {
        let d = levelDims(l);
        if (d.x * d.y <= TOP_BLOCKS) {
            return l;
        }
    }
    return 31u;
}

// Where the blocks' down words start in `masses`: past the pyramid, whose
// levels 1-3 the mass pass writes however small the world.
fn downBase() -> u32 {
    return levelBase(max(topLevel(), 3u) + 1u);
}

// A cell's pull μ: a black hole's from the UI, anything else's from its
// real density, rho, a powder's its pile's (REWRITE_PLAN.md 9-2: it took
// the table's, which holds the fixed ones at 10).
fn cellPull(v: vec4<u32>) -> f32 {
    let t = cellType(v);
    if (t == BLACK_HOLE) {
        return P.holePull;
    }
    return elems[t].rho * P.massPull;
}

var<workgroup> tile: array<vec3<f32>, 64>;

@compute @workgroup_size(8, 8)
fn mass(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let b = wid.xy * 8u + lid.xy;
    var s = vec3<f32>(0.0);
    for (var j = 0u; j < BLOCK; j++) {
        for (var i = 0u; i < BLOCK; i++) {
            let c = b * BLOCK + vec2<u32>(i, j);
            if (all(c < P.size)) {
                let m = cellPull(cells[c.y * P.size.x + c.x]);
                s += m * vec3<f32>(1.0, vec2<f32>(c) + 0.5);
            }
        }
    }
    let d0 = levelDims(0u);
    if (all(b < d0)) {
        masses[b.y * d0.x + b.x] = vec4<f32>(s, 0.0);
    }
    tile[li] = s;
    workgroupBarrier();
    // Levels 1-3 from this workgroup's 8 × 8 blocks: 4 × 4, 2 × 2, 1.
    for (var l = 1u; l <= 3u; l++) {
        let n = 8u >> l;
        let mine = all(lid.xy < vec2<u32>(n));
        var sum = vec3<f32>(0.0);
        if (mine) {
            for (var k = 0u; k < 4u; k++) {
                let q = lid.xy * 2u + vec2<u32>(k & 1u, k >> 1u);
                sum += tile[q.y * 8u + q.x];
            }
        }
        workgroupBarrier();
        if (mine) {
            tile[lid.y * 8u + lid.x] = sum;
            let at = wid.xy * n + lid.xy;
            let d = levelDims(l);
            if (all(at < d)) {
                masses[levelBase(l) + at.y * d.x + at.x] = vec4<f32>(sum, 0.0);
            }
        }
        workgroupBarrier();
    }
}

@compute @workgroup_size(256)
fn upper(@builtin(local_invocation_index) li: u32) {
    let top = topLevel();
    for (var l = 4u; l <= top; l++) {
        let d = levelDims(l);
        let below = levelDims(l - 1u);
        let base = levelBase(l);
        let baseBelow = levelBase(l - 1u);
        for (var i = li; i < d.x * d.y; i += UPPER_THREADS) {
            let at = vec2<u32>(i % d.x, i / d.x);
            var sum = vec4<f32>(0.0);
            for (var k = 0u; k < 4u; k++) {
                let q = at * 2u + vec2<u32>(k & 1u, k >> 1u);
                if (all(q < below)) {
                    sum += masses[baseBelow + q.y * below.x + q.x];
                }
            }
            masses[base + i] = sum;
        }
        storageBarrier();
        workgroupBarrier();
    }
}

// The pull of block c of level l on the point p (cells), if it has any mass,
// and its potential there (z): -μ / r, softened alike, so the pull is its
// slope.
fn pullOf(base: u32, d: vec2<u32>, c: vec2<i32>, p: vec2<f32>) -> vec3<f32> {
    if (any(c < vec2<i32>(0)) || any(c >= vec2<i32>(d))) {
        return vec3<f32>(0.0);
    }
    let s = masses[base + u32(c.y) * d.x + u32(c.x)];
    if (s.x <= 0.0) {
        return vec3<f32>(0.0);
    }
    let r = s.yz / s.x - p;
    let d2 = dot(r, r) + SOFTENING * SOFTENING;
    let inverse = 1.0 / sqrt(d2);
    return s.x * vec3<f32>(r * inverse / d2, -inverse);
}

// Down by a pull (pull.wgsl): the neighbour step it is nearest (stepIndex),
// but old's (a step, or none past 7) while the pull leans no more than
// DOWN_STICK past old's sector, so where the pull's direction lies on a
// sector's edge (round a black hole) down doesn't flicker from step to
// step, and with it how piles slump (audit 06 G1).
const DOWN_STICK: f32 = 0.0873;  // 5°

fn downStep(pull: vec2<f32>, old: u32) -> u32 {
    let eighth = 0.78539816;
    let a = atan2(pull.y, pull.x);
    let nearest = u32(i32(round(a / eighth)) + 8) & 7u;
    if (old > 7u) {
        return nearest;
    }
    var off = a - f32(old) * eighth;
    off -= round(off / 6.2831853) * 6.2831853;
    return select(nearest, old, abs(off) <= 0.5 * eighth + DOWN_STICK);
}

@compute @workgroup_size(8, 8)
fn field(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d0 = levelDims(0u);
    if (any(gid.xy >= d0)) {
        return;
    }
    let p = (vec2<f32>(gid.xy) + 0.5) * f32(BLOCK);
    let top = topLevel();
    var pull = vec3<f32>(0.0);
    // The blocks around it, directly.
    let b0 = vec2<i32>(gid.xy);
    for (var k = 0; k < 9; k++) {
        pull += pullOf(0u, d0, b0 + vec2<i32>(k % 3 - 1, k / 3 - 1), p);
    }
    // At each level, the children of the blocks around its parent that
    // aren't around it themselves: well apart from it, so far enough to take
    // whole, and covered by no finer level.
    var b = b0;
    var base = 0u;
    for (var l = 0u; l < top; l++) {
        let d = levelDims(l);
        let parent = b >> vec2<u32>(1u);
        for (var k = 0; k < 36; k++) {
            let c = parent * 2 + vec2<i32>(k % 6 - 2, k / 6 - 2);
            if (any(abs(c - b) > vec2<i32>(1))) {
                pull += pullOf(base, d, c, p);
            }
        }
        base += d.x * d.y;
        b = parent;
    }
    // And at the top, every block not around its ancestor there.
    let d = levelDims(top);
    for (var i = 0u; i < d.x * d.y; i++) {
        let c = vec2<i32>(vec2<u32>(i % d.x, i / d.x));
        if (any(abs(c - b) > vec2<i32>(1))) {
            pull += pullOf(base, d, c, p);
        }
    }
    // Down by the total pull, Earth's and the masses' (Earth's is added
    // here, so switching it works the field out again, sim.js), as last
    // worked out where it leans little past its sector; weightless where it
    // is too weak to settle anything, its step kept for when it isn't.
    let k = downBase() + gid.y * d0.x + gid.x;
    let old = u32(masses[k].x) - 1u;
    let total = vec2<f32>(0.0, P.earthG) + pull.xy;
    var word = downStep(total, old);
    if (pullShare(total) < WEIGHTLESS) {
        word = select(old, 2u, old > 7u) | DOWN_WEIGHTLESS;
    }
    masses[k] = vec4<f32>(f32((word & 7u) + 1u), 0.0, 0.0, 0.0);
    textureStore(pulls, gid.xy, vec4<f32>(pull, f32(word)));
}
