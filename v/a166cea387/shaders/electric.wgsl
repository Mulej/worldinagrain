// Electric current: batteries driving steady current around closed circuits,
// heating what it flows through and splitting electrolytes.
//
// Each conducting cell has a potential (volts). Between two touching
// conductors i and j a current
//
//     I = G (Vi - Vj + emf)    amperes, from i to j
//
// flows, where G is the conductance of the two half cells in series,
// 2 si sj / (si + sj) for cell conductances si, sj (elements.js sigma, less
// as a metal heats up), and emf is what batteries along the link drive: the
// mean of the two cells' EMF vectors (volts per cell along their painted
// direction) dotted with the step from i to j. A row of battery cells along
// their direction adds up, half a cell's worth at each end and a whole one
// across each joint between them. A link between water (or anything else
// carrying its current by ions: carriesIons) and an electrode takes back
// HALF_CELL of what drives it, splitting the water, and carries nothing
// while less drives it (the back-EMF). Charge doesn't pile up in a
// conductor, so what flows into a cell flows out of it (Kirchhoff), which
// makes the potential the solution of A V = b: A the links' conductance
// Laplacian, b what their EMFs drive (b_i = -sum of G emf over i's links).
// An open circuit settles with no current at all; a closed one carries what
// Ohm's law says.
//
// Cells touch through their four sides. Two that meet only at a corner, with
// neither of the cells beside both of them conducting, touch there too: a
// thin diagonal line of metal is a wire.
//
// The potential is solved each step by conjugate gradients (CG), from last
// step's, ELECTRIC_ITERATIONS iterations at most (sim.js takes more after a change)
// and fewer once the residual (each cell's current in and out not adding
// up) is TOL of the currents through the links (REWRITE_PLAN.md P1;
// audit/08 §5.2: in amperes, not as a share of what the batteries drive,
// thousands of times more through their own links), preconditioned
// by two-level additive Schwarz: a coarse correction from aggregates of
// cells plus Gauss-Seidel on each block's residual. The aggregates are the
// connected parts of each 8 x 8 block's strong links (G at least THETA of
// the root of the two cells' summed conductances: Vanek, Mandel & Brezina
// 1996), SLOTS of them a block at most: a wire through a block is one, two
// wires two, a plate in brine another from the brine around it, so the
// coarse problem holds every island of metal whole (Scheichl & Vainikko
// 2007; Graham & Hagger 1999), and a piece of metal floating in brine moves
// as one. The coarse problem is P'AP, the links between aggregates (its
// Galerkin rows), solved to INNER_TOL (INNER_MOST iterations at most) by
// CG in one workgroup. A link's
// current is good to its own drop, not to the potential's size, by the
// order its drop is summed in (linkDrop). In `test/schwarz.mjs` on the CPU this
// takes 24-35 iterations from nothing, 0-1 a step for a steady circuit, and
// keeps the residual to 1e-3 a step while a wire heats.
//
// Per step, with any battery in the world (sim.js leaves all this out
// without), starting from last step's potential, over 8 x 8 blocks:
//
//   assemble      each cell's conductance, EMF and aggregate (W_*), and
//                 whether the block has anything to solve
//   residual      r = b - A V; the coarse problem's rows and P'r for the
//                 block's aggregates; |I|² (the links' currents) and |r|²
//   coarse        one workgroup: the coarse correction e, A_c e = P'r (and
//                 the CG's alpha, and whether it is done)
//   precondition  V += alpha p and r -= alpha q; z = e on each aggregate
//                 plus SWEEPS symmetric Gauss-Seidel sweeps on the block's
//                 r, its ring held at zero; r.z, |r|² and P'r
//   then ELECTRIC_ITERATIONS times:
//   reduce        one workgroup: beta, from r.z
//   direction     p = z + beta p and q = A p; p.q and P'q
//   coarse, precondition
//   current       each cell's currents, over 16 x 16 blocks in workgroup
//                 memory: the current density drawn by render.wgsl, and the
//                 Joule heat and electrolysis they make, kept aside
//   effects       which then go into the cells: in a pass of their own, so
//                 that no block of the current pass reads a cell another
//                 block has already heated or split
//
// The sums over the blocks (dot products) go through a word per block
// (tileSlot), which the one-workgroup passes add up: no float atomics, and
// no pass waits on a readback. A pass with nothing left to do returns at
// once (S_DONE), so a settled circuit costs little over its dispatches.
//
// Units: volts, amperes, siemens, as for a cell a centimetre cube (CELL_M),
// whose heat capacity (J/K) is its element's heatCap. A cell Params.dx
// across conducts dx / CELL_M times as well (conductance goes as a face's
// area over its length), every one of them alike, so the potentials are the
// same at any scale and every current dx / CELL_M times what is worked out
// here: the heat it makes, I² / G, as well. That heat and the water it
// splits go into a cell (dx / CELL_M)³ times as big, so count for
// faceShare² of what they would at a centimetre (common.wgsl). A step is
// Params.dt.

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> elems: array<Elem>;
@group(0) @binding(2) var<storage, read_write> cells: array<vec4<u32>>;
// The potential (an f32; see linkDrop).
@group(0) @binding(3) var<storage, read_write> volts: array<f32>;
// Current density per cell (amperes across a cell's width, 2 x f16), for
// drawing and the hover readout; kept up for conductors only.
@group(0) @binding(4) var<storage, read_write> flow: array<u32>;
// STATS words (STAT_*: read back by sim.js), SCALARS words of the CG's
// state (S_*), TILE_WORDS per block (tileSlot), then the list of the coarse
// problem's live nodes (liveSlot).
@group(0) @binding(5) var<storage, read_write> tiles: array<atomic<u32>>;
// COARSE_WORDS per coarse node (a block's aggregate; coarseSlot): its row
// of the coarse problem and its vectors.
@group(0) @binding(6) var<storage, read_write> coarse: array<f32>;
// Per cell, W_* words of it (workSlot): what the passes after assemble read
// of it, the CG's vectors, and what the current pass leaves the effects
// pass.
@group(0) @binding(7) var<storage, read_write> work: array<u32>;
// Heat's clock per light cell (clock.wgsl), for the Joule heat and
// electrolysis.
@group(0) @binding(9) var clock: texture_2d<f32>;
// Heat put into cells as energy (heatin.wgsl): the Joule heat.
@group(0) @binding(10) var<storage, read_write> heatIn: array<atomic<u32>>;
// Per light cell: how real-time its Joule heat makes it (current's
// MAX_JOULE_RISE), f32 bits, the most of its cells', for the next step's
// clock (clock.wgsl's source).
@group(0) @binding(11) var<storage, read_write> heated: array<atomic<u32>>;

const BLOCK: i32 = ELECTRIC_BLOCK;     // 8: the blocks, and the aggregates' reach
const BLOCK_CELLS: u32 = 64u;
const SPAN: i32 = BLOCK + 2;            // a block and its ring
const SPAN_CELLS: u32 = 100u;
const_assert BLOCK_CELLS == u32(BLOCK * BLOCK) && SPAN_CELLS == u32(SPAN * SPAN);
const SLOTS: u32 = ELECTRIC_SLOTS;     // aggregates a block at most
const NO_SLOT: u32 = 0xffu;
const THETA: f32 = 0.08;
const SWEEPS: u32 = 2u;
const TOL: f32 = 1e-4;
const INNER_TOL: f32 = 1e-1;
const INNER_MOST: u32 = 30u;
const COARSE_THREADS: u32 = 64u;

// The words read back (js/shared.js ELECTRIC_STATS): battery cells seen this
// step (sim.js keeps the passes going while there are any), CG iterations
// taken, the relative residual left (f32 bits), live coarse nodes, and the
// coarse solves' iterations, all of them this step.
const STATS: u32 = ELECTRIC_STATS;
const STAT_BATTERIES: u32 = 0u;
const STAT_ITERATIONS: u32 = 1u;
const STAT_RESIDUAL: u32 = 2u;
const STAT_LIVE: u32 = 3u;
const STAT_COARSE: u32 = 4u;
// The CG's state: the passes done this step (the coarse pass counts them),
// alpha, beta, r.z and |r|² as the last precondition left them, |I|²,
// whether done, the coarse problem's live nodes, which of W_P0, W_P1 holds
// the current direction.
const SCALARS: u32 = ELECTRIC_SCALARS;
const S_STAGE: u32 = 0u;
const S_ALPHA: u32 = 1u;
const S_BETA: u32 = 2u;
const S_RZ: u32 = 3u;
const S_RR: u32 = 4u;
const S_BB: u32 = 5u;
const S_DONE: u32 = 6u;
const S_LIVE: u32 = 7u;
const S_PARITY: u32 = 8u;
const_assert S_PARITY < SCALARS;
// Per block: whether it has anything to solve, then its sums of |I|² (the
// currents through its cells' links, each counted from both ends), |r|²,
// r.z and p.q (f32 bits), and which of its aggregates are live coarse nodes
// (a bit a slot).
const TILE_WORDS: u32 = ELECTRIC_TILE_WORDS;
const T_LIVE: u32 = 0u;
const T_BB: u32 = 1u;
const T_RR: u32 = 2u;
const T_RZ: u32 = 3u;
const T_PQ: u32 = 4u;
const T_NODES: u32 = 5u;
const_assert T_NODES < TILE_WORDS;
// Per coarse node: its diagonal, its couplings to the nodes of its block's
// and the 8 around (36: by block offset (dy + 1) * 3 + dx + 1, then slot),
// P'r, P'q, the correction e, and the inner CG's r, p, q.
const COARSE_WORDS: u32 = ELECTRIC_COARSE_WORDS;
const C_DIAG: u32 = 0u;
const C_COUPLE: u32 = 1u;
const C_FR: u32 = 37u;
const C_FQ: u32 = 38u;
const C_E: u32 = 39u;
const C_R: u32 = 40u;
const C_P: u32 = 41u;
const C_Q: u32 = 42u;
const_assert C_Q < COARSE_WORDS;
// Per cell: conductance (f32), EMF (2 x f16), aggregate slot (NO_SLOT for
// none), the CG's r, z, two of p, q (f32), the current pass's effect, its
// links' interface states (FACE_*, two bits a link).
const W_SIG: u32 = 0u;
const W_EMF: u32 = 1u;
const W_AGG: u32 = 2u;
const W_R: u32 = 3u;
const W_Z: u32 = 4u;
const W_P0: u32 = 5u;
const W_Q: u32 = 7u;
const W_EFFECT: u32 = 8u;
const W_FACE: u32 = 9u;
const_assert W_FACE < ELECTRIC_WORK;
const_assert W_EFFECT < ELECTRIC_WORK;

const TAU: f32 = 6.2831853;
// The most a step's Joule heat raises a cell's temperature, °C: past it the
// step stands for less time where the cell is (heat's clock), not for less
// heat. (A dead short across a big battery would otherwise take a wire
// past melting and boiling in one step of the day's clock, six seconds of
// its heat; held to 150 °C a step, the rest was lost: REWRITE_PLAN.md P1-1.)
const MAX_JOULE_RISE: f32 = 150.0;
// Electrolysis takes its energy from the circuit (REWRITE_PLAN.md P1, the
// back-EMF; audit/08 §5.2). Splitting water takes 1.23 V and more at its
// electrodes, some 1.8 V across a cell in practice (Bard & Faulkner,
// Electrochemical Methods, 2001), half at each: so each link between a cell
// carrying ions (carriesIons) and an electrode drives HALF_CELL against the
// current through it, and carries none while less than that drives it (a
// diode with a resistance in series). Which way each such link conducts, or
// whether it is open, is taken from the step before's potentials (a lagged
// active set: assemble's faceWord), so the step's system stays linear. Of
// the I HALF_CELL a link takes, the reaction's enthalpy (water's
// thermoneutral 1.48 V, half at each electrode) goes into splitting it;
// the rest, the overpotential, into heat in the cell carrying ions.
const HALF_CELL: f32 = 0.9;
const HALF_THERMONEUTRAL: f32 = 0.74;
// Each cell's links' states (W_FACE): two bits a link, for the eight ways
// of dirOf.
const FACE_NONE: u32 = 0u;      // no interface: both carry ions, or neither
const FACE_OUT: u32 = 1u;       // conducting, current leaving the cell
const FACE_IN: u32 = 2u;        // conducting, current coming in
const FACE_OPEN: u32 = 3u;      // carrying none
const MAX_FLOW: f32 = 65000.0;        // A, the most current density kept (f16)

fn inWorld(p: vec2<i32>) -> bool {
    return all(p >= vec2<i32>(0)) && all(p < vec2<i32>(P.size));
}

fn cellIndex(p: vec2<i32>) -> u32 {
    return u32(p.y) * P.size.x + u32(p.x);
}

fn cellCount() -> u32 {
    return P.size.x * P.size.y;
}

fn blocksWide() -> u32 {
    return (P.size.x + u32(BLOCK) - 1u) / u32(BLOCK);
}

fn blockCount() -> u32 {
    return blocksWide() * ((P.size.y + u32(BLOCK) - 1u) / u32(BLOCK));
}

fn workSlot(k: u32, i: u32) -> u32 {
    return k * cellCount() + i;
}

fn workF(k: u32, i: u32) -> f32 {
    return bitcast<f32>(work[workSlot(k, i)]);
}

fn setWorkF(k: u32, i: u32, v: f32) {
    work[workSlot(k, i)] = bitcast<u32>(v);
}

fn scalarSlot(k: u32) -> u32 {
    return STATS + k;
}

fn scalar(k: u32) -> f32 {
    return bitcast<f32>(atomicLoad(&tiles[scalarSlot(k)]));
}

fn setScalar(k: u32, v: f32) {
    atomicStore(&tiles[scalarSlot(k)], bitcast<u32>(v));
}

fn tileSlot(t: u32, k: u32) -> u32 {
    return STATS + SCALARS + t * TILE_WORDS + k;
}

fn liveSlot(n: u32) -> u32 {
    return STATS + SCALARS + blockCount() * TILE_WORDS + n;
}

fn coarseSlot(node: u32, k: u32) -> u32 {
    return node * COARSE_WORDS + k;
}

// Conductance of the cell v: a metal's resistance rises with heat (resistT),
// never to under half the cold value; water conducts as well as what it
// carries lets it (solSigma: brine, acid), and as far as the cell is full
// of it (common.wgsl's amountIn: a quarter-full cell at the surface a
// quarter as well). A soil conducts through the water in its pores (its
// sigma is soaked: elements.js), by the square of how full of water they
// are (Archie's law, 1942, its saturation exponent 2), so dry not at all.
fn sigmaOf(v: vec4<u32>) -> f32 {
    let t = cellType(v);
    let e = elems[t];
    let solute = wordSolute(v.x);
    var sigma = e.sigma;
    if (holdsSolute(t) && solute != 0u) {
        sigma += elems[solute].solSigma * shareOf(wordStrength(v.x));
    } else if (holdsSoilWater(t)) {
        let full = f32(wordSoilWater(v.x)) / f32(SOIL_FULL);
        sigma *= full * full;
    } else if (e.sigma > 0.0 && e.resistT != 0.0) {
        sigma /= max(1.0 + e.resistT * (wordTempOf(v) - AMBIENT_T), 0.5);
    }
    return sigma * min(amountIn(v), 1.0);
}

// Whether a cell of t carries its current by ions in water: an electrolyte,
// and damp soil and mud, through the water they hold. Current leaving these
// into anything else that conducts (an electrode: metal, a battery) splits
// an electrolyte there; from one to another it passes on as it is.
fn carriesIons(t: u32) -> bool {
    return elems[t].electrolysis > 0.0 || holdsSoilWater(t) || t == MUD;
}

// EMF vector of a cell holding element e with aux a: volts per cell, along
// the direction it was painted in (an angle in 256ths of a turn, as fans).
fn emfOf(e: Elem, a: u32) -> vec2<f32> {
    if (e.emf == 0.0) {
        return vec2<f32>(0.0);
    }
    let angle = f32(a) / 256.0 * TAU;
    return e.emf * vec2<f32>(cos(angle), sin(angle));
}

// What drives a link's current, V_i - V_j + emf. The potentials'
// difference is exact for neighbours within a factor of two of each other
// (Sterbenz's lemma), and so is adding the EMF to it across a battery, where
// the two all but cancel; a piece of metal floating, its cells' potentials
// alike to their last bit, carries nothing (`floating.mjs`, STONE=1: 0 A).
// (Until 2026-10-25 the potential was meant to be a double float, Dekker
// 1971, its low part in a buffer of its own; but the GPU compiler,
// rearranging floating point unasked, found the two-sum's error terms
// nothing and dropped them, so the low parts were always 0: REWRITE_PLAN.md
// 8-2 found it. All along it was this order and plain f32 that worked.)
fn linkDrop(va: f32, vb: f32, emf: f32) -> f32 {
    return (va - vb) + emf;
}

// Conductance of the link between conductances a and b: half of each in series.
fn seriesG(a: f32, b: f32) -> f32 {
    return select(0.0, 2.0 * a * b / (a + b), a > 0.0 && b > 0.0);
}

// The state of a cell's link its way d, from its W_FACE word.
fn faceOf(word: u32, d: u32) -> u32 {
    return (word >> (2u * d)) & 3u;
}

// The EMF a link's interface adds, from the cell's side.
fn faceEmf(state: u32) -> f32 {
    return select(select(0.0, HALF_CELL, state == FACE_IN), -HALF_CELL, state == FACE_OUT);
}

// The state an interface link takes from what drives it, V_i - V_j + the
// batteries' EMF (linkDrop) by the step before's potentials. From the other
// cell the drop is exactly its negative (linkDrop's order), so both cells
// take the link alike, and A stays symmetric.
fn faceFrom(drop: f32) -> u32 {
    return select(select(FACE_OPEN, FACE_IN, drop < -HALF_CELL), FACE_OUT, drop > HALF_CELL);
}

// ------------------------------------------------------------ the blocks

// The block and its ring, in workgroup memory: conductance, EMF, potential,
// aggregate slot, and a vector the pass works on.
var<workgroup> sig: array<f32, SPAN_CELLS>;
var<workgroup> drive: array<u32, SPAN_CELLS>;
var<workgroup> vNear: array<f32, SPAN_CELLS>;
var<workgroup> slotOf: array<u32, SPAN_CELLS>;
var<workgroup> vec: array<f32, SPAN_CELLS>;
var<workgroup> vec2nd: array<f32, SPAN_CELLS>;
var<workgroup> face: array<u32, SPAN_CELLS>;
// Per cell of the block: union-find parents (assemble), its diagonal, its
// links' conductances and where each goes in the coarse row (residual).
var<workgroup> parent: array<atomic<u32>, BLOCK_CELLS>;
var<workgroup> diag: array<f32, BLOCK_CELLS>;
var<workgroup> rank: array<u32, BLOCK_CELLS>;
var<workgroup> linkG: array<f32, 512>;
var<workgroup> linkTo: array<u32, 512>;
var<workgroup> sums: array<vec4<f32>, BLOCK_CELLS>;
var<workgroup> blockLive: atomic<u32>;
var<workgroup> blockNodes: atomic<u32>;
var<workgroup> blockState: vec4<f32>;  // live, done, alpha or beta, parity

// The eight ways from a cell: the four sides, then the corners.
fn dirOf(d: u32) -> vec2<i32> {
    switch d {
        case 0u: { return vec2<i32>(1, 0); }
        case 1u: { return vec2<i32>(-1, 0); }
        case 2u: { return vec2<i32>(0, 1); }
        case 3u: { return vec2<i32>(0, -1); }
        case 4u: { return vec2<i32>(1, 1); }
        case 5u: { return vec2<i32>(-1, 1); }
        case 6u: { return vec2<i32>(1, -1); }
        default: { return vec2<i32>(-1, -1); }
    }
}

// Index into the block's arrays of the cell at q (block coordinates, -1 to
// BLOCK on the ring).
fn spanAt(q: vec2<i32>) -> u32 {
    return u32((q.y + 1) * SPAN + q.x + 1);
}

// The conductance of the link from the cell at k (span index) the way d
// goes, 0 for none: a corner link only where neither cell beside both
// conducts.
fn spanLink(k: u32, d: u32) -> f32 {
    let o = dirOf(d);
    let n = i32(k) + o.y * SPAN + o.x;
    if (d >= 4u && (sig[i32(k) + o.x] > 0.0 || sig[i32(k) + o.y * SPAN] > 0.0)) {
        return 0.0;
    }
    if (faceOf(face[k], d) == FACE_OPEN) {
        return 0.0;
    }
    return seriesG(sig[k], sig[n]);
}

// The batteries' EMF along that link.
fn spanBatteries(k: u32, d: u32) -> f32 {
    let o = dirOf(d);
    let n = u32(i32(k) + o.y * SPAN + o.x);
    return 0.5 * dot(unpack2x16float(drive[k]) + unpack2x16float(drive[n]), vec2<f32>(o));
}

// The EMF along it: the batteries', and an interface's back-EMF.
fn spanEmf(k: u32, d: u32) -> f32 {
    return spanBatteries(k, d) + faceEmf(faceOf(face[k], d));
}

fn spanNeighbor(k: u32, d: u32) -> u32 {
    let o = dirOf(d);
    return u32(i32(k) + o.y * SPAN + o.x);
}

// Loads what assemble left for the block and its ring (sig, drive, slotOf).
fn loadSpan(origin: vec2<i32>, li: u32) {
    for (var n = li; n < SPAN_CELLS; n += BLOCK_CELLS) {
        let q = origin - 1 + vec2<i32>(i32(n % u32(SPAN)), i32(n / u32(SPAN)));
        var s = 0.0;
        var e = 0u;
        var a = NO_SLOT;
        var f = 0u;
        if (inWorld(q)) {
            let i = cellIndex(q);
            s = workF(W_SIG, i);
            e = work[workSlot(W_EMF, i)];
            a = work[workSlot(W_AGG, i)];
            f = work[workSlot(W_FACE, i)];
        }
        sig[n] = s;
        drive[n] = e;
        slotOf[n] = a;
        face[n] = f;
    }
}

// A block's sums over its cells' four values (in sums, from each thread's
// own), the same for all of them.
fn blockSum(li: u32, v: vec4<f32>) -> vec4<f32> {
    sums[li] = v;
    workgroupBarrier();
    for (var s = BLOCK_CELLS / 2u; s > 0u; s = s / 2u) {
        if (li < s) {
            sums[li] += sums[li + s];
        }
        workgroupBarrier();
    }
    let total = workgroupUniformLoad(&sums[0]);
    workgroupBarrier();
    return total;
}

fn blockIndex(wid: vec2<u32>) -> u32 {
    return wid.y * blocksWide() + wid.x;
}

// ---------------------------------------------------------------- assemble

// The union-find's links (labeltile.wgsl): the block's cells', each its
// own to start with.
fn ufTile(i: u32) -> u32 {
    return atomicLoad(&parent[i]);
}

fn ufTileLower(i: u32, to: u32) -> u32 {
    return atomicMin(&parent[i], to);
}

// The states of the links of the cell at k (span index) that join a cell
// carrying ions to an electrode, by the step before's potentials (vNear;
// slotOf holding whether each carries ions).
fn faceWord(k: u32) -> u32 {
    var word = 0u;
    if (sig[k] > 0.0) {
        for (var d = 0u; d < 8u; d++) {
            let n = spanNeighbor(k, d);
            if (sig[n] > 0.0 && slotOf[k] != slotOf[n]) {
                word |= faceFrom(linkDrop(vNear[k], vNear[n], spanBatteries(k, d))) << (2u * d);
            }
        }
    }
    return word;
}

@compute @workgroup_size(8, 8)
fn assemble(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let origin = vec2<i32>(wid.xy) * BLOCK;
    for (var n = li; n < SPAN_CELLS; n += BLOCK_CELLS) {
        let q = origin - 1 + vec2<i32>(i32(n % u32(SPAN)), i32(n / u32(SPAN)));
        var s = 0.0;
        var e = 0u;
        var v = 0.0;
        var ionic = 0u;
        if (inWorld(q)) {
            let i = cellIndex(q);
            let c = cells[i];
            s = sigmaOf(c);
            e = pack2x16float(emfOf(elems[cellType(c)], wordHeading(c.x)));
            v = volts[i];
            ionic = select(0u, 1u, carriesIons(cellType(c)));
        }
        sig[n] = s;
        drive[n] = e;
        vNear[n] = v;
        // (slotOf, unused until the passes after, holds whether it carries
        // ions meanwhile.)
        slotOf[n] = ionic;
    }
    atomicStore(&parent[li], li);
    if (li == 0u) {
        atomicStore(&blockLive, 0u);
        // The step's CG starts afresh (the passes after count from here).
        if (all(wid.xy == vec2<u32>(0u))) {
            atomicStore(&tiles[STAT_COARSE], 0u);
            atomicStore(&tiles[STAT_BATTERIES], 0u);
            for (var k = 0u; k < SCALARS; k++) {
                atomicStore(&tiles[scalarSlot(k)], 0u);
            }
        }
    }
    workgroupBarrier();
    let k = spanAt(vec2<i32>(lid.xy));
    face[k] = faceWord(k);
    var d0 = 0.0;
    for (var d = 0u; d < 8u; d++) {
        d0 += spanLink(k, d);
    }
    diag[li] = d0;
    workgroupBarrier();
    // Strong links inside the block join their cells' aggregates (each link
    // once: right, down, and the two lower corners).
    if (d0 > 0.0) {
        for (var d = 0u; d < 8u; d++) {
            if (d == 1u || d == 3u || d >= 6u) {
                continue;
            }
            let q = vec2<i32>(lid.xy) + dirOf(d);
            if (any(q < vec2<i32>(0)) || any(q >= vec2<i32>(BLOCK))) {
                continue;
            }
            let l = u32(q.y * BLOCK + q.x);
            let g = spanLink(k, d);
            if (g > 0.0 && g >= THETA * sqrt(d0 * diag[l])) {
                ufTileJoin(li, l);
            }
        }
        atomicOr(&blockLive, 1u);
    }
    workgroupBarrier();
    // Each aggregate's root is its lowest cell; the roots in order take the
    // slots, the first SLOTS of them.
    let root = ufTileFind(li);
    var r = NO_SLOT;
    if (d0 > 0.0 && root == li) {
        var before = 0u;
        for (var m = 0u; m < li; m++) {
            if (diag[m] > 0.0 && atomicLoad(&parent[m]) == m) {
                before++;
            }
        }
        r = select(NO_SLOT, before, before < SLOTS);
    }
    rank[li] = r;
    workgroupBarrier();
    let p = origin + vec2<i32>(lid.xy);
    if (inWorld(p)) {
        let i = cellIndex(p);
        setWorkF(W_SIG, i, sig[k]);
        work[workSlot(W_EMF, i)] = drive[k];
        work[workSlot(W_FACE, i)] = face[k];
        work[workSlot(W_AGG, i)] = select(NO_SLOT, rank[root], d0 > 0.0);
    }
    if (li == 0u) {
        atomicStore(&tiles[tileSlot(blockIndex(wid.xy), T_LIVE)], atomicLoad(&blockLive));
    }
}

// ---------------------------------------------------------------- residual

// Where a link from a cell of slot `own` to the cell at span index n goes
// in the coarse row: the coupling's index (block offset, slot), LINK_DIAG
// for a link into no aggregate (it adds to the diagonal only), LINK_SAME
// for one inside the aggregate (nothing).
const LINK_DIAG: u32 = 0xfeu;
const LINK_SAME: u32 = 0xffu;

fn linkEntry(own: u32, lid: vec2<i32>, d: u32, n: u32) -> u32 {
    let other = slotOf[n];
    if (other == NO_SLOT) {
        return LINK_DIAG;
    }
    let q = lid + dirOf(d);
    let o = vec2<i32>(select(0, select(1, -1, q.x < 0), q.x < 0 || q.x >= BLOCK), select(0, select(1, -1, q.y < 0), q.y < 0 || q.y >= BLOCK));
    if (all(o == vec2<i32>(0)) && other == own) {
        return LINK_SAME;
    }
    return u32((o.y + 1) * 3 + o.x + 1) * SLOTS + other;
}

@compute @workgroup_size(8, 8)
fn residual(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let t = blockIndex(wid.xy);
    if (li == 0u) {
        blockState.x = f32(atomicLoad(&tiles[tileSlot(t, T_LIVE)]));
        atomicStore(&blockNodes, 0u);
    }
    let live = workgroupUniformLoad(&blockState).x != 0.0;
    if (!live) {
        if (li < TILE_WORDS - 1u) {
            atomicStore(&tiles[tileSlot(t, T_BB + li)], 0u);
        }
        return;
    }
    let origin = vec2<i32>(wid.xy) * BLOCK;
    loadSpan(origin, li);
    for (var n = li; n < SPAN_CELLS; n += BLOCK_CELLS) {
        let q = origin - 1 + vec2<i32>(i32(n % u32(SPAN)), i32(n / u32(SPAN)));
        vNear[n] = select(0.0, volts[cellIndex(q)], inWorld(q));
    }
    workgroupBarrier();
    let l = vec2<i32>(lid.xy);
    let k = spanAt(l);
    let own = slotOf[k];
    var r = 0.0;
    var amps = 0.0;
    for (var d = 0u; d < 8u; d++) {
        let g = select(0.0, spanLink(k, d), sig[k] > 0.0);
        let n = spanNeighbor(k, d);
        var entry = LINK_SAME;
        if (g > 0.0) {
            let emf = spanEmf(k, d);
            let current = g * linkDrop(vNear[k], vNear[n], emf);
            r -= current;
            amps += current * current;
            if (own != NO_SLOT) {
                entry = linkEntry(own, l, d, n);
            }
        }
        linkG[li * 8u + d] = g;
        linkTo[li * 8u + d] = entry;
    }
    vec[li] = r;
    workgroupBarrier();
    // The block's aggregates' rows: thread m takes entries m, m + 64, ...
    // of the SLOTS rows of 37 (the diagonal, the 36 couplings), and P'r.
    for (var m = li; m < SLOTS * 37u; m += BLOCK_CELLS) {
        let slot = m / 37u;
        let e = m % 37u;
        var total = 0.0;
        var used = false;
        for (var c = 0u; c < BLOCK_CELLS; c++) {
            if (slotOf[spanAt(vec2<i32>(i32(c % u32(BLOCK)), i32(c / u32(BLOCK))))] != slot) {
                continue;
            }
            used = true;
            for (var d = 0u; d < 8u; d++) {
                let to = linkTo[c * 8u + d];
                if (e == 0u) {
                    total += select(0.0, linkG[c * 8u + d], to != LINK_SAME);
                } else if (to == e - 1u) {
                    total -= linkG[c * 8u + d];
                }
            }
        }
        let node = t * SLOTS + slot;
        coarse[coarseSlot(node, C_DIAG + e)] = select(0.0, total, used);
    }
    if (li < SLOTS) {
        var fr = 0.0;
        for (var c = 0u; c < BLOCK_CELLS; c++) {
            if (slotOf[spanAt(vec2<i32>(i32(c % u32(BLOCK)), i32(c / u32(BLOCK))))] == li) {
                fr += vec[c];
            }
        }
        let node = t * SLOTS + li;
        coarse[coarseSlot(node, C_FR)] = fr;
        coarse[coarseSlot(node, C_E)] = 0.0;
    }
    let p = origin + l;
    if (inWorld(p)) {
        let i = cellIndex(p);
        setWorkF(W_R, i, r);
        setWorkF(W_P0, i, 0.0);
        setWorkF(W_P0 + 1u, i, 0.0);
        setWorkF(W_Q, i, 0.0);
    }
    // A live node for each aggregate the block has (its root's slot): the
    // coarse pass lists them in order (liveList).
    if (own != NO_SLOT && sig[k] > 0.0) {
        atomicOr(&blockNodes, 1u << own);
    }
    let total = blockSum(li, vec4<f32>(amps, r * r, 0.0, 0.0));
    if (li == 0u) {
        atomicStore(&tiles[tileSlot(t, T_NODES)], atomicLoad(&blockNodes));
        atomicStore(&tiles[tileSlot(t, T_BB)], bitcast<u32>(total.x));
        atomicStore(&tiles[tileSlot(t, T_RR)], bitcast<u32>(total.y));
        atomicStore(&tiles[tileSlot(t, T_RZ)], 0u);
        atomicStore(&tiles[tileSlot(t, T_PQ)], 0u);
    }
}

// ---------------------------------------------------------- one workgroup

var<workgroup> reduceBuf: array<vec2<f32>, COARSE_THREADS>;
var<workgroup> uniformF: f32;
var<workgroup> uniformV: vec2<f32>;
var<workgroup> uniformU: u32;

// Two sums over the workgroup at once, added up by one thread: a barrier
// is what costs here, not the adding (the coarse solve does a few a
// coarse iteration, and its iterations are what its time goes on).
fn wgSum2(li: u32, v: vec2<f32>) -> vec2<f32> {
    reduceBuf[li] = v;
    workgroupBarrier();
    if (li == 0u) {
        var total = vec2<f32>(0.0);
        for (var k = 0u; k < COARSE_THREADS; k++) {
            total += reduceBuf[k];
        }
        uniformV = total;
    }
    return workgroupUniformLoad(&uniformV);
}

fn wgSum(li: u32, v: f32) -> f32 {
    return wgSum2(li, vec2<f32>(v, 0.0)).x;
}

// Scalar k of the CG's, the same for the whole workgroup (as the barriers
// that follow on it need).
fn uniformScalar(li: u32, k: u32) -> f32 {
    if (li == 0u) {
        uniformF = scalar(k);
    }
    return workgroupUniformLoad(&uniformF);
}

var<workgroup> listAt: array<u32, COARSE_THREADS>;

// The coarse problem's live nodes (the residual pass's T_NODES) listed in
// their order, a run of blocks a thread: how many, the same for the whole
// workgroup. In the order the blocks' threads came to them, the coarse
// solve's sums over them came out otherwise from run to run, and two runs
// from one snapshot came apart (REWRITE_PLAN.md D-1).
fn liveList(li: u32) -> u32 {
    let run = (blockCount() + COARSE_THREADS - 1u) / COARSE_THREADS;
    let first = min(li * run, blockCount());
    let end = min(first + run, blockCount());
    var mine = 0u;
    for (var t = first; t < end; t++) {
        mine += countOneBits(atomicLoad(&tiles[tileSlot(t, T_NODES)]));
    }
    listAt[li] = mine;
    workgroupBarrier();
    if (li == 0u) {
        var before = 0u;
        for (var k = 0u; k < COARSE_THREADS; k++) {
            let n = listAt[k];
            listAt[k] = before;
            before += n;
        }
        uniformU = before;
    }
    let live = workgroupUniformLoad(&uniformU);
    var n = listAt[li];
    for (var t = first; t < end; t++) {
        let bits = atomicLoad(&tiles[tileSlot(t, T_NODES)]);
        for (var slot = 0u; slot < SLOTS; slot++) {
            if ((bits & (1u << slot)) != 0u) {
                atomicStore(&tiles[liveSlot(n)], t * SLOTS + slot);
                n++;
            }
        }
    }
    storageBarrier();
    workgroupBarrier();
    return live;
}

// The blocks' sums of word k.
fn tilesSum(li: u32, k: u32) -> f32 {
    var s = 0.0;
    for (var t = li; t < blockCount(); t += COARSE_THREADS) {
        s += bitcast<f32>(atomicLoad(&tiles[tileSlot(t, k)]));
    }
    return wgSum(li, s);
}

// The coarse problem's product, row `node`, of the vector in word w.
fn coarseRow(node: u32, w: u32) -> f32 {
    let block = node / SLOTS;
    let bw = blocksWide();
    let bh = blockCount() / bw;
    let bx = i32(block % bw);
    let by = i32(block / bw);
    var s = coarse[coarseSlot(node, C_DIAG)] * coarse[coarseSlot(node, w)];
    for (var o = 0u; o < 9u; o++) {
        let nx = bx + i32(o % 3u) - 1;
        let ny = by + i32(o / 3u) - 1;
        if (nx < 0 || ny < 0 || nx >= i32(bw) || ny >= i32(bh)) {
            continue;
        }
        let other = u32(ny) * bw + u32(nx);
        for (var slot = 0u; slot < SLOTS; slot++) {
            let c = coarse[coarseSlot(node, C_COUPLE + o * SLOTS + slot)];
            if (c != 0.0) {
                s += c * coarse[coarseSlot(other * SLOTS + slot, w)];
            }
        }
    }
    return s;
}

// The coarse correction: A_c e = f (f in C_R as given) by Jacobi-
// preconditioned CG over the live nodes, from nothing, to INNER_TOL. A
// connected part of the coarse problem with no link out of the aggregates
// has the constant for its null space, and f, the residual's sum over each
// aggregate, sums to nothing over it: the problem is consistent there.
fn coarseCorrection(li: u32, live: u32) {
    var rz = 0.0;
    var ff = 0.0;
    for (var n = li; n < live; n += COARSE_THREADS) {
        let node = atomicLoad(&tiles[liveSlot(n)]);
        let f = coarse[coarseSlot(node, C_R)];
        let d = coarse[coarseSlot(node, C_DIAG)];
        let z = select(0.0, f / d, d > 0.0);
        coarse[coarseSlot(node, C_P)] = z;
        coarse[coarseSlot(node, C_E)] = 0.0;
        rz += f * z;
        ff += f * f;
    }
    storageBarrier();
    let first = wgSum2(li, vec2<f32>(rz, ff));
    rz = first.x;
    ff = first.y;
    if (ff <= 0.0) {
        return;
    }
    for (var it = 0u; it < INNER_MOST; it++) {
        var pq = 0.0;
        for (var n = li; n < live; n += COARSE_THREADS) {
            let node = atomicLoad(&tiles[liveSlot(n)]);
            let q = coarseRow(node, C_P);
            coarse[coarseSlot(node, C_Q)] = q;
            pq += q * coarse[coarseSlot(node, C_P)];
        }
        pq = wgSum(li, pq);
        if (pq <= 0.0) {
            break;
        }
        let alpha = rz / pq;
        var next = 0.0;
        var rr = 0.0;
        for (var n = li; n < live; n += COARSE_THREADS) {
            let node = atomicLoad(&tiles[liveSlot(n)]);
            coarse[coarseSlot(node, C_E)] += alpha * coarse[coarseSlot(node, C_P)];
            let r = coarse[coarseSlot(node, C_R)] - alpha * coarse[coarseSlot(node, C_Q)];
            coarse[coarseSlot(node, C_R)] = r;
            let d = coarse[coarseSlot(node, C_DIAG)];
            next += r * select(0.0, r / d, d > 0.0);
            rr += r * r;
        }
        let sums = wgSum2(li, vec2<f32>(next, rr));
        next = sums.x;
        rr = sums.y;
        if (rr <= INNER_TOL * INNER_TOL * ff) {
            break;
        }
        if (li == 0u) {
            atomicAdd(&tiles[STAT_COARSE], 1u);
        }
        let beta = next / rz;
        rz = next;
        for (var n = li; n < live; n += COARSE_THREADS) {
            let node = atomicLoad(&tiles[liveSlot(n)]);
            let d = coarse[coarseSlot(node, C_DIAG)];
            let r = coarse[coarseSlot(node, C_R)];
            coarse[coarseSlot(node, C_P)] = select(0.0, r / d, d > 0.0) + beta * coarse[coarseSlot(node, C_P)];
        }
        storageBarrier();
    }
}

// The coarse pass. The first of the step: |b| and |r|, whether it is done
// already, and the correction for r. Then, after each direction: alpha =
// r.z / p.q, unless what the last precondition left is close enough (done),
// and the correction for r - alpha q.
@compute @workgroup_size(64)
fn coarseSolve(@builtin(local_invocation_index) li: u32) {
    if (li == 0u) {
        uniformU = atomicLoad(&tiles[scalarSlot(S_STAGE)]) | (atomicLoad(&tiles[scalarSlot(S_DONE)]) << 16u);
    }
    let state = workgroupUniformLoad(&uniformU);
    let stage = state & 0xffffu;
    if ((state >> 16u) != 0u) {
        return;
    }
    var live = 0u;
    if (stage == 0u) {
        live = liveList(li);
        if (li == 0u) {
            atomicStore(&tiles[scalarSlot(S_LIVE)], live);
        }
    } else {
        if (li == 0u) {
            uniformU = atomicLoad(&tiles[scalarSlot(S_LIVE)]);
        }
        live = workgroupUniformLoad(&uniformU);
    }
    var alpha = 0.0;
    var done = false;
    if (stage == 0u) {
        let bb = tilesSum(li, T_BB);
        let rr = tilesSum(li, T_RR);
        done = bb <= 0.0 || rr <= TOL * TOL * bb;
        if (li == 0u) {
            setScalar(S_BB, bb);
            setScalar(S_RR, rr);
        }
    } else {
        let pq = tilesSum(li, T_PQ);
        let bb = uniformScalar(li, S_BB);
        let rr = uniformScalar(li, S_RR);
        done = rr <= TOL * TOL * bb || pq <= 0.0;
        alpha = select(uniformScalar(li, S_RZ) / pq, 0.0, done);
    }
    for (var n = li; n < live; n += COARSE_THREADS) {
        let node = atomicLoad(&tiles[liveSlot(n)]);
        coarse[coarseSlot(node, C_R)] = coarse[coarseSlot(node, C_FR)] - select(0.0, alpha * coarse[coarseSlot(node, C_FQ)], stage > 0u);
    }
    storageBarrier();
    workgroupBarrier();
    if (!done) {
        coarseCorrection(li, live);
    }
    if (li == 0u) {
        setScalar(S_ALPHA, alpha);
        atomicStore(&tiles[scalarSlot(S_DONE)], select(0u, 1u, done));
        atomicStore(&tiles[scalarSlot(S_STAGE)], stage + 1u);
        atomicStore(&tiles[STAT_ITERATIONS], select(stage, stage - 1u, done && stage > 0u));
        atomicStore(&tiles[STAT_LIVE], live);
        let bb = scalar(S_BB);
        atomicStore(&tiles[STAT_RESIDUAL], bitcast<u32>(select(0.0, sqrt(scalar(S_RR) / bb), bb > 0.0)));
    }
}

// The reduce pass: beta = r.z / last r.z (none the first time), and |r|²,
// from what the last precondition left; the direction goes to the other of
// W_P0, W_P1.
@compute @workgroup_size(64)
fn reduce(@builtin(local_invocation_index) li: u32) {
    if (li == 0u) {
        uniformU = atomicLoad(&tiles[scalarSlot(S_DONE)]);
    }
    if (workgroupUniformLoad(&uniformU) != 0u) {
        return;
    }
    let rz = tilesSum(li, T_RZ);
    let rr = tilesSum(li, T_RR);
    if (li == 0u) {
        let last = scalar(S_RZ);
        setScalar(S_BETA, select(rz / last, 0.0, last == 0.0));
        setScalar(S_RZ, rz);
        setScalar(S_RR, rr);
        atomicStore(&tiles[scalarSlot(S_PARITY)], 1u - atomicLoad(&tiles[scalarSlot(S_PARITY)]));
    }
}

// --------------------------------------------------------- precondition

// The block's state from the CG's scalars (blockState: live, done, `which`
// scalar, parity), the same for all its threads.
fn loadState(t: u32, li: u32, which: u32) -> vec4<f32> {
    if (li == 0u) {
        blockState = vec4<f32>(f32(atomicLoad(&tiles[tileSlot(t, T_LIVE)])), f32(atomicLoad(&tiles[scalarSlot(S_DONE)])),
            scalar(which), f32(atomicLoad(&tiles[scalarSlot(S_PARITY)])));
    }
    return workgroupUniformLoad(&blockState);
}

// V += alpha p,
// r -= alpha q; then z = e on each aggregate plus SWEEPS symmetric
// Gauss-Seidel sweeps on the block's r (colours by the cell's x and y
// parity, forward then back, so the sweep is symmetric, as CG needs; the
// four colours keep corner neighbours apart), its ring held at zero; the
// block's r.z, |r|², and P'r for its aggregates.
@compute @workgroup_size(8, 8)
fn precondition(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let t = blockIndex(wid.xy);
    let state = loadState(t, li, S_ALPHA);
    if (state.x == 0.0 || state.y != 0.0) {
        return;
    }
    let alpha = state.z;
    let parity = u32(state.w);
    let origin = vec2<i32>(wid.xy) * BLOCK;
    loadSpan(origin, li);
    for (var n = li; n < SPAN_CELLS; n += BLOCK_CELLS) {
        vec[n] = 0.0;
    }
    let l = vec2<i32>(lid.xy);
    let k = spanAt(l);
    let p = origin + l;
    var r = 0.0;
    if (inWorld(p)) {
        let i = cellIndex(p);
        r = workF(W_R, i);
        if (alpha != 0.0) {
            volts[i] += alpha * workF(W_P0 + parity, i);
            r -= alpha * workF(W_Q, i);
        }
    }
    workgroupBarrier();
    var d0 = 0.0;
    if (sig[k] > 0.0) {
        for (var d = 0u; d < 8u; d++) {
            d0 += spanLink(k, d);
        }
    }
    let colour = u32(l.x & 1) + 2u * u32(l.y & 1);
    for (var s = 0u; s < SWEEPS; s++) {
        for (var c = 0u; c < 8u; c++) {
            let now = select(c, 7u - c, c >= 4u);
            if (colour == now && d0 > 0.0) {
                var sum = r;
                for (var d = 0u; d < 8u; d++) {
                    let q = l + dirOf(d);
                    if (all(q >= vec2<i32>(0)) && all(q < vec2<i32>(BLOCK))) {
                        sum += spanLink(k, d) * vec[spanNeighbor(k, d)];
                    }
                }
                vec[k] = sum / d0;
            }
            workgroupBarrier();
        }
    }
    var z = vec[k];
    let own = slotOf[k];
    if (own != NO_SLOT && d0 > 0.0) {
        z += coarse[coarseSlot(t * SLOTS + own, C_E)];
    }
    if (d0 <= 0.0) {
        z = 0.0;
        r = 0.0;
    }
    if (inWorld(p)) {
        let i = cellIndex(p);
        setWorkF(W_R, i, r);
        setWorkF(W_Z, i, z);
    }
    vec2nd[li] = r;
    let total = blockSum(li, vec4<f32>(r * z, r * r, 0.0, 0.0));
    if (li < SLOTS) {
        var fr = 0.0;
        for (var c = 0u; c < BLOCK_CELLS; c++) {
            if (slotOf[spanAt(vec2<i32>(i32(c % u32(BLOCK)), i32(c / u32(BLOCK))))] == li) {
                fr += vec2nd[c];
            }
        }
        coarse[coarseSlot(t * SLOTS + li, C_FR)] = fr;
    }
    if (li == 0u) {
        atomicStore(&tiles[tileSlot(t, T_RZ)], bitcast<u32>(total.x));
        atomicStore(&tiles[tileSlot(t, T_RR)], bitcast<u32>(total.y));
    }
}

// ------------------------------------------------------------ direction

// p = z + beta p over the block and its ring (the ring's worked out here
// from its z and old p, so no block waits on another's), into the other of
// W_P0, W_P1; q = A p for the block's cells; its p.q and P'q.
@compute @workgroup_size(8, 8)
fn direction(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let t = blockIndex(wid.xy);
    let state = loadState(t, li, S_BETA);
    if (state.x == 0.0 || state.y != 0.0) {
        return;
    }
    let beta = state.z;
    let parity = u32(state.w);
    let origin = vec2<i32>(wid.xy) * BLOCK;
    loadSpan(origin, li);
    for (var n = li; n < SPAN_CELLS; n += BLOCK_CELLS) {
        let q = origin - 1 + vec2<i32>(i32(n % u32(SPAN)), i32(n / u32(SPAN)));
        var pn = 0.0;
        if (inWorld(q)) {
            let i = cellIndex(q);
            pn = workF(W_Z, i);
            if (beta != 0.0) {
                pn += beta * workF(W_P0 + 1u - parity, i);
            }
        }
        vec[n] = pn;
    }
    workgroupBarrier();
    let l = vec2<i32>(lid.xy);
    let k = spanAt(l);
    var q = 0.0;
    if (sig[k] > 0.0) {
        for (var d = 0u; d < 8u; d++) {
            q += spanLink(k, d) * (vec[k] - vec[spanNeighbor(k, d)]);
        }
    }
    let p = origin + l;
    if (inWorld(p)) {
        let i = cellIndex(p);
        setWorkF(W_P0 + parity, i, vec[k]);
        setWorkF(W_Q, i, q);
    }
    vec2nd[li] = q;
    let total = blockSum(li, vec4<f32>(vec[k] * q, 0.0, 0.0, 0.0));
    if (li < SLOTS) {
        var fq = 0.0;
        for (var c = 0u; c < BLOCK_CELLS; c++) {
            if (slotOf[spanAt(vec2<i32>(i32(c % u32(BLOCK)), i32(c / u32(BLOCK))))] == li) {
                fq += vec2nd[c];
            }
        }
        coarse[coarseSlot(t * SLOTS + li, C_FQ)] = fq;
    }
    if (li == 0u) {
        atomicStore(&tiles[tileSlot(t, T_PQ)], bitcast<u32>(total.x));
    }
}

// ------------------------------------------------------------- current

const CBLOCK: i32 = 16;
const CRING: i32 = 18;

var<workgroup> csig: array<f32, 324>;
var<workgroup> cdrive: array<u32, 324>;
var<workgroup> cvolt: array<f32, 324>;
var<workgroup> ions: array<u32, 324>;    // whether it carries ions (carriesIons)
var<workgroup> cface: array<u32, 324>;   // its links' states (W_FACE)
var<workgroup> conducts: u32;

fn cring(q: vec2<i32>) -> i32 {
    return (q.y + 1) * CRING + q.x + 1;
}

fn loadRing(origin: vec2<i32>, n: u32) {
    let q = origin - 1 + vec2<i32>(i32(n % u32(CRING)), i32(n / u32(CRING)));
    var s = 0.0;
    var e = 0u;
    var v = 0.0;
    var ionic = 0u;
    var f = 0u;
    if (inWorld(q)) {
        let i = cellIndex(q);
        let c = cells[i];
        let t = cellType(c);
        s = sigmaOf(c);
        e = pack2x16float(emfOf(elems[t], wordHeading(c.x)));
        v = volts[i];
        ionic = select(0u, 1u, carriesIons(t));
        f = work[workSlot(W_FACE, i)];
    }
    csig[n] = s;
    cdrive[n] = e;
    cvolt[n] = v;
    ions[n] = ionic;
    cface[n] = f;
}

// What a cell's links add up to.
struct Flow {
    outward: f32,   // current flowing out of it, A
    inward: f32,    // and in
    heat: f32,      // I² R of its own halves of the links, W
    density: vec2<f32>,
    cathode: f32,   // current leaving an electrolyte into electrodes, A
}

var<private> acc: Flow;

// The link from the cell at k (ring index) its way w (dirOf).
fn addLink(k: i32, w: u32) {
    let d = dirOf(w);
    let n = k + d.y * CRING + d.x;
    if (w >= 4u && (csig[k + d.x] > 0.0 || csig[k + d.y * CRING] > 0.0)) {
        return;
    }
    let state = faceOf(cface[k], w);
    let s = csig[k];
    let g = seriesG(s, csig[n]);
    if (g <= 0.0 || state == FACE_OPEN) {
        return;
    }
    let emf = 0.5 * dot(unpack2x16float(cdrive[k]) + unpack2x16float(cdrive[n]), vec2<f32>(d)) + faceEmf(state);
    var i = g * linkDrop(cvolt[k], cvolt[n], emf);
    // (An interface carries current only the way its state has it: one the
    // step's potentials would run backwards turns open next step.)
    if (state == FACE_OUT) {
        i = max(i, 0.0);
    } else if (state == FACE_IN) {
        i = min(i, 0.0);
    }
    acc.outward += max(i, 0.0);
    acc.inward += max(-i, 0.0);
    acc.heat += i * i * 0.5 / s;
    acc.density += 0.5 * i * vec2<f32>(d);
    if (state != FACE_NONE && ions[k] != 0u) {
        acc.heat += abs(i) * (HALF_CELL - HALF_THERMONEUTRAL);
        acc.cathode += max(i, 0.0);
    }
}

// Whether the 16 x 16 block wid of the current pass has any of the solve's
// live blocks in it (as assemble found them this step).
fn blockConducts(wid: vec2<u32>) -> bool {
    let bw = blocksWide();
    let bh = blockCount() / bw;
    var any = 0u;
    for (var k = 0u; k < 4u; k++) {
        let b = wid * 2u + vec2<u32>(k & 1u, k >> 1u);
        if (b.x < bw && b.y < bh) {
            any |= atomicLoad(&tiles[tileSlot(b.y * bw + b.x, T_LIVE)]);
        }
    }
    return any != 0u;
}

// What the current pass leaves for the effects pass, per cell (W_EFFECT):
// the rise in its stored temperature (f32 bits, never negative), and SPLIT
// if electrolysis turns it into hydrogen.
const SPLIT: u32 = 0x80000000u;

// Per cell of the 16 x 16 blocks with conductors. Joule heat is to go into
// the cell's stored temperature, as the update pass's heat does (so a metal
// wire takes in its latent heat at 1500 °C and then melts); electrolysis is
// to turn an electrolyte cell beside the cathode into hydrogen, consuming
// it, the way boiling turns a liquid cell into steam. Both are only worked
// out here, and left for the effects pass.
@compute @workgroup_size(16, 16)
fn current(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    // Every battery counts, whether anything conducts by it yet or not: a
    // lone cell of one, its wire still to be drawn, keeps the passes on
    // (sim.js's electricLive, which nothing else turns back on but a
    // battery painted).
    let p = vec2<i32>(gid.xy);
    if (inWorld(p) && elems[cellType(cells[cellIndex(p)])].emf != 0.0) {
        atomicAdd(&tiles[STAT_BATTERIES], 1u);
    }
    if (li == 0u) {
        conducts = select(0u, 1u, blockConducts(wid.xy));
    }
    if (workgroupUniformLoad(&conducts) == 0u) {
        return;
    }
    let origin = vec2<i32>(wid.xy) * CBLOCK;
    loadRing(origin, li);
    if (li + 256u < u32(CRING * CRING)) {
        loadRing(origin, li + 256u);
    }
    workgroupBarrier();

    if (!inWorld(p)) {
        return;
    }
    let i = cellIndex(p);
    let k = cring(p - origin);
    if (csig[k] <= 0.0) {
        work[workSlot(W_EFFECT, i)] = 0u;
        return;
    }
    let v = cells[i];
    let t = cellType(v);
    let e = elems[t];
    acc = Flow();
    for (var w = 0u; w < 8u; w++) {
        addLink(k, w);
    }

    // Only current passing through a cell counts. While the potential is
    // still settling after a change (a few steps), current flows into a
    // cell that doesn't flow out (or the other way round), as if charging
    // it up; in a real conductor that is over in nanoseconds and heats
    // nothing, so the share that doesn't pass through is left out.
    let through = min(acc.inward, acc.outward);
    let peak = max(acc.inward, acc.outward);
    let share = select(0.0, through / peak, peak > 0.0);
    // (Within f16, which a dead short can pass.)
    flow[i] = pack2x16float(clamp(acc.density * share, vec2<f32>(-MAX_FLOW), vec2<f32>(MAX_FLOW)));

    // The heat for as long as a step stands for by heat's clock where the
    // cell is (clock.wgsl), as its cooling goes: on the day's clock as long
    // as it is cool, in real time once it (or something near it) glows. A
    // wire settles where the heat it makes and loses balance, either way.
    // But no longer than lets the heat raise the cell MAX_JOULE_RISE, real
    // time at the shortest: the step stands for that long there, and the
    // next step's clock round it takes it in (heated), so a short melts its
    // wire through in a few steps with all its heat. As joules (heatin.wgsl).
    let scale = faceShare(P.dx) * faceShare(P.dx);
    var tau = clockStep(textureLoad(clock, vec2<u32>(p) / 2u, 0).x, P.dayDt, P.clockSpan);
    var power = 0.0;
    if (e.emf == 0.0 && (e.flags & F_FIXED_TEMP) == 0u) {
        var g = noGas();
        if (gasHeat(t)) {
            g = gasFrom(cells[gasIndex(P.size.x * P.size.y, i)], i);
        }
        power = acc.heat * share * share * scale;
        let shortest = max(MAX_JOULE_RISE * heatCapIn(cells[i], g) / max(power, 1e-30), P.dt);
        if (shortest < tau) {
            tau = shortest;
            let real = clamp(log2(tau / P.dayDt) / P.clockSpan, 0.0, 1.0);
            let lw = (P.size.x + 1u) / 2u;
            atomicMax(&heated[(u32(p.y) / 2u) * lw + u32(p.x) / 2u], bitcast<u32>(real));
        }
    }
    let joules = power * tau;

    // Electrolysis at the cathode, where current leaves the electrolyte into
    // metal. Faraday's law: the gas given off goes with the charge passed, on
    // the same clock as the current's heat (COMPATIBILITY.md
    // `new-electrolysis`: on dt, a wire heated some 360 times faster than it
    // made gas at the day's speed). A whole cell of water at a time, at the
    // rate it is made a second, as a chance over the step (chanceOver).
    // (At the anode, brine's chlorine stays dissolved, as hypochlorite.)
    // No current reaches an electrode while less than the back-EMF drives
    // it (HALF_CELL), so none is too little to split water.
    let cathode = acc.cathode * share;
    var split = false;
    if (cathode > 0.0) {
        seedRng(i, P.frame, P.seed ^ 0xe1ec7u);
        split = rnd() < chanceOver(e.electrolysis * cathode * scale, tau);
    }
    work[workSlot(W_EFFECT, i)] = bitcast<u32>(joules) | select(0u, SPLIT, split);
}

// The current pass's heat (as energy, heatin.wgsl) and hydrogen into the
// cells, over the same blocks. A cell split into hydrogen takes no heat: the
// energy the current spends there goes into the reaction. Fresh water is
// hydrogen at once, WATER_HYDROGEN cells of it (Faraday), holding the
// water's heat; water carrying a solute is marked to split (FL_SPLIT),
// which the pairs pass does once it has handed the solute on.
@compute @workgroup_size(16, 16)
fn effects(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let p = vec2<i32>(gid.xy);
    if (!inWorld(p) || !blockConducts(wid.xy)) {
        return;
    }
    let i = cellIndex(p);
    let effect = work[workSlot(W_EFFECT, i)];
    if (effect == 0u) {
        return;
    }
    if ((effect & SPLIT) != 0u) {
        var c = unpackCell(cells[i]);
        if (holdsSolute(c.t) && soluteOf(c) != 0u) {
            c.flags |= FL_SPLIT;
            cells[i] = packCell(c);
            return;
        }
        // As much hydrogen as the cell held water, with its heat (pairs.wgsl's
        // split: a drawn cell of a particle made a whole cell's).
        let gas = Gas(0.0, 0.0, 0.0, WATER_HYDROGEN * liquidAmount(c));
        cells[i] = packCell(makeCell(HYDROGEN, AMBIENT_T + cellHeatCap(c, noGas()) * (c.temp - AMBIENT_T) / gasHeatCap(gas)));
        let w = packGas(gas);
        let k = gasIndex(P.size.x * P.size.y, i);
        if (gasHigh(i)) {
            cells[k].z = w.x;
            cells[k].w = w.y;
        } else {
            cells[k].x = w.x;
            cells[k].y = w.y;
        }
    } else {
        depositHeat(i, bitcast<f32>(effect));
    }
}
