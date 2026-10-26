// Stress in standing structure (REWRITE_PLAN.md 8-2; audit/04 §5.4's C2):
// quasi-static linear elasticity over static structure (structural solids
// not loose, which bodies.wgsl's anchoring holds), and failure where the
// stress passes the material's strength.
//
// One bilinear (Q4) plane-stress element per cell, its nodes at the cells'
// corners, two unknowns a node: its displacement over the cell's size dx, so
// strains are differences between nodes and stresses come out in pascals.
// One Poisson's ratio for every material (NU), so one unit stiffness matrix
// (KE), times the element's Young's modulus (Elem.youngs). A node on a face
// between structure and an anchor (the world's edge, a solid that isn't
// structural, settled powder under it, soil for a plant: anchorFace, as the
// anchoring takes them) is held still; a piece joined face to face and held
// by a face is rigid, so the system is never singular.
//
// The loads, per node, in Pa (the force over dx times the cell's depth dx):
// - its weight, ρ g dx / 4 from each cell it is a corner of (ρ the element's
//   real density, g the total pull at the cell in m/s²);
// - on each face meeting a fluid, the fluid's pressure p, p / 2 to each of
//   the face's two nodes, pushing in: the air's (gauge), the liquid's
//   (fluid.wgsl's, at the face) over its air's, a pile's load on its top
//   (update.wgsl's, LATERAL of it beside one) over its air's.
// - its thermal strain (8-5: thermalStress, and the skin's shock,
//   skinStress, judged apart).
// Bodies resting on it are left out (later).
//
// The solve is conjugate gradients preconditioned by each node's own 2 x 2
// block (block Jacobi), its state (the displacement, residual, direction)
// carried from step to step: each step works out the residual afresh from
// the loads as they are now and carries on along the last direction (as
// nonlinear CG does), STRESS_ITERATIONS a step (sim.js), which return at
// once once it has converged.
// Restarting it every step stalls: a slender beam's bending converges only
// all at once, some 3 iterations a cell of its length, and a restarted CG
// never gets there. When the structure changes (a node joins, leaves or is
// held differently), it starts over along the residual.
//
// Failure: each step, each cell's
// stress at its four corners (where a beam one cell thick bends most: at
// its middle a bilinear element shows none of its bending), its principal
// stresses there, and Mohr-Coulomb's σ1 / ft - σ3 / fc ≥ 1 (tension
// positive: confinement strengthens it, and compression from all round
// never breaks it; the in-plane stresses alone, as if confined across).
// The most overstressed cell of its neighbourhood breaks (its step is
// marked in the bodies buffer, and update.wgsl crumbles it into its
// breakTo the next step), which changes the structure: the anchoring
// labels it again and the solve converges again before the next breaks,
// so a crack runs and a structure comes down a piece at a time.
//
// Each step (sim.js):
//   setup       per node: whether it is held or free, its preconditioner,
//               its load; and per tile of 8 x 8 nodes, whether any is in
//               structure
//   residual    r = f - K u, z = M⁻¹ r, partial sums of r·z and f·M⁻¹f
//   begin       one workgroup: the list of the tiles with structure, in
//               order; the sums; the direction's β (0 to start over);
//               whether it has converged
//   then, STRESS_ITERATIONS times (all returning at once once converged):
//   apply       r·z (each workgroup summing the tiles' partials itself),
//               whether converged, β; the direction p = z + β p (into the
//               other of two, over its tile and the ring round it), q = K p,
//               partial p·q and r·p
//   update      α = r·p / p·q (summed likewise), u += α p (in fixed
//               point), r -= α q, z = M⁻¹ r, partial r·z
//   then:
//   measure     per cell: its failure index (in work, for the next pass)
//   crack       per cell: the most overstressed of its neighbourhood marked
//   finish      one thread: the counters for next step zeroed

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> elems: array<Elem>;
@group(0) @binding(2) var<storage, read> cells: array<vec4<u32>>;
// The air as its pass left it this step (airsample.wgsl): the pressure on
// faces meeting gas.
@group(0) @binding(3) var<storage, read> air: array<vec4<f32>>;
// The liquid's grid (fluid.wgsl): pressure, -, density (> 0 liquid).
@group(0) @binding(4) var<storage, read> field: array<vec4<f32>>;
// The piles' loads as the update pass just passed them on (update.wgsl's
// loadOut).
@group(0) @binding(5) var<storage, read> loads: array<f32>;
// The bodies buffer: structure's words (common.wgsl's structureBase), whose
// break marks this writes.
@group(0) @binding(6) var<storage, read_write> bodies: array<atomic<u32>>;
// STRESS_NODE_WORDS vec4s a node, (P.size.x + 1) x (P.size.y + 1) of them:
// the two directions; the residual and the preconditioned residual; K times
// the direction and the load; the preconditioner (the inverse of its 2 x 2
// block: xx, xy, yy) and its flags (N_*, as f32 bits).
@group(0) @binding(7) var<storage, read_write> nodes: array<vec4<f32>>;
// STRESS_STATS words read back (SS_*), STRESS_SCALARS of the CG's state
// (S_*), three partial sums a tile, a word a tile (whether it has a node in
// structure), and two words a cell (its failure index, f32 bits; how many
// judgements running it has been past its strength).
@group(0) @binding(8) var<storage, read_write> work: array<atomic<u32>>;
// The pull of the world's masses per light cell of 2 x 2 cells, cells/step²
// (gravity.wgsl).
@group(0) @binding(9) var gravity: texture_2d<f32>;
// Each node's displacement, x and y, as 64-bit fixed point in units of
// 2^-40 cell (FIX: high word, signed, then low): what the strains are
// differences of, exactly. In an f32 a tip 0.02 cells out is held to 2e-9
// cells, which in stone's 50 GPa is some 100 Pa, more than a cell's weight
// presses with, so the residual could never come under the loads; and a
// double float's error terms, which are nothing in exact arithmetic, a GPU
// compiler rearranging floating point (as they do, unasked) dropped, even
// through a 1 it couldn't know was one: integers it can't rearrange.
@group(0) @binding(10) var<storage, read_write> disp: array<vec4<u32>>;

const NU: f32 = 0.25;
// The unit square's stiffness for E = 1 and NU, plane stress, nodes in the
// order (0, 0), (1, 0), (0, 1), (1, 1) (x + 2 y), two unknowns each, x and
// y (y down): 2 x 2 Gauss points over the bilinear shape functions.
const KE = array<f32, 64>(
    0.488888889, 0.166666667, -0.288888889, -0.033333333, 0.044444444, 0.033333333, -0.244444444, -0.166666667,
    0.166666667, 0.488888889, 0.033333333, 0.044444444, -0.033333333, -0.288888889, -0.166666667, -0.244444444,
    -0.288888889, 0.033333333, 0.488888889, -0.166666667, -0.244444444, 0.166666667, 0.044444444, -0.033333333,
    -0.033333333, 0.044444444, -0.166666667, 0.488888889, 0.166666667, -0.244444444, 0.033333333, -0.288888889,
    0.044444444, -0.033333333, -0.244444444, 0.166666667, 0.488888889, -0.166666667, -0.288888889, 0.033333333,
    0.033333333, -0.288888889, 0.166666667, -0.244444444, -0.166666667, 0.488888889, -0.033333333, 0.044444444,
    -0.244444444, -0.166666667, 0.044444444, 0.033333333, -0.288888889, -0.033333333, 0.488888889, 0.166666667,
    -0.166666667, -0.244444444, -0.033333333, -0.288888889, 0.033333333, 0.044444444, 0.166666667, 0.488888889,
);

const N_ACTIVE: u32 = 1u;  // a corner of structure
const N_FIXED: u32 = 2u;   // held still

const S_RZ: u32 = 0u;       // r·z at the step's start
const S_RZ_PREV: u32 = 1u;  // r·z the last direction was made with (the next step's β)
const S_FZ: u32 = 2u;       // f·M⁻¹f, what converged is measured against
const S_BETA: u32 = 4u;     // the first direction's β (begin)
const S_DONE: u32 = 5u;     // u32: converged (the passes after return at once)
const S_DIR: u32 = 6u;      // u32: which of the two directions is the last made, at the step's start
const S_CHANGED: u32 = 7u;  // u32: nodes whose flags changed this step (start over)
const S_NODES: u32 = 8u;    // u32: nodes in structure, counted this step
const S_ITERATIONS: u32 = 9u;  // u32: updates this step
const S_JUDGE: u32 = 10u;   // u32: finite, to judge what breaks by
const S_LIVE: u32 = 11u;    // u32: tiles with structure, on the list (listSlot)
const_assert S_LIVE < STRESS_SCALARS;

// A tile's partial sums: r·z after an even and an odd iteration's update
// (the step's start's in the odd one's), p·Kp, r·p, f·M⁻¹f, and a spare.
const P_RZ0: u32 = 0u;
const P_RZ1: u32 = 1u;
const P_PQ: u32 = 2u;
const P_RP: u32 = 3u;
const P_FZ: u32 = 4u;
const P_SPARE: u32 = 5u;
const PARTIALS: u32 = 6u;

// What breaks is judged on the solve as far as it has got, each step (but on
// a solve gone non-finite). Judged only once converged, a blast was never
// judged at all: its loads, some 0.7 MPa on a wall 6 cells from the charge,
// change completely from step to step, and the solve (a wall 20 cells tall
// takes some 80 iterations from rest) never caught up; nor while its
// residual was under twice the loads (the loads taken as at least 1 kPa a
// node), as CG's residual, unlike its error, isn't monotone: it grew a
// hundredfold on its way while the solve still closed in (the CPU prototype
// in f64 alike), and the wall was judged once, at 1.7 times its strength,
// and never again. On its way the solve mostly falls short (a wall's
// bending builds up over the iterations, its loaded face pressed in first),
// so what is far past its strength shows long before; and a cell breaks only
// once it has been past it two judgements running (streakSlot), as the
// solve may overshoot for a step.
const RESTART_GROWTH: f32 = 4.0;

const TILE: i32 = 8;
const RING: i32 = TILE + 2;        // a tile of nodes and the ring round it
const RING_NODES: u32 = 100u;
const CELLS_SPAN: i32 = TILE + 1;  // the cells with a corner in the tile
const SPAN_CELLS: u32 = 81u;
const REDUCE_THREADS: u32 = 256u;

const PA_PER_AIR: f32 = 101325.0 / P_ATM;

// ---------------------------------------------------------------- layout

fn nodesWide() -> u32 {
    return P.size.x + 1u;
}

fn inNodes(n: vec2<i32>) -> bool {
    return all(n >= vec2<i32>(0)) && n.x <= i32(P.size.x) && n.y <= i32(P.size.y);
}

fn nodeIndex(n: vec2<i32>) -> u32 {
    return u32(n.y) * nodesWide() + u32(n.x);
}

fn nodeSlot(n: u32, k: u32) -> u32 {
    return n * STRESS_NODE_WORDS + k;
}

fn tilesWide() -> u32 {
    return (nodesWide() + u32(TILE) - 1u) / u32(TILE);
}

fn tileCount() -> u32 {
    return tilesWide() * ((P.size.y + 1u + u32(TILE) - 1u) / u32(TILE));
}

fn statSlot(k: u32) -> u32 {
    return k;
}

fn scalarSlot(k: u32) -> u32 {
    return STRESS_STATS + k;
}

fn partialSlot(t: u32, k: u32) -> u32 {
    return STRESS_STATS + STRESS_SCALARS + PARTIALS * t + k;
}

fn liveSlot(t: u32) -> u32 {
    return partialSlot(tileCount(), 0u) + t;
}

// The list of the tiles with structure (S_LIVE of them).
fn listSlot(i: u32) -> u32 {
    return liveSlot(tileCount()) + i;
}

fn failSlot(i: u32) -> u32 {
    return listSlot(tileCount()) + i;
}

// How many judgements running cell i has been past its strength.
fn streakSlot(i: u32) -> u32 {
    return failSlot(P.size.x * P.size.y) + i;
}

fn scalarF(k: u32) -> f32 {
    return bitcast<f32>(atomicLoad(&work[scalarSlot(k)]));
}

fn setScalarF(k: u32, v: f32) {
    atomicStore(&work[scalarSlot(k)], bitcast<u32>(v));
}

fn scalarU(k: u32) -> u32 {
    return atomicLoad(&work[scalarSlot(k)]);
}

fn setScalarU(k: u32, v: u32) {
    atomicStore(&work[scalarSlot(k)], v);
}

fn inWorld(c: vec2<i32>) -> bool {
    return all(c >= vec2<i32>(0)) && all(c < vec2<i32>(P.size));
}

fn cellIndex(c: vec2<i32>) -> u32 {
    return u32(c.y) * P.size.x + u32(c.x);
}

// ---------------------------------------------------------------- cells

fn structural(v: vec4<u32>) -> bool {
    let t = cellType(v);
    return elems[t].kind == KIND_SOLID && elems[t].breakTo != NO_ELEM;
}

// Whether c is standing structure in the solve: structural, not loose, and
// not let go by the anchoring (its support run out, frac.x <= 0).
fn standingAt(c: vec2<i32>) -> bool {
    if (!inWorld(c)) {
        return false;
    }
    let v = cells[cellIndex(c)];
    return structural(v) && !cellHas(v, FL_LOOSE) && cellFrac(v).x > 0.0;
}

fn youngsAt(c: vec2<i32>) -> f32 {
    return select(0.0, elems[cellType(cells[cellIndex(c)])].youngs, standingAt(c));
}

// The stress (Pa) a standing cell holding v would be under from its thermal
// strain, held still all round (REWRITE_PLAN.md 8-5): E α (T - T0) / (1 -
// ν), alike every way in its plane (plane stress under a strain α (T - T0)
// each way), T0 where it is unstressed: 20 °C as made, or its softening
// point once it has been past it (FL_SOFTENED: what it set at), and no
// hotter than that (past it, the stress eases as fast as it builds). So it
// pushes out on its corners as much as what holds them pushes back (setup),
// and the stress in it is what the solve finds less this (measure): heated
// against what holds it, it is squeezed; quenched on one side, that side
// shrinks against the rest and is pulled, and cracks (thermal shock,
// Kingery 1955, "Factors affecting thermal stress resistance of ceramic
// materials", J. Am. Ceram. Soc. 38, 3-15).
fn thermalStress(v: vec4<u32>) -> f32 {
    let e = elems[cellType(v)];
    let soft = softeningT(e);
    let unstressed = select(AMBIENT_T, soft, cellHas(v, FL_SOFTENED));
    return e.youngs * e.expansionLinear * (min(cellTemp(v), soft) - unstressed) / (1.0 - NU);
}

// The down step at cell c (pull.wgsl's light cell's).
fn downOfCell(c: vec2<i32>) -> vec2<i32> {
    return downAt(c);
}

// Whether standing structure at c meets an anchor across its face toward q
// (bodies.wgsl's anchorFace).
fn anchorFace(c: vec2<i32>, q: vec2<i32>) -> bool {
    if (!inWorld(q)) {
        return true;
    }
    let v = cells[cellIndex(q)];
    let t = cellType(v);
    let e = elems[t];
    if (e.kind == KIND_SOLID && e.breakTo == NO_ELEM) {
        return true;
    }
    if ((holdsSoilWater(t) || t == MUD) && cellType(cells[cellIndex(c)]) == PLANT) {
        return true;
    }
    let down = downOfCell(c);
    let d = q - c;
    let under = (d.x != 0 && d.x == down.x) || (d.y != 0 && d.y == down.y);
    return under && e.kind == KIND_POWDER && (e.flags & F_FLUFF) == 0u && length(cellVel(v)) < SETTLED_SPEED;
}

// The total pull at cell c, m/s².
fn pullAtCell(c: vec2<i32>) -> vec2<f32> {
    return pullAt(vec2<f32>(c) + 0.5) * P.dx / (P.dt * P.dt);
}

// The pressure (Pa, gauge) on standing structure at c across its face toward
// q (n = q - c, the face's outward normal), from what is at q: none from
// structure or a body's cell.
fn faceLoad(c: vec2<i32>, q: vec2<i32>) -> f32 {
    let j = cellIndex(q);
    let v = cells[j];
    let e = elems[cellType(v)];
    if (e.kind == KIND_SOLID) {
        return 0.0;
    }
    // The air's, against the open air's at that height (as update.wgsl's
    // loadAcross): structure in the open bears none of it, and a sealed
    // room's walls their gauge pressure. (Against an atmosphere, the air
    // solve's few tens of Pa of jitter a step were as much as a stone
    // cell's weight puts on each node, some 70 Pa, and the solve never came
    // under the loads.)
    let dims = airDims(P.size);
    let open = (openAirAtm(altitudeAt(vec2<f32>(q) + 0.5)) - 1.0) * P_ATM;
    let airP = (airAt(q / i32(AIR_CELL), dims).x - open) * PA_PER_AIR;
    if (e.kind == KIND_GAS) {
        return airP;
    }
    let n = vec2<f32>(q - c);
    if (e.kind == KIND_LIQUID) {
        let f = field[2u * j];
        if (f.z <= 0.0) {
            return airP;
        }
        // At the face, half a cell from the liquid's middle along the pull
        // (bodies' old faceOf), in Pa: the liquid's units are cells² a
        // step² times water's density.
        let s = P.dx / P.dt;
        let atFace = f.x - 0.5 * dot(n, pullAt(vec2<f32>(q) + 0.5)) * f.z;
        return atFace * 1000.0 * s * s + airP;
    }
    // Powder: the load of the pile it is the foot of, all of it on a face
    // under it, LATERAL of it on one beside it (update.wgsl's pileLoad).
    let down = vec2<f32>(downOfCell(c));
    let onTop = dot(n, down) < -0.5;
    let pile = loads[j] * f32(P.size.y) / LOAD_ROWS * 1000.0 * length(pullAtCell(q)) * P.dx;
    return select(LATERAL, 1.0, onTop) * pile + airP;
}

// ---------------------------------------------------------------- nodes

fn nodeFlags(n: u32) -> u32 {
    return bitcast<u32>(nodes[nodeSlot(n, 3u)].w);
}

// The direction this iteration reads (parity), or the one it writes. (Each
// is written only by its own node's thread, and read by the others only
// while it isn't being written.)
fn directionOf(n: u32, parity: u32) -> vec2<f32> {
    let w = nodes[nodeSlot(n, 0u)];
    return select(w.xy, w.zw, parity == 1u);
}

fn setDirection(n: u32, parity: u32, p: vec2<f32>) {
    let w = nodes[nodeSlot(n, 0u)];
    nodes[nodeSlot(n, 0u)] = select(vec4<f32>(p, w.zw), vec4<f32>(w.xy, p), parity == 1u);
}

// ---------------------------------------------------------------- fixed point

const FIX_UNIT: f32 = 1099511627776.0;   // 2^40 units a cell
const TWO32: f32 = 4294967296.0;

// a + d (d cells, as the sign and the magnitude split into whole 2^32s and
// the rest, so a small one keeps all its bits), on one axis: high, low.
fn fixAdd(a: vec2<u32>, d: f32) -> vec2<u32> {
    let m = abs(d) * FIX_UNIT;
    let mh = floor(m / TWO32);
    let ml = u32(m - mh * TWO32);
    let mhi = u32(mh);
    if (d >= 0.0) {
        let lo = a.y + ml;
        return vec2<u32>(a.x + mhi + select(0u, 1u, lo < a.y), lo);
    }
    let lo = a.y - ml;
    return vec2<u32>(a.x - mhi - select(0u, 1u, lo > a.y), lo);
}

// a - b in cells, exactly as integers, then to f32 (by its sign and size, so
// a small difference keeps all its bits).
fn fixDiff(a: vec2<u32>, b: vec2<u32>) -> f32 {
    var lo = a.y - b.y;
    var hi = a.x - b.x - select(0u, 1u, a.y < b.y);
    let negative = (hi & 0x80000000u) != 0u;
    if (negative) {
        lo = ~lo + 1u;
        hi = ~hi + select(0u, 1u, lo == 0u);
    }
    let m = (f32(hi) * TWO32 + f32(lo)) / FIX_UNIT;
    return select(m, -m, negative);
}

// A node's displacement less another's, both axes, in cells.
fn dispDiff(a: vec4<u32>, b: vec4<u32>) -> vec2<f32> {
    return vec2<f32>(fixDiff(a.xy, b.xy), fixDiff(a.zw, b.zw));
}

fn precondition(n: u32, r: vec2<f32>) -> vec2<f32> {
    let m = nodes[nodeSlot(n, 3u)];
    return vec2<f32>(m.x * r.x + m.y * r.y, m.y * r.x + m.z * r.y);
}

// ---------------------------------------------------------------- tiles

var<workgroup> ringVec: array<vec2<f32>, RING_NODES>;   // a direction
var<workgroup> ringDisp: array<vec4<u32>, RING_NODES>;  // the displacement (disp)
var<workgroup> spanE: array<f32, SPAN_CELLS>;
var<workgroup> sums: array<vec2<f32>, 64>;
var<workgroup> tileLive: u32;
var<workgroup> uniformFlag: u32;

fn tileOf(wid: vec3<u32>) -> u32 {
    return wid.y * tilesWide() + wid.x;
}

fn tileOrigin(wid: vec3<u32>) -> vec2<i32> {
    return vec2<i32>(wid.xy) * TILE;
}

// Whether the tile has a node in structure (as setup found it), for the
// whole workgroup.
fn liveTile(wid: vec3<u32>, li: u32) -> bool {
    if (li == 0u) {
        tileLive = atomicLoad(&work[liveSlot(tileOf(wid))]);
    }
    return workgroupUniformLoad(&tileLive) != 0u;
}

// A flag, the same for the whole workgroup.
fn uniformly(li: u32, flag: bool) -> bool {
    if (li == 0u) {
        uniformFlag = select(0u, 1u, flag);
    }
    return workgroupUniformLoad(&uniformFlag) != 0u;
}

// Whether this workgroup is the one that writes the scalars and stats (the
// first tile on the list of those with structure).
fn keeper(wid: vec3<u32>) -> bool {
    return atomicLoad(&work[listSlot(0u)]) == tileOf(wid);
}

// The Young's moduli of the cells with a corner in the tile into spanE.
fn loadSpan(origin: vec2<i32>, li: u32) {
    for (var k = li; k < SPAN_CELLS; k += 64u) {
        let c = origin - 1 + vec2<i32>(i32(k) % CELLS_SPAN, i32(k) / CELLS_SPAN);
        spanE[k] = youngsAt(c);
    }
}

// K v at node n (local lid in the tile), the cells' moduli in spanE, v in
// ringDisp (fixed) or ringVec. An element's stiffness doesn't see it move or
// turn as a whole, so each element's part is worked out on its corners'
// displacements less the node's own and less the element's turning: the
// same sum, but of terms the size of its strains. (Summed as they were,
// terms of E v, 5e10 times a tip's 0.02 cells, cancelled down to a load of
// some 70 Pa, below the f32 rounding.)
fn applyK(lid: vec2<i32>, fixed: bool) -> vec2<f32> {
    var acc = vec2<f32>(0.0);
    let me = (lid.y + 1) * RING + lid.x + 1;
    for (var e = 0; e < 4; e++) {
        let ex = e & 1;
        let ey = e >> 1;
        // The cell, as an offset into the span (origin - 1).
        let cl = lid + vec2<i32>(ex, ey);
        let E = spanE[cl.y * CELLS_SPAN + cl.x];
        if (E == 0.0) {
            continue;
        }
        let ln = u32((1 - ex) + 2 * (1 - ey));
        var d: array<vec2<f32>, 4>;
        for (var lm = 0u; lm < 4u; lm++) {
            let ml = cl + vec2<i32>(i32(lm & 1u), i32(lm >> 1u));
            let k = ml.y * RING + ml.x;
            if (fixed) {
                d[lm] = dispDiff(ringDisp[k], ringDisp[me]);
            } else {
                d[lm] = ringVec[k] - ringVec[me];
            }
        }
        // Its turning at its middle, and the corners' share of it about the
        // node (its corner ln).
        let dvdx = 0.5 * ((d[1].y - d[0].y) + (d[3].y - d[2].y));
        let dudy = 0.5 * ((d[2].x - d[0].x) + (d[3].x - d[1].x));
        let w = 0.5 * (dvdx - dudy);
        let at = vec2<f32>(f32(ln & 1u), f32(ln >> 1u));
        for (var lm = 0u; lm < 4u; lm++) {
            let r = vec2<f32>(f32(lm & 1u), f32(lm >> 1u)) - at;
            let v = d[lm] - w * vec2<f32>(-r.y, r.x);
            let row = 2u * ln * 8u + 2u * lm;
            acc.x += E * (KE[row] * v.x + KE[row + 1u] * v.y);
            acc.y += E * (KE[row + 8u] * v.x + KE[row + 9u] * v.y);
        }
    }
    return acc;
}

// Sums a vec2 over the workgroup (64 threads); thread 0 writes it as the
// tile's partial sums a and b.
fn writePartials(wid: vec3<u32>, li: u32, v: vec2<f32>, a: u32, b: u32) {
    sums[li] = v;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s >>= 1u) {
        if (li < s) {
            sums[li] += sums[li + s];
        }
        workgroupBarrier();
    }
    if (li == 0u) {
        let t = tileOf(wid);
        atomicStore(&work[partialSlot(t, a)], bitcast<u32>(sums[0].x));
        atomicStore(&work[partialSlot(t, b)], bitcast<u32>(sums[0].y));
    }
}

// Partial sums a and b over the tiles with structure (the list setup
// made), in any one workgroup of 64: each workgroup of a pass that needs
// them sums them itself, the same way, rather than one workgroup in a pass
// of its own (half the dispatches an iteration).
fn sumLive(li: u32, a: u32, b: u32) -> vec2<f32> {
    let n = atomicLoad(&work[scalarSlot(S_LIVE)]);
    var acc = vec2<f32>(0.0);
    for (var i = li; i < n; i += 64u) {
        let t = atomicLoad(&work[listSlot(i)]);
        acc += vec2<f32>(bitcast<f32>(atomicLoad(&work[partialSlot(t, a)])), bitcast<f32>(atomicLoad(&work[partialSlot(t, b)])));
    }
    sums[li] = acc;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s >>= 1u) {
        if (li < s) {
            sums[li] += sums[li + s];
        }
        workgroupBarrier();
    }
    let total = sums[0];
    workgroupBarrier();
    return total;
}

// Converged is each tile's r·z within STRESS_TOL² of its own loads'
// f·M⁻¹f (no less than FZ_FLOOR of all of them: a tile of structure
// holding up only its own weight beside one under a great load), not the
// whole world's within the world's: beside a slab of stone under lava, a
// beam's stress stopped at half its own, and steel's thermal stress (tried,
// REWRITE_PLAN.md 8-8) cracked stone in another tank (8-9). The worst of
// them, over the live tiles, partial a against P_FZ. (Its floor is as far
// as a global CG's f32 sums resolve: at 1e-6 the solve iterated on noise.)
const FZ_FLOOR: f32 = 1e-3;
fn worstLive(li: u32, a: u32, fz: f32) -> f32 {
    let n = atomicLoad(&work[scalarSlot(S_LIVE)]);
    var worst = 0.0;
    for (var i = li; i < n; i += 64u) {
        let t = atomicLoad(&work[listSlot(i)]);
        let own = max(bitcast<f32>(atomicLoad(&work[partialSlot(t, P_FZ)])), FZ_FLOOR * fz);
        worst = max(worst, bitcast<f32>(atomicLoad(&work[partialSlot(t, a)])) / max(own, 1e-30));
    }
    sums[li] = vec2<f32>(worst, 0.0);
    workgroupBarrier();
    for (var s = 32u; s > 0u; s >>= 1u) {
        if (li < s) {
            sums[li].x = max(sums[li].x, sums[li + s].x);
        }
        workgroupBarrier();
    }
    let most = sums[0].x;
    workgroupBarrier();
    return most;
}

var<workgroup> beginSums: array<vec2<f32>, REDUCE_THREADS>;
var<workgroup> beginWorst: array<f32, REDUCE_THREADS>;
var<workgroup> listAt: array<u32, REDUCE_THREADS>;
var<workgroup> listed: u32;

// The tiles with structure (setup's liveSlot) listed in their order, a run
// of tiles a thread. Listed in the order setup's workgroups came to them,
// the sums over them came out otherwise from run to run, and two runs from
// one snapshot came apart (REWRITE_PLAN.md D-1).
fn listLive(li: u32) -> u32 {
    let run = (tileCount() + REDUCE_THREADS - 1u) / REDUCE_THREADS;
    let first = min(li * run, tileCount());
    let end = min(first + run, tileCount());
    var mine = 0u;
    for (var t = first; t < end; t++) {
        mine += select(0u, 1u, atomicLoad(&work[liveSlot(t)]) != 0u);
    }
    listAt[li] = mine;
    workgroupBarrier();
    if (li == 0u) {
        var before = 0u;
        for (var k = 0u; k < REDUCE_THREADS; k++) {
            let n = listAt[k];
            listAt[k] = before;
            before += n;
        }
        listed = before;
    }
    let n = workgroupUniformLoad(&listed);
    var at = listAt[li];
    for (var t = first; t < end; t++) {
        if (atomicLoad(&work[liveSlot(t)]) != 0u) {
            atomicStore(&work[listSlot(at)], t);
            at++;
        }
    }
    storageBarrier();
    workgroupBarrier();
    return n;
}

// ---------------------------------------------------------------- passes

@compute @workgroup_size(8, 8)
fn setup(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    if (li == 0u) {
        tileLive = 0u;
    }
    workgroupBarrier();
    let nn = tileOrigin(wid) + vec2<i32>(lid.xy);
    if (inNodes(nn)) {
        let n = nodeIndex(nn);
        var inside = false;
        var fixed = false;
        var d = vec3<f32>(0.0);  // its 2 x 2 block: xx, xy, yy
        var f = vec2<f32>(0.0);
        for (var e = 0; e < 4; e++) {
            let ex = e & 1;
            let ey = e >> 1;
            let c = nn - 1 + vec2<i32>(ex, ey);
            if (!standingAt(c)) {
                continue;
            }
            inside = true;
            let cv = cells[cellIndex(c)];
            let ce = elems[cellType(cv)];
            // Painted in a stroke still being drawn: held still, as by an
            // anchor (common.wgsl's SUPPORT_HELD).
            if (cellFrac(cv).x == SUPPORT_HELD) {
                fixed = true;
            }
            let ln = u32((1 - ex) + 2 * (1 - ey));
            let row = 2u * ln * 8u + 2u * ln;
            d += ce.youngs * vec3<f32>(KE[row], KE[row + 1u], KE[row + 9u]);
            f += ce.rho * pullAtCell(c) * P.dx * 0.25;
            // Its thermal strain, out through this corner (half a face's
            // worth each way: the unit square's ∫ ∂N/∂x).
            let corner = nn - c;
            f += thermalStress(cv) * 0.5 * vec2<f32>(corner * 2 - 1);
            // Its two faces through this corner: the side one, then the top
            // or bottom one.
            for (var k = 0; k < 2; k++) {
                var step = vec2<i32>(0, corner.y * 2 - 1);
                if (k == 0) {
                    step = vec2<i32>(corner.x * 2 - 1, 0);
                }
                let q = c + step;
                if (standingAt(q)) {
                    continue;
                }
                if (anchorFace(c, q)) {
                    fixed = true;
                } else {
                    f -= faceLoad(c, q) * vec2<f32>(step) * 0.5;
                }
            }
        }
        let flags = select(0u, N_ACTIVE, inside) | select(0u, N_FIXED, fixed);
        let was = nodeFlags(n);
        if (flags != was) {
            atomicAdd(&work[scalarSlot(S_CHANGED)], 1u);
        }
        let det = d.x * d.z - d.y * d.y;
        var m = vec3<f32>(0.0);
        if (inside && !fixed && det > 0.0) {
            m = vec3<f32>(d.z, -d.y, d.x) / det;
        }
        // A node out of the solve, or held, keeps nothing; one just in it
        // starts from rest.
        let free = inside && !fixed;
        let keep = free && (was & N_ACTIVE) != 0u && (was & N_FIXED) == 0u;
        if (!keep) {
            nodes[nodeSlot(n, 0u)] = vec4<f32>(0.0);
            nodes[nodeSlot(n, 1u)] = vec4<f32>(0.0);
            disp[n] = vec4<u32>(0u);
        }
        nodes[nodeSlot(n, 2u)] = vec4<f32>(0.0, 0.0, select(vec2<f32>(0.0), f, free));
        nodes[nodeSlot(n, 3u)] = vec4<f32>(m, bitcast<f32>(flags));
        if (inside) {
            tileLive = 1u;
            atomicAdd(&work[scalarSlot(S_NODES)], 1u);
        }
    }
    workgroupBarrier();
    if (li == 0u) {
        atomicStore(&work[liveSlot(tileOf(wid))], tileLive);
    }
}

// r = f - K u, z = M⁻¹ r; partial sums of r·z (as the r·z the first
// iteration starts from: P_RZ1) and f·M⁻¹f.
@compute @workgroup_size(8, 8)
fn residual(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    if (!liveTile(wid, li)) {
        return;
    }
    let origin = tileOrigin(wid);
    loadSpan(origin, li);
    for (var k = li; k < RING_NODES; k += 64u) {
        let m = origin - 1 + vec2<i32>(i32(k) % RING, i32(k) / RING);
        ringDisp[k] = select(vec4<u32>(0u), disp[nodeIndex(m)], inNodes(m));
    }
    workgroupBarrier();
    let nn = origin + vec2<i32>(lid.xy);
    var part = vec2<f32>(0.0);
    if (inNodes(nn)) {
        let n = nodeIndex(nn);
        let flags = nodeFlags(n);
        if ((flags & N_ACTIVE) != 0u && (flags & N_FIXED) == 0u) {
            let f = nodes[nodeSlot(n, 2u)].zw;
            let r = f - applyK(vec2<i32>(lid.xy), true);
            let z = precondition(n, r);
            nodes[nodeSlot(n, 1u)] = vec4<f32>(r, z);
            part = vec2<f32>(dot(r, z), dot(f, precondition(n, f)));
        }
    }
    writePartials(wid, li, part, P_RZ1, P_FZ);
}

// The step's start: r·z afresh, the loads' size, the direction's β (0 to
// start over: the first step, or the structure changed), converged or not.
@compute @workgroup_size(256)
fn begin(@builtin(local_invocation_index) li: u32) {
    let n = listLive(li);
    if (li == 0u) {
        setScalarU(S_LIVE, n);
    }
    var acc = vec2<f32>(0.0);
    for (var i = li; i < n; i += REDUCE_THREADS) {
        let t = atomicLoad(&work[listSlot(i)]);
        acc += vec2<f32>(bitcast<f32>(atomicLoad(&work[partialSlot(t, P_RZ1)])), bitcast<f32>(atomicLoad(&work[partialSlot(t, P_FZ)])));
    }
    beginSums[li] = acc;
    workgroupBarrier();
    for (var s = REDUCE_THREADS / 2u; s > 0u; s >>= 1u) {
        if (li < s) {
            beginSums[li] += beginSums[li + s];
        }
        workgroupBarrier();
    }
    // The worst tile against its own loads (worstLive's, over this
    // workgroup's 256 threads).
    let fzAll = workgroupUniformLoad(&beginSums[0]).y;
    var worst = 0.0;
    for (var i = li; i < n; i += REDUCE_THREADS) {
        let t = atomicLoad(&work[listSlot(i)]);
        let own = max(bitcast<f32>(atomicLoad(&work[partialSlot(t, P_FZ)])), FZ_FLOOR * fzAll);
        worst = max(worst, bitcast<f32>(atomicLoad(&work[partialSlot(t, P_RZ1)])) / max(own, 1e-30));
    }
    beginWorst[li] = worst;
    workgroupBarrier();
    for (var s = REDUCE_THREADS / 2u; s > 0u; s >>= 1u) {
        if (li < s) {
            beginWorst[li] = max(beginWorst[li], beginWorst[li + s]);
        }
        workgroupBarrier();
    }
    if (li == 0u) {
        let s = beginSums[0];
        let ratio = beginWorst[0];
        // It starts over along the residual on the first step, when the
        // structure changed, and when the residual has grown more than
        // RESTART_GROWTH-fold since the last iteration: the loads changed
        // more than it had come since (a blast). Carried on then, β came to
        // a million, the direction was the last one blown up, α fell to
        // 1e-9, and the solve stood still under the blast.
        // (And when either isn't a number: carried on, a NaN in it or in the
        // directions stayed for good, β = r·z / NaN, and 0 NaN is NaN; apply
        // takes a direction started over as z alone.)
        let before = scalarF(S_RZ_PREV);
        let restart = scalarU(S_CHANGED) > 0u || !(before > 0.0) || !(s.x <= RESTART_GROWTH * before);
        setScalarF(S_RZ, s.x);
        setScalarF(S_FZ, s.y);
        setScalarF(S_BETA, select(s.x / before, 0.0, restart));
        let converged = ratio <= STRESS_TOL * STRESS_TOL;
        setScalarU(S_DONE, select(0u, 1u, converged));
        setScalarU(S_JUDGE, select(0u, 1u, s.x <= 1e30));
        setScalarU(S_ITERATIONS, 0u);
        atomicStore(&work[statSlot(SS_CONVERGED)], select(0u, 1u, converged));
        // (How far off it is, which sets how many iterations the steps take,
        // is the whole world's, and its worst tile's past STRESS_FAR_TILE:
        // by its worst tile alone, the structure scenes took 128 a step far
        // more often, 12-16 % of a step.)
        atomicStore(&work[statSlot(SS_RESIDUAL)], bitcast<u32>(sqrt(s.x / max(s.y, 1e-30))));
        atomicStore(&work[statSlot(SS_WORST)], bitcast<u32>(sqrt(ratio)));
        atomicStore(&work[statSlot(SS_ITERATIONS)], 0u);
        atomicStore(&work[statSlot(SS_NODES)], scalarU(S_NODES));
    }
}

// Iteration k (sim.js: PARITY its parity; FIRST for the first): the
// residual's r·z now (the last update's partials, or the step's start's),
// whether it has converged, and the direction's β (the step's start's
// first, which may start over); then p = z + β p into the other direction
// (over the tile and its ring, which the neighbouring tiles work out too,
// alike), q = K p, partial p·q and r·p.
override PARITY: u32 = 0u;
override FIRST: bool = false;

@compute @workgroup_size(8, 8)
fn apply(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    if (!liveTile(wid, li)) {
        return;
    }
    // Converged already (at the step's start, or an iteration before: the
    // partial sums of the parity it stopped on are then two iterations
    // stale, and read as not converged after all).
    if (uniformly(li, scalarU(S_DONE) != 0u)) {
        return;
    }
    var beta = scalarF(S_BETA);
    var done = false;
    if (!FIRST) {
        // r·z after the last update (written with the other parity) and
        // before it (this parity's: two iterations back, or the start's).
        let now = sumLive(li, P_RZ0 + 1u - PARITY, P_RZ0 + PARITY);
        let fz = scalarF(S_FZ);
        beta = now.x / max(now.y, 1e-30);
        let ratio = worstLive(li, P_RZ0 + 1u - PARITY, fz);
        done = ratio <= STRESS_TOL * STRESS_TOL;
        if (li == 0u && keeper(wid)) {
            // (Only ever set within the step: the others read it at their
            // start, and come to the same of it themselves.)
            if (done) {
                setScalarU(S_DONE, 1u);
            }
            setScalarF(S_RZ_PREV, now.x);
            atomicStore(&work[statSlot(SS_CONVERGED)], select(0u, 1u, done));
            atomicStore(&work[statSlot(SS_RESIDUAL)], bitcast<u32>(sqrt(now.x / max(fz, 1e-30))));
            atomicStore(&work[statSlot(SS_WORST)], bitcast<u32>(sqrt(ratio)));
        }
    } else if (li == 0u && keeper(wid)) {
        setScalarF(S_RZ_PREV, scalarF(S_RZ));
    }
    if (uniformly(li, done)) {
        return;
    }
    let last = scalarU(S_DIR) ^ PARITY;
    let origin = tileOrigin(wid);
    loadSpan(origin, li);
    for (var k = li; k < RING_NODES; k += 64u) {
        let m = origin - 1 + vec2<i32>(i32(k) % RING, i32(k) / RING);
        var p = vec2<f32>(0.0);
        if (inNodes(m)) {
            let n = nodeIndex(m);
            p = nodes[nodeSlot(n, 1u)].zw;
            if (beta != 0.0) {
                p += beta * directionOf(n, last);
            }
        }
        ringVec[k] = p;
    }
    workgroupBarrier();
    let nn = origin + vec2<i32>(lid.xy);
    var part = vec2<f32>(0.0);
    if (inNodes(nn)) {
        let n = nodeIndex(nn);
        let flags = nodeFlags(n);
        let p = ringVec[(i32(lid.y) + 1) * RING + i32(lid.x) + 1];
        setDirection(n, 1u - last, p);
        if ((flags & N_ACTIVE) != 0u && (flags & N_FIXED) == 0u) {
            let q = applyK(vec2<i32>(lid.xy), false);
            let w = nodes[nodeSlot(n, 2u)];
            nodes[nodeSlot(n, 2u)] = vec4<f32>(q, w.zw);
            part = vec2<f32>(dot(p, q), dot(nodes[nodeSlot(n, 1u)].xy, p));
        }
    }
    writePartials(wid, li, part, P_PQ, P_RP);
}

// Iteration k's step: α = r·p / p·Kp, the exact step along p (CG's
// r·z / p·Kp is the same within a step, but not along a direction carried
// over a step whose loads changed: the liquid's sloshing on a boiler's walls
// threw it off further each step, to residuals 1e24 times the loads), then
// u += α p (in fixed point), r -= α q, z = M⁻¹ r, partial r·z (this
// parity's). (A direction the structure doesn't resist, which a held, rigid
// piece has none of: no step.)
@compute @workgroup_size(8, 8)
fn update(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    if (!liveTile(wid, li)) {
        return;
    }
    if (uniformly(li, scalarU(S_DONE) != 0u)) {
        return;
    }
    let s = sumLive(li, P_PQ, P_RP);
    let a = select(s.y / s.x, 0.0, s.x <= 0.0);
    let to = 1u - (scalarU(S_DIR) ^ PARITY);
    let nn = tileOrigin(wid) + vec2<i32>(lid.xy);
    var part = 0.0;
    if (inNodes(nn)) {
        let n = nodeIndex(nn);
        let flags = nodeFlags(n);
        if ((flags & N_ACTIVE) != 0u && (flags & N_FIXED) == 0u) {
            let step = a * directionOf(n, to);
            let u = disp[n];
            disp[n] = vec4<u32>(fixAdd(u.xy, step.x), fixAdd(u.zw, step.y));
            let r = nodes[nodeSlot(n, 1u)].xy - a * nodes[nodeSlot(n, 2u)].xy;
            let z = precondition(n, r);
            nodes[nodeSlot(n, 1u)] = vec4<f32>(r, z);
            part = dot(r, z);
        }
    }
    writePartials(wid, li, vec2<f32>(part, 0.0), P_RZ0 + PARITY, P_SPARE);
    if (li == 0u && keeper(wid)) {
        let k = scalarU(S_ITERATIONS) + 1u;
        setScalarU(S_ITERATIONS, k);
        atomicStore(&work[statSlot(SS_ITERATIONS)], k);
    }
}

// The strain at (px, py) in cell c (0..1 each way) from its corners'
// displacements, then the stress (Pa) for its modulus E: xx, yy, xy.
fn stressAt(u: array<vec2<f32>, 4>, px: f32, py: f32, E: f32) -> vec3<f32> {
    var exx = 0.0;
    var eyy = 0.0;
    var gxy = 0.0;
    for (var l = 0u; l < 4u; l++) {
        let lx = f32(l & 1u);
        let ly = f32(l >> 1u);
        let dndx = (lx * 2.0 - 1.0) * mix(1.0 - py, py, ly);
        let dndy = mix(1.0 - px, px, lx) * (ly * 2.0 - 1.0);
        exx += dndx * u[l].x;
        eyy += dndy * u[l].y;
        gxy += dndy * u[l].x + dndx * u[l].y;
    }
    let k = E / (1.0 - NU * NU);
    return vec3<f32>(k * (exx + NU * eyy), k * (NU * exx + eyy), k * (1.0 - NU) * 0.5 * gxy);
}

// The stress along face d (0 left, 1 right, 2 top, 3 bottom) of standing
// cell c (holding v) in its skin, over what the solve finds of the cell
// (REWRITE_PLAN.md 8-5): a cell is a centimetre, and what quenches or heats
// it does so first at its face, a skin far thinner. Its face stands between
// the cell (its half, conducting) and what is across it (its film: a gas's,
// water's, or the half of a cell of anything else conducting), so is at
// the temperature they share heat at, and the skin, as much cooler than
// the cell, is pulled along the face by E α (Tcell - Tface) / (1 - ν)
// (Kingery 1955's thermal shock, with the face's Biot number). A skin
// hotter than its cell is squeezed, which stone and glass bear some ten
// times better; its spalling (a heated skin buckling off, a millimetre at
// a time) is left out, as a cell's crumbling would take a centimetre at
// once: stone against lava crumbled a row a step, the lava sinking into
// its gravel and crumbling the next. Nothing against structure, or the
// world's edge.
fn skinStress(c: vec2<i32>, v: vec4<u32>, d: u32) -> f32 {
    let q = c + select(vec2<i32>(0, select(-1, 1, d == 3u)), vec2<i32>(select(-1, 1, d == 1u), 0), d < 2u);
    if (!inWorld(q) || standingAt(q)) {
        return 0.0;
    }
    let e = elems[cellType(v)];
    if (e.expansionLinear <= 0.0) {
        return 0.0;
    }
    let j = cellIndex(q);
    let w = cells[j];
    let nt = cellType(w);
    let n = elems[nt];
    // (The temperatures they act at: water freezing is held at 0 °C while
    // its stored temperature runs down to some -80, boiling at 100 while it
    // runs up past 600.)
    let temp = tempAtCell(v, c);
    let nTemp = tempAtCell(w, q);
    var film = 2.0 * n.thermal / P.dx;
    if (n.kind == KIND_GAS) {
        film = gasFilm(nt, gasFrom(cells[gasIndex(P.size.x * P.size.y, j)], j), nTemp, temp, windSpeedAt(q));
    } else if (isWater(nt)) {
        film = waterFilm(n, nTemp, temp, e.emissivity, P.earthG * P.dx / (P.dt * P.dt), P.dx);
    }
    let half = 2.0 * e.thermal / P.dx;
    let face = (half * temp + film * nTemp) / max(half + film, 1e-9);
    let soft = softeningT(e);
    return max(e.youngs * e.expansionLinear * (min(temp, soft) - min(face, soft)) / (1.0 - NU), 0.0);
}

// Each standing cell's failure index, the worst over its corners of
// σ1 / ft - σ3 / fc (0 for anything else), with its skins' stresses along
// the faces through each corner (skinStress), and how many judgements
// running it has been 1 or more; once near enough converged to judge
// (S_JUDGE).
@compute @workgroup_size(8, 8)
fn measure(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = vec2<i32>(gid.xy);
    if (!inWorld(c) || scalarU(S_JUDGE) == 0u) {
        return;
    }
    let i = cellIndex(c);
    var worst = 0.0;
    if (standingAt(c)) {
        let e = elems[cellType(cells[i])];
        let held = thermalStress(cells[i]);
        // Hot, it is weaker (hotShare).
        let hot = hotShare(e, tempAtCell(cells[i], c));
        // Its corners' displacements less the first's (strains are
        // differences).
        var u: array<vec2<f32>, 4>;
        let u0 = disp[nodeIndex(c)];
        for (var l = 0u; l < 4u; l++) {
            u[l] = dispDiff(disp[nodeIndex(c + vec2<i32>(i32(l & 1u), i32(l >> 1u)))], u0);
        }
        var skins = vec4<f32>(0.0);
        for (var d = 0u; d < 4u; d++) {
            skins[d] = skinStress(c, cells[i], d);
        }
        worst = -1e30;
        for (var k = 0u; k < 4u; k++) {
            // (The left or right face's skin is pulled along y, the top or
            // bottom's along x.)
            let skin = vec3<f32>(skins[2u + (k >> 1u)], skins[k & 1u], 0.0);
            let s = stressAt(u, f32(k & 1u), f32(k >> 1u), e.youngs) - vec3<f32>(held, held, 0.0) + skin;
            let mid = 0.5 * (s.x + s.y);
            let r = length(vec2<f32>(0.5 * (s.x - s.y), s.z));
            worst = max(worst, ((mid + r) / e.tensile - (mid - r) / e.compressive) / hot);
        }
    }
    atomicStore(&work[failSlot(i)], bitcast<u32>(worst));
    let streak = atomicLoad(&work[streakSlot(i)]);
    atomicStore(&work[streakSlot(i)], select(0u, min(streak + 1u, 255u), worst >= 1.0));
}

fn failAt(c: vec2<i32>) -> f32 {
    if (!inWorld(c)) {
        return -1e30;
    }
    return bitcast<f32>(atomicLoad(&work[failSlot(cellIndex(c))]));
}

// A standing cell past its strength (failure index 1 or more) two
// judgements running that is the most overstressed of the cells round it
// (the lower index first, of two alike) breaks: its step marked, for
// update.wgsl to crumble it next step.
@compute @workgroup_size(8, 8)
fn crack(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = vec2<i32>(gid.xy);
    if (!inWorld(c) || scalarU(S_JUDGE) == 0u || !standingAt(c)) {
        return;
    }
    let mine = failAt(c);
    if (mine < 1.0 || atomicLoad(&work[streakSlot(cellIndex(c))]) < 2u) {
        return;
    }
    for (var k = 0; k < 9; k++) {
        if (k == 4) {
            continue;
        }
        let q = c + vec2<i32>(k % 3 - 1, k / 3 - 1);
        let other = failAt(q);
        if (other > mine || (other == mine && k < 4)) {
            return;
        }
    }
    atomicStore(&bodies[structureBreakSlot(P.size, cellIndex(c))], P.frame + 1u);
}

// The counters zeroed for next step, and which direction is the last made
// (iteration k wrote direction S_DIR ^ k ^ 1, as many as updates ran).
@compute @workgroup_size(1)
fn finish() {
    setScalarU(S_CHANGED, 0u);
    setScalarU(S_NODES, 0u);
    setScalarU(S_LIVE, 0u);
    let k = scalarU(S_ITERATIONS);
    if (k > 0u) {
        setScalarU(S_DIR, 1u - (scalarU(S_DIR) ^ ((k - 1u) & 1u)));
    }
}
