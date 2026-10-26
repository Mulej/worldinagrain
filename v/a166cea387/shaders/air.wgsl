// The air (REWRITE_PLAN.md stages 4a-4c): low-Mach flow (Rehm & Baum 1978,
// as the Fire Dynamics Simulator has it) at cell resolution (AIR_CELL 1),
// velocity on the cells' faces (x on each one's right face, y on its bottom
// face). Ordinary wind is far slower than sound, so the gas is taken to be at
// one pressure within each pressure zone, and its pressure as two parts: a
// dynamic part, found by a projection each step, and a zone's background
// pressure from the gas law. The flow is every gas's (smoke, flames,
// hydrogen ride it as markers, block.wgsl), and what gas passes through (a
// fan, air with blades in it; a black hole, which swallows it): a face is
// open where the cells across it both carry the gas (carriesAt). Liquids,
// powders and solids are walls. The carry (carry.wgsl) moves what the gas
// holds on this flow. A step:
//
//   gather          from each cell: its faces open or shut, its gas's
//                   temperature and molar mass (buoyancy), fans, black holes
//                   (sinks); each room's gas summed for its zone
//   blastPressure,  explosions' pressure waves, in a layer of their own: a
//   blastVelocity   linear wave (sim.js's BLAST_SUBSTEPS half steps a
//                   step), crossing any gas, walls reflecting it and the
//                   open edges letting it out
//   momentum        the velocity carried along by itself (semi-Lagrangian,
//                   each component from its own face), buoyancy, fans, and
//                   pushes put into the air (airsource.wgsl)
//   divergence      each cell's net outflow through its faces, less what its
//                   gas gives out to come to its zone's pressure (expansion),
//                   an explosion's gas sending that off as a blast too
//   cgStart, ...    the projection's pressure, by conjugate gradients
//                   preconditioned by multigrid V-cycles, warm-started from
//                   last step's, until what is left is under AIR_TOLERANCE
//   project         the pressure's differences taken off the faces, which
//                   leaves each cell's flow giving out what its expansion
//                   asks; and the air as the other passes read it
//                   (airsample.wgsl)
//
// Pressure zones: the rooms (sky.wgsl), every piece of the flow apart from
// the others. The open air, any reaching the world's sides or bottom, at 1
// atm at the ground and less with height (SCALE_HEIGHT); each sealed room at
// what its gas would be at evenly spread by the gas law: all it holds over
// what that would take up at 1 atm (each cell's T_ref / T), so a sealed
// room's pressure rises as a fire heats it, a pot boils in it or a liquid is
// poured into it, and stays raised (no pressure bleeds away), until what
// holds it in gives.
//
// Expansion: a cell's gas at other than its zone's pressure gives out what
// takes it there in a step (ln(p / p_zone) of its volume where it is over,
// p / p_zone - 1 where under, taking in from around it), as the
// projection's source. So a cell of water boiled into vapour (8 atm in its
// cell) spreads by the wind it makes, the air a fire heats expands away from
// it, and a cell left empty (a vacuum, where a body or liquid was) is filled
// from around it. In a sealed room (with no black hole) the room's mean is
// taken off, summed exactly (wideParts), and its pressure rises instead: what
// a piece of the flow with no way out gives out comes to none, or the
// projection could not balance it.
//
// The world's sides and bottom are open air: the potential is held at that
// of the outside there (none), so wind leaves the world and comes in. Its
// top is the top of the air (see gather). Black holes are sinks: their
// potential is held low, and air flows in.

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> elems: array<Elem>;
@group(0) @binding(2) var<storage, read> cells: array<vec4<u32>>;
// The air as the other passes read it (common.wgsl's AIR_BLOCKED).
@group(0) @binding(3) var<storage, read_write> air: array<vec4<f32>>;
// Per air cell: last step's pressure, the velocity on its right and bottom
// faces before the projection, and the gas's temperature as its buoyancy
// goes (gather's buoyantTemp).
@group(0) @binding(4) var<storage, read_write> work: array<vec4<f32>>;
// The projection's levels, each cell two vec4s: (potential, right-hand
// side, its right face's weight, its bottom face's weight), and (the
// conductance of its faces on the world's edges, 1 for a fixed potential,
// the room piece of its gas (level 0, as bits), its open share (level 0;
// none where no gas, below that where liquid or powder fills it)). A cell
// on the last column or row has its edge face's weight as its right or
// bottom face's.
@group(0) @binding(5) var<storage, read_write> mg: array<vec4<f32>>;
// The pull of the world's masses per light cell (= air cell), cells/step²
// (gravity.wgsl).
@group(0) @binding(7) var gravity: texture_2d<f32>;
@group(0) @binding(8) var<storage, read_write> sky: array<atomic<u32>>;
// Explosions' pressure waves: per air cell, overpressure (air units, as
// the air's pressure) and velocity on its right and bottom faces.
@group(0) @binding(9) var<storage, read_write> blast: array<vec4<f32>>;
// The level a multigrid pass works on, and which of the step's V-cycles.
struct Level {
    index: u32,
    cycle: u32,
    pad1: u32,
    pad2: u32,
}
@group(0) @binding(10) var<uniform> level: Level;
@group(0) @binding(11) var<uniform> B: Brush;  // the wind tool's
// The most any cell's flow gives out this step, in its volume (f32 bits,
// project), which sets how many substeps the carry takes (carry.wgsl); then
// three slots, in turn by the iteration, of the most divergence one left
// (f32 bits); then how many iterations ran this step (for probes); then, at
// AIR_MOST_CYCLES, the most of those in a step since the host last read and
// cleared it (sim.js airCycles); then, at AIR_BLAST_LIVE, a word a step by
// its parity, whether the explosions' layer held anything (blastQuiet).
@group(0) @binding(12) var<storage, read_write> status: array<atomic<u32>>;
// What other passes (and the wind tool) put into the air (airsource.wgsl),
// taken in by momentum.
@group(0) @binding(14) var<storage, read_write> source: array<atomic<u32>>;
// The liquid's grid (fluid.wgsl's field), two per cell, the first's z its
// density where the liquid counts the cell as liquid (gapAt).
@group(0) @binding(15) var<storage, read> liquid: array<vec4<f32>>;

const FAN_SPEED: f32 = 12.0;      // cells per step a fan blows at
const FAN_MIX: f32 = 0.5;         // share of the way a fan's faces are driven to it per step
const MAX_V: f32 = 12.0;          // a bound on the wind, as fast as a fan
// The momentum the air's eddies take out of it, by its speed: turbulence
// spends a flow's energy at some u³ / ℓ (Taylor 1935; Tennekes & Lumley,
// A First Course in Turbulence, 1972, §3.1), ℓ the size of the eddies that
// hold it, so the air loses its speed u at u² / ℓ, AIR_EDDY_M metres
// across, a room's (REWRITE_PLAN.md 4e-9): a buoyant draft settles where
// that matches its buoyancy, √(g ΔT / T ℓ), a chimney's 6 K some 0.45 m/s
// and a flame's gas 500 K hot some 2.4, and air stirred and left loses a
// tenth of a metre a second's eddy over some ten seconds. (A share a step,
// 2 %, held a chimney's draft to 0.07 m/s, and stilled stirred air in a
// second; none at all, every scene's air stirred on without end.
// Smagorinsky's model at a cell's size takes next to nothing out of flows
// the grid resolves.)
const AIR_EDDY_M: f32 = 1.0;
// A black hole takes gas in through a face as fast as its pull brings it in
// from the cell beside it: free fall over a cell, √(2 g) cells a step, g the
// masses' pull there (pull.wgsl), about 2 for one pulling 10 g at its edge.
// Its potential is held at less than none by that much (sinkSpeed), so the
// open air flows into it about that fast; a sealed room's gas thins evenly
// to make up what it takes (gather's ZONE_SINK_DRAW, meanExpansion). A hole
// that pulls nothing takes nothing. (A fixed -2 before, whatever it pulled:
// audit 06 A10.)
fn sinkSpeedAt(c: vec2<i32>) -> f32 {
    return sqrt(2.0 * length(fieldAt((vec2<f32>(c) + 0.5) * f32(AIR_CELL)).xy));
}

fn sinkSpeed(c: vec2<i32>) -> f32 {
    var most = 0.0;
    for (var d = 0; d < 4; d++) {
        most = max(most, sinkSpeedAt(c + select(vec2<i32>(0, (d & 1) * 2 - 1), vec2<i32>((d & 1) * 2 - 1, 0), d < 2)));
    }
    return most;
}
// Ties each sealed zone's potential, which nothing else fixes, to none: no
// more than that, as it leaves ANCHOR times the potential of divergence in
// each cell, which the wind's carry would pile up.
const ANCHOR: f32 = 1e-6;
// The coarsest level's sweeps each way, over-relaxed by COARSE_OMEGA
// (symmetric SOR: red and black in turn, then the same backwards, so still
// symmetric and positive, as the CG's preconditioner must be). 20 so take
// off as much as 60 of Gauss-Seidel's did (the CG's iterations the same in
// the scenes and the worlds the app opens on), the solve a third as long:
// one workgroup, a barrier a sweep, 33 us an iteration before (F-6).
const COARSE_SWEEPS: i32 = 20;
const COARSE_OMEGA: f32 = 1.7;
// The explosions' layer: a linear wave, its speed sqrt(KB KV) air cells a
// half step (about 2 cells a step), losing a little each half step. Its
// source is the gas an explosion makes (BLAST, common.wgsl's detonation)
// expanding: what of its volume a cell of it gives out a step (expansion)
// raises the pressure there by that times the air's bulk modulus, γ p (a
// sound wave's source is the volume put into the air: e.g. Pierce,
// Acoustics, 1989, ch. 4). Other gas expanding (boiling, burning) sends
// none: it comes out of the cell gradually, as bubbles and flames do.
const BLAST_GAMMA: f32 = 1.4;
const BLAST_KB: f32 = 0.5;
const BLAST_KV: f32 = 0.5;
const BLAST_KEEP: f32 = 0.99;
const MAX_P: f32 = 300.0;
const TAU: f32 = 6.2831853;

// ------------------------------------------------------------------ layout

fn levelDims(l: u32) -> vec2<u32> {
    var d = airDims(P.size);
    for (var k = 0u; k < l; k++) {
        d = (d + 1u) / 2u;
    }
    return d;
}

fn levelBase(l: u32) -> u32 {
    var d = airDims(P.size);
    var base = 0u;
    for (var k = 0u; k < l; k++) {
        base += d.x * d.y;
        d = (d + 1u) / 2u;
    }
    return 2u * base;
}

fn mgAt(base: u32, d: vec2<u32>, c: vec2<i32>) -> u32 {
    return base + 2u * (u32(c.y) * d.x + u32(c.x));
}

fn airIndex(c: vec2<i32>, dims: vec2<u32>) -> u32 {
    return u32(c.y) * dims.x + u32(c.x);
}

// ------------------------------------------------------------------ gather

fn inWorld(p: vec2<i32>) -> bool {
    return all(p >= vec2<i32>(0)) && all(p < vec2<i32>(P.size));
}

fn cellAt(p: vec2<i32>) -> vec4<u32> {
    return cells[u32(p.y) * P.size.x + u32(p.x)];
}

// Whether air passes through the cell at p: gas, or a solid that lets it
// through (a fan, a black hole). Not outside the world, nor a gap in the
// liquid (common.wgsl's flowsThrough).
fn passesAt(p: vec2<i32>) -> bool {
    if (!inWorld(p)) {
        return false;
    }
    let i = u32(p.y) * P.size.x + u32(p.x);
    return flowsThrough(cellType(cells[i]), liquidSide(liquid[2u * i], liquid[2u * i + 1u]));
}

// The open air's pressure at pos (cells), in atm, by its height as the pull
// has it (pull.wgsl): so it thins up and away from the Earth, and thickens
// down into a black hole's well, and with no pull is one pressure.
fn openAirAt(pos: vec2<f32>) -> f32 {
    return openAirAtm(altitudeAt(pos));
}

// What the cell at p holds of the gas (the gas plane, common.wgsl).
fn gasAt(p: vec2<i32>) -> Gas {
    let i = u32(p.y) * P.size.x + u32(p.x);
    return gasFrom(cells[gasIndex(P.size.x * P.size.y, i)], i);
}

// Whether gas at p meets the outside air: on the world's sides or bottom (its
// top is the top of the air, shut).
fn opensOut(p: vec2<i32>) -> bool {
    return p.x == 0 || p.x + 1 == i32(P.size.x) || p.y + 1 == i32(P.size.y);
}

// The outdoor air's temperature at pos (cells), at its height (pull.wgsl),
// as buoyant as it is: what buoyancy is measured against. Its vapour
// lightens it as a cell's does (buoyantTemp, by the gas's molar mass): taken
// as dry, the open air at the outdoor air's own temperature and humidity was
// some 1.2 K lighter than it, and an empty world rose through itself
// without end, in at its foot and out at its sides (4e-10).
fn ambientAt(pos: vec2<f32>) -> f32 {
    let altitude = altitudeAt(pos);
    let temp = outdoorT(altitude) + dayWarmth(P.day);
    return buoyantTemp(AIR, outdoorGas(altitude, temp), temp);
}

// Per room piece of the workgroup's tile (it lies in one: 16 cells a side
// to the tile's ROOM_TILE), its gas cells' sums (zoneSlotAt's words: amount,
// volume at 1 atm, how many, whether it opens out or has a sink).
var<workgroup> zoneAmount: array<atomic<u32>, ROOM_PIECES>;
var<workgroup> zoneVolume: array<atomic<u32>, ROOM_PIECES>;
var<workgroup> zoneCount: array<atomic<u32>, ROOM_PIECES>;
var<workgroup> zoneOpen: array<atomic<u32>, ROOM_PIECES>;
var<workgroup> zoneSinkDraw: array<atomic<u32>, ROOM_PIECES>;
var<workgroup> zonePin: array<atomic<u32>, ROOM_PIECES>;
// And per thread, its cell's piece (UNZONED for none) and its gas's moments
// (common.wgsl's ZONE_MOMENTS: C, C T, C h, C T h, C h²), summed for each
// piece by the first thread in it.
var<workgroup> momentPiece: array<u32, 64>;
var<workgroup> moments: array<array<f32, 5>, 64>;
// And per wet room piece of the tile (common.wgsl's wetWord), whether any of
// its gas opens out.
var<workgroup> wetOpen: array<atomic<u32>, ROOM_PIECES>;
// (Its bits: common.wgsl's ZONE_OUTSIDE and ZONE_SINK.)

// Whether the cell at p is part of the flow (and the carry, carry.wgsl):
// any gas, or a solid gas passes through (a fan, air with blades in it; a
// black hole, which swallows it).
fn carriesAt(p: vec2<i32>) -> bool {
    return passesAt(p);
}

// A cell's zone (gather's piece, in the solve's levels): its room piece's
// number, or UNZONED: a pocket of a tile with more than ROOM_PIECES
// (sky.wgsl's UNNUMBERED), or none at all (not gas). An unzoned cell gives
// out nothing to come to a zone's pressure: its flow balances whatever the
// rest of its room's does.
const UNZONED: u32 = 0xffffffffu;

// The piece of its tile the gas at p is in (NO_ROOM for none): its room
// byte (sky.wgsl), or for one that has none as the rooms were found (what
// changed since), the gas's beside it in the tile.
fn zoneByteAt(a: vec2<u32>) -> u32 {
    var piece = roomByteAt(a);
    let at = a % ROOM_TILE;
    if (piece == NO_ROOM && at.x > 0u) {
        piece = roomByteAt(a - vec2<u32>(1u, 0u));
    }
    if (piece == NO_ROOM && at.x + 1u < ROOM_TILE && a.x + 1u < P.size.x) {
        piece = roomByteAt(a + vec2<u32>(1u, 0u));
    }
    if (piece == NO_ROOM && at.y > 0u) {
        piece = roomByteAt(a - vec2<u32>(0u, 1u));
    }
    if (piece == NO_ROOM && at.y + 1u < ROOM_TILE && a.y + 1u < P.size.y) {
        piece = roomByteAt(a + vec2<u32>(0u, 1u));
    }
    return piece;
}

// x * ZONE_SCALE as a u32, rounded at random, so that what is summed adds up
// on average however small each part (r in [0, 1)).
fn zoneFixed(x: f32, r: f32) -> u32 {
    return u32(max(floor(x * ZONE_SCALE + r), 0.0));
}

// A cell's random number for the zone sums' rounding (a PCG hash).
fn zoneNoise(i: u32, salt: u32) -> f32 {
    return f32(pcg(i * 2654435761u + P.frame * 40503u + salt) >> 8u) / 16777216.0;
}

// What a zone's cells give out to come to its pressure (expansion) is summed
// exactly, as a 64-bit number in units of WIDE_SCALE of a cell's volume, in
// two words (low, then high, two's complement), added to with a carry from
// the low word into the high: a sealed zone's flow balances only to within
// what is left of its mean, and that is the residual its projection can't
// take off (ANCHOR only ties its potential down), so summed in an i32 at
// ZONE_SCALE, the rounding of ten thousand cells' left the solve short of
// AIR_TOLERANCE, step after step. Each cell's part (bounded by WIDE_MOST
// cells' volume) rounded at random.
const WIDE_SCALE: f32 = 16777216.0;
const WIDE_MOST: f32 = 64.0;

fn wideParts(x: f32, r: f32) -> vec2<u32> {
    let v = i32(floor(clamp(x, -WIDE_MOST, WIDE_MOST) * WIDE_SCALE + r));
    return vec2<u32>(bitcast<u32>(v), select(0u, 0xffffffffu, v < 0));
}


// The temperature a cell of t holding g at temp is as buoyant as plain air
// would be: its density under the same pressure is its molar mass over the
// air's times the air's at temp, so hydrogen (2) and vapour (18) rise, and
// air (29) only as warm as it is.
fn buoyantTemp(t: u32, g: Gas, temp: f32) -> f32 {
    return (temp + KELVIN) * elems[AIR].molar / gasMolar(t, g) - KELVIN;
}

// An air cell is a cell (AIR_CELL 1: each face of the flow open or shut, as
// the cells across it are).
fn gatherCell(a: vec2<u32>, dims: vec2<u32>, li: u32) {
    let c = vec2<i32>(a);
    let idx = airIndex(c, dims);
    let v = cellAt(c);
    let t = cellType(v);
    let e = elems[t];
    let carries = carriesAt(c);
    let fixed = t == BLACK_HOLE;
    // (A gap the liquid counts as liquid, as liquid: flowsThrough.)
    let fill = e.kind == KIND_LIQUID || e.kind == KIND_POWDER || (e.kind == KIND_GAS && !carries);
    let g = gasAt(c);
    let temp = wordTempOf(v);
    var piece = UNZONED;
    if (carries) {
        let pc = zoneByteAt(a);
        if (pc != UNNUMBERED && pc != NO_ROOM) {
            piece = roomPieceAt(P.size, a, pc);
            let wp = wetByteAt(a);
            let wet = wp < UNNUMBERED;
            if (fixed) {
                atomicOr(&zoneOpen[pc], ZONE_SINK);
                // What it takes in through the faces its gas meets it by.
                var draw = 0.0;
                for (var d = 0; d < 4; d++) {
                    let q = c + select(vec2<i32>(0, (d & 1) * 2 - 1), vec2<i32>((d & 1) * 2 - 1, 0), d < 2);
                    if (carriesAt(q) && cellType(cellAt(q)) != BLACK_HOLE) {
                        draw += sinkSpeedAt(q);
                    }
                }
                atomicAdd(&zoneSinkDraw[pc], zoneFixed(draw, zoneNoise(idx, 4u)));
            } else {
                atomicAdd(&zoneAmount[pc], zoneFixed(gasAmount(g), zoneNoise(idx, 1u)));
                atomicAdd(&zoneVolume[pc], zoneFixed((AMBIENT_T + KELVIN) / max(temp + KELVIN, 1.0), zoneNoise(idx, 2u)));
                atomicAdd(&zoneCount[pc], 1u);
                atomicMax(&zonePin[pc], ~idx);
                let cap = gasHeatCap(g);
                let h = altitudeAt((vec2<f32>(c) + 0.5) * f32(AIR_CELL));
                momentPiece[li] = piece;
                moments[li] = array<f32, 5>(cap, cap * temp, cap * h, cap * temp * h, cap * h * h);
            }
            if (opensOut(c)) {
                atomicOr(&zoneOpen[pc], ZONE_OUTSIDE);
                if (wet) {
                    atomicStore(&wetOpen[wp], 1u);
                }
            }
        }
    }

    // The flow's faces: open where the cells across one both carry the gas;
    // on the world's last column and row, where the cell does (the world's
    // edges are open air), and likewise its first column (`edge`). (The top
    // edge is the top of the air: open, a world's air a little warmer or
    // cooler than the outdoor air's at its height would rise or sink without
    // end, nothing holding it; closed, its pressure holds it, as the air
    // above would.)
    let wR = f32(carries && (a.x + 1u >= dims.x || carriesAt(c + vec2<i32>(1, 0))));
    let wD = f32(carries && (a.y + 1u >= dims.y || carriesAt(c + vec2<i32>(0, 1))));
    let edge = f32(carries && a.x == 0u);

    let at = mgAt(0u, dims, c);
    mg[at] = vec4<f32>(0.0, 0.0, wR, wD);
    // (The potential, warm from last step's, is the conjugate gradients':
    // a black hole's held.)
    if (fixed) {
        cg[idx].x = -sinkSpeed(c);
    }
    // Pressure is any gas's (and passes through what gas passes through);
    // liquid or powder carries it on (project).
    let share = select(select(0.0, -1.0, fill), 1.0, carries);
    mg[at + 1u] = vec4<f32>(edge, f32(fixed), bitcast<f32>(piece), share);
    // The gas's temperature, as buoyant as it is (a black hole's, none's:
    // the outdoor air's). (By an if: select() works out both, and
    // ambientAt, its texture reads and exps, was every gas cell's for
    // nothing.)
    var buoyant: f32;
    if (carries && !fixed) {
        buoyant = buoyantTemp(t, g, temp);
    } else {
        buoyant = ambientAt((vec2<f32>(c) + 0.5) * f32(AIR_CELL));
    }
    work[idx] = vec4<f32>(air[idx].x, 0.0, 0.0, buoyant);
}

@compute @workgroup_size(8, 8)
fn gather(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    if (li < ROOM_PIECES) {
        atomicStore(&zoneAmount[li], 0u);
        atomicStore(&zoneVolume[li], 0u);
        atomicStore(&zoneCount[li], 0u);
        atomicStore(&zoneOpen[li], 0u);
        atomicStore(&zoneSinkDraw[li], 0u);
        atomicStore(&zonePin[li], 0u);
        atomicStore(&wetOpen[li], 0u);
    }
    momentPiece[li] = UNZONED;
    if (all(gid == vec3<u32>(0u))) {
        atomicStore(&status[0], 0u);
        // cgStart's residual (what the first iteration reads), summed from
        // none, and likewise the first iteration's own.
        atomicStore(&status[1u + 2u], 0u);
        atomicStore(&status[1u + 0u], 0u);
        atomicStore(&status[4], 0u);
        atomicStore(&status[AIR_BLAST_LIVE + (P.frame & 1u)], 0u);
    }
    workgroupBarrier();
    let dims = airDims(P.size);
    if (all(gid.xy < dims)) {
        gatherCell(gid.xy, dims, li);
    }
    workgroupBarrier();
    // The moments of each piece's gas, by the first thread in it, into its
    // room's (exact sums, common.wgsl's exactParts: a workgroup's at once).
    let mp = momentPiece[li];
    var first = mp != UNZONED;
    for (var k = 0u; first && k < li; k++) {
        first = momentPiece[k] != mp;
    }
    if (first) {
        var sum = array<f32, 5>(0.0, 0.0, 0.0, 0.0, 0.0);
        for (var k = li; k < 64u; k++) {
            if (momentPiece[k] == mp) {
                for (var j = 0u; j < 5u; j++) {
                    sum[j] += moments[k][j];
                }
            }
        }
        let s = zoneSlotAt(P.size, P.frame, roomOfPiece(mp)) + ZONE_MOMENTS;
        for (var j = 0u; j < 5u; j++) {
            let w = exactParts(sum[j], EXACT_COARSE);
            let old = atomicAdd(&sky[s + 2u * j], w.x);
            atomicAdd(&sky[s + 2u * j + 1u], w.y + exactCarry(old, w.x));
        }
    }
    // Into its rooms' sums for this step; and the tile's first workgroup
    // clears its pieces' for the next.
    if (li < ROOM_PIECES) {
        let origin = wid.xy * 8u * AIR_CELL;
        let k = roomPieceAt(P.size, origin, li);
        let open = atomicLoad(&zoneOpen[li]);
        let n = atomicLoad(&zoneCount[li]);
        if (n > 0u || open != 0u) {
            let s = zoneSlotAt(P.size, P.frame, roomOfPiece(k));
            atomicAdd(&sky[s + ZONE_AMOUNT], atomicLoad(&zoneAmount[li]));
            atomicAdd(&sky[s + ZONE_VOLUME], atomicLoad(&zoneVolume[li]));
            atomicAdd(&sky[s + ZONE_COUNT], n);
            atomicOr(&sky[s + ZONE_OPEN], open);
            atomicAdd(&sky[s + ZONE_SINK_DRAW], atomicLoad(&zoneSinkDraw[li]));
            atomicMax(&sky[s + ZONE_PIN], atomicLoad(&zonePin[li]));
        }
        if (atomicLoad(&wetOpen[li]) != 0u) {
            atomicOr(&sky[wetSlotAt(P.size, P.frame, wetOfPiece(k)) + WET_OPEN], 1u);
        }
        if (all(origin % ROOM_TILE == vec2<u32>(0u))) {
            let next = zoneSlotAt(P.size, P.frame + 1u, k);
            for (var w = 0u; w < ZONE_WORDS; w++) {
                atomicStore(&sky[next + w], 0u);
            }
            let wetNext = wetSlotAt(P.size, P.frame + 1u, k);
            for (var w = 0u; w < WET_WORDS; w++) {
                atomicStore(&sky[wetNext + w], 0u);
            }
        }
    }
}

// ------------------------------------------------------------------ blasts

// A blast's faces: open between any two cells of gas (or what gas passes
// through), as the flow's are. (passesAt is none outside the world.)
fn blastOpen(c: vec2<i32>, d: vec2<i32>) -> f32 {
    return f32(passesAt(c) && passesAt(c + d));
}

// The conductance of a cell's faces on the world's edges, the top's too (they
// let a wave out: its velocity there that of an outgoing wave,
// p sqrt(KV / KB)).
fn edgeConductance(c: vec2<i32>, dims: vec2<u32>) -> f32 {
    let edges = f32(c.x == 0) + f32(u32(c.x) + 1u == dims.x) + f32(c.y == 0) + f32(u32(c.y) + 1u == dims.y);
    return edges * f32(passesAt(c));
}

// The explosions' layer at rest: nothing in it at the end of the step
// before, nor put into it then (divergence's). Its passes return at once:
// 8 a step that took 0.55 ms in a world of 549 x 275 cells, 1 ms in one of
// 762 x 395 (F-3). It comes to rest exactly: what of it is under
// BLAST_QUIET (a thousandth of the air's unit, P_ATM to the atmosphere:
// some 8 Pa) goes to none.
const BLAST_QUIET: f32 = 1e-3;

fn blastQuiet() -> bool {
    return atomicLoad(&status[AIR_BLAST_LIVE + ((P.frame + 1u) & 1u)]) == 0u;
}

fn blastLive() {
    atomicStore(&status[AIR_BLAST_LIVE + (P.frame & 1u)], 1u);
}

// What of a wave is under BLAST_QUIET, none; what is left keeps the layer
// going a step more.
fn blastKept(x: f32) -> f32 {
    if (abs(x) < BLAST_QUIET) {
        return 0.0;
    }
    blastLive();
    return x;
}

@compute @workgroup_size(8, 8)
fn blastPressure(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = airDims(P.size);
    if (any(gid.xy >= dims) || blastQuiet()) {
        return;
    }
    let c = vec2<i32>(gid.xy);
    let idx = airIndex(c, dims);
    if (!passesAt(c)) {
        blast[idx].x = 0.0;
        return;
    }
    let s = blast[idx];
    var out = blastOpen(c, vec2<i32>(1, 0)) * s.y + blastOpen(c, vec2<i32>(0, 1)) * s.z;
    if (c.x > 0) {
        out -= blastOpen(c, vec2<i32>(-1, 0)) * blast[idx - 1u].y;
    }
    if (c.y > 0) {
        out -= blastOpen(c, vec2<i32>(0, -1)) * blast[idx - dims.x].z;
    }
    out += edgeConductance(c, dims) * s.x * sqrt(BLAST_KV / BLAST_KB);
    blast[idx].x = blastKept(clamp((s.x - BLAST_KB * out) * BLAST_KEEP, -MAX_P, MAX_P));
}

@compute @workgroup_size(8, 8)
fn blastVelocity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = airDims(P.size);
    if (any(gid.xy >= dims) || blastQuiet()) {
        return;
    }
    let c = vec2<i32>(gid.xy);
    let idx = airIndex(c, dims);
    let s = blast[idx];
    var u = 0.0;
    var v = 0.0;
    if (u32(c.x) + 1u < dims.x && blastOpen(c, vec2<i32>(1, 0)) > 0.0) {
        u = (s.y - BLAST_KV * (blast[idx + 1u].x - s.x)) * BLAST_KEEP;
    }
    if (u32(c.y) + 1u < dims.y && blastOpen(c, vec2<i32>(0, 1)) > 0.0) {
        v = (s.z - BLAST_KV * (blast[idx + dims.x].x - s.x)) * BLAST_KEEP;
    }
    blast[idx].y = blastKept(u);
    blast[idx].z = blastKept(v);
}

// ---------------------------------------------------------------- momentum

// The x velocity at a point (in cells), bilinear on the right faces, which
// sit at AIR_CELL (i + 1, j + 1/2); likewise y on the bottom faces at
// AIR_CELL (i + 1/2, j + 1).
fn sampleU(q: vec2<f32>, dims: vec2<u32>) -> f32 {
    let g = q / f32(AIR_CELL) - vec2<f32>(1.0, 0.5);
    let hi = vec2<i32>(dims) - 1;
    let i0 = clamp(vec2<i32>(floor(g)), vec2<i32>(0), hi);
    let i1 = min(i0 + 1, hi);
    let f = clamp(g - vec2<f32>(i0), vec2<f32>(0.0), vec2<f32>(1.0));
    let top = mix(air[airIndex(i0, dims)].y, air[airIndex(vec2<i32>(i1.x, i0.y), dims)].y, f.x);
    let bottom = mix(air[airIndex(vec2<i32>(i0.x, i1.y), dims)].y, air[airIndex(i1, dims)].y, f.x);
    return mix(top, bottom, f.y);
}

fn sampleV(q: vec2<f32>, dims: vec2<u32>) -> f32 {
    let g = q / f32(AIR_CELL) - vec2<f32>(0.5, 1.0);
    let hi = vec2<i32>(dims) - 1;
    let i0 = clamp(vec2<i32>(floor(g)), vec2<i32>(0), hi);
    let i1 = min(i0 + 1, hi);
    let f = clamp(g - vec2<f32>(i0), vec2<f32>(0.0), vec2<f32>(1.0));
    let top = mix(air[airIndex(i0, dims)].z, air[airIndex(vec2<i32>(i1.x, i0.y), dims)].z, f.x);
    let bottom = mix(air[airIndex(vec2<i32>(i0.x, i1.y), dims)].z, air[airIndex(i1, dims)].z, f.x);
    return mix(top, bottom, f.y);
}

// Buoyancy: air at temp (°C) among the outdoor air's (at outside), under the
// pull g, accelerates against it by g (1 - T_out / T): the difference in
// density over the outdoor air's, which comes to at most g however hot.
fn lift(temp: f32, outside: f32, g: f32) -> f32 {
    return -g * (1.0 - (outside + KELVIN) / max(temp + KELVIN, 1.0));
}

// The way the fan in cell p blows (none if it isn't one).
fn fanAt(p: vec2<i32>) -> vec2<f32> {
    if (!inWorld(p)) {
        return vec2<f32>(0.0);
    }
    let v = cellAt(p);
    let angle = f32(wordHeading(v.x)) / 256.0 * TAU;
    return select(vec2<f32>(0.0), vec2<f32>(cos(angle), sin(angle)), cellType(v) == FAN);
}

// A face's velocity component u after a step of the eddies' drag at the
// air's velocity vel there (cells a step): u / (1 + |vel| / ℓ), implicit,
// ℓ in cells.
fn eddyDrag(u: f32, vel: vec2<f32>) -> f32 {
    return u / (1.0 + length(vel) * P.dx / AIR_EDDY_M);
}

@compute @workgroup_size(8, 8)
fn momentum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = airDims(P.size);
    if (any(gid.xy >= dims)) {
        return;
    }
    let c = vec2<i32>(gid.xy);
    let idx = airIndex(c, dims);
    let a = mg[mgAt(0u, dims, c)];
    let cell = f32(AIR_CELL);
    let fan = fanAt(c);
    // What was put into the air on its faces since (airsource.wgsl): taken
    // in where the face is open, gone where it isn't.
    let push = vec2<f32>(takeSource(idx, SOURCE_RIGHT), takeSource(idx, SOURCE_DOWN));
    var u = 0.0;
    var v = 0.0;
    if (u32(c.x) + 1u < dims.x && a.z > 0.0) {
        let pos = vec2<f32>(f32(c.x + 1), f32(c.y) + 0.5) * cell;
        let vel = vec2<f32>(sampleU(pos, dims), sampleV(pos, dims));
        u = sampleU(pos - vel, dims);
        let temp = 0.5 * (work[idx].w + work[idx + 1u].w);
        u = eddyDrag(u + lift(temp, ambientAt(pos), pullAt(pos).x) + push.x, vel);
        let fanR = fanAt(c + vec2<i32>(1, 0));
        if (any(fan != vec2<f32>(0.0)) || any(fanR != vec2<f32>(0.0))) {
            u = mix(u, clamp(fan.x + fanR.x, -1.0, 1.0) * FAN_SPEED, FAN_MIX);
        }
        u = clamp(u, -MAX_V, MAX_V);
    }
    if (u32(c.y) + 1u < dims.y && a.w > 0.0) {
        let pos = vec2<f32>(f32(c.x) + 0.5, f32(c.y + 1)) * cell;
        let vel = vec2<f32>(sampleU(pos, dims), sampleV(pos, dims));
        v = sampleV(pos - vel, dims);
        let temp = 0.5 * (work[idx].w + work[idx + dims.x].w);
        v = eddyDrag(v + lift(temp, ambientAt(pos), pullAt(pos).y) + push.y, vel);
        let fanD = fanAt(c + vec2<i32>(0, 1));
        if (any(fan != vec2<f32>(0.0)) || any(fanD != vec2<f32>(0.0))) {
            v = mix(v, clamp(fan.y + fanD.y, -1.0, 1.0) * FAN_SPEED, FAN_MIX);
        }
        v = clamp(v, -MAX_V, MAX_V);
    }
    work[idx].y = u;
    work[idx].z = v;
}

// The pressure, atm, of the zone a cell of gas is in (its zone, gather's
// piece) at pos (cells): a sealed room's from its gas by the gas law (its
// amount over what it would take up at 1 atm, gather's sums), the open air's
// by height (and, for want of one, an unzoned pocket's). But a pocket the
// liquid closes in is at the pressure the liquid holds it at, its wet room's
// surface's (REWRITE_PLAN.md `bubble-pressure`, rooms.wgsl's surfaceAtm):
// where it is over it, what is past it bubbles out this step (carry.wgsl's
// wetOf), and where it is under it the liquid closes in on it (the liquid's
// own solve fills a cell it has left within a step or two: as a vacuum, it
// put an atmosphere across the water beside it, which the air's push then
// threw at its top speed, 4e). (Before the surface is found, the divergence,
// its own, which comes to the same there: a sealed zone's mean expansion is
// taken off.)
fn zoneAtm(piece: u32, pos: vec2<f32>) -> f32 {
    let open = openAirAt(pos);
    if (piece == UNZONED) {
        return pocketAtm(pos, open);
    }
    let room = roomOfPiece(piece);
    let s = zoneSlotAt(P.size, P.frame, room);
    let volume = f32(atomicLoad(&sky[s + ZONE_VOLUME]));
    if ((atomicLoad(&sky[s + ZONE_OPEN]) & ZONE_OUTSIDE) != 0u || volume <= 0.0) {
        return open;
    }
    let own = f32(atomicLoad(&sky[s + ZONE_AMOUNT])) / volume;
    let wet = wetRoomAt(vec2<u32>(pos / f32(AIR_CELL)));
    if (wet == NO_WET) {
        return own;
    }
    let w = wetSlotAt(P.size, P.frame, wet);
    if (isSurface(w, room, s)) {
        return own;
    }
    let surface = surfaceAtm(w, pos);
    return select(own, surface.x, surface.y > 0.0);
}

// A pocket of gas without a zone (sky.wgsl's UNNUMBERED: its tile holds
// more pieces than ROOM_PIECES, as a cloud of grains does): its own gas's
// pressure, by the gas law, which comes to its room's as the room's gas
// fills it. (The open air's, half an atmosphere off beside a sealed room's
// pressure, threw the grains round it up and about at the most a step
// allows: test/sandcloud.mjs.) The open air's for a cell holding no gas.
fn pocketAtm(pos: vec2<f32>, open: f32) -> f32 {
    let c = clamp(vec2<i32>(floor(pos)), vec2<i32>(0), vec2<i32>(P.size) - 1);
    let i = u32(c.y) * P.size.x + u32(c.x);
    let v = cells[i];
    if (!holdsGas(cellType(v))) {
        return open;
    }
    return gasAtm(gasFrom(cells[gasIndex(P.size.x * P.size.y, i)], i), cellTemp(v));
}

// The same, gauge, P_ATM an atmosphere, as the air's pressure is kept.
fn zonePressure(piece: u32, pos: vec2<f32>) -> f32 {
    return (zoneAtm(piece, pos) - 1.0) * P_ATM;
}


// Whether air cell idx of the zone piece is its sealed zone's pin: the first
// of its gas cells (ZONE_PIN), whose potential the projection holds as a
// black hole's (fixedAt). A sealed zone, held by ANCHOR alone, was nearly
// singular: in a world of 549 x 275 cells with water moving, the pockets of
// air the water closed in left the conjugate gradients swinging, short of
// the tolerance at 64 iterations a step (a replica of the solve in f64
// alike; held each at a cell, under it in 26). What the zone's flow can't
// balance is the pin's to give out (its equation the one left out). Only
// zones of PIN_MOST cells at most, the pockets: a room held at a point took
// half as many iterations again as one held weakly everywhere by ANCHOR
// (Heated room 11 to 18), and the rooms converged before.
const PIN_MOST: u32 = 256u;

fn pinned(piece: u32, idx: u32) -> bool {
    if (!sealedZone(piece)) {
        return false;
    }
    let s = zoneSlotAt(P.size, P.frame, roomOfPiece(piece));
    return atomicLoad(&sky[s + ZONE_PIN]) == ~idx && atomicLoad(&sky[s + ZONE_COUNT]) <= PIN_MOST;
}

// What a cell's gas at p atm, in a zone at zone atm, gives out in a step to
// come to the zone's pressure, in its volume (taken in, negative): over it,
// it expands, giving out what it holds as it goes, ln(p / zone); under it,
// what flows in comes at the zone's, p / zone - 1 (a vacuum, a whole cell's).
fn expansion(p: f32, zone: f32) -> f32 {
    let ratio = p / max(zone, 1e-6);
    return select(ratio - 1.0, log(ratio), ratio > 1.0);
}

// Per room piece of the workgroup's tile, the expansion of its cells summed
// as wideParts makes it (low and high words), before it goes into the zone's
// sum: every cell of the open air adding to its two words took 0.33 ms a
// step in a world of 762 x 395 cells (F-4). Whole numbers summed, so the
// zone's sum is the same to the last unit in any grouping.
var<workgroup> sourceLow: array<atomic<u32>, ROOM_PIECES>;
var<workgroup> sourceHigh: array<atomic<u32>, ROOM_PIECES>;

// Each cell's net outflow through its faces less what its gas gives out
// (expansion): the projection's right-hand side. (The edges' faces start at
// none, and come out of the projection.) And the expansion summed: for its
// sealed room, or the open air, whose mean cgStart takes off where the gas
// has nowhere to go.
@compute @workgroup_size(8, 8)
fn divergence(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    if (li < ROOM_PIECES) {
        atomicStore(&sourceLow[li], 0u);
        atomicStore(&sourceHigh[li], 0u);
    }
    workgroupBarrier();
    let dims = airDims(P.size);
    if (all(gid.xy < dims)) {
        divergenceAt(gid);
    }
    workgroupBarrier();
    if (li < ROOM_PIECES) {
        let low = atomicLoad(&sourceLow[li]);
        let high = atomicLoad(&sourceHigh[li]);
        if (low != 0u || high != 0u) {
            let k = roomPieceAt(P.size, wid.xy * 8u * AIR_CELL, li);
            let at = zoneSlotAt(P.size, P.frame, roomOfPiece(k)) + ZONE_SOURCE;
            let old = atomicAdd(&sky[at], low);
            atomicAdd(&sky[at + 1u], high + exactCarry(old, low));
        }
    }
}

fn divergenceAt(gid: vec3<u32>) {
    let dims = airDims(P.size);
    let c = vec2<i32>(gid.xy);
    let idx = airIndex(c, dims);
    let at = mgAt(0u, dims, c);
    let a = mg[at];
    let b = mg[at + 1u];
    var out = a.z * work[idx].y + a.w * work[idx].z;
    if (c.x > 0) {
        out -= mg[mgAt(0u, dims, c - vec2<i32>(1, 0))].z * work[idx - 1u].y;
    }
    if (c.y > 0) {
        out -= mg[mgAt(0u, dims, c - vec2<i32>(0, 1))].w * work[idx - dims.x].z;
    }
    var source = 0.0;
    let piece = bitcast<u32>(b.z);
    if (pinned(piece, idx) && b.w > 0.0 && b.y <= 0.0) {
        mg[at + 1u].y = 1.0;
    }
    if (b.w > 0.0 && b.y <= 0.0 && piece != UNZONED) {
        let v = cellAt(c);
        let pos = (vec2<f32>(c) + 0.5) * f32(AIR_CELL);
        let around = zoneAtm(piece, pos);
        source = clamp(expansion(gasAtm(gasAt(c), wordTempOf(v)), around), -WIDE_MOST, WIDE_MOST);
        if (cellType(v) == BLAST && source > 0.0) {
            blast[idx].x = clamp(blast[idx].x + BLAST_GAMMA * around * P_ATM * source, -MAX_P, MAX_P);
            blastLive();
        }
        let zone = zoneSlotAt(P.size, P.frame, roomOfPiece(piece));
        let w = wideParts(source, zoneNoise(idx, 3u));
        let old = atomicAdd(&sourceLow[piece % ROOM_PIECES], w.x);
        atomicAdd(&sourceHigh[piece % ROOM_PIECES], w.y + exactCarry(old, w.x));
        // The biggest zone beside the liquid or powder of its wet room, its
        // surface where it doesn't reach the open air (rooms.wgsl's
        // isSurface); not a speck (ZONE_MIN).
        let wet = wetRoomAt(gid.xy);
        let count = atomicLoad(&sky[zone + ZONE_COUNT]);
        if (wet != NO_WET && besideJoin(c) && count >= ZONE_MIN) {
            atomicMax(&sky[wetSlotAt(P.size, P.frame, wet) + WET_MOST], count);
        }
    }
    mg[at].y = out - source;
}

// ------------------------------------------------------------ the projection

// The potential that takes the divergence off: on each level, per cell,
// sum over faces of weight (neighbour's - own) - (edges + ANCHOR) own =
// right-hand side, the world's edges at none (and on the coarse levels
// the same for the correction). A fixed cell (a black hole) keeps its
// potential (its correction none).
//
// Level 1's operator: the Galerkin one of level 0 (restrictCell) over
// COARSEN. With the correction the same over a coarse cell's four
// (piecewise constant), the Galerkin operator is about twice as stiff as the
// coarse problem it stands for (a coarse face's weight the sum of the two
// fine faces it spans, where the problem rediscretized on cells twice the
// size has their mean), and its correction about half what it should be: as
// aggregation multigrid's coarse corrections are scaled up (Blaheta 1988;
// Braess, "Towards algebraic multigrid for elliptic problems of second
// order", Computing 55, 1995), short of 2, which would leave the coarse
// modes' error undamped. Only the once: a V-cycle on a level so scaled
// overshoots its operator's inverse by as much, and scaled again at each
// level down it compounded past 2, the V-cycle no longer positive, and the
// conjugate gradients stalled. The deeper levels are the plain Galerkin
// operators of theirs.
const COARSEN: f32 = 1.8;
// ... and each level takes on CORRECT times the correction of the coarser
// level over it (up's prolongation): the same half too little at every
// level of the deeper ones, where the worlds the app opens on have six or
// seven, taken up in the correction rather than in the operators below
// (which compounded), so each level's V-cycle stays as positive as its
// coarser one's (symmetric: the coarse correction scaled the same both
// ways). With it the CG takes 35-45 % fewer iterations (F-8: Dam 18 to 10
// a step, Fog box 18 to 12, a painted 1080p world 32-40 to 19-25); 1.7 a
// little fewer still, 1.9 more again in some, nearing 2.
const CORRECT: f32 = 1.5;

fn coarsenAt(l: u32) -> f32 {
    return select(1.0, COARSEN, l >= 1u);
}

// ANCHOR for each cell of level 0 that a cell of level l spans (as the
// Galerkin coarse operator has it: with only ANCHOR, a pocket of air sealed
// inside one coarse cell would have its correction four times too much a
// level down, sixteen two down, and grow step on step), as scaled with
// level 1's.
fn anchorAt(l: u32, c: vec2<i32>) -> f32 {
    let lo = c << vec2<u32>(l);
    let span = min(lo + vec2<i32>(1i << l), vec2<i32>(airDims(P.size))) - lo;
    return ANCHOR * f32(span.x * span.y) / coarsenAt(l);
}

// The potential is solved for to AIR_TOLERANCE, the most divergence it
// leaves in any cell a step (in its volume): the wind's carry (carry.wgsl)
// stays within what it mixes only as far as the flow is free of divergence.
// By conjugate gradients, preconditioned by a multigrid V-cycle (MGPCG:
// McAdams, Sifakis & Teran, "A parallel multigrid Poisson solver for fluids
// simulation on large grids", SCA 2010), warm-started from last step's
// potential: the V-cycle alone, its constant prolongation's coarse
// corrections a poor fit, took off only a third of what was left a cycle,
// and a scene with fans or fires ran out of cycles every step. An iteration
// (up to sim.js's AIR_CYCLES a step; `level.cycle`): the V-cycle on the
// residual (level 0's potential its answer z, its right-hand side minus the
// residual), cgRho (r . z), cgBeta, cgDirection, cgApply (A p, and p . A p),
// cgAlpha, cgUpdate (the potential and the residual, and the most of it
// left, into its slot in `status`); the passes of the next skip once that
// is under the tolerance. The iterations' slots in `status`, in turn:
// iteration k reads k - 1's (cgStart's for the first), sums its own, and
// clears k + 1's. The operator, A x = -b for the divergence b, is the
// stencil's, the world's edges, black holes (their potential fixed) and
// ANCHOR as Dirichlet terms, so symmetric and positive: the V-cycle is made
// symmetric too (smoothing red then black before, black then red after).
// 1e-4: a cell's gas amount is kept in an f16, rounded at random each step
// (common.wgsl's stochastic16), to some 1e-3 of it, and that rounding is
// itself a source the projection sees (still air's divergence, 1.5e-4 a
// cell on the mean, was the rounding of its amounts): solved to 1e-5, a
// hundredth of what the state holds, still air took 7 iterations a step
// and a calm scene twice what it takes at 1e-4, the checks the same
// (F-10: the scenes 7-35 % less a step, the worlds the app opens on 8-17 %).
const AIR_TOLERANCE: f32 = 1e-4;

// Per cell of level 0: the potential x, the residual r, the direction p,
// and A p; after them, four to a vec4, the sums (cgSum): r . z, alpha,
// beta, a spare, then a partial sum per workgroup of level 0 (cgRho,
// cgApply), summed by cgBeta and cgAlpha.
@group(0) @binding(13) var<storage, read_write> cg: array<vec4<f32>>;

fn cgSumAt(k: u32) -> f32 {
    let d = airDims(P.size);
    return cg[d.x * d.y + k / 4u][k % 4u];
}

fn setCgSum(k: u32, value: f32) {
    let d = airDims(P.size);
    cg[d.x * d.y + k / 4u][k % 4u] = value;
}

fn converged() -> bool {
    return bitcast<f32>(atomicLoad(&status[1u + (level.cycle + 2u) % 3u])) < AIR_TOLERANCE;
}

// (A v) at cell c of level 0, for v the potential (which 0) or the
// direction (2) in cg.
fn applyA(c: vec2<i32>, d: vec2<u32>, which: u32) -> f32 {
    let at = mgAt(0u, d, c);
    let a = mg[at];
    let i = airIndex(c, d);
    var diag = a.z + a.w + mg[at + 1u].x + anchorAt(0u, c);
    var sum = 0.0;
    if (u32(c.x) + 1u < d.x) {
        sum += a.z * cg[i + 1u][which];
    }
    if (u32(c.y) + 1u < d.y) {
        sum += a.w * cg[i + d.x][which];
    }
    if (c.x > 0) {
        let w = mg[mgAt(0u, d, c - vec2<i32>(1, 0))].z;
        sum += w * cg[i - 1u][which];
        diag += w;
    }
    if (c.y > 0) {
        let w = mg[mgAt(0u, d, c - vec2<i32>(0, 1))].w;
        sum += w * cg[i - d.x][which];
        diag += w;
    }
    return diag * cg[i][which] - sum;
}

// Whether level 0's cell c is fixed (a black hole: its potential held).
fn fixedAt(c: vec2<i32>, d: vec2<u32>) -> bool {
    return mg[mgAt(0u, d, c) + 1u].y > 0.0;
}

// What is taken off what each of a zone's cells gives out to come to its
// pressure (divergence's sums), so that its flow balances where its gas has
// nowhere else to go: in a sealed room, the mean of it, its pressure rising
// or falling instead; in a sealed room with a black hole, as much less as
// makes up what the hole takes (its sinkSpeed a face), the room's gas
// thinning evenly into it. None for the open air (out at the world's open
// edges).
fn meanExpansion(piece: u32) -> f32 {
    if (piece == UNZONED) {
        return 0.0;
    }
    let s = zoneSlotAt(P.size, P.frame, roomOfPiece(piece));
    let n = atomicLoad(&sky[s + ZONE_COUNT]);
    let open = atomicLoad(&sky[s + ZONE_OPEN]);
    if ((open & ZONE_OUTSIDE) != 0u || n == 0u) {
        return 0.0;
    }
    let mean = exactValue(vec2<u32>(atomicLoad(&sky[s + ZONE_SOURCE]), atomicLoad(&sky[s + ZONE_SOURCE + 1u])), WIDE_SCALE) / f32(n);
    if ((open & ZONE_SINK) != 0u) {
        return mean - f32(atomicLoad(&sky[s + ZONE_SINK_DRAW])) / (ZONE_SCALE * f32(n));
    }
    return mean;
}

// The residual r of A x = -b from last step's potential (a black hole's at
// its sinkSpeed below none, gather), the zone's mean expansion taken off b where its
// gas has nowhere to go (meanExpansion), the V-cycle's right-hand side -r and
// answer none, and the most of r left, for the first iteration to read.
@compute @workgroup_size(8, 8)
fn cgStart(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = airDims(P.size);
    let c = vec2<i32>(gid.xy);
    if (any(c >= vec2<i32>(d))) {
        return;
    }
    let i = airIndex(c, d);
    let at = mgAt(0u, d, c);
    var r = 0.0;
    if (!fixedAt(c, d)) {
        let b = mg[at + 1u];
        if (b.w > 0.0) {
            mg[at].y += meanExpansion(bitcast<u32>(b.z));
        }
        r = -mg[at].y - applyA(c, d, 0u);
    }
    cg[i] = vec4<f32>(cg[i].x, r, 0.0, 0.0);
    mg[at].x = 0.0;
    mg[at].y = -r;
    atomicMax(&status[1u + 2u], bitcast<u32>(abs(r)));
}

// A partial sum over workgroups of level 0, into the sums (cgSumAt).
var<workgroup> partial: array<f32, 64>;
var<workgroup> skipping: u32;

fn sumGroup(li: u32, wid: vec2<u32>, value: f32) {
    partial[li] = value;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s >>= 1u) {
        if (li < s) {
            partial[li] += partial[li + s];
        }
        workgroupBarrier();
    }
    if (li == 0u) {
        let groups = (airDims(P.size).x + 7u) / 8u;
        setCgSum(4u + wid.y * groups + wid.x, partial[0]);
    }
}

// Whether this iteration is skipped (converged), the same for the whole
// workgroup (its barriers).
fn skipGroup(li: u32) -> bool {
    if (li == 0u) {
        skipping = u32(converged());
    }
    return workgroupUniformLoad(&skipping) != 0u;
}

// The partial sums summed, by one workgroup.
var<workgroup> wide: array<f32, 256>;

fn sumAll(li: u32) -> f32 {
    let dims = airDims(P.size);
    let n = ((dims.x + 7u) / 8u) * ((dims.y + 7u) / 8u);
    var s = 0.0;
    for (var k = li; k < n; k += 256u) {
        s += cgSumAt(4u + k);
    }
    wide[li] = s;
    workgroupBarrier();
    for (var h = 128u; h > 0u; h >>= 1u) {
        if (li < h) {
            wide[li] += wide[li + h];
        }
        workgroupBarrier();
    }
    return wide[0];
}

@compute @workgroup_size(8, 8)
fn cgRho(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) li: u32) {
    if (skipGroup(li)) {
        return;
    }
    let d = airDims(P.size);
    let c = vec2<i32>(gid.xy);
    var v = 0.0;
    if (all(c < vec2<i32>(d))) {
        v = cg[airIndex(c, d)].y * mg[mgAt(0u, d, c)].x;
    }
    sumGroup(li, wid.xy, v);
}

@compute @workgroup_size(256)
fn cgBeta(@builtin(local_invocation_index) li: u32) {
    if (skipGroup(li)) {
        return;
    }
    let rho = sumAll(li);
    if (li == 0u) {
        let before = cgSumAt(0u);
        setCgSum(2u, select(rho / before, 0.0, level.cycle == 0u || before == 0.0));
        setCgSum(0u, rho);
    }
}

// p = z + beta p (none at a fixed cell).
@compute @workgroup_size(8, 8)
fn cgDirection(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = airDims(P.size);
    let c = vec2<i32>(gid.xy);
    if (any(c >= vec2<i32>(d)) || converged()) {
        return;
    }
    let i = airIndex(c, d);
    let p = select(mg[mgAt(0u, d, c)].x + cgSumAt(2u) * cg[i].z, 0.0, fixedAt(c, d));
    cg[i].z = p;
}

// A p, and the partial sums of p . A p.
@compute @workgroup_size(8, 8)
fn cgApply(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) li: u32) {
    if (skipGroup(li)) {
        return;
    }
    let d = airDims(P.size);
    let c = vec2<i32>(gid.xy);
    var v = 0.0;
    if (all(c < vec2<i32>(d))) {
        let i = airIndex(c, d);
        let q = select(applyA(c, d, 2u), 0.0, fixedAt(c, d));
        cg[i].w = q;
        v = cg[i].z * q;
    }
    sumGroup(li, wid.xy, v);
}

@compute @workgroup_size(256)
fn cgAlpha(@builtin(local_invocation_index) li: u32) {
    if (skipGroup(li)) {
        return;
    }
    let pq = sumAll(li);
    if (li == 0u) {
        setCgSum(1u, select(cgSumAt(0u) / pq, 0.0, pq <= 0.0));
    }
}

// x += alpha p, r -= alpha A p; the V-cycle's right-hand side and answer for
// the next iteration; the most of r left.
@compute @workgroup_size(8, 8)
fn cgUpdate(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (all(gid == vec3<u32>(0u))) {
        atomicStore(&status[1u + (level.cycle + 1u) % 3u], 0u);
        if (!converged()) {
            atomicAdd(&status[4], 1u);
        }
    }
    let d = airDims(P.size);
    let c = vec2<i32>(gid.xy);
    if (any(c >= vec2<i32>(d)) || converged()) {
        return;
    }
    let i = airIndex(c, d);
    let at = mgAt(0u, d, c);
    let alpha = cgSumAt(1u);
    let v = cg[i];
    let r = v.y - alpha * v.w;
    cg[i] = vec4<f32>(v.x + alpha * v.z, r, v.z, v.w);
    mg[at].x = 0.0;
    mg[at].y = -r;
    atomicMax(&status[1u + level.cycle % 3u], bitcast<u32>(abs(r)));
}

// The V-cycle's passes on a level, by tiles (F-2): a workgroup takes a
// TILE x TILE square of level l's cells, with HALO more each side, into its
// memory, and sweeps them there red and black in turn, 2 AIR_SMOOTH sweeps
// as passes over the whole level would, each sweep leaving one more ring in
// from the edge wrong (its cells went by ones beyond, not held, or wrong
// already): the square comes out as the passes would leave it, to the bit
// but for where a compiler fuses a multiply and an add. Before them, `down`
// on its way to the coarser level, the square's residual restricted into
// the coarser cells it spans (one ring more for the residual's neighbours);
// `up` on its way back, the coarser level's correction prolonged onto it as
// it is loaded. One pass a level each way where there were 4 AIR_SMOOTH + 1:
// each cost its dispatch, about 1.7 us here, which was most of an iteration
// on the worlds the scenes have (13 to 15 passes an iteration, from 37 to
// 47). A tile reads no word another tile writes in the same pass: on the
// way down the correction starts at none (restrictCell's, cgUpdate's), so
// isn't read, and the smoothed one goes where `up` takes it from (preAt);
// on the way back that is read and the correction written.
const AIR_SMOOTH: u32 = 2u;
const TILE: u32 = 16u;
const HALO: u32 = 2u * AIR_SMOOTH + 1u;
const SPAN: u32 = TILE + 2u * HALO;
const TILE_CELLS: u32 = SPAN * SPAN;
// Per cell of the tile: the correction, the right-hand side, the right and
// lower faces' weights, and the stencil's diagonal (the operator's above, the
// edges' and the anchor's in it;
// below none for a cell not swept: fixed, or off the level).
var<workgroup> tileX: array<f32, TILE_CELLS>;
var<workgroup> tileRhs: array<f32, TILE_CELLS>;
var<workgroup> tileRight: array<f32, TILE_CELLS>;
var<workgroup> tileDown: array<f32, TILE_CELLS>;
var<workgroup> tileDiag: array<f32, TILE_CELLS>;

// Level l's cell of the tile k (from `corner`, its first held cell).
fn tileCell(corner: vec2<i32>, k: u32) -> vec2<i32> {
    return corner + vec2<i32>(i32(k % SPAN), i32(k / SPAN));
}

// Where `down` leaves a level's smoothed correction for `up`: level 0's in
// the CG's A p (cg's w, unused from cgUpdate to the next cgApply), a coarser
// level's in its spare word (restrictCell's none).
fn preAt(l: u32, c: vec2<i32>) -> f32 {
    if (l == 0u) {
        return cg[airIndex(c, airDims(P.size))].w;
    }
    return mg[mgAt(levelBase(l), levelDims(l), c) + 1u].z;
}

fn setPre(l: u32, c: vec2<i32>, x: f32) {
    if (l == 0u) {
        cg[airIndex(c, airDims(P.size))].w = x;
        return;
    }
    mg[mgAt(levelBase(l), levelDims(l), c) + 1u].z = x;
}

// The tile's cells into the workgroup's memory: on the way down their
// correction none; on the way back (`prolonged`) the smoothed one down left
// (preAt), the coarser level's correction on (prolong's: none on a fixed
// cell).
fn loadTile(l: u32, corner: vec2<i32>, li: u32, prolonged: bool) {
    let d = levelDims(l);
    let base = levelBase(l);
    for (var k = li; k < TILE_CELLS; k += TILE * TILE) {
        let c = tileCell(corner, k);
        var a = vec4<f32>(0.0);
        var diag = -1.0;
        if (all(c >= vec2<i32>(0)) && all(c < vec2<i32>(d))) {
            let at = mgAt(base, d, c);
            a = vec4<f32>(0.0, mg[at].yzw);
            let b = mg[at + 1u];
            if (prolonged) {
                a.x = preAt(l, c);
            }
            if (b.y <= 0.0) {
                if (prolonged) {
                    a.x += CORRECT * mg[mgAt(levelBase(l + 1u), levelDims(l + 1u), c / 2)].x;
                }
                diag = a.z + a.w + b.x + anchorAt(l, c);
                if (c.x > 0) {
                    diag += mg[mgAt(base, d, c - vec2<i32>(1, 0))].z;
                }
                if (c.y > 0) {
                    diag += mg[mgAt(base, d, c - vec2<i32>(0, 1))].w;
                }
            }
        }
        tileX[k] = a.x;
        tileRhs[k] = a.y;
        tileRight[k] = a.z;
        tileDown[k] = a.w;
        tileDiag[k] = diag;
    }
    workgroupBarrier();
}

// The operator's sum of the neighbours' potentials, each times its face's
// weight, at the tile's cell k (level l's c, of dims d), from the tile's.
fn tileSum(k: u32, c: vec2<i32>, d: vec2<i32>) -> f32 {
    var sum = 0.0;
    if (c.x + 1 < d.x) {
        sum += tileRight[k] * tileX[k + 1u];
    }
    if (c.y + 1 < d.y) {
        sum += tileDown[k] * tileX[k + SPAN];
    }
    if (c.x > 0) {
        sum += tileRight[k - 1u] * tileX[k - 1u];
    }
    if (c.y > 0) {
        sum += tileDown[k - SPAN] * tileX[k - SPAN];
    }
    return sum;
}

// One sweep of the tile's cells of one colour (as smoothing's passes were:
// (c.x + c.y) & 1). The outermost ring, whose neighbours the tile doesn't
// hold, is left as it is: wrong, as each sweep's next ring in is.
fn sweepTile(l: u32, corner: vec2<i32>, li: u32, color: i32) {
    let d = vec2<i32>(levelDims(l));
    for (var k = li; k < TILE_CELLS; k += TILE * TILE) {
        let t = vec2<u32>(k % SPAN, k / SPAN);
        let c = tileCell(corner, k);
        if (((c.x + c.y) & 1) != color || tileDiag[k] < 0.0 || any(t == vec2<u32>(0u)) || any(t == vec2<u32>(SPAN - 1u))) {
            continue;
        }
        tileX[k] = (tileSum(k, c, d) - tileRhs[k]) / tileDiag[k];
    }
    workgroupBarrier();
}

// The tile's own square back into level l: its cells' corrections, or, on
// the way down (`pre`), where up takes them from.
fn storeTile(l: u32, corner: vec2<i32>, li: u32, pre: bool) {
    let d = levelDims(l);
    let t = vec2<u32>(li % TILE, li / TILE) + HALO;
    let k = t.y * SPAN + t.x;
    let c = tileCell(corner, k);
    if (all(c < vec2<i32>(d)) && tileDiag[k] >= 0.0) {
        if (pre) {
            setPre(l, c, tileX[k]);
        } else {
            mg[mgAt(levelBase(l), d, c)].x = tileX[k];
        }
    }
}

// Level l + 1's cell c from its four on level l: the faces' weights (a
// coarse face's, the sum of the two it spans, as the finer level's
// operator acts on a correction the same across the four, which is how
// prolong hands it back: the Galerkin coarse operator, over COARSEN), the
// edges' likewise, whether fixed, and the error to correct, the sum of what
// is left of the equations in the four (from the tile, at its corner). The
// correction starts at none.
fn restrictCell(l: u32, corner: vec2<i32>, c: vec2<i32>) {
    let fd = levelDims(l);
    let fbase = levelBase(l);
    var rhs = 0.0;
    var edge = 0.0;
    var free = 0u;
    var held = 0u;
    for (var q = 0; q < 4; q++) {
        let f = 2 * c + vec2<i32>(q & 1, q >> 1);
        if (any(f >= vec2<i32>(fd))) {
            continue;
        }
        let at = mgAt(fbase, fd, f);
        let b = mg[at + 1u];
        if (b.y > 0.0) {
            held++;
            continue;
        }
        free++;
        // Its faces to held cells, as the world's edges are: the correction
        // there none.
        let a = mg[at];
        edge += b.x;
        if (u32(f.x) + 1u < fd.x && heldAt(fbase, fd, f + vec2<i32>(1, 0))) {
            edge += a.z;
        }
        if (u32(f.y) + 1u < fd.y && heldAt(fbase, fd, f + vec2<i32>(0, 1))) {
            edge += a.w;
        }
        if (f.x > 0 && heldAt(fbase, fd, f - vec2<i32>(1, 0))) {
            edge += mg[mgAt(fbase, fd, f - vec2<i32>(1, 0))].z;
        }
        if (f.y > 0 && heldAt(fbase, fd, f - vec2<i32>(0, 1))) {
            edge += mg[mgAt(fbase, fd, f - vec2<i32>(0, 1))].w;
        }
        let t = vec2<u32>(f - corner);
        let k = t.y * SPAN + t.x;
        rhs += tileRhs[k] - (tileSum(k, f, vec2<i32>(fd)) - tileDiag[k] * tileX[k]);
    }
    // The right face: the fine faces on the coarse cell's right column (the
    // last one there is, at the world's edge), between cells not held.
    let right = min(2 * c.x + 1, i32(fd.x) - 1);
    let bottom = min(2 * c.y + 1, i32(fd.y) - 1);
    var wR = 0.0;
    var wD = 0.0;
    for (var r = 0; r < 2; r++) {
        let y = 2 * c.y + r;
        if (y < i32(fd.y) && !heldAt(fbase, fd, vec2<i32>(right, y))
            && (u32(right) + 1u >= fd.x || !heldAt(fbase, fd, vec2<i32>(right + 1, y)))) {
            wR += mg[mgAt(fbase, fd, vec2<i32>(right, y))].z;
        }
        let x = 2 * c.x + r;
        if (x < i32(fd.x) && !heldAt(fbase, fd, vec2<i32>(x, bottom))
            && (u32(bottom) + 1u >= fd.y || !heldAt(fbase, fd, vec2<i32>(x, bottom + 1)))) {
            wD += mg[mgAt(fbase, fd, vec2<i32>(x, bottom))].w;
        }
    }
    // Held only where all its cells are (a black hole's whole, or a pin's
    // alone): the Galerkin operator of its cells not held (a pin holding the
    // coarse cell it lay in at every level slowed a sealed room's solve by
    // half).
    let fixed = free == 0u && held > 0u;
    let lc = l + 1u;
    let at = mgAt(levelBase(lc), levelDims(lc), c);
    // (Level 1 over COARSEN; the deeper levels from it as they are.)
    let scale = select(1.0, 1.0 / COARSEN, lc == 1u);
    mg[at] = vec4<f32>(0.0, select(rhs, 0.0, fixed), wR * scale, wD * scale);
    mg[at + 1u] = vec4<f32>(edge * scale, f32(fixed), 0.0, 0.0);
}

// Whether level cell f (of a level at base, dims d) is held (fixedAt's).
fn heldAt(base: u32, d: vec2<u32>, f: vec2<i32>) -> bool {
    return mg[mgAt(base, d, f) + 1u].y > 0.0;
}

// Smoothing before a level's correction (red then black), and the residual
// restricted into the coarser level (restrictCell), but on the coarsest
// level to smooth (smoothTile: a level 0 that is the coarsest itself, its
// correction left where the coarsest level's solve starts from).
fn smoothDown(wid: vec2<u32>, li: u32, restricts: bool) {
    if (skipGroup(li)) {
        return;
    }
    let l = level.index;
    let origin = vec2<i32>(wid * TILE);
    let corner = origin - vec2<i32>(i32(HALO));
    loadTile(l, corner, li, false);
    for (var i = 0u; i < AIR_SMOOTH; i++) {
        sweepTile(l, corner, li, 0);
        sweepTile(l, corner, li, 1);
    }
    storeTile(l, corner, li, restricts);
    if (restricts && li < (TILE / 2u) * (TILE / 2u)) {
        let c = origin / 2 + vec2<i32>(i32(li % (TILE / 2u)), i32(li / (TILE / 2u)));
        if (all(c < vec2<i32>(levelDims(l + 1u)))) {
            restrictCell(l, corner, c);
        }
    }
}

@compute @workgroup_size(TILE, TILE)
fn down(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    smoothDown(wid.xy, li, true);
}

@compute @workgroup_size(TILE, TILE)
fn smoothTile(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    smoothDown(wid.xy, li, false);
}

// Level l takes on the correction of the coarser level over it, and is
// smoothed after it (black then red: the V-cycle symmetric).
@compute @workgroup_size(TILE, TILE)
fn up(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    if (skipGroup(li)) {
        return;
    }
    let l = level.index;
    let corner = vec2<i32>(wid.xy * TILE) - vec2<i32>(i32(HALO));
    loadTile(l, corner, li, true);
    for (var i = 0u; i < AIR_SMOOTH; i++) {
        sweepTile(l, corner, li, 1);
        sweepTile(l, corner, li, 0);
    }
    storeTile(l, corner, li, false);
}

// The coarsest level (AIR_COARSEST cells at most, as sim.js sees to), in one
// workgroup's memory, swept over and over until it has settled.
var<workgroup> coarseA: array<vec4<f32>, AIR_COARSEST>;
var<workgroup> coarseB: array<vec2<f32>, AIR_COARSEST>;
var<workgroup> coarseDone: u32;

@compute @workgroup_size(256)
fn solveCoarsest(@builtin(local_invocation_index) li: u32) {
    if (li == 0u) {
        coarseDone = u32(converged());
    }
    if (workgroupUniformLoad(&coarseDone) != 0u) {
        return;
    }
    let l = level.index;
    let d = levelDims(l);
    let base = levelBase(l);
    let n = min(d.x * d.y, AIR_COARSEST);
    for (var j = li; j < n; j += 256u) {
        let at = base + 2u * j;
        coarseA[j] = mg[at];
        coarseB[j] = mg[at + 1u].xy;
    }
    workgroupBarrier();
    // Red and black in turn, then the same backwards (so the V-cycle, the
    // conjugate gradients' preconditioner, stays symmetric).
    for (var sweep = 0; sweep < 2 * COARSE_SWEEPS; sweep++) {
        let color = select((2 * COARSE_SWEEPS - 1 - sweep) & 1, sweep & 1, sweep < COARSE_SWEEPS);
        for (var j = li; j < n; j += 256u) {
            let c = vec2<i32>(i32(j % d.x), i32(j / d.x));
            let a = coarseA[j];
            let b = coarseB[j];
            if (((c.x + c.y) & 1) != color || b.y > 0.0) {
                continue;
            }
            var sum = 0.0;
            var diag = a.z + a.w + b.x + anchorAt(l, c);
            if (u32(c.x) + 1u < d.x) {
                sum += a.z * coarseA[j + 1u].x;
            }
            if (u32(c.y) + 1u < d.y) {
                sum += a.w * coarseA[j + d.x].x;
            }
            if (c.x > 0) {
                let o = coarseA[j - 1u];
                sum += o.z * o.x;
                diag += o.z;
            }
            if (c.y > 0) {
                let o = coarseA[j - d.x];
                sum += o.w * o.x;
                diag += o.w;
            }
            coarseA[j].x = mix(a.x, (sum - a.y) / diag, COARSE_OMEGA);
        }
        workgroupBarrier();
    }
    for (var j = li; j < n; j += 256u) {
        mg[base + 2u * j].x = coarseA[j].x;
    }
}

// ------------------------------------------------------------ the air as read

// Whether the zone of a cell (gather's piece) is a sealed one: a room whose
// potential is held by nothing but ANCHOR (not the open air's edges, nor a
// black hole's), so floats at a level of its own.
fn sealedZone(piece: u32) -> bool {
    if (piece == UNZONED) {
        return false;
    }
    let s = zoneSlotAt(P.size, P.frame, roomOfPiece(piece));
    return (atomicLoad(&sky[s + ZONE_OPEN]) & (ZONE_OUTSIDE | ZONE_SINK)) == 0u;
}

// Whether the zone of a cell of gas is a speck (common.wgsl's ZONE_MIN):
// sealed, and of fewer cells.
fn speckZone(piece: u32) -> bool {
    return sealedZone(piece)
        && atomicLoad(&sky[zoneSlotAt(P.size, P.frame, roomOfPiece(piece)) + ZONE_COUNT]) < ZONE_MIN;
}

// Per room piece of the workgroup's tile, its sealed zone's potential summed
// over the workgroup's cells, in fixed point (POTENTIAL_SCALE), before it
// goes into the zone's sum: ten thousand cells of a sealed room adding to
// one word by compare-and-swap took the scenes of one three times as long.
var<workgroup> potentialSum: array<atomic<i32>, ROOM_PIECES>;
const POTENTIAL_SCALE: f32 = 1024.0;

// Once the projection is solved, before what reads it: which zone is each
// wet room's surface, the one with WET_MOST cells beside its liquid
// (WET_SURFACE, its room's number + 1, the highest of any as many); and the
// potential summed over each sealed zone's cells of gas (ZONE_POTENTIAL),
// the level its dynamic pressure is measured from (wholePressure). The open
// air's is held at the outside's, none, at the world's edges.
@compute @workgroup_size(8, 8)
fn zonePotential(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    if (li < ROOM_PIECES) {
        atomicStore(&potentialSum[li], 0);
    }
    if (all(gid == vec3<u32>(0u))) {
        atomicMax(&status[AIR_MOST_CYCLES], atomicLoad(&status[4]));
    }
    workgroupBarrier();
    let dims = airDims(P.size);
    let c = vec2<i32>(gid.xy);
    if (all(gid.xy < dims)) {
        let b = mg[mgAt(0u, dims, c) + 1u];
        let piece = bitcast<u32>(b.z);
        if (b.w > 0.0 && b.y <= 0.0 && piece != UNZONED) {
            let room = roomOfPiece(piece);
            let zone = zoneSlotAt(P.size, P.frame, room);
            let wet = wetRoomAt(gid.xy);
            if (wet != NO_WET && besideJoin(c)) {
                let w = wetSlotAt(P.size, P.frame, wet);
                if (atomicLoad(&sky[zone + ZONE_COUNT]) == atomicLoad(&sky[w + WET_MOST])) {
                    atomicMax(&sky[w + WET_SURFACE], room + 1u);
                }
            }
            if (sealedZone(piece)) {
                let x = cg[airIndex(c, dims)].x;
                atomicAdd(&potentialSum[piece % ROOM_PIECES], i32(round(x * POTENTIAL_SCALE)));
            }
        }
    }
    workgroupBarrier();
    if (li < ROOM_PIECES) {
        let sum = atomicLoad(&potentialSum[li]);
        if (sum != 0) {
            let k = roomPieceAt(P.size, wid.xy * 8u * AIR_CELL, li);
            let at = zoneSlotAt(P.size, P.frame, roomOfPiece(k)) + ZONE_POTENTIAL;
            let w = exactWhole(sum);
            let old = atomicAdd(&sky[at], w.x);
            atomicAdd(&sky[at + 1u], w.y + exactCarry(old, w.x));
        }
    }
}

// The air's pressure in the cell c of gas (idx its index), as the other
// passes read it (P_ATM an atmosphere, gauge): its zone's background, the
// dynamic pressure the projection found, and a blast's. The potential φ
// (cells² a step) stands for ρ φ (dx / dt)², ρ the air's density at its
// zone's pressure and 20 °C, from the open air's at the world's edges, or a
// sealed zone's mean (zonePotential), which its background stands for: a
// pocket's floating level, as pressure against the water beside it, which
// the air's push (airsample.wgsl) took for an atmosphere's. A fan in a wall
// between two rooms, one zone, shows its push as the far room's pressure
// over the near one's.
fn wholePressure(c: vec2<i32>, idx: u32) -> f32 {
    let pos = (vec2<f32>(c) + 0.5) * f32(AIR_CELL);
    let piece = bitcast<u32>(mg[mgAt(0u, airDims(P.size), c) + 1u].z);
    let atm = zoneAtm(piece, pos);
    var level = 0.0;
    if (sealedZone(piece)) {
        let s = zoneSlotAt(P.size, P.frame, roomOfPiece(piece));
        level = exactValue(vec2<u32>(atomicLoad(&sky[s + ZONE_POTENTIAL]), atomicLoad(&sky[s + ZONE_POTENTIAL + 1u])), POTENTIAL_SCALE)
            / f32(max(atomicLoad(&sky[s + ZONE_COUNT]), 1u));
    }
    let s = P.dx / P.dt;
    let dynamic = elems[AIR].rho * atm * (cg[idx].x - level) * s * s / 101325.0 * P_ATM;
    return (atm - 1.0) * P_ATM + dynamic + blast[idx].x;
}

// Each sealed zone's potential taken to its mean, none, once project has
// read it (its level, zonePotential's: what the pressure is measured from,
// and the potential only from a neighbour's in the zone), so that its pin
// holds next step near none, where f32 has room for the flow's potentials.
@compute @workgroup_size(8, 8)
fn recentre(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = airDims(P.size);
    if (any(gid.xy >= dims)) {
        return;
    }
    let c = vec2<i32>(gid.xy);
    let b = mg[mgAt(0u, dims, c) + 1u];
    let piece = bitcast<u32>(b.z);
    if (b.w <= 0.0 || !sealedZone(piece)) {
        return;
    }
    let s = zoneSlotAt(P.size, P.frame, roomOfPiece(piece));
    let level = exactValue(vec2<u32>(atomicLoad(&sky[s + ZONE_POTENTIAL]), atomicLoad(&sky[s + ZONE_POTENTIAL + 1u])), POTENTIAL_SCALE)
        / f32(max(atomicLoad(&sky[s + ZONE_COUNT]), 1u));
    cg[airIndex(c, dims)].x -= level;
}

@compute @workgroup_size(8, 8)
fn project(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = airDims(P.size);
    if (any(gid.xy >= dims)) {
        return;
    }
    let c = vec2<i32>(gid.xy);
    let idx = airIndex(c, dims);
    let at = mgAt(0u, dims, c);
    let a = mg[at];
    let b = mg[at + 1u];
    let w = work[idx];
    var u = 0.0;
    var v = 0.0;
    // (Not bounded again after the projection, which would leave it no longer
    // free of divergence: the wind's carry relies on that. MAX_V bounds what
    // goes into it.)
    let phi = cg[idx].x;
    if (a.z > 0.0) {
        let next = select(0.0, cg[idx + 1u].x, u32(c.x) + 1u < dims.x);
        u = w.y - (next - phi);
    }
    if (a.w > 0.0) {
        let next = select(0.0, cg[idx + dims.x].x, u32(c.y) + 1u < dims.y);
        v = w.z - (next - phi);
    }
    // What the cell's flow gives out, all its faces' outward flows, for the
    // carry's substeps: its left and top faces as their cells' projections
    // leave them, the world's left edge's as the potential there (outside
    // at none, its velocity starting at none).
    var given = max(u, 0.0) + max(v, 0.0);
    if (c.x > 0) {
        let n = mg[mgAt(0u, dims, c - vec2<i32>(1, 0))];
        if (n.z > 0.0) {
            given += max(-(work[idx - 1u].y - (phi - cg[idx - 1u].x)), 0.0);
        }
    } else if (b.x > 0.0) {
        given += max(phi, 0.0);
    }
    if (c.y > 0) {
        let n = mg[mgAt(0u, dims, c - vec2<i32>(0, 1))];
        if (n.w > 0.0) {
            given += max(-(work[idx - dims.x].z - (phi - cg[idx - dims.x].x)), 0.0);
        }
    }
    atomicMax(&status[0], bitcast<u32>(given));
    // A speck of gas (common.wgsl's ZONE_MIN) is shown as a gap with none,
    // as the gas the liquid counts as its own is: what passes read of it is
    // what is beside it, and a body's faces don't meet it (airsample.wgsl's
    // gasCellAt).
    let speck = b.w > 0.0 && speckZone(bitcast<u32>(b.z));
    if (b.w > 0.0 && !speck) {
        air[idx] = vec4<f32>(wholePressure(c, idx), u, v, wordTempOf(cellAt(c)));
        return;
    }
    // No gas: the background pressure of the gas beside it (their zones'
    // mean); with none, the mean of the liquid or powder beside it as of
    // last step, which passes it on (the water in a pressure cooker under its
    // steam), so a liquid's boiling point and what holds it in feel it. A
    // solid passes none on. Not a blast's: passed on from cell to cell, with
    // nothing to let it go, it stayed in a heap of rubble the blast left
    // (and broke what the heap leant on, into more rubble). Nor the dynamic
    // pressure: what pushes the cell is the difference across it, its own
    // pressure none of it (airsample.wgsl).
    //
    // Liquid is at its wet room's surface's pressure (rooms.wgsl's
    // surfaceAtm: the gas over it) all through, as a liquid at rest is but
    // for its own weight's (P2's): passed on a cell a step, a tank's water
    // took that many steps to come to what the air over it had come to, and
    // the air's push threw its top up out of it (4e). And so is a powder, the
    // air in its pores (rooms.wgsl's joinsAt): a heap's, passed on, held
    // what a vacuum among its grains had once been.
    if (joinsAt(c)) {
        let wet = wetRoomAt(gid.xy);
        if (wet != NO_WET) {
            let surface = surfaceAtm(wetSlotAt(P.size, P.frame, wet), (vec2<f32>(c) + 0.5) * f32(AIR_CELL));
            if (surface.y > 0.0) {
                air[idx] = vec4<f32>((surface.x - 1.0) * P_ATM, 0.0, 0.0, AIR_BLOCKED);
                return;
            }
        }
    }
    var gas = vec2<f32>(0.0);
    var fill = vec2<f32>(0.0);
    for (var k = 0; k < 4; k++) {
        let n = c + select(vec2<i32>(0, (k & 1) * 2 - 1), vec2<i32>((k & 1) * 2 - 1, 0), k < 2);
        if (any(n < vec2<i32>(0)) || any(n >= vec2<i32>(dims))) {
            continue;
        }
        let nb = mg[mgAt(0u, dims, n) + 1u];
        let piece = bitcast<u32>(nb.z);
        if (nb.w > 0.0 && !speckZone(piece)) {
            gas += vec2<f32>(zonePressure(piece, (vec2<f32>(n) + 0.5) * f32(AIR_CELL)), 1.0);
        } else if (nb.w < 0.0) {
            fill += vec2<f32>(work[airIndex(n, dims)].x, 1.0);
        }
    }
    let p = select(select(0.0, fill.x / fill.y, fill.y > 0.0), gas.x / gas.y, gas.y > 0.0);
    // (A speck keeps a blast's wave, which is in its gas as in the gas
    // beside it: left out, the rubble's pockets by a wall a charge hit
    // shielded it.)
    let wave = select(0.0, blast[idx].x, speck);
    air[idx] = vec4<f32>(p + wave, 0.0, 0.0, select(AIR_SOLID, AIR_BLOCKED, b.w != 0.0));
}

// ------------------------------------------------------------------ wind tool

// Pushes the air along the brush stroke, through what is put into the air
// (airsource.wgsl), which the next step's momentum takes in (and bounds by
// MAX_V); the projection keeps what isn't squeezing it.
@compute @workgroup_size(8, 8)
fn wind(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = airDims(B.size);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    let center = (vec2<f32>(gid.xy) + 0.5) * f32(AIR_CELL);
    let ab = B.b - B.a;
    let h = clamp(dot(center - B.a, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0);
    if (length(center - B.a - ab * h) > B.radius + f32(AIR_CELL) * 0.5) {
        return;
    }
    let idx = gid.y * dims.x + gid.x;
    if (air[idx].w <= AIR_BLOCKED) {
        return;
    }
    pushAir(idx, ab * B.strength);
}
