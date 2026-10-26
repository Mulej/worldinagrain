// Liquids, moved as particles through a real pressure solve: FLIP on a
// staggered (MAC) grid laid over the cells (Zhu & Bridson 2005, "Animating
// sand as a fluid"; Bridson, Fluid Simulation for Computer Graphics, 2nd
// ed., 2015, chapters 5-8).
//
// The particles carry the liquid's position and velocity from step to step,
// any number of them in a cell (REWRITE_PLAN.md 7-1), sorted by cell
// between steps. The cells are a raster of them: a cell with particles in it
// where liquid can be is drawn as liquid (raster), with the cell they are in
// and how many (common.wgsl's liquidHome). Everything else about the liquid
// (type, temperature, burning, charge) lives in the cell, where the other
// passes heat it, boil it, burn it, react with it and move it like anything
// else; next step the particles take it back, following their cell wherever
// it was moved (syncParticles). There are two particle buffers: `sorted`,
// the particles by cell between steps, and `parts`, a step's.
//
// The world is worked on in 8 x 8 blocks, and only those with liquid in or
// next to them. A block dropping out is cleared, so what isn't worked on is
// always empty. Liquid moves at most a block a step, so it stays in the
// blocks worked on. A step:
//
//   sync        each drawn cell of liquid claims the particles it was drawn
//               from, wherever the other passes moved it (common.wgsl's
//               G_DEST); marks the blocks with liquid in them, and zeroes
//               the counts compact makes
//   syncParticles
//               each particle into `parts`: kept apart from the others of
//               its cell (apart), moved with its cell, taking its
//               temperature and word; none for liquid that became something
//               else (but those past a cell's worth, which stay, loose)
//   syncSeed    new liquid (painted, melted, condensed, copied) seeded
//   compact     the blocks to work on (with liquid in or next to them) and
//               those dropping out
//   retire      clears the blocks dropping out
//   clear       clears the counters of the rest
//   gather      adds the air's push, and splats each particle's velocity
//               onto the faces around it
//   sealJoin,   the gaps the liquid has left labelled, and those meeting
//   sealAnchor  the air marked open (the rest are sealed: classify)
//   classify    fluid (holds particles), air, or obstacle (solid, powder,
//               bubble, a body's cell: bodies)
//   surface     with cells small enough for it to show (sim.js's
//               surfaceTension), surface tension's pressure on the open
//               space against the liquid
//   bodyStart,  the bodies' sums emptied; the liquid's pressure carried to
//   bodyLoads   each body's surface as it is posed, and seeping in where it
//               rests on or against something (seepLoads)
//   setFaces    face velocities from the splat; an obstacle's own velocity
//               on its sides; a body's on its cells' sides; gravity; on
//               viscous liquid's, less last step's pressure push
//   viscosity   implicit diffusion of face velocities, red-black sweeps
//   unpress     the pressure push back on
//   bodyDrag,   the sweeps' drag on each body summed, and its faces
//   bodyFaces   moving with it again
//   divergence  net outflow of each fluid cell; and how crowded it is
//   cgStart,    the pressure that makes the flow divergence free, starting
//   cgPrepare   from last step's: conjugate gradients until what is left is
//   relax,      under LIQUID_TOLERANCE, preconditioned by red-black
//   multigrid,  Gauss-Seidel around a multigrid V-cycle; alongside it, the
//   cgRho ...   V-cycle relaxes a drift that spreads crowded particles out
//   cgUpdate,   into empty cells; each iteration the pressure's push on the
//   cgFinish    bodies (cgBodyP), which give way to it in the same solve
//   project     pressure differences accelerate the flow across each face,
//               and a body's faces move as it does
//   rigidify,   what the liquid did to each free body, for bodies.wgsl
//   bodyPush
//   extrapolate the projected velocity out into the air beside the liquid,
//               two faces deep, for the particles at the surface to sample
//   advect      particles take the grid's velocity and its gradient (APIC)
//               and move along the grid's flow plus the drift, stopping at
//               obstacles
//   sortCount,  the particles sorted by cell into `sorted`: counted per
//   scan...,    cell, the counts summed up in order within each block and
//   scatter     across the blocks, each particle put in its cell's run
//   rasterClaims  each cell of gas liquid comes into claims where its gas
//               is to go (airTo)
//   raster      each cell with particles where liquid can be drawn as liquid,
//               pushing any gas out of the way; elsewhere they stay as they
//               are
//   unpaint     cells no particle is in any more become air
//
// Units are cells and steps, and densities are relative to water's: water
// at depth d is under pressure GRAVITY * d.

struct Particle {
    pos: vec2<f32>,  // in cells
    vel: vec2<f32>,  // cells per step
    word: u32,       // its cell's x word, less FL_DRAWN
    temp: f32,
    // Its own, from where and when it was seeded (seedId), kept: whose word
    // a drawn cell takes (raster), and which of a cell's particles stay when
    // its count is lowered (rankById), so that neither hangs on the order
    // threads got to them (the determinism of REWRITE_PLAN.md D-1).
    id: u32,
    // The velocity's gradient around it (APIC's C): du/dx, du/dy, dv/dx,
    // dv/dy, per cell.
    affine: vec4<f32>,
}

// Per particle, its motion (position, velocity as f32 bits) in the first
// third of items, its state (word, temperature bits, and its place in its
// cell's run as sortCount counted it) in the second and its affine velocity
// (Particle's) in the third, so that neighboring threads read and write each
// as one contiguous run: a struct per particle is written field by field, a
// particle's stride apart, which is many times slower. The first `count` are
// the particles; should more be put in than there is room for, those past
// the room are lost, and the count says how many tried.
struct Particles {
    count: atomic<u32>,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    items: array<vec4<u32>>,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> elems: array<Elem>;
@group(0) @binding(2) var<storage, read_write> cells: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> air: array<vec4<f32>>;
// The particles sorted by cell, a run of them a cell (G_START, G_SORT), as
// the last step left them; and the step's, which syncParticles puts there.
@group(0) @binding(4) var<storage, read_write> sorted: Particles;
@group(0) @binding(5) var<storage, read_write> parts: Particles;
// Per cell LIQUID_CELL_WORDS (G_*): its particles as gathered, which cell's
// gas the liquid moves out of its way it was claimed for, its particles
// after advect and where their run starts in `sorted`, where they go next
// step (common.wgsl's G_DEST), the liquid it was drawn as, its smoothed
// share and densities, where its own such gas goes, and its pressure and
// drift for the relaxation's rims; then per face the splatted
// momentum and weight (WGSL has no float atomics, so the sums are fixed
// point).
@group(0) @binding(6) var<storage, read_write> counters: array<atomic<u32>>;
// Two per cell: pressure, divergence to remove, density (> 0 fluid, 0 air,
// < 0 obstacle), viscosity (also read by update.wgsl); then the drift's
// potential and the spreading out it is to make, which move particles but
// don't go into their velocity: crowding corrected through the velocity
// would keep kicking the liquid, and still water would never be still;
// whether the cell is sealed off (1, see classify), and whether it was
// liquid in each of the last two steps (bits, see stepBit). After those, per
// face: velocity, velocity before this step's forces (for FLIP), whether it
// holds one (1 from the splat, PROJECTED once projected, the extrapolated
// layers between; 0 in open air with no particle near, -1 inside an
// obstacle), and drift (until the projection sets it, the face's viscosity,
// for viscosity's sweeps: faceViscosity). Then velocity before viscosity, four faces to an
// element; then what viscosity's sweeps did to it last step, likewise.
@group(0) @binding(7) var<storage, read_write> field: array<vec4<f32>>;
// Per block: marks (MARK_LIQUID, MARK_REGION, from sync), state (ACTIVE,
// HAS_FLUID, HAS_VISCOUS, NEAR_LIQUID); then the list of blocks worked on, and of those dropping
// out; then how many of each there are (COUNT_*). The passes
// over them are dispatched a workgroup per block in the world, and those
// past the count return at once: dispatching only as many as there are,
// sized from this buffer, is many times slower in Dawn, which checks each
// such dispatch's size in a pass of its own.
@group(0) @binding(8) var<storage, read_write> blocks: array<atomic<u32>>;

// The multigrid level a pass works on (see Multigrid below), and which of
// the step's iterations (see Conjugate gradients).
struct Level {
    index: u32,
    cycle: u32,
    pad1: u32,
    pad2: u32,
}
@group(0) @binding(9) var<uniform> level: Level;
// The pull of the world's masses per light cell of 2 x 2 cells, cells/step²
// (gravity.wgsl).
@group(0) @binding(10) var gravity: texture_2d<f32>;
// What goes into the air (airsource.wgsl): gas the liquid moves into and has
// nowhere else to put (moveOutOfTheWay).
@group(0) @binding(11) var<storage, read_write> source: array<atomic<u32>>;
// Free bodies' table (bodies.wgsl's motion buffer): their pose and motion as
// last stamped, read for their cells in the solve; and after the contacts,
// what the liquid did to each (liquidSlot: bodyPush, rigidify).
@group(0) @binding(12) var<storage, read_write> motion: array<atomic<u32>>;
// Where people's bodies are (common.wgsl's occupantWord): obstacles moving
// with them (REWRITE_PLAN.md 10-1), so the liquid flows round a person and
// pushes them.
@group(0) @binding(13) var<storage, read> occupants: array<u32>;
// The bodies buffer (bodies.wgsl): which body's cup each cell was stamped
// in last step (common.wgsl's bodySweptSlot).
@group(0) @binding(14) var<storage, read> bodyWords: array<u32>;

const NONE: u32 = 0xffffffffu;
const G_COUNT: u32 = 0u;  // particles in it as gathered (for the solve)
const G_TAKEN: u32 = 1u;  // claimed for gas the liquid moves out of its way (rasterClaims), the lowest cell's ~index
const G_SORT: u32 = 2u;   // particles in it after advect: its run's length
const G_START: u32 = 3u;  // where its run starts in `sorted`
const G_TYPE: u32 = 5u;   // the liquid it was drawn as, its solute and its life (drawnAs; G_DEST, 4, is common.wgsl's)
const G_DENSITY: u32 = 6u;  // the particles' smoothed share of it, fixed point (gather; densityAt)
const G_MASS: u32 = 7u;     // the particles' densities summed, fixed point (gather; liquidIn)
const G_AIR_TO: u32 = 8u;   // where the gas the liquid comes into goes (rasterClaims), NONE for nowhere
const G_RIM: u32 = 9u;      // its pressure and drift as they were when the relaxation began (two words; relaxRims)
const FIX: f32 = 65536.0;

const BLOCK: i32 = FLUID_BLOCK;
const MARK_LIQUID: u32 = 1u;
const MARK_REGION: u32 = 2u;   // a cell of a wet held body (wetRegionAt)
const ACTIVE: u32 = 1u;
const HAS_FLUID: u32 = 2u;
const HAS_VISCOUS: u32 = 4u;
const NEAR_LIQUID: u32 = 8u;   // liquid in it or a block next to it (compact): a held body with a cell here is wet (rigidify)
// After the block lists, the words COUNT_* (js/shared.js's FLUID_COUNT): how
// many blocks are worked on and dropping out, whether any holds viscous
// liquid (not a count), the particles as last sorted, the most the
// pressure's iterations left (three slots: converged), the iterations this
// step and the most a step since the last report (sim.js reads them
// back).

// Over-relaxation: of the sweeps smoothing the pressure between multigrid
// corrections (where plain Gauss-Seidel is the better smoother), of those
// solving the coarsest level outright, and of viscosity's.
const SMOOTH_OMEGA: f32 = 1.0;
const OMEGA: f32 = 1.8;
const VISC_OMEGA: f32 = 1.6;
const SPREAD: f32 = 0.3;     // cells of drift out of a cell per cell's worth of particles over one (into it, under)
const MAX_DRIFT: f32 = 0.5;  // cells per step
// Share of last step's drift potential this step's solve starts from. In
// liquid with no air to it (a sealed box, a vacuum in it counted as liquid
// short of particles) the drift's equations have no solution: fewer
// particles than cells can't each have one. The solve then keeps raising or
// lowering the potential everywhere alike, which moves nothing, but
// without end (by 4.6 a step for a vacuum of 16 cells) until f32 can no
// longer tell the differences that do. Carried over at a little less, it
// levels off at ten steps' worth.
const DRIFT_KEEP: f32 = 0.9;
// A face's velocity as the projection left it (field's z; see extrapolate).
const PROJECTED: f32 = 2.0;
// Particles take the grid's velocity and its gradient (APIC: Jiang,
// Schroeder, Selle, Teran & Stomakhin 2015; advectSlot), the gradient kept
// at this share a step: all of it, the noise the projection leaves stirred
// still water after a few hundred steps, as FLIP's did (REWRITE_PLAN.md
// P2-2).
const AFFINE_KEEP: f32 = 0.95;
// Cells per step. Liquid moves through the cells in steps of half a cell
// (advect), not one cell per block pass as powders and gases do, so it isn't
// held to their MAX_SPEED: a jet or a fall gets up to its real speed.
const LIQUID_MAX_SPEED: f32 = 8.0;
// The half-cell steps advect takes a particle along in, at most: its
// speed clamped to LIQUID_MAX_SPEED and its drift to MAX_DRIFT.
const ADVECT_STEPS: i32 = 17;
const_assert f32(ADVECT_STEPS) >= 2.0 * (LIQUID_MAX_SPEED + MAX_DRIFT);
const IMPACT_SPEED: f32 = 2.5;  // landing speed that sets off impact-sensitive liquids

// ------------------------------------------------------------ addressing

fn cellCount() -> u32 {
    return P.size.x * P.size.y;
}

fn faceCount() -> u32 {
    return (P.size.x + 1u) * P.size.y + P.size.x * (P.size.y + 1u);
}

fn inWorld(c: vec2<i32>) -> bool {
    return all(c >= vec2<i32>(0)) && all(c < vec2<i32>(P.size));
}

fn cellIndex(c: vec2<i32>) -> u32 {
    return u32(c.y) * P.size.x + u32(c.x);
}

fn cellOf(pos: vec2<f32>) -> vec2<i32> {
    return clamp(vec2<i32>(floor(pos)), vec2<i32>(0), vec2<i32>(P.size) - 1);
}

// The cell at c and its four sides: left, right, up, down for d = 0..3.
fn beside(c: vec2<i32>, d: i32) -> vec2<i32> {
    return c + select(vec2<i32>(0, (d & 1) * 2 - 1), vec2<i32>((d & 1) * 2 - 1, 0), d < 2);
}

// u faces sit on the left side of cell (x, y), x up to W; v faces on the top
// side, y up to H.
fn uFace(x: i32, y: i32) -> u32 {
    return u32(y) * (P.size.x + 1u) + u32(x);
}

fn vFace(x: i32, y: i32) -> u32 {
    return (P.size.x + 1u) * P.size.y + u32(y) * P.size.x + u32(x);
}

fn faceInWorld(at: vec2<i32>, isU: bool) -> bool {
    let end = vec2<i32>(P.size) + select(vec2<i32>(0, 1), vec2<i32>(1, 0), isU);
    return all(at >= vec2<i32>(0)) && all(at < end);
}

fn faceIndex(at: vec2<i32>, isU: bool) -> u32 {
    return select(vFace(at.x, at.y), uFace(at.x, at.y), isU);
}

// The faces cell c sees to: its left and top ones, and its right (bottom)
// one too at the edge of the blocks worked on: the world's right (bottom)
// edge, or beside a block not worked on, whose cells are air to the solve.
// (Only a body's cells reach that far from the liquid: the faces they meet
// that air on are set and projected as the rest.) As (x, y, isU), isU < 0
// for none.
fn facesOf(c: vec2<i32>) -> array<vec3<i32>, 4> {
    let right = c.x == i32(P.size.x) - 1 || (c.x % BLOCK == BLOCK - 1 && !isActiveBlock(c / BLOCK + vec2<i32>(1, 0)));
    let bottom = c.y == i32(P.size.y) - 1 || (c.y % BLOCK == BLOCK - 1 && !isActiveBlock(c / BLOCK + vec2<i32>(0, 1)));
    return array<vec3<i32>, 4>(
        vec3<i32>(c, 1), vec3<i32>(c, 0),
        vec3<i32>(c.x + 1, c.y, select(-1, 1, right)), vec3<i32>(c.x, c.y + 1, select(-1, 0, bottom)),
    );
}

// ------------------------------------------------------------ counters

fn gridSlot(i: u32) -> u32 {
    return LIQUID_CELL_WORDS * i;
}

fn sumSlot(f: u32) -> u32 {
    return LIQUID_CELL_WORDS * cellCount() + 2u * f;
}

fn gridWord(i: u32, g: u32) -> u32 {
    return atomicLoad(&counters[gridSlot(i) + g]);
}

// The particles' smoothed share of cell c (gather's bilinear weights over a
// cell's worth, LIQUID_PER_CELL): 1 deep in still liquid, about a half at
// its surface, 0 well clear of it.
fn densityAt(c: vec2<i32>) -> f32 {
    if (!inWorld(c)) {
        return 0.0;
    }
    return f32(gridWord(cellIndex(c), G_DENSITY)) / (FIX * f32(LIQUID_PER_CELL));
}

// ------------------------------------------------------------ the field

// Outside the world is an obstacle.
fn gridAt(c: vec2<i32>) -> vec4<f32> {
    if (!inWorld(c)) {
        return vec4<f32>(0.0, 0.0, -1.0, 0.0);
    }
    return field[2u * cellIndex(c)];
}

fn driftAt(c: vec2<i32>) -> f32 {
    if (!inWorld(c)) {
        return 0.0;
    }
    return field[2u * cellIndex(c) + 1u].x;
}

fn faceSlot(f: u32) -> u32 {
    return 2u * cellCount() + f;
}

fn viscRhs(f: u32) -> f32 {
    return field[faceSlot(faceCount() + f / 4u)][f % 4u];
}

fn setViscRhs(f: u32, u: f32) {
    field[faceSlot(faceCount() + f / 4u)][f % 4u] = u;
}

fn viscLast(f: u32) -> f32 {
    return field[faceSlot(faceCount() + (faceCount() + 3u) / 4u + f / 4u)][f % 4u];
}

fn setViscLast(f: u32, u: f32) {
    field[faceSlot(faceCount() + (faceCount() + 3u) / 4u + f / 4u)][f % 4u] = u;
}

// Across a face between cells with densities a and b, how much a pressure
// difference accelerates the liquid: 1 / its density, the mean between two
// liquids; 0 against an obstacle or between two cells of air.
fn faceBeta(a: f32, b: f32) -> f32 {
    if (a < 0.0 || b < 0.0 || (a <= 0.0 && b <= 0.0)) {
        return 0.0;
    }
    return select(1.0 / max(a, b), 2.0 / (a + b), a > 0.0 && b > 0.0);
}

// The liquid's surface between a cell of it and one of air (the ghost fluid
// method: Gibou, Fedkiw, Cheng & Kang 2002): where the particles' smoothed
// share of the cells (densityAt) crosses a half, as a share of the way
// from the liquid cell's middle (da) to the air cell's (db). The pressure
// is the air's there, not at the air cell's middle, so it moves smoothly
// as the particles do, rather than a cell at a time as a cell fills.
const THETA_MIN: f32 = 0.1;

fn surfaceShare(da: f32, db: f32) -> f32 {
    return clamp((da - 0.5) / max(da - db, 1e-3), THETA_MIN, 1.0);
}

// faceBeta on the finest level, between cells of densities a and b, their
// shares da and db (shareAt), the second toward n from the first: a
// liquid-air face's over its surfaceShare. A cell in the solve as liquid
// holding no particles is full to its sides, which is where its surface
// is, half way to the air cell's middle, as for a full cell of liquid
// beside an empty one. (With the air's pressure at the air cell's middle
// there, the liquid's half a cell lower beside it, a raft rode half a row
// low.)
fn fineBeta(a: f32, b: f32, da: f32, db: f32, n: vec2<i32>) -> f32 {
    let beta = faceBeta(a, b);
    if (a > 0.0 && b == 0.0) {
        return beta / select(0.5, surfaceShare(da, db), da > 0.0);
    }
    if (b > 0.0 && a == 0.0) {
        return beta / select(0.5, surfaceShare(db, da), db > 0.0);
    }
    return beta;
}

// The share of the way from the middle of a cell of body b to that of the
// cell beside it toward n that the body fills, where its pose has it (its
// stamp is whole cells, its pose not: bodyLoads, bodyDrag): half,
// and more or less by how far its cells' middles are from their stamped
// cells' that way (bodies.wgsl's T_SUBCELL; the pivot's, which all of them
// share but for how a turned stamp's shears round).
fn bodyReach(b: u32, n: vec2<i32>) -> f32 {
    return clamp(0.5 + dot(unpack2x16float(bodyWord(b, BW_SUBCELL)), vec2<f32>(n)), 0.0, 1.0);
}

// What fineBeta takes of cell c: the particles' share of it (densityAt).
fn shareAt(c: vec2<i32>) -> f32 {
    return densityAt(c);
}

// Whether a particle at pos is in an obstacle's cell: as classified, or, in
// a block not worked on, where nothing is, by what the cell holds.
fn isObstacle(pos: vec2<f32>) -> bool {
    let c = vec2<i32>(floor(pos));
    if (gridAt(c).z < 0.0 || isOccupant(occupants[cellIndex(c)])) {
        return true;
    }
    let k = elems[wordType(cells[cellIndex(c)].x)].kind;
    return k == KIND_SOLID || k == KIND_POWDER;
}

fn isDrawn(v: vec4<u32>) -> bool {
    return cellHas(v, FL_DRAWN);
}

fn clampSpeed(v: vec2<f32>) -> vec2<f32> {
    let s = length(v);
    return select(v, v * (LIQUID_MAX_SPEED / s), s > LIQUID_MAX_SPEED);
}

// ------------------------------------------------------------ blocks

fn blocksWide() -> u32 {
    return (P.size.x + u32(BLOCK) - 1u) / u32(BLOCK);
}

fn blockCount() -> u32 {
    return blocksWide() * ((P.size.y + u32(BLOCK) - 1u) / u32(BLOCK));
}

fn inBlocks(b: vec2<i32>) -> bool {
    return all(b >= vec2<i32>(0)) && b.x < i32(blocksWide()) && b.y * i32(blocksWide()) < i32(blockCount());
}

fn blockIndex(b: vec2<i32>) -> u32 {
    return u32(b.y) * blocksWide() + u32(b.x);
}

fn stateSlot(n: u32) -> u32 {
    return blockCount() + n;
}

fn activeSlot(n: u32) -> u32 {
    return 2u * blockCount() + n;
}

fn retiredSlot(n: u32) -> u32 {
    return 3u * blockCount() + n;
}

fn countSlot(which: u32) -> u32 {
    return 4u * blockCount() + which;
}

// The n-th block worked on's particles after advect, then where its runs
// start in `sorted` (scanBlocks, scanTotals).
fn totalSlot(n: u32) -> u32 {
    return 4u * blockCount() + FLUID_STATS + n;
}

// Blocks worked on this step.
fn activeCount() -> u32 {
    return atomicLoad(&blocks[countSlot(COUNT_ACTIVE)]);
}

fn isActiveBlock(b: vec2<i32>) -> bool {
    return inBlocks(b) && (atomicLoad(&blocks[stateSlot(blockIndex(b))]) & ACTIVE) != 0u;
}

// The state bits of any of the blocks from b0 to b1.
fn statesOf(b0: vec2<i32>, b1: vec2<i32>) -> u32 {
    var bits = 0u;
    for (var y = b0.y; y <= b1.y; y++) {
        for (var x = b0.x; x <= b1.x; x++) {
            if (inBlocks(vec2<i32>(x, y))) {
                bits |= atomicLoad(&blocks[stateSlot(blockIndex(vec2<i32>(x, y)))]);
            }
        }
    }
    return bits;
}

// The block of a workgroup in a pass over the blocks worked on.
fn activeBlock(wid: u32) -> vec2<i32> {
    let n = atomicLoad(&blocks[activeSlot(wid)]);
    return vec2<i32>(i32(n % blocksWide()), i32(n / blocksWide()));
}

// The cell of a thread in a pass over the blocks worked on.
fn activeCell(wid: u32, lid: vec2<u32>) -> vec2<i32> {
    return activeBlock(wid) * BLOCK + vec2<i32>(lid);
}

// ------------------------------------------------------------ particles

fn capacity() -> u32 {
    return arrayLength(&parts.items) / 3u;
}

fn toParticle(motion: vec4<u32>, state: vec4<u32>, affine: vec4<u32>) -> Particle {
    let m = bitcast<vec4<f32>>(motion);
    return Particle(m.xy, m.zw, state.x, bitcast<f32>(state.y), state.z, bitcast<vec4<f32>>(affine));
}

fn loadParticle(k: u32) -> Particle {
    return toParticle(parts.items[k], parts.items[capacity() + k], parts.items[2u * capacity() + k]);
}

fn sortedParticle(k: u32) -> Particle {
    return toParticle(sorted.items[k], sorted.items[capacity() + k], sorted.items[2u * capacity() + k]);
}

fn storeParticle(k: u32, q: Particle) {
    parts.items[k] = bitcast<vec4<u32>>(vec4<f32>(q.pos, q.vel));
    parts.items[capacity() + k] = vec4<u32>(q.word, bitcast<u32>(q.temp), q.id, 0u);
    parts.items[2u * capacity() + k] = bitcast<vec4<u32>>(q.affine);
}

// The particles of a pass over them (a thread each, PARTICLE_GROUP to a
// workgroup, the workgroups in rows as wide as were dispatched): the
// thread's, or NONE past the step's.
const PARTICLE_GROUP: u32 = 256u;

fn threadIndex(gid: vec3<u32>, groups: vec3<u32>) -> u32 {
    return gid.y * groups.x * PARTICLE_GROUP + gid.x;
}

fn particleSlot(gid: vec3<u32>, groups: vec3<u32>) -> u32 {
    let k = threadIndex(gid, groups);
    return select(NONE, k, k < min(atomicLoad(&parts.count), capacity()));
}

// The cell of a thread in a pass over the blocks worked on, or one outside
// the world for a workgroup past them.
fn workCell(wid: u32, lid: vec2<u32>) -> vec2<i32> {
    if (wid >= activeCount()) {
        return vec2<i32>(-1);
    }
    return activeCell(wid, lid);
}

// ------------------------------------------------------------------ sync

var<workgroup> liquidHere: atomic<u32>;
var<workgroup> regionHere: atomic<u32>;

// The claim a drawn cell i makes on the particles of the cell it was drawn
// from (G_DEST): the lowest cell claiming them wins (atomicMax), and a body
// pushing them aside (DEST_ASIDE) outranks any.
fn claimOf(i: u32) -> u32 {
    return DEST_CELL | (DEST_SPAN - i);
}

fn claimedCell(d: u32) -> u32 {
    return DEST_SPAN - (d & DEST_SPAN);
}

fn isClaim(d: u32) -> bool {
    return d != DEST_KEEP && (d & DEST_ASIDE) == 0u && (d & DEST_CELL) != 0u;
}

// Whether cell i (holding v) is a cell of a held body that was wet in the
// last solve (a cell of it in a block by the liquid: rigidify, bodies.wgsl's
// B_WET): all its cells are in the solve, a block of them worked on as one
// with liquid is (MARK_REGION), for its push (heldPush), the liquid's
// pressure seeping in under it where it rests (seepAt) far from the liquid
// beside it. (Without, a block 20 wide resting on a pool's floor was lifted
// a third of Archimedes'.) A free body's cells and its cup were, for the
// rigid fluid, which had them in the solve (P3-11); a wall since P3-22, a
// free body needs it only where it meets the liquid.
fn wetRegionAt(i: u32, v: vec4<u32>) -> bool {
    let b = rigidBodyOf(v);
    return b != NONE && (bodyWord(b, BW_FLAGS) & (BF_WET | BF_HELD)) == (BF_WET | BF_HELD);
}

fn markBlockOf(pos: vec2<f32>) {
    let b = cellOf(pos) / BLOCK;
    if (inBlocks(b)) {
        atomicOr(&blocks[blockIndex(b)], MARK_LIQUID);
    }
}

// Over the whole world, in blocks: the blocks with liquid marked, the
// counts compact makes zeroed, and, with `claims`, each drawn cell's claim
// made (claimOf). syncParticles and syncSeed then fill this step's particles
// afresh.
fn syncBlock(c: vec2<i32>, wid: vec2<u32>, li: u32, claims: bool) {
    if (inWorld(c)) {
        let i = cellIndex(c);
        if (i == 0u) {
            atomicStore(&parts.count, 0u);
            atomicStore(&blocks[countSlot(COUNT_ACTIVE)], 0u);
            atomicStore(&blocks[countSlot(COUNT_RETIRED)], 0u);
            atomicStore(&blocks[countSlot(COUNT_VISCOUS)], 0u);
        }
        let v = cells[i];
        if (elems[cellType(v)].kind == KIND_LIQUID) {
            let home = liquidHome(v.w);
            if (claims && isDrawn(v) && home < cellCount()) {
                atomicMax(&counters[destSlot(home)], claimOf(i));
            }
            atomicStore(&liquidHere, 1u);
        } else if (wetRegionAt(i, v)) {
            atomicStore(&regionHere, 1u);
        }
    }
    workgroupBarrier();
    if (li == 0u && atomicLoad(&liquidHere) != 0u) {
        atomicOr(&blocks[blockIndex(vec2<i32>(wid))], MARK_LIQUID);
    }
    if (li == 0u && atomicLoad(&regionHere) != 0u) {
        atomicOr(&blocks[blockIndex(vec2<i32>(wid))], MARK_REGION);
    }
}

@compute @workgroup_size(8, 8)
fn sync(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    syncBlock(vec2<i32>(gid.xy), wid.xy, li, true);
}

// The same without the claims, for a step the liquid's passes are left out
// of (sim.js encodeFluid): the blocks marked and counted for the readback,
// which sees liquid painted since, the particles and their cells left as
// they are. (A claim made then would be made again the next step, by the
// cell from wherever it was moved to by then, and the two together pick
// either.)
@compute @workgroup_size(8, 8)
fn syncMarks(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    syncBlock(vec2<i32>(gid.xy), wid.xy, li, false);
}

// The place by id of the particle at slot k of `sorted` (id `id`, word
// `word`) among the particles of its cell h as last sorted (ties by slot):
// a cell whose count a pass lowered keeps those of the lowest ids, whatever
// order the sort put them in (REWRITE_PLAN.md D-1); and those of another
// liquid than the cell was drawn as (`drawn`) before any of its own, as
// what a pass takes of a cell is of what it was drawn as (water evaporating,
// soaking, drunk: a mixed cell lost its oil as often as its water).
fn rankById(h: u32, k: u32, id: u32, word: u32, drawn: u32) -> u32 {
    let start = gridWord(h, G_START);
    let mine = select(0u, 1u, wordType(word) == drawn);
    var r = 0u;
    for (var m = start; m < start + gridWord(h, G_SORT); m++) {
        let other = sorted.items[capacity() + m];
        let theirs = select(0u, 1u, wordType(other.x) == drawn);
        r += select(0u, 1u, theirs < mine || (theirs == mine && (other.z < id || (other.z == id && m < k))));
    }
    return r;
}

// Particles on one point move as one for good: every pass moves a particle
// by what is at its position, and the grid (the drift's too) sees a cell's
// particles only as a cell's worth, so two on one point are never parted.
// Flow into a wall, the drift into a cell short of its share and the
// escape's step out into the nearest cell bring particles together, and an
// f32's rounding (some 1e-5 cells, 1e-4 at a big world's far side) puts
// them on one point: in Rubble, 2980 particles (5 % of the water) in 1337
// groups of up to 93 stayed so for good, each group a cell's worth of water
// or more packed into one cell with a bubble round it (weird-water.json,
// 2026-10-28: 47 in a cell). So particles of a cell closer than APART are
// pushed apart, as a FLIP solver's particle correction keeps them (Ando &
// Tsuruno 2011; Ando, Thürey & Wojtan 2013), half the shortfall each, a
// pair on one point each its own way by their ids (two of one id, which a
// 32-bit hash gives a world now and then, aren't parted). APART is far
// under their spacing (half a cell at LIQUID_PER_CELL 4).
const APART: f32 = 1.0 / 256.0;
// The pushes summed exactly, in cells over APART_FIX: a cell's run is in the
// order its particles were counted (sortCount), which runs differ in, and a
// sum of f32s in another order rounds otherwise (REWRITE_PLAN.md D-1).
const APART_FIX: f32 = 16777216.0;

// The way a particle of id a is pushed from one of id b on its point, b's
// the opposite.
fn pairWay(a: u32, b: u32) -> vec2<f32> {
    let h = pcg(min(a, b) ^ pcg(max(a, b)));
    let angle = f32(h) * (6.2831853 / 4294967296.0);
    return vec2<f32>(cos(angle), sin(angle)) * select(-1.0, 1.0, a < b);
}

// Particle k of `sorted`, at pos, of id `id`, in cell h: pushed apart from
// the others of its cell closer than APART, along an axis only as far as
// keeps it in its cell. (A loop over its cell's run: a few particles, but
// as many as a cell holds.)
fn apart(k: u32, h: u32, pos: vec2<f32>, id: u32) -> vec2<f32> {
    let start = gridWord(h, G_START);
    let n = gridWord(h, G_SORT);
    if (n < 2u) {
        return pos;
    }
    var push = vec2<i32>(0);
    for (var m = start; m < start + n; m++) {
        if (m == k) {
            continue;
        }
        let d = pos - bitcast<vec4<f32>>(sorted.items[m]).xy;
        let l = length(d);
        if (l < APART) {
            let way = select(d / l, pairWay(id, sorted.items[capacity() + m].z), l == 0.0);
            push += vec2<i32>(round(0.5 * (APART - l) * APART_FIX * way));
        }
    }
    let c = floor(pos);
    let to = pos + vec2<f32>(push) / APART_FIX;
    return select(pos, to, (to >= c + APART) & (to <= c + 1.0 - APART));
}

// Particles kept in a workgroup of syncParticles or syncSeed, and where in
// `parts` its first goes.
var<workgroup> keptHere: atomic<u32>;
var<workgroup> keptBase: u32;

// Takes n places in `parts` for a thread of the workgroup (all of whose
// threads call it): the first of them, past the room if there is none.
fn takePlaces(n: u32, li: u32) -> u32 {
    if (li == 0u) {
        atomicStore(&keptHere, 0u);
    }
    workgroupBarrier();
    let at = atomicAdd(&keptHere, n);
    workgroupBarrier();
    if (li == 0u) {
        keptBase = atomicAdd(&parts.count, atomicLoad(&keptHere));
    }
    workgroupBarrier();
    return keptBase + at;
}

// A thread a particle of `sorted`: into `parts`, following the cell it was
// drawn in (G_DEST), wherever the other passes moved it, with its
// temperature, and its word if it is of the liquid the cell was drawn as
// (the rest of a mixed cell keep their own). A cell that became something
// else (boiled, frozen, burnt, eaten, painted over) took its particles with
// it, all it held (common.wgsl's liquidAmount); a cell whose count a pass
// lowered (some of it evaporated) loses as many. Those in a cell that
// wasn't drawn (liquid can't be there) stay where they are, and come out
// beside it (advect's escape). Particles moved aside by a body go aside.
// First, those of a cell too close to another are pushed apart (apart).
@compute @workgroup_size(256)
fn syncParticles(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let k = threadIndex(gid, groups);
    var keep = false;
    var q = Particle(vec2<f32>(0.0), vec2<f32>(0.0), AIR, 0.0, 0u, vec4<f32>(0.0));
    if (k < min(atomicLoad(&sorted.count), capacity())) {
        q = sortedParticle(k);
        let h = cellIndex(cellOf(q.pos));
        q.pos = apart(k, h, q.pos, q.id);
        let d = gridWord(h, G_DEST);
        if (d == DEST_KEEP) {
            keep = true;
            markBlockOf(q.pos);
        } else if ((d & DEST_ASIDE) != 0u) {
            // Not into another body's cup: the body that sent them chose the
            // cell clear of the others as they were last step, and one
            // stamped beside it in the same step can have taken it into its
            // hold since (a sealed hull a block pressed into the water took
            // in particles from 2 cells away, 2 runs in 6). Left where they
            // are, the escape lets them out clear of every body.
            let to = d & ~DEST_ASIDE;
            if (cupOf(to) == NONE) {
                q.pos = vec2<f32>(f32(to % P.size.x), f32(to / P.size.x)) + fract(q.pos);
            }
            keep = true;
            markBlockOf(q.pos);
        } else if (isClaim(d)) {
            let to = claimedCell(d);
            let v = cells[to];
            let n = liquidCount(v.w);
            if (n == LIQUID_COUNT_MAX || gridWord(h, G_SORT) <= n || rankById(h, k, q.id, q.word, cellType(v)) < n) {
                if (to != h) {
                    q.pos = vec2<f32>(f32(to % P.size.x), f32(to / P.size.x)) + fract(q.pos);
                }
                q.temp = cellTemp(v);
                let drawn = gridWord(h, G_TYPE);
                let pt = wordType(q.word);
                if (holdsSolute(pt) && pt == (drawn & 0xffu) && wordType(v.x) == pt
                    && (wordAux(q.word) == ((drawn >> 8u) & 0xffu) || wordAux(q.word) == 0u)) {
                    let life = sharedLife(select(wordLife(q.word), 0u, wordAux(q.word) == 0u), (drawn >> 16u) & 0xffu, wordLife(v.x), q.id);
                    let w = wordWithout(v.x, FL_DRAWN);
                    // Its solute the cell's, or with none left the cell's as
                    // drawn: a cell whose water's mean rounded to none has its
                    // solute let go by the passes that rewrite it, where a
                    // particle of it still holds some (REWRITE_PLAN.md 7-8:
                    // taking the cell's none erased it, Salt in water losing
                    // 1 % of its salt in 3000 steps at its brine's thin edge).
                    let solute = select((drawn >> 8u) & 0xffu, wordAux(w), wordAux(w) != 0u);
                    q.word = packWord(pt, select(0u, solute, life > 0u), life, wordShade(w), wordFlags(w));
                } else if (drawnAs(q.word) == (drawn & 0xffffu)) {
                    q.word = wordWithout(v.x, FL_DRAWN);
                }
                keep = true;
            }
        }
    }
    let m = takePlaces(select(0u, 1u, keep), li);
    if (keep && m < capacity()) {
        storeParticle(m, q);
    }
}

// A particle's id (Particle.id): a hash of the cell it was seeded in, the
// step and which of that cell's seeds it is.
fn seedId(c: vec2<i32>, j: u32) -> u32 {
    var h = ((u32(c.y) * P.size.x + u32(c.x)) * 0x9e3779b9u) ^ (P.frame * 0x85ebca6bu) ^ (j * 0xc2b2ae35u);
    h ^= h >> 16u;
    h *= 0x7feb352du;
    h ^= h >> 15u;
    return h;
}

// Where the j-th of n particles seeded in cell c starts: spread over the
// cell, in its quarters for up to four (diagonally for two), jittered.
fn seedPos(c: vec2<i32>, j: u32, n: u32) -> vec2<f32> {
    if (n == 1u) {
        return vec2<f32>(c) + 0.5;
    }
    let r = vec2<f32>(rnd(), rnd()) - 0.5;
    if (n <= 4u) {
        let quarter = array<vec2<f32>, 4>(vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0));
        return vec2<f32>(c) + 0.25 + 0.5 * quarter[j] + 0.3 * r;
    }
    return vec2<f32>(c) + 0.5 + 0.9 * r;
}

// New liquid seeded, over the whole world in blocks: a cell's worth
// (LIQUID_PER_CELL) for a cell of it that isn't drawn (painted, melted,
// condensed); as many as a drawn one counts for one that lost its claim to
// another cell copied from the same (a clone's copy), or that counts more
// than it was drawn from (a pass added to it). At the cell's velocity,
// temperature and word.
@compute @workgroup_size(8, 8)
fn syncSeed(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let c = vec2<i32>(gid.xy);
    var n = 0u;
    var v = vec4<u32>(0u);
    if (inWorld(c)) {
        let i = cellIndex(c);
        v = cells[i];
        if (elems[cellType(v)].kind == KIND_LIQUID) {
            let home = liquidHome(v.w);
            let want = liquidCount(v.w);
            if (!isDrawn(v)) {
                n = LIQUID_PER_CELL;
            } else if (home >= cellCount() || gridWord(home, G_DEST) != claimOf(i)) {
                n = max(want, 1u);
            } else if (want > gridWord(home, G_SORT) && want < LIQUID_COUNT_MAX) {
                n = want - gridWord(home, G_SORT);
            }
            seedRng(i, P.frame, P.seed ^ 0x5eedu);
        }
    }
    let m = takePlaces(n, li);
    for (var j = 0u; j < n; j++) {
        if (m + j < capacity()) {
            storeParticle(m + j, Particle(seedPos(c, j, n), cellVel(v), wordWithout(v.x, FL_DRAWN), cellTemp(v), seedId(c, j), vec4<f32>(0.0)));
        }
    }
}

// ------------------------------------------------------------------ plan

// A block is worked on with liquid in it or in a block next to it (and so
// near the liquid), or a wet held body's cells.
@compute @workgroup_size(64)
fn compact(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if (n >= blockCount()) {
        return;
    }
    let b = vec2<i32>(i32(n % blocksWide()), i32(n / blocksWide()));
    var marks = 0u;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let o = b + vec2<i32>(x, y);
            if (inBlocks(o)) {
                marks |= atomicLoad(&blocks[blockIndex(o)]);
            }
        }
    }
    let near = (marks & MARK_LIQUID) != 0u;
    let liquid = (marks & (MARK_LIQUID | MARK_REGION)) != 0u;
    let was = (atomicLoad(&blocks[stateSlot(n)]) & ACTIVE) != 0u;
    atomicStore(&blocks[stateSlot(n)], select(0u, ACTIVE, liquid) | select(0u, NEAR_LIQUID, near));
    if (liquid) {
        atomicStore(&blocks[activeSlot(atomicAdd(&blocks[countSlot(COUNT_ACTIVE)], 1u))], n);
    } else if (was) {
        atomicStore(&blocks[retiredSlot(atomicAdd(&blocks[countSlot(COUNT_RETIRED)], 1u))], n);
    }
}

// A block dropping out is left empty: air, nothing moving, nothing counted.
@compute @workgroup_size(8, 8)
fn retire(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    if (wid.x >= atomicLoad(&blocks[countSlot(COUNT_RETIRED)])) {
        return;
    }
    let n = atomicLoad(&blocks[retiredSlot(wid.x)]);
    let c = vec2<i32>(i32(n % blocksWide()), i32(n / blocksWide())) * BLOCK + vec2<i32>(lid.xy);
    if (!inWorld(c)) {
        return;
    }
    let i = cellIndex(c);
    field[2u * i] = vec4<f32>(0.0);
    field[2u * i + 1u] = vec4<f32>(0.0);
    clearCell(i);
    let fs = facesOf(c);
    for (var j = 0; j < 4; j++) {
        if (fs[j].z >= 0) {
            let f = faceIndex(fs[j].xy, fs[j].z == 1);
            field[faceSlot(f)] = vec4<f32>(0.0);
            setViscLast(f, 0.0);
            atomicStore(&counters[sumSlot(f)], 0u);
            atomicStore(&counters[sumSlot(f) + 1u], 0u);
        }
    }
    if (all(lid.xy == vec2<u32>(0u))) {
        atomicStore(&blocks[n], 0u);
    }
}

// A cell's counters at the start of a step (sync has read last step's).
fn clearCell(i: u32) {
    for (var g = 0u; g < LIQUID_CELL_WORDS; g++) {
        atomicStore(&counters[gridSlot(i) + g], 0u);
    }
}

// The counters of the blocks worked on start each step at zero, but for each
// cell's G_REGION link, its own index (sealJoin's labelling).
@compute @workgroup_size(8, 8)
fn clear(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    if (wid.x >= activeCount()) {
        return;
    }
    let c = activeCell(wid.x, lid.xy);
    if (all(lid.xy == vec2<u32>(0u))) {
        atomicStore(&blocks[blockIndex(activeBlock(wid.x))], 0u);
    }
    if (!inWorld(c)) {
        return;
    }
    let i = cellIndex(c);
    clearCell(i);
    atomicStore(&counters[gridSlot(i) + G_REGION], i);
    let fs = facesOf(c);
    for (var j = 0; j < 4; j++) {
        if (fs[j].z >= 0) {
            let f = faceIndex(fs[j].xy, fs[j].z == 1);
            atomicStore(&counters[sumSlot(f)], 0u);
            atomicStore(&counters[sumSlot(f) + 1u], 0u);
        }
    }
}

// ---------------------------------------------------------------- gather

fn isGasAt(c: vec2<i32>) -> bool {
    return inWorld(c) && elems[wordType(cells[cellIndex(c)].x)].kind == KIND_GAS;
}


// Adds a velocity component, bilinearly weighted, to the four faces around s
// (a position in that axis's face grid).
// A particle's velocity along one axis splatted onto the faces around it,
// s its position in that axis's face grid, its velocity there as at each
// face by its gradient (APIC: Jiang, Schroeder, Selle, Teran & Stomakhin
// 2015), so the grid keeps how it turns.
fn splatAxis(s: vec2<f32>, vel: f32, gradient: vec2<f32>, isU: bool) {
    let b = vec2<i32>(floor(s));
    let f = s - floor(s);
    for (var j = 0; j < 4; j++) {
        let o = vec2<i32>(j & 1, j >> 1);
        let w = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1);
        let at = b + o;
        if (w <= 0.0 || !faceInWorld(at, isU)) {
            continue;
        }
        let slot = sumSlot(faceIndex(at, isU));
        let there = vel + dot(gradient, vec2<f32>(at) - s);
        atomicAdd(&counters[slot], u32(i32(round(there * w * FIX))));
        atomicAdd(&counters[slot + 1u], u32(i32(round(w * FIX))));
    }
}

@compute @workgroup_size(256)
fn gather(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let k = particleSlot(gid, groups);
    if (k == NONE) {
        return;
    }
    var q = loadParticle(k);
    let e = elems[wordType(q.word)];
    // Blasts push on the surface; inside, the liquid passes it on. (The
    // wind's drag is the grid's, surfaceDrag.)
    let c = vec2<i32>(floor(q.pos));
    if (isGasAt(beside(c, 0)) || isGasAt(beside(c, 1)) || isGasAt(beside(c, 2)) || isGasAt(beside(c, 3))) {
        let a = sampleAir(q.pos);
        q.vel += pressurePush(a.grad, e.rho);
        parts.items[k] = bitcast<vec4<u32>>(vec4<f32>(q.pos, q.vel));
    }
    splatAxis(q.pos - vec2<f32>(0.0, 0.5), q.vel.x, q.affine.xy, true);
    splatAxis(q.pos - vec2<f32>(0.5, 0.0), q.vel.y, q.affine.zw, false);
    let home = gridSlot(cellIndex(cellOf(q.pos)));
    atomicAdd(&counters[home + G_COUNT], 1u);
    atomicAdd(&counters[home + G_MASS], u32(round(liquidDensity(q.word, q.temp) * FIX)));
    splatDensity(q.pos, liquidVolume(q.word, q.temp));
}

// The room a particle of liquid of word x at stored temperature temp takes,
// as a share of its own at its reference temperature: its thermal
// expansion (expandedAt; what a solute adds is mass, not room). The density
// projection counts each particle's share of a cell by it, so a liquid warmed
// takes more room and one cooled less: water 2.7 % more at 80 °C than at 20,
// oil 4.3 % (REWRITE_PLAN.md 7-16). A cell's amount of liquid stays its
// particles' count, its mass, and holds less of it hot.
fn liquidVolume(x: u32, temp: f32) -> f32 {
    let t = wordType(x);
    let acts = effTempIn(t, temp, phaseOf(t, wordSolute(x), wordStrength(x), 0.0));
    return 1.0 / expandedAt(t, acts);
}

// Whether the liquid's smoothed share stops at cell c (splatDensity): a
// solid or powder there, or the world's edge.
fn walls(c: vec2<i32>) -> bool {
    if (!inWorld(c) || isOccupant(occupants[cellIndex(c)])) {
        return true;
    }
    let k = elems[wordType(cells[cellIndex(c)].x)].kind;
    return k == KIND_SOLID || k == KIND_POWDER;
}

// A particle's share of the cells around it, by the same bilinear weights
// as its splat (G_DENSITY): what would fall on a wall falls back on the
// cell the particle's side of it, as if the wall were a mirror, so liquid
// against a wall is as dense as anywhere (cut off there, its surface sagged
// along every wall: REWRITE_PLAN.md P2).
fn splatDensity(pos: vec2<f32>, room: f32) {
    let s = pos - 0.5;
    let b = vec2<i32>(floor(s));
    let f = s - floor(s);
    for (var j = 0; j < 4; j++) {
        let o = vec2<i32>(j & 1, j >> 1);
        let w = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1);
        var at = b + o;
        if (walls(at)) {
            let x = vec2<i32>(b.x + 1 - o.x, at.y);
            let y = vec2<i32>(at.x, b.y + 1 - o.y);
            let xy = b + 1 - o;
            at = select(select(select(vec2<i32>(-1), xy, !walls(xy)), y, !walls(y)), x, !walls(x));
        }
        if (inWorld(at) && w > 0.0) {
            atomicAdd(&counters[gridSlot(cellIndex(at)) + G_DENSITY], u32(round(w * room * FIX)));
        }
    }
}

// -------------------------------------------------------------- the grid

fn isLiquidAt(c: vec2<i32>) -> bool {
    return inWorld(c) && elems[wordType(cells[cellIndex(c)].x)].kind == KIND_LIQUID;
}

// Whether the cell at c, holding v, is in the liquid's way: a solid or a
// powder. So is a bubble, gas with liquid on two sides or more: it
// rises by trading places with the liquid over it (block.wgsl), which
// flows around it, rather than collapsing into it as into open air. A drop
// falling through a cloud of steam goes through it. Air too (REWRITE_PLAN.md
// 7-9: left out from before every gas was part of the air, its pockets were
// holes at the surface's pressure the liquid filled in a few steps, and a
// boiled cell, air holding vapour, went with its heat to the air over the
// water), but only holding gas (a cell the liquid has left empty is the
// liquid's to fill) and with liquid two cells deep over it (a dip in the
// liquid's surface is open air, and air under a crest a cell thick a dent
// in it: as bubbles, the still tank's surface rippled on without end).
const BUBBLE_GAS: f32 = 0.25;  // a cell's worth of gas at 1 atm a cell of air holds to be a bubble

// Whether the liquid in the cell at q was falling free as last step's
// advect brought it there (fallingBit). A bubble stands for gas at the
// pressure of the liquid round it (a pressure a pocket, stage 7's, left),
// which it holds off as a wall would; liquid that presses on nothing,
// falling free, holds its gaps at the air's pressure, as open air. (Held as
// bubbles, the gaps a block of water tore open as it fell were walls at
// rest in it, the water over them standing on them: it fell at 26 to 30 %
// of free fall after 30 steps; REWRITE_PLAN.md 7-18.) Liquid new to the
// solve (painted, melted, condensed) presses: held as bubbles only under
// liquid a solve had found pressing, Still mud's speck of air was filled in
// at the first step.
fn fallingAt(q: vec2<i32>) -> bool {
    return inWorld(q) && (u32(field[2u * cellIndex(q) + 1u].w) & fallingBit(1u)) != 0u;
}
fn isObstacleCell(c: vec2<i32>, v: vec4<u32>) -> bool {
    if (isOccupant(occupants[cellIndex(c)])) {
        return true;
    }
    let t = cellType(v);
    let k = elems[t].kind;
    if (k == KIND_GAS) {
        var wet = 0;
        for (var d = 0; d < 4; d++) {
            wet += select(0, 1, isLiquidAt(beside(c, d)));
        }
        if (wet < 2) {
            return false;
        }
        if (t == AIR) {
            if (!isLiquidAt(c - downAt(c)) || !isLiquidAt(c - 2 * downAt(c)) || fallingAt(c - downAt(c))) {
                return false;
            }
            let i = cellIndex(c);
            return gasAmount(gasFrom(cells[gasIndex(P.size.x * P.size.y, i)], i)) >= BUBBLE_GAS;
        }
        return true;
    }
    return k == KIND_SOLID || k == KIND_POWDER;
}

// Particles in the cell at c, -1 for an obstacle (or outside the world).
fn countAt(c: vec2<i32>) -> i32 {
    if (!inWorld(c)) {
        return -1;
    }
    let i = cellIndex(c);
    if (isObstacleCell(c, cells[i])) {
        return -1;
    }
    return i32(atomicLoad(&counters[gridSlot(i) + G_COUNT]));
}

// Density and viscosity of the liquid in cell i: its particles' mean density
// (G_MASS, gather's), oil and water in one cell by their shares; else that
// of the liquid drawn there, water's for anything else. Its viscosity the
// liquid drawn there's, at the temperature it acts at (viscosityAt).
fn liquidIn(i: u32) -> vec2<f32> {
    let v = cells[i];
    let t = cellType(v);
    let e = elems[t];
    let acts = effTempIn(t, cellTemp(v), phaseOf(t, wordSolute(v.x), wordStrength(v.x), 0.0));
    let viscosity = select(0.0, viscosityAt(e, acts), e.kind == KIND_LIQUID);
    let n = gridWord(i, G_COUNT);
    if (n > 0u) {
        return vec2<f32>(f32(gridWord(i, G_MASS)) / (FIX * f32(n)), viscosity);
    }
    if (e.kind != KIND_LIQUID) {
        return vec2<f32>(1.0, 0.0);
    }
    return vec2<f32>(liquidDensity(v.x, cellTemp(v)), viscosity);
}

// ----------------------------------------------------------------- bodies
//
// A free body is a wall to the liquid moving at its velocity, and the
// liquid's pressure on it moves it in the same solve (Batty, Bertails &
// Bridson 2007, "A fast variational framework for accurate solid-fluid
// coupling", ACM TOG 26(3); Robinson-Mosher, Shinar, Gretarsson, Su &
// Fedkiw 2008; REWRITE_PLAN.md decision 14, P3-22). Its faces start out at
// its velocity there as it was stamped, the pull on them, and what the
// liquid's pressure at its surface as posed (bodyLoads) and the liquid's
// drag (bodyDrag) do to it; the solve then finds the pressure that leaves
// the liquid divergence free with the body giving way to it as it pushes:
// A = L + J^T M^-1 J, J p the push of the pressures p on its faces to the
// liquid (a force and a torque about its pose's centre: bodyFaceJ), M its
// mass and moment of inertia with what it carries. What it all does to the
// body (bodyPush) is its buoyancy, the liquid's drag and the push of a
// current on it, and the liquid it has to set moving to move at all (its
// added mass), the opposite done to the liquid, which flows out of its way
// as it sinks, rises or swings round. A body's cell is a wall in classify
// (its drift's z -1 less its slot): it holds no particles, which stop at it
// as at any solid (isObstacleCell). Its cup, a boat's hold, a crate's
// inside, is the space it is to the liquid and the air: air the air moves,
// or the water that comes in over its rim. (Until P3-22 the body was in the
// solve as liquid of its density, Carlson, Mucha & Turk 2004's rigid fluid,
// and its cup with it while dry, the body's: without it the solve pushed
// the thin floor of a boat up into the air over it, the water after it.
// With it a metal half circle that sank kept 329 of its 496 cup cells
// empty under the water for good, the water standing on them, and an open
// hold's air took no part in the air's flow, 31.5 °C in the sun beside the
// air's 21.)

fn bodyWord(b: u32, k: u32) -> u32 {
    return atomicLoad(&motion[MAX_BODIES + b * BODY_TABLE_WORDS + k]);
}

fn bodyF(b: u32, k: u32) -> f32 {
    return bitcast<f32>(bodyWord(b, k));
}

fn bodyV(b: u32, k: u32) -> vec2<f32> {
    return vec2<f32>(bodyF(b, k), bodyF(b, k + 1u));
}

// Word w of what the liquid did to body b this step (bodies.wgsl's
// liquidSlot): its change in velocity x, y, spin, in fixed point
// (common.wgsl's RIGID_FIX), and its cells in the solve.
fn liquidSlot(b: u32, w: u32) -> u32 {
    return MAX_BODIES * (1u + BODY_TABLE_WORDS) + BODY_CONTACTS * BODY_CONTACT_WORDS + b * BODY_LIQUID_WORDS + w;
}

// The body whose cell v is, for the solve, NONE if none: a structural solid
// marked loose with a slot (bodies.wgsl's bodyOf), whose body is alive and
// weighed (one found last step is still to be: it stands still here).
fn rigidBodyOf(v: vec4<u32>) -> u32 {
    let t = cellType(v);
    if (!cellHas(v, FL_LOOSE) || elems[t].kind != KIND_SOLID || elems[t].breakTo == NO_ELEM) {
        return NONE;
    }
    let id = cellFrac(v).x;
    if (id < 0.0 || id >= f32(MAX_BODIES)) {
        return NONE;
    }
    let b = u32(id);
    let flags = bodyWord(b, BW_FLAGS);
    if ((flags & BF_ALIVE) == 0u || (flags & BF_NEW) != 0u || bodyF(b, BW_MASS) <= 0.0) {
        return NONE;
    }
    return b;
}

// The body whose cup cell i was stamped in last step (bodies.wgsl marks it
// as moved over), if it is alive and weighed; NONE if none.
fn cupOf(i: u32) -> u32 {
    let key = bodyWords[bodySweptSlot(P.size, i, P.frame - 1u)];
    if ((key >> 12u) != claimStep(P.frame - 1u) || (key & 1u) != 0u) {
        return NONE;
    }
    let b = 2047u - ((key >> 1u) & 0x7ffu);
    if (b >= MAX_BODIES) {
        return NONE;
    }
    let flags = bodyWord(b, BW_FLAGS);
    if ((flags & BF_ALIVE) == 0u || (flags & BF_NEW) != 0u || bodyF(b, BW_MASS) <= 0.0) {
        return NONE;
    }
    return b;
}

// The free body whose seam cell i is, as stamped last step (bodies.wgsl's
// stampBody, SWEPT_SEAM: a gap the stamp's shears open between two cells of
// its wall next to each other in its shape), if it is alive and weighed;
// NONE if none. A wall of its to the liquid and the air (classify).
fn seamOf(i: u32) -> u32 {
    let key = bodyWords[bodySweptSlot(P.size, i, P.frame - 1u)];
    if ((key >> 12u) != claimStep(P.frame - 1u) || (key & 1u) == 0u) {
        return NONE;
    }
    let b = 2047u - ((key >> 1u) & 0x7ffu);
    if (b >= MAX_BODIES) {
        return NONE;
    }
    let flags = bodyWord(b, BW_FLAGS);
    if ((flags & BF_ALIVE) == 0u || (flags & BF_NEW) != 0u || bodyF(b, BW_MASS) <= 0.0) {
        return NONE;
    }
    return b;
}

// Body b's mass and moment of inertia, with what it carries (bodies.wgsl's
// massOf).
fn rigidMass(b: u32) -> f32 {
    return bodyF(b, BW_MASS) + bodyF(b, BW_CARGO);
}

fn rigidInertia(b: u32) -> f32 {
    return bodyF(b, BW_INERTIA) + bodyF(b, BW_CARGO_INERTIA);
}

// Body b's density, cells of water a cell: its mass, with what it carries,
// over its cells (the viscosity's sweeps: viscousFace).
fn bodyDensity(b: u32) -> f32 {
    return rigidMass(b) / max(f32(bodyWord(b, BW_CELLS)), 1.0);
}

// What the solve does to each body, a step's scratch (`pushes`): sets of a
// force x, y and a torque, each an exact sum of two words (common.wgsl's
// exactParts, EXACT_COARSE), so they add up the same whichever cell adds
// first: the pressure's push in an iteration of the solve, on its direction
// p (PUSH_P0 and the next, by the iteration's parity), on the pressure it
// starts from (PUSH_X) and the one it ends with (PUSH_F); the pressure at
// its surface as posed, past what the cells' give (PUSH_SURFACE); and the
// liquid's drag (PUSH_DRAG).
const PUSH_P0: u32 = 0u;
const PUSH_X: u32 = 2u;
const PUSH_F: u32 = 3u;
const PUSH_SURFACE: u32 = 4u;
const PUSH_DRAG: u32 = 5u;
const_assert BODY_PUSH_SETS == 6u;
const_assert BODY_PUSH_WORDS == 6u * BODY_PUSH_SETS;

fn pushSlot(b: u32, which: u32, k: u32) -> u32 {
    return b * BODY_PUSH_WORDS + 6u * which + 2u * k;
}

fn addPush(b: u32, which: u32, v: vec3<f32>) {
    for (var k = 0u; k < 3u; k++) {
        let w = exactParts(v[k], EXACT_COARSE);
        let at = pushSlot(b, which, k);
        let old = atomicAdd(&pushes[at], w.x);
        atomicAdd(&pushes[at + 1u], w.y + exactCarry(old, w.x));
    }
}

fn pushOf(b: u32, which: u32) -> vec3<f32> {
    var v = vec3<f32>(0.0);
    for (var k = 0u; k < 3u; k++) {
        let at = pushSlot(b, which, k);
        v[k] = exactValue(vec2<u32>(atomicLoad(&pushes[at]), atomicLoad(&pushes[at + 1u])), EXACT_COARSE);
    }
    return v;
}

// How body b gives way to a push g (a force x, y and a torque) as the
// solve takes it: as its contacts let it (bodies.wgsl's workMobility,
// worked out by the bodies' pass last step; REWRITE_PLAN.md P3-23), its
// mass and moment of inertia alone where that isn't last step's. (The
// solve runs before the contacts, and a body free to it whatever held it
// was taken to move as the contacts then stopped it: the water flowed into
// where a piece resting on another would have fallen to.)
fn giveWay(b: u32, g: vec3<f32>) -> vec3<f32> {
    if (bodyWord(b, BW_MOBILITY_AT) != P.frame - 1u) {
        return vec3<f32>(g.xy / rigidMass(b), g.z / rigidInertia(b));
    }
    let m = array<f32, 6>(bodyF(b, BW_MOBILITY), bodyF(b, BW_MOBILITY + 1u), bodyF(b, BW_MOBILITY + 2u),
        bodyF(b, BW_MOBILITY + 3u), bodyF(b, BW_MOBILITY + 4u), bodyF(b, BW_MOBILITY + 5u));
    return vec3<f32>(m[0] * g.x + m[1] * g.y + m[2] * g.z, m[1] * g.x + m[3] * g.y + m[4] * g.z, m[2] * g.x + m[4] * g.y + m[5] * g.z);
}

// What a set's push does to body b in the solve: its change in velocity and
// spin, as its contacts let it (giveWay).
fn pushChange(b: u32, which: u32) -> vec3<f32> {
    return giveWay(b, pushOf(b, which));
}

// And what it does to the body, whatever holds it: the contacts, after,
// take what they hold (bodyPush). (Held to what the contacts let it, a block
// on the bottom pressed up harder than its weight couldn't be lifted off.)
fn pushChangeFree(b: u32, which: u32) -> vec3<f32> {
    let g = pushOf(b, which);
    return vec3<f32>(g.xy / rigidMass(b), g.z / rigidInertia(b));
}

// The face from liquid cell c to body b's cell q: its normal into the body
// and its arm about the body's centre, (n, r x n), which J is made of.
fn bodyFaceJ(c: vec2<i32>, q: vec2<i32>, b: u32) -> vec3<f32> {
    let n = vec2<f32>(q - c);
    let r = vec2<f32>(c) + 0.5 + 0.5 * n - bodyV(b, BW_POS);
    return vec3<f32>(n, r.x * n.y - r.y * n.x);
}

// Liquid cell c's pressure p on the free bodies beside it, into a set (J p).
fn addPressurePush(c: vec2<i32>, p: f32, which: u32) {
    for (var d = 0; d < 4; d++) {
        let q = beside(c, d);
        let b = rigidAt(q);
        if (b != NONE) {
            addPush(b, which, bodyFaceJ(c, q, b) * p);
        }
    }
}

// (J^T M^-1 J p) at liquid cell c, p's push being a set's: how much more
// it empties through its faces to free bodies as they give way to it.
fn couplingAt(c: vec2<i32>, which: u32) -> f32 {
    var sum = 0.0;
    for (var d = 0; d < 4; d++) {
        let q = beside(c, d);
        let b = rigidAt(q);
        if (b != NONE) {
            sum += dot(bodyFaceJ(c, q, b), pushChange(b, which));
        }
    }
    return sum;
}

// Body b's velocity at x (cells) going into the solve: as last stamped, the
// pull's step, and what the pressure at its surface as posed does to it,
// and the liquid's drag, once summed; the pull's and the pushes as its
// contacts let it (giveWay: resting on the floor, it isn't taken to fall
// through it), the pull's varying across it besides.
fn bodyMoving(b: u32, x: vec2<f32>, drag: bool) -> vec2<f32> {
    let pos = bodyV(b, BW_POS);
    let pull = pullAt(pos);
    var ch = pushChange(b, PUSH_SURFACE) + giveWay(b, vec3<f32>(rigidMass(b) * pull, 0.0));
    if (drag) {
        ch += pushChange(b, PUSH_DRAG);
    }
    let r = x - pos;
    return rigidVel(b, x) + pullAt(x) - pull + ch.xy + vec2<f32>(-ch.z * r.y, ch.z * r.x);
}

// The body whose cell c is in the solve, as classify found it; NONE if none.
fn rigidAt(c: vec2<i32>) -> u32 {
    if (!inWorld(c)) {
        return NONE;
    }
    let z = field[2u * cellIndex(c) + 1u].z;
    let b = u32(max(-z, 1.0)) - 1u;
    return select(NONE, b, z < 0.0 && b < MAX_BODIES);
}

// Whether body b is held (bodies.wgsl's B_HELD): a wall to the solve, its
// cells and its cup's air obstacles standing still as the rest do, and its
// push the liquid's pressure on them (heldPush) rather than what the solve
// did to it as liquid of its density. Held by what stands still, it is at
// rest but for what the contacts let it slide. In the solve, its faces on
// what holds it moving at its velocity and the pull (as a free body's), it
// took in the liquid as a drain does, the water over a block at rest drawn
// down 0.16 to 0.33 cells a step (REWRITE_PLAN.md P3-19). (A wall moving
// at the body's velocity, pushed a step late, was a piston: water shut in
// between two hulls and a block resting on them was pressed to 268, eight
// times its weight there, and pushed them at nine cells a step a step.
// What a wall shuts in moves with it, an added mass that a push a step
// late can't hold steady: Förster, Wall & Ramm 2007.)
fn isHeld(b: u32) -> bool {
    return (bodyWord(b, BW_FLAGS) & BF_HELD) != 0u;
}

// The held body whose cell (or cup) c is, as classify found it; NONE if
// none.
fn heldAt(c: vec2<i32>) -> u32 {
    if (!inWorld(c)) {
        return NONE;
    }
    let z = field[2u * cellIndex(c) + 1u].z;
    let b = u32(max(-z, 1.0)) - 1u;
    return select(NONE, b - MAX_BODIES, z < 0.0 && b >= MAX_BODIES);
}

// The velocity of body b at x (cells), as it was moving when last stamped.
fn rigidVel(b: u32, x: vec2<f32>) -> vec2<f32> {
    let r = x - bodyV(b, BW_POS);
    let spin = bodyF(b, BW_SPIN);
    return bodyV(b, BW_VEL) + vec2<f32>(-spin * r.y, spin * r.x);
}

// Whether particles are kept out of cell c as classified: an obstacle's, or
// a body's (in the solve, but solid).
fn shutAt(c: vec2<i32>) -> bool {
    return gridAt(c).z < 0.0 || rigidAt(c) != NONE;
}

// The middle of the face at `at` on axis isU.
fn faceMiddle(at: vec2<i32>, isU: bool) -> vec2<f32> {
    return vec2<f32>(at) + select(vec2<f32>(0.5, 0.0), vec2<f32>(0.0, 0.5), isU);
}

// Whether a cell was liquid (fluid, sealed off included) as of the last two
// steps is kept in its drift's w, a bit for each parity of P.frame: classify
// reads its neighbours' from last step as they write this step's, and with
// one bit a neighbour might have written this step's already, so that how a
// cell is sealed would depend on which workgroup ran first. The bit for this
// step (back 0) or the last (back 1).
fn stepBit(back: u32) -> u32 {
    return 1u << ((P.frame + back) & 1u);
}

// And whether the liquid the step's advect brought into it was all falling
// free, its pressure where it came from under half a cell of its weight
// under the pull, two bits more: a gas cell under liquid is held as a
// bubble but where it was (isObstacleCell). Kept, as the liquid's bits are,
// a step's by the next. The liquid's, not the cell's: a block falling
// several cells a step comes into cells at its foot that held none, which
// by a bit the cells kept from the solve held their gaps as bubbles again.
// And all of it: marked by any of it (a particle sinking from a pool's top
// cell, at half a cell of its weight, would do), a box afloat lost air and
// a plank afloat never came to rest (test/sealedhold.mjs, plankrest.mjs).
fn fallingBit(back: u32) -> u32 {
    return 4u << ((P.frame + back) & 1u);
}

// Whether the empty cell at q is open to the air, as of last step: neither
// liquid nor sealed off (see classify). Blocks not worked on are open.
fn wasOpen(q: vec2<i32>) -> bool {
    if (!inWorld(q) || countAt(q) < 0) {
        return false;
    }
    return (u32(field[2u * cellIndex(q) + 1u].w) & stepBit(1u)) == 0u;
}

// Whether the empty cell at c may be a gap the liquid has left: liquid,
// sealed off or a body's last step (field's, which classify has yet to
// write), and no particles in it now.
fn mayBeSealed(c: vec2<i32>) -> bool {
    if (countAt(c) != 0) {
        return false;
    }
    let i = cellIndex(c);
    return field[2u * i].z > 0.0 || field[2u * i + 1u].z != 0.0;
}

// The gaps the liquid has left, each the cells joined through such cells
// face to face (label.wgsl's union-find, in G_REGION's words, each its own
// to start with from clear): a gap any cell of which was open to the air
// last step is open, its root's link marked REGION_OPEN, and the rest
// sealed (classify). The air's solve is incompressible, and a gap it meets
// is at its pressure within the step, however far in. (Sealed until the
// air reached it a cell at a time, a block of water falling faster than a
// cell a step left a vacuum over itself, a cell more of it every step, and
// its suction held the block to a quarter of free fall: REWRITE_PLAN.md
// 7-18.)
@compute @workgroup_size(8, 8)
fn sealJoin(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let c = workCell(wid.x, lid.xy);
    if (!inWorld(c) || !mayBeSealed(c)) {
        return;
    }
    for (var d = 0; d < 2; d++) {
        let n = c + select(vec2<i32>(0, 1), vec2<i32>(1, 0), d == 0);
        if (mayBeSealed(n)) {
            ufJoin(0u, cellIndex(c), cellIndex(n));
        }
    }
}

@compute @workgroup_size(8, 8)
fn sealAnchor(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let c = workCell(wid.x, lid.xy);
    if (!inWorld(c) || !mayBeSealed(c)) {
        return;
    }
    if (wasOpen(beside(c, 0)) || wasOpen(beside(c, 1)) || wasOpen(beside(c, 2)) || wasOpen(beside(c, 3))) {
        atomicOr(&counters[gridSlot(ufFind(0u, cellIndex(c))) + G_REGION], REGION_OPEN);
    }
}

var<workgroup> blockBits: atomic<u32>;

// The pressure a cell newly liquid starts from (classify; last step's, the
// rest's): the weight of the liquid over it in its column under the
// Earth's pull, ρ g a cell of it, from half a cell at the surface. A
// column painted at rest started from none, and the solve, a V-cycle a
// step, took tens of steps to learn its load: water fell a little and
// mud, whose still walls take gravity's step first (7-11), sank for some
// 150 steps (REWRITE_PLAN.md 7-15). It only starts the solve, which then
// settles the pressure as it does every step.
const COLUMN_MOST: i32 = 512;
fn columnWeight(p: vec2<i32>, density: f32) -> f32 {
    var over = 0;
    var q = p - vec2<i32>(0, 1);
    while (over < COLUMN_MOST && countAt(q) > 0) {
        over += 1;
        q -= vec2<i32>(0, 1);
    }
    return density * P.earthG * (f32(over) + 0.5);
}

// Each workgroup is one block, and notes what its cells hold.
@compute @workgroup_size(8, 8)
fn classify(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let p = workCell(wid.x, lid.xy);
    if (inWorld(p)) {
        let i = cellIndex(p);
        let n = countAt(p);
        var c = vec4<f32>(0.0);
        let was = field[2u * i];
        let wasDrift = field[2u * i + 1u];
        // Empty space the liquid has left, with no way in for the air: a gap
        // fast flow tears in a pipe, the dead water past a sharp corner. It
        // is a vacuum, not air at no pressure: the suction there holds the
        // column together, as it does in real water, and draws the liquid
        // back in (divergence). It is sealed while none of the gap it is in
        // meets the air (sealAnchor).
        var sealed = false;
        if (n == 0 && mayBeSealed(p)) {
            sealed = (ufLink(ufFind(0u, i)) & REGION_OPEN) == 0u;
        }
        var rigid = NONE;
        var held = NONE;
        let seam = select(seamOf(i), NONE, n < 0);
        if (seam != NONE) {
            // A gap the stamp opened in a body's wall: the body's (seamOf).
            c.z = -1.0;
            rigid = seam;
            if (isHeld(seam)) {
                held = seam;
                rigid = NONE;
            }
            sealed = false;
        } else if (n < 0) {
            c.z = -1.0;
            rigid = rigidBodyOf(cells[i]);
            // Powder in its cup is its cargo, moving with it.
            if (rigid == NONE && elems[cellType(cells[i])].kind == KIND_POWDER) {
                rigid = cupOf(i);
            }
            // A held body's a wall (isHeld).
            if (rigid != NONE && isHeld(rigid)) {
                held = rigid;
                rigid = NONE;
            }
        } else if (n > 0) {
            c = vec4<f32>(0.0, 0.0, liquidIn(i));
            // Liquid at the surface, where surface tension acts, takes on its
            // viscosity (SURFACE_DAMP).
            let sigma = waterTension();
            if (sigma > 0.0 && (countAt(beside(p, 0)) == 0 || countAt(beside(p, 1)) == 0
                || countAt(beside(p, 2)) == 0 || countAt(beside(p, 3)) == 0)) {
                c.w = max(c.w, sigma * elems[wordType(cells[i].x)].tension * SURFACE_DAMP);
            }
        } else {
            // An empty cell inside the liquid, left by particles crowding
            // into the cells around it, is liquid too, short of one
            // (divergence): as air it would hold no pressure and drag down
            // the pressure all around it. At least two sides must be liquid,
            // so the cell over the surface in a narrow tube doesn't count and
            // draw the liquid up it.
            var fluid = 0.0;
            var walls = 0.0;
            var mean = vec2<f32>(0.0);
            for (var d = 0; d < 4; d++) {
                let q = beside(p, d);
                let m = countAt(q);
                if (m < 0) {
                    walls += 1.0;
                } else if (m > 0) {
                    fluid += 1.0;
                    mean += liquidIn(cellIndex(q));
                }
            }
            if (fluid >= 2.0 && fluid + walls >= 3.0) {
                c = vec4<f32>(0.0, 0.0, mean / fluid);
            } else if (sealed) {
                c = vec4<f32>(0.0, 0.0, select(vec2<f32>(1.0, 0.0), mean / fluid, fluid > 0.0));
            }
        }
        // Both potentials start from last step's, the drift's from a little
        // less (DRIFT_KEEP). This step's liquid bit is set (stepBit) and
        // last step's kept, which the neighbours are reading.
        // (A body's cells count as liquid here: a cell it has just left
        // under the water, holding nothing (bodies.wgsl's wetLeft), is shut
        // off from the air (mayBeSealed, wasOpen), and holds the liquid's
        // pressure until the liquid flows in, rather than dropping it to
        // the air's for a step under the body, and its buoyancy with it.)
        let mine = stepBit(0u);
        let liquid = select(0u, mine, c.z > 0.0 || rigid != NONE || held != NONE);
        var drift = vec4<f32>(0.0, 0.0, select(0.0, 1.0, sealed), f32((u32(wasDrift.w) & ~(mine | fallingBit(0u))) | liquid));
        if (rigid != NONE) {
            drift.z = -1.0 - f32(rigid);
        } else if (held != NONE) {
            drift.z = -1.0 - f32(held + MAX_BODIES);
        }
        if (c.z > 0.0) {
            if (was.z > 0.0) {
                c.x = was.x;
                drift.x = wasDrift.x * DRIFT_KEEP;
            } else {
                c.x = columnWeight(p, c.z);
            }
            atomicOr(&blockBits, select(HAS_FLUID, HAS_FLUID | HAS_VISCOUS, c.w > 0.0));
        }
        field[2u * i] = c;
        field[2u * i + 1u] = drift;
    }
    workgroupBarrier();
    if (li == 0u && wid.x < activeCount()) {
        let bits = atomicLoad(&blockBits);
        atomicOr(&blocks[stateSlot(blockIndex(activeBlock(wid.x)))], bits);
        if ((bits & HAS_VISCOUS) != 0u) {
            atomicStore(&blocks[countSlot(COUNT_VISCOUS)], 1u);
        }
    }
}

// ------------------------------------------------------- surface tension
//
// A liquid's surface pulls itself in, so the pressure just inside a curved
// surface is higher than outside by σκ (σ its surface tension, κ how curved
// the surface is; Young-Laplace). Water's capillary length sqrt(σ / ρg),
// 2.7 mm, is how far that reaches against gravity: a full glass heaps up to
// about twice that over its rim, a drop smaller than it stays round, and
// water climbs a slit w wide 2 l² cos θ / w (Jurin's law; de Gennes,
// Brochard-Wyart & Quéré, Capillarity and Wetting Phenomena, 2004). So it
// only shows in a world of millimetre cells (Params.dx); at a
// centimetre a cell it is a fraction of a cell, and the pass is left out.
// It is the ghost fluid method's condition at the surface (Kang, Fedkiw &
// Liu, J. Sci. Comput. 15, 2000): the open air against the liquid is held at
// σκ instead of 0 in the pressure solve. κ comes from a smoothed outline of
// the liquid; where the surface meets a solid, the contact angle adds its
// pull, so liquid climbs glass and holds back on a waxy leaf.
// CAPILLARY_LENGTH (water's, in metres) and SURFACE_MIN_LENGTH (capillary
// lengths in cells under which it's left out) come from js/shared.js.
const MAX_CURVATURE: f32 = 1.0;        // per cell
// The smoothed outline rounds off features a few cells across, so its
// curvature comes out about half of the liquid's: a full cup heaps up half as
// high over its rim as it should. Scaled back up by this.
const CURVATURE_GAIN: f32 = 2.0;
// The contact line's pull, σ cos θ times this, per solid beside a cell of
// the surface: 2 brings water up slits one and two cells wide to Jurin's
// heights (within 10-15 %), given how the solve spreads it over the cell.
const CONTACT: f32 = 2.0;
// Done explicitly, surface tension needs time steps under about 1 / l steps
// (l the capillary length in cells; Brackbill, Kothe & Zemach, J. Comput.
// Phys. 100, 1992), and millimetre cells make l nearly 3. So liquid at the
// surface gets a viscosity of its surface tension times SURFACE_DAMP, which
// the viscosity sweeps solve implicitly: in effect the semi-implicit surface
// tension of Hysing (Int. J. Numer. Meth. Fluids 51, 2006), damping the
// waves that would otherwise grow.
const SURFACE_DAMP: f32 = 1.0;

// Water's surface tension in the pressure solve's units, ρ g l² (water's
// density being 1); 0 where it's too small to bother with.
fn waterTension() -> f32 {
    let l = CAPILLARY_LENGTH / P.dx;
    return select(0.0, elems[WATER].weight * l * l, l >= SURFACE_MIN_LENGTH);
}

// How much of the liquid's outline the cell at c is: 1 liquid, 0 open space,
// half for a solid or powder or past the world's edge (how far liquid wets a
// solid is the contact term's, in surface).
fn outline(c: vec2<i32>) -> f32 {
    if (!inWorld(c)) {
        return 0.5;
    }
    let i = cellIndex(c);
    let v = cells[i];
    if (isObstacleCell(c, v)) {
        return 0.5;
    }
    return select(0.0, 1.0, atomicLoad(&counters[gridSlot(i) + G_COUNT]) > 0u);
}

// Open space against the liquid gets the pressure σκ, from the outline's
// derivatives, Gaussian-smoothed over 5 x 5 cells: κ is the divergence of
// the outline's unit normal, -(o_xx o_y² - 2 o_x o_y o_xy + o_yy o_x²) / |∇o|³,
// positive where the liquid bulges out.
@compute @workgroup_size(8, 8)
fn surface(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let c = workCell(wid.x, lid.xy);
    if (!inWorld(c)) {
        return;
    }
    let i = cellIndex(c);
    if (field[2u * i].z != 0.0) {
        return;
    }
    var tension = 0.0;
    var contact = 0.0;  // the cosines of the contact angles of the solids beside it
    for (var d = 0; d < 4; d++) {
        let q = beside(c, d);
        if (!inWorld(q)) {
            continue;
        }
        let z = field[2u * cellIndex(q)].z;
        if (z > 0.0) {
            tension = max(tension, elems[wordType(cells[cellIndex(q)].x)].tension);
        } else if (z < 0.0) {
            contact += 2.0 * elems[wordType(cells[cellIndex(q)].x)].wetting - 1.0;
        }
    }
    if (tension <= 0.0) {
        return;
    }
    var g = vec2<f32>(0.0);
    var h = vec3<f32>(0.0);  // o_xx, o_yy, o_xy
    for (var y = -2; y <= 2; y++) {
        for (var x = -2; x <= 2; x++) {
            let k = vec2<f32>(f32(x), f32(y));
            let w = exp(-0.5 * dot(k, k)) * outline(c + vec2<i32>(x, y));
            g += k * w;
            h += vec3<f32>(k.x * k.x - 1.0, k.y * k.y - 1.0, k.x * k.y) * w;
        }
    }
    let g2 = dot(g, g);
    var kappa = 0.0;
    if (g2 >= 1e-6) {
        kappa = -(h.x * g.y * g.y - 2.0 * g.x * g.y * h.z + h.y * g.x * g.x) / (g2 * sqrt(g2));
    }
    // Where the surface meets a solid, liquid is drawn along it by σ cos θ
    // over a cell (Young), or held back for a cos θ under 0: that's what
    // makes it climb a slit, 2 l² cos θ / w (Jurin), and bead.
    let curved = clamp(kappa * CURVATURE_GAIN, -MAX_CURVATURE, MAX_CURVATURE);
    field[2u * i].x = waterTension() * tension * (curved - CONTACT * contact);
}

const POWDER_PUSH: f32 = 0.3;

// Velocity the liquid takes on beside the obstacle at c: none beside a
// solid, part of a powder's own. Powder moving through liquid trades places
// with it (block.wgsl), which already moves it out of the way; the push is
// what is left over, the wake a clump drags along. A bubble trades places
// too, but pushes nothing: gas can't drive liquid hundreds of times denser
// along, and steam rushing up a column at the updraft's speed would carry
// the liquid between its cells up with it (a meteor's lava, up its own
// steam). A free body's cells aren't obstacles but in the solve (rigid
// cells): those of one just found, still to be weighed, stand still.
fn obstacleVel(c: vec2<i32>) -> vec2<f32> {
    if (!inWorld(c)) {
        return vec2<f32>(0.0);
    }
    let held = occupants[cellIndex(c)];
    if (isOccupant(held)) {
        return occupantVel(held);
    }
    let v = cells[cellIndex(c)];
    let k = elems[cellType(v)].kind;
    if (k == KIND_SOLID || k == KIND_GAS) {
        return vec2<f32>(0.0);
    }
    return cellVel(v) * POWDER_PUSH;
}

// A face of a free body's cell: the body's velocity there going into the
// solve (bodyMoving), the pull's step and all; with the liquid's drag once
// the sweeps have summed it. Between two bodies' cells, the mean. Its
// viscosity the liquid's beside it, so the sweeps drag the body along
// (viscousFace, bodyDrag).
fn setBodyFace(f: u32, at: vec2<i32>, isU: bool, ga: vec4<f32>, gb: vec4<f32>, ra: u32, rb: u32, drag: bool) {
    let axis = select(1, 0, isU);
    let x = faceMiddle(at, isU);
    var now = 0.0;
    var u = 0.0;
    if (ra != NONE && rb != NONE) {
        now = 0.5 * (bodyMoving(ra, x, drag)[axis] + bodyMoving(rb, x, drag)[axis]);
        u = 0.5 * (rigidVel(ra, x)[axis] + rigidVel(rb, x)[axis]);
    } else {
        let b = select(rb, ra, ra != NONE);
        now = bodyMoving(b, x, drag)[axis];
        u = rigidVel(b, x)[axis];
    }
    let other = select(ga, gb, ra != NONE);
    let nu = select(0.0, other.w, (ra == NONE || rb == NONE) && other.z > 0.0);
    field[faceSlot(f)] = vec4<f32>(now, u, 1.0, nu);
    setViscRhs(f, now);
}

// The air's mass in the cell of gas c, cells of water (its gas's amount at
// air's density), as the bodies' drag takes it (bodies.wgsl's airFace).
fn airMassAt(c: vec2<i32>) -> f32 {
    let j = cellIndex(c);
    return elems[AIR].rho / 1000.0 * gasAmount(gasFrom(cells[gasIndex(cellCount(), j)], j));
}

// The drag between liquid of mass ml moving at u and air of mass ma moving
// at w along an axis, the stress between them ρ_air c |u - w| (u - w) (the
// bulk aerodynamic formula), so the share of their difference it would take
// off the air a step, were the liquid still, c |u - w| (cells a step):
// worked out for both at once (implicitly), what one loses the other gains,
// and neither past the other. The change in the liquid's velocity, and in
// the air's.
fn pairDrag(u: f32, w: f32, ml: f32, ma: f32, c: f32) -> vec2<f32> {
    if (ml <= 0.0 || ma <= 0.0 || c <= 0.0) {
        return vec2<f32>(0.0);
    }
    let mu = ml / ma;
    let k = c * abs(u - w);
    let change = (u - w) * (1.0 - 1.0 / (1.0 + k * (1.0 + 1.0 / mu)));
    return vec2<f32>(-change / (1.0 + mu), change * mu / (1.0 + mu));
}

// The drag coefficient of the wind along a liquid's surface, by the log law
// at the middle of the cell of air over it (Garratt, The Atmospheric
// Boundary Layer, 1992, ch. 3): (κ / ln(z / z0))², κ von Kármán's
// (KARMAN), z half a cell, z0 calm water's roughness, 0.1 mm: 0.011 at a
// centimetre a cell, 0.002 at a metre (the sea's, 1 to 2.5 thousandths at
// 10 m, Large & Pond 1981).
const SURFACE_ROUGHNESS: f32 = 1e-4;

fn surfaceDragCoefficient() -> f32 {
    let r = KARMAN / log(max(0.5 * P.dx / SURFACE_ROUGHNESS, 2.0));
    return r * r;
}

// The wind's drag along the liquid's surface at the face at `at` on axis
// isU (between a and b, as setFace), u its velocity there as splatted, and
// the liquid's on the wind (REWRITE_PLAN.md 4e-11): along a face between two
// cells of liquid with gas beside both, the wind's stress on the surface
// (pairDrag, surfaceDragCoefficient), with the air's face beside it, the
// air taking what the liquid gives up (airsource.wgsl's pushFace). The
// change in the face's velocity. Across the surface there is none: the
// liquid moving into the air moves it by what it puts aside (the air's
// sources, gas pushed out of cells the liquid takes) and the air's pressure
// pushes back on it (gather's pressurePush). (Each particle at the surface
// was dragged by the wind as if the air had no mass, nothing going back into
// it. With the liquid's drag (elem.drag) on the air as it was, the air the
// surface touched was stopped every step, 0.005 cells a step against the
// wind's 0.2 to 0.45 there, and the wind over a pool died twice as fast;
// and the liquid moving into the air carrying it along as well, the
// boiling water in Sealed boiler kicked its air every step, its pressure
// up to twice what gave the lid.)
fn surfaceDrag(a: vec2<i32>, b: vec2<i32>, isU: bool, ga: vec4<f32>, gb: vec4<f32>, u: f32) -> f32 {
    let dims = airDims(P.size);
    let across = select(vec2<i32>(1, 0), vec2<i32>(0, 1), isU);
    var du = 0.0;
    if (ga.z > 0.0 && gb.z > 0.0) {
        let k = surfaceDragCoefficient();
        for (var s = -1; s <= 1; s += 2) {
            let oa = a + across * s;
            let ob = b + across * s;
            if (gasCellAt(oa, dims) && gasCellAt(ob, dims)) {
                let w = select(faceV(oa, dims), faceU(oa, dims), isU);
                let drag = pairDrag(u + du, w, 0.5 * (ga.z + gb.z), 0.5 * (airMassAt(oa) + airMassAt(ob)), k);
                du += drag.x;
                pushFace(oa, select(vec2<f32>(0.0, drag.y), vec2<f32>(drag.y, 0.0), isU), dims);
            }
        }
    }
    return du;
}

// The face at `at` on axis isU, between cells a and b (b to the right of or
// below a).
fn setFace(at: vec2<i32>, isU: bool) {
    let f = faceIndex(at, isU);
    let a = at - select(vec2<i32>(0, 1), vec2<i32>(1, 0), isU);
    let axis = select(1, 0, isU);
    let ga = gridAt(a);
    let gb = gridAt(at);
    let slot = sumSlot(f);
    let weight = i32(atomicLoad(&counters[slot + 1u]));
    let ra = rigidAt(a);
    let rb = rigidAt(at);
    if (ra != NONE || rb != NONE) {
        setBodyFace(f, at, isU, ga, gb, ra, rb, false);
        return;
    }
    var u = 0.0;
    var held = 0.0;
    if (ga.z < 0.0 && gb.z < 0.0) {
        held = -1.0;
    } else if (ga.z < 0.0 || gb.z < 0.0) {
        // Liquid against an obstacle moves with it across their face (a
        // held body's still: see isHeld).
        u = obstacleVel(select(at, a, ga.z < 0.0))[axis];
        held = 1.0;
    } else if (weight > 0) {
        u = f32(i32(atomicLoad(&counters[slot]))) / f32(weight);
        held = 1.0;
    }
    var now = u;
    if (held > 0.0) {
        now += surfaceDrag(a, at, isU, ga, gb, u);
    }
    // Liquids all fall alike, under the Earth's pull and the world's masses'
    // (pull.wgsl), along both axes, as it is at the face.
    if (held > 0.0 && ga.z >= 0.0 && gb.z >= 0.0) {
        let pull = pullAt(vec2<f32>(at) + select(vec2<f32>(0.5, 0.0), vec2<f32>(0.0, 0.5), isU));
        now += select(pull.y, pull.x, isU);
    }
    var start = now;
    if (isViscousFace(ga, gb)) {
        now -= lastPush(a, at, ga, gb);
        start = now + viscLast(f);
    }
    field[faceSlot(f)] = vec4<f32>(start, u, held, faceViscosity(ga, gb));
    setViscRhs(f, now);
}

@compute @workgroup_size(8, 8)
fn setFaces(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let c = workCell(wid.x, lid.xy);
    if (!inWorld(c)) {
        return;
    }
    let fs = facesOf(c);
    for (var j = 0; j < 4; j++) {
        if (fs[j].z >= 0) {
            setFace(fs[j].xy, fs[j].z == 1);
        }
    }
}

@compute @workgroup_size(8, 8)
fn divergence(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    // The step's solve starts: its first iteration's slot for what it
    // leaves emptied, and none run yet.
    if (wid.x == 0u && all(lid.xy == vec2<u32>(0u))) {
        atomicStore(&blocks[countSlot(COUNT_RESIDUAL)], 0u);
        atomicStore(&blocks[countSlot(COUNT_CYCLES)], 0u);
    }
    let c = workCell(wid.x, lid.xy);
    if (!inWorld(c)) {
        return;
    }
    let i = cellIndex(c);
    if (field[2u * i].z <= 0.0) {
        return;
    }
    let out = field[faceSlot(uFace(c.x + 1, c.y))].x - field[faceSlot(uFace(c.x, c.y))].x
        + field[faceSlot(vFace(c.x, c.y + 1))].x - field[faceSlot(vFace(c.x, c.y))].x;
    field[2u * i].y = out;
    // Particles crowded together, more than a cell's worth (LIQUID_PER_CELL)
    // to a cell by their smoothed share (densityAt: Kugelstadt, Longva,
    // Thuerey & Bender 2019's density projection, by the same weights as the
    // splat), drift apart, and a cell holding less draws them in, so the
    // liquid keeps its volume however the pressure solve's errors add up,
    // and its cells full but at the surface. (By whole particles in a cell,
    // the drift's steps stirred the surface, which the ghost fluid's
    // pressure then passed on, growing: REWRITE_PLAN.md 7-3.)
    let n = densityAt(c);
    field[2u * i + 1u].y = select(-SPREAD * (n - posedRoom(c)), 0.0, field[2u * i + 1u].z < 0.0);
}

// The liquid liquid cell c holds beside free bodies, where their poses have
// them (bodyReach): a cell's worth, less what a body's pose fills past the
// face between, or more by what its stamped cell's pose leaves empty short
// of it, which is the liquid's but there. The drift fills it to that.
// (Their stamps are whole cells, their poses not. Filled to a cell, a cell
// under a floating plank sinking within its row held a cell of water, which
// its stamp coming down into it then put out of its way on top of what the
// plank's face had pushed out on its way there: a lift too much each time.)
fn posedRoom(c: vec2<i32>) -> f32 {
    var room = 1.0;
    for (var d = 0; d < 4; d++) {
        let q = beside(c, d);
        let b = rigidAt(q);
        if (b != NONE) {
            room += 0.5 - bodyReach(b, c - q);
        }
    }
    return clamp(room, 0.1, 1.9);
}

// The pressure and drift are solved together (the same equations, with
// different right-hand sides). Open air holds neither, and obstacles take no
// part (their faces don't move with the liquid). Red-black Gauss-Seidel
// smooths out the error over short distances, before and after a multigrid
// V-cycle (below) takes care of the long ones. Each dispatch relaxes each
// block in workgroup memory for several sweeps, its one-cell rim held as it
// was, which leaves two or three times the error along a tile's edges as
// inside it. The tiles shift by half a block every other dispatch, and each
// step starts on the other tiling from the last (sim.js), so those edges
// don't stay put and hold the solve back there. Read once and written once
// per dispatch, rather than per sweep, the field costs a fraction of the
// memory traffic. A tile's rim is read as relaxRims left it, before the
// dispatch: read from the field, it was what the tile beside it had or
// hadn't yet written there, as the GPU happened to run them, and two runs
// from the same snapshot came apart (REWRITE_PLAN.md D-1).
const RIM: i32 = BLOCK + 2;
const SWEEPS: i32 = 4;  // red-black sweep pairs per dispatch

var<workgroup> tilePot: array<vec2<f32>, 100>;  // RIM x RIM: pressure, drift
var<workgroup> tileRho: array<f32, 100>;        // density; < 0 obstacle, 0 air
var<workgroup> tileShare: array<f32, 100>;      // the particles' smoothed share (densityAt)
var<workgroup> tileLive: u32;
var<workgroup> tileOrigin: vec2<i32>;

// The block whose tile, shifted up and left by shift, workgroup wid relaxes,
// x < 0 for none: one per block worked on. Shifted by half a block, those
// tiles end half a block short of the last blocks' right and bottom edges,
// which would leave the world's last columns (rows) out of every shifted
// relaxation unless its width (height) is 1 to 4 cells past a whole number
// of blocks; the divergence left there was 5-15 times the rest's. So the
// workgroups after one per block in the world take the tiles of the column
// of blocks past the right edge (the corner too) and of the row past the
// bottom, which return at once where the world doesn't reach into them.
fn tileBlock(wid: u32, shift: i32) -> vec2<i32> {
    if (wid < activeCount()) {
        return activeBlock(wid);
    }
    let wide = i32(blocksWide());
    let high = i32(blockCount() / blocksWide());
    let j = i32(wid) - i32(blockCount());
    if (shift == 0 || j < 0 || j > high + wide) {
        return vec2<i32>(-1);
    }
    return select(vec2<i32>(j - high - 1, high), vec2<i32>(wide, j), j <= high);
}

// Over the blocks worked on, before each relaxation: each liquid cell's
// pressure and drift as they are, for the tiles' rims (G_RIM). (Only liquid
// is relaxed, and there is none outside those blocks: retire.)
@compute @workgroup_size(8, 8)
fn relaxRims(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let c = workCell(wid.x, lid.xy);
    if (!inWorld(c) || converged()) {
        return;
    }
    let i = cellIndex(c);
    if (field[2u * i].z > 0.0) {
        atomicStore(&counters[gridSlot(i) + G_RIM], bitcast<u32>(field[2u * i].x));
        atomicStore(&counters[gridSlot(i) + G_RIM + 1u], bitcast<u32>(field[2u * i + 1u].x));
    }
}

fn relaxTile(wid: u32, lid: vec2<u32>, shift: i32) {
    // A tile with no liquid in it has nothing to solve, and those past the
    // blocks worked on (and the world's edges) aren't tiles at all. One
    // thread works out where it is, for all of them.
    if (all(lid == vec2<u32>(0u))) {
        tileLive = 0u;
        let b = tileBlock(wid, shift);
        tileOrigin = b * BLOCK - shift;
        if (b.x >= 0 && all(tileOrigin < vec2<i32>(P.size)) && !converged()) {
            tileLive = statesOf(tileOrigin / BLOCK, (tileOrigin + BLOCK - 1) / BLOCK) & HAS_FLUID;
        }
    }
    if (workgroupUniformLoad(&tileLive) == 0u) {
        return;
    }
    let origin = tileOrigin;
    for (var k = i32(lid.y) * BLOCK + i32(lid.x); k < RIM * RIM; k += BLOCK * BLOCK) {
        let c = origin - 1 + vec2<i32>(k % RIM, k / RIM);
        var pot = vec2<f32>(0.0);
        var rho = -1.0;
        if (inWorld(c)) {
            let j = 2u * cellIndex(c);
            let g = field[j];
            pot = vec2<f32>(g.x, field[j + 1u].x);
            rho = g.z;
            if (rho > 0.0) {
                let r = gridSlot(cellIndex(c)) + G_RIM;
                pot = vec2<f32>(bitcast<f32>(atomicLoad(&counters[r])), bitcast<f32>(atomicLoad(&counters[r + 1u])));
            }
        }
        tilePot[k] = pot;
        tileRho[k] = rho;
        tileShare[k] = shareAt(c);
    }
    workgroupBarrier();

    let me = origin + vec2<i32>(lid);
    let t = (i32(lid.y) + 1) * RIM + i32(lid.x) + 1;
    let rho = tileRho[t];
    var rhs = vec2<f32>(0.0);
    var beta = vec4<f32>(0.0);  // to the left, right, upper and lower neighbors
    if (rho > 0.0) {
        let i = 2u * cellIndex(me);
        rhs = vec2<f32>(field[i].y, field[i + 1u].y);
        let d = tileShare[t];
        beta = vec4<f32>(
            fineBeta(rho, tileRho[t - 1], d, tileShare[t - 1], vec2<i32>(-1, 0)),
            fineBeta(rho, tileRho[t + 1], d, tileShare[t + 1], vec2<i32>(1, 0)),
            fineBeta(rho, tileRho[t - RIM], d, tileShare[t - RIM], vec2<i32>(0, -1)),
            fineBeta(rho, tileRho[t + RIM], d, tileShare[t + RIM], vec2<i32>(0, 1)),
        );
    }
    // (The pressure's held at a pinned root, which a cell of water closed in
    // by grains is, the drift's not: there it stays.)
    let diag = beta.x + beta.y + beta.z + beta.w;
    let pinned = rho > 0.0 && pinnedAt(cellIndex(me));
    let diags = vec2<f32>(diag + select(0.0, PIN, pinned), max(diag, 1e-30));
    let solving = diags.x > 0.0;
    let color = (me.x + me.y) & 1;
    for (var s = 0; s < 2 * SWEEPS; s++) {
        if (solving && color == (s & 1)) {
            let sum = beta.x * tilePot[t - 1] + beta.y * tilePot[t + 1]
                + beta.z * tilePot[t - RIM] + beta.w * tilePot[t + RIM];
            let next = mix(tilePot[t], (sum - rhs) / diags, SMOOTH_OMEGA);
            tilePot[t] = vec2<f32>(next.x, select(tilePot[t].y, next.y, diag > 0.0));
        }
        workgroupBarrier();
    }
    if (solving) {
        let i = 2u * cellIndex(me);
        field[i].x = tilePot[t].x;
        field[i + 1u].x = tilePot[t].y;
    }
}

@compute @workgroup_size(8, 8)
fn relax(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    relaxTile(wid.x, lid.xy, 0);
}

@compute @workgroup_size(8, 8)
fn relaxShifted(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    relaxTile(wid.x, lid.xy, BLOCK / 2);
}

// ------------------------------------------------------------ multigrid
//
// Relaxation alone settles the pressure locally, but only creeps across
// long distances: in a deep pool it would take hundreds of sweeps for the
// bottom to feel the weight of the water over it, and until then the water
// sloshes about. So the error left after relaxing is solved for on coarser
// and coarser grids (each cell four of the one finer), where long distances
// are short, and the correction carried back down (a multigrid V-cycle;
// McAdams, Sifakis & Teran 2010, "A parallel multigrid Poisson solver for
// fluids simulation on large grids"). A coarse cell is air if any of its
// four is, so the surface keeps its pressure; liquid if any of its four is
// (of their mean density); otherwise an obstacle. Level 0 is the field's
// grid; each coarser level keeps per cell the correction and the error it
// corrects (pressure and drift, as vec2s), then its density.
//
// The V-cycle is the preconditioner of the pressure's conjugate gradients
// (see below), the drift relaxed alongside in the same passes.
//
// The pressure is solved for to LIQUID_TOLERANCE, the most divergence it
// leaves in any cell a step (in its volume): iterations, each a V-cycle (a
// relaxation, the coarse levels and a relaxation) and the CG's passes, up
// to sim.js's FLUID_CYCLES a step (`level.cycle`), until what the last one
// left (cgUpdate) is under it; the passes of those after return at once.
// Each iteration's most goes into its slot of three from COUNT_RESIDUAL:
// iteration k reads k - 1's, sums its own, and clears k + 1's (the first's
// cleared by divergence), as the air's do (air.wgsl's converged). The first
// always runs, the drift's one V-cycle a step.
const LIQUID_TOLERANCE: f32 = 1e-2;

fn converged() -> bool {
    let slot = countSlot(COUNT_RESIDUAL + (level.cycle + 2u) % 3u);
    return level.cycle > 0u && bitcast<f32>(atomicLoad(&blocks[slot])) < LIQUID_TOLERANCE;
}

fn levelSize(l: u32) -> vec2<u32> {
    return (P.size + vec2<u32>((1u << l) - 1u)) >> vec2<u32>(l);
}

fn inLevel(l: u32, c: vec2<i32>) -> bool {
    return all(c >= vec2<i32>(0)) && all(c < vec2<i32>(levelSize(l)));
}

// Where level l (from 1) starts in the field, after the faces.
fn levelStart(l: u32) -> u32 {
    var at = 2u * cellCount() + faceCount() + 2u * ((faceCount() + 3u) / 4u);
    for (var k = 1u; k < l; k++) {
        let s = levelSize(k);
        at += 2u * s.x * s.y;
    }
    return at;
}

fn levelSlot(l: u32, c: vec2<i32>) -> u32 {
    return levelStart(l) + 2u * (u32(c.y) * levelSize(l).x + u32(c.x));
}

// Density at c on level l: > 0 liquid, 0 air, < 0 obstacle (outside too).
fn rhoAt(l: u32, c: vec2<i32>) -> f32 {
    if (!inLevel(l, c)) {
        return -1.0;
    }
    if (l == 0u) {
        return field[2u * cellIndex(c)].z;
    }
    return field[levelSlot(l, c) + 1u].x;
}

// Pressure and drift on level 0, the corrections to them on the rest.
fn potAt(l: u32, c: vec2<i32>) -> vec2<f32> {
    if (!inLevel(l, c)) {
        return vec2<f32>(0.0);
    }
    if (l == 0u) {
        let i = 2u * cellIndex(c);
        return vec2<f32>(field[i].x, field[i + 1u].x);
    }
    return field[levelSlot(l, c)].xy;
}

fn rhsAt(l: u32, c: vec2<i32>) -> vec2<f32> {
    if (l == 0u) {
        let i = 2u * cellIndex(c);
        return vec2<f32>(field[i].y, field[i + 1u].y);
    }
    return field[levelSlot(l, c)].zw;
}

struct Stencil {
    sum: vec2<f32>,  // of the neighbors' potentials, each times its face's beta
    diag: f32,       // of the betas
}

fn stencil(l: u32, c: vec2<i32>, rho: f32) -> Stencil {
    var s: Stencil;
    for (var d = 0; d < 4; d++) {
        let n = beside(c, d);
        let r = rhoAt(l, n);
        if (r < 0.0) {
            continue;
        }
        var beta = faceBeta(rho, r);
        if (l == 0u) {
            beta = fineBeta(rho, r, shareAt(c), shareAt(n), n - c);
        }
        s.sum += beta * potAt(l, n);
        s.diag += beta;
    }
    return s;
}

// What is left of the equations at c on level l, its cells 2^l wide (the
// pressure's held at a pinned root on level 0: see Conjugate gradients).
fn residualAt(l: u32, c: vec2<i32>) -> vec2<f32> {
    let rho = rhoAt(l, c);
    if (rho <= 0.0) {
        return vec2<f32>(0.0);
    }
    let s = stencil(l, c, rho);
    let h2 = f32(1u << (2u * l));
    let pin = select(vec2<f32>(0.0), vec2<f32>(PIN, 0.0), l == 0u && pinnedAt(cellIndex(c)));
    return rhsAt(l, c) - (s.sum - (s.diag + pin) * potAt(l, c)) / h2;
}

// A Gauss-Seidel (omega 1) or SOR sweep of one color at c on a coarse level.
fn smoothAt(l: u32, c: vec2<i32>, color: i32, omega: f32) {
    if (!inLevel(l, c) || ((c.x + c.y) & 1) != color || activeCount() == 0u || converged()) {
        return;
    }
    let at = levelSlot(l, c);
    let rho = field[at + 1u].x;
    if (rho <= 0.0) {
        return;
    }
    let s = stencil(l, c, rho);
    if (s.diag <= 0.0) {
        return;
    }
    let e = field[at];
    let solved = (s.sum - f32(1u << (2u * l)) * e.zw) / s.diag;
    field[at] = vec4<f32>(mix(e.xy, solved, omega), e.zw);
}

// Level l's cells from their four one level finer: what they are, and the
// error to correct, the mean of what is left of the equations there. The
// correction starts at none. A cell over a block not worked on (empty: see
// retire) is air, known without reading the four, which most are, with a
// little liquid in a big world (reading them all cost 1 ms a step at
// 900x500); a cell of a level coarser than a block covers more than one.
@compute @workgroup_size(8, 8)
fn restrictLevel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let l = level.index;
    let c = vec2<i32>(gid.xy);
    if (!inLevel(l, c) || activeCount() == 0u || converged()) {
        return;
    }
    let at = levelSlot(l, c);
    let span = 1 << l;
    if (span <= BLOCK) {
        let b = c * span / BLOCK;
        if (inBlocks(b) && (atomicLoad(&blocks[stateSlot(blockIndex(b))]) & ACTIVE) == 0u) {
            field[at] = vec4<f32>(0.0);
            field[at + 1u] = vec4<f32>(0.0);
            return;
        }
    }
    var fluid = 0.0;
    var sum = 0.0;
    var air = false;
    var r = vec2<f32>(0.0);
    for (var j = 0; j < 4; j++) {
        let k = 2 * c + vec2<i32>(j & 1, j >> 1);
        let rho = rhoAt(l - 1u, k);
        if (rho == 0.0) {
            air = true;
        } else if (rho > 0.0) {
            fluid += 1.0;
            sum += rho;
            r += residualAt(l - 1u, k);
        }
    }
    let rho = select(select(-1.0, sum / fluid, fluid > 0.0), 0.0, air);
    field[at] = vec4<f32>(0.0, 0.0, select(vec2<f32>(0.0), r * 0.25, rho > 0.0));
    field[at + 1u] = vec4<f32>(rho, 0.0, 0.0, 0.0);
}

@compute @workgroup_size(8, 8)
fn smoothRed(@builtin(global_invocation_id) gid: vec3<u32>) {
    smoothAt(level.index, vec2<i32>(gid.xy), 0, 1.0);
}

@compute @workgroup_size(8, 8)
fn smoothBlack(@builtin(global_invocation_id) gid: vec3<u32>) {
    smoothAt(level.index, vec2<i32>(gid.xy), 1, 1.0);
}

// The coarsest level, small enough for one workgroup to hold in its memory
// (MG_COARSEST cells at most, as sim.js sees to) and sweep over and over until
// it has settled, a few cells to a thread.
const COARSE_SWEEPS: i32 = 40;

var<workgroup> coarsePot: array<vec2<f32>, MG_COARSEST>;
var<workgroup> coarseRhs: array<vec2<f32>, MG_COARSEST>;
var<workgroup> coarseRho: array<f32, MG_COARSEST>;

fn coarseRhoAt(size: vec2<u32>, c: vec2<i32>) -> f32 {
    if (any(c < vec2<i32>(0)) || any(c >= vec2<i32>(size))) {
        return -1.0;
    }
    return coarseRho[u32(c.y) * size.x + u32(c.x)];
}

var<workgroup> coarseSkip: u32;

@compute @workgroup_size(256)
fn solve(@builtin(local_invocation_index) li: u32) {
    if (li == 0u) {
        coarseSkip = u32(converged());
    }
    if (workgroupUniformLoad(&coarseSkip) != 0u) {
        return;
    }
    let l = level.index;
    let size = levelSize(l);
    let n = min(size.x * size.y, MG_COARSEST);
    for (var j = li; j < n; j += 256u) {
        let c = vec2<i32>(i32(j % size.x), i32(j / size.x));
        let at = levelSlot(l, c);
        coarsePot[j] = field[at].xy;
        coarseRhs[j] = field[at].zw;
        coarseRho[j] = field[at + 1u].x;
    }
    workgroupBarrier();
    let h2 = f32(1u << (2u * l));
    for (var s = 0; s < 2 * COARSE_SWEEPS; s++) {
        for (var j = li; j < n; j += 256u) {
            let c = vec2<i32>(i32(j % size.x), i32(j / size.x));
            let rho = coarseRho[j];
            if (((c.x + c.y) & 1) != (s & 1) || rho <= 0.0) {
                continue;
            }
            var sum = vec2<f32>(0.0);
            var diag = 0.0;
            for (var d = 0; d < 4; d++) {
                let o = beside(c, d);
                let r = coarseRhoAt(size, o);
                if (r < 0.0) {
                    continue;
                }
                let beta = faceBeta(rho, r);
                sum += beta * coarsePot[u32(o.y) * size.x + u32(o.x)];
                diag += beta;
            }
            if (diag > 0.0) {
                coarsePot[j] = mix(coarsePot[j], (sum - h2 * coarseRhs[j]) / diag, OMEGA);
            }
        }
        workgroupBarrier();
    }
    for (var j = li; j < n; j += 256u) {
        let c = vec2<i32>(i32(j % size.x), i32(j / size.x));
        field[levelSlot(l, c)] = vec4<f32>(coarsePot[j], coarseRhs[j]);
    }
}

// Level l takes on the correction of the coarser level over it.
@compute @workgroup_size(8, 8)
fn prolong(@builtin(global_invocation_id) gid: vec3<u32>) {
    let l = level.index;
    let c = vec2<i32>(gid.xy);
    if (!inLevel(l, c) || activeCount() == 0u || converged()) {
        return;
    }
    let at = levelSlot(l, c);
    if (field[at + 1u].x > 0.0) {
        field[at] = vec4<f32>(field[at].xy + prolonged(l, c), field[at].zw);
    }
}

// The correction for cell c of level l from level l + 1: bilinear between
// the four coarse cells nearest its middle (9, 3, 3 and 1 sixteenths), air
// among them a correction of none, an obstacle leaving its share to the
// rest. (Each fine cell taking its coarse cell's correction whole, with
// the residuals passed down as their mean, the transfers' orders summed to
// 2, too few for this equation: Hemker 1990.)
fn prolonged(l: u32, c: vec2<i32>) -> vec2<f32> {
    let coarse = c / 2;
    let s = vec2<i32>(select(-1, 1, (c.x & 1) == 1), select(-1, 1, (c.y & 1) == 1));
    let offsets = array<vec2<i32>, 4>(vec2<i32>(0), vec2<i32>(s.x, 0), vec2<i32>(0, s.y), s);
    let weights = array<f32, 4>(9.0, 3.0, 3.0, 1.0);
    var sum = vec2<f32>(0.0);
    var weight = 0.0;
    for (var k = 0; k < 4; k++) {
        let at = coarse + offsets[k];
        let r = rhoAt(l + 1u, at);
        if (r < 0.0) {
            continue;
        }
        weight += weights[k];
        if (r > 0.0) {
            sum += weights[k] * potAt(l + 1u, at);
        }
    }
    return sum / max(weight, 1e-6);
}

// The finest level likewise, over the blocks worked on.
@compute @workgroup_size(8, 8)
fn prolongFine(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let c = workCell(wid.x, lid.xy);
    if (!inWorld(c) || converged()) {
        return;
    }
    let i = 2u * cellIndex(c);
    if (field[i].z > 0.0) {
        let e = prolonged(0u, c);
        field[i].x += e.x;
        field[i + 1u].x += e.y;
    }
}

// ---------------------------------------------------- conjugate gradients
//
// One warm-started V-cycle a step left the flow far from divergence free
// wherever the bodies were (REWRITE_PLAN.md P3-18): around a metal block
// sinking in a pool, 0.18 of a cell's volume a step in a cell, and the
// water over it went up and down by turns, -1.0 to +1.3 cells a step; in
// the owner's snapshot of metal hulls in water, 42 % of the moving water
// turned about each step. Repeated, the V-cycle took off half of what was
// left a cycle there, a tenth a cycle beside a wall a coarse cell's
// middle misses (Rafts), and stalled at 6e-3 where a boat's hold meets the
// air (Laden boat): the coarse levels see those as they aren't. So the
// pressure is solved by conjugate gradients with the V-cycle as their
// preconditioner (MGPCG: McAdams, Sifakis & Teran 2010, as the air's,
// air.wgsl), which takes out the few ways the V-cycle misses as well as
// the rest. The operator A is minus the stencil's (stencil): symmetric, a
// face's beta the same from either side (fineBeta), and positive, the
// air's pressure fixed. r is its residual, minus residualAt's; the V-cycle
// solves for z from -r (level 0's pressure its answer, its right-hand
// side), the air's pressure taken as none (cgPrepare) and put back after
// (cgFinish). The V-cycle isn't symmetric (its relaxations alternate
// between the two tilings, and the coarsest level is swept by SOR), so beta
// is the flexible one, A-orthogonal to the last direction (Notay 2000,
// "Flexible conjugate gradients", SIAM J. Sci. Comput. 22(4)).
//
// Per cell in `cg`: the pressure x, r, the direction p and A p (q). After
// the cells, per block by its index, the partial sums of a pass (by its
// index, not its place in the list of blocks worked on, which compact makes
// in no set order: summed in the same order every run); then r . z, alpha,
// beta and p . A p.
@group(0) @binding(15) var<storage, read_write> cg: array<vec4<f32>>;

// What the solve does to each body (pushSlot; js/shared.js's
// BODY_PUSH_WORDS a body).
@group(0) @binding(16) var<storage, read_write> pushes: array<atomic<u32>>;

fn partialSlot(b: vec2<i32>) -> u32 {
    return cellCount() + blockIndex(b);
}

fn cgSums() -> vec4<f32> {
    return cg[cellCount() + blockCount()];
}

// The liquid's regions, each the cells joined through liquid face to face,
// labelled at the solve's start (label.wgsl's union-find; cgJoin): one that
// meets no air holds its pressure by nothing but its walls, and its
// equations have none to give, the same pressure everywhere in it as good
// as any other. Its root is held to none as by a face of air, PIN (the
// rest go by it): without it, where the walls move (grains falling into
// the water, pushing on what they close in) the region's equations had no
// answer, and what the iterations couldn't take off it took them off
// everywhere else: Sand and water's pressure went to 1e14 and its water
// was lost (a cell of water closed in by grains, its stencil none at all,
// the V-cycle's corrections still put into it). The links are G_REGION's
// words, free from clear to rasterClaims (G_TAKEN's), put back by cgFinish;
// sealJoin labels the gaps the liquid has left in them first, and advect
// marks what came into each cell in them after (G_CAME), which scanBlocks
// takes and puts back.
// A region meeting the air has its root's link marked (REGION_OPEN), which
// leaves it a root.
const G_REGION: u32 = G_TAKEN;
const G_CAME: u32 = G_TAKEN;
const CAME_PRESSING: u32 = 1u;
const CAME_FALLING: u32 = 2u;
const REGION_OPEN: u32 = 0x80000000u;
const PIN: f32 = 1.0;

fn ufLink(at: u32) -> u32 {
    return atomicLoad(&counters[gridSlot(at) + G_REGION]);
}

fn ufLower(at: u32, to: u32) -> u32 {
    return atomicMin(&counters[gridSlot(at) + G_REGION], to);
}

// Whether liquid cell i is the root of a region meeting no air.
fn pinnedAt(i: u32) -> bool {
    return ufLink(i) == i;
}

// (A p) at liquid cell c: its stencil on the directions, none in the air.
fn applyA(c: vec2<i32>, rho: f32) -> f32 {
    var sum = 0.0;
    var diag = 0.0;
    let share = shareAt(c);
    for (var d = 0; d < 4; d++) {
        let n = beside(c, d);
        let r = rhoAt(0u, n);
        if (r < 0.0) {
            continue;
        }
        let beta = fineBeta(rho, r, share, shareAt(n), n - c);
        if (r > 0.0) {
            sum += beta * cg[cellIndex(n)].z;
        }
        diag += beta;
    }
    diag += select(0.0, PIN, pinnedAt(cellIndex(c)));
    return diag * cg[cellIndex(c)].z - sum;
}

// Whether this iteration is skipped (converged) or the workgroup has no
// block, the same for the whole workgroup (its barriers).
var<workgroup> cgSkip: u32;
var<workgroup> cgWide: array<vec2<f32>, 256>;

fn cgSkipGroup(wid: u32, li: u32) -> bool {
    if (li == 0u) {
        cgSkip = u32(converged() || wid >= activeCount());
    }
    return workgroupUniformLoad(&cgSkip) != 0u;
}

// A block's partial sums of v, into its slot.
fn sumBlock(wid: u32, li: u32, v: vec2<f32>) {
    cgWide[li] = v;
    workgroupBarrier();
    for (var h = 32u; h > 0u; h >>= 1u) {
        if (li < h) {
            cgWide[li] += cgWide[li + h];
        }
        workgroupBarrier();
    }
    if (li == 0u) {
        let b = partialSlot(activeBlock(wid));
        cg[b] = vec4<f32>(cgWide[0], 0.0, 0.0);
    }
}

// The blocks worked on's partial sums summed, in the order of their index,
// by one workgroup.
fn sumBlocks(li: u32) -> vec2<f32> {
    var s = vec2<f32>(0.0);
    for (var k = li; k < blockCount(); k += 256u) {
        if ((atomicLoad(&blocks[stateSlot(k)]) & ACTIVE) != 0u) {
            s += cg[cellCount() + k].xy;
        }
    }
    cgWide[li] = s;
    workgroupBarrier();
    for (var h = 128u; h > 0u; h >>= 1u) {
        if (li < h) {
            cgWide[li] += cgWide[li + h];
        }
        workgroupBarrier();
    }
    return cgWide[0];
}

// The solve starts: each cell's pressure as it is (last step's, or the air's
// fixed one) into x; each its own region.
@compute @workgroup_size(8, 8)
fn cgStart(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let c = workCell(wid.x, lid.xy);
    if (!inWorld(c)) {
        return;
    }
    let i = cellIndex(c);
    cg[i] = vec4<f32>(gridAt(c).x, 0.0, 0.0, 0.0);
    atomicStore(&counters[gridSlot(i) + G_REGION], i);
}

// Each liquid cell's region joined with the liquid's to its right and below.
@compute @workgroup_size(8, 8)
fn cgJoin(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let c = workCell(wid.x, lid.xy);
    if (!inWorld(c) || gridAt(c).z <= 0.0) {
        return;
    }
    for (var d = 0; d < 2; d++) {
        let n = c + select(vec2<i32>(0, 1), vec2<i32>(1, 0), d == 0);
        if (rhoAt(0u, n) > 0.0) {
            ufJoin(0u, cellIndex(c), cellIndex(n));
        }
    }
}

// A liquid cell beside the air marks its region open.
@compute @workgroup_size(8, 8)
fn cgAnchor(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let c = workCell(wid.x, lid.xy);
    if (!inWorld(c) || gridAt(c).z <= 0.0) {
        return;
    }
    for (var d = 0; d < 4; d++) {
        let n = beside(c, d);
        if (inWorld(n) && rhoAt(0u, n) == 0.0) {
            atomicOr(&counters[gridSlot(ufFind(0u, cellIndex(c))) + G_REGION], REGION_OPEN);
            return;
        }
    }
}

// The liquid's residual r = -b - A x (minus residualAt's).
@compute @workgroup_size(8, 8)
fn cgResidual(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let c = workCell(wid.x, lid.xy);
    if (!inWorld(c)) {
        return;
    }
    if (gridAt(c).z > 0.0) {
        cg[cellIndex(c)].y = -residualAt(0u, c).x - couplingAt(c, PUSH_X);
    }
}

// The V-cycle's right-hand side -r and answer none for the first
// iteration, and the air's pressure none while it solves.
@compute @workgroup_size(8, 8)
fn cgPrepare(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let c = workCell(wid.x, lid.xy);
    if (!inWorld(c)) {
        return;
    }
    let i = cellIndex(c);
    field[2u * i].x = 0.0;
    if (field[2u * i].z > 0.0) {
        field[2u * i].y = -cg[i].y;
    }
}

// The partial sums of r . z, and of z . A p for the last direction (the
// flexible beta's).
@compute @workgroup_size(8, 8)
fn cgRho(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32) {
    if (cgSkipGroup(wid.x, li)) {
        return;
    }
    let c = activeCell(wid.x, lid.xy);
    var v = vec2<f32>(0.0);
    if (inWorld(c) && gridAt(c).z > 0.0) {
        let i = cellIndex(c);
        let z = field[2u * i].x;
        v = vec2<f32>(cg[i].y * z, cg[i].w * z);
    }
    sumBlock(wid.x, li, v);
}

@compute @workgroup_size(256)
fn cgBeta(@builtin(local_invocation_index) li: u32) {
    if (cgSkipGroup(0u, li)) {
        return;
    }
    let s = sumBlocks(li);
    if (li == 0u) {
        let at = cellCount() + blockCount();
        let was = cg[at];
        let beta = select(-s.y / was.w, 0.0, level.cycle == 0u || was.w <= 0.0);
        cg[at] = vec4<f32>(s.x, was.y, beta, was.w);
    }
}

// p = z + beta p (none outside the liquid).
@compute @workgroup_size(8, 8)
fn cgDirection(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let c = workCell(wid.x, lid.xy);
    if (!inWorld(c) || converged()) {
        return;
    }
    let i = cellIndex(c);
    let p = select(0.0, field[2u * i].x + cgSums().z * cg[i].z, field[2u * i].z > 0.0);
    cg[i].z = p;
}

// A p, and the partial sums of p . A p.
@compute @workgroup_size(8, 8)
fn cgApply(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32) {
    if (cgSkipGroup(wid.x, li)) {
        return;
    }
    let c = activeCell(wid.x, lid.xy);
    var v = vec2<f32>(0.0);
    if (inWorld(c)) {
        let rho = gridAt(c).z;
        let i = cellIndex(c);
        let q = select(0.0, applyA(c, rho) + couplingAt(c, PUSH_P0 + level.cycle % 2u), rho > 0.0);
        cg[i].w = q;
        v.x = cg[i].z * q;
    }
    sumBlock(wid.x, li, v);
}

@compute @workgroup_size(256)
fn cgAlpha(@builtin(local_invocation_index) li: u32) {
    if (cgSkipGroup(0u, li)) {
        return;
    }
    let pq = sumBlocks(li).x;
    if (li == 0u) {
        let at = cellCount() + blockCount();
        let was = cg[at];
        cg[at] = vec4<f32>(was.x, select(was.x / pq, 0.0, pq <= 0.0), was.z, pq);
    }
    // (The bodies' sums of this iteration's direction, done with, emptied
    // for the iteration after next.)
    let which = PUSH_P0 + level.cycle % 2u;
    for (var k = li; k < MAX_BODIES * 6u; k += 256u) {
        atomicStore(&pushes[pushSlot(k / 6u, which, 0u) + k % 6u], 0u);
    }
}

// The pressure's push on each free body: of the pressure the solve starts
// from (cgBodyX), of each iteration's direction (cgBodyP), of the pressure
// it ends with (cgBodyF).
@compute @workgroup_size(8, 8)
fn cgBodyX(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let c = workCell(wid.x, lid.xy);
    if (inWorld(c) && gridAt(c).z > 0.0) {
        addPressurePush(c, cg[cellIndex(c)].x, PUSH_X);
    }
}

@compute @workgroup_size(8, 8)
fn cgBodyP(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32) {
    if (cgSkipGroup(wid.x, li)) {
        return;
    }
    let c = activeCell(wid.x, lid.xy);
    if (inWorld(c) && gridAt(c).z > 0.0) {
        addPressurePush(c, cg[cellIndex(c)].z, PUSH_P0 + level.cycle % 2u);
    }
}

@compute @workgroup_size(8, 8)
fn cgBodyF(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let c = workCell(wid.x, lid.xy);
    if (inWorld(c) && gridAt(c).z > 0.0) {
        addPressurePush(c, field[2u * cellIndex(c)].x, PUSH_F);
    }
}

// x += alpha p, r -= alpha A p; the V-cycle's right-hand side and answer for
// the next iteration; the most of r left, and the iterations run counted
// (project takes the most).
var<workgroup> cgMost: atomic<u32>;

@compute @workgroup_size(8, 8)
fn cgUpdate(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32) {
    if (wid.x == 0u && li == 0u) {
        atomicStore(&blocks[countSlot(COUNT_RESIDUAL + (level.cycle + 1u) % 3u)], 0u);
        if (!converged()) {
            atomicAdd(&blocks[countSlot(COUNT_CYCLES)], 1u);
        }
    }
    if (cgSkipGroup(wid.x, li)) {
        return;
    }
    let c = activeCell(wid.x, lid.xy);
    if (inWorld(c) && gridAt(c).z > 0.0) {
        let i = cellIndex(c);
        let alpha = cgSums().y;
        let v = cg[i];
        let r = v.y - alpha * v.w;
        cg[i] = vec4<f32>(v.x + alpha * v.z, r, v.z, v.w);
        field[2u * i].x = 0.0;
        field[2u * i].y = -r;
        // (Non-negative floats order as their bits do.)
        atomicMax(&cgMost, bitcast<u32>(abs(r)));
    }
    workgroupBarrier();
    if (li == 0u) {
        atomicMax(&blocks[countSlot(COUNT_RESIDUAL + level.cycle % 3u)], atomicLoad(&cgMost));
    }
}

// The solve done: the pressure back where the rest of the step (project)
// and next step (classify) read it, the air's too.
@compute @workgroup_size(8, 8)
fn cgFinish(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let c = workCell(wid.x, lid.xy);
    if (!inWorld(c)) {
        return;
    }
    let i = cellIndex(c);
    field[2u * i].x = cg[i].x;
    atomicStore(&counters[gridSlot(i) + G_REGION], 0u);
}

// Pressure differences across the face accelerate the liquid through it, and
// the drift potential's differences give the drift through it.
fn projectFace(at: vec2<i32>, isU: bool) {
    let f = faceSlot(faceIndex(at, isU));
    let a = at - select(vec2<i32>(0, 1), vec2<i32>(1, 0), isU);
    let ga = gridAt(a);
    let gb = gridAt(at);
    var face = field[f];
    let beta = fineBeta(ga.z, gb.z, shareAt(a), shareAt(at), at - a);
    if (beta > 0.0) {
        face.x -= beta * (gb.x - ga.x);
        face.z = PROJECTED;
        // No drift into or out of a body's cells, where no particle goes.
        let rigid = rigidAt(a) != NONE || rigidAt(at) != NONE;
        face.w = select(clamp(-beta * (driftAt(at) - driftAt(a)), -MAX_DRIFT, MAX_DRIFT), 0.0, rigid);
    }
    // A face between the liquid and a free body moves as the body does
    // with the pressure's push the solve found (PUSH_F).
    let ra = rigidAt(a);
    let rb = rigidAt(at);
    if ((ra == NONE) != (rb == NONE) && select(ga.z, gb.z, ra != NONE) > 0.0) {
        let b = select(rb, ra, ra != NONE);
        let ch = pushChange(b, PUSH_F);
        let r = faceMiddle(at, isU) - bodyV(b, BW_POS);
        face.x += select(ch.y + ch.z * r.x, ch.x - ch.z * r.y, isU);
        face.z = PROJECTED;
    }
    field[f] = face;
}

@compute @workgroup_size(8, 8)
fn project(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    // The most iterations any step took since the last report (sim.js's
    // fluidCycles).
    if (wid.x == 0u && all(lid.xy == vec2<u32>(0u))) {
        atomicMax(&blocks[countSlot(COUNT_MOST_CYCLES)], atomicLoad(&blocks[countSlot(COUNT_CYCLES)]));
    }
    let c = workCell(wid.x, lid.xy);
    if (!inWorld(c)) {
        return;
    }
    let fs = facesOf(c);
    for (var j = 0; j < 4; j++) {
        if (fs[j].z >= 0) {
            projectFace(fs[j].xy, fs[j].z == 1);
        }
    }
}

// What the liquid's pressure did to each held body (heldPush), summed for
// the body, and whether it is wet (a free one's push is bodyPush's): each
// block sums its cells of the lowest held body in it in fixed point in
// workgroup memory and adds that once, the rest straight into the body's
// sums (bodies.wgsl's lift takes them).
var<workgroup> rigidFirst: atomic<u32>;
var<workgroup> rigidSums: array<atomic<i32>, 3>;

// The pull's potential at pos (cells), Earth's and the masses', cells²/step²
// (pull.wgsl's altitudeAt): the liquid's pressure at rest goes up by its
// density times the fall in it.
fn potentialAt(pos: vec2<f32>) -> f32 {
    return P.earthG * (f32(P.size.y) - pos.y) + fieldAt(pos).z;
}

// How far along a held body's side the liquid's pressure is looked for where
// it rests on something (seepAt), cells.
const SEEP_REACH: i32 = 128;

// The liquid's pressure at x, on body h's face from its cell c toward n
// where what is there isn't liquid or air (what holds it up, or another
// body): the liquid's round it at that height, which seeps in between (the
// body's weight on what holds it is the contacts', not the water's). `held`
// whether h is held (its cells heldAt's) or free (rigidAt's). Along
// the body's side through its cells and cup to the first cell that isn't:
// for a face up or down, either way along its row, the nearer; for one to
// a side, back through the body. Liquid there gives its pressure, carried
// to the face's height by the liquid's weight (potentialAt); air or nothing,
// none. (With none, a metal block on the bottom of a pool felt the water
// over it and not under it, pushed down at 0.19 cells a step a step where
// Archimedes lifts it at 0.035: test/restcalm.mjs.)
fn seepAt(c: vec2<i32>, n: vec2<i32>, h: u32, held: bool, x: vec2<f32>) -> f32 {
    var ways = array<vec2<i32>, 2>(vec2<i32>(-1, 0), vec2<i32>(1, 0));
    var count = 2;
    if (n.x != 0) {
        ways[0] = -n;
        count = 1;
    }
    var nearest = SEEP_REACH + 1;
    var p = 0.0;
    for (var w = 0; w < count; w++) {
        for (var k = 1; k < nearest; k++) {
            let q = c + ways[w] * k;
            if (select(rigidAt(q), heldAt(q), held) == h) {
                continue;
            }
            let g = gridAt(q);
            if (g.z > 0.0) {
                nearest = k;
                p = g.x + g.z * (potentialAt(vec2<f32>(q) + 0.5) - potentialAt(x));
            }
            break;
        }
    }
    return p;
}

// What the liquid's pressure on held body h's cell c does to it in a step,
// as rigidify sums it (over its mass and inertia): on each face
// to what isn't the body's, the pressure there pushing it in, the cell
// beside's carried half a cell to the face by the liquid's weight, the
// air's for air, and seepAt's against the rest; about its centre of mass.
// (Its cup's air lifts it where the liquid presses on it: a diving bell.)
fn heldPush(c: vec2<i32>, h: u32) -> vec3<i32> {
    let pos = bodyV(h, BW_POS);
    var f = vec2<f32>(0.0);
    var torque = 0.0;
    for (var d = 0; d < 4; d++) {
        let n = beside(c, d) - c;
        let q = c + n;
        if (heldAt(q) == h) {
            continue;
        }
        let g = gridAt(q);
        let x = vec2<f32>(c) + 0.5 + 0.5 * vec2<f32>(n);
        var p = g.x;
        if (g.z > 0.0) {
            p += g.z * (potentialAt(vec2<f32>(q) + 0.5) - potentialAt(x));
        } else if (g.z < 0.0) {
            p = seepAt(c, n, h, true, x);
        }
        let push = -p * vec2<f32>(n);
        f += push;
        let r = x - pos;
        torque += r.x * push.y - r.y * push.x;
    }
    return vec3<i32>(round(vec3<f32>(f / rigidMass(h), torque / rigidInertia(h)) * RIGID_FIX));
}

@compute @workgroup_size(8, 8)
fn rigidify(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    if (li == 0u) {
        atomicStore(&rigidFirst, NONE);
        atomicStore(&rigidSums[0], 0);
        atomicStore(&rigidSums[1], 0);
        atomicStore(&rigidSums[2], 0);
    }
    workgroupBarrier();
    let c = workCell(wid.x, lid.xy);
    // A held body's cells are pushed by the pressure on them (heldPush).
    let h = heldAt(c);
    if (h != NONE) {
        atomicMin(&rigidFirst, h);
    }
    workgroupBarrier();
    let first = atomicLoad(&rigidFirst);
    if (h != NONE) {
        let d = heldPush(c, h);
        if ((atomicLoad(&blocks[stateSlot(blockIndex(c / BLOCK))]) & NEAR_LIQUID) != 0u) {
            atomicOr(&motion[liquidSlot(h, 3u)], LIQUID_WET);
        }
        if (h == first) {
            atomicAdd(&rigidSums[0], d.x);
            atomicAdd(&rigidSums[1], d.y);
            atomicAdd(&rigidSums[2], d.z);
        } else {
            atomicAdd(&motion[liquidSlot(h, 0u)], bitcast<u32>(d.x));
            atomicAdd(&motion[liquidSlot(h, 1u)], bitcast<u32>(d.y));
            atomicAdd(&motion[liquidSlot(h, 2u)], bitcast<u32>(d.z));
        }
    }
    workgroupBarrier();
    if (li == 0u && first != NONE) {
        for (var k = 0u; k < 3u; k++) {
            atomicAdd(&motion[liquidSlot(first, k)], bitcast<u32>(atomicLoad(&rigidSums[k])));
        }
    }
}

// The bodies' sums emptied as the step's liquid starts (bodyStart).
@compute @workgroup_size(256)
fn bodyStart(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < MAX_BODIES * BODY_PUSH_WORDS) {
        atomicStore(&pushes[gid.x], 0u);
    }
}

// The liquid's pressure on a free body where its pose puts its surface,
// past what the cell beside gives at its middle (PUSH_SURFACE): the cell's
// carried there by the liquid's weight (potentialAt). Its stamp is whole
// cells, its pose not: on the stamp's faces alone, its buoyancy went by
// whole rows as it rose or sank, and a plank afloat bobbed between two
// rows, changing over 42 times in 600 steps (REWRITE_PLAN.md P3-22; P3-15
// had it for the rigid fluid). And at a cell's middle, half a cell under
// the face, the pressure pushed a box's floor up half a row too high.
@compute @workgroup_size(8, 8)
fn bodyLoads(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let c = workCell(wid.x, lid.xy);
    if (!inWorld(c)) {
        return;
    }
    let own = rigidAt(c);
    if (own != NONE) {
        seepLoads(c, own);
        return;
    }
    let g = gridAt(c);
    if (g.z <= 0.0) {
        return;
    }
    let mid = vec2<f32>(c) + 0.5;
    for (var d = 0; d < 4; d++) {
        let q = beside(c, d);
        let b = rigidAt(q);
        if (b == NONE) {
            continue;
        }
        let n = q - c;
        let surface = mid + vec2<f32>(n) * (1.0 - bodyReach(b, -n));
        addPush(b, PUSH_SURFACE, bodyFaceJ(c, q, b) * g.z * (potentialAt(mid) - potentialAt(surface)));
    }
}

// The liquid's pressure on free body b's cell c where it rests on or
// against something (another body, a wall, a powder: no liquid between),
// into PUSH_SURFACE: the liquid's round it at that height, as on a held
// body (seepAt, heldPush), as last step's solve left it. (With none, a
// stack of single cells of metal on a pool's floor, 70 rows down, had the
// water over each pressing it down and none under it: to the solve each was
// free to sink at 2.4 cells a step, which the contacts couldn't stop, their
// poses sinking into each other till no contact held them up, and held by
// nothing, they took in the liquid over them as a drain does, a cell's worth
// every two steps packed onto the stack: REWRITE_PLAN.md P3-23.)
fn seepLoads(c: vec2<i32>, b: u32) {
    for (var d = 0; d < 4; d++) {
        let q = beside(c, d);
        if (!inWorld(q) || rigidAt(q) == b || gridAt(q).z >= 0.0) {
            continue;
        }
        let n = q - c;
        let x = vec2<f32>(c) + 0.5 + 0.5 * vec2<f32>(n);
        addPush(b, PUSH_SURFACE, bodyFaceJ(q, c, b) * seepAt(c, n, b, false, x));
    }
}

// The liquid's drag on each free body as the viscosity's sweeps left its
// faces (PUSH_DRAG): on each face of a cell of it to the liquid, the share
// of the way to the next cell's middle the body fills (bodyReach) of a
// cell of its density, times how the face moved from the body's velocity
// (Carlson et al. 2004's rigid motion, for the drag), about its centre.
@compute @workgroup_size(8, 8)
fn bodyDrag(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let c = workCell(wid.x, lid.xy);
    let b = rigidAt(c);
    if (!inWorld(c) || b == NONE) {
        return;
    }
    let pos = bodyV(b, BW_POS);
    let rho = bodyDensity(b);
    var push = vec3<f32>(0.0);
    for (var k = 0; k < 4; k++) {
        let isU = k < 2;
        let n = select(vec2<i32>(0, 1), vec2<i32>(1, 0), isU) * select(-1, 1, (k & 1) == 1);
        if (gridAt(c + n).z <= 0.0) {
            continue;
        }
        let at = c + select(vec2<i32>(0, k & 1), vec2<i32>(k & 1, 0), isU);
        let x = faceMiddle(at, isU);
        let axis = select(1, 0, isU);
        let du = bodyReach(b, n) * rho * (field[faceSlot(faceIndex(at, isU))].x - bodyMoving(b, x, false)[axis]);
        let r = x - pos;
        push[axis] += du;
        push.z += select(r.x * du, -r.y * du, isU);
    }
    if (any(push != vec3<f32>(0.0))) {
        addPush(b, PUSH_DRAG, push);
    }
}

// The bodies' faces, as the sweeps left them, moving with the bodies again,
// the drag and all: what goes into the solve is the body's own motion.
@compute @workgroup_size(8, 8)
fn bodyFaces(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let c = workCell(wid.x, lid.xy);
    if (!inWorld(c)) {
        return;
    }
    let fs = facesOf(c);
    for (var j = 0; j < 4; j++) {
        if (fs[j].z < 0) {
            continue;
        }
        let at = fs[j].xy;
        let isU = fs[j].z == 1;
        let a = at - select(vec2<i32>(0, 1), vec2<i32>(1, 0), isU);
        let ra = rigidAt(a);
        let rb = rigidAt(at);
        if (ra != NONE || rb != NONE) {
            setBodyFace(faceIndex(at, isU), at, isU, gridAt(a), gridAt(at), ra, rb, true);
        }
    }
}

// What the liquid did to each free body this step, for bodies.wgsl (its
// liquidSlot): the pressure at its surface as posed, the drag, and the
// solve's push, as changes in its velocity and spin.
@compute @workgroup_size(256)
fn bodyPush(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if (b >= MAX_BODIES) {
        return;
    }
    let ch = pushChangeFree(b, PUSH_SURFACE) + pushChangeFree(b, PUSH_DRAG) + pushChangeFree(b, PUSH_F);
    if (all(ch == vec3<f32>(0.0))) {
        return;
    }
    let d = vec3<i32>(round(ch * RIGID_FIX));
    atomicAdd(&motion[liquidSlot(b, 0u)], bitcast<u32>(d.x));
    atomicAdd(&motion[liquidSlot(b, 1u)], bitcast<u32>(d.y));
    atomicAdd(&motion[liquidSlot(b, 2u)], bitcast<u32>(d.z));
}

// The velocity the projection left, carried out into the air beside the
// liquid (Bridson 2015, ch. 4): a face that holds the splat's velocity but
// wasn't projected (both its cells air) takes the mean of those along the
// same axis around it that were (layer 1: their z PROJECTED) or that the
// first layer reached (layer 2), so a particle at the surface samples the
// projected flow, not what it splatted before the solve.
fn extrapolateFace(at: vec2<i32>, isU: bool, layer: u32) {
    let f = faceSlot(faceIndex(at, isU));
    var face = field[f];
    let a = at - select(vec2<i32>(0, 1), vec2<i32>(1, 0), isU);
    if (face.z != 1.0 || gridAt(a).z != 0.0 || gridAt(at).z != 0.0) {
        return;
    }
    let reached = select(PROJECTED, 1.5, layer == 2u);
    var sum = 0.0;
    var n = 0.0;
    for (var d = 0; d < 4; d++) {
        let q = beside(at, d);
        if (!faceInWorld(q, isU)) {
            continue;
        }
        let g = field[faceSlot(faceIndex(q, isU))];
        if (g.z >= reached) {
            sum += g.x;
            n += 1.0;
        }
    }
    if (n > 0.0) {
        face.x = sum / n;
        face.z = select(1.5, 1.25, layer == 2u);
        field[f] = face;
    }
}

fn extrapolateCell(wid: u32, lid: vec2<u32>, layer: u32) {
    let c = workCell(wid, lid);
    if (!inWorld(c)) {
        return;
    }
    let fs = facesOf(c);
    for (var j = 0; j < 4; j++) {
        if (fs[j].z >= 0) {
            extrapolateFace(fs[j].xy, fs[j].z == 1, layer);
        }
    }
}

@compute @workgroup_size(8, 8)
fn extrapolate(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    extrapolateCell(wid.x, lid.xy, 1u);
}

@compute @workgroup_size(8, 8)
fn extrapolateFurther(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    extrapolateCell(wid.x, lid.xy, 2u);
}

// The viscosity of a face between cells of grid ga and gb, neither a
// body's: the mean of the liquid's on its sides; none against an obstacle.
fn faceViscosity(ga: vec4<f32>, gb: vec4<f32>) -> f32 {
    if (ga.z < 0.0 || gb.z < 0.0) {
        return 0.0;
    }
    let fa = select(0.0, 1.0, ga.z > 0.0);
    let fb = select(0.0, 1.0, gb.z > 0.0);
    return (fa * ga.w + fb * gb.w) / max(fa + fb, 1.0);
}

// Whether viscosity's sweeps solve such a face.
fn isViscousFace(ga: vec4<f32>, gb: vec4<f32>) -> bool {
    return faceViscosity(ga, gb) > 0.0;
}

// Last step's pressure push across such a face (classify starts the
// pressure from last step's): taken off before the sweeps (setFace) and put
// back after them (unpress), so they hold back the pressure's push in the
// step it is made, as they do gravity's, and the projection then adds only
// how the pressure changed (incremental pressure correction: Goda 1979, van
// Kan 1986). Pushed on after the sweeps, as the whole of it was, the push
// went through them untouched each step: mud ran down a channel at
// Poiseuille's speed plus a step of its pressure's push, 2.6 times as fast,
// and a still column of it in a closed channel turned over for ever, down
// its middle and up its sides (REWRITE_PLAN.md 7-11).
fn lastPush(a: vec2<i32>, at: vec2<i32>, ga: vec4<f32>, gb: vec4<f32>) -> f32 {
    if ((ga.z > 0.0 && !wasLiquid(a)) || (gb.z > 0.0 && !wasLiquid(at))) {
        return 0.0;
    }
    return fineBeta(ga.z, gb.z, shareAt(a), shareAt(at), at - a) * (gb.x - ga.x);
}

// Whether the liquid cell c was liquid last step too, so classify started
// its pressure from last step's (a cell an obstacle, a bubble or air left
// starts from none: across its faces "last step's push" would be the whole
// of its neighbours' pressure, deep in the liquid, and a bubble rising
// through oil kicked the oil round it upward at 0.7 cells a step).
fn wasLiquid(c: vec2<i32>) -> bool {
    return (u32(field[2u * cellIndex(c) + 1u].w) & stepBit(1u)) != 0u;
}

// One red-black Gauss-Seidel sweep of implicit viscosity, over faces of one
// color: a face between viscous liquid moves toward the faces beside it
// along the same axis, each by the viscosity between the two (a flux, the
// same both ways): along the axis the cell's they share; across it, at
// their corner, the two faces' harmonic mean, as a viscous stress passes
// from one liquid into another, so a thin liquid barely drags a thick one
// along (water, its viscosity none here, not at all: a drop of oil in it
// rises through it and isn't held to it; by the faces' mean, as it was, a
// face of oil beside water was held to it by half oil's viscosity, and once
// the pressure's push went through the sweeps, 7-11, oil drops hung in the
// Oil and water tank: REWRITE_PLAN.md 7-12). From where the forces left
// it, before the projection (REWRITE_PLAN.md P2-3; Batty &
// Bridson 2008's order), which then takes out what it leaves diverging:
// after it, the viscous flow the particles moved along wasn't divergence
// free. Obstacles beside it, the world's edge included, hold it back (no
// slip); open air doesn't (left out). The face across an obstacle along the
// axis lies on the obstacle's surface, so is still there; one beside it
// across the flow lies half a cell into the obstacle, past its surface, so
// stands for a ghost of the face's own velocity turned back, which stills
// it at the surface: counted twice (REWRITE_PLAN.md 7-7: counted once, as
// still where it lay, a channel took half a cell more on each side, and
// oil ran down one 4 cells wide at twice Poiseuille's speed). The sweeps
// start from the forces' velocities plus what they did to the face last
// step (viscLast): in a steady flow that is already where they settle, and
// a step's few sweeps, which only creep across a thick liquid's smooth
// flows, need settle only how the flow changed. (From the forces'
// velocities alone, mud ran down a channel 20 cells wide five times as fast
// as the sweeps' own equations have it: REWRITE_PLAN.md 7-11.)
fn viscousFace(at: vec2<i32>, isU: bool, color: i32) {
    if (((at.x + at.y) & 1) != color) {
        return;
    }
    let a = at - select(vec2<i32>(0, 1), vec2<i32>(1, 0), isU);
    let ga = gridAt(a);
    let gb = gridAt(at);
    let ra = rigidAt(a);
    let rb = rigidAt(at);
    // A body's face with liquid beyond it (its viscosity the liquid's,
    // setBodyFace's) is solved too, and what the sweeps do to it the body
    // takes (bodyDrag): the liquid's drag on it, which the liquid it drags
    // along gives up. The face holds the body's mass on its side and the
    // liquid's on the other (Carlson, Mucha & Turk 2004's rigid fluid, for
    // the drag), so a sweep moves it by the liquid's viscosity over their
    // mean density. Between two of a body's cells, or a body's and an
    // obstacle's or air, it isn't.
    let surface = (ra != NONE) != (rb != NONE) && select(ga.z, gb.z, ra != NONE) > 0.0;
    if (!surface && (ga.z < 0.0 || gb.z < 0.0 || ra != NONE || rb != NONE)) {
        return;
    }
    let f = faceSlot(faceIndex(at, isU));
    let nu = field[f].w;
    if (nu <= 0.0) {
        return;
    }
    var heavy = 1.0;
    if (surface) {
        let liquid = select(ga.z, gb.z, ra != NONE);
        heavy = 2.0 * liquid / (liquid + bodyDensity(select(rb, ra, ra != NONE)));
    }
    // Through cell a and cell b, the faces across them along the axis.
    let through = vec2<f32>(select(0.0, ga.w, ga.z > 0.0), select(0.0, gb.w, gb.z > 0.0));
    var sum = 0.0;
    var n = 0.0;
    for (var d = 0; d < 4; d++) {
        let o = beside(at, d);
        // (beside: d < 2 along x; across the flow a u face's neighbours
        // above and below, a v face's to the sides.)
        let across = isU == (d >= 2);
        var other = vec4<f32>(0.0, 0.0, -1.0, 0.0);
        if (faceInWorld(o, isU)) {
            other = field[faceSlot(faceIndex(o, isU))];
        }
        var k = select(through.x, through.y, (d & 1) == 1);
        if (across) {
            k = select(2.0 * nu * other.w / max(nu + other.w, 1e-12), nu, other.z < 0.0);
        }
        if (other.z < 0.0) {
            n += k * select(1.0, 2.0, across);
        } else if (other.z > 0.0) {
            sum += k * other.x;
            n += k;
        }
    }
    field[f].x = mix(field[f].x, (viscRhs(faceIndex(at, isU)) + heavy * sum) / (1.0 + heavy * n), VISC_OMEGA);
}

var<workgroup> viscousNear: u32;

// Only blocks with viscous liquid in or next to them need a sweep (color 0
// or 1) or unpress (2).
fn viscosity(wid: u32, lid: vec2<u32>, color: i32) {
    if (all(lid == vec2<u32>(0u))) {
        viscousNear = 0u;
        if (wid < activeCount()) {
            let b = activeBlock(wid);
            viscousNear = statesOf(b - 1, b + 1) & HAS_VISCOUS;
        }
    }
    if (workgroupUniformLoad(&viscousNear) == 0u) {
        return;
    }
    let c = activeCell(wid, lid);
    if (!inWorld(c)) {
        return;
    }
    let fs = facesOf(c);
    for (var j = 0; j < 4; j++) {
        if (fs[j].z < 0) {
            continue;
        }
        if (color < 2) {
            viscousFace(fs[j].xy, fs[j].z == 1, color);
            continue;
        }
        // unpress: what the sweeps did, and the pressure's push back on.
        let at = fs[j].xy;
        let isU = fs[j].z == 1;
        let a = at - select(vec2<i32>(0, 1), vec2<i32>(1, 0), isU);
        let ga = gridAt(a);
        let gb = gridAt(at);
        let fi = faceIndex(at, isU);
        if (rigidAt(a) == NONE && rigidAt(at) == NONE && isViscousFace(ga, gb)) {
            let f = faceSlot(fi);
            setViscLast(fi, field[f].x - viscRhs(fi));
            field[f].x += lastPush(a, at, ga, gb);
        } else if (viscLast(fi) != 0.0) {
            setViscLast(fi, 0.0);
        }
    }
}

// After the sweeps (whether they ran or not), on the faces they solve: what
// they did to each, for the next step's to start from (none kept on the
// rest), and last step's pressure push back on (lastPush).
@compute @workgroup_size(8, 8)
fn unpress(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    viscosity(wid.x, lid.xy, 2);
}

@compute @workgroup_size(8, 8)
fn viscosityRed(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    viscosity(wid.x, lid.xy, 0);
}

@compute @workgroup_size(8, 8)
fn viscosityBlack(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    viscosity(wid.x, lid.xy, 1);
}

// ---------------------------------------------------------------- advect

struct Sample {
    now: f32,
    before: f32,
    weight: f32,  // of the faces that hold a velocity; 0 if none does
    drift: f32,
    gradient: vec2<f32>,  // of the velocity now, where the projection set all four faces
}

// The faces around s, a position in the axis's face grid: velocity now and
// before this step's forces, from those that hold one, and drift.
fn sampleAxis(s: vec2<f32>, isU: bool) -> Sample {
    let b = vec2<i32>(floor(s));
    let f = s - floor(s);
    var acc = vec2<f32>(0.0);
    var r: Sample;
    var held = 0;
    var gradient = vec2<f32>(0.0);
    for (var j = 0; j < 4; j++) {
        let o = vec2<i32>(j & 1, j >> 1);
        let w = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1);
        // How w changes along x and y.
        let dw = vec2<f32>(select(-1.0, 1.0, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1),
            select(1.0 - f.x, f.x, o.x == 1) * select(-1.0, 1.0, o.y == 1));
        let at = b + o;
        if (!faceInWorld(at, isU)) {
            continue;
        }
        let face = field[faceSlot(faceIndex(at, isU))];
        r.drift += w * face.w;
        if (face.z > 0.0) {
            acc += w * face.xy;
            r.weight += w;
            gradient += dw * face.x;
            held += select(0, 1, face.z == PROJECTED);
        }
    }
    if (r.weight > 0.0) {
        r.now = acc.x / r.weight;
        r.before = acc.y / r.weight;
    }
    r.gradient = select(vec2<f32>(0.0), gradient, held == 4);
    return r;
}

// How far along each axis a buried particle looks for a way out.
const ESCAPE_REACH: i32 = 16;

// The body whose cup cell i was in as stamped the step before last (the
// swept words keep each step's for the next: common.wgsl's
// bodySweptSlot), NONE if none: the side of a body's wall a particle the
// wall has since been stamped onto was on.
fn cupBefore(i: u32) -> u32 {
    let key = bodyWords[bodySweptSlot(P.size, i, P.frame - 2u)];
    if ((key >> 12u) != claimStep(P.frame - 2u) || (key & 1u) != 0u) {
        return NONE;
    }
    let b = 2047u - ((key >> 1u) & 0x7ffu);
    return select(NONE, b, b < MAX_BODIES);
}

// Whether a particle buried at c, in body `side`'s cup before it was buried
// (or NONE), may step out into the cell at q: not an obstacle's, nor a
// body's (shutAt), nor in a body's cup it wasn't in. (A cup is space to the
// liquid since P3-22: the first cell clear of a sealed hull's wall was its
// hold, and in a pressed and tipping lens hull 37 particles went in.)
fn escapesTo(q: vec2<i32>, side: u32) -> bool {
    if (!inWorld(q) || shutAt(q)) {
        return false;
    }
    let cup = cupOf(cellIndex(q));
    return cup == NONE || cup == side;
}

// A particle that something has since filled in around (a powder landing on
// it, the rest of its cell turned to ice, a body's wall) steps out into the
// nearest cell around it that isn't an obstacle, nor a body's, on the side
// of a body's wall it was on (escapesTo). Buried deeper (in a bed of gravel that
// settled over the water it sank through, under a body), it moves a cell a
// step toward the nearest way out along the four axes, up (away from the
// pull) first, or up with none within ESCAPE_REACH; where liquid can't be,
// it isn't drawn on its way. Looking only around it, 87 particles of a pool
// stayed in the gravel heaped into it.
fn escape(pos: vec2<f32>) -> vec2<f32> {
    if (!isObstacle(pos)) {
        return pos;
    }
    let c = vec2<i32>(floor(pos));
    let side = select(NONE, cupBefore(cellIndex(c)), inWorld(c));
    var best = pos;
    var bestD = 1e9;
    for (var n = 0; n < 8; n++) {
        let m = n + select(0, 1, n >= 4);
        let at = c + vec2<i32>(m % 3 - 1, m / 3 - 1);
        if (!escapesTo(at, side)) {
            continue;
        }
        let inside = clamp(pos, vec2<f32>(at) + 0.02, vec2<f32>(at) + 0.98);
        let d = distance(pos, inside);
        if (d < bestD) {
            bestD = d;
            best = inside;
        }
    }
    if (bestD < 1e9) {
        return best;
    }
    let ways = array<vec2<i32>, 4>(vec2<i32>(0, -1), vec2<i32>(-1, 0), vec2<i32>(1, 0), vec2<i32>(0, 1));
    let k = downTurns(c);
    var way = turned(ways[0], k);
    var nearest = ESCAPE_REACH + 1;
    for (var n = 0; n < 4; n++) {
        let dir = turned(ways[n], k);
        for (var r = 2; r < nearest; r++) {
            // (Nor into a body's cup it wasn't in, as the ring round it
            // isn't: up out of a hull's floor two cells thick the first cell
            // clear was its hold, air, and the water went in.)
            if (!isObstacle(vec2<f32>(c + dir * r) + 0.5) && escapesTo(c + dir * r, side)) {
                nearest = r;
                way = dir;
                break;
            }
        }
    }
    return select(pos, pos + vec2<f32>(way), inWorld(c + way));
}

// Cells of liquid per step a particle slips through the others in its cell
// against the pull (along it if denser), at g, by how much lighter it is
// than their mean: the slip of a drift-flux model (Manninen, Taivassalo &
// Kallio 1996) between liquids that don't mix, so oil in water rises
// through it, and water in oil sinks, cell or no cell. (Until 7-4, cells of
// the denser one over the lighter traded places in the block pass,
// settleLiquids.)
const SLIP: f32 = 1.0;

fn slipOf(q: Particle) -> vec2<f32> {
    let i = cellIndex(cellOf(q.pos));
    let n = gridWord(i, G_COUNT);
    if (n < 2u) {
        return vec2<f32>(0.0);
    }
    let mean = f32(gridWord(i, G_MASS)) / (FIX * f32(n));
    let lighter = (mean - liquidDensity(q.word, q.temp)) / max(mean, 1e-3);
    return -pullAt(q.pos) / elems[WATER].weight * SLIP * lighter;
}

@compute @workgroup_size(256)
fn advect(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let k = particleSlot(gid, groups);
    if (k == NONE) {
        return;
    }
    var q = loadParticle(k);
    let e = elems[wordType(q.word)];
    q.pos = escape(q.pos);
    let u = sampleAxis(q.pos - vec2<f32>(0.0, 0.5), true);
    let v = sampleAxis(q.pos - vec2<f32>(0.5, 0.0), false);
    let held = vec2<bool>(u.weight > 0.0, v.weight > 0.0);
    let flow = select(q.vel, vec2<f32>(u.now, v.now), held);
    // APIC: the grid's velocity and its gradient, the gradient kept at
    // AFFINE_KEEP (FLIP's change in the grid's velocity, with a little of
    // PIC's to damp its noise, kept still water stirring after a few hundred
    // steps, REWRITE_PLAN.md P2-2).
    var vel = clampSpeed(flow);
    let affine = AFFINE_KEEP * vec4<f32>(u.gradient, v.gradient);

    // Along the grid's velocity where it starts, plus the drift, in steps of
    // at most half a cell so nothing is skipped over. (Midpoint advection,
    // tried in P2, wasn't kept; its velocity was still sampled, two of the
    // grid's samples a particle for nothing.)
    let travel = clampSpeed(vel) + clamp(vec2<f32>(u.drift, v.drift) + slipOf(q), vec2<f32>(-MAX_DRIFT), vec2<f32>(MAX_DRIFT));
    // (At most ADVECT_STEPS, what the clamps allow: a travel not finite,
    // which nothing should make, converted to an i32 of 2^31 and hung the
    // GPU on the loop.)
    let steps = clamp(i32(ceil(max(abs(travel.x), abs(travel.y)) * 2.0)), 1, ADVECT_STEPS);
    var d = travel / f32(steps);
    var pos = q.pos;
    var impact = 0.0;
    for (var s = 0; s < steps; s++) {
        let nx = pos + vec2<f32>(d.x, 0.0);
        if (d.x != 0.0 && isObstacle(nx)) {
            if (vel.x * d.x > 0.0) {
                impact = max(impact, abs(vel.x));
                vel.x = 0.0;
            }
            d.x = 0.0;
        } else {
            pos = nx;
        }
        let ny = pos + vec2<f32>(0.0, d.y);
        if (d.y != 0.0 && isObstacle(ny)) {
            if (vel.y * d.y > 0.0) {
                impact = max(impact, abs(vel.y));
                vel.y = 0.0;
            }
            d.y = 0.0;
        } else {
            pos = ny;
        }
    }
    // It stays in the blocks worked on, which the sort and the raster go
    // over (a particle moves at most a block a step, LIQUID_MAX_SPEED, but
    // escaping as well it could get past them).
    if (!isActiveBlock(cellOf(pos) / BLOCK)) {
        pos = q.pos;
        if (!isActiveBlock(cellOf(pos) / BLOCK)) {
            pos = loadParticle(k).pos;
        }
    }
    parts.items[k] = bitcast<vec4<u32>>(vec4<f32>(pos, vel));
    parts.items[2u * capacity() + k] = bitcast<vec4<u32>>(affine);
    // Whether it was falling free, its cell's pressure as the solve left it
    // under half a cell of its weight, marked in the cell it comes to
    // (G_CAME, for scanBlocks).
    let left = field[2u * cellIndex(cellOf(q.pos))];
    if (left.z > 0.0) {
        let falling = left.x < 0.5 * left.z * length(pullAt(q.pos));
        atomicOr(&counters[gridSlot(cellIndex(cellOf(pos))) + G_CAME], select(CAME_PRESSING, CAME_FALLING, falling));
    }
    // A hard knock sets it off (in the update pass, as a flame does:
    // block.wgsl's light).
    if ((e.flags & F_IMPACT) != 0u && impact > IMPACT_SPEED) {
        parts.items[capacity() + k].x = wordWith(q.word, FL_BURN);
    }
}

// ------------------------------------------------------------------ sort

// Each particle counted in the cell it is in, its place among them kept (in
// its state's w) for scatter.
@compute @workgroup_size(256)
fn sortCount(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let k = particleSlot(gid, groups);
    if (k == NONE) {
        return;
    }
    let pos = bitcast<vec4<f32>>(parts.items[k]).xy;
    parts.items[capacity() + k].w = atomicAdd(&counters[gridSlot(cellIndex(cellOf(pos))) + G_SORT], 1u);
}

var<workgroup> scanTile: array<u32, 256>;

// Inclusive prefix sums over scanTile's first n (64 or 256) in place, all
// threads of the workgroup taking part.
fn scanInPlace(li: u32, n: u32) {
    for (var o = 1u; o < n; o <<= 1u) {
        var add = 0u;
        if (li < n && li >= o) {
            add = scanTile[li - o];
        }
        workgroupBarrier();
        if (li < n) {
            scanTile[li] += add;
        }
        workgroupBarrier();
    }
}

// A workgroup per block worked on: where each of its cells' runs starts
// among the block's, and how many particles it holds in all.
@compute @workgroup_size(8, 8)
fn scanBlocks(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32) {
    let c = workCell(wid.x, lid.xy);
    var n = 0u;
    if (inWorld(c)) {
        n = gridWord(cellIndex(c), G_SORT);
    }
    scanTile[li] = n;
    workgroupBarrier();
    scanInPlace(li, 64u);
    if (inWorld(c)) {
        let i = cellIndex(c);
        atomicStore(&counters[gridSlot(i) + G_START], scanTile[li] - n);
        // (And, the first pass over the cells after advect, whether all the
        // liquid that came into the cell was falling free, into its bits.)
        if (atomicExchange(&counters[gridSlot(i) + G_CAME], 0u) == CAME_FALLING) {
            field[2u * i + 1u].w = f32(u32(field[2u * i + 1u].w) | fallingBit(0u));
        }
    }
    if (li == 63u && wid.x < activeCount()) {
        atomicStore(&blocks[totalSlot(wid.x)], scanTile[63]);
    }
}

// One workgroup: where each block's runs start, in the order of the blocks
// worked on, and how many particles there are.
@compute @workgroup_size(256)
fn scanTotals(@builtin(local_invocation_index) li: u32) {
    let n = activeCount();
    let per = (n + 255u) / 256u;
    let lo = min(li * per, n);
    let hi = min(lo + per, n);
    var sum = 0u;
    for (var j = lo; j < hi; j++) {
        sum += atomicLoad(&blocks[totalSlot(j)]);
    }
    scanTile[li] = sum;
    workgroupBarrier();
    scanInPlace(li, 256u);
    var run = scanTile[li] - sum;
    for (var j = lo; j < hi; j++) {
        let t = atomicLoad(&blocks[totalSlot(j)]);
        atomicStore(&blocks[totalSlot(j)], run);
        run += t;
    }
    if (li == 255u) {
        atomicStore(&sorted.count, scanTile[255]);
        atomicStore(&blocks[countSlot(COUNT_PARTICLES)], scanTile[255]);
    }
}

// Each cell's run's start, the block's added.
@compute @workgroup_size(8, 8)
fn scanFinish(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let c = workCell(wid.x, lid.xy);
    if (!inWorld(c)) {
        return;
    }
    atomicAdd(&counters[gridSlot(cellIndex(c)) + G_START], atomicLoad(&blocks[totalSlot(wid.x)]));
}

// Each particle into its place in its cell's run in `sorted`.
@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let k = particleSlot(gid, groups);
    if (k == NONE) {
        return;
    }
    let motion = parts.items[k];
    let state = parts.items[capacity() + k];
    let i = cellIndex(cellOf(bitcast<vec4<f32>>(motion).xy));
    let m = gridWord(i, G_START) + state.w;
    if (m < capacity()) {
        sorted.items[m] = motion;
        sorted.items[capacity() + m] = state;
        sorted.items[2u * capacity() + m] = parts.items[2u * capacity() + k];
    }
}

// ------------------------------------------------------------------ draw

// Whether the liquid drawn in cell j last step has left it, no particle
// being in it now: free for the gas the liquid pushes out of its way.
fn leftBehind(j: u32) -> bool {
    let u = cells[j];
    return elems[cellType(u)].kind == KIND_LIQUID && gridWord(j, G_SORT) == 0u && gridWord(j, G_TAKEN) == 0u;
}

// Eighths of a turn from the Earth's down to the down at c (pull.wgsl), and
// an offset given for the Earth's down turned that far: "up first" is then
// away from the pull.
fn downTurns(c: vec2<i32>) -> i32 {
    let at = clamp(c, vec2<i32>(0), vec2<i32>(P.size) - 1);
    return i32(stepIndex(downAt(at))) - 2;
}

fn turned(o: vec2<i32>, k: i32) -> vec2<i32> {
    return turnStep(o, k);
}

// The gas plane (common.wgsl): cell i's pair, as stored.
fn gasPair(i: u32) -> vec2<u32> {
    let v = cells[gasIndex(cellCount(), i)];
    return select(v.xy, v.zw, gasHigh(i));
}

fn setGasPair(i: u32, w: vec2<u32>) {
    let j = gasIndex(cellCount(), i);
    if (gasHigh(i)) {
        cells[j].z = w.x;
        cells[j].w = w.y;
    } else {
        cells[j].x = w.x;
        cells[j].y = w.y;
    }
}

// Gas the liquid takes the place of and has nowhere to put (moveOutOfTheWay) goes
// into the air beside it, all it held and its heat (airsource.wgsl; the
// air's carry takes it in where there is gas beside the cell, and it waits
// for some where there is none), and the cell holds none: in a sealed room
// its gas is squeezed, the room's pressure rising. (Until 4c it was pooled
// for the whole world, its moisture given back to the air the liquid left
// anywhere: a bubble's water came up through the next drop.)
fn pushGasOut(i: u32, temp: f32) {
    let w = gasPair(i);
    let m = unpack2x16float(w.x);
    let r = unpack2x16float(w.y);
    pushGas(i, Gas(m.x, m.y, r.x, r.y), temp);
    setGasPair(i, vec2<u32>(0u));
}

// Gas moved from cell i into cell j: what it holds of the air goes with it,
// and i, liquid now, holds none.
fn moveGas(i: u32, j: u32) {
    setGasPair(j, gasPair(i));
    setGasPair(i, vec2<u32>(0u));
}

// Gas the liquid comes into goes into a cell beside it the liquid has left
// (which unpaint would otherwise turn into fresh air), or, for gas other than
// air, into air there too, up first: so the air keeps its oxygen and
// moisture (common.wgsl) as liquid moves through it, and a drop falling
// through humid air doesn't dry it out on its way. In two passes, so that
// which cell gets which doesn't hang on which thread came first (the
// determinism of REWRITE_PLAN.md D-1): rasterClaims has each such cell
// choose its cell and claim it (G_AIR_TO; G_TAKEN, the lowest cell
// claiming winning), and raster moves the winners' gas.

// The claim cell i makes (G_TAKEN): the lowest wins (atomicMax of this);
// never 0, which is none.
fn takenBy(i: u32) -> u32 {
    return ~i;
}

// Whether the liquid drawn in cell j last step has left it, none of its
// particles in it now (and no claim on it yet counted).
fn vacated(j: u32) -> bool {
    return elems[cellType(cells[j])].kind == KIND_LIQUID && gridWord(j, G_SORT) == 0u;
}

// The cell air (gas, with `gas`) liquid comes into at c goes to: a cell
// beside it the liquid has left, or, for other gas, air with no liquid
// coming into it too; up first. NONE for none. (A gap in the liquid, counted
// as liquid by classify, isn't moved: its cell is left for other particles,
// which keeps the liquid from opening up; what it holds goes into the air
// beside it, raster.)
fn airTo(c: vec2<i32>, gas: bool) -> u32 {
    if (!gas && gridAt(c).z > 0.0) {
        return NONE;
    }
    // Air: up, the sides, down, the diagonals. Other gas: up, the upper
    // diagonals, the sides, the lower diagonals, down.
    let airOrder = array<vec2<i32>, 8>(
        vec2<i32>(0, -1), vec2<i32>(-1, 0), vec2<i32>(1, 0), vec2<i32>(0, 1),
        vec2<i32>(-1, -1), vec2<i32>(1, -1), vec2<i32>(-1, 1), vec2<i32>(1, 1),
    );
    let gasOrder = array<vec2<i32>, 8>(
        vec2<i32>(0, -1), vec2<i32>(-1, -1), vec2<i32>(1, -1), vec2<i32>(-1, 0),
        vec2<i32>(1, 0), vec2<i32>(-1, 1), vec2<i32>(1, 1), vec2<i32>(0, 1),
    );
    let k = downTurns(c);
    for (var n = 0; n < 8; n++) {
        let at = c + turned(select(airOrder[n], gasOrder[n], gas), k);
        if (!inWorld(at)) {
            continue;
        }
        let j = cellIndex(at);
        if (vacated(j) || (gas && cellType(cells[j]) == AIR && gridWord(j, G_SORT) == 0u && !isOccupant(occupants[j]))) {
            return j;
        }
    }
    return NONE;
}

// Over the blocks worked on, a thread a cell: each cell of gas liquid comes
// into (but a bubble's) claims where its gas is to go (airTo).
@compute @workgroup_size(8, 8)
fn rasterClaims(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let c = workCell(wid.x, lid.xy);
    if (!inWorld(c)) {
        return;
    }
    let i = cellIndex(c);
    var to = NONE;
    let t = cellType(cells[i]);
    if (gridWord(i, G_SORT) > 0u && !isOccupant(occupants[i])) {
        if (t == AIR) {
            to = airTo(c, false);
        } else if (elems[t].kind == KIND_GAS && gridAt(c).z >= 0.0) {
            to = airTo(c, true);
        }
    }
    atomicStore(&counters[gridSlot(i) + G_AIR_TO], to);
    if (to != NONE) {
        atomicMax(&counters[gridSlot(to) + G_TAKEN], takenBy(i));
    }
}

// The gas in cell i moved out of the liquid's way as rasterClaims claimed,
// if its claim won; air with nowhere to go goes into the air beside it
// (pushGasOut). Whether it went somewhere (other gas staying where it is,
// and the liquid not drawn there).
fn moveOutOfTheWay(i: u32, v: vec4<u32>) -> bool {
    let to = gridWord(i, G_AIR_TO);
    if (to != NONE && gridWord(to, G_TAKEN) == takenBy(i)) {
        if (cellType(cells[to]) == AIR) {
            pushGasOut(to, wordTempOf(cells[to]));
        }
        cells[to] = v;
        moveGas(i, to);
        return true;
    }
    if (cellType(v) == AIR) {
        // (Lost where there was none, it drained a dam break's room of a
        // fifth of its vapour in 300 steps, and a room of fog raining onto
        // its floor of 14 % of its water in a minute, test/phasewater.mjs,
        // test/vapourtrack.mjs; now it waits for gas beside it.)
        pushGasOut(i, wordTempOf(v));
        return true;
    }
    return false;
}

// A workgroup per block worked on, a thread per cell: a cell with particles
// in it where liquid can be (air or other gas, which it pushes out of its
// way, or liquid) is drawn as the most numerous liquid among them, with the
// word of the first of those, their temperature (by heat capacity) and mean
// velocity, and in w the cell and how many (common.wgsl's liquidHome). Next
// step its particles follow it (syncParticles). Where liquid can't be (a
// powder or solid has come into the cell, a bubble), they stay as they are,
// not drawn (DEST_KEEP), until they find a way out (advect's escape).
@compute @workgroup_size(8, 8)
fn raster(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let c = workCell(wid.x, lid.xy);
    if (!inWorld(c)) {
        return;
    }
    let i = cellIndex(c);
    let n = gridWord(i, G_SORT);
    if (n == 0u) {
        return;
    }
    let start = gridWord(i, G_START);
    let v = cells[i];
    let t = cellType(v);
    // A person's body: not drawn, the particles left to find their way out
    // (DEST_KEEP), and liquid drawn there before the person came is air (as
    // unpaint leaves a cell the liquid has left).
    if (isOccupant(occupants[i])) {
        if (elems[t].kind == KIND_LIQUID) {
            cells[i] = vec4<u32>(AIR, bitcast<u32>(wordTempOf(v)), 0u, 0u);
            setGasPair(i, vec2<u32>(0u));
        }
        atomicStore(&counters[destSlot(i)], DEST_KEEP);
        return;
    }
    var drawn = false;
    if (elems[t].kind == KIND_LIQUID) {
        drawn = true;
    } else if (t == AIR || (elems[t].kind == KIND_GAS && gridAt(c).z >= 0.0)) {
        drawn = moveOutOfTheWay(i, v);
    }
    if (!drawn) {
        atomicStore(&counters[destSlot(i)], DEST_KEEP);
        return;
    }
    // What it is drawn as comes out the same whatever order the sort put its
    // particles in, which goes by the order threads counted them
    // (REWRITE_PLAN.md D-1): the most numerous liquid (counted for the
    // RASTER_KINDS lowest types there, ties to the lowest); its word that of
    // the particle of it with the lowest id, but for its life (a solute's
    // strength, what of a fuel has burnt), the mean of theirs that have the
    // same aux (the same solute), rounded at random so that on average none
    // is made or lost; its heat, heat capacity and velocity summed exactly
    // (common.wgsl's exactParts). (Taken in id order instead, the walk was a
    // cell's particles squared, on one thread: a person standing in a flood
    // crowds them to 36 a cell, and the step took a third longer.)
    var kinds = array<u32, RASTER_KINDS>(NONE, NONE, NONE, NONE);
    var counts = array<u32, RASTER_KINDS>(0u, 0u, 0u, 0u);
    for (var k = start; k < start + n; k++) {
        let pt = wordType(sorted.items[capacity() + k].x);
        var at = RASTER_KINDS;
        for (var m = 0u; m < RASTER_KINDS; m++) {
            if (kinds[m] == pt) {
                at = m;
            }
        }
        if (at == RASTER_KINDS && pt < kinds[RASTER_KINDS - 1u]) {
            // In among the lowest kept, in order, the highest let go (the
            // lowest types there all are counted whole whatever the order).
            at = RASTER_KINDS - 1u;
            kinds[at] = pt;
            counts[at] = 0u;
            while (at > 0u && kinds[at - 1u] > kinds[at]) {
                let t0 = kinds[at - 1u];
                let c0 = counts[at - 1u];
                kinds[at - 1u] = kinds[at];
                counts[at - 1u] = counts[at];
                kinds[at] = t0;
                counts[at] = c0;
                at--;
            }
        }
        if (at < RASTER_KINDS) {
            counts[at]++;
        }
    }
    var most = kinds[0];
    var votes = counts[0];
    for (var m = 1u; m < RASTER_KINDS; m++) {
        if (counts[m] > votes) {
            most = kinds[m];
            votes = counts[m];
        }
    }
    var word = 0u;
    var lowest = NONE;
    // Water's solute: that of its particle of the lowest id carrying one,
    // the rest of the water fresh or of it (a second solute keeps its own).
    var solute = 0u;
    var soluteId = NONE;
    let solutes = holdsSolute(most);
    for (var k = start; k < start + n; k++) {
        let state = sorted.items[capacity() + k];
        if (wordType(state.x) == most && (lowest == NONE || state.z < lowest)) {
            word = state.x;
            lowest = state.z;
        }
        if (solutes && wordType(state.x) == most && wordAux(state.x) != 0u && (soluteId == NONE || state.z < soluteId)) {
            solute = wordAux(state.x);
            soluteId = state.z;
        }
    }
    if (solutes) {
        word = packWord(wordType(word), solute, wordLife(word), wordShade(word), wordFlags(word));
    }
    var lives = vec2<f32>(0.0);
    var heat = vec2<u32>(0u);
    var capacityOf = vec2<u32>(0u);
    var velX = vec2<u32>(0u);
    var velY = vec2<u32>(0u);
    for (var k = start; k < start + n; k++) {
        let state = sorted.items[capacity() + k];
        let pt = wordType(state.x);
        // (Whole numbers, which an f32 sums exactly.) Water carrying the
        // solute, and fresh water, which carries none of it: the cell's
        // strength is the mean of all its water's, so a pass moving solute
        // in or out of it moves it in or out of all of it (syncParticles).
        if (pt == most && wordAux(state.x) == wordAux(word)) {
            lives += vec2<f32>(f32(wordLife(state.x)), 1.0);
        } else if (pt == most && solutes && wordAux(state.x) == 0u) {
            lives += vec2<f32>(0.0, 1.0);
        }
        let hc = max(elems[pt].heatCap, 1e-6);
        heat = addExact(heat, hc * bitcast<f32>(state.y));
        capacityOf = addExact(capacityOf, hc);
        let u = bitcast<vec4<f32>>(sorted.items[k]).zw;
        velX = addExact(velX, u.x);
        velY = addExact(velY, u.y);
    }
    seedRng(i, P.frame, P.seed ^ 0x7a57u);
    let life = min(u32(floor(lives.x / lives.y + rnd())), 255u);
    word = packWord(wordType(word), wordAux(word), life, wordShade(word), wordFlags(word));
    let temp = exactValue(heat, EXACT_FINE) / exactValue(capacityOf, EXACT_FINE);
    let vel = vec2<f32>(exactValue(velX, EXACT_FINE), exactValue(velY, EXACT_FINE)) / f32(n);
    cells[i] = vec4<u32>(wordWith(word, FL_DRAWN), bitcast<u32>(temp), pack2x16float(vel), liquidW(i, n));
    atomicStore(&counters[gridSlot(i) + G_TYPE], drawnAs(word) | (life << 16u));
}


// The raster counts the RASTER_KINDS lowest liquid types in a cell.
const RASTER_KINDS: u32 = 4u;

// x added to the exact sum w (common.wgsl's exactParts, EXACT_FINE), in a
// thread's own words.
fn addExact(w: vec2<u32>, x: f32) -> vec2<u32> {
    let p = exactParts(x, EXACT_FINE);
    let lo = w.x + p.x;
    return vec2<u32>(lo, w.y + p.y + select(0u, 1u, lo < p.x));
}

// What G_TYPE keeps of a drawn cell's word: its type and aux (its solute),
// and (in its third byte) its life as drawn. Next step its particles of that
// type and aux take the cell's word, as the passes left it; water's take
// the solute the passes moved (sharedLife); the rest of a mixed cell keep
// their own.
fn drawnAs(word: u32) -> u32 {
    return wordType(word) | (wordAux(word) << 8u);
}

// The life a particle of water (life, of the cell's solute or fresh) takes
// from its cell drawn at `was` and left at `now` by the passes: all the
// cell's water moving alike, each the share of the way to none, or to
// saturation, that the cell's mean moved, so their differences stay (a
// cell of brine and fresh water stays so until it mixes), none goes past
// either end, and the cell's water holds what the passes left it. Rounded
// at random, so none is made or lost on average.
fn sharedLife(life: u32, was: u32, now: u32, id: u32) -> u32 {
    if (now == was) {
        return life;
    }
    var l = f32(life);
    if (now < was) {
        l = l * f32(now) / f32(was);
    } else {
        l = 255.0 - (255.0 - l) * (255.0 - f32(now)) / (255.0 - f32(was));
    }
    seedRng(id, P.frame, P.seed ^ 0x5a17u);
    return min(u32(floor(l + rnd())), 255u);
}

@compute @workgroup_size(8, 8)
fn unpaint(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let c = workCell(wid.x, lid.xy);
    if (!inWorld(c)) {
        return;
    }
    let i = cellIndex(c);
    let v = cells[i];
    if (leftBehind(i)) {
        // The air liquid moved into was moved out of its way (moveOutOfTheWay),
        // into cells like this one where it could, into the air beside it
        // where it couldn't; so the cell left here holds nothing, a vacuum,
        // which the air around fills (air.wgsl's expansion). At the
        // temperature the liquid acted at, not its stored one (which counts
        // latent heat: water freezing, lava setting); holding nothing, it
        // takes that of what fills it.
        cells[i] = vec4<u32>(AIR, bitcast<u32>(wordTempOf(v)), 0u, 0u);
        setGasPair(i, vec2<u32>(0u));
    }
}
