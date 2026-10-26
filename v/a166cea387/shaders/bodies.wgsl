// Free bodies: pieces of structural solid that have come loose (update.wgsl)
// move as rigid bodies instead of crumbling: they fall, sink, float and
// drift, tumble, topple, roll and capsize, push each other and stack, and
// come to rest on whatever stops them.
//
// A body is kept from the step it comes loose until it comes to rest, in a
// slot of a table of bodies: its pose (where its centre of mass is, and its
// angle), its velocity and spin, its mass and moment of inertia, and where
// its shape is. Its shape is its cells, with their full state, in a grid of
// its own (body space), in a pool all bodies share. Its cells in the world
// are marked as its own (FL_LOOSE, with its slot and piece in frac; see
// common.wgsl) and carry their velocity.
//
// Each step, each body's cells are read back from the world into its shape
// (lift: whatever happened to them meanwhile, heat, fire, anything that
// destroyed them), its motion is worked out (solve), and it is stamped back
// into the world at its new pose (stamp), one to one: each of its cells into
// a world cell of its own, by a whole number of quarter turns and three
// shears (stampCell), so the world holds exactly its cells at any angle,
// however often it turns. What it moves into trades places with the cells
// it leaves (the exchange), so nothing is made or lost. A body that has come
// to rest on something that stands still rejoins the world as ordinary
// cells, squared up to the grid if it nearly is, held up by support again.
//
// Its motion: gravity at its centre of mass; the liquid's, through the
// liquid's own pressure solve, which takes its cells in as liquid of its
// density moving with it (fluid.wgsl's rigid cells, REWRITE_PLAN.md P3-3),
// so the liquid floats it, drags it, carries it along and turns it (a boat
// rights itself or capsizes, as its metacentre lies), and it moves the
// liquid out of its way; the air's pressure and drag on its faces that meet
// gas; and contact impulses where its
// cells touch, or would run into, what stands still or another body, with
// friction, at the contact, so it topples off an edge and rolls down a slope,
// and two bodies share each impulse between them, equal and opposite
// (findContact, jointSolve).
//
// Each piece of structure has a piece id (frac.y of its cells, 0 for what
// was painted): pieces only come loose together, and only form one body,
// with the cells of the same piece. A body takes a new piece id, so a body
// that has come to rest on another piece stays a piece of its own.
//
// Each step, the passes:
//
//   plan     the list of bodies to work on, and of the 8 x 8 tiles with
//            loose cells in no body in or next to them (the update pass
//            marks and lists those with such cells in them); with none
//            marked as of the last readback, sim.js runs listBodies instead,
//            and leaves out the passes up to lift
//   init     labels each loose cell in no body with the lowest one of the
//            same piece it is connected to inside its tile (union-find in
//            workgroup memory)
//   unite    joins the labels of such neighbors across the tiles' edges
//            (label.wgsl's union-find), so all of a piece's cells lead to
//            one root, its lowest cell index
//   gather   each cell takes its root as its label and adds itself to the
//            piece's sums there: its bounding box, how long it has been
//            loose, how many cells
//   found    each piece that has been loose long enough goes on a list
//   foundBodies  one workgroup: in the order of their roots, each listed
//            piece gets a body: a slot in the table, room for its shape in
//            the pool, a new piece id
//   capture  its cells are copied into its shape and marked as its own
//   lift     each body's cells are read back into its shape; its mass and
//            centre of mass, if they changed; the air's push on its faces
//            that meet gas; and what the liquid's pressure did to its
//            motion this step (fluid.wgsl's rigidify)
//   parts    a body that has lost cells is filled from its first one: if
//            that doesn't reach them all it has come apart, and it breaks
//            up into loose cells (as a piece that came loose), whose
//            labelling next step finds the pieces
//   contact  its velocity and spin from gravity, the air and the liquid, and the
//            contacts of its cells with what stands still and with other
//            bodies, into one list for all of them
//   jointSolve  one workgroup: the impulses at all the contacts together,
//            each equal and opposite on the two bodies it is between
//   integrate  its new pose, backed off if it would still overlap
//            something; whether it has come to rest; and it claims the
//            cells its new pose covers
//   stamp    the cells it no longer covers are left, and the cells it
//            covers take its shape's cells; what was in them (gas, powder
//            that wasn't resting) goes into the cells it left, matched
//            along the way it moves (the exchange); liquid goes round
//            behind it if that is open liquid, else aside; a hard landing
//            breaks the cells that hit, and what they hit
//   compact  (plan, copy) only when the pool is full: the bodies' shapes
//            are packed into its other half
//   finish   empties the lists for next step
//
// and, in the update phase after the update pass (sim.js), structure's
// anchoring (8-1; doing nothing unless the structure changed):
//   structureWatch, structureInit, structureUnite, structureAnchor,
//   structureSettle  standing structure labelled by connectivity, and what
//            no anchor holds let go (see structureWatch)
//   structureDone  one workgroup: clears the structure's changed flag
//
// The passes over the tiles worked on are dispatched a fixed number of
// workgroups, which go through the list a tile at a time; those over the
// bodies likewise go through the list a body at a time, each body's cells
// handled by one workgroup together (workgroup barriers between the stages
// of each pass). Dispatching a workgroup per tile or per body there could
// be, most returning at once, costs more than the work.
//
// Two bodies can't both stamp the same cell: the cells a body's new pose
// covers are claimed first (claimKey: the lower slot wins, a cell claimed by
// another body this step isn't left by the body it was), and the cells
// displaced matter goes to are claimed the same way. The claim words are
// kept from step to step, so a body can tell at the next lift which cells it
// was stamped into.
//
// Units are cells and steps; masses are in cells of water (rho / 1000),
// forces are the liquid's pressure (fluid.wgsl) over faces a cell wide.

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> elems: array<Elem>;
@group(0) @binding(2) var<storage, read_write> cells: array<vec4<u32>>;
// Per tile its mark and whether it is worked on; the stats read back each
// frame (BODY_STAT_*, js/shared.js); the lists of the marked tiles and of those worked on; counters
// (C_*); the pieces ripe to become bodies (foundingSlot); then per cell its
// label and its claim word; then per root the piece's sums, SUM_WORDS each;
// structure's words (common.wgsl's structureBase), and per cell which body
// last moved over it (bodySweptSlot).
@group(0) @binding(4) var<storage, read_write> bodies: array<atomic<u32>>;

// The liquid's counters (fluid.wgsl's, LIQUID_CELL_WORDS a cell): where the
// particles a cell was drawn from go next step (common.wgsl's G_DEST),
// which sendLiquid sets.
@group(0) @binding(5) var<storage, read_write> liquid: array<atomic<u32>>;
// What the passes read over and over, as plain words (atomic loads aren't
// cached, and cost many times more): the list of bodies to work on this
// step (listSlot); the table of bodies, TABLE_WORDS a slot (tableSlot); this
// step's contacts (contactSlot); and what the liquid did to each body
// (fluid.wgsl's rigidify, liquidSlot). Which body last moved over a cell is
// in the bodies buffer (bodySweptSlot).
@group(0) @binding(7) var<storage, read_write> motion: array<u32>;
// The shapes: each body's cells in a grid of its own, the cell's x word and
// temperature (an x word of 0, air, is no cell). Two halves of a cell per
// world cell each; bodies take room in one, bump-allocated, and when none
// is left those still about are packed into the other.
@group(0) @binding(8) var<storage, read_write> shapes: array<vec2<u32>>;
// The pull of the world's masses per light cell of 2 x 2 cells, cells/step²
// (gravity.wgsl).
@group(0) @binding(9) var gravity: texture_2d<f32>;
// What goes into the air (airsource.wgsl): gas a body takes the place of and
// leaves no room for (dropGas), and the drag it gives the air beside it.
@group(0) @binding(10) var<storage, read_write> source: array<atomic<u32>>;
// The air as its pass left it (air.wgsl; airsample.wgsl reads it): the
// pressure and wind on a body's faces that meet gas.
@group(0) @binding(11) var<storage, read> air: array<vec4<f32>>;

const NONE: u32 = 0xffffffffu;

// The way the total pull goes at p (cells), Earth's and the world's masses'
// (pull.wgsl's pullAt, which a body takes at its centre of mass, and lift
// how it varies across it); what a body rests on or carries is told by it.
fn pullDirAt(p: vec2<f32>) -> vec2<f32> {
    let pull = pullAt(p);
    return select(vec2<f32>(0.0, 1.0), normalize(pull), dot(pull, pull) > 1e-12);
}
// MAX_BODIES (js/shared.js) slots in the table; FREE_BODY (common.wgsl) is past them.
const_assert f32(MAX_BODIES) < FREE_BODY;
const MAX_PIECE: u32 = 2046u;    // piece ids 1..MAX_PIECE: f16 holds whole numbers up to 2048
const MAX_MOVE: f32 = 4.0;       // cells per step and axis, as for powders (MAX_SPEED)
const RIPEN_STEPS: f32 = 6.0;    // steps all of a piece must have been loose before it lets go
// Landing (REWRITE_PLAN.md P3-8). Hitting something that stands still at
// speed v (m/s), a body puts a stress of v Za Zb / (Za + Zb) into the two at
// the contact, Z = √(E ρ) each one's acoustic impedance (the flat impact of
// two elastic solids: what stores ½ ρ v² as σ² / 2E): the cells that hit
// break where that passes their impact strength (elements.js impact), and
// brittle ones (glass, ice) shatter right through, with the hardest blow
// any of its cells took. It breaks what it lands on by the same measure
// (another body doesn't count: it gives way). A wall is rigid; a powder at
// rest gives as a granular bed (POWDER_MODULUS). So glass breaks off a fall
// onto stone that brings it to 6.6 m/s, and not onto wood: at a centimetre
// a cell, where bodies fall at 2.4 m/s at most (MAX_MOVE), nothing does.
const SHARD_SPEED: f32 = 0.5;     // fragments of what a body breaks start out falling
const SHATTER_REACH: f32 = 1.5;   // cells a brittle solid hit shatters around the spot, per sqrt(stress / impact strength)
const SHATTER_MAX: f32 = 6.0;
const POWDER_MODULUS: f32 = 5e7;  // Pa: a settled granular bed's stiffness, about sand's
const RIGID_IMPEDANCE: f32 = 1e15;
// Contacts. A body's cells are taken as round, half a cell across, at the
// middles of its shape's cells (not of the cells they are stamped into).
// They keep CONTACT_DIST from the middle of a cell that stands still, along
// the contact's normal, which keeps them out of its cell at any angle, and
// BODY_DIST from another body's (their real middles too). Contacts are
// speculative (Paul Firth, "Speculative contacts: a continuous collision
// engine approach", 2011): a gap may close this step, no more, so nothing
// is caught a step late. All bodies' contacts are solved together
// (jointSolve), each body's mass split evenly between its own (Tonge,
// Benevolenskiy & Voroshilov 2012, "Mass splitting for jitter-free parallel
// rigid body simulation", ACM TOG 31(4)), SOLVE_ITERATIONS rounds, with
// Coulomb friction FRICTION; a cell found overlapping is pushed out by
// PUSH_OUT of the overlap a step. A pose that would leave at most
// OVERLAP_SLACK cells overlapping is taken (they aren't stamped for the
// step).
const CONTACT_DIST: f32 = 0.75;
const BODY_DIST: f32 = 1.0;
const OVERLAP_SLACK: u32 = 3u;
const FRICTION: f32 = 0.6;
const PUSH_OUT: f32 = 0.3;
const SOLVE_ITERATIONS: u32 = 8u;
const CONTACT_CAP: u32 = 256u;    // contacts a body keeps in workgroup memory, for its landing (all go into jointSolve's list, up to BODY_CONTACTS)
// At rest: going less than REST_SPEED a step for REST_STEPS steps on end
// (its cells too, as it turns), on something that stands still and not
// afloat, a body rejoins the world: squared up to the grid if it is within
// SNAP_ANGLE of it and nothing is in the way, else as it lies.
const REST_SPEED: f32 = 0.05;
const REST_STEPS: u32 = 8u;
const SNAP_ANGLE: f32 = 0.1;
// A body turns at most MAX_SPIN radians a step, and its cells move at most
// MAX_MOVE cells a step as it turns.
const MAX_SPIN: f32 = 0.25;
const QUARTER_TURN: f32 = 1.5707963;
const TAU: f32 = 6.2831853;
// Displaced powder is pushed on ahead through at most this many cells of
// loose material to the first free one.
const PUSH_REACH: i32 = 8;

// The words read back by sim.js after the tiles' marks and states,
// BODY_STAT_* (js/shared.js's BODY_STAT): tiles marked by the update pass
// so far; tiles worked on this step; tiles marked as of the last step the
// passes ran; bodies in the table; whether the pool is full (1 to be
// packed, 2 being packed this step).
//
// The counters, C_* (js/shared.js's BODY_COUNTER): bodies on this step's
// list; pieces listed to become bodies (found); cells of the pool's half in
// use taken; piece ids handed out; the half of the pool in use. Then tallies, for testing: liquid a
// body's cell was left out for, having nowhere to go (C_KEPT); powder and
// gas let go for the same reason (C_LOST_POWDER, C_LOST_GAS).
const COUNTERS: u32 = BODY_COUNTERS;
const_assert C_LOST_GAS < COUNTERS;

// A piece's sums, SUM_WORDS per root.
const S_BOX: u32 = 0u;     // its bounding box: lowest x, y, then highest
const S_RIPE: u32 = 4u;    // steps its youngest loose cell has been loose
const S_CELLS: u32 = 5u;   // cells
const S_SLOT: u32 = 6u;    // the body it became, if it did
const SUM_WORDS: u32 = BODY_SUM_WORDS;
const_assert S_SLOT < SUM_WORDS;

// The table of bodies, TABLE_WORDS per slot (f32 bits unless noted).
const T_POS: u32 = BW_POS;  // x, y: centre of mass in the world, as stamped
const T_ANGLE: u32 = 2u;
const T_VEL: u32 = BW_VEL;  // x, y
const T_SPIN: u32 = BW_SPIN; // angular velocity, radians per step
const T_MASS: u32 = BW_MASS; // in cells of water
const T_INERTIA: u32 = BW_INERTIA; // about the centre of mass
const T_COM: u32 = 8u;      // x, y: centre of mass in the shape's grid
const T_RADIUS: u32 = 10u;  // farthest the shape's corners are from it
const T_SHAPE: u32 = 11u;   // u32: its first cell in the pool
const T_SIZE: u32 = 12u;    // u32: the shape's grid, width | height << 16
const T_CELLS: u32 = BW_CELLS;  // u32: cells in the shape
const T_BOX: u32 = 14u;     // i32 x 4: the world cells it was stamped over, lowest x, y, highest x, y
const T_FLAGS: u32 = BW_FLAGS; // u32: B_*
const T_FRAME: u32 = BW_FRAME; // u32: the step it was last stamped
const T_PIECE: u32 = 20u;   // u32
const T_REST: u32 = 21u;    // u32: steps on end it has been at rest
const T_FORCE: u32 = 22u;   // from lift: the air's push x, y, its torque and the cargo's weight's
const T_SUBCELL: u32 = BW_SUBCELL;  // u32, from stamp: Stamp's offset, as two f16 (pack2x16float)
const T_LIQUID: u32 = 26u;  // from lift: what the liquid did to its velocity x, y and spin this step, its cells in the liquid's solve
const T_NEXT: u32 = 30u;    // from solve: the new pose, x, y, angle
const T_PACKED: u32 = 30u;  // u32: where packing the pool moves its shape to (T_NEXT's word, free by then)
const T_NEXTVEL: u32 = 33u; // x, y, spin
const T_NEWBOX: u32 = 36u;  // i32 x 4
const T_IMPACT: u32 = 40u;  // the hardest landing this step: approach speed squared, the normal x, y
const T_STAMP: u32 = 43u;   // how it was last stamped (Stamp): anchor x, y (i32), quarter turns (u32), shears (f32 x 2)
const T_PIVOT: u32 = 48u;   // i32 x 2: the shape cell it turns about as stamped
const T_MOVED_N: u32 = 50u; // u32: its cells last stamped elsewhere (see MOVED_CAP)
const T_MOVED: u32 = 51u;   // u32 x 2 each: the shape cell, the world cell it went to
const T_PUSH: u32 = 99u;    // from jointSolve: the push out of overlaps, x, y, spin
const T_SUPPORT: u32 = 102u; // u32, from jointSolve: resting on something that stands still (T_SUPPORT_STILL), held (T_SUPPORT_HELD), resting on it or a body that is (T_SUPPORT_RESTING)
const T_SUPPORT_STILL: u32 = 1u;
const T_SUPPORT_HELD: u32 = 2u;
const T_SUPPORT_RESTING: u32 = 4u;
const T_COUNT: u32 = 103u;  // u32, from contact: its contacts this step; jointSolve's: the rows it is in
const T_CARGO: u32 = BW_CARGO;    // from lift: the mass of the powder resting on it (it carries it: massOf)
const T_CARGO_INERTIA: u32 = BW_CARGO_INERTIA;  // and its moment of inertia about the centre of mass
const T_MOBILITY: u32 = BW_MOBILITY;  // from integrate: how it gives way to a push as its contacts let it, xx, xy, xs, yy, ys, ss (workMobility)
const T_MOBILITY_AT: u32 = BW_MOBILITY_AT;  // u32: the step that was worked out
const TABLE_WORDS: u32 = BODY_TABLE_WORDS;
const_assert T_MOVED + 2u * MOVED_CAP <= T_PUSH;
const_assert T_MOBILITY_AT < TABLE_WORDS;

const B_ALIVE: u32 = BF_ALIVE;
const B_NEW: u32 = BF_NEW;    // captured last step: its mass and motion are still to be worked out
const B_REJOIN: u32 = 4u;     // stamped this step as ordinary cells, and gone
const B_PARTS: u32 = BF_PARTS; // lost cells this step (lift): the parts pass looks whether it is still in one piece
const B_WET: u32 = BF_WET;    // held, a cell of it in a block by the liquid in its last solve: the next takes in all its cells (fluid.wgsl's wetRegionAt)
const B_SUPPORTED: u32 = 16u; // resting on something that stands still
const B_HELD: u32 = BF_HELD;  // resting on that, or on a body that is, two steps (jointSolve): a wall to the liquid's solve, not in it (fluid.wgsl's isHeld)
const B_RESTED: u32 = 128u;   // resting so this step (jointSolve), as B_HELD goes by

// Contact kinds (obstacleKind).
const OB_NONE: u32 = 0u;
const OB_STATIC: u32 = 1u;
const OB_BODY: u32 = 2u;

const BODY_THREADS: u32 = 256u;
// A cell its stamp puts where something stands still (the stamp is a cell
// or so off the pose the contacts keep clear of what it rests on), or that
// another body or something that can't be moved holds, goes into the
// nearest cell of gas within SEEK_REACH instead, and the body keeps where:
// up to MOVED_CAP of them a step, the rest left out of the world for the
// step (still in its shape).
const MOVED_CAP: u32 = 24u;
const CARGO_REACH: i32 = 32;  // cells up a column of powder on a body it carries
// A cell of a body's cup: one with none of its cells in it (AIR), between
// two of them in its row and over one in its column, in the shape's grid,
// as a boat's hull holds the air inside it and a crate's walls the air shut
// in them (weigh). The liquid's solve takes it as the body's (fluid.wgsl's
// rigid cells), which would otherwise push the floor of a hull alone up
// into the air over it. Marked in the shape's second word.
const CUP: u32 = 1u;
// A body carries what is in its cup (REWRITE_PLAN.md P3-5): the powder in
// each cell of it, as last stamped, goes where that cell is stamped now, as
// its own cells go (integrate claims the cells for it; stampBody's
// cargoTake, cargoRoom, cargoPut). Each thread keeps what its shape cells
// carry, CARGO_EACH at most (the rest stays where it is for the step).
const CARGO_EACH: u32 = 4u;
const SEEK_REACH: i32 = 4;
// The parts pass's bitmaps (two, of the shape's cells and of those reached
// from its first), in words: a shape's grid of more cells than they hold
// (49152, its rows padded to whole words) breaks up whenever it loses
// cells, as it may have come apart.
const PARTS_WORDS: u32 = 1536u;

// ------------------------------------------------------------ addressing

fn cellCount() -> u32 {
    return P.size.x * P.size.y;
}

fn inWorld(c: vec2<i32>) -> bool {
    return all(c >= vec2<i32>(0)) && all(c < vec2<i32>(P.size));
}

fn cellIndex(c: vec2<i32>) -> u32 {
    return u32(c.y) * P.size.x + u32(c.x);
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

// What cell i holds of the air going with it to cell j, i holding none.
fn moveGas(i: u32, j: u32) {
    setGasPair(j, gasPair(i));
    setGasPair(i, vec2<u32>(0u));
}

// The gas in cell i, at p, with nowhere to go: put into the air beside it,
// all it holds and its heat (airsource.wgsl), and i holds none.
fn dropGas(p: vec2<i32>, i: u32) {
    pushGas(i, gasFrom(cells[gasIndex(cellCount(), i)], i), cellTemp(cells[i]));
    setGasPair(i, vec2<u32>(0u));
}

fn tileCount() -> u32 {
    return bodyTileCount(P.size);
}

fn activeSlot(n: u32) -> u32 {
    return tileCount() + n;
}

fn statSlot(k: u32) -> u32 {
    return bodyMarkedSlot(P.size) + k;
}

fn activeListSlot(k: u32) -> u32 {
    return bodyMarkListSlot(P.size) + tileCount() + k;
}

fn counterSlot(k: u32) -> u32 {
    return activeListSlot(tileCount()) + k;
}

// The pieces ripe to become bodies this step (found), C_FOUNDING of them.
fn foundingSlot(k: u32) -> u32 {
    return counterSlot(COUNTERS) + k;
}

fn labelSlot(i: u32) -> u32 {
    return foundingSlot(MAX_BODIES) + i;
}

fn claimSlot(i: u32) -> u32 {
    return labelSlot(cellCount()) + i;
}

fn sumSlot(r: u32, k: u32) -> u32 {
    return claimSlot(cellCount()) + r * SUM_WORDS + k;
}

fn sum(r: u32, k: u32) -> u32 {
    return atomicLoad(&bodies[sumSlot(r, k)]);
}

// The swept word's low bit: a body's seam, not its cup (stampBody); and how
// far apart two of its cells next to each other in its shape are looked
// at for one.
const SWEPT_SEAM: u32 = 1u;
const SEAM_REACH: i32 = 3;

// Body b moves over cell i this step (common.wgsl's bodySweptSlot).
fn sweep(i: u32, b: u32) {
    atomicMax(&bodies[bodySweptSlot(P.size, i, P.frame)], bodySweptKey(P.frame, b));
}

// Whether body b was the one kept as moving over cell i last step.
fn sweptLast(i: u32, b: u32) -> bool {
    return atomicLoad(&bodies[bodySweptSlot(P.size, i, P.frame - 1u)]) == bodySweptKey(P.frame - 1u, b);
}

// In motion.
fn listSlot(k: u32) -> u32 {
    return k;
}

fn tableSlot(b: u32, k: u32) -> u32 {
    return MAX_BODIES + b * TABLE_WORDS + k;
}

// Word w of contact k in this step's list (jointSolve), after the table.
fn contactSlot(k: u32, w: u32) -> u32 {
    return MAX_BODIES * (1u + TABLE_WORDS) + k * BODY_CONTACT_WORDS + w;
}

// Word w of what the liquid did to body b this step, after the contacts
// (fluid.wgsl's rigidify, its liquidSlot): its change in velocity x, y and
// spin, in fixed point (common.wgsl's RIGID_FIX), and its cells in the
// liquid's solve.
fn liquidSlot(b: u32, w: u32) -> u32 {
    return contactSlot(BODY_CONTACTS, 0u) + b * BODY_LIQUID_WORDS + w;
}

// Body b's mass and moment of inertia with what it carries (T_CARGO): the
// powder resting on it moves with it (REWRITE_PLAN.md P3-4).
fn massOf(b: u32) -> f32 {
    return tableF(b, T_MASS) + tableF(b, T_CARGO);
}

fn inertiaOf(b: u32) -> f32 {
    return tableF(b, T_INERTIA) + tableF(b, T_CARGO_INERTIA);
}

fn tableF(b: u32, k: u32) -> f32 {
    return bitcast<f32>(motion[tableSlot(b, k)]);
}

fn tableV(b: u32, k: u32) -> vec2<f32> {
    return vec2<f32>(tableF(b, k), tableF(b, k + 1u));
}

fn tableU(b: u32, k: u32) -> u32 {
    return motion[tableSlot(b, k)];
}

fn tableI(b: u32, k: u32) -> vec2<i32> {
    return vec2<i32>(bitcast<i32>(motion[tableSlot(b, k)]), bitcast<i32>(motion[tableSlot(b, k + 1u)]));
}

fn setF(b: u32, k: u32, v: f32) {
    motion[tableSlot(b, k)] = bitcast<u32>(v);
}

fn setV(b: u32, k: u32, v: vec2<f32>) {
    setF(b, k, v.x);
    setF(b, k + 1u, v.y);
}

fn setU(b: u32, k: u32, v: u32) {
    motion[tableSlot(b, k)] = v;
}

fn setI(b: u32, k: u32, v: vec2<i32>) {
    motion[tableSlot(b, k)] = bitcast<u32>(v.x);
    motion[tableSlot(b, k + 1u)] = bitcast<u32>(v.y);
}

// ---------------------------------------------------------------- claims
//
// A claim word is the step (common.wgsl's claimStep, never 0; sim.js
// clears the words when it wraps) over a priority, so the latest step's claims always win, and
// within a step the higher priority: a body's is 2047 less its slot, so the
// lower slot wins; matter displaced to a cell claims it at priority 0, which
// any body's claim beats, and a cell a body leaves is held at 1 for its
// exchange (claimLeft), and for what goes round behind it (claimBehind).
// The lowest bit marks a cell a body claimed but couldn't stamp. Claims are
// made with atomicMax.

const FILL_PRIORITY: u32 = 0u;
const LEFT_PRIORITY: u32 = 1u;

fn claimKey(frame: u32, priority: u32) -> u32 {
    return (claimStep(frame) << 12u) | (priority << 1u);
}

fn bodyKey(frame: u32, b: u32) -> u32 {
    return claimKey(frame, 2047u - b);
}

fn claimOf(i: u32) -> u32 {
    return atomicLoad(&bodies[claimSlot(i)]);
}

// Whether a body claimed cell i this step.
fn claimedByBody(k: u32) -> bool {
    return (k >> 12u) == claimStep(P.frame) && ((k >> 1u) & 0x7ffu) > LEFT_PRIORITY;
}

// Holds cell i, which a body has just left, for its exchange.
fn claimLeft(i: u32) {
    atomicMax(&bodies[claimSlot(i)], claimKey(P.frame, LEFT_PRIORITY));
}

// Whether cell i is still held for the exchange.
fn heldLeft(i: u32) -> bool {
    return claimOf(i) == claimKey(P.frame, LEFT_PRIORITY);
}

// Claims free cell i for matter displaced into it; false if a body or other
// matter has it this step, or it is held for an exchange.
fn claimFill(i: u32) -> bool {
    let key = claimKey(P.frame, FILL_PRIORITY);
    return atomicMax(&bodies[claimSlot(i)], key) < key;
}

// The same for what goes round behind a body, which may also take a cell
// held for its exchange (one less for that).
fn claimBehind(i: u32) -> bool {
    let held = atomicCompareExchangeWeak(&bodies[claimSlot(i)], claimKey(P.frame, LEFT_PRIORITY), claimKey(P.frame, FILL_PRIORITY));
    return held.exchanged || claimFill(i);
}

// ----------------------------------------------------------------- cells

fn kindOf(v: vec4<u32>) -> u32 {
    return elems[cellType(v)].kind;
}

// isStructural (common.wgsl) for what is in v, reading only the two fields
// it needs: these are read for every cell a body looks at, and copying out
// an Elem whole costs many times more.
fn structural(v: vec4<u32>) -> bool {
    let t = cellType(v);
    return elems[t].kind == KIND_SOLID && elems[t].breakTo != NO_ELEM;
}

// Whether v is one of body b's cells.
fn isBodyCell(v: vec4<u32>, b: u32) -> bool {
    return cellHas(v, FL_LOOSE) && cellFrac(v).x == f32(b) && structural(v);
}

// The body v is a cell of, NONE if none.
fn bodyOf(v: vec4<u32>) -> u32 {
    if (!cellHas(v, FL_LOOSE) || !structural(v)) {
        return NONE;
    }
    let id = cellFrac(v).x;
    if (id < 0.0 || id >= f32(MAX_BODIES)) {
        return NONE;
    }
    return u32(id);
}

// Whether v is loose but in no body: its support has run out, or the body it
// was part of broke up.
fn isUnowned(v: vec4<u32>) -> bool {
    if (!structural(v)) {
        return false;
    }
    let f = cellFrac(v);
    if (cellHas(v, FL_LOOSE)) {
        return f.x == FREE_BODY;
    }
    return f.x <= 0.0;
}

fn isUnownedAt(c: vec2<i32>) -> bool {
    return inWorld(c) && isUnowned(cells[cellIndex(c)]);
}

fn pieceOf(v: vec4<u32>) -> f32 {
    return cellFrac(v).y;
}

// Whether the loose cell at q belongs with one of piece `piece`.
fn samePieceAt(q: vec2<i32>, piece: f32) -> bool {
    if (!inWorld(q)) {
        return false;
    }
    let w = cells[cellIndex(q)];
    return isUnowned(w) && pieceOf(w) == piece;
}

// What the cell at q is to body b: nothing in its way (gas, liquid, powder
// that isn't resting on anything or is too light to hold up a solid, as for
// support, and its own cells), something that stands still (solids, resting
// powder, the world's edge, loose cells in no body yet), or another body.
fn obstacleKind(q: vec2<i32>, b: u32) -> u32 {
    if (!inWorld(q)) {
        return OB_STATIC;
    }
    return obstacleOf(cells[cellIndex(q)], b);
}

// The same for a cell in the world, given what is in it.
fn obstacleOf(v: vec4<u32>, b: u32) -> u32 {
    let t = cellType(v);
    let kind = elems[t].kind;
    if (kind == KIND_SOLID) {
        if (!cellHas(v, FL_LOOSE) || elems[t].breakTo == NO_ELEM) {
            return OB_STATIC;
        }
        let id = cellFrac(v).x;
        if (id == f32(b)) {
            return OB_NONE;
        }
        return select(OB_BODY, OB_STATIC, id >= f32(MAX_BODIES));
    }
    if (kind == KIND_POWDER && cellHas(v, FL_GROUNDED) && (elems[t].flags & F_FLUFF) == 0u) {
        return OB_STATIC;
    }
    return OB_NONE;
}

fn cross2(a: vec2<f32>, b: vec2<f32>) -> f32 {
    return a.x * b.y - a.y * b.x;
}

// The velocity of a point at r from the centre of a body spinning at spin.
fn spinVel(spin: f32, r: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(-spin * r.y, spin * r.x);
}

// The cell one step along axis (0 x, 1 y) in direction dir.
fn axisStep(axis: u32, dir: i32) -> vec2<i32> {
    return select(vec2<i32>(dir, 0), vec2<i32>(0, dir), axis == 1u);
}

// ----------------------------------------------------------------- poses
//
// A body's pose maps its shape's grid into the world: shape coords s (the
// cell (a, b) of the grid spans a..a+1, b..b+1) go to at + R(angle) (s -
// com), which its contacts and the liquid's push go by. Its cells are
// stamped into the world's one to one (a Stamp): shape cell s goes to the
// world cell stampCell gives, by a whole number of quarter turns about the
// pivot (a shape cell picked when it is found) and the rest of its angle,
// at most an eighth of a turn either way, as three shears (Alan Paeth, "A
// fast algorithm for general raster rotation", Graphics Interface '86):
// x += round(alpha y), y += round(beta x), x += round(alpha y), with alpha
// = -tan(rest / 2), beta = sin(rest), whose product is the rotation. Each
// shear moves whole rows or columns by whole cells, so each, and the three
// together, map the grid's cells one to one onto the world's (and back,
// unstampCell); the pivot goes to the world cell its middle is in at the
// pose, and every cell to within about a cell of where the pose puts it.

struct Pose {
    at: vec2<f32>,
    rot: vec2<f32>,  // cos, sin
    com: vec2<f32>,
    size: vec2<i32>,
    off: u32,
}

struct Stamp {
    anchor: vec2<i32>,  // the world cell the pivot goes to
    pivot: vec2<i32>,
    quarter: u32,       // whole quarter turns, 0..3
    shear: vec2<f32>,   // alpha, beta
    offset: vec2<f32>,  // the pivot's middle at the pose less its cell's, each way -0.5 to 0.5 (T_SUBCELL)
}

// Square to the grid (a multiple of a quarter turn) the rotation is exact,
// so that the shape maps one to one onto the world's cells.
fn rotationOf(angle: f32) -> vec2<f32> {
    let r = vec2<f32>(cos(angle), sin(angle));
    if (abs(r.x) < 1e-6) {
        return vec2<f32>(0.0, sign(r.y));
    }
    if (abs(r.y) < 1e-6) {
        return vec2<f32>(sign(r.x), 0.0);
    }
    return r;
}

// The angle taken into -pi..pi.
fn wrapAngle(a: f32) -> f32 {
    return a - round(a / TAU) * TAU;
}

fn rotate(v: vec2<f32>, rot: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(rot.x * v.x - rot.y * v.y, rot.y * v.x + rot.x * v.y);
}

// Where the middle of shape cell s is in the world.
fn cellWorld(s: vec2<i32>, q: Pose) -> vec2<f32> {
    return q.at + rotate(vec2<f32>(s) + 0.5 - q.com, q.rot);
}

// How a body at (at, angle), its centre of mass at com in its grid, is
// stamped turning about pivot.
fn stampAt(at: vec2<f32>, angle: f32, com: vec2<f32>, pivot: vec2<i32>) -> Stamp {
    var st: Stamp;
    let turns = round(angle / QUARTER_TURN);
    let rest = angle - turns * QUARTER_TURN;
    st.quarter = u32(i32(turns) & 3);
    st.shear = vec2<f32>(-tan(0.5 * rest), sin(rest));
    st.pivot = pivot;
    let middle = at + rotate(vec2<f32>(pivot) + 0.5 - com, rotationOf(angle));
    st.anchor = vec2<i32>(floor(middle));
    st.offset = middle - floor(middle) - 0.5;
    return st;
}

fn quarterTurned(o: vec2<i32>, quarter: u32) -> vec2<i32> {
    switch quarter {
        case 1u: {
            return vec2<i32>(-o.y, o.x);
        }
        case 2u: {
            return -o;
        }
        case 3u: {
            return vec2<i32>(o.y, -o.x);
        }
        default: {
            return o;
        }
    }
}

// The world cell shape cell s is stamped into.
fn stampCell(s: vec2<i32>, st: Stamp) -> vec2<i32> {
    let o = quarterTurned(s - st.pivot, st.quarter);
    let x1 = o.x + i32(round(st.shear.x * f32(o.y)));
    let y1 = o.y + i32(round(st.shear.y * f32(x1)));
    let x2 = x1 + i32(round(st.shear.x * f32(y1)));
    return st.anchor + vec2<i32>(x2, y1);
}

// The shape cell stamped into world cell p: stampCell undone, exactly.
fn unstampCell(p: vec2<i32>, st: Stamp) -> vec2<i32> {
    let d = p - st.anchor;
    let x1 = d.x - i32(round(st.shear.x * f32(d.y)));
    let y0 = d.y - i32(round(st.shear.y * f32(x1)));
    let x0 = x1 - i32(round(st.shear.x * f32(y0)));
    return st.pivot + quarterTurned(vec2<i32>(x0, y0), (4u - st.quarter) & 3u);
}

fn tableStamp(b: u32) -> Stamp {
    var st: Stamp;
    st.anchor = tableI(b, T_STAMP);
    st.quarter = tableU(b, T_STAMP + 2u);
    st.shear = tableV(b, T_STAMP + 3u);
    st.pivot = tableI(b, T_PIVOT);
    return st;
}

fn setStamp(b: u32, st: Stamp) {
    setI(b, T_STAMP, st.anchor);
    setU(b, T_STAMP + 2u, st.quarter);
    setV(b, T_STAMP + 3u, st.shear);
    setU(b, T_SUBCELL, pack2x16float(st.offset));
}

// ---------------------------------------------------------- the tiles

// Every body there is goes on the list of bodies to work on; and how many
// tiles were marked is noted for sim.js to read back.
fn listBody(i: u32) -> u32 {
    if (i < MAX_BODIES && (tableU(i, T_FLAGS) & B_ALIVE) != 0u) {
        let k = atomicAdd(&bodies[counterSlot(C_LISTED)], 1u);
        motion[listSlot(k)] = i;
    }
    let marked = atomicLoad(&bodies[statSlot(BODY_STAT_MARKED)]);
    if (i == 0u) {
        atomicStore(&bodies[statSlot(BODY_STAT_MARKED_RUN)], marked);
    }
    return marked;
}

// With nothing loose in no body as of the last readback, sim.js leaves out
// the passes that find new bodies: this lists the bodies only, and the
// tiles stay marked for when they run.
@compute @workgroup_size(64)
fn listBodies(@builtin(global_invocation_id) gid: vec3<u32>) {
    listBody(gid.x);
}

// Each marked tile, and the tiles around it, go on the list to work on, once
// each. The marks are spent: the update pass marks the tiles with loose
// cells in no body in them afresh next step.
@compute @workgroup_size(64)
fn plan(@builtin(global_invocation_id) gid: vec3<u32>) {
    // (The list found makes starts empty: foundBodies empties it, and a
    // snapshot from before D-1 holds the stack of free slots there.)
    if (gid.x == 0u) {
        atomicStore(&bodies[counterSlot(C_FOUNDING)], 0u);
    }
    let marked = listBody(gid.x);
    if (gid.x >= marked) {
        return;
    }
    let tile = atomicLoad(&bodies[bodyMarkListSlot(P.size) + gid.x]);
    atomicStore(&bodies[tile], 0u);
    let wide = i32(bodyTilesWide(P.size));
    let high = i32(tileCount()) / wide;
    let at = vec2<i32>(i32(tile) % wide, i32(tile) / wide);
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let t = at + vec2<i32>(dx, dy);
            if (any(t < vec2<i32>(0)) || t.x >= wide || t.y >= high) {
                continue;
            }
            let n = u32(t.y * wide + t.x);
            if (atomicExchange(&bodies[activeSlot(n)], 1u) == 0u) {
                let k = atomicAdd(&bodies[statSlot(BODY_STAT_ACTIVE)], 1u);
                atomicStore(&bodies[activeListSlot(k)], n);
            }
        }
    }
}

// The cell of thread lid in the k-th tile worked on.
fn tileCell(k: u32, lid: vec2<u32>) -> vec2<i32> {
    let n = atomicLoad(&bodies[activeListSlot(k)]);
    let wide = bodyTilesWide(P.size);
    return vec2<i32>(vec2<u32>(n % wide, n / wide) * BODY_TILE + lid);
}

fn tilesWorked() -> u32 {
    return atomicLoad(&bodies[statSlot(BODY_STAT_ACTIVE)]);
}

// Empties the lists, and lets the pool start over once no body is left in it;
// after packing it, the bodies' shapes are where they were packed to.
fn finishStep() {
    atomicStore(&bodies[statSlot(BODY_STAT_ACTIVE)], 0u);
    atomicStore(&bodies[counterSlot(C_LISTED)], 0u);
    atomicStore(&bodies[counterSlot(C_CONTACTS)], 0u);
    if (atomicLoad(&bodies[statSlot(BODY_STAT_COMPACT)]) == 2u) {
        for (var b = 0u; b < MAX_BODIES; b++) {
            if ((tableU(b, T_FLAGS) & B_ALIVE) != 0u) {
                setU(b, T_SHAPE, tableU(b, T_PACKED));
            }
        }
        atomicXor(&bodies[counterSlot(C_HALF)], 1u);
        atomicStore(&bodies[statSlot(BODY_STAT_COMPACT)], 0u);
    }
    if (atomicLoad(&bodies[statSlot(BODY_STAT_ALIVE)]) == 0u) {
        atomicStore(&bodies[counterSlot(C_POOL)], 0u);
    }
}

// Packing the pool. Bodies come and go, each taking room for its shape at the
// top of the half of the pool in use; the room of those gone is only
// regained when none is left, or when the half is full and the bodies still
// about are packed into the other half: plan works out where each goes, in
// the order of their slots (a prefix sum of their sizes), copy moves them
// there and empties where they were, and finish has them take the new place.
var<workgroup> packSizes: array<u32, 256>;
var<workgroup> packing: u32;

@compute @workgroup_size(256)
fn compactPlan(@builtin(local_invocation_index) li: u32) {
    if (li == 0u) {
        packing = atomicLoad(&bodies[statSlot(BODY_STAT_COMPACT)]);
    }
    if (workgroupUniformLoad(&packing) == 0u) {
        return;
    }
    // Four slots a thread.
    var size = 0u;
    for (var k = 0u; k < 4u; k++) {
        let b = li * 4u + k;
        if ((tableU(b, T_FLAGS) & B_ALIVE) != 0u) {
            let s = tableU(b, T_SIZE);
            size += (s & 0xffffu) * (s >> 16u);
        }
    }
    packSizes[li] = size;
    workgroupBarrier();
    if (li == 0u) {
        var total = 0u;
        for (var k = 0u; k < 256u; k++) {
            let s = packSizes[k];
            packSizes[k] = total;
            total += s;
        }
        atomicStore(&bodies[counterSlot(C_POOL)], total);
        atomicStore(&bodies[statSlot(BODY_STAT_COMPACT)], 2u);
    }
    workgroupBarrier();
    let capacity = arrayLength(&shapes) / 2u;
    var at = (1u - atomicLoad(&bodies[counterSlot(C_HALF)])) * capacity + packSizes[li];
    for (var k = 0u; k < 4u; k++) {
        let b = li * 4u + k;
        if ((tableU(b, T_FLAGS) & B_ALIVE) != 0u) {
            setU(b, T_PACKED, at);
            let s = tableU(b, T_SIZE);
            at += (s & 0xffffu) * (s >> 16u);
        }
    }
}

@compute @workgroup_size(256)
fn compactCopy(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    if (atomicLoad(&bodies[statSlot(BODY_STAT_COMPACT)]) != 2u) {
        return;
    }
    for (var b = wid.x; b < MAX_BODIES; b += groups.x) {
        if ((tableU(b, T_FLAGS) & B_ALIVE) == 0u) {
            continue;
        }
        let old = tableU(b, T_SHAPE);
        let packed = tableU(b, T_PACKED);
        let s = tableU(b, T_SIZE);
        let n = (s & 0xffffu) * (s >> 16u);
        for (var i = li; i < n; i += BODY_THREADS) {
            shapes[packed + i] = shapes[old + i];
            shapes[old + i] = vec2<u32>(0u);
        }
    }
}

@compute @workgroup_size(1)
fn finish() {
    atomicStore(&bodies[statSlot(BODY_STAT_MARKED)], 0u);
    finishStep();
}

// After listBodies: the marked tiles' list is kept.
@compute @workgroup_size(1)
fn finishListed() {
    finishStep();
}

// ---------------------------------------------------------------- labels

// A tile's loose cells are joined up in workgroup memory first, much faster
// than in the bodies buffer: each takes the tile's lowest loose cell of its
// piece it is connected to inside the tile as its label (the lowest by cell
// index too), and only those have sums. The unite pass then only has to
// join labels across the tiles' edges.
var<workgroup> local: array<atomic<u32>, 64>;  // NONE where not loose
var<workgroup> localPiece: array<f32, 64>;     // its piece, -1 where not loose
var<workgroup> tilesShared: u32;

// The union-find's links (labeltile.wgsl, label.wgsl): a tile's cells' in
// workgroup memory, and the cells' labels in the bodies buffer, loose
// pieces' (labelSlot) and structure's (structureLabelSlot).
fn ufTile(i: u32) -> u32 {
    return atomicLoad(&local[i]);
}

fn ufTileLower(i: u32, to: u32) -> u32 {
    return atomicMin(&local[i], to);
}

fn ufLink(at: u32) -> u32 {
    return atomicLoad(&bodies[at]);
}

fn ufLower(at: u32, to: u32) -> u32 {
    return atomicMin(&bodies[at], to);
}

// Joins loose cells i and j of the tile if they are of one piece.
fn localJoin(i: u32, j: u32) {
    if (localPiece[j] == localPiece[i]) {
        ufTileJoin(i, j);
    }
}

fn zeroSums(i: u32) {
    atomicStore(&bodies[sumSlot(i, S_BOX)], NONE);
    atomicStore(&bodies[sumSlot(i, S_BOX + 1u)], NONE);
    atomicStore(&bodies[sumSlot(i, S_BOX + 2u)], 0u);
    atomicStore(&bodies[sumSlot(i, S_BOX + 3u)], 0u);
    atomicStore(&bodies[sumSlot(i, S_RIPE)], NONE);
    atomicStore(&bodies[sumSlot(i, S_CELLS)], 0u);
    atomicStore(&bodies[sumSlot(i, S_SLOT)], NONE);
}

@compute @workgroup_size(8, 8)
fn init(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    if (li == 0u) {
        tilesShared = tilesWorked();
    }
    let count = workgroupUniformLoad(&tilesShared);
    for (var k = wid.x; k < count; k += groups.x) {
        let c = tileCell(k, lid.xy);
        var loose = false;
        var piece = -1.0;
        if (inWorld(c)) {
            let v = cells[cellIndex(c)];
            loose = isUnowned(v);
            piece = select(-1.0, pieceOf(v), loose);
        }
        atomicStore(&local[li], select(NONE, li, loose));
        localPiece[li] = piece;
        workgroupBarrier();
        // Right, down and both diagonals down, inside the tile.
        if (loose) {
            if (lid.x < 7u) {
                localJoin(li, li + 1u);
            }
            if (lid.y < 7u) {
                if (lid.x > 0u) {
                    localJoin(li, li + 7u);
                }
                localJoin(li, li + 8u);
                if (lid.x < 7u) {
                    localJoin(li, li + 9u);
                }
            }
        }
        workgroupBarrier();
        if (loose) {
            let root = ufTileFind(li);
            let i = cellIndex(c);
            atomicStore(&bodies[labelSlot(i)], cellIndex(c - vec2<i32>(lid.xy) + vec2<i32>(i32(root % 8u), i32(root / 8u))));
            if (root == li) {
                zeroSums(i);
            }
        }
        workgroupBarrier();
    }
}

// The root of i's tree (label.wgsl).
fn find(i: u32) -> u32 {
    return ufFind(labelSlot(0u), i);
}

fn join(i: u32, j: u32) {
    ufJoin(labelSlot(0u), i, j);
}

fn joinLoose(c: vec2<i32>, q: vec2<i32>, piece: f32) {
    if (samePieceAt(q, piece)) {
        join(cellIndex(c), cellIndex(q));
    }
}

// Right, down and both diagonals down, where they are in another tile: every
// pair of neighbors across the tiles' edges once.
fn uniteCell(c: vec2<i32>, lid: vec2<u32>) {
    if (!isUnownedAt(c)) {
        return;
    }
    let piece = pieceOf(cells[cellIndex(c)]);
    if (lid.x == 7u) {
        joinLoose(c, c + vec2<i32>(1, 0), piece);
    }
    if (lid.x == 0u || lid.y == 7u) {
        joinLoose(c, c + vec2<i32>(-1, 1), piece);
    }
    if (lid.y == 7u) {
        joinLoose(c, c + vec2<i32>(0, 1), piece);
    }
    if (lid.x == 7u || lid.y == 7u) {
        joinLoose(c, c + vec2<i32>(1, 1), piece);
    }
}

@compute @workgroup_size(8, 8)
fn unite(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    for (var k = wid.x; k < tilesWorked(); k += groups.x) {
        uniteCell(tileCell(k, lid.xy), lid.xy);
    }
}

// ----------------------------------------------------------------- anchoring
//
// Static structure (structural solids not loose) is held up only if it is
// connected to an anchor (REWRITE_PLAN.md 8-1; audit/04 §5.4's C1): the
// world's edge, a solid that isn't structural (wall, a machine), settled
// powder under it, or for a plant the soil its roots are in. Only face to
// face: squares joined at a corner alone turn about it (in the stress solve
// they share one node, a hinge, 8-2), so a piece joined face to face is
// rigid, and held by a face it is held at two nodes at least. Whenever it changes (structureWatch:
// each tile's hash of where its static structure is and which of it
// touches an anchor, against last step's: sand dug out from under a tower
// changes no structure),
// it is labelled (union-find over the cells beside one another, face to
// face, all pieces together), each piece's root is marked if any of its
// cells meets an anchor face to face, and the cells of every
// piece left unmarked are loose at once, with their support run out, which
// the body passes then take: a tower cut off at its foot falls, rather than
// hanging there until its support has run down a cell a step. How far
// structure can reach out from its anchors is the stress solve's to say
// (stress.wgsl: what can't carry its load cracks). In the bodies buffer after the
// sums: whether it changed, a hash per tile, then per cell its label and
// whether it is a marked root.

fn structureSlot() -> u32 {
    return structureBase(P.size);
}

fn structureHashSlot(tile: u32) -> u32 {
    return structureSlot() + 1u + tile;
}

fn structureLabelSlot(i: u32) -> u32 {
    return structureSlot() + 1u + tileCount() + i;
}

fn structureAnchorSlot(i: u32) -> u32 {
    return structureLabelSlot(cellCount()) + i;
}

fn isStatic(v: vec4<u32>) -> bool {
    return structural(v) && !cellHas(v, FL_LOOSE);
}

fn isStaticAt(c: vec2<i32>) -> bool {
    return inWorld(c) && isStatic(cells[cellIndex(c)]);
}

// Static structure not let go (its support run out, frac.x <= 0, waiting
// for the body passes to take it): what the pieces are labelled and
// anchored over, the cells the stress solve takes (stress.wgsl's
// standingAt). (Labelled as well, a piece let go could join one that wasn't
// and hold it by an anchor the stress solve doesn't see.)
fn standingAt(c: vec2<i32>) -> bool {
    return isStaticAt(c) && cellFrac(cells[cellIndex(c)]).x > 0.0;
}

fn structureFind(i: u32) -> u32 {
    return ufFind(structureLabelSlot(0u), i);
}

fn structureJoin(i: u32, j: u32) {
    ufJoin(structureLabelSlot(0u), i, j);
}

// Whether static structure at c meets an anchor across its face toward q
// (q - c a unit step): the world's edge or a solid that isn't structural,
// settled powder (not fluff) under it (along its down), or, for a plant,
// soil: any it grows in, that holds water, or mud (it was dirt and mud
// alone, so a seedling on ash or sand was let go the step it sprouted).
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
    let down = downAt(c);
    let d = q - c;
    let under = (d.x != 0 && d.x == down.x) || (d.y != 0 && d.y == down.y);
    return under && e.kind == KIND_POWDER && (e.flags & F_FLUFF) == 0u && length(cellVel(v)) < SETTLED_SPEED;
}

fn anchoredAt(c: vec2<i32>) -> bool {
    // Painted in a stroke still being drawn (common.wgsl's SUPPORT_HELD).
    if (cellFrac(cells[cellIndex(c)]).x == SUPPORT_HELD) {
        return true;
    }
    for (var k = 0; k < 4; k++) {
        if (anchorFace(c, c + besideStep(k))) {
            return true;
        }
    }
    return false;
}

var<workgroup> structureChanged: u32;
var<workgroup> watchHash: atomic<u32>;

fn structureCell(wid: vec3<u32>, lid: vec3<u32>) -> vec2<i32> {
    return vec2<i32>(wid.xy * BODY_TILE + lid.xy);
}

// Whether the structure changed this step, for the whole workgroup.
fn structureDirty(li: u32) -> bool {
    if (li == 0u) {
        structureChanged = atomicLoad(&bodies[structureSlot()]);
    }
    return workgroupUniformLoad(&structureChanged) != 0u;
}

@compute @workgroup_size(8, 8)
fn structureWatch(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    if (li == 0u) {
        atomicStore(&watchHash, 0u);
    }
    workgroupBarrier();
    let c = structureCell(wid, lid);
    if (isStaticAt(c)) {
        let anchored = select(0u, 0x9e3779b9u, anchoredAt(c));
        atomicAdd(&watchHash, ((cellIndex(c) + 1u) * 2654435761u) ^ anchored);
    }
    workgroupBarrier();
    if (li == 0u) {
        let slot = structureHashSlot(wid.y * bodyTilesWide(P.size) + wid.x);
        let hash = atomicLoad(&watchHash);
        if (atomicLoad(&bodies[slot]) != hash) {
            atomicStore(&bodies[slot], hash);
            atomicStore(&bodies[structureSlot()], 1u);
        }
    }
}

@compute @workgroup_size(8, 8)
fn structureInit(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    if (!structureDirty(li)) {
        return;
    }
    let c = structureCell(wid, lid);
    let fixed = standingAt(c);
    atomicStore(&local[li], select(NONE, li, fixed));
    workgroupBarrier();
    // Face to face, standing to standing only. (Joined to whatever was
    // beside it, a cell of air right of one cell and under another joined
    // the two, which met only at a corner: a cell a turned stamp left
    // hanging off a resting block by its corner was held by the block's
    // anchor, and in the stress solve, turning on that one node, it was a
    // mechanism whose solve ran away to displacements of thousands of
    // cells, and cracked the block: Tumbling block, a run in five.)
    if (fixed) {
        if (lid.x < 7u && standingAt(c + vec2<i32>(1, 0))) {
            ufTileJoin(li, li + 1u);
        }
        if (lid.y < 7u && standingAt(c + vec2<i32>(0, 1))) {
            ufTileJoin(li, li + 8u);
        }
    }
    workgroupBarrier();
    if (inWorld(c)) {
        let i = cellIndex(c);
        var label = NONE;
        if (fixed) {
            let root = ufTileFind(li);
            label = cellIndex(c - vec2<i32>(lid.xy) + vec2<i32>(i32(root % 8u), i32(root / 8u)));
        }
        atomicStore(&bodies[structureLabelSlot(i)], label);
        atomicStore(&bodies[structureAnchorSlot(i)], 0u);
    }
}

@compute @workgroup_size(8, 8)
fn structureUnite(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    if (!structureDirty(li)) {
        return;
    }
    let c = structureCell(wid, lid);
    if (!standingAt(c)) {
        return;
    }
    let i = cellIndex(c);
    if (lid.x == 7u && standingAt(c + vec2<i32>(1, 0))) {
        structureJoin(i, cellIndex(c + vec2<i32>(1, 0)));
    }
    if (lid.y == 7u && standingAt(c + vec2<i32>(0, 1))) {
        structureJoin(i, cellIndex(c + vec2<i32>(0, 1)));
    }
}

@compute @workgroup_size(8, 8)
fn structureAnchor(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    if (!structureDirty(li)) {
        return;
    }
    let c = structureCell(wid, lid);
    if (standingAt(c) && anchoredAt(c)) {
        atomicStore(&bodies[structureAnchorSlot(structureFind(cellIndex(c)))], 1u);
    }
}

// The tile of cell p marked as holding loose cells in no body, as the
// update pass marks them (update.wgsl's markLoose), for plan to work on.
fn markTile(p: vec2<i32>) {
    let tile = bodyTileOf(vec2<u32>(p), P.size);
    if (atomicLoad(&bodies[tile]) == 0u && atomicExchange(&bodies[tile], 1u) == 0u) {
        let n = atomicAdd(&bodies[bodyMarkedSlot(P.size)], 1u);
        atomicStore(&bodies[bodyMarkListSlot(P.size) + n], tile);
    }
}

// The cells of a piece no anchor holds have their support run out
// (update.wgsl's: as if they had been loose for COLLAPSE_STEPS), and their
// tiles are marked: the update pass has marked this step's already, and
// the labelling (init, unite) has to work on every tile with loose cells in
// no body in it. (Unmarked, a piece let go here had its cells in the tiles
// next to marked ones labelled, and unite joined them to their neighbours
// in the tiles beyond, whose labels were last step's or none: their root
// was cell 0, its sums never emptied, and capture took the cells into
// whichever body was in slot 0, outside its shape, where none moved them.)
@compute @workgroup_size(8, 8)
fn structureSettle(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    if (!structureDirty(li)) {
        return;
    }
    let c = structureCell(wid, lid);
    if (!standingAt(c)) {
        return;
    }
    let i = cellIndex(c);
    if (atomicLoad(&bodies[structureAnchorSlot(structureFind(i))]) == 0u) {
        var cell = unpackCell(cells[i]);
        cell.frac.x = min(cell.frac.x, -P.supportMax);
        cells[i] = packCell(cell);
        markTile(c);
    }
}

@compute @workgroup_size(1)
fn structureDone() {
    atomicStore(&bodies[structureSlot()], 0u);
}

// ----------------------------------------------------------------- sums
//
// A tile's cells add themselves up in workgroup memory first, one set of
// sums for each piece in the tile, kept at the slot of its first cell there;
// that cell's thread then adds them to the piece's. Thousands of cells
// adding to the same few words in the bodies buffer would wait on each other.

var<workgroup> slotRoot: array<u32, 64>;
var<workgroup> tileSums: array<atomic<u32>, 64u * SUM_WORDS>;

// How long the loose cell v has been loose, from how far its support has run
// down (update.wgsl); a cell of a body that broke up has been for long enough.
fn looseSteps(v: vec4<u32>) -> u32 {
    if (cellHas(v, FL_LOOSE)) {
        return u32(RIPEN_STEPS);
    }
    return u32(-cellFrac(v).x * COLLAPSE_STEPS / P.supportMax);
}

fn gatherCell(c: vec2<i32>, slot: u32) {
    let v = cells[cellIndex(c)];
    let piece = pieceOf(v);
    let at = slot * SUM_WORDS;
    atomicAdd(&tileSums[at + S_CELLS], 1u);
    atomicMin(&tileSums[at + S_RIPE], looseSteps(v));
    // Its bounding box, from the cells on its edges.
    if (!samePieceAt(c - vec2<i32>(1, 0), piece)) {
        atomicMin(&tileSums[at + S_BOX], u32(c.x));
    }
    if (!samePieceAt(c - vec2<i32>(0, 1), piece)) {
        atomicMin(&tileSums[at + S_BOX + 1u], u32(c.y));
    }
    if (!samePieceAt(c + vec2<i32>(1, 0), piece)) {
        atomicMax(&tileSums[at + S_BOX + 2u], u32(c.x));
    }
    if (!samePieceAt(c + vec2<i32>(0, 1), piece)) {
        atomicMax(&tileSums[at + S_BOX + 3u], u32(c.y));
    }
}

fn flushSums(slot: u32, r: u32) {
    let at = slot * SUM_WORDS;
    atomicAdd(&bodies[sumSlot(r, S_CELLS)], atomicLoad(&tileSums[at + S_CELLS]));
    atomicMin(&bodies[sumSlot(r, S_RIPE)], atomicLoad(&tileSums[at + S_RIPE]));
    for (var k = 0u; k < 2u; k++) {
        let lo = atomicLoad(&tileSums[at + S_BOX + k]);
        if (lo != NONE) {
            atomicMin(&bodies[sumSlot(r, S_BOX + k)], lo);
        }
        atomicMax(&bodies[sumSlot(r, S_BOX + 2u + k)], atomicLoad(&tileSums[at + S_BOX + 2u + k]));
    }
}

@compute @workgroup_size(8, 8)
fn gather(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    if (li == 0u) {
        tilesShared = tilesWorked();
    }
    let count = workgroupUniformLoad(&tilesShared);
    for (var k = wid.x; k < count; k += groups.x) {
        let c = tileCell(k, lid.xy);
        let loose = isUnownedAt(c);
        var r = NONE;
        if (loose) {
            let i = cellIndex(c);
            r = find(i);
            atomicStore(&bodies[labelSlot(i)], r);
        }
        slotRoot[li] = r;
        for (var w = 0u; w < SUM_WORDS; w++) {
            let none = w == S_RIPE || w == S_BOX || w == S_BOX + 1u;
            atomicStore(&tileSums[li * SUM_WORDS + w], select(0u, NONE, none));
        }
        workgroupBarrier();
        var slot = li;
        if (loose) {
            for (var j = 0u; j < li; j++) {
                if (slotRoot[j] == r) {
                    slot = j;
                    break;
                }
            }
            gatherCell(c, slot);
        }
        workgroupBarrier();
        if (loose && slot == li) {
            flushSums(li, r);
        }
        workgroupBarrier();
    }
}

// ------------------------------------------------------------ new bodies

// Bodies are founded in the order of their pieces' roots (their lowest
// cells), each taking the lowest free slot, room in the pool after the last
// one's and the next piece id: taken as threads came to them, which piece
// got which slot, room and id differed from run to run, and so did what
// came of them (REWRITE_PLAN.md D-1). So found only lists the pieces ripe
// to become bodies, and foundBodies, one workgroup, founds them.

// A slot given back: free again for foundBodies (which looks for slots no
// living body holds).
fn giveSlot(b: u32) {
    setU(b, T_FLAGS, 0u);
    atomicSub(&bodies[statSlot(BODY_STAT_ALIVE)], 1u);
}

// Each piece that has been loose long enough, at its root's thread, goes on
// the list to become a body (the first MAX_BODIES of them: more at once,
// which those are is the GPU's race).
fn foundCell(c: vec2<i32>) {
    if (!inWorld(c)) {
        return;
    }
    let i = cellIndex(c);
    if (!isUnowned(cells[i]) || atomicLoad(&bodies[labelSlot(i)]) != i) {
        return;
    }
    if (f32(sum(i, S_RIPE)) < RIPEN_STEPS) {
        return;
    }
    let k = atomicAdd(&bodies[counterSlot(C_FOUNDING)], 1u);
    if (k < MAX_BODIES) {
        atomicStore(&bodies[foundingSlot(k)], i);
    }
}

@compute @workgroup_size(8, 8)
fn found(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    for (var k = wid.x; k < tilesWorked(); k += groups.x) {
        foundCell(tileCell(k, lid.xy));
    }
}

// The size of the box of the piece whose root is i.
fn foundSize(i: u32) -> vec2<i32> {
    let lo = vec2<i32>(i32(sum(i, S_BOX)), i32(sum(i, S_BOX + 1u)));
    let hi = vec2<i32>(i32(sum(i, S_BOX + 2u)), i32(sum(i, S_BOX + 3u)));
    return hi - lo + 1;
}

// The piece whose root is i becomes body b, its shape at off in the pool.
// It starts out where its cells are, unturned; its mass, centre of mass and
// motion are worked out by the lift pass, next step. Until then a body
// touching it reads it as a uniform box of water of its cells, with nothing
// pushing it: a slot taken again would otherwise still hold the last
// body's, and a fresh one none (0, which the contacts divide by).
fn foundBody(i: u32, b: u32, off: u32, piece: u32) {
    let lo = vec2<i32>(i32(sum(i, S_BOX)), i32(sum(i, S_BOX + 1u)));
    let hi = vec2<i32>(i32(sum(i, S_BOX + 2u)), i32(sum(i, S_BOX + 3u)));
    let size = hi - lo + 1;
    let middle = vec2<f32>(size) * 0.5;
    setV(b, T_POS, vec2<f32>(lo) + middle);
    setF(b, T_ANGLE, 0.0);
    setV(b, T_VEL, vec2<f32>(0.0));
    setF(b, T_SPIN, 0.0);
    setV(b, T_COM, middle);
    setU(b, T_SHAPE, off);
    setU(b, T_SIZE, u32(size.x) | (u32(size.y) << 16u));
    setU(b, T_CELLS, sum(i, S_CELLS));
    let mass = f32(sum(i, S_CELLS));
    setF(b, T_MASS, mass);
    setF(b, T_INERTIA, mass * dot(vec2<f32>(size), vec2<f32>(size)) / 12.0);
    setF(b, T_RADIUS, length(middle));
    setV(b, T_FORCE, vec2<f32>(0.0));
    setF(b, T_FORCE + 2u, 0.0);
    setV(b, T_LIQUID, vec2<f32>(0.0));
    setV(b, T_LIQUID + 2u, vec2<f32>(0.0));
    setF(b, T_CARGO, 0.0);
    setF(b, T_CARGO_INERTIA, 0.0);
    for (var k = 0u; k < BODY_LIQUID_WORDS; k++) {
        motion[liquidSlot(b, k)] = 0u;
    }
    setI(b, T_BOX, lo);
    setI(b, T_BOX + 2u, hi);
    // Its cells are where they are: stamped square, turning about the cell
    // its middle is in, as if last step (capture claims them so), so the
    // stamp this step leaves those it no longer covers.
    let pivot = vec2<i32>(floor(middle));
    setI(b, T_PIVOT, pivot);
    var st: Stamp;
    st.anchor = lo + pivot;
    st.pivot = pivot;
    st.quarter = 0u;
    st.shear = vec2<f32>(0.0);
    setStamp(b, st);
    setU(b, T_MOVED_N, 0u);
    // (What the joint solve reads of a body that its contacts may find in
    // its founding step: the slot's last body's were left.)
    setV(b, T_NEXTVEL, vec2<f32>(0.0));
    setF(b, T_NEXTVEL + 2u, 0.0);
    setV(b, T_PUSH, vec2<f32>(0.0));
    setF(b, T_PUSH + 2u, 0.0);
    setU(b, T_COUNT, 0u);
    setU(b, T_FLAGS, B_ALIVE | B_NEW);
    setU(b, T_FRAME, P.frame - 1u);
    setU(b, T_PIECE, piece);
    setU(b, T_REST, 0u);
    atomicStore(&bodies[sumSlot(i, S_SLOT)], b);
}

const FOUND_EACH: u32 = MAX_BODIES / BODY_THREADS;  // list entries and slots a thread
var<workgroup> foundRoots: array<u32, MAX_BODIES>;
var<workgroup> foundFree: array<u32, MAX_BODIES>;
var<workgroup> foundRun: array<u32, BODY_THREADS>;
var<workgroup> foundCounts: vec4<u32>;  // listed, the sort's size, free slots, founded

// An exclusive prefix sum of foundRun (by one thread: a step's worth, once
// a step); the total.
fn foundScan(li: u32) -> u32 {
    workgroupBarrier();
    if (li == 0u) {
        var total = 0u;
        for (var k = 0u; k < BODY_THREADS; k++) {
            let n = foundRun[k];
            foundRun[k] = total;
            total += n;
        }
        foundCounts.w = total;
    }
    workgroupBarrier();
    return foundCounts.w;
}

// The listed pieces, sorted by root (bitonic), become bodies in that order:
// the lowest free slots, their shapes one after the other in the pool, the
// next piece ids; as many as there are slots and room for (the pool is
// packed at the end of the step once sim.js reads it is full, and the rest
// try again).
@compute @workgroup_size(256)
fn foundBodies(@builtin(local_invocation_index) li: u32) {
    if (li == 0u) {
        let n = min(atomicLoad(&bodies[counterSlot(C_FOUNDING)]), MAX_BODIES);
        var size = 1u;
        while (size < n) {
            size <<= 1u;
        }
        foundCounts = vec4<u32>(n, select(size, 0u, n == 0u), 0u, 0u);
    }
    let counts = workgroupUniformLoad(&foundCounts);
    if (counts.x == 0u) {
        return;
    }
    for (var k = li; k < counts.y; k += BODY_THREADS) {
        foundRoots[k] = select(NONE, atomicLoad(&bodies[foundingSlot(k)]), k < counts.x);
    }
    workgroupBarrier();
    for (var k = 2u; k <= counts.y; k <<= 1u) {
        for (var j = k >> 1u; j > 0u; j >>= 1u) {
            for (var i = li; i < counts.y; i += BODY_THREADS) {
                let l = i ^ j;
                if (l > i && (foundRoots[i] > foundRoots[l]) == ((i & k) == 0u)) {
                    let r = foundRoots[i];
                    foundRoots[i] = foundRoots[l];
                    foundRoots[l] = r;
                }
            }
            workgroupBarrier();
        }
    }
    // The free slots, lowest first.
    var free = 0u;
    for (var k = 0u; k < FOUND_EACH; k++) {
        free += select(1u, 0u, (tableU(li * FOUND_EACH + k, T_FLAGS) & B_ALIVE) != 0u);
    }
    foundRun[li] = free;
    let freeCount = foundScan(li);
    var at = foundRun[li];
    for (var k = 0u; k < FOUND_EACH; k++) {
        let b = li * FOUND_EACH + k;
        if ((tableU(b, T_FLAGS) & B_ALIVE) == 0u) {
            foundFree[at] = b;
            at++;
        }
    }
    // Their room in the pool, in order.
    workgroupBarrier();
    var room = 0u;
    for (var k = 0u; k < FOUND_EACH; k++) {
        let j = li * FOUND_EACH + k;
        if (j < counts.x) {
            let size = foundSize(foundRoots[j]);
            room += u32(size.x * size.y);
        }
    }
    foundRun[li] = room;
    foundScan(li);
    let capacity = arrayLength(&shapes) / 2u;
    let top = atomicLoad(&bodies[counterSlot(C_POOL)]);
    let base = atomicLoad(&bodies[counterSlot(C_HALF)]) * capacity;
    let pieces = atomicLoad(&bodies[counterSlot(C_PIECE)]);
    var off = foundRun[li];
    var founded = 0u;
    var full = false;
    for (var k = 0u; k < FOUND_EACH; k++) {
        let j = li * FOUND_EACH + k;
        if (j >= counts.x) {
            break;
        }
        let i = foundRoots[j];
        let size = foundSize(i);
        let n = u32(size.x * size.y);
        if (top + off + n > capacity) {
            full = true;
            break;
        }
        if (j >= freeCount) {
            break;
        }
        foundBody(i, foundFree[j], base + top + off, 1u + (pieces + j) % MAX_PIECE);
        off += n;
        founded = j + 1u;
    }
    if (full) {
        atomicMax(&bodies[statSlot(BODY_STAT_COMPACT)], 1u);
    }
    // (Those founded are the first of the order: the thread with the last
    // one has its end.)
    foundRun[li] = founded;
    workgroupBarrier();
    if (li == 0u) {
        var last = 0u;
        var used = 0u;
        for (var k = 0u; k < BODY_THREADS; k++) {
            last = max(last, foundRun[k]);
        }
        for (var j = 0u; j < last; j++) {
            let size = foundSize(foundRoots[j]);
            used += u32(size.x * size.y);
        }
        atomicStore(&bodies[counterSlot(C_POOL)], top + used);
        atomicStore(&bodies[counterSlot(C_PIECE)], pieces + last);
        atomicAdd(&bodies[statSlot(BODY_STAT_ALIVE)], last);
        atomicStore(&bodies[counterSlot(C_FOUNDING)], 0u);
    }
}

// Each cell of a new body goes into its shape, and is marked as its own,
// claimed by it as if stamped there last step. Its velocity stays: a piece
// of a body that broke up goes on as it was moving.
fn captureCell(c: vec2<i32>) {
    if (!inWorld(c)) {
        return;
    }
    let i = cellIndex(c);
    let v = cells[i];
    if (!isUnowned(v)) {
        return;
    }
    let b = sum(atomicLoad(&bodies[labelSlot(i)]), S_SLOT);
    if (b == NONE) {
        return;
    }
    let s = c - tableI(b, T_BOX);
    let size = tableU(b, T_SIZE);
    // Only into its grid: a cell its label sent to a body whose box doesn't
    // hold it is left unowned, for the labelling to take up again next
    // step. (Written past the grid, into another body's shape or none, it
    // was a cell no body moved, marked as the body's: the owner's
    // bodies-weird.snap, 119 cells of steel in a strip 88 wide marked as a
    // body of 27 by 22, review F140.)
    if (any(s < vec2<i32>(0)) || any(s >= vec2<i32>(i32(size & 0xffffu), i32(size >> 16u)))) {
        return;
    }
    shapes[tableU(b, T_SHAPE) + u32(s.y) * (size & 0xffffu) + u32(s.x)] = vec2<u32>(v.x, v.y);
    var cell = unpackCell(v);
    cell.flags |= FL_LOOSE;
    cell.frac = vec2<f32>(f32(b), f32(tableU(b, T_PIECE)));
    cells[i] = packCell(cell);
    atomicStore(&bodies[claimSlot(i)], bodyKey(P.frame - 1u, b));
}

// The last pass over the tiles also takes them off the list to work on.
@compute @workgroup_size(8, 8)
fn capture(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    for (var k = wid.x; k < tilesWorked(); k += groups.x) {
        captureCell(tileCell(k, lid.xy));
        if (li == 0u) {
            atomicStore(&bodies[activeSlot(atomicLoad(&bodies[activeListSlot(k)]))], 0u);
        }
    }
}

// ------------------------------------------------------------ the bodies
//
// The passes over the bodies have a workgroup take a body at a time; its
// thread 0 reads the body's slot into workgroup memory (head), and all of
// them go through its cells together.

struct Head {
    pos: vec2<f32>,
    angle: f32,
    spin: f32,
    vel: vec2<f32>,
    com: vec2<f32>,
    size: vec2<i32>,
    lo: vec2<i32>,
    hi: vec2<i32>,
    mass: f32,
    inertia: f32,
    radius: f32,
    off: u32,
    frame: u32,
    piece: u32,
    rest: u32,
    cells: u32,
    force: vec4<f32>,   // the air's push x, y, its torque (T_FORCE)
    liquid: vec4<f32>,  // what the liquid did to its velocity x, y, spin, its cells in the solve (T_LIQUID)
    stamp: Stamp,  // as it was last stamped
}

var<workgroup> head: Head;
var<workgroup> headFlags: u32;
var<workgroup> listed: u32;
// Its cells last stamped elsewhere (T_MOVED): the shape cell (its index in
// the grid), the world cell.
var<workgroup> movedShape: array<u32, MOVED_CAP>;
var<workgroup> movedCell: array<u32, MOVED_CAP>;
var<workgroup> movedCount: u32;

fn loadHead(b: u32) {
    head.pos = tableV(b, T_POS);
    head.angle = tableF(b, T_ANGLE);
    head.spin = tableF(b, T_SPIN);
    head.vel = tableV(b, T_VEL);
    head.com = tableV(b, T_COM);
    let size = tableU(b, T_SIZE);
    head.size = vec2<i32>(i32(size & 0xffffu), i32(size >> 16u));
    head.lo = tableI(b, T_BOX);
    head.hi = tableI(b, T_BOX + 2u);
    head.mass = massOf(b);
    head.inertia = inertiaOf(b);
    head.radius = tableF(b, T_RADIUS);
    head.off = tableU(b, T_SHAPE);
    head.frame = tableU(b, T_FRAME);
    head.piece = tableU(b, T_PIECE);
    head.rest = tableU(b, T_REST);
    head.cells = tableU(b, T_CELLS);
    head.force = vec4<f32>(tableV(b, T_FORCE), tableF(b, T_FORCE + 2u), 0.0);
    head.liquid = vec4<f32>(tableV(b, T_LIQUID), tableV(b, T_LIQUID + 2u));
    head.stamp = tableStamp(b);
    headFlags = tableU(b, T_FLAGS);
    movedCount = min(tableU(b, T_MOVED_N), MOVED_CAP);
    for (var k = 0u; k < movedCount; k++) {
        movedShape[k] = tableU(b, T_MOVED + 2u * k);
        movedCell[k] = tableU(b, T_MOVED + 2u * k + 1u);
    }
}

// The world cell the head body's shape cell i (at s in its grid) was last
// stamped into: by its stamp, or where it went instead (MOVED_CAP).
fn stampedAt(i: u32, s: vec2<i32>) -> vec2<i32> {
    for (var k = 0u; k < movedCount; k++) {
        if (movedShape[k] == i) {
            let j = movedCell[k];
            return vec2<i32>(i32(j % P.size.x), i32(j / P.size.x));
        }
    }
    return stampCell(s, head.stamp);
}

// Whether nothing body b's cells can't pass (what stands still, another
// body) lies on the line from p to q, between them: what it puts at q
// doesn't go through a wall.
fn clearPath(p: vec2<i32>, q: vec2<i32>, b: u32) -> bool {
    let d = q - p;
    let n = max(abs(d.x), abs(d.y));
    for (var k = 1; k < n; k++) {
        let c = p + vec2<i32>(round(vec2<f32>(d) * f32(k) / f32(n)));
        if (obstacleKind(c, b) != OB_NONE) {
            return false;
        }
    }
    return true;
}

// The nearest cell of gas to p within SEEK_REACH with a clear line to it
// from the middle of body b (clearPath: not through a wall, however far
// into it the body's pose or its stamp reaches), claimed for matter put
// there (claimFill), going round each ring from its top left corner; from
// p itself (`first` 0) or the ring round it (1); (-1, -1) if none. With
// `cups`, none of the head body's cup as stamped by st (a cell of a crate's
// wall put inside it left a hole in the wall, which let the water in).
fn seekGas(p: vec2<i32>, first: i32, b: u32, st: Stamp, cups: bool) -> vec2<i32> {
    let middle = vec2<i32>(floor(head.pos));
    if (first == 0 && seekable(p, middle, b, st, cups)) {
        return p;
    }
    for (var r = 1; r <= SEEK_REACH; r++) {
        for (var k = 0; k < 8 * r; k++) {
            let side = k / (2 * r);
            let along = k % (2 * r);
            var q = p + vec2<i32>(-r, -r);
            switch side {
                case 0: { q += vec2<i32>(along, 0); }
                case 1: { q += vec2<i32>(2 * r, along); }
                case 2: { q += vec2<i32>(2 * r - along, 2 * r); }
                default: { q += vec2<i32>(0, 2 * r - along); }
            }
            if (seekable(q, middle, b, st, cups)) {
                return q;
            }
        }
    }
    return vec2<i32>(-1);
}

fn seekable(q: vec2<i32>, middle: vec2<i32>, b: u32, st: Stamp, cups: bool) -> bool {
    if (!inWorld(q) || kindOf(cells[cellIndex(q)]) != KIND_GAS) {
        return false;
    }
    if (cups) {
        let s = unstampCell(q, st);
        if (all(s >= vec2<i32>(0)) && all(s < head.size) && shapes[head.off + u32(s.y * head.size.x + s.x)].y == CUP) {
            return false;
        }
    }
    return clearPath(middle, q, b) && claimFill(cellIndex(q));
}

// Shape cell i of the head body's grid (counting along its rows).
fn shapeCell(i: u32) -> vec2<i32> {
    let w = u32(head.size.x);
    return vec2<i32>(i32(i % w), i32(i / w));
}

// Its cells in its grid, head.size.x * head.size.y.
fn shapeCells() -> u32 {
    return u32(head.size.x * head.size.y);
}

fn poseOf(at: vec2<f32>, angle: f32) -> Pose {
    var q: Pose;
    q.at = at;
    q.rot = rotationOf(angle);
    q.com = head.com;
    q.size = head.size;
    q.off = head.off;
    return q;
}

// Three sums over the workgroup's threads at once, the same for all of them
// (for the lift pass): one tree, a third of the barriers.
struct Sums {
    a: vec4<f32>,
    b: vec4<f32>,
    c: vec4<f32>,
}

var<workgroup> reduceBuf3: array<Sums, BODY_THREADS>;

fn reduceSums(li: u32, v: Sums) -> Sums {
    reduceBuf3[li] = v;
    workgroupBarrier();
    for (var s = BODY_THREADS / 2u; s > 0u; s = s / 2u) {
        if (li < s) {
            let o = reduceBuf3[li + s];
            reduceBuf3[li].a += o.a;
            reduceBuf3[li].b += o.b;
            reduceBuf3[li].c += o.c;
        }
        workgroupBarrier();
    }
    let total = workgroupUniformLoad(&reduceBuf3[0]);
    workgroupBarrier();
    return total;
}

fn listedBodies(li: u32) -> u32 {
    if (li == 0u) {
        listed = atomicLoad(&bodies[counterSlot(C_LISTED)]);
    }
    return workgroupUniformLoad(&listed);
}

// Empties the body's shape and gives back its slot.
fn freeBody(b: u32, li: u32) {
    let n = u32(head.size.x * head.size.y);
    for (var i = li; i < n; i += BODY_THREADS) {
        shapes[head.off + i] = vec2<u32>(0u);
    }
    if (li == 0u) {
        giveSlot(b);
    }
}

// ------------------------------------------------------------------ lift

var<workgroup> lost: atomic<u32>;
var<workgroup> shrinking: u32;

// The air against the face of body cell p (of v) toward n, where gas is
// there (REWRITE_PLAN.md 4e-3): its pressure, in the body's table units
// (what pushes it is how that differs over its faces: liftBody), and its
// drag, worked out once for the body and that cell's air as 4e-2's for
// powders: their relative velocity at the face w goes to w / (1 + k (1 +
// μ)), k the cell's element's rate and μ the body's mass over the air's,
// the air taking the most of it (airsource.wgsl's pushIn: a falling body
// moves the air ahead of it out of its way) and the body the rest. None
// for one just found, which has no velocity of its own yet.
struct AirFace {
    open: bool,     // gas is there
    p: f32,
    drag: vec2<f32>,
}

fn airFace(p: vec2<i32>, n: vec2<i32>, at: vec2<f32>, v: vec4<u32>) -> AirFace {
    let q = p + n;
    let dims = airDims(P.size);
    if (!gasCellAt(q, dims)) {
        return AirFace(false, 0.0, vec2<f32>(0.0));
    }
    var force = vec2<f32>(0.0);
    let j = cellIndex(q);
    let air = elems[AIR].rho / 1000.0 * gasAmount(gasFrom(cells[gasIndex(cellCount(), j)], j));
    if (air > 0.0 && head.mass > 0.0) {
        let mu = head.mass / air;
        let w = head.vel + spinVel(head.spin, at) - windIn(q, n, dims);
        let change = w * (1.0 - 1.0 / (1.0 + elems[cellType(v)].drag * (1.0 + mu)));
        let du = change * mu / (1.0 + mu);
        pushIn(q, n, du, dims);
        force -= air * du;
    }
    return AirFace(true, inLiquidUnits(airAt(q, dims).x), force);
}

// Whether world cell q is inside the head body as last stamped: in its cup,
// or a cell of its own stamped elsewhere (MOVED_CAP). Its stamp is a
// bijection of the grid (three shears), so q is some one cell of its shape's
// grid. The gas in it pushes the body's walls apart, not the body along: a
// sealed crate's air, however pressed, moves it none. (Summed, the turned
// stamp's two cells of air in a corner of the Rafts crate that are no cell
// of its cup, a sealed zone of their own beside the cup the liquid's solve
// takes for its own, rose to 1.5 atm as a turn squeezed them and threw the
// crate at the tank's wall, where it shattered: one run in six.)
fn insideBody(q: vec2<i32>) -> bool {
    let s = unstampCell(q, head.stamp);
    if (any(s < vec2<i32>(0)) || any(s >= head.size)) {
        return false;
    }
    let shape = shapes[head.off + u32(s.y * head.size.x + s.x)];
    return shape.y == CUP || wordType(shape.x) != AIR;
}

fn liftBody(b: u32, li: u32) {
    if (li == 0u) {
        loadHead(b);
        atomicStore(&lost, 0u);
    }
    let flags = workgroupUniformLoad(&headFlags);
    if ((flags & B_ALIVE) == 0u) {
        return;
    }
    let fresh = (flags & B_NEW) != 0u;
    let q = poseOf(head.pos, head.angle);
    let key = bodyKey(head.frame, b);
    var force = vec4<f32>(0.0);  // the air's drag x, y and the masses' pull past its centre's; their torque and the cargo's weight's; the cargo's moment of inertia
    let centred = fieldAt(head.pos).xy;
    var push = vec4<f32>(0.0);   // a new body's momentum x, y, angular momentum, mass
    var stamped = vec4<f32>(0.0);  // its cells as stamped: their mass times where from the pose's centre, x, y; mass; the cargo's mass
    // Over its faces that meet gas: the air's pressure times the face's
    // normal (x, y), its torque, and the pressure; the normals, their
    // torques, and how many.
    var airP = vec4<f32>(0.0);
    var airN = vec4<f32>(0.0);
    let n = shapeCells();
    for (var i = li; i < n; i += BODY_THREADS) {
        let si = head.off + i;
        if (wordType(shapes[si].x) == AIR) {
            // Powder in its cup as last stamped is its cargo, however it
            // lies (cargoOf).
            if (!fresh && shapes[si].y == CUP) {
                let c = stampCell(shapeCell(i), head.stamp);
                if (inWorld(c) && kindOf(cells[cellIndex(c)]) == KIND_POWDER) {
                    let carried = cargoOf(c, q.at);
                    force += vec4<f32>(0.0, 0.0, carried.zw);
                    stamped.w += carried.x;
                }
            }
            continue;
        }
        let p = stampedAt(i, shapeCell(i));
        if (!inWorld(p)) {
            continue;
        }
        let j = cellIndex(p);
        // Not stamped here last step: another body had the cell, or there
        // was something in it that couldn't be moved.
        if (claimOf(j) != key) {
            continue;
        }
        let v = cells[j];
        if (!isBodyCell(v, b)) {
            // Burnt, melted, broken, dissolved, painted over: gone.
            shapes[si] = vec2<u32>(0u);
            atomicAdd(&lost, 1u);
            continue;
        }
        shapes[si] = vec2<u32>(v.x, v.y);
        let r = vec2<f32>(p) + 0.5 - q.at;
        let m = elems[cellType(v)].rho / 1000.0;
        stamped += vec4<f32>(m * r, m, 0.0);
        // The world's masses' pull on the cell past what freeMotion gives it
        // at the body's centre of mass: a field varying across it turns it
        // (the gravity-gradient torque; Goldstein, Classical Mechanics), the
        // end of a bar nearer a black hole pulled harder, turning it to
        // point at it. (Taken at its centre alone, a bar falling at 45°
        // toward a hole kept its angle where a rigid one turns 8° in 40
        // steps; REWRITE_PLAN.md P3-20.) Earth's pull, the same everywhere,
        // turns nothing.
        if (!fresh) {
            let tide = m * (fieldAt(vec2<f32>(p) + 0.5).xy - centred);
            force += vec4<f32>(tide, cross2(r, tide), 0.0);
        }
        if (fresh) {
            let u = cellVel(v);
            push += vec4<f32>(m * u, m * cross2(r, u), m);
        }
        // The powder resting on it outside its cup (REWRITE_PLAN.md P3-4),
        // each column of it on top of one of its cells, CARGO_REACH at most
        // (the cup's is counted above). (The pile's load, update.wgsl's
        // loadOut, passes only through powder at rest, which on a bobbing
        // boat its sand mostly isn't: its cargo came and went between none
        // and half of it.)
        let up = -downAt(p);
        var on = p + up;
        for (var k = 0; !fresh && k < CARGO_REACH && inWorld(on); k++) {
            if (inCupAs(on, head.stamp) || kindOf(cells[cellIndex(on)]) != KIND_POWDER) {
                break;
            }
            let carried = cargoOf(on, q.at);
            force += vec4<f32>(0.0, 0.0, carried.zw);
            stamped.w += carried.x;
            push.w += 1.0;  // (a new body's mass, which a new one carries none of)
            on += up;
        }
        for (var d = 0u; !fresh && d < 4u; d++) {
            let nd = select(vec2<i32>(0, i32(d & 1u) * 2 - 1), vec2<i32>(i32(d & 1u) * 2 - 1, 0), d < 2u);
            if (insideBody(p + nd)) {
                continue;
            }
            let at = r + 0.5 * vec2<f32>(nd);
            let f = airFace(p, nd, at, v);
            if (f.open) {
                let nf = vec2<f32>(nd);
                force += vec4<f32>(f.drag, cross2(at, f.drag), 0.0);
                airP += vec4<f32>(f.p * nf, f.p * cross2(at, nf), f.p);
                airN += vec4<f32>(nf, cross2(at, nf), 1.0);
            }
        }
    }
    // The air's pressure pushes it by how it differs over its faces: from its
    // mean over them, as gas gets round into the cracks where it touches
    // what it rests on (its cells meet that face to face, leaving no gas
    // between: the pressure of a sealed room pressed it into the wall it
    // leant on, as a suction cup).
    let airTotals = reduceSums(li, Sums(airP, airN, vec4<f32>(0.0)));
    let mean = airTotals.a.w / max(airTotals.b.w, 1.0);
    let totals = reduceSums(li, Sums(force, push, stamped));
    force = totals.a - vec4<f32>(airTotals.a.xyz - mean * airTotals.b.xyz, 0.0);
    push = totals.b;
    stamped = totals.c;

    // Lost cells: it is weighed again, and the parts pass looks whether it
    // is still in one piece (B_PARTS).
    if (li == 0u) {
        shrinking = atomicLoad(&lost);
    }
    let shrank = workgroupUniformLoad(&shrinking) > 0u;
    if (shrank || fresh) {
        if (!weigh(b, li, fresh, push)) {
            return;
        }
    }
    if (li == 0u) {
        // The air's torque about the centre of mass, wherever weigh moved
        // it to.
        let torque = force.z - cross2(head.pos - q.at, force.xy);
        setV(b, T_FORCE, force.xy);
        setF(b, T_FORCE + 2u, torque);
        setF(b, T_CARGO, stamped.w);
        setF(b, T_CARGO_INERTIA, force.w);
        // What the liquid did to it, as the liquid's pass found it this step
        // (the body was in its solve as it was stamped), taken and the sums
        // emptied for the next.
        var liquid = vec4<f32>(0.0);
        for (var k = 0u; k < 3u; k++) {
            liquid[k] = f32(bitcast<i32>(motion[liquidSlot(b, k)])) / RIGID_FIX;
            motion[liquidSlot(b, k)] = 0u;
        }
        let wet = (motion[liquidSlot(b, 3u)] & LIQUID_WET) != 0u;
        motion[liquidSlot(b, 3u)] = 0u;
        // Its turning, about the middle of its cells as stamped, which the
        // liquid pushed on, rather than its pose's centre of mass, a cell or
        // so from it, which the liquid summed it about: the stamp's being
        // off from the pose shifted the line its buoyancy pushes along by
        // as much, which turned it, one way more than the other (a floating
        // block rocked a third of a radian either way and hung on one side).
        if (stamped.z > 0.0) {
            let off = stamped.xy / stamped.z - (head.pos - q.at);
            liquid.z -= cross2(off, liquid.xy) * head.mass / head.inertia;
        }
        setV(b, T_LIQUID, liquid.xy);
        setV(b, T_LIQUID + 2u, liquid.zw);
        // Wet (held: rigidify's LIQUID_WET): all its cells are in the next
        // solve (fluid.wgsl's wetRegionAt).
        setU(b, T_FLAGS, (headFlags & ~(B_NEW | B_WET)) | select(0u, B_PARTS, shrank) | select(0u, B_WET, wet));
    }
}

// The body's mass, centre of mass and moment of inertia from its shape, and
// for a new one its velocity, from its cells' (its momentum, and its angular
// momentum about the new centre of mass). Its pose is moved with its centre
// of mass so its cells stay where they are. Each cell is a square of the
// element's density (in cells of water), with I = m / 6 about its own
// middle. False, and the body is gone, if it has no cells left.
fn weigh(b: u32, li: u32, fresh: bool, push: vec4<f32>) -> bool {
    let size = head.size;
    let middle = vec2<f32>(size) * 0.5;
    let n = u32(size.x * size.y);
    var sums = vec4<f32>(0.0);  // mass, mass-weighted x, y from the middle of the grid, m r^2
    var count = 0.0;
    for (var i = li; i < n; i += BODY_THREADS) {
        let s = shapes[head.off + i];
        if (wordType(s.x) == AIR) {
            continue;
        }
        let m = elems[wordType(s.x)].rho / 1000.0;
        let d = vec2<f32>(f32(i % u32(size.x)), f32(i / u32(size.x))) + 0.5 - middle;
        sums += vec4<f32>(m, m * d, m * dot(d, d));
        count += 1.0;
    }
    let totals = reduceSums(li, Sums(sums, vec4<f32>(count, 0.0, 0.0, 0.0), vec4<f32>(0.0)));
    sums = totals.a;
    let cellsLeft = totals.b.x;
    if (cellsLeft < 0.5) {
        freeBody(b, li);
        return false;
    }
    if (li == 0u) {
        let mass = sums.x;
        let offset = sums.yz / mass;
        let com = middle + offset;
        let inertia = sums.w - mass * dot(offset, offset) + mass / 6.0;
        let q = poseOf(head.pos, head.angle);
        let at = head.pos + rotate(com - head.com, q.rot);
        var radius = 0.0;
        for (var k = 0u; k < 4u; k++) {
            let corner = vec2<f32>(f32(k & 1u), f32(k >> 1u)) * vec2<f32>(size) - com;
            radius = max(radius, length(corner));
        }
        if (fresh) {
            // Momentum and angular momentum were summed about the old pose.
            let total = max(push.w, 1e-6);
            let vel = push.xy / total;
            let spin = (push.z - cross2(at - head.pos, push.xy)) / inertia;
            setV(b, T_VEL, vel);
            setF(b, T_SPIN, spin);
            head.vel = vel;
            head.spin = spin;
        }
        setF(b, T_MASS, mass);
        setF(b, T_INERTIA, inertia);
        setV(b, T_COM, com);
        setF(b, T_RADIUS, radius);
        setU(b, T_CELLS, u32(cellsLeft));
        setV(b, T_POS, at);
        head.pos = at;
        head.com = com;
        head.mass = mass + tableF(b, T_CARGO);
        head.inertia = inertia + tableF(b, T_CARGO_INERTIA);
        head.cells = u32(cellsLeft);
    }
    workgroupBarrier();
    // Its cup (CUP).
    for (var i = li; i < n; i += BODY_THREADS) {
        let si = head.off + i;
        if (wordType(shapes[si].x) != AIR) {
            continue;
        }
        let s = vec2<i32>(i32(i % u32(size.x)), i32(i / u32(size.x)));
        let cup = cupWall(s, vec2<i32>(-1, 0)) && cupWall(s, vec2<i32>(1, 0)) && cupWall(s, vec2<i32>(0, 1));
        shapes[si] = vec2<u32>(0u, select(0u, CUP, cup));
    }
    workgroupBarrier();
    return true;
}

// The powder at c a body carries (its cargo, REWRITE_PLAN.md P3-4: it rides
// with the body, update.wgsl's support): its mass, which joins the body's,
// its weight's torque about at, the body's centre of mass (gravity on the
// whole falls with the body), and its moment of inertia about it.
fn cargoOf(c: vec2<i32>, at: vec2<f32>) -> vec4<f32> {
    let m = elems[cellType(cells[cellIndex(c)])].rho / 1000.0;
    let r = vec2<f32>(c) + 0.5 - at;
    return vec4<f32>(m, 0.0, cross2(r, m * pullAt(vec2<f32>(c) + 0.5)), m * dot(r, r));
}

// Whether one of the head body's cells lies from s along d in its shape's
// grid (for its cup).
fn cupWall(s: vec2<i32>, d: vec2<i32>) -> bool {
    for (var t = s + d; all(t >= vec2<i32>(0)) && all(t < head.size); t += d) {
        if (wordType(shapes[head.off + u32(t.y * head.size.x + t.x)].x) != AIR) {
            return true;
        }
    }
    return false;
}

// A body that has come apart (the parts pass) lets go of its cells: they
// stay loose, in no body, moving as they were (FREE_BODY), and the
// labelling next step finds the pieces they make and makes each a body. A cell its last stamp
// left out goes into the nearest cell of gas (seekGas), its gas into the
// air beside it, rather than going with the shape.

fn breakUp(b: u32, li: u32) {
    let n = shapeCells();
    // Its cells left out of its stamp first, while those in the world are
    // still its own, which seekGas's path from its middle goes through:
    // FREE_BODY, they are walls to it, and a cell it couldn't find room
    // for beyond them went with its shape.
    for (var i = li; i < n; i += BODY_THREADS) {
        let shape = shapes[head.off + i];
        if (wordType(shape.x) == AIR) {
            continue;
        }
        let p = stampedAt(i, shapeCell(i));
        if (inWorld(p) && isBodyCell(cells[cellIndex(p)], b)) {
            continue;
        }
        let q = seekGas(p, 1, b, head.stamp, false);
        if (q.x >= 0) {
            let j = cellIndex(q);
            dropGas(q, j);
            var cell = unpackCell(vec4<u32>(shape.x, shape.y, 0u, 0u));
            cell.flags |= FL_LOOSE;
            cell.vel = head.vel;
            cell.frac = vec2<f32>(FREE_BODY, f32(head.piece));
            cells[j] = packCell(cell);
        }
    }
    storageBarrier();
    workgroupBarrier();
    for (var i = li; i < n; i += BODY_THREADS) {
        if (wordType(shapes[head.off + i].x) == AIR) {
            continue;
        }
        let p = stampedAt(i, shapeCell(i));
        if (inWorld(p) && isBodyCell(cells[cellIndex(p)], b)) {
            let j = cellIndex(p);
            var cell = unpackCell(cells[j]);
            cell.frac = vec2<f32>(FREE_BODY, f32(head.piece));
            cells[j] = packCell(cell);
        }
    }
    workgroupBarrier();
    freeBody(b, li);
}

@compute @workgroup_size(256)
fn lift(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let count = listedBodies(li);
    for (var k = wid.x; k < count; k += groups.x) {
        liftBody(motion[listSlot(k)], li);
        workgroupBarrier();
    }
}

// ------------------------------------------------------------------ parts
//
// Whether a body that lost cells this step (B_PARTS, from lift) is still in
// one piece: its shape's cells, 8-connected as a piece's are (init), are
// filled from its first one, in bitmaps in workgroup memory (each row of
// its grid in whole words, a cell a bit), until the fill grows no more. A
// cell it didn't reach is in another piece, and the body breaks up (the
// labelling next step makes each piece a body). (Lift tested each cell lost
// by its eight neighbours, the others lost that step already gone: a black
// hole eating across a bar took the last four cells of its top row in a
// step, each with one cell of the bar beside it or none, and the two halves
// went on as one body about the hole, its centre of mass in it.)

var<workgroup> partsFull: array<u32, PARTS_WORDS>;
var<workgroup> partsReached: array<atomic<u32>, PARTS_WORDS>;
var<workgroup> partsFirst: atomic<u32>;
var<workgroup> partsGrew: atomic<u32>;
var<workgroup> partsMore: u32;       // whether it grew (or has cells it didn't reach), for all of them
var<workgroup> partsSize: vec3<u32>;  // words a row, rows, words

// The cells reached in word x of row y, with those beside them along the
// row (across the word's ends too); none outside the grid.
fn partsRow(y: i32, x: u32, size: vec3<u32>) -> u32 {
    if (y < 0 || y >= i32(size.y)) {
        return 0u;
    }
    let at = u32(y) * size.x + x;
    let r = atomicLoad(&partsReached[at]);
    var d = r | (r << 1u) | (r >> 1u);
    if (x > 0u) {
        d |= atomicLoad(&partsReached[at - 1u]) >> 31u;
    }
    if (x + 1u < size.x) {
        d |= atomicLoad(&partsReached[at + 1u]) << 31u;
    }
    return d;
}

// Whether any thread grew the fill (or found a cell it didn't reach), for
// the whole workgroup, and none again.
fn partsAny(li: u32) -> bool {
    workgroupBarrier();
    if (li == 0u) {
        partsMore = atomicLoad(&partsGrew);
        atomicStore(&partsGrew, 0u);
    }
    return workgroupUniformLoad(&partsMore) != 0u;
}

fn partsBody(b: u32, li: u32) {
    if (li == 0u) {
        headFlags = tableU(b, T_FLAGS);
        if ((headFlags & (B_ALIVE | B_PARTS)) == (B_ALIVE | B_PARTS)) {
            loadHead(b);
            let wide = (u32(head.size.x) + 31u) / 32u;
            partsSize = vec3<u32>(wide, u32(head.size.y), wide * u32(head.size.y));
            atomicStore(&partsFirst, NONE);
            atomicStore(&partsGrew, 0u);
        }
    }
    let flags = workgroupUniformLoad(&headFlags);
    if ((flags & (B_ALIVE | B_PARTS)) != (B_ALIVE | B_PARTS)) {
        return;
    }
    let size = workgroupUniformLoad(&partsSize);
    var whole = size.z <= PARTS_WORDS;
    if (whole) {
        let w = u32(head.size.x);
        for (var k = li; k < size.z; k += BODY_THREADS) {
            let row = head.off + (k / size.x) * w;
            let x0 = (k % size.x) * 32u;
            var bits = 0u;
            for (var x = x0; x < min(x0 + 32u, w); x++) {
                if (wordType(shapes[row + x].x) != AIR) {
                    bits |= 1u << (x - x0);
                }
            }
            partsFull[k] = bits;
            atomicStore(&partsReached[k], 0u);
            if (bits != 0u) {
                atomicMin(&partsFirst, k);
            }
        }
        workgroupBarrier();
        if (li == 0u) {
            let first = atomicLoad(&partsFirst);
            if (first != NONE) {
                atomicStore(&partsReached[first], partsFull[first] & (~partsFull[first] + 1u));
            }
        }
        workgroupBarrier();
        // Each round, each word takes in the cells beside those reached, in
        // its row and the rows either side (corners too), then the rest of
        // each run of cells along its row they lead into. What a round reads
        // of a word another thread is growing is its cells before or after:
        // either, the fill only grows, and it ends once a round grows none.
        loop {
            for (var k = li; k < size.z; k += BODY_THREADS) {
                let full = partsFull[k];
                let r = atomicLoad(&partsReached[k]);
                if (r == full) {
                    continue;
                }
                let y = i32(k / size.x);
                let x = k % size.x;
                var n = (r | partsRow(y - 1, x, size) | partsRow(y, x, size) | partsRow(y + 1, x, size)) & full;
                loop {
                    let m = (n | (n << 1u) | (n >> 1u)) & full;
                    if (m == n) {
                        break;
                    }
                    n = m;
                }
                if (n != r) {
                    atomicOr(&partsReached[k], n);
                    atomicStore(&partsGrew, 1u);
                }
            }
            if (!partsAny(li)) {
                break;
            }
        }
        for (var k = li; k < size.z; k += BODY_THREADS) {
            if ((partsFull[k] & ~atomicLoad(&partsReached[k])) != 0u) {
                atomicStore(&partsGrew, 1u);
            }
        }
        whole = !partsAny(li);
    }
    if (!whole) {
        breakUp(b, li);
        return;
    }
    if (li == 0u) {
        setU(b, T_FLAGS, flags & ~B_PARTS);
    }
}

@compute @workgroup_size(256)
fn parts(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let count = listedBodies(li);
    for (var k = wid.x; k < count; k += groups.x) {
        partsBody(motion[listSlot(k)], li);
        workgroupBarrier();
    }
}

// ----------------------------------------------------------------- contacts

// A body's contacts as its contact pass finds them, for its hardest
// landing's normal: per contact r (from the centre of mass) and normal; the
// gap (f32 bits) and how hard it lands (f32 bits; -1 against another body).
var<workgroup> contactAt: array<vec4<f32>, CONTACT_CAP>;
var<workgroup> contactGap: array<vec4<u32>, CONTACT_CAP>;
var<workgroup> contacts: atomic<u32>;
var<workgroup> impact: atomic<u32>;     // the hardest approach, speed squared as f32 bits
var<workgroup> overlaps: atomic<u32>;
var<workgroup> overlapCount: u32;
// The box the cells a body's new pose covers make: lowest x, y, highest x, y.
var<workgroup> boxSpan: array<atomic<i32>, 4>;
var<workgroup> stepState: vec4<f32>;    // the motion solved: vel x, y, spin, and how hard it landed
var<workgroup> nextPose: vec4<f32>;     // the pose it moves to: x, y, angle, -
var<workgroup> snapPose: vec4<f32>;     // the same squared up to the grid
var<workgroup> snapping: u32;
var<workgroup> impactNormal: vec2<f32>;

// A body's velocity and spin this step before anything stops it: gravity,
// the air's push, and what the liquid's pressure did to it, which is its
// buoyancy (about the centre of buoyancy, which rights a floating body or
// capsizes it, depending on where its metacentre lies), the liquid's drag
// and the push of a current, and the liquid it has to set moving to move
// (its added mass), all at once: the liquid's as the liquid's solve, which
// moved the body with it, found it (fluid.wgsl's bodyPush; REWRITE_PLAN.md
// P3-22).
fn freeMotion(pos: vec2<f32>, vel0: vec2<f32>, spin0: f32, mass: f32, inertia: f32, force: vec4<f32>, liquid: vec4<f32>) -> vec3<f32> {
    let vel = vel0 + pullAt(pos) + force.xy / mass + liquid.xy;
    let spin = spin0 + force.z / inertia + liquid.z;
    return vec3<f32>(clamp(vel, vec2<f32>(-MAX_MOVE), vec2<f32>(MAX_MOVE)), spin);
}

// Where the middle of the shape cell body o has stamped at world cell q is,
// at the pose it was stamped at: within about a cell of q's middle.
fn bodyCellAt(o: u32, q: vec2<i32>) -> vec2<f32> {
    let at = tableV(o, T_POS);
    let rot = rotationOf(tableF(o, T_ANGLE));
    let com = tableV(o, T_COM);
    let s = vec2<f32>(unstampCell(q, tableStamp(o)));
    return at + rotate(s + 0.5 - com, rot);
}

// How far the middle c of a body cell can move along n before it touches
// what is at q: a cell that stands still, CONTACT_DIST from its middle, or
// another body's cell, taken as round like this one's, a cell apart. Both
// bodies measure it the same way, from the middles of their cells.
fn clearance(c: vec2<f32>, q: vec2<i32>, kind: u32, n: vec2<f32>) -> f32 {
    if (kind == OB_BODY) {
        let o = bodyOf(cells[cellIndex(q)]);
        return dot(c - bodyCellAt(o, q), n) - BODY_DIST;
    }
    return dot(c - vec2<f32>(q) - 0.5, n) - CONTACT_DIST;
}

// Finds what body cell p (whose middle is at c, moving at u) touches or
// will run into this step, and adds the contact. Its 8 neighbours give the
// normal, from the occupancy gradient of what is there (axis neighbours
// weigh twice the diagonal ones), and the gap, the least distance along it;
// moving fast, it looks on ahead along its way too.
//
// Another body moves at its own velocity, and gives way as much as its mass
// and moment of inertia let it (split between this body's contacts, as this
// body's are): each of the two works out the same impulse in all and takes
// its own share, so they push each other apart and share their momentum.
// But stacks are solved from the ground up (shock propagation: Guendelman,
// Bridson & Fedkiw 2003, "Nonconvex rigid bodies with stacking", ACM TOG
// 22(3)): a body resting on something that stands still, directly or
// through a stack of others, is taken to stand still by one landing on it
// from above, and it pays no heed to that one in turn.
fn findContact(p: vec2<i32>, c: vec2<f32>, u: vec2<f32>, b: u32) {
    var grad = vec2<f32>(0.0);
    var kinds = 0u;
    var otherCell = vec2<i32>(-1);
    var inside = true;
    var around = 0u;
    for (var d = 0u; d < 9u; d++) {
        let o = vec2<i32>(i32(d % 3u) - 1, i32(d / 3u) - 1);
        if (d == 4u) {
            continue;
        }
        let q = p + o;
        var kind = OB_STATIC;
        if (inWorld(q)) {
            let v = cells[cellIndex(q)];
            inside = inside && isBodyCell(v, b);
            kind = obstacleOf(v, b);
        } else {
            inside = false;
        }
        // Two bits per neighbour, for the gap below.
        around |= kind << (2u * d);
        if (kind == OB_NONE) {
            continue;
        }
        let weight = select(1.0, 2.0, o.x == 0 || o.y == 0);
        grad -= weight * vec2<f32>(o);
        kinds |= kind;
        if (kind == OB_BODY) {
            otherCell = q;
        }
    }
    // Only its outside cells touch anything.
    if (inside) {
        return;
    }
    // The normal from the ring past those too, weighed alike by distance
    // (2 / |o|²): a slope of stairs reads as its slope, not as a tread here
    // and a corner there (a block slid down stairs of 2 in 3 at a speed of
    // its own, bumping, and stuck on them at 33.7°, where μ 0.6 slides it).
    if (kinds != 0u) {
        for (var d = 0u; d < 25u; d++) {
            let o = vec2<i32>(i32(d % 5u) - 2, i32(d / 5u) - 2);
            if (max(abs(o.x), abs(o.y)) < 2) {
                continue;
            }
            let q = p + o;
            if (!inWorld(q) || obstacleOf(cells[cellIndex(q)], b) != OB_NONE) {
                grad -= 2.0 / f32(dot(o, o)) * vec2<f32>(o);
            }
        }
    }
    var n = vec2<f32>(0.0);
    var gap = 0.0;
    if (kinds != 0u && dot(grad, grad) > 1e-6) {
        n = normalize(grad);
        gap = 1e9;
        for (var d = 0u; d < 9u; d++) {
            let o = vec2<i32>(i32(d % 3u) - 1, i32(d / 3u) - 1);
            let kind = (around >> (2u * d)) & 3u;
            if (kind != OB_NONE) {
                gap = min(gap, clearance(c, p + o, kind, n));
            }
        }
    } else {
        // Nothing beside it (or it is wedged in all round): ahead, as far
        // as it gets this step, along its way and along each axis it moves
        // on (what it meets coming from the side while both fall lies
        // across, not along its way); the nearest it meets.
        kinds = 0u;
        gap = 1e9;
        for (var ray = 0u; ray < 3u; ray++) {
            var way = u;
            if (ray > 0u) {
                way[2u - ray] = 0.0;
            }
            let speed = length(way);
            if (speed < 0.5) {
                continue;
            }
            let dir = way / speed;
            for (var k = 1; k <= i32(ceil(speed)) + 1; k++) {
                let t = vec2<i32>(floor(c + dir * f32(k)));
                if (inWorld(t) && isBodyCell(cells[cellIndex(t)], b)) {
                    // Its own cells this way: what it meets, they meet first.
                    if (any(t != p)) {
                        break;
                    }
                    continue;
                }
                let kind = obstacleKind(t, b);
                if (kind != OB_NONE) {
                    let clear = clearance(c, t, kind, -dir);
                    if (clear < gap) {
                        n = -dir;
                        gap = clear;
                        kinds = kind;
                        otherCell = t;
                    }
                    break;
                }
            }
        }
        if (kinds == 0u) {
            return;
        }
    }
    // Against something that stands still: how hard it lands, if it does
    // this step (-1 against another body), and against another body, where
    // on that one it touches.
    let still = (kinds & OB_STATIC) != 0u;
    var landing = -1.0;
    var other = 0xffffu;
    var rb = vec2<f32>(0.0);
    if (still) {
        let approach = -dot(u, n);
        landing = 0.0;
        if (approach > max(gap, 0.0)) {
            landing = approach;
        }
    } else {
        other = bodyOf(cells[cellIndex(otherCell)]);
        rb = bodyCellAt(other, otherCell) - tableV(other, T_POS);
    }
    let k = atomicAdd(&contacts, 1u);
    if (k < CONTACT_CAP) {
        contactAt[k] = vec4<f32>(c - head.pos, n);
        contactGap[k] = vec4<u32>(bitcast<u32>(gap), 0u, 0u, bitcast<u32>(landing));
        // How hard it lands, by the contacts kept, whose normal goes with
        // it (contactBody): by all, the hardest past CONTACT_CAP had its
        // speed and the last landing's normal, or none.
        if (landing > 0.0) {
            atomicMax(&impact, bitcast<u32>(landing * landing));
        }
    }
    // Into this step's list for all bodies (jointSolve).
    let g = atomicAdd(&bodies[counterSlot(C_CONTACTS)], 1u);
    if (g < BODY_CONTACTS) {
        let r = c - head.pos;
        motion[contactSlot(g, 0u)] = b | (other << 16u);
        motion[contactSlot(g, 1u)] = bitcast<u32>(r.x);
        motion[contactSlot(g, 2u)] = bitcast<u32>(r.y);
        motion[contactSlot(g, 3u)] = bitcast<u32>(rb.x);
        motion[contactSlot(g, 4u)] = bitcast<u32>(rb.y);
        motion[contactSlot(g, 5u)] = bitcast<u32>(n.x);
        motion[contactSlot(g, 6u)] = bitcast<u32>(n.y);
        motion[contactSlot(g, 7u)] = bitcast<u32>(gap);
        motion[contactSlot(g, 8u)] = bitcast<u32>(landing);
        motion[contactSlot(g, 9u)] = 0u;
        motion[contactSlot(g, 10u)] = 0u;
        motion[contactSlot(g, 11u)] = 0u;
    }
}

// In how many of the cells the shape would be stamped over at pose q there
// is something standing still (or the world's edge), or another body as it
// stands now.
fn overlapsAt(at: vec2<f32>, angle: f32, b: u32, li: u32) -> u32 {
    if (li == 0u) {
        atomicStore(&overlaps, 0u);
    }
    workgroupBarrier();
    // By the cells its cells' middles are in at that pose, as the contacts
    // go (its stamp may be a cell or so off, and a cell it can't stamp goes
    // elsewhere: MOVED_CAP).
    let q = poseOf(at, angle);
    let n = shapeCells();
    for (var i = li; i < n; i += BODY_THREADS) {
        if (wordType(shapes[head.off + i].x) != AIR && obstacleKind(vec2<i32>(floor(cellWorld(shapeCell(i), q))), b) != OB_NONE) {
            atomicAdd(&overlaps, 1u);
        }
    }
    workgroupBarrier();
    if (li == 0u) {
        overlapCount = atomicLoad(&overlaps);
    }
    return workgroupUniformLoad(&overlapCount);
}

// How many of the head body b's cells are in the world as it stands.
fn cellsInWorld(b: u32, li: u32) -> u32 {
    if (li == 0u) {
        atomicStore(&overlaps, 0u);
    }
    workgroupBarrier();
    let n = shapeCells();
    for (var i = li; i < n; i += BODY_THREADS) {
        if (wordType(shapes[head.off + i].x) == AIR) {
            continue;
        }
        let p = stampedAt(i, shapeCell(i));
        if (inWorld(p) && isBodyCell(cells[cellIndex(p)], b)) {
            atomicAdd(&overlaps, 1u);
        }
    }
    workgroupBarrier();
    if (li == 0u) {
        overlapCount = atomicLoad(&overlaps);
    }
    return workgroupUniformLoad(&overlapCount);
}

// Its motion before anything stops it (freeMotion), as the contacts start
// from it (T_NEXTVEL), and its contacts, into the list for all bodies;
// how hard it lands, and on what normal (stamp breaks what it hits).
fn contactBody(b: u32, li: u32) {
    if (li == 0u) {
        loadHead(b);
        atomicStore(&contacts, 0u);
        atomicStore(&impact, 0u);
    }
    let flags = workgroupUniformLoad(&headFlags);
    if ((flags & B_ALIVE) == 0u) {
        return;
    }
    let free = freeMotion(head.pos, head.vel, head.spin, head.mass, head.inertia, head.force, head.liquid);
    let q = poseOf(head.pos, head.angle);
    let n = shapeCells();
    for (var i = li; i < n; i += BODY_THREADS) {
        if (wordType(shapes[head.off + i].x) == AIR) {
            continue;
        }
        let s = shapeCell(i);
        let p = stampedAt(i, s);
        if (!inWorld(p) || !isBodyCell(cells[cellIndex(p)], b)) {
            continue;
        }
        // Around the cell its middle is in at the pose, which its stamp may
        // be a cell or so off.
        let c = cellWorld(s, q);
        findContact(vec2<i32>(floor(c)), c, free.xy + spinVel(free.z, c - head.pos), b);
    }
    workgroupBarrier();
    let found = min(atomicLoad(&contacts), CONTACT_CAP);
    let hard = atomicLoad(&impact);
    for (var k = li; k < found; k += BODY_THREADS) {
        let landing = bitcast<f32>(contactGap[k].w);
        if (hard != 0u && landing > 0.0 && bitcast<u32>(landing * landing) == hard) {
            setV(b, T_IMPACT + 1u, contactAt[k].zw);
        }
    }
    if (li == 0u) {
        setV(b, T_NEXTVEL, free.xy);
        setF(b, T_NEXTVEL + 2u, free.z);
        setV(b, T_PUSH, vec2<f32>(0.0));
        setF(b, T_PUSH + 2u, 0.0);
        setU(b, T_SUPPORT, 0u);
        setU(b, T_COUNT, atomicLoad(&contacts));
        setF(b, T_IMPACT, bitcast<f32>(hard));
    }
}

@compute @workgroup_size(256)
fn contact(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let count = listedBodies(li);
    for (var k = wid.x; k < count; k += groups.x) {
        contactBody(motion[listSlot(k)], li);
        workgroupBarrier();
    }
}

// ------------------------------------------------------------- jointSolve
//
// All the bodies' contacts at once, in one workgroup (REWRITE_PLAN.md P3-2;
// audit/07 §5.4): SOLVE_ITERATIONS rounds of Jacobi over them, each contact
// working out its impulse from the two bodies' velocities as the last round
// left them and putting it on both, equal and opposite (Newton's third law
// by construction: each body solved its own before, against a guess at the
// other's motion, and a chain of bodies passed a push on one a step), each
// body's mass split between its contacts as it counts them (Tonge,
// Benevolenskiy & Voroshilov 2012). A contact's normal impulse may close
// its gap, no more (speculative: Firth 2011); friction FRICTION of it at
// most; then as many rounds of the push out of overlaps (PUSH_OUT of the
// overlap a step), kept apart from the velocity so it doesn't make them
// bounce. What a round changes is summed per body in fixed point (no float
// atomics). With it, shock propagation's flags (read across workgroups as
// they were written) are gone: a stack is solved whole.

var<workgroup> jointSums: array<atomic<i32>, 3072>;  // 3 x MAX_BODIES: what a round changes, x, y, spin
var<workgroup> jointCount: u32;
// The bodies resting on what stands still (on it, or on a body that is), a
// bit a body: those on it, then by turns those resting as of the round
// before and as of this one (jointSolve).
const HELD_ROUNDS: u32 = 8u;
var<workgroup> stillBits: array<atomic<u32>, 32>;
var<workgroup> restingBits: array<array<atomic<u32>, 32>, 2>;
const_assert MAX_BODIES <= 32u * 32u;

fn restingIn(which: u32, b: u32) -> bool {
    return (atomicLoad(&restingBits[which][b / 32u]) & (1u << (b % 32u))) != 0u;
}

const JOINT_FIX: f32 = 1048576.0;
const_assert 3u * MAX_BODIES <= 3072u;

struct JointContact {
    a: u32,
    b: u32,       // 0xffff: something that stands still
    ra: vec2<f32>,
    rb: vec2<f32>,
    n: vec2<f32>,
    gap: f32,
    landing: f32,
}

fn jointContact(k: u32) -> JointContact {
    var c: JointContact;
    let w = motion[contactSlot(k, 0u)];
    c.a = w & 0xffffu;
    c.b = w >> 16u;
    c.ra = vec2<f32>(bitcast<f32>(motion[contactSlot(k, 1u)]), bitcast<f32>(motion[contactSlot(k, 2u)]));
    c.rb = vec2<f32>(bitcast<f32>(motion[contactSlot(k, 3u)]), bitcast<f32>(motion[contactSlot(k, 4u)]));
    c.n = vec2<f32>(bitcast<f32>(motion[contactSlot(k, 5u)]), bitcast<f32>(motion[contactSlot(k, 6u)]));
    c.gap = bitcast<f32>(motion[contactSlot(k, 7u)]);
    c.landing = bitcast<f32>(motion[contactSlot(k, 8u)]);
    return c;
}

fn jointAdd(b: u32, d: vec3<f32>) {
    let q = vec3<i32>(round(d * JOINT_FIX));
    if (any(q != vec3<i32>(0))) {
        atomicAdd(&jointSums[3u * b], q.x);
        atomicAdd(&jointSums[3u * b + 1u], q.y);
        atomicAdd(&jointSums[3u * b + 2u], q.z);
    }
}

// A body's velocity (pass 0) or push (1), as rounds so far have it.
fn jointState(b: u32, which: u32) -> vec3<f32> {
    let w = select(T_NEXTVEL, T_PUSH, which == 1u);
    return vec3<f32>(tableV(b, w), tableF(b, w + 2u));
}

// How much a body gives way at r along d, per unit of impulse, its mass split
// count ways.
fn jointGive(b: u32, r: vec2<f32>, d: vec2<f32>) -> f32 {
    let rd = cross2(r, d);
    return f32(max(tableU(b, T_COUNT), 1u)) * (1.0 / massOf(b) + rd * rd / inertiaOf(b));
}

@compute @workgroup_size(256)
fn jointSolve(@builtin(local_invocation_index) li: u32) {
    if (li == 0u) {
        jointCount = min(atomicLoad(&bodies[counterSlot(C_CONTACTS)]), BODY_CONTACTS);
    }
    let found = workgroupUniformLoad(&jointCount);
    if (found == 0u) {
        return;
    }
    // Each body's mass split over the rows it is in, as either side
    // (Tonge et al.), not over the contacts it found itself: an interface
    // between two bodies is listed by both, and each row pushed them apart
    // as if it were the only one, twice what stops them. Two bars meeting
    // end-on swapped their speeds each round, and after the eight rounds
    // were back where they started, the one going on into the other.
    for (var w = li; w < 3u * MAX_BODIES; w += 256u) {
        atomicStore(&jointSums[w], 0);
    }
    workgroupBarrier();
    for (var k = li; k < found; k += 256u) {
        let c = jointContact(k);
        atomicAdd(&jointSums[3u * c.a], 1);
        if (c.b != 0xffffu) {
            atomicAdd(&jointSums[3u * c.b], 1);
        }
    }
    workgroupBarrier();
    for (var b = li; b < MAX_BODIES; b += 256u) {
        let rows = atomicLoad(&jointSums[3u * b]);
        if (rows > 0) {
            setU(b, T_COUNT, u32(rows));
        }
    }
    storageBarrier();
    workgroupBarrier();
    for (var which = 0u; which < 2u; which++) {
        for (var it = 0u; it < SOLVE_ITERATIONS; it++) {
            for (var w = li; w < 3u * MAX_BODIES; w += 256u) {
                atomicStore(&jointSums[w], 0);
            }
            workgroupBarrier();
            for (var k = li; k < found; k += 256u) {
                let c = jointContact(k);
                let still = c.b == 0xffffu;
                let sa = jointState(c.a, which);
                var sb = vec3<f32>(0.0);
                if (!still) {
                    sb = jointState(c.b, which);
                }
                let rel = sa.xy + spinVel(sa.z, c.ra) - select(sb.xy + spinVel(sb.z, c.rb), vec2<f32>(0.0), still);
                let kn = jointGive(c.a, c.ra, c.n) + select(jointGive(c.b, c.rb, c.n), 0.0, still);
                var j = vec2<f32>(0.0);
                if (which == 0u) {
                    let sums = vec2<f32>(bitcast<f32>(motion[contactSlot(k, 9u)]), bitcast<f32>(motion[contactSlot(k, 10u)]));
                    let t = vec2<f32>(-c.n.y, c.n.x);
                    let normal = max(sums.x + (-max(c.gap, 0.0) - dot(rel, c.n)) / kn, 0.0);
                    let kt = jointGive(c.a, c.ra, t) + select(jointGive(c.b, c.rb, t), 0.0, still);
                    let limit = FRICTION * normal;
                    let across = clamp(sums.y - dot(rel, t) / kt, -limit, limit);
                    motion[contactSlot(k, 9u)] = bitcast<u32>(normal);
                    motion[contactSlot(k, 10u)] = bitcast<u32>(across);
                    j = (normal - sums.x) * c.n + (across - sums.y) * t;
                } else {
                    let sum = bitcast<f32>(motion[contactSlot(k, 11u)]);
                    let out = max(sum + (-c.gap * PUSH_OUT - dot(rel, c.n)) / kn, 0.0);
                    motion[contactSlot(k, 11u)] = bitcast<u32>(out);
                    j = (out - sum) * c.n;
                }
                jointAdd(c.a, vec3<f32>(j / massOf(c.a), cross2(c.ra, j) / inertiaOf(c.a)));
                if (!still) {
                    jointAdd(c.b, vec3<f32>(-j / massOf(c.b), -cross2(c.rb, j) / inertiaOf(c.b)));
                }
            }
            workgroupBarrier();
            for (var b = li; b < MAX_BODIES; b += 256u) {
                let d = vec3<f32>(f32(atomicLoad(&jointSums[3u * b])), f32(atomicLoad(&jointSums[3u * b + 1u])),
                    f32(atomicLoad(&jointSums[3u * b + 2u]))) / JOINT_FIX;
                if (any(d != vec3<f32>(0.0))) {
                    let w = select(T_NEXTVEL, T_PUSH, which == 1u);
                    setV(b, w, tableV(b, w) + d.xy);
                    setF(b, w + 2u, tableF(b, w + 2u) + d.z);
                }
            }
            storageBarrier();
            workgroupBarrier();
        }
    }
    // Resting on something that stands still, pushing on it from below
    // (against the pull at its middle); and resting on that or on a body
    // that is, a body further up a stack each round (the round before's
    // read, so the same however the threads go). Held, so two steps
    // running, and let go when not so two steps running, a body is a wall
    // to the liquid's solve, the liquid's pressure on it summed for its push
    // (fluid.wgsl's isHeld): in the solve as liquid of its density, its
    // faces on what holds it moving at its velocity and the pull, which the
    // contacts stop, it took in the liquid as a drain does, the water over
    // it drawn down at 0.16 to 0.33 cells a step for as long as it rested
    // (REWRITE_PLAN.md P3-19). (Held and let go with each touch, a hull
    // afloat in shallow water grazing the bottom changed every step or two,
    // and the water moving with it met a wall at each change, which threw it
    // a cell a step sideways.)
    for (var w = li; w < 32u; w += 256u) {
        atomicStore(&stillBits[w], 0u);
        atomicStore(&restingBits[0][w], 0u);
    }
    workgroupBarrier();
    for (var k = li; k < found; k += 256u) {
        let c = jointContact(k);
        if (c.b != 0xffffu || bitcast<f32>(motion[contactSlot(k, 9u)]) <= 0.0) {
            continue;
        }
        let up = dot(c.n, pullDirAt(tableV(c.a, T_POS)));
        if (c.landing >= 0.0 && up < -0.5) {
            atomicOr(&stillBits[c.a / 32u], 1u << (c.a % 32u));
        }
        if (up < -0.5) {
            atomicOr(&restingBits[0][c.a / 32u], 1u << (c.a % 32u));
        }
    }
    workgroupBarrier();
    for (var round = 0u; round < HELD_ROUNDS; round++) {
        let was = round % 2u;
        let now = 1u - was;
        for (var w = li; w < 32u; w += 256u) {
            atomicStore(&restingBits[now][w], atomicLoad(&restingBits[was][w]));
        }
        workgroupBarrier();
        for (var k = li; k < found; k += 256u) {
            let c = jointContact(k);
            if (c.b == 0xffffu || bitcast<f32>(motion[contactSlot(k, 9u)]) <= 0.0) {
                continue;
            }
            let up = dot(c.n, pullDirAt(tableV(c.a, T_POS)));
            if (up < -0.5 && restingIn(was, c.b)) {
                atomicOr(&restingBits[now][c.a / 32u], 1u << (c.a % 32u));
            }
            if (up > 0.5 && restingIn(was, c.a)) {
                atomicOr(&restingBits[now][c.b / 32u], 1u << (c.b % 32u));
            }
        }
        workgroupBarrier();
    }
    for (var b = li; b < MAX_BODIES; b += 256u) {
        let still = (atomicLoad(&stillBits[b / 32u]) & (1u << (b % 32u))) != 0u;
        let resting = restingIn(HELD_ROUNDS % 2u, b);
        let flags = tableU(b, T_FLAGS);
        let was = (flags & B_HELD) != 0u;
        let rested = (flags & B_RESTED) != 0u;
        let held = (resting && (rested || was)) || (was && rested);
        if (still || resting || held) {
            setU(b, T_SUPPORT, select(0u, T_SUPPORT_STILL, still) | select(0u, T_SUPPORT_HELD, held)
                | select(0u, T_SUPPORT_RESTING, resting));
        }
    }
}

// ------------------------------------------------------------- integrate

// How body b gives way to a push in the liquid's next solve, as the
// contacts that pressed this step let it (T_MOBILITY; fluid.wgsl's
// giveWay; REWRITE_PLAN.md P3-23): its mobility M^-1 (velocity x, y and
// spin from a force x, y and a torque), less what those contacts hold it
// from, M^-1/2 P M^-1/2, P taking out of a velocity scaled by M^1/2 the
// directions each such contact holds: along its normal, and across it where
// friction held it (inside its cone), each a row (its direction and arm)
// scaled by M^-1/2, made a unit. The rows' outer products summed in fixed
// point (MOBILITY_FIX), alike whatever order the contacts were listed in;
// the directions held those of that sum's eigenvectors (Jacobi's rotations)
// over HELD_EIGEN. To the liquid's solve, which ran before the contacts, a
// body was free whatever held it: a piece resting on another was taken to
// fall under its weight and the water's push, and the water flowed into
// where it would have gone, which the contacts then stopped it going (in
// Rubble a pile of 1000 cells' worth of water and more packed onto pieces of
// a heap by step 150, 9000 cells of the tank empty under the water).
const MOBILITY_FIX: f32 = 65536.0;
const HELD_EIGEN: f32 = 0.05;
// Friction's share of its limit (FRICTION times the normal) under which a
// contact held across it rather than slid.
const STICK: f32 = 0.99;
var<workgroup> heldRows: array<atomic<i32>, 6>;

fn addHeldRow(d: vec3<f32>) {
    let l = length(d);
    if (l <= 0.0) {
        return;
    }
    let e = d / l;
    let o = round(vec3<f32>(e.x * e.x, e.x * e.y, e.x * e.z) * MOBILITY_FIX);
    let p = round(vec3<f32>(e.y * e.y, e.y * e.z, e.z * e.z) * MOBILITY_FIX);
    atomicAdd(&heldRows[0], i32(o.x));
    atomicAdd(&heldRows[1], i32(o.y));
    atomicAdd(&heldRows[2], i32(o.z));
    atomicAdd(&heldRows[3], i32(p.x));
    atomicAdd(&heldRows[4], i32(p.y));
    atomicAdd(&heldRows[5], i32(p.z));
}

// The projection onto the directions symmetric s holds (its eigenvectors
// over HELD_EIGEN), by Jacobi's rotations.
fn heldProjection(s: mat3x3<f32>) -> mat3x3<f32> {
    var a = s;
    var v = mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0));
    for (var sweep = 0; sweep < 8; sweep++) {
        for (var k = 0; k < 3; k++) {
            let p = select(0, 1, k == 2);
            let q = select(2, 1, k == 0);
            if (abs(a[p][q]) < 1e-9) {
                continue;
            }
            let theta = 0.5 * atan2(2.0 * a[p][q], a[q][q] - a[p][p]);
            let c = cos(theta);
            let sn = sin(theta);
            var r = mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0));
            // (Columns first: the rotation's row p, column q is sn.)
            r[p][p] = c;
            r[q][q] = c;
            r[q][p] = sn;
            r[p][q] = -sn;
            a = transpose(r) * a * r;
            v = v * r;
        }
    }
    var held = mat3x3<f32>();
    for (var k = 0; k < 3; k++) {
        if (a[k][k] > HELD_EIGEN) {
            held += outerProduct(v[k], v[k]);
        }
    }
    return held;
}

fn outerProduct(x: vec3<f32>, y: vec3<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(x * y.x, x * y.y, x * y.z);
}

fn workMobility(b: u32, li: u32) {
    if (li < 6u) {
        atomicStore(&heldRows[li], 0);
    }
    workgroupBarrier();
    let found = min(atomicLoad(&bodies[counterSlot(C_CONTACTS)]), BODY_CONTACTS);
    let scale = vec3<f32>(inverseSqrt(massOf(b)), inverseSqrt(massOf(b)), inverseSqrt(inertiaOf(b)));
    for (var k = li; k < found; k += BODY_THREADS) {
        let c = jointContact(k);
        let normal = bitcast<f32>(motion[contactSlot(k, 9u)]);
        if ((c.a != b && c.b != b) || normal <= 0.0) {
            continue;
        }
        let r = select(c.rb, c.ra, c.a == b);
        addHeldRow(vec3<f32>(c.n, cross2(r, c.n)) * scale);
        let across = bitcast<f32>(motion[contactSlot(k, 10u)]);
        if (abs(across) < STICK * FRICTION * normal) {
            let t = vec2<f32>(-c.n.y, c.n.x);
            addHeldRow(vec3<f32>(t, cross2(r, t)) * scale);
        }
    }
    workgroupBarrier();
    if (li == 0u) {
        let w = array<f32, 6>(
            f32(atomicLoad(&heldRows[0])), f32(atomicLoad(&heldRows[1])), f32(atomicLoad(&heldRows[2])),
            f32(atomicLoad(&heldRows[3])), f32(atomicLoad(&heldRows[4])), f32(atomicLoad(&heldRows[5])),
        );
        let s = mat3x3<f32>(vec3<f32>(w[0], w[1], w[2]), vec3<f32>(w[1], w[3], w[4]), vec3<f32>(w[2], w[4], w[5])) * (1.0 / MOBILITY_FIX);
        let free = mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0)) - heldProjection(s);
        let half = mat3x3<f32>(vec3<f32>(scale.x, 0.0, 0.0), vec3<f32>(0.0, scale.y, 0.0), vec3<f32>(0.0, 0.0, scale.z));
        let m = half * free * half;
        setF(b, T_MOBILITY, m[0][0]);
        setF(b, T_MOBILITY + 1u, m[1][0]);
        setF(b, T_MOBILITY + 2u, m[2][0]);
        setF(b, T_MOBILITY + 3u, m[1][1]);
        setF(b, T_MOBILITY + 4u, m[2][1]);
        setF(b, T_MOBILITY + 5u, m[2][2]);
        setU(b, T_MOBILITY_AT, P.frame);
    }
}

// Its velocity and push as jointSolve left them, clamped; its new pose,
// backed off if it would still overlap something; whether it has come to
// rest (and rejoins); and the cells its new pose covers, claimed.
fn integrateBody(b: u32, li: u32) {
    if (li == 0u) {
        loadHead(b);
    }
    let flags = workgroupUniformLoad(&headFlags);
    if ((flags & B_ALIVE) == 0u) {
        return;
    }
    workMobility(b, li);
    var vel = clamp(tableV(b, T_NEXTVEL), vec2<f32>(-MAX_MOVE), vec2<f32>(MAX_MOVE));
    let fastest = min(MAX_SPIN, MAX_MOVE / max(head.radius, 1.0));
    var spin = clamp(tableF(b, T_NEXTVEL + 2u), -fastest, fastest);
    let pushing = vec3<f32>(tableV(b, T_PUSH), tableF(b, T_PUSH + 2u));
    // Where it gets to, pushed out of any overlap as well. If it would
    // overlap something there after all (a contact missed, bodies pushed
    // together through a chain of them faster than the contacts pass the
    // push on), and in more than a few cells, and in more than where it is,
    // it is backed off, and slowed as much. The few aren't stamped this
    // step, and the contacts push it out of them: the steps of a slope would
    // otherwise stop a body rolling down it dead.
    // (The few as many as a quarter of its cells, at most OVERLAP_SLACK: a
    // body of a cell or two could otherwise take a pose wholly in a wall,
    // and come out past it, packed in with grains and other bodies where
    // its contacts found no face to hold it.)
    let slack = min(OVERLAP_SLACK, workgroupUniformLoad(&head).cells / 4u);
    let travel = vec4<f32>(vel + pushing.xy, spin + pushing.z, 0.0);
    var share = 1.0;
    let full = overlapsAt(head.pos + travel.xy, head.angle + travel.z, b, li);
    if (full > slack) {
        let now = overlapsAt(head.pos, head.angle, b, li);
        if (full > now) {
            let half = overlapsAt(head.pos + travel.xy * 0.5, head.angle + travel.z * 0.5, b, li);
            share = select(0.0, 0.5, half <= max(slack, now));
        }
    }
    // A body none of whose cells is in the world (each lost its cell to
    // another body's claim, or to something it couldn't move, and found no
    // gas near to go to instead: MOVED_CAP) is buried, and held where it is
    // by what it is in. It found no contacts, and overlapping as much
    // wherever it went, went as if nothing were there: through the floor
    // of a sealed box packed with snow and salt (scenes.js Frozen brine).
    if (cellsInWorld(b, li) == 0u) {
        share = 0.0;
    }
    vel *= share;
    spin *= share;

    if (li == 0u) {
        let support = tableU(b, T_SUPPORT);
        let restFlag = (support & T_SUPPORT_STILL) != 0u;
        // By how far it goes, its push out of overlaps too: the contacts can
        // leave it some speed into what it rests on, which the push takes
        // back (a block tilted on the world's bottom edge kept 0.08 cells a
        // step into it, never coming to rest, where it didn't move).
        let slow = length(travel.xy * share) < REST_SPEED && abs(travel.z * share) * head.radius < REST_SPEED;
        var rest = 0u;
        if (restFlag && slow) {
            rest = head.rest + 1u;
        }
        var newFlags = flags & ~(B_REJOIN | B_SUPPORTED | B_HELD | B_RESTED);
        if (restFlag) {
            newFlags |= B_SUPPORTED;
        }
        if ((support & T_SUPPORT_HELD) != 0u) {
            newFlags |= B_HELD;
        }
        if ((support & T_SUPPORT_RESTING) != 0u) {
            newFlags |= B_RESTED;
        }
        if (rest >= REST_STEPS) {
            newFlags |= B_REJOIN;
        }
        headFlags = newFlags;
        setU(b, T_FLAGS, newFlags);
        setU(b, T_REST, rest);
        nextPose = vec4<f32>(head.pos + travel.xy * share, wrapAngle(head.angle + travel.z * share), 0.0);
        // Coming to rest nearly square to the grid: squared up, its cells'
        // middles on the world's.
        let square = round(nextPose.z / QUARTER_TURN) * QUARTER_TURN;
        snapping = select(0u, 1u, rest >= REST_STEPS && abs(nextPose.z - square) < SNAP_ANGLE);
        let snapped = poseOf(nextPose.xy, square);
        let corner = cellWorld(vec2<i32>(0), snapped);
        snapPose = vec4<f32>(nextPose.xy + floor(corner) + 0.5 - corner, square, 0.0);
        setV(b, T_NEXTVEL, vel);
        setF(b, T_NEXTVEL + 2u, spin);
    }
    if (workgroupUniformLoad(&snapping) != 0u) {
        let at = snapPose;
        if (overlapsAt(at.xy, at.z, b, li) == 0u && li == 0u) {
            nextPose = at;
        }
    }
    workgroupBarrier();
    // It rejoins only with every cell stamped: with more overlapping what is
    // there than can go elsewhere (MOVED_CAP; one left out of the stamp would
    // be lost with its shape), it rests on as a body until the contacts have
    // pushed it clearer.
    if ((workgroupUniformLoad(&headFlags) & B_REJOIN) != 0u) {
        let pose = workgroupUniformLoad(&nextPose);
        if (overlapsAt(pose.xy, pose.z, b, li) > MOVED_CAP / 2u && li == 0u) {
            headFlags &= ~B_REJOIN;
            setU(b, T_FLAGS, headFlags);
        }
        workgroupBarrier();
    }
    // Claims the cells its new pose covers, and finds the box they make.
    let to = stampAt(nextPose.xy, nextPose.z, head.com, head.stamp.pivot);
    let key = bodyKey(P.frame, b);
    if (li == 0u) {
        atomicStore(&boxSpan[0], 0x7fffffff);
        atomicStore(&boxSpan[1], 0x7fffffff);
        atomicStore(&boxSpan[2], -0x7fffffff);
        atomicStore(&boxSpan[3], -0x7fffffff);
    }
    workgroupBarrier();
    let m = shapeCells();
    for (var i = li; i < m; i += BODY_THREADS) {
        if (wordType(shapes[head.off + i].x) == AIR) {
            continue;
        }
        let p = stampCell(shapeCell(i), to);
        if (inWorld(p)) {
            atomicMax(&bodies[claimSlot(cellIndex(p))], key);
            atomicMin(&boxSpan[0], p.x);
            atomicMin(&boxSpan[1], p.y);
            atomicMax(&boxSpan[2], p.x);
            atomicMax(&boxSpan[3], p.y);
        }
    }
    // And those its cargo goes to (stampBody's cargoTake picks the same
    // ones), what it leaves held for the exchange.
    var carried = 0u;
    for (var i = li; i < m && carried < CARGO_EACH; i += BODY_THREADS) {
        if (!cargoMoves(i, head.stamp, to)) {
            continue;
        }
        carried++;
        atomicMax(&bodies[claimSlot(cellIndex(stampCell(shapeCell(i), to)))], key);
        claimLeft(cellIndex(stampCell(shapeCell(i), head.stamp)));
    }
    workgroupBarrier();
    if (li == 0u) {
        setV(b, T_NEXT, nextPose.xy);
        setF(b, T_NEXT + 2u, nextPose.z);
        setI(b, T_NEWBOX, vec2<i32>(atomicLoad(&boxSpan[0]), atomicLoad(&boxSpan[1])));
        setI(b, T_NEWBOX + 2u, vec2<i32>(atomicLoad(&boxSpan[2]), atomicLoad(&boxSpan[3])));
    }
}

@compute @workgroup_size(256)
fn integrate(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let count = listedBodies(li);
    for (var k = wid.x; k < count; k += groups.x) {
        integrateBody(motion[listSlot(k)], li);
        workgroupBarrier();
    }
}

// ----------------------------------------------------------------- leaving

// Whether a cell the body leaves at p is in the liquid (liquid beside it on
// a face), and so the liquid's to fill: gas it moves into goes into the
// cells it leaves only where they are in the air, as it does through the
// air (the exchange), and with none there, into the air beside the body
// (displace). Matched by place alone, the air over a body rising out of the
// water went into the cells its floor left under the water; the water came
// in after it the next step, its air pushed into the cell's air source
// (fluid.wgsl's pushGasOut), which the carry takes in only where there is
// gas over it up through the liquid, and over it was the body's floor: under
// the Rafts crate 10 cells' worth more waited for good each row it rose,
// 124 to 168 in all, 317 to 404 in the scene (REWRITE_PLAN.md P3-17).
// Inside its cup the exchange is the cup's own (exchangeKey): the air in a
// hold stays in it, whatever is beside it.
fn wetLeft(p: vec2<i32>) -> bool {
    for (var d = 0u; d < 4u; d++) {
        let q = p + select(vec2<i32>(0, i32(d & 1u) * 2 - 1), vec2<i32>(i32(d & 1u) * 2 - 1, 0), d < 2u);
        if (inWorld(q) && kindOf(cells[cellIndex(q)]) == KIND_LIQUID) {
            return true;
        }
    }
    return false;
}

// A cell a body has left: nothing, a vacuum, which the air around fills
// (air.wgsl's expansion; the gas the body took the place of went into the
// air beside it, dropGas). As warm as the air beside it, if any, to be
// drawn so until it is filled (holding nothing, it takes the temperature of
// what fills it).
fn vacatedAir(p: vec2<i32>) -> vec4<u32> {
    var temp = AMBIENT_T;
    for (var d = 0u; d < 4u; d++) {
        let q = p + select(vec2<i32>(0, i32(d & 1u) * 2 - 1), vec2<i32>(i32(d & 1u) * 2 - 1, 0), d < 2u);
        if (inWorld(q)) {
            let w = cells[cellIndex(q)];
            if (cellType(w) == AIR) {
                temp = cellTemp(w);
                break;
            }
        }
    }
    setGasPair(cellIndex(p), vec2<u32>(0u));
    return packCell(makeCell(AIR, temp));
}

// ----------------------------------------------------------------- stamp

// What the solid v at cell i breaks into, carried along at vel, owing the
// cells it makes beyond itself (formed).
fn broken(v: vec4<u32>, vel: vec2<f32>, i: u32) -> vec4<u32> {
    let c = unpackCell(v);
    seedRng(i, P.frame, P.seed ^ 0xb40cu);
    var f = formed(convert(c, elems[c.t].breakTo), c);
    f.vel = vel;
    return packCell(f);
}

// Whether the impact stress p (Pa) breaks the solid v.
fn breaksAt(v: vec4<u32>, p: f32) -> bool {
    let e = elems[cellType(v)];
    return p > e.impact && (e.flags & F_INDESTRUCTIBLE) == 0u && e.breakTo != NO_ELEM;
}

// A material's acoustic impedance, √(E ρ) (kg/(m²·s)): a solid's by its
// Young's modulus, rigid for one without (a wall, the machines); a powder's
// as a granular bed.
fn impedance(e: Elem) -> f32 {
    let rho = e.rho;
    if (e.kind == KIND_POWDER) {
        return sqrt(POWDER_MODULUS * rho);
    }
    return select(RIGID_IMPEDANCE, sqrt(e.youngs * rho), e.youngs > 0.0);
}

// The stress (Pa) a blow at speed (m/s) puts into a and b where they meet.
fn contactStress(speed: f32, a: Elem, b: Elem) -> f32 {
    let za = impedance(a);
    let zb = impedance(b);
    return speed * za * zb / (za + zb);
}

// For the head body's cell at p of element e, landing hard this step: the
// cell ahead of it along the blow, whether it is something standing still,
// and the stress of the blow there (0 if it hit nothing).
struct Blow {
    ahead: vec2<i32>,
    hit: bool,
    stress: f32,
}

fn blowAt(p: vec2<i32>, e: Elem, b: u32, speed: f32) -> Blow {
    let into = -impactNormal;
    let axis = select(0u, 1u, abs(into.y) > abs(into.x));
    let ahead = p + axisStep(axis, select(-1, 1, into[axis] >= 0.0));
    let hit = obstacleKind(ahead, b) == OB_STATIC && inWorld(ahead);
    var stress = 0.0;
    if (hit) {
        stress = contactStress(speed, e, elems[cellType(cells[cellIndex(ahead)])]);
    }
    return Blow(ahead, hit, stress);
}

var<workgroup> hardest: atomic<u32>;   // the hardest blow a cell of the body took (f32 bits, Pa)
var<workgroup> unwritten: atomic<u32>; // of a body rejoining, cells of it its stamp has nowhere to write
var<workgroup> unwrittenCount: u32;

// A brittle solid standing still, hit at c by impact pressure p, shatters
// around the spot as far as SHATTER_REACH * sqrt(p / breakP) cells (up to
// SHATTER_MAX): the crack runs out through it, as a brittle body landing
// shatters right through. A falling block goes through a pane of glass, and
// what's left of the pane without support comes down after it.
fn shatterAround(c: vec2<i32>, t: u32, p: f32) {
    let reach = min(SHATTER_REACH * sqrt(p / elems[t].impact), SHATTER_MAX);
    let r = i32(ceil(reach));
    for (var y = -r; y <= r; y++) {
        for (var x = -r; x <= r; x++) {
            let q = c + vec2<i32>(x, y);
            if (!inWorld(q) || f32(x * x + y * y) > reach * reach) {
                continue;
            }
            let i = cellIndex(q);
            let v = cells[i];
            if (cellType(v) == t && !cellHas(v, FL_LOOSE)) {
                cells[i] = broken(v, pullDirAt(vec2<f32>(q) + 0.5) * SHARD_SPEED, i);
            }
        }
    }
}

// Where liquid the head body b moves into at p, moving at u there, goes out
// of its way (REWRITE_PLAN.md P3-14; audit/07 §5.5's "pushed out along the
// SDF"): the nearest cell within OUT_REACH outside the body as it is stamped
// now, on the side of its wall p was (in its cup as it was stamped, or not:
// `inside`; a bowl's water stays in the bowl, the water outside a hull out
// of it, P3-22), ahead of it first, holding what `kind` is (liquid: joining the liquid
// there, its crowding the projection's density spreads and the flow it sets
// carrying it round; gas, for new liquid, which moves as a cell), in no
// other body (claimed this step, or moved over by one last step). (-1, -1)
// if none. (Into gas too, a raft bobbing up into a wave's cell put it on
// its deck, ahead of it: it rode half a row low and shed its cargo.)
// (It went round behind the body, across its whole length, or OUT_REACH clear
// of its box ahead or aside, by hand, the audit's B1: ahead of a block
// sinking into the water on a sealed hull was the hull's inside, and the
// water went in there, 5 to 14 rows down through its top in a step.)
const OUT_REACH: i32 = i32(MAX_MOVE) + 1;

fn outsideOf(p: vec2<i32>, u: vec2<f32>, b: u32, key: u32, kind: u32, inside: bool) -> vec2<i32> {
    let ahead = select(vec2<f32>(0.0), normalize(u), dot(u, u) > 1e-8);
    for (var r = 1; r <= OUT_REACH; r++) {
        var best = vec2<i32>(-1);
        var most = -1e9;
        for (var k = 0; k < 8 * r; k++) {
            let side = k / (2 * r);
            let along = k % (2 * r);
            var q = p + vec2<i32>(-r, -r);
            switch side {
                case 0: { q += vec2<i32>(along, 0); }
                case 1: { q += vec2<i32>(2 * r, along); }
                case 2: { q += vec2<i32>(2 * r - along, 2 * r); }
                default: { q += vec2<i32>(0, 2 * r - along); }
            }
            if (!outsideFor(q, b, key, kind, inside)) {
                continue;
            }
            // The nearest of the ring, then the most ahead (the ring's order
            // breaking ties, so alike in any run).
            let o = vec2<f32>(q - p);
            let score = dot(o, ahead) / length(o) - length(o);
            if (score > most) {
                most = score;
                best = q;
            }
        }
        if (best.x >= 0) {
            return best;
        }
    }
    return vec2<i32>(-1);
}

// Whether liquid the head body b (claiming as key) puts out of its way can go
// into cell q, holding what `kind` is, on the side of its wall it was
// (outsideOf).
fn outsideFor(q: vec2<i32>, b: u32, key: u32, kind: u32, inside: bool) -> bool {
    if (!inWorld(q)) {
        return false;
    }
    let j = cellIndex(q);
    let k = claimOf(j);
    if (k == key || inCupAs(q, stampTo) != inside || claimedByBody(k)) {
        return false;
    }
    let swept = atomicLoad(&bodies[bodySweptSlot(P.size, j, P.frame - 1u)]);
    if ((swept >> 12u) == claimStep(P.frame - 1u) && swept != bodySweptKey(P.frame - 1u, b)) {
        return false;
    }
    return kindOf(cells[j]) == kind;
}

// Liquid v, drawn from its particles, sent to cell q: its particles go there
// at the liquid's next sync (fluid.wgsl's syncParticles, DEST_ASIDE), or stay
// where they are with q none (DEST_KEEP), to be let out of whatever is there
// by the liquid's escape. False if v isn't drawn (new liquid, its particles
// not seeded yet).
fn sendLiquid(v: vec4<u32>, q: vec2<i32>) -> bool {
    let home = liquidHome(v.w);
    if (!cellHas(v, FL_DRAWN) || home >= cellCount()) {
        return false;
    }
    atomicStore(&liquid[destSlot(home)], select(DEST_KEEP, DEST_ASIDE | cellIndex(q), q.x >= 0));
    return true;
}

// Whether one of the head body's cells is over shape cell s in its column,
// in its shape (a crate's lid over its inside, none over a boat's hold).
fn lidOver(s: vec2<i32>) -> bool {
    for (var y = s.y - 1; y >= 0; y--) {
        if (wordType(shapes[head.off + u32(y * head.size.x + s.x)].x) != AIR) {
            return true;
        }
    }
    return false;
}

// The head body's new stamp, as stampBody stamps it.
var<workgroup> stampTo: Stamp;

// Whether world cell q is in the head body's cup (CUP) as stamped by st, or
// as newly stamped.
fn inCupAs(q: vec2<i32>, st: Stamp) -> bool {
    let s = unstampCell(q, st);
    return all(s >= vec2<i32>(0)) && all(s < head.size) && shapes[head.off + u32(s.y * head.size.x + s.x)].y == CUP;
}

fn inCup(q: vec2<i32>) -> bool {
    return inCupAs(q, stampTo);
}

// Whether one of the head body's cells is at world cell q as stamped by st.
fn coversAs(q: vec2<i32>, st: Stamp) -> bool {
    let s = unstampCell(q, st);
    return all(s >= vec2<i32>(0)) && all(s < head.size) && wordType(shapes[head.off + u32(s.y * head.size.x + s.x)].x) != AIR;
}

// Whether the head body's shape cell i is a cell of its cup holding powder
// as stamped by `was`, which the stamp `to` moves: its cargo (CARGO_EACH).
fn cargoMoves(i: u32, was: Stamp, to: Stamp) -> bool {
    if (shapes[head.off + i].y != CUP) {
        return false;
    }
    let s = shapeCell(i);
    let p = stampCell(s, was);
    let q = stampCell(s, to);
    return inWorld(p) && inWorld(q) && any(p != q) && kindOf(cells[cellIndex(p)]) == KIND_POWDER;
}

// What this thread's shape cells carry, while the body moves: each one's
// shape cell, cell and gas pair, and those with nowhere to go yet.
var<private> cargoShape: array<u32, CARGO_EACH>;
var<private> cargoCell: array<vec4<u32>, CARGO_EACH>;
var<private> cargoGas: array<vec2<u32>, CARGO_EACH>;
var<private> cargoCount: u32;
var<private> cargoHomeless: u32;

// Its cargo taken up: each cell of it as last stamped (`was`) left holding
// nothing, and the exchange's if nothing of the body's comes into it (the
// exchange keeps only those still held for it, heldLeft). Another body
// taking a cell of it this step takes what is in it.
fn cargoTake(li: u32, n: u32, was: Stamp, to: Stamp, key: u32) {
    cargoCount = 0u;
    cargoHomeless = 0u;
    var seen = 0u;
    for (var i = li; i < n && seen < CARGO_EACH; i += BODY_THREADS) {
        if (!cargoMoves(i, was, to)) {
            continue;
        }
        seen++;
        let p = stampCell(shapeCell(i), was);
        let j = cellIndex(p);
        let k = claimOf(j);
        if (claimedByBody(k) && k != key) {
            continue;
        }
        cargoShape[cargoCount] = i;
        cargoCell[cargoCount] = cells[j];
        cargoGas[cargoCount] = gasPair(j);
        cargoCount++;
        cells[j] = vacatedAir(p);
        let at = atomicAdd(&leftCount, 1u);
        if (at < EXCHANGE_CAP) {
            leftCells[at] = j;
            leftKeys[at] = exchangeKey(p, inCupAs(p, to));
        }
    }
}

// Room made for it where it goes (`to`), as for the body's own cells: what
// is in a cell there goes the way a cell of the body would send it (the
// exchange, displace), the air in it into the air beside it, and the
// body's own cell, last stamped there, is covered. Where something stands
// still, a body's cell is now (a turned stamp isn't one to one) or another
// body has it, the cargo has nowhere to go yet (cargoPut).
fn cargoRoom(to: Stamp, toPose: Pose, vel: vec2<f32>, spin: f32, b: u32, key: u32) {
    for (var c = 0u; c < cargoCount; c++) {
        let q = stampCell(shapeCell(cargoShape[c]), to);
        let j = cellIndex(q);
        if (claimOf(j) != key || coversAs(q, to) || obstacleKind(q, b) == OB_STATIC) {
            cargoHomeless |= 1u << c;
            continue;
        }
        let old = cells[j];
        let kind = kindOf(old);
        if (isBodyCell(old, b) || kind == KIND_SOLID) {
            continue;
        }
        if (kind == KIND_GAS) {
            dropGas(q, j);
            continue;
        }
        if (kind == KIND_POWDER) {
            let at = atomicAdd(&intoCount, 1u);
            if (at < EXCHANGE_CAP) {
                intoCells[at] = j;
                intoKeys[at] = exchangeKey(q, inCupAs(q, head.stamp));
                continue;
            }
        }
        let u = vel + spinVel(spin, vec2<f32>(q) + 0.5 - toPose.at);
        if (!displace(old, q, u, b, key)) {
            cargoHomeless |= 1u << c;
        }
    }
}

// The cargo put down, or, with nowhere to go, into the nearest cell of gas
// (seekGas); let go only with none near. Its velocity is its own across
// the body's (update.wgsl's inCupOf: it grips the body as it would the
// ground), which the body's motion doesn't add to: carried by the stamp,
// it would move as far again.
fn cargoPut(to: Stamp, b: u32) {
    for (var c = 0u; c < cargoCount; c++) {
        var q = stampCell(shapeCell(cargoShape[c]), to);
        if ((cargoHomeless & (1u << c)) != 0u) {
            q = seekGas(q, 0, b, to, false);
            if (q.x < 0) {
                atomicAdd(&bodies[counterSlot(C_LOST_POWDER)], 1u);
                continue;
            }
            dropGas(q, cellIndex(q));
        }
        let j = cellIndex(q);
        cells[j] = cargoCell[c];
        setGasPair(j, cargoGas[c]);
    }
}

// From cell p of body b's new pose, back the way it came (step), the first
// cell it doesn't cover now: where what it moved into goes round to. Past
// the body's new box there is none.
fn behindOf(p: vec2<i32>, step: vec2<i32>, key: u32) -> vec2<i32> {
    var q = p;
    loop {
        q -= step;
        if (!inWorld(q) || any(q < head.lo - 1) || any(q > head.hi + 1)) {
            q = vec2<i32>(-1);
            break;
        }
        if (claimOf(cellIndex(q)) != key) {
            break;
        }
    }
    return q;
}

// Moves v, which body b's cell at p (moving at u) takes the place of, out of
// its way. Liquid goes out into the nearest liquid outside the body
// (outsideOf), or, drawn with none near, stays where it is, in the body's
// cell, for the liquid's escape to let out; powder is pushed on ahead to the
// first free cell past whatever loose material is ahead of it, else goes
// round behind; gas, air too, goes round behind, into a cell the body has
// left (holding nothing, vacatedAir), and powder or gas with nowhere to go
// is let go. Liquid is never let go: false if, undrawn (new this step, its
// particles not seeded), it has nowhere to go, and the body's cell is left
// out there instead. What a gas (or ash) holds of the air goes with it, and
// what is let go, and the air a cell is moved into, goes into the air beside
// it (moveGas, dropGas). (Air let go ahead of a body and none behind it, a
// body falling into water left a vacuum over itself and squeezed the air in
// the pockets beside it, and its own wood broke between them.)
fn displace(v: vec4<u32>, p: vec2<i32>, u: vec2<f32>, b: u32, key: u32) -> bool {
    let kind = kindOf(v);
    let i = cellIndex(p);
    if (kind == KIND_SOLID) {
        return true;
    }
    var axis = select(0u, 1u, abs(u.y) > abs(u.x));
    var dir = select(-1, 1, u[axis] >= 0.0);
    if (dot(u, u) < 1e-8) {
        // Covered without moving, as its footprint changed with its angle:
        // away from its middle.
        let out = vec2<f32>(p) + 0.5 - head.pos;
        axis = select(0u, 1u, abs(out.y) > abs(out.x));
        dir = select(-1, 1, out[axis] >= 0.0);
    }
    let step = axisStep(axis, dir);
    if (kind == KIND_LIQUID) {
        let inside = inCupAs(p, head.stamp);
        if (sendLiquid(v, outsideOf(p, u, b, key, KIND_LIQUID, inside))) {
            return true;
        }
        // New liquid: the cell itself, into gas.
        let q = outsideOf(p, u, b, key, KIND_GAS, inside);
        if (q.x >= 0 && claimFill(cellIndex(q))) {
            dropGas(q, cellIndex(q));
            cells[cellIndex(q)] = v;
            return true;
        }
        atomicAdd(&bodies[counterSlot(C_KEPT)], 1u);
        return false;
    }
    let back = behindOf(p, step, key);
    // Never into its own cup: a crate's wall moving into sand would put it
    // inside the crate.
    let backFree = back.x >= 0 && wordType(cells[cellIndex(back)].x) == AIR && !inCup(back);
    if (kind == KIND_POWDER) {
        var t = p;
        for (var k = 0; k < PUSH_REACH; k++) {
            t += step;
            if (!inWorld(t)) {
                break;
            }
            let j = cellIndex(t);
            let w = cells[j];
            let wk = kindOf(w);
            if (wk == KIND_GAS) {
                if (claimFill(j)) {
                    var moved = unpackCell(v);
                    moved.vel = u;
                    dropGas(t, j);
                    cells[j] = packCell(moved);
                    moveGas(i, j);
                    return true;
                }
                break;
            }
            if (wk == KIND_SOLID) {
                break;
            }
        }
    }
    // (Gas not into one in the liquid: wetLeft.)
    if (backFree && (kind == KIND_POWDER || !wetLeft(back)) && claimBehind(cellIndex(back))) {
        dropGas(back, cellIndex(back));
        cells[cellIndex(back)] = v;
        moveGas(i, cellIndex(back));
        return true;
    }
    dropGas(p, i);
    atomicAdd(&bodies[counterSlot(select(C_LOST_GAS, C_LOST_POWDER, kind == KIND_POWDER))], 1u);
    return true;
}

// Powder v at p that the head body moves into and carries (its cargo) goes
// to the first cell of gas up the column over it, against the pull, through
// at most CARGO_REACH cells of powder, moving at u. Whether it went.
fn liftCargo(v: vec4<u32>, p: vec2<i32>, u: vec2<f32>) -> bool {
    let up = -downAt(p);
    var t = p;
    for (var k = 0; k < CARGO_REACH; k++) {
        t += up;
        if (!inWorld(t)) {
            return false;
        }
        let j = cellIndex(t);
        let w = cells[j];
        let wk = kindOf(w);
        if (wk == KIND_GAS) {
            if (!claimFill(j)) {
                return false;
            }
            var moved = unpackCell(v);
            moved.vel = u;
            dropGas(t, j);
            cells[j] = packCell(moved);
            moveGas(cellIndex(p), j);
            return true;
        }
        if (wk != KIND_POWDER) {
            return false;
        }
    }
    return false;
}

// The exchange (stampBody): the cells a body leaves (of its old footprint,
// not its new one, nor another body's now, nor one in the liquid outside its
// cup, wetLeft) and those it moves into holding gas or powder are each put
// in order across the way it moves and then along it, and what was in each
// cell it moves into goes into the cell it leaves that is matched with it:
// moving straight, into the cell it left in the same line, behind it. Its
// old and new footprints are as many cells (the stamp is one to one), so
// there is a cell left for each moved into, but those something else has
// taken (another body, matter painted in, powder or gas displaced round
// behind it) or the liquid fills: what has nowhere to go is displaced as
// before (displace), and a cell left that nothing comes into holds nothing,
// which the air around fills, or the liquid. At most EXCHANGE_CAP cells
// either way; the rest are displaced so too.
const EXCHANGE_CAP: u32 = 512u;
var<workgroup> leftCells: array<u32, EXCHANGE_CAP>;
var<workgroup> leftKeys: array<u32, EXCHANGE_CAP>;
var<workgroup> intoCells: array<u32, EXCHANGE_CAP>;
var<workgroup> intoKeys: array<u32, EXCHANGE_CAP>;
var<workgroup> leftCount: atomic<u32>;
var<workgroup> intoCount: atomic<u32>;
var<workgroup> leftKept: atomic<u32>;
var<workgroup> exchangeSizes: vec4<u32>;  // cells left, moved into, kept of those left, the sort's size
var<workgroup> newMovedCount: atomic<u32>;
var<workgroup> newMovedShape: array<u32, MOVED_CAP>;
var<workgroup> newMovedCell: array<u32, MOVED_CAP>;
var<workgroup> exchangeAxes: vec4<f32>;   // across the way it moves, along it

// A cell's place in the exchange's order: inside the body's cup first (a
// cell left, in its cup as it is now; one moved into, in its cup as it
// was), then across the way the body moves, then along it, from its
// middle. So what is shut in a hull or a crate stays in it as it moves, the
// air a wall moves into going where a wall has left inside: matched by
// place alone, the air in the cells of a crate its top moved into went out
// over it, and the cells its floor left were left empty, a row a row it
// moved, and a crate settling on water was a vacuum inside in 20 steps.
fn exchangeKey(p: vec2<i32>, inside: bool) -> u32 {
    let r = vec2<f32>(p) + 0.5 - nextPose.xy;
    let across = clamp(i32(round(dot(r, exchangeAxes.xy))) + 1024, 0, 2047);
    let along = clamp(i32(round(dot(r, exchangeAxes.zw))) + 1024, 0, 2047);
    return select(1u << 22u, 0u, inside) | (u32(across) << 11u) | u32(along);
}

// Whether a cell keyed ka, index ca comes before one keyed kb, cb: by key,
// and cells of the same key by index. (Two cells can round to one key, and
// the sort isn't stable: by key alone, which of them went first followed
// the order the threads listed them in, and two runs from one snapshot came
// apart; REWRITE_PLAN.md D-1.)
fn exchangeAfter(ka: u32, ca: u32, kb: u32, cb: u32) -> bool {
    return ka > kb || (ka == kb && ca > cb);
}

// Sorts the cells left and those moved into by their keys, the first n of
// each (n a power of two, the same for the whole workgroup; bitonic).
fn sortExchange(li: u32, n: u32) {
    for (var k = 2u; k <= n; k <<= 1u) {
        for (var j = k >> 1u; j > 0u; j >>= 1u) {
            for (var i = li; i < n; i += BODY_THREADS) {
                let l = i ^ j;
                if (l > i) {
                    let up = (i & k) == 0u;
                    if (exchangeAfter(leftKeys[i], leftCells[i], leftKeys[l], leftCells[l]) == up) {
                        let key = leftKeys[i];
                        let cell = leftCells[i];
                        leftKeys[i] = leftKeys[l];
                        leftCells[i] = leftCells[l];
                        leftKeys[l] = key;
                        leftCells[l] = cell;
                    }
                    if (exchangeAfter(intoKeys[i], intoCells[i], intoKeys[l], intoCells[l]) == up) {
                        let key = intoKeys[i];
                        let cell = intoCells[i];
                        intoKeys[i] = intoKeys[l];
                        intoCells[i] = intoCells[l];
                        intoKeys[l] = key;
                        intoCells[l] = cell;
                    }
                }
            }
            workgroupBarrier();
        }
    }
}

// Where the head body's shape cell i is written at stamp `to`: its cell
// there, or where it went instead (MOVED_CAP: newMovedShape, the first
// movedNow of them).
fn stampedTo(i: u32, to: Stamp, movedNow: u32) -> vec2<i32> {
    var p = stampCell(shapeCell(i), to);
    for (var k = 0u; k < movedNow; k++) {
        if (newMovedShape[k] == i) {
            p = vec2<i32>(i32(newMovedCell[k] % P.size.x), i32(newMovedCell[k] / P.size.x));
        }
    }
    return p;
}

fn stampBody(b: u32, li: u32) {
    if (li == 0u) {
        loadHead(b);
        nextPose = vec4<f32>(tableV(b, T_NEXT), tableF(b, T_NEXT + 2u), 0.0);
        stepState = vec4<f32>(tableV(b, T_NEXTVEL), tableF(b, T_NEXTVEL + 2u), tableF(b, T_IMPACT));
        impactNormal = tableV(b, T_IMPACT + 1u);
        // The box the displacement looks in is the new one.
        head.lo = tableI(b, T_NEWBOX);
        head.hi = tableI(b, T_NEWBOX + 2u);
        atomicStore(&leftCount, 0u);
        atomicStore(&intoCount, 0u);
        atomicStore(&leftKept, 0u);
        atomicStore(&newMovedCount, 0u);
        let v = stepState.xy;
        let along = select(vec2<f32>(0.0, 1.0), normalize(v), dot(v, v) > 1e-6);
        exchangeAxes = vec4<f32>(-along.y, along.x, along);
    }
    let flags = workgroupUniformLoad(&headFlags);
    if ((flags & B_ALIVE) == 0u) {
        return;
    }
    let rejoin = (flags & B_REJOIN) != 0u;
    let was = head.stamp;
    let to = stampAt(nextPose.xy, nextPose.z, head.com, was.pivot);
    if (li == 0u) {
        stampTo = to;
    }
    let toPose = poseOf(nextPose.xy, nextPose.z);
    let vel = stepState.xy;
    let spin = stepState.z;
    let hard = stepState.w;
    let key = bodyKey(P.frame, b);
    let n = shapeCells();

    // The cells it leaves: they hold nothing for now (what a cell of the
    // body held of the air, a fan's, goes into the air beside it).
    for (var i = li; i < n; i += BODY_THREADS) {
        if (wordType(shapes[head.off + i].x) == AIR) {
            continue;
        }
        let p = stampedAt(i, shapeCell(i));
        if (!inWorld(p)) {
            continue;
        }
        let j = cellIndex(p);
        let k = claimOf(j);
        // Its own again (not its cup now, where its cargo may go, or not:
        // left so, its cell would be there twice), or another body's now,
        // which stamps it.
        if (!isBodyCell(cells[j], b) || (k == key && !inCupAs(p, to)) || (claimedByBody(k) && k != key)) {
            continue;
        }
        if (holdsGas(cellType(cells[j]))) {
            dropGas(p, j);
        }
        cells[j] = vacatedAir(p);
        claimLeft(j);
        // (Not one in the liquid outside its cup, which the liquid fills:
        // wetLeft.)
        if (inCupAs(p, to) || !wetLeft(p)) {
            let at = atomicAdd(&leftCount, 1u);
            if (at < EXCHANGE_CAP) {
                leftCells[at] = j;
                leftKeys[at] = exchangeKey(p, inCupAs(p, to));
            }
        }
    }
    cargoTake(li, n, was, to, key);
    workgroupBarrier();

    // The cells it moves into: liquid goes out of its way (displace) now,
    // gas and powder are listed for the exchange. Where it
    // can't go (something standing still is there, painted in or what it
    // rests on, another body has it, or liquid that couldn't be moved), its
    // cell goes into the nearest gas instead (MOVED_CAP), whose gas is the
    // exchange's.
    for (var i = li; i < n; i += BODY_THREADS) {
        let shape = shapes[head.off + i];
        let p = stampCell(shapeCell(i), to);
        if (wordType(shape.x) == AIR) {
            // Its cup, for the liquid's solve. Liquid in a cell of it under
            // its lid that wasn't last step is liquid the body has passed
            // over (moving more than a cell, or turning), and is put out of
            // it, as liquid it moves into is: left there it was inside the
            // crate. (Liquid that got into its cup by itself, as into a
            // boat, stays: under an open top it came in over the rim, which
            // a hold being space the water comes into lets it since P3-22,
            // and put out it bailed the Laden boat dry as it pitched. And in
            // its cup as it was stamped, as a body just found was where its
            // cells are: nothing marks a new one's cup as swept, and a tank
            // of water coming loose had all its water put out of it, crammed
            // into a few cells beside it.)
            if (shape.y == CUP && inWorld(p)) {
                let j = cellIndex(p);
                let ours = sweptLast(j, b) || inCupAs(p, was);
                sweep(j, b);
                let v = cells[j];
                if (!ours && kindOf(v) == KIND_LIQUID && cellHas(v, FL_DRAWN) && lidOver(shapeCell(i))) {
                    // Into liquid outside it, or failing that gas: left in
                    // its cup it was inside the crate.
                    let u = vel + spinVel(spin, vec2<f32>(p) + 0.5 - toPose.at);
                    var q = outsideOf(p, u, b, key, KIND_LIQUID, false);
                    if (q.x < 0) {
                        q = outsideOf(p, u, b, key, KIND_GAS, false);
                    }
                    if (q.x >= 0 && sendLiquid(v, q)) {
                        cells[j] = vacatedAir(p);
                    }
                }
            }
            continue;
        }
        // (Past the world's edge it goes elsewhere too: left out, it was
        // lost when the body rejoined.)
        if (inWorld(p)) {
            let j = cellIndex(p);
            // Moved over (and whether it was its own, or its cup, last step).
            let wasOurs = sweptLast(j, b);
            sweep(j, b);
            let ours = claimOf(j) == key;
            if (ours) {
                let old = cells[j];
                if (isBodyCell(old, b)) {
                    continue;
                }
                if (obstacleKind(p, b) != OB_STATIC) {
                    let kind = kindOf(old);
                    // Another body's cell, which it takes with it.
                    if (kind == KIND_SOLID) {
                        continue;
                    }
                    // Its cargo, powder that was in its cup or that it rises
                    // into, is lifted (liftCargo) rather than traded to where it
                    // left: a boat bobbing up put the sand in its hold under its
                    // floor, a cell row at a time, until none was left in it.
                    let along = vel + spinVel(spin, vec2<f32>(p) + 0.5 - toPose.at);
                    if (kind == KIND_POWDER && (wasOurs || dot(along, pullDirAt(vec2<f32>(p) + 0.5)) < 0.0)
                        && liftCargo(old, p, along)) {
                        continue;
                    }
                    if (kind != KIND_LIQUID) {
                        let at = atomicAdd(&intoCount, 1u);
                        if (at < EXCHANGE_CAP) {
                            intoCells[at] = j;
                            intoKeys[at] = exchangeKey(p, inCupAs(p, was));
                            continue;
                        }
                    }
                    let u = vel + spinVel(spin, vec2<f32>(p) + 0.5 - toPose.at);
                    if (displace(old, p, u, b, key)) {
                        continue;
                    }
                }
                atomicOr(&bodies[claimSlot(j)], 1u);
            }
        }
        let k = atomicAdd(&newMovedCount, 1u);
        if (k >= MOVED_CAP) {
            continue;
        }
        // Where it went last step, if it went elsewhere then too and has just
        // left it (so a body at rest keeps it there); else the nearest gas
        // from where its middle is at the pose, which the stamp may have left
        // free (a cell of a turned body's stamp can be a cell or so off,
        // leaving a gap beside another).
        var q = vec2<i32>(-1);
        for (var m = 0u; m < movedCount; m++) {
            if (movedShape[m] == i) {
                let back = atomicCompareExchangeWeak(&bodies[claimSlot(movedCell[m])], claimKey(P.frame, LEFT_PRIORITY), key);
                if (back.exchanged) {
                    q = vec2<i32>(i32(movedCell[m] % P.size.x), i32(movedCell[m] / P.size.x));
                }
            }
        }
        let again = q.x >= 0;
        if (!again) {
            q = seekGas(vec2<i32>(floor(cellWorld(shapeCell(i), toPose))), 0, b, to, true);
        }
        newMovedShape[k] = select(NONE, i, q.x >= 0);
        if (q.x < 0) {
            continue;
        }
        let jq = cellIndex(q);
        newMovedCell[k] = jq;
        // (A cell it has just left holds nothing.)
        if (again) {
            continue;
        }
        atomicMax(&bodies[claimSlot(jq)], key);
        let at = atomicAdd(&intoCount, 1u);
        if (at < EXCHANGE_CAP) {
            intoCells[at] = jq;
            intoKeys[at] = exchangeKey(q, inCupAs(q, was));
        } else {
            dropGas(q, jq);
        }
    }
    cargoRoom(to, toPose, vel, spin, b, key);
    workgroupBarrier();

    // The cells left that nothing else has taken (powder or gas gone round
    // behind it, matter displaced by another body) are the exchange's.
    if (li == 0u) {
        let lefts = min(atomicLoad(&leftCount), EXCHANGE_CAP);
        let intos = min(atomicLoad(&intoCount), EXCHANGE_CAP);
        var size = 1u;
        while (size < max(lefts, intos)) {
            size <<= 1u;
        }
        exchangeSizes = vec4<u32>(lefts, intos, 0u, size);
    }
    let sizes = workgroupUniformLoad(&exchangeSizes);
    for (var i = li; i < sizes.w; i += BODY_THREADS) {
        if (i < sizes.x && heldLeft(leftCells[i])) {
            atomicAdd(&leftKept, 1u);
        } else {
            leftKeys[i] = NONE;
        }
        if (i >= sizes.y) {
            intoKeys[i] = NONE;
        }
    }
    workgroupBarrier();
    sortExchange(li, select(sizes.w, 0u, sizes.y == 0u));
    let kept = atomicLoad(&leftKept);
    for (var r = li; r < sizes.y; r += BODY_THREADS) {
        let j = intoCells[r];
        let old = cells[j];
        if (r < kept) {
            let k = leftCells[r];
            cells[k] = old;
            moveGas(j, k);
        } else {
            let p = vec2<i32>(i32(j % P.size.x), i32(j / P.size.x));
            let u = vel + spinVel(spin, vec2<f32>(p) + 0.5 - toPose.at);
            if (!displace(old, p, u, b, key)) {
                atomicOr(&bodies[claimSlot(j)], 1u);
            }
        }
    }
    workgroupBarrier();

    // Its cells, at its new pose, or where they went instead; landing hard,
    // the hardest blow any of them takes first.
    let movedNow = min(atomicLoad(&newMovedCount), MOVED_CAP);
    let speed = sqrt(hard) * P.dx / P.dt;
    if (li == 0u) {
        atomicStore(&hardest, 0u);
        atomicStore(&unwritten, 0u);
    }
    workgroupBarrier();
    // It rejoins only with every cell of it written: one its stamp put in
    // something standing still, with no gas near to go to instead, or past
    // the world's edge, went with its shape (freeBody), and a block come to
    // rest tilted on the floor rejoined a cell or two short. It rests on as
    // a body instead, till the contacts have pushed it clearer (trying
    // again each step it is still at rest).
    for (var i = li; rejoin && i < n; i += BODY_THREADS) {
        if (wordType(shapes[head.off + i].x) == AIR) {
            continue;
        }
        let p = stampedTo(i, to, movedNow);
        if (!inWorld(p) || claimOf(cellIndex(p)) != key) {
            atomicAdd(&unwritten, 1u);
        }
    }
    workgroupBarrier();
    if (li == 0u) {
        unwrittenCount = atomicLoad(&unwritten);
        if (rejoin && unwrittenCount != 0u) {
            headFlags &= ~B_REJOIN;
            setU(b, T_FLAGS, headFlags);
        }
    }
    let rejoining = rejoin && workgroupUniformLoad(&unwrittenCount) == 0u;
    for (var i = li; hard > 0.0 && i < n; i += BODY_THREADS) {
        let shape = shapes[head.off + i];
        if (wordType(shape.x) == AIR) {
            continue;
        }
        let p = stampedTo(i, to, movedNow);
        if (inWorld(p)) {
            atomicMax(&hardest, bitcast<u32>(blowAt(p, elems[wordType(shape.x)], b, speed).stress));
        }
    }
    workgroupBarrier();
    let hardestBlow = bitcast<f32>(atomicLoad(&hardest));
    for (var i = li; i < n; i += BODY_THREADS) {
        let shape = shapes[head.off + i];
        if (wordType(shape.x) == AIR) {
            continue;
        }
        let p = stampedTo(i, to, movedNow);
        if (!inWorld(p)) {
            continue;
        }
        let j = cellIndex(p);
        if (claimOf(j) != key) {
            continue;
        }
        var cell = unpackCell(vec4<u32>(shape.x, shape.y, 0u, 0u));
        let u = vel + spinVel(spin, vec2<f32>(p) + 0.5 - toPose.at);
        if (rejoining) {
            cell.flags &= ~FL_LOOSE;
            cell.vel = vec2<f32>(0.0);
            cell.frac = vec2<f32>(P.supportMax, f32(head.piece));
        } else {
            cell.flags |= FL_LOOSE;
            cell.vel = u;
            cell.frac = vec2<f32>(f32(b), f32(head.piece));
        }
        var out = packCell(cell);
        // A hard landing: the cells that hit something (and, if brittle,
        // all of them, by the hardest blow) may break, and so may what they
        // hit.
        if (hard > 0.0) {
            let e = elems[cell.t];
            let blow = blowAt(p, e, b, speed);
            if (blow.hit) {
                let a = cellIndex(blow.ahead);
                let t = cells[a];
                if (kindOf(t) == KIND_SOLID && breaksAt(t, blow.stress)) {
                    cells[a] = broken(t, pullDirAt(vec2<f32>(blow.ahead) + 0.5) * SHARD_SPEED, a);
                    if ((elems[cellType(t)].flags & F_BRITTLE) != 0u) {
                        shatterAround(blow.ahead, cellType(t), blow.stress);
                    }
                }
            }
            let brittle = (e.flags & F_BRITTLE) != 0u;
            if ((blow.hit && breaksAt(out, blow.stress)) || (brittle && breaksAt(out, hardestBlow))) {
                out = broken(out, u, j);
            }
        }
        cells[j] = out;
    }
    workgroupBarrier();
    cargoPut(to, b);
    workgroupBarrier();
    // Its cargo outside its cup, each column of powder on top of one of its
    // cells (lift's), is marked as its, as its cup is, for the liquid's
    // solve: it rides with it there too. (As an obstacle, the faces between
    // a raft and the sand on it moved with the raft, so its columns under
    // the sand fell freely in the solve, and took none of the liquid's
    // pressure: a raft loaded over three quarters of its deck sank six rows
    // too deep.)
    for (var i = li; !rejoining && i < n; i += BODY_THREADS) {
        if (wordType(shapes[head.off + i].x) == AIR) {
            continue;
        }
        let p = stampCell(shapeCell(i), to);
        if (!inWorld(p) || !isBodyCell(cells[cellIndex(p)], b)) {
            continue;
        }
        let up = -downAt(p);
        var on = p + up;
        for (var k = 0; k < CARGO_REACH && inWorld(on); k++) {
            let jo = cellIndex(on);
            if (inCup(on) || kindOf(cells[jo]) != KIND_POWDER) {
                break;
            }
            sweep(jo, b);
            on += up;
        }
    }
    // Its seams: where two cells next to each other in its shape (across or
    // at a corner) land more than a cell apart, the cells between them are
    // its too, for the liquid and the air (fluid.wgsl's seamOf: a wall of
    // its), marked as moved over with the seam bit. Turning, the stamp's
    // three shears put two cells of a wall a cell thick two apart for a step
    // or two (P3-13's note); once a hold was space the water comes into,
    // P3-22, a lens hull a cell thick tipped by a stone dropped on it took
    // the water in through them, opened at 0.3 rad, and sank.
    for (var i = li; !rejoining && i < n; i += BODY_THREADS) {
        if (wordType(shapes[head.off + i].x) == AIR) {
            continue;
        }
        let s = shapeCell(i);
        let p = stampCell(s, to);
        for (var e = 0; e < 4; e++) {
            let d = select(select(vec2<i32>(1, -1), vec2<i32>(1, 1), e == 2), select(vec2<i32>(0, 1), vec2<i32>(1, 0), e == 0), e < 2);
            let t = s + d;
            if (any(t < vec2<i32>(0)) || any(t >= head.size) || wordType(shapes[head.off + u32(t.y * head.size.x + t.x)].x) == AIR) {
                continue;
            }
            let q = stampCell(t, to);
            var c = p;
            for (var k = 0; k < SEAM_REACH && max(abs(q - c).x, abs(q - c).y) > 1; k++) {
                c += sign(q - c);
                if (inWorld(c) && !isBodyCell(cells[cellIndex(c)], b)) {
                    atomicMax(&bodies[bodySweptSlot(P.size, cellIndex(c), P.frame)], bodySweptKey(P.frame, b) | SWEPT_SEAM);
                }
            }
        }
    }
    workgroupBarrier();
    if (rejoining) {
        freeBody(b, li);
    } else if (li == 0u) {
        setV(b, T_POS, nextPose.xy);
        setF(b, T_ANGLE, nextPose.z);
        setV(b, T_VEL, vel);
        setF(b, T_SPIN, spin);
        setI(b, T_BOX, head.lo);
        setI(b, T_BOX + 2u, head.hi);
        setU(b, T_FRAME, P.frame);
        setStamp(b, to);
        // Kept by shape cell, not in the order the threads listed them
        // (REWRITE_PLAN.md D-1).
        let listed = min(atomicLoad(&newMovedCount), MOVED_CAP);
        var kept = 0u;
        var after = 0u;
        loop {
            var at = NONE;
            for (var k = 0u; k < listed; k++) {
                let s = newMovedShape[k];
                if (s != NONE && (kept == 0u || s > after) && (at == NONE || s < newMovedShape[at])) {
                    at = k;
                }
            }
            if (at == NONE) {
                break;
            }
            setU(b, T_MOVED + 2u * kept, newMovedShape[at]);
            setU(b, T_MOVED + 2u * kept + 1u, newMovedCell[at]);
            after = newMovedShape[at];
            kept++;
        }
        setU(b, T_MOVED_N, kept);
    }
}

@compute @workgroup_size(256)
fn stamp(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let count = listedBodies(li);
    for (var k = wid.x; k < count; k += groups.x) {
        stampBody(motion[listSlot(k)], li);
        workgroupBarrier();
    }
}
