// Shared cell format, RNG and element helpers. Element ids and the Elem struct
// are generated from js/elements.js and prepended to this file at load time.
// Every module that includes this file must declare `elems`.

struct Params {
    size: vec2<u32>,
    frame: u32,
    seed: u32,
    offset: vec2<u32>,  // Margolus block offset for this pass
    supportMax: f32,    // a standing structural cell's support (frac.x), the unit anchoring lets go by
    // The world's scale: a cell is a cube dx metres on a side (js/shared.js
    // has what goes by it).
    dx: f32,
    // The time of day: the share of a day gone since midnight, and how much
    // of a day passes per step (0 while the clock is stopped).
    day: f32,
    dayRate: f32,
    // Gravity (gravity.wgsl): the pull of a black hole's cell, and of any
    // other cell per kg/m³ of its element's rho.
    holePull: f32,
    massPull: f32,
    // The Earth's own pull, straight down the screen, cells/step²: water's
    // weight (g at any scale, see dt), or 0 with it switched off.
    earthG: f32,
    // Seconds a step stands for: of motion (real time), longer in a bigger
    // world (js/shared.js stepSeconds), and on the day's clock
    // (dayStepSeconds), never less than dt; and log2(dt / dayDt), for heat's
    // clock in between (clockStep).
    dt: f32,
    dayDt: f32,
    clockSpan: f32,
    // Conduction in stages (heat.wgsl): how many there are this step (0:
    // none, faces too stiff for update.wgsl held to HEAT_SHARE_MAX there),
    // and the most J/K a step a face passes in them, as a share of the
    // smaller heat capacity of its two cells.
    heatStages: u32,
    heatLimit: f32,
    // 1 while a stroke of the brush is being drawn (sim.js holding): what it
    // paints is held where it is (SUPPORT_HELD) until it is let go.
    holding: u32,
    pad: f32,
}

// Down (GRAVITY_PLAN.md): each cell's is the total pull on it, the Earth's
// (Params.earthG) and the world's masses' (gravity.wgsl), rounded to one of
// its 8 neighbour steps, and what rests, piles, hangs, drains and rises does
// so along it. Weights (Elem.weight) are given against water's, g at a
// centimetre a cell; each takes the total pull as its weight is of that.

// How much a weight counts for under that pull: the pull as a share of g.
// Under WEIGHTLESS (with the Earth's switched off, far from any black hole)
// nothing settles or slumps: it floats. DOWN_WEIGHTLESS marks it in a down
// word (gravity.wgsl, pull.wgsl).
const WEIGHTLESS: f32 = 1e-3;
const DOWN_WEIGHTLESS: u32 = 8u;

fn pullShare(pull: vec2<f32>) -> f32 {
    return length(pull) / elems[WATER].weight;
}

// The 8 neighbour steps by index, from +x round toward +y (clockwise on
// screen): (1, 0), (1, 1), (0, 1), (-1, 1), ... Arithmetic rather than a
// table, which an index known only at run time would push out of registers.
fn stepIndex(d: vec2<i32>) -> u32 {
    if (d.y == 0) {
        return select(4u, 0u, d.x > 0);
    }
    if (d.x == 0) {
        return select(6u, 2u, d.y > 0);
    }
    if (d.y > 0) {
        return select(3u, 1u, d.x > 0);
    }
    return select(5u, 7u, d.x > 0);
}

fn stepX(i: u32) -> i32 {
    let j = (i + 1u) & 7u;
    return select(select(0, -1, j >= 4u && j <= 6u), 1, j <= 2u);
}

fn stepAt(i: u32) -> vec2<i32> {
    return vec2<i32>(stepX(i), stepX((i + 6u) & 7u));
}

// Step d turned by k eighths of a turn, k > 0 clockwise on screen: from
// Earth's down (0, 1), k = -2 is +x, ±1 the diagonals below, ±3 those above.
fn turnStep(d: vec2<i32>, k: i32) -> vec2<i32> {
    return stepAt((stepIndex(d) + u32(k + 8)) & 7u);
}

struct Brush {
    a: vec2<f32>,
    b: vec2<f32>,
    size: vec2<u32>,
    origin: vec2<u32>,  // first cell of the dispatched bounding box
    radius: f32,
    mode: u32,
    elem: u32,
    replace: u32,   // OVER_*: what it paints (heats, ...) over
    seed: u32,
    strength: f32,  // paint: fill chance; heat/cool: degrees; wind: gain
    angle: f32,     // stroke direction, used to aim fans
    shape: u32,     // SHAPE_*
    temp: f32,      // paint: the temperature it is painted at, °C, or under TEMP_OWN its element's own
    only: u32,      // OVER_ONLY's element
    dx: f32,        // the world's scale, metres a cell (Params.dx), for the open air's pressure
    pad: u32,
}

// The brush's MODE_*, SHAPE_* and OVER_* (what it paints over) and TEMP_OWN
// come from js/shared.js.

// A cell is stored as vec4<u32> (its layout and accessors, cellType,
// cellTemp, wordAux and the like, come from js/shared.js):
//   x: type (8) | aux (8) | life (8) | shade (4) | flags (4); what aux, life
//      and a plant's shade hold by the class of element is its record
//      (js/cellrecord.js, read and written through its accessors:
//      soluteOf(c), withSugar(c, v), wordSoilWater(x) and the like)
//   y: temperature, f32 bits
//   z: velocity in cells per step, 2 x f16
//   w: sub-cell position carried between passes, 2 x f16; structural
//      solids keep their support there instead (frac.x) and the piece of
//      structure they belong to (frac.y, see bodies.wgsl), and while they
//      move with a free body (FL_LOOSE) its slot in frac.x; liquid is moved
//      as particles (fluid.wgsl), and its cells keep the index of the
//      particle they were drawn from there instead (Cell.w); air has no use
//      for it
// and what a gas holds of the air, after all the cells in the same buffer
// (the gas plane, Gas below).
struct Cell {
    t: u32,      // element id
    aux: u32,    // by class (js/cellrecord.js): a solute, a charge, a heading, ...
    life: u32,   // by class: what has burnt, sugar, soil water, a strength, ...
    shade: u32,  // per-particle color variation (a plant's lean in its top two bits)
    flags: u32,  // FL_*
    temp: f32,   // °C
    vel: vec2<f32>,
    frac: vec2<f32>,
    w: u32,      // the w word as stored: a drawn liquid cell's particles (liquidHome)
}

// The flags (FL_*, js/shared.js), a bit meaning different things for
// different elements:
//   1 FL_BURN: burning.
//   2 FL_DEW: air against a surface colder than itself, where
//     mist collects (update.wgsl). FL_STRESSED: ice that owes cells (see
//     formed) and has no room to grow them, gas or liquid, beside it
//     (update.wgsl), so it presses on what is (block.wgsl). FL_ROOTED: soil
//     a plant's roots run through (pairs.wgsl), which pass its water up
//     to the plant. FL_SPLIT: water electrolysis is splitting
//     (electric.wgsl), once it has handed its solute on (pairs.wgsl).
//   4 FL_DRAWN: liquid drawn from the particles in it (fluid.wgsl's
//     raster), which w says (liquidHome); liquid without it is new, and is
//     seeded with particles. FL_WET: powder under liquid, through the
//     powder over it (update.wgsl). FL_SOFTENED: structure that has been
//     past its softening point (softeningT), and so is unstressed there,
//     not where it was made (update.wgsl; stress.wgsl's thermal strain).
//   8 FL_GROUNDED: powder resting, through whatever is under it, on the
//     ground. FL_LOOSE: a structural solid whose support ran out, moving with
//     the free body it is part of (bodies.wgsl), carrying its velocity in vel
//     and the body's slot in frac.x instead of support; or FREE_BODY there,
//     if the body broke up, until its pieces become bodies of their own.
const FREE_BODY: f32 = 2047.0;

// A drawn liquid cell's w (fluid.wgsl's raster): the cell its particles were
// in when it was drawn, and how many there were. Whatever moves the cell
// (block.wgsl's trades, the bodies' displacing) moves w with it, and next
// step its particles follow (fluid.wgsl's syncParticles); a pass that takes
// some of its liquid lowers the count. LIQUID_COUNT_MAX means as many or
// more.
const LIQUID_COUNT_BITS: u32 = 8u;
const LIQUID_COUNT_MAX: u32 = 255u;

fn liquidHome(w: u32) -> u32 {
    return w >> LIQUID_COUNT_BITS;
}

fn liquidCount(w: u32) -> u32 {
    return w & LIQUID_COUNT_MAX;
}

fn liquidW(home: u32, count: u32) -> u32 {
    return (home << LIQUID_COUNT_BITS) | min(count, LIQUID_COUNT_MAX);
}

// A drawn liquid cell's home when it was drawn from no particles: a pass
// made it holding more than a cell (formed), and the liquid's seeding gives
// it as many as it counts (fluid.wgsl's syncSeed).
const LIQUID_NO_HOME: u32 = 0xffffffu;

// The liquid in cell c (a cell of liquid), in cells: its particles' worth
// if drawn, a cell's if new (seeded with a cell's worth). A pass that turns
// it into something else takes all of it (fluid.wgsl's syncParticles drops
// its particles), and has to account for as much.
fn liquidAmount(c: Cell) -> f32 {
    if ((c.flags & FL_DRAWN) == 0u) {
        return 1.0;
    }
    return f32(liquidCount(c.w)) / f32(LIQUID_PER_CELL);
}

// Fresh water (a drawn cell of WATER carrying nothing) giving up `taken`
// cells of itself (to the air, the soil, a plant): a particle at a time
// (LIQUID_PER_CELL to a cell), as likely as what is over is of one, its
// count lowered (fluid.wgsl's syncParticles drops as many); whether that
// was the last of it. Anything else goes whole, as likely as `taken` is of
// what it holds. (A particle at a time, the water taken matches what the
// other side gets four times as closely as a cell at a time.)
fn takeLiquid(c: ptr<function, Cell>, taken: f32) -> bool {
    let fresh = ((*c).flags & FL_DRAWN) != 0u && (*c).t == WATER && soluteOf(*c) == 0u;
    if (!fresh) {
        return rnd() < taken / liquidAmount(*c);
    }
    let n = liquidCount((*c).w);
    let gone = u32(floor(taken * f32(LIQUID_PER_CELL) + rnd()));
    if (gone >= n) {
        return true;
    }
    (*c).w = liquidW(liquidHome((*c).w), n - gone);
    return false;
}

// The same for a cell as stored, 1 for anything but liquid.
fn amountIn(v: vec4<u32>) -> f32 {
    if (elems[cellType(v)].kind != KIND_LIQUID || !cellHas(v, FL_DRAWN)) {
        return 1.0;
    }
    return f32(liquidCount(v.w)) / f32(LIQUID_PER_CELL);
}

// People's bodies in the world (people.wgsl stamps them, sim.js clears the
// mask before it each step; REWRITE_PLAN.md 10-1): per cell, the person's
// slot plus one in the low 8 bits (none: 0), and its velocity in 12-bit
// signed 128ths of a cell a step in the next 12 and the 12 after. Powders
// and gases don't move into such a cell (block.wgsl), the liquid takes it
// as an obstacle moving so (fluid.wgsl), and light as opaque.
const OCCUPANT_SCALE: f32 = 128.0;

fn occupantWord(slot: u32, vel: vec2<f32>) -> u32 {
    let q = vec2<i32>(clamp(round(vel * OCCUPANT_SCALE), vec2<f32>(-2047.0), vec2<f32>(2047.0)));
    return (slot + 1u) | ((u32(q.x) & 0xfffu) << 8u) | ((u32(q.y) & 0xfffu) << 20u);
}

fn isOccupant(w: u32) -> bool {
    return (w & 0xffu) != 0u;
}

fn occupantVel(w: u32) -> vec2<f32> {
    // (The 12-bit fields sign-extended by shifting them to the top first.)
    let x = i32(w << 12u) >> 20u;
    let y = i32(w) >> 20u;
    return vec2<f32>(f32(x), f32(y)) / OCCUPANT_SCALE;
}

// Where the particles a cell was drawn from go next step, in the liquid's
// counters (fluid.wgsl's G_DEST, LIQUID_CELL_WORDS a cell): nowhere, 0 (what
// it was drawn as is gone), nowhere else (it wasn't drawn: liquid can't be
// there), aside (the bodies' sendLiquid: | the cell), or to the cell drawn
// from them now (| DEST_SPAN less that cell, so that of two cells copied
// from one the first takes them).
const DEST_KEEP: u32 = 0xffffffffu;
const DEST_ASIDE: u32 = 0x80000000u;
const DEST_CELL: u32 = 0x40000000u;
const DEST_SPAN: u32 = 0x3fffffffu;
const G_DEST: u32 = 4u;

fn destSlot(cell: u32) -> u32 {
    return LIQUID_CELL_WORDS * cell + G_DEST;
}

// NO_ELEM, AMBIENT_T and KELVIN come from js/shared.js.
const NEVER: f32 = 1e8;

// Movement: a particle moves at most one cell per block pass, so the number
// of passes per step (BLOCK_PASSES, js/shared.js) is also the speed limit.
const MAX_SPEED: f32 = BLOCK_PASSES;
const SETTLED_SPEED: f32 = 0.05;  // a powder slower than this counts as resting

// The air grid has one cell per AIR_CELL x AIR_CELL particle cells
// (js/shared.js; air.wgsl). Each air cell is vec4<f32>(pressure, velocity
// on its right face, velocity on its bottom face, the air's temperature);
// cells with no gas in them store AIR_BLOCKED as temperature, and solids
// AIR_SOLID, under it: a solid holds apart the gas on its two sides, and
// what it reports of their pressure pushes nothing beside it
// (airsample.wgsl).
const AIR_BLOCKED: f32 = -10000.0;
const AIR_SOLID: f32 = -20000.0;

// Structural solids (those with fragments to crumble into) keep a support in
// frac.x: P.supportMax standing (new ones start with more, which the update
// pass clamps), and -P.supportMax once the anchoring has let them go
// (bodies.wgsl: joined to nothing that holds them), as if loose for
// COLLAPSE_STEPS, long enough to fall as one free body. What breaks them is
// their stress (stress.wgsl).
const SUPPORT_NEW: f32 = 10000.0;
const COLLAPSE_STEPS: f32 = 40.0;
// Structure the brush paints (brush.wgsl) is held where it is while the
// stroke it is painted in goes on (P.holding): it counts as anchored
// (bodies.wgsl's anchoredAt) and its nodes are held still in the stress
// solve (stress.wgsl's setup), so a shape drawn in the air stays put until
// the pointer is let go, then falls as one piece. The update pass clamps it
// as it does SUPPORT_NEW the first step no stroke is going on.
const SUPPORT_HELD: f32 = 20000.0;

// The body passes (bodies.wgsl) only look for new bodies in the 8 x 8 tiles
// with loose cells in no body in or next to them. The update pass marks the
// tiles with such cells in them and lists them, in the bodies buffer: a mark
// per tile at its start; past a word per tile of the body passes' own, how
// many tiles are marked (which sim.js reads back, with how many bodies there
// are, to leave out the body passes when there is nothing loose anywhere),
// and BODY_STAT_WORDS on, the list. BODY_TILE and the layout's counts come
// from js/shared.js.

fn bodyTilesWide(size: vec2<u32>) -> u32 {
    return (size.x + BODY_TILE - 1u) / BODY_TILE;
}

fn bodyTileCount(size: vec2<u32>) -> u32 {
    return bodyTilesWide(size) * ((size.y + BODY_TILE - 1u) / BODY_TILE);
}

fn bodyTileOf(p: vec2<u32>, size: vec2<u32>) -> u32 {
    return (p.y / BODY_TILE) * bodyTilesWide(size) + p.x / BODY_TILE;
}

fn bodyMarkedSlot(size: vec2<u32>) -> u32 {
    return 2u * bodyTileCount(size);
}

fn bodyMarkListSlot(size: vec2<u32>) -> u32 {
    return 2u * bodyTileCount(size) + BODY_STAT_WORDS;
}

// Structure's words in the bodies buffer, after the bodies' own (their
// header, a label and a claim word a cell, BODY_SUM_WORDS of sums a cell):
// whether the structure changed this step, a hash a tile, and three words a
// cell, its label, whether it is a root an anchor holds (bodies.wgsl's
// anchoring), and the step the stress solve last found it overstressed
// (stress.wgsl; update.wgsl crumbles it the next step).
fn structureBase(size: vec2<u32>) -> u32 {
    let cells = size.x * size.y;
    return 4u * bodyTileCount(size) + BODY_STAT_WORDS + BODY_COUNTERS + MAX_BODIES + (2u + BODY_SUM_WORDS) * cells;
}

fn structureBreakSlot(size: vec2<u32>, i: u32) -> u32 {
    return structureBase(size) + 1u + bodyTileCount(size) + 2u * size.x * size.y + i;
}

// What the liquid's pass sums for a held body (fluid.wgsl's rigidify,
// bodies.wgsl's lift takes it): this bit of the last word is set if a cell
// of it was in a block by the liquid (the body is wet: B_WET).
const LIQUID_WET: u32 = 0x80000000u;

// After structure's, per cell which body last moved over it (bodies.wgsl's
// stamp), in two words, one for even steps and one for odd: the step's low
// 20 bits over 2047 less the body's slot (bodySweptKey), put by atomicMax,
// so that of two bodies moving over a cell in one step the lower slot is the
// one kept (as claims go), whichever thread came first; and last step's is
// read from the other word, which nothing writes this step. (Written over
// and read in one word, which body a cell went to, and whether a body found
// a cell its own from last step, followed the order the GPU ran the bodies
// in: REWRITE_PLAN.md D-1.)
fn bodySweptSlot(size: vec2<u32>, i: u32, frame: u32) -> u32 {
    return structureBreakSlot(size, size.x * size.y) + 2u * i + (frame & 1u);
}

fn bodySweptKey(frame: u32, b: u32) -> u32 {
    return (claimStep(frame) << 12u) | ((2047u - b) << 1u);
}

// The step as a claim word (bodies.wgsl's claimKey) and a swept word keep
// it, in their top 20 bits: 1 to CLAIM_PERIOD, never 0, which a cleared word
// is (as the step's low 20 bits, once every 2^20 steps every claim of
// matter at priority 0 was 0, and failed); sim.js clears the words as it
// comes round (CLAIM_STEPS).
const CLAIM_PERIOD: u32 = 0xfffffu;

// What the liquid did to a body this step (fluid.wgsl's rigidify, into
// bodies.wgsl's liquidSlot words), its change in velocity and spin, in
// fixed point at this scale.
const RIGID_FIX: f32 = 16777216.0;

fn claimStep(frame: u32) -> u32 {
    return frame % CLAIM_PERIOD + 1u;
}

// Electric charge: 4 and 3 are active (spreading, CHARGE_ACTIVE from
// js/shared.js), 2 and 1 are cooldown. A static pulse's few millijoules
// warm nothing measurably (REWRITE_PLAN.md P1, sparks), and light only what
// a spark can (sparkLights).
const CHARGE_NEW: u32 = 4u;

// The gas plane: what a cell of gas holds (REWRITE_PLAN.md 4c), as amounts.
// Every gas is the air with something in it (its kind a marker riding the
// air's flow): a flame the air it burns in, smoke the air its soot is in,
// hydrogen a gas of its own; ash holds the air of its pores, and a fan the
// air in it (holdsGas). Kept after the cells in the cells buffer (so it goes
// with them, the update pass's src and dst alike), two words a cell, a
// cell's pair at gasIndex and gasHigh:
//   x: vapour and mist, 2 x f16 (vapour units, js/shared.js WATER_VAPOUR)
//   y: the share of its dry gas's oxygen used up (0 fresh, 1 none left),
//      f16, and the dry gas's amount, f16: a cell of air at 1 atm and 20 °C
//      holds 1
// What a gas holds goes with it as a pass moves it, and the air's carry
// moves it between cells (carry.wgsl); a pass that turns a gas into
// something else puts it into the air beside it (airsource.wgsl). All zero
// is nothing, a vacuum: what anything else holds (the update pass writes it
// so), and what a cell of gas a liquid or body leaves starts with, until the
// air around fills it; gas made out of something else (water boiling, wood
// burning) holds what it was made into (madeGas, update.wgsl phaseGas).
struct Gas {
    vapour: f32,
    mist: f32,
    lost: f32,
    amount: f32,
}

// Index of the vec4 holding cell i's pair, among count cells, and whether
// it is the second pair (zw) in it.
fn gasIndex(count: u32, i: u32) -> u32 {
    return count + (i >> 1u);
}

fn gasHigh(i: u32) -> bool {
    return (i & 1u) != 0u;
}

fn gasFrom(v: vec4<u32>, i: u32) -> Gas {
    let w = select(v.xy, v.zw, gasHigh(i));
    let m = unpack2x16float(w.x);
    let r = unpack2x16float(w.y);
    return Gas(m.x, m.y, r.x, r.y);
}

fn packGas(g: Gas) -> vec2<u32> {
    return vec2<u32>(pack2x16float(vec2<f32>(g.vapour, g.mist)), pack2x16float(vec2<f32>(g.lost, g.amount)));
}

// Nothing: a vacuum.
fn noGas() -> Gas {
    return Gas(0.0, 0.0, 0.0, 0.0);
}

// The dry air's amount in a cell of air at temp (°C) under atm atmospheres.
fn airAmount(atm: f32, temp: f32) -> f32 {
    return atm * (AMBIENT_T + KELVIN) / max(temp + KELVIN, 1.0);
}

// Vapour units as an amount of gas, on the dry air's scale: VAPOUR_MAX of
// them is a cell of gas at 1 atm and the boiling point (BOIL_K).
fn vapourAmount(vapour: f32) -> f32 {
    return vapour * (AMBIENT_T + KELVIN) / (BOIL_K * VAPOUR_MAX);
}

// All the gas a cell holds, as an amount: its dry gas and its vapour (mist is
// water, no gas).
fn gasAmount(g: Gas) -> f32 {
    return g.amount + vapourAmount(g.vapour);
}

// The pressure, atm, of what a cell of gas holds at temp (°C), by the gas
// law: as much as it would be at 1 atm and 20 °C, times the temperature's
// rise in kelvin.
fn gasAtm(g: Gas, temp: f32) -> f32 {
    return gasAmount(g) * max(temp + KELVIN, 0.0) / (AMBIENT_T + KELVIN);
}

// The heat capacity, J/K, of what a cell of gas holds: the dry gas's (the
// air's, elements.js GAS_HEAT_CAP, a cell of it at 1 atm and 20 °C), the
// vapour's (VAPOUR_HEAT_CAP a cell of water's worth) and the mist's, which
// is water. None in a vacuum.
fn gasHeatCap(g: Gas) -> f32 {
    return elems[AIR].heatCap * g.amount + (VAPOUR_HEAT_CAP * g.vapour + elems[WATER].heatCap * g.mist) / WATER_VAPOUR;
}

// Gas made where there was none (what burns away into smoke or ash, an
// explosive going off, water split into hydrogen), its oxygen used up by
// lost: a cell of it at 1 atm and temp (°C), which the air's flow lets
// expand or squeezes to the pressure around it (air.wgsl's expansion).
// How much gas a thing gives off by what it is is stage 5's
// (REWRITE_PLAN.md).
fn madeGas(temp: f32, lost: f32) -> Gas {
    return Gas(0.0, 0.0, lost, airAmount(1.0, temp));
}

// The molar mass, g/mol, of what a cell of t holding g holds: its dry gas's
// (t's, the air's for air) and its vapour's, 18, by amount. The oxygen an
// oxygen carrier's dry gas has used went into as much carbon dioxide, mole
// for mole as carbon burns, 44 g/mol for oxygen's 32 (the water burning
// makes is its vapour): so burnt air at the outdoor air's temperature is
// heavier, 31.5 all burnt, and pools low as smoke cools (REWRITE_PLAN.md
// 5g; the gas plane has no CO₂ amount of its own: the oxygen used stands
// for it).
const CO2_GAIN: f32 = 0.2095 * 12.0;  // fresh air's oxygen by moles, times 44 - 32
fn gasMolar(t: u32, g: Gas) -> f32 {
    let vapour = vapourAmount(g.vapour);
    let all = g.amount + vapour;
    let dry = elems[t].molar + select(0.0, CO2_GAIN * g.lost, carriesOxygen(t));
    return select(elems[AIR].molar, (g.amount * dry + vapour * 18.0) / all, all > 0.0);
}

// x as an f16 rounded at random, given r in [0, 1): up to the f16 above it as
// often as x lies up the way to it from the one below, so that what is
// rounded adds up on average however small the change (a little vapour
// evaporating into nearly saturated air each step would otherwise round
// away, while the water it came from still runs out). (Dithered by a
// fraction of x's own f16 step instead, rounding leaned towards none by up
// to a quarter of a step where x was just past a power of two.)
fn stochastic16(x: f32, r: f32) -> f32 {
    let bits = pack2x16float(vec2<f32>(x, 0.0)) & 0xffffu;
    let near = unpack2x16float(bits).x;
    if (near == x || abs(near) > 65000.0) {
        return near;
    }
    // The f16 on x's side of `near`: a step in magnitude down (x between
    // none and near) or up.
    let magnitude = bits & 0x7fffu;
    let down = abs(x) < abs(near) && magnitude > 0u;
    let other = unpack2x16float((bits & 0x8000u) | select(magnitude + 1u, magnitude - 1u, down)).x;
    let lo = min(near, other);
    let hi = max(near, other);
    return select(lo, hi, r < (x - lo) / (hi - lo));
}

// Whether the air's flow passes through a cell of t (air.wgsl, carry.wgsl,
// and the rooms, sky.wgsl): gas, or a solid that lets it through (a fan, a
// black hole); but not gas the liquid counts as liquid (fluid.wgsl's
// classify: an empty cell inside it, short of the liquid that left it, and
// a vacuum a fast flow tears open), liquid its density there in the liquid's
// grid. As the liquid has it, so a gap its churning leaves isn't a pocket of
// nothing pulling the water around it in.
fn flowsThrough(t: u32, liquid: f32) -> bool {
    let e = elems[t];
    return (e.kind == KIND_GAS && liquid <= 0.0) || (e.kind == KIND_SOLID && (e.flags & F_AIR_PASS) != 0u);
}

// What the air's passes take for a cell's density in the liquid's grid, from
// its cell c and drift d there (fluid.wgsl's field): a body's cells and cup
// as liquid, held (its drift's z below none) as well as free. A held body's
// cup is a wall to the liquid's solve (fluid.wgsl's isHeld), its density
// there below none, and its air the body's, kept out of the air's flow and
// rooms as P3-12 has a floating body's (REWRITE_PLAN.md P3-19).
fn liquidSide(c: vec4<f32>, d: vec4<f32>) -> f32 {
    return select(c.z, 1.0, d.z < 0.0);
}

// Whether a cell of t holds air in the gas plane: a gas; ash, in its pores;
// and what the air passes through and carries (a fan, air with blades in it;
// carry.wgsl), so a fan broken into scrap hands its air's moisture on as a
// gas turned into something else does.
fn holdsGas(t: u32) -> bool {
    let e = elems[t];
    return e.kind == KIND_GAS || t == ASH || (e.kind == KIND_SOLID && (e.flags & F_AIR_PASS) != 0u && t != BLACK_HOLE);
}

// The air's oxygen. Things only burn with enough of it around: below about
// 15 % oxygen, against fresh air's 21 % (SMOTHER, js/shared.js), flaming
// combustion of ordinary fuels in air at 20 °C goes out (Babrauskas,
// Ignition Handbook, 2003); see flameReach.

// The air's moisture: vapour and mist, in units of 1 / WATER_VAPOUR of a
// cell of water, what boiling water becomes too (js/shared.js: water
// expands VAPOUR_EXPANSION times as it boils, so air at 20 °C holds
// VAPOUR_20, about 210 times real air's vapour by mass). WATER_VAPOUR,
// VAPOUR_20, VAPOUR_MAX (the most a cell of gas holds at 1 atm, all of it
// vapour) and OUTDOOR_HUMIDITY come from js/shared.js. Every gas takes it in
// and gives it off (update.wgsl).
// A snowflake is half a cell of water (see formed), WATER_VAPOUR / 2 of
// vapour and mist, but the air it forms out of may have held a little more by
// then, and ice or snow growing into air (block.wgsl) takes in the air's.
// That extra is kept in the ice's or snow's life, in 1 / FROZEN_EXTRA units of
// vapour (js/shared.js; up to 4 units), and comes back as water when it melts.
fn holdsVapour(t: u32) -> bool {
    return elems[t].kind == KIND_GAS;
}

// Water's latent heat of evaporation, J per vapour unit: what water gives up
// in evaporating and gets back as its vapour condenses (one water: 2257 J a
// cell of it).
fn vapourLatent() -> f32 {
    return elems[WATER].latentUp / WATER_VAPOUR;
}

// Ice or snow's life with the vapour and mist (in vapour units) of air it
// takes in added to it.
fn frozenExtra(life: u32, moisture: f32) -> u32 {
    return u32(clamp(f32(life) + moisture * FROZEN_EXTRA + rnd(), 0.0, 255.0));
}

// What gives its oxygen to what burns, and evens it out with the rest:
// air, flames and smoke, and ash, in the air of its pores. (A fireball's gas
// and hydrogen have none to give.)
fn carriesOxygen(t: u32) -> bool {
    return t == AIR || t == FIRE || t == SMOKE || t == ASH;
}

// Share of fresh air's oxygen in a cell of type t holding g.
fn oxygenOf(t: u32, g: Gas) -> f32 {
    return select(0.0, 1.0 - g.lost, carriesOxygen(t));
}

// The oxygen's pressure in a cell of type t holding g at temp, as a share
// of fresh air's at 1 atm and 20 °C: what breathing goes by (people.wgsl).
// (By its freshness alone, oxygenOf, a vacuum was fresh air: people
// breathed easily in a room a black hole had emptied.)
fn oxygenPressure(t: u32, g: Gas, temp: f32) -> f32 {
    return oxygenOf(t, g) * g.amount * max(temp + KELVIN, 0.0) / (AMBIENT_T + KELVIN);
}

// Fire (REWRITE_PLAN.md stage 5; js/shared.js's OXYGEN_HEAT and
// CRITICAL_FLAME_T). The oxygen a cell of t holding g has to burn, in cells of
// fresh air's: what its dry air (g.amount, 1 a cell of air at 1 atm and
// 20 °C) has left; ash's, in its pores.
fn oxygenHeld(t: u32, g: Gas) -> f32 {
    return select(0.0, (1.0 - g.lost) * g.amount, carriesOxygen(t));
}

// Burning heats its gas no hotter than FLAME_MAX_T: past about 2000 °C the
// products of combustion come apart again as fast as they form, and their
// heat capacity climbs (the adiabatic flame temperatures of fuels burning
// in air are near it: Turns, An Introduction to Combustion, 2012, ch. 2),
// while the gases' heat capacity here is a cold gas's (js/shared.js).
const FLAME_MAX_T: f32 = 2000.0;

// How hot burning all the oxygen it holds would make a cell of t holding g
// at temp: a flame burns in its gas where that reaches CRITICAL_FLAME_T and
// it isn't at FLAME_MAX_T already (holdsFlame). Air short of oxygen, cold,
// or heavy with vapour or mist (whose heat capacity takes some of the heat),
// holds no flame; hot air holds one with less.
fn flameReach(t: u32, g: Gas, temp: f32) -> f32 {
    let cap = gasHeatCap(g);
    return select(temp, temp + oxygenHeld(t, g) * OXYGEN_HEAT / cap, cap > 0.0 && carriesOxygen(t));
}

fn holdsFlame(t: u32, g: Gas, temp: f32) -> bool {
    return temp < FLAME_MAX_T && flameReach(t, g, temp) >= CRITICAL_FLAME_T;
}

// The oxygen a flame can burn in a cell of t holding g at temp, in cells of
// fresh air's: what it holds, as far as that heats it to FLAME_MAX_T; none
// where it holds no flame. What burns against it (pairs.wgsl) burns that,
// in 255ths of a cell of fuel, oxygenNeed / 255 of oxygen each: where it is
// less than that, nothing does, and the fuel goes out (update.wgsl).
fn oxygenToBurn(t: u32, g: Gas, temp: f32) -> f32 {
    if (!holdsFlame(t, g, temp)) {
        return 0.0;
    }
    return min(oxygenHeld(t, g), (FLAME_MAX_T - temp) * heatCapOf(t, g) / OXYGEN_HEAT);
}

// Fuel reaches the oxygen in ash's pores at this share of the rate it would
// fresh air's (slowing a fire, not smothering it): what burns against ash
// uses its pores' air up, and that is made good from the air around (ash
// carries oxygen; fresh ash has none left), so a sealed box's fire goes out
// once its air is used up, ash or not.
const ASH_OXYGEN: f32 = 0.8;

// A flame is gas burning: below this it no longer glows (the Draper point,
// 525 °C), and a flame shown there dies down (update.wgsl).
const FLAME_T: f32 = 500.0;
// An explosion's gas (BLAST) shows as a fireball while it and the gas beside
// it are at least this hot (update.wgsl, block.wgsl).
const FIREBALL_T: f32 = 1000.0;

// The vapour pressure over water at temp (°C), atm: the Magnus formula
// (Alduchov & Eskridge 1996, "Improved Magnus form approximation of
// saturation vapor pressure"), 1 atm at 100 °C; held at -60 °C below and
// 200 °C above.
fn vapourPressure(temp: f32) -> f32 {
    let t = clamp(temp, -60.0, 200.0);
    return exp(17.625 * t / (243.04 + t) - 17.625 * 100.0 / 343.04);
}

// The temperature (°C) water boils at under atm atmospheres: vapourPressure
// turned round (held at 200 °C, as it is).
fn boilingAt(atm: f32) -> f32 {
    let y = log(max(atm, 1e-6)) + 17.625 * 100.0 / 343.04;
    return select(200.0, clamp(243.04 * y / (17.625 - y), -60.0, 200.0), y < 17.0);
}

// Vapour that saturates a cell of gas at temp (°C) under atm atmospheres:
// the vapour pressure, at most atm (past its boiling point there water's
// vapour is all the gas there can be), as a density by the ideal gas law:
// VAPOUR_MAX at 100 °C and 1 atm. It roughly doubles every 11 °C: VAPOUR_20
// at 20 °C. (js/shared.js saturation is the same.) For the fresh air a scene
// or the outdoors starts with; what a cell of gas holds goes by vapourRoom.
fn saturationAt(temp: f32, atm: f32) -> f32 {
    return VAPOUR_MAX * min(vapourPressure(temp), atm) * BOIL_K / (max(temp, -60.0) + KELVIN);
}

// At 1 atm.
fn saturation(temp: f32) -> f32 {
    return saturationAt(temp, 1.0);
}

// The vapour (units) gas holding dry of its own gas (amount) holds at temp
// (°C) at its zone's pressure, atm: as much as makes its partial pressure
// the water's vapour pressure there (Dalton's law: the vapour's share of
// the gas, n_v / (n_dry + n_v), is e_s / p), dry e_s / (p - e_s) of it,
// whatever room the gas takes up; past the boiling point at atm, without
// end (VAPOUR_ENDLESS: the water boils). None with no gas of its own: vapour
// alone condenses down to its boiling point. 1 % over saturationAt for air
// at 1 atm and 20 °C.
const VAPOUR_ENDLESS: f32 = 1e30;

fn vapourRoom(dry: f32, temp: f32, atm: f32) -> f32 {
    let es = vapourPressure(temp);
    if (es >= atm) {
        return VAPOUR_ENDLESS;
    }
    return dry * es / (atm - es) * BOIL_K * VAPOUR_MAX / (AMBIENT_T + KELVIN);
}

// Fresh air's vapour (makeCell), the outdoor air's at 20 °C.
const OUTDOOR_VAPOUR: f32 = OUTDOOR_HUMIDITY * VAPOUR_20;

// The open air cools with height, 6.5 °C a kilometre (the ICAO standard
// atmosphere): nothing in a world a centimetre a cell, a few degrees over a
// landscape a few metres a cell, which keeps the snow on its peaks. Altitude
// is in metres up from the bottom of the world (Params.dx).
// LAPSE_RATE, °C per metre, comes from js/shared.js.
// Rising air cools as it expands by g / cp, 9.8 °C a kilometre (the dry
// adiabatic lapse rate; Wallace & Hobbs, Atmospheric Science, 2006, ch. 3).
const DRY_ADIABATIC: f32 = 0.0098;  // °C per metre

// The open air's temperature at an altitude, over the whole day; through the
// day it is dayWarmth warmer or cooler than this.
fn outdoorT(altitude: f32) -> f32 {
    return AMBIENT_T - LAPSE_RATE * altitude;
}

// The open air's vapour at an altitude: as humid at any height, so the colder
// air up high isn't fog. Clouds form where moister air rises into it. It
// doesn't follow the day: as the air cools at night it gets more humid, and
// dew and mist come out of it by dawn.
fn outdoorVapour(altitude: f32) -> f32 {
    return OUTDOOR_HUMIDITY * saturation(outdoorT(altitude));
}

// The open air's gas at an altitude, at temp: as humid as the outdoor air
// there (outdoorVapour), and as much dry air as the open air's pressure
// leaves room for beside its vapour, never less than none (js/shared.js
// outdoorAirGas is the same: what a cleared world's air is).
fn outdoorGas(altitude: f32, temp: f32) -> Gas {
    let vapour = outdoorVapour(altitude);
    return Gas(vapour, 0.0, 0.0, max(airAmount(openAirAtm(altitude), temp) - vapourAmount(vapour), 0.0));
}

// The open air's pressure at an altitude, in atm: 1 at the bottom of the
// world, less with height by a scale height of R T / (M g) at 20 °C
// (SCALE_HEIGHT, js/shared.js).
fn openAirAtm(altitude: f32) -> f32 {
    return exp(-altitude / SCALE_HEIGHT);
}

// The day. The sun rises on the left (east) at 6:00, stands overhead at noon
// and sets on the right at 18:00; day is the share of the day gone since
// midnight (Params.day).
const SUNRISE: f32 = 0.25;
const WARMEST: f32 = 0.625;       // 15:00, when the outdoor air is warmest
const OUTDOOR_SWING: f32 = 5.0;   // °C the outdoor air is warmer than outdoorT then, and cooler at dawn

// Unit vector from the world toward the sun (y grows downward); it points
// below the horizon at night.
fn sunDirection(day: f32) -> vec2<f32> {
    let a = (day - SUNRISE) * 6.2831853;
    return vec2<f32>(-cos(a), -sin(a));
}

// Direct sunlight, for a surface square to it: its colour and brightness,
// 1 at noon, reddened and dimmed low over the horizon (through more air),
// and none at night.
fn sunlight(day: f32) -> vec3<f32> {
    let height = -sunDirection(day).y;
    let tint = mix(vec3<f32>(1.0, 0.45, 0.18), vec3<f32>(1.0, 0.96, 0.9), smoothstep(0.0, 0.45, height));
    return tint * smoothstep(-0.02, 0.15, height);
}

// What the open sky shines with: blue by day, a red glow around sunrise and
// sunset, and at night a faint glow of moon and stars, so night isn't black.
fn skylight(day: f32) -> vec3<f32> {
    let height = -sunDirection(day).y;
    let night = vec3<f32>(0.018, 0.022, 0.045);
    let noon = vec3<f32>(0.2, 0.31, 0.52);
    let dusk = vec3<f32>(0.32, 0.15, 0.1) * exp(-40.0 * height * height);
    return mix(night, noon, smoothstep(-0.12, 0.35, height)) + dusk;
}

// How much warmer the open air is than outdoorT at the time of day: coolest
// at dawn and warmest in mid-afternoon, OUTDOOR_SWING either way. So the
// outdoor air's temperature is outdoorT(altitude) + dayWarmth(day): gas
// under open sky mixes toward it (update.wgsl), the air's buoyancy is
// measured from it (air.wgsl), and the sky's and the ground's thermal
// radiance follow it (light.wgsl).
fn dayWarmth(day: f32) -> f32 {
    // Since dawn: warming until WARMEST, cooling through the evening and night.
    let u = fract(day - SUNRISE);
    let rise = WARMEST - SUNRISE;
    let w = select(cos(3.1415927 * (u - rise) / (1.0 - rise)), -cos(3.1415927 * u / rise), u < rise);
    return OUTDOOR_SWING * w;
}

// Two clocks for heat. A step is the share of a day the clock moves on in it
// (Params.dayDt: 6 seconds at the normal speed, 2 slow, 24 fast, and 6 with
// the clock stopped), and a step of motion in real time (Params.dt: 1/60 of
// a second at a centimetre a cell). Heat moves at its real rate (W) times
// how long a step stands for: near the outdoor temperature on the day's
// clock, so the sun warms the ground, the night sky chills it and dew forms
// in a day as long as the day is; around hot things in real time, so lava
// stays molten for minutes and red-hot steel glows a long while, as they
// would. Which clock, where, is the clock field (clock.wgsl): how real-time
// heat is, 0 to 1, as real-time as the hottest matter within CLOCK_REACH
// cells is (realness: none to DAY_HEAT, all from REAL_HEAT), less over
// CLOCK_REACH cells more.
const DAY_HEAT: f32 = 150.0;   // °C
const REAL_HEAT: f32 = 500.0;  // °C, about where things start to glow
const CLOCK_REACH: f32 = 8.0;  // cells

fn realness(temp: f32) -> f32 {
    return clamp((temp - DAY_HEAT) / (REAL_HEAT - DAY_HEAT), 0.0, 1.0);
}

// Seconds a step stands for at a realness of real: from the day's clock,
// day seconds a step, to real time, evenly in the logarithm, span being
// log2 of real time's step over the day's (Params.dayDt and clockSpan).
fn clockStep(real: f32, day: f32, span: f32) -> f32 {
    return day * exp2(real * span);
}

// The chance that something happening to a whole cell at `rate` a second
// (a Poisson process) has happened in `tau` seconds, the clock its process
// runs on (REWRITE_PLAN.md stage 2's table): 1 - e^(-rate tau), so how
// often it happens goes by time, not by steps or how often a pass looks
// (Gillespie, J. Chem. Phys. 115, 2001).
fn chanceOver(rate: f32, tau: f32) -> f32 {
    return 1.0 - exp(-rate * tau);
}

// Heat at the world's scale. A cell holds its heat per CELL_M cube of it
// (js/shared.js), as the element table gives heat capacities, and what
// reaches it through a face is worked out for a face CELL_M across. A cell
// dx across has faces (dx / CELL_M)² as big, and (dx / CELL_M)³ cubes to
// share what comes through them: faceShare each. Conduction across a face
// goes down with the distance between the cells' middles, dx, too: a share
// faceShare² (update.wgsl heatPassed).
fn faceShare(dx: f32) -> f32 {
    return CELL_M / dx;
}

// Conduction. The conductance between a cell of e and one of n beside it,
// W/K (per CELL_M cube, see faceShare), in a world of cells dx across:
// through the two halves in series, by their thermal conductivities, or
// where a gas meets anything else, through the gas's film, film (W/(m²·K),
// gasFilm). Nothing for anything that doesn't conduct (walls, void).
fn faceConductance(e: Elem, n: Elem, film: f32, dx: f32) -> f32 {
    return conductanceOf(e, n, e.thermal, n.thermal, film, dx);
}

// The same, their thermal conductivities ke and kn as they are (a soil's
// by its water: thermalOf).
fn conductanceOf(e: Elem, n: Elem, ke: f32, kn: f32, film: f32, dx: f32) -> f32 {
    if (ke <= 0.0 || kn <= 0.0) {
        return 0.0;
    }
    let share = faceShare(dx);
    let series = CELL_M * 2.0 * ke * kn / (ke + kn) * share * share;
    return select(series, CELL_M * CELL_M * film * share, (e.kind == KIND_GAS) != (n.kind == KIND_GAS));
}

// The film between a cell of gas t holding g at temp (°C) and a surface at
// surface: its own (Elem.film, still air's few W/(m²·K)), and where its
// vapour condenses on the surface (its partial pressure over the water's at
// the surface's temperature) a condensing film's, which the air in it holds
// back: Uchida's correlation, 380 W (the air's mass over the vapour's)^-0.707
// (Uchida, Oyama & Togo 1965, as containment codes take it), up to pure
// steam's (Steam's film, elements.js). So steam gives its heat up to a cold
// surface fast, and a bubble of it in cold water condenses, while humid air
// barely does. And wind past the surface, wind m/s, adds forced convection
// to its own film, WIND_FILM a metre a second (Watmuff, Charters & Proctor
// 1977, wind over a surface outdoors: h = 2.8 + 3.0 v W/(m²·K); REWRITE_PLAN.md
// 4e-7): a fan's 7 m/s cools a surface five times as fast as still air.
const WIND_FILM: f32 = 3.0;
fn gasFilm(t: u32, g: Gas, temp: f32, surface: f32, wind: f32) -> f32 {
    let e = elems[t];
    let own = e.film + WIND_FILM * wind;
    let vapour = vapourAmount(g.vapour);
    if (vapour <= 0.0 || vapour * max(temp + KELVIN, 1.0) / (AMBIENT_T + KELVIN) <= vapourPressure(surface)) {
        return own;
    }
    let air = g.amount * e.molar / (vapour * 18.0);
    let condensing = select(elems[STEAM].film, min(380.0 * pow(air, -0.707), elems[STEAM].film), air > 0.0);
    return own + condensing;
}

// The film across the face between a cell of t holding g at temp and one of
// nt holding ng at nTemp, where one of them is gas: the gas's (gasFilm), the
// wind in that gas wind m/s.
fn faceFilm(t: u32, g: Gas, temp: f32, nt: u32, ng: Gas, nTemp: f32, wind: f32) -> f32 {
    if (elems[t].kind == KIND_GAS) {
        return gasFilm(t, g, temp, nTemp, wind);
    }
    return gasFilm(nt, ng, nTemp, temp, wind);
}

// Water against a surface (REWRITE_PLAN.md 4e-5; Incropera, Fundamentals of
// Heat and Mass Transfer, ch. 9 and 10), W/(m²·K) of the step between them:
// what its own flow carries, which still water half a cell deep (its
// conduction) only floors. Below boiling, natural convection, turbulent
// over a hot plate: Nu = 0.15 Ra^(1/3), so h = 0.15 k (g β ΔT / (ν α))^(1/3),
// whatever the size (WATER_CONVECTION: water's k, β, ν, α near 50 °C;
// some 800 W/(m²·K) at 10 °C, 1700 at 100). With the surface past water's
// boiling point, the boiling curve by the surface's superheat ΔTe:
// nucleate boiling (Rohsenow's, q = NUCLEATE ΔTe³, for a common surface,
// Csf 0.013), up to the critical heat flux (Zuber's, ZUBER_FLUX at 1 atm,
// more the colder the water, Ivey & Morris's SUBCOOLED_CHF); film boiling
// (Bromley's, the vapour film's conduction, with its radiation) past the
// Leidenfrost point, which colder water pushes far up (Dhir & Purohit 1978:
// 201 + 8 ΔTsub K: water at 20 °C wets a surface up to some 940 °C, so hot
// metal or glass quenched in it boils it off fast, and lava, hotter, is
// held off by its steam); between the two, transition boiling (log-log).
// Both scale with the Earth's pull g (m/s²): convection as g^(1/3),
// nucleate boiling as g^(1/2), the critical flux as g^(1/4), film boiling
// as g^(1/4). Its boiling point is at 1 atm (Elem.hiT); where it boils goes
// by the pressure (update.wgsl). dx: the world's cells' size, m.
const WATER_CONVECTION: f32 = 168.0;   // 0.15 k (β / (ν α))^(1/3), W/(m²·K^(4/3)·(m/s²)^(1/3))
const NUCLEATE: f32 = 137.0;           // W/(m²·K³), at 9.81 m/s²
const ZUBER_FLUX: f32 = 1.26e6;        // W/m², saturated, 1 atm, 9.81 m/s²
const SUBCOOLED_CHF: f32 = 0.0469;     // 0.1 (ρl/ρv)^0.75 cp / hfg, per K of subcooling
const BROMLEY: f32 = 727.0;            // the vapour film's kv³ ρv (ρl - ρv) / μv
const STANDARD_G: f32 = 9.81;          // m/s², what those are at
fn waterFilm(we: Elem, water: f32, surface: f32, emissivity: f32, g: f32, dx: f32) -> f32 {
    let step = abs(surface - water);
    let floor = 2.0 * we.thermal / dx;
    if (step <= 0.0) {
        return floor;
    }
    let pull = max(g, 1e-3);
    var q = WATER_CONVECTION * pow(pull * step, 1.0 / 3.0) * step;
    let superheat = surface - we.hiT;
    if (superheat > 0.0) {
        let subcooled = max(we.hiT - water, 0.0);
        let top = ZUBER_FLUX * pow(pull / STANDARD_G, 0.25) * (1.0 + SUBCOOLED_CHF * subcooled);
        let nucleate = NUCLEATE * sqrt(pull / STANDARD_G);
        let critical = pow(top / nucleate, 1.0 / 3.0);
        let leidenfrost = 201.0 + 8.0 * subcooled;
        var boiling = 0.0;
        if (superheat <= critical) {
            boiling = nucleate * superheat * superheat * superheat;
        } else if (superheat >= leidenfrost) {
            boiling = filmBoiling(superheat, surface, we.hiT, emissivity, pull, dx);
        } else {
            let along = (log(superheat) - log(critical)) / (log(leidenfrost) - log(critical));
            boiling = exp(mix(log(top), log(filmBoiling(leidenfrost, we.hiT + leidenfrost, we.hiT, emissivity, pull, dx)), along));
        }
        q = max(q, boiling);
    }
    return max(q / step, floor);
}

// Film boiling's heat flux (W/m²) at superheat (K) from a surface at
// surface (°C) to water boiling at boil (°C): Bromley's conduction across
// the vapour film over a body dx across (its latent heat with the vapour's
// superheat, 0.8 cp,v ΔT), plus three quarters of the surface's radiation
// across it (Bromley's combination, as h^(4/3) = hc^(4/3) + hr h^(1/3)).
fn filmBoiling(superheat: f32, surface: f32, boil: f32, emissivity: f32, g: f32, dx: f32) -> f32 {
    let latent = 2.257e6 + 1600.0 * superheat;
    let conducted = 0.62 * pow(BROMLEY * g * latent / (dx * superheat), 0.25);
    let s = surface + KELVIN;
    let b = boil + KELVIN;
    let radiated = emissivity * 5.67e-8 * (s * s * s * s - b * b * b * b) / superheat;
    return (conducted + 0.75 * radiated) * superheat;
}

// The density (relative to water's) of liquid of word x at stored
// temperature temp: at the temperature it acts at (effTempIn, not counting
// latent heat it holds: boiling water past its boiling point is at it), so
// where warm liquid sits beside cold the projection can't balance gravity
// with pressure alone, and the warm liquid rises: convection. What water
// carries makes it denser (solDensity), so brine sinks under fresh water.
fn liquidDensity(x: u32, temp: f32) -> f32 {
    let t = wordType(x);
    let solute = wordSolute(x);
    let acts = effTempIn(t, temp, phaseOf(t, solute, wordStrength(x), 0.0));
    var density = elems[t].density / 100.0 * expandedAt(t, acts);
    if (holdsSolute(t) && solute != 0u) {
        density += elems[solute].solDensity * shareOf(wordStrength(x));
    }
    return density;
}

// A liquid's convection across heat's step (REWRITE_PLAN.md 4e-8), as a
// sealed room's air has its own (update.wgsl's mixOf, 4e-4). Heat goes on
// heat's clock, τ a step (up to the day's 6 s), but the liquid moves a real
// step, dt, at a time: water beside steel warmer than it took the steel's
// heat and kept it, a cell thick, and a pool cooled at its surface stayed
// cold on top. Over the rest of heat's step, τ - dt, two cells of liquid
// one over the other, the upper the denser (liquidDensity: warm water under
// cold, above 4 °C; brine over fresh), turn over as a parcel a cell across
// does under its buoyancy, in t = √(dx ρ / (g Δρ)): a share 1 - e^(-(τ -
// dt) / t) of their difference evens out, a conductance (W/K per CELL_M
// cube) of that share of their heat capacities in series over τ (convective
// adjustment: Bryan 1969, J. Comput. Phys. 4; Marotzke 1991). Lighter over
// denser is stable and left to conduction, and a step on real time (τ =
// dt) to the flow alone. Two side by side (beside), the denser slumps under
// the lighter as a lock exchange does, its front at half √(g' dx): in twice
// that time, whichever is the denser. (Beside a warm wall the water is
// carried off sideways as well as up: with only the cells one over the
// other turning over, steel in water lost heat at 120 W/(m²·K) where
// natural convection has some 1000.) upper and lower are the two's words
// and stored temperatures (for beside, either way), capUpper and capLower
// their heat capacities, g the Earth's pull (m/s², straight down the
// screen), dx the cells' size (m), dt the flow's step.
// At most TURNOVER_MAX of the smaller heat capacity a step, under what
// update.wgsl passes without heat.wgsl's stages (HEAT_SHARE_MAX), so the
// turnover makes no face stiff by itself: two cells even out up to 40 % of
// what they differ by a step, against a turnover a fraction of a second
// long. (At the whole share every pool's faces went to the stages, which
// then ran each frame: the Sealed boiler 6 % slower.)
const TURNOVER_MAX: f32 = 0.8 * HEAT_SHARE_MAX;
fn turnoverConductance(upper: u32, upperTemp: f32, lower: u32, lowerTemp: f32, capUpper: f32, capLower: f32,
    g: f32, dx: f32, tau: f32, dt: f32, beside: bool) -> f32 {
    let lag = tau - dt;
    if (lag <= 0.0 || g <= 0.0 || capUpper <= 0.0 || capLower <= 0.0) {
        return 0.0;
    }
    let ru = liquidDensity(upper, upperTemp);
    let rl = liquidDensity(lower, lowerTemp);
    if (ru == rl || (ru < rl && !beside)) {
        return 0.0;
    }
    let turn = select(1.0, 2.0, beside) * sqrt(dx * 0.5 * (ru + rl) / (g * abs(ru - rl)));
    let share = (1.0 - exp(-lag / turn)) * capUpper * capLower / (capUpper + capLower);
    return min(share, TURNOVER_MAX * min(capUpper, capLower)) / tau;
}

// The conductance (W/K, per CELL_M cube) across the face between a cell of t
// holding g at temp (°C) and one of nt holding ng at nTemp, as the heat
// passes (update.wgsl) and its stages (heat.wgsl) both take it: water
// against condensed matter but water (a solid, or another liquid: lava,
// oil, molten metal) through water's film (waterFilm, the Earth's pull
// earthPull m/s²) in series with the other's half cell; a gas against
// anything else through the gas's film (the wind in the gas, wind m/s);
// else conduction through the two halves (faceConductance), a soil's by
// the water it holds (word and nWord the two cells' words: thermalOf).
// (Powder in water stays by conduction: what it does there goes on between
// its grains.)
fn cellFaceConductance(t: u32, word: u32, g: Gas, temp: f32, nt: u32, nWord: u32, ng: Gas, nTemp: f32, earthPull: f32, dx: f32,
    wind: f32) -> f32 {
    let e = elems[t];
    let n = elems[nt];
    let wetted = n.kind == KIND_SOLID || (n.kind == KIND_LIQUID && !isWater(nt));
    let wetting = e.kind == KIND_SOLID || (e.kind == KIND_LIQUID && !isWater(t));
    if ((isWater(t) && wetted) || (isWater(nt) && wetting)) {
        let water = select(nt, t, isWater(t));
        let other = select(t, nt, isWater(t));
        let we = elems[water];
        let oe = elems[other];
        if (we.thermal <= 0.0 || oe.thermal <= 0.0) {
            return 0.0;
        }
        let film = waterFilm(we, select(nTemp, temp, isWater(t)), select(temp, nTemp, isWater(t)), oe.emissivity, earthPull, dx);
        let share = faceShare(dx);
        let k = thermalOf(other, select(word, nWord, isWater(t)));
        return 1.0 / (1.0 / (CELL_M * CELL_M * film * share) + 1.0 / (CELL_M * 2.0 * k * share * share));
    }
    return conductanceOf(e, n, thermalOf(t, word), thermalOf(nt, nWord), faceFilm(t, g, temp, nt, ng, nTemp, wind), dx);
}

// Seconds a step stands for in heat passing between cells of e and n, tau by
// the faster of their clocks (clock.wgsl). Heaters and coolers (F_FIXED_TEMP),
// held at their temperature by the power they take, are stoves and
// freezers: they pass heat on the day's clock (day) wherever they are, as
// the pot on a stove boils in a time-lapse along with the day (in real time
// a 600 °C element would take most of an hour), though the glass under it
// gets hot enough to put the pot on real time.
fn faceSeconds(e: Elem, n: Elem, tau: f32, day: f32) -> f32 {
    return select(tau, day, ((e.flags | n.flags) & F_FIXED_TEMP) != 0u);
}

// Conduction is worked out explicitly (update.wgsl) across a face that
// passes no more in a step than HEAT_SHARE_MAX (js/shared.js) of the smaller
// heat capacity of its two cells per degree between them: four such faces
// never take a cell past what is around it (forward Euler is monotone that
// far). Heat across a stiffer face would go further than that in a step
// (metal on the day's clock; in small cells, nearly anything): those go in
// stages (heat.wgsl), as far as heat really would.

// Thermal radiance of a perfect emitter at temp (°C): σT⁴, in units of one
// at 1000 K. About 0.0074 at 20 °C, 0.73 for a flame, 7.8 for lava at
// 1400 °C. Light and radiant heat are worked out in these units (light.wgsl,
// update.wgsl).
fn thermalRadiance(temp: f32) -> f32 {
    let k = max(temp + KELVIN, 0.0) / 1000.0;
    let k2 = k * k;
    return k2 * k2;
}

// Share of visible light a cell of type t reflects, in its colour; mist
// reflects nearly all of it (fog glows in the light).
const ALBEDO: f32 = 0.8;
const MIST_ALBEDO: f32 = 0.85;

fn albedo(t: u32) -> vec3<f32> {
    if (t == AIR) {
        return vec3<f32>(MIST_ALBEDO);
    }
    let e = elems[t];
    return vec3<f32>(e.r, e.g, e.b) * ALBEDO;
}

// Plants (REWRITE_PLAN.md 5d; update.wgsl, block.wgsl, pairs.wgsl). A plant
// cell keeps the sugar it has made in its life (up to 255; a new cell costs
// GROW_SUGAR of it), and the water it holds in its aux: PLANT_WATER turgid,
// a fresh plant's 0.64 g, each unit PLANT_UNIT vapour units. Below PLANT_WILT
// it wilts: its stomata shut, and it neither makes sugar nor gives off
// water. The side that is brighter, which it grows toward, is in its
// shade's top two bits (PLANT_LEFT, PLANT_RIGHT); the rest vary its colour.
const PLANT_WATER: u32 = 255u;
const PLANT_WILT: u32 = 64u;
const PLANT_UNIT: f32 = 0.64 * WATER_VAPOUR / 255.0;
const GROW_SUGAR: u32 = 16u;
const PLANT_LEFT: u32 = 4u;
const PLANT_RIGHT: u32 = 8u;

// Visible light let through by a CELL_M cube of air holding this much mist
// (vapour units, see WATER_VAPOUR): thick fog (MIST_THICK in shade.wgsl)
// passes about 80 % a cell. (clarityAt makes it a cell's.) Not real fog's:
// the air holds some 210 times real air's water (js/shared.js), so this is
// tuned for the look, per unit of what saturates air at 20 °C.
const MIST_EXTINCTION: f32 = 0.04 / VAPOUR_20;

fn mistClarity(mist: f32) -> f32 {
    return exp(-MIST_EXTINCTION * mist);
}

// Mist in the thermal infrared: half as much as it dims light. Water takes
// in the infrared about 10 µm strongly (its absorption index there some
// 0.05: Hale & Querry, Applied Optics 12:555, 1973), so a fog droplet a few
// microns across takes in about as much of it as falls on it (an
// absorption efficiency near 1) where it scatters light about twice that
// (an extinction efficiency near 2): real fog is close to opaque in the
// thermal infrared over tens of metres, and keeps a night's cold in (it
// stops what the ground gives off to the sky). Tuned with it, as
// MIST_EXTINCTION is.
const MIST_IR: f32 = 0.5 * MIST_EXTINCTION;

// The share of light a cell dx across lets through, clear being a CELL_M
// cube's (the element table's clarity and irClarity, mistClarity): what
// stops light does so by the metre (Beer-Lambert), so a cell twice as deep
// lets through clear², and one a tenth as deep the tenth root of it. Exactly
// clear at a centimetre.
fn clarityAt(clear: f32, dx: f32) -> f32 {
    let depth = dx / CELL_M;
    return select(exp2(log2(max(clear, 1e-30)) * depth), clear, depth == 1.0);
}

// What a cell of t lets through, visible light and thermal infrared, in a
// world of cells dx across; g is what it holds if it is a gas (its gas
// plane's).
// The light passes (light.wgsl) and what its cells take in of it
// (update.wgsl radiantHeat) both go by this.
// A gas's mist dims what passes (mistClarity; in the infrared MIST_IR), and
// its vapour takes in the infrared, VAPOUR_IR a metre a vapour unit (a cell
// of water's worth, 64 of them, 4 a metre, as steam's was).
const VAPOUR_IR: f32 = 4.0 / 64.0;

fn cellClarity(t: u32, g: Gas, dx: f32) -> vec2<f32> {
    let e = elems[t];
    var clear = vec2<f32>(e.clarity, e.irClarity);
    if (e.kind == KIND_GAS) {
        clear *= vec2<f32>(mistClarity(g.mist), exp(-VAPOUR_IR * g.vapour * CELL_M - MIST_IR * g.mist));
    }
    return vec2<f32>(clarityAt(clear.x, dx), clarityAt(clear.y, dx));
}

// The same for cell c: a cell of liquid by how much it holds
// (liquidAmount), as that much of the way across it, the rest open.
fn clarityOf(c: Cell, g: Gas, dx: f32) -> vec2<f32> {
    let depth = select(dx, dx * liquidAmount(c), elems[c.t].kind == KIND_LIQUID);
    return cellClarity(c.t, g, depth);
}

fn unpackCell(v: vec4<u32>) -> Cell {
    var c: Cell;
    c.t = cellType(v);
    c.aux = cellAux(v);
    c.life = cellLife(v);
    c.shade = cellShade(v);
    c.flags = cellFlags(v);
    c.temp = cellTemp(v);
    c.vel = cellVel(v);
    c.frac = cellFrac(v);
    c.w = v.w;
    return c;
}

// A liquid cell's w is a particle index, kept bit for bit (it wouldn't survive
// a round trip through f16).
fn packCell(c: Cell) -> vec4<u32> {
    let x = packWord(c.t, c.aux, c.life, c.shade, c.flags);
    let w = select(pack2x16float(c.frac), c.w, elems[wordType(c.t)].kind == KIND_LIQUID);
    return vec4<u32>(x, bitcast<u32>(c.temp), pack2x16float(c.vel), w);
}

fn airDims(size: vec2<u32>) -> vec2<u32> {
    return (size + AIR_CELL - 1u) / AIR_CELL;
}

// Rooms (sky.wgsl's room passes): the gas (and what the air flows through),
// joined up into rooms, the gas connected to each other, the air's pressure
// zones (air.wgsl): the open air one of them, the one that reaches the
// world's sides or bottom. Found per ROOM_TILE square of the world first, as
// pieces of the tile (up to ROOM_PIECES a tile, the biggest first; any more,
// UNNUMBERED, are pockets without a zone of their own), then the pieces
// joined across the tiles' edges into trees whose roots are the rooms' first
// pieces. The sky buffer holds, after the skyline (a word per column): a
// byte per cell, its piece in its tile, UNNUMBERED or NO_ROOM, four cells a
// word along a row of a tile; and a word per piece, the lower-numbered piece
// it is joined to (itself for a room's first), as a piece's number is tile *
// ROOM_PIECES + its piece. Then the ventilation (ventilationAt), and the
// zones' sums (zoneSlotAt). ROOM_TILE (a multiple of the 8 cells a side of
// the air's workgroups) and ROOM_PIECES come from js/shared.js.
const_assert ROOM_TILE % 8u == 0u;
const_assert ROOM_PIECES <= 64u;
const NO_ROOM: u32 = 255u;
const UNNUMBERED: u32 = ROOM_PIECES;

fn roomTiles(size: vec2<u32>) -> vec2<u32> {
    return (size + ROOM_TILE - 1u) / ROOM_TILE;
}

fn roomPieceCount(size: vec2<u32>) -> u32 {
    let tiles = roomTiles(size);
    return tiles.x * tiles.y * ROOM_PIECES;
}

// Index in the sky buffer of the word holding cell p's room byte.
fn roomWord(size: vec2<u32>, p: vec2<u32>) -> u32 {
    let tiles = roomTiles(size);
    let tile = p / ROOM_TILE;
    let at = p % ROOM_TILE;
    return size.x + (tile.y * tiles.x + tile.x) * (ROOM_TILE * ROOM_TILE / 4u) + at.y * (ROOM_TILE / 4u) + at.x / 4u;
}

// Index in the sky buffer of piece k's word: the piece it is joined to.
fn roomLink(size: vec2<u32>, k: u32) -> u32 {
    let tiles = roomTiles(size);
    return size.x + tiles.x * tiles.y * (ROOM_TILE * ROOM_TILE / 4u) + k;
}

// The number of the piece cell p's room byte names.
fn roomPieceAt(size: vec2<u32>, p: vec2<u32>, piece: u32) -> u32 {
    let tiles = roomTiles(size);
    let tile = p / ROOM_TILE;
    return (tile.y * tiles.x + tile.x) * ROOM_PIECES + piece;
}

// Index in the sky buffer of column x's ventilation (sky.wgsl): after the
// rooms' words, a word per column, the row its open air is in the wind
// above; then as many again for the pass that works it out
// (ventilationAt(size, size.x + x)).
fn ventilationAt(size: vec2<u32>, x: u32) -> u32 {
    return roomLink(size, roomPieceCount(size)) + x;
}

// The air's pressure zones (air.wgsl), after the ventilation: per room (its
// first piece's number), two slots, by the step's parity, of ZONE_WORDS
// (js/shared.js) words, sums over its gas cells in fixed point (1 /
// ZONE_SCALE), rounded at random so that they add up: ZONE_AMOUNT, all the
// gas they hold (gasAmount); ZONE_VOLUME, what they would take up at 1 atm
// as a share of their volume at 20 °C (T_ref / T, each); ZONE_COUNT, how
// many; ZONE_OPEN, whether any meets the outside air; ZONE_SOURCE, what of
// their gas they give out (+) or take in (-) to come to the zone's pressure
// (air.wgsl's expansion), as a 64-bit number in two words, low then high
// (air.wgsl's wideParts: exact, its mean what makes the zone's flow balance);
// ZONE_SINK_DRAW, what the black holes in it take in a step (air.wgsl's
// sinkSpeed, a face its gas meets one by), in cells' volume, ZONE_SCALE;
// ZONE_POTENTIAL (exact, at air.wgsl's POTENTIAL_SCALE), a sealed one's
// projection potential summed (air.wgsl's zonePotential), the level its
// dynamic pressure is measured from; from ZONE_MOMENTS, its gas's moments
// (exact, EXACT_COARSE, two words each; air.wgsl's gather): Σ C, Σ C T,
// Σ C h, Σ C T h and Σ C h², C a cell's heat capacity, T its temperature
// (°C) and h its height along the pull (pull.wgsl's altitudeAt, metres);
// ZONE_MIX_CAP and ZONE_MIX_HEAT (exact, EXACT_FINE), what its cells mix
// by on heat's clock, s C and s C T (update.wgsl's zoneHeat;
// REWRITE_PLAN.md 4e-4); and ZONE_PIN, the first of its gas cells by index
// (the index's complement, the most of them), whose potential a sealed
// zone's projection holds (air.wgsl's pinned, F-5).
const ZONE_AMOUNT: u32 = 0u;
const ZONE_VOLUME: u32 = 1u;
const ZONE_COUNT: u32 = 2u;
const ZONE_OPEN: u32 = 3u;
const ZONE_SOURCE: u32 = 4u;
const ZONE_SINK_DRAW: u32 = 6u;
const ZONE_POTENTIAL: u32 = 7u;
const ZONE_MOMENTS: u32 = 9u;
const ZONE_MIX_CAP: u32 = 19u;
const ZONE_MIX_HEAT: u32 = 21u;
const ZONE_PIN: u32 = 23u;
const_assert ZONE_PIN < ZONE_WORDS;
const ZONE_SCALE: f32 = 1024.0;
// Bits of ZONE_OPEN: the room meets the outside air (and is part of it), or
// has a black hole in it (somewhere for its gas to go, so it isn't squeezed
// as a sealed room is).
const ZONE_OUTSIDE: u32 = 1u;
const ZONE_SINK: u32 = 2u;
// A sealed zone of fewer cells is a speck of gas between matter, not a room:
// a cell a body has just left (a vacuum till filled), a notch of a turned
// stamp's steps, a corner of a crate its cup leaves out. What reads the air
// takes it as a gap holding none (air.wgsl's project: what is beside it,
// and no body's face meets it), and it is no wet room's surface. (By the
// gas law over a cell or two, swung by a cell's worth of gas pushed in or a
// vacuum, specks put an atmosphere across what they touched: a sealed glass
// crate afloat, whose inside took the pressure of a one-cell vacuum as its
// wet room's surface, shattered whole; two cells of air in its corner rose
// to 1.5 atm and threw it; loose wood by a notch crumbled. At the open air's
// pressure instead, in a sealed room at another, they pressed a block into
// the slope it lay on with the room's difference.)
const ZONE_MIN: u32 = 4u;

fn zoneSlotAt(size: vec2<u32>, frame: u32, room: u32) -> u32 {
    return ventilationAt(size, 2u * size.x) + ZONE_WORDS * ((frame % 2u) * roomPieceCount(size) + room);
}

// Wet rooms (REWRITE_PLAN.md `bubble-pressure`): the rooms found again with
// the liquid joining them (sky.wgsl's room passes with WET), so that the gas
// the liquid closes in is in one with the gas over the liquid. After the
// zones' sums, laid out as the rooms are: a byte per cell (wetWord), a word
// per piece (wetLink); then per wet room (its first piece's number) two
// slots, by the step's parity, of WET_WORDS (js/shared.js) words: WET_OPEN,
// whether it reaches the open air (air.wgsl's gather); WET_MOST, the most
// cells of any of its zones beside the liquid or a powder (air.wgsl's
// divergence), and WET_SURFACE, that zone's room's number + 1
// (zonePotential): the zone the liquid's free surface faces where it doesn't
// reach the open air (rooms.wgsl's surfaceAtm); WET_TAKERS, how many cells
// take in what its pockets give out, as much each (carry.wgsl's wetOf); from
// WET_GIVEN,
// what they give out (exact, EXACT_FINE, two words each; as
// airsource.wgsl's gas: amount, vapour, mist, oxygen used, heat); and
// WET_GIVERS (exact, EXACT_FINE), what the cells giving it out weigh
// together, to have it back where nothing takes it.
const WET_OPEN: u32 = 0u;
const WET_MOST: u32 = 1u;
const WET_SURFACE: u32 = 2u;
const WET_TAKERS: u32 = 3u;
const WET_GIVEN: u32 = 4u;
const WET_GIVERS: u32 = 14u;
const_assert WET_GIVERS + 1u < WET_WORDS;

fn wetBase(size: vec2<u32>) -> u32 {
    return ventilationAt(size, 2u * size.x) + 2u * ZONE_WORDS * roomPieceCount(size) - size.x;
}

fn wetWord(size: vec2<u32>, p: vec2<u32>) -> u32 {
    return wetBase(size) + roomWord(size, p);
}

fn wetLink(size: vec2<u32>, k: u32) -> u32 {
    return wetBase(size) + roomLink(size, k);
}

fn wetSlotAt(size: vec2<u32>, frame: u32, room: u32) -> u32 {
    return wetLink(size, roomPieceCount(size)) + WET_WORDS * ((frame % 2u) * roomPieceCount(size) + room);
}

// Exact sums (REWRITE_PLAN.md D-1): what many threads add into one place in
// a pass, as a 64-bit two's complement number of units of 1 / scale in two
// words, low then high, added to with a carry from the low word into the
// high (air.wgsl's wideParts the same, rounded at random; a module adds
// with its own buffer, as
// `let w = exactParts(x, scale); let old = atomicAdd(&buf[at], w.x);
// atomicAdd(&buf[at + 1u], w.y + exactCarry(old, w.x));`). Integers add up
// the same in any order: an f32 added to by compare-and-swap came out
// otherwise as the GPU ran the threads in another order, and two runs from
// one snapshot came apart. Each part is rounded to the nearest unit (none
// is lost as the sum grows, as an f32's low bits are). EXACT_FINE keeps
// sums within ±2^31, EXACT_COARSE within ±2^39.
const EXACT_FINE: f32 = 4294967296.0;
const EXACT_COARSE: f32 = 16777216.0;

fn exactParts(x: f32, scale: f32) -> vec2<u32> {
    let s = round(clamp(x * scale, -9.2e18, 9.2e18));
    if (abs(s) < 2147483648.0) {
        let v = i32(s);
        return vec2<u32>(bitcast<u32>(v), select(0u, 0xffffffffu, v < 0));
    }
    // (Past 2^31 an f32 is a whole multiple of 2^8: the low word exact.)
    let hi = floor(s / 4294967296.0);
    return vec2<u32>(u32(s - hi * 4294967296.0), bitcast<u32>(i32(hi)));
}

// A whole number n (an i32) as exactParts gives one, n units.
fn exactWhole(n: i32) -> vec2<u32> {
    return vec2<u32>(bitcast<u32>(n), select(0u, 0xffffffffu, n < 0));
}

// The carry into the high word of adding add to the low word that was old.
fn exactCarry(old: u32, add: u32) -> u32 {
    return select(0u, 1u, old + add < old);
}

// The sum in words w (low, high), in units of 1 / scale.
fn exactValue(w: vec2<u32>, scale: f32) -> f32 {
    let l = bitcast<i32>(w.x);
    let h = bitcast<i32>(w.y);
    if ((h == 0 && l >= 0) || (h == -1 && l < 0)) {
        return f32(l) / scale;
    }
    return (f32(h) * 4294967296.0 + f32(w.x)) / scale;
}

var<private> rngState: u32;

fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

fn seedRng(a: u32, b: u32, c: u32) {
    rngState = pcg(a ^ pcg(b ^ pcg(c)));
}

fn rndU() -> u32 {
    rngState = pcg(rngState);
    return rngState;
}

fn rnd() -> f32 {
    return f32(rndU() >> 8u) * (1.0 / 16777216.0);
}

fn initLife(t: u32) -> u32 {
    switch t {
        case FIRE: { return 14u + rndU() % 40u; }
        case SMOKE: { return 60u + rndU() % 140u; }
        case BLAST: { return 2u + rndU() % 4u; }
        // What a seed has stored: growth for a seedling of 10-25 cells,
        // which in the dark is all it grows (a bean's seedling grows some
        // centimetres on its seed before it needs light).
        case SEED: { return 10u + rndU() % 16u; }
        default: { return 0u; }
    }
}

// P_ATM (js/shared.js) is the air grid's pressure (relative to the open air,
// which is at one atmosphere) per atmosphere: a sealed pot's glass lid
// (breakP 12) gives way at about an atmosphere over, about where a pressure
// cooker runs. MIN_ATM is the least absolute pressure, in atmospheres, a
// boiling point is worked out for: a black hole's suction comes close to a
// vacuum.

// The log of the absolute pressure in atmospheres, for air pressure p:
// what boiling points go by (transitionsAt).
fn atmLog(p: f32) -> f32 {
    return log(max(1.0 + p / P_ATM, MIN_ATM));
}

// Element e's (hiT, loT) under air pressure p, given as atmLog(p). Where it
// has a boilShift, its boiling transition (a liquid's hiT, a vapour's loT)
// moves with the pressure as the Clausius-Clapeyron relation has it:
// 1/T = 1/T0 - ln(p / p0) / boilShift, in kelvin, T0 being the table's
// temperature at one atmosphere. Water boils at about 121 °C two atmospheres
// up (a pressure cooker) and at about 70 °C at a third of one (on a high
// mountain). The stored temperature measures heat either way, so a moved
// boiling point turns latent heat taken in into plain heat, or back, and
// none is lost.
fn transitionsAt(e: Elem, lnAtm: f32) -> vec2<f32> {
    var t = vec2<f32>(e.hiT, e.loT);
    if (e.boilShift > 0.0 && lnAtm != 0.0) {
        let gas = e.kind == KIND_GAS;
        let t0 = select(e.hiT, e.loT, gas) + KELVIN;
        let boil = 1.0 / (1.0 / t0 - lnAtm / e.boilShift) - KELVIN;
        t = select(vec2<f32>(boil, e.loT), vec2<f32>(e.hiT, boil), gas);
    }
    return t;
}

// A liquid's kinematic viscosity (cells² a step) at the temperature it acts
// at, temp: its table's at viscosityT, thicker colder and runnier hotter by
// e every viscosityE kelvin, an Arrhenius law fitted over its working range
// (none: the same at any temperature). Lava cooling from 1400 °C to its
// solidus thickens from some 5 Pa·s to 1.6e4, so a flow crusts over and its
// hot core runs on under its crust; glass works stiffer as it cools
// (REWRITE_PLAN.md 7-17). At most VISCOSITY_MOST.
const VISCOSITY_MOST: f32 = 1e4;
fn viscosityAt(e: Elem, temp: f32) -> f32 {
    if (e.viscosityE <= 0.0) {
        return e.viscosity;
    }
    return min(e.viscosity * exp((e.viscosityT - temp) / e.viscosityE), VISCOSITY_MOST);
}

// A byte of the record's share of full: a solute's strength of saturated,
// what of a fuel has burnt.
fn shareOf(byte: u32) -> f32 {
    return f32(byte) / 255.0;
}

// A cell's (hiT, loT): under air pressure lnAtm (transitionsAt), and for
// water, moved by the solute it carries (aux) as much as it has of it (life):
// brine freezes lower and boils higher.
fn phaseOf(t: u32, solute: u32, strength: u32, lnAtm: f32) -> vec2<f32> {
    var tr = transitionsAt(elems[t], lnAtm);
    if (holdsSolute(t) && solute != 0u) {
        let s = elems[solute];
        tr += vec2<f32>(s.solBoil, s.solFreeze) * shareOf(strength);
    }
    return tr;
}

// The temperature a cell of t acts at, its stored temperature being stored
// and its phase points tr = (hiT, loT), as phaseOf gives them. A cell's heat
// is kept as a stored temperature: its energy is heatCap times that, so a
// cell taking in or giving off heat changes it by the heat over heatCap,
// and past a phase change it holds at the transition point while its
// stored temperature runs on, measuring the latent heat taken in (or given
// off) so far; see changePhase in update.wgsl. A cell that has given off all
// of its latentDown but can't change yet (brine with nowhere to put its
// salt) has frozen, and cools on as what it freezes into, gram for gram
// (Elem.fill). One that has taken
// in all of its latentUp but can't change yet (brine or mud with no air to
// boil off into) is superheated, and warms on as itself. (upSpan, downSpan
// and downScale are worked out from the rest in js/elements.js.)
fn effTempIn(t: u32, stored: f32, tr: vec2<f32>) -> f32 {
    let e = elems[t];
    return clamp(stored, tr.y, tr.x) + max(stored - tr.x - e.upSpan, 0.0)
        - max(tr.y - e.downSpan - stored, 0.0) * e.downScale;
}

// The temperature cell c acts at, at 1 atm, with what it carries (water's
// solute) moving its phase points: what every pass reads of a cell's
// temperature but update.wgsl's and block.wgsl's phase changes, which take
// the air's pressure too (effTempIn with phaseOf at it).
fn tempOf(c: Cell) -> f32 {
    return effTempIn(c.t, c.temp, phaseOf(c.t, soluteOf(c), strengthOf(c), 0.0));
}

fn wordTempOf(v: vec4<u32>) -> f32 {
    let t = cellType(v);
    return effTempIn(t, cellTemp(v), phaseOf(t, wordSolute(v.x), wordStrength(v.x), 0.0));
}

// The stored temperature of a cell of t at absolute zero: for something
// that gives off latent heat as it cools (water freezing), far enough below
// its freezing point to give it all off and then cool to absolute zero as
// what it freezes into, gram for gram (effTempIn, at 1 atm; Elem.zero):
// water's -218, its ice's heat capacity half its own. (Taken as -273
// wherever that was lower, the cool brush could take water below absolute
// zero, acting at -382.)
fn storedZero(t: u32) -> f32 {
    return elems[t].zero;
}

// A liquid's density at the temperature it acts at, temp, as a share of its
// table density: thermal expansion from its reference temperature, 20 °C,
// or its freezing point for a melt that only exists hotter (see
// `expansion`).
fn expandedAt(t: u32, temp: f32) -> f32 {
    let e = elems[t];
    let d = temp - max(AMBIENT_T, e.loT);
    return clamp(1.0 - e.expansion * d - e.expansion2 * d * d, 0.8, 1.05);
}

// Stored temperature at which a cell has taken in all of its latentUp, its
// boiling (or melting) point being hiT (phaseOf).
fn latentFull(e: Elem, hiT: f32) -> f32 {
    return hiT + e.latentUp / e.heatCap;
}

// Soil water. A porous powder (cohesive: sand, dirt, ash) holds water in its
// life, up to SOIL_FULL soaked: then it holds SOIL_HOLD of a cell of water,
// the water in its pores, so a cell of water is WATER_UNITS of it. It soaks
// up water beside it, passes it on through the grains and drains it down
// (Richards' equation, pairs.wgsl), gives it off as vapour at the surface and
// to the roots of plants (update.wgsl); damp, its grains cling (block.wgsl).
const SOIL_FULL: u32 = 255u;
const SOIL_HOLD: f32 = 0.4;
const WATER_UNITS: f32 = 255.0 / SOIL_HOLD;

// How tightly a soil of e holding life (SOIL_FULL soaked) holds its water
// (REWRITE_PLAN.md 5f): its share of the water it can give up (effective
// saturation: what it holds past its residual soilR, of what it would
// soaked), the suction that holds it (its head, m of water, none soaked,
// SOIL_DRY dry), and how fast water flows through it (m/s): van Genuchten's
// curves (1980, Soil Sci. Soc. Am. J. 44) and Mualem's conductivity (1976).
const SOIL_DRY: f32 = -1000.0;
fn soilSaturation(e: Elem, life: f32) -> f32 {
    let theta = SOIL_HOLD * life / f32(SOIL_FULL);
    return clamp((theta - e.soilR) / (SOIL_HOLD - e.soilR), 0.0, 1.0);
}

fn soilHead(e: Elem, se: f32) -> f32 {
    if (se >= 1.0) {
        return 0.0;
    }
    if (se <= 0.0) {
        return SOIL_DRY;
    }
    let m = 1.0 - 1.0 / e.soilN;
    return max(-pow(pow(se, -1.0 / m) - 1.0, 1.0 / e.soilN) / e.soilAlpha, SOIL_DRY);
}

fn soilConductivity(e: Elem, se: f32) -> f32 {
    if (se <= 0.0) {
        return 0.0;
    }
    let m = 1.0 - 1.0 / e.soilN;
    let f = 1.0 - pow(max(1.0 - pow(se, 1.0 / m), 0.0), m);
    return e.soilK * sqrt(se) * f * f;
}

// Roots draw water only down to the wilting point, a suction of 1.5 MPa
// (153 m of water; Hillel, Environmental Soil Physics, 1998): the life a
// soil of e holds there, and whether it holds more.
const WILTING_HEAD: f32 = -153.0;
fn soilWiltLife(e: Elem) -> f32 {
    let m = 1.0 - 1.0 / e.soilN;
    let se = pow(1.0 + pow(-e.soilAlpha * WILTING_HEAD, e.soilN), -m);
    return (e.soilR + se * (SOIL_HOLD - e.soilR)) / SOIL_HOLD * f32(SOIL_FULL);
}

fn soilDrinkable(e: Elem, life: u32) -> bool {
    return f32(life) > soilWiltLife(e);
}

fn porous(e: Elem) -> bool {
    return e.kind == KIND_POWDER && e.cohesion > 0.0;
}

// Whether c is water carrying a solute (aux), or a liquid that leaves the
// rest of itself behind when its water goes (RESIDUE: mud).
fn leavesResidue(c: Cell, e: Elem) -> bool {
    return (e.flags & F_RESIDUE) != 0u || (holdsSolute(c.t) && soluteOf(c) != 0u);
}

// What a liquid leaves behind once its water is gone, boiled off or
// evaporated: a RESIDUE liquid's hiTo (dirt), the solute water carried as a
// grain holding just what the water held (salt; a whole grain is
// 1 / solubility cells' fill, its life counting up what has gone), fumes
// (acid), or nothing. Of as much liquid as the cell held (liquidAmount):
// a quarter of a cell of mud leaves a cell of dirt one time in four, a
// quarter of a cell of brine a quarter of the salt.
fn residueOf(c: Cell, e: Elem, temp: f32) -> Cell {
    let amount = select(1.0, liquidAmount(c), e.kind == KIND_LIQUID);
    if ((e.flags & F_RESIDUE) != 0u && rnd() < amount) {
        // Mud is soaked dirt and a cell of water more (SOIL_FULL).
        var r = makeCell(e.hiTo, temp);
        if (holdsSoilWater(e.hiTo)) {
            r = withSoilWater(r, SOIL_FULL);
        }
        return r;
    }
    let solute = soluteOf(c);
    if (holdsSolute(c.t) && solute != 0u) {
        let s = elems[solute];
        if (s.solResidue != NO_ELEM) {
            var r = makeCell(s.solResidue, temp);
            if (s.solResidue == solute && holdsDissolved(solute)) {
                r = withDissolved(r, u32(clamp(255.0 - f32(strengthOf(c)) * s.solubility * amount + rnd(), 0.0, 255.0)));
            }
            return r;
        }
    }
    var air = makeCell(AIR, temp);
    air.frac = vec2<f32>(0.0);
    return air;
}

// Whether a cell of t's heat is what its gas plane holds (gasHeatCap): a gas,
// and what the air passes through and carries (a fan, air with blades in
// it); not ash, a powder with a little air in its pores.
fn gasHeat(t: u32) -> bool {
    return holdsGas(t) && t != ASH;
}

// The heat capacity, J/K, of a cell of t holding g (its gas plane): what it
// holds, for a gas (none for a vacuum); the element's, for the rest.
fn heatCapOf(t: u32, g: Gas) -> f32 {
    return select(elems[t].heatCap, gasHeatCap(g), gasHeat(t));
}

// The same for the cell c (or v, as stored): a cell of liquid's by how much
// it holds (liquidAmount), a quarter of a cell of water a quarter of a
// cell's; and soil's with the water in its pores (REWRITE_PLAN.md 5h: its
// dry grains' alone, soaked sand held 1.3 J/(cm³·K) where it holds 3.0).
fn cellHeatCap(c: Cell, g: Gas) -> f32 {
    let cap = heatCapOf(c.t, g);
    if (holdsSoilWater(c.t)) {
        return cap + soilWaterCap(soilWaterOf(c));
    }
    if (holdsDissolved(c.t)) {
        return cap * grainShare(dissolvedOf(c));
    }
    return select(cap, cap * liquidAmount(c), elems[c.t].kind == KIND_LIQUID);
}

fn heatCapIn(v: vec4<u32>, g: Gas) -> f32 {
    let t = cellType(v);
    if (holdsSoilWater(t)) {
        return heatCapOf(t, g) + soilWaterCap(wordSoilWater(v.x));
    }
    if (holdsDissolved(t)) {
        return heatCapOf(t, g) * grainShare(wordDissolved(v.x));
    }
    return heatCapOf(t, g) * amountIn(v);
}

// What is left of a grain that has dissolved `dissolved` (255 all of it):
// its heat capacity as much of a cell's (a part of a grain had a whole
// cell's, and merging two ran the one they made far hotter than either).
fn grainShare(dissolved: u32) -> f32 {
    return f32(255u - min(dissolved, 254u)) / 255.0;
}

// The heat capacity of the water a soil holding life (SOIL_FULL soaked)
// holds, J/(cm³·K): SOIL_HOLD of a cell of water, soaked.
fn soilWaterCap(life: u32) -> f32 {
    return elems[WATER].heatCap * SOIL_HOLD * f32(life) / f32(SOIL_FULL);
}

// A soil's thermal conductivity holding life of water (Johansen's model:
// elements.js soilKSat): dry, its table's; soaked, soilKSat; between, by its
// Kersten number. Anything else, its table's.
fn thermalOf(t: u32, word: u32) -> f32 {
    let e = elems[t];
    if (!holdsSoilWater(t) || e.soilKSat <= 0.0) {
        return e.thermal;
    }
    let sr = f32(wordSoilWater(word)) / f32(SOIL_FULL);
    let ke = select(0.0, max(1.0 + e.soilKe * log2(sr) * 0.30103, 0.0), sr > 0.0);
    return e.thermal + (e.soilKSat - e.thermal) * ke;
}

fn isStructural(e: Elem) -> bool {
    return e.kind == KIND_SOLID && e.breakTo != NO_ELEM;
}

// Where a structural solid softens (°C): half its melting point in kelvin,
// past which it creeps (glass's strain point, stone's and steel's creep),
// and stress from heating it further eases as fast as it builds (Ashby,
// Materials Selection in Mechanical Design: creep from about 0.3-0.5 of
// the melting point); never for one that doesn't melt (wood chars first).
// Ice, at 0 °C, is past it at any temperature it holds: it creeps (as
// glaciers do), so takes no thermal stress.
fn softeningT(e: Elem) -> f32 {
    return select(NEVER, 0.5 * (e.hiT + KELVIN) - KELVIN, e.hiT < NEVER);
}

// The share of its strength a structural solid e keeps at temp (°C)
// (REWRITE_PLAN.md 8-7): all of it up to hotStart, half at hotHalf, and
// 1 / (1 + x^HOT_SHAPE) x of the way from one to the other: HOT_SHAPE fit to
// structural steel's (EN 1993-1-2, Table 3.1: 0.78 at 500 °C, 0.47 at 600,
// 0.23 at 700, 0.11 at 800 against 0.85, 0.47, 0.23, 0.12 so).
const HOT_SHAPE: f32 = 2.6;
fn hotShare(e: Elem, temp: f32) -> f32 {
    let x = max(temp - e.hotStart, 0.0) / max(e.hotHalf - e.hotStart, 1.0);
    return select(1.0 / (1.0 + pow(x, HOT_SHAPE)), 1.0, x <= 0.0);
}

fn makeCell(t: u32, temp: f32) -> Cell {
    var c: Cell;
    c.t = t;
    c.aux = 0u;
    c.life = initLife(t);
    c.shade = rndU() & (SHADES - 1u);
    c.flags = 0u;
    c.temp = temp;
    let e = elems[t];
    // New solids start fully supported and settle from there.
    if (isStructural(e)) {
        c.frac.x = SUPPORT_NEW;
    }
    return c;
}

// An explosive c, of e, going off (REWRITE_PLAN.md 5b): its cell becomes the
// gas it makes (BLAST), blastGas cells of it and blastVapour units of water
// as vapour, none of it oxygen, holding all its heat: its heat of explosion
// (blast) and what heat it held (from AMBIENT_T, at its stored temperature,
// which counts any latent heat). So little gas holding so much is thousands
// of degrees hot, and at tens of atmospheres in its cell, as real energies
// make it with the world's gas compressed (decision 5). The air's flow
// spreads it at once (air.wgsl's expansion), and as it expands it sends a
// blast wave off through the air (air.wgsl's blast layer), so a charge shut
// in where its gas can't spread hits harder than one in the open; it mixes
// and cools as it goes.
struct Detonation {
    c: Cell,
    g: Gas,
}

fn detonation(c0: Cell, e: Elem) -> Detonation {
    // (A cell of a liquid explosive, of as much as it holds: liquidAmount.)
    let amount = select(1.0, liquidAmount(c0), e.kind == KIND_LIQUID);
    let g = Gas(e.blastVapour * amount, 0.0, 1.0, e.blastGas * amount);
    var c = makeCell(BLAST, AMBIENT_T + (e.blast + e.heatCap * (c0.temp - AMBIENT_T)) * amount / gasHeatCap(g));
    c.vel = c0.vel;
    return Detonation(c, g);
}

fn isWater(t: u32) -> bool {
    return t == WATER || t == SALTWATER;
}

// The forms of a substance hold different masses a cell (Elem.fill, from
// js/elements.js SUBSTANCES), so they take up different room: a cell of
// water freezes into 1.11 cells of ice (ice is 9 % lighter, as real ice
// is) or 2 of snow, a cell of stone melts into 1.2 of lava, a cell of
// lava setting while it flies makes 1.25 of gravel. n has just turned from
// `old` into n.t, by a phase change or breaking up. Turning into a lighter
// form, it owes the cells that makes beyond itself (owedOf), to be grown
// into the room beside it (block.wgsl); a fractional cell is owed at
// random, in the right share on average (owedOf, the record's: ice's and
// snow's in aux, the other forms that can owe one, F_OWES, in life).
// Turning into a denser form, it stays at random in the right share and is
// air otherwise; a liquid instead holds its share in particles, more than a
// cell or less (what it owed then in it too). What `old` still owed goes
// with it, and ice and snow can
// hold a little more water than their share (FROZEN_EXTRA). A cell of
// liquid holds as much as its particles (liquidAmount): a quarter of a
// cell of water freezes into a cell of ice one time in 3.6, and is air the
// rest. n's temperature is the same either way: changePhase gives it the
// energy for its grams.
fn formed(n: Cell, old: Cell) -> Cell {
    return formedTimes(n, old, 1.0);
}

// The same for k times old's grams (brine freezing, its salt's share of
// the time a grain instead, as ice the rest with its water made up).
fn formedTimes(n: Cell, old: Cell, k: f32) -> Cell {
    var was = elems[old.t].fill * k;
    let to = elems[n.t].fill;
    if (was <= 0.0 || to <= 0.0 || n.t == old.t) {
        return n;
    }
    if (elems[old.t].kind == KIND_LIQUID) {
        was *= liquidAmount(old);
    }
    var r = was * f32(1u + owedOf(old)) / to;
    if (holdsFrozenExtra(old.t)) {
        r += f32(frozenExtraOf(old)) / FROZEN_EXTRA / WATER_VAPOUR / to;
    }
    var c = n;
    if (elems[n.t].kind == KIND_LIQUID) {
        // A liquid holds what it is in particles, more than a cell or less:
        // drawn from none, it is seeded with as many as it counts (ice
        // owing what it couldn't fit thawed into a cell of water, and lost
        // the rest; less than a cell, it was a whole cell or none, four
        // times as rough).
        let count = u32(floor(r * f32(LIQUID_PER_CELL) + rnd()));
        if (count == 0u) {
            c = makeCell(AIR, n.temp);
            c.frac = vec2<f32>(0.0);
        } else {
            c.flags |= FL_DRAWN;
            c.w = liquidW(LIQUID_NO_HOME, count);
        }
    } else if (r > 1.0 && holdsOwed(n.t)) {
        c = withOwed(c, owedOf(c) + u32(floor(r - 1.0 + rnd())));
    } else if (rnd() >= r) {
        c = makeCell(AIR, n.temp);
        c.frac = vec2<f32>(0.0);
    }
    return c;
}

fn flammable(e: Elem) -> bool {
    return e.igniteT < NEVER;
}

// What a spark lights where a pulse reaches it (block.wgsl): a flammable
// gas or an explosive, whose least ignition energy a spark's millijoules
// pass (hydrogen in air 0.02 mJ; black powder, which a flint's sparks set
// off). Solid and liquid fuels want a flame's heat held on them: a spark on
// a log, or on oil under its flash point, goes out.
fn sparkLights(e: Elem) -> bool {
    return flammable(e) && (e.kind == KIND_GAS || e.blast > 0.0);
}

// Whether a particle of type m can move into a cell holding t, trading places
// with it, in the block pass. dy is the vertical direction of the move (-1 up,
// 0 sideways, 1 down). Liquid isn't moved there at all (fluid.wgsl moves it
// as particles), but it gets out of the way: powder moves through it any way
// it is pushed (by gravity, and by the liquid's pressure, which floats light
// powder up; update.wgsl), and gas bubbles up through it. Powders push
// through gas; gases only trade places with air, or with other gases they
// are lighter/heavier than in the right direction.
fn canEnter(m: u32, t: u32, dy: i32) -> bool {
    if (m == t) {
        return false;
    }
    let me = elems[m];
    let te = elems[t];
    if (me.kind == KIND_SOLID || te.kind == KIND_SOLID || me.kind == KIND_LIQUID) {
        return false;
    }
    if (me.kind == KIND_GAS) {
        if (t == AIR) {
            return true;
        }
        if (te.kind == KIND_LIQUID) {
            return dy < 0;
        }
        if (te.kind != KIND_GAS) {
            return false;
        }
        if (dy < 0) {
            return te.density > me.density;
        }
        if (dy > 0) {
            return te.density < me.density;
        }
        return false;
    }
    return te.kind == KIND_GAS || te.kind == KIND_LIQUID;
}
