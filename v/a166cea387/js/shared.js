// What the host and the shaders have to agree on, defined once: constants,
// enums and buffer layout counts. The host imports them from here; the
// shaders get them from sharedPrelude(), which sim.js puts after the element
// table's prelude (elements.js) in every pipeline that includes common.wgsl.
// A shader may still call one by a shorter local name, as an alias of the
// shared one (fluid.wgsl's BLOCK is FLUID_BLOCK); where a shader works one
// out for itself, a const_assert there checks it against the value here.

import * as shared from "./shared.js";

// ------------------------------------------------------------ time and space

// main.js runs STEPS_PER_SECOND steps a second at 1x. A cell is a cube dx
// metres on a side (the world's scale; Params.dx), and a step dt seconds of
// motion: STEP_SECONDS in a world of CELL_M cells, as the square root of dx
// in any other (stepSeconds). Gravity is fixed in cells per step² (water's
// weight, elements.js), and that makes it g at every scale, so things fall,
// slosh and splash alike in cells and steps whatever the scale, a big world's
// in slow motion (Froude similarity; Heller 2011, "Scale effects in physical
// hydraulic engineering models", J. Hydraulic Research 49(3)). What goes by
// physics rather than by the look of it, heat and current, goes by dt and dx
// (REWRITE_PLAN.md, stage 2, has which processes do). The element table's
// heat capacities and latent heats are a CELL_M cube's: the shaders keep a
// cell's heat per CELL_M cube of it, so a stored temperature means the same
// at any scale.
// The day's clock moves on DAY_STEPS steps a day at its normal speed: four
// minutes, a step 6 seconds of the day (dayStepSeconds; common.wgsl's
// clockStep).
export const STEPS_PER_SECOND = 60;
export const STEP_SECONDS = 1 / STEPS_PER_SECOND;
export const CELL_M = 0.01;
export const DAY_STEPS = 14400;
export const SECONDS_PER_DAY = 86400;
export const DAY_STEP = SECONDS_PER_DAY / DAY_STEPS;

// Seconds of motion a step stands for in a world of cells dx metres across.
export function stepSeconds(dx) {
    return STEP_SECONDS * Math.sqrt(dx / CELL_M);
}

// Seconds a step stands for on the day's clock, the day moving on dayRate of
// itself a step: with the clock stopped, its normal speed's (heat goes on at
// that pace), and never less than a step of motion, dt.
export function dayStepSeconds(dayRate, dt) {
    const day = dayRate > 0 ? dayRate * SECONDS_PER_DAY : DAY_STEP;
    return Math.max(day, dt);
}
export const AMBIENT_T = 20;     // °C
export const KELVIN = 273.15;
export const LAPSE_RATE = 0.0065;  // °C per metre (common.wgsl)
// Conduction across a face passing more than this share of the smaller heat
// capacity of its two cells a step, per degree between them, is stiff: it
// goes in heat.wgsl's stages, which sim.js sets by how stiff it gets
// (common.wgsl, update.wgsl).
export const HEAT_SHARE_MAX = 0.25;
// The air grid's pressure per atmosphere, and the least absolute pressure a
// boiling point is worked out for, in atmospheres (common.wgsl).
export const P_ATM = 12;
export const MIN_ATM = 0.02;

// ------------------------------------------------------------ cells

// A cell is one vec4<u32>, CELL_BYTES. Its x word packs the element and the
// small fields beside it: type, aux and life (8 bits each; what aux and life
// mean depends on the element, see Cell in common.wgsl), shade (colour
// variation, or a plant's dryness) and flags (CELL_FLAGS), each [first bit,
// bits]. y is its stored temperature (f32 bits), z its velocity (2 x f16,
// cells a step), w its frac (2 x f16) or, for a liquid, the index of the
// particle it was drawn from. The shaders read and write it through the
// accessors sharedPrelude() makes from this (cellType, cellTemp, wordAux,
// ...), the host through cellFields(). After all the cells, in the same
// buffer, the gas plane (Gas in common.wgsl): GAS_BYTES a cell, cell i's
// at count * CELL_BYTES + i * GAS_BYTES: vapour and mist (2 x f16), then the
// share of its oxygen used up and its amount of gas (2 x f16; an amount of 1
// is a cell of air at 1 atm and 20 °C, and all of it 0 a vacuum).
export const CELL_BYTES = 16;
export const GAS_BYTES = 8;
export const cellsBufferBytes = (count) => count * CELL_BYTES + Math.ceil(count / 2) * 2 * GAS_BYTES;
export const CELL_FIELDS = { type: [0, 8], aux: [8, 8], life: [16, 8], shade: [24, 4], flags: [28, 4] };
// The flags' bits, each with a meaning per kind of element (common.wgsl has
// what each means): burning; air with dew on it, ice with no room to grow,
// soil a plant has rooted into, water electrolysis is splitting; liquid drawn
// from a particle, powder under liquid, structure that has been past its
// softening point; powder resting on the ground, structure loose in a free
// body.
export const CELL_FLAGS = {
    BURN: 1,
    DEW: 2, STRESSED: 2, ROOTED: 2, SPLIT: 2,
    DRAWN: 4, WET: 4, SOFTENED: 4,
    GROUNDED: 8, LOOSE: 8,
};

// The element of cell i, from the words of a cells buffer (a Uint32Array).
export function cellTypeAt(u, i) {
    const [at, bits] = CELL_FIELDS.type;
    return (u[4 * i] >>> at) & ((1 << bits) - 1);
}

// A cell's fields, from the words of a cells buffer (as a Uint32Array) and
// the same as a Float32Array, at cell i.
export function cellFields(u, f, i) {
    const x = u[4 * i];
    const field = ([at, bits]) => (x >>> at) & ((1 << bits) - 1);
    return {
        type: field(CELL_FIELDS.type), aux: field(CELL_FIELDS.aux), life: field(CELL_FIELDS.life),
        shade: field(CELL_FIELDS.shade), flags: field(CELL_FIELDS.flags), temp: f[4 * i + 1],
        vel: [f16Value(u[4 * i + 2] & 0xffff), f16Value(u[4 * i + 2] >>> 16)],
        frac: [f16Value(u[4 * i + 3] & 0xffff), f16Value(u[4 * i + 3] >>> 16)],
        w: u[4 * i + 3],
    };
}

export const NO_ELEM = 255;
// A brush's temperature under this means the element's own.
export const TEMP_OWN = -1e29;
// Electric charge: 4 and 3 are active (spreading), 2 and 1 are cooldown.
export const CHARGE_ACTIVE = 3;
// Block passes a step (block.wgsl): a particle moves at most one cell a pass.
export const BLOCK_PASSES = 4;

// Water as vapour (common.wgsl), in units of 1 / WATER_VAPOUR of a cell of
// water, one scale for the air's vapour and mist and for what water boils
// into (REWRITE_PLAN.md decision 3; since 4c steam is the air's vapour, not
// a kind of cell). Water expands VAPOUR_EXPANSION times as it boils at 1 atm
// (real water about 1600 times; compressed here as the heat ratios are), so
// a cell of pure vapour at 1 atm and its boiling point, BOIL_K kelvin, holds
// VAPOUR_MAX: the most air at 1 atm holds. Below that, what saturates air
// follows the vapour pressure (saturation): VAPOUR_20 at 20 °C, some 1 / 35
// of VAPOUR_MAX. The outdoor air's humidity is relative to that. Air painted
// at 20 °C holds OUTDOOR_HUMIDITY * VAPOUR_20 of it (OUTDOOR_AIR_FRAC); a
// cleared world's, as much of what saturates it at the outdoor air's
// temperature at its height, and the open air's dry air there
// (outdoorAirGas, below). Vapour's heat capacity, VAPOUR_HEAT_CAP,
// is a cell of water's as vapour (2 J/(g K), a gram a cell), compressed as
// the air's is (GAS_HEAT_CAP, below, the same for a cell of air at 1 atm
// and 20 °C and for VAPOUR_MAX units).
export const WATER_VAPOUR = 64;
export const VAPOUR_EXPANSION = 8;
export const VAPOUR_MAX = WATER_VAPOUR / VAPOUR_EXPANSION;
export const BOIL_K = 100 + KELVIN;
export const VAPOUR_HEAT_CAP = 2.0;
export const OUTDOOR_HUMIDITY = 0.5;
// Ice and snow keep the moisture of air they take in in their life, in
// 1 / FROZEN_EXTRA units (common.wgsl frozenExtra).
export const FROZEN_EXTRA = 64;

// Vapour saturating air at temp (°C) under atm atmospheres, in units a cell:
// the Magnus formula for the vapour pressure over water (Alduchov & Eskridge
// 1996, "Improved Magnus form approximation of saturation vapor pressure"),
// at most atm (past its boiling point there water's vapour is all the gas),
// made a density by the ideal gas law: VAPOUR_MAX at 100 °C and 1 atm.
// common.wgsl's saturationAt is the same.
export function saturation(temp, atm = 1) {
    const t = Math.max(temp, -60);
    const tm = Math.min(t, 200);
    const es = Math.exp(17.625 * tm / (243.04 + tm) - 17.625 * 100 / 343.04);
    return VAPOUR_MAX * Math.min(es, atm) * BOIL_K / (t + KELVIN);
}

export const VAPOUR_20 = saturation(20);
export const OUTDOOR_AIR_FRAC = f16Bits(OUTDOOR_HUMIDITY * VAPOUR_20);

// The world's gas is GAS_COMPRESSION times as dense as real gas: water's
// vapour is (it expands VAPOUR_EXPANSION times on boiling, not 1600: 1000 /
// VAPOUR_EXPANSION kg/m³ at 100 °C against real steam's 0.598), and every
// gas with it. So is a gas's heat capacity, GAS_HEAT_CAP a cell at 1 atm
// (real air's, steam's and the like's some 1.2 kJ/(m³·K)), as a steam
// unit's is, and the oxygen it holds: a cell of fresh air holds AIR_OXYGEN
// grams, real air's 0.279 mg a cubic centimetre (1.204 kg/m³ at 20 °C, 23.1 %
// of it oxygen by mass) times as much. Compressed alike, the latent heat the
// air's moisture gives off condensing heats it as it would real air, and
// what burns in it heats it as much (REWRITE_PLAN.md decision 3, stage 5).
export const GAS_COMPRESSION = 1000 / VAPOUR_EXPANSION / 0.598;
export const GAS_HEAT_CAP = 1.2e-3 * GAS_COMPRESSION;
export const AIR_GRAMS = 1.204e-3 * GAS_COMPRESSION;
export const AIR_OXYGEN = AIR_GRAMS * 0.2314;
// The moles of gas in a cell at 1 atm and 20 °C (the ideal gas law,
// compressed), and the cells of gas half a mole makes: what an equivalent of
// acid gives off eating a metal (hydrogen) or a carbonate (carbon dioxide).
export const AIR_MOLES = 101325e-6 / (8.314 * (AMBIENT_T + KELVIN)) * GAS_COMPRESSION;
export const GAS_PER_EQUIVALENT = 0.5 / AIR_MOLES;
// The cells of hydrogen a cell of water split by electrolysis makes: a gram
// of it, 55.5 mmol, a mole of hydrogen to a mole of water (Faraday).
export const WATER_HYDROGEN = 1 / 18.015 / AIR_MOLES;

// Fire (REWRITE_PLAN.md stage 5). What burns gives OXYGEN_HEAT joules for
// each cell of fresh air's oxygen it uses up, whatever it is: 13.1 kJ a gram
// of oxygen (Huggett 1980, "Estimation of rate of heat release by means of
// oxygen consumption measurements", Fire and Materials 4(2)), which heats
// that air by some 3000 °C. Flaming goes out where the gas beside it
// couldn't be heated to CRITICAL_FLAME_T by burning the oxygen it holds:
// at 20 °C, where it holds less than SMOTHER of fresh air's, about 15 %
// oxygen against 21 % (Babrauskas, Ignition Handbook, 2003), and hotter or
// moister gas at more or less (the critical flame temperature of FDS:
// McGrattan et al., Fire Dynamics Simulator Technical Reference Guide, NIST
// SP 1018). Fire goes FIRE_PACE times faster than real (decided 2026-10-24,
// to keep fires to their pace before stage 5): what burns burns, and is
// lit by a flame, in a FIRE_PACE-th of the real time the element table
// gives (burnTime), on heat's clock; its heat and oxygen are as real.
export const OXYGEN_HEAT = 13.1e3 * AIR_OXYGEN;
export const SMOTHER = 0.71;
export const CRITICAL_FLAME_T = AMBIENT_T + SMOTHER * OXYGEN_HEAT / GAS_HEAT_CAP;
export const FIRE_PACE = 60;

// The open air's pressure at an altitude (m), in atm: 1 at the bottom of the
// world, less with height by a scale height of R T / (M g) at 20 °C.
// common.wgsl's openAirAtm is the same.
export const SCALE_HEIGHT = 8400;

export function openAirAtm(altitude) {
    return Math.exp(-altitude / SCALE_HEIGHT);
}

// The outdoor air's temperature at an altitude (m): 20 °C at the bottom of
// the world, lapsing LAPSE_RATE a metre (common.wgsl's outdoorT).
export function outdoorT(altitude) {
    return AMBIENT_T - LAPSE_RATE * altitude;
}

// The gas plane's two words for the open air at an altitude (m), at temp
// (the outdoor air's there, if not given): as humid as the outdoor air is
// at that temperature, its vapour and no mist; none of its oxygen used, and
// as much dry air as its pressure leaves room for beside its vapour, never
// less than none (common.wgsl's outdoorGas is the same; brush.wgsl's
// paintedGas likewise takes the vapour's room out of the dry air's).
export function outdoorAirGas(altitude, temp = outdoorT(altitude)) {
    const vapour = OUTDOOR_HUMIDITY * saturation(temp);
    const all = openAirAtm(altitude) * (AMBIENT_T + KELVIN) / Math.max(temp + KELVIN, 1);
    const dry = Math.max(all - vapour * (AMBIENT_T + KELVIN) / (BOIL_K * VAPOUR_MAX), 0);
    return [f16Bits(vapour), (f16Bits(dry) << 16) >>> 0];
}

// ------------------------------------------------------------ air, rooms

// Particle cells per air cell side (air.wgsl: one, each face of the flow open
// or shut as the cells across it are), and the most cells the air's
// projection's coarsest level has, which one workgroup solves.
export const AIR_CELL = 1;
export const AIR_COARSEST = 256;
// The most substeps of the wind's carry a step (carry.wgsl), even: half a
// cell each for the fastest wind (air.wgsl's MAX_V, 12 cells a step, a little
// more after the projection).
export const CARRY_STEPS = 32;
// The word of the air's status (air.wgsl) that holds the most iterations its
// projection took in a step since the host last read it (sim.js airCycles).
export const AIR_MOST_CYCLES = 5;
// The two words of the air's status (air.wgsl), by the step's parity, that
// say whether its explosions' layer holds anything: the next step's passes
// of it run only if so (blastQuiet).
export const AIR_BLAST_LIVE = 6;
// The rooms (sky.wgsl), the air's pressure zones (air.wgsl): found per
// ROOM_TILE square, up to ROOM_PIECES pieces a tile (a pocket past that has
// no zone: grains of a sand cloud hold many); a zone's sums ZONE_WORDS words
// (common.wgsl zoneSlotAt), and a wet room's (the rooms joined through
// liquid) WET_WORDS (wetSlotAt).
export const ROOM_TILE = 32;
export const ROOM_PIECES = 64;
export const ZONE_WORDS = 24;
export const WET_WORDS = 16;

// ------------------------------------------------------------ liquids

// Liquids (fluid.wgsl) are worked on in blocks of FLUID_BLOCK cells a side.
// After the blocks' lists, these words of the blocks buffer, FLUID_STATS of
// them, read back each frame: the blocks worked on and dropping out,
// whether any holds viscous liquid, the particles as last sorted; the most
// divergence the pressure's iterations left, a slot each for three in turn
// (fluid.wgsl's converged), the iterations this step took, and the most any
// step took since the last report (sim.js's fluidCycles).
export const FLUID_BLOCK = 8;
export const FLUID_COUNT = { ACTIVE: 0, RETIRED: 1, VISCOUS: 2, PARTICLES: 3, RESIDUAL: 4, CYCLES: 7, MOST_CYCLES: 8 };
export const FLUID_STATS = 9;
// A cell of new liquid is seeded with LIQUID_PER_CELL particles (fluid.wgsl's
// syncSeed), and a cell's worth of liquid is that many. The particle buffers
// hold LIQUID_ROOM cells' worth of them a cell of the world.
export const LIQUID_PER_CELL = 4;
export const LIQUID_ROOM = 1.25;
// The liquid's words per cell in its counters buffer (fluid.wgsl's G_*); the
// bodies set where a cell's particles go (G_DEST) as they push it aside.
export const LIQUID_CELL_WORDS = 11;
// The multigrid's coarsest level, solved by one workgroup: at most this many cells.
export const MG_COARSEST = 512;
// Water's capillary length (m), and the capillary lengths in cells under
// which surface tension is left out.
export const CAPILLARY_LENGTH = 0.0027;
export const SURFACE_MIN_LENGTH = 0.3;
// Von Kármán's constant, of the log law over a surface (update.wgsl's
// Shields speed, fluid.wgsl's surfaceDragCoefficient).
export const KARMAN = 0.41;

// ------------------------------------------------------------ bodies

// Free bodies (bodies.wgsl): tiles of BODY_TILE cells a side to look for new
// ones in; a table of MAX_BODIES slots of BODY_TABLE_WORDS; BODY_SUM_WORDS of
// sums per cell (as a piece's root). In the header, after a mark and a state
// per tile: the stats read back each frame (BODY_STAT), the rest of
// BODY_STAT_WORDS, the counters (BODY_COUNTER).
export const BODY_TILE = 8;
export const MAX_BODIES = 1024;
export const BODY_TABLE_WORDS = 116;
export const BODY_SUM_WORDS = 8;
export const BODY_STAT = { MARKED: 0, ACTIVE: 1, MARKED_RUN: 2, ALIVE: 3, COMPACT: 4 };
export const BODY_STATS = 5;
export const BODY_STAT_WORDS = 8;
export const BODY_COUNTER = { LISTED: 0, FOUNDING: 1, POOL: 2, PIECE: 3, HALF: 4, KEPT: 5, LOST_POWDER: 6, LOST_GAS: 7, CONTACTS: 8 };
export const BODY_COUNTERS = 9;
// The contacts all bodies found this step, solved together (bodies.wgsl's
// jointSolve): at most BODY_CONTACTS of BODY_CONTACT_WORDS each.
export const BODY_CONTACTS = 16384;
export const BODY_CONTACT_WORDS = 12;
// What the liquid's pass (fluid.wgsl) reads of a body's slot in the table:
// the words of its pose and motion (bodies.wgsl's T_*), its cells, and its
// flags (B_*); and after the contacts, BODY_LIQUID_WORDS a slot for what
// the liquid does to it in a step (REWRITE_PLAN.md P3-3, P3-22), and
// whether a cell of it was in a block by the liquid in its last solve, held
// (WET: the next takes all its cells in, P3-11), and whether it rests on what
// stands still, or on a body that does (HELD: a wall to the liquid's
// solve, P3-19); and how it gives way to a push as its contacts let it, six
// words of a symmetric 3 x 3 over velocity x, y and spin, and the step that
// was worked out (MOBILITY, MOBILITY_AT: P3-23). A snapshot as it loads (snapshot.js's
// migrateBodies) reads the flags and the step a body was last stamped
// (FRAME), and asks the parts pass to look at every body (PARTS).
export const BODY_WORD = {
    POS: 0, VEL: 3, SPIN: 5, MASS: 6, INERTIA: 7, CELLS: 13, FLAGS: 18, FRAME: 19, SUBCELL: 25, CARGO: 105, CARGO_INERTIA: 106, MOBILITY: 108,
    MOBILITY_AT: 114,
};
export const BODY_FLAG = { ALIVE: 1, NEW: 2, PARTS: 8, HELD: 32, WET: 64 };
export const BODY_LIQUID_WORDS = 4;
// The liquid's solve's sums of what it does to each body, a step's scratch
// (fluid.wgsl's pushes): BODY_PUSH_SETS sets of a force x, y and a torque,
// each an exact sum of two words.
export const BODY_PUSH_SETS = 6;
export const BODY_PUSH_WORDS = 6 * BODY_PUSH_SETS;

// ------------------------------------------------------------ structure

// Piles (update.wgsl): a resting powder cell bears its own weight, density /
// 100, and the resting powder's on it, counted as if the world were
// LOAD_ROWS tall; one beside a wall presses on it with LATERAL of that.
// What it bears presses on the structure under or beside it (stress.wgsl).
export const LOAD_ROWS = 200;
export const LATERAL = 0.4;
// The stress solve (stress.wgsl, REWRITE_PLAN.md 8-2): STRESS_ITERATIONS CG
// iterations encoded a step, which return at once once it has converged (its
// residual under STRESS_TOL of the loads, preconditioned norms), and
// STRESS_ITERATIONS_FAR while the last report said it was far off (its
// residual over STRESS_FAR of the loads; 8-6): a load rising fast (a
// boiler's steam under its lid) left the residual at half the loads step
// after step, and the lid gave anywhere from its strength to three times it.
// (By the residual over STRESS_TOL alone, structure warming or cooling,
// whose thermal strain drifts on, was never quite converged, and took the
// 128 all the time, 10-14 % of a step.) (Fewer once a report said it had converged were too late
// for a blast: the reports come a frame or more behind, and a check
// stepping 300 steps in long frames met it with 4 a step, which never
// caught up; nor, before 6-5, did the harness's reports come back at all.)
// Far off too while its worst tile's residual is over STRESS_FAR_TILE of
// that tile's own loads (8-9): beside a steel pot, its own weight and its
// steam's push converged, the world's residual stayed under STRESS_FAR
// while the pot's glass lid, its load rising each step, stayed at half its
// own, and gave at up to three times the pressure that breaks it. (At
// STRESS_FAR, by the worst tile, structure warming or cooling took the 128
// most steps: 8-8.)
// STRESS_STATS words of stats read back, then STRESS_SCALARS of the CG's
// state, then six words a tile of 8 x 8 nodes (its partial sums), a word a
// tile (whether it has any structure), a place on the list of those that
// do, and two words a cell (its failure index, its streak);
// STRESS_NODE_WORDS vec4s a node (and its displacement apart, in fixed
// point).
export const STRESS_ITERATIONS = 32;
export const STRESS_ITERATIONS_FAR = 128;
export const STRESS_FAR = 0.1;
export const STRESS_FAR_TILE = 0.3;
export const STRESS_TOL = 0.02;
export const STRESS_STATS = 5;
export const STRESS_STAT = { CONVERGED: 0, RESIDUAL: 1, ITERATIONS: 2, NODES: 3, WORST: 4 };
export const STRESS_SCALARS = 16;
export const STRESS_NODE_WORDS = 4;

// ------------------------------------------------------------ electric current

// Electric current (electric.wgsl; REWRITE_PLAN.md P1): blocks of
// ELECTRIC_BLOCK cells a side, each with up to ELECTRIC_SLOTS aggregates
// (coarse nodes, ELECTRIC_COARSE_WORDS words each), ELECTRIC_ITERATIONS CG
// iterations a step at most; ELECTRIC_STATS words of stats, then
// ELECTRIC_SCALARS of the CG's state, ELECTRIC_TILE_WORDS a block and a word
// a coarse node (the live ones' list); ELECTRIC_WORK working words a cell.
export const ELECTRIC_BLOCK = 8;
export const ELECTRIC_SLOTS = 4;
export const ELECTRIC_ITERATIONS = 3;
// ... and ELECTRIC_ITERATIONS_FAR for ELECTRIC_FAR_STEPS steps after anything
// is painted (a circuit just drawn), or while the last report had the
// residual over ELECTRIC_FAR of the currents (one broken or remade). (Not
// for any residual over a mark: in brine splitting at an electrode the
// circuit changes every step, more iterations hardly settle it more, and
// the step cost several times as much.)
export const ELECTRIC_ITERATIONS_FAR = 8;
export const ELECTRIC_FAR_STEPS = 10;
export const ELECTRIC_FAR = 0.1;
export const ELECTRIC_COARSE_WORDS = 48;
export const ELECTRIC_STATS = 5;
export const ELECTRIC_SCALARS = 16;
export const ELECTRIC_TILE_WORDS = 6;
export const ELECTRIC_WORK = 10;

// ------------------------------------------------------------ gravity

// Gravity (gravity.wgsl): blocks of level 0 are light cells, GRAVITY_BLOCK
// cells a side; the pyramid tops out at GRAVITY_TOP_BLOCKS blocks.
export const GRAVITY_BLOCK = 2;
export const GRAVITY_TOP_BLOCKS = 16;

// ------------------------------------------------------------ people

// People (person.wgsl, people.wgsl): MAX_PEOPLE slots; requests to place and
// erase them, MAX_PERSON_REQUESTS at most a frame. What a person is doing: a
// free slot; standing about; strolling, never far from home; running from
// danger, or on fire; talking to someone beside it; asleep at home, at night;
// lying where it fell, until it fades. Its flags: facing left; burning;
// standing on something; in a liquid past its waist; a current through it
// this step; hurt this step; no air to breathe (under a liquid, buried, in
// smoke or a stale room).
export const MAX_PEOPLE = 64;
export const MAX_PERSON_REQUESTS = 32;
export const REQUEST = { PLACE: 1, ERASE: 2, ERASE_BOX: 3 };
export const PERSON_STATE = { NONE: 0, IDLE: 1, WALK: 2, FLEE: 3, CHAT: 4, SLEEP: 5, DEAD: 6 };
export const PERSON_FLAG = { LEFT: 1, BURNING: 2, GROUND: 4, SWIM: 8, SHOCK: 16, HURT: 32, UNDER: 64 };

// ------------------------------------------------------------ brush and views

// The brush (brush.wgsl): what it does (SOAK: soil under water soaked
// through, once a scene is built; VAPOUR: the air's vapour set to strength
// times what saturates it at its temperature, for scenes); its shape:
// within its radius of the
// segment a-b (CAPSULE), cell centres inside the box a-b (RECT), within its
// radius measured square, the larger of x and y (SQUARE), as the capsule
// thinning out to its edge (SPRAY), the cells set in its mask, a flood fill,
// within the box a-b (MASK); and what it paints over: only empty space (air),
// anything, or only the one element Brush.only (and erasing, only that).
export const BRUSH = { PAINT: 0, HEAT: 1, COOL: 2, SPARK: 3, WIND: 4, SOAK: 5, VAPOUR: 6 };
export const SHAPE = { CAPSULE: 0, RECT: 1, SQUARE: 2, SPRAY: 3, MASK: 4 };
export const OVER = { EMPTY: 0, ALL: 1, ONLY: 2 };
export const VIEW = { NORMAL: 0, HEAT: 1, PRESSURE: 2, OXYGEN: 3, HUMIDITY: 4, LIT: 5, GRAVITY: 6 };

// ------------------------------------------------------------ for the shaders

const U = "u32";
const I = "i32";
const F = "f32";

// The constants above the shaders get, with their WGSL type.
const CONSTANTS = {
    CELL_M: F, DAY_STEP: F, AMBIENT_T: F, KELVIN: F, LAPSE_RATE: F, HEAT_SHARE_MAX: F, P_ATM: F, MIN_ATM: F,
    NO_ELEM: U, TEMP_OWN: F, CHARGE_ACTIVE: U, BLOCK_PASSES: F,
    WATER_VAPOUR: F, VAPOUR_20: F, VAPOUR_MAX: F, BOIL_K: F, VAPOUR_HEAT_CAP: F, OUTDOOR_HUMIDITY: F, FROZEN_EXTRA: F,
    SCALE_HEIGHT: F, OXYGEN_HEAT: F, SMOTHER: F, CRITICAL_FLAME_T: F, FIRE_PACE: F, GAS_PER_EQUIVALENT: F, GAS_COMPRESSION: F,
    WATER_HYDROGEN: F,
    AIR_CELL: U, AIR_COARSEST: U, CARRY_STEPS: U, AIR_MOST_CYCLES: U, AIR_BLAST_LIVE: U, ROOM_TILE: U, ROOM_PIECES: U, ZONE_WORDS: U, WET_WORDS: U,
    FLUID_BLOCK: I, FLUID_STATS: U, LIQUID_PER_CELL: U, LIQUID_CELL_WORDS: U, MG_COARSEST: U, CAPILLARY_LENGTH: F, KARMAN: F, SURFACE_MIN_LENGTH: F,
    BODY_TILE: U, MAX_BODIES: U, BODY_CONTACTS: U, BODY_CONTACT_WORDS: U, BODY_LIQUID_WORDS: U, BODY_PUSH_SETS: U, BODY_PUSH_WORDS: U,
    LOAD_ROWS: F, LATERAL: F, STRESS_TOL: F, STRESS_STATS: U, STRESS_SCALARS: U, STRESS_NODE_WORDS: U, BODY_TABLE_WORDS: U, BODY_SUM_WORDS: U, BODY_STATS: U, BODY_STAT_WORDS: U, BODY_COUNTERS: U,
    ELECTRIC_BLOCK: I, ELECTRIC_SLOTS: U, ELECTRIC_ITERATIONS: U, ELECTRIC_COARSE_WORDS: U, ELECTRIC_STATS: U, ELECTRIC_SCALARS: U, ELECTRIC_TILE_WORDS: U, ELECTRIC_WORK: U,
    GRAVITY_BLOCK: U, GRAVITY_TOP_BLOCKS: U,
    MAX_PEOPLE: U, MAX_PERSON_REQUESTS: U,
};

// The enums above the shaders get, each value as a u32 named with its prefix.
const ENUMS = {
    FLUID_COUNT: "COUNT_", BODY_STAT: "BODY_STAT_", BODY_COUNTER: "C_", BODY_WORD: "BW_", BODY_FLAG: "BF_", STRESS_STAT: "SS_",
    REQUEST: "REQUEST_", PERSON_STATE: "P_", PERSON_FLAG: "PF_",
    BRUSH: "MODE_", SHAPE: "SHAPE_", OVER: "OVER_", VIEW: "VIEW_",
};

function literal(value, type) {
    if (type === U) {
        return `${value}u`;
    }
    if (type === I) {
        return String(value);
    }
    const s = String(value);
    return /[.e]/.test(s) ? s : `${s}.0`;
}

export function sharedPrelude() {
    const lines = ["// Generated from js/shared.js"];
    for (const [name, type] of Object.entries(CONSTANTS)) {
        lines.push(`const ${name}: ${type} = ${literal(shared[name], type)};`);
    }
    for (const [name, prefix] of Object.entries(ENUMS)) {
        for (const [key, value] of Object.entries(shared[name])) {
            lines.push(`const ${prefix}${key}: u32 = ${value}u;`);
        }
    }
    return lines.join("\n") + "\n" + cellPrelude();
}

// The cell's accessors: wordType(x) and the like for the x word, cellType(v)
// and the like for the whole cell; packWord for packCell (common.wgsl).
function cellPrelude() {
    const name = (field) => field[0].toUpperCase() + field.slice(1);
    const arg = (field) => (field === "type" ? "t" : field);  // `type` is reserved in WGSL
    const lines = [
        `const CELL_BYTES: u32 = ${CELL_BYTES}u;`,
        `const FLAG_SHIFT: u32 = ${CELL_FIELDS.flags[0]}u;`,
        `const SHADES: u32 = ${1 << CELL_FIELDS.shade[1]}u;`,
    ];
    for (const [flag, bit] of Object.entries(CELL_FLAGS)) {
        lines.push(`const FL_${flag}: u32 = ${bit}u;`);
    }
    const packed = [];
    for (const [field, [at, bits]] of Object.entries(CELL_FIELDS)) {
        const mask = `0x${((1 << bits) - 1).toString(16)}u`;
        const shifted = at === 0 ? "x" : `(x >> ${at}u)`;
        const value = at + bits === 32 ? `x >> ${at}u` : `${shifted} & ${mask}`;
        lines.push(`fn word${name(field)}(x: u32) -> u32 { return ${value}; }`);
        lines.push(`fn cell${name(field)}(v: vec4<u32>) -> u32 { return word${name(field)}(v.x); }`);
        packed.push(at === 0 ? `(${arg(field)} & ${mask})` : `((${arg(field)} & ${mask}) << ${at}u)`);
    }
    const args = Object.keys(CELL_FIELDS).map((field) => `${arg(field)}: u32`).join(", ");
    lines.push(
        `fn packWord(${args}) -> u32 { return ${packed.join(" | ")}; }`,
        "fn wordHas(x: u32, flag: u32) -> bool { return (wordFlags(x) & flag) != 0u; }",
        "fn cellHas(v: vec4<u32>, flag: u32) -> bool { return wordHas(v.x, flag); }",
        "fn wordWith(x: u32, flag: u32) -> u32 { return x | (flag << FLAG_SHIFT); }",
        "fn wordWithout(x: u32, flag: u32) -> u32 { return x & ~(flag << FLAG_SHIFT); }",
        "fn cellTemp(v: vec4<u32>) -> f32 { return bitcast<f32>(v.y); }",
        "fn cellVel(v: vec4<u32>) -> vec2<f32> { return unpack2x16float(v.z); }",
        "fn cellFrac(v: vec4<u32>) -> vec2<f32> { return unpack2x16float(v.w); }",
    );
    return lines.join("\n") + "\n";
}

// The f16 with bits h.
export function f16Value(h) {
    const sign = h >> 15 ? -1 : 1;
    const exp = (h >> 10) & 31;
    const mant = h & 1023;
    if (exp === 31) {
        return mant ? NaN : sign * Infinity;
    }
    return exp === 0 ? sign * mant * 2 ** -24 : sign * (1 + mant / 1024) * 2 ** (exp - 15);
}

// The bits of x as an f16, rounded to the nearest: subnormal below 2^-14,
// infinite past the largest.
export function f16Bits(x) {
    const sign = x < 0 || Object.is(x, -0) ? 0x8000 : 0;
    const a = Math.abs(x);
    if (a === 0) {
        return sign;
    }
    if (a >= 65520) {
        return sign | 0x7c00;
    }
    if (a < 2 ** -14) {
        return sign | Math.round(a / 2 ** -24);
    }
    let exp = Math.floor(Math.log2(a));
    if (a / 2 ** exp < 1) {
        exp--;
    }
    let mant = Math.round((a / 2 ** exp - 1) * 1024);
    if (mant === 1024) {
        mant = 0;
        exp++;
    }
    return sign | ((exp + 15) << 10) | mant;
}
