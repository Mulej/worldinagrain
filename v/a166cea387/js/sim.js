// GPU resources and per-frame command encoding for the simulation.

import { recordPrelude } from "./cellrecord.js";
import { ELEMENTS, ID, KIND, packElements, wgslPrelude } from "./elements.js";
import {
    AIR_BLAST_LIVE, AIR_CELL, AIR_COARSEST, AIR_MOST_CYCLES, AMBIENT_T, CARRY_STEPS, BLOCK_PASSES, BODY_COUNTERS, BODY_STAT, BODY_STAT_WORDS,
    BODY_CONTACT_WORDS, BODY_CONTACTS, BODY_LIQUID_WORDS, BODY_PUSH_WORDS, BODY_STATS, BODY_SUM_WORDS, BODY_TABLE_WORDS, BODY_TILE, BRUSH, CAPILLARY_LENGTH, CELL_BYTES, CELL_FLAGS, CELL_M,
    cellFields, cellsBufferBytes, cellTypeAt, CHARGE_ACTIVE, DAY_STEPS, dayStepSeconds, ELECTRIC_BLOCK, ELECTRIC_COARSE_WORDS,
    ELECTRIC_FAR, ELECTRIC_FAR_STEPS, ELECTRIC_ITERATIONS, ELECTRIC_ITERATIONS_FAR, ELECTRIC_SCALARS, ELECTRIC_SLOTS, ELECTRIC_STATS, ELECTRIC_TILE_WORDS, ELECTRIC_WORK, f16Value, FLUID_BLOCK, FLUID_COUNT, FLUID_STATS, GAS_BYTES,
    GRAVITY_BLOCK, GRAVITY_TOP_BLOCKS, HEAT_SHARE_MAX, KELVIN, LAPSE_RATE, LIQUID_CELL_WORDS, LIQUID_PER_CELL, LIQUID_ROOM, MAX_BODIES,
    MAX_PEOPLE, MAX_PERSON_REQUESTS, MG_COARSEST, outdoorAirGas, outdoorT, OVER, PERSON_FLAG, PERSON_STATE,
    REQUEST, ROOM_PIECES, ROOM_TILE, SHAPE, sharedPrelude, stepSeconds, STRESS_FAR, STRESS_FAR_TILE, STRESS_ITERATIONS, STRESS_ITERATIONS_FAR,
    STRESS_NODE_WORDS, STRESS_SCALARS, STRESS_STAT, STRESS_STATS, SURFACE_MIN_LENGTH, TEMP_OWN, VIEW, WET_WORDS,
    ZONE_WORDS,
} from "./shared.js";
import { persistentResources } from "./passes.js";
import { GpuTimer } from "./timer.js";

export { BRUSH, CELL_FLAGS, CHARGE_ACTIVE, LAPSE_RATE, PERSON_FLAG, PERSON_STATE, SHAPE, VIEW };

const SLOT_BYTES = 256;  // minUniformBufferOffsetAlignment
const MAX_SLOTS = 64;
const PARAMS_BYTES = 80;
const BRUSH_BYTES = 80;
const AIR_BYTES = 16;
// The air's projection (air.wgsl): levels at most (the finest included, so
// the levels' uniform slots, MG_LEVELS + 1, hold them).
const AIR_LEVELS = 9;
// The tiles its V-cycle's passes take a level in (air.wgsl's TILE).
const AIR_TILE = 16;
// Iterations of the projection's conjugate gradients a step at most: as many
// as it takes to bring the divergence left under air.wgsl's AIR_TOLERANCE
// (the gas's expansion is met only as far as the flow is solved). With every
// gas's expansion in it (4c) a sealed room's flow takes 10 to 40; at 16, a
// warm pool's never got there.
const AIR_CYCLES = 64;
// ... of which a step encodes twice as many as the last report's steps took
// at most, and AIR_CYCLES_SPARE more (airCycles; F-1): the iterations after
// the solve is done return at once, but each of their 30-50 passes still
// costs its dispatch (about 1.7 us here), and 64 of them were most of a
// step. The scenes take 6 to 40, 14 on the mean; a step takes 12-15 more
// now and then, where the CG stalls near the tolerance. All AIR_CYCLES after
// anything is painted, and once a report says a step used all it was given.
const AIR_CYCLES_SPARE = 8;
// Explosions' pressure waves: half steps a step (air.wgsl's blast layer, a
// cell each: 2 cells a step).
const BLAST_SUBSTEPS = 4;
// Liquids (fluid.wgsl).
const PARTICLE_BYTES = 48;   // a particle's motion, state and affine velocity in Particles
const PARTICLES_HEADER = 16;
const PARTICLE_GROUP = 256;  // threads a workgroup in the passes over the particles (fluid.wgsl)
const FIELD_CELL_BYTES = 32; // two vec4s per cell
const FIELD_FACE_BYTES = 16; // a vec4 per face, then a quarter as much again, twice
const MG_LEVELS = 8;         // coarse levels at most
// The pressure solve per step: relaxation dispatches before and after each
// multigrid V-cycle (each several sweeps over each block), and iterations of
// the conjugate gradients the V-cycle preconditions at most, as many as it
// takes to bring the divergence left under fluid.wgsl's LIQUID_TOLERANCE
// (the scenes with bodies 3 to 5 on the mean, 9 at most).
// Then red-black sweep pairs of viscosity.
const PRE_RELAX = 1;
const POST_RELAX = 1;
const FLUID_CYCLES = 16;
// ... of which a step encodes as many as the last report's steps took at
// most, and FLUID_CYCLES_SPARE more (fluidCycles): those after the solve is
// done return at once, but each of their 35-50 passes still costs its
// dispatch. (Twice as many, as the air's, AIR_CYCLES_SPARE, cost Dam and
// Deep hull 0.3-0.5 ms a step more, the most a frame's steps took already
// over their mean by half.)
const FLUID_CYCLES_SPARE = 2;
const VISCOSITY_ITERATIONS = 8;
// Gravity (gravity.wgsl): the field of the world's masses, worked out every
// GRAVITY_EVERY steps and after anything is painted. What pulls noticeably,
// a black hole, stays put and doesn't grow as it eats, and the rest pulls
// with around a ten-billionth of g, so twice a second keeps up with it;
// each time takes ~0.7 ms at 530 x 260. Its blocks are light cells
// (GRAVITY_BLOCK), and its pyramid tops out at GRAVITY_TOP_BLOCKS blocks.
const GRAVITY_EVERY = 30;
const NEWTON_G = 6.674e-11;  // m³ / (kg s²)
// A black hole pulls this many g at its edge unless the UI says otherwise.
export const DEFAULT_BLACK_HOLE_PULL = 10;
// How far down the open air the wind gets (sky.wgsl's ventilation) is found
// again every ROOM_EVERY steps: it changes rarely. (The rooms, the air's
// pressure zones, every step.)
const ROOM_EVERY = 4;
// A cell is a centimetre unless a scene or the scale setting says otherwise
// (Params.dx in common.wgsl).
export const DEFAULT_METRES_PER_CELL = CELL_M;
const SUPPORT_FRACTION = 0.3;
const MIN_SUPPORT = 48;
// Electric current (electric.wgsl): the potential is solved over blocks of
// ELECTRIC_BLOCK cells (shared.js has the layout), and the current pass works
// over blocks of CURRENT_BLOCK cells.
const CURRENT_BLOCK = 16;
const FLOW_BYTES = 4;
// Elements that drive current (batteries): painting one turns the electric
// passes on (see electricLive).
const EMF_ELEMENTS = new Set(ELEMENTS.filter((e) => e.emf !== 0).map((e) => e.index));
// Free bodies (bodies.wgsl; shared.js has the buffer's layout): per cell a
// label, a claim word and BODY_SUM_WORDS of sums, and a pool for their
// shapes, two halves of two words per world cell.
const BODY_GROUPS = 128;     // workgroups the passes over the tiles worked on go through them with
const BODY_WORKGROUPS = 64;  // and those over the bodies
// The stress solve's passes (stress.wgsl), as its header lists them: [name,
// entry point, override constants]; apply and update for each parity of
// the iteration, apply for the first apart.
const STRESS_PASSES = [
    ["setup", "setup"], ["residual", "residual"], ["begin", "begin"],
    ["applyFirst", "apply", { PARITY: 0, FIRST: 1 }], ["apply0", "apply", { PARITY: 0 }], ["apply1", "apply", { PARITY: 1 }],
    ["update0", "update", { PARITY: 0 }], ["update1", "update", { PARITY: 1 }],
    ["measure", "measure"], ["crack", "crack"], ["finish", "finish"],
];
// With no structure as of the last report, the stress solve runs only every
// STRESS_LOOK_EVERY steps (and after painting), to find out whether there is.
const STRESS_LOOK_EVERY = 16;

const BODY_PASSES = [
    "plan", "init", "unite", "gather", "found", "foundBodies", "capture", "lift", "parts", "contact", "jointSolve", "integrate", "stamp", "finish",
    "listBodies", "finishListed", "compactPlan", "compactCopy",
    "structureWatch", "structureInit", "structureUnite", "structureAnchor", "structureSettle", "structureDone",
];
// The claim words hold the step as common.wgsl's claimStep has it, which
// comes round every CLAIM_STEPS steps (CLAIM_PERIOD); they are cleared as
// it does, or old claims would outrank new ones.
// People (person.wgsl, people.wgsl): a slot each, of PERSON_BYTES, and the
// requests to place and erase them, sent before a frame's steps.
const PERSON_BYTES = 80;
const CLAIM_STEPS = 2 ** 20 - 1;
// What the passes that can be left out (liquid, bodies, current, people)
// have to do is read back each frame, into one of REPORTS staging buffers,
// and taken in a frame or two late (readReport): the liquid's block counts,
// the bodies' stats, the electric stats (the batteries first) and the
// people. Each of those has an epoch (Simulation.epochs), bumped whenever
// work for it comes from outside its passes (the world cleared or loaded, a
// liquid or a battery painted, a person placed): the parts of a report asked
// for before the last bump are of a world that is no more, and are let go.
// A report of the bodies asked for more than REPORT_PATIENCE stepped frames
// ago is too old to go by: their passes run anyway.
// The report also holds the stiffest face of conduction the frame's update
// passes found, which sets the next frames' heat stages, and the most
// iterations the air's projection took in a step since the last report,
// which sets how many the next frames' steps encode (airCycles).
const REPORTS = 3;
const REPORT_FLUID = 0;
const REPORT_BODIES = REPORT_FLUID + FLUID_STATS;
const REPORT_ELECTRIC = REPORT_BODIES + BODY_STATS;
const REPORT_HEAT = REPORT_ELECTRIC + ELECTRIC_STATS;
const REPORT_STRESS = REPORT_HEAT + 1;
const REPORT_AIR = REPORT_STRESS + STRESS_STATS;
const REPORT_PEOPLE = REPORT_AIR + 1;
const REPORT_BYTES = REPORT_PEOPLE * 4 + MAX_PEOPLE * PERSON_BYTES;
const REPORT_PATIENCE = 3;
// Conduction across stiff faces (heat.wgsl), in s stages of damped
// first-order Runge-Kutta-Chebyshev, s from 2 to HEAT_MAX_STAGES: as few as
// take the stiffest face the last report found (heatStageCount).
// HEAT_DAMPING is ε, how far below 1 the stiffest wiggles' amplitude is
// held each step (to about 1 / cosh √(2ε)), at a little of the stages'
// reach.
const HEAT_DAMPING = 0.15;
const HEAT_MAX_STAGES = 32;
const HEAT_STAGE_BYTES = 16;
// With no stages, heat.wgsl's prepare still looks for stiff faces every
// HEAT_LOOK_EVERY steps (update.wgsl holds them to what stays monotone
// until there are stages).
const HEAT_LOOK_EVERY = 4;

// For s stages: each stage's weights (heat.wgsl's Stage: the stage before,
// the one before that, what the stage before passes), and how stiff a face
// they take, as J/K a step over the smaller heat capacity of its cells. A
// step's stage values follow the Chebyshev polynomials T_j at
// w0 + w1 z (z the step's eigenvalue), shifted so the first derivative at
// z = 0 is 1 (first order) and damped by w0 = 1 + ε / s²; stable while
// w0 + w1 z stays in [-1, 1], to z = -(1 + w0) / w1, about (2 - 4ε / 3) s²,
// and Gershgorin's bound puts a cell whose four faces each pass k of its
// heat capacity at z = -8k at most.
function chebyshevStages(s) {
    const w0 = 1 + HEAT_DAMPING / (s * s);
    const T = [1, w0];
    const U = [1, 2 * w0];
    for (let j = 2; j <= s; j++) {
        T.push(2 * w0 * T[j - 1] - T[j - 2]);
        U.push(2 * w0 * U[j - 1] - U[j - 2]);
    }
    const w1 = T[s] / (s * U[s - 1]);
    const stages = [[1, 0, w1 / w0]];
    for (let j = 2; j <= s; j++) {
        stages.push([2 * w0 * T[j - 1] / T[j], -T[j - 2] / T[j], 2 * w1 * T[j - 1] / T[j]]);
    }
    return { stages, limit: (1 + w0) / w1 / 8 };
}
const HEAT_CHEBYSHEV = Array.from({ length: HEAT_MAX_STAGES + 1 }, (_, s) => (s ? chebyshevStages(s) : null));
// Light and radiant heat (light.wgsl), worked out every LIGHT_EVERY steps
// over light cells of LIGHT_CELL x LIGHT_CELL cells, through cascades of
// probes: cascade k has a probe per 2^k light cells a side, each casting
// 4 x 2^k rays, out to 2^(k+1) - 1/2 light cells. There are MIN_CASCADES at
// least, and as many as reach CASCADE_REACH of the world's longer side;
// past that, a ray sees the sky or the ground (a diagonal across the whole
// world sees too little of the far corner to matter, and one more cascade
// would cost as much as each of the others).
export const LIGHT_EVERY = 4;
// Radiant heat's directions and the most cells along a world's shorter
// side its sweep's workgroup holds (radiate.wgsl's RAD_DIRS and SPAN).
const RADIANT_DIRS = 16;
const RADIANT_SPAN = 2048;
const SUN_EVERY = 2;  // light refreshes per sun sweep
const LIGHT_CELL = 2;
// The most light cells along a line the sun's sweep holds (light.wgsl's
// SWEEP_MAX), either way: a world's rows and its columns are both lines.
const SUN_SPAN = 2048;
const MIN_CASCADES = 2;
const MAX_CASCADES = 16;
const CASCADE_REACH = 0.9;

// The clock. A day lasts DAY_STEPS steps at normal speed (four minutes at
// 60 steps a second); DAY_SPEEDS are the clock's speeds: stopped, slow,
// normal and fast.
export const DAY_SPEEDS = [0, 1 / 3, 1, 4];
export const CLOCK = { STOPPED: 0, SLOW: 1, NORMAL: 2, FAST: 3 };

export const MAX_STEPS_PER_FRAME = Math.floor(MAX_SLOTS / (1 + BLOCK_PASSES));
// A brush that paints again within a frame's steps of the last time carries
// on its stroke (sim.paint): a held brush paints a segment a frame.
const STROKE_GAP = MAX_STEPS_PER_FRAME;
// Phases of a frame, each encoded as its own passes so they can be timed.
export const PHASES = [
    "light", "gravity", "air", "carry", "fluid", "heat", "update", "stress", "bodies", "move", "people", "electric", "draw",
];
// The most passes a step begins (encodeSteps): one per simulation phase
// above, and heat's (the clock, then the stages), update's (the pairs, then
// the cells) and the bodies' (the structure, then the bodies) in two each.
const STEP_PASSES = 15;
const DRAW_PASSES = 2;   // glow, then the render pass

// A flood fill (fill) takes in at most this many cells.
const FILL_MAX = 1 << 22;

// What the GPU has to allow: the storage buffers the liquid's passes bind
// (fluid.wgsl's, the most any pass does: createPipelines checks), and per
// cell the bytes of the largest buffer. (Browsers tell what they allow in
// steps, 8, 10, 16, 31: more than 10 asks for 16, which Safari doesn't
// give.) The largest buffer is
// the liquid's grid: a cell's two vec4s and a third as much again for the
// coarse levels, two faces' vec4s and half as much again (two quarter-words
// a face since 7-11), and a little for the faces along the edges; or its particles, LIQUID_ROOM cells' worth
// a cell. (The bodies' is a label, a claim word and BODY_SUM_WORDS of sums a
// cell.)
export const STORAGE_BUFFERS_NEEDED = 14;
const FIELD_BYTES_PER_CELL = Math.ceil(FIELD_CELL_BYTES * 4 / 3 + 2 * FIELD_FACE_BYTES * 1.5) + 2;
const PARTICLE_BYTES_PER_CELL = Math.ceil(LIQUID_PER_CELL * LIQUID_ROOM * PARTICLE_BYTES);
const LARGEST_CELL_BYTES = Math.max(FIELD_BYTES_PER_CELL, PARTICLE_BYTES_PER_CELL, (2 + BODY_SUM_WORDS) * 4);

// The layouts the developer's toolbox (devtools.js) reads the buffers by,
// kept here with the buffers.
export const LAYOUT = {
    AIR_BYTES, FLOW_BYTES, PARTICLE_BYTES, PARTICLES_HEADER, FIELD_CELL_BYTES, FIELD_FACE_BYTES, LIGHT_CELL, PERSON_BYTES,
};

// The most cells a world can have on a device with these limits: its largest
// buffer has to fit in one binding, and the passes dispatched with a
// workgroup per fluid block, in one dimension, can't ask for more than 65535.
// Of those, FLUID_EDGE_GROUPS are kept for the blocks the world's edges cut
// short and the shifted relaxation's tiles past them, about (w + h) / 4 in
// all: enough for worlds up to 16000 cells across and down together.
const FLUID_EDGE_GROUPS = 4096;
export function maxCells(limits) {
    const bytes = Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize);
    return Math.min(Math.floor(bytes / LARGEST_CELL_BYTES), (65535 - FLUID_EDGE_GROUPS) * FLUID_BLOCK * FLUID_BLOCK);
}

const scratchF32 = new Float32Array(1);
const scratchU32 = new Uint32Array(scratchF32.buffer);

function f32Bits(value) {
    scratchF32[0] = value;
    return scratchU32[0];
}

// Maps a line in the concatenated source back to "file:line".
function locate(parts, line) {
    let start = 1;
    for (const [name, text] of parts) {
        const count = text.split("\n").length;
        if (line < start + count) {
            return `${name}.wgsl:${line - start + 1}`;
        }
        start += count;
    }
    return `?:${line}`;
}

async function compile(device, label, parts) {
    const code = parts.map(([, text]) => text).join("\n");
    const module = device.createShaderModule({ label, code });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === "error");
    if (errors.length > 0) {
        const lines = errors.map((m) => `${locate(parts, m.lineNum)}: ${m.message}`);
        throw new Error(`Shader "${label}" failed to compile:\n${lines.join("\n")}`);
    }
    return module;
}

// Each shader module, [label, parts]: the sources joined, in order, to make
// it (also read by test/naga.mjs).
export function shaderModules(sources) {
    const prelude = ["prelude", wgslPrelude()];
    const part = (name) => [name, sources[name]];
    const base = [prelude, ["shared", sharedPrelude()], ["record", recordPrelude()], part("common")];
    return [
        ["update", [...base, part("airsample"), part("pull"), part("airsource"), part("label"), part("update")]],
        ["heat", [...base, part("airsample"), part("heat")]],
        ["pairs", [...base, part("airsample"), part("pull"), part("airsource"), part("pairs")]],
        ["sky", [...base, part("label"), part("labeltile"), part("sky")]],
        ["fluid", [...base, part("airsample"), part("pull"), part("airsource"), part("label"), part("fluid")]],
        ["bodies", [...base, part("airsample"), part("pull"), part("airsource"), part("label"), part("labeltile"), part("bodies")]],
        ["block", [...base, part("airsample"), part("pull"), part("airsource"), part("block")]],
        ["electric", [...base, part("heatin"), part("labeltile"), part("electric")]],
        ["air", [...base, part("label"), part("rooms"), part("pull"), part("airsource"), part("air")]],
        ["carry", [...base, part("label"), part("rooms"), part("pull"), part("airsource"), part("carry")]],
        ["brush", [...base, part("airsource"), part("brush")]],
        ["glow", [...base, part("shade"), part("glow")]],
        ["blur", [part("blur")]],
        ["render", [...base, part("shade"), part("render")]],
        ["light", [...base, part("shade"), part("light")]],
        ["clock", [...base, part("clock")]],
        ["gravity", [...base, part("gravity")]],
        ["people", [...base, part("airsample"), part("pull"), part("airsource"), part("person"), part("people")]],
        ["figures", [...base, part("shade"), part("person"), part("figures")]],
        ["stress", [...base, part("airsample"), part("pull"), part("stress")]],
        ["radiate", [...base, part("radiate")]],
    ];
}

export async function createPipelines(device, format, sources) {
    const [
        update, heat, pairs, sky, fluid, bodies, block, electric, air, carry, brush, glow, blur, render, light, clock, gravity,
        people, figures, stress, radiate,
    ] = await Promise.all(shaderModules(sources).map(([label, parts]) => compile(device, label, parts)));

    // Sim passes take a dynamic offset into one uniform buffer holding a
    // parameter slot per pass, so a whole frame is written with one upload.
    const COMPUTE = GPUShaderStage.COMPUTE;
    const params = { binding: 0, visibility: COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: PARAMS_BYTES } };
    const readOnly = (binding) => ({ binding, visibility: COMPUTE, buffer: { type: "read-only-storage" } });
    const writable = (binding) => ({ binding, visibility: COMPUTE, buffer: { type: "storage" } });
    // (No pass binds more storage buffers than STORAGE_BUFFERS_NEEDED says,
    // which the app asks the GPU for.)
    const layout = (label, entries) => {
        const storage = entries.filter((e) => e.buffer?.type === "storage" || e.buffer?.type === "read-only-storage").length;
        if (storage > STORAGE_BUFFERS_NEEDED) {
            throw new Error(`The ${label} passes bind ${storage} storage buffers; STORAGE_BUFFERS_NEEDED says ${STORAGE_BUFFERS_NEEDED}`);
        }
        return device.createBindGroupLayout({ label, entries });
    };
    const texture = (binding, sampleType = "unfilterable-float") => ({ binding, visibility: COMPUTE, texture: { sampleType } });
    const storageTexture = (binding, format, access = "write-only") => ({ binding, visibility: COMPUTE, storageTexture: { access, format } });
    const updateLayout = layout("update", [
        params, readOnly(1), readOnly(2), writable(3), readOnly(4), readOnly(5), writable(6), readOnly(7), writable(8),
        writable(9), texture(10), texture(11), texture(12), texture(13), writable(14), writable(15), texture(16),
    ]);
    // Radiant heat (radiate.wgsl): the cells, this time's and the last
    // time's takings and edges, and the per-cell result for update.wgsl.
    const radiateLayout = layout("radiate", [
        params, readOnly(1), readOnly(2), writable(3), readOnly(4), writable(5), readOnly(6), storageTexture(7, "rgba32float"),
    ]);
    // Conduction across stiff faces in stages (heat.wgsl), each stage's
    // weights a slot of its own.
    const heatStage = { binding: 10, visibility: COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: HEAT_STAGE_BYTES } };
    const heatLayout = layout("heat", [
        params, readOnly(1), writable(2), readOnly(3), texture(4), writable(5), writable(6), writable(7), writable(8),
        writable(9), heatStage, writable(11),
    ]);
    // Gravity (gravity.wgsl): its field is read by the passes that move
    // things (update, fluid, bodies, air) and the Gravity view.
    const gravityLayout = layout("gravity", [params, readOnly(1), readOnly(2), writable(3), storageTexture(4, "rgba32float")]);
    // Light (light.wgsl): its passes' own layouts, as one texture is written
    // by one pass and read by the next.
    const cascadeSlot = { binding: 11, visibility: COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 16 } };
    const cascadeEntries = [
        params, texture(4, "uint"), texture(5), texture(8, "float"), storageTexture(9, "rgba16float"), cascadeSlot,
        texture(19, "uint"),
    ];
    const lightLayouts = {
        build: layout("build", [
            params, readOnly(1), readOnly(2), storageTexture(3, "rgba32uint"), texture(5), texture(17, "uint"),
            storageTexture(18, "rgba32uint"), readOnly(22),
        ]),
        sun: layout("sun", [params, texture(4, "uint"), storageTexture(6, "r32float")]),
        bounce: layout("bounce", [params, readOnly(1), readOnly(2), texture(4, "uint"), texture(7), writable(21)]),
        cascade: layout("cascade", cascadeEntries),
        gather: layout("gather", [
            params, texture(4, "uint"), texture(7), texture(8, "float"), cascadeSlot,
            storageTexture(12, "rgba16float"), storageTexture(13, "r32float"), storageTexture(20, "rgba32uint"), writable(21),
        ]),
    };
    // Heat's clock (clock.wgsl): each light cell's hottest matter, then
    // across and down.
    const clockLayout = layout("clock", [
        params, readOnly(1), readOnly(2), writable(3), writable(4), storageTexture(5, "r32float", "read-write"), writable(6),
    ]);
    const bodiesLayout = layout("bodies", [
        params, readOnly(1), writable(2), writable(4), writable(5), writable(7), writable(8),
        texture(9), writable(10), readOnly(11),
    ]);
    const skyLayout = layout("sky", [params, readOnly(1), readOnly(2), writable(3), readOnly(4)]);
    const level = { binding: 9, visibility: COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 16 } };
    const fluidLayout = layout("fluid", [
        params, readOnly(1), writable(2), readOnly(3), writable(4), writable(5), writable(6), writable(7), writable(8), level,
        texture(10), writable(11), writable(12), readOnly(13), readOnly(14), writable(15), writable(16),
    ]);
    // (The block pass and the pairs pass, which reads heat's clock too.)
    const blockLayout = layout("block", [params, readOnly(1), writable(2), readOnly(3), texture(4), writable(5), texture(6), readOnly(7)]);
    const electricLayout = layout("electric", [
        params, readOnly(1), writable(2), writable(3), writable(4), writable(5), writable(6), writable(7),
        texture(9), writable(10), writable(11),
    ]);
    // The stress solve (stress.wgsl): params, elements, cells, air, the
    // liquid's field, the piles' loads, the bodies buffer (structure's
    // words), its nodes and working words, gravity, its nodes' displacements.
    const stressLayout = layout("stress", [
        params, readOnly(1), readOnly(2), readOnly(3), readOnly(4), readOnly(5), writable(6), writable(7), writable(8), texture(9),
        writable(10),
    ]);
    // The air (air.wgsl): the air, its working state and the projection's
    // levels, gravity, the zones' sums in the sky buffer, the blasts' layer,
    // the level and iteration a pass of the solve works on, the solve's
    // status, and its conjugate gradients' vectors.
    const airLevel = { binding: 10, visibility: COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 16 } };
    const airLayout = layout("air", [
        params, readOnly(1), readOnly(2), writable(3), writable(4), writable(5), texture(7), writable(8), writable(9),
        airLevel, writable(12), writable(13), writable(14), readOnly(15),
    ]);
    // The wind's carry (carry.wgsl): the cells, carried in place; the air;
    // the fastest wind; the carried air, scratch; the substep; what other
    // passes put into the air, taken in; and the projection's potential (the
    // conjugate gradients' vectors), the flow out of the world's left edge.
    const carrySubstep = { binding: 6, visibility: COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 16 } };
    const carryLayout = layout("carry", [
        params, readOnly(1), writable(2), readOnly(3), readOnly(4), writable(5), carrySubstep, writable(7), readOnly(8),
        readOnly(9), texture(10), writable(11),
    ]);
    // People (people.wgsl): the cells, which a person on fire sets alight;
    // the air, the current density and gravity, which push, shock and pull
    // them; the people themselves; the requests; what goes into the air
    // (airsource.wgsl), the drag they give it; and the radiant heat that
    // reaches them (light.wgsl).
    const peopleLayout = layout("people", [
        params, readOnly(1), writable(2), readOnly(3), readOnly(4), texture(5), writable(6), readOnly(7), writable(8),
        texture(9), writable(10), texture(11),
    ]);

    const computePipeline = (label, module, entryPoint, bindLayout, constants = {}) => device.createComputePipelineAsync({
        label,
        layout: bindLayout ? device.createPipelineLayout({ bindGroupLayouts: [bindLayout] }) : "auto",
        compute: { module, entryPoint, constants },
    });

    const fluidEntries = [
        "sync", "syncMarks", "syncParticles", "syncSeed", "retire", "clear", "gather", "sealJoin", "sealAnchor", "classify",
        "bodyStart", "bodyLoads", "setFaces", "bodyDrag", "bodyFaces",
        "divergence", "relaxRims", "relax", "relaxShifted", "restrictLevel", "smoothRed", "smoothBlack", "solve", "prolong",
        "prolongFine", "cgStart", "cgJoin", "cgAnchor", "cgBodyX", "cgResidual", "cgPrepare", "cgRho", "cgBeta", "cgDirection", "cgBodyP",
        "cgApply", "cgAlpha", "cgUpdate", "cgFinish", "cgBodyF", "bodyPush",
        "project", "viscosityRed", "viscosityBlack", "unpress", "advect", "sortCount", "scanBlocks",
        "scanTotals", "scanFinish", "scatter", "rasterClaims", "raster", "unpaint", "compact", "surface", "extrapolate",
        "extrapolateFurther", "rigidify",
    ];
    const fluidPipes = Promise.all(fluidEntries.map((entry) => computePipeline(entry, fluid, entry, fluidLayout)));
    const bodyPipes = Promise.all(BODY_PASSES.map((entry) => computePipeline(entry, bodies, entry, bodiesLayout)));
    const stressPipes = Promise.all(STRESS_PASSES.map(([name, entry, constants]) => computePipeline(
        `stress ${name}`, stress, entry, stressLayout, constants,
    )));
    const lightEntries = Object.keys(lightLayouts);
    const lightPipes = Promise.all(lightEntries.map((entry) => computePipeline(entry, light, entry, lightLayouts[entry])));
    const gravityEntries = ["mass", "upper", "field"];
    const gravityPipes = Promise.all(gravityEntries.map((entry) => computePipeline(entry, gravity, entry, gravityLayout)));
    const airEntries = [
        "gather", "blastPressure", "blastVelocity", "momentum", "divergence", "down", "smoothTile", "up",
        "solveCoarsest", "cgStart", "cgRho", "cgBeta", "cgDirection", "cgApply", "cgAlpha", "cgUpdate",
        "zonePotential", "project", "recentre",
    ];
    const airPipes = Promise.all(airEntries.map((entry) => computePipeline(`air ${entry}`, air, entry, airLayout)));
    // (flow from the scratch's set S into T, and from T into S.)
    const carryEntries = [["rise"], ["load"], ["flow", { FROM_T: 0 }], ["flow", { FROM_T: 1 }], ["settle"]];
    const carryNames = ["rise", "load", "flowToT", "flowToS", "settle"];
    const carryPipes = Promise.all(carryEntries.map(([entry, constants], i) => computePipeline(
        `carry ${carryNames[i]}`, carry, entry, carryLayout, constants,
    )));
    const [
        lightList, clockSource, clockAcross, clockDown, gravityList, updatePipe, zoneHeatPipe, heatPrepare, heatStagePipe,
        pairsPipe, skyPipe,
        ventilationPipe, roomsPipe, roomUnitePipe, wetRoomsPipe, wetUnitePipe, fluidList, bodyList, blockPipe, assemble, residual, coarseSolve,
        precondition, reduce, direction, current, effects, airList, carryList, wind, brushPipe, glowPipe,
        shinePipe, blurH, blurV, renderPipe, peoplePipe, placePipe, figuresPipe, stressList, radiateSweep, radiateResolve,
    ] = await Promise.all([
        lightPipes,
        computePipeline("clockSource", clock, "source", clockLayout),
        computePipeline("clockAcross", clock, "across", clockLayout),
        computePipeline("clockDown", clock, "down", clockLayout),
        gravityPipes,
        computePipeline("update", update, "main", updateLayout),
        computePipeline("zoneHeat", update, "zoneHeat", updateLayout),
        computePipeline("heatPrepare", heat, "prepare", heatLayout),
        computePipeline("heatStage", heat, "stage", heatLayout),
        // Same bindings as the block pass: params, elements, cells, air.
        computePipeline("pairs", pairs, "main", blockLayout),
        computePipeline("sky", sky, "main", skyLayout),
        computePipeline("ventilation", sky, "ventilation", skyLayout),
        computePipeline("rooms", sky, "rooms", skyLayout),
        computePipeline("roomUnite", sky, "roomUnite", skyLayout),
        computePipeline("wetRooms", sky, "rooms", skyLayout, { WET: 1 }),
        computePipeline("wetUnite", sky, "roomUnite", skyLayout, { WET: 1 }),
        fluidPipes,
        bodyPipes,
        computePipeline("block", block, "main", blockLayout),
        computePipeline("assemble", electric, "assemble", electricLayout),
        computePipeline("residual", electric, "residual", electricLayout),
        computePipeline("coarseSolve", electric, "coarseSolve", electricLayout),
        computePipeline("precondition", electric, "precondition", electricLayout),
        computePipeline("reduce", electric, "reduce", electricLayout),
        computePipeline("direction", electric, "direction", electricLayout),
        computePipeline("current", electric, "current", electricLayout),
        computePipeline("effects", electric, "effects", electricLayout),
        airPipes,
        carryPipes,
        computePipeline("wind", air, "wind"),
        computePipeline("brush", brush, "main"),
        computePipeline("glow", glow, "main"),
        computePipeline("shine", glow, "shine"),
        computePipeline("blurH", blur, "blurH"),
        computePipeline("blurV", blur, "blurV"),
        device.createRenderPipelineAsync({
            label: "render",
            layout: "auto",
            vertex: { module: render, entryPoint: "vs" },
            fragment: { module: render, entryPoint: "fs", targets: [{ format }] },
            primitive: { topology: "triangle-list" },
        }),
        computePipeline("people", people, "main", peopleLayout),
        computePipeline("place", people, "place", peopleLayout),
        // The people over the world, a quad each, blended (fading, ghosts).
        device.createRenderPipelineAsync({
            label: "figures",
            layout: "auto",
            vertex: { module: figures, entryPoint: "vs" },
            fragment: {
                module: figures,
                entryPoint: "fs",
                targets: [{
                    format,
                    blend: {
                        color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" },
                        alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
                    },
                }],
            },
            primitive: { topology: "triangle-list" },
        }),
        stressPipes,
        computePipeline("radiate sweep", radiate, "sweep", radiateLayout),
        computePipeline("radiate resolve", radiate, "resolve", radiateLayout),
    ]);

    const elementData = packElements();
    const elemBuf = device.createBuffer({
        label: "elements",
        size: elementData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(elemBuf, 0, elementData);

    return {
        light: Object.fromEntries(lightEntries.map((entry, i) => [entry, lightList[i]])),
        lightLayouts,
        clock: { source: clockSource, across: clockAcross, down: clockDown },
        clockLayout,
        gravity: Object.fromEntries(gravityEntries.map((entry, i) => [entry, gravityList[i]])),
        gravityLayout,
        update: updatePipe,
        zoneHeat: zoneHeatPipe,
        updateLayout,
        heat: { prepare: heatPrepare, stage: heatStagePipe },
        heatLayout,
        pairs: pairsPipe,
        sky: skyPipe,
        ventilation: ventilationPipe,
        rooms: roomsPipe,
        roomUnite: roomUnitePipe,
        wetRooms: wetRoomsPipe,
        wetUnite: wetUnitePipe,
        skyLayout,
        fluid: Object.fromEntries(fluidEntries.map((entry, i) => [entry, fluidList[i]])),
        fluidLayout,
        bodies: Object.fromEntries(BODY_PASSES.map((entry, i) => [entry, bodyList[i]])),
        bodiesLayout,
        block: blockPipe,
        blockLayout,
        electric: { assemble, residual, coarseSolve, precondition, reduce, direction, current, effects },
        electricLayout,
        stress: Object.fromEntries(STRESS_PASSES.map(([name], i) => [name, stressList[i]])),
        stressLayout,
        radiate: { sweep: radiateSweep, resolve: radiateResolve },
        radiateLayout,
        air: Object.fromEntries(airEntries.map((entry, i) => [entry, airList[i]])),
        airLayout,
        carry: Object.fromEntries(carryNames.map((name, i) => [name, carryList[i]])),
        carryLayout,
        wind,
        brush: brushPipe,
        glow: glowPipe,
        shine: shinePipe,
        blurH,
        blurV,
        render: renderPipe,
        people: peoplePipe,
        place: placePipe,
        peopleLayout,
        figures: figuresPipe,
        elemBuf,
        sampler: device.createSampler({ magFilter: "linear", minFilter: "linear" }),
    };
}

// Blocks in gravity.wgsl's pyramid over a world of width × height cells:
// every level up to its top (at most GRAVITY_TOP_BLOCKS blocks), and at
// least up to level 3, which its mass pass writes whatever the world's size.
function gravityBlocks(width, height) {
    const n0 = [Math.ceil(width / GRAVITY_BLOCK), Math.ceil(height / GRAVITY_BLOCK)];
    let total = 0;
    let topped = false;
    for (let l = 0; !topped || l <= 3; l++) {
        const [w, h] = n0.map((n) => Math.ceil(n / 2 ** l));
        total += w * h;
        topped ||= w * h <= GRAVITY_TOP_BLOCKS;
    }
    return total;
}

// Probes a side in cascade k over light cells (lw, lh), and the directions
// each casts rays in (light.wgsl).
function probes(k, lw, lh) {
    return [Math.ceil(lw / 2 ** k), Math.ceil(lh / 2 ** k)];
}

function directions(k) {
    return 4 * 2 ** k;
}

export class Simulation {
    // Throws if a world width x height can't be made, before anything is
    // made for it: radiant heat's wavefront holds the world's shorter side.
    static check(width, height) {
        if (Math.min(width, height) > RADIANT_SPAN) {
            throw new Error(`radiant heat sweeps at most ${RADIANT_SPAN} cells along a world's shorter side`);
        }
        if (Math.max(width, height) > SUN_SPAN * LIGHT_CELL) {
            throw new Error(`the sun sweeps at most ${SUN_SPAN * LIGHT_CELL} cells along a world's side`);
        }
    }

    // clock: the time of day to start at and the clock's speed (see
    // setClock); 9:00 with the clock running if there is none.
    // metresPerCell: the world's scale, which clear() lays its air out by.
    constructor(device, pipes, width, height, clock = { day: 9 / 24, speed: CLOCK.NORMAL }, metresPerCell = DEFAULT_METRES_PER_CELL) {
        Simulation.check(width, height);
        this.device = device;
        this.pipes = pipes;
        this.earthGravity = true;  // the Earth pulls (see setEarthGravity)
        this.width = width;
        this.height = height;
        this.clock = { ...clock };
        this.airWidth = Math.ceil(width / AIR_CELL);
        this.airHeight = Math.ceil(height / AIR_CELL);
        // A standing structural cell's support (frac.x, common.wgsl), which
        // anchoring takes below -supportMax to let a piece go, and from which
        // a loose cell's steps loose are told (bodies.wgsl's looseSteps).
        // (Until 8-2, the longest span structure could reach.)
        this.supportMax = Math.max(MIN_SUPPORT, Math.round(SUPPORT_FRACTION * Math.min(width, height)));
        // Whether a stroke of the brush is being drawn (main.js): structure
        // it paints holds where it is until it is let go (common.wgsl's
        // SUPPORT_HELD).
        this.holding = false;
        this.paintedAt = null;  // the step the brush last painted at (paint: a stroke going on)
        this.metresPerCell = metresPerCell;
        this.cur = 0;          // index of the cell buffer holding the latest state
        this.frameNo = 0;
        this.passNo = 0;
        this.phaseX = 0;
        // The random choices the steps make come from here (see seed).
        this.rngState = (Math.random() * 0x100000000) >>> 0;
        this.readPending = false;
        this.onHover = null;
        this.timer = device.features.has("timestamp-query")
            ? new GpuTimer(device, STEP_PASSES * MAX_STEPS_PER_FRAME + DRAW_PASSES)
            : null;

        const U = GPUBufferUsage;
        // The cells, and after them the gas plane (shared.js GAS_BYTES).
        this.cells = [0, 1].map((i) => device.createBuffer({
            label: `cells${i}`,
            size: cellsBufferBytes(width * height),
            usage: U.STORAGE | U.COPY_SRC | U.COPY_DST,
        }));
        // The air (air.wgsl): air[0] as the other passes read it; air[1] its
        // working state; the projection's levels, down to one workgroup's
        // worth (AIR_COARSEST cells), two vec4s a cell; the blasts' layer.
        this.air = [0, 1].map((i) => device.createBuffer({
            label: `air${i}`,
            size: this.airWidth * this.airHeight * AIR_BYTES,
            usage: U.STORAGE | U.COPY_SRC | U.COPY_DST,
        }));
        this.airLevels = [[this.airWidth, this.airHeight]];
        while (this.airLevels.length < AIR_LEVELS) {
            const [lw, lh] = this.airLevels[this.airLevels.length - 1];
            if (lw * lh <= AIR_COARSEST) {
                break;
            }
            this.airLevels.push([Math.ceil(lw / 2), Math.ceil(lh / 2)]);
        }
        this.airProjection = device.createBuffer({
            label: "airProjection",
            size: this.airLevels.reduce((n, [lw, lh]) => n + lw * lh, 0) * 2 * AIR_BYTES,
            usage: U.STORAGE | U.COPY_SRC | U.COPY_DST,
        });
        this.blast = device.createBuffer({
            label: "blast",
            size: this.airWidth * this.airHeight * AIR_BYTES,
            usage: U.STORAGE | U.COPY_SRC | U.COPY_DST,
        });
        // The air's status (air.wgsl): the most any cell's flow gives out
        // this step, which sets the wind's carry's substeps (carry.wgsl), and
        // the solve's; and per cell four vec4s of the carry's scratch (its
        // air through the substeps, twice).
        this.airFastest = device.createBuffer({ label: "airFastest", size: 32, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
        this.wakeBlast();
        // The air's projection's iterations a step, as the last report has
        // them (AIR_CYCLES_SPARE), and the liquid's (FLUID_CYCLES_SPARE).
        this.airCycles = AIR_CYCLES;
        this.fluidCycles = FLUID_CYCLES;
        this.carryScratch = device.createBuffer({ label: "carryScratch", size: width * height * 4 * 16, usage: U.STORAGE });
        // Liquid particles and their grid (fluid.wgsl): the particles sorted
        // by cell between steps (particles[0]) and a step's (particles[1]),
        // room for LIQUID_ROOM cells' worth a cell; per cell the liquid's
        // words (LIQUID_CELL_WORDS), then per face two.
        const cellCount = width * height;
        const faceCount = (width + 1) * height + width * (height + 1);
        this.particleSlots = Math.ceil(cellCount * LIQUID_PER_CELL * LIQUID_ROOM);
        this.blockCount = Math.ceil(width / FLUID_BLOCK) * Math.ceil(height / FLUID_BLOCK);
        // The shifted relaxation also takes the tiles of the column of blocks
        // past the world's right edge, the corner and the row past its
        // bottom (tileBlock in fluid.wgsl).
        this.edgeTiles = Math.ceil(width / FLUID_BLOCK) + Math.ceil(height / FLUID_BLOCK) + 1;
        // Which tiling the relaxation starts each step on, the other one from
        // the last step's (see encodeFluid).
        this.shiftedFirst = false;
        const storage = (label, size, usage = 0) => device.createBuffer({
            label, size, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST | usage,
        });
        this.particles = [0, 1].map((i) => storage(`particles${i}`, PARTICLES_HEADER + this.particleSlots * PARTICLE_BYTES));
        this.counters = storage("counters", (LIQUID_CELL_WORDS * cellCount + 2 * faceCount) * 4);
        // The multigrid's coarse levels, each half the last on a side, down
        // to one a single workgroup can solve.
        this.levels = [];
        for (let l = 1; l <= MG_LEVELS; l++) {
            const size = [Math.ceil(width / 2 ** l), Math.ceil(height / 2 ** l)];
            this.levels.push(size);
            if (size[0] * size[1] <= MG_COARSEST) {
                break;
            }
        }
        const levelCells = this.levels.reduce((n, [lw, lh]) => n + lw * lh, 0);
        this.field = storage("field", (cellCount + levelCells) * FIELD_CELL_BYTES
            + (faceCount + 2 * Math.ceil(faceCount / 4)) * FIELD_FACE_BYTES);
        // Per block marks and state, the lists of blocks worked on and
        // dropping out, how many there are of each, and whether any liquid
        // is viscous. Those last are read back every frame, a frame or two
        // late, to leave out what there is no liquid for (see encodeFluid).
        this.blocks = storage("blocks", (5 * this.blockCount + FLUID_STATS) * 4);
        // The pressure's conjugate gradients (fluid.wgsl's cg): per cell the
        // pressure, residual, direction and what the operator makes of it;
        // per block the partial sums of a pass; then the sums. Written
        // before it is read each step.
        this.fluidCg = storage("fluidCg", (cellCount + this.blockCount + 1) * 16);
        // What the liquid's solve does to each body, summed (fluid.wgsl's
        // pushes): a step's scratch, cleared as the step's liquid starts.
        this.bodyPushes = storage("bodyPushes", MAX_BODIES * BODY_PUSH_WORDS * 4);
        // Free bodies (bodies.wgsl): the header (per tile its mark and
        // whether it is worked on, the stats read back, the lists of the
        // marked tiles and those worked on, the counters and the free
        // slots), then per cell its label and its claim word, and per cell
        // (as a piece's root) the piece's sums, structure's words, and per
        // cell when a body last moved over it and which (two words, by the
        // step's parity); as plain words, the list of bodies worked on, the
        // table of bodies, the contacts and what the liquid did to each;
        // and the pool of their shapes.
        this.bodyTiles = Math.ceil(width / BODY_TILE) * Math.ceil(height / BODY_TILE);
        this.bodyHeader = 4 * this.bodyTiles + BODY_STAT_WORDS + BODY_COUNTERS + MAX_BODIES;
        // (After the sums, structure's words: common.wgsl's structureBase.)
        this.bodies = storage("bodies", (this.bodyHeader + (2 + BODY_SUM_WORDS) * cellCount + 1 + this.bodyTiles + 5 * cellCount) * 4);
        this.bodyMotion = storage("bodyMotion",
            (MAX_BODIES * (1 + BODY_TABLE_WORDS + BODY_LIQUID_WORDS) + BODY_CONTACTS * BODY_CONTACT_WORDS) * 4);
        this.bodyShapes = storage("bodyShapes", 2 * cellCount * 8);
        // Which of the passes that can be left out have work to do, as the
        // last report taken in found (readReport): the liquid's (any block
        // with liquid, any viscous), the bodies' (anything loose, anything
        // marked to look at, the pool full), the current's (electricLive)
        // and the people's (peopleLive). Until a report says otherwise, they
        // run.
        this.reports = Array.from({ length: REPORTS }, (_, k) => ({
            buffer: device.createBuffer({ label: `report${k}`, size: REPORT_BYTES, usage: U.MAP_READ | U.COPY_DST }),
            pending: false,
        }));
        this.epochs = { fluid: 0, bodies: 0, electric: 0, people: 0, heat: 0, stress: 0, air: 0 };
        this.framesStepped = 0;  // frames with steps in them so far
        this.reportedAt = 0;     // the frame (framesStepped) the newest report of the bodies taken in was asked for
        this.fluidStats = { active: 1, viscous: 1 };
        this.bodiesLoose = true;
        this.bodiesMarked = true;
        this.bodiesFull = false;
        // The passes over the particles: a thread each there is room for, in
        // rows of at most 32768 workgroups.
        const particleGroups = Math.ceil(this.particleSlots / PARTICLE_GROUP);
        this.particleGroups = [Math.min(particleGroups, 32768), Math.ceil(particleGroups / 32768)];
        // The substep each of the wind's carry's passes works on, a slot each.
        this.levelBuf = device.createBuffer({ label: "levels", size: SLOT_BYTES * CARRY_STEPS, usage: U.UNIFORM | U.COPY_DST });
        const levelData = new Uint32Array(SLOT_BYTES / 4 * CARRY_STEPS);
        for (let k = 0; k < CARRY_STEPS; k++) {
            levelData[SLOT_BYTES / 4 * k] = k;
        }
        device.queue.writeBuffer(this.levelBuf, 0, levelData);
        // The level and iteration each pass of the liquid's pressure solve
        // works on (fluid.wgsl's Level): a slot per level and iteration.
        const fluidLevelSlots = (MG_LEVELS + 1) * FLUID_CYCLES;
        this.fluidLevelBuf = device.createBuffer({
            label: "fluidLevels", size: SLOT_BYTES * fluidLevelSlots, usage: U.UNIFORM | U.COPY_DST,
        });
        const fluidLevelData = new Uint32Array(SLOT_BYTES / 4 * fluidLevelSlots);
        for (let cycle = 0; cycle < FLUID_CYCLES; cycle++) {
            for (let l = 0; l <= MG_LEVELS; l++) {
                fluidLevelData.set([l, cycle], SLOT_BYTES / 4 * (cycle * (MG_LEVELS + 1) + l));
            }
        }
        device.queue.writeBuffer(this.fluidLevelBuf, 0, fluidLevelData);
        // The air's projection's (air.wgsl): a slot per level and V-cycle,
        // (level, cycle).
        this.airLevelBuf = device.createBuffer({
            label: "airLevels", size: SLOT_BYTES * AIR_LEVELS * AIR_CYCLES, usage: U.UNIFORM | U.COPY_DST,
        });
        const airLevelData = new Uint32Array(SLOT_BYTES / 4 * AIR_LEVELS * AIR_CYCLES);
        for (let cycle = 0; cycle < AIR_CYCLES; cycle++) {
            for (let l = 0; l < AIR_LEVELS; l++) {
                airLevelData.set([l, cycle], SLOT_BYTES / 4 * (cycle * AIR_LEVELS + l));
            }
        }
        device.queue.writeBuffer(this.airLevelBuf, 0, airLevelData);
        // The air's conjugate gradients (air.wgsl): per air cell the
        // potential (warm from last step's), residual, direction and what the
        // operator makes of it; then the sums, four to a vec4, four and a
        // partial per 8 x 8 workgroup.
        const airGroups = Math.ceil(this.airWidth / 8) * Math.ceil(this.airHeight / 8);
        this.airCg = device.createBuffer({
            label: "airCg",
            size: (this.airWidth * this.airHeight + Math.ceil((4 + airGroups) / 4)) * 16,
            usage: U.STORAGE | U.COPY_SRC | U.COPY_DST,
        });
        // What other passes put into the air (airsource.wgsl): SOURCE_WORDS
        // (16) words an air cell, exact sums of two words each.
        this.airSource = device.createBuffer({
            label: "airSource", size: this.airWidth * this.airHeight * 64, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST,
        });
        // First non-gas row per column (sky.wgsl): what is under open sky;
        // then each cell's piece, a byte; what each piece is joined to
        // (common.wgsl, ROOM_TILE); how far down the wind gets in each
        // column, with as many words again for working it out (ventilationAt
        // in common.wgsl); and the air's pressure zones' sums, two slots of
        // ZONE_WORDS per piece (zoneSlotAt); then the wet rooms' bytes,
        // links and two slots of WET_WORDS per piece (wetSlotAt). Read back
        // by the test harness.
        this.roomTiles = [Math.ceil(width / ROOM_TILE), Math.ceil(height / ROOM_TILE)];
        const pieces = this.roomTiles[0] * this.roomTiles[1] * ROOM_PIECES;
        const roomWords = pieces / ROOM_PIECES * ROOM_TILE * ROOM_TILE / 4 + pieces;
        this.sky = device.createBuffer({
            label: "sky",
            size: (width + roomWords + 2 * width + 2 * ZONE_WORDS * pieces + roomWords + 2 * WET_WORDS * pieces) * 4,
            usage: U.STORAGE | U.COPY_SRC | U.COPY_DST,
        });
        // Pile load per cell (update.wgsl), read from one and written to the
        // other, alongside the cells: loads[cur] goes with cells[cur].
        this.loads = [0, 1].map((i) => device.createBuffer({
            label: `load${i}`,
            size: width * height * 4,
            usage: U.STORAGE | U.COPY_SRC | U.COPY_DST,
        }));
        // Electricity (electric.wgsl): the potential and current density per
        // cell, and the stats and tile flags. The passes only run while there
        // may be a battery in the world: from when one is painted until a
        // readback of the battery count, a frame or two late, finds none.
        // The passes' working words (ELECTRIC_WORK a cell: what assemble
        // makes of each cell, the CG's vectors, the current's effects, which
        // way its links to electrodes conduct) go in a buffer of their own.
        this.volts = storage("volts", cellCount * 4);
        this.electricWork = storage("electricWork", cellCount * ELECTRIC_WORK * 4);
        this.flow = storage("flow", cellCount * FLOW_BYTES);
        this.electricTiles = [Math.ceil(width / ELECTRIC_BLOCK), Math.ceil(height / ELECTRIC_BLOCK)];
        const blockCount = this.electricTiles[0] * this.electricTiles[1];
        this.electric = storage("electric", (ELECTRIC_STATS + ELECTRIC_SCALARS + (ELECTRIC_TILE_WORDS + ELECTRIC_SLOTS) * blockCount) * 4);
        this.coarse = storage("coarse", blockCount * ELECTRIC_SLOTS * ELECTRIC_COARSE_WORDS * 4);
        this.electricLive = false;
        // The CG's residual as of the last report (electric.wgsl's
        // STAT_RESIDUAL), and steps left of more iterations since painting.
        this.electricResidual = 0;
        this.electricFarSteps = ELECTRIC_FAR_STEPS;
        // The stress solve (stress.wgsl): STRESS_NODE_WORDS vec4s a node at
        // the cells' corners, its displacement (fixed point), and its working
        // words (stats, the CG's state, six partial sums, a word and a place
        // on the list a tile of 8 x 8 nodes, two words a cell).
        this.stressTiles = [Math.ceil((width + 1) / 8), Math.ceil((height + 1) / 8)];
        const stressTileCount = this.stressTiles[0] * this.stressTiles[1];
        this.stressNodes = storage("stressNodes", (width + 1) * (height + 1) * STRESS_NODE_WORDS * 16);
        this.stressDisp = storage("stressDisp", (width + 1) * (height + 1) * 16);
        this.stressWork = storage("stressWork", (STRESS_STATS + STRESS_SCALARS + 8 * stressTileCount + 2 * cellCount) * 4);
        // Whether there is structure, as of the last report.
        this.stressLive = true;
        this.stressFar = true;
        this.params = device.createBuffer({ label: "params", size: SLOT_BYTES * MAX_SLOTS, usage: U.UNIFORM | U.COPY_DST });
        this.paramData = new ArrayBuffer(SLOT_BYTES * MAX_SLOTS);
        this.brushBuf = device.createBuffer({ label: "brush", size: BRUSH_BYTES, usage: U.UNIFORM | U.COPY_DST });
        this.brushData = new ArrayBuffer(BRUSH_BYTES);
        // A flood fill's cells, a bit each (fill, brush.wgsl's SHAPE_MASK).
        this.fillMask = device.createBuffer({ label: "fillMask", size: Math.ceil(width * height / 32) * 4, usage: U.STORAGE | U.COPY_DST });
        this.fillPending = false;
        this.destroyed = false;
        this.viewBuf = device.createBuffer({ label: "view", size: 80, usage: U.UNIFORM | U.COPY_DST });
        this.viewData = new ArrayBuffer(80);
        // People (people.wgsl): their slots; the requests to place and erase
        // them, queued (addPerson, erasePeople) until the next frame; and
        // those the last report found, for the hover readout (personAt).
        this.people = device.createBuffer({ label: "people", size: MAX_PEOPLE * PERSON_BYTES, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
        // Where people's bodies are, a word a cell (common.wgsl's
        // occupantWord): the people pass stamps them as it ends each step,
        // on the mask cleared before it, for the next step's passes.
        this.occupants = device.createBuffer({ label: "occupants", size: width * height * 4, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
        this.occupantsStamped = false;
        this.personRequests = device.createBuffer({
            label: "personRequests", size: 16 + MAX_PERSON_REQUESTS * 32, usage: U.STORAGE | U.COPY_DST,
        });
        this.requests = [];
        this.peopleSeen = [];
        this.peopleLive = false;
        this.glowBuf = device.createBuffer({ label: "glow", size: 16, usage: U.UNIFORM | U.COPY_DST });
        // The hovered cell, its air cell, its potential and current density,
        // its gas, and its stress's failure index (stress.wgsl).
        this.readBuf = device.createBuffer({
            label: "readback", size: CELL_BYTES + AIR_BYTES + 4 + FLOW_BYTES + GAS_BYTES + 4, usage: U.MAP_READ | U.COPY_DST,
        });
        device.queue.writeBuffer(this.glowBuf, 0, new Uint32Array([width, height, 0, 0]));

        this.glowSize = [Math.ceil(width / 2), Math.ceil(height / 2)];
        this.glowTex = [0, 1].map((i) => device.createTexture({
            label: `glow${i}`,
            size: this.glowSize,
            format: "rgba16float",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        }));
        const glowViews = this.glowTex.map((t) => t.createView());
        // Per cell, what the renderer draws shiny ones from (glow.wgsl's shine).
        this.shineTex = device.createTexture({
            label: "shine",
            size: [width, height],
            format: "rgba16float",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        const shineView = this.shineTex.createView();

        // Light (light.wgsl), per light cell: what it gives off and lets
        // through, the sunlight reaching it, the light reaching it, and its
        // radiant heat; and two textures the cascades take turns in (cascade
        // k in lightCascades[k & 1]), each big enough for any of them:
        // cascade 0 keeps cascade 1's eight directions for every light cell,
        // in blocks four across and two down.
        const [lw, lh] = this.lightSize = [Math.ceil(width / LIGHT_CELL), Math.ceil(height / LIGHT_CELL)];
        this.cascadeCount = Math.max(MIN_CASCADES, Math.ceil(Math.log2(CASCADE_REACH * Math.max(lw, lh) + 0.5)));
        let cascadeWidth = 4 * lw;
        for (let k = 1; k < this.cascadeCount; k++) {
            cascadeWidth = Math.max(cascadeWidth, directions(k) * probes(k, lw, lh)[0]);
        }
        const T = GPUTextureUsage;
        const texture = (label, size, format, usage = 0) => device.createTexture({
            label, size, format, usage: T.STORAGE_BINDING | T.TEXTURE_BINDING | T.COPY_SRC | usage,
        });
        this.lightScene = texture("lightScene", this.lightSize, "rgba32uint");
        this.sunTex = texture("sun", this.lightSize, "r32float", T.COPY_DST);
        this.lightTex = texture("light", this.lightSize, "rgba16float", T.COPY_DST);
        this.heatTex = texture("heat", this.lightSize, "r32float", T.COPY_DST);
        // Radiant heat (radiate.wgsl): per cell what it took in and its
        // emitting area; the takings in fixed point (what each cell takes
        // in, then has reach it, then reflects out of each face), this
        // time's and the last time's; and what
        // leaves the world's edges (per direction, its sides' rows, then its
        // top's and bottom's columns), the last time's coming in at the
        // sides. A wavefront holds the cells along the shorter side
        // (Simulation.check).
        this.radiantTex = texture("radiant", [width, height], "rgba32float", T.COPY_DST);
        this.radiantTaken = [0, 1].map((k) => device.createBuffer({
            label: `radiantTaken${k}`, size: 7 * width * height * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        }));
        this.radiantEdges = [0, 1].map((k) => device.createBuffer({
            label: `radiantEdges${k}`, size: RADIANT_DIRS * (width + height + 2) * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        }));
        this.radiantNow = 0;
        // Heat's clock per light cell (clock.wgsl), and its first passes':
        // each light cell's hottest matter, and the most along its row.
        this.clockTex = texture("clock", this.lightSize, "r32float", T.COPY_DST);
        this.clockHot = device.createBuffer({ label: "clockHot", size: lw * lh * 4, usage: GPUBufferUsage.STORAGE });
        // Heat put into cells as energy, J a cell, an exact sum in two words
        // (heatin.wgsl).
        this.heatIn = device.createBuffer({
            label: "heatIn", size: width * height * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        this.clockAlong = device.createBuffer({ label: "clockAlong", size: lw * lh * 4, usage: GPUBufferUsage.STORAGE });
        // How real-time the current's heat made each light cell last step,
        // for heat's clock (electric.wgsl, clock.wgsl's source).
        this.clockHeated = device.createBuffer({
            label: "clockHeated", size: lw * lh * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        // Conduction across stiff faces (heat.wgsl): per cell, its faces' J/K
        // a step and its heat capacity, its phase points, and the stage
        // temperatures, two in turn; per tile of 8 x 8 cells whether it has
        // a stiff face; the stiffest face found (read back with the report,
        // cleared before each frame); each stage's weights.
        const heatBuffer = (label, size, usage = 0) => device.createBuffer({ label, size, usage: GPUBufferUsage.STORAGE | usage });
        this.heatFaces = heatBuffer("heatFaces", cellCount * 16);
        this.heatPhases = heatBuffer("heatPhases", cellCount * 8);
        this.heatStage = [0, 1].map((k) => heatBuffer(`heatStage${k}`, cellCount * 8));
        this.heatTileBuf = heatBuffer("heatTiles", Math.ceil(width / 8) * Math.ceil(height / 8) * 4);
        this.heatStiffBuf = heatBuffer("heatStiffness", 4, GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
        this.heatWeights = device.createBuffer({
            label: "heatWeights", size: SLOT_BYTES * HEAT_MAX_STAGES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.heatStages = 0;     // stages this frame (heatStageCount)
        this.heatStiffness = 0;  // the stiffest face the last report found
        this.heatLooked = false; // whether the steps since the stiffest face was last read looked for stiff faces
        // Mirrors: the light reaching each light cell from each quarter, and
        // what its shiny cells show as a mirror each way.
        this.quartersTex = texture("quarters", this.lightSize, "rgba32uint", T.COPY_DST);
        this.mirrorTex = texture("mirror", this.lightSize, "rgba32uint");
        // The sunlight mirrors throw on each light cell (light.wgsl's bounce).
        this.bounced = device.createBuffer({ label: "bounced", size: lw * lh * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        this.lightCascades = [0, 1].map((i) => texture(`cascade${i}`, [cascadeWidth, 2 * lh], "rgba16float"));
        // Gravity (gravity.wgsl): the field per light cell, and the pyramid
        // of block masses it is worked out from.
        this.gravityTex = texture("gravity", this.lightSize, "rgba32float", T.COPY_DST);
        // The pyramid's blocks, then per light cell its down word as last
        // worked out (gravity.wgsl's downBase).
        const lightCells = Math.ceil(width / GRAVITY_BLOCK) * Math.ceil(height / GRAVITY_BLOCK);
        this.gravityMasses = device.createBuffer({
            label: "gravityMasses", size: (gravityBlocks(width, height) + lightCells) * 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        this.blackHolePull = DEFAULT_BLACK_HOLE_PULL;
        this.gravityStale = 1;
        // Each cascade's level, and how many there are, a slot each.
        this.cascadeBuf = device.createBuffer({ label: "cascades", size: SLOT_BYTES * MAX_CASCADES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const cascadeData = new Uint32Array(SLOT_BYTES / 4 * MAX_CASCADES);
        for (let k = 0; k < MAX_CASCADES; k++) {
            cascadeData[SLOT_BYTES / 4 * k] = k;
            cascadeData[SLOT_BYTES / 4 * k + 1] = this.cascadeCount;
        }
        device.queue.writeBuffer(this.cascadeBuf, 0, cascadeData);
        this.lightStale = 2;  // times the light is to be worked out before the next step (see clear)
        this.lightRefreshes = 0;

        const bind = (layout, entries) => device.createBindGroup({
            layout,
            entries: Object.entries(entries).map(([binding, resource]) => ({ binding: Number(binding), resource })),
        });
        const slot = { buffer: this.params, size: PARAMS_BYTES };
        const elems = { buffer: pipes.elemBuf };
        const cell = (i) => ({ buffer: this.cells[i] });
        const air = (i) => ({ buffer: this.air[i] });
        const auto = (pipeline) => pipeline.getBindGroupLayout(0);

        // Passes that touch cells have a bind group per cell buffer; `cur` picks one.
        const field = { buffer: this.field };
        const sky = { buffer: this.sky };
        const load = (i) => ({ buffer: this.loads[i] });
        const particles = (i) => ({ buffer: this.particles[i] });
        const bodies = { buffer: this.bodies };
        const lightView = this.lightTex.createView();
        const heatView = this.heatTex.createView();
        const gravityView = this.gravityTex.createView();
        const heatClock = this.clockTex.createView();
        const radiantView = this.radiantTex.createView();
        this.updateBG = [0, 1].map((i) => bind(pipes.updateLayout, {
            0: slot, 1: elems, 2: cell(i), 3: cell(1 - i), 4: air(0), 5: field, 6: sky, 7: load(i), 8: load(1 - i),
            9: bodies, 10: heatView, 11: lightView, 12: gravityView, 13: heatClock, 14: { buffer: this.heatIn },
            15: { buffer: this.airSource }, 16: radiantView,
        }));
        // Radiant heat, by cell buffer (cur) and by which takings and edges
        // are this time's (radiantNow).
        this.radiateBG = [0, 1].map((i) => [0, 1].map((k) => bind(pipes.radiateLayout, {
            0: slot, 1: elems, 2: cell(i), 3: { buffer: this.radiantTaken[k] }, 4: { buffer: this.radiantTaken[1 - k] },
            5: { buffer: this.radiantEdges[k] }, 6: { buffer: this.radiantEdges[1 - k] }, 7: radiantView,
        })));
        this.gravityBG = [0, 1].map((i) => bind(pipes.gravityLayout, {
            0: slot, 1: elems, 2: cell(i), 3: { buffer: this.gravityMasses }, 4: gravityView,
        }));
        // Light: cascade k writes lightCascades[k & 1] and reads the one above
        // it from the other; gather reads cascade 0.
        const ll = pipes.lightLayouts;
        const sceneView = this.lightScene.createView();
        const sunView = this.sunTex.createView();
        const cascadeViews = this.lightCascades.map((t) => t.createView());
        const cascadeSlot = { buffer: this.cascadeBuf, size: 16 };
        const quartersView = this.quartersTex.createView();
        const mirrorView = this.mirrorTex.createView();
        const occupants = { buffer: this.occupants };
        this.buildBG = [0, 1].map((i) => bind(ll.build, {
            0: slot, 1: elems, 2: cell(i), 3: sceneView, 5: lightView, 17: quartersView, 18: mirrorView, 22: occupants,
        }));
        const cascadeEntries = (i) => ({
            0: slot, 4: sceneView, 5: lightView, 8: cascadeViews[1 - i], 9: cascadeViews[i], 11: cascadeSlot,
            19: mirrorView,
        });
        this.cascadeBG = [0, 1].map((i) => bind(ll.cascade, cascadeEntries(i)));
        this.sunBG = bind(ll.sun, { 0: slot, 4: sceneView, 6: sunView });
        const bounced = { buffer: this.bounced };
        this.bounceBG = [0, 1].map((i) => bind(ll.bounce, { 0: slot, 1: elems, 2: cell(i), 4: sceneView, 7: sunView, 21: bounced }));
        this.gatherBG = bind(ll.gather, {
            0: slot, 4: sceneView, 7: sunView, 8: cascadeViews[0], 11: cascadeSlot,
            12: lightView, 13: heatView, 20: quartersView, 21: bounced,
        });
        this.clockBG = [0, 1].map((i) => bind(pipes.clockLayout, {
            0: slot, 1: elems, 2: cell(i), 3: { buffer: this.clockHot }, 4: { buffer: this.clockAlong }, 5: heatClock,
            6: { buffer: this.clockHeated },
        }));
        // Heat's stages: per cell buffer, and per stage before and after in
        // turn (heatStage[k] before, the other after).
        this.heatBG = [0, 1].map((i) => [0, 1].map((k) => bind(pipes.heatLayout, {
            0: slot, 1: elems, 2: cell(i), 3: air(0), 4: heatClock, 5: { buffer: this.heatFaces },
            6: { buffer: this.heatPhases }, 7: { buffer: this.heatStage[k] }, 8: { buffer: this.heatStage[1 - k] },
            9: { buffer: this.heatTileBuf }, 10: { buffer: this.heatWeights, size: HEAT_STAGE_BYTES },
            11: { buffer: this.heatStiffBuf },
        })));
        // The bodies set where the particles of liquid they push aside go
        // (the liquid's counters, fluid.wgsl's G_DEST).
        this.bodiesBG = [0, 1].map((i) => bind(pipes.bodiesLayout, {
            0: slot, 1: elems, 2: cell(i), 4: bodies, 5: { buffer: this.counters },
            7: { buffer: this.bodyMotion }, 8: { buffer: this.bodyShapes }, 9: gravityView, 10: { buffer: this.airSource },
            11: air(0),
        }));
        this.skyBG = [0, 1].map((i) => bind(pipes.skyLayout, { 0: slot, 1: elems, 2: cell(i), 3: sky, 4: field }));
        this.fluidBG = [0, 1].map((i) => bind(pipes.fluidLayout, {
            0: slot, 1: elems, 2: cell(i), 3: air(0), 4: particles(0), 5: particles(1),
            6: { buffer: this.counters }, 7: field, 8: { buffer: this.blocks }, 9: { buffer: this.fluidLevelBuf, size: 16 },
            10: gravityView, 11: { buffer: this.airSource }, 12: { buffer: this.bodyMotion }, 13: occupants,
            14: { buffer: this.bodies }, 15: { buffer: this.fluidCg }, 16: { buffer: this.bodyPushes },
        }));
        this.blockBG = [0, 1].map((i) => bind(pipes.blockLayout, {
            0: slot, 1: elems, 2: cell(i), 3: air(0), 4: gravityView, 5: { buffer: this.airSource }, 6: heatClock,
            7: occupants,
        }));
        this.electricBG = [0, 1].map((i) => bind(pipes.electricLayout, {
            0: slot, 1: elems, 2: cell(i), 3: { buffer: this.volts }, 4: { buffer: this.flow }, 5: { buffer: this.electric },
            6: { buffer: this.coarse }, 7: { buffer: this.electricWork }, 9: heatClock,
            10: { buffer: this.heatIn }, 11: { buffer: this.clockHeated },
        }));
        // The stress solve, after the update pass (cur flipped): its cells
        // and the loads it just passed on (stress.wgsl).
        this.stressBG = [0, 1].map((i) => bind(pipes.stressLayout, {
            0: slot, 1: elems, 2: cell(i), 3: air(0), 4: field, 5: load(i), 6: bodies,
            7: { buffer: this.stressNodes }, 8: { buffer: this.stressWork }, 9: gravityView, 10: { buffer: this.stressDisp },
        }));
        this.airBG = [0, 1].map((i) => bind(pipes.airLayout, {
            0: slot, 1: elems, 2: cell(i), 3: air(0), 4: air(1), 5: { buffer: this.airProjection }, 7: gravityView, 8: sky,
            9: { buffer: this.blast }, 10: { buffer: this.airLevelBuf, size: 16 }, 12: { buffer: this.airFastest },
            13: { buffer: this.airCg }, 14: { buffer: this.airSource }, 15: field,
        }));
        this.carryBG = [0, 1].map((i) => bind(pipes.carryLayout, {
            0: slot, 1: elems, 2: cell(i), 3: air(0), 4: { buffer: this.airFastest }, 5: { buffer: this.carryScratch },
            6: { buffer: this.levelBuf, size: 16 }, 7: { buffer: this.airSource }, 8: { buffer: this.airCg }, 9: field,
            10: gravityView, 11: sky,
        }));
        this.windBG = bind(auto(pipes.wind), { 3: air(0), 11: { buffer: this.brushBuf }, 14: { buffer: this.airSource } });
        this.brushBG = [0, 1].map((i) => bind(auto(pipes.brush), {
            0: { buffer: this.brushBuf }, 1: elems, 2: cell(i), 3: { buffer: this.fillMask },
            4: { buffer: this.airSource },
        }));
        this.glowBG = [0, 1].map((i) => bind(auto(pipes.glow), {
            0: { buffer: this.glowBuf }, 1: elems, 2: cell(i), 3: glowViews[0],
        }));
        this.shineBG = [0, 1].map((i) => bind(auto(pipes.shine), {
            0: { buffer: this.glowBuf }, 1: elems, 2: cell(i), 4: lightView, 5: shineView, 6: sunView,
        }));
        this.blurHBG = bind(auto(pipes.blurH), { 0: glowViews[0], 1: glowViews[1] });
        this.blurVBG = bind(auto(pipes.blurV), { 0: glowViews[1], 1: glowViews[0] });
        this.renderBG = [0, 1].map((i) => bind(auto(pipes.render), {
            0: { buffer: this.viewBuf }, 1: elems, 2: cell(i), 3: glowViews[0], 4: pipes.sampler, 5: air(0),
            6: { buffer: this.flow }, 7: lightView, 8: sunView, 9: gravityView, 10: shineView,
        }));
        const people = { buffer: this.people };
        this.peopleBG = [0, 1].map((i) => bind(pipes.peopleLayout, {
            0: slot, 1: elems, 2: cell(i), 3: air(0), 4: { buffer: this.flow }, 5: gravityView, 6: people,
            7: { buffer: this.personRequests }, 8: { buffer: this.airSource }, 9: radiantView, 10: occupants,
            11: heatClock,
        }));
        this.figuresBG = [0, 1].map((i) => bind(auto(pipes.figures), {
            0: { buffer: this.viewBuf }, 1: elems, 2: cell(i), 3: people, 4: lightView, 5: pipes.sampler,
        }));

        this.clear();
    }

    // A world of air, the open air's at each height: as cold as the outdoor
    // air there (outdoorT, which buoyancy goes by), as humid, and as much of
    // it as the open air's pressure there (outdoorAirGas; air.wgsl
    // openAirAt, a cell's centre). With lapse false, 20 °C all the way up
    // (a scene, which cools all it has built, its air too, by its height:
    // scenes.js loadScene). (At 20 °C all the way up, a cleared world 14 km
    // tall rose in a storm for some 300 steps.)
    clear(lapse = true) {
        const count = this.width * this.height;
        const data = new Uint32Array(cellsBufferBytes(count) / 4);
        for (let y = 0; y < this.height; y++) {
            const altitude = (this.height - y - 0.5) * this.metresPerCell;
            const temp = lapse ? outdoorT(altitude) : AMBIENT_T;
            const [moist, rest] = outdoorAirGas(altitude, temp);
            for (let x = 0; x < this.width; x++) {
                const i = y * this.width + x;
                data[i * CELL_BYTES / 4 + 1] = f32Bits(temp);
                const at = (count * CELL_BYTES + i * GAS_BYTES) / 4;
                data[at] = moist;
                data[at + 1] = rest;
            }
        }
        for (const buffer of this.cells) {
            this.device.queue.writeBuffer(buffer, 0, data);
        }
        // Still air, no pressure, no blast, nothing put into it; no load on
        // anything; no heat on its way into any cell, nor any current's heat
        // making the clock real-time; radiant heat none taken in or
        // reflected yet (and the world's sides seeing the outdoor air's
        // radiance at 20 °C come in until the first sweep says what goes
        // out of them, below).
        this.zero([...this.air, this.airProjection, this.airCg, this.blast, this.airSource, ...this.loads, this.heatIn, ...this.radiantTaken,
            this.clockHeated]);
        const ambientRadiance = ((AMBIENT_T + KELVIN) / 1000) ** 4;
        for (const buffer of this.radiantEdges) {
            this.device.queue.writeBuffer(buffer, 0, new Float32Array(buffer.size / 4).fill(ambientRadiance));
        }
        this.device.queue.writeTexture({ texture: this.radiantTex }, new Float32Array(4 * count), { bytesPerRow: 16 * this.width }, [this.width, this.height]);
        // No particles, no block
        // worked on, and nothing loose. No potential, no current, no battery,
        // and no metal floating (the electric working words are written
        // before they are read but for that, electric.wgsl's floatSlot).
        const electric = [this.volts, this.flow, this.electric, this.coarse, this.electricWork];
        this.zero([...this.particles, this.counters, this.field, this.blocks, ...electric, this.stressNodes, this.stressWork, this.stressDisp, this.occupants]);
        this.clearBodies();
        // Reports still on their way are of the world before (newEpoch): the
        // liquid's and the bodies' passes run until one of this world says
        // there is nothing for them, as for a new world. Conduction has no
        // stages until one finds a stiff face.
        this.newEpoch("fluid", "bodies", "electric", "people", "heat", "stress", "air");
        this.stressLive = true;
        this.stressFar = true;
        this.airCycles = AIR_CYCLES;
        this.fluidCycles = FLUID_CYCLES;
        this.wakeBlast();
        this.heatStiffness = 0;
        // (Nor the last world's stiffest face or most iterations, yet to be
        // read.)
        this.heatLooked = false;
        this.device.queue.writeBuffer(this.heatStiffBuf, 0, new Uint32Array(1));
        this.device.queue.writeBuffer(this.airFastest, AIR_MOST_CYCLES * 4, new Uint32Array(1));
        this.fluidStats = { active: 1, viscous: 1 };
        this.bodiesLoose = true;
        this.bodiesMarked = true;
        this.bodiesFull = false;
        this.reportedAt = this.framesStepped;
        this.electricLive = false;
        // No light, and none of the last world's radiant heat, until the light
        // is worked out again before the next step: twice, as each time
        // takes what is reflected from the time before.
        const [lw, lh] = this.lightSize;
        for (const [texture, bytes] of [[this.lightTex, 8], [this.heatTex, 4], [this.quartersTex, 16]]) {
            this.device.queue.writeTexture({ texture }, new Uint8Array(lw * lh * bytes), { bytesPerRow: lw * bytes }, [lw, lh]);
        }
        this.zero([this.bounced]);
        // Heat on the day's clock everywhere until the light finds what is hot.
        this.device.queue.writeTexture({ texture: this.clockTex }, new Float32Array(lw * lh), { bytesPerRow: lw * 4 }, [lw, lh]);
        this.lightStale = 2;
        // No pull from the world's masses until it is worked out, before the
        // next step: down the screen (pull.wgsl's down word, 2).
        const field = new Float32Array(lw * lh * 4);
        for (let k = 3; k < field.length; k += 4) {
            field[k] = 2;
        }
        this.device.queue.writeTexture({ texture: this.gravityTex }, field, { bytesPerRow: lw * 16 }, [lw, lh]);
        // and no down word kept from the last world's (gravity.wgsl's DOWN_STICK).
        this.device.queue.writeBuffer(this.gravityMasses, 0, new Uint8Array(this.gravityMasses.size));
        this.gravityStale = 1;
        // Nobody, and nobody on the way.
        this.device.queue.writeBuffer(this.people, 0, new Uint32Array(MAX_PEOPLE * PERSON_BYTES / 4));
        this.requests = [];
        this.peopleSeen = [];
        this.peopleLive = false;
    }

    // Drops what has been put into the air and is yet to be taken in
    // (airsource.wgsl): for a scene just built, the gas its painting pushed
    // aside, long gone into the open air in a world that had always stood so.
    dropAirSource() {
        this.zero([this.airSource]);
    }

    // Buffers to all zeros, on the GPU, in the queue's order. (Uploaded as
    // typed arrays of zeros their size, the particles' alone 70 MB at
    // 549 x 275, a scene took the most of its load in it.)
    zero(buffers) {
        const enc = this.device.createCommandEncoder();
        for (const buffer of buffers) {
            enc.clearBuffer(buffer);
        }
        this.device.queue.submit([enc.finish()]);
    }

    // Work has come for these subsystems from outside their passes: reports
    // asked for before now are of them as they no longer are (readReport).
    // The explosions' layer's next passes run (air.wgsl's blastQuiet), as
    // they must where its state came from elsewhere (a new world, a
    // snapshot): with nothing in it they leave it as it is, and the step
    // after skips them.
    wakeBlast() {
        this.device.queue.writeBuffer(this.airFastest, AIR_BLAST_LIVE * 4, new Uint32Array([1, 1]));
    }

    newEpoch(...subsystems) {
        for (const s of subsystems) {
            this.epochs[s]++;
        }
    }

    // How hard a black hole pulls at its edge, in Earth's g (gravity.wgsl).
    setBlackHolePull(g) {
        this.blackHolePull = g;
        this.gravityStale = 1;
    }

    // Whether the Earth pulls, straight down the screen (Params.earthG). Off,
    // only the world's own masses do (gravity.wgsl): out in space, with a
    // black hole the one pull worth the name. Down goes by both, so the field
    // is worked out again.
    setEarthGravity(on) {
        this.earthGravity = on;
        this.gravityStale = 1;
    }

    // Sets the time of day (a share of the day since midnight) and the
    // clock's speed (CLOCK); either left out stays as it is.
    setClock(day = this.clock.day, speed = this.clock.speed) {
        this.clock = { day: ((day % 1) + 1) % 1, speed };
        this.lightStale = Math.max(this.lightStale, 1);
    }

    // Share of a day that passes per step.
    dayRate() {
        return DAY_SPEEDS[this.clock.speed] / DAY_STEPS;
    }

    // Seconds a step stands for: of motion, and on the day's clock
    // (js/shared.js).
    stepSeconds() {
        return stepSeconds(this.metresPerCell);
    }

    dayStepSeconds() {
        return dayStepSeconds(this.dayRate(), this.stepSeconds());
    }

    // Starts the random choices the steps make (the GPU's seed for each
    // pass, the block passes' phase, the brush's spray, a person's looks)
    // over from n: two runs from the same snapshot and seed make the same
    // ones. A world starts from a random seed.
    seed(n) {
        this.rngState = n >>> 0;
    }

    // A random u32 (mulberry32, Tommy Ettinger's, public domain).
    random() {
        this.rngState = (this.rngState + 0x6d2b79f5) >>> 0;
        let t = this.rngState;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return (t ^ (t >>> 14)) >>> 0;
    }

    // The simulation's state on the GPU, by the names passes.js's table of
    // passes uses: a buffer or texture, or a pair of them (the cells, the
    // pile loads, the particles: `cur` says which is the latest).
    resources() {
        return {
            cells: this.cells, loads: this.loads, air: this.air[0], airScratch: this.air[1], airCg: this.airCg, airSource: this.airSource,
            airProjection: this.airProjection, blast: this.blast, airFastest: this.airFastest,
            carryScratch: this.carryScratch, sky: this.sky, particles: this.particles, counters: this.counters, field: this.field, blocks: this.blocks,
            fluidCg: this.fluidCg, bodyPushes: this.bodyPushes,
            levels: this.levelBuf, fluidLevels: this.fluidLevelBuf, airLevels: this.airLevelBuf, bodies: this.bodies, bodyMotion: this.bodyMotion, bodyShapes: this.bodyShapes,
            volts: this.volts, flow: this.flow, electric: this.electric, coarse: this.coarse,
            stressNodes: this.stressNodes, stressWork: this.stressWork, stressDisp: this.stressDisp,
            electricWork: this.electricWork, people: this.people, occupants: this.occupants, personRequests: this.personRequests,
            lightScene: this.lightScene, sun: this.sunTex, light: this.lightTex, heat: this.heatTex,
            quarters: this.quartersTex, mirror: this.mirrorTex, cascades: this.lightCascades,
            bounced: this.bounced, cascadeSlots: this.cascadeBuf, clock: this.clockTex, clockHot: this.clockHot, clockAlong: this.clockAlong, clockHeated: this.clockHeated,
            heatIn: this.heatIn, radiant: this.radiantTex, radiantTaken: this.radiantTaken, radiantEdges: this.radiantEdges,
            heatFaces: this.heatFaces, heatPhases: this.heatPhases, heatStage0: this.heatStage[0],
            heatStage1: this.heatStage[1], heatTiles: this.heatTileBuf, heatStiffness: this.heatStiffBuf,
            heatWeights: this.heatWeights,
            gravity: this.gravityTex,
            gravityMasses: this.gravityMasses, glow: this.glowTex, shine: this.shineTex, glowParams: this.glowBuf,
            elements: this.pipes.elemBuf, params: this.params, brush: this.brushBuf, fillMask: this.fillMask,
            view: this.viewBuf,
        };
    }

    // What a snapshot keeps (snapshot.js): the resources one step leaves for
    // the next (passes.js's persistentResources), by name; a pair as name0
    // and name1.
    snapshotParts() {
        const resources = this.resources();
        const parts = { buffers: {}, textures: {} };
        for (const name of persistentResources(new Set(Object.keys(resources)))) {
            const r = resources[name];
            for (const [key, one] of Array.isArray(r) ? r.map((x, i) => [`${name}${i}`, x]) : [[name, r]]) {
                parts[one instanceof GPUTexture ? "textures" : "buffers"][key] = one;
            }
        }
        return parts;
    }

    // The rest of what a snapshot keeps: which buffers of each pair are the
    // latest, the step and pass counters, the random state, the settings,
    // when the light and gravity are next worked out, and which passes run
    // (as last read back).
    snapshotState() {
        return {
            cur: this.cur, frameNo: this.frameNo, passNo: this.passNo, phaseX: this.phaseX,
            shiftedFirst: this.shiftedFirst, rng: this.rngState,
            clock: { ...this.clock }, metresPerCell: this.metresPerCell, blackHolePull: this.blackHolePull,
            earthGravity: this.earthGravity,
            lightStale: this.lightStale, lightRefreshes: this.lightRefreshes, gravityStale: this.gravityStale,
            fluidStats: { ...this.fluidStats }, bodiesLoose: this.bodiesLoose, bodiesMarked: this.bodiesMarked,
            bodiesFull: this.bodiesFull, electricLive: this.electricLive, peopleLive: this.peopleLive,
            heatStiffness: this.heatStiffness,
            // (How many CG iterations the current takes, and whether the
            // stress solve runs, go by these: left as the page had them, two
            // loads of one snapshot ran apart; REWRITE_PLAN.md D-1.)
            electricResidual: this.electricResidual, electricFarSteps: this.electricFarSteps, stressLive: this.stressLive,
            stressFar: this.stressFar, airCycles: this.airCycles, fluidCycles: this.fluidCycles,
            // Which of radiant heat's takings is this time's (two loads of
            // one snapshot parted at the next light when it was left as the
            // page had it; REWRITE_PLAN.md D-2).
            radiantNow: this.radiantNow,
        };
    }

    // Takes on the state snapshotState gave, once a snapshot's buffers are
    // in. Reports still on their way are of the world before, and are let
    // go (newEpoch), as are queued people; what the snapshot says the
    // passes have to do counts as a fresh report.
    restoreState(s) {
        Object.assign(this, {
            cur: s.cur, frameNo: s.frameNo, passNo: s.passNo, phaseX: s.phaseX, shiftedFirst: s.shiftedFirst,
            rngState: s.rng, clock: { ...s.clock }, metresPerCell: s.metresPerCell, blackHolePull: s.blackHolePull,
            earthGravity: s.earthGravity, lightStale: s.lightStale, lightRefreshes: s.lightRefreshes,
            gravityStale: s.gravityStale, fluidStats: { ...s.fluidStats }, bodiesLoose: s.bodiesLoose,
            bodiesMarked: s.bodiesMarked, bodiesFull: s.bodiesFull, electricLive: s.electricLive,
            peopleLive: s.peopleLive, heatStiffness: s.heatStiffness ?? 0,
            electricResidual: s.electricResidual ?? 0, electricFarSteps: s.electricFarSteps ?? ELECTRIC_FAR_STEPS,
            radiantNow: s.radiantNow ?? 0, airCycles: s.airCycles ?? AIR_CYCLES, fluidCycles: s.fluidCycles ?? FLUID_CYCLES,
        });
        this.newEpoch("fluid", "bodies", "electric", "people", "heat", "stress", "air");
        this.wakeBlast();
        // The occupants mask loaded may hold someone no longer there (erased
        // while paused): cleared before the next step, whoever is there
        // stamping it again.
        this.occupantsStamped = true;
        this.stressLive = s.stressLive ?? true;
        this.stressFar = s.stressFar ?? true;
        this.reportedAt = this.framesStepped;
        this.requests = [];
        this.peopleSeen = [];
    }

    // No bodies: the header and the cells' labels and claims cleared, all
    // slots free, the table and the pool empty.
    clearBodies() {
        // (Structure's words after the sums too: its marks and hashes.)
        this.zero([this.bodies, this.bodyMotion, this.bodyShapes]);
    }

    destroy() {
        const buffers = [
            ...this.cells, ...this.air, this.airProjection, this.blast, this.airFastest, this.carryScratch, ...this.loads,
            this.airCg, this.airSource, this.sky,
            ...this.particles, this.counters, this.field, this.blocks, this.fluidCg, this.bodyPushes, this.bodies, this.bodyMotion, this.bodyShapes,
            this.stressNodes, this.stressWork, this.stressDisp,
            this.levelBuf, this.fluidLevelBuf, this.airLevelBuf, ...this.reports.map((r) => r.buffer),
            this.volts, this.electricWork, this.flow, this.electric, this.coarse,
            this.params, this.brushBuf, this.viewBuf, this.glowBuf, this.readBuf, this.cascadeBuf,
            this.gravityMasses, this.bounced, this.people, this.occupants, this.personRequests, this.fillMask, this.clockHot, this.clockAlong, this.clockHeated, this.heatIn,
            this.heatFaces, this.heatPhases, ...this.heatStage, this.heatTileBuf, this.heatStiffBuf, this.heatWeights,
            ...this.radiantTaken, ...this.radiantEdges,
        ];
        this.destroyed = true;
        for (const b of buffers) {
            b.destroy();
        }
        const textures = [
            ...this.glowTex, this.shineTex, this.lightScene, this.sunTex, this.lightTex, this.heatTex,
            this.quartersTex, this.mirrorTex, ...this.lightCascades, this.gravityTex, this.clockTex, this.radiantTex,
        ];
        for (const t of textures) {
            t.destroy();
        }
        this.timer?.destroy();
    }

    // Mean GPU milliseconds per frame for each phase since the last call, or
    // null if the device can't time passes.
    takeTimings() {
        return this.timer?.take() ?? null;
    }

    // Applies one brush stroke segment (or a box, or a fill's mask, its box
    // from a to b) immediately. replace: over anything, or (the default)
    // only into empty space; or with `target`, only over that element.
    // temp: what it is painted at, °C, its element's own if left out.
    paint(brush) {
        // A held stroke paints a segment a frame (ui.js takeBrushes). Only
        // its first, or a brush after a gap (or several between two steps:
        // a scene, a paused world), re-arms what a change to the world calls
        // for below; the rest go by the reports, which take the air back to
        // all its iterations if they run out (F-1) and the stress solve far
        // off if it is. (Re-armed every frame, a held brush kept the air at
        // 64 iterations, the stress solve far off and gravity's field
        // recomputed for as long as it was held: test/heldbrush.mjs.)
        const fresh = this.paintedAt === null || this.frameNo === this.paintedAt || this.frameNo - this.paintedAt > STROKE_GAP;
        this.paintedAt = this.frameNo;
        // Of what a brush paints or erases, only black holes pull enough to
        // show at once (ordinary matter's pull, ~1e-10 g, is below the
        // Gravity view's floor; the field takes it in within GRAVITY_EVERY
        // steps): a stroke that may paint one, or paint over one, recomputes
        // the field (the wind, heating and the rest leave it as it is:
        // audit 06 G5).
        if (brush.mode === BRUSH.PAINT && fresh && (brush.elem === ID.BLACK_HOLE || brush.replace)) {
            this.gravityStale = 1;
        }
        const shape = brush.shape ?? SHAPE.CAPSULE;
        const pad = shape === SHAPE.RECT || shape === SHAPE.MASK ? 0 : brush.radius + 1;
        const x0 = Math.max(0, Math.floor(Math.min(brush.ax, brush.bx) - pad));
        const y0 = Math.max(0, Math.floor(Math.min(brush.ay, brush.by) - pad));
        const x1 = Math.min(this.width - 1, Math.ceil(Math.max(brush.ax, brush.bx) + pad));
        const y1 = Math.min(this.height - 1, Math.ceil(Math.max(brush.ay, brush.by) + pad));
        const affectsCells = x0 <= x1 && y0 <= y1;
        if (!affectsCells && brush.mode !== BRUSH.WIND) {
            return;
        }
        // A battery painted turns the current on, and liquid the liquid's
        // passes, until a report since says there is nothing for them.
        const painted = ELEMENTS[ID[ELEMENTS[brush.elem]?.paintAs] ?? brush.elem];
        if (brush.mode === BRUSH.PAINT && EMF_ELEMENTS.has(brush.elem)) {
            this.electricLive = true;
            this.newEpoch("electric");
        }
        // Anything painted may change a circuit: the next steps take more
        // iterations.
        if (brush.mode === BRUSH.PAINT && fresh) {
            this.electricFarSteps = ELECTRIC_FAR_STEPS;
        }
        if (brush.mode === BRUSH.PAINT && painted?.kind === KIND.LIQUID) {
            this.fluidStats = { active: 1, viscous: 1 };
            this.newEpoch("fluid");
        }
        // Anything painted may be or hold up structure: the stress solve
        // runs until a report since says there is none, and as if far off
        // where the stroke may make or take away structure (a structural
        // solid painted, or anything painted over what is there).
        if (brush.mode === BRUSH.PAINT && fresh) {
            this.stressLive = true;
            this.stressFar ||= brush.replace || (painted?.kind === KIND.SOLID && painted.breakTo != null);
            this.newEpoch("stress");
        }
        // Any brush may part a room or set its gas going (heat, wind): the
        // air's projection gets all its iterations until a report since;
        // and the liquid's all its (liquid poured, or a wall erased under
        // it).
        if (fresh) {
            this.airCycles = AIR_CYCLES;
            this.newEpoch("air");
            this.fluidCycles = FLUID_CYCLES;
            this.newEpoch("fluid");
        }

        const f = new Float32Array(this.brushData);
        const u = new Uint32Array(this.brushData);
        f[0] = brush.ax;
        f[1] = brush.ay;
        f[2] = brush.bx;
        f[3] = brush.by;
        u[4] = this.width;
        u[5] = this.height;
        u[6] = x0;
        u[7] = y0;
        f[8] = brush.radius;
        u[9] = brush.mode;
        // A preset (salt water) is painted as the element it stands for,
        // carrying its solute: the solute in elem's second byte, how much of
        // it in the third (moist, 1 saturated, the default). Soil goes in the
        // second byte the water it holds (moist, 0 dry to 1 soaked).
        const preset = ELEMENTS[brush.elem];
        u[10] = preset?.paintAs
            ? (ID[preset.paintAs] | (ID[preset.paintSolute] << 8) | (Math.round((brush.moist ?? 1) * 255) << 16)) >>> 0
            : brush.elem | (Math.round((brush.moist ?? 0) * 255) << 8);
        u[11] = brush.target != null ? OVER.ONLY : brush.replace ? OVER.ALL : OVER.EMPTY;
        u[12] = this.random();
        f[13] = brush.strength;
        f[14] = brush.angle ?? 0;
        u[15] = shape;
        f[16] = brush.temp ?? TEMP_OWN;
        u[17] = brush.target ?? 0;
        f[18] = this.metresPerCell;
        this.device.queue.writeBuffer(this.brushBuf, 0, this.brushData);

        const enc = this.device.createCommandEncoder();
        const pass = enc.beginComputePass();
        if (brush.mode === BRUSH.WIND) {
            pass.setPipeline(this.pipes.wind);
            pass.setBindGroup(0, this.windBG);
            pass.dispatchWorkgroups(Math.ceil(this.airWidth / 8), Math.ceil(this.airHeight / 8));
        }
        if (affectsCells) {
            pass.setPipeline(this.pipes.brush);
            pass.setBindGroup(0, this.brushBG[this.cur]);
            pass.dispatchWorkgroups(Math.ceil((x1 - x0 + 1) / 8), Math.ceil((y1 - y0 + 1) / 8));
        }
        pass.end();
        this.device.queue.submit([enc.finish()]);
    }

    // A flood fill from the cell at (ax, ay): everything joined to it side by
    // side that is the same element it is (up to FILL_MAX cells), painted
    // with the brush (its mode, element, temperature), every cell of it, over
    // whatever is there. The cells are read back for it first, so it lands a
    // frame or two later; one at a time.
    fill(brush) {
        const x = Math.floor(brush.ax);
        const y = Math.floor(brush.ay);
        const { width: w, height: h } = this;
        if (this.fillPending || x < 0 || y < 0 || x >= w || y >= h) {
            return;
        }
        this.fillPending = true;
        const n = w * h;
        const read = this.device.createBuffer({ label: "fillRead", size: n * CELL_BYTES, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const enc = this.device.createCommandEncoder();
        enc.copyBufferToBuffer(this.cells[this.cur], 0, read, 0, n * CELL_BYTES);
        this.device.queue.submit([enc.finish()]);
        read.mapAsync(GPUMapMode.READ).then(() => {
            const words = new Uint32Array(read.getMappedRange());
            const start = y * w + x;
            const type = cellTypeAt(words, start);
            const bits = new Uint32Array(Math.ceil(n / 32));
            const stack = new Int32Array(Math.min(n, FILL_MAX) + 4);
            let [x0, y0, x1, y1] = [x, y, x, y];
            let top = 0;
            let count = 0;
            stack[top++] = start;
            bits[start >>> 5] |= 1 << (start & 31);
            while (top > 0 && count < FILL_MAX) {
                const i = stack[--top];
                count++;
                const cx = i % w;
                const cy = (i - cx) / w;
                x0 = Math.min(x0, cx);
                x1 = Math.max(x1, cx);
                y0 = Math.min(y0, cy);
                y1 = Math.max(y1, cy);
                for (const [nx, ny] of [[cx - 1, cy], [cx + 1, cy], [cx, cy - 1], [cx, cy + 1]]) {
                    const j = ny * w + nx;
                    if (nx < 0 || ny < 0 || nx >= w || ny >= h || (bits[j >>> 5] & (1 << (j & 31))) || cellTypeAt(words, j) !== type) {
                        continue;
                    }
                    if (top < stack.length) {
                        bits[j >>> 5] |= 1 << (j & 31);
                        stack[top++] = j;
                    }
                }
            }
            read.unmap();
            read.destroy();
            if (!this.destroyed) {
                this.device.queue.writeBuffer(this.fillMask, 0, bits);
                this.paint({
                    ...brush, ax: x0, ay: y0, bx: x1 + 1, by: y1 + 1, radius: 0, shape: SHAPE.MASK,
                    strength: brush.mode === BRUSH.PAINT ? 1 : brush.strength, replace: true, target: null,
                });
            }
            this.fillPending = false;
        }, () => {
            read.destroy();
            this.fillPending = false;
        });
    }

    // Advances `steps` simulation steps, then renders into `target`. With
    // `timing`, the GPU time of each phase is measured (see takeTimings).
    frame({ steps, view, target, hover, timing = false }) {
        this.timer?.begin(timing);
        const enc = this.device.createCommandEncoder();
        this.encodeRequests(enc);
        if (steps > 0) {
            this.encodeStepping(enc, Math.min(steps, MAX_STEPS_PER_FRAME));
        } else if (this.lightStale || this.gravityStale) {
            // Paused, after a new scene, the clock was set or something was
            // painted: the light and gravity are worked out for their views
            // all the same.
            this.writeSlot(0, 0, 0);
            if (this.lightStale) {
                this.encodeLight(enc, [0]);
            }
            if (this.gravityStale) {
                this.encodeGravity(enc, [0]);
            }
            this.device.queue.writeBuffer(this.params, 0, this.paramData, 0, SLOT_BYTES);
        }
        this.encodeGlow(enc, view.mode);
        this.encodeRender(enc, view, target);
        const read = hover && !this.readPending ? hover : null;
        if (read) {
            const cellIndex = read.y * this.width + read.x;
            const airIndex = Math.floor(read.y / AIR_CELL) * this.airWidth + Math.floor(read.x / AIR_CELL);
            const at = CELL_BYTES + AIR_BYTES;
            enc.copyBufferToBuffer(this.cells[this.cur], cellIndex * CELL_BYTES, this.readBuf, 0, CELL_BYTES);
            enc.copyBufferToBuffer(this.air[0], airIndex * AIR_BYTES, this.readBuf, CELL_BYTES, AIR_BYTES);
            enc.copyBufferToBuffer(this.volts, cellIndex * 4, this.readBuf, at, 4);
            enc.copyBufferToBuffer(this.flow, cellIndex * FLOW_BYTES, this.readBuf, at + 4, FLOW_BYTES);
            const gasAt = this.width * this.height * CELL_BYTES + cellIndex * GAS_BYTES;
            enc.copyBufferToBuffer(this.cells[this.cur], gasAt, this.readBuf, at + 4 + FLOW_BYTES, GAS_BYTES);
            const failAt = (STRESS_STATS + STRESS_SCALARS + 8 * this.stressTiles[0] * this.stressTiles[1] + cellIndex) * 4;
            enc.copyBufferToBuffer(this.stressWork, failAt, this.readBuf, at + 4 + FLOW_BYTES + GAS_BYTES, 4);
        }
        // A report of what the passes that can be left out have to do, into
        // a staging buffer not already waiting to be read: every frame that
        // steps, and while there are people, every frame (their hover
        // readout).
        if (steps > 0) {
            this.framesStepped++;
        }
        const report = steps > 0 || this.peopleLive ? this.reports.find((r) => !r.pending) : null;
        if (report) {
            const b = report.buffer;
            enc.copyBufferToBuffer(this.blocks, 4 * this.blockCount * 4, b, REPORT_FLUID * 4, FLUID_STATS * 4);
            enc.copyBufferToBuffer(this.bodies, 2 * this.bodyTiles * 4, b, REPORT_BODIES * 4, BODY_STATS * 4);
            enc.copyBufferToBuffer(this.electric, 0, b, REPORT_ELECTRIC * 4, ELECTRIC_STATS * 4);
            enc.copyBufferToBuffer(this.stressWork, 0, b, REPORT_STRESS * 4, STRESS_STATS * 4);
            // The stiffest face and the most iterations the air took are
            // the most since they were last read: taken and cleared only by
            // a report on steps (one on none, a paused world's for its
            // people, would drop them), kept until then if every report was
            // busy.
            if (steps > 0) {
                enc.copyBufferToBuffer(this.heatStiffBuf, 0, b, REPORT_HEAT * 4, 4);
                enc.clearBuffer(this.heatStiffBuf, 0, 4);
                enc.copyBufferToBuffer(this.airFastest, AIR_MOST_CYCLES * 4, b, REPORT_AIR * 4, 4);
                enc.clearBuffer(this.airFastest, AIR_MOST_CYCLES * 4, 4);
                enc.clearBuffer(this.blocks, (4 * this.blockCount + FLUID_COUNT.MOST_CYCLES) * 4, 4);
            }
            enc.copyBufferToBuffer(this.people, 0, b, REPORT_PEOPLE * 4, MAX_PEOPLE * PERSON_BYTES);
        }
        this.timer?.resolve(enc);
        this.device.queue.submit([enc.finish()]);
        this.timer?.collect();
        if (read) {
            this.readCell(read, this.electricLive);
        }
        if (report) {
            this.readReport(report, steps > 0, steps > 0 && this.heatLooked);
            this.heatLooked &&= steps === 0;
        }
    }

    // How many stages conduction across stiff faces takes this frame
    // (heat.wgsl): as few as take the stiffest face the last report found,
    // none for none; their weights into their slots when that changes.
    heatStageCount() {
        const q = this.heatStiffness;
        let s = 0;
        if (q > HEAT_SHARE_MAX) {
            s = 2;
            while (s < HEAT_MAX_STAGES && HEAT_CHEBYSHEV[s].limit < q) {
                s++;
            }
        }
        if (s === this.heatStages) {
            return;
        }
        this.heatStages = s;
        if (s > 0) {
            const data = new ArrayBuffer(SLOT_BYTES * s);
            HEAT_CHEBYSHEV[s].stages.forEach((weights, j) => {
                new Float32Array(data, j * SLOT_BYTES, 3).set(weights);
                new Uint32Array(data, j * SLOT_BYTES + 12, 1)[0] = j === s - 1 ? 1 : 0;
            });
            this.device.queue.writeBuffer(this.heatWeights, 0, data);
        }
    }

    // Places a person with its feet at (x, y), in cells, lifted out of
    // anything solid there, in the next frame; seed, its looks (and name,
    // people.js), at random if left out. Up to MAX_PEOPLE at once: past that,
    // nobody new.
    addPerson(x, y, seed = this.random()) {
        this.queueRequest(REQUEST.PLACE, x, y, seed >>> 0);
    }

    // Takes away everyone whose body comes within radius cells of (x, y), or
    // of the stroke from there to (bx, by).
    erasePeople(x, y, radius, bx = x, by = y) {
        this.queueRequest(REQUEST.ERASE, x, y, f32Bits(radius), bx, by);
    }

    // Takes away everyone whose body is in the box from (x0, y0) to (x1, y1).
    erasePeopleBox(x0, y0, x1, y1) {
        this.queueRequest(REQUEST.ERASE_BOX, x0, y0, 0, x1, y1);
    }

    // A request: its kind, a point, the seed or the radius, and a second
    // point (people.wgsl's place).
    queueRequest(kind, x, y, w, bx = x, by = y) {
        if (this.requests.length < MAX_PERSON_REQUESTS) {
            this.requests.push([kind, f32Bits(x), f32Bits(y), w, f32Bits(bx), f32Bits(by), 0, 0]);
        }
    }

    // The requests queued since the last frame (people.wgsl's place), before
    // its steps, on params slot 0 (of which it needs only the world's size:
    // written here for a world that hasn't stepped yet; the steps' writes
    // come after, and win).
    encodeRequests(enc) {
        if (this.requests.length === 0) {
            return;
        }
        this.writeSlot(0, 0, 0);
        this.device.queue.writeBuffer(this.params, 0, this.paramData, 0, SLOT_BYTES);
        const data = new Uint32Array(4 + 8 * this.requests.length);
        data[0] = this.requests.length;
        this.requests.forEach((r, k) => data.set(r, 4 + 8 * k));
        if (this.requests.some(([kind]) => kind === REQUEST.PLACE)) {
            this.peopleLive = true;
            this.newEpoch("people");
        }
        this.device.queue.writeBuffer(this.personRequests, 0, data);
        this.requests = [];
        const pass = enc.beginComputePass({ label: "place" });
        pass.setPipeline(this.pipes.place);
        pass.setBindGroup(0, this.peopleBG[this.cur], [0]);
        pass.dispatchWorkgroups(1);
        pass.end();
    }

    // Takes in a report (see REPORTS) asked for this frame, once it is back:
    // of each subsystem whose epoch hasn't moved on since, what its passes
    // have to do. The liquid's, the bodies' and the current's parts only
    // tell anything of a frame that stepped (the passes ran up to them), the
    // stiffest face of conduction only of one that looked for it (looked:
    // heat.wgsl's prepare ran).
    // With no battery left the current goes off, and with it its potential
    // (a battery painted later would drive current through what was left
    // of it) and its current density, which people feel whether the passes
    // run or not.
    readReport(report, stepped, looked) {
        report.pending = true;
        const epochs = { ...this.epochs };
        const airCycles = this.airCycles;
        const fluidCycles = this.fluidCycles;
        const frame = this.framesStepped;
        report.landing = report.buffer.mapAsync(GPUMapMode.READ).then(() => {
            const range = report.buffer.getMappedRange();
            const u = new Uint32Array(range);
            const f = new Float32Array(range);
            const current = (s) => epochs[s] === this.epochs[s];
            if (stepped && current("fluid")) {
                this.fluidStats = {
                    active: u[REPORT_FLUID + FLUID_COUNT.ACTIVE] + u[REPORT_FLUID + FLUID_COUNT.PARTICLES],
                    viscous: u[REPORT_FLUID + FLUID_COUNT.VISCOUS],
                };
                // Likewise the liquid's iterations.
                const most = u[REPORT_FLUID + FLUID_COUNT.MOST_CYCLES];
                this.fluidCycles = most >= fluidCycles ? FLUID_CYCLES : Math.min(most + FLUID_CYCLES_SPARE, FLUID_CYCLES);
            }
            if (stepped && current("bodies")) {
                const stat = (k) => u[REPORT_BODIES + k];
                this.bodiesMarked = Boolean(stat(BODY_STAT.MARKED) || stat(BODY_STAT.MARKED_RUN));
                this.bodiesLoose = Boolean(this.bodiesMarked || stat(BODY_STAT.ALIVE));
                this.bodiesFull = Boolean(stat(BODY_STAT.COMPACT));
                this.reportedAt = Math.max(this.reportedAt, frame);
            }
            if (looked && current("heat")) {
                this.heatStiffness = f[REPORT_HEAT];
            }
            if (stepped && current("electric")) {
                this.electricResidual = f[REPORT_ELECTRIC + 2];
            }
            // A step that used all the iterations it was given may have
            // wanted more.
            if (stepped && current("air")) {
                const most = u[REPORT_AIR];
                this.airCycles = most >= airCycles ? AIR_CYCLES : Math.min(2 * most + AIR_CYCLES_SPARE, AIR_CYCLES);
            }
            if (stepped && current("stress")) {
                this.stressLive = u[REPORT_STRESS + STRESS_STAT.NODES] > 0;
                this.stressFar = f[REPORT_STRESS + STRESS_STAT.RESIDUAL] > STRESS_FAR
                    || f[REPORT_STRESS + STRESS_STAT.WORST] > STRESS_FAR_TILE;
            }
            if (stepped && current("electric") && this.electricLive && u[REPORT_ELECTRIC] === 0) {
                this.electricLive = false;
                for (const target of [this.volts, this.coarse, this.flow]) {
                    this.device.queue.writeBuffer(target, 0, new Uint32Array(target.size / 4));
                }
            }
            if (current("people")) {
                const seen = [];
                for (let k = 0; k < MAX_PEOPLE; k++) {
                    const at = REPORT_PEOPLE + k * PERSON_BYTES / 4;
                    if (u[at + 4] === PERSON_STATE.NONE) {
                        continue;
                    }
                    seen.push({
                        slot: k, x: f[at], y: f[at + 1], vx: f[at + 2], vy: f[at + 3], state: u[at + 4], flags: u[at + 5],
                        health: f[at + 7], breath: f[at + 8], seed: u[at + 9], cause: u[at + 15],
                    });
                }
                this.peopleSeen = seen;
                this.peopleLive = seen.length > 0;
            }
            report.buffer.unmap();
            report.pending = false;
        }, () => {
            report.pending = false;
        });
    }

    // Once every report asked for so far is taken in. The app takes each in
    // when it lands, a frame or two late as the GPU runs; the test harness
    // waits on this before each step, so its steps go by the same reports
    // however busy the GPU is (REWRITE_PLAN.md D-2: under load two seeded
    // runs parted at the second step, the first report in or not).
    async reportsTaken() {
        await Promise.all(this.reports.filter((r) => r.pending).map((r) => r.landing));
    }

    // The person whose body is at cell (x, y), as last read back, or null.
    personAt(x, y) {
        return this.peopleSeen.find((p) => Math.abs(x + 0.5 - p.x) <= 2.5 && y < p.y + 0.5 && y >= p.y - (p.state === PERSON_STATE.DEAD || p.state === PERSON_STATE.SLEEP ? 5.5 : 10.5)) ?? null;
    }

    beginPass(enc, phase) {
        return enc.beginComputePass({ label: phase, timestampWrites: this.timer?.writes(phase) });
    }

    writeSlot(slot, ox, oy) {
        const u = new Uint32Array(this.paramData, slot * SLOT_BYTES, PARAMS_BYTES / 4);
        u[0] = this.width;
        u[1] = this.height;
        u[2] = this.frameNo;
        u[3] = this.random();
        u[4] = ox;
        u[5] = oy;
        u[6] = f32Bits(this.supportMax);
        u[7] = f32Bits(this.metresPerCell);
        u[8] = f32Bits(this.clock.day);
        u[9] = f32Bits(this.dayRate());
        // Gravity's pulls μ (gravity.wgsl): a cell of mass m pulls G m / r²
        // at r (r in metres), which in cells per step² is G m Δt² / (Δx³ r²)
        // with r in cells: with m = ρ Δx³ (ρ in kg/m³, the element's rho),
        // G ρ Δt² per cell at a cell's distance. A black hole's cells each
        // pull k g / π, so a disc of them pulls k g at its edge.
        const dt = this.stepSeconds();
        u[10] = f32Bits(this.blackHolePull * ELEMENTS[ID.WATER].weight / Math.PI);
        u[11] = f32Bits(NEWTON_G * dt * dt);
        u[12] = f32Bits(this.earthGravity ? ELEMENTS[ID.WATER].weight : 0);
        const dayDt = this.dayStepSeconds();
        u[13] = f32Bits(dt);
        u[14] = f32Bits(dayDt);
        u[15] = f32Bits(Math.log2(dt / dayDt));
        u[16] = this.heatStages;
        u[17] = f32Bits(this.heatStages ? HEAT_CHEBYSHEV[this.heatStages].limit : HEAT_SHARE_MAX);
        u[18] = this.holding ? 1 : 0;
        u[19] = 0;
    }

    // A frame's steps (at most MAX_STEPS_PER_FRAME), with what they take
    // from the step before.
    encodeStepping(enc, steps) {
        this.heatStageCount();
        this.encodeSteps(enc, steps);
    }

    // What a step of this world as it is takes, in ms, { cpu, gpu }: its
    // encoding, and the GPU's running it. n steps encoded, then run and
    // waited for, then 2n, and the differences over n (what waiting takes
    // drops out). Frames overlap the two (the next one is encoded while the
    // GPU runs this one), so a frame's steps take the longer of them. n is
    // at most MAX_STEPS_PER_FRAME / 2, so each run is one command buffer.
    // The world steps on 3n steps. (main.js picks the grid by it.)
    async timeSteps(n) {
        const run = async (count) => {
            const start = performance.now();
            const enc = this.device.createCommandEncoder();
            this.encodeStepping(enc, count);
            const buffer = enc.finish();
            const encoded = performance.now();
            this.device.queue.submit([buffer]);
            await this.device.queue.onSubmittedWorkDone();
            return { cpu: encoded - start, gpu: performance.now() - encoded };
        };
        const steps = Math.min(n, MAX_STEPS_PER_FRAME / 2);
        const once = await run(steps);
        const twice = await run(2 * steps);
        return { cpu: Math.max(twice.cpu - once.cpu, 0) / steps, gpu: Math.max(twice.gpu - once.gpu, 0) / steps };
    }

    encodeSteps(enc, steps) {
        const { width: w, height: h } = this;
        const cellGroups = [Math.ceil(w / 8), Math.ceil(h / 8)];
        const blockGroups = [Math.ceil((Math.floor(w / 2) + 1) / 8), Math.ceil((Math.floor(h / 2) + 1) / 8)];
        let slot = 0;
        for (let s = 0; s < steps; s++) {
            this.frameNo++;
            this.clock.day = (this.clock.day + this.dayRate()) % 1;
            this.writeSlot(slot, 0, 0);
            const offset = [slot * SLOT_BYTES];

            // The light, every LIGHT_EVERY steps, from the cells as they are
            // at the start of the step.
            if (this.lightStale || this.frameNo % LIGHT_EVERY === 0) {
                this.encodeLight(enc, offset);
            }
            if (this.gravityStale || this.frameNo % GRAVITY_EVERY === 0) {
                this.encodeGravity(enc, offset);
            }
            // Heat's clock (clock.wgsl), every step: each light cell's
            // hottest matter, then across and down.
            let pass = this.beginPass(enc, "heat");
            pass.setBindGroup(0, this.clockBG[this.cur], offset);
            const lightGroups = this.lightSize.map((n) => Math.ceil(n / 8));
            for (const pipe of [this.pipes.clock.source, this.pipes.clock.across, this.pipes.clock.down]) {
                pass.setPipeline(pipe);
                pass.dispatchWorkgroups(...lightGroups);
            }
            pass.end();

            // The rooms (sky.wgsl), on the cells the air's flow is about to be
            // found for, every step (the skyline over them is the update
            // phase's, which nothing before it reads: here it was done twice):
            // the air's pressure zones are them (air.wgsl), and a sealed room
            // taken for the open air (one just built or painted shut) has
            // nowhere for what its gas gives out to go. Then the wet rooms,
            // the rooms joined through the liquid, the same way.
            pass = this.beginPass(enc, "air");
            pass.setBindGroup(0, this.skyBG[this.cur], offset);
            for (const pipe of [this.pipes.rooms, this.pipes.roomUnite, this.pipes.wetRooms, this.pipes.wetUnite]) {
                pass.setPipeline(pipe);
                pass.dispatchWorkgroups(...this.roomTiles);
            }
            this.encodeAir(pass, offset);
            pass.end();

            // The wind's carry (carry.wgsl), on the very cells the air's flow
            // was just found for: their air into scratch, as many substeps
            // as the fastest wind takes (CARRY_STEPS encoded, the substep's
            // number in the levels' slots, from set S into T and back), and
            // back into the cells.
            pass = this.beginPass(enc, "carry");
            const carry = this.pipes.carry;
            const onSubstep = (k) => pass.setBindGroup(0, this.carryBG[this.cur], [offset[0], SLOT_BYTES * k]);
            onSubstep(0);
            pass.setPipeline(carry.rise);
            pass.dispatchWorkgroups(...cellGroups);
            pass.setPipeline(carry.load);
            pass.dispatchWorkgroups(...cellGroups);
            for (let k = 0; k < CARRY_STEPS; k++) {
                onSubstep(k);
                pass.setPipeline(k % 2 === 0 ? carry.flowToT : carry.flowToS);
                pass.dispatchWorkgroups(...cellGroups);
            }
            onSubstep(0);
            pass.setPipeline(carry.settle);
            pass.dispatchWorkgroups(...cellGroups);
            pass.end();

            pass = this.beginPass(enc, "fluid");
            this.encodeFluid(pass, offset);
            pass.end();

            // Pairs of neighbors settling what passes between them, in place
            // (pairs.wgsl): a pair per invocation, along x on even steps and
            // y on odd ones, as P.frame picks there. Then the skyline again,
            // on the cells as the liquid left them: the update pass needs it
            // for heat exchange with the outside; and every ROOM_EVERY steps
            // how far down it the wind gets, for the outside air's oxygen
            // (one workgroup, sky.wgsl).
            pass = this.beginPass(enc, "update");
            pass.setPipeline(this.pipes.pairs);
            pass.setBindGroup(0, this.blockBG[this.cur], offset);
            const pairsVertical = this.frameNo % 2 === 1;
            pass.dispatchWorkgroups(
                Math.ceil(Math.ceil((pairsVertical ? h : w) / 2) / 8),
                Math.ceil((pairsVertical ? w : h) / 8),
            );
            pass.setPipeline(this.pipes.sky);
            pass.setBindGroup(0, this.skyBG[this.cur], offset);
            pass.dispatchWorkgroups(Math.ceil(w / 64));
            if (this.frameNo % ROOM_EVERY === 0) {
                pass.setPipeline(this.pipes.ventilation);
                pass.dispatchWorkgroups(1);
            }
            pass.end();
            // Conduction across stiff faces, in stages (heat.wgsl), on the
            // cells as the update pass then finds them: every step while
            // there are stages, and every HEAT_LOOK_EVERY to find out whether
            // there should be.
            if (this.heatStages > 0 || this.frameNo % HEAT_LOOK_EVERY === 0) {
                pass = this.beginPass(enc, "heat");
                this.encodeHeat(pass, offset, cellGroups);
                pass.end();
                this.heatLooked = true;
            }
            // Each sealed room's air's mixing over heat's step, summed on the
            // cells as the update pass is about to take them (update.wgsl's
            // zoneHeat), then the update pass.
            pass = this.beginPass(enc, "update");
            pass.setBindGroup(0, this.updateBG[this.cur], offset);
            pass.setPipeline(this.pipes.zoneHeat);
            pass.dispatchWorkgroups(...cellGroups);
            pass.setPipeline(this.pipes.update);
            pass.dispatchWorkgroups(...cellGroups);
            pass.end();
            this.cur ^= 1;

            // Structure's anchoring (bodies.wgsl's structureWatch on), on the
            // cells the update pass just wrote, every step: it does nothing
            // unless the structure changed.
            pass = this.beginPass(enc, "bodies");
            pass.setBindGroup(0, this.bodiesBG[this.cur], offset);
            const st = this.pipes.bodies;
            for (const pipe of [st.structureWatch, st.structureInit, st.structureUnite, st.structureAnchor, st.structureSettle]) {
                pass.setPipeline(pipe);
                pass.dispatchWorkgroups(...cellGroups);
            }
            pass.setPipeline(st.structureDone);
            pass.dispatchWorkgroups(1);
            pass.end();
            if (this.stressLive || this.frameNo % STRESS_LOOK_EVERY === 0) {
                this.encodeStress(enc, offset, cellGroups);
            }

            // Free bodies, on the cells the update pass just wrote, with its
            // parameters (bodies.wgsl). With nothing loose anywhere, as of a
            // frame or two ago, they are left out; a piece coming loose since
            // waits for the next readback (it was going to hang there a while
            // first anyway). Frames stepped faster than readbacks come back
            // can't wait for them.
            if (this.frameNo % CLAIM_STEPS === 0) {
                const cellCount = w * h;
                enc.clearBuffer(this.bodies, (this.bodyHeader + cellCount) * 4, cellCount * 4);
                // (And which body moved over each cell, keyed by the step
                // too: common.wgsl's bodySweptSlot, at the buffer's end.)
                enc.clearBuffer(this.bodies, this.bodies.size - 2 * cellCount * 4, 2 * cellCount * 4);
            }
            if (this.bodiesRun()) {
                pass = this.beginPass(enc, "bodies");
                this.encodeBodies(pass, offset);
                pass.end();
            }
            slot++;

            // Both block offsets flip every pass, so a particle moving steadily
            // stays in the slot that can step its way. Occasionally shifting the
            // x phase lets the other diagonal pairs meet too.
            if (this.random() & 1) {
                this.phaseX ^= 1;
            }
            pass = this.beginPass(enc, "move");
            pass.setPipeline(this.pipes.block);
            for (let p = 0; p < BLOCK_PASSES; p++) {
                this.passNo++;
                this.writeSlot(slot, (this.passNo + this.phaseX) & 1, this.passNo & 1);
                pass.setBindGroup(0, this.blockBG[this.cur], [slot * SLOT_BYTES]);
                pass.dispatchWorkgroups(...blockGroups);
                slot++;
            }
            pass.end();

            // The people, on the cells as they have moved (people.wgsl): a
            // workgroup each, stamping their bodies on the occupants mask
            // cleared for them; with none, cleared once.
            if (this.peopleLive || this.occupantsStamped) {
                enc.clearBuffer(this.occupants);
                this.occupantsStamped = this.peopleLive;
            }
            if (this.peopleLive) {
                pass = this.beginPass(enc, "people");
                pass.setPipeline(this.pipes.people);
                pass.setBindGroup(0, this.peopleBG[this.cur], offset);
                pass.dispatchWorkgroups(MAX_PEOPLE);
                pass.end();
            }

            if (this.electricLive) {
                this.encodeElectric(enc, offset);
            }
        }
        this.device.queue.writeBuffer(this.params, 0, this.paramData, 0, slot * SLOT_BYTES);
    }

    // The air for one step (air.wgsl, in the order its header lists): gather,
    // the blasts' half steps, momentum and divergence, then the projection
    // by a multigrid V-cycle, its finest level smoothed before and after,
    // each coarser one on the way down and up, the coarsest solved by one
    // workgroup; then the projection itself.
    encodeAir(pass, offset) {
        const a = this.pipes.air;
        const deepest = this.airLevels.length - 1;
        // The level and V-cycle a pass works on: its slot in airLevelBuf.
        const onLevel = (l, cycle = 0) => pass.setBindGroup(0, this.airBG[this.cur], [
            offset[0], SLOT_BYTES * (cycle * AIR_LEVELS + l),
        ]);
        const run = (pipeline, l) => {
            const [lw, lh] = this.airLevels[l];
            pass.setPipeline(pipeline);
            pass.dispatchWorkgroups(Math.ceil(lw / 8), Math.ceil(lh / 8));
        };
        // The V-cycle's passes, a workgroup to a tile of the level.
        const tiles = (pipeline, l) => {
            const [lw, lh] = this.airLevels[l];
            pass.setPipeline(pipeline);
            pass.dispatchWorkgroups(Math.ceil(lw / AIR_TILE), Math.ceil(lh / AIR_TILE));
        };
        const single = (pipeline) => {
            pass.setPipeline(pipeline);
            pass.dispatchWorkgroups(1);
        };
        onLevel(0);
        run(a.gather, 0);
        for (let i = 0; i < BLAST_SUBSTEPS; i++) {
            run(a.blastPressure, 0);
            run(a.blastVelocity, 0);
        }
        run(a.momentum, 0);
        run(a.divergence, 0);
        run(a.cgStart, 0);
        // Conjugate gradients, each iteration preconditioned by a V-cycle,
        // until what is left is under air.wgsl's AIR_TOLERANCE (the passes of
        // the iterations after skip), airCycles at most.
        // A V-cycle: each level but the coarsest smoothed and its residual
        // restricted into the next (down), the coarsest solved, and each
        // level back up taking on the correction and smoothed (up).
        for (let cycle = 0; cycle < this.airCycles; cycle++) {
            for (let l = 0; l < deepest; l++) {
                onLevel(l, cycle);
                tiles(a.down, l);
            }
            onLevel(deepest, cycle);
            if (deepest === 0) {
                tiles(a.smoothTile, 0);
            }
            single(a.solveCoarsest);
            for (let l = deepest - 1; l >= 0; l--) {
                onLevel(l, cycle);
                tiles(a.up, l);
            }
            run(a.cgRho, 0);
            single(a.cgBeta);
            run(a.cgDirection, 0);
            run(a.cgApply, 0);
            single(a.cgAlpha);
            run(a.cgUpdate, 0);
        }
        onLevel(0);
        run(a.zonePotential, 0);
        run(a.project, 0);
        run(a.recentre, 0);
    }

    // Conduction across stiff faces (heat.wgsl): which they are and what
    // they pass, then the stages, before and after taking turns.
    encodeHeat(pass, offset, groups) {
        pass.setPipeline(this.pipes.heat.prepare);
        pass.setBindGroup(0, this.heatBG[this.cur][0], [offset[0], 0]);
        pass.dispatchWorkgroups(...groups);
        pass.setPipeline(this.pipes.heat.stage);
        for (let j = 1; j <= this.heatStages; j++) {
            pass.setBindGroup(0, this.heatBG[this.cur][(j - 1) & 1], [offset[0], (j - 1) * SLOT_BYTES]);
            pass.dispatchWorkgroups(...groups);
        }
    }

    // Light and radiant heat (light.wgsl), in the order its header lists:
    // build, bounce and gather a thread per light cell, the sun sweep one
    // workgroup, and each cascade from the farthest down a thread per probe
    // and direction.
    encodeLight(enc, offset) {
        const l = this.pipes.light;
        const [lw, lh] = this.lightSize;
        const cellGroups = [Math.ceil(lw / 8), Math.ceil(lh / 8)];
        // Radiant heat's takings start from none each time (radiate.wgsl).
        const k = this.radiantNow;
        enc.clearBuffer(this.radiantTaken[k]);
        const pass = this.beginPass(enc, "light");
        pass.setPipeline(this.pipes.radiate.sweep);
        pass.setBindGroup(0, this.radiateBG[this.cur][k], offset);
        pass.dispatchWorkgroups(RADIANT_DIRS);
        pass.setPipeline(this.pipes.radiate.resolve);
        pass.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8));
        this.radiantNow = 1 - k;
        pass.setPipeline(l.build);
        pass.setBindGroup(0, this.buildBG[this.cur], offset);
        pass.dispatchWorkgroups(...cellGroups);
        // The sun's sweep is one workgroup taking its time, and the sun moves
        // slowly: every SUN_EVERY times the light is worked out (and whenever
        // it is out of date).
        if (this.lightStale || this.lightRefreshes % SUN_EVERY === 0) {
            pass.setPipeline(l.sun);
            pass.setBindGroup(0, this.sunBG, offset);
            pass.dispatchWorkgroups(1);
        }
        this.lightRefreshes++;
        // Mirrors in the sun throw it on.
        pass.setPipeline(l.bounce);
        pass.setBindGroup(0, this.bounceBG[this.cur], offset);
        pass.dispatchWorkgroups(...cellGroups);
        // Cascade 0 is cascade 1's directions for every light cell.
        pass.setPipeline(l.cascade);
        for (let k = this.cascadeCount - 1; k >= 0; k--) {
            const [pw, ph] = probes(k, lw, lh);
            pass.setBindGroup(0, this.cascadeBG[k & 1], [offset[0], SLOT_BYTES * k]);
            pass.dispatchWorkgroups(Math.ceil(directions(Math.max(k, 1)) * pw / 8), Math.ceil(ph / 8));
        }
        pass.setPipeline(l.gather);
        pass.setBindGroup(0, this.gatherBG, [offset[0], 0]);
        pass.dispatchWorkgroups(...cellGroups);
        pass.end();
        this.lightStale = Math.max(0, this.lightStale - 1);
    }

    // Gravity (gravity.wgsl): the blocks' masses and the pyramid over them,
    // then the field.
    encodeGravity(enc, offset) {
        const groups = this.lightSize.map((n) => Math.ceil(n / 8));
        const pass = this.beginPass(enc, "gravity");
        pass.setBindGroup(0, this.gravityBG[this.cur], offset);
        pass.setPipeline(this.pipes.gravity.mass);
        pass.dispatchWorkgroups(...groups);
        pass.setPipeline(this.pipes.gravity.upper);
        pass.dispatchWorkgroups(1);
        pass.setPipeline(this.pipes.gravity.field);
        pass.dispatchWorkgroups(...groups);
        pass.end();
        this.gravityStale = 0;
    }

    // Whether the bodies' report is too old to go by (REPORT_PATIENCE).
    bodiesStale() {
        return this.framesStepped - this.reportedAt > REPORT_PATIENCE;
    }

    // Whether the body passes run this step: something was loose as of the
    // last report, or that report is too old to go by.
    bodiesRun() {
        return this.bodiesLoose || this.bodiesStale();
    }

    // Electric current for one step (electric.wgsl), on the cells as they
    // are after moving: the conductances and aggregates, the residual and
    // the coarse problem, the coarse correction and the preconditioner, then
    // the CG's iterations (ELECTRIC_ITERATIONS, or ELECTRIC_ITERATIONS_FAR
    // just after painting, or while the last report had it far off),
    // each a reduce, a direction, a coarse correction and a precondition,
    // whose passes return at once once it is done; then the currents, and
    // the heat and hydrogen they make put into the cells. Only while there
    // may be a battery.
    encodeElectric(enc, offset) {
        const e = this.pipes.electric;
        const pass = this.beginPass(enc, "electric");
        pass.setBindGroup(0, this.electricBG[this.cur], offset);
        const over = (pipe, groups) => {
            pass.setPipeline(pipe);
            pass.dispatchWorkgroups(...groups);
        };
        over(e.assemble, this.electricTiles);
        over(e.residual, this.electricTiles);
        over(e.coarseSolve, [1]);
        over(e.precondition, this.electricTiles);
        const far = this.electricFarSteps > 0 || this.electricResidual > ELECTRIC_FAR;
        this.electricFarSteps = Math.max(this.electricFarSteps - 1, 0);
        const iterations = far ? ELECTRIC_ITERATIONS_FAR : ELECTRIC_ITERATIONS;
        for (let k = 0; k < iterations; k++) {
            over(e.reduce, [1]);
            over(e.direction, this.electricTiles);
            over(e.coarseSolve, [1]);
            over(e.precondition, this.electricTiles);
        }
        const blocks = [Math.ceil(this.width / CURRENT_BLOCK), Math.ceil(this.height / CURRENT_BLOCK)];
        over(e.current, blocks);
        over(e.effects, blocks);
        pass.end();
    }

    // Stress in standing structure for one step (stress.wgsl, in the order
    // its header lists), on the cells the update pass and the anchoring just
    // left: setup over the tiles of nodes, the residual and the CG's start,
    // its STRESS_ITERATIONS iterations (STRESS_ITERATIONS_FAR while the last
    // report said it was far off: STRESS_FAR, STRESS_FAR_TILE), each an apply and an update (by the
    // iteration's parity), returning at once once converged; then the
    // cracks.
    encodeStress(enc, offset, cellGroups) {
        const st = this.pipes.stress;
        const pass = this.beginPass(enc, "stress");
        pass.setBindGroup(0, this.stressBG[this.cur], offset);
        const over = (pipe, groups) => {
            pass.setPipeline(pipe);
            pass.dispatchWorkgroups(...groups);
        };
        over(st.setup, this.stressTiles);
        over(st.residual, this.stressTiles);
        over(st.begin, [1]);
        for (let k = 0; k < (this.stressFar ? STRESS_ITERATIONS_FAR : STRESS_ITERATIONS); k++) {
            over(k === 0 ? st.applyFirst : (k & 1 ? st.apply1 : st.apply0), this.stressTiles);
            over(k & 1 ? st.update1 : st.update0, this.stressTiles);
        }
        over(st.measure, cellGroups);
        over(st.crack, cellGroups);
        over(st.finish, [1]);
        pass.end();
    }

    // Whether a cell is small enough for surface tension to show
    // (waterTension in fluid.wgsl); at a centimetre it isn't, and its pass is
    // left out.
    surfaceTension() {
        return CAPILLARY_LENGTH / this.metresPerCell >= SURFACE_MIN_LENGTH;
    }

    // Liquids for one step (fluid.wgsl), in the order its header lists. The
    // passes over the blocks worked on get a workgroup per block in the
    // world, and those over the particles a thread for each there is room
    // for; the ones with nothing to do return at once. With no liquid and no
    // particles anywhere, as of a frame or two ago, only the bookkeeping
    // runs (syncMarks, which leaves the particles and their cells as they
    // are); liquid painted since holds still until the next readback notices
    // it. Not while the body passes may run, though (bodiesRun), whose
    // rigid cells the solve moves, and not on readbacks too old to go by.
    // Viscosity's sweeps likewise wait for viscous liquid.
    encodeFluid(pass, offset) {
        const f = this.pipes.fluid;
        // The level and iteration a pass works on: its slot in
        // fluidLevelBuf.
        const onLevel = (l, cycle = 0) => pass.setBindGroup(0, this.fluidBG[this.cur], [offset[0], SLOT_BYTES * (cycle * (MG_LEVELS + 1) + l)]);
        const run = (pipeline, ...groups) => {
            pass.setPipeline(pipeline);
            pass.dispatchWorkgroups(...groups);
        };
        const blocks = (pipeline) => run(pipeline, this.blockCount);
        const particles = (pipeline) => run(pipeline, ...this.particleGroups);
        const world = [Math.ceil(this.width / FLUID_BLOCK), Math.ceil(this.height / FLUID_BLOCK)];
        onLevel(0);
        const full = this.fluidStats.active || this.bodiesRun();
        if (!full) {
            run(f.syncMarks, ...world);
            run(f.compact, Math.ceil(this.blockCount / 64));
            blocks(f.retire);
            return;
        }
        run(f.sync, ...world);
        particles(f.syncParticles);
        run(f.syncSeed, ...world);
        run(f.compact, Math.ceil(this.blockCount / 64));
        blocks(f.retire);
        blocks(f.clear);
        particles(f.gather);
        blocks(f.sealJoin);
        blocks(f.sealAnchor);
        blocks(f.classify);
        if (this.surfaceTension()) {
            blocks(f.surface);
        }
        // The bodies' sums emptied, and the liquid's pressure at each body's
        // surface as it is posed (fluid.wgsl's bodyLoads), for its faces.
        run(f.bodyStart, Math.ceil(MAX_BODIES * BODY_PUSH_WORDS / 256));
        blocks(f.bodyLoads);
        blocks(f.setFaces);
        // Viscosity before the projection, which takes out what it leaves
        // diverging (fluid.wgsl's viscousFace); the pressure's push, taken
        // off the faces it solves (setFaces), back on after it (unpress),
        // which runs even when the sweeps don't (the readback saying there
        // is no viscous liquid yet, when there is).
        const viscous = this.fluidStats.viscous || this.surfaceTension();
        for (let i = 0; viscous && i < VISCOSITY_ITERATIONS; i++) {
            blocks(f.viscosityRed);
            blocks(f.viscosityBlack);
        }
        blocks(f.unpress);
        // The liquid's drag on the bodies, as the sweeps left their faces,
        // summed, and their faces moving with them again (bodyDrag).
        if (viscous) {
            blocks(f.bodyDrag);
            blocks(f.bodyFaces);
        }
        blocks(f.divergence);

        // The pressure (and drift): relaxed, then corrected from the coarse
        // levels in V-cycles, relaxed again. The relaxations alternate
        // between the two tilings, and each step starts on the other one from
        // the last: with one relaxation before the V-cycle and one after, the
        // last one of every step would otherwise be on the same tiling, and
        // its seams, where the solve is least settled, always in the same
        // place.
        this.shiftedFirst = !this.shiftedFirst;
        let shifted = this.shiftedFirst;
        const relax = (times) => {
            for (let i = 0; i < times; i++) {
                blocks(f.relaxRims);
                if (shifted) {
                    run(f.relaxShifted, this.blockCount + this.edgeTiles);
                } else {
                    blocks(f.relax);
                }
                shifted = !shifted;
            }
        };
        // A sweep of smoothing on each level on the way down and on the way
        // up. (Passing the error down unsmoothed, its jagged parts alias
        // into the coarse levels' corrections, and in a fast flow the cycle
        // runs away.)
        const deepest = this.levels.length;
        // By conjugate gradients, the V-cycle their preconditioner, until
        // what is left is under the tolerance (fluid.wgsl's converged),
        // fluidCycles iterations encoded.
        blocks(f.cgStart);
        blocks(f.cgJoin);
        blocks(f.cgAnchor);
        blocks(f.cgBodyX);
        blocks(f.cgResidual);
        blocks(f.cgPrepare);
        for (let cycle = 0; cycle < this.fluidCycles; cycle++) {
            const onCoarse = (l, pipeline) => {
                const [lw, lh] = this.levels[l - 1];
                onLevel(l, cycle);
                run(pipeline, Math.ceil(lw / 8), Math.ceil(lh / 8));
            };
            onLevel(0, cycle);
            relax(PRE_RELAX);
            for (let l = 1; l <= deepest; l++) {
                onCoarse(l, f.restrictLevel);
                if (l < deepest) {
                    onCoarse(l, f.smoothRed);
                    onCoarse(l, f.smoothBlack);
                }
            }
            onLevel(deepest, cycle);
            run(f.solve, 1);
            for (let l = deepest - 1; l >= 1; l--) {
                onCoarse(l, f.prolong);
                onCoarse(l, f.smoothRed);
                onCoarse(l, f.smoothBlack);
            }
            onLevel(0, cycle);
            blocks(f.prolongFine);
            relax(POST_RELAX);
            blocks(f.cgRho);
            run(f.cgBeta, 1);
            blocks(f.cgDirection);
            blocks(f.cgBodyP);
            blocks(f.cgApply);
            run(f.cgAlpha, 1);
            blocks(f.cgUpdate);
        }
        onLevel(0);
        blocks(f.cgFinish);
        blocks(f.cgBodyF);
        blocks(f.project);
        // What the liquid did to the bodies in it, for theirs (fluid.wgsl's
        // rigidify and bodyPush; bodies.wgsl's lift takes it).
        if (this.bodiesRun()) {
            blocks(f.rigidify);
            run(f.bodyPush, Math.ceil(MAX_BODIES / 256));
        }
        blocks(f.extrapolate);
        blocks(f.extrapolateFurther);
        particles(f.advect);
        // Sorted by cell into particles[0], and drawn from there.
        particles(f.sortCount);
        blocks(f.scanBlocks);
        run(f.scanTotals, 1);
        blocks(f.scanFinish);
        particles(f.scatter);
        blocks(f.rasterClaims);
        blocks(f.raster);
        blocks(f.unpaint);
    }

    // Free bodies for one step (bodies.wgsl), in the order its header lists:
    // plan with a thread per tile that could be marked or slot in the table,
    // the passes over the tiles worked on with BODY_GROUPS workgroups that go
    // through them, those over the bodies with BODY_WORKGROUPS, and a single
    // thread to finish. With nothing loose in no
    // body as of a fresh enough readback, the passes that look for new
    // bodies are left out; the update pass's marks wait for them.
    encodeBodies(pass, offset) {
        const b = this.pipes.bodies;
        const run = (pipelines, groups) => {
            for (const pipeline of pipelines) {
                pass.setPipeline(pipeline);
                pass.dispatchWorkgroups(groups);
            }
        };
        const look = this.bodiesMarked || this.bodiesStale();
        pass.setBindGroup(0, this.bodiesBG[this.cur], offset);
        run([look ? b.plan : b.listBodies], Math.ceil(Math.max(this.bodyTiles, MAX_BODIES) / 64));
        if (look) {
            run([b.init, b.unite, b.gather, b.found], BODY_GROUPS);
            run([b.foundBodies], 1);
            run([b.capture], BODY_GROUPS);
        }
        run([b.lift, b.parts, b.contact], BODY_WORKGROUPS);
        run([b.jointSolve], 1);
        run([b.integrate, b.stamp], BODY_WORKGROUPS);
        // The pool of shapes is packed once a readback says it is full.
        if (this.bodiesFull) {
            run([b.compactPlan], 1);
            run([b.compactCopy], BODY_WORKGROUPS);
        }
        run([look ? b.finish : b.finishListed], 1);
    }

    // The bloom, and shiny cells' colours in the view drawn (glow.wgsl).
    encodeGlow(enc, mode) {
        this.device.queue.writeBuffer(this.glowBuf, 8, new Uint32Array([mode === VIEW.LIT ? 1 : 0, f32Bits(this.clock.day)]));
        const [gw, gh] = this.glowSize;
        const x = Math.ceil(gw / 8);
        const y = Math.ceil(gh / 8);
        const pass = this.beginPass(enc, "draw");
        pass.setPipeline(this.pipes.glow);
        pass.setBindGroup(0, this.glowBG[this.cur]);
        pass.dispatchWorkgroups(x, y);
        pass.setPipeline(this.pipes.blurH);
        pass.setBindGroup(0, this.blurHBG);
        pass.dispatchWorkgroups(x, y);
        pass.setPipeline(this.pipes.blurV);
        pass.setBindGroup(0, this.blurVBG);
        pass.dispatchWorkgroups(x, y);
        pass.setPipeline(this.pipes.shine);
        pass.setBindGroup(0, this.shineBG[this.cur]);
        pass.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8));
        pass.end();
    }

    encodeRender(enc, view, target) {
        const f = new Float32Array(this.viewData);
        const u = new Uint32Array(this.viewData);
        f[0] = this.width;
        f[1] = this.height;
        f[2] = view.originX;
        f[3] = view.originY;
        f[4] = view.scale;
        f[5] = view.time;
        f[6] = view.mouseX;
        f[7] = view.mouseY;
        f[8] = view.lineX;
        f[9] = view.lineY;
        f[10] = view.radius;
        u[11] = view.cursor;
        u[12] = view.mode;
        u[13] = this.electricLive ? 1 : 0;
        f[14] = this.clock.day;
        f[16] = view.canvasWidth;
        f[17] = view.canvasHeight;
        this.device.queue.writeBuffer(this.viewBuf, 0, this.viewData);

        const pass = enc.beginRenderPass({
            colorAttachments: [{ view: target, loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 1] }],
            timestampWrites: this.timer?.writes("draw"),
        });
        pass.setPipeline(this.pipes.render);
        pass.setBindGroup(0, this.renderBG[this.cur]);
        pass.draw(3);
        // The people over it, a quad each (figures.wgsl).
        if (this.peopleLive) {
            pass.setPipeline(this.pipes.figures);
            pass.setBindGroup(0, this.figuresBG[this.cur]);
            pass.draw(6, MAX_PEOPLE);
        }
        pass.end();
    }

    // With `electric`, the potential and current density are read too (null
    // while no currents are being solved): the current as worked out for a
    // cell CELL_M across, and through one this world's size (electric.wgsl).
    readCell(pos, electric) {
        this.readPending = true;
        const buffer = this.readBuf;
        buffer.mapAsync(GPUMapMode.READ).then(() => {
            const range = buffer.getMappedRange();
            const cell = cellFields(new Uint32Array(range, 0, 4), new Float32Array(range, 0, 4), 0);
            const [pressure] = new Float32Array(range, CELL_BYTES, 1);
            const [volts] = new Float32Array(range, CELL_BYTES + AIR_BYTES, 1);
            const flow = new Uint32Array(range, CELL_BYTES + AIR_BYTES + 4, 1)[0];
            const [jx, jy] = [f16Value(flow & 0xffff), f16Value(flow >>> 16)];
            const [moist, held] = new Uint32Array(range, CELL_BYTES + AIR_BYTES + 4 + FLOW_BYTES, 2);
            const [stress] = new Float32Array(range, CELL_BYTES + AIR_BYTES + 4 + FLOW_BYTES + GAS_BYTES, 1);
            buffer.unmap();
            this.readPending = false;
            this.onHover?.({
                x: pos.x,
                y: pos.y,
                type: cell.type,
                aux: cell.aux,
                life: cell.life,
                flags: cell.flags,
                temp: cell.temp,
                vx: cell.vel[0],
                vy: cell.vel[1],
                fracX: cell.frac[0],
                fracY: cell.frac[1],
                // What it holds of the air (the gas plane).
                vapour: f16Value(moist & 0xffff),
                mist: f16Value(moist >>> 16),
                // (As breathing goes by it, the oxygen's pressure against
                // fresh air's: common.wgsl's oxygenPressure.)
                oxygen: (1 - f16Value(held & 0xffff)) * f16Value(held >>> 16) * Math.max(cell.temp + KELVIN, 0) / (AMBIENT_T + KELVIN),
                pressure,
                // How near standing structure is to breaking: its failure
                // index, 1 at its strength (stress.wgsl's measure).
                stress,
                electric: electric ? { volts, amps: Math.hypot(jx, jy) * this.metresPerCell / CELL_M } : null,
            });
        }, () => {
            this.readPending = false;
        });
    }
}
