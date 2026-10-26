// The passes of a frame and of each step in it, in order, and what each
// reads and writes of the simulation's state, by name (Simulation.resources
// maps the names to the GPU's buffers and textures). From it,
// persistentResources(): what is left from one step for the next, which a
// snapshot has to keep (Simulation.snapshotParts) — written down once here,
// where each pass is, rather than listed apart and kept in step by hand.
//
// A pass runs `when` its group does: "" every step (every frame, for the
// frame's own passes); "light" when the light is worked out, and
// "light/sun" only some of those times; "gravity", "rooms" (the
// ventilation) and "claims" on
// their cadences; "fluid", "bodies", "people", "electric" and "stress" while they
// have work (sim.js's reports), "heat" too and every few steps besides;
// "place" when people are placed or erased. A
// group runs whenever one it starts with does ("light/sun" only with
// "light", all with ""). `overwrites` it writes in full, `writes` in part.
// A resource read before anything that runs whenever the reader does has
// overwritten it is kept from step to step.

// Written by the host before every use (heatStiffness cleared each time a
// report reads it, then raised by the steps: what it carries is for the
// host), and never changed after creation: neither carries anything from
// one step to the next that the steps read.
const HOST_INPUTS = ["params", "personRequests", "brush", "fillMask", "view", "heatStiffness", "heatWeights"];
const CONSTANTS = ["elements", "levels", "fluidLevels", "airLevels", "cascadeSlots", "glowParams"];

const BEFORE_STEPS = [
    { name: "place", when: "place", reads: ["personRequests", "people", "cells", "air", "flow", "gravity", "radiant"], writes: ["people"] },
];

const STEP = [
    // radiate.wgsl, with the light: what each cell takes in (and reflects),
    // from the last time's reflections and what went out of the world's
    // sides; then per cell for update.wgsl
    { name: "radiate sweep", when: "light", reads: ["cells", "radiantTaken", "radiantEdges"], writes: ["radiantTaken", "radiantEdges"] },
    { name: "radiate resolve", when: "light", reads: ["cells", "radiantTaken"], overwrites: ["radiant"] },
    // light.wgsl
    { name: "light build", when: "light", reads: ["cells", "light", "quarters", "occupants"], overwrites: ["lightScene", "mirror"] },
    { name: "light sun", when: "light/sun", reads: ["lightScene"], overwrites: ["sun"] },
    { name: "light bounce", when: "light", reads: ["cells", "lightScene", "sun"], overwrites: ["bounced"] },
    // The farthest cascade marches to the horizon; each nearer one merges
    // the one beyond it, written just before.
    { name: "light cascade (farthest)", when: "light", reads: ["lightScene", "light", "mirror"], overwrites: ["cascades"] },
    { name: "light cascades", when: "light", reads: ["lightScene", "light", "mirror", "cascades"], writes: ["cascades"] },
    { name: "light gather", when: "light", reads: ["lightScene", "sun", "cascades", "bounced"], overwrites: ["light", "heat", "quarters"] },
    // gravity.wgsl
    // (The masses are worked out afresh each time, but after them each
    // light cell keeps the down word it was last given, which the field
    // reads for its hysteresis, gravity.wgsl's DOWN_STICK.)
    { name: "gravity mass", when: "gravity", reads: ["cells"], writes: ["gravityMasses"] },
    { name: "gravity upper", when: "gravity", reads: ["gravityMasses"], writes: ["gravityMasses"] },
    { name: "gravity field", when: "gravity", reads: ["gravityMasses"], writes: ["gravityMasses"], overwrites: ["gravity"] },
    // clock.wgsl, every step: each light cell's hottest matter, across, then
    // down, fading what was there before
    { name: "clock source", when: "", reads: ["cells", "clockHeated"], overwrites: ["clockHot", "clockHeated"] },
    { name: "clock across", when: "", reads: ["clockHot"], overwrites: ["clockAlong"] },
    { name: "clock down", when: "", reads: ["clockAlong", "clock"], writes: ["clock"] },
    // sky.wgsl: the rooms, the air's pressure zones (the skyline is the
    // update phase's, below)
    { name: "rooms", when: "", reads: ["cells", "sky", "field"], writes: ["sky"] },
    // air.wgsl: gather (its working state afresh, the zones' sums into the
    // sky), the blasts' layer, momentum from last step's air, the projection
    // warm-started from last step's potential (airCg), the air
    // (and the pushes other passes and the wind tool put into the air,
    // airSource, taken in and cleared)
    {
        name: "air", when: "",
        reads: ["cells", "air", "airProjection", "airCg", "blast", "sky", "gravity", "airLevels", "airSource", "field"],
        writes: ["air", "airProjection", "airCg", "blast", "sky", "airSource"], overwrites: ["airScratch", "airFastest"],
    },
    // carry.wgsl, on the cells the air's flow was found for: the cells' gas
    // and the gas other passes pushed aside (airSource, taken in and
    // cleared) into scratch, the substeps there, then back into the cells
    {
        name: "carry", when: "", reads: ["cells", "air", "airFastest", "airCg", "airSource", "levels", "field", "gravity", "sky"],
        writes: ["cells", "airSource"], overwrites: ["carryScratch"],
    },
    // fluid.wgsl (its bookkeeping, syncMarks to retire, runs even with no
    // liquid); the gas it takes the place of into the air (airSource)
    {
        name: "fluid", when: "fluid",
        reads: ["cells", "air", "particles", "counters", "field", "blocks", "fluidLevels", "gravity", "airSource", "bodyMotion", "occupants", "bodies"],
        writes: ["cells", "particles", "counters", "field", "blocks", "airSource"], overwrites: ["fluidCg", "bodyPushes"],
    },
    // pairs.wgsl (the hydrogen of brine split where its salt crystallises
    // into the air, airSource), sky.wgsl (the skyline, on the cells the
    // liquid left)
    { name: "pairs", when: "", reads: ["cells", "air", "gravity", "clock", "airSource"], writes: ["cells", "airSource"] },
    { name: "sky", when: "", reads: ["cells"], writes: ["sky"] },
    { name: "ventilation", when: "rooms", reads: ["cells", "sky"], writes: ["sky"] },
    // heat.wgsl: prepare, then the stages; the first writes all that the
    // rest read of heatStage1
    {
        name: "heat prepare", when: "heat", reads: ["cells", "air", "clock", "heatStiffness"],
        writes: ["heatStiffness"], overwrites: ["heatFaces", "heatPhases", "heatStage0", "heatTiles"],
    },
    {
        name: "heat stage 1", when: "heat", reads: ["cells", "heatFaces", "heatPhases", "heatStage0", "heatTiles", "heatWeights"],
        overwrites: ["heatStage1"],
    },
    {
        name: "heat stages", when: "heat",
        reads: ["cells", "heatFaces", "heatPhases", "heatStage0", "heatStage1", "heatTiles", "heatWeights"],
        writes: ["cells", "heatStage0", "heatStage1"],
    },
    // update.wgsl: reads cells[cur] and loads[cur], writes all of the other
    // two (then cur flips); its tile marks into the bodies; takes in the heat
    // other passes put in (heatIn) and clears it; gas it turns into
    // something else into the air (airSource)
    {
        name: "update", when: "",
        reads: ["cells", "loads", "air", "field", "sky", "bodies", "heat", "light", "gravity", "clock", "heatIn", "airSource", "radiant"],
        writes: ["cells", "loads", "bodies", "heatIn", "airSource"],
    },
    { name: "clear claims", when: "claims", writes: ["bodies"] },
    // bodies.wgsl's anchoring (structureWatch on): structure no anchor holds
    // let go
    { name: "structure", when: "", reads: ["cells", "bodies", "gravity"], writes: ["cells", "bodies"] },
    // stress.wgsl: the CG's state carried from step to step; a crack's step
    // marked in the bodies buffer, for next step's update
    {
        name: "stress", when: "stress",
        reads: ["cells", "air", "field", "loads", "bodies", "stressNodes", "stressWork", "stressDisp", "gravity"],
        writes: ["bodies", "stressNodes", "stressWork", "stressDisp"],
    },
    // bodies.wgsl
    {
        name: "bodies", when: "bodies",
        reads: ["cells", "field", "bodies", "counters", "bodyMotion", "bodyShapes", "gravity", "airSource"],
        writes: ["cells", "bodies", "counters", "bodyMotion", "bodyShapes", "airSource"],
    },
    // block.wgsl, BLOCK_PASSES times
    { name: "move", when: "", reads: ["cells", "air", "gravity", "airSource", "occupants"], writes: ["cells", "airSource"] },
    // people.wgsl
    // (its bodies stamped on the occupants mask, cleared before it)
    { name: "people", when: "people", reads: ["cells", "air", "flow", "gravity", "people", "radiant", "clock"], writes: ["cells", "people", "occupants"] },
    // electric.wgsl: the Joule heat into heatIn, for next step's update
    {
        name: "electric", when: "electric",
        reads: ["cells", "volts", "electric", "coarse", "electricWork", "clock", "heatIn"],
        writes: ["cells", "volts", "flow", "electric", "coarse", "electricWork", "heatIn", "clockHeated"],
    },
];

const AFTER_STEPS = [
    // glow.wgsl, blur.wgsl, render.wgsl, figures.wgsl
    { name: "glow", when: "", reads: ["cells", "glowParams"], overwrites: ["glow"] },
    { name: "shine", when: "", reads: ["cells", "light", "sun", "glowParams"], overwrites: ["shine"] },
    { name: "blur", when: "", reads: ["glow"], writes: ["glow"] },
    { name: "render", when: "", reads: ["view", "cells", "glow", "air", "flow", "light", "sun", "gravity", "shine"] },
    { name: "figures", when: "people", reads: ["view", "cells", "people", "light"] },
    // what the host reads back: the hovered cell, and the report
    { name: "hover", when: "", reads: ["cells", "air", "volts", "flow"] },
    { name: "report", when: "", reads: ["blocks", "bodies", "electric", "heatStiffness", "people", "stressWork"] },
];

// Every pass but the GPU-wide inputs reads `elements` and `params`; they
// are left out above.
export const PASSES = [...BEFORE_STEPS, ...STEP, ...AFTER_STEPS];

// The names of the resources one step leaves for the next. Throws on a
// name `known` doesn't have (a typo in the table).
export function persistentResources(known) {
    const kept = new Set();
    PASSES.forEach((pass, i) => {
        for (const name of [...(pass.reads ?? []), ...(pass.writes ?? []), ...(pass.overwrites ?? [])]) {
            if (!known.has(name)) {
                throw new Error(`passes.js: ${pass.name} names "${name}", which the simulation has no resource for`);
            }
        }
        for (const name of pass.reads ?? []) {
            if (HOST_INPUTS.includes(name) || CONSTANTS.includes(name)) {
                continue;
            }
            const covered = PASSES.slice(0, i).some((p) => (p.overwrites ?? []).includes(name) && pass.when.startsWith(p.when));
            if (!covered) {
                kept.add(name);
            }
        }
    });
    return kept;
}
