// The developer's toolbox, kept out of sight: ` (the backquote key) opens
// and closes it, and ?dev in the address opens it with the page. Nothing in
// the page points to it, and the Shortcuts list leaves it out. It is built
// the first time it is opened, so until then the page holds none of it.
//
// Cell: everything the world stores for one cell, word for word, read back
// from each buffer and texture that keeps something per cell, beside what
// each word means as the shaders read it. The cell is the pointer's, or one
// pinned with Ctrl+click (⌘+click on a Mac) or typed in, moved a cell at a
// time with the arrow keys. What hangs on what that read finds (the cell's
// particles, its body, its person, its zone and wet room) is read straight
// after it: a frame or two later while the world runs, the same state while
// it is paused.
// World: the host's state (snapshotState: what the reports adapt), the
// GPU's counters the reports copy, and every resource with its size.
// Bodies, People: their tables, a row each, and one picked word for word.
// Steps: a step or ten of a paused world, and a trace (the cell read after
// each step of a run of them, downloaded as JSON).
// In the console, `sand`: the world (sim), the device, the toolbox, and
// read(buffer, offset, bytes) for any buffer's words.

import { cellRecord } from "./cellrecord.js";
import { ELEMENTS, KIND } from "./elements.js";
import { describePerson } from "./people.js";
import {
    AIR_CELL, BODY_CONTACT_WORDS, BODY_CONTACTS, BODY_FLAG, BODY_LIQUID_WORDS, BODY_PUSH_WORDS, BODY_STAT, BODY_STATS, BODY_SUM_WORDS,
    BODY_TABLE_WORDS, CELL_BYTES, CELL_FLAGS, cellFields, ELECTRIC_STATS, ELECTRIC_WORK, f16Value, FLUID_BLOCK, FLUID_COUNT, FLUID_STATS,
    GAS_BYTES, LIQUID_CELL_WORDS, LIQUID_PER_CELL, MAX_BODIES, MAX_PEOPLE, P_ATM, PERSON_FLAG, PERSON_STATE, ROOM_PIECES, ROOM_TILE,
    STRESS_NODE_WORDS, STRESS_SCALARS, STRESS_STAT, STRESS_STATS, WET_WORDS, ZONE_WORDS,
} from "./shared.js";
import { LAYOUT, MAX_STEPS_PER_FRAME } from "./sim.js";

const REFRESH_MS = 100;       // the inspector's reads while it is live, at most one in this
const LIST_REFRESH_MS = 500;  // the bodies' and people's tables'
const MAX_PARTICLES = 16;     // of a cell's run of particles, the first this many read
const TRACE_STEPS = 100;      // a trace's steps unless another number is typed

// What the shaders keep to themselves that the words are read by:
// fluid.wgsl's FIX, the G_* counters, its blocks' bits and PROJECTED;
// common.wgsl's exact sums (EXACT_FINE, EXACT_COARSE), RIGID_FIX, DEST_*,
// AIR_BLOCKED, CLAIM_PERIOD and the ZONE_* and WET_* words; air.wgsl's
// WIDE_SCALE; radiate.wgsl's RAD_FIXED; stress.wgsl's displacements' unit
// (2^-40 cell) and N_* flags; bodies.wgsl's T_* words, B_* flags and
// FREE_BODY; electric.wgsl's W_* words and NO_SLOT. Mirrored by hand: one
// changed there reads wrong here (the raw words are right whatever).
const FIX = 65536;
const EXACT_FINE = 2 ** 32;
const EXACT_COARSE = 2 ** 24;
const RIGID_FIX = 2 ** 24;
const WIDE_SCALE = 2 ** 24;
const RAD_FIXED = 262144;
const DISP_UNIT = 2 ** 40;
const AIR_BLOCKED = -10000;
const DEST = { KEEP: 0xffffffff, ASIDE: 0x80000000, CELL: 0x40000000, SPAN: 0x3fffffff };
const G = { COUNT: 0, TAKEN: 1, SORT: 2, START: 3, DEST: 4, TYPE: 5, DENSITY: 6, MASS: 7, AIR_TO: 8, RIM: 9 };
const NONE = 0xffffffff;
const NO_SLOT = 0xff;
const FREE_BODY = 2047;
const CLAIM_PERIOD = 0xfffff;
const BLOCK_MARKS = [[1, "liquid"], [2, "held body's region"]];
const BLOCK_STATE = [[1, "active"], [2, "has fluid"], [4, "has viscous"], [8, "near liquid"]];
const BODY_FLAGS = [
    [BODY_FLAG.ALIVE, "alive"], [BODY_FLAG.NEW, "new"], [4, "rejoin"], [BODY_FLAG.PARTS, "parts"], [16, "supported"],
    [BODY_FLAG.HELD, "held"], [BODY_FLAG.WET, "wet"], [128, "rested"],
];
const SUPPORT = [[1, "still"], [2, "held"], [4, "resting"]];
const NODE_FLAGS = [[1, "active"], [2, "fixed"]];
const PUSH_SETS = ["pressure (even iterations)", "pressure (odd iterations)", "solution", "forces", "surface", "drag"];
const KIND_NAMES = Object.fromEntries(Object.entries(KIND).map(([name, k]) => [k, name.toLowerCase()]));
const STATE_NAMES = Object.fromEntries(Object.entries(PERSON_STATE).map(([name, k]) => [k, name.toLowerCase()]));
const PERSON_FLAGS = Object.entries(PERSON_FLAG).map(([name, b]) => [b, name.toLowerCase()]);
const GROUPS = ["Cell", "People", "Air", "Heat", "Light", "Liquid", "Bodies", "Structure", "Current", "Rooms"];
const TEXEL_BYTES = { "rgba32uint": 16, "rgba32float": 16, "rgba16float": 8, "r32float": 4 };

// ------------------------------------------------------------ numbers

const scratch = new DataView(new ArrayBuffer(4));

function f32(w) {
    scratch.setUint32(0, w >>> 0, true);
    return scratch.getFloat32(0, true);
}

// A float to 9 significant digits, enough to tell an f32 from its
// neighbours; an f16 to 5.
function num(v) {
    return Number.isFinite(v) ? String(Number(v.toPrecision(9))) : String(v);
}

function half(h) {
    const v = f16Value(h & 0xffff);
    return Number.isFinite(v) ? String(Number(v.toPrecision(5))) : String(v);
}

function halves(w) {
    return `${half(w & 0xffff)}, ${half(w >>> 16)}`;
}

function hex(w) {
    return `0x${(w >>> 0).toString(16).padStart(8, "0")}`;
}

// An exact sum's two words, low then high (common.wgsl's exactValue), in
// units of 1 / scale.
function exactSum(lo, hi, scale) {
    return ((hi | 0) * 2 ** 32 + (lo >>> 0)) / scale;
}

// The names of the bits set in v, from [bit, name] pairs.
function bitNames(v, names) {
    const set = names.filter(([b]) => v & b).map(([, name]) => name);
    return set.length ? set.join(", ") : "none";
}

// A cell's flags by every name each bit has (shared.js's CELL_FLAGS: a bit
// means one thing to each kind of element).
function cellFlagNames(flags) {
    const set = [1, 2, 4, 8].filter((b) => flags & b).map((b) => Object.keys(CELL_FLAGS).filter((k) => CELL_FLAGS[k] === b).join("/"));
    return set.length ? set.join(", ") : "none";
}

function elementName(t) {
    const e = ELEMENTS[t];
    return e ? `${t} ${e.id}` : `${t} (none)`;
}

// A cell index as (x, y).
function at(sim, i) {
    return i < sim.width * sim.height ? `${i} (${i % sim.width}, ${Math.floor(i / sim.width)})` : `${i} (outside)`;
}

// How each word is shown raw.
const FORMATS = {
    hex,
    u32: (w) => String(w >>> 0),
    i32: (w) => String(w | 0),
    f32: (w) => num(f32(w)),
    f16: (w) => `${half(w & 0xffff)} | ${half(w >>> 16)}`,
};

const escapeHtml = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

// ------------------------------------------------------------ reading back

// A part of a buffer to read (bytes at byte offset), or a texel of a
// texture, named as the inspector shows where its words came from.
function span(buffer, name, offset, bytes) {
    return { buffer, name, offset, bytes };
}

function texel(texture, name, x, y) {
    return { texture, name, x, y, bytes: TEXEL_BYTES[texture.format] };
}

// Why a part can't be read, or null if it can.
function unreadable(r) {
    if (r.texture) {
        if (!(r.texture.usage & GPUTextureUsage.COPY_SRC)) {
            return "not copyable";
        }
        if (!r.bytes) {
            return `format ${r.texture.format} not known`;
        }
        return r.x < 0 || r.y < 0 || r.x >= r.texture.width || r.y >= r.texture.height ? "outside the texture" : null;
    }
    if (!(r.buffer.usage & GPUBufferUsage.COPY_SRC)) {
        return "not copyable";
    }
    if (r.offset < 0 || r.bytes <= 0 || r.offset % 4 || r.bytes % 4 || r.offset + r.bytes > r.buffer.size) {
        return `outside the buffer (${r.offset}+${r.bytes} of ${r.buffer.size})`;
    }
    return null;
}

// The parts' words (a Uint32Array each, or a string saying why not), read
// in one copy into a buffer of their own: a texel at a 256-byte boundary,
// as copyTextureToBuffer asks. Errors go to the caller, not the page's
// error box (an error scope).
export async function readParts(device, parts) {
    const places = [];
    let size = 0;
    for (const r of parts) {
        if (unreadable(r)) {
            places.push(-1);
            continue;
        }
        size = Math.ceil(size / (r.texture ? 256 : 4)) * (r.texture ? 256 : 4);
        places.push(size);
        size += r.bytes;
    }
    if (size === 0) {
        return parts.map(unreadable);
    }
    device.pushErrorScope("validation");
    const staging = device.createBuffer({ label: "devtools", size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder({ label: "devtools" });
    parts.forEach((r, k) => {
        if (places[k] < 0) {
            return;
        }
        if (r.texture) {
            enc.copyTextureToBuffer({ texture: r.texture, origin: [r.x, r.y] }, { buffer: staging, offset: places[k] }, [1, 1]);
        } else {
            enc.copyBufferToBuffer(r.buffer, r.offset, staging, places[k], r.bytes);
        }
    });
    device.queue.submit([enc.finish()]);
    const error = await device.popErrorScope();
    if (error) {
        staging.destroy();
        throw new Error(error.message);
    }
    await staging.mapAsync(GPUMapMode.READ);
    const all = staging.getMappedRange().slice(0);
    staging.destroy();
    return parts.map((r, k) => (places[k] < 0 ? unreadable(r) : new Uint32Array(all, places[k], r.bytes / 4)));
}

// ------------------------------------------------------------ where a cell's words are

// Cell (x, y)'s index in each of the world's grids, and where the buffers
// laid out by hand keep its words (each from the shader function named).
function placeOf(sim, x, y) {
    const W = sim.width;
    const H = sim.height;
    const n = W * H;
    const tiles = sim.roomTiles;
    const roomBytes = tiles[0] * tiles[1] * ROOM_TILE * ROOM_TILE / 4;
    const pieces = tiles[0] * tiles[1] * ROOM_PIECES;
    const vent = W + roomBytes + pieces;
    const zones = vent + 2 * W;
    const wet = zones + 2 * ZONE_WORDS * pieces;
    const tile = Math.floor(y / ROOM_TILE) * tiles[0] + Math.floor(x / ROOM_TILE);
    // (bodies.wgsl's structureSlot, common.wgsl's structureBase.)
    const structure = sim.bodyHeader + (2 + BODY_SUM_WORDS) * n;
    const lx = Math.floor(x / LAYOUT.LIGHT_CELL);
    const ly = Math.floor(y / LAYOUT.LIGHT_CELL);
    return {
        x, y, W, H, n, i: y * W + x,
        air: Math.floor(y / AIR_CELL) * sim.airWidth + Math.floor(x / AIR_CELL),
        lx, ly, light: ly * sim.lightSize[0] + lx,
        block: Math.floor(y / FLUID_BLOCK) * Math.ceil(W / FLUID_BLOCK) + Math.floor(x / FLUID_BLOCK),
        // Its faces, left, right, top and bottom (fluid.wgsl's uFace, vFace).
        faces: [y * (W + 1) + x, y * (W + 1) + x + 1, (W + 1) * H + y * W + x, (W + 1) * H + (y + 1) * W + x],
        // Its corners' nodes (stress.wgsl: (W + 1) × (H + 1) of them).
        corners: [[0, 0], [1, 0], [0, 1], [1, 1]],
        nodes: [[0, 0], [1, 0], [0, 1], [1, 1]].map(([dx, dy]) => (y + dy) * (W + 1) + x + dx),
        // stress.wgsl's failSlot: after the stats, the CG's scalars, and per
        // tile its partial sums, whether it is live and the list of them.
        failures: STRESS_STATS + STRESS_SCALARS + 8 * sim.stressTiles[0] * sim.stressTiles[1],
        // common.wgsl's rooms: per cell a byte (its piece of its tile), the
        // pieces' links (union-find), the ventilation, the zones' slots; then
        // the wet rooms' bytes, links and slots, each of two by step parity.
        rooms: { bytes: W, links: W + roomBytes, slots: zones, words: ZONE_WORDS },
        wetRooms: { bytes: wet, links: wet + roomBytes, slots: wet + roomBytes + pieces, words: WET_WORDS },
        vent, pieces, tile,
        byteWord: tile * ROOM_TILE * ROOM_TILE / 4 + (y % ROOM_TILE) * ROOM_TILE / 4 + Math.floor((x % ROOM_TILE) / 4),
        structure: structure + 1 + sim.bodyTiles,
    };
}

// ------------------------------------------------------------ what a cell's words mean

// Each section of the inspector: the parts it reads (from the place, and
// for those of the second read, what the first found), and its rows, each
// [label, first word, words, what they say]. Its words are its parts'
// words one after another.
const CELL_SECTIONS = [
    {
        key: "cell", group: "Cell", title: "Cell",
        parts: (p, sim) => [span(sim.cells[sim.cur], `cells[${sim.cur}]`, p.i * CELL_BYTES, CELL_BYTES)],
        rows: (u, f, p, sim) => {
            const c = cellFields(u, f, 0);
            const e = ELEMENTS[c.type];
            const drawn = e?.kind === KIND.LIQUID && c.flags & CELL_FLAGS.DRAWN;
            const roles = Object.entries(cellRecord(c)).map(([k, v]) => `${k} ${v}`).join(" · ");
            const loose = e?.kind === KIND.SOLID && c.flags & CELL_FLAGS.LOOSE;
            return [
                ["type", 0, 1, `${elementName(c.type)}${e ? ` (${KIND_NAMES[e.kind]})` : ""}`],
                ["aux, life, shade", 0, 1, `aux ${c.aux} · life ${c.life} · shade ${c.shade}`],
                ["flags", 0, 1, `${c.flags}: ${cellFlagNames(c.flags)}`],
                ["roles", 0, 1, roles || "none (cellrecord.js)"],
                ["temperature", 1, 1, `${num(c.temp)} °C stored (past a phase point it holds the latent heat)`],
                ["velocity", 2, 1, `${halves(u[2])} cells/step`],
                drawn
                    ? ["liquid home", 3, 1, `home ${at(sim, u[3] >>> 8)}, ${u[3] & 0xff} of ${LIQUID_PER_CELL} particles (liquidHome)`]
                    : ["frac", 3, 1, `${halves(u[3])}${loose ? ` (body slot ${half(u[3])}, piece ${half(u[3] >>> 16)})` : ""}`],
            ];
        },
    },
    {
        key: "gas", group: "Cell", title: "Gas plane",
        parts: (p, sim) => [span(sim.cells[sim.cur], `cells[${sim.cur}] gas`, p.n * CELL_BYTES + p.i * GAS_BYTES, GAS_BYTES)],
        rows: (u) => [
            ["vapour, mist", 0, 1, `vapour ${half(u[0])} · mist ${half(u[0] >>> 16)}`],
            ["oxygen lost, amount", 1, 1, `lost ${half(u[1])} · amount ${half(u[1] >>> 16)} (1: a cell of air at 1 atm, 20 °C)`],
        ],
    },
    {
        key: "occupant", group: "Cell", title: "Occupant",
        parts: (p, sim) => [span(sim.occupants, "occupants", p.i * 4, 4)],
        rows: (u) => {
            const w = u[0];
            if ((w & 0xff) === 0) {
                return [["person", 0, 1, "none"]];
            }
            const vx = ((w << 12) | 0) >> 20;
            const vy = (w | 0) >> 20;
            return [["person", 0, 1, `slot ${(w & 0xff) - 1}, moving ${num(vx / 128)}, ${num(vy / 128)} cells/step`]];
        },
    },
    {
        key: "air", group: "Air", title: "Air cell",
        parts: (p, sim) => [span(sim.air[0], "air", p.air * LAYOUT.AIR_BYTES, LAYOUT.AIR_BYTES)],
        rows: (u, f) => [
            ["pressure", 0, 1, `${num(f[0])} (gauge, ${P_ATM} an atmosphere: ${num(f[0] / P_ATM)} atm)`],
            ["right face", 1, 1, `${num(f[1])} cells/step`],
            ["bottom face", 2, 1, `${num(f[2])} cells/step`],
            ["temperature", 3, 1, f[3] <= AIR_BLOCKED ? "no gas (AIR_BLOCKED)" : `${num(f[3])} °C`],
        ],
    },
    {
        key: "airCg", group: "Air", title: "Air's projection",
        parts: (p, sim) => [span(sim.airCg, "airCg", p.air * 16, 16)],
        rows: (u, f) => [
            ["potential", 0, 1, num(f[0])], ["residual", 1, 1, num(f[1])], ["direction", 2, 1, num(f[2])], ["A·p", 3, 1, num(f[3])],
        ],
    },
    {
        key: "airSource", group: "Air", title: "Air source (exact sums)",
        parts: (p, sim) => [span(sim.airSource, "airSource", p.air * 64, 64)],
        rows: (u) => {
            const sum = (k) => num(exactSum(u[2 * k], u[2 * k + 1], EXACT_FINE));
            return [
                ["right face", 0, 2, `${sum(0)} cells/step added`], ["bottom face", 2, 2, `${sum(1)} cells/step added`],
                ["amount", 4, 2, sum(2)], ["vapour", 6, 2, sum(3)], ["mist", 8, 2, sum(4)], ["oxygen lost", 10, 2, sum(5)],
                ["heat", 12, 2, `${sum(6)} J`], ["(unused)", 14, 2, ""],
            ];
        },
    },
    {
        key: "blast", group: "Air", title: "Blast layer",
        parts: (p, sim) => [span(sim.blast, "blast", p.air * 16, 16)],
        rows: (u, f) => [
            ["overpressure", 0, 1, num(f[0])], ["right face", 1, 1, `${num(f[1])} cells/step`],
            ["bottom face", 2, 1, `${num(f[2])} cells/step`], ["w", 3, 1, num(f[3])],
        ],
    },
    {
        key: "heat", group: "Heat", title: "Heat in, load",
        parts: (p, sim) => [span(sim.heatIn, "heatIn", p.i * 8, 8), span(sim.loads[sim.cur], `loads[${sim.cur}]`, p.i * 4, 4)],
        rows: (u, f) => [
            ["heat coming in", 0, 2, `${num(exactSum(u[0], u[1], EXACT_FINE))} J (heatin.wgsl)`],
            ["pile's load", 2, 1, num(f[2])],
        ],
    },
    {
        key: "clock", group: "Heat", title: "Heat's clock (light cell)",
        parts: (p, sim) => [texel(sim.clockTex, "clock", p.lx, p.ly), span(sim.clockHeated, "clockHeated", p.light * 4, 4)],
        rows: (u, f) => [
            ["realness", 0, 1, `${num(f[0])} (0 the day's clock, 1 real time)`],
            ["current's heat", 1, 1, `${num(f[1])} (how real-time the current's heat made it)`],
        ],
    },
    {
        key: "radiant", group: "Heat", title: "Radiant heat",
        parts: (p, sim) => {
            const k = 1 - sim.radiantNow;
            const taken = sim.radiantTaken[k];
            return [
                texel(sim.radiantTex, "radiant", p.x, p.y),
                span(taken, `radiantTaken${k}`, p.i * 4, 4), span(taken, `radiantTaken${k} reach`, (p.n + p.i) * 4, 4),
                span(taken, `radiantTaken${k} faces`, (2 * p.n + 4 * p.i) * 4, 16), span(taken, `radiantTaken${k} peak`, (6 * p.n + p.i) * 4, 4),
            ];
        },
        rows: (u, f) => [
            ["took in", 0, 1, num(f[0])], ["emitting area", 1, 1, num(f[1])], ["thermal radiance", 2, 1, num(f[2])],
            ["mean radiance", 3, 1, `${num(f[3])} · peak realness ${u[3] & 0xff} of 255 (low mantissa bits)`],
            ["taken (fixed)", 4, 1, num(u[4] / RAD_FIXED)], ["reached (fixed)", 5, 1, num(u[5] / RAD_FIXED)],
            ["faces reflect", 6, 4, `right ${num(u[6] / RAD_FIXED)} · left ${num(u[7] / RAD_FIXED)} · down ${num(u[8] / RAD_FIXED)} · up ${num(u[9] / RAD_FIXED)}`],
            ["peak", 10, 1, `${u[10]}`],
        ],
    },
    {
        key: "light", group: "Light", title: "Light (light cell)",
        parts: (p, sim) => [
            texel(sim.lightTex, "light", p.lx, p.ly), texel(sim.sunTex, "sun", p.lx, p.ly), texel(sim.heatTex, "heat", p.lx, p.ly),
            texel(sim.gravityTex, "gravity", p.lx, p.ly),
        ],
        rows: (u, f) => [
            ["light r, g", 0, 1, halves(u[0])], ["light b, a", 1, 1, halves(u[1])],
            ["direct sun", 2, 1, `${num(f[2])} (1 full sunlight square to it)`],
            ["heat (texture)", 3, 1, num(f[3])],
            ["gravity's pull", 4, 2, `${num(f[4])}, ${num(f[5])} cells/step²`], ["gravity z, w", 6, 2, `${num(f[6])}, ${num(f[7])}`],
        ],
    },
    {
        key: "lightPacked", group: "Light", title: "Light, packed (light cell)",
        parts: (p, sim) => [
            texel(sim.lightScene, "lightScene", p.lx, p.ly), texel(sim.quartersTex, "quarters", p.lx, p.ly),
            texel(sim.mirrorTex, "mirror", p.lx, p.ly),
        ],
        rows: (u) => ["scene", "quarters", "mirror"].flatMap((name, k) => [0, 1, 2, 3].map((c) => [`${name}.${"xyzw"[c]}`, 4 * k + c, 1, `f16s ${halves(u[4 * k + c])}`])),
    },
    {
        key: "electric", group: "Current", title: "Current",
        parts: (p, sim) => [
            span(sim.volts, "volts", p.i * 4, 4), span(sim.flow, "flow", p.i * LAYOUT.FLOW_BYTES, LAYOUT.FLOW_BYTES),
            span(sim.electricWork, "electricWork", p.i * ELECTRIC_WORK * 4, ELECTRIC_WORK * 4),
        ],
        rows: (u, f) => [
            ["potential", 0, 1, `${num(f[0])} V`], ["current density", 1, 1, halves(u[1])],
            ["conductance", 2, 1, num(f[2])], ["EMF", 3, 1, `${halves(u[3])} V/cell`],
            ["aggregate slot", 4, 1, u[4] === NO_SLOT ? "none" : String(u[4])],
            ["r, z", 5, 2, `${num(f[5])}, ${num(f[6])}`], ["p (two)", 7, 2, `${num(f[7])}, ${num(f[8])}`], ["q", 9, 1, num(f[9])],
            ["effect", 10, 1, num(f[10])], ["links", 11, 1, `${(u[11] >>> 0).toString(2).padStart(8, "0")} (two bits a link)`],
        ],
    },
    {
        key: "stress", group: "Structure", title: "Stress",
        parts: (p, sim) => [
            span(sim.stressWork, "stressWork failure", (p.failures + p.i) * 4, 4),
            span(sim.stressWork, "stressWork streak", (p.failures + p.n + p.i) * 4, 4),
            ...p.nodes.map((node, k) => span(sim.stressNodes, `stressNodes corner ${p.corners[k]}`, node * STRESS_NODE_WORDS * 16, STRESS_NODE_WORDS * 16)),
            ...p.nodes.map((node, k) => span(sim.stressDisp, `stressDisp corner ${p.corners[k]}`, node * 16, 16)),
        ],
        rows: (u, f, p) => [
            ["failure index", 0, 1, `${num(f[0])} (1 at its strength)`],
            ["past it", 1, 1, `${u[1]} judgements running`],
            ...p.corners.flatMap(([dx, dy], k) => {
                const a = 2 + 16 * k;
                const name = `node (${p.x + dx}, ${p.y + dy})`;
                return [
                    [`${name} directions`, a, 4, `${num(f[a])}, ${num(f[a + 1])} · ${num(f[a + 2])}, ${num(f[a + 3])}`],
                    [`${name} r, z`, a + 4, 4, `${num(f[a + 4])}, ${num(f[a + 5])} · ${num(f[a + 6])}, ${num(f[a + 7])}`],
                    [`${name} K·p, load`, a + 8, 4, `${num(f[a + 8])}, ${num(f[a + 9])} · ${num(f[a + 10])}, ${num(f[a + 11])}`],
                    [`${name} M⁻¹, flags`, a + 12, 4, `${num(f[a + 12])}, ${num(f[a + 13])}, ${num(f[a + 14])} · ${bitNames(u[a + 15], NODE_FLAGS)}`],
                ];
            }),
            ...p.corners.map(([dx, dy], k) => {
                const a = 66 + 4 * k;
                return [`displacement (${p.x + dx}, ${p.y + dy})`, a, 4,
                    `${num(exactSum(u[a + 1], u[a], DISP_UNIT))}, ${num(exactSum(u[a + 3], u[a + 2], DISP_UNIT))} cells`];
            }),
        ],
    },
    {
        key: "bodyWords", group: "Structure", title: "Bodies' and structure's words",
        parts: (p, sim) => [
            span(sim.bodies, "bodies label", (sim.bodyHeader + p.i) * 4, 4),
            span(sim.bodies, "bodies claim", (sim.bodyHeader + p.n + p.i) * 4, 4),
            span(sim.bodies, "bodies sums", (sim.bodyHeader + 2 * p.n + BODY_SUM_WORDS * p.i) * 4, BODY_SUM_WORDS * 4),
            span(sim.bodies, "structure label", (p.structure + p.i) * 4, 4),
            span(sim.bodies, "structure anchor", (p.structure + p.n + p.i) * 4, 4),
            span(sim.bodies, "structure break", (p.structure + 2 * p.n + p.i) * 4, 4),
            span(sim.bodies, "swept (even, odd)", (p.structure + 3 * p.n + 2 * p.i) * 4, 8),
        ],
        rows: (u, f, p, sim) => {
            const claim = (w) => (w === 0 ? "none" : `step ${w >>> 12} · priority ${(w >>> 1) & 0x7ff} · bit ${w & 1}`);
            // (A swept word is the step's that last wrote it: the last step's
            // is the one the liquid reads, an older one is history.)
            const now = sim.frameNo % CLAIM_PERIOD + 1;
            const swept = (w) => (w === 0 ? "none" : `step ${w >>> 12}${(w >>> 12) === now ? " (the last step's)" : ` (${now - (w >>> 12)} steps ago)`}`
                + ` · body ${FREE_BODY - ((w >>> 1) & 0x7ff)}${w & 1 ? " · seam" : ""}`);
            const b = BODY_SUM_WORDS;
            return [
                ["label", 0, 1, u[0] === NONE ? "none" : at(sim, u[0])],
                ["claim", 1, 1, claim(u[1])],
                ["sums (root)", 2, b, `${b} words, as summed for a root`],
                ["structure label", 2 + b, 1, u[2 + b] === NONE ? "none" : at(sim, u[2 + b])],
                ["anchored root", 3 + b, 1, String(u[3 + b])],
                ["break", 4 + b, 1, u[4 + b] === 0 ? "none" : `step ${u[4 + b]}`],
                ["swept, even steps", 5 + b, 1, swept(u[5 + b])],
                ["swept, odd steps", 6 + b, 1, swept(u[6 + b])],
            ];
        },
    },
    {
        key: "field", group: "Liquid", title: "Liquid grid",
        parts: (p, sim) => [
            span(sim.field, "field", p.i * LAYOUT.FIELD_CELL_BYTES, LAYOUT.FIELD_CELL_BYTES),
            ...p.faces.map((face, k) => span(sim.field, `field ${["left", "right", "top", "bottom"][k]} face`, (2 * p.n + face) * LAYOUT.FIELD_FACE_BYTES, LAYOUT.FIELD_FACE_BYTES)),
        ],
        rows: (u, f, p, sim) => {
            const z = f[6];
            const sealed = z === 1 ? "sealed" : z < 0 ? (-1 - z >= MAX_BODIES ? `held body ${-1 - z - MAX_BODIES}` : `body ${-1 - z}`) : "open";
            const b = f[7] >>> 0;
            const parity = sim.frameNo & 1;
            const bit = (k, back) => ((b >>> (k + ((parity + back) & 1))) & 1 ? "yes" : "no");
            const held = (h) => (h === 2 ? "projected" : h === 1 ? "splat" : h === 0 ? "open air" : h < 0 ? "obstacle" : "extrapolated");
            return [
                ["pressure", 0, 1, num(f[0])], ["divergence", 1, 1, num(f[1])],
                ["density", 2, 1, `${num(f[2])} (> 0 fluid, 0 air, < 0 obstacle)`], ["viscosity", 3, 1, num(f[3])],
                ["drift", 4, 1, num(f[4])], ["drift's spread", 5, 1, num(f[5])], ["sealed, body", 6, 1, `${num(z)}: ${sealed}`],
                ["bits", 7, 1, `${b}: liquid this step ${bit(0, 0)}, last ${bit(0, 1)} · falling this ${bit(2, 0)}, last ${bit(2, 1)}`],
                ...["left", "right", "top", "bottom"].map((side, k) => {
                    const a = 8 + 4 * k;
                    return [`${side} face`, a, 4, `${num(f[a])} cells/step · before ${num(f[a + 1])} · ${held(f[a + 2])} (${num(f[a + 2])}) · ${num(f[a + 3])}`];
                }),
            ];
        },
    },
    {
        key: "counters", group: "Liquid", title: "Liquid counters",
        parts: (p, sim) => [
            span(sim.counters, "counters", p.i * LIQUID_CELL_WORDS * 4, LIQUID_CELL_WORDS * 4),
            ...p.faces.map((face, k) => span(sim.counters, `face sums ${["left", "right", "top", "bottom"][k]}`, (LIQUID_CELL_WORDS * p.n + 2 * face) * 4, 8)),
        ],
        rows: (u, f, p, sim) => {
            const dest = (w) => {
                if (w === DEST.KEEP) {
                    return "keep (not drawn)";
                }
                if (w === 0) {
                    return "nowhere (gone)";
                }
                if (w & DEST.ASIDE) {
                    return `aside: ${at(sim, w & ~DEST.ASIDE)}`;
                }
                return `${w & DEST.CELL ? "cell" : "?"} ${at(sim, DEST.SPAN - (w & DEST.SPAN))}`;
            };
            const t = u[G.TYPE];
            const a = LIQUID_CELL_WORDS;
            return [
                ["gathered", G.COUNT, 1, `${u[G.COUNT]} particles`],
                ["taken, came, region", G.TAKEN, 1, `${hex(u[G.TAKEN])} (claim, or came pressing 1 / falling 2, or region)`],
                ["run", G.SORT, 2, `${u[G.SORT]} particles from ${u[G.START]} in sorted`],
                ["next step", G.DEST, 1, dest(u[G.DEST])],
                ["drawn as", G.TYPE, 1, `${elementName(t & 0xff)} · aux ${(t >>> 8) & 0xff}`],
                ["share", G.DENSITY, 1, num(u[G.DENSITY] / (FIX * LIQUID_PER_CELL))],
                ["densities", G.MASS, 1, num(u[G.MASS] / FIX)],
                ["gas goes to", G.AIR_TO, 1, u[G.AIR_TO] === NONE ? "nowhere" : at(sim, u[G.AIR_TO])],
                ["rim's pressure, drift", G.RIM, 2, `${num(f[G.RIM])}, ${num(f[G.RIM + 1])}`],
                ...["left", "right", "top", "bottom"].map((side, k) => [`${side} face's splat`, a + 2 * k, 2,
                    `momentum ${num((u[a + 2 * k] | 0) / FIX)} · weight ${num(u[a + 2 * k + 1] / FIX)}`]),
            ];
        },
    },
    {
        key: "fluidCg", group: "Liquid", title: "Liquid's solve, block",
        parts: (p, sim) => [
            span(sim.fluidCg, "fluidCg", p.i * 16, 16),
            span(sim.blocks, "blocks marks", p.block * 4, 4), span(sim.blocks, "blocks state", (sim.blockCount + p.block) * 4, 4),
        ],
        rows: (u, f, p) => [
            ["pressure", 0, 1, num(f[0])], ["residual", 1, 1, num(f[1])], ["direction", 2, 1, num(f[2])], ["A·p", 3, 1, num(f[3])],
            ["block marks", 4, 1, `block ${p.block}: ${bitNames(u[4], BLOCK_MARKS)}`], ["block state", 5, 1, bitNames(u[5], BLOCK_STATE)],
        ],
    },
    {
        key: "sky", group: "Rooms", title: "Sky and rooms",
        parts: (p, sim) => [
            span(sim.sky, "skyline", p.x * 4, 4), span(sim.sky, "ventilation", (p.vent + p.x) * 4, 4),
            span(sim.sky, "ventilation (its pass's)", (p.vent + p.W + p.x) * 4, 4),
            span(sim.sky, "room byte", (p.rooms.bytes + p.byteWord) * 4, 4), span(sim.sky, "wet room byte", (p.wetRooms.bytes + p.byteWord) * 4, 4),
        ],
        rows: (u, f, p) => {
            const piece = (w) => {
                const b = (w >>> (8 * (p.x % 4))) & 0xff;
                return b === 255 ? "none" : b >= ROOM_PIECES ? `unnumbered (${b})` : `piece ${b} of tile ${p.tile}`;
            };
            return [
                ["skyline", 0, 1, `column ${p.x}: the first row not gas ${u[0] & 0xffff} · the open air ends at row ${u[0] >>> 16}`],
                ["ventilation", 1, 2, `in the wind above row ${u[1] | 0} (its pass's scratch ${num(f[2])})`],
                ["room", 3, 1, `${piece(u[3])} (byte ${p.x % 4})`], ["wet room", 4, 1, `${piece(u[4])} (byte ${p.x % 4})`],
            ];
        },
    },
];

// The second read: what the first found the cell's words to point to.
const FOUND_SECTIONS = [
    {
        key: "particles", group: "Liquid", title: "Particles (its run in sorted)",
        parts: (p, sim, first) => {
            const c = first.counters?.u;
            const count = c ? Math.min(c[G.SORT], MAX_PARTICLES) : 0;
            const start = c?.[G.START] ?? 0;
            if (count === 0 || start + count > sim.particleSlots) {
                return null;
            }
            const region = (k) => LAYOUT.PARTICLES_HEADER + 16 * (k * sim.particleSlots + start);
            return [
                span(sim.particles[0], "particles0 motion", region(0), 16 * count),
                span(sim.particles[0], "particles0 state", region(1), 16 * count),
                span(sim.particles[0], "particles0 affine", region(2), 16 * count),
            ];
        },
        // (Read as three runs, motion, state and affine; shown a particle at
        // a time.)
        rows: (u, f, p, sim, first) => {
            const c = first.counters.u;
            const count = Math.min(c[G.SORT], MAX_PARTICLES);
            const rows = [];
            for (let k = 0; k < count; k++) {
                const m = 4 * k;
                const s = 4 * (count + k);
                const a = 4 * (2 * count + k);
                const word = u[s];
                rows.push(
                    [`#${c[G.START] + k} at`, m, 2, `${num(f[m])}, ${num(f[m + 1])}`],
                    [`#${c[G.START] + k} velocity`, m + 2, 2, `${num(f[m + 2])}, ${num(f[m + 3])} cells/step`],
                    [`#${c[G.START] + k} word`, s, 1, `${elementName(word & 0xff)} · aux ${(word >>> 8) & 0xff} · life ${(word >>> 16) & 0xff}`],
                    [`#${c[G.START] + k} temp, id, rank`, s + 1, 3, `${num(f[s + 1])} °C · id ${hex(u[s + 2])} · rank ${u[s + 3]}`],
                    [`#${c[G.START] + k} affine`, a, 4, `${num(f[a])}, ${num(f[a + 1])}, ${num(f[a + 2])}, ${num(f[a + 3])}`],
                );
            }
            if (c[G.SORT] > count) {
                rows.push(["", 0, 0, `and ${c[G.SORT] - count} more`]);
            }
            return rows;
        },
    },
    {
        key: "body", group: "Bodies", title: "Body",
        parts: (p, sim, first) => {
            const b = bodyAt(first, sim);
            return b === null ? null : bodyParts(sim, b);
        },
        rows: (u, f, p, sim, first) => bodyRows(u, f, bodyAt(first, sim)),
    },
    {
        key: "person", group: "People", title: "Person",
        // (The one whose body the occupant word names, or failing that, as
        // the hover readout finds one, whose figure is drawn over the cell:
        // the dead and sleeping lie on the ground, in no cells.)
        parts: (p, sim, first) => {
            const w = first.occupant?.u[0] ?? 0;
            const slot = (w & 0xff) !== 0 ? (w & 0xff) - 1 : sim.personAt(p.x, p.y)?.slot;
            return slot === undefined ? null : [span(sim.people, `people slot ${slot}`, slot * LAYOUT.PERSON_BYTES, LAYOUT.PERSON_BYTES)];
        },
        rows: (u, f) => personRows(u, f),
    },
    {
        key: "zone", group: "Rooms", title: "Zone (this step's slot)",
        parts: (p, sim, first) => roomSlot(p, sim, first, "rooms", "roomLinks", 3),
        rows: (u, f, p, sim, first) => {
            // (common.wgsl's ZONE_* words: sums in 1 / ZONE_SCALE, 1024,
            // and exact ones, air.wgsl's WIDE_SCALE and POTENTIAL_SCALE,
            // 1024.)
            const room = roomOf(p, first, "rooms", "roomLinks", 3);
            const pair = (k, scale) => num(exactSum(u[k], u[k + 1], scale));
            return [
                ["room", 0, 0, `root piece ${room}`],
                ["amount, volume", 0, 2, `${num(u[0] / 1024)}, ${num(u[1] / 1024)}: ${num(u[0] / Math.max(u[1], 1))} atm`],
                ["cells", 2, 1, String(u[2])], ["open", 3, 1, bitNames(u[3], [[1, "outside"], [2, "sink"]])],
                ["source", 4, 2, pair(4, WIDE_SCALE)], ["sinks draw", 6, 1, num(u[6] / 1024)], ["potential", 7, 2, pair(7, 1024)],
                ...["ΣC", "ΣCT", "ΣCh", "ΣCTh", "ΣCh²"].map((name, k) => [name, 9 + 2 * k, 2, pair(9 + 2 * k, EXACT_COARSE)]),
                ["mixing ΣC, ΣCT", 19, 4, `${pair(19, EXACT_FINE)}, ${pair(21, EXACT_FINE)}`],
                ["pinned cell", 23, 1, u[23] === 0 ? "none" : at(sim, ~u[23] >>> 0)],
            ];
        },
    },
    {
        key: "wetRoom", group: "Rooms", title: "Wet room (this step's slot)",
        parts: (p, sim, first) => roomSlot(p, sim, first, "wetRooms", "wetLinks", 4),
        rows: (u, f, p, sim, first) => {
            // (common.wgsl's WET_* words.)
            const room = roomOf(p, first, "wetRooms", "wetLinks", 4);
            const pair = (k) => num(exactSum(u[k], u[k + 1], EXACT_FINE));
            return [
                ["room", 0, 0, `root piece ${room}`],
                ["open", 0, 1, String(u[0])], ["most cells", 1, 1, String(u[1])],
                ["surface", 2, 1, u[2] === 0 ? "none" : `room ${u[2] - 1}`], ["takers", 3, 1, String(u[3])],
                ...["amount", "vapour", "mist", "oxygen used", "heat"].map((name, k) => [`given: ${name}`, 4 + 2 * k, 2, pair(4 + 2 * k)]),
                ["givers' weight", 14, 2, pair(14)],
            ];
        },
    },
];

// The links of each kind of room, read whole with the first read (a word a
// piece: a few tens of KB), to find a cell's room's root.
function linkParts(p, sim) {
    return {
        roomLinks: span(sim.sky, "room links", p.rooms.links * 4, p.pieces * 4),
        wetLinks: span(sim.sky, "wet room links", p.wetRooms.links * 4, p.pieces * 4),
    };
}

// The root piece of the cell's room (rooms.wgsl's union-find), or null for
// none.
function roomOf(p, first, kind, links, byteRow) {
    const w = first.sky?.u[byteRow];
    const l = first[links]?.u;
    if (w === undefined || !l) {
        return null;
    }
    const b = (w >>> (8 * (p.x % 4))) & 0xff;
    if (b >= ROOM_PIECES) {
        return null;
    }
    let k = p.tile * ROOM_PIECES + b;
    for (let guard = 0; l[k] < k && guard < p.pieces; guard++) {
        k = l[k];
    }
    return k;
}

function roomSlot(p, sim, first, kind, links, byteRow) {
    const room = roomOf(p, first, kind, links, byteRow);
    if (room === null) {
        return null;
    }
    const r = p[kind];
    return [span(sim.sky, `${kind === "rooms" ? "zone" : "wet room"} slot ${room}`, (r.slots + r.words * ((sim.frameNo % 2) * p.pieces + room)) * 4, r.words * 4)];
}

// The body whose cell this is: the slot a loose cell of structure names, or
// the body that moved over it in the last step (an older swept word is
// history: weird-water.json's cell named a body that had swept it 1300 steps
// before).
function bodyAt(first, sim) {
    const u = first.cell?.u;
    if (u) {
        const c = cellFields(u, new Float32Array(u.buffer, u.byteOffset, 4), 0);
        const e = ELEMENTS[c.type];
        const slot = f16Value(u[3] & 0xffff);
        if (e?.kind === KIND.SOLID && c.flags & CELL_FLAGS.LOOSE && slot < MAX_BODIES && Number.isInteger(slot)) {
            return slot;
        }
    }
    const w = first.bodyWords?.u;
    if (w) {
        const key = w[5 + BODY_SUM_WORDS + (sim.frameNo & 1)];
        if (key !== 0 && key >>> 12 === sim.frameNo % CLAIM_PERIOD + 1) {
            const b = FREE_BODY - ((key >>> 1) & 0x7ff);
            return b < MAX_BODIES ? b : null;
        }
    }
    return null;
}

// Body b's table row (bodies.wgsl's tableSlot), its push sums (fluid.wgsl's
// pushSlot) and its liquid words (liquidSlot).
function bodyParts(sim, b) {
    const liquid = MAX_BODIES * (1 + BODY_TABLE_WORDS) + BODY_CONTACTS * BODY_CONTACT_WORDS + b * BODY_LIQUID_WORDS;
    return [
        span(sim.bodyMotion, `bodyMotion table ${b}`, (MAX_BODIES + b * BODY_TABLE_WORDS) * 4, BODY_TABLE_WORDS * 4),
        span(sim.bodyPushes, `bodyPushes ${b}`, b * BODY_PUSH_WORDS * 4, BODY_PUSH_WORDS * 4),
        span(sim.bodyMotion, `bodyMotion liquid ${b}`, liquid * 4, BODY_LIQUID_WORDS * 4),
    ];
}

// bodies.wgsl's T_* words, then the pushes and the liquid's words.
function bodyRows(u, f, b) {
    const fs = (k, n) => Array.from({ length: n }, (_, j) => num(f[k + j])).join(", ");
    const is = (k, n) => Array.from({ length: n }, (_, j) => String(u[k + j] | 0)).join(", ");
    const T = BODY_TABLE_WORDS;
    const moved = Math.min(u[50], 24);
    const rows = [
        ["slot", 0, 0, String(b)],
        ["position", 0, 2, `${fs(0, 2)} (centre of mass, as stamped)`], ["angle", 2, 1, `${num(f[2])} rad`],
        ["velocity", 3, 2, `${fs(3, 2)} cells/step`], ["spin", 5, 1, `${num(f[5])} rad/step`],
        ["mass", 6, 1, `${num(f[6])} cells of water`], ["inertia", 7, 1, num(f[7])],
        ["centre in shape", 8, 2, fs(8, 2)], ["radius", 10, 1, num(f[10])],
        ["shape", 11, 2, `from ${u[11]} in the pool, ${u[12] & 0xffff} × ${u[12] >>> 16}`], ["cells", 13, 1, String(u[13])],
        ["box", 14, 4, is(14, 4)], ["flags", 18, 1, bitNames(u[18], BODY_FLAGS)], ["stamped at step", 19, 1, String(u[19])],
        ["piece", 20, 1, String(u[20])], ["at rest", 21, 1, `${u[21]} steps`],
        ["air's push, torque", 22, 3, fs(22, 3)], ["subcell", 25, 1, halves(u[25])],
        ["liquid's change", 26, 4, `${fs(26, 3)} · ${num(f[29])} cells in the solve`],
        ["next pose", 30, 3, fs(30, 3)], ["next velocity", 33, 3, fs(33, 3)], ["new box", 36, 4, is(36, 4)],
        ["impact", 40, 3, `speed² ${num(f[40])} · normal ${fs(41, 2)}`],
        ["stamp", 43, 5, `anchor ${is(43, 2)} · ${u[45]} quarter turns · shears ${fs(46, 2)}`], ["pivot", 48, 2, is(48, 2)],
        ["moved cells", 50, 1, String(u[50])],
        ...Array.from({ length: moved }, (_, k) => [`moved #${k}`, 51 + 2 * k, 2, `shape ${u[51 + 2 * k]} → world ${u[52 + 2 * k]}`]),
        ["overlap push", 99, 3, fs(99, 3)], ["support", 102, 1, bitNames(u[102], SUPPORT)], ["contacts", 103, 1, String(u[103])],
        ["(104)", 104, 1, ""], ["cargo", 105, 2, `mass ${num(f[105])} · inertia ${num(f[106])}`], ["(107)", 107, 1, ""],
        ["mobility", 108, 6, `${fs(108, 3)} · ${fs(111, 2)} · ${num(f[113])} (xx, xy, xs · yy, ys · ss: as its contacts let it)`],
        ["mobility at", 114, 1, `step ${u[114]}`], ["(115)", 115, 1, ""],
    ];
    const pushes = PUSH_SETS.map((name, s) => {
        const a = T + 6 * s;
        const v = [0, 1, 2].map((k) => num(exactSum(u[a + 2 * k], u[a + 2 * k + 1], EXACT_COARSE)));
        return [`push: ${name}`, a, 6, v.join(", ")];
    });
    const l = T + BODY_PUSH_WORDS;
    const liquid = [
        ["liquid slot", l, 3, `Δv ${num((u[l] | 0) / RIGID_FIX)}, ${num((u[l + 1] | 0) / RIGID_FIX)} · Δspin ${num((u[l + 2] | 0) / RIGID_FIX)}`],
        ["liquid slot, wet", l + 3, 1, u[l + 3] & 0x80000000 ? "wet (LIQUID_WET)" : "dry"],
    ];
    return [...rows, ...pushes, ...liquid];
}

// person.wgsl's struct Person.
function personRows(u, f) {
    const p = {
        seed: u[9], state: u[4], cause: u[15], health: f[7], flags: u[5], breath: f[8],
    };
    return [
        ["", 0, 0, describePerson(p)],
        ["feet", 0, 2, `${num(f[0])}, ${num(f[1])}`], ["velocity", 2, 2, `${num(f[2])}, ${num(f[3])} cells/step`],
        ["state", 4, 1, STATE_NAMES[u[4]] ?? String(u[4])], ["flags", 5, 1, bitNames(u[5], PERSON_FLAGS)],
        ["timer", 6, 1, `${num(f[6])} steps`], ["health, breath", 7, 2, `${num(f[7])}, ${num(f[8])}`],
        ["seed", 9, 1, hex(u[9])], ["home", 10, 1, num(f[10])], ["stride", 11, 1, num(f[11])], ["burn", 12, 1, `${num(f[12])} steps`],
        ["fell from", 13, 1, num(f[13])], ["since", 14, 1, `${num(f[14])} steps`], ["cause", 15, 1, String(u[15])],
        ["skin, core", 16, 2, `${num(f[16])}, ${num(f[17])} °C`], ["burn dose", 18, 1, num(f[18])], ["(pad)", 19, 1, ""],
    ];
}

// ------------------------------------------------------------ the toolbox

export class DevTools {
    // sim: the world now (main.js makes a new one for each size); toCell:
    // page to cell coordinates; cellBox: a cell's square on the page.
    constructor({ device, controls, sim, toCell, cellBox }) {
        this.device = device;
        this.controls = controls;
        this.sim = sim;
        this.toCell = toCell;
        this.cellBox = cellBox;
        this.root = null;      // the panel, once built
        this.open = false;
        this.tab = "cell";
        this.format = "hex";
        this.live = true;      // read again every REFRESH_MS
        this.pinned = null;    // the cell pinned, or null to follow the pointer
        this.result = null;    // the last read: { tab, ... }
        this.error = "";
        this.reading = false;
        this.dirty = true;
        this.lastRead = 0;
        this.stepsLeft = 0;
        this.trace = null;     // { steps, rows, need }: need, a read before the next step
        this.folded = new Set(["lightPacked", "stress", "airCg", "blast"]);
        this.lastTrace = null; // the last trace's reads
        this.bodySlot = null;
        this.personSlot = null;
        window.addEventListener("keydown", (e) => this.onKey(e), true);
        window.addEventListener("pointerdown", (e) => this.onPointer(e), true);
        if (new URLSearchParams(location.search).has("dev")) {
            this.toggle(true);
        }
    }

    toggle(open = !this.open) {
        this.open = open;
        if (open && !this.root) {
            this.build();
        }
        if (this.root) {
            this.root.hidden = !open;
            this.marker.hidden = true;
        }
        if (open) {
            const dev = this;
            window.sand = {
                get sim() {
                    return dev.sim();
                },
                device: this.device, controls: this.controls, devtools: this,
                read: async (buffer, offset = 0, bytes = buffer.size - offset) => (await readParts(this.device, [span(buffer, "", offset, bytes)]))[0],
            };
            this.dirty = true;
        }
    }

    build() {
        const root = this.root = document.createElement("div");
        root.id = "devtools";
        root.innerHTML = `
            <header><span>Dev toolbox</span><span class="dev-status"></span><button data-act="close" title="Close (\`)">×</button></header>
            <nav>${["cell", "world", "bodies", "people"].map((t) => `<button data-tab="${t}">${t}</button>`).join("")}</nav>
            <div class="dev-bar">
                <label>x <input type="number" name="x" min="0"></label>
                <label>y <input type="number" name="y" min="0"></label>
                <button data-act="follow" title="Follow the pointer again (Ctrl+click pins a cell)">follow</button>
                <select name="format" title="How the raw words are shown">${Object.keys(FORMATS).map((k) => `<option>${k}</option>`).join("")}</select>
                <label title="Read again ten times a second"><input type="checkbox" name="live" checked> live</label>
            </div>
            <div class="dev-bar">
                <button data-act="step" title="Pause and step once">step</button>
                <button data-act="step10" title="Pause and step ten times">step 10</button>
                <button data-act="trace" title="Pause, then read the cell after each of these steps">trace</button>
                <input type="number" name="traceSteps" min="1" value="${TRACE_STEPS}" title="Steps to trace">
                <button data-act="copy" title="Copy what is shown, as JSON">copy</button>
                <a class="dev-download" hidden></a>
            </div>
            <div class="dev-body"></div>`;
        document.body.append(root);
        this.marker = document.createElement("div");
        this.marker.className = "dev-marker";
        this.marker.hidden = true;
        document.body.append(this.marker);
        this.body = root.querySelector(".dev-body");
        this.status = root.querySelector(".dev-status");
        this.inputs = { x: root.querySelector("[name=x]"), y: root.querySelector("[name=y]") };
        // Keys typed here are the toolbox's, not the app's shortcuts.
        root.addEventListener("keydown", (e) => {
            if (e.code !== "Backquote") {
                e.stopPropagation();
            }
        });
        root.addEventListener("click", (e) => this.onClick(e));
        root.addEventListener("change", (e) => {
            const t = e.target;
            if (t.name === "format") {
                this.format = t.value;
                this.render();
            } else if (t.name === "live") {
                this.live = t.checked;
            } else if (t.name === "x" || t.name === "y") {
                this.pin(Number(this.inputs.x.value), Number(this.inputs.y.value));
            }
        });
        // A section folded or opened stays so.
        root.addEventListener("toggle", (e) => {
            const key = e.target.dataset?.key;
            if (key) {
                if (e.target.open) {
                    this.folded.delete(key);
                } else {
                    this.folded.add(key);
                }
            }
        }, true);
        this.setTab("cell");
    }

    setTab(tab) {
        this.tab = tab;
        for (const b of this.root.querySelectorAll("nav button")) {
            b.classList.toggle("active", b.dataset.tab === tab);
        }
        this.result = null;
        this.dirty = true;
        this.render();
    }

    onClick(e) {
        const b = e.target.closest("button, [data-body], [data-person]");
        if (!b) {
            return;
        }
        if (b.dataset.tab) {
            this.setTab(b.dataset.tab);
            return;
        }
        if (b.dataset.body !== undefined) {
            this.bodySlot = Number(b.dataset.body);
            this.dirty = true;
            return;
        }
        if (b.dataset.person !== undefined) {
            this.personSlot = Number(b.dataset.person);
            this.dirty = true;
            return;
        }
        const act = b.dataset.act;
        if (act === "close") {
            this.toggle(false);
        } else if (act === "follow") {
            this.pinned = null;
            this.dirty = true;
        } else if (act === "step" || act === "step10") {
            this.pause();
            this.stepsLeft += act === "step" ? 1 : 10;
        } else if (act === "trace") {
            this.startTrace(Math.max(1, Math.floor(Number(this.root.querySelector("[name=traceSteps]").value) || TRACE_STEPS)));
        } else if (act === "copy") {
            this.copy();
        } else if (act === "pinAt") {
            this.pin(Number(b.dataset.x), Number(b.dataset.y));
            this.setTab("cell");
        }
    }

    // Backquote opens and closes it (not while typing in a field); with it
    // open and a cell pinned, the arrow keys move the pin.
    onKey(e) {
        const typing = e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLSelectElement;
        if (e.ctrlKey || e.metaKey || e.altKey || typing) {
            return;
        }
        if (e.code === "Backquote" || e.key === "`") {
            e.preventDefault();
            e.stopPropagation();
            this.toggle();
            return;
        }
        const arrow = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key];
        if (this.open && this.pinned && arrow) {
            e.preventDefault();
            e.stopPropagation();
            this.pin(this.pinned.x + arrow[0], this.pinned.y + arrow[1]);
        }
    }

    // Ctrl+click (⌘+click) on the world pins the cell, in place of painting.
    onPointer(e) {
        if (!this.open || e.target !== this.controls.canvas || !(e.ctrlKey || e.metaKey) || e.button !== 0) {
            return;
        }
        e.preventDefault();
        e.stopPropagation();
        const c = this.toCell(e.clientX, e.clientY);
        this.pin(Math.floor(c.x), Math.floor(c.y));
        if (this.tab !== "cell") {
            this.setTab("cell");
        }
    }

    pin(x, y) {
        const sim = this.sim();
        if (!Number.isFinite(x) || !Number.isFinite(y)) {
            return;
        }
        this.pinned = { x: Math.min(Math.max(Math.round(x), 0), sim.width - 1), y: Math.min(Math.max(Math.round(y), 0), sim.height - 1) };
        this.dirty = true;
    }

    pause() {
        if (!this.controls.state.paused) {
            this.controls.state.paused = true;
            this.controls.sync();
        }
    }

    // The cell inspected: the pinned one (kept in the world), or the
    // pointer's.
    target() {
        const sim = this.sim();
        if (this.pinned) {
            this.pinned.x = Math.min(this.pinned.x, sim.width - 1);
            this.pinned.y = Math.min(this.pinned.y, sim.height - 1);
            return this.pinned;
        }
        return this.controls.hoverCell();
    }

    // Steps of a paused world the toolbox asks for this frame (main.js): a
    // trace's one at a time, each once the step before it has been read.
    takeSteps() {
        if (this.trace) {
            if (this.trace.need || this.reading) {
                return 0;
            }
            this.trace.need = true;
            return 1;
        }
        const n = Math.min(this.stepsLeft, MAX_STEPS_PER_FRAME);
        this.stepsLeft -= n;
        return n;
    }

    // After each frame (main.js): the marker on the pinned cell, and a read
    // if one is due.
    afterFrame(steps) {
        if (!this.open) {
            return;
        }
        this.placeMarker();
        if (this.reading) {
            return;
        }
        if (this.trace) {
            if (this.trace.need) {
                this.refresh();
            }
            return;
        }
        const wait = this.tab === "bodies" || this.tab === "people" ? LIST_REFRESH_MS : REFRESH_MS;
        if (this.dirty || (steps > 0 && this.controls.state.paused) || (this.live && performance.now() - this.lastRead >= wait)) {
            this.refresh();
        }
    }

    placeMarker() {
        const box = this.pinned && this.tab === "cell" ? this.cellBox(this.pinned.x, this.pinned.y) : null;
        this.marker.hidden = !box;
        if (box) {
            const size = Math.max(box.size, 4);
            Object.assign(this.marker.style, {
                left: `${box.left + box.size / 2 - size / 2 - 1}px`, top: `${box.top + box.size / 2 - size / 2 - 1}px`,
                width: `${size}px`, height: `${size}px`,
            });
        }
    }

    async refresh() {
        const sim = this.sim();
        if (!sim || sim.destroyed) {
            return;
        }
        this.reading = true;
        this.dirty = false;
        this.lastRead = performance.now();
        // (A read begun before a trace began isn't one of its reads.)
        const trace = this.trace;
        const tab = trace ? "cell" : this.tab;
        let got = null;
        try {
            const result = tab === "cell" ? await this.readCell(sim)
                : tab === "world" ? await this.readWorld(sim)
                    : tab === "bodies" ? await this.readBodies(sim)
                        : await this.readPeople(sim);
            // (A world made meanwhile, of a scene or a snapshot, is read
            // afresh. With the pointer off the world, the last cell read
            // stays.)
            got = result && { tab, ...result };
            if (this.sim() === sim && (result || tab !== "cell")) {
                this.result = got;
            }
            this.error = "";
        } catch (err) {
            this.error = err.message;
        } finally {
            this.reading = false;
        }
        if (trace && trace === this.trace) {
            this.traced(got);
        }
        this.render();
    }

    // Sections' parts read, and their words and rows.
    async readSections(sim, defs, p, first = {}, extra = {}) {
        const plans = defs.map((d) => ({ d, parts: d.parts(p, sim, first) })).filter((s) => s.parts);
        const all = plans.flatMap((s) => s.parts);
        const extras = Object.entries(extra);
        const words = await readParts(this.device, [...all, ...extras.map(([, r]) => r)]);
        const out = {};
        let k = 0;
        for (const { d, parts } of plans) {
            const got = words.slice(k, k + parts.length);
            k += parts.length;
            const bad = got.findIndex((w) => typeof w === "string");
            const section = {
                key: d.key, group: d.group, title: d.title,
                sources: parts.map((r) => (r.texture ? `${r.name} (${r.x}, ${r.y})` : `${r.name} @ ${r.offset}`)),
            };
            if (bad >= 0) {
                section.error = `${parts[bad].name}: ${got[bad]}`;
            } else {
                const u = new Uint32Array(got.reduce((n, w) => n + w.length, 0));
                got.reduce((n, w) => (u.set(w, n), n + w.length), 0);
                section.u = u;
                section.f = new Float32Array(u.buffer);
            }
            out[d.key] = section;
        }
        extras.forEach(([key], j) => {
            const w = words[k + j];
            out[key] = typeof w === "string" ? { error: w } : { u: w };
        });
        for (const d of defs) {
            const s = out[d.key];
            if (s?.u) {
                try {
                    s.rows = d.rows(s.u, s.f, p, sim, first);
                } catch (err) {
                    s.error = `couldn't decode: ${err.message}`;
                }
            }
        }
        return out;
    }

    async readCell(sim) {
        const cell = this.target();
        if (!cell) {
            return null;
        }
        const p = placeOf(sim, cell.x, cell.y);
        const step = sim.frameNo;
        const first = await this.readSections(sim, CELL_SECTIONS, p, {}, linkParts(p, sim));
        if (sim.destroyed) {
            return null;
        }
        const later = sim.frameNo;
        const found = await this.readSections(sim, FOUND_SECTIONS, p, first);
        const sections = [...CELL_SECTIONS, ...FOUND_SECTIONS].map((d) => first[d.key] ?? found[d.key]).filter(Boolean)
            .sort((a, b) => GROUPS.indexOf(a.group) - GROUPS.indexOf(b.group));
        return { cell: { x: cell.x, y: cell.y, i: p.i }, pinned: !!this.pinned, step, later, sections };
    }

    async readWorld(sim) {
        const counts = [
            span(sim.particles[0], "particles0 count", 0, 4),
            span(sim.blocks, "blocks counts", 4 * sim.blockCount * 4, FLUID_STATS * 4),
            span(sim.bodies, "bodies stats", 2 * sim.bodyTiles * 4, BODY_STATS * 4),
            span(sim.electric, "electric stats", 0, ELECTRIC_STATS * 4),
            span(sim.stressWork, "stress stats", 0, STRESS_STATS * 4),
        ];
        const [particles, fluid, bodies, electric, stress] = await readParts(this.device, counts);
        const named = (u, names) => (typeof u === "string" ? u : Object.entries(names).map(([k, i]) => `${k.toLowerCase()} ${u[i]}`).join(" · "));
        const gpu = [
            ["particles", typeof particles === "string" ? particles : `${particles[0]} of ${sim.particleSlots} slots`],
            ["liquid", named(fluid, FLUID_COUNT)],
            ["bodies", named(bodies, BODY_STAT)],
            ["current", typeof electric === "string" ? electric : Array.from(electric, (w) => hex(w)).join(" ")],
            ["stress", named(stress, STRESS_STAT)],
        ];
        const state = Object.entries({
            size: `${sim.width} × ${sim.height}`, ...sim.snapshotState(),
            heatStages: sim.heatStages, epochs: sim.epochs, framesStepped: sim.framesStepped, reportedAt: sim.reportedAt,
            particleSlots: sim.particleSlots, blockCount: sim.blockCount, bodyTiles: sim.bodyTiles, lightSize: sim.lightSize,
            airLevels: sim.airLevels, levels: sim.levels,
        }).map(([k, v]) => [k, typeof v === "object" ? JSON.stringify(v) : String(v)]);
        let total = 0;
        const resources = [];
        for (const [name, r] of Object.entries(sim.resources())) {
            for (const [key, one] of Array.isArray(r) ? r.map((x, i) => [`${name}${i}`, x]) : [[name, r]]) {
                const isTexture = one instanceof GPUTexture;
                const bytes = isTexture ? one.width * one.height * (TEXEL_BYTES[one.format] ?? 0) : one.size;
                total += bytes;
                resources.push([key, isTexture ? `${one.width} × ${one.height} ${one.format}` : "buffer", bytes]);
            }
        }
        return { gpu, state, resources, total };
    }

    async readBodies(sim) {
        const [table] = await readParts(this.device, [span(sim.bodyMotion, "bodyMotion table", MAX_BODIES * 4, MAX_BODIES * BODY_TABLE_WORDS * 4)]);
        if (typeof table === "string") {
            throw new Error(table);
        }
        const f = new Float32Array(table.buffer, table.byteOffset, table.length);
        const list = [];
        for (let b = 0; b < MAX_BODIES; b++) {
            const a = b * BODY_TABLE_WORDS;
            if (table[a + 18] & BODY_FLAG.ALIVE) {
                list.push({
                    slot: b, x: f[a], y: f[a + 1], angle: f[a + 2], vx: f[a + 3], vy: f[a + 4], spin: f[a + 5], mass: f[a + 6],
                    cells: table[a + 13], flags: table[a + 18], rest: table[a + 21], support: table[a + 102],
                });
            }
        }
        let detail = null;
        if (this.bodySlot !== null) {
            const parts = bodyParts(sim, this.bodySlot);
            const words = await readParts(this.device, parts);
            const bad = words.find((w) => typeof w === "string");
            if (bad) {
                throw new Error(bad);
            }
            const u = new Uint32Array(words.reduce((n, w) => n + w.length, 0));
            words.reduce((n, w) => (u.set(w, n), n + w.length), 0);
            const fu = new Float32Array(u.buffer);
            detail = { key: "bodyDetail", group: "Bodies", title: `Body ${this.bodySlot}`, sources: parts.map((r) => `${r.name} @ ${r.offset}`), u, f: fu, rows: bodyRows(u, fu, this.bodySlot) };
        }
        return { list, detail };
    }

    async readPeople(sim) {
        const [all] = await readParts(this.device, [span(sim.people, "people", 0, MAX_PEOPLE * LAYOUT.PERSON_BYTES)]);
        if (typeof all === "string") {
            throw new Error(all);
        }
        const f = new Float32Array(all.buffer, all.byteOffset, all.length);
        const words = LAYOUT.PERSON_BYTES / 4;
        const list = [];
        let detail = null;
        for (let k = 0; k < MAX_PEOPLE; k++) {
            const u = all.slice(k * words, (k + 1) * words);
            if (u[4] === PERSON_STATE.NONE) {
                continue;
            }
            const fu = new Float32Array(u.buffer);
            list.push({ slot: k, x: f[k * words], y: f[k * words + 1], text: describePerson({ seed: u[9], state: u[4], cause: u[15], health: fu[7], flags: u[5], breath: fu[8] }) });
            if (k === this.personSlot) {
                detail = { key: "personDetail", group: "People", title: `Person ${k}`, sources: [`people @ ${k * LAYOUT.PERSON_BYTES}`], u, f: fu, rows: personRows(u, fu) };
            }
        }
        return { list, detail };
    }

    // A trace of the cell inspected (pinned for it): the state it starts
    // from, then a read after each step.
    startTrace(steps) {
        const cell = this.target();
        if (!cell) {
            this.controls.showToast("Point at a cell or pin one to trace it");
            return;
        }
        this.pinned = { ...cell };
        this.pause();
        this.stepsLeft = 0;
        this.trace = { steps, rows: [], need: true };
        this.root.querySelector(".dev-download").hidden = true;
        this.setTab("cell");
    }

    // A read of a trace taken (null if it failed): kept, and the trace done
    // after its last step, or a read that failed.
    traced(r) {
        const t = this.trace;
        t.need = false;
        if (r?.sections) {
            t.rows.push(this.toJson(r));
        }
        if (t.rows.length > t.steps || !r?.sections) {
            this.trace = null;
            this.lastTrace = t.rows;
            const blob = new Blob([JSON.stringify(t.rows, null, 1)], { type: "application/json" });
            const a = this.root.querySelector(".dev-download");
            URL.revokeObjectURL(a.href);
            a.href = URL.createObjectURL(blob);
            const c = t.rows[0]?.cell;
            a.download = `trace-${c ? `${c.x}-${c.y}-` : ""}${t.rows[0]?.step ?? 0}.json`;
            a.textContent = `trace (${t.rows.length} reads)`;
            a.hidden = false;
            console.log("devtools trace", t.rows);
        }
    }

    // The inspector's read as plain data: each section's sources, words (as
    // shown) and rows.
    toJson(r) {
        const show = FORMATS[this.format];
        return {
            step: r.step, cell: r.cell,
            sections: r.sections.map((s) => ({
                key: s.key, title: s.title, sources: s.sources, error: s.error,
                words: s.u ? Array.from(s.u, (w) => show(w)) : undefined,
                rows: s.rows?.map(([label, , , text]) => [label, text]),
            })),
        };
    }

    async copy() {
        const r = this.result;
        if (!r) {
            return;
        }
        const data = r.tab === "cell" ? this.toJson(r) : r;
        const text = JSON.stringify(data, (k, v) => (v instanceof Uint32Array || v instanceof Float32Array ? Array.from(v) : v), 1);
        try {
            await navigator.clipboard.writeText(text);
            this.controls.showToast("Copied");
        } catch {
            console.log(text);
            this.controls.showToast("Couldn't copy: it's in the console");
        }
    }

    // ------------------------------------------------------------ drawing

    render() {
        if (!this.root || !this.open) {
            return;
        }
        const sim = this.sim();
        if (!sim) {
            return;
        }
        const r = this.result;
        const t = this.trace;
        this.status.textContent = `step ${sim.frameNo}${t ? ` · tracing ${t.rows.length}/${t.steps + 1}` : ""}${this.error ? ` · ${this.error}` : ""}`;
        for (const bar of this.root.querySelectorAll(".dev-bar")) {
            bar.querySelectorAll("input[name=x], input[name=y], [data-act=follow]").forEach((el) => {
                el.disabled = this.tab !== "cell";
            });
        }
        if (this.tab === "cell" && r?.cell && document.activeElement !== this.inputs.x && document.activeElement !== this.inputs.y) {
            this.inputs.x.value = r.cell.x;
            this.inputs.y.value = r.cell.y;
        }
        // (Not drawn over a selection in it: it could be copied.)
        const selection = window.getSelection();
        if (selection && !selection.isCollapsed && this.body.contains(selection.anchorNode)) {
            return;
        }
        const html = this.tab === "cell" ? this.cellHtml(r)
            : this.tab === "world" ? this.worldHtml(r)
                : this.tab === "bodies" ? this.listHtml(r, "body") : this.listHtml(r, "person");
        if (html !== this.shown) {
            this.shown = html;
            this.body.innerHTML = html;
        }
    }

    sectionHtml(s) {
        const show = FORMATS[this.format];
        const raw = (k, n) => (s.u ? Array.from(s.u.subarray(k, k + n), show).join(" ") : "");
        let last = -1;
        const rows = (s.rows ?? []).map(([label, k, n, text]) => {
            // (Words shown once, by the first row reading them.)
            const words = n > 0 && k !== last ? raw(k, n) : "";
            if (n > 0) {
                last = k;
            }
            return `<tr><th>${escapeHtml(label)}</th><td class="dev-raw">${escapeHtml(words)}</td><td>${escapeHtml(text)}</td></tr>`;
        }).join("");
        const error = s.error ? `<p class="dev-error">${escapeHtml(s.error)}</p>` : "";
        return `<details data-key="${s.key}"${this.folded.has(s.key) ? "" : " open"}><summary>${escapeHtml(s.title)} `
            + `<span class="dev-src">${escapeHtml(s.sources.join(" · "))}</span></summary>${error}<table>${rows}</table></details>`;
    }

    cellHtml(r) {
        if (!r || r.tab !== "cell") {
            return `<p class="dev-note">Point at the world, or Ctrl+click a cell to pin it.</p>`;
        }
        const head = `<p class="dev-where">(${r.cell.x}, ${r.cell.y}) · cell ${r.cell.i} · ${r.pinned ? "pinned (arrow keys move it)" : "under the pointer"}`
            + ` · read at step ${r.step}${r.later !== r.step ? `, what it points to at step ${r.later}` : ""}</p>`;
        let group = "";
        const parts = r.sections.map((s) => {
            const heading = s.group !== group ? `<h3>${escapeHtml(s.group)}</h3>` : "";
            group = s.group;
            return heading + this.sectionHtml(s);
        });
        return head + parts.join("");
    }

    worldHtml(r) {
        if (!r || r.tab !== "world") {
            return "";
        }
        const table = (rows) => `<table>${rows.map(([k, v]) => `<tr><th>${escapeHtml(k)}</th><td>${escapeHtml(v)}</td></tr>`).join("")}</table>`;
        const mb = (bytes) => `${(bytes / 2 ** 20).toFixed(2)} MB`;
        return `<h3>GPU counters</h3>${table(r.gpu)}<h3>Host state</h3>${table(r.state)}`
            + `<h3>Resources · ${mb(r.total)}</h3>${table(r.resources.map(([k, kind, bytes]) => [k, `${kind} · ${bytes} B (${mb(bytes)})`]))}`;
    }

    listHtml(r, kind) {
        if (!r || r.tab !== (kind === "body" ? "bodies" : "people")) {
            return "";
        }
        const pick = (x, y) => `<button data-act="pinAt" data-x="${Math.floor(x)}" data-y="${Math.floor(y)}" title="Inspect this cell">(${Math.floor(x)}, ${Math.floor(y)})</button>`;
        const rows = kind === "body"
            ? r.list.map((b) => `<tr data-body="${b.slot}" class="${b.slot === this.bodySlot ? "active" : ""}"><th>${b.slot}</th>`
                + `<td>${pick(b.x, b.y)}</td><td>${b.cells} cells · ${num(b.mass)} · v ${num(b.vx)}, ${num(b.vy)} · ω ${num(b.spin)}</td>`
                + `<td>${escapeHtml(bitNames(b.flags, BODY_FLAGS))} · rest ${b.rest} · ${escapeHtml(bitNames(b.support, SUPPORT))}</td></tr>`)
            : r.list.map((q) => `<tr data-person="${q.slot}" class="${q.slot === this.personSlot ? "active" : ""}"><th>${q.slot}</th>`
                + `<td>${pick(q.x, q.y - 1)}</td><td>${escapeHtml(q.text)}</td></tr>`);
        const none = `<p class="dev-note">${kind === "body" ? "No bodies." : "No people."}</p>`;
        const list = r.list.length ? `<table class="dev-list">${rows.join("")}</table>` : none;
        return `<p class="dev-note">${r.list.length} ${kind === "body" ? "alive" : "in the world"}: click a row for its words.</p>${list}`
            + (r.detail ? this.sectionHtml(r.detail) : "");
    }
}
