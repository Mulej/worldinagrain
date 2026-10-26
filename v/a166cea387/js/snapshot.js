// Snapshots: a world saved whole, to pick up exactly where it left off (the
// app's Save and Load, the test harness's saveSnapshot and loadSnapshot).
//
// A snapshot is gzipped: MAGIC, the header's length in bytes (u32), the
// header (JSON: the format, the world's size, the elements by id, the
// simulation's own state, and the name and size of each part), then each
// part's bytes in the header's order, each padded to 4. The parts are the
// GPU buffers and textures that carry over from one step to the next
// (Simulation.snapshotParts), a texture's rows packed tight. A snapshot only
// loads into a world of its size, with the same elements and every part the
// same size: one saved by a checkout that lays a buffer out differently is
// refused rather than misread. The header also says what scale water's
// vapour was kept at (WATER_SCALE), and one saved at another is brought to
// this one's as it loads (migrateWater); and whether the gases' state was in
// the gas plane (GAS_PLANE), which one from before is moved into
// (migrateGas), and one from before the gas had an amount given it
// (migrateAmounts); and whether its bodies were kept whole (BODIES), which
// one from before has looked over (migrateBodies). One whose bodies' table
// was OLD_TABLE_WORDS a slot has each slot widened (migrateTable).

import { ELEMENTS, FLAG, ID, KIND } from "./elements.js";
import {
    AMBIENT_T, BODY_FLAG, BODY_TABLE_WORDS, BODY_WORD, BOIL_K, CELL_BYTES, CELL_FIELDS, CELL_FLAGS, CELL_M, MAX_BODIES, MAX_PEOPLE, cellsBufferBytes,
    cellTypeAt, f16Bits, f16Value, FLUID_STATS,
    FROZEN_EXTRA, GAS_BYTES, KELVIN, openAirAtm, VAPOUR_20, VAPOUR_EXPANSION, VAPOUR_MAX, WATER_VAPOUR,
} from "./shared.js";

const MAGIC = "SANDSNAP";
const FORMAT = 1;
// Parts kept since snapshots were first saved, which an older one lacks:
// loaded, they start empty, as a cleared world's do. electricWork's float
// flags (2026-10-22) settle again within a step; heat's clock (2026-10-22)
// is found again the next step; heat on its way into cells (heatIn,
// 2026-10-23) is a step's Joule heat, lost once; radiant heat (radiant and
// its takings and edges, stage 6) is worked out again within LIGHT_EVERY
// steps, what was reflected lost once; where people's bodies are
// (occupants, 10-1) is stamped again by their next step; the down each
// light cell was last given (gravityMasses, 2026-10-27, its hysteresis)
// starts from none; how real-time the current's heat made each light cell
// (clockHeated, P1-1) is found again by the next step's current.
export const ADDED = new Set(["electricWork", "clock", "heatIn", "radiant", "radiantTaken0", "radiantTaken1", "radiantEdges0", "radiantEdges1", "occupants",
    "gravityMasses", "clockHeated"]);
// Parts the steps make again from the cells within a step or a few (the
// air settles, the rooms are found again): one laid out differently by an
// older version (the air's grid, REWRITE_PLAN.md stage 4a), or missing
// (what was put into the air, at most a step's, stage C), loads empty.
// The stress solve's state too (REWRITE_PLAN.md 8-2): empty, it starts over.
// And the light's share of sunlight per light cell (heat), worked out again
// within LIGHT_EVERY steps, which held the infrared too until stage 6; and
// the current's blocks and CG state (electric), worked out afresh each step
// (a word more a block since D-1), and its working words a cell (a word
// more since the back-EMF, P1). Heat on its way into cells (heatIn), an
// f32 a cell before D-1's exact sums, loads empty: a step's Joule heat, lost
// once. The radiant heat's takings, a word a cell more since 6-5, load
// empty too: what was reflected, lost once, and the next sweep's in.
const REBUILT = new Set(["air", "airProjection", "airCg", "blast", "sky", "airSource", "stressNodes", "stressWork", "stressDisp", "heat", "electric", "heatIn",
    "radiantTaken0", "radiantTaken1", "electricWork"]);
// The liquid's parts, laid out otherwise before its particles were sorted
// by cell (REWRITE_PLAN.md 7-1): one saved so loads with none, and its
// liquid cells seeded afresh (migrateLiquid).
const LIQUID_PARTS = ["particles0", "particles1", "counters", "blocks"];
// Parts an older version kept that this one doesn't, passed over: the
// potential's low part (halfway, until 2026-10-25), always 0 (electric.wgsl
// linkDrop); the cascades' balance of radiant heat (ledger, until stage 6,
// radiate.wgsl's now).
export const DROPPED = new Set(["halfway", "ledger"]);
// Water's vapour scale: what saturates air at 20 °C (vapour units), and the
// share of a unit ice and snow keep their extra in. Snapshots from before
// the header said (REWRITE_PLAN.md stage 3a) were at OLD_WATER_SCALE.
const WATER_SCALE = { vapour20: VAPOUR_20, frozenExtra: FROZEN_EXTRA };
const OLD_WATER_SCALE = { vapour20: 4, frozenExtra: 8 };
// The gas plane's layout (common.wgsl's Gas), in the header from stage 3d
// on. Before it, air kept its vapour and mist in its w word and air, flames,
// smoke and ash their oxygen used up in aux (of 255). Layout 2 (stage 4c)
// adds the dry gas's amount (the second word's high f16, none before), and
// steam is the air's vapour, not cells of its own.
const GAS_PLANE = 2;
// The bodies' state as REWRITE_PLAN.md P3-9 keeps it, in the header from
// then on: each body in one piece, each loose cell in the world its body's.
const BODIES = 1;
// A loose cell's slot in no body (common.wgsl's FREE_BODY).
const FREE_BODY = 2047;
const TEXEL_BYTES = { "r32float": 4, "rgba16float": 8, "rgba32float": 16, "rgba32uint": 16 };

const align = (n, to) => Math.ceil(n / to) * to;

async function gzip(bytes, stream) {
    return new Uint8Array(await new Response(new Blob([bytes]).stream().pipeThrough(stream)).arrayBuffer());
}

// The world as it is, read back from the GPU and packed. Its state is taken
// as the copies are sent, so the frames that run on while they come back
// don't get mixed into it. Each part comes back through a staging buffer of
// its own: all of them in one would be some 1.2 KB a cell, past what a GPU
// gives one buffer at sizes the app makes (maxCells allows for its largest
// buffer, not their sum).
export async function saveSnapshot(sim) {
    const { buffers, textures } = sim.snapshotParts();
    const device = sim.device;
    const stage = (size) => device.createBuffer({ label: "snapshot", size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const items = [];
    for (const [name, buffer] of Object.entries(buffers)) {
        items.push({ name, buffer, bytes: buffer.size, staging: stage(buffer.size) });
    }
    for (const [name, texture] of Object.entries(textures)) {
        const row = texture.width * TEXEL_BYTES[texture.format];
        const stride = align(row, 256);
        items.push({ name, texture, row, stride, bytes: row * texture.height, staging: stage(stride * texture.height) });
    }
    const enc = device.createCommandEncoder();
    for (const item of items) {
        if (item.buffer) {
            enc.copyBufferToBuffer(item.buffer, 0, item.staging, 0, item.bytes);
        } else {
            const { texture } = item;
            enc.copyTextureToBuffer({ texture }, { buffer: item.staging, bytesPerRow: item.stride }, [texture.width, texture.height]);
        }
    }
    device.queue.submit([enc.finish()]);
    const header = {
        format: FORMAT,
        width: sim.width,
        height: sim.height,
        elements: ELEMENTS.map((e) => e.id),
        water: WATER_SCALE,
        gas: GAS_PLANE,
        bodies: BODIES,
        state: sim.snapshotState(),
        parts: items.map(({ name, bytes, texture }) => ({
            name, bytes, ...(texture && { width: texture.width, height: texture.height, format: texture.format }),
        })),
    };
    try {
        await Promise.all(items.map((item) => item.staging.mapAsync(GPUMapMode.READ)));
        const head = new TextEncoder().encode(JSON.stringify(header));
        const start = MAGIC.length + 4 + align(head.length, 4);
        const out = new Uint8Array(start + items.reduce((n, item) => n + align(item.bytes, 4), 0));
        out.set(new TextEncoder().encode(MAGIC), 0);
        new DataView(out.buffer).setUint32(MAGIC.length, head.length, true);
        out.set(head, MAGIC.length + 4);
        let at = start;
        for (const item of items) {
            const read = new Uint8Array(item.staging.getMappedRange());
            if (item.buffer) {
                out.set(read.subarray(0, item.bytes), at);
            } else {
                for (let y = 0; y < item.texture.height; y++) {
                    out.set(read.subarray(y * item.stride, y * item.stride + item.row), at + y * item.row);
                }
            }
            item.staging.destroy();
            at += align(item.bytes, 4);
        }
        return await gzip(out, new CompressionStream("gzip"));
    } finally {
        for (const item of items) {
            item.staging.destroy();
        }
    }
}

// A snapshot unpacked: { header, parts }, parts a Map from name to bytes.
// Throws if it isn't one this checkout can read.
export async function readSnapshot(bytes) {
    let raw;
    try {
        raw = await gzip(bytes, new DecompressionStream("gzip"));
    } catch {
        throw new Error("not a snapshot");
    }
    if (raw.length < MAGIC.length + 4 || new TextDecoder().decode(raw.subarray(0, MAGIC.length)) !== MAGIC) {
        throw new Error("not a snapshot");
    }
    const length = new DataView(raw.buffer, raw.byteOffset).getUint32(MAGIC.length, true);
    const header = JSON.parse(new TextDecoder().decode(raw.subarray(MAGIC.length + 4, MAGIC.length + 4 + length)));
    if (header.format !== FORMAT) {
        throw new Error(`saved in snapshot format ${header.format}; this version reads format ${FORMAT}`);
    }
    if (header.elements.join() !== ELEMENTS.map((e) => e.id).join()) {
        throw new Error("saved with a different set of elements");
    }
    const parts = new Map();
    let at = MAGIC.length + 4 + align(length, 4);
    for (const part of header.parts) {
        if (at + part.bytes > raw.length) {
            throw new Error("the snapshot is cut short");
        }
        parts.set(part.name, raw.subarray(at, at + part.bytes));
        at += align(part.bytes, 4);
    }
    return { header, parts };
}

// The parts of a snapshot saved at another water scale (WATER_SCALE),
// brought to this one's: the air's vapour and mist as the same share of
// what saturates it (both cell buffers), and ice's and snow's extra in this
// scale's shares (as much of it as their life holds). The rest is the same.
// (The liquid's pooled moisture goes with migrateAmounts.)
function migrateWater(sim, header, parts) {
    const was = header.water ?? OLD_WATER_SCALE;
    if (was.vapour20 === WATER_SCALE.vapour20 && was.frozenExtra === WATER_SCALE.frozenExtra) {
        return parts;
    }
    const vapour = WATER_SCALE.vapour20 / was.vapour20;
    const frozen = WATER_SCALE.frozenExtra / was.frozenExtra;
    const [lifeAt, lifeBits] = CELL_FIELDS.life;
    const lifeMask = ((1 << lifeBits) - 1) << lifeAt;
    const scaled = (half) => f16Bits(f16Value(half) * vapour);
    const out = new Map(parts);
    for (const name of ["cells0", "cells1"]) {
        const bytes = parts.get(name);
        if (!bytes) {
            continue;
        }
        const copy = bytes.slice();
        const u = new Uint32Array(copy.buffer, copy.byteOffset, copy.length / 4);
        for (let i = 0; i < u.length / 4; i++) {
            const t = cellTypeAt(u, i);
            if (t === ID.AIR) {
                const w = u[4 * i + 3];
                u[4 * i + 3] = (scaled(w & 0xffff) | (scaled(w >>> 16) << 16)) >>> 0;
            } else if (t === ID.ICE || t === ID.SNOW) {
                const life = Math.min(Math.round(((u[4 * i] & lifeMask) >>> lifeAt) * frozen), (1 << lifeBits) - 1);
                u[4 * i] = ((u[4 * i] & ~lifeMask) | (life << lifeAt)) >>> 0;
            }
        }
        out.set(name, copy);
    }
    return out;
}

// The parts of a snapshot from before the gas plane (GAS_PLANE), its gases'
// state moved into it: air's vapour and mist, and the oxygen air, flames,
// smoke and ash had used up, in both cell buffers (their old places
// cleared).
function migrateGas(sim, header, parts) {
    if ((header.gas ?? 0) >= 1) {
        return parts;
    }
    const count = sim.width * sim.height;
    const out = new Map(parts);
    const [auxAt, auxBits] = CELL_FIELDS.aux;
    const auxMask = ((1 << auxBits) - 1) << auxAt;
    const carriers = new Set([ID.AIR, ID.FIRE, ID.SMOKE, ID.ASH]);
    for (const name of ["cells0", "cells1"]) {
        const bytes = parts.get(name);
        if (!bytes || bytes.length !== count * CELL_BYTES) {
            continue;
        }
        const grown = new Uint8Array(cellsBufferBytes(count));
        grown.set(bytes);
        const u = new Uint32Array(grown.buffer);
        for (let i = 0; i < count; i++) {
            const t = cellTypeAt(u, i);
            if (!carriers.has(t)) {
                continue;
            }
            const gas = (count * CELL_BYTES + i * GAS_BYTES) / 4;
            const lost = ((u[4 * i] & auxMask) >>> auxAt) / 255;
            u[gas + 1] = f16Bits(lost);
            u[4 * i] = (u[4 * i] & ~auxMask) >>> 0;
            if (t === ID.AIR) {
                u[gas] = u[4 * i + 3];
                u[4 * i + 3] = 0;
            }
        }
        out.set(name, grown);
    }
    return out;
}

// The parts of a snapshot from before the gas had an amount (GAS_PLANE 2,
// stage 4c), given one: every cell holding gas the open air's at its height
// and temperature (as js/shared.js outdoorAirGas and brush.wgsl's
// paintedGas have it), less its vapour's share; a sealed room's pressure
// comes back from it. Steam becomes air holding its water as vapour (a unit
// of steam was a cell of vapour at 1 atm and the boiling point, VAPOUR_MAX
// units), none of it dry gas. The liquid's pooled moisture (the blocks'
// last FLUID_POOL words, gone) is lost: what air it had closed over held.
function migrateAmounts(sim, header, parts) {
    if (header.gas === GAS_PLANE) {
        return parts;
    }
    const count = sim.width * sim.height;
    const out = new Map(parts);
    const [typeAt, typeBits] = CELL_FIELDS.type;
    const typeMask = ((1 << typeBits) - 1) << typeAt;
    const [auxAt, auxBits] = CELL_FIELDS.aux;
    const auxMask = ((1 << auxBits) - 1) << auxAt;
    const dx = header.state?.metresPerCell ?? CELL_M;
    const tRef = AMBIENT_T + KELVIN;
    const vapourAmount = (v) => v * tRef / (BOIL_K * VAPOUR_MAX);
    const holdsGas = (e) => e.kind === KIND.GAS || e.id === "ASH" || (e.flags & FLAG.AIR_PASS && e.id !== "BLACK_HOLE");
    for (const name of ["cells0", "cells1"]) {
        const bytes = parts.get(name);
        if (!bytes || bytes.length !== cellsBufferBytes(count)) {
            continue;
        }
        const copy = bytes.slice();
        const u = new Uint32Array(copy.buffer, copy.byteOffset, copy.length / 4);
        const f = new Float32Array(copy.buffer, copy.byteOffset, copy.length / 4);
        for (let i = 0; i < count; i++) {
            const t = cellTypeAt(u, i);
            if (!holdsGas(ELEMENTS[t])) {
                continue;
            }
            const gas = (count * CELL_BYTES + i * GAS_BYTES) / 4;
            let vapour = f16Value(u[gas] & 0xffff);
            const mist = f16Value(u[gas] >>> 16);
            const lost = u[gas + 1] & 0xffff;
            let amount;
            if (t === ID.STEAM) {
                vapour += ((u[4 * i] & auxMask) >>> auxAt) * WATER_VAPOUR / VAPOUR_EXPANSION;
                amount = 0;
                u[4 * i] = ((u[4 * i] & ~typeMask & ~auxMask) | (ID.AIR << typeAt)) >>> 0;
            } else {
                const altitude = (sim.height - Math.floor(i / sim.width) - 0.5) * dx;
                amount = Math.max(openAirAtm(altitude) * tRef / Math.max(f[4 * i + 1] + KELVIN, 1) - vapourAmount(vapour), 0);
            }
            u[gas] = (f16Bits(vapour) | (f16Bits(mist) << 16)) >>> 0;
            u[gas + 1] = (lost | (f16Bits(amount) << 16)) >>> 0;
        }
        out.set(name, copy);
    }
    // (The liquid's counts after the blocks' lists were 4 words then:
    // migrateBlocks.)
    const blocks = parts.get("blocks");
    const fluidPool = 3, stats = 4;
    if (blocks && blocks.length === (4 * sim.blockCount + stats + fluidPool) * 4) {
        out.set("blocks", blocks.slice(0, (4 * sim.blockCount + stats) * 4));
    }
    return out;
}

// The parts of a snapshot whose liquid is laid out otherwise (from before
// 7-1, or with another LIQUID_PER_CELL, or another LIQUID_CELL_WORDS: 7-3's
// smoothed share, 7-4's densities, D-1's claims and rims): the liquid's
// parts left out, to load
// empty, and its liquid cells no longer drawn (FL_DRAWN), so that they are
// seeded with particles afresh as new liquid is, at rest. (Drawn, they
// would claim particles there are none of by what their w meant before.)
function migrateLiquid(sim, parts) {
    const saved = parts.get("particles0");
    const counters = parts.get("counters");
    if (!saved || (saved.length === sim.particles[0].size && counters?.length === sim.counters.size)) {
        return parts;
    }
    const out = new Map(parts);
    for (const name of LIQUID_PARTS) {
        out.delete(name);
    }
    const [flagsAt] = CELL_FIELDS.flags;
    const drawn = CELL_FLAGS.DRAWN << flagsAt;
    for (const name of ["cells0", "cells1"]) {
        const bytes = parts.get(name);
        if (!bytes) {
            continue;
        }
        const copy = bytes.slice();
        const u = new Uint32Array(copy.buffer, copy.byteOffset, copy.length / 4);
        for (let i = 0; i < sim.width * sim.height; i++) {
            if (ELEMENTS[cellTypeAt(u, i)]?.kind === KIND.LIQUID) {
                u[4 * i] = (u[4 * i] & ~drawn) >>> 0;
                u[4 * i + 3] = 0;
            }
        }
        out.set(name, copy);
    }
    out.set(MIGRATED_LIQUID, true);
    return out;
}
const MIGRATED_LIQUID = Symbol("liquid migrated");

// The parts of a snapshot from before people's bodies held heat (10-2: 16
// words a person, 20 since): each person's record with its skin at 34 °C,
// its core at 37, no burn.
function migratePeople(sim, parts) {
    const saved = parts.get("people");
    if (!saved || saved.length !== MAX_PEOPLE * 64 || sim.people.size === saved.length) {
        return parts;
    }
    const words = sim.people.size / 4 / MAX_PEOPLE;
    const from = new Uint32Array(saved.buffer, saved.byteOffset, saved.length / 4);
    const out = new Uint32Array(sim.people.size / 4);
    const f = new Float32Array(out.buffer);
    for (let k = 0; k < MAX_PEOPLE; k++) {
        out.set(from.subarray(16 * k, 16 * k + 16), words * k);
        f[words * k + 16] = 34;
        f[words * k + 17] = 37;
    }
    return new Map(parts).set("people", new Uint8Array(out.buffer));
}

// The parts of a snapshot from before P3-18, when the liquid's counts after
// the blocks' lists were 4 words, not FLUID_STATS: the pressure's V-cycles'
// slots and counts put in after them, empty (a step's first cycle always
// runs, and clears what the next reads).
function migrateBlocks(sim, parts) {
    const saved = parts.get("blocks");
    const added = (FLUID_STATS - 4) * 4;
    if (!saved || saved.length !== sim.blocks.size - added) {
        return parts;
    }
    const at = (4 * sim.blockCount + 4) * 4;
    const grown = new Uint8Array(sim.blocks.size);
    grown.set(saved.subarray(0, at));
    grown.set(saved.subarray(at), at + added);
    return new Map(parts).set("blocks", grown);
}

// The parts of a snapshot from before D-1, when which body last moved over
// each cell was two words a cell at the start of bodyMotion, and is now
// two (by the step's parity) at the end of the bodies buffer: dropped from
// the one, and none in the other (no cell's cup is taken as last step's).
// (Its bodies' table OLD_TABLE_WORDS a slot, widened after: migrateTable.)
function migrateSwept(sim, parts) {
    const cellBytes = 2 * 4 * sim.width * sim.height;
    const bodies = parts.get("bodies"), motion = parts.get("bodyMotion");
    const sizes = [sim.bodyMotion.size, sim.bodyMotion.size - TABLE_GROWTH].map((size) => size + cellBytes);
    if (!bodies || !motion || bodies.length !== sim.bodies.size - cellBytes || !sizes.includes(motion.length)) {
        return parts;
    }
    const grown = new Uint8Array(sim.bodies.size);
    grown.set(bodies);
    return new Map(parts).set("bodies", grown).set("bodyMotion", motion.slice(cellBytes));
}

// The parts of a snapshot from before P3-23, its bodies' table
// OLD_TABLE_WORDS a slot: each slot widened, the words after it none (no
// mobility worked out: MOBILITY_AT 0, the liquid's solve takes each body as
// its mass alone has it until the bodies' pass has run), the contacts and
// the liquid's words after the table moved along.
const OLD_TABLE_WORDS = 108;
const TABLE_GROWTH = MAX_BODIES * (BODY_TABLE_WORDS - OLD_TABLE_WORDS) * 4;

function migrateTable(sim, parts) {
    const saved = parts.get("bodyMotion");
    if (!saved || saved.length !== sim.bodyMotion.size - TABLE_GROWTH) {
        return parts;
    }
    const old = new Uint32Array(saved.slice().buffer);
    const grown = new Uint32Array(sim.bodyMotion.size / 4);
    grown.set(old.subarray(0, MAX_BODIES));
    for (let b = 0; b < MAX_BODIES; b++) {
        const from = MAX_BODIES + b * OLD_TABLE_WORDS;
        grown.set(old.subarray(from, from + OLD_TABLE_WORDS), MAX_BODIES + b * BODY_TABLE_WORDS);
    }
    grown.set(old.subarray(MAX_BODIES * (1 + OLD_TABLE_WORDS)), MAX_BODIES * (1 + BODY_TABLE_WORDS));
    return new Map(parts).set("bodyMotion", new Uint8Array(grown.buffer));
}

// The parts of a snapshot from before viscosity's sweeps kept what they did
// to each face (REWRITE_PLAN.md 7-11: a quarter of a word a face more in the
// liquid's field, after the velocities before viscosity, the coarse levels
// after it): none kept, the sweeps start from the forces' velocities for a
// step.
function migrateField(sim, parts) {
    const saved = parts.get("field");
    const faces = (sim.width + 1) * sim.height + sim.width * (sim.height + 1);
    const quarter = Math.ceil(faces / 4) * 16;
    if (!saved || saved.length !== sim.field.size - quarter) {
        return parts;
    }
    const at = sim.width * sim.height * 32 + faces * 16 + quarter;
    const grown = new Uint8Array(sim.field.size);
    grown.set(saved.subarray(0, at));
    grown.set(saved.subarray(at), at + quarter);
    return new Map(parts).set("field", grown);
}

// The parts of a snapshot from before P3-9 (header.bodies), which may hold
// a body in two pieces (lift took the cells lost in a step one at a time)
// and loose cells no body has (structure let go where the labelling didn't
// look, taken into slot 0 outside its shape): each body is looked at whole
// in its first step (BODY_FLAG.PARTS, bodies.wgsl's parts pass), and each
// loose cell its body didn't stamp in its last step (its claim word isn't
// the body's key, bodies.wgsl's bodyKey) is let go as a broken body's are,
// to become a body of its own.
function migrateBodies(sim, header, parts) {
    const name = `cells${header.state.cur}`;
    const saved = parts.get(name), bodies = parts.get("bodies"), motion = parts.get("bodyMotion");
    if ((header.bodies ?? 0) >= BODIES || !saved || bodies?.length !== sim.bodies.size || motion?.length !== sim.bodyMotion.size) {
        return parts;
    }
    const n = sim.width * sim.height;
    const table = new Uint32Array(motion.slice().buffer);
    const word = (b, k) => MAX_BODIES + b * BODY_TABLE_WORDS + k;
    for (let b = 0; b < MAX_BODIES; b++) {
        if (table[word(b, BODY_WORD.FLAGS)] & BODY_FLAG.ALIVE) {
            table[word(b, BODY_WORD.FLAGS)] |= BODY_FLAG.PARTS;
        }
    }
    const claims = new Uint32Array(bodies.slice((sim.bodyHeader + n) * 4, (sim.bodyHeader + 2 * n) * 4).buffer);
    const cells = new Uint32Array(saved.slice().buffer);
    for (let i = 0; i < n; i++) {
        const x = cells[4 * i];
        const slot = f16Value(cells[4 * i + 3] & 0xffff);
        if (!((x >>> CELL_FIELDS.flags[0]) & CELL_FLAGS.LOOSE) || ELEMENTS[cellTypeAt(cells, i)].kind !== KIND.SOLID || !(slot < MAX_BODIES)) {
            continue;
        }
        const alive = table[word(slot, BODY_WORD.FLAGS)] & BODY_FLAG.ALIVE;
        const key = (((table[word(slot, BODY_WORD.FRAME)] & 0xfffff) << 12) | ((FREE_BODY - slot) << 1)) >>> 0;
        if (!alive || claims[i] !== key) {
            cells[4 * i + 3] = ((cells[4 * i + 3] & 0xffff0000) | f16Bits(FREE_BODY)) >>> 0;
        }
    }
    return new Map(parts).set(name, new Uint8Array(cells.buffer)).set("bodyMotion", new Uint8Array(table.buffer));
}

// Loads a snapshot (readSnapshot's) into sim, a world of its size. Throws,
// with nothing changed, if any part doesn't fit.
export function restoreSnapshot(sim, { header, parts: saved }) {
    if (header.width !== sim.width || header.height !== sim.height) {
        throw new Error(`the snapshot is of a ${header.width}×${header.height} world, this one is ${sim.width}×${sim.height}`);
    }
    const parts = migrateBodies(sim, header, migrateField(sim, migrateTable(sim,
        migrateSwept(sim, migratePeople(sim, migrateBlocks(sim, migrateLiquid(sim, migrateAmounts(sim, header, migrateGas(sim, header, migrateWater(sim, header, saved))))))))));
    const { buffers, textures } = sim.snapshotParts();
    const rebuilt = (name) => REBUILT.has(name) || (parts.has(MIGRATED_LIQUID) && LIQUID_PARTS.includes(name));
    const fits = (name, bytes) => parts.get(name)?.length === bytes || (!parts.has(name) && ADDED.has(name)) || rebuilt(name);
    const misfits = [
        ...Object.entries(buffers).filter(([name, buffer]) => !fits(name, buffer.size)),
        ...Object.entries(textures).filter(([name, t]) => !fits(name, t.width * t.height * TEXEL_BYTES[t.format])),
    ].map(([name]) => name);
    const known = new Set([...Object.keys(buffers), ...Object.keys(textures)]);
    misfits.push(...[...parts.keys()].filter((name) => typeof name === "string" && !known.has(name) && !DROPPED.has(name)));
    if (misfits.length) {
        throw new Error(`saved by a version that lays out ${misfits.join(", ")} differently`);
    }
    const queue = sim.device.queue;
    for (const [name, buffer] of Object.entries(buffers)) {
        const saved = parts.get(name);
        queue.writeBuffer(buffer, 0, saved?.length === buffer.size ? saved : new Uint8Array(buffer.size));
    }
    for (const [name, texture] of Object.entries(textures)) {
        const row = texture.width * TEXEL_BYTES[texture.format];
        // (A rebuilt texture saved in another format loads empty, as a
        // rebuilt buffer of another size does: its bytes mean something
        // else.)
        const saved = parts.get(name);
        const data = saved?.length === row * texture.height ? saved : new Uint8Array(row * texture.height);
        queue.writeTexture({ texture }, data, { bytesPerRow: row }, [texture.width, texture.height]);
    }
    sim.restoreState(header.state);
}
