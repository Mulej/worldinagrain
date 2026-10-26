// The people pass (person.wgsl), once a step after the cells have moved: a
// workgroup per person. Its threads take in what is around the body
// together, a cell or a few each (a thread alone, reading them one after
// another, took a third of a millisecond a step for a handful of people);
// then the first is harmed by it, makes up its mind what to do, and moves.
// And the place pass, before a frame's steps, adds the people asked for and
// takes away those erased (sim.js's addPerson and erasePeople).

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> elems: array<Elem>;
@group(0) @binding(2) var<storage, read_write> cells: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> air: array<vec4<f32>>;
// Current density per cell, 2 x f16 (electric.wgsl): what shocks.
@group(0) @binding(4) var<storage, read> flow: array<u32>;
// The pull of the world's masses per light cell of 2 x 2 cells, cells/step²
// (gravity.wgsl).
@group(0) @binding(5) var gravity: texture_2d<f32>;
@group(0) @binding(6) var<storage, read_write> people: array<Person, MAX_PEOPLE>;
// The place pass's requests: how many in the first word, then two vec4s
// each: REQUEST_*, x and y (cells, as f32 bits), and the seed or the
// radius; then a second point's x and y (an erasing stroke's other end, a
// box's other corner).
@group(0) @binding(7) var<storage, read> requests: array<vec4<u32>>;
// What goes into the air (airsource.wgsl): the drag a person gives it.
@group(0) @binding(8) var<storage, read_write> source: array<atomic<u32>>;
// Radiant heat per cell (radiate.wgsl): the mean radiance reaching it (w),
// what a body there feels.
@group(0) @binding(9) var radiant: texture_2d<f32>;
// Where people's bodies are (common.wgsl's occupantWord): each stamps its
// own as it ends its step (stamp), for the next step's passes.
@group(0) @binding(10) var<storage, read_write> occupants: array<atomic<u32>>;
// Heat's clock (clock.wgsl): how real a step is about it, which its body's
// heat goes by, as the cells round it do (decision 1; REWRITE_PLAN.md 10-2).
@group(0) @binding(11) var clock: texture_2d<f32>;

// REQUEST_* and MAX_PERSON_REQUESTS come from js/shared.js.

// Moving, in cells and steps. The world is a centimetre a cell, so a person
// is ten centimetres tall, and falls and jumps as quickly as that; it walks
// about its height a second, which reads as a stroll.
const WALK_SPEED: f32 = 0.2;
const RUN_SPEED: f32 = 0.45;
const SWIM_SPEED: f32 = 0.12;
const GRIP: f32 = 0.3;           // share of the way to the speed it wants it gets a step on its feet (or swimming)
const AIR_CONTROL: f32 = 0.03;   // and in the air (it moves at most MAX_SPEED, as anything does)
const MAX_STEP: i32 = 3;         // a ledge it steps up onto as it walks
const MAX_JUMP: i32 = 8;         // and the highest it jumps up onto
const SAFE_DROP: i32 = 12;       // the deepest drop it walks off, calm
const FLEE_DROP: i32 = 30;       // and running for its life
const WADE: i32 = 2;             // liquid it wades into, calm, deep (deeper, it turns back)
const ROAM: f32 = 50.0;          // how far from home it strolls
const WIND_DRAG: f32 = 0.01;     // share of the way to the wind's speed it is blown a step
// Drag between a person (m) and the air its body stands in (wind, its
// middle's), worked out once for the two (REWRITE_PLAN.md 4e-2, as
// update.wgsl's for powders): their relative velocity w goes to
// w / (1 + k (1 + μ)), k WIND_DRAG and μ its mass over that air's, each
// taking its share of the change by the other's mass. Its new velocity; the
// air's share goes into the faces of its body's cells (airsource.wgsl), for
// the next step's air. A person is hundreds of times heavier than the air
// it stands in: walking, it takes that air along; a fan's stream stops at
// it.
fn windDrag(m: Person, wind: vec2<f32>) -> vec2<f32> {
    let dims = airDims(P.size);
    let cx = i32(floor(m.pos.x));
    let feet = i32(ceil(m.pos.y));
    var amount = 0.0;
    for (var y = feet - BODY_H; y < feet; y++) {
        for (var x = cx - BODY_HALF; x <= cx + BODY_HALF; x++) {
            let q = vec2<i32>(x, y);
            if (gasCellAt(q, dims)) {
                amount += gasAmount(gasFrom(cells[gasIndex(P.size.x * P.size.y, u32(y) * P.size.x + u32(x))], u32(y) * P.size.x + u32(x)));
            }
        }
    }
    let air = elems[AIR].rho * amount;
    if (air <= 0.0) {
        return m.vel;
    }
    let mu = PERSON_RHO * f32(BODY_H * (2 * BODY_HALF + 1)) / air;
    let w = m.vel - wind;
    let change = w * (1.0 - 1.0 / (1.0 + WIND_DRAG * (1.0 + mu)));
    let du = change * mu / (1.0 + mu);
    for (var y = feet - BODY_H; y < feet; y++) {
        for (var x = cx - BODY_HALF; x <= cx + BODY_HALF; x++) {
            let q = vec2<i32>(x, y);
            if (gasCellAt(q, dims)) {
                pushAir(u32(y) * dims.x + u32(x), du);
            }
        }
    }
    return m.vel - change / (1.0 + mu);
}

// A person's density, kg/m³, for what the air's pressure does to them
// (airsample.wgsl's pressurePush): about water's. Weighing under water
// finds 1.0 to 1.1 g/cm³ with the lungs' air left out; with a breath in, a
// person floats, so a little under water's (an estimate).
const PERSON_RHO: f32 = 985.0;
const FLOAT: f32 = 1.6;          // a liquid's push up on it, all of it under, against its weight: it floats 1/FLOAT under
const LIQUID_DRAG: f32 = 0.1;    // share of the way to the liquid's flow it is carried a step

// Harm, as a share of its health a step.
const SAFE_FALL: f32 = 30.0;     // cells it falls unhurt; DEADLY_FALL more kills it
const DEADLY_FALL: f32 = 60.0;
// Its body's heat (REWRITE_PLAN.md 10-2): a skin and a core, each with a
// heat capacity (its 30 cells' worth of water-like tissue, the skin some 15 %
// of it). The skin takes heat from what touches it, as a cell does from a
// cell beside it (common.wgsl's faceConductance, its tissue as water's,
// through a gas's film), from the radiant heat reaching it, and from its
// clothes' flames when they burn; it passes heat to the core, through the
// tissue and the blood, over about a minute. The core makes its own heat,
// more while it is cold (shivering) and less, down to giving it off, while
// it is warm (sweating), to hold it at 37 °C. Burns go by Henriques' damage
// integral (Henriques, "Studies of thermal injury V", Arch. Pathol. 43, 1947,
// 489-502: A e^(-ΔE/RT) a second past 44 °C, 1 a blistering burn) at its
// skin's surface: the hotter of the skin as a whole and where something
// touches it, the surface there at once at the touching temperature of the
// two (feel): a liquid's own, flowing over it as in Henriques' and Moritz's
// experiments; a solid's by the two's thermal effusivities, √(k ρ c), so
// 60 °C steel burns where 60 °C wood doesn't; a gas's through its film. The
// core too cold or too hot does harm as it stays so.
// Thermal radiance (σT⁴ in units of one at 1000 K) to W/m² is
// RADIANCE_FLUX; what the radiant heat at its middle (radiate.wgsl)
// brings its skin over what its skin gives back, over its body's faces.
const RADIANCE_FLUX: f32 = 56700.0;
const BODY_FACES: f32 = 26.0;     // its 3 x 10 cells' faces to the world
const BODY_EMISSIVITY: f32 = 0.98;
const SKIN_CAP: f32 = 19.0;       // J/K
const CORE_CAP: f32 = 106.0;      // J/K
const SKIN_CORE: f32 = 0.05;      // W/K: the skin's heat into the core, tissue and blood (some 19 W/(m²·K))
const SKIN_EFFUSIVITY: f32 = 1170.0;  // W s^½/(m²·K): skin's √(k ρ c)
const SKIN_FILM: f32 = 250.0;     // W/(m²·K): into the skin from its surface, through its outer layers
const BASAL: f32 = 0.3;           // W it makes at rest (a mouse's, as heavy)
const REGULATE: f32 = 1.0;        // W/K more it makes for each °C its core is under 37 (less over)
const SHIVER_MAX: f32 = 1.0;      // W, at most, more it makes so
const SWEAT_MAX: f32 = 1.0;       // and less
const FLAME_FILM: f32 = 60.0;     // W/(m²·K) from its clothes' flames, burning, at FIRE's temperature
const BURN_FROM: f32 = 44.0;      // °C its skin is burnt past
const HENRIQUES_LN_A: f32 = 226.8;  // ln of A = 3.1e98 a second
const HENRIQUES_E: f32 = 75500.0;   // ΔE / R, K
const DOSE_DEATH: f32 = 3.0;      // burn it dies of (three times blistering), its health going as it is taken
const CORE_COLD: f32 = 35.0;      // °C its core is hypothermic under
const CORE_HOT: f32 = 41.0;       // and has heatstroke over
const COLD_HURT: f32 = 0.02;      // health a second per °C past those (5 °C under: ten seconds)
const HEAT_HURT: f32 = 0.05;
const IGNITE_T: f32 = 300.0;     // what touches it this hot sets its clothes alight
const IGNITE_AIR_T: f32 = 500.0; // or gas this hot
const CATCH: f32 = 0.05;         // chance a step that a flame touching it does
const BURN_STEPS: f32 = 300.0;       // how long its clothes burn
const SPREAD: f32 = 0.12;            // chance a step it sets the air beside it alight, burning
const BREATH_STEPS: f32 = 600.0;     // it holds its breath ten seconds
const BREATHE_IN: f32 = 1.0 / 120.0; // and gets it back in two
const CHOKE_HARM: f32 = 1.0 / 180.0;
const STALE: f32 = 0.6;          // oxygen (of fresh air's) short of which it gasps (about 12 %; SMOTHER, flames', is 0.71)
const CRUSH: i32 = 12;           // cells of its body something solid can take up before it is crushed
const CRUSH_HARM: f32 = 1.0 / 600.0; // a step per cell past CRUSH
const BLAST_HARM: f32 = 0.35;
const BLAST_FLING: f32 = 0.6;
const ACID_HARM: f32 = 1.0 / 120.0;  // in full-strength acid
const SHOCK_AMPS: f32 = 0.1;     // current through a conductor it touches that stops a heart
const SHOCK_HARM: f32 = 1.0 / 30.0;
const HEAL: f32 = 1.0 / 3600.0;  // a minute to heal from nothing

// Danger, as it looks around every SCAN_EVERY steps: within SCAN_REACH
// cells either side, flames and burning things, lava, blasts, acid, voids
// and black holes, anything hot past DANGER_T, a flood coming, each weighing
// less with distance. Past FLEE_AT it runs.
const SCAN_EVERY: u32 = 4u;
const SCAN_REACH: i32 = 24;
const DANGER_T: f32 = 100.0;
const FLOOD_SPEED: f32 = 1.5;
const FLEE_AT: f32 = 0.3;
const FLEE_STEPS: f32 = 180.0;
const PANIC_TURN: f32 = 0.02;    // chance a step that it turns, on fire with no water in sight
// Bedtime, by how high the sun is (sunDirection, -1 at midnight): it goes
// home to sleep after about 21:00 and gets up about 5:00.
const BEDTIME_SUN: f32 = -0.7;
const WAKE_SUN: f32 = -0.2;
// Company.
const CHAT_REACH: f32 = 8.0;
const CHAT_NEAR: f32 = 3.0;
const CHAT_STEPS: f32 = 240.0;
const CHAT_AGAIN: f32 = 900.0;
const CHAT_EVERY: u32 = 8u;

// What it has done this step, for after it has moved.
var<private> bumped: bool;      // walked into something
var<private> fell: f32;         // cells it fell, landing
var<private> harm: f32;
var<private> worst: f32;
var<private> cause: u32;

fn cellAt(x: i32, y: i32) -> Cell {
    if (x >= 0 && x < i32(P.size.x) && y >= 0 && y < i32(P.size.y)) {
        return unpackCell(cells[u32(y) * P.size.x + u32(x)]);
    }
    // Past the sides and the bottom of the world, wall; over its top, air.
    var c: Cell;
    c.t = select(WALL, AIR, y < 0 && x >= 0 && x < i32(P.size.x));
    c.temp = AMBIENT_T;
    return c;
}

fn typeAt(x: i32, y: i32) -> u32 {
    if (x >= 0 && x < i32(P.size.x) && y >= 0 && y < i32(P.size.y)) {
        return wordType(cells[u32(y) * P.size.x + u32(x)].x);
    }
    return select(WALL, AIR, y < 0 && x >= 0 && x < i32(P.size.x));
}

// What it can't walk through: solids and powders, but not foliage.
fn blocks(t: u32) -> bool {
    let k = elems[t].kind;
    return (k == KIND_SOLID || k == KIND_POWDER) && t != PLANT;
}

fn blockedIn(x0: i32, x1: i32, y0: i32, y1: i32) -> bool {
    for (var y = y0; y <= y1; y++) {
        for (var x = x0; x <= x1; x++) {
            if (blocks(typeAt(x, y))) {
                return true;
            }
        }
    }
    return false;
}

// Its three columns at row y.
fn blockedRow(cx: i32, y: i32) -> bool {
    return blockedIn(cx - BODY_HALF, cx + BODY_HALF, y, y);
}

fn isMolten(t: u32) -> bool {
    return t == LAVA || t == MOLTEN_METAL || t == MOLTEN_GLASS;
}

// Sets the cell at (x, y) to c if it is air.
fn putInAir(x: i32, y: i32, c: Cell) {
    if (x < 0 || x >= i32(P.size.x) || y < 0 || y >= i32(P.size.y)) {
        return;
    }
    let i = u32(y) * P.size.x + u32(x);
    if (wordType(cells[i].x) == AIR) {
        // At the air's temperature, as a flame is the air it burns in
        // (block.wgsl): nothing paid for heat put in (a flame at FIRE's
        // 800 °C, ash and smoke at 300).
        var n = c;
        n.temp = cellTemp(cells[i]);
        cells[i] = packCell(n);
    }
}

// Sums and extremes the threads gather in workgroup memory: temperatures as
// ordered bits (ordered), sums of fractions as fixed point.
const FIXED: f32 = 1024.0;

fn ordered(f: f32) -> u32 {
    let b = bitcast<u32>(f);
    return select(b | 0x80000000u, ~b, (b >> 31u) == 1u);
}

fn unordered(u: u32) -> f32 {
    return bitcast<f32>(select(~u, u & 0x7fffffffu, (u >> 31u) == 1u));
}

// What touches it: its body (its three columns, BODY_H rows) and a cell
// round it (two columns either side, a row above and below): FELT_CELLS, a
// thread each (feel), gathered (felt).
const FELT_W: u32 = 5u;
const FELT_CELLS: u32 = 60u;  // FELT_W by BODY_H + 2
const FELT_FIRE: u32 = 1u;
const FELT_MOLTEN: u32 = 2u;
const FELT_GONE: u32 = 4u;
const FELT_SHOCK: u32 = 8u;
const FELT_FACE_LIQUID: u32 = 16u;
const FELT_FACE_SOLID: u32 = 32u;
const FELT_FACE_LEFT: u32 = 64u;   // something solid against its face's row, left
const FELT_FACE_RIGHT: u32 = 128u; // and right

var<workgroup> feltHot: atomic<u32>;
var<workgroup> feltHotAir: atomic<u32>;
var<workgroup> feltCold: atomic<u32>;
var<workgroup> feltFlags: atomic<u32>;
var<workgroup> feltSolid: atomic<u32>;
var<workgroup> feltLiquid: atomic<u32>;      // rows of its body with liquid in it or beside it (bits)
var<workgroup> feltLiquidCells: atomic<u32>;
// The conductance from what touches it to its skin (millionths of a W/K) and
// that times its temperature (ten-thousandths of a W), summed (feel).
var<workgroup> feltConductance: atomic<u32>;
var<workgroup> feltHeat: atomic<i32>;
var<workgroup> feltContact: atomic<u32>;  // the hottest its skin's surface is where something touches it (ordered)
var<workgroup> feltAcid: atomic<u32>;
var<workgroup> feltOxygen: atomic<u32>;
var<workgroup> feltBlast: array<atomic<i32>, 3>;
var<workgroup> feltCurrent: array<atomic<i32>, 2>;

struct Sense {
    hottest: f32,       // of what isn't gas
    hottestAir: f32,    // and of the gas about it
    coldest: f32,       // of the gas
    fire: bool,         // a flame, something burning or molten touching it
    molten: bool,       // lava or molten metal or glass in its body
    blast: vec3<f32>,   // the way away from the blast cells touching it (xy), and how many (z)
    acid: f32,          // the strongest acid in its body
    gone: bool,         // a void or a black hole touching it
    shock: bool,
    solid: i32,         // cells of its body taken up by something solid
    liquid: i32,        // and of its middle column, by a liquid
    current: vec2<f32>, // the liquid's flow in that, summed
    headLiquid: bool,
    headSolid: bool,
    oxygen: f32,        // at its face, of fresh air's: none in a liquid, steam or rock
    conductance: f32,   // W/K from what touches it to its skin
    heat: f32,          // that times their temperatures, W (with °C)
    contact: f32,       // the hottest its skin's surface is where something touches it
}

// Where the cell dx across its body and dy along it from the row over its
// head (feel's) is in the world: standing, or lying (lying).
fn bodyCell(dx: i32, dy: i32, cx: i32, feet: i32) -> vec2<i32> {
    if (lying(person)) {
        let left = (person.flags & PF_LEFT) != 0u;
        return vec2<i32>(cx - 5 + select(BODY_H - 1 - dy, dy, left), feet - 3 - dx);
    }
    return vec2<i32>(cx + dx, feet - BODY_H + dy);
}

// Asleep or dead it lies on its back along the ground, its head the way it
// faces, as figures.wgsl draws it: its three columns rows feet - 4 to feet -
// 2, its BODY_H rows columns cx - 5 to cx + 4. (Its cells in the occupants
// mask and what it felt were its standing body's: sand rested on the air
// over it, and under water to its chest it breathed at its standing face.)
fn lying(m: Person) -> bool {
    return m.state == P_SLEEP || m.state == P_DEAD;
}

// Cell k of what touches it.
fn feel(k: u32, cx: i32, feet: i32) {
    let dx = i32(k % FELT_W) - 2;
    let dy = i32(k / FELT_W) - 1;  // from the row over its head
    let at = bodyCell(dx, dy, cx, feet);
    let x = at.x;
    let y = at.y;
    let c = cellAt(x, y);
    let e = elems[c.t];
    let inBody = abs(dx) <= BODY_HALF && dy >= 0 && dy < BODY_H;
    let temp = tempOf(c);
    if (e.kind == KIND_GAS) {
        atomicMax(&feltHotAir, ordered(temp));
        atomicMin(&feltCold, ordered(temp));
    } else {
        atomicMax(&feltHot, ordered(temp));
    }
    var flags = 0u;
    if (c.t == FIRE || (c.flags & FL_BURN) != 0u) {
        flags |= FELT_FIRE;
    }
    if (isMolten(c.t)) {
        flags |= FELT_FIRE;
    }
    if (c.t == VOID || c.t == BLACK_HOLE) {
        flags |= FELT_GONE;
    }
    if (e.sigma > 0.0 && x >= 0 && x < i32(P.size.x) && y >= 0 && y < i32(P.size.y)) {
        let j = unpack2x16float(flow[u32(y) * P.size.x + u32(x)]);
        flags |= select(0u, FELT_SHOCK, dot(j, j) > SHOCK_AMPS * SHOCK_AMPS);
    }
    if (c.t == BLAST) {
        let mid = select(vec2<i32>(cx, feet - BODY_H / 2), vec2<i32>(cx, feet - 3), lying(person));
        atomicAdd(&feltBlast[0], mid.x - x);
        atomicAdd(&feltBlast[1], mid.y - y);
        atomicAdd(&feltBlast[2], 1);
    }
    // What touches its body (beside it, over its head, under its feet), and
    // how well its heat passes to its skin at the skin's temperature now.
    let side = abs(dx) == BODY_HALF + 1 && dy >= 0 && dy < BODY_H;
    let ends = abs(dx) <= BODY_HALF && (dy == -1 || dy == BODY_H);
    if (side || ends) {
        var film = e.film;
        if (e.kind == KIND_GAS && x >= 0 && x < i32(P.size.x) && y >= 0 && y < i32(P.size.y)) {
            let i = u32(y) * P.size.x + u32(x);
            film = gasFilm(c.t, gasFrom(cells[gasIndex(P.size.x * P.size.y, i)], i), temp, person.skin, windSpeedAt(vec2<i32>(x, y)));
        }
        let g = faceConductance(elems[WATER], e, film, P.dx);
        atomicAdd(&feltConductance, u32(g * 1e6));
        atomicAdd(&feltHeat, i32(g * clamp(temp, -273.0, 3000.0) * 1e4));
        // Its skin's surface there, at once (see BURN_FROM).
        var surface = temp;
        if (e.kind == KIND_GAS) {
            surface = person.skin + (temp - person.skin) * film / (film + SKIN_FILM);
        } else if (e.kind != KIND_LIQUID) {
            let effusivity = sqrt(max(e.thermal * e.heatCap * 1e6, 0.0));
            surface = (SKIN_EFFUSIVITY * person.skin + effusivity * temp) / (SKIN_EFFUSIVITY + effusivity);
        }
        if (e.thermal > 0.0) {
            atomicMax(&feltContact, ordered(surface));
        }
    }
    // Liquid and what it carries in its body or just beside it: since
    // people are in the grid (REWRITE_PLAN.md 10-1) the liquid flows round
    // their bodies, and what is beside a row of it is what it is in.
    let rowed = dy >= 0 && dy < BODY_H && (inBody || abs(dx) == BODY_HALF + 1);
    if (inBody && blocks(c.t)) {
        atomicAdd(&feltSolid, 1u);
    }
    if (rowed) {
        if (holdsSolute(c.t) && soluteOf(c) == ACID) {
            atomicMax(&feltAcid, strengthOf(c));
        }
        if (e.kind == KIND_LIQUID && (dx == 0 || !inBody)) {
            atomicOr(&feltLiquid, 1u << u32(dy));
            atomicAdd(&feltLiquidCells, 1u);
            atomicAdd(&feltCurrent[0], i32(c.vel.x * FIXED));
            atomicAdd(&feltCurrent[1], i32(c.vel.y * FIXED));
        }
        if (isMolten(c.t)) {
            flags |= FELT_MOLTEN;
        }
    }
    // Its face's row beside it: with sand (or anything solid) on both sides
    // it is buried to the face (its own cells stay clear: 10-1).
    if (dy == 2 && abs(dx) == BODY_HALF + 1 && blocks(c.t)) {
        flags |= select(FELT_FACE_RIGHT, FELT_FACE_LEFT, dx < 0);
    }
    // Its face: the sprite's eyes' row (figures.wgsl).
    if (dx == 0 && dy == 2) {
        flags |= select(0u, FELT_FACE_LIQUID, e.kind == KIND_LIQUID);
        flags |= select(0u, FELT_FACE_SOLID, blocks(c.t));
        // The air's oxygen, from what it holds (the gas plane, after the cells).
        var gas = noGas();
        if (carriesOxygen(c.t) && x >= 0 && x < i32(P.size.x) && y >= 0 && y < i32(P.size.y)) {
            let i = u32(y) * P.size.x + u32(x);
            gas = gasFrom(cells[gasIndex(P.size.x * P.size.y, i)], i);
        }
        atomicStore(&feltOxygen, bitcast<u32>(oxygenPressure(c.t, gas, c.temp)));
    }
    if (flags != 0u) {
        atomicOr(&feltFlags, flags);
    }
}

fn felt() -> Sense {
    var s: Sense;
    s.hottest = unordered(atomicLoad(&feltHot));
    s.hottestAir = unordered(atomicLoad(&feltHotAir));
    s.coldest = unordered(atomicLoad(&feltCold));
    let flags = atomicLoad(&feltFlags);
    s.fire = (flags & FELT_FIRE) != 0u;
    s.molten = (flags & FELT_MOLTEN) != 0u;
    s.gone = (flags & FELT_GONE) != 0u;
    s.shock = (flags & FELT_SHOCK) != 0u;
    // Its face in a liquid: the row of its eyes (figures.wgsl), in it or
    // beside it.
    s.headLiquid = (flags & FELT_FACE_LIQUID) != 0u || (atomicLoad(&feltLiquid) & 4u) != 0u;
    s.headSolid = (flags & FELT_FACE_SOLID) != 0u || (flags & (FELT_FACE_LEFT | FELT_FACE_RIGHT)) == (FELT_FACE_LEFT | FELT_FACE_RIGHT);
    s.blast = vec3<f32>(f32(atomicLoad(&feltBlast[0])), f32(atomicLoad(&feltBlast[1])), f32(atomicLoad(&feltBlast[2])));
    s.acid = shareOf(atomicLoad(&feltAcid));
    s.solid = i32(atomicLoad(&feltSolid));
    // Rows of it in liquid, and the liquid's flow there, as the sum over
    // that many rows of its mean.
    s.liquid = i32(countOneBits(atomicLoad(&feltLiquid)));
    s.conductance = f32(atomicLoad(&feltConductance)) * 1e-6;
    s.contact = unordered(atomicLoad(&feltContact));
    s.heat = f32(atomicLoad(&feltHeat)) * 1e-4;
    let cells = f32(max(atomicLoad(&feltLiquidCells), 1u));
    s.current = vec2<f32>(f32(atomicLoad(&feltCurrent[0])), f32(atomicLoad(&feltCurrent[1]))) / FIXED / cells * f32(s.liquid);
    s.oxygen = bitcast<f32>(atomicLoad(&feltOxygen));
    return s;
}

// The ground the columns either side of it offer, just ahead (where it
// steps) and two further (where a jump lands it): how far up from its feet
// (WALL_RISE: higher than it can jump, a wall; DROP: nothing within
// FLEE_DROP below), and whether there is a liquid on it deeper than it
// wades, or one it wouldn't set foot in at all: acid, lava, molten metal or
// glass. AHEAD_ROWS of each of the four columns from MAX_JUMP + 1 over its
// feet, a thread a cell (lookAhead), each column's first solid row and first
// liquid one (with whether it is bad, the low bit) gathered (ahead).
const WALL_RISE: i32 = 99;
const DROP: i32 = -99;
const AHEAD_ROWS: u32 = 40u;  // MAX_JUMP + 1 + FLEE_DROP + 1
const AHEAD_CELLS: u32 = 160u;
const NOTHING: u32 = 0xffffffffu;

var<workgroup> aheadSolid: array<atomic<u32>, 4>;
var<workgroup> aheadLiquid: array<atomic<u32>, 4>;

struct Ahead {
    rise: i32,
    deep: bool,
    bad: bool,
}

// Column c: 0 just ahead to the right, 1 a jump's length that way, 2 and 3
// to the left.
fn aheadColumn(c: u32, cx: i32) -> i32 {
    return cx + select(1, -1, c >= 2u) * (BODY_HALF + 1 + 2 * i32(c & 1u));
}

fn lookAhead(k: u32, cx: i32, feet: i32) {
    let c = k / AHEAD_ROWS;
    let r = k % AHEAD_ROWS;
    let x = aheadColumn(c, cx);
    let y = feet - MAX_JUMP - 1 + i32(r);
    let t = typeAt(x, y);
    if (blocks(t)) {
        atomicMin(&aheadSolid[c], r);
    } else if (elems[t].kind == KIND_LIQUID) {
        let bad = isMolten(t) || (holdsSolute(t) && soluteOf(cellAt(x, y)) == ACID);
        atomicMin(&aheadLiquid[c], (r << 1u) | select(0u, 1u, bad));
    }
}

fn ahead(c: u32) -> Ahead {
    var a: Ahead;
    let solid = atomicLoad(&aheadSolid[c]);
    let liquid = atomicLoad(&aheadLiquid[c]);
    let wet = liquid != NOTHING && (liquid >> 1u) < solid;
    a.bad = wet && (liquid & 1u) != 0u;
    if (solid == 0u) {
        a.rise = WALL_RISE;
    } else if (solid == NOTHING) {
        a.rise = DROP;
        a.deep = wet;
    } else {
        a.rise = MAX_JUMP + 1 - i32(solid);
        a.deep = wet && i32(solid) - i32(liquid >> 1u) > WADE;
    }
    return a;
}

// Danger about it, every SCAN_EVERY steps: SCAN_CELLS, every other cell
// within SCAN_REACH either side and from 18 over its feet to 2 under, a few
// a thread (scan), gathered (seen).
const SCAN_W: u32 = 25u;  // SCAN_REACH * 2 / 2 + 1
const SCAN_CELLS: u32 = 275u;  // SCAN_W by 11 rows

var<workgroup> scanDanger: atomic<i32>;
var<workgroup> scanSide: atomic<i32>;
var<workgroup> scanWater: atomic<u32>;  // nearest water: its distance, doubled, plus 1 on the right

fn scan(k: u32, cx: i32, feet: i32) {
    let dx = 2 * i32(k % SCAN_W) - SCAN_REACH;
    let dy = 2 * i32(k / SCAN_W) - 18;
    let c = cellAt(cx + dx, feet + dy);
    let e = elems[c.t];
    var w = 0.0;
    if (c.t == FIRE || (c.flags & FL_BURN) != 0u || c.t == VOID || c.t == BLACK_HOLE) {
        w = 1.0;
    } else if (isMolten(c.t)) {
        w = 1.5;
    } else if (c.t == BLAST) {
        w = 3.0;
    } else if (holdsSolute(c.t) && soluteOf(c) == ACID) {
        w = 0.4 * shareOf(strengthOf(c));
    } else {
        if (e.kind == KIND_LIQUID && dot(c.vel, c.vel) > FLOOD_SPEED * FLOOD_SPEED) {
            w = 0.6;
        }
        let temp = tempOf(c);
        if (temp > DANGER_T) {
            w = max(w, min((temp - DANGER_T) / 200.0, 1.5));
        }
    }
    if (w > 0.0) {
        let near = w / (1.0 + length(vec2<f32>(f32(dx), f32(dy))) / 6.0);
        atomicAdd(&scanDanger, i32(near * FIXED));
        atomicAdd(&scanSide, i32(near * f32(dx) * FIXED));
    }
    if (isWater(c.t) && soluteOf(c) != ACID) {
        atomicMin(&scanWater, u32(2 * abs(dx)) + select(0u, 1u, dx >= 0));
    }
}

// The danger (x, weighted), the side it is on (y, -1 or 1), and the side the
// nearest water is on (z, 0 for none in sight).
fn seen() -> vec3<f32> {
    let water = atomicLoad(&scanWater);
    return vec3<f32>(
        f32(atomicLoad(&scanDanger)) / FIXED,
        select(-1.0, 1.0, atomicLoad(&scanSide) >= 0),
        select(select(-1.0, 1.0, (water & 1u) != 0u), 0.0, water == NOTHING),
    );
}

fn hurt(amount: f32, why: u32) {
    harm += amount;
    if (amount > worst) {
        worst = amount;
        cause = why;
    }
}

fn facing(m: Person) -> i32 {
    return select(1, -1, (m.flags & PF_LEFT) != 0u);
}

fn face(m: Person, dir: f32) -> Person {
    var n = m;
    n.flags = select(n.flags & ~PF_LEFT, n.flags | PF_LEFT, dir < 0.0);
    return n;
}

// Moves it by its velocity, a cell at a time, up or down and then along,
// stopping at what blocks it; along, with canStep, it steps up onto a ledge
// up to MAX_STEP high. Sets bumped and fell.
fn moveBody(p: Person, canStep: bool) -> Person {
    var m = p;
    bumped = false;
    fell = 0.0;
    let n = i32(ceil(abs(m.vel.y)));
    for (var k = 0; k < n; k++) {
        let ny = m.pos.y + m.vel.y / f32(n);
        let cx = i32(floor(m.pos.x));
        if (m.vel.y > 0.0) {
            // Its feet reaching a row: on it, if it is solid.
            let row = i32(ceil(ny)) - 1;
            if (row > i32(ceil(m.pos.y)) - 1 && blockedRow(cx, row)) {
                m.pos.y = f32(row);
                fell = m.pos.y - m.fallFrom;
                m.vel.y = 0.0;
                break;
            }
        } else {
            // Its head reaching one: it stops under it.
            let row = i32(ceil(ny)) - BODY_H;
            if (row < i32(ceil(m.pos.y)) - BODY_H && blockedRow(cx, row)) {
                m.pos.y = f32(row + 1 + BODY_H);
                m.vel.y = 0.0;
                break;
            }
        }
        m.pos.y = ny;
    }
    let along = i32(ceil(abs(m.vel.x)));
    for (var k = 0; k < along; k++) {
        let nx = m.pos.x + m.vel.x / f32(along);
        let cx = i32(floor(m.pos.x));
        let ncx = i32(floor(nx));
        if (ncx != cx) {
            let edge = ncx + select(-BODY_HALF, BODY_HALF, ncx > cx);
            let feet = i32(ceil(m.pos.y));
            if (blockedIn(edge, edge, feet - BODY_H, feet - 1)) {
                var up = 0;
                for (var h = 1; canStep && h <= MAX_STEP; h++) {
                    if (!blockedIn(ncx - BODY_HALF, ncx + BODY_HALF, feet - h - BODY_H, feet - h - 1)) {
                        up = h;
                        break;
                    }
                }
                if (up == 0) {
                    bumped = true;
                    m.vel.x = 0.0;
                    break;
                }
                m.pos.y = f32(feet - up);
            }
        }
        m.pos.x = nx;
    }
    return m;
}

// Burnt up: a little ash where it stood and a puff of smoke, and its slot freed.
fn ashes(m: Person) {
    let cx = i32(floor(m.pos.x));
    let feet = i32(ceil(m.pos.y));
    for (var dy = 1; dy <= 3; dy++) {
        for (var dx = -BODY_HALF; dx <= BODY_HALF; dx++) {
            putInAir(cx + dx, feet - dy, makeCell(ASH, 300.0));
        }
    }
    for (var dy = BODY_H - 3; dy < BODY_H; dy++) {
        putInAir(cx, feet - dy, makeCell(SMOKE, 300.0));
    }
}

fn die(p: Person, why: u32) -> Person {
    var m = p;
    m.cause = why;
    m.flags &= ~(PF_BURNING | PF_SHOCK);
    m.health = 0.0;
    if (why == DIED_BURNT) {
        ashes(m);
        m.state = P_NONE;
        return m;
    }
    m.state = P_DEAD;
    m.timer = DEAD_STEPS;
    m.since = 0.0;
    return m;
}

// (The pull on it where it is, the Earth's and the world's masses', is
// pull.wgsl's pullAt.) A person stands up the screen, on what is under its
// feet: what it jumps against is the pull's share down the screen there,
// none where it pulls up or sideways only.
fn footing(m: Person) -> f32 {
    return max(pullAt(m.pos - vec2<f32>(0.0, f32(BODY_H) * 0.5)).y, 0.0);
}

var<workgroup> person: Person;
var<workgroup> personState: u32;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let who = wid.x;
    if (li == 0u) {
        person = people[who];
        personState = person.state;
        atomicStore(&feltHot, ordered(-1e9));
        atomicStore(&feltContact, ordered(-1e9));
        atomicStore(&feltHotAir, ordered(-1e9));
        atomicStore(&feltCold, ordered(1e9));
        atomicStore(&scanWater, NOTHING);
        for (var c = 0u; c < 4u; c++) {
            atomicStore(&aheadSolid[c], NOTHING);
            atomicStore(&aheadLiquid[c], NOTHING);
        }
    }
    if (workgroupUniformLoad(&personState) == P_NONE) {
        return;
    }
    // What is about it, the threads together.
    let cx = i32(floor(person.pos.x));
    let feet = i32(ceil(person.pos.y));
    let scanning = (P.frame + who) % SCAN_EVERY == 0u;
    if (li < FELT_CELLS) {
        feel(li, cx, feet);
    }
    for (var k = li; scanning && k < SCAN_CELLS; k += 64u) {
        scan(k, cx, feet);
    }
    for (var k = li; k < AHEAD_CELLS; k += 64u) {
        lookAhead(k, cx, feet);
    }
    workgroupBarrier();
    if (li != 0u) {
        return;
    }

    // The rest the first thread alone.
    var m = person;
    seedRng(P.seed, P.frame, who ^ m.seed);
    harm = 0.0;
    worst = 0.0;
    cause = 0u;
    let s = felt();
    if (s.gone) {
        // Swallowed whole.
        people[who].state = P_NONE;
        return;
    }
    m.flags &= ~(PF_SHOCK | PF_HURT | PF_UNDER | PF_SWIM);
    let swimming = s.liquid * 2 > BODY_H;
    if (swimming) {
        m.flags |= PF_SWIM;
    }

    // What pushes and pulls it: gravity, the wind and the air's pressure (a
    // blast's above all), a liquid it is in, which floats it and carries it
    // along, and a blast going off against it.
    let middle = m.pos - vec2<f32>(0.0, f32(BODY_H) * 0.5);
    let a = sampleAir(middle);
    let pull = pullAt(middle);
    m.vel += pull + pressurePush(a.grad, PERSON_RHO);
    m.vel = windDrag(m, a.vel);
    if (s.liquid > 0) {
        // Floated against the pull, whichever way it goes.
        m.vel -= pull * FLOAT * f32(s.liquid) / f32(BODY_H);
        m.vel += (s.current / f32(s.liquid) - m.vel) * LIQUID_DRAG;
    }
    if (s.blast.z > 0.0) {
        m.vel += normalize(s.blast.xy + vec2<f32>(0.0, -0.5)) * BLAST_FLING;
    }

    if (m.state == P_DEAD) {
        // It lies where it fell, or floats, until it fades.
        if (s.molten) {
            people[who].state = P_NONE;
            return;
        }
        m.vel.x *= 0.9;
        m.vel = clamp(m.vel, vec2<f32>(-MAX_SPEED), vec2<f32>(MAX_SPEED));
        m = moveBody(m, false);
        m.since += 1.0;
        m.timer -= 1.0;
        if (m.timer <= 0.0) {
            m.state = P_NONE;
        }
        keep(who, m);
        return;
    }

    // Harm.
    if (s.molten) {
        people[who] = die(m, DIED_BURNT);
        return;
    }
    // Its body's heat over the seconds the step stands for in heat about it
    // (heat's clock at its middle: real time by hot matter, the day's away
    // from it, as for the cells it touches, which would otherwise cool or
    // warm hundreds of times faster than it): the skin's balance taken
    // implicitly in what passes linearly (from what touches it, into the
    // core), the radiant heat and its clothes' flames as they are now; then
    // the core's.
    let mid = clamp(vec2<i32>(cx, feet - BODY_H / 2), vec2<i32>(0), vec2<i32>(P.size) - 1);
    let tau = clockStep(textureLoad(clock, vec2<u32>(mid) / 2u, 0).x, P.dayDt, P.clockSpan);
    let area = BODY_FACES * CELL_M * CELL_M;
    let lit = clamp(vec2<i32>(cx, feet - BODY_H / 2), vec2<i32>(0), vec2<i32>(textureDimensions(radiant)) - 1);
    // (Its 8 lowest bits are radiate.wgsl's, how real-time the hottest
    // thing reaching it is: left off.)
    let reaching = bitcast<f32>(bitcast<u32>(textureLoad(radiant, lit, 0).w) & ~0xffu);
    let rays = BODY_EMISSIVITY * RADIANCE_FLUX * (max(reaching, 0.0) - thermalRadiance(m.skin)) * area;
    let flames = select(0.0, FLAME_FILM * area * (elems[FIRE].spawnT - m.skin), (m.flags & PF_BURNING) != 0u);
    let k = tau / SKIN_CAP;
    m.skin = (m.skin + k * (s.heat + SKIN_CORE * m.core + rays + flames)) / (1.0 + k * (s.conductance + SKIN_CORE));
    let made = BASAL + clamp(REGULATE * (37.0 - m.core), -SWEAT_MAX, SHIVER_MAX);
    m.core += (SKIN_CORE * (m.skin - m.core) + made) * tau / CORE_CAP;
    // (Burnt by its own flames, or by what it touches, the light or the heat about it.)
    let scorched = select(DIED_HEAT, DIED_BURNT, (m.flags & PF_BURNING) != 0u);
    let surface = max(m.skin, s.contact);
    if (surface > BURN_FROM) {
        let taken = min(exp(HENRIQUES_LN_A - HENRIQUES_E / (surface + KELVIN)) * tau, DOSE_DEATH);
        m.dose += taken;
        hurt(taken / DOSE_DEATH, scorched);
    }
    if (m.core < CORE_COLD) {
        hurt((CORE_COLD - m.core) * COLD_HURT * tau, DIED_COLD);
    }
    if (m.core > CORE_HOT) {
        hurt((m.core - CORE_HOT) * HEAT_HURT * tau, DIED_HEAT);
    }
    let wet = s.liquid >= 3;
    if ((m.flags & PF_BURNING) == 0u && !wet
        && ((s.fire && rnd() < CATCH) || s.hottest > IGNITE_T || s.hottestAir > IGNITE_AIR_T)) {
        m.flags |= PF_BURNING;
        m.burn = BURN_STEPS * (0.8 + 0.4 * rnd());
    }
    if ((m.flags & PF_BURNING) != 0u) {
        m.burn -= 1.0;
        if (wet || m.burn <= 0.0) {
            m.flags &= ~PF_BURNING;
        } else {
            // It sets the air beside it alight, and what that touches.
            if (rnd() < SPREAD) {
                let side = select(-2, 2, rnd() < 0.5);
                let up = 1 + i32(rnd() * f32(BODY_H));
                putInAir(cx + side, feet - up, makeCell(FIRE, elems[FIRE].spawnT));
            }
        }
    }
    if (s.blast.z > 0.0) {
        hurt(BLAST_HARM, DIED_BLAST);
    }
    if (s.acid > 0.0) {
        hurt(s.acid * ACID_HARM, DIED_ACID);
    }
    if (s.shock) {
        m.flags |= PF_SHOCK;
        hurt(SHOCK_HARM, DIED_SHOCK);
    }
    if (s.solid > CRUSH) {
        hurt(f32(s.solid - CRUSH) * CRUSH_HARM, DIED_CRUSHED);
    }
    // Breathing: under a liquid or buried it holds its breath, in stale air
    // or smoke it gasps, the more the staler; out of breath it chokes.
    let held = s.headLiquid || s.headSolid;
    if (held || s.oxygen < STALE) {
        m.flags |= PF_UNDER;
        m.breath -= select(1.0 - s.oxygen / STALE, 1.0, held) / BREATH_STEPS;
    } else {
        m.breath = min(m.breath + BREATHE_IN, 1.0);
    }
    if (m.breath <= 0.0) {
        m.breath = 0.0;
        hurt(CHOKE_HARM, select(DIED_CHOKED, DIED_DROWNED, s.headLiquid));
    }

    // Making up its mind.
    let sun = -sunDirection(P.day).y;
    let night = sun < BEDTIME_SUN;
    let burning = (m.flags & PF_BURNING) != 0u;
    m.timer -= 1.0;
    m.since = max(m.since - 1.0, 0.0);
    if (scanning) {
        let d = seen();
        if (burning && d.z != 0.0) {
            // On fire, for the water.
            m.state = P_FLEE;
            m.timer = FLEE_STEPS;
            m = face(m, d.z);
        } else if (d.x > FLEE_AT) {
            m.state = P_FLEE;
            m.timer = FLEE_STEPS;
            m = face(m, -d.y);
        }
    }
    if (burning) {
        if (m.state != P_FLEE) {
            m.state = P_FLEE;
            m.timer = FLEE_STEPS;
        }
        if (rnd() < PANIC_TURN) {
            m = face(m, -f32(facing(m)));
        }
    }
    let homeward = select(1.0, -1.0, m.home < m.pos.x);
    switch m.state {
        case P_IDLE: {
            if (rnd() < 0.004) {
                m = face(m, -f32(facing(m)));
            }
            if (m.timer <= 0.0) {
                m.state = P_WALK;
                m.timer = 120.0 + 360.0 * rnd();
                m = face(m, select(select(-1.0, 1.0, rnd() < 0.5), homeward, abs(m.pos.x - m.home) > ROAM * 0.5));
            }
        }
        case P_WALK: {
            if (abs(m.pos.x - m.home) > ROAM && f32(facing(m)) != homeward) {
                m = face(m, homeward);
            }
            if (m.timer <= 0.0) {
                m.state = P_IDLE;
                m.timer = 60.0 + 180.0 * rnd();
            }
        }
        case P_FLEE: {
            if (m.timer <= 0.0 && !burning) {
                m.state = P_IDLE;
                m.timer = 60.0;
            }
        }
        case P_CHAT: {
            if (m.timer <= 0.0) {
                m.state = P_IDLE;
                m.timer = 30.0 + 60.0 * rnd();
                m.since = CHAT_AGAIN;
            }
        }
        case P_SLEEP: {
            if (sun > WAKE_SUN) {
                m.state = P_IDLE;
                m.timer = 30.0 + 120.0 * rnd();
            }
        }
        default: {}
    }
    // At night it goes home and sleeps.
    if (night && (m.state == P_IDLE || m.state == P_WALK)) {
        if (abs(m.pos.x - m.home) > 3.0) {
            m.state = P_WALK;
            m.timer = 60.0;
            m = face(m, homeward);
        } else if ((m.flags & PF_GROUND) != 0u) {
            m.state = P_SLEEP;
        }
    }
    // Someone about, free to talk, as it is: it stops for a chat.
    if ((m.state == P_IDLE || m.state == P_WALK) && !night && m.since <= 0.0 && P.frame % CHAT_EVERY == 0u) {
        // (Read as the others' workgroups go: some as they were, some as
        // they are now, all the same for this.)
        var best = CHAT_REACH + 1.0;
        var other = 0.0;
        for (var k = 0u; k < MAX_PEOPLE; k++) {
            let q = people[k];
            let dx = q.pos.x - m.pos.x;
            if (k != who && (q.state == P_IDLE || q.state == P_WALK) && q.since <= 0.0 && abs(q.pos.y - m.pos.y) < 2.0
                && abs(dx) >= CHAT_NEAR && abs(dx) < best) {
                best = abs(dx);
                other = dx;
            }
        }
        if (best <= CHAT_REACH) {
            m.state = P_CHAT;
            m.timer = CHAT_STEPS;
            m = face(m, other);
        }
    }

    // Where it is going, and what is in its way.
    let grounded = (m.flags & PF_GROUND) != 0u;
    let fleeing = m.state == P_FLEE;
    var want = 0.0;
    if (m.state == P_WALK) {
        want = WALK_SPEED;
    } else if (fleeing) {
        want = RUN_SPEED;
    }
    if (swimming && want > 0.0) {
        want = SWIM_SPEED;
    }
    if (want > 0.0 && (grounded || swimming)) {
        // A wall it can't jump, a drop too deep, water too deep to wade,
        // acid or lava, there or where a jump would land it: it turns
        // back. Swimming, only from a wall; a bank it can reach, it climbs
        // out onto.
        let side = select(0u, 2u, facing(m) < 0);
        let look = ahead(side);
        let drop = select(SAFE_DROP, FLEE_DROP, fleeing);
        var turn = look.rise > MAX_JUMP || look.bad
            || (!swimming && (look.rise < -drop || (look.deep && !fleeing)));
        if (!turn && look.rise > MAX_STEP) {
            let landing = ahead(side + 1u);
            turn = landing.bad || (landing.deep && !fleeing);
        }
        if (turn && fleeing && !burning) {
            // Cornered: back the way it came is what it is running from. It
            // cowers.
            want = 0.0;
        } else if (turn) {
            m = face(m, -f32(facing(m)));
        } else if (look.rise > MAX_STEP && footing(m) > 0.0) {
            m.vel.y = -sqrt(2.0 * footing(m) * (f32(look.rise) + 1.5));
        }
    }
    let aim = want * f32(facing(m));
    m.vel.x += (aim - m.vel.x) * select(AIR_CONTROL, GRIP, grounded || swimming);
    m.vel = clamp(m.vel, vec2<f32>(-MAX_SPEED), vec2<f32>(MAX_SPEED));

    // Moving. Something solid in its body (sand poured over it), with room
    // over its head, it climbs up out of.
    if (s.solid > 0 && !blockedRow(cx, feet - BODY_H - 1) && P.frame % 2u == 0u) {
        m.pos.y -= 1.0;
    }
    let was = m.pos.x;
    m = moveBody(m, grounded || swimming);
    let ncx = i32(floor(m.pos.x));
    let onGround = abs(m.pos.y - round(m.pos.y)) < 0.01 && blockedRow(ncx, i32(round(m.pos.y)));
    m.flags = select(m.flags & ~PF_GROUND, m.flags | PF_GROUND, onGround);
    if (onGround || swimming) {
        if (onGround && fell > SAFE_FALL) {
            hurt((fell - SAFE_FALL) / DEADLY_FALL, DIED_FALL);
        }
        m.fallFrom = m.pos.y;
    } else {
        m.fallFrom = min(m.fallFrom, m.pos.y);
    }
    if (bumped && want > 0.0) {
        if (fleeing && onGround && footing(m) > 0.0) {
            m.vel.y = -sqrt(2.0 * footing(m) * (f32(MAX_JUMP) + 1.5));
        } else if (night && onGround && m.state == P_WALK) {
            // Something between it and home: it sleeps where it is.
            m.state = P_SLEEP;
        } else if (!fleeing) {
            m = face(m, -f32(facing(m)));
        }
    }
    if (onGround || swimming) {
        m.stride += abs(m.pos.x - was);
    }

    // Its health.
    m.health -= harm;
    if (harm > 0.0) {
        m.flags |= PF_HURT;
    } else {
        m.health = min(m.health + HEAL, 1.0);
    }
    if (m.health <= 0.0) {
        m = die(m, cause);
    }
    keep(who, m);
}

// Person m, slot who, as it ends its step: written back, and, there still,
// its body's cells marked in the occupants mask where it now stands
// (common.wgsl's occupantWord), for the next step's passes to keep out of
// and flow round.
fn keep(who: u32, m: Person) {
    people[who] = m;
    if (m.state == P_NONE) {
        return;
    }
    let cx = i32(floor(m.pos.x));
    let feet = i32(ceil(m.pos.y));
    let word = occupantWord(who, m.vel);
    // Standing, or lying (lying()).
    var lo = vec2<i32>(cx - BODY_HALF, feet - BODY_H);
    var hi = vec2<i32>(cx + BODY_HALF, feet - 1);
    if (lying(m)) {
        lo = vec2<i32>(cx - 5, feet - 3 - BODY_HALF);
        hi = vec2<i32>(cx + 4, feet - 3 + BODY_HALF);
    }
    // (Where two overlap, the greater word, whichever workgroup is first:
    // as the last to store it, the cell and the liquid's obstacle there
    // went by the GPU's timing, and two runs from one seed parted.)
    for (var y = lo.y; y <= hi.y; y++) {
        for (var x = lo.x; x <= hi.x; x++) {
            if (x >= 0 && x < i32(P.size.x) && y >= 0 && y < i32(P.size.y)) {
                atomicMax(&occupants[u32(y) * P.size.x + u32(x)], word);
            }
        }
    }
}

// A new person, its feet at `at`, lifted out of anything solid it was put into.
fn newPerson(at: vec2<f32>, seed: u32) -> Person {
    var m: Person;
    m.pos = vec2<f32>(floor(at.x) + 0.5, round(at.y));
    let cx = i32(floor(m.pos.x));
    for (var k = 0; k < 4 * BODY_H; k++) {
        let feet = i32(m.pos.y);
        if (!blockedIn(cx - BODY_HALF, cx + BODY_HALF, feet - BODY_H, feet - 1)) {
            break;
        }
        m.pos.y -= 1.0;
    }
    m.state = P_IDLE;
    m.flags = select(0u, PF_LEFT, (seed & 1u) != 0u);
    m.timer = 30.0 + f32(seed % 90u);
    m.health = 1.0;
    m.breath = 1.0;
    m.seed = seed;
    m.home = m.pos.x;
    m.fallFrom = m.pos.y;
    m.since = f32(seed % 600u);  // not all of them ready to chat at once
    m.skin = 34.0;
    m.core = 37.0;
    return m;
}

// How far apart the segments a-b and c-d come: 0 where they cross, else
// the least of their ends' distances from the other.
fn segmentsApart(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>, d: vec2<f32>) -> f32 {
    let side = vec4<f32>(cross2(b - a, c - a), cross2(b - a, d - a), cross2(d - c, a - c), cross2(d - c, b - c));
    if (side.x * side.y < 0.0 && side.z * side.w < 0.0) {
        return 0.0;
    }
    return min(min(pointApart(a, c, d), pointApart(b, c, d)), min(pointApart(c, a, b), pointApart(d, a, b)));
}

fn cross2(u: vec2<f32>, v: vec2<f32>) -> f32 {
    return u.x * v.y - u.y * v.x;
}

// How far p is from the segment a-b.
fn pointApart(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let ab = b - a;
    let t = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0);
    return length(p - a - ab * t);
}

@compute @workgroup_size(1)
fn place() {
    let n = min(requests[0].x, MAX_PERSON_REQUESTS);
    for (var k = 0u; k < n; k++) {
        let r = requests[1u + 2u * k];
        let at = vec2<f32>(bitcast<f32>(r.y), bitcast<f32>(r.z));
        let to = bitcast<vec2<f32>>(requests[2u + 2u * k].xy);
        if (r.x == REQUEST_ERASE) {
            // Everyone whose body comes within the radius of the stroke.
            let radius = bitcast<f32>(r.w);
            for (var j = 0u; j < MAX_PEOPLE; j++) {
                let q = people[j];
                let body = segmentsApart(at, to, vec2<f32>(q.pos.x, q.pos.y - f32(BODY_H)), q.pos);
                if (q.state != P_NONE && body <= radius + 1.5) {
                    people[j].state = P_NONE;
                }
            }
        } else if (r.x == REQUEST_ERASE_BOX) {
            // Everyone whose body is in the box, a cell either way.
            let lo = min(at, to) - 1.0;
            let hi = max(at, to) + 1.0;
            for (var j = 0u; j < MAX_PEOPLE; j++) {
                let q = people[j];
                if (q.state != P_NONE && q.pos.x >= lo.x && q.pos.x <= hi.x && q.pos.y >= lo.y && q.pos.y - f32(BODY_H) <= hi.y) {
                    people[j].state = P_NONE;
                }
            }
        } else if (r.x == REQUEST_PLACE) {
            for (var j = 0u; j < MAX_PEOPLE; j++) {
                if (people[j].state == P_NONE) {
                    people[j] = newPerson(at, r.w);
                    break;
                }
            }
        }
    }
}
