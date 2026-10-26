// Margolus pass: each invocation owns one 2x2 block and is the only one that
// touches those cells, so swaps and reactions are race-free. The block grid is
// shifted by P.offset between passes so every neighbor pair gets its turn.
//
// Slot layout:  0 1
//               2 3     (y grows downward)
//
// The block's cells live in four registers (s0..s3), packed as stored, and
// slot(i) / setSlot(i, c) pick one with select() rather than indexing an
// array: an array indexed by a runtime value can't stay in registers on
// most GPUs and spills to much slower memory.

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> elems: array<Elem>;
@group(0) @binding(2) var<storage, read_write> cells: array<vec4<u32>>;
// For the air pressure, which moves boiling points (transitionsAt).
@group(0) @binding(3) var<storage, read> air: array<vec4<f32>>;
// The pull of the world's masses per light cell of 2 x 2 cells (gravity.wgsl),
// for down (pull.wgsl).
@group(0) @binding(4) var gravity: texture_2d<f32>;
// What goes into the air (airsource.wgsl): the gas a reaction turns into
// something else held (handOn).
@group(0) @binding(5) var<storage, read_write> source: array<atomic<u32>>;
// Where people's bodies are (common.wgsl's occupantWord): nothing moves into
// their cells (REWRITE_PLAN.md 10-1), so sand poured on a person rests on
// them.
@group(0) @binding(7) var<storage, read> occupants: array<u32>;

// A flame against a surface heats it with some 50 kW/m² (Drysdale, An
// Introduction to Fire Dynamics, 2011, ch. 6: 20-60 for flames a few
// centimetres to a metre high), and two cells beside each other meet about
// twice a step in this pass (lightChance).
const FLAME_FLUX: f32 = 5e4;
const PILOT_TRIES: f32 = 2.0;
// Chances per pairing that a growing tip grows: straight up, or up while
// leaning toward the light; diagonally up toward the light; and branching off
// to a side.
const PLANT_RISE: f32 = 0.05;
const PLANT_RISE_LEANING: f32 = 0.02;
const PLANT_BEND: f32 = 0.04;
const PLANT_BRANCH: f32 = 0.008;
const FROST_CRACK: f32 = 0.01;  // chance per step, over its hardness, that ice with no room to grow cracks what holds it in

const SCATTER: f32 = 0.1;       // random sideways kick of powders that land hard
const BOUNCE: f32 = 0.4;        // share of speed kept (reversed) after a sideways hit
const IMPACT_SPEED: f32 = 2.5;  // landing speed that sets off impact-sensitive elements

var<private> s0: vec4<u32>;
var<private> s1: vec4<u32>;
var<private> s2: vec4<u32>;
var<private> s3: vec4<u32>;
// Which slot each slot's cell came from, two bits each (swapSlots), for
// the gas plane to go with them once they have moved (main).
var<private> perm: u32;
const UNMOVED: u32 = 0xe4u;  // 3, 2, 1, 0
var<private> moved: u32;
var<private> acting: u32;   // one bit per slot that can act on a neighbor (canAct)
var<private> inWorld: u32;  // one bit per slot; blocks at the edges hang over it
var<private> origin: vec2<i32>;  // the block's top left cell

fn isInWorld(i: u32) -> bool {
    return ((inWorld >> i) & 1u) != 0u;
}

// Down (common.wgsl): the block's, from the total pull at its first cell,
// set in main; and whether that pull is too weak to settle anything
// (WEIGHTLESS), when nothing slumps or slides.
var<private> blockDown: vec2<i32>;
var<private> weightless: bool;

// A step's sign along down (-1 up, 0 across, 1 down) and across it (+ on
// the side +x is under the Earth's down).
fn stepDown(dx: i32, dy: i32) -> i32 {
    return sign(dx * blockDown.x + dy * blockDown.y);
}

fn stepSide(dx: i32, dy: i32) -> i32 {
    let side = turnStep(blockDown, -2);
    return sign(dx * side.x + dy * side.y);
}

fn slotRaw(i: u32) -> vec4<u32> {
    return select(select(s0, s1, i == 1u), select(s2, s3, i == 3u), i >= 2u);
}

fn setSlotRaw(i: u32, v: vec4<u32>) {
    s0 = select(s0, v, i == 0u);
    s1 = select(s1, v, i == 1u);
    s2 = select(s2, v, i == 2u);
    s3 = select(s3, v, i == 3u);
}

fn slot(i: u32) -> Cell {
    return unpackCell(slotRaw(i));
}

fn setSlot(i: u32, c: Cell) {
    setSlotRaw(i, packCell(c));
}

fn slotType(i: u32) -> u32 {
    return cellType(slotRaw(i));
}

// The chance per contact that a flame lights what it touches, of qe at
// temp: as soon as its surface would reach igniteT. Under a flame's heat
// flux (FLAME_FLUX) a thermally thick solid's surface gets there in
// (π/4) kρc (igniteT - temp)² / flux² seconds (Drysdale ch. 6), kρc its
// thermal conductivity times its heat capacity: wood in about 3 s from cold,
// coal in 36, sawdust in 0.6, a damp green plant in 17; at FIRE_PACE
// (js/shared.js), in real time (a flame is hot, heat's clock real around
// it). A flammable gas is lit at once.
fn lightChance(qe: Elem, temp: f32) -> f32 {
    if (qe.kind == KIND_GAS) {
        return 1.0;
    }
    let rise = max(qe.igniteT - temp, 0.0);
    let seconds = 0.785398 * qe.thermal * qe.heatCap * 1e6 * rise * rise / (FLAME_FLUX * FLAME_FLUX);
    return 1.0 - exp(-FIRE_PACE * P.dt / PILOT_TRIES / max(seconds, 1e-6));
}

// Whether a flame passes between slots i and j: one of the block's other
// two holds gas a flame burns in. One burning surface lights another only
// through the flame in the gas beside them both, not through a solid
// (wood burning outside lights the next wood along, not the wood within).
fn flameBetween(i: u32, j: u32) -> bool {
    for (var k = 0u; k < 4u; k++) {
        if (k != i && k != j && isInWorld(k)) {
            let c = slot(k);
            if (holdsFlame(c.t, slotGas(k), c.temp)) {
                return true;
            }
        }
    }
    return false;
}

// q, of qe, lit by a flame or a spark: alight, if it burns (its life counts
// what of it burns from now; a plant's or seed's growth budget goes); an
// explosive, to go off in the update pass (common.wgsl's detonation).
fn light(q: Cell, qe: Elem) -> Cell {
    var c = q;
    if (qe.blast > 0.0) {
        c.flags |= FL_BURN;
    } else if (qe.burnTime > 0.0) {
        c.flags |= FL_BURN;
        if (holdsSugar(c.t) || holdsStore(c.t)) {
            c = withBurnt(c, 0u);
        }
    }
    return c;
}

fn hasMoved(i: u32) -> bool {
    return ((moved >> i) & 1u) != 0u;
}

fn swapSlots(i: u32, j: u32) {
    let a = slotRaw(i);
    setSlotRaw(i, slotRaw(j));
    setSlotRaw(j, a);
    moved |= (1u << i) | (1u << j);
    let pi = (perm >> (2u * i)) & 3u;
    let pj = (perm >> (2u * j)) & 3u;
    perm = (perm & ~((3u << (2u * i)) | (3u << (2u * j)))) | (pj << (2u * i)) | (pi << (2u * j));
}

// Two gases trade places as their kinds only (REWRITE_PLAN.md 4c: a gas's
// kind is a marker riding the air's flow): the kind, its life, aux and
// flags (x), velocity (z) and offset (w) move; the temperature (y) and what
// the cells hold of the air (the gas plane) stay, the air's carry moving
// those (carry.wgsl).
fn tradeKinds(i: u32, j: u32) {
    let a = slotRaw(i);
    let b = slotRaw(j);
    setSlotRaw(i, vec4<u32>(b.x, a.y, b.z, b.w));
    setSlotRaw(j, vec4<u32>(a.x, b.y, a.z, a.w));
    moved |= (1u << i) | (1u << j);
}

// The gas plane (common.wgsl) of slot i's cell, as the block started: what
// reactions read and write, before anything has moved.
fn slotIndex(i: u32) -> u32 {
    let p = origin + vec2<i32>(i32(i & 1u), i32(i >> 1u));
    return u32(p.y) * P.size.x + u32(p.x);
}

fn gasPairAt(k: u32) -> vec2<u32> {
    let v = cells[gasIndex(P.size.x * P.size.y, k)];
    return select(v.xy, v.zw, gasHigh(k));
}

fn setGasPairAt(k: u32, w: vec2<u32>) {
    let j = gasIndex(P.size.x * P.size.y, k);
    if (gasHigh(k)) {
        cells[j].z = w.x;
        cells[j].w = w.y;
    } else {
        cells[j].x = w.x;
        cells[j].y = w.y;
    }
}

fn slotGas(i: u32) -> Gas {
    let k = slotIndex(i);
    return gasFrom(cells[gasIndex(P.size.x * P.size.y, k)], k);
}

fn setSlotGas(i: u32, g: Gas) {
    setGasPairAt(slotIndex(i), packGas(g));
}

// ---------------------------------------------------------------- reactions

// Slot i's cell having turned from `was` into `now` in a reaction: a gas
// turned into something else (air a plant grows into, or a clone fills)
// puts what it held into the air beside it (airsource.wgsl), and holds
// none. Gas turned into gas (a flame spreading into air, a fireball into
// smoke) keeps what it held.
fn handOn(i: u32, was: Cell, now: Cell) {
    if (!holdsGas(was.t) || holdsGas(now.t)) {
        return;
    }
    pushGas(slotIndex(i), slotGas(i), was.temp);
    setSlotGas(i, noGas());
}

// p acts on q. Both are written back, so either side may change.
fn react(pi: u32, qi: u32) {
    var p = slot(pi);
    if (p.t == AIR || !isInWorld(qi)) {
        return;
    }
    var q = slot(qi);
    let pWas = p;
    let qWas = q;
    let pe = elems[p.t];
    let qe = elems[q.t];
    let dx = i32(qi & 1u) - i32(pi & 1u);
    let dy = i32(qi >> 1u) - i32(pi >> 1u);

    // Something burning shows flames in the air beside it where that is
    // burning hot (its burning heats it: pairs.wgsl), and its flames light
    // what can burn beside it. A flame is the air it burns in (its gas
    // plane) and as hot; it does nothing but show it, and light things.
    // Water against it puts it out (update.wgsl), the heat going into the
    // water as any heat does.
    if ((p.flags & FL_BURN) != 0u) {
        if (q.t == AIR) {
            let air = slotGas(qi);
            if (q.temp >= FLAME_T && holdsFlame(q.t, air, q.temp)
                && rnd() < pe.burnFire * oxygenOf(q.t, air)) {
                q = makeCell(FIRE, q.temp);
            }
        } else if (flammable(qe) && (q.flags & FL_BURN) == 0u && flameBetween(pi, qi)
            && rnd() < lightChance(qe, tempOf(q))) {
            q = light(q, qe);
        }
    }

    // Active charge jumps to idle conductors and lights what a spark lights
    // (sparkLights: gas, explosives).
    if (holdsCharge(p.t) && chargeOf(p) >= CHARGE_ACTIVE) {
        if (holdsCharge(q.t)) {
            if (chargeOf(q) == 0u) {
                q = withCharge(q, CHARGE_NEW);
            }
        } else if (sparkLights(qe) && (q.flags & FL_BURN) == 0u) {
            q = light(q, qe);
        }
    }

    // The cells it owes (formed in common.wgsl: ice or snow from water, lava
    // from stone, gravel from lava setting while it flies...): grown into air
    // beside it, or handed on up through the same element above it (and now
    // and then sideways, evening out), so they gather at the top: under the
    // water still freezing over ice, which has room to rise, or at the
    // surface, which heaves up. Ice and snow take the air's moisture into
    // what they grow (frozenExtra); the rest of the air they grow into, and
    // all of it for the rest, goes into the air beside it (handOn). Ice with
    // no room anywhere (FL_STRESSED, update.wgsl)
    // presses on what holds it in, which can crack, the ice squeezing into
    // the crack: a full bottle left in the cold bursts, and water frozen in a
    // sealed crack wedges it wider, to let in more water for the next frost.
    let owed = owedOf(p);
    if (owed > 0u) {
        let stressed = p.t == ICE && (p.flags & FL_STRESSED) != 0u;
        if (q.t == AIR) {
            q = makeCell(p.t, p.temp);
            if (holdsFrozenExtra(p.t)) {
                // What it can't keep (rounded at random, and past 255)
                // stays in the air, handed on.
                var air = slotGas(qi);
                let held = air.vapour + air.mist;
                q = withFrozenExtra(q, frozenExtra(0u, held));
                let rest = select(0.0, max(1.0 - f32(frozenExtraOf(q)) / FROZEN_EXTRA / held, 0.0), held > 0.0);
                air.vapour *= rest;
                air.mist *= rest;
                setSlotGas(qi, air);
            }
            p = withOwed(p, owed - 1u);
        } else if (q.t == p.t && owedOf(q) < 255u && ((stepDown(dx, dy) < 0 && stepSide(dx, dy) == 0)
            || (stepDown(dx, dy) == 0 && owedOf(q) + 1u < owed && rnd() < 0.25))) {
            q = withOwed(q, owedOf(q) + 1u);
            p = withOwed(p, owed - 1u);
        } else if (stressed && q.t != ICE && isStructural(qe) && (qe.flags & F_INDESTRUCTIBLE) == 0u
            && rnd() < FROST_CRACK / max(qe.hardness, 1.0)) {
            // It cracks: crumbles into its fragments (glass into shards),
            // its grams and heat kept as update.wgsl's crumble keeps them,
            // and the ice grows into the room they leave. (The ice took the
            // cell, the glass's grams and heat gone with it.)
            var f = convert(q, qe.breakTo);
            if (qe.fill <= 0.0 || elems[f.t].fill <= 0.0) {
                f.temp = AMBIENT_T + qe.heatCap * (q.temp - AMBIENT_T) / elems[f.t].heatCap;
            }
            q = formed(f, q);
        }
    }

    switch p.t {
        case FIRE: {
            if (isWater(q.t) || q.t == ICE || q.t == SNOW) {
                if (rnd() < 0.3) {
                    // Against water, a flame goes out: it leaves air or
                    // smoke, the air it burned in, as when it dies out
                    // (update.wgsl), its heat going into the water as any
                    // gas's does.
                    p = makeCell(select(AIR, SMOKE, rnd() < 0.5), p.temp);
                }
            } else if (flammable(qe) && (q.flags & FL_BURN) == 0u && rnd() < lightChance(qe, tempOf(q))) {
                q = light(q, qe);
            }
        }
        case BLAST: {
            // An explosion's gas, burning hot: it lights what it touches as a
            // flame does, setting off other explosives, and shows as a
            // fireball through the air beside it as hot as FIREBALL_T (its
            // heat spread there by the air's flow). What it breaks, its
            // pressure does (air.wgsl's blast layer).
            if (flammable(qe) && (q.flags & FL_BURN) == 0u && rnd() < lightChance(qe, tempOf(q))) {
                q = light(q, qe);
            } else if (q.t == AIR && q.temp >= FIREBALL_T) {
                q = makeCell(BLAST, q.temp);
            }
        }
        case PLANT: {
            // Not while either is burning: a burning plant's life counts what
            // of it has burnt.
            if (((p.flags | q.flags) & FL_BURN) != 0u) {
                break;
            }
            // Up and across as its down has them (the same as dy and dx under
            // Earth's).
            let rising = stepDown(dx, dy) < 0;
            let across = stepSide(dx, dy);
            if (q.t == PLANT && sugarOf(p) > sugarOf(q) && rising) {
                // Sugar climbs the stalk toward the growing tip: half of what
                // it has more than the cell above it (so its roots keep
                // theirs; pairs.wgsl evens it out every way besides).
                let moved = (sugarOf(p) - sugarOf(q)) / 2u;
                q = withSugar(q, sugarOf(q) + moved);
                p = withSugar(p, sugarOf(p) - moved);
            } else if (q.t == AIR && sugarOf(p) >= GROW_SUGAR && plantWaterOf(p) >= PLANT_WILT * 3u / 2u && rising
                && (p.flags & FL_LOOSE) == 0u) {
                // The growing tip, with the sugar for a cell and turgid (half
                // as watered again as it wilts at), grows: up, toward the
                // brighter side if one is (a shoot bending to the light), and
                // now and then it branches. The new cell has half its water
                // (either may wilt until its roots catch up), and takes over the support
                // of the cell it grew from, so a broken-off sprig can't hold
                // itself up by growing. (A plant falling loose doesn't grow:
                // its frac.x is its body's.)
                let lean = leanOf(p);
                let toward = (across < 0 && lean == PLANT_LEFT) || (across > 0 && lean == PLANT_RIGHT);
                if ((across == 0 && rnd() < select(PLANT_RISE, PLANT_RISE_LEANING, lean != 0u))
                    || (toward && rnd() < PLANT_BEND) || (across != 0 && !toward && rnd() < PLANT_BRANCH)) {
                    q = withPlantWater(makeCell(PLANT, q.temp), plantWaterOf(p) / 2u);
                    p = withPlantWater(p, plantWaterOf(p) - plantWaterOf(q));
                    p = withSugar(p, sugarOf(p) - GROW_SUGAR);
                    q.frac.x = p.frac.x;
                }
            }
        }
        case CLONE: {
            // Copies what it first touches, and what water it touches carries
            // (its solute in the clone's life), saturated.
            if (copiesOf(p) == 0u) {
                if (q.t != AIR && q.t != BLAST && (qe.flags & F_INDESTRUCTIBLE) == 0u) {
                    p = withCopiesSolute(withCopies(p, q.t), select(0u, soluteOf(q), holdsSolute(q.t)));
                }
            } else if (q.t == AIR && rnd() < 0.12) {
                // As made new (a flame's or smoke's life), and water with
                // the solute it copied, saturated. Gas it makes is the air
                // it fills; anything else pushes that aside (handOn).
                q = makeCell(copiesOf(p), elems[copiesOf(p)].spawnT);
                if (copiesSoluteOf(p) != 0u) {
                    q = withStrength(withSolute(q, copiesSoluteOf(p)), 255u);
                }
            }
        }
        case VOID, BLACK_HOLE: {
            // What it takes leaves nothing (a vacuum the air fills); a gas,
            // only its kind, its air staying as it was.
            if (q.t != AIR && (qe.flags & F_INDESTRUCTIBLE) == 0u) {
                q = makeCell(AIR, select(AMBIENT_T, q.temp, holdsGas(q.t)));
            }
        }
        default: {}
    }

    handOn(pi, pWas, p);
    handOn(qi, qWas, q);
    setSlot(pi, p);
    setSlot(qi, q);
}

// Whether c can act on anything in react(). Most cells (sand, water,
// stone, wall...) never do, and blocks where no cell can skip reactions,
// which are most of this pass's work. Keep in step with react().
fn canAct(c: Cell) -> bool {
    if ((c.flags & FL_BURN) != 0u || (holdsCharge(c.t) && chargeOf(c) >= CHARGE_ACTIVE)) {
        return true;
    }
    // Owing cells to grow (formed in common.wgsl).
    if (owedOf(c) > 0u) {
        return true;
    }
    switch c.t {
        case FIRE, BLAST, CLONE, VOID, BLACK_HOLE: {
            return true;
        }
        case PLANT: {
            return sugarOf(c) > 0u;
        }
        default: {
            return false;
        }
    }
}

// The six pairs of slots in a block, a nibble each with the first slot in the
// low two bits: 0-1, 2-3, 0-2, 1-3, 0-3, 1-2.
const PAIRS: u32 = 0x9cd8e4u;

fn reactAll() {
    let start = rndU() % 6u;
    for (var n = 0u; n < 6u; n++) {
        let nib = (PAIRS >> (4u * ((start + n) % 6u))) & 0xfu;
        let pair = vec2<u32>(nib & 3u, nib >> 2u);
        let order = select(pair, pair.yx, rnd() < 0.5);
        if (((acting >> order.x) & 1u) != 0u) {
            react(order.x, order.y);
        }
        if (((acting >> order.y) & 1u) != 0u) {
            react(order.y, order.x);
        }
    }
}

// ----------------------------------------------------------------- movement
//
// Every powder and gas particle carries a velocity (cells per step) and a
// sub-cell offset. Each pass adds velocity / BLOCK_PASSES to the offset; once it
// reaches half a cell the particle tries to step that way. The block offsets
// alternate every pass, so a particle moving steadily stays in the slot that
// can move its way. Liquid moves in fluid.wgsl; here it only gets out of the
// way, trading places with powder pushing through it or bubbles rising.

// (Air moves only as a bubble rising through liquid, with the velocity the
// update pass gives it there: REWRITE_PLAN.md 7-9.)
fn isMover(c: Cell) -> bool {
    let k = elems[c.t].kind;
    return (c.t != AIR || any(c.vel != vec2<f32>(0.0))) && k != KIND_SOLID && k != KIND_LIQUID;
}

fn stepOf(f: f32) -> i32 {
    if (f >= 0.5) {
        return 1;
    }
    if (f <= -0.5) {
        return -1;
    }
    return 0;
}

// Slot reached from slot i by (dx, dy), or -1 if that is outside this block.
fn slotAt(i: u32, dx: i32, dy: i32) -> i32 {
    let x = i32(i & 1u) + dx;
    let y = i32(i >> 1u) + dy;
    if (x < 0 || x > 1 || y < 0 || y > 1) {
        return -1;
    }
    return y * 2 + x;
}

// Whether slot j (in the block) holds a solid.
fn solidSlot(j: i32) -> bool {
    return j >= 0 && isInWorld(u32(j)) && elems[slotType(u32(j))].kind == KIND_SOLID;
}

// 1: moved. 0: blocked. -1: can't try this pass (outside the block, or the
// target already moved), so the particle just waits.
fn tryMove(i: u32, dx: i32, dy: i32) -> i32 {
    let j = slotAt(i, dx, dy);
    if (j < 0 || hasMoved(u32(j))) {
        return -1;
    }
    let k = u32(j);
    // The world's edge is a wall, and so is a person's body.
    if (!isInWorld(k) || !canEnter(slotType(i), slotType(k), stepDown(dx, dy)) || isOccupant(occupants[slotIndex(k)])) {
        return 0;
    }
    // Nor through the corner where two solids touch (a turned body's walls
    // are steps that meet so: the sand in a boat's hold ran out under it).
    if (dx != 0 && dy != 0 && solidSlot(slotAt(i, dx, 0)) && solidSlot(slotAt(i, 0, dy))) {
        return 0;
    }
    if (holdsVapour(slotType(i)) && holdsVapour(slotType(k))) {
        tradeKinds(i, k);
    } else {
        swapSlots(i, k);
    }
    var c = slot(k);
    c.frac -= vec2<f32>(f32(dx), f32(dy));
    setSlot(k, c);
    return 1;
}

// All of it in the frame of the block's down (blockDown): a particle's down g
// is the block's, or its opposite for one that rises. Where down is
// diagonal, the way on past what blocks it is either axis step; a slide is
// down turned an eighth either way.

// Slips past whatever is in the way along g, diagonally to it. Powders only
// sometimes (spread), so piles keep a slope. Damp grains (moisture in life;
// update.wgsl) cling together, so a damp pile keeps a steep face; quite
// damp, they hardly ever let go. Soaked through under water they don't
// cling. (Only cohesive powders keep moisture in life: salt and snow keep
// other things there.)
fn slide(i: u32, g: vec2<i32>) -> bool {
    let m = slot(i);
    let e = elems[m.t];
    let a = turnStep(g, 1);
    let b = turnStep(g, -1);
    let aIn = slotAt(i, a.x, a.y) >= 0;
    let bIn = slotAt(i, b.x, b.y) >= 0;
    if (!aIn && !bIn) {
        return false;
    }
    let gu = normalize(vec2<f32>(g));
    let across = vec2<f32>(gu.y, -gu.x);
    let sideVel = dot(m.vel, across);
    // This block offers one way past, or (down diagonal) both: then the way
    // it already moves, or either.
    var s = select(b, a, aIn);
    if (aIn && bIn) {
        let aWay = dot(vec2<f32>(a), across) * sideVel > 0.0;
        s = select(b, a, select(rnd() < 0.5, aWay, abs(sideVel) > 0.2));
    }
    let sSide = dot(vec2<f32>(s), across);
    if (abs(sideVel) > 0.2 && sign(sideVel) != sign(sSide)) {
        return false;
    }
    let damp = select(0.0, shareOf(soilWaterOf(m)), holdsSoilWater(m.t) && (m.flags & FL_WET) == 0u);
    let chance = select(1.0, e.spread * pow(1.0 - damp, 4.0 * e.cohesion), e.kind == KIND_POWDER);
    if (e.kind == KIND_POWDER && !slips(i, e, s, g)) {
        return false;
    }
    if (rnd() >= chance || tryMove(i, s.x, s.y) != 1) {
        return false;
    }
    let k = u32(slotAt(i, s.x, s.y));
    var c = slot(k);
    c.frac = vec2<f32>(0.0);
    let cs = dot(c.vel, across);
    c.vel += (sign(sSide) * max(abs(cs), slideSpeed(i, e)) - cs) * across;
    setSlot(k, c);
    return true;
}

// Whether a grain of e in slot i, pressed along g, slips by step s: the
// pull's part along the step more than its friction (Elem.friction) times
// its part across it (Coulomb, REWRITE_PLAN.md 9-1). Down a 45° step under a
// straight pull every powder with a friction under 1 slips, as the lattice
// has it; along a flat floor under a pull 21° off down (a black hole beside
// it, its down rounded to the diagonal) sand doesn't: it did, 3 cells.
fn slips(i: u32, e: Elem, s: vec2<i32>, g: vec2<i32>) -> bool {
    let p = origin + vec2<i32>(i32(i & 1u), i32(i >> 1u));
    var pull = pullAt(vec2<f32>(p) + 0.5);
    if (dot(pull, vec2<f32>(g)) < 0.0) {
        pull = -pull;
    }
    let step = normalize(vec2<f32>(s));
    let along = dot(pull, step);
    return along > e.friction * length(pull - along * step);
}

// The speed along the ground a grain of e in slot i slides off with, down a
// cell's 45° step: what its weight gains it down the step, less its friction
// (Elem.friction), so speed² = 2 g (sin 45° - friction cos 45°) √2, of which
// across is its share cos 45°: √(g (1 - friction)). Sand 0.30 cells a step at
// g, sawdust (friction 1) none. (Until 9-1, every grain slid off at 0.3.)
fn slideSpeed(i: u32, e: Elem) -> f32 {
    let p = origin + vec2<i32>(i32(i & 1u), i32(i >> 1u));
    let g = length(pullAt(vec2<f32>(p) + 0.5)) * max(e.weight, 0.0) / elems[WATER].weight;
    return sqrt(max(g * (1.0 - e.friction), 0.0));
}

// The particle in slot j is in the way of slot i's move along down (gu). If
// it is itself moving the same way, this isn't a hit: when it's at least as
// fast, slot i just waits for it to clear; when slot i is faster and has
// caught up, the two share their momentum (an inelastic collision, weighted
// by density), so neither stops dead. Returns false for an ordinary hit, on
// something resting or coming the other way.
fn catchUp(i: u32, j: i32, gu: vec2<f32>) -> bool {
    if (j < 0) {
        return false;
    }
    let k = u32(j);
    var a = slot(i);
    var b = slot(k);
    let va = dot(a.vel, gu);
    let vb = dot(b.vel, gu);
    if (!isMover(b) || va * vb <= 0.0 || abs(vb) < SETTLED_SPEED) {
        return false;
    }
    if (abs(vb) < abs(va)) {
        let ma = elems[a.t].rho;
        let mb = elems[b.t].rho;
        let v = (ma * va + mb * vb) / (ma + mb);
        a.vel += (v - va) * gu;
        b.vel += (v - vb) * gu;
        setSlot(i, a);
        setSlot(k, b);
    }
    return true;
}

// A blocked step s: along down (or against it, rising) a landing, across it
// a bounce.
fn collide(i: u32, s: vec2<i32>, gu: vec2<f32>) {
    var m = slot(i);
    let e = elems[m.t];
    let across = vec2<f32>(gu.y, -gu.x);
    if (stepDown(s.x, s.y) != 0) {
        let impact = abs(dot(m.vel, gu));
        if ((e.flags & F_IMPACT) != 0u && impact > IMPACT_SPEED) {
            m = light(m, e);
        }
        if (e.kind == KIND_POWDER) {
            m.vel += across * (rnd() - 0.5) * impact * SCATTER;
        }
        m.vel -= dot(m.vel, gu) * gu;
        m.frac -= dot(m.frac, gu) * gu;
    } else {
        let vs = dot(m.vel, across);
        m.vel -= (1.0 + BOUNCE) * vs * across;
        m.frac -= dot(m.frac, across) * across;
    }
    setSlot(i, m);
}

fn moveSlot(i: u32) {
    let m = slot(i);
    if (hasMoved(i) || !isMover(m)) {
        return;
    }
    let weight = elems[m.t].weight;
    let s = vec2<i32>(stepOf(m.frac.x), stepOf(m.frac.y));
    let g = select(blockDown, -blockDown, weight < 0.0);  // "down" for this particle
    let gu = normalize(vec2<f32>(blockDown));

    if (all(s == vec2<i32>(0))) {
        // At rest: look for a way past what it rests on, so piles slump;
        // weightless, it stays where it is.
        if (weight != 0.0 && !weightless) {
            let below = slotAt(i, g.x, g.y);
            if (below >= 0 && !canEnter(m.t, slotType(u32(below)), select(1, -1, weight < 0.0))) {
                _ = slide(i, g);
            }
        }
        return;
    }
    if (tryMove(i, s.x, s.y) != 0) {
        return;
    }
    if (s.x != 0 && s.y != 0) {
        // Its two axis steps, the one more along its down first.
        let v = vec2<i32>(0, s.y);
        let h = vec2<i32>(s.x, 0);
        let vFirst = abs(dot(v, g)) >= abs(dot(h, g));
        let first = select(h, v, vFirst);
        let second = select(v, h, vFirst);
        let r1 = tryMove(i, first.x, first.y);
        if (r1 == 1) {
            return;
        }
        let r2 = tryMove(i, second.x, second.y);
        if (r2 == 1) {
            return;
        }
        let caught = r1 == 0 && catchUp(i, slotAt(i, first.x, first.y), gu);
        if (r1 == 0 && !caught) {
            collide(i, first, gu);
        }
        if (r2 == 0) {
            collide(i, second, gu);
        }
        return;
    }
    if (stepDown(s.x, s.y) != 0) {
        // Behind something moving the same way: wait, don't slip around it.
        if (catchUp(i, slotAt(i, s.x, s.y), gu)) {
            return;
        }
        if (dot(s, g) > 0 && !weightless && slide(i, g)) {
            return;
        }
    }
    collide(i, s, gu);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<i32>(P.size);
    origin = vec2<i32>(gid.xy) * 2 - vec2<i32>(P.offset);
    // Blocks hanging over the world's edge run too; their outside slots act
    // as walls, so the edge columns move as freely as any other.
    var types = 0u;
    inWorld = 0u;
    for (var s = 0u; s < 4u; s++) {
        let p = origin + vec2<i32>(i32(s & 1u), i32(s >> 1u));
        var v = vec4<u32>(AIR, 0u, 0u, 0u);
        if (all(p >= vec2<i32>(0)) && all(p < size)) {
            inWorld |= 1u << s;
            v = cells[u32(p.y * size.x + p.x)];
            types |= v.x;
        }
        setSlotRaw(s, v);
    }
    // Plain air has nothing to do here; the air grid moves it.
    if ((types & 0xffu) == AIR) {
        return;
    }
    let at = clamp(origin, vec2<i32>(0), size - 1);
    blockDown = downAt(at);
    weightless = weightlessAt(at);

    var movers = 0u;
    acting = 0u;
    for (var s = 0u; s < 4u; s++) {
        let c = slot(s);
        if (isMover(c)) {
            movers |= 1u << s;
        }
    }
    for (var s = 0u; s < 4u; s++) {
        let c = slot(s);
        if (c.t != AIR && canAct(c)) {
            acting |= 1u << s;
        }
    }
    // Nothing here moves and nothing reacts: the block stays as it is.
    if (movers == 0u && acting == 0u) {
        return;
    }

    seedRng(gid.y * 65536u + gid.x, P.frame, P.seed);
    moved = 0u;
    perm = UNMOVED;
    if (acting != 0u) {
        reactAll();
    }

    for (var s = 0u; s < 4u; s++) {
        var c = slot(s);
        if (isMover(c)) {
            c.frac = clamp(c.frac + c.vel / BLOCK_PASSES, vec2<f32>(-1.5), vec2<f32>(1.5));
            setSlot(s, c);
        }
    }
    let start = rndU() & 3u;
    for (var k = 0u; k < 4u; k++) {
        moveSlot((start + k) & 3u);
    }

    for (var s = 0u; s < 4u; s++) {
        if (isInWorld(s)) {
            let p = origin + vec2<i32>(i32(s & 1u), i32(s >> 1u));
            cells[u32(p.y * size.x + p.x)] = slotRaw(s);
        }
    }
    // What the cells that moved hold of the air goes with them (the gas
    // plane; the reactions above wrote theirs where they started). Only
    // slots in the world ever trade places.
    if (perm != UNMOVED) {
        var pairs: array<vec2<u32>, 4>;
        for (var s = 0u; s < 4u; s++) {
            if (isInWorld(s)) {
                pairs[s] = gasPairAt(slotIndex(s));
            }
        }
        for (var s = 0u; s < 4u; s++) {
            let came = (perm >> (2u * s)) & 3u;
            if (came != s && isInWorld(s)) {
                setGasPairAt(slotIndex(s), pairs[came]);
            }
        }
    }
}
