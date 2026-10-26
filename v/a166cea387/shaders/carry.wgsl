// The carry (REWRITE_PLAN.md stages 4b, 4c): what the gas holds, moved with
// the flow, none of it made or lost on the way: the dry gas's amount, vapour,
// mist, the oxygen used up of its dry gas, and their heat.
//
// Flux form at cell resolution, on the projected flow (air.wgsl): through
// each face between two cells that carry the gas, a share of a cell's volume
// a substep, the flow across the face times the substep, goes from the cell
// upwind into the other (the donor cell: Godunov's, first order), with what
// it holds. What one cell gives the other takes, so all of it is kept. Heat
// goes as energy, and a cell's temperature is its heat over its heat
// capacity, which is what it holds (common.wgsl gasHeatCap): so a cell's new
// temperature is a mix of its own and what flowed in, never past any of
// them, whatever divergence the flow has. The flow gives out what a cell's
// gas expands by (air.wgsl's expansion), and takes in what one squeezed or
// empty takes: so a boiled cell's vapour spreads, a heated cell's air thins
// and a vacuum fills. No cell gives more than it holds while its outflow is
// at most its volume a substep: the substeps are as many as the most any
// cell gives out (air.wgsl's project) takes to come within CARRY_REACH.
//
// What carries the gas: any gas, and what gas passes through (a fan: air with
// blades in it). Liquids, powders and solids are walls; a black hole
// swallows what flows into it. The world's sides and bottom are the open
// air: what flows out is gone, and what flows in is the outside air's at that
// height (fresh, as humid as the outdoor air is, at its pressure there); its
// top is the top of the air, shut.
//
// Gas pushed aside by a pass that put something else in its place
// (airsource.wgsl's pushGas) is taken in here, at load: by its cell if that
// holds gas, else by the cell beside it in the biggest zone of gas (takerOf);
// and in liquid with no gas beside it, it rises through the liquid to the
// gas over it (a bubble's air the water closed over, RISE_REACH cells at
// most). Where something solid stops its way up through the liquid (a
// floating body's floor), it goes along under it first to where it can
// rise (rise, before load). (Air pushed into rock with no gas beside it
// waits there until it has some.)
//
// Gas the liquid closes in is at the liquid's pressure (REWRITE_PLAN.md
// `bubble-pressure`): a sealed pocket over its wet room's pressure (the
// rooms joined through the liquid, common.wgsl's wetWord) gives what is past
// it, at load, to the wet room, which its takers take in at settle (wetOf).
//
// Air rising (against the pull, by its height, pull.wgsl) expands and cools,
// and sinking warms, by DRY_ADIABATIC: faster
// than the open air cools with height (LAPSE_RATE), so a rising warm pocket
// soon isn't warm any more, and a landscape's air doesn't churn itself to one
// temperature top to bottom. The mist of dew stays on its surface, out of
// the wind.
//
// Passes, straight after the air's (on the very cells its flow was found
// for): load (the cells' gas into scratch, unrounded, with what was pushed
// into it), then each substep flow (from one scratch set into the other),
// then settle (back into the cells, rounded into their f16s once a step).

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> elems: array<Elem>;
@group(0) @binding(2) var<storage, read_write> cells: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> air: array<vec4<f32>>;
// The most any cell gives out this step (air.wgsl's project, f32 bits).
@group(0) @binding(4) var<storage, read> fastest: array<u32>;
// Per cell two sets of two vec4s (Held): the gas as one substep leaves it,
// and as the next makes it.
@group(0) @binding(5) var<storage, read_write> scratch: array<vec4<u32>>;
// Which of the step's substeps this is.
struct Substep {
    index: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}
@group(0) @binding(6) var<uniform> substep: Substep;
// What other passes put into the air (airsource.wgsl): the gas pushed aside.
@group(0) @binding(7) var<storage, read_write> source: array<atomic<u32>>;
// The projection's potential (air.wgsl), per cell's x: on the world's left
// edge, what flows out through it.
@group(0) @binding(8) var<storage, read> potential: array<vec4<f32>>;
// The liquid's grid (fluid.wgsl's field), for the gaps it counts as liquid
// (common.wgsl's flowsThrough).
@group(0) @binding(9) var<storage, read> liquid: array<vec4<f32>>;
// The pull of the world's masses per light cell of 2 x 2 cells
// (gravity.wgsl), for down and height (pull.wgsl): which way gas in liquid
// rises, how much air rising cools, and the outside air at the edges.
@group(0) @binding(10) var gravity: texture_2d<f32>;

// The rooms and the air's zones' sums (sky.wgsl, air.wgsl): where gas pushed
// aside goes (takerOf).
@group(0) @binding(11) var<storage, read_write> sky: array<atomic<u32>>;

// How far up through liquid gas pushed into it rises in a step to the gas
// over it (load).
const RISE_REACH: i32 = 64;
// flow from set T into S (true), or from S into T.
override FROM_T: bool = true;

// The most of its volume a cell gives out in a substep. (CARRY_STEPS, the
// most substeps a step, comes from js/shared.js.)
const CARRY_REACH: f32 = 1.0;

// This step's substeps (sim.js encodes CARRY_STEPS; the rest do nothing; an
// even number, so the gas ends in S).
fn substeps() -> u32 {
    let k = clamp(u32(ceil(bitcast<f32>(fastest[0]) / CARRY_REACH)), 1u, CARRY_STEPS);
    return k + (k & 1u);
}

fn inWorld(p: vec2<i32>) -> bool {
    return all(p >= vec2<i32>(0)) && all(p < vec2<i32>(P.size));
}

fn cellIndex(p: vec2<i32>) -> u32 {
    return u32(p.y) * P.size.x + u32(p.x);
}

// What a cell holds of the gas, unrounded between the passes (rounded every
// substep, the f16s' rounding would add up): its heat (J, its heat capacity
// times its temperature in °C), its dry gas's amount, its vapour and mist,
// the oxygen used up of its dry gas (the share times the amount); its
// temperature as it was (kept by a cell left holding nothing); whether it
// carries the gas, swallows it (a black hole), and is dew.
struct Held {
    heat: f32,
    amount: f32,
    vapour: f32,
    mist: f32,
    lost: f32,
    temp: f32,
    carries: bool,
    swallows: bool,
    dew: bool,
}

const HELD_S: u32 = 0u;
const HELD_T: u32 = 2u;

fn readHeld(which: u32, i: u32) -> Held {
    let a = bitcast<vec4<f32>>(scratch[4u * i + which]);
    let b = scratch[4u * i + which + 1u];
    let flags = b.z;
    return Held(a.x, a.y, a.z, a.w, bitcast<f32>(b.x), bitcast<f32>(b.y), (flags & 1u) != 0u, (flags & 2u) != 0u,
        (flags & 4u) != 0u);
}

fn writeHeld(which: u32, i: u32, h: Held) {
    scratch[4u * i + which] = bitcast<vec4<u32>>(vec4<f32>(h.heat, h.amount, h.vapour, h.mist));
    let flags = select(0u, 1u, h.carries) | select(0u, 2u, h.swallows) | select(0u, 4u, h.dew);
    scratch[4u * i + which + 1u] = vec4<u32>(bitcast<u32>(h.lost), bitcast<u32>(h.temp), flags, 0u);
}

// What a cell holding h holds, as the gas plane has it.
fn heldGas(h: Held) -> Gas {
    return Gas(h.vapour, h.mist, select(0.0, h.lost / h.amount, h.amount > 0.0), h.amount);
}

// The outside air beyond the world's sides and bottom, in cell q (outside
// the world), at its height (pull.wgsl).
fn outsideHeld(q: vec2<i32>) -> Held {
    let altitude = altitudeAt(vec2<f32>(q) + 0.5);
    let temp = outdoorT(altitude) + dayWarmth(P.day);
    let g = outdoorGas(altitude, temp);
    return Held(gasHeatCap(g) * temp, g.amount, g.vapour, 0.0, 0.0, temp, true, false, false);
}

// Whether the cell at p carries the gas: what the flow passes through
// (common.wgsl's flowsThrough, air.wgsl's carriesAt) but a black hole, which
// swallows it.
fn carriesAt(p: vec2<i32>) -> bool {
    if (!inWorld(p)) {
        return false;
    }
    let i = cellIndex(p);
    let t = cellType(cells[i]);
    return flowsThrough(t, liquidSide(liquid[2u * i], liquid[2u * i + 1u])) && t != BLACK_HOLE;
}

// The four cells beside one, in the order gas pushed out of it looks for one
// to take it in: up, left, right, down.
fn beside(p: vec2<i32>, k: i32) -> vec2<i32> {
    return p + select(select(vec2<i32>(1, 0), vec2<i32>(0, 1), k == 3), select(vec2<i32>(0, -1), vec2<i32>(-1, 0), k == 1), k < 2);
}

// The cells of the pressure zone the gas at p is in (air.wgsl's, this step's
// sums in the sky): none for a pocket with no zone (sky.wgsl's UNNUMBERED).
fn zoneCells(p: vec2<i32>) -> u32 {
    let a = vec2<u32>(p);
    let piece = roomByteAt(a);
    if (piece == NO_ROOM || piece == UNNUMBERED) {
        return 0u;
    }
    return atomicLoad(&sky[zoneSlotAt(P.size, P.frame, roomOfPiece(roomPieceAt(P.size, a, piece))) + ZONE_COUNT]);
}

// Which of the cells beside q (beside's k) takes in the gas pushed out of
// it: of those that carry gas, the one in the biggest zone, the first of
// them in beside's order; -1 for none. Pushed gas goes where there is most
// room for it: the open air over a pocket a body or the water has closed
// off beside it, which a cell's worth would squeeze to twice its pressure,
// and it would break what holds it in.
fn takerOf(q: vec2<i32>) -> i32 {
    var best = -1;
    var most = 0u;
    for (var m = 0; m < 4; m++) {
        let r = beside(q, m);
        if (carriesAt(r)) {
            let n = zoneCells(r) + 1u;
            if (n > most) {
                most = n;
                best = m;
            }
        }
    }
    return best;
}

// What the gas at p does with the pressure the liquid holds it at
// (REWRITE_PLAN.md `bubble-pressure`), from this step's zones' and wet rooms'
// sums (air.wgsl), alike at load and at settle: its wet room's slot (NO_WET
// for none); the share of all it holds it gives out (give); and how much of
// what is given out it takes in (take, over all its wet room's takers'). A
// pocket (any zone but its wet room's surface, rooms.wgsl's isSurface) over
// the surface's pressure (the open air's at its height, or a sealed tank's
// air over its water) gives what is past it, which bubbles out round what
// holds it in; the surface's cells beside the liquid take it in, as much
// each, where the bubbles come up. A pocket under it stays so: the liquid
// would close in on it (P2). (volume: its volume at 1 atm, as air.wgsl's
// gather has it: what it weighs as a giver.)
struct Wet {
    slot: u32,
    give: f32,
    take: f32,
    volume: f32,
}

fn wetOf(p: vec2<i32>) -> Wet {
    let a = vec2<u32>(p);
    let piece = roomByteAt(a);
    let room = wetRoomAt(a);
    if (piece >= UNNUMBERED || room == NO_WET) {
        return Wet(NO_WET, 0.0, 0.0, 0.0);
    }
    let zoneRoom = roomOfPiece(roomPieceAt(P.size, a, piece));
    let zone = zoneSlotAt(P.size, P.frame, zoneRoom);
    let slot = wetSlotAt(P.size, P.frame, room);
    let volume = (AMBIENT_T + KELVIN) / max(wordTempOf(cells[cellIndex(p)]) + KELVIN, 1.0);
    var w = Wet(slot, 0.0, 0.0, volume);
    if (isSurface(slot, zoneRoom, zone)) {
        w.take = select(0.0, 1.0, besideJoin(p));
        return w;
    }
    let surface = surfaceAtm(slot, vec2<f32>(p) + 0.5);
    let zoneVolume = f32(atomicLoad(&sky[zone + ZONE_VOLUME]));
    if (surface.y <= 0.0 || zoneVolume <= 0.0) {
        return w;
    }
    let own = f32(atomicLoad(&sky[zone + ZONE_AMOUNT])) / zoneVolume;
    if (own > surface.x) {
        w.give = 1.0 - surface.x / own;
    }
    return w;
}

// x added to the exact sum in the sky's words at, at + 1 (common.wgsl's
// exactParts, EXACT_FINE; as airsource.wgsl's addSource).
fn addSky(at: u32, x: f32) {
    if (x == 0.0) {
        return;
    }
    let w = exactParts(x, EXACT_FINE);
    let old = atomicAdd(&sky[at], w.x);
    atomicAdd(&sky[at + 1u], w.y + exactCarry(old, w.x));
}

fn skyExact(at: u32) -> f32 {
    return exactValue(vec2<u32>(atomicLoad(&sky[at]), atomicLoad(&sky[at + 1u])), EXACT_FINE);
}

// Gas pushed into cell j's slot (airsource.wgsl) added to h, and none left.
fn takeIn(h: ptr<function, Held>, j: u32) {
    (*h).amount += takeSource(j, SOURCE_AMOUNT);
    (*h).vapour += takeSource(j, SOURCE_VAPOUR);
    (*h).mist += takeSource(j, SOURCE_MIST);
    (*h).lost += takeSource(j, SOURCE_LOST);
    (*h).heat += takeSource(j, SOURCE_HEAT);
}

// Gas waiting in liquid (airsource.wgsl's pushGas) whose way up through it
// something solid stops, a floating body's floor or a roof over a pool:
// along under it, the nearer way first, to where it can rise on, and on up,
// to the liquid's cell under the gas there, for load to take in (takerOf).
// Left where it was, it waited for good: the air of a pool painted under a
// floating box waited in the water under it, 19.8 cells' worth a row, 790
// in all, for as long as the box floated there, and a hold that was air
// took it in as the box's floor came down into it (REWRITE_PLAN.md P3-21).
// Where it can't go on within SLIDE_REACH (a diving bell's air), it waits.
const SLIDE_REACH: i32 = 64;
const SLIDE_TURNS: i32 = 8;

// The liquid's cell under gas that gas waiting at q goes to, along under
// what stops it; (-1, -1) for where it is (it rises straight up to the gas,
// as load takes it, or can't go on).
fn slideTo(q: vec2<i32>) -> vec2<i32> {
    var p = q;
    var slid = false;
    for (var turn = 0; turn < SLIDE_TURNS; turn++) {
        let up = -downAt(p);
        for (var k = 0; k < RISE_REACH && liquidAt(p + up); k++) {
            p += up;
        }
        if (carriesAt(p + up)) {
            return select(vec2<i32>(-1), p, slid);
        }
        // Stopped: along under what stops it, through the liquid, to the
        // nearest cell either way with liquid or gas over it.
        let side = vec2<i32>(-up.y, up.x);
        var found = vec2<i32>(-1);
        var open = vec2<bool>(true, true);
        for (var s = 1; s <= SLIDE_REACH && found.x < 0 && any(open); s++) {
            for (var w = 0; w < 2; w++) {
                if (!open[w] || found.x >= 0) {
                    continue;
                }
                let r = p + side * select(s, -s, w == 0);
                if (!liquidAt(r)) {
                    open[w] = false;
                } else if (liquidAt(r + up) || carriesAt(r + up)) {
                    found = r;
                }
            }
        }
        if (found.x < 0) {
            return vec2<i32>(-1);
        }
        p = found;
        slid = true;
    }
    return vec2<i32>(-1);
}

// Whether cell i holds gas waiting in its slot (airsource.wgsl).
fn waitsIn(i: u32) -> bool {
    for (var k = SOURCE_AMOUNT; k <= SOURCE_HEAT; k++) {
        let at = SOURCE_WORDS * i + 2u * k;
        if (atomicLoad(&source[at]) != 0u || atomicLoad(&source[at + 1u]) != 0u) {
            return true;
        }
    }
    return false;
}

// Gas waiting in liquid cell q with no gas beside it, moved along under what
// stops its way up (slideTo), its words whole (exact sums: nothing rounds).
// Each moves its own into a cell with gas over it, which moves none: so what
// a cell holds after doesn't hang on which ran first.
@compute @workgroup_size(8, 8)
fn rise(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid.xy >= P.size)) {
        return;
    }
    let q = vec2<i32>(gid.xy);
    let i = gid.y * P.size.x + gid.x;
    if (!liquidAt(q) || !waitsIn(i)) {
        return;
    }
    for (var m = 0; m < 4; m++) {
        if (carriesAt(beside(q, m))) {
            return;
        }
    }
    let to = slideTo(q);
    if (to.x < 0) {
        return;
    }
    let j = cellIndex(to);
    for (var k = SOURCE_AMOUNT; k <= SOURCE_HEAT; k++) {
        let off = SOURCE_WORDS * i + 2u * k;
        let lo = atomicExchange(&source[off], 0u);
        let hi = atomicExchange(&source[off + 1u], 0u);
        let at = SOURCE_WORDS * j + 2u * k;
        let old = atomicAdd(&source[at], lo);
        atomicAdd(&source[at + 1u], hi + exactCarry(old, lo));
    }
}

// The cells' gas into S, with what was pushed into it.
@compute @workgroup_size(8, 8)
fn load(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid.xy >= P.size)) {
        return;
    }
    let p = vec2<i32>(gid.xy);
    let i = gid.y * P.size.x + gid.x;
    let v = cells[i];
    let t = cellType(v);
    var h = Held(0.0, 0.0, 0.0, 0.0, 0.0, cellTemp(v), false, t == BLACK_HOLE, cellHas(v, FL_DEW));
    if (h.swallows) {
        // Into a black hole: gone.
        var gone = h;
        takeIn(&gone, i);
    } else if (carriesAt(p)) {
        let g = gasFrom(cells[gasIndex(P.size.x * P.size.y, i)], i);
        h = Held(gasHeatCap(g) * h.temp, g.amount, g.vapour, g.mist, g.lost * g.amount, h.temp, true, false, h.dew);
        takeIn(&h, i);
        // And what was pushed out of the cells beside it that don't carry
        // gas, where it is the one beside them that takes it (takerOf).
        for (var k = 0; k < 4; k++) {
            let q = beside(p, k);
            if (!inWorld(q) || carriesAt(q) || cellType(cells[cellIndex(q)]) == BLACK_HOLE) {
                continue;
            }
            let m = takerOf(q);
            if (m >= 0 && all(beside(q, m) == p)) {
                takeIn(&h, cellIndex(q));
            }
        }
        // And what rises to it through the liquid under it: what was pushed
        // into liquid with no gas beside it, as far down as the liquid goes.
        let dn = downAt(p);
        var q = p + dn;
        for (var k = 0; k < RISE_REACH && liquidAt(q); k++) {
            var gasBeside = false;
            for (var m = 0; m < 4; m++) {
                gasBeside = gasBeside || carriesAt(beside(q, m));
            }
            if (!gasBeside) {
                takeIn(&h, cellIndex(q));
            }
            q += dn;
        }
        // What is past its wet room's pressure, given out (wetOf): all it
        // holds by as much, but its dew's mist, which stays (and the mist's
        // heat, as flow has it).
        let wet = wetOf(p);
        if (wet.give > 0.0) {
            let f = wet.give;
            let heat = f * h.heat * heatGoing(h);
            let mist = select(f * h.mist, 0.0, h.dew);
            addSky(wet.slot + WET_GIVEN, f * h.amount);
            addSky(wet.slot + WET_GIVEN + 2u, f * h.vapour);
            addSky(wet.slot + WET_GIVEN + 4u, mist);
            addSky(wet.slot + WET_GIVEN + 6u, f * h.lost);
            addSky(wet.slot + WET_GIVEN + 8u, heat);
            addSky(wet.slot + WET_GIVERS, f * wet.volume);
            h.heat -= heat;
            h.amount -= f * h.amount;
            h.vapour -= f * h.vapour;
            h.mist -= mist;
            h.lost -= f * h.lost;
        }
        if (wet.take > 0.0) {
            atomicAdd(&sky[wet.slot + WET_TAKERS], 1u);
        }
    }
    writeHeld(HELD_S, i, h);
}


// The flow across cell d's face toward d + side (a unit step), out of d
// positive: the air's right and bottom faces' (air.wgsl); on the world's
// left edge, where no face's is kept, the potential there (what flows out
// through it, the outside's at none); on its top, none.
fn windOut(d: vec2<i32>, side: vec2<i32>) -> f32 {
    let w = P.size.x;
    let i = cellIndex(d);
    if (side.x > 0) {
        return air[i].y;
    }
    if (side.y > 0) {
        return air[i].z;
    }
    if (side.y < 0) {
        return select(0.0, -air[i - w].z, d.y > 0);
    }
    if (d.x > 0) {
        return -air[i - 1u].y;
    }
    return potential[i].x;
}

// The flow out of cell d through its face toward d + side in a substep of h
// of a step, as a share of what d holds: what the projection asks, unless
// all d's faces together ask for more than it holds (a flow the projection
// left short of its tolerance, a cell giving out more than the substeps
// take, CARRY_STEPS): then each scaled alike so that they give out all of
// it, and none more. Both cells across a face work it out alike, so what
// one gives the other takes. (Given out past what it held, a cell was left
// holding less than nothing, and its temperature went with it.)
fn shareOut(d: vec2<i32>, side: vec2<i32>, h: f32) -> f32 {
    var total = 0.0;
    for (var k = 0; k < 4; k++) {
        total += max(windOut(d, sideOf(k)) * h, 0.0);
    }
    return windOut(d, side) * h / max(total, 1.0);
}

// Left, right, up, down.
fn sideOf(k: i32) -> vec2<i32> {
    return select(vec2<i32>(0, (k & 1) * 2 - 1), vec2<i32>((k & 1) * 2 - 1, 0), k < 2);
}

// The share of h's heat that goes with its gas: all of it, but for dew, whose
// mist stays (flow).
fn heatGoing(h: Held) -> f32 {
    if (!h.dew) {
        return 1.0;
    }
    let cap = gasHeatCap(heldGas(h));
    return select(1.0, 1.0 - elems[WATER].heatCap * h.mist / WATER_VAPOUR / cap, cap > 0.0);
}

@compute @workgroup_size(8, 8)
fn flow(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid.xy >= P.size) || substep.index >= substeps()) {
        return;
    }
    let inSet = select(HELD_S, HELD_T, FROM_T);
    let outSet = select(HELD_T, HELD_S, FROM_T);
    let d = vec2<i32>(gid.xy);
    let i = cellIndex(d);
    let own = readHeld(inSet, i);
    if (!own.carries) {
        writeHeld(outSet, i, own);
        return;
    }
    let h = 1.0 / f32(substeps());
    var next = own;
    // How far down (metres, by the pull's height, pull.wgsl) the gas moving
    // through its faces goes, half a cell's way each side of them.
    let height = altitudeAt(vec2<f32>(d) + 0.5);
    var sinking = 0.0;
    for (var k = 0; k < 4; k++) {
        let side = sideOf(k);
        let wind = windOut(d, side) * h;
        sinking += 0.5 * wind * (height - altitudeAt(vec2<f32>(d + side) + 0.5));
        if (wind > 0.0) {
            // Its own gas goes out, but the mist of its dew (and its heat).
            let out = shareOut(d, side, h);
            next.heat -= out * own.heat * heatGoing(own);
            next.amount -= out * own.amount;
            next.vapour -= out * own.vapour;
            next.mist -= select(out * own.mist, 0.0, own.dew);
            next.lost -= out * own.lost;
        } else if (wind < 0.0) {
            let q = d + side;
            var n: Held;
            var into = -wind;
            if (inWorld(q)) {
                n = readHeld(inSet, cellIndex(q));
                into = shareOut(q, -side, h);
            } else {
                n = outsideHeld(q);
            }
            if (!n.carries) {
                // (Nothing comes out of a black hole: the flow out of one,
                // which there shouldn't be, brings no change.)
                continue;
            }
            next.heat += into * n.heat * heatGoing(n);
            next.amount += into * n.amount;
            next.vapour += into * n.vapour;
            next.mist += select(into * n.mist, 0.0, n.dew);
            next.lost += into * n.lost;
        }
    }
    // Gas moving down by `sinking` metres warms as it is squeezed.
    next.heat += gasHeatCap(heldGas(own)) * DRY_ADIABATIC * sinking;
    next.amount = max(next.amount, 0.0);
    next.vapour = max(next.vapour, 0.0);
    next.mist = max(next.mist, 0.0);
    next.lost = clamp(next.lost, 0.0, next.amount);
    writeHeld(outSet, i, next);
}

// Four numbers in [0, 1) for cell i this step (a PCG hash).
fn noise(i: u32) -> vec4<f32> {
    var h = i * 747796405u + P.frame * 2891336453u + P.seed;
    var out = vec4<f32>(0.0);
    for (var k = 0; k < 4; k++) {
        h = h * 747796405u + 2891336453u;
        let word = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
        out[k] = f32((word >> 22u) ^ word) / 4294967296.0;
    }
    return out;
}

// The gas as the substeps left it (S) into the cells, with its share of what
// its wet room's pockets gave out (wetOf: over all its takers', or, with
// none to take it, back to the givers, as each gave by volume), its
// temperature its heat over its heat capacity (a cell left holding nothing
// keeps its own), rounded at random into their f16s (common.wgsl's
// stochastic16), once a step.
@compute @workgroup_size(8, 8)
fn settle(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid.xy >= P.size)) {
        return;
    }
    let i = gid.y * P.size.x + gid.x;
    var h = readHeld(HELD_S, i);
    if (!h.carries) {
        return;
    }
    let wet = wetOf(vec2<i32>(gid.xy));
    if (wet.slot != NO_WET) {
        let takers = f32(atomicLoad(&sky[wet.slot + WET_TAKERS]));
        let givers = skyExact(wet.slot + WET_GIVERS);
        let back = select(0.0, wet.give * wet.volume / givers, givers > 0.0);
        let share = select(back, wet.take / takers, takers > 0.0);
        if (share > 0.0) {
            h.amount += share * skyExact(wet.slot + WET_GIVEN);
            h.vapour += share * skyExact(wet.slot + WET_GIVEN + 2u);
            h.mist += share * skyExact(wet.slot + WET_GIVEN + 4u);
            h.lost += share * skyExact(wet.slot + WET_GIVEN + 6u);
            h.heat += share * skyExact(wet.slot + WET_GIVEN + 8u);
        }
    }
    let g = heldGas(h);
    let heatCap = gasHeatCap(g);
    let temp = select(h.temp, h.heat / heatCap, heatCap > 1e-6);
    let r = noise(i);
    let w = packGas(Gas(stochastic16(g.vapour, r.x), stochastic16(g.mist, r.y), stochastic16(g.lost, r.z),
        stochastic16(g.amount, r.w)));
    cells[i].y = bitcast<u32>(temp);
    let k = gasIndex(P.size.x * P.size.y, i);
    if (gasHigh(i)) {
        cells[k].z = w.x;
        cells[k].w = w.y;
    } else {
        cells[k].x = w.x;
        cells[k].y = w.y;
    }
}
