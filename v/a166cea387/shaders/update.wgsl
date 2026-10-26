// Per-cell pass, once per step: heat (conduction, exchange with the outside
// under open sky; the wind has carried the air's already, carry.wgsl),
// moisture, oxygen, ignition and burning, phase
// changes with latent heat, shattering under pressure, crushing under piles,
// lifetimes, and the velocity of powders and gases (weight, wind drag,
// pressure push, buoyancy in liquid, resting). Liquids move as particles
// (fluid.wgsl). Reads `src` and writes `dst` so neighbor reads are stable;
// pile loads go from `loadIn` to `loadOut` the same way.

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> elems: array<Elem>;
@group(0) @binding(2) var<storage, read> src: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> dst: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> air: array<vec4<f32>>;
// Two per cell from fluid.wgsl, the first being liquid pressure, -, density
// (> 0 where liquid was), -.
@group(0) @binding(5) var<storage, read> fluid: array<vec4<f32>>;
// The skyline from sky.wgsl, a word per column, then the zones' sums.
@group(0) @binding(6) var<storage, read_write> sky: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read> loadIn: array<f32>;
@group(0) @binding(8) var<storage, read_write> loadOut: array<f32>;
// Tile marks for the body passes (bodies.wgsl), set here for loose cells in
// no body.
@group(0) @binding(9) var<storage, read_write> bodies: array<atomic<u32>>;
// Light (light.wgsl), per light cell of 2 x 2 cells: the share of the
// sunlight it takes in each of its cells gets per unit it takes in, and the
// visible light reaching it (rgb), which plants grow by.
@group(0) @binding(10) var heat: texture_2d<f32>;
@group(0) @binding(11) var light: texture_2d<f32>;
// The pull of the world's masses per light cell, cells/step² (gravity.wgsl).
@group(0) @binding(12) var gravity: texture_2d<f32>;
// Heat's clock per light cell, how real-time it is (clock.wgsl).
@group(0) @binding(13) var clock: texture_2d<f32>;
// Heat other passes put into cells as energy (heatin.wgsl), J, an exact sum
// in two words a cell: taken in here, and cleared.
@group(0) @binding(14) var<storage, read_write> heatIn: array<u32>;
// What it puts into the air (airsource.wgsl): gas a drop of mist leaves, or a
// cell holding gas turned into something else held.
@group(0) @binding(15) var<storage, read_write> source: array<atomic<u32>>;
// Radiant heat (radiate.wgsl), as last worked out: per cell what it took in
// (thermal radiance through its faces) and its emitting area.
@group(0) @binding(16) var radiant: texture_2d<f32>;

// Down (pull.wgsl): the total pull on this cell, the Earth's and the world's
// masses', down's step, and it as a unit vector, set at the start of
// updateCell.
var<private> localPull: vec2<f32>;
var<private> localDown: vec2<i32>;
var<private> localDir: vec2<f32>;

fn downStep() -> vec2<i32> {
    return localDown;
}

// Down's step turned by k eighths of a turn (turnStep): -2 is "right", the
// side +x is under the Earth's down.
fn downTurned(k: i32) -> vec2<i32> {
    return turnStep(localDown, k);
}

fn downDir() -> vec2<f32> {
    return localDir;
}

// A velocity (or offset) along down and across it, and set either.
fn alongDown(v: vec2<f32>) -> f32 {
    return dot(v, localDir);
}

fn alongSide(v: vec2<f32>) -> f32 {
    return dot(v, vec2<f32>(localDir.y, -localDir.x));
}

fn withAlongDown(v: vec2<f32>, a: f32) -> vec2<f32> {
    return v + (a - dot(v, localDir)) * localDir;
}

fn withAlongSide(v: vec2<f32>, s: f32) -> vec2<f32> {
    let side = vec2<f32>(localDir.y, -localDir.x);
    return v + (s - dot(v, side)) * side;
}

fn inGrid(q: vec2<i32>) -> bool {
    return all(q >= vec2<i32>(0)) && all(q < vec2<i32>(P.size));
}

// Whether the neighbour of around() at offset off is the one mist settles
// from (side -1) or into (1): along down's nearer axis, its vertical one for
// a diagonal down.
fn mistFrom(off: vec2<i32>, side: i32) -> bool {
    let dn = localDown;
    let axis = select(vec2<i32>(dn.x, 0), vec2<i32>(0, dn.y), dn.y != 0);
    return all(off == axis * side);
}

// Coulomb friction (REWRITE_PLAN.md 9-1): a resting grain's speed along the
// ground (relative to what it rests on) loses up to its element's friction
// (tan of its angle of repose) times what presses it down a step, its own
// weight (in liquid, less the liquid's buoyancy) and the load of the pile on
// it, the pull's part along down for each: whatever pushes it along the
// ground by less than that, it stays put (static friction); pushed by more,
// it slides, slowed by as much (kinetic). (Losing 60 % of its speed a step,
// as until 9-1, any push along the ground made it creep: on a flat floor
// under a pull 20° off down, 0.06 cells a step.)
fn coulomb(along: f32, ride: f32, grip: f32) -> f32 {
    let rel = along - ride;
    return ride + sign(rel) * max(abs(rel) - grip, 0.0);
}
const FLYING_SPEED: f32 = 0.3;
const CRUMBLE_SPEED: f32 = 0.5;    // fragments of a broken solid start out falling
const MAX_STORED_T: f32 = 9999.0;  // a bound on runaway heating, far past anything real here
// Piles. A resting powder cell bears its own weight, rho / 1000 (its pile's: a
// cell of water-dense pile weighs 1), plus that of the resting powder on
// it; weights are counted as if the world were LOAD_ROWS tall, so a pile
// filling half of it weighs the same at any resolution (LOAD_ROWS and
// LATERAL, the share of its load a pile presses sideways with, from
// js/shared.js). A loose structural solid under a pile, or beside one,
// crumbles once the load passes its crushP; standing structure takes it as
// stress (stress.wgsl).
// A powder cell in a liquid is a parcel a cell across of its grains with
// the liquid in the pores between them (REWRITE_PLAN.md 9-2): as dense as
// its pile (rho) and that liquid together, and dragged through the liquid
// as a grain that size is, by Ferguson & Church's law ("A simple universal
// equation for grain settling velocity", J. Sedimentary Research 74(6),
// 2004): per mass of the liquid it displaces, C1 nu / D^2 + (3/4) C2 |u| / D,
// Stokes' drag and a natural grain's drag coefficient. Sand's parcel at 1 cm
// settles through water at some 0.36 m/s, as a handful let go in it does;
// its grains alone at 0.07 (0.5 mm), as they rain out of a cloud.
const FC_C1: f32 = 18.0;
const FC_C2: f32 = 1.0;
const WATER_NU: f32 = 1e-6;       // water's kinematic viscosity, m²/s (CRC Handbook, 20 °C); the least a liquid's here
// The most the liquid's pressure gradient around a powder cell counts for,
// in multiples of still water's. A clump of sand hitting water spikes the
// pressure right at its front; the grains behind are shielded from it, so
// each grain feeling all of it would fling the clump back out.
const MAX_WET_PUSH: f32 = 2.0;
// A bubble a cell across rises through liquid at a spherical-cap bubble's
// terminal speed, 0.711 sqrt(g d) (Davies & Taylor 1950, Proc. R. Soc. A
// 200; Clift, Grace & Weber, Bubbles, Drops and Particles, 1978, ch. 8):
// 0.22 m/s at a centimetre, measured alike. A step's length grows with the
// square root of the cell's (js/shared.js stepSeconds), so it is the same
// BUBBLE_RISE sqrt(pull) cells a step at any scale, pull in cells a step².
// In a thick liquid it creeps instead, at Hadamard and Rybczynski's speed
// for a gas bubble, g d² / 12ν (Clift, Grace & Weber ch. 3): ν the liquid's
// beside it, cells² a step, d a cell. The two limits combine as
// Jamialahmadi, Branch & Müller-Steinhagen (1994, Chem. Eng. Res. Des. 72)
// combine them, U1 U2 / sqrt(U1² + U2²): water's bubble is the cap's (0.371
// cells a step at a centimetre), oil's (ν 0.3) 0.074, mud's (ν 100) 2e-4,
// which stays where it is. (At the cap's speed whatever the liquid, a
// bubble left in mud rose through it as through water, and mud let go of
// every bubble painted into it: REWRITE_PLAN.md 7-10.)
const BUBBLE_RISE: f32 = 0.711;
// Chance per step that hydrogen where the wind gets to under open sky
// (ventilated) escapes, mixing away into the atmosphere; under a roof, or
// deep in a hollow, it gathers.
const HYDROGEN_ESCAPE: f32 = 0.004;

// Oxygen (see the gas plane and flameReach in common.wgsl). What burns
// takes it (pairs.wgsl).
const OXYGEN_SPREAD: f32 = 0.2;   // share of the difference with each neighbor evened out per step
const SKY_OXYGEN: f32 = 0.3;      // share of the loss the outside air makes good per step where ventilated
// What chars (charTo) held hot without a flame chars, as in a charcoal kiln:
// the smothered core of a log, a pier standing in lava. Wood's pyrolysis
// goes by three first-order reactions, into gas, tar and char, A e^(-E/RT)
// each (Thurner & Mann 1981, Ind. Eng. Chem. Process Des. Dev. 20): about
// 0.0005 of it a second at 300 °C, 0.003 at 350, 0.05 at 450, in real time;
// here at fire's pace (FIRE_PACE, js/shared.js), on heat's clock.
fn pyrolysisRate(temp: f32) -> f32 {
    let rt = 8.314 * max(temp + KELVIN, 1.0);
    return 1.44e4 * exp(-88.6e3 / rt) + 4.13e6 * exp(-112.7e3 / rt) + 7.38e5 * exp(-106.5e3 / rt);
}
// Moisture (see WATER_VAPOUR in common.wgsl), in vapour units. What
// evaporates gives up water's latent heat (vapourLatent, the same for steam),
// and air its vapour condenses in (into mist, or a drop) gets it back.
// Share of the difference with each neighbor evened out per step. Together
// with what the wind takes (4 * WIND_MOISTURE_MAX) it must stay under all
// of a cell's vapour, or it would go negative.
const VAPOUR_SPREAD: f32 = 0.1;
// Evaporation across a water surface: the vapour density the air beside it
// is short of saturation at the water's temperature, carried off at
// MASS_TRANSFER metres a second. That is the air's film coefficient (5
// W/(m²·K)) over real air's heat capacity (1200 J/(m³·K)), by the analogy
// between heat and mass transfer (Lewis number about 1; Incropera et al.,
// Fundamentals of Heat and Mass Transfer, §6.7), on heat's clock: on the
// day's, the air against a pond all but saturates in a step, and what limits
// it is the vapour spreading away and the latent heat the water pays.
const MASS_TRANSFER: f32 = 0.0042;

// The share of the air's shortfall evaporating across a face in a step of
// tau seconds (MASS_TRANSFER over a cell's depth), as an exponential
// approach that, faces four at a time, never takes the air past saturation.
fn evaporationShare(tau: f32) -> f32 {
    return (1.0 - exp(-4.0 * MASS_TRANSFER * tau / P.dx)) * 0.25;
}
// The outdoor air's vapour mixes with gas under open sky as its heat does
// (outsideHeat): this share of the difference in a step of the normal day's.
const SKY_VAPOUR: f32 = 0.01;
// Mist settles at the Stokes speed of fog droplets about 15 µm across, 6.8
// mm/s under Earth's g (2 r² ρ g / 9 μ), as the pull there is of g, on the
// motion clock: at 1 cm a cell and 1/60 s a step, 0.0114 of it a step.
const MIST_SETTLING: f32 = 0.0068;
const MIST_SPREAD: f32 = 0.05;    // share of the difference with each neighbor mixed per step (fog spreading)
// Mist past MIST_GATHER_MIN, twice what saturates air at 20 °C, draws in the
// mist around it, the thinner the faster (droplets colliding and growing
// into raindrops), until it makes a whole drop. Thinner mist only drifts,
// spreads, settles and evaporates.
const MIST_GATHER: f32 = 0.02;
const MIST_GATHER_MIN: f32 = 2.0 * VAPOUR_20;
const DROP_MIST: f32 = 56.0;      // mist that makes a drop, along with the cell's vapour
const DEW_SLACK: f32 = 1.0;       // °C a surface may be warmer than the misty air against it and still collect dew
const DEW_GATHER: f32 = 0.08;     // share of the mist beside it misty air against a cold surface draws in per step
// Sediment: liquid flowing over a resting grain fast enough lifts it into
// the flow (see shieldsSpeed); it settles again where the flow slows, so
// streams cut channels and build deltas.
const LIFT: f32 = 0.3;            // upward push per step per cell/step of flow past the critical speed

// Whether liquid t is (or holds) water that evaporates: brine leaves its
// salt and mud its dirt (hiTo, as when they boil off; see RESIDUE).
fn holdsWater(t: u32) -> bool {
    return isWater(t) || t == MUD;
}

// Water in cell c giving off `vapour` units this step (into the air beside
// it: moistAir there), whether that was the last of it: fresh water a
// particle at a time, anything else whole (common.wgsl's takeLiquid).
fn evaporated(c: ptr<function, Cell>, vapour: f32) -> bool {
    return takeLiquid(c, vapour / WATER_VAPOUR);
}

// What the four cells beside a cell add up to: heat flowing in, and what they
// hold of the air's oxygen and moisture compared with it.
struct Around {
    // Heat passed in from the cells beside it (heatPassed), summed. Both
    // sides work out the same amount, so heat is conserved. Air holds so
    // little heat that it barely cools a lava flow, while warming quickly
    // itself.
    energy: f32,
    // For something that burns: whether the gas beside it holds a flame that
    // can burn some of it (a 255th of a cell, oxygenToBurn), and whether
    // water (or ice, snow, mud) is against it.
    flame: bool,
    wet: bool,
    deficit: f32,      // oxygen lost by the carriers beside it less this one's, by amount (around), summed
    vapour: f32,       // vapour held by the air beside it less this cell's, summed
    mist: f32,         // mist settling in and gathering, less what it gives up
    coldest: f32,      // the coldest solid or powder beside a cell of air (1e9 for none)
    // Vapour passing per step from water (or damp soil) to the air beside
    // it: into this cell of air, or out of this cell of water. Both sides
    // work it out the same way, so the water a cell of water loses (all at
    // once, at random, at this rate on average) is the vapour the air gains.
    evaporation: f32,
    room: bool,   // gas or liquid beside it, which ice owing cells can grow or push into
    // For something that can catch fire (an igniteT), the hottest its
    // surface is where it touches something that can't (contactTemp), or
    // its own temperature.
    contact: f32,
    // The temperatures of the cells beside it that hold heat, summed, and
    // how many: what a cell of nothing is as warm as.
    warmth: vec2<f32>,
}

// Two bodies brought together meet, at once, at the mean of their
// temperatures weighted by their thermal effusivities, √(k ρc), whatever
// their temperatures through do after: wood (about 400 J/(m²·K·s^½))
// against lava (about 2200) is at about 980 °C where they touch, and wood
// against hot metal at the metal's temperature. What catches fire is that
// surface, long before a centimetre of wood (a cell, whose temperature is
// its mean) has warmed through: wood touching lava chars and lights in a
// moment, as it does, rather than over minutes. Air and flames, with next to
// no effusivity, don't light anything by touch, and fuel doesn't light
// fuel: fire spreads by the flames it gives off (block.wgsl) and the heat it
// conducts, not by the step in temperature between two cells of the same
// burning log.
fn contactTemp(e: Elem, heatCap: f32, temp: f32, n: Elem, nCap: f32, nTemp: f32) -> f32 {
    let own = sqrt(e.thermal * heatCap);
    let other = sqrt(n.thermal * nCap);
    return select(temp, (own * temp + other * nTemp) / (own + other), own + other > 0.0);
}

// Whether the cell at index i is a gap in a body of liquid, which the fluid
// pass counts as liquid (fluid.wgsl): water doesn't evaporate into it. (It
// would fill with vapour, lost again as the gap closes.) Not a body's cup,
// which the liquid's solve takes for the body's (its drift's z the body's
// slot, below none): the air shut in a hull or a crate is air. (Taken for a
// gap, what it held went out into the air around the step the body first
// touched water, and its inside was a vacuum, as warm as whatever was
// beside it or the sun left it: a floating hull's hold in the heat view.)
fn isGap(i: u32) -> bool {
    return fluid[2u * i].z > 0.0 && fluid[2u * i + 1u].z >= 0.0;
}

// The air's mass beside the cell at c, in cells of air at 1 atm and 20 °C
// (gasAmount): the gas of the cells beside it that windBeside reads.
fn airBeside(c: vec2<i32>) -> f32 {
    let dims = airDims(P.size);
    var amount = 0.0;
    for (var k = 0; k < 4; k++) {
        let q = c + besideStep(k);
        if (gasCellAt(q, dims)) {
            amount += gasAmount(gasSrc(u32(q.y) * P.size.x + u32(q.x)));
        }
    }
    return amount;
}

// The air beside the cell at c pushed by du (cells a step), on the faces
// windBeside reads it from (airsource.wgsl's pushIn, for the next step's
// air), so what it reads there next is du more.
fn pushBeside(c: vec2<i32>, du: vec2<f32>) {
    let dims = airDims(P.size);
    for (var k = 0; k < 4; k++) {
        let d = besideStep(k);
        if (gasCellAt(c + d, dims)) {
            pushIn(c + d, d, du, dims);
        }
    }
}

// The gas cell i holds as of last step (common.wgsl's gas plane), and this
// step's.
fn gasSrc(i: u32) -> Gas {
    return gasFrom(src[gasIndex(P.size.x * P.size.y, i)], i);
}

fn setGasDst(i: u32, g: Gas) {
    let w = packGas(g);
    let k = gasIndex(P.size.x * P.size.y, i);
    if (gasHigh(i)) {
        dst[k].z = w.x;
        dst[k].w = w.y;
    } else {
        dst[k].x = w.x;
        dst[k].y = w.y;
    }
}

// The air's convection over heat's step (REWRITE_PLAN.md 4e-4). Heat goes on
// heat's clock, τ a step (up to the day's 6 s), but the flow moves the air a
// real step, dt, at a time: the air beside a warm surface piled up what it
// got, and the surface, against warm air, shed its own too slowly. So over
// the rest of heat's step, τ - dt, a sealed room's air mixes as its natural
// convection would turn it over: each cell of its gas toward the room's
// mean by s = 1 - exp(-(τ - dt) / t), t = H / U the turnover time, H the
// room's height and U = √(g H ΔT / T) its buoyant speed, ΔT how much warmer
// its air is below than above: the slope of its temperature on its height
// along the pull, across H (from its moments, air.wgsl's gather, common.wgsl's
// ZONE_MOMENTS; H for a room of even height √12 times its spread). Air
// warmer above than below is stable and stays so, as a warm layer under a
// ceiling does; a room heated from below turns over. The mean is weighted
// by s C (zoneHeat), so what the cells give and take sums to none; a cell on
// real time (s none) is left to the flow, and so is the open air (its
// outdoor exchange, outsideHeat, goes on heat's clock already) and a room
// with no pull to stir it.
struct Mix {
    slot: u32,
    share: f32,
}

// The union-find's links (label.wgsl): the room pieces', in the sky.
fn ufLink(at: u32) -> u32 {
    return atomicLoad(&sky[at]);
}

fn ufLower(at: u32, to: u32) -> u32 {
    return atomicMin(&sky[at], to);
}

// The first piece of the room piece k is in (common.wgsl's roomLink).
fn roomFirstOf(k: u32) -> u32 {
    return ufFind(roomLink(P.size, 0u), k);
}

// The zone slot of the gas at p as this step's rooms found it (sky.wgsl),
// or NO_ZONE: no room (gas since, or a gap the liquid counts as its own), or
// a pocket past its tile's ROOM_PIECES.
const NO_ZONE: u32 = 0xffffffffu;

fn zoneSlotOfCell(p: vec2<u32>) -> u32 {
    let byte = (atomicLoad(&sky[roomWord(P.size, p)]) >> (8u * (p.x % 4u))) & 0xffu;
    if (byte >= UNNUMBERED) {
        return NO_ZONE;
    }
    return zoneSlotAt(P.size, P.frame, roomFirstOf(roomPieceAt(P.size, p, byte)));
}

// The room's zone slot and the share s the gas of t at p (its clock's
// realness real) mixes by this step; none but for a sealed room's gas.
fn mixOf(p: vec2<u32>, t: u32, real: f32) -> Mix {
    var m = Mix(0u, 0.0);
    let i = p.y * P.size.x + p.x;
    if (!flowsThrough(t, liquidSide(fluid[2u * i], fluid[2u * i + 1u])) || t == BLACK_HOLE) {
        return m;
    }
    let z = zoneSlotOfCell(p);
    if (z == NO_ZONE || (atomicLoad(&sky[z + ZONE_OPEN]) & (ZONE_OUTSIDE | ZONE_SINK)) != 0u) {
        return m;
    }
    let lag = max(clockAt(real) - P.dt, 0.0);
    let cap = skyExact(z + ZONE_MOMENTS, EXACT_COARSE);
    if (lag <= 0.0 || cap <= 0.0) {
        return m;
    }
    let temp = skyExact(z + ZONE_MOMENTS + 2u, EXACT_COARSE) / cap;
    let height = skyExact(z + ZONE_MOMENTS + 4u, EXACT_COARSE) / cap;
    let spread = skyExact(z + ZONE_MOMENTS + 8u, EXACT_COARSE) / cap - height * height;
    if (spread <= 0.0) {
        return m;
    }
    let cov = skyExact(z + ZONE_MOMENTS + 6u, EXACT_COARSE) / cap - temp * height;
    let tall = sqrt(12.0 * spread);
    let warmer = -cov / spread * tall;
    if (warmer <= 0.0) {
        return m;
    }
    // (Height along the pull in metres of Earth's g: g H is EARTH_G H.)
    let speed = sqrt(EARTH_G * tall * warmer / (temp + KELVIN));
    m.slot = z;
    m.share = 1.0 - exp(-lag * speed / tall);
    return m;
}

// The heat (J) the cell of t at p (heat capacity heatCap, at temp) takes in
// from its room's mixing this step (mixOf).
fn mixHeatOf(p: vec2<u32>, t: u32, real: f32, heatCap: f32, temp: f32) -> f32 {
    let m = mixOf(p, t, real);
    if (m.share <= 0.0) {
        return 0.0;
    }
    let cap = skyExact(m.slot + ZONE_MIX_CAP, EXACT_FINE);
    if (cap <= 0.0) {
        return 0.0;
    }
    let mean = skyExact(m.slot + ZONE_MIX_HEAT, EXACT_FINE) / cap;
    return m.share * heatCap * (mean - temp);
}

// The exact sum (common.wgsl's exactParts) in the sky's words at, at + 1.
fn skyExact(at: u32, scale: f32) -> f32 {
    return exactValue(vec2<u32>(atomicLoad(&sky[at]), atomicLoad(&sky[at + 1u])), scale);
}

// x added to the exact sum in the sky's words at, at + 1.
fn addSkyExact(at: u32, x: f32, scale: f32) {
    let w = exactParts(x, scale);
    let old = atomicAdd(&sky[at], w.x);
    atomicAdd(&sky[at + 1u], w.y + exactCarry(old, w.x));
}

// Per thread of zoneHeat's workgroup: its cell's room's zone slot (NO_MIX
// for none) and what it mixes by, s C and s C T.
const NO_MIX: u32 = 0xffffffffu;
var<workgroup> mixSlot: array<u32, 64>;
var<workgroup> mixCap: array<f32, 64>;
var<workgroup> mixHeat: array<f32, 64>;

// Each sealed room's s C and s C T summed (ZONE_MIX_CAP, ZONE_MIX_HEAT), over
// the cells as the update pass is about to take them, with the heat
// capacity and temperature it takes them at: a workgroup's cells of a room
// first, by the first of its threads in it, then into the room's words.
@compute @workgroup_size(8, 8)
fn zoneHeat(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    var slot = NO_MIX;
    var cap = 0.0;
    var heat = 0.0;
    if (all(gid.xy < P.size)) {
        let i = gid.y * P.size.x + gid.x;
        let c = unpackCell(src[i]);
        let m = mixOf(gid.xy, c.t, realnessAt(gid.xy));
        if (m.share > 0.0) {
            let lnAtm = atmLog(sampleAir(vec2<f32>(gid.xy) + 0.5).p);
            let temp = effTempIn(c.t, c.temp, phaseOf(c.t, soluteOf(c), strengthOf(c), lnAtm));
            slot = m.slot;
            cap = m.share * cellHeatCap(c, gasSrc(i));
            heat = cap * temp;
        }
    }
    mixSlot[li] = slot;
    mixCap[li] = cap;
    mixHeat[li] = heat;
    workgroupBarrier();
    if (slot == NO_MIX) {
        return;
    }
    for (var k = 0u; k < li; k++) {
        if (mixSlot[k] == slot) {
            return;
        }
    }
    var sumCap = 0.0;
    var sumHeat = 0.0;
    for (var k = li; k < 64u; k++) {
        if (mixSlot[k] == slot) {
            sumCap += mixCap[k];
            sumHeat += mixHeat[k];
        }
    }
    addSkyExact(slot + ZONE_MIX_CAP, sumCap, EXACT_FINE);
    addSkyExact(slot + ZONE_MIX_HEAT, sumHeat, EXACT_FINE);
}

// The Earth's pull, m/s² (water's film goes by it: common.wgsl's waterFilm).
fn earthPull() -> f32 {
    return P.earthG * P.dx / (P.dt * P.dt);
}

// How fast heat moves: two clocks, and the clock field between them
// (clock.wgsl).
fn realnessAt(p: vec2<u32>) -> f32 {
    return textureLoad(clock, p / 2u, 0).x;
}

fn clockAt(real: f32) -> f32 {
    return clockStep(real, P.dayDt, P.clockSpan);
}

// J (per CELL_M cube, see faceShare) a cell of e (heat capacity heatCap) at
// temp takes in per step from a cell of n (nCap) at nTemp beside it, by
// their conductance g (cellFaceConductance in common.wgsl) and how long a
// step stands for at the faster of their clocks, tau seconds (see around),
// so both work it out alike. Only across a
// face mild enough for it (HEAT_SHARE_MAX); a stiffer one went in
// heat.wgsl's stages before this pass, or with none this step (heat.wgsl's
// are taken on once its prepare pass has found stiff faces, a frame or so
// later), passes what stays monotone.
fn heatPassed(e: Elem, n: Elem, heatCap: f32, nCap: f32, temp: f32, nTemp: f32, tau: f32, conductance: f32) -> f32 {
    let g = conductance * faceSeconds(e, n, tau, P.dayDt);
    let small = min(heatCap, nCap);
    if (small <= 0.0) {
        return 0.0;
    }
    if (g <= HEAT_SHARE_MAX * small) {
        return g * (nTemp - temp);
    }
    return select(HEAT_SHARE_MAX * small * (nTemp - temp), 0.0, P.heatStages > 0u);
}

// lnAtm is the air pressure at the cell (as atmLog); a neighbour's
// temperature goes by its own (tempAtCell), as it works it out itself.
// real is its realness (realnessAt), tau the seconds a step stands for by it
// (clockAt), lost the share of its oxygen used up and amount its dry air's
// (what it evens out with the cells beside it). Everything is traded from
// what the cells hold as the wind has carried them (carry.wgsl), this
// cell's and its neighbours' alike, so both sides of each exchange agree.
fn around(p: vec2<i32>, c: Cell, e: Elem, temp: f32, heatCap: f32, lnAtm: f32, real: f32, tau: f32,
    lost: f32, amount: f32) -> Around {
    var a: Around;
    let atm = exp(lnAtm);
    let carrier = carriesOxygen(c.t);
    let water = holdsWater(c.t);
    let vapourous = holdsVapour(c.t);
    let i = u32(p.y) * P.size.x + u32(p.x);
    let gap = vapourous && isGap(i);
    let was = gasSrc(i);
    let own = vec2<f32>(was.vapour, was.mist);
    let held = own.x;
    let dew = c.t == AIR && (c.flags & FL_DEW) != 0u;
    a.coldest = 1e9;
    // Damp soil gives off its water as water does, the less the drier it
    // is: its surface is only that wet.
    let soil = porous(e);
    let wetness = select(f32(soilWaterOf(c)) / f32(SOIL_FULL), 1.0, water);
    // A plant cell with its stomata open gives off water as a leaf does.
    let leaf = holdsPlantWater(c.t) && transpiring(p, packWord(c.t, c.aux, c.life, c.shade, c.flags));
    let fuel = e.igniteT < NEVER;
    a.contact = temp;
    // The share of its mist settling into the air below a step (MIST_SETTLING).
    let settling = min(MIST_SETTLING * pullShare(localPull) * P.dt / P.dx, 0.25);
    for (var d = 0; d < 4; d++) {
        // Left, right, above, below.
        let q = p + select(vec2<i32>(0, (d & 1) * 2 - 1), vec2<i32>((d & 1) * 2 - 1, 0), d < 2);
        if (any(q < vec2<i32>(0)) || any(q >= vec2<i32>(P.size))) {
            continue;
        }
        let j = u32(q.y) * P.size.x + u32(q.x);
        let v = src[j];
        let t = cellType(v);
        let n = elems[t];
        let nTemp = tempAtCell(v, q);
        // What it holds of the gas (the gas plane): its heat capacity, if a
        // gas's, and what trades oxygen or moisture with this cell, or burns.
        var ng = noGas();
        if (holdsGas(t)) {
            ng = gasSrc(j);
        }
        let nCap = heatCapIn(v, ng);
        if (nCap > 0.0) {
            a.warmth += vec2<f32>(nTemp, 1.0);
        }
        // The faster of the two clocks: the neighbour's only differs from
        // this cell's out of its light cell, and only matters if faster.
        var faceTau = tau;
        if (any(q / 2 != p / 2)) {
            let nReal = realnessAt(vec2<u32>(q));
            if (nReal > real) {
                faceTau = clockAt(nReal);
            }
        }
        // (The wind past a gas's face with anything else: its film's.)
        var wind = 0.0;
        if ((e.kind == KIND_GAS) != (n.kind == KIND_GAS)) {
            wind = windSpeedAt(select(q, p, e.kind == KIND_GAS));
        }
        let mine = packWord(c.t, c.aux, c.life, c.shade, c.flags);
        var conductance = cellFaceConductance(c.t, mine, was, temp, t, v.x, ng, nTemp, earthPull(), P.dx, wind);
        // A liquid beside a liquid, turning over across heat's step
        // (common.wgsl's turnoverConductance; d 2 the cell above).
        // (By an if: select() works out both.)
        if (e.kind == KIND_LIQUID && n.kind == KIND_LIQUID) {
            if (d == 2) {
                conductance += turnoverConductance(v.x, cellTemp(v), mine, c.temp, nCap, heatCap, earthPull(), P.dx, faceTau, P.dt, false);
            } else {
                conductance += turnoverConductance(mine, c.temp, v.x, cellTemp(v), heatCap, nCap, earthPull(), P.dx, faceTau, P.dt, d < 2);
            }
        }
        a.energy += heatPassed(e, n, heatCap, nCap, temp, nTemp, faceTau, conductance);
        if (fuel) {
            if (n.igniteT >= NEVER) {
                a.contact = max(a.contact, contactTemp(e, heatCap, temp, n, nCap, nTemp));
            }
            // (As pairs.wgsl's burn needs it: its oxygen need times what
            // the cell holds of the fuel.)
            a.flame = a.flame || oxygenToBurn(t, ng, nTemp) >= e.oxygenNeed * select(1.0, liquidAmount(c), e.kind == KIND_LIQUID) / 255.0;
            a.wet = a.wet || holdsWater(t) || t == ICE || t == SNOW;
        }
        // Oxygen evens out by amount, as much going each way: so much of the
        // less of the two cells' dry air trades places, its oxygen with it
        // (a hot cell holds less air than a cold one).
        if (carrier && carriesOxygen(t) && amount > 0.0) {
            a.deficit += min(ng.amount, amount) / amount * (ng.lost - lost);
        }
        // (A gap in the liquid, which the liquid and the air count as liquid,
        // common.wgsl's flowsThrough, trades no moisture: a cell of nothing,
        // short of the liquid that left it.)
        if (vapourous && !gap) {
            if (holdsVapour(t) && !isGap(j)) {
                let f = vec2<f32>(ng.vapour, ng.mist);
                // Whether either holds dew, as of last step so the two cells
                // agree (see FL_DEW). Mist collects against a cold surface
                // from the mist around it, and stays there: dew settles on
                // the surface, out of the wind.
                let nDew = t == AIR && cellHas(v, FL_DEW);
                let afloat = select(vec2<f32>(1.0), vec2<f32>(1.0, 0.0), dew || nDew);
                // Spreading, settling and gathering; dew gathers the mist
                // beside it.
                a.vapour += f.x - own.x;
                a.mist += MIST_SPREAD * (f.y - own.y) * afloat.y;
                if (mistFrom(q - p, -1)) {
                    a.mist += f.y * settling * afloat.y;
                } else if (mistFrom(q - p, 1)) {
                    a.mist -= own.y * settling * afloat.y;
                }
                if (dew && !nDew) {
                    a.mist += f.y * DEW_GATHER;
                } else if (nDew && !dew) {
                    a.mist -= own.y * DEW_GATHER;
                } else if (own.y > f.y && (own.y >= MIST_GATHER_MIN || dew)) {
                    a.mist += f.y * MIST_GATHER;
                } else if (f.y > own.y && (f.y >= MIST_GATHER_MIN || nDew)) {
                    a.mist -= own.y * MIST_GATHER;
                }
            } else if (holdsWater(t)) {
                a.evaporation += evaporationShare(faceTau) * max(saturationAt(nTemp, atm) - held, 0.0);
            } else if (holdsSoilWater(t)) {
                a.evaporation += evaporationShare(faceTau) * f32(wordSoilWater(v.x)) / f32(SOIL_FULL)
                    * max(saturationAt(nTemp, atm) - held, 0.0);
            } else if (holdsPlantWater(t) && transpiring(q, v.x)) {
                a.evaporation += transpirationShare(faceTau, wordPlantWater(v.x)) * max(saturationAt(nTemp, atm) - held, 0.0);
            }
            if (n.kind == KIND_SOLID || n.kind == KIND_POWDER) {
                a.coldest = min(a.coldest, nTemp);
            }
        } else if ((water || soil || leaf) && holdsVapour(t) && !isGap(j)) {
            // (At the gas's pressure, its temperature by its boiling point
            // there, as the gas works it out: the two sides agree on it.)
            let theirs = atmLog(airAt(q / i32(AIR_CELL), airDims(P.size)).x);
            let surface = effTempIn(c.t, c.temp, phaseOf(c.t, soluteOf(c), strengthOf(c), theirs));
            let share = select(evaporationShare(faceTau) * wetness, transpirationShare(faceTau, plantWaterOf(c)), leaf);
            a.evaporation += share * max(saturationAt(surface, exp(theirs)) - ng.vapour, 0.0);
        }
        a.room = a.room || n.kind == KIND_GAS || n.kind == KIND_LIQUID;
    }
    return a;
}

// Whether the cell at p is outdoors (REWRITE_PLAN.md 8-4): gas of the open
// air, its zone reaching the world's open edges (air.wgsl's opensOut, the
// pressure it is at), and under the open sky, over its column's roof
// (sky.wgsl's skyline, which looks through foliage): what mixes with the
// outside air's heat, moisture and oxygen. Air under a roof but open to the
// world's side is at the outside air's pressure, but sheltered (a room with
// a door, a crawlspace under a raised box); air under the sky but in no zone
// of the open air (a pocket leaves close in, a gap the liquid counts as its
// own) isn't outdoors either. (Until 8-4 the skyline alone said so.)
fn outdoors(p: vec2<u32>) -> bool {
    if (p.y >= (atomicLoad(&sky[p.x]) >> 16u)) {
        return false;
    }
    let z = zoneSlotOfCell(p);
    return z != NO_ZONE && (atomicLoad(&sky[z + ZONE_OPEN]) & ZONE_OUTSIDE) != 0u;
}

// Whether the cell at p, outdoors, is where the wind gets to (ventilation in
// sky.wgsl), so takes the outside air's oxygen, and hydrogen there escapes:
// not deep in a hollow open to the sky, a roofless room, a pit, whose oxygen
// is made good from above only as fast as it spreads. (The ventilation is
// worked out every few steps, so ask outdoors(p), as of now, first.)
fn ventilated(p: vec2<u32>) -> bool {
    return p.y < atomicLoad(&sky[ventilationAt(P.size, p.x)]);
}

// The height of cell p's middle, metres, as the pull has it (pull.wgsl's
// altitudeAt).
fn altitude(p: vec2<u32>) -> f32 {
    return altitudeAt(vec2<f32>(p) + 0.5);
}

// Heat exchanged with the air outside, which only reaches gas under open sky
// (sky.wgsl): it mixes with the outdoor air, at 20 °C less with height
// (outdoorT) and warmer by day than by night (dayWarmth). Anything
// sheltered, like air in a sealed room, keeps its heat. (What is under the
// sky trades heat with it by radiation; see radiantHeat.) Elem.ambient is
// the share of the difference made up in a step of the normal day's,
// DAY_STEP seconds; in a step of tau, as much as that many of them would.
//
// And gas beside a surface, warmer than the outdoor air, is carried off by
// its own buoyancy over the part of heat's step the flow doesn't cover, τ -
// dt (REWRITE_PLAN.md 4e-6, the open air's side of 4e-4): a parcel a cell
// across rises its own height in √(dx T / (g ΔT)), 0.07 s for a centimetre
// 60 K warm, and the open air takes its place, so the surface's film (its
// natural convection, Elem.film) meets the outdoor air, not a layer of its
// own warmed air. (Over the 6 s of the day's step, the coal of the Tabletop
// in the sun came to 106 °C against the 82 its sunlight, its radiation to
// the sky and its film's 5 W/(m²·K) balance at.) Air colder than outdoors
// lies where it is, as a night's cold pools; air beside anything past
// DAY_HEAT, or near it (its own clock at all real), is left to the flow, so
// a heater's plume carries its heat up (that face trades heat on the
// hotter's faster clock); and air clear of surfaces keeps its heat, rising.
fn outsideHeat(p: vec2<u32>, e: Elem, heatCap: f32, temp: f32, tau: f32, real: f32) -> f32 {
    if (e.kind != KIND_GAS || !outdoors(p)) {
        return 0.0;
    }
    let outside = outdoorT(altitude(p)) + dayWarmth(P.day);
    var share = 1.0 - pow(1.0 - e.ambient, tau / DAY_STEP);
    let lag = tau - P.dt;
    let warmer = temp - outside;
    if (real <= 0.0 && lag > 0.0 && warmer > 0.0 && besideSurface(p) <= DAY_HEAT) {
        let rise = sqrt(P.dx * (temp + KELVIN) / (max(earthPull(), 1e-6) * warmer));
        share = max(share, 1.0 - exp(-lag / rise));
    }
    return share * heatCap * (outside - temp);
}

// The hottest of what isn't gas beside the cell at p (its faces), °C; past
// any temperature (NEVER) with none.
fn besideSurface(p: vec2<u32>) -> f32 {
    var hottest = -NEVER;
    for (var k = 0; k < 4; k++) {
        let q = vec2<i32>(p) + besideStep(k);
        if (inGrid(q)) {
            let v = src[u32(q.y) * P.size.x + u32(q.x)];
            if (elems[cellType(v)].kind != KIND_GAS) {
                hottest = max(hottest, wordTempOf(v));
            }
        }
    }
    return select(hottest, NEVER, hottest == -NEVER);
}

// Radiant heat: W a cell takes in per unit of thermal radiance (common.wgsl's
// thermalRadiance, σT⁴ in units of one at 1000 K) through a face CELL_M
// across (see faceShare): σ (5.67e-8 W/(m²·K⁴)) times (1000 K)⁴ times its
// area. Per step, times how long a step stands for: by the faster of the
// cell's clock and the clock of what reaches it (as real-time as a body at
// the temperature that would give it off), so what something hot gives off
// in real time is taken in on the same clock however far it shines.
const RADIANT: f32 = 5.67;
// Sunlight, W a cell takes in that is all of its light cell's share of full
// sunlight square to it: 1000 W/m² (a clear day's at the ground) across a
// face CELL_M across. Per step, times how long a step stands for by the
// cell's clock: the day's for the ground it warms, real time for lava
// (which would otherwise take in hours of sun for its seconds of cooling).
const SUN_HEAT: f32 = 1000.0 * CELL_M * CELL_M;
const LUMA: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

// Heat a cell takes in by radiation: the thermal infrared it took in less
// what it gives off (radiate.wgsl), and the sunlight it doesn't let through
// or reflect, its share of what its light cell takes in (light.wgsl). So a lava pool warms
// a cave's walls, a fire sets wood alight across a gap, snow in the sun
// melts and in the shade doesn't, and glass, which lets the sunlight in but
// not the heat back out, keeps the frost off a glasshouse's beds. Walls
// (emissivity 0) take in and give off none. real is the cell's realness, tau
// the seconds a step stands for by it, mist what mist it holds.
fn radiantHeat(p: vec2<u32>, c: Cell, e: Elem, temp: f32, heatCap: f32, real: f32, tau: f32, g: Gas) -> f32 {
    let h = textureLoad(heat, p / 2u, 0);
    // As the light passes take it: what it lets through at this scale.
    let clear = clarityOf(c, g, P.dx);
    let share = faceShare(P.dx);
    // The infrared it took in when it was last worked out (radiate.wgsl),
    // less what it gives off at its temperature now: its emitting area
    // times its thermal radiance. What it takes in, as the temperature it
    // would give it off at, sets its clock with its own (decision 1).
    let ir = textureLoad(radiant, p, 0);
    var infrared = 0.0;
    if (ir.y > 0.0) {
        let settled = 1000.0 * sqrt(sqrt(max(ir.x / ir.y, 0.0))) - KELVIN;
        // On the clock of the hottest thing it sees (how real-time that is,
        // radiate.wgsl's peakSlot), the clock that gives it off: what it
        // takes in as a whole, a hot plate's share and a cool room's, is
        // cooler, and on that clock it took in what a 500 °C plate gave off
        // in real time three to six times over, ten cells off
        // (REWRITE_PLAN.md 6-5).
        let hottest = f32(bitcast<u32>(ir.w) & 0xffu) / 255.0;
        infrared = RADIANT * share * clockAt(max(max(real, realness(settled)), hottest)) * (ir.x - ir.y * thermalRadiance(temp));
        // What it takes in is a few steps old: never past where it would
        // give off as much, or a cell holding little heat (a flame, smoke)
        // would overshoot and swing.
        let room = (settled - temp) * heatCap;
        infrared = select(max(infrared, room), min(infrared, room), room > 0.0);
    }
    // A matte surface sends back its albedo of the sunlight on it; a clear
    // solid or liquid (glass, water, ice) takes in what it doesn't let
    // through, much of it the sun's near infrared (about a tenth of
    // sunlight in a centimetre of window glass).
    let reflect = select(dot(albedo(c.t), LUMA), 0.0, e.clarity > 0.0 && e.kind != KIND_GAS);
    let sun = SUN_HEAT * share * tau * h.x * (1.0 - clear.x) * (1.0 - reflect);
    return infrared + sun;
}

// Plants grow by light (REWRITE_PLAN.md 5d; block.wgsl, pairs.wgsl). A plant
// cell with its stomata open makes PHOTO_SUGAR of sugar a step in full
// sunlight, as much less as the light on it is (growth goes as the light a
// leaf takes in: Monteith 1977), and breathes RESPIRE_SUGAR of it away a
// step at 20 °C, twice as much each 10 °C warmer (Q10 2): so a plant lives
// in the dark as long as its sugar lasts. Out of sugar it starves, and dried
// out it dies, each as likely a step as PLANT_DEATH (into sawdust). Its
// stomata are open in light past GROW_LIGHT (open shade will do, deep shade
// and night won't) while it is watered past PLANT_WILT, the wider the more
// watered (fully open, some 5 mm/s of conductance, STOMATA; Nobel,
// Physicochemical and Environmental Plant Physiology, 2009); open, it gives
// off water through them in series with the air's film (transpiration), by
// what a real leaf's air would lack of saturation: the world's air holds
// GAS_COMPRESSION times real air's water (decision 3), which a water surface
// evaporates by, but a leaf's water comes up from its roots, which real
// soil feeds as real leaves need (about 0.2 g/cm² a day in the sun). It
// pays water's latent heat, so a leaf in the sun is cooled and a sealed
// terrarium's air grows damp. A side brighter than the other by LEAN_LIGHT
// is the one it grows toward.
const GROW_LIGHT: f32 = 0.06;
const LEAN_LIGHT: f32 = 1.25;
const PHOTO_SUGAR: f32 = 0.05;
const RESPIRE_SUGAR: f32 = 0.005;
const PLANT_DEATH: f32 = 0.001;
const STOMATA: f32 = 5e-3;
const SPROUT_CHANCE: f32 = 0.02;  // per step, that a seed where it can drink sprouts

// The share of the air's shortfall a leaf holding water gives off across a
// face in a step of tau seconds: through its stomata as open as it is
// watered past PLANT_WILT, by a real leaf's air's shortfall (see STOMATA).
fn transpirationShare(tau: f32, water: u32) -> f32 {
    let open = STOMATA * clamp((f32(water) - f32(PLANT_WILT)) / f32(PLANT_WATER - PLANT_WILT), 0.0, 1.0);
    let g = 1.0 / (1.0 / MASS_TRANSFER + 1.0 / max(open, 1e-9));
    return min(g * tau / P.dx / GAS_COMPRESSION, 0.25);
}

// Whether the plant cell at p holding v has its stomata open.
fn transpiring(p: vec2<i32>, v: u32) -> bool {
    return wordPlantWater(v) >= PLANT_WILT && !wordHas(v, FL_BURN) && lightAt(p) >= GROW_LIGHT;
}

fn lightAt(p: vec2<i32>) -> f32 {
    let q = clamp(p / 2, vec2<i32>(0), vec2<i32>(textureDimensions(light)) - 1);
    return dot(textureLoad(light, q, 0).rgb, LUMA);
}

// Whether there is fresh water, mud or soil past its wilting point beside the
// cell at p (or diagonally: plants grow diagonally too): what a seed needs to
// sprout.
fn canDrink(p: vec2<i32>) -> bool {
    for (var k = 0; k < 9; k++) {
        let r = p + vec2<i32>(k % 3 - 1, k / 3 - 1);
        if (k == 4 || any(r < vec2<i32>(0)) || any(r >= vec2<i32>(P.size))) {
            continue;
        }
        let v = src[u32(r.y) * P.size.x + u32(r.x)].x;
        let t = wordType(v);
        if ((holdsSolute(t) && wordSolute(v) == 0u) || t == MUD || (holdsSoilWater(t) && soilDrinkable(elems[t], wordSoilWater(v)))) {
            return true;
        }
    }
    return false;
}

// A plant cell at p, at temp: it leans toward its brighter side, makes sugar
// with its stomata open and breathes it away (see PHOTO_SUGAR), and starves
// or dries out.
fn plantNeeds(p: vec2<u32>, c: Cell, temp: f32) -> Cell {
    var n = c;
    let q = vec2<i32>(p);
    // Its sides, across its down (block.wgsl's plants grow by them).
    let left = lightAt(q - downTurned(-2) * 2);
    let right = lightAt(q + downTurned(-2) * 2);
    var lean = 0u;
    if (right > left * LEAN_LIGHT + 0.01) {
        lean = PLANT_RIGHT;
    } else if (left > right * LEAN_LIGHT + 0.01) {
        lean = PLANT_LEFT;
    }
    n = withLean(n, lean);
    let light = lightAt(q);
    var sugar = f32(sugarOf(c)) - RESPIRE_SUGAR * exp2((temp - AMBIENT_T) / 10.0);
    if (plantWaterOf(c) >= PLANT_WILT && light >= GROW_LIGHT) {
        sugar += PHOTO_SUGAR * min(light, 1.0);
    }
    n = withSugar(n, u32(clamp(floor(sugar + rnd()), 0.0, 255.0)));
    if ((sugarOf(n) == 0u || plantWaterOf(c) == 0u) && rnd() < PLANT_DEATH) {
        return convert(c, SAWDUST);
    }
    return n;
}

// The vapour a cell of gas holding dry of its own gas has room for at temp
// under its zone's pressure atm: its dry gas's at atm (vapourRoom, Dalton's
// law), or if more, the vapour pressure's in the cell as it is (a cell's
// worth at 1 atm, saturationAt with no cap): the two are one for gas at its
// zone's pressure, and gas squeezed into a cell will spread before it
// condenses, while gas thinned (a cell the air is filling) has room for the
// vapour pressure's worth now. (By Dalton's alone a cell holding next to
// nothing had room for none: what vapour it took in condensed into it, and
// heated it to the boiling point.)
fn roomFor(dry: f32, temp: f32, atm: f32) -> f32 {
    return max(vapourRoom(dry, temp, atm), saturationAt(temp, VAPOUR_ENDLESS));
}

// Saturation adjustment (Soong & Ogura, J. Atmos. Sci. 30, 1973; as cloud
// models do it): gas at temp holding vapour and mist (units), of heat
// capacity heatCap, and dry of its own gas, brought to where its vapour
// just saturates it, what condenses giving off water's latent heat into it
// and what evaporates taking it out: heatCap (t - temp) = latent (vapour -
// room(t)), or all of the mist evaporated if the gas can take it at the
// temperature that leaves. Its room for vapour, room(t), is its dry gas's at
// its zone's pressure, atm (vapourRoom, by Dalton's law): so a cell of water
// just boiled, all of it vapour, condenses down to its boiling point there
// and no further, and gas squeezed or spread doesn't condense or dry for it.
// Found between the temperatures with all of the mist evaporated and all of
// the vapour condensed (and below the boiling point, past which the gas has
// room for it all), by halving: room(t) is as steep as it likes near the
// boiling point, with little gas of its own, where Newton's method strayed.
// Implicit, so stable however fast, and the water and heat come out exact:
// the vapour is what the heat leaves at the temperature found. Returns the
// temperature, vapour and mist.
fn saturationAdjust(temp: f32, vapour: f32, mist: f32, heatCap: f32, atm: f32, dry: f32) -> vec3<f32> {
    let latent = vapourLatent();
    let water = vapour + mist;
    let evaporated = temp - latent * mist / heatCap;
    if (water <= roomFor(dry, evaporated, atm)) {
        return vec3<f32>(evaporated, water, 0.0);
    }
    var lo = evaporated;
    var hi = min(temp + latent * vapour / heatCap, max(boilingAt(atm), evaporated));
    for (var k = 0; k < 20; k++) {
        let t = 0.5 * (lo + hi);
        if (heatCap * (t - temp) < latent * (vapour - min(roomFor(dry, t, atm), water))) {
            lo = t;
        } else {
            hi = t;
        }
    }
    let v = clamp(vapour - heatCap * (hi - temp) / latent, 0.0, water);
    return vec3<f32>(temp + latent * (vapour - v) / heatCap, v, water - v);
}

// A cell of gas takes in the vapour from the water and gas beside it (and
// under open sky mixes with the outdoor air: even deep in a hollow, where its
// oxygen doesn't, see ventilated); what is past saturation at its
// temperature condenses into mist, and mist evaporates again into gas that
// isn't saturated. Mist settles and gathers (see Around); enough of it in one
// cell makes a drop of rain, or a flake of snow below freezing. So humid air
// cooling, on a cold window or rising off a hot bath into the cold, fogs up
// and drips. p is where it is (its row for the outdoor air's vapour there).
// tau is how long a step stands for, by heat's clock there; atm the
// pressure of its zone. Every gas does all this with the moisture it holds
// (g, the gas plane's, as the carry has left it), but only air gathers dew.
fn moistAir(c: Cell, g: ptr<function, Gas>, nb: Around, outside: bool, p: vec2<u32>, tau: f32, atm: f32) -> Cell {
    var n = c;
    var vapour = (*g).vapour + VAPOUR_SPREAD * nb.vapour + nb.evaporation;
    var mist = (*g).mist + nb.mist;
    if (outside) {
        let share = 1.0 - pow(1.0 - SKY_VAPOUR, tau / DAY_STEP);
        vapour += (outdoorVapour(altitude(p)) - vapour) * share;
    }
    // (What a cell trades with those beside it a step is less than it holds,
    // so these hold none less than none but its rounding.)
    vapour = max(vapour, 0.0);
    mist = max(mist, 0.0);
    // Condensing, the vapour gives the gas its latent heat; mist evaporating
    // takes it back: to saturation at the temperature that leaves it.
    let heatCap = gasHeatCap(Gas(vapour, mist, (*g).lost, (*g).amount));
    if (heatCap <= 0.0) {
        *g = noGas();
        return n;
    }
    let adjusted = saturationAdjust(c.temp, vapour, mist, heatCap, atm, (*g).amount);
    n.temp = adjusted.x;
    vapour = adjusted.y;
    mist = adjusted.z;
    // The drop takes the cell's vapour with it, mist and vapour making up a
    // cell of water between them; a snowflake, half that (a cell of snow is
    // half a cell of water; see formed), keeping any more (FROZEN_EXTRA). The
    // rest of its gas (its dry gas, and vapour past a drop's) is pushed into
    // the gas around (airsource.wgsl), as anything in a gas's place pushes
    // its gas.
    let drop = select(WATER, SNOW, c.temp < 0.0);
    let share = elems[drop].rho / elems[WATER].rho;
    if (mist >= DROP_MIST * share && mist + vapour >= WATER_VAPOUR * share) {
        let more = mist + vapour - WATER_VAPOUR * share;
        // What of the drop was still vapour condenses into it, warming it.
        var d = makeCell(drop, n.temp);
        d.temp += vapourLatent() * max(WATER_VAPOUR * share - mist, 0.0) / elems[drop].heatCap;
        // A flake's water freezes too, giving off what melting it takes back
        // (its latentUp: none was, and each fog's snow melted lost it).
        if (drop == SNOW) {
            d.temp += elems[SNOW].latentUp / elems[SNOW].heatCap;
        }
        // (What a snowflake's life can't keep, past FROZEN_EXTRA's most, goes
        // with the rest as vapour.)
        var rest = Gas(more, 0.0, (*g).lost, (*g).amount);
        if (drop == SNOW) {
            d = withFrozenExtra(d, frozenExtra(0u, more));
            rest.vapour = max(more - f32(frozenExtraOf(d)) / FROZEN_EXTRA, 0.0);
        }
        pushGas(p.y * P.size.x + p.x, rest, n.temp);
        *g = noGas();
        return d;
    }
    (*g).vapour = dither16(vapour);
    (*g).mist = dither16(mist);
    // Misty, so saturated, against a surface no warmer than itself, mist
    // collects here however thin (Around): dew, and frost below freezing,
    // on a cold window or fog-wet ground. Read by the cells beside it next
    // step, so both sides of each exchange agree on it.
    if (c.t == AIR) {
        n.flags &= ~FL_DEW;
        if (mist > 0.01 && nb.coldest <= c.temp + DEW_SLACK) {
            n.flags |= FL_DEW;
        }
    }
    return n;
}

// x (none below 0) as it will be stored in an f16 (the gas plane), rounded
// at random (common.wgsl's stochastic16).
fn dither16(x: f32) -> f32 {
    if (x <= 0.0) {
        return 0.0;
    }
    return stochastic16(x, rnd());
}

// The share of its oxygen used up, as stored (an f16, rounded at random so
// that small changes add up on average instead of being lost).
fn oxygenLost(x: f32) -> f32 {
    return dither16(clamp(x, 0.0, 1.0));
}

// What is under a particle (above it, for rising gases): whether it blocks the
// particle, and how fast the blocker itself is moving out of the way.
struct Support {
    blocked: bool,
    speed: f32,
    body: bool,        // what holds it up is a free body's cell (loose structure)
    ride: vec2<f32>,   // whose velocity it is
    cup: bool,         // and it is in the body's cup (inCupOf)
}

fn support(x: u32, y: u32, c: Cell, up: bool) -> Support {
    let q = vec2<i32>(i32(x), i32(y)) + select(downStep(), -downStep(), up);
    if (!inGrid(q)) {
        return Support(true, 0.0, false, vec2<f32>(0.0), false);
    }
    let other = src[u32(q.y) * P.size.x + u32(q.x)];
    let o = elems[cellType(other)];
    if (canEnter(c.t, cellType(other), select(1, -1, up))) {
        return Support(false, 0.0, false, vec2<f32>(0.0), false);
    }
    let vy = alongDown(cellVel(other));
    var speed = max(select(vy, -vy, up), 0.0);
    // The same powder under it, falling and not resting on anything (see
    // onGround), gains its weight this step just as this one does; counting
    // that, a clump in mid-air falls as one, instead of each cell lagging
    // further and further behind the one under it and the clump drawing out
    // into streaks. Once it's moving, or from rest where open space shows
    // under the clump (inMidAir): a stack merely waiting to learn it's on the
    // ground (that climbs a cell per step past GROUND_LOOK) would otherwise
    // pick up speed a row at a time and look like it was falling. (From rest
    // too only since REWRITE_PLAN.md 9-4: a block of sand let go set off a
    // row a step, its top 47 cells behind its front by step 24.)
    let resting = cellHas(other, FL_GROUNDED);
    if (!up && o.kind == KIND_POWDER && cellType(other) == c.t && !resting
        && (speed >= SETTLED_SPEED || inMidAir(x, y, c.t))) {
        speed += max(o.weight, 0.0) * pullShare(localPull);
    }
    let body = !up && isStructural(o) && cellHas(other, FL_LOOSE);
    return Support(true, speed, body, cellVel(other), body && inCupOf(x, y, other));
}

const GROUND_LOOK: u32 = 16u;  // cells of its own kind looked down through for what it rests on

// Whether the cell at (x, y), on a cell of a free body, is in the body's
// cup (bodies.wgsl's CUP): one of the body's own cells (frac's x, its slot)
// along its row on both sides, within CUP_LOOK. The body carries what is in
// its cup as its cells are carried (its stamp, REWRITE_PLAN.md P3-5), so
// that gripping its floor, it doesn't move along at the body's speed as
// well.
const CUP_LOOK: i32 = 64;
fn inCupOf(x: u32, y: u32, body: vec4<u32>) -> bool {
    let slot = unpack2x16float(body.w).x;
    let side = turnStep(downStep(), 2);
    var walls = 0u;
    for (var k = 0u; k < 2u; k++) {
        let d = select(side, -side, k == 1u);
        var q = vec2<i32>(i32(x), i32(y));
        for (var n = 0; n < CUP_LOOK; n++) {
            q += d;
            if (!inGrid(q)) {
                break;
            }
            let v = src[u32(q.y) * P.size.x + u32(q.x)];
            let o = elems[cellType(v)];
            if (isStructural(o) && cellHas(v, FL_LOOSE)) {
                walls += select(0u, 1u, unpack2x16float(v.w).x == slot);
                break;
            }
            if (o.kind == KIND_SOLID) {
                break;
            }
        }
    }
    return walls == 2u;
}

// Whether, looking down through its own kind as onGround does, the cell at
// (x, y) sees open space under its clump within GROUND_LOOK, nothing it
// rests on on the way (support: a clump in mid-air from rest).
fn inMidAir(x: u32, y: u32, t: u32) -> bool {
    var q = vec2<i32>(i32(x), i32(y));
    for (var d = 1u; d <= GROUND_LOOK; d++) {
        q += downStep();
        if (!inGrid(q)) {
            return false;
        }
        let v = src[u32(q.y) * P.size.x + u32(q.x)];
        if (cellType(v) != t) {
            return elems[cellType(v)].kind == KIND_GAS;
        }
        if (cellHas(v, FL_GROUNDED)) {
            return false;
        }
    }
    return false;
}

// Whether the powder t at (x, y), blocked by what is under it, rests on
// something rather than falling with it: on a solid or the world's floor, on
// some other material (whatever that is doing), or on its own kind that rests
// on something in turn. Looking down through its own kind, a cell that knew
// it rested on something last step (FL_GROUNDED) settles it; open space (a
// clump in mid-air, each cell resting on the one under it) means it doesn't,
// so the clump falls as one instead of spreading out as though it had landed.
// Past GROUND_LOOK cells the flag carries it on up, a step at a time. On a
// free body (FL_LOOSE) it rides along instead: resting, it would stand still
// in the body's way (bodies.wgsl), and a tower would hang in mid-air on the
// shards in its own broken windows.
fn onGround(x: u32, y: u32, t: u32) -> bool {
    var q = vec2<i32>(i32(x), i32(y));
    for (var d = 1u; d <= GROUND_LOOK; d++) {
        q += downStep();
        if (!inGrid(q)) {
            return true;
        }
        let v = src[u32(q.y) * P.size.x + u32(q.x)];
        if (cellType(v) != t) {
            let o = elems[cellType(v)];
            return o.kind != KIND_GAS && !(isStructural(o) && cellHas(v, FL_LOOSE));
        }
        if (cellHas(v, FL_GROUNDED)) {
            return true;
        }
    }
    return false;
}

// What the liquid around a powder cell does to it, from the four cells beside
// it: the pressure gradient across it, which floats it, and the velocity of
// the liquid there, which drags it along.
struct Wet {
    grad: vec2<f32>,
    vel: vec2<f32>,
    count: f32,      // liquid cells beside it; none, and it isn't wet
    density: f32,    // theirs on average
    viscosity: f32,  // theirs on average, cells² a step (no less than water's)
}

// Pressure at q, and whether there is one to read: the liquid's as solved in
// fluid.wgsl, none in open gas, unknown in a solid or powder.
fn pressureAt(q: vec2<i32>) -> vec2<f32> {
    if (any(q < vec2<i32>(0)) || any(q >= vec2<i32>(P.size))) {
        return vec2<f32>(0.0);
    }
    let i = u32(q.y) * P.size.x + u32(q.x);
    let f = fluid[2u * i];
    if (f.z > 0.0) {
        return vec2<f32>(f.x, 1.0);
    }
    return vec2<f32>(0.0, select(0.0, 1.0, elems[wordType(src[i].x)].kind == KIND_GAS));
}

// Pressure difference per cell between b and a, or `fallback` without both.
fn gradient(a: vec2<f32>, b: vec2<f32>, fallback: f32) -> f32 {
    return select(fallback, (b.x - a.x) / 2.0, a.y > 0.0 && b.y > 0.0);
}

// Whether the powder at p is under liquid: right under it (or under a gap in
// it that the fluid pass counted as liquid), or under powder that was under
// liquid as of last step (FL_WET), so a clump at the bottom of a pond is
// soaked through a cell further each step.
fn submerged(p: vec2<i32>) -> bool {
    let q = p - downStep();
    if (!inGrid(q)) {
        return false;
    }
    let i = u32(q.y) * P.size.x + u32(q.x);
    let v = src[i];
    let k = elems[cellType(v)].kind;
    return k == KIND_LIQUID || liquidSide(fluid[2u * i], fluid[2u * i + 1u]) > 0.0 || (k == KIND_POWDER && cellHas(v, FL_WET));
}

// The speed of the liquid half a cell over a bed of powder e's grains that
// lifts them off it, cells a step: Shields' criterion, the bed's shear
// stress over the grains' submerged weight a unit of area past a critical
// value, θ = u*² / (R g D) > θc, by Soulsby & Whitehouse's fit of θc to the
// grains' dimensionless size D* = D (R g / ν²)^(1/3) (Soulsby, Dynamics of
// Marine Sands, 1997, eq. 77), and the flow over the bed by the law of the
// wall over grains (u = u* ln(30 z / ks) / κ, Nikuradse's ks = 2.5 D) half a
// cell up, where the cell beside it has it. R is the grains' density over
// the liquid's less 1, g the pull, ν the liquid's (cells² a step); D is the
// grains' size (grain, mm). Medium sand (0.5 mm) in water at 1 cm cells is
// lifted by 0.18 m/s, fine gravel (6 mm) by 0.41, as Hjulström's curve has
// them; the same in cells and steps at any scale but for the grains, which
// don't grow with the cells.
fn shieldsSpeed(e: Elem, liquid: f32, nu: f32, g: f32) -> f32 {
    let r = max(e.density / 100.0 / liquid - 1.0, 0.0);
    let d = e.grain * 1e-3 / P.dx;
    let dStar = d * pow(r * g / (nu * nu), 1.0 / 3.0);
    let theta = 0.3 / (1.0 + 1.2 * dStar) + 0.055 * (1.0 - exp(-0.02 * dStar));
    let uStar = sqrt(theta * r * g * d);
    return uStar * max(log(30.0 * 0.5 / (2.5 * d)), 1.0) / KARMAN;
}

// Whether the cell two over p (up the pull) is liquid: air a cell under the
// liquid's top, under a crest a cell thick, is a dent in the surface, not a
// bubble (REWRITE_PLAN.md 7-9).
fn deepUnder(p: vec2<i32>) -> bool {
    let q = p - 2 * downStep();
    if (!inGrid(q)) {
        return false;
    }
    return elems[cellType(src[u32(q.y) * P.size.x + u32(q.x)])].kind == KIND_LIQUID;
}

fn wetness(p: vec2<i32>, soaked: bool) -> Wet {
    var w: Wet;
    var density = 0.0;
    for (var d = 0; d < 4; d++) {
        let q = p + select(vec2<i32>(0, (d & 1) * 2 - 1), vec2<i32>((d & 1) * 2 - 1, 0), d < 2);
        if (any(q < vec2<i32>(0)) || any(q >= vec2<i32>(P.size))) {
            continue;
        }
        let v = src[u32(q.y) * P.size.x + u32(q.x)];
        let e = elems[cellType(v)];
        if (e.kind == KIND_LIQUID) {
            w.vel += cellVel(v);
            density += e.density / 100.0;
            let acts = effTempIn(cellType(v), cellTemp(v), phaseOf(cellType(v), wordSolute(v.x), wordStrength(v.x), 0.0));
            w.viscosity += max(viscosityAt(e, acts), WATER_NU * P.dt / (P.dx * P.dx));
            w.count += 1.0;
        }
    }
    if (w.count == 0.0) {
        // Inside a soaked clump the water in between the grains floats each
        // of them (taken to be water), and the clump rises or sinks as one.
        if (soaked) {
            w.grad = localPull;
        }
        return w;
    }
    w.vel /= w.count;
    w.viscosity /= w.count;
    w.density = density * 100.0 / w.count;
    // Where it rests on (or under) more powder there is no pressure to read
    // on that side; the liquid's own weight (under the pull here) stands in,
    // as if it were still.
    let fallback = density / w.count * localPull;
    let still = length(fallback);
    let grad = vec2<f32>(
        gradient(pressureAt(p - vec2<i32>(1, 0)), pressureAt(p + vec2<i32>(1, 0)), fallback.x),
        gradient(pressureAt(p - vec2<i32>(0, 1)), pressureAt(p + vec2<i32>(0, 1)), fallback.y),
    );
    let most = MAX_WET_PUSH * still;
    let size = length(grad);
    w.grad = select(grad, grad * (most / size), size > most);
    return w;
}

// What the air's pressure bears on the cell at p: the difference between
// the pressures on its two sides, left and right or above and below,
// whichever is more. On a side, the pressure of what is there as its air
// cell has it (air.wgsl): gas, or liquid or powder, which pass it on; a
// solid (more of what bears it) or the world's edge, the open air's at
// that height. So a pressure cooker's wall bears its gauge pressure, and a
// face a blast hits its overpressure, whatever is behind it; a girder or a
// wall between two rooms at one pressure (a blast all round it) bears none.
// (What a whole standing wall bears, by its thickness and span, is the
// stress solve's: stress.wgsl. This breaks what is loose, and what isn't
// structure.)
fn loadAcross(p: vec2<i32>) -> f32 {
    let dims = airDims(P.size);
    let open = (openAirAtm(altitude(vec2<u32>(p))) - 1.0) * P_ATM;
    var load = 0.0;
    for (var axis = 0; axis < 2; axis++) {
        var side = vec2<f32>(open);
        for (var s = 0; s < 2; s++) {
            let step = s * 2 - 1;
            let q = p + select(vec2<i32>(0, step), vec2<i32>(step, 0), axis == 0);
            if (inGrid(q) && elems[wordType(src[u32(q.y) * P.size.x + u32(q.x)].x)].kind != KIND_SOLID) {
                side[s] = airAt(q / i32(AIR_CELL), dims).x;
            }
        }
        load = max(load, abs(side.x - side.y));
    }
    return load;
}

// Last step's pile load on the powder at p; none outside the world.
fn loadAt(p: vec2<i32>) -> f32 {
    if (any(p < vec2<i32>(0)) || any(p >= vec2<i32>(P.size))) {
        return 0.0;
    }
    return loadIn[u32(p.y) * P.size.x + u32(p.x)];
}

// What piled powder presses on the solid at p with: the whole load of a pile
// on top of it, or LATERAL of one beside it (a deep pile presses outward on
// its walls). LATERAL is the at-rest earth pressure coefficient 1 - sin φ
// (Jáky) for sand's angle of friction, φ ≈ 35°.
fn pileLoad(p: vec2<i32>) -> f32 {
    let side = max(loadAt(p + downTurned(2)), loadAt(p + downTurned(-2)));
    return max(loadAt(p - downStep()), side * LATERAL);
}

// A solid breaking into its fragments; they start out falling, along with the
// body they broke off if it was moving (a free body's vel is its own).
fn crumble(c: Cell, e: Elem) -> Cell {
    var f = convert(c, e.breakTo);
    f.vel = withAlongDown(c.vel, alongDown(c.vel) + CRUMBLE_SPEED);
    // What isn't a substance's form (wood into sawdust) keeps its heat
    // (from AMBIENT_T) cell for cell; a substance's forms do gram for gram
    // (formed).
    if (e.fill <= 0.0 || elems[f.t].fill <= 0.0) {
        f.temp = AMBIENT_T + e.heatCap * (c.temp - AMBIENT_T) / elems[f.t].heatCap;
    }
    return formed(f, c);
}

// Marks the tile holding p for the body passes and lists it, once (reading
// the mark first: most loose cells find their tile marked already, and a
// read is much cheaper than an atomic exchange).
fn markLoose(p: vec2<u32>) {
    let tile = bodyTileOf(p, P.size);
    if (atomicLoad(&bodies[tile]) == 0u && atomicExchange(&bodies[tile], 1u) == 0u) {
        let n = atomicAdd(&bodies[bodyMarkedSlot(P.size)], 1u);
        atomicStore(&bodies[bodyMarkListSlot(P.size) + n], tile);
    }
}

// Phase change. Something that solidifies while flying or falling becomes the
// solid's fragments instead (lava -> gravel, freezing rain -> snow), so no
// solid ends up hanging in mid-air.
fn transition(c: Cell, t: u32) -> Cell {
    let e = elems[t];
    if (e.kind == KIND_SOLID && e.breakTo != NO_ELEM && abs(alongDown(c.vel)) > FLYING_SPEED) {
        return convert(c, e.breakTo);
    }
    var n = convert(c, t);
    // A solid that forms where it is (lava setting into stone, water
    // freezing) starts with no support of its own, as something built does:
    // whether it stands is the anchoring's to say (bodies.wgsl), joined to
    // what holds it or not (a crust over water joined to the bank stands, if
    // its strength bears it, stress.wgsl; one that isn't comes loose: stone
    // sinks, ice floats).
    if (isStructural(e) && !isStructural(elems[c.t])) {
        n.frac.x = 0.0;
    }
    return n;
}

// The grams in a cell of type b over those in one of a (Elem.fill), for heat
// kept gram for gram across a change of form; 1 where either has none.
fn gramsTo(a: u32, b: u32) -> f32 {
    let ga = elems[a].fill;
    let gb = elems[b].fill;
    return select(1.0, gb / ga, ga > 0.0 && gb > 0.0);
}

// What a cell of t holds once it has changed phase into gas (the gas
// plane): water boils into vapour, a cell of water's WATER_VAPOUR units in
// the air of its cell (8 atm there until it has spread, air.wgsl's
// expansion); anything else (fluff charring into smoke) into gas made of it
// where it changes, its oxygen used up (madeGas).
fn phaseGas(t: u32) -> Gas {
    if (isWater(t)) {
        return Gas(WATER_VAPOUR * gramsTo(WATER, t), 0.0, 0.0, 0.0);
    }
    return madeGas(elems[t].hiT, 1.0);
}

// Latent heat. Past hiT a cell holds at hiT (effTempIn) while the heat it
// takes in builds up in its stored temperature; once that adds up to
// latentUp it becomes hiTo, and whatever is left over warms the new phase,
// gram for gram (a cell of the new form holds gramsTo of the old, and as
// many more or fewer cells of it are made, formed). Cooling past loT works
// the same way with latentDown. A liquid that leaves a residue (mud, brine)
// boils off only into open air (pairs.wgsl), so once it has taken in all of
// its latent heat it holds, its stored temperature running on with the heat
// it takes in, which goes with its vapour when it boils off. Brine can't
// freeze with its salt in it: it passes it on to water beside it first
// (pairs.wgsl), and holds, freezing on as it cools; past the coldest brine
// can be (saturated), its salt crystallises out. Water boils into vapour in
// the air in its cell (phaseGas), whose heat capacity takes what is left
// over; the air's moisture condenses as mist (moistAir), not by a phase
// change. Phase points are at the air pressure and for what water carries
// (lnAtm; see phaseOf).
fn changePhase(c: Cell, e: Elem, lnAtm: f32) -> Cell {
    let tr = phaseOf(c.t, soluteOf(c), strengthOf(c), lnAtm);
    if (e.hiTo != NO_ELEM && c.temp > tr.x) {
        if (leavesResidue(c, e)) {
            return c;
        }
        let spare = (c.temp - tr.x) * e.heatCap - e.latentUp;
        if (spare >= 0.0) {
            var n = transition(c, e.hiTo);
            n.temp = tr.x + spare * gramsTo(c.t, n.t) / heatCapOf(n.t, phaseGas(c.t));
            return formed(n, c);
        }
    } else if (e.loTo != NO_ELEM && c.temp < tr.y) {
        let spare = (tr.y - c.temp) * e.heatCap - e.latentDown;
        if (spare >= 0.0) {
            if (holdsSolute(c.t) && soluteOf(c) != 0u) {
                let s = elems[soluteOf(c)];
                if (effTempIn(c.t, c.temp, tr) > e.loT + s.solFreeze) {
                    return c;
                }
                // Brine past its coldest freezing point freezes whole: its
                // salt crystallises and its water freezes, a cell one or the
                // other. A whole grain of salt as often as the cell holds a
                // grain of it (residueOf's share), and ice holding all its
                // water the rest of the time, made up for the times it was
                // salt (formedTimes): both kept on the whole. (It was all
                // salt, its water gone: a tank of brine frozen and thawed
                // came back a heap of salt.)
                if (s.solResidue == soluteOf(c)) {
                    let salt = min(f32(strengthOf(c)) / 255.0 * s.solubility * liquidAmount(c), 0.99);
                    if (rnd() < salt) {
                        return withDissolved(makeCell(soluteOf(c), effTempIn(c.t, c.temp, tr)), 0u);
                    }
                    var n = transition(c, e.loTo);
                    n.temp = tr.y - spare * gramsTo(c.t, n.t) / elems[n.t].heatCap;
                    return formedTimes(n, c, 1.0 / (1.0 - salt));
                }
            }
            var n = transition(c, e.loTo);
            n.temp = tr.y - spare * gramsTo(c.t, n.t) / elems[n.t].heatCap;
            return formed(n, c);
        }
    }
    return c;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < P.size.x && gid.y < P.size.y) {
        updateCell(gid);
    }
}

fn updateCell(gid: vec3<u32>) {
    localPull = pullAt(vec2<f32>(gid.xy) + 0.5);
    localDown = downAt(vec2<i32>(gid.xy));
    // A step's length is 1 or √2.
    localDir = vec2<f32>(localDown) * select(1.0, 0.70710678, localDown.x != 0 && localDown.y != 0);
    let w = P.size.x;
    let h = P.size.y;
    let i = gid.y * w + gid.x;
    seedRng(i, P.frame, P.seed);
    var c = unpackCell(src[i]);
    let was = c.t;
    let e = elems[c.t];
    let pos = vec2<f32>(gid.xy) + 0.5;
    let a = sampleAir(pos);
    // What it holds of the air, as the wind has carried it (carry.wgsl) and
    // as it goes (the gas plane).
    var g = gasSrc(i);

    // Heat moves as energy; each cell's temperature changes by what it gains
    // over its heat capacity.
    let lnAtm = atmLog(a.p);
    let temp = effTempIn(c.t, c.temp, phaseOf(c.t, soluteOf(c), strengthOf(c), lnAtm));
    let heatCap = cellHeatCap(c, g);
    let real = realnessAt(gid.xy);
    let tau = clockAt(real);
    let nb = around(vec2<i32>(gid.xy), c, e, temp, heatCap, lnAtm, real, tau, g.lost, g.amount);
    // Water and damp soil give up the latent heat of the vapour they give off
    // (vapourLatent), which the air gets back as it condenses; for air (any
    // gas that holds vapour), nb.evaporation is the vapour it takes in.
    let latent = select(vapourLatent() * nb.evaporation, 0.0, holdsVapour(c.t));
    // And the heat other passes put into it as energy since (heatin.wgsl).
    // A vacuum (a cell of gas holding nothing) has no heat capacity to take
    // any in, and trades none: what was put in waits for gas to come in.
    let deposited = exactValue(vec2<u32>(heatIn[2u * i], heatIn[2u * i + 1u]), EXACT_FINE);
    // And its room's air's mixing over heat's step (mixOf, 4e-4).
    let energy = nb.energy + outsideHeat(gid.xy, e, heatCap, temp, tau, real)
        + radiantHeat(gid.xy, c, e, temp, heatCap, real, tau, g) - latent + deposited
        + mixHeatOf(gid.xy, c.t, real, heatCap, temp);
    let outside = outdoors(gid.xy);
    if (heatCap > 0.0) {
        c.temp += energy / heatCap;
        heatIn[2u * i] = 0u;
        heatIn[2u * i + 1u] = 0u;
    }
    if ((e.flags & F_FIXED_TEMP) != 0u) {
        c.temp = e.fixedT;
    }

    // Oxygen: flames, air and smoke even out what they have lost between
    // them; burning things beside them draw on it, a flame uses up its own,
    // and where the wind gets to under open sky the outside air makes it
    // good.
    let vented = outside && ventilated(gid.xy);
    if (carriesOxygen(c.t)) {
        var lost = g.lost + OXYGEN_SPREAD * nb.deficit;
        if (vented) {
            lost -= lost * SKY_OXYGEN;
        }
        g.lost = oxygenLost(lost);
    }

    // Moisture: water evaporates into the air beside it, the air holds what
    // its temperature lets it and the rest condenses into mist. A gap in the
    // liquid, which the liquid and the air count as liquid (common.wgsl's
    // flowsThrough), holds no gas: what it holds goes into the air
    // (airsource.wgsl; up through the liquid, carry.wgsl).
    if (holdsVapour(c.t) && isGap(i)) {
        pushGas(i, g, c.temp);
        g = noGas();
    } else if (holdsVapour(c.t)) {
        c = moistAir(c, &g, nb, outside, gid.xy, tau, exp(lnAtm));
    } else if (holdsWater(c.t) && evaporated(&c, nb.evaporation)) {
        // The last of it: its vapour has gone into the air beside it, leaving
        // whatever it carried (residueOf), at the temperature the water was
        // (not its stored one, which for brine held at its boiling point
        // counts the latent heat taken in: 640 °C salt).
        c = residueOf(c, e, temp);
    } else if (holdsSoilWater(c.t) && c.t == was) {
        // Damp soil dries by what it gave off.
        let dried = floor(nb.evaporation * (WATER_UNITS / WATER_VAPOUR) + rnd());
        c = withSoilWater(c, u32(max(f32(soilWaterOf(c)) - dried, 0.0)));
    } else if (holdsPlantWater(c.t)) {
        // A leaf loses the water it gave off.
        c = withPlantWater(c, u32(max(f32(plantWaterOf(c)) - floor(nb.evaporation / PLANT_UNIT + rnd()), 0.0)));
    }
    // What it holds of the air (the gas plane) is settled now, written so as
    // not to be kept through the rest (the pass is near the most registers
    // it can have without slowing down); the few changes below write it
    // again. A flame's starving is all of it that is wanted below. Anything
    // but a gas holds none, so the last of some water drying up (residueOf)
    // leaves a cell of nothing, which the air around fills.
    if (!holdsGas(was)) {
        g = noGas();
    }
    setGasDst(i, g);
    // A cell of gas holding nothing has no heat to have a temperature by: it
    // is as warm as what is beside it (as the light, people and the views
    // find it), until gas comes in and brings its own (carry.wgsl).
    if (gasHeat(c.t) && gasHeatCap(g) <= 0.0 && nb.warmth.y > 0.0) {
        c.temp = nb.warmth.x / nb.warmth.y;
    }
    // A flame (FIRE) shows burning gas: in gas that couldn't burn, or isn't
    // hot enough to glow, it dies down fast.
    let starved = !holdsFlame(c.t, g, c.temp) || c.temp < FLAME_T;

    if (holdsCharge(c.t) && chargeOf(c) > 0u) {
        c = withCharge(c, chargeOf(c) - 1u);
    }

    // It lights, or chars, hot through or where it touches something hot
    // (contactTemp), or where a flame touches it (block.wgsl). Alight, it
    // burns while gas beside it holds a flame that can burn some of it
    // (oxygenToBurn) and nothing wet is against it (a wet surface stays at
    // about 100 °C, however hot the cell is through). The burning itself, the
    // oxygen it takes and the heat it gives that gas, goes face by face in the
    // pairs pass (pairs.wgsl), which counts what of it has burnt in its life
    // and turns it into burnTo when it has all gone. Smothered or doused, it
    // goes out but stays as hot as it is, and catches again if, still past
    // igniteT, it gets air to burn in: a smouldering room flaring up when the
    // door opens.
    let hottest = max(effTempIn(c.t, c.temp, phaseOf(c.t, soluteOf(c), strengthOf(c), lnAtm)), nb.contact);
    let flaming = nb.flame && !nb.wet;
    if (e.blast > 0.0 && ((c.flags & FL_BURN) != 0u || hottest > e.igniteT)) {
        // An explosive lit (by a flame, a spark, a hard knock: block.wgsl,
        // fluid.wgsl) or past igniteT goes off: its cell the gas it makes
        // (common.wgsl's detonation).
        let d = detonation(c, e);
        c = d.c;
        setGasDst(i, d.g);
    } else if ((c.flags & FL_BURN) != 0u) {
        if (!flaming) {
            c.flags &= ~FL_BURN;
            // A plant's or seed's life counted what of it burnt; it doesn't
            // grow on that as a budget (its sugar, its store).
            if (holdsSugar(c.t)) {
                c = withSugar(c, 0u);
            } else if (holdsStore(c.t)) {
                c = withStore(c, 0u);
            }
        }
    } else if (hottest > e.igniteT && e.burnTime > 0.0 && flaming) {
        c.flags |= FL_BURN;
        // Catching again after being smothered, it goes on from what of it
        // had burnt; a plant's or seed's growth budget goes up in smoke.
        if (holdsSugar(c.t) || holdsStore(c.t)) {
            c = withBurnt(c, 0u);
        }
    } else if (e.charTo != NO_ELEM && !nb.wet && rnd() < chanceOver(FIRE_PACE * pyrolysisRate(hottest), tau)) {
        // Not alight, it chars (pyrolysisRate; not under water, which keeps
        // its surface at about 100 °C): its char by its yield, or the gas its
        // volatiles make, unburnt, as smoke. Either holds all its heat, on
        // average what it held (each as likely as its share).
        let heat = e.heatCap * (c.temp - AMBIENT_T);
        if (rnd() < e.charYield) {
            c = convert(c, e.charTo);
            c.temp = AMBIENT_T + heat / elems[c.t].heatCap;
        } else {
            let volatiles = Gas(0.0, 0.0, 1.0, e.charGas);
            c = convert(c, SMOKE);
            c.temp = AMBIENT_T + heat / gasHeatCap(volatiles);
            setGasDst(i, volatiles);
        }
    }

    // What the air's pressure across it breaks (standing structure takes it
    // as stress instead: stress.wgsl).
    let breakP = elems[c.t].breakP;
    let standing = isStructural(elems[c.t]) && (c.flags & FL_LOOSE) == 0u;
    if (!standing && breakP < NEVER && (elems[c.t].flags & F_INDESTRUCTIBLE) == 0u && loadAcross(vec2<i32>(gid.xy)) > breakP) {
        c = crumble(c, elems[c.t]);
    }

    // Structure past its softening point: unstressed there from now on, as
    // it cools (stress.wgsl's thermal strain; thermal shock is that strain
    // against what holds it, REWRITE_PLAN.md 8-5).
    if (isStructural(elems[c.t]) && c.temp > softeningT(elems[c.t])) {
        c.flags |= FL_SOFTENED;
    }

    // Frost: ice still owing cells (formed) with no room beside it to grow
    // them into presses on what holds it in (block.wgsl).
    if (c.t == ICE) {
        c.flags = select(c.flags & ~FL_STRESSED, c.flags | FL_STRESSED, owedOf(c) > 0u && !nb.room);
    }

    if ((c.flags & FL_BURN) == 0u) {
        let before = c.t;
        let taken = select(1.0, liquidAmount(c), elems[c.t].kind == KIND_LIQUID);
        c = changePhase(c, elems[c.t], lnAtm);
        // Boiled or charred into gas, it holds what it became (phaseGas), of
        // as much as it was (a cell of liquid holding more or less than a
        // cell's worth, liquidAmount). (A form shrinking leaves a cell of air too, formed,
        // but never into what it boils into: water, the one that boils into
        // air, grows as it freezes.)
        if (!holdsGas(before) && holdsGas(c.t) && c.t == elems[before].hiTo) {
            let g = phaseGas(before);
            setGasDst(i, Gas(g.vapour * taken, g.mist * taken, g.lost, g.amount * taken));
        }
    }

    switch c.t {
        case FIRE: {
            if (rnd() < 0.65) {
                // Starved, a flame dies down fast. What it leaves is the air it
                // burned in, which it held all along (the gas plane): its
                // moisture, and its oxygen less what it burned.
                let fade = select(1u, 4u, starved);
                if (lifetimeOf(c) >= fade) {
                    c = withLifetime(c, lifetimeOf(c) - fade);
                } else {
                    c = convert(c, select(AIR, SMOKE, rnd() < 0.15));
                }
            }
        }
        case SMOKE: {
            if (rnd() < 0.5) {
                if (lifetimeOf(c) > 0u) {
                    c = withLifetime(c, lifetimeOf(c) - 1u);
                } else {
                    // As a flame dying out leaves it (FIRE).
                    c = convert(c, AIR);
                }
            }
        }
        case BLAST: {
            // The gas an explosive made, shown for the few steps it is
            // burning hot (block.wgsl spreads the show through gas as hot),
            // then the gas it is, some of it smoke, as a flame dying leaves;
            // cooler than FIREBALL_T at once.
            if (lifetimeOf(c) > 0u && c.temp >= FIREBALL_T) {
                c = withLifetime(c, lifetimeOf(c) - 1u);
            } else {
                c = convert(c, select(AIR, SMOKE, rnd() < 0.15));
            }
        }
        case HYDROGEN: {
            if (vented && rnd() < HYDROGEN_ESCAPE) {
                c = convert(c, AIR);
            }
        }
        case PLANT: {
            if ((c.flags & FL_BURN) == 0u) {
                c = plantNeeds(gid.xy, c, c.temp);
            }
        }
        case SEED: {
            // Only once it has landed: a seed sprouting in mid-air would grow
            // a plant with nothing under it. Where it can drink it sprouts, in
            // the dark too, into a seedling with the sugar the seed stored
            // (for 10-25 cells of growth) and no water yet, which its roots
            // then drink.
            if ((c.flags & FL_BURN) == 0u && rnd() < SPROUT_CHANCE && length(c.vel) < SETTLED_SPEED
                && canDrink(vec2<i32>(gid.xy))) {
                let store = min(storeOf(c) * GROW_SUGAR, 255u);
                c = withSugar(convert(c, PLANT), store);
            }
        }
        default: {}
    }

    // Structural solids keep their support in frac.x (and their piece in
    // frac.y). Standing, what holds them up is their being joined to an
    // anchor (bodies.wgsl's anchoring, which runs out the support of a
    // piece nothing holds, at once) and what breaks them is the stress solve
    // (stress.wgsl, which marks the step it found one past its strength):
    // it crumbles into its fragments, which then fall. A piece let go comes
    // loose, and the body passes (bodies.wgsl) take over: it falls (floats,
    // drifts) as one body (FL_LOOSE) until it comes to rest on something
    // again. Cells of a body that broke up are loose too until their pieces
    // are bodies of their own; loose, a pile on them crushes them past their
    // crushP, as nothing works out their stress.
    let se = elems[c.t];
    if (isStructural(se)) {
        if ((c.flags & FL_LOOSE) != 0u) {
            if (pileLoad(vec2<i32>(gid.xy)) > se.crushP) {
                c = crumble(c, se);
            } else if (c.frac.x == FREE_BODY) {
                markLoose(gid.xy);
            }
        } else if (atomicLoad(&bodies[structureBreakSlot(P.size, i)]) == P.frame) {
            c = crumble(c, se);
        } else {
            // (Painted in a stroke still being drawn, it is held.)
            let letGo = c.frac.x <= -0.5 * P.supportMax;
            let held = c.frac.x == SUPPORT_HELD && P.holding != 0u;
            c.frac = vec2<f32>(select(P.supportMax, c.frac.x, letGo || held), c.frac.y);
            if (letGo) {
                markLoose(gid.xy);
            }
        }
    }

    // Velocity. Air itself is moved by the air grid, and liquid as particles
    // (fluid.wgsl), whose velocity its cell only shows. A solid only moves
    // with a free body, which keeps its own. Another gas's kind (smoke, a
    // flame) is a marker riding the air's flow (REWRITE_PLAN.md 4c): it moves
    // at the wind's velocity, give or take its turbulence (spread), a whole
    // cell at a time (block.wgsl), what it holds staying with the air
    // (carry.wgsl); what lifts it (a hot flame, light hydrogen) lifts the
    // air (air.wgsl's buoyancy).
    let me = elems[c.t];
    if (me.kind == KIND_SOLID) {
        if ((c.flags & FL_LOOSE) == 0u) {
            c.vel = vec2<f32>(0.0);
        }
    } else if (c.t == AIR && !(submerged(vec2<i32>(gid.xy)) && wetness(vec2<i32>(gid.xy), false).count >= 2.0
        && deepUnder(vec2<i32>(gid.xy)))) {
        // (Air under liquid with liquid round it is a bubble, and rises as
        // one, below: REWRITE_PLAN.md 7-9.)
        c.vel = vec2<f32>(0.0);
    } else if (me.kind == KIND_GAS) {
        var v = a.vel;
        // Under liquid, a bubble: it rises through the liquid around it
        // (trading places with it, block.wgsl) at a cell-wide bubble's
        // speed. Under soaked powder or a wet body with no liquid beside it,
        // there is none to rise through (and no viscosity to creep by: its
        // speed came out NaN), so it goes with the air.
        if (submerged(vec2<i32>(gid.xy))) {
            let wet = wetness(vec2<i32>(gid.xy), false);
            if (wet.count > 0.0) {
                let pull = length(localPull);
                let cap = BUBBLE_RISE * sqrt(pull);
                let creep = pull / (12.0 * wet.viscosity);
                v = wet.vel - downDir() * cap * creep / max(sqrt(cap * cap + creep * creep), 1e-12);
            }
        }
        v = withAlongSide(v, alongSide(v) + (rnd() - 0.5) * me.spread);
        let speed = length(v);
        c.vel = select(v, v * (MAX_SPEED / speed), speed > MAX_SPEED);
    } else if (me.kind != KIND_LIQUID) {
        var v = c.vel;
        var entrained = false;
        let weight = me.weight;
        // The total pull, the Earth's and the world's masses', as its weight
        // is of g: sand falls toward a black hole.
        v += localPull * (weight / elems[WATER].weight);
        // Drag between it and the air beside it, worked out once for the
        // two (REWRITE_PLAN.md 4e-2): their relative velocity w goes to
        // w / (1 + k (1 + μ)), k its element's rate and μ its mass over the
        // air's beside it, each taking its share of the change by the other's
        // mass, so momentum is kept. A cell of powder is a cell of it at its
        // bulk density, hundreds of times the air's: the air beside it goes
        // along with it more than it is held back. One resting on the ground
        // is part of it.
        let beside = windBeside(vec2<i32>(gid.xy));
        let air = elems[AIR].rho * airBeside(vec2<i32>(gid.xy));
        if (beside.z > 0.0 && air > 0.0 && (c.flags & FL_GROUNDED) == 0u) {
            let mu = me.rho / air;
            let w = v - beside.xy;
            let change = w * (1.0 - 1.0 / (1.0 + me.drag * (1.0 + mu)));
            v -= change / (1.0 + mu);
            pushBeside(vec2<i32>(gid.xy), change * mu / (1.0 + mu));
        }
        v += pressurePush(a.grad, me.rho);
        // In liquid, a powder is pushed by the liquid's pressure, which
        // floats it (fully, it sinks only as far as it is denser), and
        // dragged along by the liquid's flow.
        var buoyed = 1.0;  // the share of its weight that presses it down (coulomb)
        if (me.kind == KIND_POWDER) {
            let soaked = submerged(vec2<i32>(gid.xy));
            c.flags &= ~FL_WET;
            if (soaked) {
                c.flags |= FL_WET;
            }
            let wet = wetness(vec2<i32>(gid.xy), soaked);
            // The parcel (see FC_C1), its pores' liquid what is beside it, or
            // inside a soaked clump water.
            let liquid = select(1.0, wet.density / 100.0, wet.count > 0.0);
            let pile = me.rho / 1000.0;
            let parcel = pile + (1.0 - pile / (me.density / 100.0)) * liquid;
            v -= wet.grad / parcel;
            if (soaked && wet.count > 0.0) {
                buoyed = max(1.0 - liquid / parcel, 0.0);
            }
            if (wet.count > 0.0) {
                // Its drag, a cell across: D 1 in cells and steps, taken
                // implicitly (it closes at most the whole gap a step).
                let rel = wet.vel - v;
                let k = liquid / parcel * (FC_C1 * wet.viscosity + 0.75 * FC_C2 * length(rel));
                v += rel * (k / (1.0 + k));
                // Fast enough, the flow over a grain lifts it off the bed
                // and carries it (see shieldsSpeed).
                let flow = length(wet.vel - c.vel);
                let critical = shieldsSpeed(me, liquid, wet.viscosity, length(localPull));
                if (flow > critical) {
                    entrained = true;
                    v = withAlongDown(v, alongDown(v) - LIFT * (flow - critical));
                }
            }
        }
        // What it rests on doesn't hold it if it is on its way off it: a
        // grain of sawdust floating up from the bottom of a pond.
        let up = weight < 0.0;
        let falling = alongDown(v);
        let leaving = select((falling < -SETTLED_SPEED), (falling > SETTLED_SPEED), up);
        let s = support(gid.x, gid.y, c, up);
        let grounded = s.blocked && !leaving && onGround(gid.x, gid.y, c.t);
        c.flags &= ~FL_GROUNDED;
        if (grounded) {
            c.flags |= FL_GROUNDED;
        }
        if (s.blocked && !leaving) {
            // Can't fall faster than what it rests on, so falling clumps stay together.
            if (weight < 0.0) {
                v = withAlongDown(v, max(falling, -s.speed));
            } else {
                v = withAlongDown(v, min(falling, s.speed));
            }
            // What presses it down a step, its weight and the pile's on it
            // (loadAt in cells of water-dense weight over LOAD_ROWS of the
            // world's height), times its friction: its grip (coulomb).
            let pressing = max(dot(localPull, downDir()), 0.0);
            let piled = loadAt(vec2<i32>(gid.xy) - downStep()) * f32(h) / LOAD_ROWS / (me.rho / 1000.0);
            let grip = me.friction * pressing * (buoyed * max(weight, 0.0) / elems[WATER].weight + piled);
            if (s.speed < 0.05 && grounded) {
                // Truly resting: powders grip (unless the water is carrying
                // them off).
                c.frac = withAlongDown(c.frac, 0.0);
                if (!entrained) {
                    v = withAlongSide(v, coulomb(alongSide(v), 0.0, grip));
                }
            }
            // On a free body it rides along, gripping it as it would the
            // ground but moving with it (REWRITE_PLAN.md P3-4): a raft
            // carries its sand sideways too. The body carries its mass
            // (bodies.wgsl's lift). In the body's cup the body's stamp
            // carries it (P3-5), so it grips the body as it would the
            // ground, its velocity its own across the body's.
            if (s.body && !entrained) {
                v = withAlongSide(v, coulomb(alongSide(v), select(alongSide(s.ride), 0.0, s.cup), grip));
            }
        }
        let speed = length(v);
        c.vel = select(v, v * (MAX_SPEED / speed), speed > MAX_SPEED);
    }

    // A resting powder passes on the load of the resting powder on top of it
    // as of last step, plus its own weight, so a pile's weight builds up
    // downward a cell per step. Anything else carries none on.
    var load = 0.0;
    if (me.kind == KIND_POWDER && abs(alongDown(c.vel)) < SETTLED_SPEED) {
        load = me.rho / 1000.0 * LOAD_ROWS / f32(h) + loadAt(vec2<i32>(gid.xy) - downStep());
    }
    loadOut[i] = load;

    // Nothing gets colder than absolute zero, however much the radiation
    // and conduction of a step overshoot.
    c.temp = clamp(c.temp, storedZero(c.t), MAX_STORED_T);
    dst[i] = packCell(c);

    // A cell holding gas that has turned into something else since its gas
    // was written (a fan broken into scrap) puts what it held into the air
    // beside it (airsource.wgsl), and holds none.
    if (holdsGas(was) && !holdsGas(c.t)) {
        pushGas(i, gasFrom(dst[gasIndex(w * h, i)], i), c.temp);
        setGasDst(i, noGas());
    }
}
