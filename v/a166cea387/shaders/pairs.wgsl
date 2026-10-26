// Pairs pass, once per step before the update pass: two neighbors settling
// something between them at once. Every cell is paired with one neighbor,
// along one axis, the axis and the pairing changing every step so each cell
// meets each of its four neighbors in turn; one invocation owns a pair and is
// the only one touching its two cells, so whatever passes between them adds
// up exactly. So far that is what water carries (a solute: its element in
// aux, how much in life, 255 being saturated; see elements.js): salt
// dissolving into water, brine and fresh water mixing, freezing water pushing
// out its salt, ice melting into brine or under salt, acid eating what it
// reacts with, and brine or mud boiling off into air and leaving the rest
// behind, the grains it leaves merging; and what burns (stage 5).
// And the water in soil (see SOIL_FULL): soaked up from water and mud,
// spreading through the grains and draining down, drunk by plants' roots.
// (In the update pass itself, all this code slows every other cell's work.)

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> elems: array<Elem>;
@group(0) @binding(2) var<storage, read_write> cells: array<vec4<u32>>;
// For the air pressure, which moves boiling points (phaseOf).
@group(0) @binding(3) var<storage, read> air: array<vec4<f32>>;
// The pull of the world's masses per light cell (gravity.wgsl), for down
// (common.wgsl).
@group(0) @binding(4) var gravity: texture_2d<f32>;
// What goes into the air (airsource.wgsl): the hydrogen saturated brine
// split by electrolysis makes (split).
@group(0) @binding(5) var<storage, read_write> source: array<atomic<u32>>;
// Heat's clock per light cell (clock.wgsl): how real-time a place is, which
// the rates of what passes between water and what it touches go by.
@group(0) @binding(6) var clock: texture_2d<f32>;

// The seconds a face's visit stands for, by heat's clock (stage 2b): a pair
// comes round every 4 steps, at the faster of its two cells' clocks.
fn visitTau(pa: vec2<u32>, pb: vec2<u32>) -> f32 {
    let real = max(textureLoad(clock, pa / 2u, 0).x, textureLoad(clock, pb / 2u, 0).x);
    return 4.0 * clockStep(real, P.dayDt, P.clockSpan);
}

const MELT_IN_BRINE: f32 = 0.05;  // chance per pairing that ice melts into brine colder than it can freeze at
// The chance per pairing that soaked dirt takes in the fresh water beside it
// as a cell of water more, turning to mud.
const SOAK_MUD: f32 = 0.02;
// A plant cell short of water drinks up to ROOT_TAKE units of it a pairing
// from the fresh water, mud or soil beside it, soil down to its wilting
// point (soilWiltLife: what is left the soil holds too tightly), and its
// water evens out with the plant cells beside it at PLANT_FLOW of their
// difference a pairing (the sap rising): so a plant's roots water its
// leaves; its sugar evens out the same way. A plant short of water to grow
// on (under GROW_WATER) roots into the soil under it, and its roots spread on
// down through soil holding less than they could draw (ROOT_SPREAD a
// pairing): rooted soil (FL_ROOTED) keeps its grains, and passes the water
// it holds past its wilting point up the roots, ROOT_FLOW of it a pairing,
// to the rooted soil above or the plant, so a plant drinks from as deep as
// its roots go.
const ROOT_TAKE: f32 = 32.0;
const PLANT_FLOW: f32 = 0.25;
const GROW_WATER: u32 = 96u;
const ROOT_SPREAD: f32 = 0.05;
const ROOT_FLOW: f32 = 0.25;

struct Pair {
    a: Cell,
    b: Cell,
}

// Empty air where a cell of liquid was, at the temperature the liquid acted at.
fn emptied(c: Cell) -> Cell {
    var air = makeCell(AIR, tempOf(c));
    air.frac = vec2<f32>(0.0);
    return air;
}

// Water w (fresh) or mud with its water gone: nothing, or soaked dirt.
fn drained(w: Cell) -> Cell {
    if (w.t == MUD) {
        return residueOf(w, elems[MUD], tempOf(w));
    }
    return emptied(w);
}

// A porous powder s and the fresh water or mud w beside it. Not yet soaked,
// s soaks through, taking up all it has room for, and w goes as likely as
// that room is of the water it holds, fresh water a particle at a time
// (common.wgsl's takeLiquid): on average as much water goes as the soil
// takes up. Soaked dirt takes water in as mud
// (a cell of water more).
fn soak(s0: Cell, w0: Cell, drain: i32, tau: f32) -> Pair {
    var s = s0;
    var w = w0;
    let held = soilWaterOf(s);
    if (held < SOIL_FULL) {
        // Free water is at no suction: it soaks in by the soil's pull and its
        // own weight (drain: 1 with the water above) through their face, at
        // the mean of the soil's conductivity soaked and as it is (Green &
        // Ampt 1911), and goes as likely as what soaked in is of a cell.
        let e = elems[s.t];
        let se = soilSaturation(e, f32(held));
        let k = 0.5 * (e.soilK + soilConductivity(e, se));
        let q = k * (-soilHead(e, se) / P.dx + f32(drain));
        let has = WATER_UNITS * liquidAmount(w);
        let took = clamp(q * tau / P.dx * WATER_UNITS, 0.0, min(f32(SOIL_FULL - held), has));
        let whole = u32(min(floor(took + rnd()), f32(SOIL_FULL - held)));
        s = withWaterHeat(s, whole, tempOf(w));
        s = withSoilWater(s, held + whole);
        if (takeLiquid(&w, took / WATER_UNITS)) {
            w = drained(w);
        }
    } else if (s.t == DIRT && w.t == WATER && liquidAmount(w) == 1.0 && rnd() < SOAK_MUD) {
        // (A full cell of water: mud holds one.)
        s = makeCell(MUD, s.temp);
        w = emptied(w);
    }
    return Pair(s, w);
}

// Two porous grains, a above b (drain 1), b above a (-1) or neither (0):
// water flows between them as Richards' equation has it (1931): through
// their face at the mean of their conductivities, q = K ((ψa - ψb) / dx +
// drain) m/s, by the difference in their suction and height, over the
// visit's tau seconds. The suction's share no more than brings their
// heads level (the two curves' slopes), the height's no more than the
// one has to give or the other room for. So water soaks from wet soil into
// dry, drains out of soil wetter than it holds against gravity (its field
// capacity comes of the curves), and quickly through sand, slowly through
// dirt, and soil beside sand draws water from it.
fn soilWater(a0: Cell, b0: Cell, drain: i32, tau: f32) -> Pair {
    var a = a0;
    var b = b0;
    let ea = elems[a.t];
    let eb = elems[b.t];
    let la = f32(soilWaterOf(a));
    let lb = f32(soilWaterOf(b));
    let sa = soilSaturation(ea, la);
    let sb = soilSaturation(eb, lb);
    let ha = soilHead(ea, sa);
    let hb = soilHead(eb, sb);
    let k = 0.5 * (soilConductivity(ea, sa) + soilConductivity(eb, sb));
    let units = tau / P.dx * WATER_UNITS;
    let full = f32(SOIL_FULL);
    // What brings their heads level: for one soil, as much water in each;
    // for two, found along their curves (the head grows with the water).
    var level = 0.5 * (la - lb);
    if (a.t != b.t) {
        var lo = select(-min(lb, full - la), 0.0, ha >= hb);
        var hi = select(0.0, min(la, full - lb), ha >= hb);
        for (var n = 0; n < 8; n++) {
            let mid = 0.5 * (lo + hi);
            let gap = soilHead(ea, soilSaturation(ea, la - mid)) - soilHead(eb, soilSaturation(eb, lb + mid));
            if (gap > 0.0) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        level = 0.5 * (lo + hi);
    }
    let suction = clamp(k * (ha - hb) / P.dx * units, -abs(level), abs(level));
    let weight = k * f32(drain) * units;
    let moved = clamp(floor(suction + weight + rnd()), -min(lb, full - la), min(la, full - lb));
    // (The water taking its heat to the one it goes into: REWRITE_PLAN.md 5h.)
    if (moved > 0.0) {
        b = withWaterHeat(b, u32(moved), tempOf(a));
    } else if (moved < 0.0) {
        a = withWaterHeat(a, u32(-moved), tempOf(b));
    }
    a = withSoilWater(a, u32(la - moved));
    b = withSoilWater(b, u32(lb + moved));
    return Pair(a, b);
}

// Soil s taking `units` of water (SOIL_FULL soaked) at temp: its
// temperature the mean of its own, at its heat capacity with its water as
// it was, and the water's (REWRITE_PLAN.md 5h: the water brought no heat,
// so warm water soaking into cold sand warmed it none).
fn withWaterHeat(s: Cell, units: u32, temp: f32) -> Cell {
    var n = s;
    if (units == 0u) {
        return n;
    }
    let cap = cellHeatCap(s, noGas());
    let more = soilWaterCap(units);
    n.temp = (cap * tempOf(s) + more * temp) / (cap + more);
    return n;
}

// Plant cell p drinking from the soil, fresh water or mud x beside it
// (ROOT_TAKE), not while burning (its life counts what of it has burnt):
// soil gives what its pores hold past its wilting point; water or mud gives
// what was drunk (common.wgsl's takeLiquid).
fn drink(p0: Cell, x0: Cell, drain: i32) -> Pair {
    var p = p0;
    var x = x0;
    let water = plantWaterOf(p);
    if ((p.flags & FL_BURN) != 0u || water >= PLANT_WATER) {
        return Pair(p, x);
    }
    let soil = holdsSoilWater(x.t);
    // Short of water to grow on, it roots into the soil under it.
    if (drain > 0 && soil && water < GROW_WATER) {
        x.flags |= FL_ROOTED;
    }
    var took = min(f32(PLANT_WATER - water), ROOT_TAKE);
    // Soil's units a unit of the plant's water.
    let perUnit = PLANT_UNIT / WATER_VAPOUR * WATER_UNITS;
    if (soil) {
        let held = soilWaterOf(x);
        let left = f32(held) - soilWiltLife(elems[x.t]);
        took = floor(min(took, max(left, 0.0) / perUnit));
        x = withSoilWater(x, held - u32(min(floor(took * perUnit + rnd()), f32(held))));
    } else {
        took = min(took, liquidAmount(x) * WATER_VAPOUR / PLANT_UNIT);
        if (takeLiquid(&x, took * PLANT_UNIT / WATER_VAPOUR)) {
            x = drained(x);
        }
    }
    p = withPlantWater(p, water + u32(took));
    return Pair(p, x);
}

// Two porous grains, rooted a above b: the roots pass b's water past its
// wilting point up (ROOT_FLOW), and spread down into b while a is drier
// than it is (ROOT_SPREAD).
fn roots(a0: Cell, b0: Cell) -> Pair {
    var a = a0;
    var b = b0;
    let la = soilWaterOf(a);
    let lb = soilWaterOf(b);
    if ((b.flags & FL_ROOTED) != 0u) {
        let left = f32(lb) - soilWiltLife(elems[b.t]);
        let moved = clamp(floor(left * ROOT_FLOW + rnd()), 0.0, f32(SOIL_FULL - la));
        // (The water taking its heat up with it, as soilWater's does.)
        a = withWaterHeat(a, u32(moved), tempOf(b));
        a = withSoilWater(a, la + u32(moved));
        b = withSoilWater(b, lb - u32(moved));
    } else if (lb > la && rnd() < ROOT_SPREAD) {
        b.flags |= FL_ROOTED;
    }
    return Pair(a, b);
}

// Two plant cells beside each other: their water and sugar even out
// (PLANT_FLOW), not while either burns (its life counts what has burnt).
fn sap(a0: Cell, b0: Cell) -> Pair {
    var a = a0;
    var b = b0;
    if (((a.flags | b.flags) & FL_BURN) != 0u) {
        return Pair(a, b);
    }
    let wa = f32(plantWaterOf(a));
    let wb = f32(plantWaterOf(b));
    let flow = floor((wa - wb) * PLANT_FLOW + rnd());
    a = withPlantWater(a, u32(wa - flow));
    b = withPlantWater(b, u32(wb + flow));
    let sa = f32(sugarOf(a));
    let sb = f32(sugarOf(b));
    let sugar = floor((sa - sb) * PLANT_FLOW + rnd());
    a = withSugar(a, u32(sa - sugar));
    b = withSugar(b, u32(sb + sugar));
    return Pair(a, b);
}

// What is left of a grain of a soluble element, in 255ths of the fill of a
// cell of water: a whole grain fills 1 / solubility cells. What of it has
// dissolved counts up as it goes, to 255 when it's gone.
fn grainLeft(g: Cell) -> f32 {
    return f32(255u - dissolvedOf(g)) * (1.0 / elems[g.t].solubility);
}

// A grain having given up `moved` of what it holds: on to the next, or gone.
fn grainLess(g: Cell, moved: f32) -> Cell {
    var n = g;
    let used = f32(dissolvedOf(g)) + floor(moved * elems[g.t].solubility + rnd());
    if (used >= 255.0) {
        n = makeCell(AIR, g.temp);
        n.frac = vec2<f32>(0.0);
    } else {
        n = withDissolved(n, u32(used));
    }
    return n;
}

// How far water w has frozen, by the latent heat it has given off at its
// freezing point: 0 liquid, 1 all of it (slush between).
fn frozenShare(w: Cell) -> f32 {
    let e = elems[WATER];
    return (phaseOf(WATER, soluteOf(w), strengthOf(w), 0.0).y - w.temp) * e.heatCap / e.latentDown;
}

// Whether water w has given off all its latent heat freezing but holds on to
// its solute, which it must pass on before it can turn to ice.
fn frozenOut(w: Cell) -> bool {
    return soluteOf(w) != 0u && frozenShare(w) >= 1.0;
}

// Brine slushier than this melts no more ice until it has warmed: a freezing
// mixture melts ice only as fast as heat comes into it.
const MELT_SLUSH: f32 = 0.5;

// Two cells of water: what they carry evens out (different solutes don't
// mix; pure water takes on its neighbor's) by the solute's diffusivity
// (Fick: the two relax to their mean at 2 solD / dx² over the visit's tau
// seconds), as slowly as molecular diffusion is (salt a centimetre in about
// a day: most mixing is the water moving, so brine poured under fresh water
// stays layered); and one done freezing pushes its solute out into the
// other. What moves is solute, a cell's strength times how much water it
// holds (liquidAmount): a quarter of a cell of brine giving a full cell of
// fresh water as much as raises it a quarter as far.
fn mixSolutes(a: Cell, b: Cell, tau: f32) -> Pair {
    var x = a;
    var y = b;
    let xs = soluteOf(x);
    let ys = soluteOf(y);
    if (xs != ys && xs != 0u && ys != 0u) {
        return Pair(x, y);
    }
    let solute = max(xs, ys);
    let xl = strengthOf(x);
    let yl = strengthOf(y);
    let xOut = frozenOut(x);
    let yOut = frozenOut(y);
    let xa = liquidAmount(x);
    let ya = liquidAmount(y);
    var flow = 0.0;  // from x to y, in 255ths of a cell's
    if (xOut && !yOut) {
        flow = min(f32(xl) * xa, f32(255u - yl) * ya);
    } else if (yOut && !xOut) {
        flow = -min(f32(yl) * ya, f32(255u - xl) * xa);
    } else {
        let share = 1.0 - exp(-2.0 * elems[solute].solD * tau / (P.dx * P.dx));
        flow = (f32(xl) - f32(yl)) * share * xa * ya / (xa + ya);
    }
    // Rounded at random, the same steps of strength each way between cells
    // holding as much, so none is made or lost; between cells holding
    // different amounts, none on average.
    let r = rnd();
    var xn = u32(clamp(floor(f32(xl) - flow / xa + r), 0.0, 255.0));
    var yn = u32(clamp(floor(f32(yl) + flow / ya + r), 0.0, 255.0));
    if (xa == ya) {
        let steps = i32(floor(flow / xa + r));
        xn = u32(clamp(i32(xl) - steps, 0, 255));
        yn = u32(clamp(i32(yl) + steps, 0, 255));
    }
    x = withSolute(withStrength(x, xn), select(0u, solute, xn > 0u));
    y = withSolute(withStrength(y, yn), select(0u, solute, yn > 0u));
    return Pair(x, y);
}

// Ice (or snow) melted into a cell of water carrying solute s, `strength` of
// it: as much heat as the ice had, less the latent heat to melt it, so the
// water comes out colder than the ice was. Snow is half a cell of water, so
// it is water only half the time (formed), with twice the heat. Ice too cold
// for that brine to be at all liquid, all its latent heat given off at its
// freezing point (frozenOut), stays ice: so a freezing mixture melts only as
// much ice as the heat around pays for.
fn meltInto(ice: Cell, s: u32, strength: u32) -> Cell {
    let ie = elems[ice.t];
    let we = elems[WATER];
    let energy = (ice.temp - ie.hiT) * ie.heatCap - ie.latentUp;
    let stored = energy * (we.rho / ie.rho) / we.heatCap;
    let freezing = we.loT + elems[s].solFreeze * shareOf(strength);
    if ((freezing - stored) * we.heatCap >= we.latentDown) {
        return ice;
    }
    // As much water as the ice held, what it owed and what it kept past its
    // share too (formed; this took a cell of ice for its own grams alone).
    return formed(withStrength(withSolute(makeCell(WATER, stored), select(0u, s, strength > 0u)), strength), ice);
}

// Water w (or mud) at pw and some other cell o beside it, at po: soil
// soaking it up, a plant drinking it, a grain dissolving into the water, ice
// melting into brine colder than it can freeze at, acid eating o, brine or
// mud done boiling giving its water off into air.
fn withWater(w0: Cell, o0: Cell, pw: vec2<u32>, po: vec2<u32>, tau: f32, drain: i32) -> Pair {
    var w = w0;
    var o = o0;
    let oe = elems[o.t];
    let solute = soluteOf(w);
    let strength = strengthOf(w);
    // Only fresh water: brine keeps its salt, and plants can't drink it.
    if (w.t == MUD || (holdsSolute(w.t) && solute == 0u)) {
        if (holdsSoilWater(o.t)) {
            let s = soak(o, w, drain, tau);
            return Pair(s.b, s.a);
        }
        if (holdsPlantWater(o.t)) {
            let s = drink(o, w, 0);
            return Pair(s.b, s.a);
        }
    }
    if (holdsSolute(w.t)) {
        if (holdsDissolved(o.t)) {
            // Dissolving as fast as the water is short of saturation
            // (dissolveRate: Noyes & Whitney 1897), what of a grain is left at
            // most.
            if ((solute == 0u || solute == o.t) && strength < 255u) {
                // (Into as much water as the cell holds, liquidAmount.)
                let amount = liquidAmount(w);
                let share = 1.0 - exp(-oe.dissolveRate * tau / P.dx);
                let moved = floor(min(f32(255u - strength) * share * amount + rnd(), grainLeft(o)));
                if (moved > 0.0) {
                    let raised = u32(min(floor(moved / amount + rnd()), f32(255u - strength)));
                    w = withStrength(withSolute(w, o.t), strength + raised);
                    // What dissolves takes its heat into the water (over
                    // 20 °C, as a crumbling cell keeps its: its capacity
                    // goes with it, grainShare).
                    let before = cellHeatCap(o, noGas());
                    let grainTemp = o.temp;
                    o = grainLess(o, moved);
                    let gone = before - select(cellHeatCap(o, noGas()), 0.0, o.t == AIR);
                    w.temp += gone * (grainTemp - AMBIENT_T) / cellHeatCap(w, noGas());
                }
            }
            return Pair(w, o);
        }
        if ((o.t == ICE || o.t == SNOW) && solute != 0u) {
            if (tempOf(o) > phaseOf(WATER, solute, strength, 0.0).y && frozenShare(w) < MELT_SLUSH
                && rnd() < MELT_IN_BRINE) {
                // Half the brine's solute goes with it, if it is water.
                let half = u32(floor(f32(strength) * 0.5 + rnd()));
                o = meltInto(o, solute, half);
                if (o.t == WATER) {
                    w = withSolute(withStrength(w, strength - half), select(0u, solute, strength > half));
                }
            }
            return Pair(w, o);
        }
        let s = elems[solute];
        if (solute != 0u && s.acidEq > 0.0 && oe.acidDemand > 0.0
            && (oe.flags & (F_ACID_PROOF | F_INDESTRUCTIBLE)) == 0u
            && rnd() < s.solCorrode * shareOf(strength) / (1.0 + oe.hardness)) {
            // Acid eating o by stoichiometry (5e): a cell of o takes
            // acidDemand equivalents. Holding that many, the acid eats it
            // whole; holding fewer, it is spent, and eats it as likely as it
            // holds of them. Eaten, o is the gas it gives off (half a mole an
            // equivalent: hydrogen from a metal, carbon dioxide, air holding no
            // oxygen, from a carbonate), with its heat; the heat of reaction
            // goes into the acid.
            // (As much as the cell of it holds: liquidAmount.)
            let amount = liquidAmount(w);
            let have = s.acidEq * shareOf(strength) * amount;
            let used = min(have, oe.acidDemand);
            let eaten = have >= oe.acidDemand || rnd() < have / oe.acidDemand;
            let left = u32(max(f32(strength) - floor(used / (s.acidEq * amount) * 255.0 + rnd()), 0.0));
            w = withSolute(withStrength(w, left), select(0u, solute, left > 0u));
            var heat = used * oe.acidHeat;
            if (eaten) {
                let oTemp = tempOf(o);
                let gas = Gas(0.0, 0.0, select(0.0, 1.0, oe.acidGas == AIR), oe.acidDemand * GAS_PER_EQUIVALENT);
                o = makeCell(oe.acidGas, oTemp);
                o.frac = vec2<f32>(0.0);
                setGasAtCell(po.y * P.size.x + po.x, gas);
                heat += (oe.heatCap - gasHeatCap(gas)) * (oTemp - AMBIENT_T);
            }
            // (Into as much as the cell holds, as the acid it used was.)
            w.temp += heat / (elems[w.t].heatCap * amount);
            return Pair(w, o);
        }
    }
    // Brine or mud that has taken in all of its latent heat (changePhase in
    // update.wgsl holds it, its stored temperature running on: superheated,
    // effTempIn) boils its water off, a cell of water's vapour (as
    // update.wgsl's phaseGas), into the air beside it and leaves what it
    // carried, the two sharing the heat it took in past that at one
    // temperature; the vapour mixes into the air there, heat and all.
    let we = elems[w.t];
    if (o.t == AIR && leavesResidue(w, we)) {
        let tr = phaseOf(w.t, solute, strength, atmLog(airPressureAt(vec2<f32>(pw) + 0.5)));
        let full = latentFull(we, tr.x);
        if (w.temp >= full) {
            var left = residueOf(w, we, tr.x);
            let vapour = Gas(WATER_VAPOUR * liquidAmount(w), 0.0, 0.0, 0.0);
            let caps = gasHeatCap(vapour) + select(elems[left.t].heatCap, 0.0, left.t == AIR);
            left.temp = tr.x + (w.temp - full) * we.heatCap * liquidAmount(w) / caps;
            let io = po.y * P.size.x + po.x;
            var g = gasAtCell(io);
            let air = gasHeatCap(g);
            o.temp = (air * o.temp + gasHeatCap(vapour) * left.temp) / (air + gasHeatCap(vapour));
            g.vapour += vapour.vapour;
            setGasAtCell(io, g);
            w = left;
        }
    }
    return Pair(w, o);
}

// What cell i holds of the air (the gas plane, common.wgsl).
fn gasAtCell(i: u32) -> Gas {
    return gasFrom(cells[gasIndex(P.size.x * P.size.y, i)], i);
}

fn setGasAtCell(i: u32, g: Gas) {
    let w = packGas(g);
    let k = gasIndex(P.size.x * P.size.y, i);
    if (gasHigh(i)) {
        cells[k].z = w.x;
        cells[k].w = w.y;
    } else {
        cells[k].x = w.x;
        cells[k].y = w.y;
    }
}

// Salt (any soluble grain) on ice: brine forms, colder than the ice can stay
// frozen against, so the ice melts into it, the heat for it taken from the
// ice and so from around: a freezing mixture. Not below the coldest the
// saturated brine can be (salt's -21 °C), at which nothing melts.
fn grainOnIce(g0: Cell, i0: Cell, tau: f32) -> Pair {
    var g = g0;
    var i = i0;
    let ge = elems[g.t];
    if (tempOf(i) > elems[WATER].loT + ge.solFreeze && rnd() < 1.0 - exp(-ge.dissolveRate * tau / P.dx)) {
        let moved = floor(min(255.0, grainLeft(g)));
        i = meltInto(i, g.t, u32(moved));
        if (i.t == WATER) {
            g = grainLess(g, moved);
        }
    }
    return Pair(g, i);
}

// Fire (REWRITE_PLAN.md stage 5): burning fuel f and the oxygen carrier o
// beside it, at pf and po (f burning, FL_BURN, as the update pass lights and
// puts it out). Where a flame can burn in o's gas (oxygenToBurn), f takes the
// oxygen it burns from it: as fast as a face of fresh air burns a cell of f
// through in burnTime real seconds (FIRE_PACE times faster here; slower for a
// bigger cell, whose faces are smaller for what it holds, faceShare), in
// proportion to the oxygen o holds, ash's pores giving theirs at ASH_OXYGEN
// of that. A face comes round every 4 steps, in real time (burning cells
// make heat's clock real, clock.wgsl). What burns, in whole 255ths of a cell
// of f (its life counts them up), uses oxygenNeed cells of fresh air's
// oxygen a cell, and gives o its oxygenHeat a cell of oxygen (Huggett's
// OXYGEN_HEAT) and the water it makes, as vapour, which comes in at
// AMBIENT_T (what the heat is reckoned from) and is heated with the rest,
// no hotter than FLAME_MAX_T. Its volatiles burnt (charAt), what chars is
// its char (charTo, charYield of it at random, the gas they left else);
// burnt through, anything is its ash (burnTo, ashYield of it, the gas its
// burning made else). What heat it held beyond what it is then (from
// AMBIENT_T) went with the gas it burnt into, to o; turned into something
// that holds gas (smoke, ash's pores, air), it holds gas made of it, its
// oxygen used up. A cell of liquid fuel holds as much as its particles
// (liquidAmount), and burns through as much faster or slower.
fn burn(f0: Cell, o0: Cell, pf: vec2<u32>, po: vec2<u32>) -> Pair {
    var f = f0;
    var o = o0;
    let fe = elems[f.t];
    let amount = select(1.0, liquidAmount(f), fe.kind == KIND_LIQUID);
    let need = fe.oxygenNeed * amount;
    let io = po.y * P.size.x + po.x;
    var g = gasAtCell(io);
    // What it can burn: the oxygen o holds, as far as that heats o to
    // FLAME_MAX_T (oxygenToBurn), at a rate by all o holds.
    let room = oxygenToBurn(o.t, g, tempOf(o));
    if (fe.burnTime <= 0.0 || room <= 0.0) {
        return Pair(f, o);
    }
    let rate = fe.oxygenNeed * FIRE_PACE / fe.burnTime * faceShare(P.dx) * select(1.0, ASH_OXYGEN, o.t == ASH);
    let want = min(oxygenHeld(o.t, g) * (1.0 - exp(-rate * 4.0 * P.dt)), room);
    // As far as it chars at, if it chars (5c).
    let limit = select(255u, u32(fe.charAt), fe.charTo != NO_ELEM);
    let burnt = burntOf(f);
    let units = min(min(floor(want * 255.0 / need + rnd()), floor(room * 255.0 / need)),
        f32(limit - min(burnt, limit)));
    if (units <= 0.0) {
        return Pair(f, o);
    }
    let used = units * need / 255.0;
    var heat = used * fe.oxygenHeat;
    f = withBurnt(f, burnt + u32(units));
    if (burntOf(f) >= limit) {
        let fTemp = tempOf(f);
        let fCap = elems[f.t].heatCap * amount;
        if (fe.charTo != NO_ELEM) {
            // Its volatiles burnt, it is charcoal glowing on, by its yield,
            // or the gas they left.
            f = convert(f, select(AIR, fe.charTo, rnd() < fe.charYield));
        } else {
            // Burnt through: its ash, by its yield, or the gas its burning
            // made.
            f = convert(f, select(AIR, fe.burnTo, elems[fe.burnTo].kind == KIND_GAS || rnd() < fe.ashYield));
        }
        var fg = noGas();
        if (holdsGas(f.t)) {
            fg = madeGas(f.temp, 1.0);
            setGasAtCell(pf.y * P.size.x + pf.x, fg);
        }
        heat += (fCap - heatCapOf(f.t, fg)) * (fTemp - AMBIENT_T);
    }
    let cap = heatCapOf(o.t, g);
    let vapour = used * fe.burnVapour;
    g.lost = stochastic16(min(g.lost + used / g.amount, 1.0), rnd());
    g.vapour = stochastic16(g.vapour + vapour, rnd());
    setGasAtCell(io, g);
    let capNow = heatCapOf(o.t, g);
    o.temp = (cap * o.temp + (capNow - cap) * AMBIENT_T + heat) / capNow;
    return Pair(f, o);
}

// A flammable gas h (hydrogen), lit, and the oxygen carrier o beside it, at
// ph and po: as burn, but what burns is the gas h holds (its amount, a cell
// of it at 1 atm and 20 °C oxygenNeed cells of fresh air's oxygen), as fast
// as its flame runs through a cell (burnTime), and the heat of what of it
// burns (from AMBIENT_T) goes with the rest to o. Burnt out, h is a cell of
// nothing, which the air around fills.
fn burnGas(h0: Cell, o0: Cell, ph: vec2<u32>, po: vec2<u32>) -> Pair {
    var h = h0;
    var o = o0;
    let he = elems[h.t];
    let ih = ph.y * P.size.x + ph.x;
    let io = po.y * P.size.x + po.x;
    var gh = gasAtCell(ih);
    var g = gasAtCell(io);
    let room = oxygenToBurn(o.t, g, tempOf(o));
    if (he.burnTime <= 0.0 || room <= 0.0 || gh.amount <= 0.0) {
        return Pair(h, o);
    }
    let rate = he.oxygenNeed * FIRE_PACE / he.burnTime * faceShare(P.dx);
    let used = min(min(oxygenHeld(o.t, g) * (1.0 - exp(-rate * 4.0 * P.dt)), room), gh.amount * he.oxygenNeed);
    let burnt = used / he.oxygenNeed;
    let left = Gas(gh.vapour, gh.mist, gh.lost, gh.amount - burnt);
    let heat = used * he.oxygenHeat + (gasHeatCap(gh) - gasHeatCap(left)) * (h.temp - AMBIENT_T);
    if (left.amount < 1e-3) {
        // What it held but the hydrogen (its moisture) stays.
        h = makeCell(AIR, h.temp);
        setGasAtCell(ih, Gas(gh.vapour, gh.mist, 0.0, 0.0));
    } else {
        setGasAtCell(ih, Gas(left.vapour, left.mist, left.lost, stochastic16(left.amount, rnd())));
    }
    let cap = heatCapOf(o.t, g);
    g.lost = stochastic16(min(g.lost + used / g.amount, 1.0), rnd());
    g.vapour = stochastic16(g.vapour + used * he.burnVapour, rnd());
    setGasAtCell(io, g);
    let capNow = heatCapOf(o.t, g);
    o.temp = (cap * o.temp + (capNow - cap) * AMBIENT_T + heat) / capNow;
    return Pair(h, o);
}

// Whether a cell holding v burns at a face to a cell of type t: alight, and
// t an oxygen carrier.
fn burnsAgainst(v: vec4<u32>, t: u32) -> bool {
    return cellHas(v, FL_BURN) && carriesOxygen(t) && elems[cellType(v)].burnTime > 0.0;
}

// f, lit, burning against the oxygen carrier o: a gas burns what it holds.
fn burnAt(f: Cell, o: Cell, pf: vec2<u32>, po: vec2<u32>) -> Pair {
    if (elems[f.t].kind == KIND_GAS) {
        return burnGas(f, o, pf, po);
    }
    return burn(f, o, pf, po);
}

// Two grains of a soluble element (a above b, drain 1; b above a, -1; or
// neither) where each holds less than a grain (what water left of it,
// residueOf): the upper (or a) goes into the lower, if it fits, and is air.
// So a pan of sea water dried leaves a crust as thick as its salt, not as
// deep as its water was.
fn mergeGrains(a: Cell, b: Cell, drain: i32) -> Pair {
    let ha = 255u - dissolvedOf(a);
    let hb = 255u - dissolvedOf(b);
    let held = ha + hb;
    if (held > 255u) {
        return Pair(a, b);
    }
    // Their heat together, each's capacity as much as is left of it
    // (grainShare): the mean of their temperatures by that.
    let temp = (f32(ha) * a.temp + f32(hb) * b.temp) / f32(max(held, 1u));
    if (drain > 0) {
        var keep = withDissolved(b, 255u - held);
        keep.temp = temp;
        return Pair(emptied(a), keep);
    }
    var keep = withDissolved(a, 255u - held);
    keep.temp = temp;
    return Pair(keep, emptied(b));
}

// Water w that electrolysis is splitting (FL_SPLIT, electric.wgsl) beside the
// cell o: it hands its solute on to o if o is water that can take it (fresh,
// or carrying the same), as much as o has room for, and with none left it is
// the hydrogen it splits into, WATER_HYDROGEN cells of it (Faraday), holding
// its heat. So brine at a cathode bubbles hydrogen and the brine beside it
// grows saltier. Where o is water already as salty as it can be, the salt
// comes out of solution: w is its residue (a grain holding just its salt)
// and its hydrogen joins the gas beside it (airsource.wgsl), as gas holding
// no oxygen (the gas plane keeps no kinds of gas apart).
fn split(w0: Cell, o0: Cell, pw: vec2<u32>) -> Pair {
    var w = w0;
    var o = o0;
    let solute = soluteOf(w);
    if (solute != 0u && holdsSolute(o.t) && (soluteOf(o) == 0u || soluteOf(o) == solute)) {
        let moved = min(strengthOf(w), 255u - strengthOf(o));
        o = withStrength(withSolute(o, solute), strengthOf(o) + moved);
        let left = strengthOf(w) - moved;
        w = withSolute(withStrength(w, left), select(0u, solute, left > 0u));
        let gas = Gas(0.0, 0.0, 0.0, WATER_HYDROGEN * liquidAmount(w));
        let temp = AMBIENT_T + cellHeatCap(w, noGas()) * (w.temp - AMBIENT_T) / gasHeatCap(gas);
        if (left == 0u) {
            w = makeCell(HYDROGEN, temp);
            setGasAtCell(pw.y * P.size.x + pw.x, gas);
        } else if (moved == 0u) {
            let residue = residueOf(w, elems[w.t], tempOf(w));
            pushGas(pw.y * P.size.x + pw.x, Gas(0.0, 0.0, 1.0, gas.amount), tempOf(w));
            w = residue;
        }
    }
    return Pair(w, o);
}

// The pair (a, b) at (pa, pb), settled; drain as soilWater's, tau the
// seconds the visit stands for (visitTau).
fn settle(a: Cell, b: Cell, pa: vec2<u32>, pb: vec2<u32>, drain: i32, tau: f32) -> Pair {
    if (a.t == WATER && (a.flags & FL_SPLIT) != 0u) {
        return split(a, b, pa);
    }
    if (b.t == WATER && (b.flags & FL_SPLIT) != 0u) {
        let s = split(b, a, pb);
        return Pair(s.b, s.a);
    }
    if ((a.flags & FL_BURN) != 0u && carriesOxygen(b.t)) {
        return burnAt(a, b, pa, pb);
    }
    if ((b.flags & FL_BURN) != 0u && carriesOxygen(a.t)) {
        let s = burnAt(b, a, pb, pa);
        return Pair(s.b, s.a);
    }
    if (a.t == WATER && b.t == WATER) {
        return mixSolutes(a, b, tau);
    }
    let ae = elems[a.t];
    let be = elems[b.t];
    if (a.t == WATER || (ae.flags & F_RESIDUE) != 0u) {
        return withWater(a, b, pa, pb, tau, drain);
    }
    if (b.t == WATER || (be.flags & F_RESIDUE) != 0u) {
        let s = withWater(b, a, pb, pa, tau, -drain);
        return Pair(s.b, s.a);
    }
    if (ae.solubility > 0.0 && a.t == b.t) {
        return mergeGrains(a, b, drain);
    }
    let aSoil = porous(ae);
    let bSoil = porous(be);
    if (aSoil && bSoil) {
        // Roots running down through the soil carry its water up; the rest
        // flows as Richards' equation has it.
        if (drain > 0 && (a.flags & FL_ROOTED) != 0u) {
            let r = roots(a, b);
            return soilWater(r.a, r.b, drain, tau);
        }
        if (drain < 0 && (b.flags & FL_ROOTED) != 0u) {
            let r = roots(b, a);
            let s = soilWater(r.b, r.a, drain, tau);
            return s;
        }
        return soilWater(a, b, drain, tau);
    }
    if (a.t == PLANT && b.t == PLANT) {
        return sap(a, b);
    }
    if (a.t == PLANT && bSoil) {
        return drink(a, b, drain);
    }
    if (b.t == PLANT && aSoil) {
        let s = drink(b, a, -drain);
        return Pair(s.b, s.a);
    }
    let aIce = a.t == ICE || a.t == SNOW;
    let bIce = b.t == ICE || b.t == SNOW;
    if (ae.solubility > 0.0 && bIce) {
        return grainOnIce(a, b, tau);
    }
    if (be.solubility > 0.0 && aIce) {
        let s = grainOnIce(b, a, tau);
        return Pair(s.b, s.a);
    }
    return Pair(a, b);
}

// The pairing goes by step: frame % 4 picks the axis (odd: vertical) and
// which of a cell's two neighbors along it (frame / 2 % 2 shifts the pairs by
// one). gid.x counts pairs along the axis, gid.y lines across it.
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = P.frame % 4u;
    let vertical = (k & 1u) == 1u;
    let along = gid.x * 2u + (k >> 1u);
    let pa = select(vec2<u32>(along, gid.y), vec2<u32>(gid.y, along), vertical);
    let pb = pa + select(vec2<u32>(1u, 0u), vec2<u32>(0u, 1u), vertical);
    if (any(pb >= P.size)) {
        return;
    }
    let ia = pa.y * P.size.x + pa.x;
    let ib = pb.y * P.size.x + pb.x;
    let va = cells[ia];
    let vb = cells[ib];
    let ta = cellType(va);
    let tb = cellType(vb);
    // Only water, mud, a soluble grain or soil (with soil or a plant) ever
    // settles anything, or something burning against an oxygen carrier.
    let ea = elems[ta];
    let eb = elems[tb];
    let soilA = porous(ea);
    let soilB = porous(eb);
    if (ta != WATER && tb != WATER && ((ea.flags | eb.flags) & F_RESIDUE) == 0u
        && ea.solubility <= 0.0 && eb.solubility <= 0.0
        && !(soilA && (soilB || tb == PLANT)) && !(soilB && ta == PLANT) && !(ta == PLANT && tb == PLANT)
        && !burnsAgainst(va, tb) && !burnsAgainst(vb, ta)) {
        return;
    }
    seedRng(ia, P.frame, P.seed ^ 0x5a1f5a1fu);
    // Which of the two is above the other, by the pair's step along down;
    // weightless, neither.
    let step = vec2<i32>(pb) - vec2<i32>(pa);
    let drain = select(sign(dot(step, downAt(vec2<i32>(pa)))), 0, weightlessAt(vec2<i32>(pa)));
    let s = settle(unpackCell(va), unpackCell(vb), pa, pb, drain, visitTau(pa, pb));
    cells[ia] = packCell(s.a);
    cells[ib] = packCell(s.b);
}
