// Applies a tool to every cell inside the brush (SHAPE_*, js/shared.js): a
// capsule from B.a to B.b, round or square, or a spray; a box; or the cells
// a flood fill found (sim.js's fill), set in the mask.

const TAU: f32 = 6.2831853;
// A spray paints at most this share of the cells at its middle a frame (it
// goes on painting while held), thinning out to none at its edge; heating
// and cooling, it is soft-edged the same way.
const SPRAY: f32 = 0.08;

@group(0) @binding(0) var<uniform> B: Brush;
@group(0) @binding(1) var<storage, read> elems: array<Elem>;
@group(0) @binding(2) var<storage, read_write> cells: array<vec4<u32>>;
// SHAPE_MASK's cells, a bit each, row by row over the world.
@group(0) @binding(3) var<storage, read> mask: array<u32>;
// What goes into the air (airsource.wgsl): gas painted over.
@group(0) @binding(4) var<storage, read_write> source: array<atomic<u32>>;

// What cell i holds of the air (the gas plane, common.wgsl).
fn gasAt(i: u32) -> Gas {
    return gasFrom(cells[gasIndex(B.size.x * B.size.y, i)], i);
}

fn setGasAt(i: u32, g: Gas) {
    let w = packGas(g);
    let k = gasIndex(B.size.x * B.size.y, i);
    if (gasHigh(i)) {
        cells[k].z = w.x;
        cells[k].w = w.y;
    } else {
        cells[k].x = w.x;
        cells[k].y = w.y;
    }
}

// What a cell of t painted at temp (°C) in row y holds of the air: the open
// air's pressure at its height (openAirAtm), of its own gas and its vapour
// (the outdoor air's, OUTDOOR_VAPOUR, for what holds air: air, a flame,
// smoke, ash's pores, a fan; none for hydrogen or a fireball's gas); steam
// all of it vapour. Nothing for anything else.
fn paintedGas(t: u32, temp: f32, y: u32, steam: bool) -> Gas {
    if (!holdsGas(t)) {
        return noGas();
    }
    let all = airAmount(openAirAtm((f32(B.size.y - y) - 0.5) * B.dx), temp);
    if (steam) {
        return Gas(all * BOIL_K * VAPOUR_MAX / (AMBIENT_T + KELVIN), 0.0, 0.0, 0.0);
    }
    let vapour = select(0.0, OUTDOOR_VAPOUR, carriesOxygen(t) || elems[t].kind == KIND_SOLID);
    return Gas(vapour, 0.0, 0.0, max(all - vapourAmount(vapour), 0.0));
}

// Whether a point q from the segment's start is within r of the segment d
// measured square: whether some point of the segment has it within r both
// across and down.
fn inSquare(q: vec2<f32>, d: vec2<f32>, r: f32) -> bool {
    var lo = 0.0;
    var hi = 1.0;
    for (var k = 0; k < 2; k++) {
        if (abs(d[k]) < 1e-6) {
            if (abs(q[k]) > r) {
                return false;
            }
        } else {
            let t0 = (q[k] - r) / d[k];
            let t1 = (q[k] + r) / d[k];
            lo = max(lo, min(t0, t1));
            hi = min(hi, max(t0, t1));
        }
    }
    return lo <= hi;
}

// How far into the brush the cell centred at p is: 1 inside it (at the
// middle of a spray, which thins to 0 at its edge), below 0 outside.
fn coverage(p: vec2<f32>, cell: vec2<u32>) -> f32 {
    let inBox = all(p >= min(B.a, B.b)) && all(p <= max(B.a, B.b));
    switch B.shape {
        case SHAPE_RECT: {
            return select(-1.0, 1.0, inBox);
        }
        case SHAPE_MASK: {
            let i = cell.y * B.size.x + cell.x;
            return select(-1.0, 1.0, inBox && ((mask[i >> 5u] >> (i & 31u)) & 1u) != 0u);
        }
        case SHAPE_SQUARE: {
            return select(-1.0, 1.0, inSquare(p - B.a, B.b - B.a, B.radius));
        }
        default: {
            let ab = B.b - B.a;
            let h = clamp(dot(p - B.a, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0);
            let d = length(p - B.a - ab * h) / max(B.radius, 1e-6);
            if (d > 1.0) {
                return -1.0;
            }
            return select(1.0, 1.0 - d * d, B.shape == SHAPE_SPRAY);
        }
    }
}

// Dispatched over the brush's bounding box only, starting at B.origin.
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid.xy + B.origin;
    if (cell.x >= B.size.x || cell.y >= B.size.y) {
        return;
    }
    let p = vec2<f32>(cell) + 0.5;
    let cover = coverage(p, cell);
    if (cover < 0.0) {
        return;
    }
    let i = cell.y * B.size.x + cell.x;
    seedRng(i, B.seed, 0x5eedu);
    var c = unpackCell(cells[i]);
    let e = elems[c.t];
    // Painting (erasing, heating, ...) over just the one element.
    if (B.replace == OVER_ONLY && c.t != B.only && B.mode != MODE_WIND && B.mode != MODE_SOAK && B.mode != MODE_VAPOUR) {
        return;
    }
    let spray = B.shape == SHAPE_SPRAY;

    switch B.mode {
        case MODE_PAINT: {
            // A preset (salt water) comes as water carrying its solute, in
            // elem's second byte, as strong as the third says (255
            // saturated); soil as holding the water in the second (see
            // SOIL_FULL).
            let elem = B.elem & 0xffu;
            let extra = (B.elem >> 8u) & 0xffu;
            let strength = max((B.elem >> 16u) & 0xffu, 1u);
            let soil = holdsSoilWater(elem);
            let same = c.t == elem && (!isWater(elem) || (soluteOf(c) == extra && (extra == 0u || strengthOf(c) == strength)));
            if (elem != AIR && (same || (B.replace == OVER_EMPTY && c.t != AIR))) {
                return;
            }
            if (rnd() >= B.strength * select(1.0, SPRAY * cover, spray)) {
                return;
            }
            let own = B.temp < -1e28;
            if (elem == AIR && holdsGas(c.t)) {
                // Erasing a gas (or a fan) leaves the air it was: what it
                // held, as warm as it was unless painted at a temperature.
                c = makeCell(AIR, select(B.temp, c.temp, own));
                cells[i] = packCell(c);
                return;
            }
            // What it is painted over held of the air goes into the air
            // there (airsource.wgsl): a gas painted over a gas joins it, and
            // anything else pushes it into the gas beside it. What is painted
            // holds the open air's at its height (paintedGas); steam is air
            // holding vapour (elements.js), at its own temperature. Gas
            // painted where gas was pushed out and waits to go (a wall
            // painted and erased again before the air has taken it) is that
            // gas again, at its own temperature.
            let was = c;
            if (holdsGas(was.t)) {
                pushGas(i, gasAt(i), was.temp);
            }
            let steam = elem == STEAM;
            c = makeCell(select(elem, AIR, steam), select(B.temp, elems[elem].spawnT, own));
            if (isStructural(elems[c.t])) {
                c.frac.x = SUPPORT_HELD;
            }
            var g = paintedGas(c.t, c.temp, cell.y, steam);
            if (holdsGas(c.t) && !holdsGas(was.t)) {
                let back = Gas(takeSource(i, SOURCE_VAPOUR), takeSource(i, SOURCE_MIST), takeSource(i, SOURCE_LOST),
                    takeSource(i, SOURCE_AMOUNT));
                let heat = takeSource(i, SOURCE_HEAT);
                if (back.amount + back.vapour + back.mist > 0.0) {
                    g = Gas(back.vapour, back.mist, select(0.0, back.lost / back.amount, back.amount > 0.0), back.amount);
                    c.temp = heat / max(gasHeatCap(g), 1e-6);
                }
            }
            // A blast painted is a cell of gunpowder going off there
            // (common.wgsl's detonation).
            if (elem == BLAST) {
                let d = detonation(makeCell(GUNPOWDER, AMBIENT_T), elems[GUNPOWDER]);
                c = d.c;
                g = d.g;
            }
            setGasAt(i, g);
            if (soil) {
                c = withSoilWater(c, extra);
            } else if (extra != 0u) {
                c = withStrength(withSolute(c, extra), strength);
            } else if (holdsPlantWater(elem)) {
                // A plant painted is watered, with sugar to grow on.
                c = withSugar(withPlantWater(c, PLANT_WATER), 128u);
            } else if (holdsHeading(elem)) {
                // Fans blow, and batteries drive current (+ ahead), the
                // way the stroke was heading.
                c = withHeading(c, u32(fract(B.angle / TAU) * 256.0) & 255u);
            }
        }
        case MODE_HEAT: {
            // (Up to 5000 °C: never down, from something hotter, a blast.)
            c.temp = max(c.temp, min(c.temp + B.strength * select(1.0, cover, spray), 5000.0));
        }
        case MODE_COOL: {
            // Down to absolute zero (storedZero). Never warms anything.
            c.temp = min(c.temp, max(c.temp - B.strength * select(1.0, cover, spray), storedZero(c.t)));
        }
        case MODE_WIND: {
            // The wind tool pushes the air (air.wgsl's wind, into its
            // source), which carries the gases and drags what is loose
            // (update.wgsl): nothing here. (It shoved loose cells directly
            // too, grains hopping up the screen whatever the pull: audit 06
            // A16, G4.)
            return;
        }
        case MODE_SOAK: {
            // Soil with water over it, through the soil above (a seabed, a
            // pond's bottom), soaked through, as it would long have been.
            if (!porous(e)) {
                return;
            }
            var y = i32(cell.y) - 1;
            while (y >= 0 && porous(elems[wordType(cells[u32(y) * B.size.x + cell.x].x)])) {
                y--;
            }
            if (y < 0) {
                return;
            }
            let t = wordType(cells[u32(y) * B.size.x + cell.x].x);
            if (t != WATER && t != MUD) {
                return;
            }
            c = withSoilWater(c, SOIL_FULL);
        }
        case MODE_VAPOUR: {
            // A gas that takes vapour in (common.wgsl's holdsVapour) holding
            // B.strength times what saturates it at its temperature (at 1
            // atm); its mist and oxygen as they were, and its pressure: the
            // vapour takes the place of as much of its dry gas.
            if (!holdsVapour(c.t)) {
                return;
            }
            var g = gasAt(i);
            let vapour = B.strength * saturation(tempOf(c));
            g.amount = max(g.amount + vapourAmount(g.vapour) - vapourAmount(vapour), 0.0);
            g.vapour = vapour;
            setGasAt(i, g);
            return;
        }
        case MODE_SPARK: {
            if (holdsCharge(c.t)) {
                if (chargeOf(c) == 0u) {
                    c = withCharge(c, CHARGE_NEW);
                }
            } else if (e.burnTime > 0.0 || e.blast > 0.0) {
                // Lit, as a flame lights it (block.wgsl's light): its life
                // counts what of it burns from now; an explosive goes off in
                // the update pass.
                c.flags |= FL_BURN;
                if (holdsSugar(c.t) || holdsStore(c.t)) {
                    c = withBurnt(c, 0u);
                }
            }
        }
        default: {}
    }
    cells[i] = packCell(c);
}
