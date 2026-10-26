// Element appearance, shared by the renderer and the glow pass.

const BG: vec3<f32> = vec3<f32>(0.045, 0.05, 0.065);
const MIST: vec3<f32> = vec3<f32>(0.72, 0.76, 0.8);
const MIST_THICK: f32 = 6.0 * VAPOUR_20;  // mist (vapour units) that looks as thick as fog gets: six times what saturates air at 20 °C
const BATTERY_RIB: f32 = 3.0;  // cells per - to + rib drawn on batteries
// Current is drawn on a log scale of its density, A per cell width: from
// faint at CURRENT_FAINT (milliamps in brine) to full at CURRENT_FULL (the
// kiloamps that make a thin metal wire glow).
const CURRENT_FAINT: f32 = -4.0;  // log10 A
const CURRENT_FULL: f32 = 3.5;
const CURRENT_COLOR: vec3<f32> = vec3<f32>(0.55, 0.8, 1.0);
const DRY_PLANT: vec3<f32> = vec3<f32>(0.6, 0.52, 0.22);  // a plant about to wither of thirst: straw
// The Lit view (render.wgsl) shows light as the eye takes it in, 1 -
// exp(-EXPOSURE x light): a sunlit field and firelight in a dark cave both
// read.
const EXPOSURE: f32 = 2.5;
const LIT_FLOOR: f32 = 0.04;  // the least anything is seen in, so shapes show in the dark

fn flameColor(heat: f32) -> vec3<f32> {
    return mix(vec3<f32>(0.7, 0.1, 0.02), vec3<f32>(1.0, 0.82, 0.3), clamp(heat, 0.0, 1.0));
}

fn flicker(c: Cell, time: f32) -> f32 {
    return 0.8 + 0.2 * sin(time * 17.0 + f32(c.shade) * 1.7);
}

// Rough incandescence ramp: dark red -> orange -> yellow -> white.
fn blackbody(t: f32) -> vec3<f32> {
    let x = clamp((t - 450.0) / 1800.0, 0.0, 1.0);
    let a = vec3<f32>(0.55, 0.04, 0.0);
    let b = vec3<f32>(1.0, 0.32, 0.02);
    let c = vec3<f32>(1.0, 0.78, 0.32);
    let d = vec3<f32>(1.0, 0.97, 0.88);
    if (x < 0.33) {
        return mix(a, b, x / 0.33);
    }
    if (x < 0.66) {
        return mix(b, c, (x - 0.33) / 0.33);
    }
    return mix(c, d, (x - 0.66) / 0.34);
}

fn glowAmount(t: f32) -> f32 {
    return smoothstep(450.0, 1100.0, t);
}

// The share brighter or darker that shade (0..SHADES - 1) makes a cell of an
// element varying by colorVar.
fn shadeVariation(shade: u32, colorVar: f32) -> f32 {
    return (f32(shade) / f32(SHADES - 1u) - 0.5) * 2.0 * colorVar;
}

// A shade from a cell's position, for a cell whose shade holds something else.
fn cellHash(pos: vec2<f32>) -> u32 {
    var h = (u32(pos.x) * 0x8da6b343u) ^ (u32(pos.y) * 0xd8163841u);
    h ^= h >> 15u;
    h *= 0x2c1b3c6du;
    h ^= h >> 12u;
    return h & (SHADES - 1u);
}

// Water is tinted by what it carries, the more of it the more (solColor). A
// plant yellows as it dries out (its water: PLANT_WATER turgid), and varies
// by where it is (cellColor), not by its shade. (Its shade's low two bits,
// 0 to 3, taken as 0 to SHADES - 1, drew every plant 9 to 15 % darker.)
fn baseColor(c: Cell) -> vec3<f32> {
    let e = elems[c.t];
    var rgb = vec3<f32>(e.r, e.g, e.b);
    if (holdsSolute(c.t) && soluteOf(c) != 0u) {
        let s = elems[soluteOf(c)];
        rgb = mix(rgb, vec3<f32>(s.solR, s.solG, s.solB), shareOf(strengthOf(c)));
    }
    if (holdsPlantWater(c.t)) {
        let dry = clamp((f32(PLANT_WATER) - f32(plantWaterOf(c))) / f32(PLANT_WATER - PLANT_WILT / 2u), 0.0, 1.0);
        return mix(rgb, DRY_PLANT, dry * dry);
    }
    return rgb * (1.0 + shadeVariation(c.shade, e.colorVar));
}

// Current flowing through a conductor (density in A per cell width, along
// the conventional current, + to -): a faint blue-white tint, the stronger
// the denser the current, with bright dashes drifting along with it, the
// faster the stronger it is.
fn currentColor(col: vec3<f32>, density: vec2<f32>, pos: vec2<f32>, time: f32) -> vec3<f32> {
    let amps = length(density);
    if (amps <= 0.0) {
        return col;
    }
    let level = clamp((log2(amps) * 0.30103 - CURRENT_FAINT) / (CURRENT_FULL - CURRENT_FAINT), 0.0, 1.0);
    let wave = 0.5 + 0.5 * sin(dot(pos, density / amps) * 1.2 - time * (3.0 + 9.0 * level));
    let dash = wave * wave * wave;
    return mix(col, CURRENT_COLOR, level * (0.15 + 0.45 * dash));
}

// Steam is the air's vapour (REWRITE_PLAN.md 4c), drawn as Steam's colour
// (elements.js) by how much of the gas is vapour: none at the outdoor air's,
// most of the way where it is all vapour (and denser for more of it); a
// cell holding next to nothing (one a liquid has just left) as faint as
// what little it holds.
fn steamColor() -> vec3<f32> {
    let e = elems[STEAM];
    return vec3<f32>(e.r, e.g, e.b);
}

fn steamHaze(g: Gas) -> f32 {
    let vapour = vapourAmount(g.vapour);
    let share = vapour / max(vapour + g.amount, 1e-6);
    return 0.42 * smoothstep(0.05, 0.6, share) * min(vapour + g.amount, 1.0) + 0.45 * clamp((vapour - 1.0) / 7.0, 0.0, 1.0);
}

// pos is the cell position, for patterns that run across neighboring cells;
// current is the cell's current density (zero where none flows); base is
// its colour as baseColor gives it, or for a shiny cell, what it shows of
// the light around it (render.wgsl); g, what it holds of the air (the gas
// plane, common.wgsl).
fn cellColor(cell: Cell, pos: vec2<f32>, time: f32, current: vec2<f32>, base: vec3<f32>, g: Gas) -> vec3<f32> {
    var c = cell;
    let e = elems[c.t];
    var col = base;
    switch c.t {
        case AIR: {
            // Faint tint so hot and cold air currents are visible, steam
            // (air mostly vapour) as a haze, and mist as fog.
            col = BG
                + vec3<f32>(0.22, 0.06, 0.015) * smoothstep(120.0, 1200.0, c.temp)
                + vec3<f32>(0.03, 0.07, 0.14) * (1.0 - smoothstep(-80.0, 0.0, c.temp));
            col = mix(col, steamColor(), steamHaze(g));
            col = mix(col, MIST, clamp(g.mist / MIST_THICK, 0.0, 0.85));
        }
        case FIRE: {
            col = flameColor(f32(lifetimeOf(c)) / 45.0) * flicker(c, time);
        }
        case BLAST: {
            // An explosion's gas: white-hot to orange as it cools.
            col = mix(vec3<f32>(1.0, 0.55, 0.15), vec3<f32>(1.0, 0.97, 0.8), clamp((c.temp - FIREBALL_T) / 3000.0, 0.0, 1.0));
        }
        case SMOKE: {
            col = mix(BG, col, clamp(f32(lifetimeOf(c)) / 90.0, 0.15, 0.85));
        }
        case GLASS, SHARDS: {
            col = mix(BG, col, 0.6);
        }
        case FAN: {
            // Bands drifting in the blowing direction.
            let angle = f32(headingOf(c)) / 256.0 * 6.2831853;
            let along = dot(pos, vec2<f32>(cos(angle), sin(angle)));
            col *= 0.75 + 0.25 * sin(along * 1.8 - time * 14.0);
        }
        case BATTERY: {
            // Ribs across its polarity, each shading from a dark, bluish -
            // side to a bright, reddish + side, like a row of cells.
            let angle = f32(headingOf(c)) / 256.0 * 6.2831853;
            let along = dot(pos + 0.5, vec2<f32>(cos(angle), sin(angle)));
            let ramp = fract(along / BATTERY_RIB);
            col *= 0.35 + 0.85 * ramp;
            col = mix(col, vec3<f32>(0.2, 0.3, 0.85), 0.55 * (1.0 - ramp) * (1.0 - ramp));
            col = mix(col, vec3<f32>(1.0, 0.3, 0.15), 0.5 * ramp * ramp);
        }
        case HYDROGEN: {
            // Thin and pale against the sky; in a liquid, a bright bubble.
            col = mix(BG, col, 0.55);
        }
        case BLACK_HOLE: {
            col = vec3<f32>(0.02, 0.0, 0.04) + vec3<f32>(0.12, 0.03, 0.2) * (0.5 + 0.5 * sin(time * 4.0 + f32(c.shade)));
        }
        case WATER, SALTWATER, ACID, OIL, NITRO: {
            col *= 0.94 + 0.06 * sin(time * 2.5 + f32(c.shade) * 0.8);
        }
        case PLANT: {
            // Its shade's top bits are the way it leans (common.wgsl's
            // PLANT_LEFT): it varies by where it is instead.
            c.shade = cellHash(pos);
            col *= 1.0 + shadeVariation(c.shade, e.colorVar);
            if (sugarOf(c) > 0u) {
                col = mix(col, vec3<f32>(0.55, 0.9, 0.35), 0.5);
            }
        }
        case CLONE: {
            if (copiesOf(c) != 0u) {
                let s = elems[copiesOf(c)];
                col = mix(col, vec3<f32>(s.r, s.g, s.b), 0.35);
            }
        }
        case VOID: {
            col *= 0.7 + 0.3 * sin(time * 3.0 + f32(c.shade));
        }
        default: {}
    }
    // Damp grains (their soil water) look darker, as wet sand does.
    if (holdsSoilWater(c.t)) {
        col *= 1.0 - 0.35 * shareOf(soilWaterOf(c));
    }
    if (holdsCharge(c.t) && chargeOf(c) > 0u) {
        if (chargeOf(c) >= CHARGE_ACTIVE) {
            col = mix(col, vec3<f32>(1.0, 0.95, 0.65), 0.85);
        } else {
            col = mix(col, vec3<f32>(0.5, 0.72, 1.0), 0.35);
        }
    }
    if ((c.flags & FL_BURN) != 0u) {
        col = mix(col, flameColor(f32(c.shade) / f32(SHADES - 1u)), 0.6 * flicker(c, time));
    }
    // Under the incandescence, so a wire the current makes red-hot looks it.
    col = currentColor(col, current, pos, time);
    if (e.kind != KIND_GAS) {
        let t = tempOf(c);
        col = mix(col, blackbody(t), glowAmount(t) * 0.9);
    }
    return col;
}

// Light a cell gives off, fed into the bloom.
fn emission(c: Cell) -> vec3<f32> {
    let e = elems[c.t];
    var em = vec3<f32>(0.0);
    if (c.t == FIRE) {
        em = flameColor(f32(lifetimeOf(c)) / 45.0) * 0.8;
    } else if (c.t == BLAST) {
        em = vec3<f32>(1.0, 0.7, 0.35) * 1.5;
    } else if (c.t == AIR) {
        em = vec3<f32>(0.6, 0.15, 0.02) * smoothstep(400.0, 1600.0, c.temp) * 0.15;
    } else if (e.kind != KIND_GAS) {
        let t = tempOf(c);
        em = blackbody(t) * glowAmount(t) * 0.8;
    }
    if ((c.flags & FL_BURN) != 0u) {
        em += flameColor(0.6) * 0.5;
    }
    if (holdsCharge(c.t) && chargeOf(c) >= CHARGE_ACTIVE) {
        em += vec3<f32>(1.0, 0.9, 0.5) * 0.6;
    }
    if (holdsSolute(c.t) && soluteOf(c) == ACID) {
        em += vec3<f32>(0.15, 0.5, 0.05) * 0.1 * shareOf(strengthOf(c));
    } else if (c.t == BLACK_HOLE) {
        em += vec3<f32>(0.4, 0.12, 0.7) * 0.5;
    }
    return em;
}
