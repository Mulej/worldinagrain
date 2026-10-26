// Element definitions. This is the single source of truth: the table is packed
// into a GPU buffer, and the element ids plus the Elem struct are emitted as a
// WGSL prelude so shaders and JS can never disagree on numbering or layout.

import { REACTIONS } from "./reactions.js";
import { AIR_GRAMS, AIR_OXYGEN, CELL_M, GAS_COMPRESSION, GAS_HEAT_CAP, NO_ELEM, WATER_VAPOUR } from "./shared.js";

export const KIND = { SOLID: 1, POWDER: 2, LIQUID: 3, GAS: 4 };

export const FLAG = {
    CONDUCT: 1,        // carries a spark's static pulse: conductors with sigma at least SPARK_SIGMA
    ACID_PROOF: 2,     // not dissolved by acid
    INDESTRUCTIBLE: 4, // ignores void, acid and pressure
    FIXED_TEMP: 8,     // temperature pinned to fixedT
    AIR_PASS: 16,      // solid that air flows through (fans, black holes)
    IMPACT: 32,        // ignites when it hits something hard
    FLUFF: 64,         // powder too light to hold up a solid resting on it
    RESIDUE: 128,      // its water boils off as steam into open air, leaving hiTo (block.wgsl)
    BRITTLE: 256,      // structural solid that shatters right through on a hard landing (bodies.wgsl)
    OWES: 512,         // a form a cell of another becomes more than one of: the cells it owes in its life (formed in common.wgsl)
};

export const GROUPS = ["Basic", "Thermal", "Combustion", "Chemistry & Life", "Exotic"];

const T_NEVER = 1e9;
// Earth's gravity in cells per step², the default weight of powders and
// liquids: 9.81 m/s² at 60 steps a second and a centimetre a cell. Lighter
// weights (light grains held back by the air, gases floating up in it) are
// shares of it, as their drag and buoyancy scale with it.
const GRAVITY = 9.81 / 3600 / 0.01;
// A gas's heat capacity, J/K a cell at 1 atm, is GAS_HEAT_CAP (js/shared.js):
// compressed with the gas, some 210 times real gas's density (0.25 J/K a
// cubic centimetre, as a steam unit's is). Compressed alike, the latent heat
// the air's moisture gives off condensing heats it as it would real air:
// saturated air cooled a degree gives off about 2.2 times the heat it lost
// (Wallace & Hobbs, Atmospheric Science, 2006, ch. 3), so fog forms at its
// real pace (REWRITE_PLAN.md decision 3).
// The least conductance (S per cell) that carries a spark's pulse. Metals and
// electrolytes do; fresh water, mud and the like barely conduct and don't.
// Batteries are left out whatever their sigma: their aux holds their polarity,
// not a charge.
const SPARK_SIGMA = 0.01;
// Conductance (S) words describing it start at, best first: metal, molten
// metal, scrap and electrolytes, fresh water and mud.
const CONDUCTION_WORDS = [[1e4, "well"], [2000, "moderately"], [SPARK_SIGMA, "poorly"], [0, "barely"]];

// Field order defines the GPU struct layout. Every field is 4 bytes.
const FIELDS = [
    ["kind", "u32"],
    // What sinks through what, g/cm³ × 100 (water 100): a powder's grains'
    // (see rho, and the check where rho is worked out).
    ["density", "f32"],
    // Its real density, kg/m³ (RHO, or a substance's fill): what the air's
    // pressure and drag do to it go by it (REWRITE_PLAN.md 4e).
    ["rho", "f32"],
    ["weight", "f32"],    // vertical acceleration, cells/step²; negative rises
    ["drag", "f32"],      // how strongly wind carries it, 0..1 per step
    ["spread", "f32"],    // powders: slide chance; gases: turbulence
    // Liquids: kinematic viscosity, cells² per step. Velocity diffuses
    // through the liquid this fast and stops against walls, so thick
    // liquids heap up and creep instead of running flat (fluid.wgsl). Each
    // liquid's real one at a centimetre a cell, ν dt / dx² (dt 1/60 s), per
    // cell at any other scale (REWRITE_PLAN.md stage 2's table); none for
    // those no thicker than water (water's own, 1e-6 m²/s, is 1.7e-4 here,
    // which the grid's flow doesn't resolve): REWRITE_PLAN.md 7-14. That at
    // viscosityT (°C), thicker colder and runnier hotter by e every
    // viscosityE kelvin (none: the same at any temperature; common.wgsl's
    // viscosityAt, 7-17).
    ["viscosity", "f32"],
    ["viscosityT", "f32"],
    ["viscosityE", "f32"],
    // Heat. A cell's temperature changes by the energy it gains over heatCap.
    // thermal is the material's real thermal conductivity, W/(m·K): two
    // cells pass heat through their halves in series, by how long a step
    // stands for (heat's clock, clock.wgsl). A gas against anything else
    // passes it through a film instead: film, W/(m²·K) (still air's few,
    // condensing steam's far more). heatCap is the real volumetric heat
    // capacity but for gases, whose is compressed with them (GAS_HEAT_CAP):
    // a powder's its pile's, its grains' specific heat over its rho, not its
    // grains' density (REWRITE_PLAN.md 9-5: sawdust 0.21 × 2.0 J/(g·K) as wood
    // here, charcoal 0.25 × 1.0, coal 0.83 × 1.26, salt 1.2 × 0.86, wood ash
    // 0.6 × 0.8; Incropera's table A.3 and the CRC Handbook's ranges).
    ["thermal", "f32"],
    ["film", "f32"],
    ["heatCap", "f32"],   // volumetric heat capacity, J/(cm³·K)
    ["ambient", "f32"],   // gases under open sky: share of the gap to 20 °C mixed away per step of the normal day's
    // Phase changes. Past hiT (or loT) a cell holds at that temperature while
    // it takes in latentUp (gives off latentDown), in J/cm³, then becomes
    // hiTo (loTo). See changePhase in update.wgsl. For the forms of a
    // substance (SUBSTANCES) these follow from its latent heats per gram and
    // the form's fill, the grams a cell of it holds, so a cell changing form
    // keeps its mass and energy (formed in common.wgsl); 0 for the rest,
    // which change cell for cell.
    ["fill", "f32"],
    ["hiT", "f32"],
    ["hiTo", "u32"],
    ["latentUp", "f32"],
    ["loT", "f32"],
    ["loTo", "u32"],
    ["latentDown", "f32"],
    // Derived, for the temperature a cell acts at (effTempIn in common.wgsl):
    // how far past hiT (below loT) the stored temperature runs while it takes
    // in latentUp (gives off latentDown), huge without hiTo (loTo); and how
    // much faster than the stored temperature it cools once condensed, as
    // what it condenses into, gram for gram (heatCap over the grams' heatCap);
    // and its stored temperature at absolute zero, below -273 by the latent
    // heat it gives off on the way.
    ["upSpan", "f32"],
    ["downSpan", "f32"],
    ["downScale", "f32"],
    ["zero", "f32"],
    // A gas's molar mass, g/mol (air.wgsl's buoyancy: the dry gas in a cell
    // marked as it is this; the air, 29).
    ["molar", "f32"],
    // Burning (REWRITE_PLAN.md stage 5; js/shared.js's OXYGEN_HEAT). Past
    // igniteT, with gas beside it a flame can burn in, it burns: at each face
    // to such gas it takes the oxygen it burns from it, oxygenNeed cells of
    // fresh air's for a cell of it (burnO2, the grams of oxygen a gram of it
    // burns with, by the grams a cell holds), as fast as one face of fresh
    // air burns a cell of it through in burnTime real seconds (FIRE_PACE
    // times faster here), and gives that gas the heat, oxygenHeat a cell of
    // oxygen (burnHeat a gram: Huggett's 13.1 kJ but for hydrogen), and the
    // water it makes, burnVapour vapour units a cell of oxygen (burnWater,
    // grams of water a gram of oxygen; pairs.wgsl). Its life counts what of
    // it has burnt, to 255, and then it is burnTo (a gas, hydrogen, burns
    // away what it holds). A flame touching it lights it as soon as its
    // surface would reach igniteT (block.wgsl). burnFire: the chance per
    // contact that burning, it shows a flame in the hot gas beside it.
    ["igniteT", "f32"],
    ["burnTime", "f32"],
    ["oxygenNeed", "f32"],
    ["burnVapour", "f32"],
    ["oxygenHeat", "f32"],
    ["burnTo", "u32"],
    ["burnFire", "f32"],
    // Char (REWRITE_PLAN.md 5c). What chars (wood, sawdust, plants, seeds)
    // burns its volatiles first, oxygenNeed's share but what its char will
    // burn with: at a life of charAt it is charTo (charcoal), charYield cells
    // of it a cell (charMass, its grams a gram), at random in that share and
    // the gas its volatiles leave otherwise. Held hot without a flame, it
    // chars as fast as its pyrolysis goes (update.wgsl), its volatiles going
    // unburnt, charGas cells of gas (smoke). What has burnt through is
    // burnTo, ashYield cells of it a cell (ashMass, its grams a gram), or
    // the gas its burning made; a gas burnTo (oil's smoke) always.
    ["charTo", "u32"],
    ["charYield", "f32"],
    ["charAt", "f32"],
    ["charGas", "f32"],
    ["ashYield", "f32"],
    // Explosives (REWRITE_PLAN.md 5b): lit, or past igniteT, a cell goes off
    // into the gas it makes, blastGas cells of it (gasYield, cm³ a gram at
    // 0 °C and 1 atm, compressed as the world's gas is) and blastVapour
    // units of water (blastWater, grams a gram), holding blast joules (its
    // heat of explosion, blastHeat J a gram); 0 for what doesn't explode.
    ["blast", "f32"],
    ["blastGas", "f32"],
    ["blastVapour", "f32"],
    ["hardness", "f32"],  // resistance to acid
    ["breakP", "f32"],    // solids: shatter into breakTo above this pressure
    ["breakTo", "u32"],
    ["crushP", "f32"],    // loose structural solids: crumble into breakTo under this much piled powder (update.wgsl)
    // Structural solids standing as structure (stress.wgsl, REWRITE_PLAN.md
    // 8-2): Young's modulus and the strengths in tension and in compression,
    // Pa (STRENGTH). They take over from breakP and crushP for structure;
    // those stay for loose cells. And its linear
    // thermal expansion, 1/K (8-5): heated or cooled against what holds it,
    // it is stressed (thermal shock). And how it weakens when hot (8-7,
    // common.wgsl's hotShare): its strengths whole up to hotStart (°C), half
    // at hotHalf, and less past that. And the stress an impact breaks it at,
    // Pa (bodies.wgsl's landings, REWRITE_PLAN.md P3-8): a strength under a
    // blow, from what breaks it so, not a static one.
    ["youngs", "f32"],
    ["tensile", "f32"],
    ["compressive", "f32"],
    ["expansionLinear", "f32"],
    ["hotStart", "f32"],
    ["hotHalf", "f32"],
    ["impact", "f32"],
    ["flags", "u32"],
    ["spawnT", "f32"],
    ["fixedT", "f32"],
    // Powders: their grains' size, mm. Liquid flowing over a bed picks
    // grains up once it outruns a speed that grows with their size and
    // their excess density over the liquid (Shields: update.wgsl's
    // shieldsSpeed), so fine soil washes away first and gravel only in a
    // flood. Medium sand 0.5 (0.25-0.5, Wentworth's scale), fine gravel 6
    // (4-8), soil's aggregates 0.2, table salt's crystals 0.4, crushed coal
    // and glass 2, scrap 3, wood ash 0.05, a wheat seed 4 (estimates within
    // those ranges).
    ["grain", "f32"],
    // Powders: how much moisture binds the grains (0 none, 1 a damp grain
    // never slides): damp sand holds a steep face, dry or soaked sand doesn't.
    ["cohesion", "f32"],
    // Powders: their friction, grain on grain, tan of their angle of repose
    // (REPOSE): resting, a grain stays put while what pushes it along the
    // ground is within this times what presses it down, and slides past
    // that, slowing by as much (update.wgsl, block.wgsl's slide).
    ["friction", "f32"],
    // Electricity (electric.wgsl). sigma is a cell's electrical conductance
    // between opposite faces, in siemens; 0 is an insulator. A cell is taken
    // to be a centimetre cube, so it is the bulk conductivity in S/cm (real
    // iron is about 1e5, sea water 0.05; metals are compressed down here the
    // way heat ratios are). resistT: share by which its resistance rises per
    // °C above 20 °C (metals about 0.005), so a hot wire carries less. emf:
    // volts a cell drives current with, along the direction stored in its
    // aux (batteries). electrolysis: an electrolyte's cells split into
    // hydrogen per coulomb of current leaving them into an electrode (a
    // conductor that isn't an electrolyte), WATER_HYDROGEN cells of it a cell
    // of water (Faraday's mole for mole), its solute handed on to the water
    // beside it first (pairs.wgsl); 0 for anything else. Faraday's law has a
    // coulomb split 9.3e-5 of a cell; water's 40 is the pace a battery's few
    // amps bubble hydrogen at here, as fire's is (FIRE_PACE).
    ["sigma", "f32"],
    ["resistT", "f32"],
    ["emf", "f32"],
    ["electrolysis", "f32"],
    // Liquids: thermal expansion. Density falls by `expansion` per °C above
    // the liquid's reference temperature (20 °C, or its freezing point for a
    // melt), and by `expansion2` per °C² on top: water's pair puts its
    // densest at 4 °C, as in real water. Warm liquid beside cold is lighter,
    // so heated liquid rises and convects (fluid.wgsl).
    ["expansion", "f32"],
    ["expansion2", "f32"],
    // Boiling under pressure: latent heat of vaporisation over the gas
    // constant (L/R, in kelvin; water's is about 4890), for a liquid whose hiT
    // is its boiling point or a vapour whose loT is its condensing point. Its
    // boiling point then moves with the air pressure (transitionsAt in
    // common.wgsl); 0 keeps it fixed.
    ["boilShift", "f32"],
    // Dissolving in water (REWRITE_PLAN.md 5e; pairs.wgsl). A cell of water
    // carries one solute: its element in aux, how much in life, 255 being
    // saturated (or, for acid, full strength). solubility: cells of it that
    // saturate a cell of water (0: it doesn't dissolve, or mixes with water as
    // a liquid, like acid); dissolveRate: how fast a grain dissolves into the
    // water beside it, m/s, as the water is short of saturation (Noyes &
    // Whitney 1897: the solute's diffusivity over the film it crosses);
    // solD: its diffusivity in water, m²/s, how fast it evens out between two
    // cells of water (Fick). At saturation, the water is solDensity denser (a
    // share of water's density), freezes solFreeze and boils solBoil °C off
    // water's points, conducts solSigma more and is tinted solColor. An acid
    // holds acidEq equivalents a cell at full strength, and eats what reacts
    // with it (acidDemand) at solCorrode a pairing at full strength over its
    // hardness + 1, using itself up by the equivalents it eats. With its
    // water gone, what's left is solResidue: the solute (salt crystallises),
    // or fumes.
    ["solubility", "f32"],
    ["dissolveRate", "f32"],
    ["solD", "f32"],
    ["acidEq", "f32"],
    ["solDensity", "f32"],
    ["solFreeze", "f32"],
    ["solBoil", "f32"],
    ["solSigma", "f32"],
    ["solCorrode", "f32"],
    ["solR", "f32"],
    ["solG", "f32"],
    ["solB", "f32"],
    ["solResidue", "u32"],
    // What an acid eats: acidDemand equivalents of it to a cell (acidGrams,
    // the grams an equivalent eats: iron 27.9 as Fe2+, a carbonate 50), giving
    // off acidHeat J an equivalent and acidGas (hydrogen from a metal, carbon
    // dioxide, as air holding no oxygen, from a carbonate), half a mole an
    // equivalent (GAS_PER_EQUIVALENT); 0 for what it doesn't react with.
    ["acidDemand", "f32"],
    ["acidHeat", "f32"],
    ["acidGas", "u32"],
    // Soil water (REWRITE_PLAN.md 5f; porous powders, pairs.wgsl): how
    // tightly its pores hold water and how fast it flows through them, by van
    // Genuchten's curves (1980): soilR, the water it never gives up (a share
    // of the cell, its residual content; a soaked cell holds SOIL_HOLD),
    // soilAlpha (1/m) and soilN their shape, soilK its conductivity soaked
    // (m/s).
    ["soilR", "f32"],
    ["soilAlpha", "f32"],
    ["soilN", "f32"],
    ["soilK", "f32"],
    // And how well it conducts heat as wet as it is, by Johansen's model
    // (1975; Farouki, Thermal Properties of Soils, 1981): its thermal
    // conductivity dry (thermal) and soaked (soilKSat, its grains' and the
    // water's geometric mean by its pores), between them by its Kersten
    // number, 1 + soilKe log10 Sr (Sr its saturation; 0.7 for coarse soil, 1
    // for fine). (Its heat capacity counts the water it holds too:
    // common.wgsl's cellHeatCap.)
    ["soilKSat", "f32"],
    ["soilKe", "f32"],
    // Surface tension (fluid.wgsl's surface pass), which only shows in a
    // world of millimetre cells. tension: a liquid's, as a share of water's
    // (72 mN/m; CRC Handbook). wetting: how far liquid creeps onto a solid or
    // powder, (1 + cos θ) / 2 for its contact angle θ with water: 1 for
    // clean glass, which water spreads over and climbs; 0.5 for 90°; lower
    // for waxy leaves, which it beads on.
    ["tension", "f32"],
    ["wetting", "f32"],
    // Light and radiant heat (light.wgsl). clarity: the share of visible
    // light (sunlight, flames, glowing matter) a centimetre cube of it lets
    // through (from its extinction, see OPTICS); irClarity
    // the same for thermal infrared, what everything gives off at its
    // temperature (glass lets light in but not the heat back out). What a
    // cell doesn't let through it absorbs, or reflects: emissivity is the
    // share of thermal infrared it absorbs, and gives off as much of what a
    // perfect emitter at its temperature would (Kirchhoff's law); the rest it
    // reflects. Visible light is reflected in the cell's colour: scattered
    // every way, but by a polished surface (metal), its shine of it as a
    // mirror does, tinted by its colour.
    ["clarity", "f32"],
    ["irClarity", "f32"],
    ["emissivity", "f32"],
    ["shine", "f32"],
    ["r", "f32"],
    ["g", "f32"],
    ["b", "f32"],
    ["colorVar", "f32"],
];

const DEFAULTS = {
    group: null,
    note: "",
    kind: KIND.SOLID,
    density: 1000,
    rho: null,
    weight: GRAVITY,
    drag: 0.03,
    spread: 0,
    viscosity: 0,
    viscosityT: 20,
    viscosityE: 0,
    thermal: 1,
    film: 5,
    heatCap: 2,
    ambient: 0,
    hiT: T_NEVER,
    hiTo: null,
    fill: 0,
    latentUp: 0,
    loT: -T_NEVER,
    loTo: null,
    latentDown: 0,
    molar: 29,
    igniteT: T_NEVER,
    burnTime: 0,
    burnO2: 0,
    burnWater: 0,
    burnHeat: 13.1e3,
    burnTo: "AIR",
    burnFire: 0,
    charTo: null,
    charMass: 0,
    ashMass: 0,
    blastHeat: 0,
    gasYield: 0,
    blastWater: 0,
    hardness: 2,
    breakP: T_NEVER,
    breakTo: null,
    crushP: T_NEVER,
    youngs: 0,
    tensile: 0,
    expansionLinear: 0,
    hotStart: T_NEVER,
    hotHalf: T_NEVER,
    impact: T_NEVER,
    compressive: 0,
    flags: 0,
    spawnT: 20,
    fixedT: 0,
    grain: 0.5,
    cohesion: 0,
    sigma: 0,
    resistT: 0,
    emf: 0,
    electrolysis: 0,
    expansion: 0,
    expansion2: 0,
    boilShift: 0,
    solubility: 0,
    dissolveRate: 0,
    solD: 0,
    acidEq: 0,
    acidGrams: 0,
    acidHeat: 0,
    acidGas: null,
    soilR: 0,
    soilAlpha: 0,
    soilN: 0,
    soilK: 0,
    soilKSat: 0,
    soilKe: 0,
    solDensity: 0,
    solFreeze: 0,
    solBoil: 0,
    solSigma: 0,
    solCorrode: 0,
    solColor: [0, 0, 0],
    solResidue: null,
    tension: 0,
    wetting: 0.5,
    // A preset painted as another element carrying this solute (sim.js
    // paint: saturated, or as strong as the brush's moist says): salt water
    // is water with salt in it. JS only.
    paintAs: null,
    paintSolute: null,
    // Light and radiant heat: opaque to both bands (extinction per metre,
    // see OPTICS; clarity and irClarity follow from it), and a good emitter,
    // as most non-metals are (0.9 or so).
    extinction: Infinity,
    irExtinction: Infinity,
    clarity: 0,
    irClarity: 0,
    emissivity: 0.9,
    shine: 0,
    color: [1, 0, 1],
    colorVar: 0.08,
    paint: null, // brush fill chance per frame; null = derived from kind
};

const DEFS = [
    {
        // Real air holds about 1/3500 of water's heat per volume; it's
        // compressed here so air still carries some heat. Thermal
        // conductivities here are real ones (Engineering ToolBox's tables,
        // Incropera's Fundamentals of Heat and Mass Transfer, appendix A),
        // for room temperature; a surface's film in still air, 5 W/(m²·K)
        // (Incropera's table 1.1: 2-25 for free convection in gases).
        id: "AIR", name: "Air",
        kind: KIND.GAS, density: 1, weight: 0, drag: 0, thermal: 0.026, film: 5, heatCap: GAS_HEAT_CAP, ambient: 0.003, hardness: 0,
        color: [0.045, 0.05, 0.065], colorVar: 0,
    },

    // Basic
    {
        id: "WALL", name: "Wall", group: "Basic", note: "Indestructible, blocks heat and air",
        thermal: 0, flags: FLAG.INDESTRUCTIBLE | FLAG.ACID_PROOF | FLAG.FIXED_TEMP, fixedT: 20, wetting: 0.8,
        color: [0.4, 0.41, 0.45], colorVar: 0.04,
    },
    {
        id: "SAND", name: "Sand", group: "Basic", note: "Holds a steep face when damp; flowing water carries it off",
        kind: KIND.POWDER, density: 265, spread: 0.9, thermal: 0.3, heatCap: 1.3, hardness: 1, cohesion: 0.95, wetting: 0.95,
        hiT: 1250, hiTo: "MOLTEN_GLASS", latentUp: 220,
        color: [0.87, 0.75, 0.48], colorVar: 0.1,
    },
    {
        // Fresh water barely conducts (tap water's 0.002 S/m or so; pure water
        // a thousandth of that): what current gets through splits it, but far
        // too slowly to see. Salt makes it an electrolyte.
        // Its expansion fits real water's densities (CRC Handbook): densest
        // at 4 °C, 0.1 % lighter at 20 °C and 4 % lighter at 100 °C. So the
        // 4 °C water sinks to the bottom of a freezing lake, and the lake
        // freezes from the top down.
        id: "WATER", name: "Water", group: "Basic",
        kind: KIND.LIQUID, density: 100, drag: 0.03, thermal: 0.6, heatCap: 4.18, hardness: 2,
        // Boils into vapour in the air in its cell (update.wgsl changePhase).
        hiT: 100, hiTo: "AIR", latentUp: 2257,
        loT: 0, loTo: "ICE", latentDown: 334,
        flags: FLAG.ACID_PROOF, sigma: 2e-5, electrolysis: 40, expansion: 1.42e-4, expansion2: 4.45e-6, boilShift: 4890,
        tension: 1,
        color: [0.16, 0.42, 0.86], colorVar: 0.04,
    },
    {
        id: "STONE", name: "Stone", group: "Basic",
        density: 300, thermal: 2.5, heatCap: 2.2, hardness: 6, breakP: 50, breakTo: "GRAVEL", crushP: 150,
        wetting: 0.9,
        hiT: 1000, hiTo: "LAVA", latentUp: 1200,
        color: [0.47, 0.45, 0.44], colorVar: 0.1,
    },
    {
        id: "GRAVEL", name: "Gravel", group: "Basic",
        kind: KIND.POWDER, density: 265, drag: 0.02, spread: 0.5, thermal: 0.7, heatCap: 1.6, hardness: 5, grain: 6,
        hiT: 1000, hiTo: "LAVA", latentUp: 800,
        color: [0.52, 0.5, 0.48], colorVar: 0.22,
    },
    {
        id: "DIRT", name: "Dirt", group: "Basic", note: "Soaks up water; soaked through, turns it to mud",
        kind: KIND.POWDER, density: 260, spread: 0.4, thermal: 0.6, heatCap: 1.3, hardness: 1, grain: 0.2, cohesion: 0.8,
        wetting: 0.95,
        color: [0.42, 0.29, 0.18], colorVar: 0.12,
    },

    // Thermal
    {
        // Water grows 11 % as it freezes, a cell of it into 1.11 cells of ice,
        // so each takes 300 J/cm³ to melt to water's 334 given off freezing
        // (formed in common.wgsl). Snow is half a cell of water.
        id: "ICE", name: "Ice", group: "Thermal", note: "Water expands as it freezes: a full bottle in the cold bursts",
        density: 90, thermal: 2.2, heatCap: 1.9, hardness: 3, spawnT: -30, breakP: 20, breakTo: "SNOW", crushP: 40,
        flags: FLAG.BRITTLE,
        hiT: 0, hiTo: "WATER", latentUp: 300,
        color: [0.68, 0.86, 0.98], colorVar: 0.05,
    },
    {
        id: "SNOW", name: "Snow", group: "Thermal",
        kind: KIND.POWDER, density: 92, weight: GRAVITY / 2, drag: 0.15, spread: 0.3, thermal: 0.15, heatCap: 1.0,
        hardness: 0.5, spawnT: -15,
        hiT: 0, hiTo: "WATER", latentUp: 167,
        color: [0.94, 0.96, 1.0], colorVar: 0.03,
    },
    {
        // Steam is the air's vapour (REWRITE_PLAN.md 4c): water boils into it
        // in the cell it was in, a cell's worth of water VAPOUR_EXPANSION
        // times as much gas as a cell of air (js/shared.js), which the air's
        // flow spreads. Painted (brush.wgsl), a cell of air all of it vapour
        // at the open air's pressure, at spawnT; there are no cells of Steam.
        // Its film is pure steam's condensing on a cold surface (a few
        // thousand W/(m²·K)), what vapour condensing out of any gas comes to
        // with no air in it (common.wgsl gasFilm).
        id: "STEAM", name: "Steam", group: "Thermal", note: "Water boils into 8 times its volume; confined, it builds pressure",
        kind: KIND.GAS, density: 0.4, weight: 0, thermal: 0.025, film: 3000, heatCap: GAS_HEAT_CAP, hardness: 0, spawnT: 140,
        molar: 18,
        color: [0.78, 0.8, 0.85], colorVar: 0.08,
    },
    {
        id: "LAVA", name: "Lava", group: "Thermal",
        kind: KIND.LIQUID, density: 250, drag: 0.005, viscosity: 50, thermal: 1.5, heatCap: 3.3, hardness: 3,
        // basalt: some 800 Pa·s at 1150 °C, tenfold for each 115 K it cools
        // near its liquidus (Giordano, Russell & Dingwell 2008's model)
        viscosityT: 1150, viscosityE: 50,
        spawnT: 1400, expansion: 1e-4,  // silicate melts: roughly 1e-4 per °C
        tension: 5,  // silicate melts: about 0.35 N/m
        loT: 1000, loTo: "STONE", latentDown: 1200,
        color: [0.95, 0.32, 0.05], colorVar: 0.1,
    },
    {
        id: "GLASS", name: "Glass", group: "Thermal", note: "Shatters easily under pressure, or heated or cooled too suddenly",
        density: 250, thermal: 1.0, heatCap: 2.1, hardness: 3, flags: FLAG.ACID_PROOF | FLAG.BRITTLE, breakP: 12, breakTo: "SHARDS", crushP: 120,
        // Like borosilicate lab glass, which takes a hot plate but not a
        // red-hot beaker plunged into cold water.
        wetting: 0.97,
        hiT: 1300, hiTo: "MOLTEN_GLASS", latentUp: 250,
        color: [0.62, 0.8, 0.84], colorVar: 0.03,
    },
    {
        id: "SHARDS", name: "Glass shards", group: "Thermal",
        kind: KIND.POWDER, density: 250, drag: 0.02, spread: 0.6, thermal: 0.3, heatCap: 1.6, hardness: 3, flags: FLAG.ACID_PROOF,
        grain: 2,
        hiT: 1300, hiTo: "MOLTEN_GLASS", latentUp: 200,
        color: [0.7, 0.86, 0.9], colorVar: 0.18,
    },
    {
        id: "MOLTEN_GLASS", name: "Molten glass", group: "Thermal",
        kind: KIND.LIQUID, density: 240, drag: 0.005, viscosity: 300, thermal: 1.5, heatCap: 2.6, hardness: 2,
        // soda-lime glass, as its Vogel-Fulcher-Tammann law has it (log η =
        // -2.6 + 4500 / (T - 260 °C), Pa·s) from 1000 to 1500 °C: 4500 Pa·s
        // at 990 °C, e every 89 K
        viscosityT: 990, viscosityE: 89,
        spawnT: 1500, flags: FLAG.ACID_PROOF, expansion: 1e-4, tension: 4,
        loT: 1150, loTo: "GLASS", latentDown: 250,
        color: [1.0, 0.62, 0.25], colorVar: 0.05,
    },
    {
        // Iron's resistance rises about 0.5% per degree, five times over by
        // red heat: a thin wire across a battery heats until it passes too
        // little current to get hotter (a lamp filament), but a thin link
        // in a thick wire, which can't hold the current back, runs away.
        id: "METAL", name: "Metal", group: "Thermal", note: "Polished, it mirrors light; a thin wire carrying a heavy current glows, and melts through",
        density: 780, thermal: 50, heatCap: 3.5, hardness: 15, sigma: 3e4, resistT: 0.005, wetting: 0.67,
        breakP: 110, breakTo: "SCRAP", crushP: 400,
        hiT: 1500, hiTo: "MOLTEN_METAL", latentUp: 1900,
        color: [0.56, 0.59, 0.65], colorVar: 0.05,
    },
    {
        // Current only gets through loose scrap where the pieces touch: a
        // thirtieth of what it would through solid metal, and those contacts
        // are where a heap carrying a heavy current gets hot.
        id: "SCRAP", name: "Metal scrap", group: "Thermal",
        kind: KIND.POWDER, density: 780, drag: 0.01, spread: 0.4, thermal: 5, heatCap: 2.8, hardness: 10, sigma: 1000, resistT: 0.005,
        grain: 3,
        hiT: 1500, hiTo: "MOLTEN_METAL", latentUp: 1700,
        color: [0.5, 0.52, 0.57], colorVar: 0.2,
    },
    {
        // Liquid iron conducts heat about as well as the solid just short of
        // melting, some 30 W/(m·K). It expands 1.3e-4 per °C, and is about
        // as runny as water, 5.6 mPa·s at 1550 °C, 8e-7 m²/s (Assael et al.
        // 2006, "Reference data for the density and viscosity of liquid
        // aluminum and liquid iron"): no viscosity here, as water's.
        id: "MOLTEN_METAL", name: "Molten metal", group: "Thermal",
        kind: KIND.LIQUID, density: 700, drag: 0.003, viscosity: 0, thermal: 30, heatCap: 5.7, hardness: 4, spawnT: 1700,
        sigma: 2500, expansion: 1.3e-4,
        tension: 25,  // liquid iron: about 1.8 N/m
        loT: 1500, loTo: "METAL", latentDown: 1900,
        color: [1.0, 0.55, 0.2], colorVar: 0.05,
    },
    {
        id: "HEATER", name: "Heater", group: "Thermal", note: "Holds 600 °C",
        thermal: 20, heatCap: 4, flags: FLAG.FIXED_TEMP | FLAG.INDESTRUCTIBLE | FLAG.ACID_PROOF, fixedT: 600, spawnT: 600,
        color: [0.8, 0.22, 0.15], colorVar: 0.03,
    },
    {
        id: "COOLER", name: "Cooler", group: "Thermal", note: "Holds -100 °C",
        thermal: 20, heatCap: 4, flags: FLAG.FIXED_TEMP | FLAG.INDESTRUCTIBLE | FLAG.ACID_PROOF, fixedT: -100, spawnT: -100,
        color: [0.3, 0.55, 0.95], colorVar: 0.03,
    },

    // Combustion. What burns takes the oxygen its composition does, burnO2
    // grams a gram, and makes burnWater grams of water a gram of oxygen:
    // wood about CH1.44O0.66 (Drysdale, An Introduction to Fire Dynamics,
    // 2011, ch. 1), bituminous coal about CH0.8O0.1, charcoal nearly carbon,
    // oil CH2, a seed's starch and oil, and a plant's dry fifth (the water
    // that makes up the rest of it is stage 5's plants'). Its heat is the
    // oxygen's (OXYGEN_HEAT): wood's 10.8 kJ a cell, 18 MJ/kg, oil's 39 kJ,
    // 45 MJ/kg, as real fuels have (Babrauskas in the SFPE Handbook). Its
    // burnTime is in real seconds, kept to fires' pace before stage 5 times
    // FIRE_PACE: a cell of wood in 320 s, a centimetre charred at 1.9 mm/min,
    // a brisk fire's (Eurocode 5 designs timber for 0.65). Wood, sawdust and
    // seeds char into a quarter of their mass of charcoal (Di Blasi 2008),
    // a fresh plant into a twentieth (it is four fifths water), and leave
    // their ash: half a percent of wood's mass, 3 % of charcoal's, a tenth of
    // coal's.
    {
        id: "WOOD", name: "Wood", group: "Combustion",
        density: 60, thermal: 0.13, heatCap: 1.2, hardness: 3, breakP: 35, breakTo: "SAWDUST", crushP: 80, wetting: 0.75,
        igniteT: 280, burnTime: 320, burnO2: 1.37, burnWater: 0.39, burnTo: "ASH", burnFire: 0.12,
        charTo: "CHARCOAL", charMass: 0.25, ashMass: 0.005,
        color: [0.47, 0.3, 0.15], colorVar: 0.12,
    },
    {
        id: "SAWDUST", name: "Sawdust", group: "Combustion",
        kind: KIND.POWDER, density: 50, weight: GRAVITY * 5 / 6, drag: 0.1, spread: 0.5, thermal: 0.08, heatCap: 0.42, hardness: 1, flags: FLAG.FLUFF,
        igniteT: 200, burnTime: 30, burnO2: 1.37, burnWater: 0.39, burnTo: "ASH", burnFire: 0.15,
        charTo: "CHARCOAL", charMass: 0.25, ashMass: 0.005,
        color: [0.72, 0.55, 0.32], colorVar: 0.15,
    },
    {
        id: "COAL", name: "Coal", group: "Combustion", note: "Burns long and hot",
        kind: KIND.POWDER, density: 130, spread: 0.5, thermal: 0.25, heatCap: 1.05, grain: 2, wetting: 0.35,
        igniteT: 350, burnTime: 510, burnO2: 2.56, burnWater: 0.2, burnTo: "ASH", burnFire: 0.06, ashMass: 0.1,
        color: [0.13, 0.13, 0.14], colorVar: 0.1,
    },
    {
        // What wood and plants char into when held hot without the air to
        // flame (pyrolysis, update.wgsl): light, glowing more than flaming,
        // and burning long but no hotter than the wood fire it came from.
        id: "CHARCOAL", name: "Charcoal", group: "Combustion", note: "Wood charred without air; glows long and steady",
        kind: KIND.POWDER, density: 40, drag: 0.05, spread: 0.5, thermal: 0.08, heatCap: 0.25,
        igniteT: 350, burnTime: 400, burnO2: 2.4, burnWater: 0.05, burnTo: "ASH", burnFire: 0.04, ashMass: 0.03,
        color: [0.1, 0.09, 0.09], colorVar: 0.12,
    },
    {
        id: "OIL", name: "Oil", group: "Combustion", note: "Floats on water",
        kind: KIND.LIQUID, density: 87, drag: 0.03, viscosity: 0.015, thermal: 0.15, heatCap: 1.6, hardness: 1,
        igniteT: 180, burnTime: 50, burnO2: 3.43, burnWater: 0.375, burnTo: "SMOKE", burnFire: 0.5,
        // olive oil (CRC Handbook): 32 mN/m; 84 mPa·s at 20 °C, 9.2e-5 m²/s,
        // some a third of it at 50
        viscosityT: 20, viscosityE: 30,
        expansion: 7.2e-4, tension: 0.45,
        color: [0.55, 0.4, 0.08], colorVar: 0.05,
    },
    {
        // A gas is the air with something in it (REWRITE_PLAN.md 4c): a flame
        // the air it burns in, smoke the air its soot is in, hydrogen a gas of
        // its own (molar). It rides the air's flow and rises by the air's
        // buoyancy (its temperature and molar mass), not a weight of its own.
        id: "FIRE", name: "Fire", group: "Combustion", note: "Short-lived, heats what it touches",
        kind: KIND.GAS, density: 0.3, weight: 0, drag: 0.35, spread: 0.35, thermal: 0.08, film: 25, heatCap: GAS_HEAT_CAP,
        hardness: 0, spawnT: 800,
        color: [1.0, 0.55, 0.12], colorVar: 0.1,
    },
    {
        id: "SMOKE", name: "Smoke", group: "Combustion",
        kind: KIND.GAS, density: 0.6, weight: 0, drag: 0.3, spread: 0.2, thermal: 0.03, heatCap: GAS_HEAT_CAP, ambient: 0.01,
        hardness: 0, spawnT: 80,
        color: [0.28, 0.28, 0.3], colorVar: 0.15,
    },
    {
        id: "ASH", name: "Ash", group: "Combustion",
        kind: KIND.POWDER, density: 60, weight: GRAVITY * 5 / 6, drag: 0.08, spread: 0.6, thermal: 0.1, heatCap: 0.48,
        hardness: 0.5, hiT: 1000, hiTo: "SMOKE", latentUp: 100, flags: FLAG.FLUFF, grain: 0.05, cohesion: 0.5,
        color: [0.62, 0.6, 0.58], colorVar: 0.1,
    },
    {
        id: "GUNPOWDER", name: "Gunpowder", group: "Combustion",
        kind: KIND.POWDER, density: 170, spread: 0.7, thermal: 0.2, heatCap: 1.4, hardness: 1, grain: 0.5,
        // Black powder: 2.9 MJ/kg, 280 litres of gas a kilogram (most of
        // its products are solids, as smoke), little water (Cooper,
        // Explosives Engineering, 1996).
        igniteT: 160, blastHeat: 2900, gasYield: 280, blastWater: 0.02,
        color: [0.24, 0.23, 0.25], colorVar: 0.25,
    },
    {
        id: "NITRO", name: "Nitro", group: "Combustion", note: "Explodes when heated or dropped hard",
        kind: KIND.LIQUID, density: 159, drag: 0.03, thermal: 0.2, heatCap: 2.4, hardness: 1, flags: FLAG.IMPACT,
        // Nitroglycerin: 6.3 MJ/kg, 716 litres of gas a kilogram, a third
        // of it water (C3H5N3O9: 3 CO2, 2.5 H2O, 1.5 N2, 0.25 O2; Cooper).
        igniteT: 70, blastHeat: 6300, gasYield: 469, blastWater: 0.198, tension: 0.7,
        color: [0.82, 0.9, 0.35], colorVar: 0.05,
    },
    {
        // What electrolysis gives off (electric.wgsl). A fourteenth as dense
        // as air, it rises fast and gathers under whatever roof it finds;
        // under open sky it slowly escapes (update.wgsl). Lit by a flame, a
        // spark or anything past 500 °C, it burns with the oxygen of the air
        // beside it into water, 8 g of oxygen a gram and 242 kJ a mole (15.1
        // kJ a gram of oxygen), as fast as its flame runs through a cell; a
        // jar of it with no air doesn't burn.
        id: "HYDROGEN", name: "Hydrogen", group: "Combustion", note: "Rises fast and gathers under a roof; burns in air into water",
        kind: KIND.GAS, density: 0.07, weight: 0, drag: 0.3, spread: 0.25, thermal: 0.18, heatCap: GAS_HEAT_CAP, ambient: 0.01,
        hardness: 0, igniteT: 500, burnTime: 0.3, burnO2: 8, burnWater: 1.125, burnHeat: 15.1e3, burnFire: 0.3, molar: 2,
        color: [0.8, 0.86, 1.0], colorVar: 0.05,
    },

    // Chemistry & life
    {
        // Acid is a solute: water carrying it, at full strength as painted.
        // Concentrated hydrochloric acid (37 %, 12 equivalents a litre: CRC
        // Handbook): it eats metal, fizzing hydrogen, and stone, fizzing
        // carbon dioxide, an equivalent of itself for each it eats, so a cell
        // of it eats a 23rd of a cell of iron or a fifth of one of stone, and
        // leaves wood, sand and glass be; mixed into more water it weakens.
        // Boiled dry, it fumes away.
        id: "ACID", name: "Acid", group: "Chemistry & Life", note: "Eats metal, fizzing hydrogen, and stone, using itself up; diluted in water it weakens",
        kind: KIND.LIQUID, density: 118, drag: 0.03, thermal: 0.5, heatCap: 3.8, hardness: 1, flags: FLAG.ACID_PROOF,
        paintAs: "WATER", paintSolute: "ACID",
        solDensity: 0.18, solFreeze: -30, solBoil: 40, solSigma: 0.3, solCorrode: 1, solD: 3.3e-9, acidEq: 0.012,
        solColor: [0.45, 0.95, 0.2], solResidue: "SMOKE",
        color: [0.45, 0.95, 0.2], colorVar: 0.08,
    },
    {
        // Common salt dissolves to 36 g in 100 g of water at any temperature
        // (35.7 at 0 °C, 39 at 100 °C); its grains pile at about 1.2 g/cm³,
        // so a third of a cell of it saturates a cell of water. Saturated
        // brine is 1.2 g/cm³, freezes at -21 °C (the eutectic; sea water, a
        // seventh as salty, at -1.9 °C), boils near 108 °C and conducts about
        // 20 S/m (CRC Handbook). Salt on ice makes brine colder than the ice,
        // so the ice melts, taking the heat for it from around: a freezing
        // mixture, -21 °C at best.
        id: "SALT", name: "Salt", group: "Chemistry & Life", note: "Dissolves in water, a third of a cell to saturate one; melts ice, chilling it",
        kind: KIND.POWDER, density: 216, spread: 0.8, thermal: 1.0, heatCap: 1.03, hardness: 1, grain: 0.4,
        // Dissolving across a still film some 0.1 mm thick at its
        // diffusivity, 1.5e-9 m²/s (CRC Handbook).
        solubility: 0.3, dissolveRate: 1.5e-5, solD: 1.5e-9, solDensity: 0.2, solFreeze: -21, solBoil: 8, solSigma: 0.2,
        solColor: [0.27, 0.53, 0.8], solResidue: "SALT",
        color: [0.93, 0.93, 0.95], colorVar: 0.05,
    },
    {
        id: "SALTWATER", name: "Salt water", group: "Chemistry & Life",
        note: "Water saturated with salt: heavier, freezes lower, conducts, and leaves its salt when it dries",
        kind: KIND.LIQUID, density: 120, thermal: 0.6, heatCap: 4.0,
        paintAs: "WATER", paintSolute: "SALT",
        color: [0.27, 0.53, 0.8], colorVar: 0.05,
    },
    {
        id: "MUD", name: "Mud", group: "Chemistry & Life",
        // Soaked dirt and a cell of water more (SOIL_FULL in common.wgsl),
        // which it gives up into drier soil beside it. Conducts through its
        // water, as dirt does (SOILS). Dries like brine
        // boils: at 100 °C it takes in the latent heat, then gives its water
        // off into open air as steam and turns back into soaked dirt.
        kind: KIND.LIQUID, density: 170, drag: 0.01, viscosity: 100, thermal: 1.2, heatCap: 3.0, hardness: 1,
        flags: FLAG.RESIDUE, sigma: 5e-4, expansion: 1.42e-4, expansion2: 4.45e-6, boilShift: 4890, tension: 1,
        hiT: 100, hiTo: "DIRT", latentUp: 2257,
        color: [0.32, 0.23, 0.15], colorVar: 0.08,
    },
    {
        id: "SEED", name: "Seed", group: "Chemistry & Life", note: "Sprouts in water or damp soil",
        kind: KIND.POWDER, density: 130, drag: 0.08, spread: 0.6, thermal: 0.15, heatCap: 1.5, hardness: 1, grain: 4,
        igniteT: 200, burnTime: 20, burnO2: 1.26, burnWater: 0.45, burnTo: "ASH", burnFire: 0.1,
        charTo: "CHARCOAL", charMass: 0.25, ashMass: 0.03,
        color: [0.8, 0.68, 0.35], colorVar: 0.15, paint: 0.05,
    },
    {
        id: "PLANT", name: "Plant", group: "Chemistry & Life", note: "Grows toward the light, drinking from damp soil; roots grip dirt and mud. Withers in the dark or dry soil",
        density: 80, thermal: 0.4, heatCap: 3.0, hardness: 2, breakP: 10, breakTo: "SAWDUST", crushP: 40,
        wetting: 0.35,  // waxy leaves: water beads on them
        igniteT: 230, burnTime: 80, burnO2: 0.24, burnWater: 0.47, burnTo: "ASH", burnFire: 0.2,
        charTo: "CHARCOAL", charMass: 0.05, ashMass: 0.01,
        color: [0.22, 0.66, 0.22], colorVar: 0.15,
    },

    // Exotic
    {
        id: "CLONE", name: "Clone", group: "Exotic", note: "Copies the first thing that touches it",
        thermal: 1.0, flags: FLAG.INDESTRUCTIBLE | FLAG.ACID_PROOF,
        color: [0.85, 0.6, 0.15], colorVar: 0.05,
    },
    {
        id: "VOID", name: "Void", group: "Exotic", note: "Deletes anything that touches it",
        thermal: 0, flags: FLAG.INDESTRUCTIBLE | FLAG.ACID_PROOF,
        color: [0.2, 0.05, 0.25], colorVar: 0.05,
    },
    {
        // Each cell is a 1.5 V cell, + toward where the stroke was heading;
        // cells in a row along that add up, side by side they share the load.
        // Its internal resistance (1/sigma per cell) limits the current, so a
        // thin wire across it gets far hotter than a thick bar.
        id: "BATTERY", name: "Battery", group: "Exotic",
        note: "Drives current around a closed circuit; + is the way you dragged while painting it",
        thermal: 1.0, flags: FLAG.INDESTRUCTIBLE | FLAG.ACID_PROOF, sigma: 4000, emf: 1.5,
        color: [0.95, 0.85, 0.25], colorVar: 0.04,
    },
    {
        id: "FAN", name: "Fan", group: "Exotic", note: "Blows air the way you dragged while painting it",
        // Air with blades in it: the wind carries what its air holds through
        // it (carry.wgsl), and its heat is its air's, the blades too thin to
        // count.
        thermal: 0.026, film: 5, heatCap: GAS_HEAT_CAP, hardness: 8, flags: FLAG.AIR_PASS, breakP: 80, breakTo: "SCRAP",
        crushP: 300,
        color: [0.55, 0.66, 0.72], colorVar: 0.04,
    },
    {
        id: "BLACK_HOLE", name: "Black hole", group: "Exotic", note: "Pulls everything in, as hard as the Black hole slider says, and swallows what it touches",
        thermal: 0, flags: FLAG.AIR_PASS | FLAG.INDESTRUCTIBLE | FLAG.ACID_PROOF,
        color: [0.02, 0.0, 0.04], colorVar: 0,
    },
    {
        id: "BLAST", name: "Blast", group: "Exotic", note: "Raw explosion",
        kind: KIND.GAS, density: 0.2, weight: 0, drag: 0.5, thermal: 0.1, film: 25, heatCap: GAS_HEAT_CAP, hardness: 0, spawnT: 1800,
        color: [1.0, 0.9, 0.6], colorVar: 0, paint: 1,
    },
];

// Light and radiant heat (light.wgsl): how much an element stops of visible
// light and of thermal infrared, as extinction coefficients per metre (a
// beam crossing x metres of it keeps exp(-k x) of itself, Beer-Lambert),
// and its emissivity, where they differ from DEFAULTS (opaque, emissivity
// 0.9), kept apart from DEFS. A cell takes a centimetre cube's, clarity and
// irClarity (CELL_M), and a cell of another size as much as that depth of
// it lets through (clarityAt in common.wgsl). The values are real ones:
// - water: pure water absorbs 0.02-0.06 /m of blue-green light and 0.3-0.6
//   of red (Pope & Fry, Applied Optics 36(33), 1997); 0.1 /m for the eye's
//   mix with a little scattering. A pond is nearly clear, the sea dims over
//   tens of metres.
// - ice: pure ice absorbs less than water does (Warren & Brandt, J.
//   Geophys. Res. 113, 2008); the bubbles and cracks of lake ice scatter
//   more, about 1 /m (an estimate).
// - steam: water vapour is clear to visible light (what is seen of steam is
//   its droplets, the air's mist). A centimetre of it at 1 atm and 100 °C
//   gives off about 4 % of a black body's thermal radiation (Hottel's
//   charts; Modest, Radiative Heat Transfer, 3rd ed. 2013, ch. 11): 4 /m,
//   taken as grey (over metres its bands saturate, and this overstates it).
// - glass: clear float glass lets through about 0.90 of visible light 4 mm
//   thick and 0.86 10 mm thick (makers' data), so about 7 /m inside it (the
//   rest is reflected at its faces, which a cell can't tell from its
//   depth); it stops thermal infrared past 4.5 µm or so within a fraction
//   of a millimetre (Rubin, Solar Energy Materials 12, 1985). Molten glass
//   lets through some of its own glow, which peaks near 2 µm, where clear
//   glass melts absorb about 100 /m (Viskanta & Anderson, Advances in Heat
//   Transfer 11, 1975).
// - glass shards: a heap of fragments reflects some 4 % at each face a beam
//   crosses, a few faces a centimetre: about 50 /m (an estimate).
// - flames and blasts: their soot glowing, about 3.72 fv C0 T / C2 per
//   metre (fv the soot's share of the volume, 1e-7 to 1e-6): 0.3-3 /m
//   (Modest, ch. 11), so 1 /m, in both bands. A flame a centimetre deep
//   gives off 1 % of a black body's radiation, a fire a metre deep two
//   thirds of it.
// - smoke: soot cooled, which stops visible light at 8.7 m² a gram
//   (Mulholland & Croarkin, Fire and Materials 24, 2000): about 1 /m at
//   0.1 g/m³; thermal infrared some twenty times less (soot absorbs about
//   as one over the wavelength).
// - plants: foliage stops light as its leaves cover the way, about half
//   their area per metre (Monsi & Saeki, Japanese Journal of Botany 14,
//   1953): 1 /m for 2 m² of leaves per m³; as much of the infrared (a leaf
//   lets none of it through).
// - oil (olive) and nitro, yellow: about 15 and 5 /m (estimates); both
//   opaque to thermal infrared.
// Air's mist (mistClarity in common.wgsl) isn't real fog's yet: the air
// holds thousands of times real air's water (REWRITE_PLAN.md stage 3).
// Snow is translucent by scattering, which this doesn't model (what isn't
// let through is taken in), so it stays opaque and reflects its albedo.
// Thermal emissivities are about 0.95 for water, ice and snow, 0.9 for
// glass, wood, soil and stone, and 0.1 to 0.3 for bare metal (Incropera et
// al., Fundamentals of Heat and Mass Transfer, table A.11). Wall blocks
// heat, so it reflects all of it (emissivity 0); a black hole takes in all
// light. Polished metal reflects nearly all the light it reflects as a
// mirror does (shine), tinted by its colour; scrap is scratched dull, and
// molten metal has a skin of oxide on it. A fan is air with blades in it
// (its heat air's), its blades too thin to block light or give off heat.
const OPTICS = {
    AIR: { extinction: 0, irExtinction: 0, emissivity: 0 },
    FAN: { extinction: 0, irExtinction: 0, emissivity: 0 },
    WALL: { emissivity: 0 },
    WATER: { extinction: 0.1, emissivity: 0.96 },
    ICE: { extinction: 1, emissivity: 0.97 },
    SNOW: { emissivity: 0.97 },
    STEAM: { extinction: 0, irExtinction: 4, emissivity: 1 },
    GLASS: { extinction: 7 },
    SHARDS: { extinction: 50 },
    MOLTEN_GLASS: { extinction: 7, irExtinction: 100 },
    METAL: { emissivity: 0.2, shine: 0.85 },
    SCRAP: { emissivity: 0.3, shine: 0.45 },
    MOLTEN_METAL: { emissivity: 0.3, shine: 0.5 },
    OIL: { extinction: 15 },
    FIRE: { extinction: 1, irExtinction: 1, emissivity: 1 },
    SMOKE: { extinction: 1, irExtinction: 0.05, emissivity: 1 },
    NITRO: { extinction: 5 },
    HYDROGEN: { extinction: 0, irExtinction: 0, emissivity: 0 },
    // (Acid and salt water are painted as water carrying them, and take its.)
    PLANT: { extinction: 1, irExtinction: 1 },
    BLACK_HOLE: { emissivity: 1 },
    BLAST: { extinction: 1, irExtinction: 1, emissivity: 1 },
};

// Real densities, g/cm³, of what isn't a substance's form (whose fill is
// its density, SUBSTANCES): the air's pressure pushes a cell of it by
// -∇p / ρ, and its drag shares out what moves by its mass (REWRITE_PLAN.md
// 4e). A powder's is its bulk density, the grains with the air between them.
// From the Engineering ToolBox's tables of densities of solids, liquids and
// bulk materials unless said:
// - gases at 1 atm: dry air at 20 °C 0.001204; steam at 100 °C 0.000598;
//   hydrogen 0.0000838; a flame's gas and a blast's at about 1100 K, air's
//   times 293 / 1100; smoke at about 100 °C, air's times 293 / 373.
// - soil 1.3 (dry, 1.1-1.5), mud 1.7 (1.6-1.9 wet), wood 0.6 (softwoods
//   0.4-0.7), sawdust 0.21, coal 0.83 (bituminous, broken), charcoal 0.25
//   (lump), wood ash 0.6, black powder 1.0 (loose), table salt 1.2 (bulk),
//   wheat 0.77 (bulk, for seeds); crude oil 0.87, nitroglycerin 1.59 (CRC
//   Handbook), hydrochloric acid at 37 % 1.18, saturated brine 1.2 (CRC
//   Handbook).
// - a plant's leaves and stems, 0.8 (fresh plant tissue mostly water and
//   air spaces: an estimate); a fan, air with blades in it, 0.3 (an
//   estimate); a battery, an AA alkaline cell's 23 g in 8.1 cm³, 2.8.
// - wall, heater, cooler, clone, void and black hole, which the air can't
//   move (held, or a wall's): concrete's 2.4, the heater's and cooler's
//   steel's 7.8.
const RHO = {
    AIR: 0.001204,
    STEAM: 0.000598,
    HYDROGEN: 0.0000838,
    FIRE: 0.001204 * 293 / 1100,
    BLAST: 0.001204 * 293 / 1100,
    SMOKE: 0.001204 * 293 / 373,
    WALL: 2.4,
    DIRT: 1.3,
    MUD: 1.7,
    WOOD: 0.6,
    SAWDUST: 0.21,
    COAL: 0.83,
    CHARCOAL: 0.25,
    ASH: 0.6,
    GUNPOWDER: 1.0,
    SALT: 1.2,
    SEED: 0.77,
    OIL: 0.87,
    NITRO: 1.59,
    ACID: 1.18,
    SALTWATER: 1.2,
    PLANT: 0.8,
    FAN: 0.3,
    BATTERY: 2.8,
    HEATER: 7.8,
    COOLER: 7.8,
    CLONE: 2.4,
    VOID: 2.4,
    BLACK_HOLE: 2.4,
};

// Whether an element carries a spark's pulse follows from its conductance.
function sparkFlag(e) {
    return e.sigma >= SPARK_SIGMA && e.emf === 0 ? FLAG.CONDUCT : 0;
}

// Substances that change form (REWRITE_PLAN.md stage 3c): each one's latent
// heats per gram, and the grams a cell of each of its forms holds, its fill
// (its density, but for steam a whole boiled cell of water). A form's
// latentUp and latentDown follow from them, so it takes the same heat to
// melt (or boil) a gram whichever form it is in, and a cell changing form
// keeps its mass and energy (formed and changePhase in the shaders). Rock
// 400 J/g, about what lava flow models take (3.5-4 x 10⁵ J/kg); silica
// 137.5, sand's as the table had it (quartz about 140); iron 244 (247, CRC
// Handbook); water 334 to melt and 2257 to boil.
const SUBSTANCES = {
    rock: { fusion: 400, forms: { STONE: 3.0, GRAVEL: 2.0, LAVA: 2.5 } },
    silica: { fusion: 137.5, forms: { SAND: 1.6, GLASS: 2.5, SHARDS: 2.4, MOLTEN_GLASS: 2.4 } },
    iron: { fusion: 244, forms: { METAL: 7.8, SCRAP: 7.0, MOLTEN_METAL: 7.0 } },
    water: { fusion: 334, boiling: 2257, forms: { ICE: 0.9, SNOW: 0.5, WATER: 1.0 } },
};

// The substance and fill of each form, and their latent heats by the
// substance's per gram: boiling between a liquid and a gas, melting else.
// A form a cell of a fuller one turns into (by a phase change, setting
// while flying, or breaking) owes the rest in cells (FLAG.OWES), in its
// life; ice and snow keep theirs in aux, as before. So does a solid a
// liquid sets into: a cell of liquid can hold more than a cell of it (its
// particles crowded, or owing what it couldn't fit), which the solid then
// owes (2026-10-27: lava set into stone that owed nothing, and a crucible
// of stone melted and set again lost a sixth of it).
function withSubstance(e, byId) {
    const entry = Object.entries(SUBSTANCES).find(([, s]) => e.id in s.forms);
    if (!entry) {
        return e;
    }
    const [, s] = entry;
    const fill = s.forms[e.id];
    const latent = (other) => {
        const o = byId[other];
        const gas = e.kind === KIND.GAS || o.kind === KIND.GAS;
        return (gas ? s.boiling : s.fusion) * fill;
    };
    const out = { ...e, fill };
    if (e.hiTo && e.hiTo in s.forms) {
        out.latentUp = latent(e.hiTo);
    }
    if (e.loTo && e.loTo in s.forms) {
        out.latentDown = latent(e.loTo);
    }
    return out;
}

function owesFlag(e, byId) {
    if (e.id === "ICE" || e.id === "SNOW" || !e.fill) {
        return 0;
    }
    const from = Object.values(byId).filter((o) => (o.fill > e.fill
        && [o.hiTo, o.loTo, o.breakTo, byId[o.loTo]?.breakTo].includes(e.id)) || (o.kind === KIND.LIQUID && o.loTo === e.id));
    return from.length ? FLAG.OWES : 0;
}

// Structural solids' stiffness and strength (stress.wgsl, REWRITE_PLAN.md
// 8-2): Young's modulus E, the strengths in tension and in compression, Pa,
// and the linear thermal expansion α, 1/K (8-5), from handbook ranges
// (Ashby, Materials Selection in Mechanical Design, 4th ed., 2011, unless
// said), which vary by a factor of several between sources and samples;
// each picked within its range:
// - stone as granite or a hard limestone: E 50 GPa (20-70), tension 10 MPa
//   (granite 7-25 pulled directly), compression 130 MPa (30-250), α 8e-6
//   (granite 6-9e-6);
// - ice near its melting point: E 9 GPa, tension 1 MPa (0.7-3.1),
//   compression 5 MPa (5-25, colder is stronger) (Petrovic, "Mechanical
//   properties of ice and snow", J. Mater. Sci. 38, 2003, 1-6), α 5.1e-5;
// - borosilicate glass, as lab glass (the element's note): E 64 GPa,
//   tension 40 MPa (what a sound piece bears; flaws set it, 7-100),
//   compression 1 GPa, α 3.3e-6 (soda-lime's is 9e-6): it takes a step of
//   about 140 °C (Kingery's E α ΔT / (1 - ν) at its tension), soda-lime
//   about 50;
// - steel (metal, and a fan's blades): E 200 GPa, both 250 MPa (mild
//   steel's yield);
// - wood as a softwood, taken as if the same every way: E 11 GPa (along
//   the grain), tension 40 MPa (along it 60-100, across 2-3), compression
//   30 MPa (along it 30-50);
// - a plant's green stems: E 0.5 GPa, tension 5 MPa, compression 3 MPa
//   (herbaceous stems, 0.1-3 GPa; an estimate).
// Only the brittle ones take a thermal strain: steel (α 1.2e-5) yields
// under a strain it is held to, which relieves it, where the solve, having
// no plasticity, would crack it (a pressure cooker's pot at 120 °C, a wire
// at 300); wood (5e-6 along the grain) and stems give likewise. Left out
// until the solve yields.
// How each weakens when hot (hotStart, hotHalf; REWRITE_PLAN.md 8-7): steel
// as structural steel's yield strength in fire (EN 1993-1-2, Table 3.1:
// whole to 400 °C, 0.47 at 600, 0.11 at 800); stone as siliceous concrete
// (EN 1992-1-2, Table 3.1: whole to 100 °C, 0.6 at 500, 0.3 at 700); glass
// from its strain point (some 500 °C) to its softening point (720), where
// it sags under its own weight; wood by EN 1995-1-2's Annex B (0.65 of its
// strength in tension at 100 °C), as plants. Ice keeps its strength to its
// melting point (it creeps instead: softeningT).
// And the stress a blow breaks each at (impact): stone, wood and plants
// crush where they are struck (their compressive strengths); glass cracks
// from its flaws at about its tensile strength (a marble survives a drop of
// a metre or two onto stone, 30 MPa so); ice at 20 MPa, its compressive
// strength under a fast load (5 to 25 MPa as the load is quicker:
// Petrovic, "Mechanical properties of ice and snow", J. Mater. Sci. 38,
// 2003); steel dents, which cells don't show, rather than breaking (none).
const STRENGTH = {
    STONE: { youngs: 50e9, tensile: 10e6, compressive: 130e6, expansionLinear: 8e-6, hotStart: 100, hotHalf: 550, impact: 130e6 },
    ICE: { youngs: 9e9, tensile: 1e6, compressive: 5e6, expansionLinear: 5.1e-5, impact: 20e6 },
    GLASS: { youngs: 64e9, tensile: 40e6, compressive: 1e9, expansionLinear: 3.3e-6, hotStart: 500, hotHalf: 650, impact: 40e6 },
    METAL: { youngs: 200e9, tensile: 250e6, compressive: 250e6, hotStart: 400, hotHalf: 590 },
    FAN: { youngs: 200e9, tensile: 250e6, compressive: 250e6, hotStart: 400, hotHalf: 590 },
    WOOD: { youngs: 11e9, tensile: 40e6, compressive: 30e6, hotStart: 20, hotHalf: 120, impact: 30e6 },
    PLANT: { youngs: 0.5e9, tensile: 5e6, compressive: 3e6, hotStart: 20, hotHalf: 120, impact: 3e6 },
};

// Powders' angles of repose, degrees, dry, poured (Al-Hashemi & Al-Amoudi,
// "A review on the angle of repose of granular materials", Powder
// Technology 330, 2018, 397-417, from its ranges per material; ash as fly
// ash, scrap and shards as angular crushed metal and glass, seeds as
// cereal grains). Their tangent is their friction (REWRITE_PLAN.md 9-1).
const REPOSE = {
    SAND: 34, GRAVEL: 38, DIRT: 40, SNOW: 40, SHARDS: 40, SCRAP: 42, SAWDUST: 45, COAL: 37, CHARCOAL: 40,
    ASH: 40, GUNPOWDER: 35, SALT: 32, SEED: 28,
};

// How soils hold and pass water (stage 5f): van Genuchten's parameters for
// the soil classes of Carsel & Parrish (1988, Water Resources Research 24):
// sand as sand, dirt as loam, ash as silt loam. And sigma, the current they
// pass soaked, through the water in their pores (electric.wgsl's sigmaOf,
// less the drier): Archie's law (1942), the water's conductivity times
// their pores' share (SOIL_HOLD, 0.4) to the power m, 1.3 for loose rounded
// grains and 2 for platy, clayey ones. Sand's water is the water that
// soaked it (2e-5 S/cm); dirt's a soil's solution, its salts dissolved
// (some 1e-3 S/cm, a usual soil's 1 dS/m; mud's 5e-4 is the same water
// through a slurry's pores, 0.7 of it); ash's its potash dissolved, a
// strong lye (taken as 1e-2 S/cm, potassium carbonate's at 0.05 mol/l).
// Their thermal conductivity soaked (Johansen 1975): the grains' (quartz
// 7.7 W/(m·K), other minerals 2, by the soil's quartz: sand's most of it,
// 6.5; a loam's some 0.4 of it, 3.4; ash's 1, an estimate) and water's
// (0.6) geometric mean by the pores (0.4, SOIL_HOLD; ash's 0.7): 2.5, 1.7,
// 0.7; their tables' thermal is dry soil's, Johansen's (0.137 ρd + 64.7) /
// (2700 - 0.947 ρd) by their dry density: sand 0.24, dirt 0.16, ash 0.07.
const SOILS = {
    SAND: { soilR: 0.045, soilAlpha: 14.5, soilN: 2.68, soilK: 8.25e-5, sigma: 6e-6, thermal: 0.24, soilKSat: 2.5, soilKe: 0.7 },
    DIRT: { soilR: 0.078, soilAlpha: 3.6, soilN: 1.56, soilK: 2.89e-6, sigma: 1.6e-4, thermal: 0.16, soilKSat: 1.7, soilKe: 1 },
    ASH: { soilR: 0.067, soilAlpha: 2.0, soilN: 1.41, soilK: 1.25e-6, sigma: 1.6e-3, thermal: 0.07, soilKSat: 0.7, soilKe: 1 },
};

// What acid eats (stage 5e): the grams an equivalent of acid eats, the heat
// it gives off an equivalent, and the gas it fizzes. Iron as Fe2+ (55.8 g a
// mole, 2 equivalents), 89 kJ a mole; stone, gravel and the ash taken as
// carbonates (CaCO3, 50 g an equivalent; 15 kJ a mole; dirt a tenth of it,
// ash a fifth, as potash): CRC Handbook. Anything else it leaves be.
const ACIDS = {
    METAL: { acidGrams: 27.9, acidHeat: 44.5e3, acidGas: "HYDROGEN" },
    SCRAP: { acidGrams: 27.9, acidHeat: 44.5e3, acidGas: "HYDROGEN" },
    MOLTEN_METAL: { acidGrams: 27.9, acidHeat: 44.5e3, acidGas: "HYDROGEN" },
    STONE: { acidGrams: 50, acidHeat: 7.5e3, acidGas: "AIR" },
    GRAVEL: { acidGrams: 50, acidHeat: 7.5e3, acidGas: "AIR" },
    DIRT: { acidGrams: 500, acidHeat: 7.5e3, acidGas: "AIR" },
    ASH: { acidGrams: 345, acidHeat: 7.5e3, acidGas: "AIR" },
};

const MERGED = DEFS.map((d, index) => ({ ...DEFAULTS, ...d, ...OPTICS[d.id], ...ACIDS[d.id], ...SOILS[d.id], ...STRENGTH[d.id], index }));
const MERGED_BY_ID = Object.fromEntries(MERGED.map((e) => [e.id, e]));
const FORMED = MERGED.map((e) => withSubstance(e, MERGED_BY_ID));
const FORMED_BY_ID = Object.fromEntries(FORMED.map((e) => [e.id, e]));

// The grams a cell of e holds (a gas's compressed as the world's gas is),
// and the cells of fresh air's oxygen it burns with.
const cellGramsOf = (e) => (e.fill || RHO[e.id]) * (e.kind === KIND.GAS ? GAS_COMPRESSION : 1);
const oxygenNeedOf = (e) => cellGramsOf(e) * e.burnO2 / AIR_OXYGEN;

export const ELEMENTS = FORMED.map((e) => {
    // What a centimetre cube of it lets through (OPTICS).
    const clarity = Math.exp(-e.extinction * CELL_M);
    const irClarity = Math.exp(-e.irExtinction * CELL_M);
    // The derived phase fields (FIELDS).
    const lo = FORMED_BY_ID[e.loTo];
    const grams = lo && e.fill && lo.fill ? lo.fill / e.fill : 1;
    const phase = {
        upSpan: e.hiTo ? e.latentUp / e.heatCap : 1e30,
        downSpan: lo ? e.latentDown / e.heatCap : 1e30,
        downScale: lo ? e.heatCap * grams / lo.heatCap : 0,
        zero: lo ? Math.max(e.loT - (e.latentDown + (e.loT + 273) * lo.heatCap / grams) / e.heatCap, -273) : -273,
    };
    // Its density, kg/m³: a substance's form's fill, else RHO's.
    const rho = 1000 * (e.fill || RHO[e.id]);
    if (!(rho > 0)) {
        throw new Error(`No density for ${e.id}`);
    }
    // A liquid or a solid has one density: what sinks through what and the
    // liquid's solve go by its table density, its mass by rho, and they are
    // the same (REWRITE_PLAN.md 7-6: oil's solve took the table's 0.80 where
    // it is 0.87). Not the fixed ones (DEFAULTS' 10, heavier than anything,
    // so nothing sinks through them). A powder has two: its grains' (the
    // table's: sinking or floating in a liquid, its buoyancy and drag there,
    // Shields), and its pile's, the grains with the air between them (rho: a
    // cell's mass, its load, a body's cargo), no more than its grains' (9-2).
    if ((e.kind === KIND.LIQUID || e.kind === KIND.SOLID) && e.density !== DEFAULTS.density
        && Math.abs(10 * e.density - rho) > 1e-6 * rho) {
        throw new Error(`${e.id}'s table density ${e.density / 100} isn't its real one, ${rho / 1000}`);
    }
    if (e.kind === KIND.POWDER && 10 * e.density < rho) {
        throw new Error(`${e.id}'s grains (${e.density / 100}) are lighter than its pile (${rho / 1000})`);
    }
    // The grams a cell holds (a gas's compressed as the world's gas is), the
    // cells of fresh air's oxygen it burns with, the vapour units of water it
    // makes and the heat it gives a cell of oxygen (a gram of water is
    // WATER_VAPOUR units: a cell of water holds a gram); going off, its heat
    // of explosion a cell, and the gas and water it makes.
    const cellGrams = cellGramsOf(e);
    const char = FORMED_BY_ID[e.charTo];
    const ash = FORMED_BY_ID[e.burnTo];
    const charYield = char ? cellGrams * e.charMass / cellGramsOf(char) : 0;
    const burning = {
        oxygenNeed: oxygenNeedOf(e),
        charYield,
        charAt: char ? 255 * (1 - charYield * oxygenNeedOf(char) / oxygenNeedOf(e)) : 255,
        charGas: cellGrams * (1 - e.charMass) / AIR_GRAMS,
        ashYield: ash.kind === KIND.GAS ? 1 : cellGrams * e.ashMass / cellGramsOf(ash),
        burnVapour: e.burnWater * AIR_OXYGEN * WATER_VAPOUR,
        oxygenHeat: e.burnHeat * AIR_OXYGEN,
        blast: cellGrams * e.blastHeat,
        blastGas: cellGrams * e.gasYield * (293.15 / 273.15) / GAS_COMPRESSION,
        blastVapour: cellGrams * e.blastWater * WATER_VAPOUR,
        acidDemand: e.acidGrams > 0 ? cellGrams / e.acidGrams : 0,
    };
    if (e.burnTime > 0 && !(burning.oxygenNeed > 0)) {
        throw new Error(`No oxygen to burn for ${e.id}`);
    }
    return {
        ...e, rho, clarity, irClarity, ...phase, ...burning,
        friction: e.id in REPOSE ? Math.tan(REPOSE[e.id] * Math.PI / 180) : 0,
        flags: e.flags | sparkFlag(e) | owesFlag(e, FORMED_BY_ID),
    };
});
export const ID = Object.fromEntries(ELEMENTS.map((e) => [e.id, e.index]));

for (const id of [...Object.keys(OPTICS), ...Object.keys(RHO), ...Object.keys(ACIDS), ...Object.keys(SOILS), ...Object.keys(REPOSE)]) {
    if (!(id in ID)) {
        throw new Error(`Optics or density given for unknown element: ${id}`);
    }
}

// The reaction table's columns (js/reactions.js) are the table's.
for (const r of REACTIONS) {
    for (const column of r.columns) {
        if (!FIELDS.some(([name]) => name === column)) {
            throw new Error(`Reaction ${r.id} names ${column}, which isn't a column of the element table`);
        }
    }
}

if (ELEMENTS.length >= NO_ELEM) {
    throw new Error("Too many elements for an 8-bit id");
}

function ref(name) {
    if (name == null) {
        return NO_ELEM;
    }
    if (!(name in ID)) {
        throw new Error(`Unknown element reference: ${name}`);
    }
    return ID[name];
}

export function packElements() {
    const buffer = new ArrayBuffer(ELEMENTS.length * FIELDS.length * 4);
    const view = new DataView(buffer);
    let offset = 0;
    for (const e of ELEMENTS) {
        const values = {
            ...e,
            hiTo: ref(e.hiTo),
            loTo: ref(e.loTo),
            burnTo: ref(e.burnTo),
            charTo: ref(e.charTo),
            breakTo: ref(e.breakTo),
            solResidue: ref(e.solResidue),
            acidGas: ref(e.acidGas),
            solR: e.solColor[0],
            solG: e.solColor[1],
            solB: e.solColor[2],
            r: e.color[0],
            g: e.color[1],
            b: e.color[2],
        };
        for (const [name, type] of FIELDS) {
            if (type === "u32") {
                view.setUint32(offset, values[name], true);
            } else {
                view.setFloat32(offset, values[name], true);
            }
            offset += 4;
        }
    }
    return buffer;
}

export function wgslPrelude() {
    const lines = ["// Generated from js/elements.js"];
    for (const e of ELEMENTS) {
        lines.push(`const ${e.id}: u32 = ${e.index}u;`);
    }
    for (const [name, value] of Object.entries(KIND)) {
        lines.push(`const KIND_${name}: u32 = ${value}u;`);
    }
    for (const [name, value] of Object.entries(FLAG)) {
        lines.push(`const F_${name}: u32 = ${value}u;`);
    }
    lines.push("struct Elem {");
    for (const [name, type] of FIELDS) {
        lines.push(`    ${name}: ${type},`);
    }
    lines.push("}");
    return lines.join("\n") + "\n";
}

export function paintDensity(e) {
    if (e.paint != null) {
        return e.paint;
    }
    switch (e.kind) {
        case KIND.POWDER: return 0.35;
        case KIND.LIQUID: return 0.5;
        case KIND.GAS: return 0.3;
        default: return 1;
    }
}

function nameOf(id) {
    return ELEMENTS[ID[id]].name.toLowerCase();
}

// Tooltip text derived from the same numbers the simulation uses: its
// reactions' lines (js/reactions.js), then how it holds together and
// conducts.
export function describe(e) {
    const parts = [];
    if (e.note) {
        parts.push(e.note);
    }
    for (const r of REACTIONS) {
        const line = r.applies(e) ? r.text(e, nameOf) : null;
        if (line) {
            parts.push(line);
        }
    }
    if (e.breakTo) {
        const mpa = (pa) => `${+(pa / 1e6).toPrecision(2)} MPa`;
        parts.push("Stands while joined face to face to something fixed (a wall, the ground, the world's edge); cut off, it comes loose as one piece, which tumbles, rolls, floats and stacks");
        if (e.tensile > 0) {
            parts.push(`Cracks into ${nameOf(e.breakTo)} where its own weight, what presses on it, or heating or cooling against what holds it stress it past ${mpa(e.tensile)} pulling or ${mpa(e.compressive)} crushing`
                + (e.hotHalf < T_NEVER ? `, half that by ${e.hotHalf} °C` : ""));
        } else {
            parts.push(`Crumbles into ${nameOf(e.breakTo)} under pressure`);
        }
        if (e.flags & FLAG.BRITTLE) {
            parts.push("Shatters on a hard landing");
        }
    }
    if (e.emf > 0) {
        parts.push(`${e.emf} V per cell`);
    } else if (e.sigma > 0) {
        const [, word] = CONDUCTION_WORDS.find(([least]) => e.sigma >= least);
        parts.push(`Conducts electricity ${word}${SOILS[e.id] ? " when soaked, less the drier, dry not at all" : ""}`);
    }
    return parts.join(". ");
}
