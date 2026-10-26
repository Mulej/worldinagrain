// Test scenes (REWRITE_PLAN.md decision 12). Each is a small world built
// to show one seam between subsystems (COMPATIBILITY.md): what it should do
// is its hint, and test/checks.mjs runs it and measures whether it does.
// A scene that tests what a later stage builds fails its check until then.
//
// Each has a world of its own size, in cells (the app and the harness make a
// world that size), and is laid out in cells, x across and y down from the
// top, boxes inclusive. Cells are a centimetre unless a scene has a `height`
// (metres, the world's: Params.metresPerCell). Structural solids need a path
// to the ground or to Wall, which anchors, and hold spans up to
// sim.supportMax.

import { ID } from "./elements.js";
import { BRUSH, CLOCK, DEFAULT_BLACK_HOLE_PULL, DEFAULT_METRES_PER_CELL, LAPSE_RATE, SHAPE } from "./sim.js";

const DOWN = Math.PI / 2;
const RIGHT = 0;

// Deterministic randomness, so the same people every time.
function seededRandom(seed) {
    let s = seed >>> 0;
    return () => {
        s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
        return s / 4294967296;
    };
}

class Builder {
    constructor(sim) {
        this.sim = sim;
        this.w = sim.width;
        this.h = sim.height;
        this.personSeeds = seededRandom(0x9e3779b9);
    }

    // opts: angle (fans and batteries: the way they push), temp (°C, else
    // the element's own), moist (soil's water, 0 dry to 1 soaked; a
    // solution's strength, 1 saturated).
    paint(name, shape, ax, ay, bx, by, radius, opts = {}) {
        if (!(name in ID)) {
            throw new Error(`Unknown element: ${name}`);
        }
        this.sim.paint({
            ax, ay, bx, by, radius, shape, angle: opts.angle ?? 0,
            mode: BRUSH.PAINT, elem: ID[name], replace: true, strength: 1, moist: opts.moist, temp: opts.temp,
        });
    }

    // The cells from (x0, y0) to (x1, y1), both included.
    rect(name, x0, y0, x1, y1, opts) {
        this.paint(name, SHAPE.RECT, x0, y0, x1 + 1, y1 + 1, 0, opts);
    }

    disc(name, x, y, radius, opts) {
        this.paint(name, SHAPE.CAPSULE, x + 0.5, y + 0.5, x + 0.5, y + 0.5, radius, opts);
    }

    // A box of `name` with walls `wall` cells thick, air inside.
    box(name, x0, y0, x1, y1, wall = 1, opts) {
        this.rect(name, x0, y0, x1, y1, opts);
        this.rect("AIR", x0 + wall, y0 + wall, x1 - wall, y1 - wall);
    }

    // An open-topped tank of `name`, walls `wall` thick, air inside.
    tank(name, x0, y0, x1, y1, wall = 1) {
        this.rect(name, x0, y0, x1, y1);
        this.rect("AIR", x0 + wall, y0, x1 - wall, y1 - wall);
    }

    // Wall along the bottom, `rows` deep.
    floor(rows = 2) {
        this.rect("WALL", 0, this.h - rows, this.w - 1, this.h - 1);
    }

    heat(x0, y0, x1, y1, degrees) {
        this.tool(BRUSH.HEAT, x0, y0, x1, y1, degrees);
    }

    // The air's vapour at `share` of what saturates it.
    vapour(x0, y0, x1, y1, share) {
        this.tool(BRUSH.VAPOUR, x0, y0, x1, y1, share);
    }

    tool(mode, x0, y0, x1, y1, strength) {
        this.sim.paint({
            ax: x0, ay: y0, bx: x1 + 1, by: y1 + 1, radius: 0, shape: SHAPE.RECT, angle: 0,
            mode, elem: ID.AIR, replace: true, strength,
        });
    }

    // A person standing at x on the ground under y.
    person(x, y) {
        this.sim.addPerson(x, y, Math.floor(this.personSeeds() * 2 ** 32));
    }
}

// ---------------------------------------------------------------- air

// A flue of Wall 8 cells wide and 1.1 m tall over a heater, open at its top
// and at its foot's side, at night: the warm air in it rises, drawing the
// room's air in at its foot (the stack effect), held back by the air's
// eddies by its speed (REWRITE_PLAN.md 4e-9).
function chimney(b) {
    b.rect("WALL", 0, 190, 119, 199);
    b.rect("WALL", 54, 70, 55, 175);
    b.rect("WALL", 64, 70, 65, 189);
    b.rect("HEATER", 56, 186, 63, 189);
}

function heatedRoom(b) {
    b.floor();
    b.box("WALL", 20, 20, 99, 69, 2);
    b.rect("HEATER", 22, 66, 97, 67);
}

function fanRoom(b) {
    b.floor();
    b.box("WALL", 5, 15, 74, 69, 2);
    b.rect("FAN", 38, 60, 40, 67, { angle: RIGHT });
    b.box("WALL", 85, 15, 154, 69, 2);
    b.rect("WALL", 119, 17, 120, 67);
    b.rect("FAN", 119, 40, 120, 44, { angle: RIGHT });
}

// (The fan is held by a rod of wall from the world's edge: hung in mid-air,
// it is held by nothing, and comes loose and falls at once, REWRITE_PLAN.md
// 8-1; stood on a pedestal, it blocked the air that passed under it, and
// the smoke came back round. The rod blocks a row.)
function smokeInWind(b) {
    b.floor();
    b.rect("WALL", 0, 65, 19, 65);
    b.rect("FAN", 20, 55, 23, 75, { angle: RIGHT });
    b.rect("SMOKE", 40, 62, 45, 68);
}

// A block of water let fall into a tank: its splash shuts pockets of air in
// it, sealed off by the water as it closes over them (F-5).
function splash(b) {
    b.floor();
    b.tank("GLASS", 10, 40, 149, 97);
    b.rect("WATER", 11, 84, 148, 96);
    b.rect("WATER", 40, 5, 100, 35);
}

// Nothing but the open air, at an hour its outdoor air is at the day's mean
// (22:30: dayWarmth none), so as warm and as humid as a cleared world's.
function stillAir() {}

function plume(b) {
    b.floor(4);
    b.rect("HEATER", 74, 114, 85, 115);
    b.rect("SMOKE", 20, 100, 30, 110, { temp: 20 });
}

function holeRoom(b) {
    b.floor();
    b.box("WALL", 30, 20, 89, 79, 3);
    b.disc("BLACK_HOLE", 60, 50, 2);
}

function chargeAndWall(b) {
    b.floor(4);
    [1, 2, 4, 8].forEach((thick, k) => {
        const x = 50 * k;
        if (k < 3) {
            b.rect("WALL", x + 48, 60, x + 49, 95);
        }
        b.rect("NITRO", x + 8, 93, x + 10, 95, { temp: 100 });
        b.rect("STONE", x + 16, 76, x + 15 + thick, 95);
    });
}

// ---------------------------------------------------------------- water

function fogBox(b) {
    b.floor();
    b.box("WALL", 10, 10, 149, 89, 2);
    b.vapour(12, 12, 147, 87, 4);
}

function coldLid(b) {
    b.floor();
    b.box("WALL", 10, 10, 149, 89, 2);
    b.rect("COOLER", 50, 12, 109, 13);
    b.vapour(12, 14, 147, 87, 0.9);
}

function warmPool(b) {
    b.floor();
    b.box("WALL", 10, 10, 149, 89, 2);
    b.rect("WATER", 12, 78, 147, 87, { temp: 60 });
    b.vapour(12, 12, 147, 77, 0.5);
}

// (The vessels stand on the floor: hung over it, only the heater held the
// steel up, and its lid came loose and fell in.)
function sealedBoiler(b) {
    b.floor();
    b.box("METAL", 38, 50, 81, 97, 2);
    b.rect("AIR", 40, 50, 79, 50);
    b.rect("GLASS", 40, 51, 79, 51);
    b.rect("WATER", 40, 74, 79, 95);
    b.rect("HEATER", 58, 94, 61, 95);
}

function pressureCooker(b) {
    b.floor();
    b.box("METAL", 30, 56, 89, 97, 2);
    b.rect("WATER", 32, 78, 87, 95);
    b.rect("HEATER", 58, 94, 61, 95);
}

// ---------------------------------------------------------------- fire

// The box holds the oxygen to burn a third of its wood: wood takes some 14
// cells of fresh air's oxygen a cell (stage 5). Its walls let no heat out,
// so its air heats up as the fire burns, a flame burning on in less oxygen
// the hotter it is (holdsFlame), until it is as hot as burning makes
// anything (FLAME_MAX_T), and the fire goes out with oxygen and wood left.
function fireBox(b) {
    b.floor();
    b.box("WALL", 48, 46, 111, 89, 2);
    b.rect("WOOD", 60, 76, 99, 87, { temp: 520 });
}

// A bed of embers (coal hot through, which sinks) the water poured on it
// covers. Wet, it can't burn; hot through, it would light again wherever it
// was left uncovered, there being no cooling a centimetre of it through in
// seconds (and wood, which floats, would float up and burn on the water).
function dousing(b) {
    b.floor();
    b.box("WALL", 10, 10, 109, 69, 2);
    b.rect("COAL", 45, 63, 74, 67, { temp: 520 });
    b.rect("WATER", 45, 14, 74, 29);
}

// Two equal charges of gunpowder going off on the floor, one shut in a
// sealed box of Wall, one in the open, each with a gauge 12 cells off
// (checks.mjs): the gas it makes can't get away from the one shut in.
function charges(b) {
    b.floor();
    b.box("WALL", 10, 36, 69, 77, 2);
    b.rect("GUNPOWDER", 25, 72, 27, 77, { temp: 200 });
    b.rect("GUNPOWDER", 115, 72, 117, 77, { temp: 200 });
}

// Two sealed jars of Wall, each with a hot steel rod in it (from its wall,
// which holds it up): one of hydrogen over air, one of hydrogen alone.
function hydrogenJars(b) {
    b.floor();
    b.box("WALL", 10, 10, 49, 49, 2);
    b.rect("HYDROGEN", 12, 12, 47, 29);
    b.rect("METAL", 12, 28, 31, 31, { temp: 700 });
    b.box("WALL", 70, 10, 109, 49, 2);
    b.rect("HYDROGEN", 72, 12, 107, 47);
    b.rect("METAL", 72, 28, 91, 31, { temp: 700 });
}

// A sealed box of Wall packed with wood, a heater for its floor: no air
// in it at all.
function kiln(b) {
    b.floor();
    b.box("WALL", 20, 10, 79, 49, 2);
    b.rect("WOOD", 22, 12, 77, 47);
    b.rect("HEATER", 22, 46, 77, 47);
}

// A stack of wood in the open on the floor, lit by two rows of flames on top.
function bonfire(b) {
    b.floor();
    b.rect("WOOD", 44, 70, 55, 77);
    b.rect("FIRE", 44, 68, 55, 69);
}

function lavaAndWater(b) {
    b.floor(4);
    b.rect("STONE", 0, 80, 159, 95);
    b.rect("STONE", 28, 66, 29, 79);
    b.rect("STONE", 70, 66, 71, 79);
    b.rect("LAVA", 30, 72, 69, 79);
    b.rect("WATER", 30, 40, 69, 49);
}

// Two blocks of steel at 800 °C under water, one basin of it at 20 °C, the
// other just off the boil: the cold water wets the steel and boils off it,
// quenching it in seconds; the boiling water's steam holds it off (film
// boiling, the Leidenfrost effect), and it cools several times slower
// (REWRITE_PLAN.md 4e-5: water's film by the boiling curve).
function hotSteel(b) {
    b.floor();
    for (const [x0, temp] of [[10, 20], [70, 99]]) {
        b.rect("WALL", x0, 50, x0 + 1, 77);
        b.rect("WALL", x0 + 38, 50, x0 + 39, 77);
        b.rect("WATER", x0 + 2, 54, x0 + 37, 77, { temp });
        b.rect("METAL", x0 + 17, 68, x0 + 22, 77, { temp: 800 });
    }
}

// ---------------------------------------------------------------- solutes

// A sealed box of wall brim-full of fresh water, no air in it, to freeze
// with the cool tool and thaw with the heat tool. Ice takes more room than
// its water, and with none to take it owes the rest (common.wgsl's
// formed); thawed, the water is all there again.
function frozenBox(b) {
    b.floor();
    b.box("WALL", 30, 20, 89, 71, 2);
    b.rect("WATER", 32, 22, 87, 69);
}

// A sealed crucible of wall packed with stone, to melt with the heat tool
// and let set again: lava takes more room than its stone, and with none to
// take it owes the rest (formed); set again, the stone is all there.
function crucible(b) {
    b.floor();
    b.box("WALL", 40, 30, 79, 69, 2);
    b.rect("STONE", 42, 32, 77, 67);
}

// A basin of stone holding water, to freeze with the cool tool. Freezing,
// the water stays at 0 °C while its latent heat goes (its stored
// temperature runs down some 80 below): the stone beside it feels it at
// 0 °C, some 5-8 MPa of thermal stress in its skin, under its strength
// (stress.wgsl's skinStress took the stored temperature, 20-30 MPa, and
// cracked it).
function frozenPond(b) {
    b.floor();
    b.rect("STONE", 20, 40, 99, 77);
    b.rect("WATER", 24, 44, 95, 73);
}

// A sealed glass bottle brim-full of water, to freeze with the cool tool:
// the ice, with no room to take, cracks the glass, which crumbles into
// shards (block.wgsl), as a full bottle left out in a frost bursts.
function frozenBottle(b) {
    b.floor();
    b.box("GLASS", 40, 30, 79, 77, 2);
    b.rect("WATER", 42, 32, 77, 75);
}

// A sealed box of wall brim-full of saturated brine, to freeze with the
// cool tool past salt's coldest (-21 °C) and thaw with the heat tool: it
// freezes into ice and grains of salt, and thaws into brine again, its
// water all there (update.wgsl's changePhase).
function frozenBrine(b) {
    b.floor();
    b.box("WALL", 30, 20, 89, 71, 2);
    b.rect("SALTWATER", 32, 22, 87, 69, { moist: 1 });
}

// A glass tank of fresh water with a heap of salt on its bottom.
function saltWater(b) {
    b.floor();
    b.tank("GLASS", 20, 20, 99, 77);
    b.rect("WATER", 21, 30, 98, 76);
    b.rect("SALT", 45, 70, 74, 76);
}

// A layer of coal on the ground (Wall, which takes no heat), in the open
// under the noon sun, the clock stopped.
function sunlitCoal(b) {
    b.floor(4);
    b.rect("COAL", 0, 74, 119, 75);
}

// Saturated brine 10 rows deep under fresh water, in a glass tank.
function brineLayer(b) {
    b.floor();
    b.tank("GLASS", 20, 20, 99, 77);
    b.rect("WATER", 21, 30, 98, 66);
    b.rect("SALTWATER", 21, 67, 98, 76, { moist: 1 });
}

// Two glass tanks of acid, one over a steel slab, one over a block of stone.
function acid(b) {
    b.floor();
    b.tank("GLASS", 10, 30, 59, 77);
    b.rect("METAL", 11, 72, 58, 76);
    b.rect("ACID", 11, 52, 58, 71);
    b.tank("GLASS", 70, 30, 119, 77);
    b.rect("STONE", 71, 72, 118, 76);
    b.rect("ACID", 71, 52, 118, 71);
}

// Two glass tanks of dry soil, sand and dirt, each with water poured on it.
function infiltration(b) {
    b.floor();
    b.tank("GLASS", 10, 30, 51, 97);
    b.rect("SAND", 11, 50, 50, 96, { moist: 0 });
    b.rect("WATER", 11, 40, 50, 45);
    b.tank("GLASS", 68, 30, 109, 97);
    b.rect("DIRT", 69, 50, 108, 96, { moist: 0 });
    b.rect("WATER", 69, 40, 108, 45);
}

// ---------------------------------------------------------------- liquids and bodies

function stillTank(b) {
    b.floor();
    b.tank("GLASS", 20, 30, 99, 77);
    b.rect("WATER", 21, 50, 98, 76);
}

function dam(b) {
    b.floor();
    b.rect("WATER", 10, 30, 59, 97);
}

function floaters(b) {
    b.floor();
    b.tank("GLASS", 20, 40, 139, 97);
    b.rect("WATER", 21, 60, 138, 96);
    b.rect("WOOD", 40, 28, 51, 35);
    b.rect("METAL", 100, 28, 105, 35);
}

function sandIntoWater(b) {
    b.floor();
    b.tank("GLASS", 20, 40, 99, 97);
    b.rect("WATER", 21, 60, 98, 96);
    b.rect("SAND", 45, 15, 74, 34);
}

// A tall block of wood hung loose just over a steep slope in a sealed box
// of foggy air, with sand falling beyond it: it drops onto the slope,
// topples, and tumbles down it through the fog onto the floor, where the
// sand falls. (Wood, as no landing at a body's speed breaks it: stone
// chips into gravel, a cell of it into one or two at random.)
function tumblingBlock(b) {
    b.floor();
    b.box("WALL", 10, 10, 149, 89, 2);
    b.vapour(12, 12, 147, 87, 1.5);
    for (let x = 12; x < 52; x++) {
        b.rect("WALL", x, 88 - (52 - x), x, 87);
    }
    b.rect("WOOD", 16, 40, 19, 49);
    b.rect("SAND", 90, 30, 101, 45);
}

// Cells 10 cm across (the scene's height): a block of glass over a stone
// floor, one over a wooden floor, and one of stone over stone, each let
// fall 4 m.
function drops(b) {
    b.floor(4);
    b.rect("STONE", 5, 90, 64, 95);
    b.rect("WOOD", 70, 90, 129, 95);
    b.rect("STONE", 135, 90, 194, 95);
    b.rect("GLASS", 30, 40, 37, 43);
    b.rect("GLASS", 95, 40, 102, 43);
    b.rect("STONE", 160, 40, 167, 43);
}

// Three stone blocks 10 by 3 on slopes of stairs going down to the right:
// 1 in 2 (26.6°), 3 in 4 (36.9°) and 1 in 1 (45°).
function slopes(b) {
    b.floor();
    for (const [x0, rise, run] of [[5, 1, 2], [70, 3, 4], [135, 1, 1]]) {
        for (let x = 0; x < 60; x++) {
            b.rect("WALL", x0 + x, 25 + Math.floor((x * rise) / run), x0 + x, 97);
        }
        const top = 25 + Math.floor((8 * rise) / run);
        b.rect("STONE", x0 + 8, top - 5, x0 + 17, top - 3);
    }
}

// Six stone crates hung one over another with a gap between each, every
// other one a cell to the side: they fall, the lowest lands, and each lands
// on the one below it.
function crateStack(b) {
    b.floor();
    b.rect("WALL", 0, 90, 119, 99);
    for (let k = 0; k < 6; k++) {
        const y = 82 - k * 9;
        b.rect("STONE", 50 + (k % 2), y, 59 + (k % 2), y + 3);
    }
}

// A wide tank of water with, let go just over it, a raft of wood, a tall
// post of wood and a sealed crate of glass. The liquid's own pressure floats
// them (REWRITE_PLAN.md P3-3): the raft level, six tenths of it under; the
// post, too tall and thin to float upright, tips over and floats on its
// side; the crate, though glass sinks, on the air shut in it, dry inside.
function rafts(b) {
    b.floor();
    b.tank("GLASS", 10, 40, 209, 97);
    b.rect("WATER", 11, 60, 208, 96);
    b.rect("WOOD", 16, 55, 35, 58);
    b.rect("WOOD", 100, 44, 103, 57);
    b.box("GLASS", 176, 49, 191, 58, 1);
}

// A wide wooden hull with high sides and an open hold, let go with its floor
// 38 rows under the water (as if pushed down and let go): it rises, bobs
// and floats, the water calm round it (REWRITE_PLAN.md P3-11: the middle of
// its hold, more than a block or two from the water, was out of the
// liquid's solve, air at no pressure under the waterline there, and the
// water came up through its floor at it, the hull thrown about and the
// water at hundreds of cells a step).
function deepHull(b) {
    b.floor();
    b.tank("WALL", 5, 20, 234, 147, 3);
    b.rect("WATER", 8, 60, 231, 144);
    b.rect("WOOD", 60, 30, 179, 97);
    b.rect("AIR", 66, 30, 173, 89);
}

// Two wooden rafts with a rail at each end let go in a tank of water,
// about as deep as they float, one with a heap of gravel on it. The gravel
// rides on the raft and weighs it down: it floats lower by the gravel's
// weight (REWRITE_PLAN.md P3-4).
function cargo(b) {
    b.floor();
    b.tank("GLASS", 10, 40, 189, 97);
    b.rect("WATER", 11, 60, 188, 96);
    for (const [x, y] of [[25, 57], [115, 58]]) {
        b.rect("WOOD", x, y, x + 39, y + 7);
        b.rect("WOOD", x, y - 3, x, y - 1);
        b.rect("WOOD", x + 39, y - 3, x + 39, y - 1);
    }
    b.rect("GRAVEL", 129, 56, 140, 57);
}

// Five hundred loose bits of stone and metal, a cell or two each and a cell
// apart, let go in a tank of water: they sink and settle into a heap, the
// water flowing round and between them (REWRITE_PLAN.md 7-19: particles of
// it brought onto one point moved as one for good, 2980 of them in groups
// of up to 93, each a cell's worth or more packed into one cell with a
// bubble round it).
function rubble(b) {
    b.floor();
    b.tank("GLASS", 10, 20, 189, 117);
    b.rect("WATER", 11, 30, 188, 116);
    for (let y = 36; y < 80; y += 3) {
        for (let x = 50; x < 150; x += 3) {
            b.rect((x + y) % 2 ? "STONE" : "METAL", x, y, x + (x * 7 + y * 3) % 2, y + (x * 5 + y) % 2);
        }
    }
}

// (Glass two cells thick: a cell thick, 78 tall, holding 67 of water, its
// walls bore some 30 MPa at their foot, near glass's 40, and cracked at the
// corner: REWRITE_PLAN.md 8-2.)
// A wooden tank full of water, held by nothing, over the floor.
function fallingTank(b) {
    b.floor();
    b.rect("WOOD", 40, 30, 41, 61);
    b.rect("WOOD", 80, 30, 81, 61);
    b.rect("WOOD", 40, 62, 81, 63);
    b.rect("WATER", 42, 36, 79, 61);
}

// A wooden boat, its hold full of sand to its middle, let go over a tank
// of water. Its sides stand 9 rows over its waterline: the water comes
// into a hold over its rim (REWRITE_PLAN.md P3-22), and with 3.5 rows, its
// rim went under as it plunged and pitched 0.2 radians, and it swamped
// and sank.
function ladenBoat(b) {
    b.floor();
    b.tank("GLASS", 10, 40, 189, 97);
    b.rect("WATER", 11, 65, 188, 96);
    b.rect("WOOD", 80, 50, 111, 51);
    b.rect("WOOD", 80, 32, 81, 49);
    b.rect("WOOD", 110, 32, 111, 49);
    b.rect("SAND", 82, 44, 109, 49);
}

function bubble(b) {
    b.floor();
    // Walls 3 cells thick: 2 were at their strength at the bottom corners
    // under the water's weight (failure index 0.9, cracking now and then;
    // REWRITE_PLAN.md 7-1), and the scene is the bubble's, not the tank's.
    b.box("GLASS", 18, 18, 81, 99, 3);
    b.rect("WATER", 21, 30, 78, 96);
    b.rect("AIR", 45, 82, 54, 91);
}

// Oil under water: a sealed glass tank, a layer of oil on its floor and
// water over it. The oil, lighter, rises through the water and floats on
// it, the two keeping apart, neither made nor lost (REWRITE_PLAN.md 7-4).
function oilAndWater(b) {
    b.floor();
    b.box("GLASS", 18, 18, 81, 99, 3);
    b.rect("OIL", 21, 81, 78, 96);
    b.rect("WATER", 21, 45, 78, 80);
}

// Two blocks of lava either side of a wall on a floor, one at 1400 °C and one
// at 1100: lava thickens some tenfold for each 115 K it cools (REWRITE_PLAN.md
// 7-17), so the hot one runs out like syrup and the cooler one barely
// creeps.
function hotLava(b) {
    b.rect("WALL", 0, 56, 199, 59);
    b.rect("WALL", 98, 20, 101, 55);
    b.rect("LAVA", 78, 36, 97, 55, { temp: 1400 });
    b.rect("LAVA", 102, 36, 121, 55, { temp: 1100 });
}

// Two tanks of oil the same, one at 20 °C and one at 80 °C, at night: the
// warm oil takes more room, by its thermal expansion, 7.2e-4 a degree, some
// 4 % (REWRITE_PLAN.md 7-16).
function warmOil(b) {
    b.floor();
    b.rect("WALL", 0, 10, 3, 77);
    b.rect("WALL", 46, 10, 53, 77);
    b.rect("WALL", 96, 10, 99, 77);
    b.rect("OIL", 4, 38, 45, 77, { temp: 20 });
    b.rect("OIL", 54, 38, 95, 77, { temp: 80 });
}

// Mud (ν 100 cells² a step) fed from a reservoir down a channel 20 cells
// wide into a basin: a thick liquid's slow flow, at plane Poiseuille
// flow's speed under its pull and the pressure's fall along the channel
// (REWRITE_PLAN.md 7-11).
function thickFlow(b) {
    b.floor();
    b.rect("WALL", 0, 4, 3, 39);
    b.rect("WALL", 76, 4, 79, 39);
    b.rect("WALL", 0, 40, 29, 140);
    b.rect("WALL", 50, 40, 79, 140);
    b.rect("MUD", 4, 4, 75, 39);
    b.rect("MUD", 30, 40, 49, 140);
}

// A column of mud in a tank, a speck of air left in it: the mud stands
// still from the first (REWRITE_PLAN.md 7-11, 7-15), and the speck stays a
// bubble where it is, which in mud that thick creeps up at Hadamard and
// Rybczynski's 2e-4 cells a step (7-10).
function stillMud(b) {
    b.floor();
    b.tank("WALL", 18, 20, 41, 97, 2);
    b.rect("MUD", 20, 36, 39, 95);
    b.rect("AIR", 29, 60, 30, 61);
}

// Two blocks of stone let go in two tanks, one of mud, one of water: in
// water a stone sinks at its drag's terminal speed, most of a metre a
// second; through mud this thick it creeps down as a cylinder in Stokes
// flow does (Lamb's), some hundredth of a cell a step (REWRITE_PLAN.md
// 7-13).
function sinking(b) {
    b.floor();
    b.tank("WALL", 10, 20, 55, 117, 2);
    b.tank("WALL", 64, 20, 109, 117, 2);
    b.rect("MUD", 12, 30, 53, 115);
    b.rect("WATER", 66, 30, 107, 115);
    b.rect("STONE", 31, 36, 34, 39);
    b.rect("STONE", 85, 36, 88, 39);
}

// A block of steel at 80 °C on a plinth in a tank of water at 20 °C, at
// night: on the day's clock the water round it turns over as natural
// convection does (REWRITE_PLAN.md 4e-8), carrying its heat up and away.
function warmSteel(b) {
    b.floor();
    b.tank("WALL", 0, 20, 119, 97, 4);
    b.rect("WATER", 4, 24, 115, 93);
    b.rect("WALL", 57, 65, 62, 93);
    b.rect("METAL", 55, 55, 64, 64, { temp: 80 });
}

// ---------------------------------------------------------------- structure

function loadedBeam(b) {
    b.floor(4);
    b.rect("STONE", 0, 80, 159, 95);
    b.rect("WALL", 40, 50, 44, 79);
    b.rect("WALL", 115, 50, 119, 79);
    b.rect("STONE", 45, 50, 114, 53);
    b.rect("SAND", 60, 15, 99, 44);
}

// Three steel beams a cell deep across 56 cells between piers, each under
// a pile of scrap 30 rows high: one at 20 °C, one at 450, one at 650
// (cells 6 cm across: the scene's height).
function hotBeams(b) {
    b.floor();
    [20, 450, 650].forEach((temp, k) => {
        const x = 5 + 65 * k;
        b.rect("WALL", x, 20, x + 3, 97);
        b.rect("WALL", x + 60, 20, x + 63, 97);
        b.rect("METAL", x + 4, 60, x + 59, 60, { temp });
        b.rect("SCRAP", x + 4, 30, x + 59, 59);
    });
}

function cutTower(b) {
    b.floor(4);
    b.rect("STONE", 0, 100, 119, 115);
    b.rect("STONE", 55, 40, 64, 95);
}

// A stone slab across two wall posts (REWRITE_PLAN.md 8-1): taking one
// post away leaves it held up by the other, connected to it; taking both
// leaves it connected to nothing, and it falls at once.
function posts(b) {
    b.floor(4);
    b.rect("WALL", 50, 70, 52, 95);
    b.rect("WALL", 68, 70, 70, 95);
    b.rect("STONE", 46, 66, 74, 69);
}

// Two steel bars side by side in the air, the left held by nothing, the
// right on a wall post under its far end. The left falls at once, as a
// body; taken off its post, the right falls too, every cell of it
// (REWRITE_PLAN.md P3-9: let go as the left was becoming a body, the cells
// of the right in the tile beside the left's were taken into the left's
// slot, in no body's shape, and hung in the air for good).
function twoBars(b) {
    b.floor(4);
    b.rect("METAL", 20, 50, 27, 52);
    b.rect("METAL", 36, 50, 100, 52);
    b.rect("WALL", 98, 53, 100, 75);
}

// A floor and nothing else, to draw on (the check draws a bar of steel in
// the air, a stretch a step, as the app's draw tool does while the pointer
// is down): what a stroke paints stays where it is until it is let go, then
// falls as one piece (REWRITE_PLAN.md P3-10: let go as it was painted, each
// step's stretch came loose and fell on its own as the next was drawn).
function drawnBar(b) {
    b.floor(4);
}

function hotSlab(b) {
    b.floor();
    b.rect("STONE", 0, 50, 119, 77);
    b.rect("STONE", 38, 40, 39, 49);
    b.rect("STONE", 80, 40, 81, 49);
    b.rect("LAVA", 40, 44, 79, 49);
}

// Two blocks of glass at 250 °C on the floor, one left to cool in the air,
// the other in a basin water pours into: quenched, its skin shrinks against
// its hot inside and cracks (REWRITE_PLAN.md 8-5: thermal shock as thermal
// strain, borosilicate's some 140 °C step at its strength); the one in the
// air cools too slowly to. The basin's walls stand well over its water, so
// the splash of its boiling stays in: over walls level with it, water ran
// along the floor to the other block in some runs, and its touch all but
// cracked it.
function quench(b) {
    b.floor();
    b.rect("GLASS", 20, 68, 29, 77, { temp: 250 });
    b.rect("WALL", 72, 44, 73, 77);
    b.rect("WALL", 96, 44, 97, 77);
    b.rect("GLASS", 80, 68, 89, 77, { temp: 250 });
    b.rect("WATER", 74, 40, 95, 52);
}

// ---------------------------------------------------------------- current and light

function brineCell(b) {
    b.floor();
    b.tank("GLASS", 40, 40, 79, 77);
    // Sea water, a seventh as salty as saturated brine.
    b.rect("SALTWATER", 41, 50, 78, 76, { moist: 0.14 });
    b.rect("WALL", 44, 20, 47, 29);
    b.rect("WALL", 72, 20, 75, 29);
    b.rect("METAL", 48, 28, 71, 29);
    b.rect("METAL", 48, 30, 49, 70);
    b.rect("METAL", 70, 30, 71, 70);
    b.rect("BATTERY", 58, 28, 61, 29, { angle: RIGHT });
}

function hotWire(b) {
    b.floor();
    b.rect("WALL", 20, 62, 25, 77);
    b.rect("WALL", 94, 62, 99, 77);
    b.rect("METAL", 20, 30, 99, 31);
    b.rect("METAL", 20, 30, 21, 61);
    b.rect("METAL", 98, 30, 99, 61);
    b.rect("METAL", 20, 60, 99, 61);
    b.rect("AIR", 55, 61, 64, 61);
    b.rect("BATTERY", 20, 40, 21, 43, { angle: DOWN });
}

// Three steel wires, each sparked at its near end: one to a block of wood,
// one to a heap of gunpowder, one into a glass box of hydrogen over air;
// and a fourth like the first, not sparked, in the same sun.
function sparks(b) {
    b.floor();
    b.rect("WALL", 5, 52, 37, 53);
    b.rect("METAL", 5, 50, 30, 51);
    b.rect("WOOD", 31, 44, 37, 51);
    b.rect("WALL", 45, 52, 77, 53);
    b.rect("METAL", 45, 50, 70, 51);
    b.rect("GUNPOWDER", 71, 48, 77, 51);
    b.box("GLASS", 115, 30, 145, 52);
    b.rect("HYDROGEN", 116, 31, 144, 41);
    b.rect("WALL", 85, 43, 114, 44);
    b.rect("METAL", 85, 41, 125, 42);
    b.rect("WALL", 160, 52, 192, 53);
    b.rect("METAL", 160, 50, 185, 51);
    b.rect("WOOD", 186, 44, 192, 51);
    for (const [x, y] of [[5, 50], [45, 50], [85, 41]]) {
        b.tool(BRUSH.SPARK, x, y, x + 1, y + 1, 1);
    }
}

// Three tanks of fresh water, two steel electrodes in each, driven by a
// battery of one cell, two and four (1.5, 3 and 6 V).
function splittingWater(b) {
    b.floor();
    [1, 2, 4].forEach((n, k) => {
        const x = 10 + 52 * k;
        b.tank("GLASS", x, 40, x + 39, 77);
        b.rect("WATER", x + 1, 50, x + 38, 76);
        b.rect("WALL", x + 4, 20, x + 7, 29);
        b.rect("WALL", x + 32, 20, x + 35, 29);
        b.rect("METAL", x + 8, 28, x + 31, 29);
        b.rect("METAL", x + 8, 30, x + 9, 70);
        b.rect("METAL", x + 30, 30, x + 31, 70);
        const x0 = x + 20 - Math.ceil(n / 2);
        b.rect("BATTERY", x0, 28, x0 + n - 1, 29, { angle: RIGHT });
    });
}

// Three tanks of dirt, dry, damp and twice as damp, a battery's two steel
// rods in each.
function dampGround(b) {
    b.floor();
    [0, 0.4, 0.8].forEach((moist, k) => {
        const x = 10 + 52 * k;
        b.tank("GLASS", x, 40, x + 39, 77);
        b.rect("DIRT", x + 1, 46, x + 38, 76, { moist });
        b.rect("WALL", x + 6, 20, x + 9, 29);
        b.rect("WALL", x + 30, 20, x + 33, 29);
        b.rect("METAL", x + 10, 28, x + 29, 29);
        b.rect("METAL", x + 10, 30, x + 11, 70);
        b.rect("METAL", x + 28, 30, x + 29, 70);
        b.rect("BATTERY", x + 18, 28, x + 21, 29, { angle: RIGHT });
    });
}

function greenhouse(b) {
    b.floor(4);
    b.rect("STONE", 0, 85, 199, 95);
    b.rect("GLASS", 120, 55, 179, 85);
    b.rect("AIR", 121, 56, 178, 84);
}

function glowAcrossGap(b) {
    b.box("WALL", 5, 5, 114, 54, 2);
    b.rect("STONE", 20, 25, 29, 52, { temp: 1000 });
    b.rect("WOOD", 50, 25, 59, 52);
}

// ---------------------------------------------------------------- world

function seedling(b) {
    b.floor();
    b.rect("DIRT", 0, 80, 99, 97, { moist: 0.6 });
    for (const x of [30, 50, 70]) {
        b.rect("SEED", x, 78, x + 1, 79);
    }
}

// A sealed glass box of damp dirt with seeds on it, in the sun. (Its glass
// two cells thick: a cell thick, its lid, 80 across, cracked as the sun
// warmed the air sealed under it, at a tenth of an atmosphere over:
// REWRITE_PLAN.md 8-2.)
function terrarium(b) {
    b.floor();
    b.box("GLASS", 19, 29, 100, 78, 2);
    b.rect("DIRT", 21, 64, 98, 76, { moist: 0.7 });
    for (const x of [35, 55, 75]) {
        b.rect("SEED", x, 62, x + 1, 63);
    }
}

function villager(b) {
    b.floor();
    b.rect("STONE", 0, 85, 199, 97);
    b.rect("WALL", 30, 60, 31, 84);
    b.person(40, 84);
    b.rect("WOOD", 46, 70, 58, 84, { temp: 520 });
    b.person(140, 84);
    b.rect("WATER", 150, 30, 175, 84);
}

// A villager in a narrow pit, and sand poured over it: the sand fills the pit
// round the person and lies on them, never in their body, and buried to the
// face they hold their breath (REWRITE_PLAN.md 10-1: till then the sand
// poured into the body, which climbed out half a cell a step or was
// crushed by the cells of it in it).
function buried(b) {
    b.floor();
    b.rect("STONE", 0, 85, 119, 97);
    b.rect("WALL", 55, 66, 56, 84);
    b.rect("WALL", 64, 66, 65, 84);
    b.person(60, 84);
    b.rect("SAND", 50, 25, 70, 50);
}

// Three villagers boxed in, one on a floor of steel at 60 °C, one on wood at
// 60 °C, one in water at 75 °C: the steel burns its feet in seconds (its
// skin's surface where they touch near 58 °C, steel passing heat so much
// better than skin), the wood doesn't (41 °C), and the water scalds at once
// (REWRITE_PLAN.md 10-2: Henriques' burns at the skin's surface).
function hotFloors(b) {
    b.floor();
    b.rect("STONE", 0, 85, 159, 97);
    for (const [x0, floor] of [[10, "METAL"], [60, "WOOD"]]) {
        b.rect("WALL", x0, 60, x0 + 1, 84);
        b.rect("WALL", x0 + 19, 60, x0 + 20, 84);
        b.rect(floor, x0 + 2, 82, x0 + 18, 84, { temp: 60 });
        b.person(x0 + 10, 82);
    }
    b.rect("WALL", 110, 60, 111, 84);
    b.rect("WALL", 129, 60, 130, 84);
    b.rect("WATER", 112, 76, 128, 84, { temp: 75 });
    b.person(120, 84);
}

function holeInSpace(b) {
    b.disc("BLACK_HOLE", 80, 80, 3);
    b.rect("SAND", 77, 35, 82, 40);
    b.rect("SAND", 77, 120, 82, 125);
    b.rect("SAND", 35, 77, 40, 82);
    b.rect("SAND", 120, 77, 125, 82);
}

// A steel bar falling onto a small black hole, in space: the hole eats a
// slot through its middle as it passes, and cut in two it goes on as two
// bodies, each pulled on its own (REWRITE_PLAN.md P3-9: the halves went on
// as one body about the hole, its middle in it).
function barHole(b) {
    b.disc("BLACK_HOLE", 80, 80, 2);
    b.rect("METAL", 65, 40, 95, 45);
}

// Single grains of sand on a floor beside a small black hole, level with
// it, every other cell from 6 to 40 cells off: near it the hole's pull along
// the floor is more than a grain's friction (tan 34° of its weight) and
// draws it in; farther, it is less, and the grain stays put however long
// (REWRITE_PLAN.md 9-1: losing a share of its speed a step, grains on a
// floor wandered a few cells under any push along it).
function sandByHole(b) {
    b.floor();
    b.disc("BLACK_HOLE", 30, 76, 3);
    for (let x = 36; x <= 70; x += 2) {
        b.rect("SAND", x, 77, x, 77);
    }
}

export const SCENE_GROUPS = ["Air", "Water", "Fire", "Solutes", "Liquids and bodies", "Structure", "Current and light", "World"];

// Experiments with the clock stopped (stopClock) play out the same whenever
// they are watched; the rest need the sun.
export const SCENES = [
    // Air
    {
        id: "chimney", group: "Air", name: "Chimney", size: [120, 200], hour: 0, stopClock: true, build: chimney,
        hint: "A heater at the foot of a flue warms its air, which rises out of the top and draws the room's air in at the foot: a chimney's draft.",
    },
    {
        id: "heated-room", group: "Air", name: "Heated room", size: [120, 80], hour: 12, stopClock: true, build: heatedRoom,
        hint: "A sealed box with a heater for a floor. The air warms, its pressure follows the gas law, and none of its water is made or lost. Press P for the pressure.",
    },
    {
        id: "fan-room", group: "Air", name: "Fan room", size: [160, 80], hour: 12, stopClock: true, build: fanRoom,
        hint: "Left, a fan in a sealed room stirs the air round without changing its pressure. Right, a fan set in the wall between two sealed rooms pushes air from one into the other, so the right room's pressure rises above the left's. Press P.",
    },
    {
        id: "smoke-wind", group: "Air", name: "Smoke in the wind", size: [200, 100], hour: 12, stopClock: true, build: smokeInWind,
        hint: "A fan blows across open ground. The puff of smoke in front of it goes downwind with the air.",
    },
    {
        id: "splash", group: "Air", name: "Splash", size: [160, 100], hour: 12, stopClock: true, build: splash,
        hint: "A block of water falls into a tank of water. Its splash shuts pockets of air in the water, each a room of its own until it rises out.",
    },
    {
        id: "still-air", group: "Air", name: "Still air", size: [200, 100], hour: 22.5, stopClock: true, build: stillAir,
        hint: "Nothing but the open air, at night, as warm and as humid as the air outside it is: nothing moves it, and it stays still.",
    },
    {
        id: "plume", group: "Air", name: "Plume", size: [160, 120], hour: 12, stopClock: true, build: plume,
        hint: "Hot air rises from the heater in a plume. The smoke on the left is no warmer than the air around it, so it hangs where it is.",
    },
    {
        id: "hole-room", group: "Air", name: "Black hole room", size: [120, 100], hour: 12, stopClock: true, build: holeRoom,
        hint: "A small black hole in a sealed room swallows the air that flows into it, so the room's pressure falls. Press P.",
    },
    {
        id: "charge-wall", group: "Air", name: "Charge and wall", size: [200, 100], hour: 12, stopClock: true, build: chargeAndWall,
        hint: "Four equal charges, each against a stone wall 1, 2, 4 and 8 cells thick. The thinner the wall, the more of it breaks; the thickest holds.",
    },

    // Water
    {
        id: "fog-box", group: "Water", name: "Fog box", size: [160, 100], hour: 12, stopClock: true, build: fogBox,
        hint: "Sealed air holding four times the vapour it can at 20 °C. The extra condenses into a fog, which slowly settles, and the box keeps every bit of its water. Press M for the moisture.",
    },
    {
        id: "cold-lid", group: "Water", name: "Cold lid", size: [160, 100], hour: 12, stopClock: true, build: coldLid,
        hint: "Humid air sealed in under a cooler. Dew and frost gather on the cold strip, and the box keeps its water.",
    },
    {
        id: "warm-pool", group: "Water", name: "Warm pool", size: [160, 100], hour: 12, stopClock: true, build: warmPool,
        hint: "A warm pool sealed in a box evaporates until the air over it is saturated, cooling as it pays the vapour's latent heat. No water is made or lost.",
    },
    {
        id: "sealed-boiler", group: "Water", name: "Sealed boiler", size: [120, 100], hour: 12, stopClock: true, build: sealedBoiler,
        hint: "A heater in the water of a sealed steel boiler with a glass lid. As it boils the steam's pressure builds, and the water boils hotter than 100 °C, until the glass lid gives where the pressure bends it past its strength, about half an atmosphere over. Press P.",
    },
    {
        id: "pressure-cooker", group: "Water", name: "Pressure cooker", size: [120, 100], hour: 12, stopClock: true, build: pressureCooker,
        hint: "A heater in the water of a sealed steel pot. The steam presses on the water too, so it boils hotter than 100 °C, and the steel holds an atmosphere over and more. Press P.",
    },

    // Fire
    {
        id: "fire-box", group: "Fire", name: "Fire box", size: [160, 100], hour: 12, stopClock: true, build: fireBox,
        hint: "A wood fire in a sealed box. It burns until it has used up the oxygen, and the heat it gives is what that oxygen burns. Press O for the oxygen.",
    },
    {
        id: "dousing", group: "Fire", name: "Dousing", size: [120, 80], hour: 12, stopClock: true, build: dousing,
        hint: "Water poured on a fire puts it out; the water that boils away takes its latent heat from the fire.",
    },
    {
        id: "charges", group: "Fire", name: "Charges", size: [160, 80], hour: 12, stopClock: true, build: charges,
        hint: "Two equal charges of gunpowder go off, one shut in a sealed box, one in the open. The gas it makes can't get away from the one shut in: it hits harder, and its box stays pressed. Press P.",
    },
    {
        id: "hydrogen-jars", group: "Fire", name: "Hydrogen jars", size: [120, 60], hour: 12, stopClock: true, build: hydrogenJars,
        hint: "Two sealed jars with a red-hot steel rod in each: one of hydrogen over air, one of hydrogen alone. The hydrogen meeting the air burns into water; with no air, it doesn't burn at all.",
    },
    {
        id: "kiln", group: "Fire", name: "Charcoal kiln", size: [100, 60], hour: 12, stopClock: true, build: kiln,
        hint: "Wood packed into a sealed box on a heater, with no air: it can't burn, so it chars, into a little over half as much charcoal and the smoke its volatiles make.",
    },
    {
        id: "bonfire", group: "Fire", name: "Bonfire", size: [100, 80], hour: 12, build: bonfire,
        hint: "A stack of wood lit in the open. It burns down through glowing charcoal and leaves only a little ash, as burnt wood does.",
    },
    {
        id: "lava-water", group: "Fire", name: "Lava and water", size: [160, 100], hour: 12, stopClock: true, build: lavaAndWater,
        hint: "Water poured onto lava boils into steam, and the lava's skin sets into stone.",
    },
    {
        id: "crucible", group: "Fire", name: "Melted and set", size: [120, 100], hour: 12, stopClock: true, build: crucible,
        hint: "A sealed crucible packed with stone. Melt it with the heat tool (H), then cool it (C): the lava takes more room than its stone and keeps what it can't fit, so the stone sets again whole.",
    },
    {
        id: "hot-steel", group: "Water", name: "Hot steel", size: [120, 80], hour: 12, stopClock: true, build: hotSteel,
        hint: "Steel at 800 °C under cold water and under water just off the boil: the cold water boils off it and quenches it in seconds; the boiling water's own steam holds it off, and it cools far slower.",
    },
    {
        id: "frozen-box", group: "Water", name: "Frozen and thawed", size: [120, 80], hour: 12, stopClock: true, build: frozenBox,
        hint: "A sealed box brim-full of water. Freeze it with the cool tool (C) and thaw it with the heat tool (H): the ice takes more room than its water and keeps what it can't fit, so the water comes back whole.",
    },
    {
        id: "frozen-pond", group: "Water", name: "Frozen pond", size: [120, 80], hour: 0, stopClock: true, build: frozenPond,
        hint: "A stone basin holding water. Cool the water (C) and it freezes at 0 °C, giving off its latent heat while it stays there: the stone beside it feels 0 °C, and doesn't crack.",
    },
    {
        id: "frozen-bottle", group: "Water", name: "Frozen bottle", size: [120, 80], hour: 12, stopClock: true, build: frozenBottle,
        hint: "A sealed glass bottle brim-full of water. Freeze it with the cool tool (C): the ice has no room to grow, so it cracks the glass, which breaks into shards.",
    },

    // Solutes
    {
        id: "frozen-brine", group: "Solutes", name: "Frozen brine", size: [120, 80], hour: 12, stopClock: true, build: frozenBrine,
        hint: "A sealed box brim-full of saturated brine. Cool it (C) past -21 °C and it freezes into ice and grains of salt; warm it (H) and it is brine again, all its water there.",
    },
    {
        id: "salt-water", group: "Solutes", name: "Salt in water", size: [120, 80], hour: 12, stopClock: true, build: saltWater,
        hint: "A heap of salt on the bottom of a tank of fresh water dissolves as fast as the water beside it is short of saturation. The brine stays at the bottom, and not a grain's worth of salt is made or lost.",
    },
    {
        id: "brine-layer", group: "Solutes", name: "Brine layer", size: [120, 80], hour: 12, stopClock: true, build: brineLayer,
        hint: "Saturated brine under fresh water. Salt diffuses as slowly as it does in still water, a centimetre in about a day, so the two stay layered.",
    },
    {
        id: "acid", group: "Solutes", name: "Acid", size: [130, 80], hour: 12, stopClock: true, build: acid,
        hint: "Acid over steel and over stone. It eats each as far as it has the strength, a cell of steel taking 23 cells of acid and a cell of stone 5, steel fizzing hydrogen and stone carbon dioxide, and the acid warms.",
    },

    {
        id: "infiltration", group: "Solutes", name: "Infiltration", size: [120, 100], hour: 12, stopClock: true, build: infiltration,
        hint: "Water poured on dry sand and on dry dirt soaks in as their pores pull it and its weight takes it down: quickly through sand, slowly through dirt, which holds more of it against gravity.",
    },

    // Liquids and bodies
    {
        id: "still-tank", group: "Liquids and bodies", name: "Still tank", size: [120, 80], hour: 12, stopClock: true, build: stillTank,
        hint: "A tank of still water comes to rest and stays there, its pressure growing with depth. Press P.",
    },
    {
        id: "dam", group: "Liquids and bodies", name: "Dam", size: [200, 100], hour: 12, stopClock: true, build: dam,
        hint: "A column of water let go slumps and spreads across the floor. Not a drop is lost.",
    },
    {
        id: "floaters", group: "Liquids and bodies", name: "Floaters", size: [160, 100], hour: 12, stopClock: true, build: floaters,
        hint: "A block of wood and a block of steel dropped into water: the wood floats, the steel sinks, and the water keeps every drop.",
    },
    {
        id: "sand-water", group: "Liquids and bodies", name: "Sand into water", size: [120, 100], hour: 12, stopClock: true, build: sandIntoWater,
        hint: "Sand poured into a tank of water sinks to the bottom, and the water rises by the sand's volume.",
    },
    {
        id: "tumbling-block", group: "Liquids and bodies", name: "Tumbling block", size: [160, 100], hour: 12, stopClock: true,
        build: tumblingBlock,
        hint: "A block of wood comes loose over a slope in a sealed box of fog and tumbles down it. It keeps every cell at every angle, and the fog and the falling sand it meets are pushed aside, none of them lost.",
    },
    {
        id: "drops", group: "Liquids and bodies", name: "Drops", size: [200, 100], height: 10, hour: 12, stopClock: true, build: drops,
        hint: "Blocks let fall 4 m (cells 10 cm across). A blow stresses both things it joins by how stiff and dense each is: glass breaks on stone, not on wood, and stone on stone holds.",
    },
    {
        id: "slopes", group: "Liquids and bodies", name: "Slopes", size: [200, 100], hour: 12, stopClock: true, build: slopes,
        hint: "Stone blocks on slopes of 27°, 37° and 45°. Their friction, 0.6, holds them up to 31°; past it they slide, the steeper the faster.",
    },
    {
        id: "crate-stack", group: "Liquids and bodies", name: "Crate stack", size: [120, 100], hour: 12, stopClock: true,
        build: crateStack,
        hint: "Six stone crates hung one over another fall and land on each other, and stand as a stack, every cell kept, none sunk into another.",
    },
    {
        id: "falling-tank", group: "Liquids and bodies", name: "Falling tank", size: [120, 80], hour: 12, stopClock: true, build: fallingTank,
        hint: "A wooden tank of water, held by nothing, falls to the floor with its water in it.",
    },
    {
        id: "laden-boat", group: "Liquids and bodies", name: "Laden boat", size: [200, 100], hour: 12, stopClock: true, build: ladenBoat,
        hint: "A wooden boat with sand in its hold dropped into a tank. It carries what is in its hold as it moves, so not a grain spills however it pitches.",
    },
    {
        id: "deep-hull", group: "Liquids and bodies", name: "Deep hull", size: [240, 150], hour: 12, stopClock: true, build: deepHull,
        hint: "A wide wooden hull with high sides, let go with its floor deep under the water. It rises, bobs and floats, and the water stays calm round it.",
    },
    {
        id: "rafts", group: "Liquids and bodies", name: "Rafts", size: [220, 100], hour: 12, stopClock: true, build: rafts,
        hint: "A raft and a tall post of wood and a sealed glass crate dropped into water. The raft floats level, six tenths under; the post tips over and floats on its side; the crate floats on the air inside it, though glass sinks, and stays dry.",
    },
    {
        id: "cargo", group: "Liquids and bodies", name: "Cargo", size: [200, 100], hour: 12, stopClock: true, build: cargo,
        hint: "Two wooden rafts let go in water, one carrying a heap of gravel between its rails. The gravel rides on it, and the loaded raft floats lower by its weight.",
    },
    {
        id: "rubble", group: "Liquids and bodies", name: "Rubble", size: [200, 120], hour: 12, stopClock: true, build: rubble,
        hint: "Five hundred bits of stone and metal let go in a tank of water sink and settle into a heap, and the water flows back round them, none of it left packed into a lump.",
    },
    {
        id: "bubble", group: "Liquids and bodies", name: "Bubble", size: [100, 100], hour: 12, stopClock: true, build: bubble,
        hint: "A pocket of air trapped at the bottom of a sealed tank rises to the surface and joins the air above, all of it.",
    },
    {
        id: "oil-water", group: "Liquids and bodies", name: "Oil and water", size: [100, 100], hour: 12, stopClock: true, build: oilAndWater,
        hint: "A layer of oil under water in a sealed tank rises through it and floats on top, the two keeping apart, neither made nor lost.",
    },
    {
        id: "hot-lava", group: "Liquids and bodies", name: "Hot and cool lava", size: [200, 60], hour: 12, stopClock: true, build: hotLava,
        hint: "Two blocks of lava either side of a wall, one hotter than the other: the hot lava runs out across the floor, the cooler, far thicker, hardly moves.",
    },
    {
        id: "warm-oil", group: "Liquids and bodies", name: "Warm oil", size: [100, 80], hour: 0, stopClock: true, build: warmOil,
        hint: "Two tanks of the same oil, one cold and one hot: the hot oil stands higher, taking more room as it expands.",
    },
    {
        id: "thick-flow", group: "Liquids and bodies", name: "Thick flow", size: [80, 200], hour: 12, stopClock: true, build: thickFlow,
        hint: "Mud runs from a reservoir down a narrow channel, slowest by the walls and fastest down the middle, at the speed viscous flow between two walls has.",
    },
    {
        id: "still-mud", group: "Liquids and bodies", name: "Still mud", size: [60, 100], hour: 12, stopClock: true, build: stillMud,
        hint: "A column of mud with a speck of air in it stands still from the first; the air stays where it is, too thick a liquid for a bubble that small to rise through.",
    },
    {
        id: "sinking", group: "Liquids and bodies", name: "Sinking", size: [120, 120], hour: 12, stopClock: true, build: sinking,
        hint: "Two stones, one in mud, one in water: the one in water drops to the bottom at once, the one in mud creeps down, held by the mud's drag.",
    },
    {
        id: "warm-steel", group: "Liquids and bodies", name: "Warm steel", size: [120, 100], hour: 0, stopClock: true, build: warmSteel,
        hint: "Steel at 80 °C in cold water warms it, and the warmed water rises off it in a plume, carrying the heat away, as natural convection does.",
    },

    // Structure
    {
        id: "loaded-beam", group: "Structure", name: "Loaded beam", size: [160, 100], hour: 12, stopClock: true, build: loadedBeam,
        hint: "Sand dropped on a stone beam between two piers. The beam bears the pile or breaks under it, and the rock below stays solid.",
    },
    {
        id: "hot-beams", group: "Structure", name: "Hot beams", size: [200, 100], height: 6, hour: 12, stopClock: true, build: hotBeams,
        hint: "Three steel beams under the same pile of scrap, at 20, 450 and 650 °C. Steel keeps its strength to 400 °C and has lost two thirds of it by 650, so the hottest beam gives way under the load the others bear.",
    },
    {
        id: "cut-tower", group: "Structure", name: "Cut tower", size: [120, 120], hour: 12, stopClock: true, build: cutTower,
        hint: "A stone tower with its foot cut away falls as one piece and lands whole.",
    },
    {
        id: "posts", group: "Structure", name: "Posts", size: [120, 100], hour: 12, stopClock: true, build: posts,
        hint: "A stone slab across two wall posts. Take one post away and the slab still stands on the other; take both and it falls at once.",
    },
    {
        id: "drawn-bar", group: "Structure", name: "Drawn in the air", size: [160, 80], hour: 12, stopClock: true, build: drawnBar,
        hint: "An empty room to draw in. Draw steel in the air in one stroke: it stays where you draw it until you let go, then falls as one piece.",
    },
    {
        id: "two-bars", group: "Structure", name: "Two bars", size: [160, 80], hour: 12, stopClock: true, build: twoBars,
        hint: "Two steel bars in the air: the left, held by nothing, falls at once. Take away the right one's post and it falls too, every cell of it.",
    },
    {
        id: "sunlit-coal", group: "Air", name: "Sunlit coal", size: [120, 80], hour: 12, stopClock: true, build: sunlitCoal,
        hint: "Coal in the noon sun warms until what it gives off to the sky and to the open air over it takes all the sunlight it takes in. The air over it stays the outdoor air's: what it warms rises and the open air takes its place.",
    },
    {
        id: "hot-slab", group: "Structure", name: "Hot slab", size: [120, 80], hour: 12, stopClock: true, build: hotSlab,
        hint: "A pool of lava on solid stone: the stone under it heats up but stays whole.",
    },
    {
        id: "quench", group: "Structure", name: "Quench", size: [120, 80], hour: 12, stopClock: true, build: quench,
        hint: "Two blocks of glass at 250 °C: water poured over one cracks it, its skin shrinking against its hot inside; the one in the air cools whole.",
    },

    // Current and light
    {
        id: "brine-cell", group: "Current and light", name: "Brine cell", size: [120, 80], hour: 12, stopClock: true, build: brineCell,
        hint: "A battery drives current through brine between two steel electrodes, and hydrogen bubbles off, as much as the charge passed makes. Hover a wire for volts and amps.",
    },
    {
        id: "hot-wire", group: "Current and light", name: "Hot wire", size: [120, 80], hour: 12, stopClock: true, build: hotWire,
        hint: "A battery drives current round a steel loop with one thin link. The link carries the same current through less metal, so it heats far more than the rest.",
    },
    {
        id: "sparks", group: "Current and light", name: "Sparks", size: [200, 80], hour: 12, stopClock: true, build: sparks,
        hint: "A static pulse runs along three steel wires. Its few millijoules set off the gunpowder and the hydrogen at their ends, but can't light the wood, and warm nothing. Press S to send another.",
    },
    {
        id: "splitting-water", group: "Current and light", name: "Splitting water", size: [170, 80], hour: 12, stopClock: true, build: splittingWater,
        hint: "Three tanks of fresh water between steel electrodes, driven by one battery cell, two and four. Splitting water takes some 1.8 V at the electrodes, so one cell's 1.5 V drives no current at all; two drive what their 3 V has past it, and four three and a half times that. Fresh water conducts little, so hydrogen comes off slowly. Hover the water between the electrodes for its current.",
    },
    {
        id: "damp-ground", group: "Current and light", name: "Damp ground", size: [170, 80], hour: 12, stopClock: true, build: dampGround,
        hint: "Three tanks of dirt, dry, damp and twice as damp, each with a battery driving current between two steel rods in it. Soil conducts through the water in its pores, by the square of how full of it they are (Archie's law): not at all dry, and four times as well twice as damp. Hover the dirt between the rods for the current through it.",
    },
    {
        id: "greenhouse", group: "Current and light", name: "Greenhouse", size: [200, 100], hour: 7, build: greenhouse,
        hint: "Two patches of ground at dawn, one under a glass hut. The glass lets the sunlight in and keeps the warmed air, so by noon it is warmer inside than out. Press T for the heat.",
    },
    {
        id: "glow-gap", group: "Current and light", name: "Glow across a gap", size: [120, 60], hour: 12, stopClock: true, build: glowAcrossGap,
        hint: "Red-hot stone facing a block of wood across a gap of air, sealed in. The wood warms in the stone's glow, and no heat is made or lost.",
    },

    // World
    {
        id: "seedling", group: "World", name: "Seedling", size: [100, 100], hour: 8, build: seedling,
        hint: "Seeds on damp soil in the morning sun sprout and grow on the water they draw from it.",
    },
    {
        id: "terrarium", group: "World", name: "Terrarium", size: [120, 80], hour: 9, build: terrarium,
        hint: "Seeds sprouting in a sealed glass box of damp soil, in the sun. The plants draw the soil's water up through their roots and give it off through their leaves, so the air grows damp, and not a drop leaves the box.",
    },
    {
        id: "villager", group: "World", name: "Villager", size: [200, 100], hour: 12, stopClock: true, build: villager,
        hint: "Two villagers: one caught between a wall and a fire, which burns them, and one beside a column of water that comes down and sweeps them along.",
    },
    {
        id: "buried", group: "World", name: "Buried", size: [120, 100], hour: 12, stopClock: true, build: buried,
        hint: "Sand poured over a villager in a pit fills round them and lies on them, never in their body; buried to the face, they hold their breath.",
    },
    {
        id: "hot-floors", group: "World", name: "Hot floors", size: [160, 100], hour: 12, stopClock: true, build: hotFloors,
        hint: "Villagers on steel and on wood both at 60 °C, and in water at 75 °C: the steel burns their feet in seconds, the wood doesn't, the water scalds at once.",
    },
    {
        id: "sand-hole", group: "World", name: "Sand by a hole", size: [160, 80], hour: 12, stopClock: true, holePull: 4, build: sandByHole,
        hint: "Single grains of sand on a floor beside a small black hole. Where the hole's pull along the floor beats a grain's friction it is drawn in; farther off it stays put.",
    },
    {
        id: "hole-space", group: "World", name: "Hole in space", size: [160, 160], hour: 12, stopClock: true, earthGravity: false, holePull: 4, build: holeInSpace,
        hint: "No Earth below, only a small black hole pulling. Sand on all four sides falls toward it and is swallowed. Press F for the pull.",
    },
    {
        id: "bar-hole", group: "World", name: "Bar and hole", size: [160, 160], hour: 12, stopClock: true, earthGravity: false, holePull: 4, build: barHole,
        hint: "No Earth below, and a steel bar falling onto a small black hole. The hole eats a slot through the bar's middle; cut in two, each half falls on as a body of its own.",
    },
];

// Rows cooled at a time to start a tall scene at its height's temperature.
const LAPSE_BAND = 8;

// Builds the scene into sim, which is its size (sceneSize): from its hour of
// the day, with the clock running or stopped there.
export function loadScene(sim, scene) {
    // Its scale first: a cleared world's air is the open air's at its height.
    sim.metresPerCell = scene.height ? scene.height / sim.height : DEFAULT_METRES_PER_CELL;
    sim.clear(false);
    sim.setClock(scene.hour / 24, scene.stopClock ? CLOCK.STOPPED : CLOCK.NORMAL);
    sim.setBlackHolePull(scene.holePull ?? DEFAULT_BLACK_HOLE_PULL);
    sim.setEarthGravity(scene.earthGravity ?? true);
    scene.build(new Builder(sim));
    // Soil under water starts soaked through, as it would long have been.
    sim.paint({
        ax: 0, ay: 0, bx: sim.width, by: sim.height, radius: 0, shape: SHAPE.RECT,
        angle: 0, mode: BRUSH.SOAK, elem: ID.AIR, replace: true, strength: 1,
    });
    // Everything starts as much colder than it was built as the open air is
    // at its height (outdoorT in common.wgsl).
    for (let y = 0; scene.height && y < sim.height; y += LAPSE_BAND) {
        const altitude = (sim.height - y - LAPSE_BAND / 2) * sim.metresPerCell;
        const cooler = LAPSE_RATE * altitude;
        if (cooler >= 0.5) {
            sim.paint({
                ax: 0, ay: y, bx: sim.width, by: Math.min(y + LAPSE_BAND, sim.height), radius: 0, shape: SHAPE.RECT,
                angle: 0, mode: BRUSH.COOL, elem: ID.AIR, replace: true, strength: cooler,
            });
        }
    }
    // As it would long have stood: the air its building pushed aside (a
    // room's walls painted into the open air) has gone, not into the room.
    sim.dropAirSource();
}

// The world a scene is built in: [width, height] in cells.
export function sceneSize(scene) {
    return scene.size;
}
