// The cell's record (REWRITE_PLAN.md 5d-2; audit/10 §5.2): what a cell's
// aux and life (a byte each, CELL_FIELDS in js/shared.js) and its shade's
// top bits hold, by the class of its element, declared once here. Each
// field holds one meaning for an element, or one for each state it can be
// in (a plant's life is its sugar, or, burning, what of it has burnt),
// which this module checks as it loads. From it come the shaders'
// accessors (recordPrelude: soluteOf(c) and withSolute(c, v) for a Cell,
// wordSolute(x) for its packed word, and a class test for each role, such
// as holdsSolute(t)); convert(c, t), what a cell keeps when it turns into
// another element; and the host's cellRecord(cell), the same read for the
// hover and the harness.

import { ELEMENTS, FLAG, KIND } from "./elements.js";
import { CELL_FIELDS, CELL_FLAGS } from "./shared.js";

// The classes. `of` picks the elements that are in it from the table;
// `wgsl` makes the same test in WGSL on an element id.
const growing = (e) => e.id === "PLANT" || e.id === "SEED";
const CLASSES = {
    water: { of: (e) => e.id === "WATER", wgsl: (t) => `${t} == WATER` },
    soil: { of: (e) => e.kind === KIND.POWDER && e.cohesion > 0, wgsl: (t) => `porous(elems[${t}])` },
    soluble: {
        of: (e) => e.kind === KIND.POWDER && e.solubility > 0,
        wgsl: (t) => `elems[${t}].kind == KIND_POWDER && elems[${t}].solubility > 0.0`,
    },
    // What burns as a solid or liquid (a gas burns what it holds of the gas
    // plane: pairs.wgsl's burnGas), but what grows.
    fuel: {
        of: (e) => e.burnTime > 0 && e.kind !== KIND.GAS && !growing(e),
        wgsl: (t) => `elems[${t}].burnTime > 0.0 && elems[${t}].kind != KIND_GAS && ${t} != PLANT && ${t} != SEED`,
    },
    growing: { of: growing, wgsl: (t) => `${t} == PLANT || ${t} == SEED` },
    plant: { of: (e) => e.id === "PLANT", wgsl: (t) => `${t} == PLANT` },
    seed: { of: (e) => e.id === "SEED", wgsl: (t) => `${t} == SEED` },
    frozen: { of: (e) => e.id === "ICE" || e.id === "SNOW", wgsl: (t) => `${t} == ICE || ${t} == SNOW` },
    owing: { of: (e) => (e.flags & FLAG.OWES) !== 0, wgsl: (t) => `(elems[${t}].flags & F_OWES) != 0u` },
    conductor: { of: (e) => (e.flags & FLAG.CONDUCT) !== 0, wgsl: (t) => `(elems[${t}].flags & F_CONDUCT) != 0u` },
    heading: { of: (e) => e.id === "FAN" || e.emf !== 0, wgsl: (t) => `${t} == FAN || elems[${t}].emf != 0.0` },
    clone: { of: (e) => e.id === "CLONE", wgsl: (t) => `${t} == CLONE` },
    marker: { of: (e) => ["FIRE", "SMOKE", "BLAST"].includes(e.id), wgsl: (t) => `${t} == FIRE || ${t} == SMOKE || ${t} == BLAST` },
};

// The roles: the field each is in (mask: the bits of it, else all), the
// class that holds it, and what it means; `burning` (true or false), only
// while the cell is alight or not. A role in more than one class (owed) is
// read from whichever the cell is in. `carry`: kept when a cell turns into
// another element of the same class (convert).
export const ROLES = [
    { name: "solute", field: "aux", class: "water", carry: true, note: "the element it carries dissolved (SALT, ACID), 0 none" },
    { name: "strength", field: "life", class: "water", carry: true, note: "how much of its solute, 255 saturated" },
    { name: "soilWater", field: "life", class: "soil", note: "the water it holds, SOIL_FULL soaked (stage 5f)" },
    { name: "dissolved", field: "life", class: "soluble", note: "what of the grain has dissolved, 255 all of it" },
    { name: "burnt", field: "life", class: "fuel", note: "what of it has burnt, 255 all (a char from charAt)" },
    { name: "burnt", field: "life", class: "growing", burning: true, note: "what of it has burnt, alight, its growth gone up in smoke" },
    { name: "sugar", field: "life", class: "plant", burning: false, note: "its sugar, GROW_SUGAR a new cell (5d-1)" },
    { name: "plantWater", field: "aux", class: "plant", note: "its water, PLANT_WATER turgid, wilting under PLANT_WILT" },
    { name: "lean", field: "shade", mask: 0xc, class: "plant", note: "the side it leans to, PLANT_LEFT or PLANT_RIGHT (its shade the low bits)" },
    { name: "store", field: "life", class: "seed", burning: false, note: "the growth it has stored, in cells (initLife)" },
    { name: "owed", field: "aux", class: "frozen", note: "the cells it owes, freezing having made more (formed)" },
    { name: "owed", field: "life", class: "owing", note: "the cells it owes, being a lighter form than it was (formed)" },
    { name: "frozenExtra", field: "life", class: "frozen", note: "the water it holds past its share, 1 / FROZEN_EXTRA of a vapour unit each" },
    { name: "charge", field: "aux", class: "conductor", note: "a spark's static charge: CHARGE_ACTIVE and up spreading, then cooling down" },
    { name: "heading", field: "aux", class: "heading", note: "the way it pushes, in 1/256 turns" },
    { name: "copies", field: "aux", class: "clone", note: "the element it copies, 0 none yet" },
    { name: "copiesSolute", field: "life", class: "clone", note: "the solute of the water it copies" },
    { name: "lifetime", field: "life", class: "marker", note: "the steps it has left (initLife)" },
];

// What carries over of the flags on a type change: burning, into what
// burns (wood alight crumbling into sawdust alight, a fuel charring on
// into its char).
const CARRIED_FLAGS = [{ flag: "BURN", wgsl: (t) => `flammable(elems[${t}])` }];

const fieldMask = (role) => role.mask ?? (1 << CELL_FIELDS[role.field][1]) - 1;
const capital = (s) => s[0].toUpperCase() + s.slice(1);

function holds(role, e, burning) {
    return CLASSES[role.class].of(e) && (role.burning === undefined || role.burning === burning);
}

// Each element's roles, alight and not: a field's bits held by two roles at
// once is a mistake in the table above.
for (const e of ELEMENTS) {
    for (const burning of [false, true]) {
        const held = {};
        for (const role of ROLES.filter((r) => holds(r, e, burning))) {
            const bits = held[role.field] ?? 0;
            if (bits & fieldMask(role)) {
                throw new Error(`cellrecord.js: ${e.id}'s ${role.field} holds ${role.name} and another role at once`);
            }
            held[role.field] = bits | fieldMask(role);
        }
    }
}
for (const role of ROLES) {
    if (!(role.class in CLASSES) || !(role.field in CELL_FIELDS) || role.field === "type" || role.field === "flags") {
        throw new Error(`cellrecord.js: role ${role.name} names an unknown class or field`);
    }
}

// The roles a cell of element e holds, alight (burning) or not.
export function rolesOf(e, burning) {
    return ROLES.filter((role) => holds(role, e, burning));
}

// The roles that apply to a cell ({ type, aux, life, shade, flags }, as
// js/shared.js's cellFields reads it), by name: { solute: 31, strength: 36 }.
export function cellRecord(cell) {
    const e = ELEMENTS[cell.type];
    const record = {};
    for (const role of e ? rolesOf(e, (cell.flags & CELL_FLAGS.BURN) !== 0) : []) {
        record[role.name] = cell[role.field] & fieldMask(role);
    }
    return record;
}

// The shaders' accessors and convert (REWRITE_PLAN.md 5d-2). A role's
// class test, holds<Role>(t), takes no state: a plant's life is its sugar
// or, alight, what of it has burnt, and what reads it knows which.
export function recordPrelude() {
    const lines = ["// Generated from js/cellrecord.js: the cell's record"];
    const names = [...new Set(ROLES.map((r) => r.name))];
    for (const name of names) {
        const roles = ROLES.filter((r) => r.name === name);
        const Name = capital(name);
        const test = (r, t = "t") => CLASSES[r.class].wgsl(t);
        for (const r of roles) {
            lines.push(`// ${name}: ${r.note} (${r.field}, ${r.class})`);
        }
        const getter = (r, v) => (r.mask ? `(${v}.${r.field} & ${r.mask}u)` : `${v}.${r.field}`);
        const word = (r) => (r.mask ? `(word${capital(r.field)}(x) & ${r.mask}u)` : `word${capital(r.field)}(x)`);
        const setter = (r) => (r.mask
            ? `n.${r.field} = (n.${r.field} & ~${r.mask}u) | (v & ${r.mask}u);`
            : `n.${r.field} = min(v, ${fieldMask(r)}u);`);
        lines.push(`fn holds${Name}(t: u32) -> bool { return ${roles.map((r) => `(${test(r)})`).join(" || ")}; }`);
        // One field for every class that holds it (all the same bits): read
        // and written as it is.
        if (roles.every((r) => r.field === roles[0].field && r.mask === roles[0].mask)) {
            const [r] = roles;
            lines.push(`fn ${name}Of(c: Cell) -> u32 { return ${getter(r, "c")}; }`);
            lines.push(`fn word${Name}(x: u32) -> u32 { return ${word(r)}; }`);
            lines.push(`fn with${Name}(c: Cell, v: u32) -> Cell { var n = c; ${setter(r)} return n; }`);
        } else {
            // In whichever class the cell is in, the first that holds it; 0
            // (and nothing written) in none.
            const branches = (make) => roles.map((r, k) => `${k ? " else " : ""}if (${test(r, "c.t")}) { ${make(r)} }`).join("");
            lines.push(`fn ${name}Of(c: Cell) -> u32 { ${branches((r) => `return ${getter(r, "c")};`)} return 0u; }`);
            lines.push(`fn with${Name}(c: Cell, v: u32) -> Cell { var n = c; ${branches(setter)} return n; }`);
        }
    }
    // convert: what a cell c keeps turning into t.
    const carried = ROLES.filter((r) => r.carry);
    const classes = [...new Set(carried.map((r) => r.class))];
    lines.push(
        "// What a cell c keeps when it turns into t: its temperature (the caller",
        "// works out its heat), its velocity, what of its record carries into t's",
        "// class (the solute water carries), and burning into what burns. The rest",
        "// is a new t's (makeCell).",
        "fn convert(c: Cell, t: u32) -> Cell {",
        "    var n = makeCell(t, c.temp);",
        "    n.vel = c.vel;",
    );
    for (const cls of classes) {
        const of = CLASSES[cls].wgsl;
        const fields = carried.filter((r) => r.class === cls).map((r) => `n.${r.field} = c.${r.field};`);
        lines.push(`    if ((${of("c.t")}) && (${of("t")})) { ${fields.join(" ")} }`);
    }
    for (const f of CARRIED_FLAGS) {
        lines.push(`    if ((c.flags & FL_${f.flag}) != 0u && ${f.wgsl("t")}) { n.flags |= FL_${f.flag}; }`);
    }
    lines.push("    return n;", "}");
    return lines.join("\n") + "\n";
}
