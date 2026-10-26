// What the hover readout says about a person (person.wgsl): their name, from
// their seed (hashed, as their looks are from it in figures.wgsl, so seeds
// close together don't share one), and what they are doing or died of.

import { PERSON_FLAG, PERSON_STATE } from "./sim.js";

const NAMES = [
    "Ada", "Ahmed", "Aiko", "Alba", "Amara", "Anders", "Ana", "Arjun", "Bea", "Bram", "Chen", "Clara", "Dario", "Dina",
    "Eli", "Ema", "Femi", "Freya", "Goran", "Hana", "Ines", "Ivo", "Jaro", "Juno", "Kai", "Kemal", "Lea", "Lin", "Luka",
    "Maja", "Mateo", "Mila", "Nadia", "Nika", "Noor", "Oli", "Omar", "Pia", "Quinn", "Rafa", "Rina", "Rosa", "Sami",
    "Sanna", "Sofia", "Tariq", "Tea", "Theo", "Uma", "Vera", "Vid", "Wen", "Xena", "Yara", "Yusuf", "Zala", "Zoran",
    "Iris", "Leon", "Mira", "Niko", "Petra", "Tomo", "Ula",
];

const DOING = {
    [PERSON_STATE.IDLE]: "standing about",
    [PERSON_STATE.WALK]: "out for a walk",
    [PERSON_STATE.FLEE]: "running for it",
    [PERSON_STATE.CHAT]: "chatting",
    [PERSON_STATE.SLEEP]: "asleep",
};

// By person.wgsl's DIED_*.
const DIED = [
    "dead", "died of the heat", "burnt", "drowned", "choked", "crushed", "fell to their death", "froze to death",
    "electrocuted", "blown up", "dissolved in acid",
];

export function nameOf(seed) {
    let h = Math.imul(seed ^ (seed >>> 16), 0x45d9f3b);
    h = Math.imul(h ^ (h >>> 16), 0x45d9f3b);
    return NAMES[((h ^ (h >>> 16)) >>> 0) % NAMES.length];
}

export function describePerson(p) {
    const name = nameOf(p.seed);
    if (p.state === PERSON_STATE.DEAD) {
        return `${name} · ${DIED[p.cause] ?? DIED[0]}`;
    }
    const tags = [DOING[p.state] ?? "", `health ${Math.round(p.health * 100)}%`];
    if (p.flags & PERSON_FLAG.BURNING) {
        tags.push("on fire");
    }
    if (p.flags & PERSON_FLAG.SWIM) {
        tags.push("swimming");
    }
    if (p.flags & PERSON_FLAG.UNDER || p.breath < 1) {
        tags.push(`breath ${Math.round(p.breath * 100)}%`);
    }
    return `${name} · ${tags.join(" · ")}`;
}
