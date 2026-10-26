// People: little figures ten cells tall that live in the world (people.wgsl
// moves them, figures.wgsl draws them). They aren't cells: they stand on
// what is solid, wade and swim in liquids and walk through gas and foliage,
// and the world doesn't make room for them, but it can bury, burn, drown,
// choke, crush, shock and blow them up. Each has a slot in a buffer of
// MAX_PEOPLE.

const BODY_H: i32 = 10;         // cells tall
const BODY_HALF: i32 = 1;       // its body is the three columns either side of its middle one

// MAX_PEOPLE, what a person is doing (Person.state, P_*) and its flags
// (Person.flags, PF_*) come from js/shared.js.

// What it died of (Person.cause; names in js/people.js).
const DIED_HEAT: u32 = 1u;
const DIED_BURNT: u32 = 2u;
const DIED_DROWNED: u32 = 3u;
const DIED_CHOKED: u32 = 4u;
const DIED_CRUSHED: u32 = 5u;
const DIED_FALL: u32 = 6u;
const DIED_COLD: u32 = 7u;
const DIED_SHOCK: u32 = 8u;
const DIED_BLAST: u32 = 9u;
const DIED_ACID: u32 = 10u;

// Steps a body lies before it fades, and how long the fading takes.
const DEAD_STEPS: f32 = 900.0;
const FADE_STEPS: f32 = 120.0;
const GHOST_STEPS: f32 = 150.0; // its ghost rises from it for this long

// 80 bytes (PERSON_BYTES in sim.js).
struct Person {
    pos: vec2<f32>,     // its feet: x the middle of its middle column, y the bottom of its body (cells)
    vel: vec2<f32>,     // cells/step
    state: u32,
    flags: u32,
    timer: f32,         // steps left at what it is doing
    health: f32,        // 0..1
    breath: f32,        // 0..1
    seed: u32,          // its looks and its own randomness
    home: f32,          // the x it lives at
    stride: f32,        // its walk, in cells walked (for the legs)
    burn: f32,          // steps it has left on fire
    fallFrom: f32,      // y it last stood or swam at, for how far it has fallen
    since: f32,         // steps since it died; before that, until it will chat again
    cause: u32,         // what it died of
    // Its body's heat (REWRITE_PLAN.md 10-2; people.wgsl): its skin's
    // temperature and its core's (°C), and the burn its skin has taken
    // (Henriques' damage integral, 1 a blistering burn).
    skin: f32,
    core: f32,
    dose: f32,
    pad: f32,
}
