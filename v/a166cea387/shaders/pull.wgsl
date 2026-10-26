// The pull, down and height where a pass is (REWRITE_PLAN.md 4d), all read
// from the gravity pass's field (gravity.wgsl) per light cell of
// GRAVITY_BLOCK × GRAVITY_BLOCK cells: the masses' pull (xy, cells/step²),
// their potential (z, cells²/step²), and down (w): the neighbour step the
// total pull, Earth's and theirs, is nearest (stepIndex), kept while it
// leans a few degrees past its sector (gravity.wgsl's downStep), with
// DOWN_WEIGHTLESS (common.wgsl) added where it is too weak to settle
// anything (WEIGHTLESS). One down for every pass. A module including this
// binds `gravity: texture_2d<f32>` and declares `P`.

// The g heights are measured by, m/s² (altitudeAt).
const EARTH_G: f32 = 9.81;

fn gravityTexel(c: vec2<i32>) -> vec4<f32> {
    let size = vec2<i32>(textureDimensions(gravity));
    return textureLoad(gravity, clamp(c, vec2<i32>(0), size - 1), 0);
}

// The masses' field (pull and potential) at pos (cells), bilinear between
// the light cells' middles (rgba32float isn't filterable without a
// feature, so by hand), so what moves feels it change smoothly.
fn fieldAt(pos: vec2<f32>) -> vec3<f32> {
    let q = pos / f32(GRAVITY_BLOCK) - 0.5;
    let i = vec2<i32>(floor(q));
    let f = q - floor(q);
    let top = mix(gravityTexel(i).xyz, gravityTexel(i + vec2<i32>(1, 0)).xyz, f.x);
    let bottom = mix(gravityTexel(i + vec2<i32>(0, 1)).xyz, gravityTexel(i + vec2<i32>(1, 1)).xyz, f.x);
    return mix(top, bottom, f.y);
}

// The total pull at pos (cells), Earth's and the masses', cells/step².
fn pullAt(pos: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(0.0, P.earthG) + fieldAt(pos).xy;
}

// Cell p's light cell's down word.
fn downWordAt(p: vec2<i32>) -> u32 {
    return u32(gravityTexel(p / i32(GRAVITY_BLOCK)).w);
}

// Down at cell p, a neighbour step.
fn downAt(p: vec2<i32>) -> vec2<i32> {
    return stepAt(downWordAt(p) & 7u);
}

// Whether the pull at cell p is too weak to settle anything (WEIGHTLESS).
fn weightlessAt(p: vec2<i32>) -> bool {
    return (downWordAt(p) & DOWN_WEIGHTLESS) != 0u;
}

// The height at pos (cells), metres, as the pull has it: its potential,
// Earth's and the masses', over Earth's g (geopotential height; Wallace &
// Hobbs, Atmospheric Science, 2006, ch. 1). Up from the world's bottom row
// with Earth's pull on, and lower near a mass by its well; with Earth's off,
// from nothing far from any mass, down into a black hole's. The open air's
// pressure and temperature go by it, and so does air rising or sinking.
fn altitudeAt(pos: vec2<f32>) -> f32 {
    let phi = P.earthG * (f32(P.size.y) - pos.y) + fieldAt(pos).z;
    let v = P.dx / P.dt;
    return phi * v * v / EARTH_G;
}
