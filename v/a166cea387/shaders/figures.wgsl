// Draws the people (person.wgsl) over the world, in the render pass after
// it: a quad each, its pixels a cell to the sprite's pixel, so they are as
// blocky as the world. Standing, a person is 5 x 10 cells; asleep or dead it
// lies along the ground; a bubble over its head says it is scared, asleep
// or talking, and a ghost rises from the dead. What is solid in front of it
// hides it (buried in sand, behind a wall it fell into); a liquid tints it.
// Its colours come from its seed.

// As render.wgsl's.
struct View {
    grid: vec2<f32>,
    origin: vec2<f32>,
    scale: f32,
    time: f32,
    mouse: vec2<f32>,
    lineStart: vec2<f32>,
    radius: f32,
    cursor: u32,
    mode: u32,
    electric: u32,
    day: f32,
    pad2: u32,
    canvas: vec2<f32>,
}

// VIEW_* come from js/shared.js.

@group(0) @binding(0) var<uniform> V: View;
@group(0) @binding(1) var<storage, read> elems: array<Elem>;
@group(0) @binding(2) var<storage, read> cells: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> people: array<Person, MAX_PEOPLE>;
// The light reaching each light cell of 2 x 2 cells (light.wgsl), for the
// Lit view.
@group(0) @binding(4) var lightTex: texture_2d<f32>;
@group(0) @binding(5) var lightSampler: sampler;

// Sprites: a row of cells a word, a hex digit a cell, the leftmost its back
// (it faces right). 0 nothing, 1 hair, 2 skin, 3 eye, 4 shirt, 5 trousers,
// 6 boots; the bubbles' 8 white, 9 red, 10 blue, 11 dark.
const UPPER: array<u32, 7> = array<u32, 7>(0x01110u, 0x11220u, 0x02230u, 0x00200u, 0x04440u, 0x44444u, 0x24442u);
const UPPER_ARMS_UP: array<u32, 7> = array<u32, 7>(0x01110u, 0x11220u, 0x22232u, 0x40204u, 0x44444u, 0x04440u, 0x04440u);
const LEGS_STAND: array<u32, 3> = array<u32, 3>(0x05550u, 0x05050u, 0x06060u);
const LEGS_STRIDE: array<u32, 3> = array<u32, 3>(0x05550u, 0x50005u, 0x60006u);
const LEGS_PASS: array<u32, 3> = array<u32, 3>(0x05550u, 0x00550u, 0x00660u);
const BUBBLE_ROWS: i32 = 8;
const BUBBLE_ALERT: array<u32, 8> = array<u32, 8>(0x08880u, 0x88988u, 0x88988u, 0x88988u, 0x88888u, 0x88988u, 0x08880u, 0x08000u);
const BUBBLE_SLEEP: array<u32, 8> = array<u32, 8>(0x08880u, 0x8AAA8u, 0x888A8u, 0x88A88u, 0x8A888u, 0x8AAA8u, 0x08880u, 0x08000u);
const BUBBLE_CHAT: array<u32, 8> = array<u32, 8>(0x00000u, 0x00000u, 0x08880u, 0x88888u, 0x8B8B8u, 0x88888u, 0x08880u, 0x08000u);
const LEGS_STILL: u32 = 0u;
const LEGS_APART: u32 = 1u;
const LEGS_TOGETHER: u32 = 2u;

const GHOST_RISE: f32 = 0.06;  // cells a step

struct Out {
    @builtin(position) pos: vec4<f32>,
    @location(0) g: vec2<f32>,
    @location(1) @interpolate(flat) who: u32,
}

// Where a person is drawn from: its middle column, and the row its feet
// stand on.
fn anchor(m: Person) -> vec2<i32> {
    return vec2<i32>(i32(floor(m.pos.x)), i32(round(m.pos.y)));
}

@vertex
fn vs(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> Out {
    var o: Out;
    o.who = i;
    let m = people[i];
    if (m.state == P_NONE) {
        o.pos = vec4<f32>(-2.0, -2.0, 0.0, 1.0);
        return o;
    }
    // Room for its bubble and its ghost over its head, and lying down.
    let corner = vec2<f32>(select(0.0, 1.0, v == 1u || v == 4u || v == 5u), select(0.0, 1.0, v == 2u || v == 3u || v == 5u));
    let at = vec2<f32>(anchor(m));
    let g = mix(at + vec2<f32>(-6.0, -f32(BODY_H) - 18.0), at + vec2<f32>(7.0, 1.0), corner);
    let px = V.origin + g * V.scale;
    o.pos = vec4<f32>(px.x / V.canvas.x * 2.0 - 1.0, 1.0 - px.y / V.canvas.y * 2.0, 0.0, 1.0);
    o.g = g;
    return o;
}

fn digit(row: u32, x: i32) -> u32 {
    return (row >> (4u * u32(4 - x))) & 0xfu;
}

// The sprite's cell (x, y), x from its back, y from the top of its head.
fn spriteAt(x: i32, y: i32, armsUp: bool, legs: u32) -> u32 {
    if (x < 0 || x > 4 || y < 0 || y >= BODY_H) {
        return 0u;
    }
    if (y < 7) {
        var upper = UPPER;
        var raised = UPPER_ARMS_UP;
        return digit(select(upper[y], raised[y], armsUp), x);
    }
    var still = LEGS_STAND;
    var apart = LEGS_STRIDE;
    var together = LEGS_PASS;
    let row = select(select(still[y - 7], together[y - 7], legs == LEGS_TOGETHER), apart[y - 7], legs == LEGS_APART);
    return digit(row, x);
}

fn bubbleAt(x: i32, y: i32, state: u32) -> u32 {
    if (x < 0 || x > 4 || y < 0 || y >= BUBBLE_ROWS) {
        return 0u;
    }
    var alert = BUBBLE_ALERT;
    var sleep = BUBBLE_SLEEP;
    var chat = BUBBLE_CHAT;
    let row = select(select(alert[y], sleep[y], state == P_SLEEP), chat[y], state == P_CHAT);
    return digit(row, x);
}

// Colour k of the person with this seed (hashed, so seeds close together
// look nothing alike).
fn colorOf(k: u32, seed: u32) -> vec3<f32> {
    let looks = pcg(seed);
    var skins = array<vec3<f32>, 6>(
        vec3<f32>(0.98, 0.84, 0.71), vec3<f32>(0.93, 0.75, 0.6), vec3<f32>(0.78, 0.57, 0.42),
        vec3<f32>(0.6, 0.42, 0.3), vec3<f32>(0.45, 0.3, 0.2), vec3<f32>(0.33, 0.22, 0.15),
    );
    var hairs = array<vec3<f32>, 6>(
        vec3<f32>(0.1, 0.08, 0.07), vec3<f32>(0.3, 0.18, 0.1), vec3<f32>(0.5, 0.32, 0.15),
        vec3<f32>(0.9, 0.78, 0.45), vec3<f32>(0.7, 0.3, 0.12), vec3<f32>(0.72, 0.72, 0.74),
    );
    var shirts = array<vec3<f32>, 10>(
        vec3<f32>(0.85, 0.2, 0.2), vec3<f32>(0.2, 0.45, 0.9), vec3<f32>(0.25, 0.7, 0.3), vec3<f32>(0.95, 0.8, 0.2),
        vec3<f32>(0.6, 0.3, 0.8), vec3<f32>(0.95, 0.5, 0.15), vec3<f32>(0.15, 0.7, 0.7), vec3<f32>(0.95, 0.5, 0.7),
        vec3<f32>(0.92, 0.92, 0.92), vec3<f32>(0.15, 0.2, 0.45),
    );
    var trousers = array<vec3<f32>, 5>(
        vec3<f32>(0.2, 0.3, 0.55), vec3<f32>(0.35, 0.25, 0.15), vec3<f32>(0.15, 0.15, 0.17),
        vec3<f32>(0.45, 0.45, 0.48), vec3<f32>(0.3, 0.38, 0.2),
    );
    switch k {
        case 1u: { return hairs[(looks >> 3u) % 6u]; }
        case 2u: { return skins[looks % 6u]; }
        case 3u: { return vec3<f32>(0.08, 0.06, 0.06); }
        case 4u: { return shirts[(looks >> 6u) % 10u]; }
        case 5u: { return trousers[(looks >> 10u) % 5u]; }
        case 6u: { return vec3<f32>(0.2, 0.13, 0.08); }
        case 8u: { return vec3<f32>(0.95, 0.95, 0.93); }
        case 9u: { return vec3<f32>(0.9, 0.12, 0.1); }
        case 10u: { return vec3<f32>(0.25, 0.4, 0.9); }
        default: { return vec3<f32>(0.2, 0.2, 0.22); }
    }
}

fn noise(p: vec2<i32>, t: u32) -> f32 {
    return f32(pcg((u32(p.x) * 73856093u) ^ (u32(p.y) * 19349663u) ^ pcg(t)) & 0xffffu) / 65535.0;
}

fn blocks(t: u32) -> bool {
    let k = elems[t].kind;
    return (k == KIND_SOLID || k == KIND_POWDER) && t != PLANT;
}

@fragment
fn fs(in: Out) -> @location(0) vec4<f32> {
    let m = people[in.who];
    let cell = vec2<i32>(floor(in.g));
    let at = anchor(m);
    let left = (m.flags & PF_LEFT) != 0u;
    let dead = m.state == P_DEAD;
    let burning = (m.flags & PF_BURNING) != 0u;
    let frame = u32(V.time * 12.0);

    // Its body: standing, or lying on its back along the ground with its
    // head the way it faces, its front up.
    var k = 0u;
    if (m.state == P_SLEEP || dead) {
        let lx = cell.x - (at.x - 5);
        let ly = cell.y - (at.y - 5);
        if (lx >= 0 && lx < 10 && ly >= 0 && ly < 5) {
            k = spriteAt(4 - ly, select(9 - lx, lx, left), false, LEGS_STILL);
        }
        if (k == 3u && m.state == P_SLEEP) {
            k = 2u;  // eyes shut
        }
    } else {
        let x = cell.x - (at.x - 2);
        let armsUp = m.state == P_FLEE || burning || (m.flags & PF_SWIM) != 0u;
        var legs = LEGS_STILL;
        if ((m.flags & PF_SWIM) != 0u) {
            legs = select(LEGS_APART, LEGS_TOGETHER, (frame / 3u) % 2u == 0u);
        } else if ((m.flags & PF_GROUND) == 0u) {
            legs = LEGS_APART;
        } else if (m.state == P_WALK || m.state == P_FLEE) {
            legs = select(LEGS_APART, LEGS_TOGETHER, u32(m.stride / 2.0) % 2u == 0u);
        }
        k = spriteAt(select(x, 4 - x, left), cell.y - (at.y - BODY_H), armsUp, legs);
    }

    var col = vec3<f32>(0.0);
    var alpha = 1.0;
    if (k != 0u) {
        // What is in front of it.
        if (cell.x >= 0 && cell.y >= 0 && cell.x < i32(V.grid.x) && cell.y < i32(V.grid.y)) {
            let c = unpackCell(cells[u32(cell.y) * u32(V.grid.x) + u32(cell.x)]);
            if (blocks(c.t)) {
                discard;
            }
            col = colorOf(k, m.seed);
            if (elems[c.t].kind == KIND_LIQUID) {
                col = mix(col, baseColor(c), 0.45);
            }
        } else {
            col = colorOf(k, m.seed);
        }
        if (dead) {
            col = mix(col, vec3<f32>(dot(col, vec3<f32>(0.3, 0.5, 0.2))), 0.6) * 0.8;
            alpha = clamp(m.timer / FADE_STEPS, 0.0, 1.0);
        }
        if (burning) {
            col = mix(col, mix(vec3<f32>(1.0, 0.35, 0.05), vec3<f32>(1.0, 0.85, 0.3), noise(cell, frame)), 0.55);
        }
        if ((m.flags & PF_HURT) != 0u && (frame / 2u) % 2u == 0u) {
            col = mix(col, vec3<f32>(0.9, 0.1, 0.1), 0.35);
        }
        if ((m.flags & PF_SHOCK) != 0u && frame % 2u == 0u) {
            col = vec3<f32>(0.85, 0.92, 1.0);
        }
    } else {
        // Flames over its head, on fire.
        let up = (at.y - BODY_H) - cell.y;
        if (burning && abs(cell.x - at.x) <= 1 && up >= 0 && up < 4 && noise(cell, frame) < 0.6 - 0.15 * f32(up)) {
            col = mix(vec3<f32>(1.0, 0.3, 0.05), vec3<f32>(1.0, 0.9, 0.35), noise(cell, frame + 7u));
        } else if (!dead && (m.state == P_FLEE || m.state == P_SLEEP
            || (m.state == P_CHAT && (u32(m.timer / 60.0) + in.who) % 2u == 0u))) {
            // Its bubble, over its head on the side it faces (the glyph
            // the right way round, its tail to the head).
            let bob = select(0, 1, m.state == P_SLEEP && fract(V.time * 0.7) < 0.5);
            let y = cell.y - (at.y - BODY_H - BUBBLE_ROWS - bob);
            var x = cell.x - select(at.x, at.x - 4, left);
            if (left && y == BUBBLE_ROWS - 1) {
                x = 4 - x;
            }
            let b = bubbleAt(x, y, m.state);
            if (b == 0u) {
                discard;
            }
            return vec4<f32>(colorOf(b, m.seed), 0.9);
        } else if (dead && m.since < GHOST_STEPS) {
            // Its ghost, rising, waving, fading.
            let rise = i32(4.0 + m.since * GHOST_RISE);
            let sway = i32(round(sin(m.since * 0.1)));
            let x = cell.x - (at.x - 2 + sway);
            if (spriteAt(select(x, 4 - x, left), cell.y - (at.y - BODY_H - rise), true, LEGS_TOGETHER) == 0u) {
                discard;
            }
            return vec4<f32>(0.85, 0.9, 1.0, 0.45 * (1.0 - m.since / GHOST_STEPS));
        } else {
            discard;
        }
    }

    // As the view shows the world.
    if (V.mode == VIEW_LIT) {
        let dims = vec2<f32>(textureDimensions(lightTex));
        let light = textureSampleLevel(lightTex, lightSampler, in.g / (2.0 * dims), 0.0).rgb;
        col = col * (1.0 - exp(-EXPOSURE * light) + LIT_FLOOR);
        if (burning) {
            col += vec3<f32>(0.6, 0.25, 0.05);
        }
    } else if (V.mode == VIEW_HEAT) {
        // Warm-blooded: about 37 °C, or burning hot.
        col = select(select(vec3<f32>(0.62, 0.12, 0.14), vec3<f32>(0.25, 0.25, 0.3), dead), vec3<f32>(1.0, 0.8, 0.2), burning);
    } else if (V.mode != 0u) {
        alpha *= 0.5;
    }
    return vec4<f32>(col, alpha);
}
