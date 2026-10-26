// Radiant heat (REWRITE_PLAN.md stage 6): the thermal infrared everything
// gives off at its temperature, carried cell to cell by finite-volume
// discrete ordinates (Chui & Raithby, Numer. Heat Transfer B 23:269, 1993;
// Modest, Radiative Heat Transfer, 3rd ed., 2013), conservative by
// construction: what every cell takes in less what it gives off, and what
// crosses the world's edges, come to none. (Visible light, and the sun's
// heat, are light.wgsl's.)
//
// RAD_DIRS directions in the plane, offset from the axes, each weighted so
// that radiance the same every way puts that much through a face (the
// thermal radiance units of common.wgsl's thermalRadiance, σT⁴ over σ
// (1000 K)⁴). Along a direction d = (μ, η) a cell takes what comes in
// through its two upwind faces and sends its radiance I out through its two
// downwind faces, the step scheme:
//   I = (μ I_x + η I_y + κ B) / (μ + η + κ),
// κ the optical depth across it (its absorptance, clarity and emissivity,
// cellClarity's: Kirchhoff, it gives off as it takes in) and B its own
// thermal radiance. An opaque cell is a boundary: it takes in ε of what
// reaches its faces from cells that aren't opaque, sends ε B out of them,
// and what each face reflected the last time (1 - ε) out of that face
// again, evenly over the directions leaving it; a face between two opaque
// cells carries nothing. So touching solids trade heat only by conduction,
// and nothing gets through a wall a cell thick, a mirror or not, or two
// walls meeting at a corner. (Reflected out of every face, what reached
// one side of a wall a cell thick went out of the other: a perfect mirror
// passed half of a hot room's radiation to the cold room beside it.) (test/radproto.mjs: an isothermal box takes in
// none, the budget closes to rounding, a lone block loses ε σ A (T⁴ - T_e⁴)
// to a part in a thousand. Light.wgsl's cascades, which this takes over the
// heat from, gave 0.17 of it: test/radiantarea.mjs.)
//
// A direction's cells depend only on the cells before them on both axes, so
// it is swept by its anti-diagonals, the cells of one all at once, one
// workgroup a direction, all RAD_DIRS at the same time. What comes in at the
// world's top is the sky's (skyRadiance, Swinbank's), at its bottom the
// ground's, and at its sides what went out of that side the last time in
// the mirrored direction: the world goes on as its mirror image.
//
// Passes, when the light is worked out (sim.js):
//   sweep    per direction its anti-diagonals: per cell what it takes in
//            (and an opaque one reflects), added up over the directions in
//            fixed point (taken), and what leaves the world's edges
//   resolve  per cell what it took in and how well it gives off (its
//            emitting area: what it gives off is that times its thermal
//            radiance), for update.wgsl, which pays out the difference at
//            its own temperature every step until the next time

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> elems: array<Elem>;
@group(0) @binding(2) var<storage, read> cells: array<vec4<u32>>;
// This time's: per cell what it takes in, then what reaches it through its
// faces (for the mean radiance about it: what a person standing there
// feels), then what each of its faces reflects (opaque; faceSlot), in fixed
// point (RAD_FIXED), then how real-time the hottest thing reaching it any
// one way is (peakSlot). Cleared before the sweep.
@group(0) @binding(3) var<storage, read_write> taken: array<atomic<u32>>;
// The last time's, for what opaque cells reflect out again.
@group(0) @binding(4) var<storage, read> takenBefore: array<u32>;
// What leaves the world's sides, per direction and row (radiance), then its
// top and bottom, per direction and column, then per direction what came
// in and what went out across all its edges (through a face, summed); and
// the last time's, for what comes in at the sides.
@group(0) @binding(5) var<storage, read_write> edges: array<f32>;
@group(0) @binding(6) var<storage, read> edgesBefore: array<f32>;
// Per cell (resolve): what it took in, its emitting area, its thermal
// radiance as the sweep took it, and the mean radiance reaching it, in its
// 8 lowest mantissa bits how real-time the hottest thing reaching it any
// one way is (peakSlot, 0 to 255: update.wgsl's radiantHeat).
@group(0) @binding(7) var radiantOut: texture_storage_2d<rgba32float, write>;

const RAD_DIRS: u32 = 16u;
// 2^18 a unit of radiance through a face, to 16384 of them: a cell's
// takings and the radiance reaching it, summed over its faces and ways, at
// most some 10000 °C all round (a blast's 7000-odd, the heat tool's 5000).
// (At 2^22, to 1024, they wrapped round past some 3700 °C all round, and a
// grain at the heart of a fireball, or in a cavity the heat tool had taken
// to 5000 °C, read a fraction of what reached it.) A unit is 57 kW/m²
// (σ (1000 K)⁴), so its resolution is a fifth of a W/m².
const RAD_FIXED: f32 = 262144.0;
// A cell letting through less than this of the infrared crossing it is
// opaque: a surface, not a medium.
const OPAQUE_CLEAR: f32 = 1e-3;
// What the workgroup holds of a wavefront: the cells along the world's
// shorter side (sim.js keeps it within this).
const SPAN: u32 = 2048u;
const PI: f32 = 3.1415927;
// A clear sky's long-wave radiation from the air temperature T (K) near the
// ground, 5.31e-13 T⁶ W/m² (Swinbank, "Long-wave radiation from clear
// skies", Q. J. Royal Meteorological Society 89:339, 1963), in thermal
// radiance units (σ (1000 K)⁴ = 56700 W/m²): some 15 K colder than the air
// at 20 °C. (Light.wgsl took the sky 30 K colder, for its cascades, which
// coupled a fifth as strongly: under the real coupling that took a glass
// roof in the sun at noon to 13 °C.)
const RAD_SWINBANK: f32 = 9.365e-18;

fn radDir(m: u32) -> vec2<f32> {
    let a = (f32(m) + 0.5) * 2.0 * PI / f32(RAD_DIRS);
    return vec2<f32>(cos(a), sin(a));
}

// The weight that makes radiance I the same every way put I through a
// face: 1 over the sum, over the directions toward it, of |d . n|.
fn radWeight() -> f32 {
    var half = 0.0;
    for (var m = 0u; m < RAD_DIRS; m++) {
        half += max(radDir(m).x, 0.0);
    }
    return 1.0 / half;
}

// A cell's infrared optics: opaque or not, its optical depth across (not
// opaque), its emissivity (opaque), and its thermal radiance.
struct Optics {
    opaque: bool,
    kappa: f32,
    eps: f32,
    radiance: f32,
    hot: u32,      // how real-time its temperature is (hotOf)
}

fn opticsAt(p: vec2<u32>) -> Optics {
    let i = p.y * P.size.x + p.x;
    let c = unpackCell(cells[i]);
    let e = elems[c.t];
    var g = noGas();
    if (holdsVapour(c.t)) {
        g = gasFrom(cells[gasIndex(P.size.x * P.size.y, i)], i);
    }
    let clear = clarityOf(c, g, P.dx).y;
    let temp = effTempIn(c.t, c.temp, phaseOf(c.t, soluteOf(c), strengthOf(c), 0.0));
    var o: Optics;
    o.opaque = clear < OPAQUE_CLEAR;
    o.eps = e.emissivity;
    // What it doesn't let through it takes in as far as it is a good
    // emitter (the rest passes as if it had let it through).
    o.kappa = -log(max(1.0 - (1.0 - clear) * e.emissivity, 1e-6));
    o.radiance = thermalRadiance(temp);
    o.hot = hotOf(temp);
    return o;
}

// The sweep's wavefronts carry in each value's 8 lowest mantissa bits (its
// precision 16 bits still) how real-time the hottest thing that gave off,
// or passed or sent on, light along the ray is: its realness (common.wgsl)
// as 0 to 255. A cell takes in radiant heat on the clock of the hottest
// thing it sees (peakSlot; update.wgsl's radiantHeat), the clock that gives
// it off, so what something hot gives off in real time is taken in in real
// time however far it shines (REWRITE_PLAN.md 6-5). (Its brightness
// wouldn't do: a 500 °C steel plate, ε 0.2, shines as dimly as a 380 °C
// black one.)
fn hotOf(temp: f32) -> u32 {
    return u32(round(realness(temp) * 255.0));
}

fn waveValue(v: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(v) & ~0xffu);
}

fn waveHot(v: f32) -> u32 {
    return bitcast<u32>(v) & 0xffu;
}

// (Rounded to the nearest at its 16 bits: cut off, every value lost a
// little, all the same way, and the books came out 0.07 % short.)
fn wavePack(v: f32, hot: u32) -> f32 {
    return bitcast<f32>(((bitcast<u32>(v) + 0x80u) & ~0xffu) | hot);
}

fn opaqueAt(q: vec2<i32>) -> bool {
    if (any(q < vec2<i32>(0)) || any(q >= vec2<i32>(P.size))) {
        return false;
    }
    return opticsAt(vec2<u32>(q)).opaque;
}

// An opaque cell's faces open to what isn't opaque (or the world's edge).
fn openFaces(p: vec2<u32>) -> f32 {
    let q = vec2<i32>(p);
    var n = 0.0;
    for (var k = 0; k < 4; k++) {
        let d = select(vec2<i32>(0, 1 - 2 * (k & 1)), vec2<i32>(1 - 2 * (k & 1), 0), k < 2);
        n += select(1.0, 0.0, opaqueAt(q + d));
    }
    return n;
}

fn skyRadiance() -> f32 {
    let near = max(outdoorT(f32(P.size.y) * P.dx) + dayWarmth(P.day) + KELVIN, 0.0);
    let n3 = near * near * near;
    return RAD_SWINBANK * n3 * n3;
}

fn groundRadiance() -> f32 {
    return thermalRadiance(outdoorT(0.0) + dayWarmth(P.day));
}

fn sideSlot(m: u32, y: u32) -> u32 {
    return m * P.size.y + y;
}

fn endSlot(m: u32, x: u32) -> u32 {
    return RAD_DIRS * P.size.y + m * P.size.x + x;
}

fn tallySlot(m: u32) -> u32 {
    return RAD_DIRS * (P.size.x + P.size.y) + 2u * m;
}

// In the takings: what reached cell i, and what its face f reflects (0
// right, 1 left, 2 down, 3 up).
fn reachSlot(i: u32) -> u32 {
    return P.size.x * P.size.y + i;
}

fn faceSlot(i: u32, f: u32) -> u32 {
    return 2u * P.size.x * P.size.y + 4u * i + f;
}

// How real-time the hottest thing reaching cell i any one way is (hotOf;
// REWRITE_PLAN.md 6-5): the clock it takes radiant heat in on is that
// thing's (update.wgsl's radiantHeat).
fn peakSlot(i: u32) -> u32 {
    return 6u * P.size.x * P.size.y + i;
}

// What face f of cell i reflected the last time, going out of it now.
fn reflectedBefore(i: u32, f: u32) -> f32 {
    return f32(takenBefore[faceSlot(i, f)]) / RAD_FIXED;
}

// The four of them together.
fn allReflectedBefore(i: u32) -> f32 {
    var sum = 0.0;
    for (var f = 0u; f < 4u; f++) {
        sum += reflectedBefore(i, f);
    }
    return sum;
}

// The direction mirrored across the world's sides (x the other way).
fn mirrored(m: u32) -> u32 {
    return (RAD_DIRS / 2u + RAD_DIRS - 1u - m) % RAD_DIRS;
}

// A quantity through a face in fixed point, rounded.
fn fixed(v: f32) -> u32 {
    return u32(v * RAD_FIXED + 0.5);
}

// Radiance the same every way out of a cell puts this much through its
// faces: the sum over the directions of w (μ + η).
fn everyWay(wt: f32) -> f32 {
    var sum = 0.0;
    for (var m = 0u; m < RAD_DIRS; m++) {
        let d = radDir(m);
        sum += wt * (abs(d.x) + abs(d.y));
    }
    return sum;
}

// Two wavefronts' radiance by place along the shorter side; an opaque
// cell's as -1 - what its faces send out.
// (Then what came in and went out across the world's edges, per thread.)
var<workgroup> wave: array<f32, 2u * SPAN>;

@compute @workgroup_size(256)
fn sweep(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let m = wid.x;
    let d = radDir(m);
    let mu = abs(d.x);
    let eta = abs(d.y);
    let wt = radWeight();
    let n = P.size;
    // The sweep's own frame: x' and y' from the upwind corner; its
    // wavefronts indexed along the shorter side (by y' if it is the y side).
    let byY = n.y < n.x;
    // The faces a cell sends out of along d (x, then y); it takes in
    // through the other two.
    let fx = select(1u, 0u, d.x > 0.0);
    let fy = select(3u, 2u, d.y > 0.0);
    let sky = skyRadiance();
    let ground = groundRadiance();
    let fronts = n.x + n.y - 1u;
    var crossed = vec2<f32>(0.0);
    for (var k = 0u; k < fronts; k++) {
        let cur = (k & 1u) * SPAN;
        let prev = SPAN - cur;
        // the wavefront's cells: x' + y' = k
        let lo = select(0u, k - (n.y - 1u), k >= n.y);
        let hi = min(k, n.x - 1u);
        let count = hi - lo + 1u;
        for (var t = li; t < count; t += 256u) {
            let xs = lo + t;
            let ys = k - xs;
            let x = select(n.x - 1u - xs, xs, d.x > 0.0);
            let y = select(n.y - 1u - ys, ys, d.y > 0.0);
            let p = vec2<u32>(x, y);
            let here = select(xs, ys, byY);
            let o = opticsAt(p);
            // In through the upwind x face: the world's side, or the cell
            // before along x (on the wavefront before, one back along it if
            // indexed by x', the same place if by y').
            var ix = 0.0;
            var openX = true;
            var hotX = 0u;
            if (xs == 0u) {
                ix = edgesBefore[sideSlot(mirrored(m), y)];
                crossed.x += wt * mu * ix;
            } else {
                let packed = wave[prev + select(here - 1u, here, byY)];
                let v = waveValue(packed);
                openX = !(v < 0.0 && o.opaque);
                hotX = select(0u, waveHot(packed), openX);
                // (an opaque cell's own, and what its face toward this one
                // reflected)
                ix = select(v, -1.0 - v + reflectedBefore(y * n.x + select(x + 1u, x - 1u, d.x > 0.0), fx), v < 0.0);
            }
            var iy = 0.0;
            var openY = true;
            var hotY = 0u;
            if (ys == 0u) {
                iy = select(ground, sky, d.y > 0.0);
                crossed.x += wt * eta * iy;
            } else {
                let packed = wave[prev + select(here, here - 1u, byY)];
                let v = waveValue(packed);
                openY = !(v < 0.0 && o.opaque);
                hotY = select(0u, waveHot(packed), openY);
                iy = select(v, -1.0 - v + reflectedBefore(select(y + 1u, y - 1u, d.y > 0.0) * n.x + x, fy), v < 0.0);
            }
            let i = y * n.x + x;
            let hotIn = max(hotX, hotY);
            atomicMax(&taken[peakSlot(i)], hotIn);
            var outX = 0.0;
            var outY = 0.0;
            if (o.opaque) {
                let reachX = select(0.0, mu * ix, openX);
                let reachY = select(0.0, eta * iy, openY);
                atomicAdd(&taken[reachSlot(i)], fixed(wt * (reachX + reachY)));
                atomicAdd(&taken[i], fixed(wt * o.eps * (reachX + reachY)));
                // each face reflects what came in through it (the upwind ones)
                atomicAdd(&taken[faceSlot(i, 1u - fx)], fixed(wt * (1.0 - o.eps) * reachX));
                atomicAdd(&taken[faceSlot(i, 5u - fy)], fixed(wt * (1.0 - o.eps) * reachY));
                // what its downwind faces send out: its own (the cells after
                // it add what each face reflected the last time)
                let own = o.eps * o.radiance;
                // (What it sends on: its own, and, shiny, what it reflects.)
                let hotOut = max(select(0u, o.hot, o.eps > 0.0), select(0u, hotIn, o.eps < 0.5));
                wave[cur + here] = wavePack(-1.0 - own, hotOut);
                outX = own + reflectedBefore(i, fx);
                outY = own + reflectedBefore(i, fy);
            } else {
                // (what it reflected when it was opaque, since gone, as a
                // medium gives off, evenly every way)
                let out = (mu * ix + eta * iy + o.kappa * o.radiance) / (mu + eta + o.kappa) + allReflectedBefore(i) / everyWay(wt);
                let absorbed = wt * (mu * ix + eta * iy) * o.kappa / (mu + eta + o.kappa);
                atomicAdd(&taken[i], fixed(max(absorbed, 0.0)));
                atomicAdd(&taken[reachSlot(i)], fixed(wt * (mu * ix + eta * iy)));
                wave[cur + here] = wavePack(out, max(hotIn, select(0u, o.hot, o.kappa > 0.0)));
                outX = out;
                outY = out;
            }
            // What leaves the world's far side and its top or bottom.
            if (xs == n.x - 1u) {
                edges[sideSlot(m, y)] = outX;
                crossed.y += wt * mu * outX;
            }
            if (ys == n.y - 1u) {
                edges[endSlot(m, x)] = outY;
                crossed.y += wt * eta * outY;
            }
        }
        workgroupBarrier();
    }
    wave[2u * li] = crossed.x;
    wave[2u * li + 1u] = crossed.y;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (li < s) {
            wave[2u * li] += wave[2u * (li + s)];
            wave[2u * li + 1u] += wave[2u * (li + s) + 1u];
        }
        workgroupBarrier();
    }
    if (li == 0u) {
        edges[tallySlot(m)] = wave[0];
        edges[tallySlot(m) + 1u] = wave[1];
    }
}

// What a cell gives off per unit of its thermal radiance (its emitting
// area, in faces): an opaque one ε through each open face; a medium, along
// each direction, (μ + η) κ / (μ + η + κ) of its radiance, weighted.
fn emittingArea(p: vec2<u32>, o: Optics) -> f32 {
    if (o.opaque) {
        return o.eps * openFaces(p);
    }
    if (o.kappa <= 0.0) {
        return 0.0;
    }
    let wt = radWeight();
    var area = 0.0;
    for (var m = 0u; m < RAD_DIRS; m++) {
        let d = radDir(m);
        let s = abs(d.x) + abs(d.y);
        area += wt * s * o.kappa / (s + o.kappa);
    }
    return area;
}

@compute @workgroup_size(8, 8)
fn resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid.xy >= P.size)) {
        return;
    }
    let p = gid.xy;
    let i = p.y * P.size.x + p.x;
    let o = opticsAt(p);
    var took = f32(atomicLoad(&taken[i])) / RAD_FIXED;
    let reaching = f32(atomicLoad(&taken[reachSlot(i)])) / RAD_FIXED / everyWay(radWeight());
    // What a face of an opaque cell reflected the last time that can't go
    // out of it now (what it faces has come to be opaque) it takes in.
    if (o.opaque) {
        let q = vec2<i32>(p);
        for (var f = 0u; f < 4u; f++) {
            let d = select(vec2<i32>(0, 5 - 2 * i32(f)), vec2<i32>(1 - 2 * i32(f), 0), f < 2u);
            if (opaqueAt(q + d)) {
                took += reflectedBefore(i, f);
            }
        }
    }
    textureStore(radiantOut, p, vec4<f32>(took, emittingArea(p, o), o.radiance, wavePack(reaching, atomicLoad(&taken[peakSlot(i)]))));
}
