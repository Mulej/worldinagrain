// Conduction across stiff faces, in stages (REWRITE_PLAN.md stage 2c).
//
// update.wgsl passes heat explicitly (forward Euler), a step at a time,
// across every face mild enough for it: one that passes no more than
// HEAT_SHARE_MAX of the smaller heat capacity of its two cells per degree
// between them (common.wgsl). Across a stiffer face heat would go further in
// a step than one explicit step can take it without overshooting: steel or
// a steam film on the day's clock, nearly anything in millimetre cells. Those
// faces go here instead, before the update pass, in s stages of damped
// first-order Runge-Kutta-Chebyshev: each stage an explicit step across them,
// mixed with the two before it by weights from Chebyshev polynomials, so
// that s stages are stable up to about 1.8 s² times what one explicit step
// takes (rather than s times) and the stiffest wiggles are damped
// (Verwer, Hundsdorfer & Sommeijer, "Convergence properties of the
// Runge-Kutta-Chebyshev method", Numer. Math. 57, 1990; Hundsdorfer &
// Verwer, Numerical Solution of Time-Dependent Advection-Diffusion-Reaction
// Equations, Springer 2003). Each stage moves heat between two cells by one
// amount worked out alike on both sides, and the weights add up to one, so
// heat is conserved exactly. The host sets the stages (sim.js,
// heatStageCount) by the stiffest face prepare found a frame or so before;
// a face stiffer than they take (Params.heatLimit, until the host catches
// up) passes as much as they do. With no stages, prepare still runs every
// few steps (sim.js HEAT_LOOK_EVERY) to find stiff faces, which update.wgsl
// meanwhile holds to what stays monotone.
//
// prepare: per cell, the J/K a step passed across its right and lower faces
//          if stiff (0 if not), its heat capacity and phase points, its
//          stored and effective temperatures; per workgroup tile, whether
//          any of its cells has a stiff face (the stages leave the rest);
//          and the stiffest face of all, for the host.
// stage:   one stage, from the stage before's temperatures (and its own the
//          stage before that); the last writes the stored temperatures back.

struct Stage {
    mu: f32,    // weight of the stage before's stored temperature
    nu: f32,    // of the one before that
    heat: f32,  // of what the stage before's temperatures pass
    last: u32,  // the last stage, which writes the cells
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> elems: array<Elem>;
@group(0) @binding(2) var<storage, read_write> cells: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> air: array<vec4<f32>>;
// Heat's clock per light cell (clock.wgsl).
@group(0) @binding(4) var clock: texture_2d<f32>;
// Per cell: J/K passed a step across its right and lower faces, if stiff,
// and its heat capacity; its phase points (hiT, loT; phaseOf).
@group(0) @binding(5) var<storage, read_write> faces: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> phases: array<vec2<f32>>;
// Stage temperatures per cell (stored, effective), two buffers in turn: the
// stage before's, read, and this one's, written over the stage before that's.
@group(0) @binding(7) var<storage, read_write> before: array<vec2<f32>>;
@group(0) @binding(8) var<storage, read_write> after: array<vec2<f32>>;
// Per tile of 8 x 8 cells (a workgroup): 1 if any of its cells has a stiff
// face.
@group(0) @binding(9) var<storage, read_write> tiles: array<u32>;
@group(0) @binding(10) var<uniform> S: Stage;
// The stiffest face prepare has found since the host cleared it (before
// each frame), as J/K a step over the smaller heat capacity of its two
// cells, as its bits (positive floats order as their bits do).
@group(0) @binding(11) var<storage, read_write> stiffness: array<atomic<u32>>;

var<workgroup> stiffHere: atomic<u32>;
var<workgroup> stiffestHere: atomic<u32>;

fn tileIndex(wid: vec2<u32>) -> u32 {
    return wid.y * ((P.size.x + 7u) / 8u) + wid.x;
}

fn realnessAt(p: vec2<u32>) -> f32 {
    return textureLoad(clock, p / 2u, 0).x;
}

// What cell i, of type t, holds of the air (the gas plane), if it holds gas.
fn gasAtCell(i: u32, t: u32) -> Gas {
    var g = noGas();
    if (holdsGas(t)) {
        g = gasFrom(cells[gasIndex(P.size.x * P.size.y, i)], i);
    }
    return g;
}

// The heat capacity of cell i, of type t: a gas's what it holds (the gas
// plane; heatCapOf, none in a vacuum), liquid's by how much (heatCapIn).
fn heatCapAt(i: u32, t: u32) -> f32 {
    return heatCapIn(cells[i], gasAtCell(i, t));
}

// Right, down, left, up.
fn faceStep(d: u32) -> vec2<i32> {
    return select(vec2<i32>(0, select(1, -1, d == 3u)), vec2<i32>(select(1, -1, d == 2u), 0), (d & 1u) == 0u);
}

@compute @workgroup_size(8, 8)
fn prepare(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let inside = all(gid.xy < P.size);
    let i = gid.y * P.size.x + gid.x;
    if (inside) {
        let p = vec2<i32>(gid.xy);
        let v = cells[i];
        let t = cellType(v);
        let e = elems[t];
        let heatCap = heatCapAt(i, t);
        let real = realnessAt(gid.xy);
        // Each face as update.wgsl's heatPassed has it, so the two agree on
        // which are stiff.
        var passed = vec2<f32>(0.0);
        var stiffest = 0.0;
        for (var d = 0u; d < 4u; d++) {
            let q = p + faceStep(d);
            if (any(q < vec2<i32>(0)) || any(q >= vec2<i32>(P.size))) {
                continue;
            }
            let k = u32(q.y) * P.size.x + u32(q.x);
            let w = cells[k];
            let nt = cellType(w);
            let n = elems[nt];
            let tau = clockStep(max(real, realnessAt(vec2<u32>(q))), P.dayDt, P.clockSpan);
            let pull = P.earthG * P.dx / (P.dt * P.dt);
            var wind = 0.0;
            if ((e.kind == KIND_GAS) != (n.kind == KIND_GAS)) {
                wind = windSpeedAt(select(q, p, e.kind == KIND_GAS));
            }
            let nCap = heatCapAt(k, nt);
            var conductance = cellFaceConductance(t, v.x, gasAtCell(i, t), tempAtCell(v, p), nt, w.x, gasAtCell(k, nt), tempAtCell(w, q), pull, P.dx, wind);
            // A liquid turning over (update.wgsl's, d 3 the cell above).
            if (e.kind == KIND_LIQUID && n.kind == KIND_LIQUID) {
                if (d == 3u) {
                    conductance += turnoverConductance(w.x, cellTemp(w), v.x, cellTemp(v), nCap, heatCap, pull, P.dx, tau, P.dt, false);
                } else {
                    conductance += turnoverConductance(v.x, cellTemp(v), w.x, cellTemp(w), heatCap, nCap, pull, P.dx, tau, P.dt, (d & 1u) == 0u);
                }
            }
            let g = conductance * faceSeconds(e, n, tau, P.dayDt);
            let small = min(heatCap, nCap);
            if (small > 0.0 && g > HEAT_SHARE_MAX * small) {
                stiffest = max(stiffest, g / small);
                if (d < 2u) {
                    passed[d] = min(g, P.heatLimit * small);
                }
            }
        }
        // With no stages this step, only looking for stiff faces.
        if (P.heatStages > 0u) {
            faces[i] = vec4<f32>(passed, heatCap, 0.0);
        }
        if (stiffest > 0.0) {
            atomicOr(&stiffHere, 1u);
            atomicMax(&stiffestHere, bitcast<u32>(stiffest));
        }
    }
    workgroupBarrier();
    let stiffTile = atomicLoad(&stiffHere) != 0u;
    if (li == 0u) {
        tiles[tileIndex(wid.xy)] = select(0u, 1u, stiffTile);
        let most = atomicLoad(&stiffestHere);
        if (most > 0u) {
            atomicMax(&stiffness[0], most);
        }
    }
    // What the stages read of a cell beyond its faces, only in a tile they
    // work on (they read a neighbour's across a stiff face only, whose
    // tile is one): its phase points at its own air pressure (update.wgsl
    // takes its neighbours' at its own, for the faces it works out), and
    // its temperatures.
    if (inside && stiffTile && P.heatStages > 0u) {
        let v = cells[i];
        let t = cellType(v);
        let tr = phaseOf(t, wordSolute(v.x), wordStrength(v.x), atmLog(airPressureAt(vec2<f32>(gid.xy) + 0.5)));
        phases[i] = tr;
        before[i] = vec2<f32>(cellTemp(v), effTempIn(t, cellTemp(v), tr));
    }
}

@compute @workgroup_size(8, 8)
fn stage(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    if (tiles[tileIndex(wid.xy)] == 0u || any(gid.xy >= P.size)) {
        return;
    }
    let w = P.size.x;
    let i = gid.y * w + gid.x;
    let f = faces[i];
    let own = before[i];
    // What the stiff faces pass at the stage before's temperatures. A
    // neighbour across one is in a tile with a stiff face, so has them.
    var heat = 0.0;
    var stiff = false;
    if (f.x > 0.0) {
        heat += f.x * (before[i + 1u].y - own.y);
        stiff = true;
    }
    if (f.y > 0.0) {
        heat += f.y * (before[i + w].y - own.y);
        stiff = true;
    }
    if (gid.x > 0u) {
        let k = faces[i - 1u].x;
        if (k > 0.0) {
            heat += k * (before[i - 1u].y - own.y);
            stiff = true;
        }
    }
    if (gid.y > 0u) {
        let k = faces[i - w].y;
        if (k > 0.0) {
            heat += k * (before[i - w].y - own.y);
            stiff = true;
        }
    }
    // The first stage has no stage before the one before.
    let older = select(own.x, after[i].x, S.nu != 0.0);
    let stored = S.mu * own.x + S.nu * older + select(0.0, S.heat * heat / f.z, stiff);
    after[i] = vec2<f32>(stored, effTempIn(cellType(cells[i]), stored, phases[i]));
    if (S.last != 0u && stiff) {
        cells[i].y = bitcast<u32>(stored);
    }
}
