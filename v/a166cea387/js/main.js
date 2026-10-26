// Bootstrap: WebGPU setup, world sizing and the frame loop.

import { DevTools } from "./devtools.js";
import { loadScene, sceneSize } from "./scenes.js";
import { STEPS_PER_SECOND } from "./shared.js";
import { CLOCK, createPipelines, LIGHT_EVERY, MAX_STEPS_PER_FRAME, maxCells, PHASES, Simulation, STORAGE_BUFFERS_NEEDED } from "./sim.js";
import { readSnapshot, restoreSnapshot, saveSnapshot } from "./snapshot.js";
import { Tour } from "./tour.js";
import { Controls } from "./ui.js";

const SHADERS = [
    "common", "shade", "airsample", "pull", "label", "labeltile", "rooms", "heatin", "airsource", "update", "heat", "pairs", "sky", "fluid", "bodies", "block", "electric", "air", "carry", "stress", "radiate",
    "brush", "glow", "blur", "render", "light", "clock", "gravity", "person", "people", "figures",
];
const MAX_FRAME_DT = 0.25;
const INTERVAL_EASE = 0.02;  // share of a longer frame interval taken into the estimate per frame
const LIGHT_SHARE = 0.5;     // of a frame's time its work may take for the frame to say what the display's is
const MIN_GRID = 16;
const START_CLOCK = { day: 9 / 24, speed: CLOCK.NORMAL };  // a fresh world's morning

// The grid (CSS pixels a cell) is picked for the computer the app opens on
// (pickGrid): the finest of these whose empty world's step takes the GPU at
// most STEP_BUDGET_MS. A busy world takes up to about twice an empty one's,
// which still leaves room in a 60 Hz frame (16.7 ms) for the step and the
// drawing. (What encoding a step takes the CPU is the same at any grid.) One picked under World > Grid is kept instead, from then on
// (GRID_KEY). ?budget=ms in the address sets the budget (to see what a
// slower computer gets).
const AUTO_GRIDS = [3, 4, 6, 8];  // Normal, Coarse, Chunky, Huge
const STEP_BUDGET_MS = Number(new URLSearchParams(location.search).get("budget")) || 8;
const SETTLE_FRAMES = 8;  // frames a world runs before its step is timed (pickGrid)
const GRID_KEY = "sand.grid";

// The grid picked under World > Grid, or null.
function pickedGrid() {
    try {
        const cell = Number(localStorage.getItem(GRID_KEY));
        return cell > 0 ? cell : null;
    } catch {
        return null;
    }
}

function keepGrid(cell) {
    try {
        localStorage.setItem(GRID_KEY, String(cell));
    } catch {
        // (No storage, as in some private windows: picked again each time.)
    }
}

const canvas = document.getElementById("view");
const stage = document.getElementById("stage");

// The performance readout (G): frame and step rates, and GPU time per phase.
function perfReport(fps, stepsPerSecond, stepsPerFrame, timings) {
    const lines = [`${Math.round(fps)} fps · ${Math.round(stepsPerSecond)} steps/s · ${stepsPerFrame.toFixed(1)} steps/frame`];
    if (!timings) {
        lines.push("GPU times need the timestamp-query feature");
    } else if (timings.frames === 0) {
        lines.push("GPU times pending");
    } else {
        const ms = PHASES.map((phase) => timings.ms[phase] ?? 0);
        const total = ms.reduce((a, b) => a + b, 0);
        lines.push(`GPU ${total.toFixed(1)} ms/frame`);
        lines.push(PHASES.map((phase, i) => `${phase} ${ms[i].toFixed(1)}`).join(" · "));
    }
    return lines.join("\n");
}

function showError(message) {
    const el = document.getElementById("error");
    el.textContent = message;
    el.hidden = false;
}

// Asked of the server every time (it answers "unchanged" if so): a shader the
// browser kept from before would not match the JS that drives it. Found from
// this module, not the page (the packaged app keeps them both under v/<hash>/,
// tools/package.mjs).
async function loadShaders() {
    const texts = await Promise.all(SHADERS.map(async (name) => {
        const res = await fetch(new URL(`../shaders/${name}.wgsl`, import.meta.url), { cache: "no-cache" });
        if (!res.ok) {
            // The packaged page's recovery (tools/package.mjs STALE_PAGE): a
            // page and modules the browser kept, their version's folder gone
            // since, reload once into the site's version.
            window.filesMissing?.();
            throw new Error(`Failed to load shaders/${name}.wgsl (${res.status})`);
        }
        return res.text();
    }));
    return Object.fromEntries(SHADERS.map((name, i) => [name, texts[i]]));
}

async function start() {
    if (!navigator.gpu) {
        throw new Error(
            "WebGPU is not available in this browser.\n\n"
            + "Use a recent Chrome/Chromium or Edge. On Linux you may need to enable "
            + "chrome://flags/#enable-unsafe-webgpu (and Vulkan).",
        );
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) {
        throw new Error("No WebGPU adapter available.");
    }
    // Timestamp queries power the per-phase GPU times in the readout. The
    // liquid's passes bind more storage buffers than the default limit of 8,
    // and big worlds need bigger buffers than the default 128 MiB: ask for
    // what the adapter has (see maxCells).
    if (adapter.limits.maxStorageBuffersPerShaderStage < STORAGE_BUFFERS_NEEDED) {
        throw new Error(
            `This browser lets a GPU shader use ${adapter.limits.maxStorageBuffersPerShaderStage} storage buffers, `
            + `and the simulation needs ${STORAGE_BUFFERS_NEEDED}.\n\n`
            + "It runs in Chrome or Edge on most computers; Safari doesn't allow that many yet.",
        );
    }
    const requiredFeatures = adapter.features.has("timestamp-query") ? ["timestamp-query"] : [];
    const requiredLimits = {
        maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
    };
    const device = await adapter.requestDevice({ requiredFeatures, requiredLimits });
    device.lost.then((info) => {
        if (info.reason !== "destroyed") {
            showError(`GPU device lost: ${info.message}`);
        }
    });
    device.addEventListener("uncapturederror", (e) => {
        console.error(e.error.message);
        showError(e.error.message);
    });

    const context = canvas.getContext("webgpu");
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: "opaque" });

    const pipes = await createPipelines(device, format, await loadShaders());
    let sim = null;

    // Fit the grid into the canvas, centered, preserving aspect ratio.
    const transform = () => {
        const scale = Math.min(canvas.width / sim.width, canvas.height / sim.height);
        return {
            scale,
            originX: (canvas.width - sim.width * scale) / 2,
            originY: (canvas.height - sim.height * scale) / 2,
        };
    };

    const toCell = (clientX, clientY) => {
        const rect = canvas.getBoundingClientRect();
        const t = transform();
        const px = (clientX - rect.left) * (canvas.width / rect.width);
        const py = (clientY - rect.top) * (canvas.height / rect.height);
        return { x: (px - t.originX) / t.scale, y: (py - t.originY) / t.scale };
    };

    // A cell's square on the page (client pixels): the developer's
    // toolbox marks the cell it has pinned.
    const cellBox = (x, y) => {
        const rect = canvas.getBoundingClientRect();
        const t = transform();
        const k = rect.width / canvas.width;
        return { left: rect.left + (t.originX + x * t.scale) * k, top: rect.top + (t.originY + y * t.scale) * k, size: t.scale * k };
    };

    const resizeCanvas = () => {
        const dpr = window.devicePixelRatio || 1;
        canvas.width = Math.max(1, Math.round(stage.clientWidth * dpr));
        canvas.height = Math.max(1, Math.round(stage.clientHeight * dpr));
    };

    const controls = new Controls({
        canvas,
        toCell,
        gridSize: () => ({ width: sim.width, height: sim.height }),
        onClear: () => sim.clear(),
        onResolution: (cellSize) => {
            newWorld(cellSize);
            keepGrid(cellSize);
        },
        // A scene is built in a world of its own size, drawn as big as fits.
        onScene: (scene) => {
            const [width, height] = sceneSize(scene);
            if (width !== sim.width || height !== sim.height) {
                newWorld(controls.state.cellSize, [width, height]);
            }
            loadScene(sim, scene);
            return [sim.metresPerCell, sim.blackHolePull, sim.earthGravity];
        },
        onScale: (metres) => {
            sim.metresPerCell = metres;
        },
        onHolePull: (g) => sim.setBlackHolePull(g),
        onEarthGravity: (on) => sim.setEarthGravity(on),
        onClock: (day, speed) => sim.setClock(day, speed),
        clock: () => sim?.clock ?? START_CLOCK,
        onSaveSnapshot: () => saveSnapshot(sim),
        // A snapshot of a world of another size gets a world of its size,
        // drawn as big as fits.
        onLoadSnapshot: async (bytes) => {
            const snapshot = await readSnapshot(bytes);
            const { width, height } = snapshot.header;
            if (width !== sim.width || height !== sim.height) {
                if (width * height > largest) {
                    throw new Error(`a ${width}×${height} world is too big for this GPU`);
                }
                newWorld(controls.state.cellSize, [width, height], (next) => restoreSnapshot(next, snapshot));
            } else {
                restoreSnapshot(sim, snapshot);
            }
            return [sim.metresPerCell, sim.blackHolePull, sim.earthGravity];
        },
    });

    // A world too big for the GPU's buffers (a large screen at Fine) gets
    // coarser cells. Errors shown for the last world don't hold for the new
    // one. With size ([width, height]), a world that size. A world that
    // can't be made throws before the old one goes. With fill, which puts
    // something in the new world (a snapshot) and may throw, the old world
    // goes only once that has worked: until then there are two (without
    // one, the old goes first, as two may not fit in the GPU's memory).
    const largest = maxCells(device.limits);
    const newWorld = (cellSize, size = null, fill = null) => {
        let cell = cellSize;
        const dims = () => size ?? [
            Math.max(MIN_GRID, Math.floor(stage.clientWidth / cell)),
            Math.max(MIN_GRID, Math.floor(stage.clientHeight / cell)),
        ];
        while (!size && dims()[0] * dims()[1] > largest) {
            cell++;
        }
        const [width, height] = dims();
        Simulation.check(width, height);
        // The time of day and the world's settings carry over to the new one.
        const clock = sim?.clock ?? START_CLOCK;
        if (!fill) {
            sim?.destroy();
        }
        const next = new Simulation(device, pipes, width, height, clock, controls.state.scale);
        next.setBlackHolePull(controls.state.holePull);
        next.setEarthGravity(controls.state.earthGravity);
        if (fill) {
            try {
                fill(next);
            } catch (err) {
                next.destroy();
                throw err;
            }
            sim?.destroy();
        }
        sim = next;
        sim.onHover = (cell) => controls.showHover(cell, sim.personAt(cell.x, cell.y));
        document.getElementById("grid").textContent = `${width}×${height}`;
        document.getElementById("error").hidden = true;
        controls.repoint();
    };

    // The grid to open with: one picked under World > Grid, or Normal and
    // then pickGrid's.
    const resolution = document.getElementById("resolution");
    const setGrid = (cell) => {
        controls.state.cellSize = cell;
        resolution.value = String(cell);
    };
    const picked = pickedGrid();
    if (picked && [...resolution.options].some((o) => Number(o.value) === picked)) {
        setGrid(picked);
    }

    resizeCanvas();
    newWorld(controls.state.cellSize);
    // The developer's toolbox, out of sight until ` or ?dev (devtools.js).
    const devtools = new DevTools({ device, controls, sim: () => sim, toCell, cellBox });

    // The finest grid this computer keeps up with, by what a step of the
    // empty world takes the GPU (timeSteps): too slow, the finest coarser one
    // its cells, taken as costing alike, say fits, measured again (its step
    // costs more than that: a step has a part that doesn't shrink with the
    // world). Before the first frame, so nothing else is on the GPU, and
    // the world is still empty.
    const pickGrid = async () => {
        // Frames of a step each first, each after the reports the ones before
        // asked for, as the app's are: a world's first steps work out its
        // light and gravity afresh, and encode all the air's iterations
        // (sim.js airCycles) until a report says how many it needs, three
        // times what an empty world's take once it has one. Then
        // LIGHT_EVERY steps, the light once in them.
        const settle = async () => {
            for (let k = 0; k < SETTLE_FRAMES; k++) {
                await sim.reportsTaken();
                sim.frame({
                    steps: 1, target: context.getCurrentTexture().createView(), hover: null,
                    view: {
                        ...transform(), ...controls.cursor(), time: performance.now() / 1000, mode: controls.state.view,
                        canvasWidth: canvas.width, canvasHeight: canvas.height,
                    },
                });
                await device.queue.onSubmittedWorkDone();
            }
        };
        const measure = async () => {
            await settle();
            return (await sim.timeSteps(LIGHT_EVERY)).gpu;
        };
        // (Said only if it takes a while: a fast computer is done at once.)
        const notice = setTimeout(() => controls.showToast("Finding a grid this computer keeps up with…"), 600);
        const cellsAt = (cell) => Math.floor(stage.clientWidth / cell) * Math.floor(stage.clientHeight / cell);
        const coarsest = AUTO_GRIDS[AUTO_GRIDS.length - 1];
        let cell = controls.state.cellSize;
        let cost = await measure();
        while (cost > STEP_BUDGET_MS && cell < coarsest) {
            const scaled = (g) => cost * cellsAt(g) / cellsAt(cell);
            cell = AUTO_GRIDS.find((g) => g > cell && scaled(g) <= STEP_BUDGET_MS) ?? coarsest;
            setGrid(cell);
            newWorld(cell);
            cost = await measure();
        }
        clearTimeout(notice);
        if (cell !== AUTO_GRIDS[0]) {
            const name = [...resolution.options].find((o) => Number(o.value) === cell).textContent;
            controls.showToast(`Using the ${name} grid (bigger cells) so this computer keeps up. Change it under World ▸ Grid.`);
        } else {
            controls.hideToast();
        }
    };
    // (The panel waits meanwhile: a scene or snapshot loaded, or a grid
    // picked, would put a world in place of the one being timed.)
    if (!picked) {
        const panel = document.getElementById("panel");
        panel.inert = true;
        try {
            await pickGrid();
        } finally {
            panel.inert = false;
        }
    }
    // The tour, the first time the app is opened here (the ? button again).
    const tour = new Tour(controls);
    tour.openFirstTime();
    new ResizeObserver(() => {
        resizeCanvas();
        controls.repoint();
    }).observe(stage);

    const fpsEl = document.getElementById("fps");
    const perfEl = document.getElementById("perf");
    let last = performance.now();
    let acc = 0;
    let fpsFrames = 0;
    let fpsSteps = 0;
    let fpsStart = last;
    let lastTimings = null;  // shown again while a slow frame's times are still on the way
    // The display's frame interval: taken down at once to any shorter one,
    // and only slowly up, so a dropped frame now and then doesn't count; and
    // up only by a frame whose work, the CPU's or the GPU's, the longer,
    // took under LIGHT_SHARE of it (a display slower than 60 Hz shows as
    // frames the computer idles through). One the steps made long says how
    // long the steps are, not what the display shows: taken as the display's,
    // when every frame ran late (a step costing more than a frame), it let in
    // more steps a frame, which made the frames later still, up to
    // MAX_STEPS_PER_FRAME at 2 frames a second, the world no faster than at
    // one a frame (test/framepace.mjs).
    let interval = 1 / 60;
    let cpuBusy = 0;  // the last frame's work on the CPU, seconds
    let gpuBusy = 0;  // the latest frame's on the GPU as far as known: from its submission till it was done

    const frame = (now) => {
        const start = performance.now();
        const elapsed = Math.max(now - last, 0) / 1000;
        const dt = Math.min(elapsed, MAX_FRAME_DT);
        last = now;
        if (elapsed > 0.001) {
            if (elapsed < interval) {
                interval = elapsed;
            } else if (Math.max(cpuBusy, gpuBusy) < LIGHT_SHARE * elapsed) {
                interval += (dt - interval) * INTERVAL_EASE;
            }
        }

        // Fixed simulation rate independent of the display refresh rate.
        const s = controls.state;
        let steps;
        if (s.paused) {
            acc = 0;
            steps = controls.takeStep() ? 1 : devtools.takeSteps();
        } else {
            acc += dt * STEPS_PER_SECOND * s.speed;
            steps = Math.floor(acc + 0.25);  // tolerance absorbs vsync jitter
            acc -= steps;
            // No catching up. When steps cost more GPU time than a frame
            // lasts, extra steps to make up for a late frame only make the
            // next frame later still, until the frame rate collapses. Past
            // what the speed setting needs per frame at the display's frame
            // rate (two at 30 Hz, one or two at 50), the backlog is dropped:
            // the simulation slows down and frames stay smooth.
            const perFrame = s.speed * STEPS_PER_SECOND * interval;
            const maxSteps = Math.min(MAX_STEPS_PER_FRAME, Math.max(1, Math.ceil(perFrame - 0.05)));
            if (steps > maxSteps) {
                steps = maxSteps;
                acc = 0;
            }
        }

        sim.holding = controls.drawing();
        for (const brush of controls.takeBrushes()) {
            if (brush.person) {
                sim.addPerson(brush.x, brush.y);
                continue;
            }
            if (brush.fill) {
                sim.fill(brush);
                continue;
            }
            sim.paint(brush);
            // The eraser takes the people it touches: those in a box, or
            // along a stroke.
            if (brush.erasePeople && brush.box) {
                sim.erasePeopleBox(brush.ax, brush.ay, brush.bx, brush.by);
            } else if (brush.erasePeople) {
                sim.erasePeople(brush.ax, brush.ay, brush.radius, brush.bx, brush.by);
            }
        }
        sim.frame({
            steps,
            target: context.getCurrentTexture().createView(),
            view: {
                ...transform(), ...controls.cursor(), time: now / 1000, mode: s.view,
                canvasWidth: canvas.width, canvasHeight: canvas.height,
            },
            hover: controls.hoverCell(),
            timing: s.perf,
        });
        const submitted = performance.now();
        cpuBusy = (submitted - start) / 1000;
        device.queue.onSubmittedWorkDone().then(() => {
            gpuBusy = (performance.now() - submitted) / 1000;
        });
        devtools.afterFrame(steps);
        controls.showClock();
        tour.update(now);

        fpsFrames++;
        fpsSteps += steps;
        if (now - fpsStart >= 500) {
            const seconds = (now - fpsStart) / 1000;
            fpsEl.textContent = `${Math.round(fpsFrames / seconds)} fps`;
            const timings = sim.takeTimings();
            if (timings?.frames) {
                lastTimings = timings;
            }
            if (s.perf) {
                const shown = timings && (lastTimings ?? timings);
                perfEl.textContent = perfReport(fpsFrames / seconds, fpsSteps / seconds, fpsSteps / fpsFrames, shown);
            }
            fpsFrames = 0;
            fpsSteps = 0;
            fpsStart = now;
        }
        requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
}

start().catch((err) => {
    console.error(err);
    showError(err.message);
});
