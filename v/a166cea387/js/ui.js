// Sidebar controls, pointer input and keyboard shortcuts.

import { cellRecord } from "./cellrecord.js";
import { ELEMENTS, GROUPS, ID, KIND, describe, paintDensity } from "./elements.js";
import { describePerson } from "./people.js";
import { SCENE_GROUPS, SCENES } from "./scenes.js";
import { KELVIN, MIN_ATM, P_ATM, saturation } from "./shared.js";
import { BRUSH, CELL_FLAGS, CHARGE_ACTIVE, DEFAULT_BLACK_HOLE_PULL, DEFAULT_METRES_PER_CELL, SHAPE, VIEW } from "./sim.js";

// Icons, drawn in the button's text colour (style.css sizes them).
const svg = (body) => `<svg viewBox="0 0 24 24" aria-hidden="true">${body}</svg>`;
const ICONS = {
    draw: svg('<path d="M4 20l4-1L19 8l-3-3L5 16z"/><path d="M14 7l3 3"/>'),
    erase: svg('<path d="M8 20h12"/><path d="M4.5 14.5l9-9a2 2 0 0 1 3 0l2 2a2 2 0 0 1 0 3L11 18H8z"/><path d="M9 10l5 5"/>'),
    wind: svg('<path d="M3 8h10a3 3 0 1 0-3-3"/><path d="M3 12h15a3 3 0 1 1-3 3"/><path d="M3 16h6"/>'),
    heat: svg('<path d="M12 21c-3.5 0-6-2.5-6-6 0-3.5 3.5-5 4-10 3 2 4.5 4.5 4.5 7 1-1 1.5-2 1.5-3 1.5 1.5 2 3.5 2 6 0 3.5-2.5 6-6 6z"/>'),
    cool: svg('<path d="M12 3v18M4.2 7.5l15.6 9M4.2 16.5l15.6-9"/><path d="M9.5 4.5L12 6l2.5-1.5M9.5 19.5L12 18l2.5 1.5"/>'),
    spark: svg('<path d="M13 2L5 14h6l-1 8 8-12h-6z"/>'),
    person: svg('<circle cx="12" cy="5" r="2.5"/><path d="M12 8v7M7.5 11h9M12 15l-3.5 6M12 15l3.5 6"/>'),
    round: svg('<circle cx="12" cy="12" r="7"/>'),
    square: svg('<rect x="5" y="5" width="14" height="14" rx="1"/>'),
    spray: svg('<circle cx="12" cy="12" r="1.2"/><circle cx="7" cy="9" r="1"/><circle cx="16.5" cy="8" r="1"/><circle cx="8.5" cy="16" r="1"/><circle cx="16" cy="15.5" r="1"/><circle cx="12" cy="5.5" r="0.8"/><circle cx="5" cy="13.5" r="0.8"/><circle cx="19" cy="12" r="0.8"/><circle cx="12" cy="19" r="0.8"/>'),
    box: svg('<rect x="4" y="6" width="16" height="12" rx="1" stroke-dasharray="3 2.5"/><path d="M15 15l5 5"/>'),
    fill: svg('<path d="M4 11l7-7 8 8-7 7z"/><path d="M4 11h15"/><path d="M20 15c0 1.6 1.2 2.3 1.2 3.4a1.2 1.2 0 0 1-2.4 0c0-1.1 1.2-1.8 1.2-3.4z"/>'),
    pause: svg('<path d="M9 5v14M15 5v14"/>'),
    play: svg('<path d="M7 5l12 7-12 7z"/>'),
    step: svg('<path d="M6 5l9 7-9 7z"/><path d="M18 5v14"/>'),
    clear: svg('<path d="M4 7h16M10 11v6M14 11v6"/><path d="M6 7l1 13h10l1-13M9 7V4h6v3"/>'),
};

const TOOLS = [
    { id: "draw", label: "Draw", key: "d", hint: "Paint the selected element" },
    { id: "erase", label: "Erase", key: "e", hint: "Remove anything (also right mouse)" },
    { id: "wind", label: "Wind", key: "w", hint: "Drag to push air" },
    { id: "heat", label: "Heat", key: "h", hint: "Raise temperature" },
    { id: "cool", label: "Cool", key: "c", hint: "Lower temperature" },
    { id: "spark", label: "Spark", key: "s", hint: "Zap conductors with a static pulse, which sets off gas and explosives it reaches; ignite fuel" },
    {
        id: "person", label: "Person", key: "v",
        hint: "Place a person: they stroll near home, chat, sleep at home at night and run from danger",
    },
];

// Brush shapes. A round, square or spray brush paints along the stroke
// (Shift: a straight line); a box is dragged out and painted on release; a
// fill paints everything joined to the clicked cell that is the same as it.
const SHAPES = [
    { id: "round", label: "Round", key: "1", hint: "A round brush" },
    { id: "square", label: "Square", key: "2", hint: "A square brush" },
    { id: "spray", label: "Spray", key: "3", hint: "Scattered dots, thinning out to the edge; hold to build up" },
    { id: "box", label: "Box", key: "4", hint: "Drag out a box, filled on release" },
    { id: "fill", label: "Fill", key: "5", hint: "Fill everything joined to the clicked cell that is the same as it" },
];

// What the brush paints (erases, heats, ...) over.
const OVERS = [
    { id: "empty", label: "Empty", hint: "Paint only into empty space" },
    { id: "all", label: "All", hint: "Paint over anything" },
    { id: "only", label: "Only", hint: "Paint only over the element chosen below" },
];

// Which brush options each tool has.
const SHAPED = new Set(["draw", "erase", "heat", "cool", "spark"]);
const DENSITY_TOOLS = new Set(["draw", "erase"]);

// The outline drawn for the brush (render.wgsl's CURSOR_*).
const CURSOR = { ROUND: 1, SQUARE: 2, SPRAY: 3, BOX: 4, FILL: 5, LINE: 16 };

// Where a placed person's feet go, below the pointer: their middle at it.
const PERSON_DROP = 5;

const TEMP_RATE = 40;    // degrees per frame for the heat and cool tools
const WIND_GAIN = 0.6;   // air speed added per cell of drag
const AIM_DISTANCE = 0.75; // cells the pointer must move before the stroke direction updates
const MIN_SIZE = 1;
const MAX_SIZE = 40;
const TOAST_MS = 7000;
const MIN_AMPS = 1e-6;   // less current than this through a cell isn't shown

// Element e's [hiT, loT] under air pressure p, as transitionsAt() in common.wgsl.
function transitionsAt(e, p) {
    if (!(e.boilShift > 0) || p === 0) {
        return [e.hiT, e.loT];
    }
    const gas = e.kind === KIND.GAS;
    const t0 = (gas ? e.loT : e.hiT) + KELVIN;
    const atm = Math.max(1 + p / P_ATM, MIN_ATM);
    const boil = 1 / (1 / t0 - Math.log(atm) / e.boilShift) - KELVIN;
    return gas ? [e.hiT, boil] : [boil, e.loT];
}

// The scale slider runs from 1 mm to 50 m a cell, evenly in the log.
const SCALE_MIN_LOG = -3;
const SCALE_LOG_SPAN = Math.log10(50) - SCALE_MIN_LOG;
const SCALE_STEPS = 1000;

function scaleFromSlider(value) {
    return 10 ** (SCALE_MIN_LOG + (value / SCALE_STEPS) * SCALE_LOG_SPAN);
}

function sliderFromScale(metres) {
    return Math.round(((Math.log10(metres) - SCALE_MIN_LOG) / SCALE_LOG_SPAN) * SCALE_STEPS);
}

// The black hole slider runs from 1 g to 100 g at its edge, evenly in the log.
const HOLE_LOG_SPAN = 2;

function holePullFromSlider(value) {
    return 10 ** ((value / SCALE_STEPS) * HOLE_LOG_SPAN);
}

function sliderFromHolePull(g) {
    return Math.round((Math.log10(g) / HOLE_LOG_SPAN) * SCALE_STEPS);
}

// A length in metres in the handiest unit: "4 mm", "1 cm", "5.5 m".
function formatLength(metres) {
    const [value, unit] = metres < 0.01 ? [metres * 1000, "mm"] : metres < 1 ? [metres * 100, "cm"] : [metres, "m"];
    return `${value < 10 ? +value.toFixed(1) : Math.round(value)} ${unit}`;
}

function cssColor(rgb) {
    const [r, g, b] = rgb.map((c) => Math.round(Math.min(1, c) * 255));
    return `rgb(${r} ${g} ${b})`;
}

// Current with a metric prefix, three significant digits: "2.41 kA", "35.0 mA".
function formatAmps(amps) {
    const prefixes = [[1e3, "k"], [1, ""], [1e-3, "m"], [1e-6, "µ"]];
    const [scale, prefix] = prefixes.find(([s]) => amps >= s) ?? prefixes[prefixes.length - 1];
    return `${(amps / scale).toPrecision(3)} ${prefix}A`;
}

const MINUTES_PER_DAY = 1440;

// "9:05" for a time of day given as a share of the day since midnight.
function formatClock(day) {
    const minutes = Math.floor(day * MINUTES_PER_DAY) % MINUTES_PER_DAY;
    return `${Math.floor(minutes / 60)}:${String(minutes % 60).padStart(2, "0")}`;
}

// A button with an icon and a tooltip naming its shortcut.
function iconButton(className, icon, label, hint, key) {
    const button = document.createElement("button");
    button.className = `${className} icon`;
    button.innerHTML = icon;
    button.setAttribute("aria-label", label);
    button.title = `${label}: ${hint}${key ? ` (${key.toUpperCase()})` : ""}`;
    return button;
}

export class Controls {
    // onClock(day, speed) sets the time of day and the clock's speed (either
    // undefined to leave it); clock() is the simulation's clock as it is.
    // onSaveSnapshot() is the world as a snapshot's bytes (snapshot.js);
    // onLoadSnapshot(bytes) loads one, and returns the world's scale, black
    // hole pull and Earth's gravity as onScene does.
    constructor({
        canvas, toCell, gridSize, onClear, onResolution, onScene, onScale, onHolePull, onEarthGravity, onClock, clock,
        onSaveSnapshot, onLoadSnapshot,
    }) {
        this.canvas = canvas;
        this.toCell = toCell;
        this.gridSize = gridSize;
        this.onClear = onClear;
        this.onResolution = onResolution;
        this.onScene = onScene;  // loads a scene, returns the world's scale, black hole pull and Earth's gravity
        this.onScale = onScale;
        this.onHolePull = onHolePull;
        this.onEarthGravity = onEarthGravity;
        this.onClock = onClock;
        this.clock = clock;
        this.onSaveSnapshot = onSaveSnapshot;
        this.onLoadSnapshot = onLoadSnapshot;
        this.clockShown = null;  // the minute and speed shown, to touch the DOM only when they change

        this.state = {
            tool: "draw",
            element: ID.SAND,
            size: 6,
            shape: "round",
            density: null,  // share of the brush's cells painted, or null for the element's own (paintDensity)
            temp: null,     // °C painted at, or null for the element's own
            over: "empty",  // OVERS
            overElement: ID.SAND,
            view: VIEW.NORMAL,
            paused: false,
            speed: 1,
            perf: false,  // performance readout
            cellSize: Number(document.getElementById("resolution").value),
            scale: DEFAULT_METRES_PER_CELL,  // metres a cell
            holePull: DEFAULT_BLACK_HOLE_PULL,  // g at a black hole's edge
            earthGravity: true,  // the Earth pulls
        };
        // x and y in cells, from clientX and clientY on the page (kept to
        // work them out again when the world or the canvas changes size).
        // A stroke is a line (Shift) or a box (the box shape) while dragged.
        this.pointer = {
            inside: false, down: false, line: false, box: false, button: 0,
            x: 0, y: 0, startX: 0, startY: 0, lastX: 0, lastY: 0, clientX: 0, clientY: 0,
        };
        // Direction of the current stroke, used to aim fans. Starts pointing up.
        this.aim = { x: 0, y: 0, angle: -Math.PI / 2 };
        // Brushes are built when the input happens, so a tap or line that
        // starts and ends between two frames still uses the settings of that moment.
        this.queue = [];
        this.stepRequested = false;
        this.hoverType = null;  // the element under the pointer, as last read back (for the eyedropper)
        // Strokes finished by tool (one with the right button is erase's),
        // with fills and people placed, and scenes loaded, and the last
        // stroke's tool and element: what the tour watches for (tour.js).
        this.counts = { draw: 0, erase: 0, wind: 0, heat: 0, cool: 0, spark: 0, person: 0, scenes: 0 };
        this.lastStroke = null;

        this.hoverEl = document.getElementById("hover");
        this.perfEl = document.getElementById("perf");
        this.toastEl = document.getElementById("toast");
        this.toastTimer = 0;
        this.buildScenes();
        this.buildBrush();
        this.buildPalette();
        this.bindSettings();
        this.bindPointer();
        this.bindKeys();
        this.sync();
    }

    // One collapsible section per scene group, all closed to start with.
    buildScenes() {
        const container = document.getElementById("scenes");
        for (const group of SCENE_GROUPS) {
            const scenes = SCENES.filter((s) => s.group === group);
            const section = document.createElement("details");
            section.className = "scene-group";
            const summary = document.createElement("summary");
            const count = document.createElement("span");
            count.className = "count";
            count.textContent = String(scenes.length);
            summary.append(group, count);
            const grid = document.createElement("div");
            grid.className = "scenes";
            for (const scene of scenes) {
                const button = document.createElement("button");
                button.className = "scene";
                button.dataset.scene = scene.id;
                button.textContent = scene.name;
                button.title = scene.hint;
                button.addEventListener("click", () => {
                    this.state.paused = false;
                    // The scene sets the world's scale, its black holes'
                    // pull and whether the Earth pulls (loadScene).
                    [this.state.scale, this.state.holePull, this.state.earthGravity] = this.onScene(scene);
                    this.counts.scenes++;
                    this.sync();
                    this.showToast(scene.hint);
                });
                grid.append(button);
            }
            section.append(summary, grid);
            container.append(section);
        }
    }

    showToast(text) {
        this.toastEl.textContent = text;
        this.toastEl.classList.add("visible");
        clearTimeout(this.toastTimer);
        this.toastTimer = setTimeout(() => this.toastEl.classList.remove("visible"), TOAST_MS);
    }

    hideToast() {
        clearTimeout(this.toastTimer);
        this.toastEl.classList.remove("visible");
    }

    // The brush card: the tools, the shapes and what it paints over.
    buildBrush() {
        const tools = document.getElementById("tools");
        for (const tool of TOOLS) {
            const button = iconButton("tool", ICONS[tool.id], tool.label, tool.hint, tool.key);
            button.dataset.tool = tool.id;
            button.addEventListener("click", () => this.setTool(tool.id));
            tools.append(button);
        }
        const shapes = document.getElementById("shapes");
        for (const shape of SHAPES) {
            const button = iconButton("shape", ICONS[shape.id], shape.label, shape.hint, shape.key);
            button.dataset.shape = shape.id;
            button.addEventListener("click", () => this.setShape(shape.id));
            shapes.append(button);
        }
        const over = document.getElementById("over");
        for (const option of OVERS) {
            const button = document.createElement("button");
            button.className = "over";
            button.dataset.over = option.id;
            button.textContent = option.label;
            button.title = option.hint;
            button.addEventListener("click", () => {
                this.state.over = option.id;
                this.sync();
            });
            over.append(button);
        }
        // The element "Only" paints over: any of them, by group.
        const only = document.getElementById("over-element");
        for (const group of GROUPS) {
            const optgroup = document.createElement("optgroup");
            optgroup.label = group;
            for (const e of ELEMENTS.filter((el) => el.group === group && !el.paintAs)) {
                const option = document.createElement("option");
                option.value = String(e.index);
                option.textContent = e.name;
                optgroup.append(option);
            }
            only.append(optgroup);
        }
        only.addEventListener("change", () => {
            this.state.overElement = Number(only.value);
            this.state.over = "only";
            this.sync();
        });
    }

    buildPalette() {
        const container = document.getElementById("palette");
        for (const group of GROUPS) {
            const section = document.createElement("section");
            section.className = "element-group";
            const heading = document.createElement("h3");
            heading.textContent = group;
            const grid = document.createElement("div");
            grid.className = "elements";
            for (const e of ELEMENTS.filter((el) => el.group === group)) {
                const button = document.createElement("button");
                button.className = "element";
                button.dataset.element = String(e.index);
                button.dataset.name = e.name.toLowerCase();
                button.style.setProperty("--swatch", cssColor(e.color));
                button.textContent = e.name;
                button.title = `${e.name}: ${describe(e)}`;
                button.addEventListener("click", () => this.setElement(e.index));
                grid.append(button);
            }
            section.append(heading, grid);
            container.append(section);
        }
    }

    // Shows only the elements whose name has the search in it, and the
    // groups with any left.
    filterPalette(query) {
        const q = query.trim().toLowerCase();
        for (const section of document.querySelectorAll(".element-group")) {
            let shown = 0;
            for (const button of section.querySelectorAll(".element")) {
                const match = !q || button.dataset.name.includes(q);
                button.hidden = !match;
                shown += match ? 1 : 0;
            }
            section.hidden = shown === 0;
        }
    }

    bindSettings() {
        const s = this.state;
        const size = document.getElementById("size");
        size.addEventListener("input", () => this.setSize(Number(size.value)));

        // Density and temperature follow their sliders; their values, clicked,
        // go back to the element's own.
        const density = document.getElementById("density");
        density.addEventListener("input", () => {
            s.density = Number(density.value) / 100;
            this.sync();
        });
        document.getElementById("density-out").addEventListener("click", () => {
            s.density = null;
            this.sync();
        });
        const temp = document.getElementById("temp");
        temp.addEventListener("input", () => {
            s.temp = Number(temp.value);
            this.sync();
        });
        document.getElementById("temp-out").addEventListener("click", () => {
            s.temp = null;
            this.sync();
        });

        // The search: typed into, it filters the elements; Enter picks the
        // first left, Escape clears it.
        const search = document.getElementById("search");
        search.addEventListener("input", () => this.filterPalette(search.value));
        search.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                const first = [...document.querySelectorAll(".element")].find((b) => !b.hidden);
                if (first) {
                    this.setElement(Number(first.dataset.element));
                }
                search.blur();
            } else if (e.key === "Escape") {
                search.value = "";
                this.filterPalette("");
                search.blur();
            }
        });

        const speed = document.getElementById("speed");
        speed.addEventListener("change", () => {
            s.speed = Number(speed.value);
        });

        const resolution = document.getElementById("resolution");
        resolution.addEventListener("change", () => {
            s.cellSize = Number(resolution.value);
            this.onResolution(s.cellSize);
        });

        const scale = document.getElementById("scale");
        scale.addEventListener("input", () => {
            s.scale = scaleFromSlider(Number(scale.value));
            this.onScale(s.scale);
            this.sync();
        });

        const holePull = document.getElementById("hole-pull");
        holePull.addEventListener("input", () => {
            s.holePull = holePullFromSlider(Number(holePull.value));
            this.onHolePull(s.holePull);
            this.sync();
        });

        const earthGravity = document.getElementById("earth-gravity");
        earthGravity.addEventListener("change", () => {
            s.earthGravity = earthGravity.checked;
            this.onEarthGravity(s.earthGravity);
        });

        const view = document.getElementById("view-mode");
        view.addEventListener("change", () => {
            s.view = Number(view.value);
        });

        // The time of day follows the slider while it is dragged; let go,
        // it gives up the focus so it follows the clock again.
        const clock = document.getElementById("clock");
        clock.addEventListener("input", () => {
            this.onClock(Number(clock.value) / MINUTES_PER_DAY, undefined);
            this.showClock();
        });
        clock.addEventListener("change", () => clock.blur());
        const daySpeed = document.getElementById("day-speed");
        daySpeed.addEventListener("change", () => {
            this.onClock(undefined, Number(daySpeed.value));
            this.showClock();
        });

        document.getElementById("pause").addEventListener("click", () => this.togglePause());
        const step = document.getElementById("step");
        step.innerHTML = ICONS.step;
        step.addEventListener("click", () => this.requestStep());
        const clear = document.getElementById("clear");
        clear.innerHTML = ICONS.clear;
        clear.addEventListener("click", () => this.onClear());

        // Snapshots: saved as a download, named for when; loaded from a file
        // picked, setting the world's settings as a scene does (the pause
        // stays as it is).
        document.getElementById("snapshot-save").addEventListener("click", async () => {
            try {
                const bytes = await this.onSaveSnapshot();
                const now = new Date();
                const two = (n) => String(n).padStart(2, "0");
                const name = `sand-${now.getFullYear()}-${two(now.getMonth() + 1)}-${two(now.getDate())}`
                    + `-${two(now.getHours())}${two(now.getMinutes())}${two(now.getSeconds())}.snap`;
                const link = document.createElement("a");
                link.href = URL.createObjectURL(new Blob([bytes], { type: "application/octet-stream" }));
                link.download = name;
                link.click();
                setTimeout(() => URL.revokeObjectURL(link.href), 1000);
                this.showToast(`Saved ${name} (${(bytes.length / 1e6).toFixed(1)} MB)`);
            } catch (err) {
                this.showToast(`Couldn't save the snapshot: ${err.message}`);
            }
        });
        const file = document.getElementById("snapshot-file");
        document.getElementById("snapshot-load").addEventListener("click", () => file.click());
        file.addEventListener("change", async () => {
            const picked = file.files[0];
            file.value = "";
            if (!picked) {
                return;
            }
            try {
                [s.scale, s.holePull, s.earthGravity] = await this.onLoadSnapshot(new Uint8Array(await picked.arrayBuffer()));
                this.sync();
                this.showToast(`Loaded ${picked.name}`);
            } catch (err) {
                this.showToast(`Couldn't load ${picked.name}: ${err.message}`);
            }
        });

        // Keep focus off clicked buttons so Space doesn't re-trigger them, and
        // off a select once it's set: letters typed afterwards would pick its
        // options by name (C for Coarse, clearing the world) instead of
        // reaching the shortcuts, which ignore keys aimed at a select.
        document.getElementById("panel").addEventListener("click", (e) => {
            const control = e.target.closest("button, input[type=checkbox]");
            control?.blur();
        });
        for (const select of document.querySelectorAll("#panel select")) {
            select.addEventListener("change", () => select.blur());
        }
    }

    bindPointer() {
        const c = this.canvas;
        const p = this.pointer;
        // Drawing takes the focus from whatever control had it (preventing
        // the pointer's default below would otherwise leave it there).
        c.tabIndex = -1;
        c.addEventListener("pointerdown", (e) => {
            if (e.button === 1) {
                // The middle button picks up the element under the pointer.
                e.preventDefault();
                this.pickHovered();
                return;
            }
            if (e.button !== 0 && e.button !== 2) {
                return;
            }
            c.focus({ preventScroll: true });
            e.preventDefault();
            this.updatePointer(e);
            const tool = e.button === 2 ? "erase" : this.state.tool;
            // A person, one a click, in the world (a click in the margin
            // round it places nobody), their body (three cells wide) all in
            // it; a fill, one a click.
            if (tool === "person") {
                const { width, height } = this.gridSize();
                if (p.x >= 0 && p.x < width && p.y >= 0 && p.y < height) {
                    this.queue.push({ person: true, x: Math.min(Math.max(p.x, 1.5), width - 1.5), y: p.y + PERSON_DROP });
                    this.noteStroke(e.button);
                }
                return;
            }
            if (SHAPED.has(tool) && this.state.shape === "fill") {
                this.queue.push({ ...this.makeBrush(e.button, p.x, p.y, p.x, p.y, true), fill: true });
                this.noteStroke(e.button);
                return;
            }
            c.setPointerCapture(e.pointerId);
            p.down = true;
            p.button = e.button;
            p.box = SHAPED.has(tool) && this.state.shape === "box";
            p.line = !p.box && e.shiftKey;
            p.startX = p.lastX = this.aim.x = p.x;
            p.startY = p.lastY = this.aim.y = p.y;
        });
        c.addEventListener("pointermove", (e) => this.updatePointer(e));
        const end = () => {
            if (!p.down) {
                return;
            }
            if (p.box) {
                this.queue.push(this.makeBox(p.button, p.startX, p.startY, p.x, p.y));
            } else if (p.line) {
                this.queue.push(this.makeBrush(p.button, p.startX, p.startY, p.x, p.y, true));
            } else {
                this.queue.push(this.makeBrush(p.button, p.lastX, p.lastY, p.x, p.y, false));
            }
            this.noteStroke(p.button);
            p.down = false;
            p.line = false;
            p.box = false;
        };
        c.addEventListener("pointerup", end);
        c.addEventListener("pointercancel", end);
        c.addEventListener("pointerleave", () => {
            p.inside = false;
            this.hoverEl.textContent = "";
        });
        c.addEventListener("contextmenu", (e) => e.preventDefault());
        c.addEventListener("wheel", (e) => {
            e.preventDefault();
            // A sideways swipe on a trackpad scrolls in x only.
            if (e.deltaY === 0) {
                return;
            }
            this.setSize(this.state.size + (e.deltaY < 0 ? 1 : -1));
        }, { passive: false });
    }

    bindKeys() {
        window.addEventListener("keydown", (e) => {
            // (AltGr, which some layouts type [ and ] with, reports Ctrl and
            // Alt too: its keys are the characters they type.)
            if ((e.ctrlKey || e.metaKey || e.altKey) && !e.getModifierState("AltGraph")) {
                return;
            }
            // Typing in the search box is just typing.
            if (e.target instanceof HTMLInputElement && (e.target.type === "search" || e.target.type === "text")) {
                return;
            }
            // A focused select keeps the arrow keys, Enter and Space, which
            // work it, but not letters: those would pick its options by name
            // (C for Coarse, which clears the world). They're shortcuts, or
            // nothing.
            if (e.target instanceof HTMLSelectElement) {
                if (e.key.length !== 1 || e.key === " ") {
                    return;
                }
                e.preventDefault();
            }
            const key = e.key.toLowerCase();
            const tool = TOOLS.find((t) => t.key === key);
            const shape = SHAPES.find((t) => t.key === key);
            const s = this.state;
            if (tool) {
                this.setTool(tool.id);
            } else if (shape) {
                this.setShape(shape.id);
            } else if (key === " ") {
                this.togglePause();
            } else if (key === "n") {
                this.requestStep();
            } else if (key === "t") {
                s.view = s.view === VIEW.HEAT ? VIEW.NORMAL : VIEW.HEAT;
            } else if (key === "p") {
                s.view = s.view === VIEW.PRESSURE ? VIEW.NORMAL : VIEW.PRESSURE;
            } else if (key === "o") {
                s.view = s.view === VIEW.OXYGEN ? VIEW.NORMAL : VIEW.OXYGEN;
            } else if (key === "m") {
                s.view = s.view === VIEW.HUMIDITY ? VIEW.NORMAL : VIEW.HUMIDITY;
            } else if (key === "l") {
                s.view = s.view === VIEW.LIT ? VIEW.NORMAL : VIEW.LIT;
            } else if (key === "f") {
                s.view = s.view === VIEW.GRAVITY ? VIEW.NORMAL : VIEW.GRAVITY;
            } else if (key === "r") {
                s.over = s.over === "empty" ? "all" : "empty";
            } else if (key === "i") {
                this.pickHovered();
            } else if (key === "/") {
                document.getElementById("search").focus();
            } else if (key === "g") {
                s.perf = !s.perf;
                this.perfEl.textContent = "measuring…";
            } else if (key === "[") {
                this.setSize(s.size - 1);
            } else if (key === "]") {
                this.setSize(s.size + 1);
            } else {
                return;
            }
            e.preventDefault();
            this.sync();
        });
    }

    // A stroke finished (or a fill, or a person placed), counted by its tool.
    noteStroke(button) {
        const tool = button === 2 ? "erase" : this.state.tool;
        this.counts[tool]++;
        this.lastStroke = { tool, element: this.state.element };
    }

    updatePointer(e) {
        const p = this.pointer;
        p.clientX = e.clientX;
        p.clientY = e.clientY;
        const pos = this.toCell(e.clientX, e.clientY);
        p.x = pos.x;
        p.y = pos.y;
        p.inside = true;
        const dx = p.x - this.aim.x;
        const dy = p.y - this.aim.y;
        if (p.down && Math.hypot(dx, dy) >= AIM_DISTANCE) {
            this.aim = { x: p.x, y: p.y, angle: Math.atan2(dy, dx) };
        }
    }

    setTool(id) {
        this.state.tool = id;
        this.sync();
    }

    setShape(id) {
        this.state.shape = id;
        if (!SHAPED.has(this.state.tool)) {
            this.state.tool = "draw";
        }
        this.sync();
    }

    setElement(index) {
        this.state.element = index;
        this.setTool("draw");
    }

    // The eyedropper: the element under the pointer, as last read back, to
    // draw with.
    pickHovered() {
        if (this.hoverType != null && ELEMENTS[this.hoverType]) {
            this.setElement(this.hoverType);
        }
    }

    setSize(size) {
        this.state.size = Math.max(MIN_SIZE, Math.min(MAX_SIZE, Math.round(size)));
        this.sync();
    }

    togglePause() {
        this.state.paused = !this.state.paused;
        this.sync();
    }

    requestStep() {
        this.state.paused = true;
        this.stepRequested = true;
        this.sync();
    }

    takeStep() {
        const step = this.stepRequested;
        this.stepRequested = false;
        return step;
    }

    // Reflects state into the DOM.
    sync() {
        const s = this.state;
        const e = ELEMENTS[s.element];
        const tool = TOOLS.find((t) => t.id === s.tool);
        for (const el of document.querySelectorAll(".tool")) {
            el.classList.toggle("active", el.dataset.tool === s.tool);
        }
        for (const el of document.querySelectorAll(".shape")) {
            el.classList.toggle("active", el.dataset.shape === s.shape);
        }
        for (const el of document.querySelectorAll(".element")) {
            el.classList.toggle("active", Number(el.dataset.element) === s.element);
        }
        // What the brush card says it does, and which options apply.
        const drawing = s.tool === "draw";
        const swatch = document.getElementById("brush-swatch");
        swatch.style.setProperty("--swatch", drawing ? cssColor(e.color) : "transparent");
        swatch.innerHTML = drawing ? "" : ICONS[s.tool];
        document.getElementById("brush-name").textContent = drawing ? e.name : tool.label;
        document.getElementById("brush-note").textContent = drawing ? describe(e) : tool.hint;
        const shaped = SHAPED.has(s.tool);
        document.getElementById("shape-row").hidden = !shaped;
        document.getElementById("size-row").hidden = s.tool === "person" || (shaped && (s.shape === "box" || s.shape === "fill"));
        document.getElementById("density-row").hidden = !DENSITY_TOOLS.has(s.tool) || s.shape === "fill";
        document.getElementById("temp-row").hidden = !drawing;
        document.getElementById("over-row").hidden = !shaped || s.shape === "fill";
        document.getElementById("over-element").hidden = !shaped || s.shape === "fill" || s.over !== "only";
        // Only drawing paints into empty space; the rest go over anything.
        const over = drawing || s.over !== "empty" ? s.over : "all";
        for (const el of document.querySelectorAll(".over")) {
            el.classList.toggle("active", el.dataset.over === over);
            el.hidden = el.dataset.over === "empty" && !drawing;
        }
        document.getElementById("over-element").value = String(s.overElement);
        document.getElementById("size").value = String(s.size);
        document.getElementById("size-out").textContent = String(s.size);
        const density = s.density ?? paintDensity(e);
        document.getElementById("density").value = String(Math.round(density * 100));
        document.getElementById("density-out").textContent = s.density == null ? "auto" : `${Math.round(density * 100)}%`;
        const temp = s.temp ?? e.spawnT ?? 20;
        document.getElementById("temp").value = String(temp);
        document.getElementById("temp-out").textContent = s.temp == null ? "own" : `${Math.round(temp)} °C`;

        document.getElementById("view-mode").value = String(s.view);
        document.getElementById("scale").value = String(sliderFromScale(s.scale));
        document.getElementById("scale-out").textContent = formatLength(s.scale);
        document.getElementById("hole-pull").value = String(sliderFromHolePull(s.holePull));
        document.getElementById("hole-pull-out").textContent = `${+s.holePull.toPrecision(2)} g`;
        document.getElementById("earth-gravity").checked = s.earthGravity;
        const pause = document.getElementById("pause");
        pause.innerHTML = s.paused ? ICONS.play : ICONS.pause;
        pause.title = s.paused ? "Play (Space)" : "Pause (Space)";
        pause.setAttribute("aria-label", s.paused ? "Play" : "Pause");
        pause.classList.toggle("active", s.paused);
        this.perfEl.hidden = !s.perf;
        this.showClock();
    }

    // Shows the time of day and the clock's speed, as they change (the
    // slider is left alone while it is being dragged).
    showClock() {
        const { day, speed } = this.clock();
        const minute = Math.floor(day * MINUTES_PER_DAY) % MINUTES_PER_DAY;
        if (this.clockShown?.minute === minute && this.clockShown?.speed === speed) {
            return;
        }
        this.clockShown = { minute, speed };
        const slider = document.getElementById("clock");
        if (document.activeElement !== slider) {
            slider.value = String(minute);
        }
        document.getElementById("clock-out").textContent = formatClock(day);
        document.getElementById("day-speed").value = String(speed);
    }

    radius() {
        return this.state.size - 0.5;
    }

    // The brush for a stroke segment from a to b (a straight line, a box or
    // a fill: isWhole, painted once at full density unless set otherwise).
    makeBrush(button, ax, ay, bx, by, isWhole) {
        const s = this.state;
        const tool = button === 2 ? "erase" : s.tool;
        const shape = SHAPED.has(tool) ? s.shape : "round";
        const brush = {
            ax, ay, bx, by,
            radius: this.radius(),
            shape: { square: SHAPE.SQUARE, spray: SHAPE.SPRAY }[shape] ?? SHAPE.CAPSULE,
            mode: BRUSH.PAINT,
            elem: ID.AIR,
            replace: true,
            strength: s.density ?? 1,
            angle: this.aim.angle,
            erasePeople: tool === "erase",
        };
        if (SHAPED.has(tool) && s.over === "only") {
            brush.target = s.overElement;
        }
        switch (tool) {
            case "draw": {
                const e = ELEMENTS[s.element];
                brush.elem = e.index;
                brush.replace = s.over !== "empty";
                brush.strength = s.density ?? (isWhole || shape === "spray" ? 1 : paintDensity(e));
                if (s.temp != null) {
                    brush.temp = s.temp;
                }
                break;
            }
            case "heat":
                brush.mode = BRUSH.HEAT;
                brush.strength = TEMP_RATE;
                break;
            case "cool":
                brush.mode = BRUSH.COOL;
                brush.strength = TEMP_RATE;
                break;
            case "spark":
                brush.mode = BRUSH.SPARK;
                break;
            case "wind":
                brush.mode = BRUSH.WIND;
                brush.strength = WIND_GAIN;
                break;
            default:
                break;
        }
        return brush;
    }

    // A box from one corner to the other: the cells whose middles are in it.
    makeBox(button, ax, ay, bx, by) {
        const brush = this.makeBrush(button, Math.min(ax, bx), Math.min(ay, by), Math.max(ax, bx), Math.max(ay, by), true);
        return { ...brush, shape: SHAPE.RECT, radius: 0, box: true };
    }

    // Whether a stroke of the draw tool is going on: what it paints stays
    // where it is until the pointer is let go (sim.holding).
    drawing() {
        const p = this.pointer;
        return p.down && p.button === 0 && this.state.tool === "draw";
    }

    // Stroke segments to apply this frame. A held brush keeps applying even
    // when the pointer is still, so powders pile up and heat accumulates.
    takeBrushes() {
        const p = this.pointer;
        const brushes = this.queue;
        this.queue = [];
        if (p.down && !p.line && !p.box) {
            brushes.push(this.makeBrush(p.button, p.lastX, p.lastY, p.x, p.y, false));
            p.lastX = p.x;
            p.lastY = p.y;
        }
        return brushes;
    }

    // The brush's outline for the renderer (render.wgsl's CURSOR_*).
    cursor() {
        const p = this.pointer;
        const s = this.state;
        const tool = p.down && p.button === 2 ? "erase" : s.tool;
        const out = { mouseX: p.x, mouseY: p.y, lineX: p.startX, lineY: p.startY, radius: this.radius(), cursor: 0 };
        if (!p.inside && !p.down) {
            return out;
        }
        if (tool === "person") {
            // Where the person would stand.
            return { ...out, mouseX: p.x + 2.5, mouseY: p.y + PERSON_DROP, lineX: p.x - 2.5, lineY: p.y - PERSON_DROP, cursor: CURSOR.BOX };
        }
        const shape = SHAPED.has(tool) ? s.shape : "round";
        if (shape === "fill") {
            return { ...out, cursor: CURSOR.FILL };
        }
        if (shape === "box") {
            // Dragging, the box; before, a cross where it starts.
            return p.down ? { ...out, cursor: CURSOR.BOX } : { ...out, cursor: CURSOR.FILL };
        }
        const kind = { square: CURSOR.SQUARE, spray: CURSOR.SPRAY }[shape] ?? CURSOR.ROUND;
        return { ...out, cursor: kind | (p.down && p.line ? CURSOR.LINE : 0) };
    }

    // The world or the canvas changed size: the pointer is over another cell
    // now, though it hasn't moved. A stroke in progress goes on from there.
    repoint() {
        const p = this.pointer;
        if (!p.inside && !p.down) {
            return;
        }
        const pos = this.toCell(p.clientX, p.clientY);
        p.x = p.lastX = pos.x;
        p.y = p.lastY = pos.y;
        this.aim.x = p.x;
        this.aim.y = p.y;
    }

    hoverCell() {
        const p = this.pointer;
        if (!p.inside) {
            return null;
        }
        const { width, height } = this.gridSize();
        const x = Math.floor(p.x);
        const y = Math.floor(p.y);
        if (x < 0 || y < 0 || x >= width || y >= height) {
            // Over the border around the world.
            if (this.hoverEl.textContent) {
                this.hoverEl.textContent = "";
            }
            this.hoverType = null;
            return null;
        }
        return { x, y };
    }

    // The hovered cell, or the person there (sim.js's personAt) if anyone.
    showHover(cell, person = null) {
        const e = ELEMENTS[cell.type];
        if (!e || !this.pointer.inside) {
            return;
        }
        this.hoverType = cell.type;
        if (person) {
            this.hoverEl.textContent = describePerson(person);
            return;
        }
        // What its aux and life hold, by its class (js/cellrecord.js).
        const record = cellRecord(cell);
        let name = e.name;
        if (record.copies) {
            name += ` of ${ELEMENTS[record.copies]?.name.toLowerCase() ?? "?"}`;
        }
        // Water carrying a solute (255 saturated) goes by its preset's name
        // ("Salt water", "Acid"), and the solute moves its phase points
        // (phaseOf in common.wgsl).
        const solute = record.solute ? ELEMENTS[record.solute] : null;
        const strength = solute ? record.strength / 255 : 0;
        if (solute) {
            name = ELEMENTS.find((p) => p.paintSolute === solute.id)?.name ?? `${solute.name} water`;
        }
        // Past a phase change the stored temperature measures the latent heat
        // taken in or given off so far; the cell itself sits at the transition
        // point, or past all of its latentDown, cools on as what it condenses
        // into, gram for gram, or past all of its latentUp, superheated, warms
        // on as itself (effTempIn in common.wgsl). Pressure moves boiling
        // points.
        let [hiT, loT] = transitionsAt(e, cell.pressure);
        if (solute) {
            hiT += solute.solBoil * strength;
            loT += solute.solFreeze * strength;
        }
        const temp = Math.min(Math.max(cell.temp, loT), hiT) + Math.max(cell.temp - hiT - e.upSpan, 0)
            - Math.max(loT - e.downSpan - cell.temp, 0) * e.downScale;
        const tags = [`${temp.toFixed(0)} °C`];
        if (solute) {
            tags.push(`${Math.round(strength * 100)}% ${solute.solubility > 0 ? "saturated" : "strength"}`);
            if (Math.abs(loT - e.loT) >= 1) {
                tags.push(`freezes at ${loT.toFixed(0)} °C`);
            }
        }
        if (e.hiTo && cell.temp > hiT) {
            const done = Math.min((cell.temp - hiT) * e.heatCap / e.latentUp, 1);
            tags.push(`→ ${ELEMENTS[ID[e.hiTo]].name.toLowerCase()} ${Math.round(done * 100)}%`);
        } else if (e.loTo && cell.temp < loT) {
            const done = Math.min((loT - cell.temp) * e.heatCap / e.latentDown, 1);
            tags.push(`→ ${ELEMENTS[ID[e.loTo]].name.toLowerCase()} ${Math.round(done * 100)}%`);
        }
        // A boiling point the pressure (or a solute) has moved by a degree or more.
        if (e.boilShift > 0 && Math.abs(hiT - e.hiT) >= 1) {
            tags.push(`boils at ${hiT.toFixed(0)} °C`);
        }
        tags.push(`pressure ${cell.pressure.toFixed(1)}`);
        // Altitude, in a world big enough for it to matter (outdoorT in
        // common.wgsl).
        const { height } = this.gridSize();
        if (height * this.state.scale >= 100) {
            tags.push(`${Math.round((height - cell.y - 0.5) * this.state.scale)} m up`);
        }
        // Air, flames and smoke give what oxygen the air they hold has left
        // to what burns; every gas holds vapour and mist (the gas plane in
        // common.wgsl).
        if (e.id === "AIR" || e.id === "FIRE" || e.id === "SMOKE") {
            tags.push(`oxygen ${Math.round(cell.oxygen * 100)}%`);
        }
        // The water soil holds, 255 soaked (SOIL_FULL in common.wgsl).
        if (record.soilWater > 0) {
            tags.push(`${record.soilWater > 200 ? "wet" : "damp"} ${Math.round((record.soilWater / 255) * 100)}%`);
        }
        if (e.kind === KIND.GAS) {
            tags.push(`humidity ${Math.round(cell.vapour / saturation(temp) * 100)}%`);
            if (cell.mist >= 0.5) {
                tags.push("misty");
            }
        }
        const speed = Math.hypot(cell.vx, cell.vy);
        if (speed >= 0.05) {
            tags.push(`speed ${speed.toFixed(1)}`);
        }
        if (cell.flags & CELL_FLAGS.BURN) {
            tags.push("burning");
        }
        // A plant's water and sugar (update.wgsl's plantNeeds).
        if ("sugar" in record) {
            tags.push(`water ${Math.round((record.plantWater / 255) * 100)}%`, `sugar ${record.sugar}`);
        }
        // What of a fuel has burnt, to 255 (pairs.wgsl).
        if (record.burnt > 0) {
            tags.push(`burnt ${Math.round((record.burnt / 255) * 100)}%`);
        }
        if (record.charge >= CHARGE_ACTIVE) {
            tags.push("charged");
        }
        // Standing structure: how near its stress is to its strength.
        if (e.tensile > 0 && !(cell.flags & CELL_FLAGS.LOOSE) && cell.stress >= 0.01) {
            tags.push(`stressed ${Math.round(cell.stress * 100)}%`);
        }
        // Potential, and the current through the cell (its density across a
        // cell's width: a one-cell wire's whole current).
        // (A dry soil conducts nothing: elements.js SOILS.)
        if (cell.electric && e.sigma > 0 && record.soilWater !== 0) {
            tags.push(`${cell.electric.volts.toFixed(2)} V`);
            if (cell.electric.amps >= MIN_AMPS) {
                tags.push(formatAmps(cell.electric.amps));
            }
        }
        this.hoverEl.textContent = `${name} · ${tags.join(" · ")}`;
    }
}
