// The guided tour: a card over the world that takes a newcomer through the
// basics a step at a time, pointing at the control each step is about and
// waiting for them to try it (drawing, picking water, heating, erasing, a
// view, pausing, a scene). It starts by itself the first time the app is
// opened, and the ? button or key shows it again.

import { ID } from "./elements.js";
import { VIEW } from "./sim.js";

const SEEN_KEY = "sand.tourSeen";
const DWELL_MS = 1500;  // how long a step's done mark shows before the next

// Each step: its title and text, the control it points at (a selector, none
// for the world itself), what it sets up as it begins (enter(controls)), and
// when it is done: done(controls, start, memory), start being what the
// controls were as the step began (see snapshot) and memory the step's own
// scratch. A step with no done waits for Next.
const STEPS = [
    {
        title: "Welcome to Sand",
        text: "A world of tiny cells where sand piles up, water flows, fire burns and air moves, all by real physics. "
            + "This takes a minute: try each thing as it asks.",
        next: "Start",
    },
    {
        title: "Draw",
        text: "Sand is picked. Drag across the world to pour some.",
        enter: (c) => c.setElement(ID.SAND),
        done: (c, start) => c.counts.draw > start.counts.draw,
    },
    {
        title: "Pick an element",
        target: '.element[data-name="water"]',
        text: "Pick Water in the list, then pour it over your sand. Every element behaves its own way: hover one to read about it.",
        done: (c, start) => c.counts.draw > start.counts.draw && c.lastStroke?.element === ID.WATER,
    },
    {
        title: "The brush",
        target: "#brush",
        text: "This card shows what you're holding. Scroll over the world, or press [ and ], to change the brush's size. "
            + "Shape picks a round, square or spray brush, a box, or a fill.",
        done: (c, start) => c.state.size !== start.size,
    },
    {
        title: "Heat things up",
        target: '.tool[data-tool="heat"]',
        text: "Tools do more than paint. Pick Heat (H) and hold it on the water until it boils away as steam.",
        done: (c, start) => c.counts.heat > start.counts.heat,
    },
    {
        title: "Erase",
        target: '.tool[data-tool="erase"]',
        text: "Drag with the right mouse button to erase anything, whatever tool you hold (or pick Erase, E).",
        done: (c, start) => c.counts.erase > start.counts.erase,
    },
    {
        title: "See the unseen",
        target: "#view-mode",
        text: "The view menu shows what you can't see: heat, pressure, oxygen, moisture, light and gravity. Try Heat (T), "
            + "then go back to Normal.",
        done: (c, start, memory) => {
            memory.looked ||= c.state.view !== VIEW.NORMAL;
            return memory.looked && c.state.view === VIEW.NORMAL;
        },
    },
    {
        title: "Pause and step",
        target: "#pause",
        text: "Space pauses the world so you can look closely, and N moves it on one step at a time. "
            + "Pause it, then press Space again to play.",
        done: (c, start, memory) => {
            memory.paused ||= c.state.paused;
            return memory.paused && !c.state.paused;
        },
    },
    {
        title: "Scenes",
        target: "#scenes",
        text: "Scenes are ready-made experiments: a dam bursting, a fire in a box, a raft in a tank. "
            + "Open a group and pick one; each tells you what to watch for.",
        done: (c, start) => c.counts.scenes > start.counts.scenes,
    },
    {
        title: "That's the tour",
        target: "#tour-open",
        text: "Every key is listed under Shortcuts at the bottom of the panel. Press ? any time to see this again. Have fun!",
        next: "Done",
    },
];

// What the controls were as a step began, for its done to compare with.
function snapshot(controls) {
    return { counts: { ...controls.counts }, size: controls.state.size };
}

export class Tour {
    constructor(controls) {
        this.controls = controls;
        this.card = document.getElementById("tour");
        this.progress = document.getElementById("tour-progress");
        this.stage = document.getElementById("stage");
        this.title = document.getElementById("tour-title");
        this.text = document.getElementById("tour-text");
        this.nextButton = document.getElementById("tour-next");
        this.skipButton = document.getElementById("tour-skip");
        this.index = -1;
        this.start = null;
        this.memory = {};
        this.doneAt = 0;     // when this step was done (0: not yet)
        this.target = null;  // the element pointed at

        this.nextButton.addEventListener("click", () => this.next());
        this.skipButton.addEventListener("click", () => this.close());
        document.getElementById("tour-open").addEventListener("click", () => this.open());
        window.addEventListener("keydown", (e) => {
            if (e.ctrlKey || e.metaKey || e.altKey || (e.target instanceof HTMLInputElement && e.target.type === "search")) {
                return;
            }
            if (e.key === "?") {
                e.preventDefault();
                this.open();
            } else if (e.key === "Escape" && this.isOpen()) {
                e.preventDefault();
                this.close();
            }
        });
    }

    isOpen() {
        return this.index >= 0;
    }

    // Shows the tour the first time the app is opened in this browser.
    openFirstTime() {
        let seen = false;
        try {
            seen = localStorage.getItem(SEEN_KEY) === "1";
        } catch {
            // (No storage, as in some private windows: every time, then.)
        }
        if (!seen) {
            this.open();
        }
    }

    open() {
        this.show(0);
    }

    close() {
        this.point(null);
        this.index = -1;
        this.card.hidden = true;
        this.stage.classList.remove("touring");
        try {
            localStorage.setItem(SEEN_KEY, "1");
        } catch {
            // (See openFirstTime.)
        }
    }

    next() {
        if (this.index + 1 >= STEPS.length) {
            this.close();
        } else {
            this.show(this.index + 1);
        }
    }

    show(index) {
        const step = STEPS[index];
        this.index = index;
        step.enter?.(this.controls);
        this.start = snapshot(this.controls);
        this.memory = {};
        this.doneAt = 0;
        this.card.hidden = false;
        this.stage.classList.add("touring");
        this.card.classList.remove("done");
        this.progress.textContent = index === 0 ? "Tour" : `${index} of ${STEPS.length - 1}`;
        this.title.textContent = step.title;
        this.text.textContent = step.text;
        this.nextButton.textContent = step.next ?? "Next";
        this.skipButton.hidden = index === STEPS.length - 1;
        this.point(step.target ? document.querySelector(step.target) : null);
    }

    // Marks the element a step is about, in view.
    point(el) {
        this.target?.classList.remove("tour-target");
        this.target = el;
        if (el) {
            el.classList.add("tour-target");
            el.scrollIntoView({ block: "nearest", behavior: "smooth" });
        }
    }

    // Each frame: whether the step being shown is done, and on to the next a
    // moment after it is.
    update(now) {
        const step = STEPS[this.index];
        if (!step?.done) {
            return;
        }
        if (!this.doneAt && step.done(this.controls, this.start, this.memory)) {
            this.doneAt = now;
            this.card.classList.add("done");
        }
        if (this.doneAt && now - this.doneAt >= DWELL_MS) {
            this.next();
        }
    }
}
