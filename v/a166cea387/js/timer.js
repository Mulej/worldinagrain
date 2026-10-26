// GPU time per phase of a frame, from timestamp queries. Reading the results
// back never stalls a frame: while one frame's timings are still on their way
// back, the frames after it go untimed.

const QUERY_BYTES = 8;

export class GpuTimer {
    constructor(device, maxPasses) {
        this.maxPasses = maxPasses;
        this.querySet = device.createQuerySet({ label: "timestamps", type: "timestamp", count: maxPasses * 2 });
        const size = maxPasses * 2 * QUERY_BYTES;
        const U = GPUBufferUsage;
        this.resolveBuf = device.createBuffer({ label: "timestamps", size, usage: U.QUERY_RESOLVE | U.COPY_SRC });
        this.readBuf = device.createBuffer({ label: "timestamps-read", size, usage: U.MAP_READ | U.COPY_DST });
        this.passes = [];     // phase of each timed pass in the frame being encoded
        this.timing = false;  // whether the frame being encoded is timed
        this.pending = false; // a readback is in flight
        this.totals = {};
        this.frames = 0;
        this.overrun = false; // said that a frame had more passes than slots
    }

    // Call before encoding a frame.
    begin(enabled) {
        this.timing = Boolean(enabled) && !this.pending;
        this.passes.length = 0;
    }

    // timestampWrites for a pass of the given phase, or undefined if untimed.
    // A frame with more passes than there are slots for times only its first
    // ones, which would leave the later phases short: said once, so the
    // count can be put right (sim.js STEP_PASSES).
    writes(phase) {
        if (!this.timing) {
            return undefined;
        }
        if (this.passes.length >= this.maxPasses) {
            if (!this.overrun) {
                this.overrun = true;
                console.warn(`GPU timer: more than ${this.maxPasses} passes in a frame; ${phase} and later passes go untimed`);
            }
            return undefined;
        }
        const i = this.passes.length;
        this.passes.push(phase);
        return { querySet: this.querySet, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 };
    }

    // Call after encoding the frame's passes, before finishing the encoder.
    resolve(enc) {
        if (!this.timing || this.passes.length === 0) {
            return;
        }
        const count = this.passes.length * 2;
        enc.resolveQuerySet(this.querySet, 0, count, this.resolveBuf, 0);
        enc.copyBufferToBuffer(this.resolveBuf, 0, this.readBuf, 0, count * QUERY_BYTES);
    }

    // Call after submitting the frame.
    collect() {
        if (!this.timing || this.passes.length === 0) {
            return;
        }
        this.pending = true;
        const passes = [...this.passes];
        this.readBuf.mapAsync(GPUMapMode.READ).then(() => {
            const t = new BigUint64Array(this.readBuf.getMappedRange());
            for (let i = 0; i < passes.length; i++) {
                const ns = t[2 * i + 1] > t[2 * i] ? Number(t[2 * i + 1] - t[2 * i]) : 0;
                this.totals[passes[i]] = (this.totals[passes[i]] ?? 0) + ns / 1e6;
            }
            this.frames++;
            this.readBuf.unmap();
            this.pending = false;
        }, () => {
            // The world was replaced and the buffer destroyed mid-flight.
            this.pending = false;
        });
    }

    // Mean milliseconds per timed frame for each phase since the last call.
    take() {
        const ms = {};
        for (const [phase, total] of Object.entries(this.totals)) {
            ms[phase] = total / this.frames;
        }
        const result = { frames: this.frames, ms };
        this.totals = {};
        this.frames = 0;
        return result;
    }

    destroy() {
        this.querySet.destroy();
        this.resolveBuf.destroy();
        this.readBuf.destroy();
    }
}
