// Heat put into cells as energy (REWRITE_PLAN.md stage C, COMPATIBILITY.md
// `heat-in`): a pass that heats a cell other than by trading with its
// neighbours (Joule heat; reactions, stage 5) adds the joules to the cell's
// slot of heatIn, and the next update pass takes them in over the cell's
// heat capacity (and clears the slot), as it takes in what conduction and
// radiation pass it. Nothing else writes a cell's temperature for it, so
// the energy is the same on both sides whatever the cell has become by then
// (a wire that melted into a drop is heated as the drop).
//
// Two words a cell, an exact sum (common.wgsl's exactParts, EXACT_FINE:
// WGSL has no f32 atomics, and an f32 added to by compare-and-swap came out
// otherwise as the threads came). A module including this binds
// `heatIn: array<atomic<u32>>`.

fn depositHeat(i: u32, joules: f32) {
    if (joules == 0.0) {
        return;
    }
    let w = exactParts(joules, EXACT_FINE);
    let old = atomicAdd(&heatIn[2u * i], w.x);
    atomicAdd(&heatIn[2u * i + 1u], w.y + exactCarry(old, w.x));
}
