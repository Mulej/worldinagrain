// One union-find for every labelling (REWRITE_PLAN.md 8-3): the rooms and
// the wet rooms (sky.wgsl, rooms.wgsl), standing structure and loose pieces
// (bodies.wgsl), the electric solve's aggregates (electric.wgsl). A label
// is a link toward a lower index, and an index whose link isn't lower is a
// root: so a link left over from before, or none (0xffffffff), can't make a
// loop. Joining lowers the higher root's link to the lower one with
// atomicMin, and if another thread lowered it first, joins what that one
// linked it to: lock-free (Playne & Hawick 2018, "A New Algorithm for
// Parallel Connected-Component Labelling on GPUs", IEEE TPDS 29(6)).
//
// Only the members of a labelling are joined: what is a member, and what
// joins what (faces only, or corners too; one piece's cells only), is the
// caller's to test before it joins. (8-1's structure joined each of its
// cells to whatever was beside it, and through a cell of air two cells
// that met only at a corner were one piece.)
//
// This part is the union-find over links in a storage buffer, by offset (a
// labelling's links at base + i); labeltile.wgsl the same over a tile's in
// workgroup memory. A module including this one defines the words it keeps
// them in:
//   fn ufLink(at: u32) -> u32                 the link at word at
//   fn ufLower(at: u32, to: u32) -> u32       atomicMin of it with to, the old link

// The root of i, of the links at base + i.
fn ufFind(base: u32, i: u32) -> u32 {
    var x = i;
    loop {
        let up = ufLink(base + x);
        if (up >= x) {
            break;
        }
        x = up;
    }
    return x;
}

fn ufJoin(base: u32, i: u32, j: u32) {
    var a = i;
    var b = j;
    loop {
        a = ufFind(base, a);
        b = ufFind(base, b);
        if (a == b) {
            return;
        }
        let lo = min(a, b);
        let hi = max(a, b);
        let old = ufLower(base + hi, lo);
        if (old == hi) {
            return;
        }
        a = old;
        b = lo;
    }
}
