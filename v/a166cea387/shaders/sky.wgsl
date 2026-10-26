// Skyline: for each column, the row of the first cell that isn't gas, counting
// down from the top of the world (the world's height if there is none); the
// steps no longer read it (a surface's radiation to the sky is radiate.wgsl's),
// the harness's skyColumns does. In the high 16 bits, the row where the open air ends,
// looking through foliage too, which air blows through: gas above it is
// outdoors and mixes with the outside air's heat and moisture, and gas below
// it is sheltered, in a room or a cave. Of the open air, only what the wind
// gets down to takes the outside air's oxygen (ventilation).
//
// Passes, in sim.js order: rooms and roomUnite (the gas's rooms, labelled
// and joined; again with WET for the wet rooms), at the air phase's head;
// then in the update phase main (the skyline), and ventilation.

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> elems: array<Elem>;
@group(0) @binding(2) var<storage, read> cells: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> sky: array<atomic<u32>>;
// The liquid's grid (fluid.wgsl's field), for the gaps it counts as liquid
// (common.wgsl's flowsThrough).
@group(0) @binding(4) var<storage, read> liquid: array<vec4<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.size.x) {
        return;
    }
    var y = 0u;
    while (y < P.size.y && elems[wordType(cells[y * P.size.x + gid.x].x)].kind == KIND_GAS) {
        y++;
    }
    let surface = y;
    loop {
        if (y >= P.size.y) {
            break;
        }
        let t = wordType(cells[y * P.size.x + gid.x].x);
        if (t != PLANT && elems[t].kind != KIND_GAS) {
            break;
        }
        y++;
    }
    atomicStore(&sky[gid.x], surface | (y << 16u));
}

// Ventilation, every ROOM_EVERY steps (sim.js), after the skyline: how far
// down each column's open air the wind gets, mixing it with the outside air
// (its oxygen, and hydrogen escaping, update.wgsl's ventilated). Not the
// bottom of a deep hollow open to the sky: wind over a gap between walls
// deeper than about 0.7 of its width skims over it, turning the air in it
// over in a slow eddy of its own that barely trades with the air above
// (street canyons' "skimming flow": Oke 1988, Energy and Buildings 11,
// 103-113), so a roofless room's air, or a pit's, is left as it was, to be
// made good from above only as fast as it spreads. (Its moisture still mixes
// with the outside air's: that is what keeps what the wind makes of it in
// check outdoors, update.wgsl's moistAir.) A hollow is what would hold water
// poured over the
// world, up to the level it spills over its lower side: the lower of the
// highest ground to its left and to its right (in rows, the open air's end,
// the skyline's high half: the greater of the least to the left and to the
// right). The wind gets to all the air above that level, and in the hollow
// as far as an upside-down cone, its sides WIND_REACH rows down a column
// across, can be let down into it from above (the open air's end closed by
// that cone, a morphological closing): it reaches the middle of a hollow
// WIND_REACH / 2 of its width down, so all of one less than 0.7 times as
// deep as it is wide but for its bottom corners, within 0.7 of their height
// of its walls (their eddies), and none of a narrow shaft below that. (A
// wall or a tree on open ground shelters nothing.) The dilation and erosion
// each take a running minimum or maximum from the left and from the right,
// as scans over the columns, the hollows' level too: a thread's run of
// columns in turn, over the threads' totals in workgroup memory, then the
// run again.
const WIND_REACH: f32 = 1.4;
const VENT_THREADS: u32 = 256u;
const VENT_FAR: f32 = 1e9;

var<workgroup> ventLeft: array<vec2<f32>, VENT_THREADS>;
var<workgroup> ventRight: array<vec2<f32>, VENT_THREADS>;

// From each thread's totals, the running minimum (sign 1) or maximum (-1) of
// all threads' to its left (ventLeft) and to its right (ventRight), its own
// included: Hillis-Steele.
fn ventScan(li: u32, sign: f32) {
    for (var s = 1u; s < VENT_THREADS; s <<= 1u) {
        var left = sign * ventLeft[li];
        var right = sign * ventRight[li];
        if (li >= s) {
            left = min(left, sign * ventLeft[li - s]);
        }
        if (li + s < VENT_THREADS) {
            right = min(right, sign * ventRight[li + s]);
        }
        workgroupBarrier();
        ventLeft[li] = sign * left;
        ventRight[li] = sign * right;
        workgroupBarrier();
    }
}

@compute @workgroup_size(VENT_THREADS)
fn ventilation(@builtin(local_invocation_index) li: u32) {
    let w = P.size.x;
    let run = (w + VENT_THREADS - 1u) / VENT_THREADS;
    let x0 = min(li * run, w);
    let x1 = min(x0 + run, w);

    // Dilation (x): the cone set on the ground beside each column, the
    // highest it is anywhere (rows count down, so the least): min over i of
    // o[i] + k |x - i|, as k x + min over i <= x of (o[i] - k i), and -k x +
    // min over i >= x of (o[i] + k i). And the hollows' level (y): the
    // greater of min over i <= x of o[i] and min over i >= x of o[i].
    var left = vec2<f32>(VENT_FAR);
    var right = vec2<f32>(VENT_FAR);
    for (var x = x0; x < x1; x++) {
        let o = f32(atomicLoad(&sky[x]) >> 16u);
        left = min(left, vec2<f32>(o - WIND_REACH * f32(x), o));
        right = min(right, vec2<f32>(o + WIND_REACH * f32(x), o));
    }
    ventLeft[li] = left;
    ventRight[li] = right;
    workgroupBarrier();
    ventScan(li, 1.0);
    left = select(vec2<f32>(VENT_FAR), ventLeft[max(li, 1u) - 1u], li > 0u);
    right = select(vec2<f32>(VENT_FAR), ventRight[min(li + 1u, VENT_THREADS - 1u)], li + 1u < VENT_THREADS);
    workgroupBarrier();
    // From the left, into the working words and the ventilation's own.
    for (var x = x0; x < x1; x++) {
        let o = f32(atomicLoad(&sky[x]) >> 16u);
        left = min(left, vec2<f32>(o - WIND_REACH * f32(x), o));
        atomicStore(&sky[ventilationAt(P.size, w + x)], bitcast<u32>(WIND_REACH * f32(x) + left.x));
        atomicStore(&sky[ventilationAt(P.size, x)], bitcast<u32>(left.y));
    }
    // From the right, and the erosion's totals as it goes: max over its run
    // of d[j] + k j and of d[j] - k j.
    var rise = -VENT_FAR;
    var fall = -VENT_FAR;
    for (var x = x1; x > x0; x--) {
        let o = f32(atomicLoad(&sky[x - 1u]) >> 16u);
        right = min(right, vec2<f32>(o + WIND_REACH * f32(x - 1u), o));
        let at = ventilationAt(P.size, w + x - 1u);
        let d = min(bitcast<f32>(atomicLoad(&sky[at])), right.x - WIND_REACH * f32(x - 1u));
        atomicStore(&sky[at], bitcast<u32>(d));
        let level = ventilationAt(P.size, x - 1u);
        atomicStore(&sky[level], bitcast<u32>(max(bitcast<f32>(atomicLoad(&sky[level])), right.y)));
        rise = max(rise, d + WIND_REACH * f32(x - 1u));
        fall = max(fall, d - WIND_REACH * f32(x - 1u));
    }

    // Erosion: the cone let down from above as far as the dilated ground
    // lets it, the lowest it gets anywhere over each column (the most): max
    // over j of d[j] - k |x - j|, as -k x + max over j <= x of (d[j] + k j),
    // and k x + max over j >= x of (d[j] - k j); or the hollows' level, if
    // lower.
    ventLeft[li] = vec2<f32>(rise, 0.0);
    ventRight[li] = vec2<f32>(fall, 0.0);
    workgroupBarrier();
    ventScan(li, -1.0);
    rise = select(-VENT_FAR, ventLeft[max(li, 1u) - 1u].x, li > 0u);
    fall = select(-VENT_FAR, ventRight[min(li + 1u, VENT_THREADS - 1u)].x, li + 1u < VENT_THREADS);
    for (var x = x0; x < x1; x++) {
        let d = bitcast<f32>(atomicLoad(&sky[ventilationAt(P.size, w + x)]));
        rise = max(rise, d + WIND_REACH * f32(x));
        let at = ventilationAt(P.size, x);
        atomicStore(&sky[at], bitcast<u32>(max(bitcast<f32>(atomicLoad(&sky[at])), rise - WIND_REACH * f32(x))));
    }
    for (var x = x1; x > x0; x--) {
        let d = bitcast<f32>(atomicLoad(&sky[ventilationAt(P.size, w + x - 1u)]));
        fall = max(fall, d - WIND_REACH * f32(x - 1u));
        let at = ventilationAt(P.size, x - 1u);
        let reach = max(bitcast<f32>(atomicLoad(&sky[at])), fall + WIND_REACH * f32(x - 1u));
        // Rows above this one are in the wind; never below the open air's end.
        let open = atomicLoad(&sky[x - 1u]) >> 16u;
        atomicStore(&sky[at], min(u32(max(ceil(reach), 0.0)), open));
    }
}

// Rooms (see ROOM_TILE in common.wgsl), every step before the air's (sim.js),
// after the skyline, in two passes: rooms, then roomUnite. (The air pass
// follows each piece's links to its room's first piece itself.)
//
// rooms: a workgroup per tile, each thread four cells along a row. The
// tile's cells of gas are joined up in workgroup memory, as
// bodies.wgsl's init joins a tile's loose cells: each takes the lowest
// cell of its piece (by index in the tile) as its label. The pieces are
// numbered the tile's biggest first, the rest in the order of their lowest
// cells, up to ROOM_PIECES; any more are UNNUMBERED (pockets of air between
// the grains of a sand cloud, which give out nothing to come to a pressure
// of a zone they have none of, air.wgsl). Each piece starts out joined to
// itself.
//
// With WET, the same over the gas, the liquid and powders, the wet rooms
// (common.wgsl's wetWord): the gas the liquid closes in, or a powder's
// grains, is in one with the gas over it (REWRITE_PLAN.md
// `bubble-pressure`).
const ROOM_CELLS: u32 = ROOM_TILE * ROOM_TILE;
const NO_CELL: u32 = 0xffffffffu;
override WET: bool = false;

// The sky's word holding cell p's byte, and piece k's link, of the rooms or
// (WET) the wet rooms.
fn labelWord(p: vec2<u32>) -> u32 {
    return select(roomWord(P.size, p), wetWord(P.size, p), WET);
}

fn labelLink(k: u32) -> u32 {
    return select(roomLink(P.size, k), wetLink(P.size, k), WET);
}

// Whether cell i is in a room: what the air's flow passes through
// (common.wgsl's flowsThrough: gas, a fan, a black hole), so that a zone is
// a piece of the flow's own, the open air too; and with WET the liquid as
// well, the gaps it counts as its own, and powders.
fn inRoom(i: u32) -> bool {
    let t = wordType(cells[i].x);
    let kind = elems[t].kind;
    return flowsThrough(t, liquidSide(liquid[2u * i], liquid[2u * i + 1u])) || (WET && (kind == KIND_LIQUID || kind == KIND_GAS || kind == KIND_POWDER));
}

var<workgroup> roomParent: array<atomic<u32>, ROOM_CELLS>;  // NO_CELL where not in a room
var<workgroup> roomSize: array<atomic<u32>, ROOM_CELLS>;    // cells of the piece whose lowest cell this is
var<workgroup> roomRoots: array<atomic<u32>, ROOM_CELLS / 32u>;  // a bit for each piece's lowest cell
var<workgroup> roomBefore: array<u32, ROOM_CELLS / 32u>;  // pieces' lowest cells in the words before
var<workgroup> roomBiggest: atomic<u32>;  // the biggest piece: its size, then 1023 less its lowest cell
var<workgroup> roomAny: atomic<u32>;
var<workgroup> roomAnyShared: u32;

// The union-find's links (labeltile.wgsl, label.wgsl): the tile's cells'
// in workgroup memory, the pieces' in the sky.
fn ufTile(i: u32) -> u32 {
    return atomicLoad(&roomParent[i]);
}

fn ufTileLower(i: u32, to: u32) -> u32 {
    return atomicMin(&roomParent[i], to);
}

fn ufLink(at: u32) -> u32 {
    return atomicLoad(&sky[at]);
}

fn ufLower(at: u32, to: u32) -> u32 {
    return atomicMin(&sky[at], to);
}

@compute @workgroup_size(256)
fn rooms(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let row = li / (ROOM_TILE / 4u);
    let col = (li % (ROOM_TILE / 4u)) * 4u;
    let origin = wid.xy * ROOM_TILE + vec2<u32>(col, row);
    let first = row * ROOM_TILE + col;
    var mine = 0u;
    for (var k = 0u; k < 4u; k++) {
        let p = origin + vec2<u32>(k, 0u);
        var room = false;
        if (all(p < P.size)) {
            room = inRoom(p.y * P.size.x + p.x);
        }
        atomicStore(&roomParent[first + k], select(NO_CELL, first + k, room));
        atomicStore(&roomSize[first + k], 0u);
        mine |= u32(room);
    }
    if (li < ROOM_CELLS / 32u) {
        atomicStore(&roomRoots[li], 0u);
    }
    if (li < ROOM_PIECES) {
        let k = (wid.y * roomTiles(P.size).x + wid.x) * ROOM_PIECES + li;
        atomicStore(&sky[labelLink(k)], k);
    }
    if (mine != 0u) {
        atomicStore(&roomAny, 1u);
    }
    workgroupBarrier();
    if (li == 0u) {
        roomAnyShared = atomicLoad(&roomAny);
    }
    // Most tiles (out in the open, or solid) have no rooms at all.
    if (workgroupUniformLoad(&roomAnyShared) == 0u) {
        if (all(origin < P.size)) {
            atomicStore(&sky[labelWord(origin)], 0xffffffffu);
        }
        return;
    }
    // Right and down, inside the tile.
    for (var k = 0u; k < 4u; k++) {
        let i = first + k;
        if (atomicLoad(&roomParent[i]) == NO_CELL) {
            continue;
        }
        if (col + k + 1u < ROOM_TILE && atomicLoad(&roomParent[i + 1u]) != NO_CELL) {
            ufTileJoin(i, i + 1u);
        }
        if (row + 1u < ROOM_TILE && atomicLoad(&roomParent[i + ROOM_TILE]) != NO_CELL) {
            ufTileJoin(i, i + ROOM_TILE);
        }
    }
    workgroupBarrier();
    var roots = vec4<u32>(NO_CELL);
    for (var k = 0u; k < 4u; k++) {
        let i = first + k;
        if (atomicLoad(&roomParent[i]) != NO_CELL) {
            roots[k] = ufTileFind(i);
            atomicAdd(&roomSize[roots[k]], 1u);
            if (roots[k] == i) {
                atomicOr(&roomRoots[i / 32u], 1u << (i % 32u));
            }
        }
    }
    workgroupBarrier();
    // The biggest piece, and how many pieces' lowest cells come before each
    // word's.
    for (var k = 0u; k < 4u; k++) {
        if (roots[k] == first + k) {
            atomicMax(&roomBiggest, (atomicLoad(&roomSize[first + k]) << 10u) | (ROOM_CELLS - 1u - first - k));
        }
    }
    if (li < ROOM_CELLS / 32u) {
        var n = 0u;
        for (var w = 0u; w < li; w++) {
            n += countOneBits(atomicLoad(&roomRoots[w]));
        }
        roomBefore[li] = n;
    }
    workgroupBarrier();
    // A piece's number: 0 for the biggest, the rest by how many pieces'
    // lowest cells come before their own.
    let biggest = ROOM_CELLS - 1u - (atomicLoad(&roomBiggest) & (ROOM_CELLS - 1u));
    var word = 0u;
    for (var k = 0u; k < 4u; k++) {
        var piece = NO_ROOM;
        let r = roots[k];
        if (r != NO_CELL) {
            let n = roomBefore[r / 32u] + countOneBits(atomicLoad(&roomRoots[r / 32u]) & ((1u << (r % 32u)) - 1u));
            piece = min(select(n + select(0u, 1u, biggest > r), 0u, r == biggest), UNNUMBERED);
        }
        word |= piece << (8u * k);
    }
    if (all(origin < P.size)) {
        atomicStore(&sky[labelWord(origin)], word);
    }
}

// The piece cell p (in the world) is in, by number, or NONE_PIECE: for none,
// and for a pocket past the tile's ROOM_PIECES (UNNUMBERED, which as a
// number would be the next tile's first piece, and join it to what it met).
const NONE_PIECE: u32 = 0xffffffffu;

fn pieceOfCell(p: vec2<u32>) -> u32 {
    let piece = (atomicLoad(&sky[labelWord(p)]) >> (8u * (p.x % 4u))) & 0xffu;
    return select(roomPieceAt(P.size, p, piece), NONE_PIECE, piece >= UNNUMBERED);
}

// roomUnite: a workgroup per tile, its first 32 threads down its right
// edge and the rest along its bottom edge: where sheltered air meets the
// next tile's, their pieces are joined (label.wgsl).
@compute @workgroup_size(64)
fn roomUnite(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = li % ROOM_TILE;
    let down = li >= ROOM_TILE;
    let p = wid.xy * ROOM_TILE + select(vec2<u32>(ROOM_TILE - 1u, i), vec2<u32>(i, ROOM_TILE - 1u), down);
    let q = p + select(vec2<u32>(1u, 0u), vec2<u32>(0u, 1u), down);
    if (any(q >= P.size)) {
        return;
    }
    let a = pieceOfCell(p);
    let b = pieceOfCell(q);
    if (a != NONE_PIECE && b != NONE_PIECE) {
        ufJoin(labelLink(0u), a, b);
    }
}
