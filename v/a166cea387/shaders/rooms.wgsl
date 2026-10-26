// The rooms (sky.wgsl), as the air's pressure zones read them (air.wgsl): a
// cell's piece byte and the room a piece is in. A module including this binds `sky`, `cells`, `elems` and `liquid`
// (the liquid's grid), and includes label.wgsl.

fn roomByteAt(p: vec2<u32>) -> u32 {
    return (atomicLoad(&sky[roomWord(P.size, p)]) >> (8u * (p.x % 4u))) & 0xffu;
}

// The union-find's links (label.wgsl): the pieces', in the sky.
fn ufLink(at: u32) -> u32 {
    return atomicLoad(&sky[at]);
}

fn ufLower(at: u32, to: u32) -> u32 {
    return atomicMin(&sky[at], to);
}

// The room piece k (by number) is in: the first piece of those it is joined
// to (sky.wgsl's roomUnite).
fn roomOfPiece(k: u32) -> u32 {
    return ufFind(roomLink(P.size, 0u), k);
}

// The same of the wet rooms (common.wgsl's wetWord: the rooms joined
// through the liquid).
fn wetByteAt(p: vec2<u32>) -> u32 {
    return (atomicLoad(&sky[wetWord(P.size, p)]) >> (8u * (p.x % 4u))) & 0xffu;
}

fn wetOfPiece(k: u32) -> u32 {
    return ufFind(wetLink(P.size, 0u), k);
}

// Whether the cell at p is in the liquid: liquid, or a gap the liquid counts
// as its own (common.wgsl's flowsThrough), as the wet rooms are joined
// through (sky.wgsl).
fn liquidAt(p: vec2<i32>) -> bool {
    if (any(p < vec2<i32>(0)) || any(p >= vec2<i32>(P.size))) {
        return false;
    }
    let i = u32(p.y) * P.size.x + u32(p.x);
    let e = elems[wordType(cells[i].x)];
    return e.kind == KIND_LIQUID || (e.kind == KIND_GAS && liquidSide(liquid[2u * i], liquid[2u * i + 1u]) > 0.0);
}

// Whether the cell at p joins the gas on its sides into one wet room
// (sky.wgsl's rooms with WET): liquid (liquidAt), which moves to even out
// what it closes in, or a powder, whose grains the air passes between: a
// centimetre of sand lets a cell of air through it in a few milliseconds
// at an atmosphere's difference (Darcy's law, k Δp / (μ L), with clean
// sand's permeability 1e-12 to 1e-9 m², Bear, Dynamics of Fluids in Porous
// Media, 1972), so a pocket among the grains is at the air's pressure
// around it. (Through a metre of soil it would take some seconds: not
// modelled.) (Held as their own, vacuums left among falling grains pulled
// a heap of sand together with an atmosphere's push, 4e.)
fn joinsAt(p: vec2<i32>) -> bool {
    if (liquidAt(p)) {
        return true;
    }
    if (any(p < vec2<i32>(0)) || any(p >= vec2<i32>(P.size))) {
        return false;
    }
    return elems[wordType(cells[u32(p.y) * P.size.x + u32(p.x)].x)].kind == KIND_POWDER;
}

// Whether the gas at p meets the liquid or a powder (joinsAt): where a wet
// room's surface takes in what its pockets give out.
fn besideJoin(p: vec2<i32>) -> bool {
    return joinsAt(p + vec2<i32>(1, 0)) || joinsAt(p - vec2<i32>(1, 0)) || joinsAt(p + vec2<i32>(0, 1))
        || joinsAt(p - vec2<i32>(0, 1));
}

// The wet room the gas at p is in (by its first piece's number), or
// NO_WET: none found for it (a pocket past its tile's ROOM_PIECES).
const NO_WET: u32 = 0xffffffffu;

fn wetRoomAt(p: vec2<u32>) -> u32 {
    let piece = wetByteAt(p);
    if (piece >= UNNUMBERED) {
        return NO_WET;
    }
    return wetOfPiece(roomPieceAt(P.size, p, piece));
}

// A wet room's surface (REWRITE_PLAN.md `bubble-pressure`): the zone the
// liquid's free surface faces, at whose pressure the liquid holds the gas it
// closes in. The open air, where the wet room reaches it; else its biggest
// zone beside the liquid (common.wgsl's WET_MOST and WET_SURFACE), a sealed
// tank's air over its water. (The gas of all its zones together, by the gas
// law, counted the cells the water had just left as part of it, a vacuum
// each: the pressure fell, and the air over the water went down into them.)
// Whether the zone of room `room` (its slot z) is wet room slot w's surface.
fn isSurface(w: u32, room: u32, z: u32) -> bool {
    if ((atomicLoad(&sky[z + ZONE_OPEN]) & ZONE_OUTSIDE) != 0u) {
        return true;
    }
    return atomicLoad(&sky[w + WET_OPEN]) == 0u && atomicLoad(&sky[w + WET_SURFACE]) == room + 1u;
}

// The pressure, atm, of wet room slot w's surface at pos (cells), and 1; or
// none and 0 where it has none (yet: WET_SURFACE is found after the air's
// divergence).
fn surfaceAtm(w: u32, pos: vec2<f32>) -> vec2<f32> {
    if (atomicLoad(&sky[w + WET_OPEN]) != 0u) {
        return vec2<f32>(openAirAtm(altitudeAt(pos)), 1.0);
    }
    let surface = atomicLoad(&sky[w + WET_SURFACE]);
    if (surface == 0u) {
        return vec2<f32>(0.0);
    }
    let z = zoneSlotAt(P.size, P.frame, surface - 1u);
    let volume = f32(atomicLoad(&sky[z + ZONE_VOLUME]));
    if (volume <= 0.0) {
        return vec2<f32>(0.0);
    }
    return vec2<f32>(f32(atomicLoad(&sky[z + ZONE_AMOUNT])) / volume, 1.0);
}
