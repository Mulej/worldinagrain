// The union-find of label.wgsl over a tile's links in workgroup memory, by
// index in the tile. A module including this defines them:
//   fn ufTile(i: u32) -> u32                  i's link
//   fn ufTileLower(i: u32, to: u32) -> u32    atomicMin of it with to, the old link

fn ufTileFind(i: u32) -> u32 {
    var x = i;
    loop {
        let up = ufTile(x);
        if (up >= x) {
            break;
        }
        x = up;
    }
    return x;
}

fn ufTileJoin(i: u32, j: u32) {
    var a = i;
    var b = j;
    loop {
        a = ufTileFind(a);
        b = ufTileFind(b);
        if (a == b) {
            return;
        }
        let lo = min(a, b);
        let hi = max(a, b);
        let old = ufTileLower(hi, lo);
        if (old == hi) {
            return;
        }
        a = old;
        b = lo;
    }
}
