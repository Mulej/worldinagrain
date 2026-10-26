// Separable gaussian blur for the bloom. Compiled standalone (no prelude).

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;

const RADIUS: i32 = 6;
const SIGMA2: f32 = 18.0;  // 2 * sigma^2 with sigma = 3

fn blur(gid: vec2<u32>, dir: vec2<i32>) {
    let dims = vec2<i32>(textureDimensions(src));
    let p = vec2<i32>(gid);
    if (p.x >= dims.x || p.y >= dims.y) {
        return;
    }
    var sum = vec3<f32>(0.0);
    var total = 0.0;
    for (var k = -RADIUS; k <= RADIUS; k++) {
        let q = clamp(p + dir * k, vec2<i32>(0), dims - 1);
        let w = exp(-f32(k * k) / SIGMA2);
        sum += textureLoad(src, q, 0).rgb * w;
        total += w;
    }
    textureStore(dst, p, vec4<f32>(sum / total, 1.0));
}

@compute @workgroup_size(8, 8)
fn blurH(@builtin(global_invocation_id) gid: vec3<u32>) {
    blur(gid.xy, vec2<i32>(1, 0));
}

@compute @workgroup_size(8, 8)
fn blurV(@builtin(global_invocation_id) gid: vec3<u32>) {
    blur(gid.xy, vec2<i32>(0, 1));
}
