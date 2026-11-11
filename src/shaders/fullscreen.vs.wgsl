// One triangle covering the screen: clip (-1,-1), (3,-1), (-1,3). Draw with draw(3) and no vertex buffer.
@vertex
fn main(@builtin(vertex_index) vertexIdx: u32) -> @builtin(position) vec4f {
    let uv = vec2f(f32((vertexIdx << 1u) & 2u), f32(vertexIdx & 2u));
    return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}
