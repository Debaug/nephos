struct AffineDecomposition {
    angle: f32,
    shear: f32,
    scale: vec2f,
    translation: vec2f,
}

struct Affine {
    decomposition: AffineDecomposition,
    computed: mat3x3f,
}

fn hash(val: u32) -> u32 {
    return (val + 1) * 1580030168; // * (2^32 - 1) / e
}

// in range [0, 1)
// FIXME: better generation
fn random_float(idx: u32) -> f32 {
    return f32(idx % 0xFFFF) / f32(0xFFFF);
}
