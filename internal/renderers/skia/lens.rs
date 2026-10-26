// Copyright © SixtyFPS GmbH <info@slint.dev>
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Slint-Royalty-free-2.0 OR LicenseRef-Slint-Software-3.0

//! The lens of a `BackdropBlur` with `refraction`: its rounded edge bends the backdrop as the
//! edge of thick glass does, in a band `bezel` wide, and brightens toward the rim.

use skia_safe::{ImageFilter, RuntimeEffect, runtime_effect::RuntimeShaderBuilder};

/// `p` is in the rectangle's own coordinates, from its top left corner. At `d` in from the edge
/// the backdrop shows from further inside, along the edge's normal, by a quarter circle's
/// profile: `refraction` at the edge, nothing at `bezel` in. Red bends `dispersion` more and blue
/// as much less. The light comes from the top left and is `edge_light` strong: a sharp glint
/// along the rim, bright where the edge faces the light and fainter where it faces away (the
/// two opposite glints of thick glass); just inside the lit rim a dark band, where thick glass
/// turns the light away; and inside the edge that faces away from the light, a soft bright lip
/// where the light that passed through the body gathers. The body itself stays clear.
const SKSL: &str = r#"
uniform shader backdrop;
uniform float2 size;
uniform float4 radii;
uniform float refraction;
uniform float bezel;
uniform float dispersion;
uniform float edge_light;

// Where the light comes from: the top left, mostly from above.
const float2 LIGHT = float2(-0.54, -0.84);

float edge_distance(float2 p) {
    float2 h = size * 0.5;
    float2 c = p - h;
    float r = c.x < 0.0 ? (c.y < 0.0 ? radii.x : radii.w) : (c.y < 0.0 ? radii.y : radii.z);
    r = min(r, min(h.x, h.y));
    float2 q = abs(c) - h + r;
    return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r;
}

half4 main(float2 p) {
    float d = -edge_distance(p);
    if (d >= bezel) {
        return backdrop.eval(p);
    }
    float2 n = normalize(float2(
        edge_distance(p + float2(0.5, 0.0)) - edge_distance(p - float2(0.5, 0.0)),
        edge_distance(p + float2(0.0, 0.5)) - edge_distance(p - float2(0.0, 0.5))) + 1e-5);
    float t = 1.0 - clamp(d / bezel, 0.0, 1.0);
    float2 bend = n * (1.0 - sqrt(1.0 - t * t)) * refraction;
    half4 c = backdrop.eval(p - bend);
    if (dispersion > 0.0) {
        c.r = backdrop.eval(p - bend * (1.0 + dispersion)).r;
        c.b = backdrop.eval(p - bend * (1.0 - dispersion)).b;
    }
    float lit = max(dot(n, LIGHT), 0.0);
    float away = max(-dot(n, LIGHT), 0.0);
    float glint = exp(-d / max(1.2, bezel * 0.05))
        * (0.22 + 1.3 * lit * lit + 0.75 * away * away * away);
    float dz = (d - 0.2 * bezel) / (0.12 * bezel);
    float dark = exp(-dz * dz) * (0.25 + 0.75 * lit);
    float cz = (d - 0.32 * bezel) / (0.2 * bezel);
    float lip = 0.32 * exp(-cz * cz) * away * sqrt(away);
    c.rgb *= half(max(1.0 - 0.45 * edge_light * dark, 0.0));
    c.rgb = min(c.rgb + half3(edge_light * (glint + lip)) * c.a, c.aaa);
    return c;
}
"#;

thread_local! {
    static EFFECT: Option<RuntimeEffect> = RuntimeEffect::make_for_shader(SKSL, None)
        .map_err(|error| i_slint_core::debug_log!("Skia renderer: the backdrop lens does not compile: {error}"))
        .ok();
}

/// The lens over `input` (the blurred backdrop, or the backdrop itself where it is `None`) for a
/// rectangle `size` big with corners `radii` (top left, top right, bottom right, bottom left),
/// all in physical pixels, with `bezel` above 0. `None` if the lens cannot be made, which leaves
/// the backdrop unbent. Skia reads the input as far round each pixel as the lens reaches.
pub fn filter(
    input: Option<ImageFilter>,
    size: [f32; 2],
    radii: [f32; 4],
    refraction: f32,
    bezel: f32,
    dispersion: f32,
    edge_light: f32,
) -> Option<ImageFilter> {
    EFFECT.with(|effect| {
        let mut builder = RuntimeShaderBuilder::new(effect.clone()?);
        builder.set_uniform_float("size", &size).ok()?;
        builder.set_uniform_float("radii", &radii).ok()?;
        builder.set_uniform_float("refraction", &[refraction]).ok()?;
        builder.set_uniform_float("bezel", &[bezel]).ok()?;
        builder.set_uniform_float("dispersion", &[dispersion]).ok()?;
        builder.set_uniform_float("edge_light", &[edge_light]).ok()?;
        let reach = refraction * (1. + dispersion.max(0.));
        skia_safe::image_filters::runtime_shader_with_options(
            &builder, reach, "backdrop", input, false,
        )
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_lens_compiles() {
        if let Err(error) = skia_safe::RuntimeEffect::make_for_shader(super::SKSL, None) {
            panic!("{error}");
        }
    }
}
