// Copyright © SixtyFPS GmbH <info@slint.dev>
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Slint-Royalty-free-2.0 OR LicenseRef-Slint-Software-3.0

//! Kidō's HDR spike (S17 in Tokahiro/kidou-dev's docs/spikes.md), only with the
//! `kidou-hdr-spike` feature: where the environment variable `KIDOU_SPIKE_HDR` is `scrgb` or
//! `scrgb-g22`, a window's swap chain is 16-bit float in the extended linear sRGB colour space
//! (scRGB), where 1.0 is 80 nits. Slint still draws each frame into an 8-bit layer as it draws
//! into an 8-bit swap chain without the spike, so its blending stays the same; a shader then
//! decodes the layer (the sRGB curve with `scrgb`, gamma 2.2 with `scrgb-g22`) and scales it so
//! that its white is Windows' SDR white. With `KIDOU_SPIKE_HDR_LINES` at `1`, three lines at the
//! top right show 1, 2 and 4 times SDR white.

use std::cell::{Cell, RefCell};
use std::time::{Duration, Instant};

use skia_safe::{Color4f, Data, Paint, Rect, RuntimeEffect, gpu};
use wgpu_30 as wgpu;

/// scRGB's 1.0, in nits.
const SCRGB_ONE: f32 = 80.;

/// A frame after a pause at least this long reads the SDR white level again: Windows tells a
/// window of no change to it (Microsoft Learn, "Use DirectX with Advanced Color on high/standard
/// dynamic range displays", "IDXGIOutput6"), and reading it on every frame of an animation would
/// ask Windows each time.
const REST: Duration = Duration::from_millis(500);

/// The layer, premultiplied and sRGB encoded, decoded to linear and scaled by `white`.
const SKSL: &str = r#"
uniform shader layer;
uniform float white;
uniform float gamma22;

float3 srgb(float3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, float3(2.4)), step(0.04045, c));
}

half4 main(float2 p) {
    float4 c = layer.eval(p);
    if (c.a <= 0.0) {
        return half4(0.0);
    }
    float3 e = c.rgb / c.a;
    float3 l = gamma22 > 0.5 ? pow(e, float3(2.2)) : srgb(e);
    return half4(half3(l * white * c.a), half(c.a));
}
"#;

pub(crate) struct Hdr {
    gamma22: bool,
    lines: bool,
    effect: RuntimeEffect,
    /// SDR white in scRGB's units.
    white: Cell<f32>,
    last_frame: Cell<Option<Instant>>,
    layer: RefCell<Option<skia_safe::Surface>>,
}

impl Hdr {
    /// The spike as `KIDOU_SPIKE_HDR` asks for it, `None` where it asks for none.
    pub(crate) fn from_env() -> Option<Self> {
        let mode = std::env::var("KIDOU_SPIKE_HDR").ok()?;
        let gamma22 = match mode.as_str() {
            "scrgb" => false,
            "scrgb-g22" => true,
            _ => {
                log::info!("hdr: KIDOU_SPIKE_HDR is {mode:?}, so the frame stays 8-bit");
                return None;
            }
        };
        let effect = RuntimeEffect::make_for_shader(SKSL, None)
            .map_err(|error| log::error!("hdr: the decoding shader does not compile: {error}"))
            .ok()?;
        Some(Self {
            gamma22,
            lines: std::env::var("KIDOU_SPIKE_HDR_LINES").is_ok_and(|lines| lines == "1"),
            effect,
            white: Cell::new(1.),
            last_frame: Cell::new(None),
            layer: RefCell::new(None),
        })
    }

    /// Asks for the 16-bit swap chain in scRGB where the surface offers it; `false` leaves
    /// `config` as it was.
    pub(crate) fn choose(
        &self,
        capabilities: &wgpu::SurfaceCapabilities,
        config: &mut wgpu::SurfaceConfiguration,
    ) -> bool {
        let format = wgpu::TextureFormat::Rgba16Float;
        let offered = capabilities
            .color_spaces(format)
            .contains(wgpu::SurfaceColorSpaces::EXTENDED_SRGB_LINEAR);
        if offered {
            config.format = format;
            config.color_space = wgpu::SurfaceColorSpace::ExtendedSrgbLinear;
        }
        log::info!(
            "hdr: swap chain {:?}, {} decode{}",
            config.format,
            if self.gamma22 { "gamma 2.2" } else { "sRGB" },
            if offered { "" } else { " (Rgba16Float in scRGB is not offered)" }
        );
        offered
    }

    /// Reads SDR white from Windows, through wgpu, and logs it with `why`.
    pub(crate) fn read_white(
        &self,
        surface: &wgpu::Surface<'_>,
        adapter: &wgpu::Adapter,
        why: &str,
    ) {
        let info = surface.display_hdr_info(adapter);
        let hdr = info.coarse.and_then(|coarse| coarse.high_dynamic_range);
        let luminance = info.luminance.unwrap_or_default();
        let white = match (hdr, luminance.sdr_white_nits) {
            (Some(true), Some(nits)) if nits > 0. => nits / SCRGB_ONE,
            // An SDR output shows scRGB's 1.0 as its white and clips above it.
            _ => 1.,
        };
        self.white.set(white);
        log::info!(
            "hdr: {why}: output HDR {hdr:?}, SDR white {:?} nits, peak {:?} nits, full frame {:?} \
             nits, scale {white}",
            luminance.sdr_white_nits,
            luminance.max_nits,
            luminance.max_full_frame_nits,
        );
    }

    /// Whether this frame comes after a rest, when the white level is read again.
    pub(crate) fn after_rest(&self) -> bool {
        let now = Instant::now();
        let rested = self.last_frame.get().is_none_or(|last| now - last >= REST);
        self.last_frame.set(Some(now));
        rested
    }

    /// The 8-bit layer Slint draws a `width` × `height` frame into, made anew when the size
    /// changes.
    pub(crate) fn layer(
        &self,
        gr_context: &mut gpu::DirectContext,
        width: i32,
        height: i32,
    ) -> Option<std::cell::RefMut<'_, skia_safe::Surface>> {
        let mut layer = self.layer.borrow_mut();
        if layer.as_ref().is_none_or(|layer| (layer.width(), layer.height()) != (width, height)) {
            let info = skia_safe::ImageInfo::new(
                (width, height),
                skia_safe::ColorType::RGBA8888,
                skia_safe::AlphaType::Premul,
                crate::srgb_color_space(),
            );
            *layer = gpu::surfaces::render_target(
                gr_context,
                gpu::Budgeted::Yes,
                &info,
                None,
                gpu::SurfaceOrigin::TopLeft,
                None,
                false,
                None,
            );
        }
        std::cell::RefMut::filter_map(layer, Option::as_mut).ok()
    }

    /// Draws `layer` decoded and scaled onto `target`, the swap chain's linear surface, and the
    /// test lines above it.
    pub(crate) fn compose(&self, layer: &mut skia_safe::Surface, target: &skia_safe::Canvas) {
        let image = layer.image_snapshot();
        let white = self.white.get();
        let uniforms = [white, if self.gamma22 { 1. } else { 0. }];
        let bytes: Vec<u8> = uniforms.iter().flat_map(|value| value.to_ne_bytes()).collect();
        let shader =
            image.to_raw_shader(None, skia_safe::SamplingOptions::default(), None).and_then(
                |layer| self.effect.make_shader(Data::new_copy(&bytes), &[layer.into()], None),
            );
        let Some(shader) = shader else {
            log::error!("hdr: the decoding shader could not be made");
            return;
        };
        let mut paint = Paint::default();
        paint.set_shader(shader);
        paint.set_blend_mode(skia_safe::BlendMode::Src);
        target.draw_paint(&paint);

        if self.lines {
            let linear = skia_safe::ColorSpace::new_srgb_linear();
            let right = image.width() as f32 - 16.;
            for (row, times) in [1., 2., 4.].into_iter().enumerate() {
                let level = white * times;
                let mut line = Paint::new(Color4f::new(level, level, level, 1.), &linear);
                line.set_blend_mode(skia_safe::BlendMode::Src);
                let top = 16. + row as f32 * 12.;
                target.draw_rect(Rect::new(right - 160., top, right, top + 4.), &line);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_decoding_shader_compiles() {
        if let Err(error) = skia_safe::RuntimeEffect::make_for_shader(super::SKSL, None) {
            panic!("{error}");
        }
    }
}
