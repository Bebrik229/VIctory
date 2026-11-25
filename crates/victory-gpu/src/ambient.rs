//! Constant indirect diffuse light while ray-traced probes do not run.
//!
//! Grand Theft Auto VI's indirect diffuse light comes from probes it fills
//! by tracing rays, which do not run here yet: its program that resolves
//! them (`CS_ScreenProbes_TemporalResolve_*`) leaves next to nothing, and
//! whatever the sun does not reach directly is black (or uninitialized
//! garbage such as bright magenta foliage and indoor color blocks). Right
//! after that program, its output is replaced by a constant (it used to be
//! a floor under what the program wrote, but that is garbage: see the
//! shader). The constant is the same everywhere: the light in these images
//! does not go with the title's exposure (well-lit surfaces hold about the
//! same in a shot at -5.5 EV and in one at -9.75). The title's own lighting
//! then applies it with the surfaces' colors.
//!
//! `VICTORY_AMBIENT=<level>` sets it (0.6 by default, 0 for none); `off`
//! keeps the title's own output.
//! shortcut: the same light everywhere, from everywhere; gone once rays run.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering};
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::s;

use crate::host::{HostList, host};
use crate::object::This;

/// The programs it follows, by the start of their name.
pub(crate) const PROGRAM: &[u8] = b"CS_ScreenProbes_TemporalResolve";
/// Where they write the indirect diffuse light (R9G9B9E5 as 32-bit
/// integers): root parameter and place in its table.
pub(crate) const LIGHT: (u8, u32) = (1, 176);

/// 0.6 looked balanced in run a2. `VICTORY_AMBIENT=off` leaves what the
/// title's resolve wrote; 0 is no indirect diffuse light at all.
const DEFAULT: f32 = 0.6;

/// Parses `VICTORY_AMBIENT`: `off` (case-insensitive) returns `None`, a
/// finite non-negative float overrides the level, and unset/empty/invalid
/// values fall back to [`DEFAULT`]. Pure, for testing.
fn parse_level(value: Option<&str>) -> Option<f32> {
    let trimmed = value.map(str::trim).unwrap_or("");
    if trimmed.eq_ignore_ascii_case("off") {
        return None;
    }
    let level = trimmed
        .parse::<f32>()
        .ok()
        .filter(|l| l.is_finite() && *l >= 0.0)
        .unwrap_or(DEFAULT);
    Some(level)
}

/// The constant indirect diffuse level; `None` if `VICTORY_AMBIENT=off`.
pub(crate) fn level() -> Option<f32> {
    static LEVEL: OnceLock<Option<f32>> = OnceLock::new();
    *LEVEL.get_or_init(|| parse_level(std::env::var("VICTORY_AMBIENT").ok().as_deref()))
}

const SOURCE: &str = r#"
cbuffer Floor : register(b0) { float level; float3 tint; };
RWTexture2D<uint> light : register(u0);
uint pack(float3 c) {
    c = clamp(c, 0.0, 65408.0);
    float most = max(c.r, max(c.g, c.b));
    int e = clamp(int(floor(log2(max(most, 1e-9)))) + 16, 0, 31);
    float scale = exp2(float(24 - e));
    if (most * scale >= 511.5) { e = min(e + 1, 31); scale = exp2(float(24 - e)); }
    uint3 m = min(uint3(c * scale + 0.5), 511u);
    return m.r | (m.g << 9) | (m.b << 18) | (uint(e) << 27);
}
[numthreads(8, 8, 1)]
void main(uint3 at : SV_DispatchThreadID) {
    uint w, h;
    light.GetDimensions(w, h);
    if (at.x >= w || at.y >= h) return;
    // What the resolve left is not kept: without rays its inputs are never
    // written and it holds whatever their memory did (bright magenta on
    // translucent surfaces, blocks of colours indoors: run v6, 2026-11-25).
    light[at.xy] = pack(tint * level);
}
"#;

/// Views, used in turn: more than the passes of every list in flight.
const SLOTS: u32 = 1024;

struct Pass {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    views: ID3D12DescriptorHeap,
    step: u32,
    next: AtomicU32,
}
// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Pass {}
// SAFETY: as above.
unsafe impl Sync for Pass {}

/// Serializes the root signature directly instead of embedding `[RootSignature]`
/// in HLSL so it works under both native D3D12 and Wine/VKD3D.
fn make_root_signature(device: &ID3D12Device) -> windows::core::Result<ID3D12RootSignature> {
    let range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
        NumDescriptors: 1,
        BaseShaderRegister: 0,
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: 0,
    };
    let params = [
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: 4,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
    ];
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        NumStaticSamplers: 0,
        pStaticSamplers: std::ptr::null(),
        Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
    };
    let mut blob: Option<ID3DBlob> = None;
    // SAFETY: `desc` points at live slices for the duration of the call.
    unsafe {
        D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None)?;
        let blob = blob.ok_or_else(windows::core::Error::empty)?;
        let bytes =
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
        device.CreateRootSignature(0, bytes)
    }
}

fn make(device: &ID3D12Device) -> windows::core::Result<Pass> {
    let code = crate::host::compile(SOURCE, s!("main"), s!("cs_5_1"))?;
    let root = make_root_signature(device)?;
    // SAFETY: plain D3D12 compute PSO and descriptor heap creation.
    unsafe {
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: code.as_ptr().cast(),
                BytecodeLength: code.len(),
            },
            ..Default::default()
        })?;
        let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        let views = device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: kind,
            NumDescriptors: SLOTS,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        })?;
        Ok(Pass {
            root,
            pso,
            views,
            step: device.GetDescriptorHandleIncrementSize(kind),
            next: AtomicU32::new(0),
        })
    }
}

fn uav_barrier(list: &ID3D12GraphicsCommandList, resource: &ID3D12Resource) {
    let barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                // SAFETY: borrowed COM pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(resource) },
            }),
        },
    };
    // SAFETY: an open list; `resource` lives through the call and is kept by the list.
    unsafe { list.ResourceBarrier(&[barrier]) };
}

/// Replaces what `light` holds with constant indirect diffuse light, in `list`,
/// after the resolve program wrote it as an image. Returns whether anything was
/// recorded (if true, the list's descriptor heap, compute root signature, and
/// pipeline state were replaced by ours).
pub(crate) fn raise(list: &HostList, light: This) -> bool {
    let (Some(level), Some(host)) = (level(), host()) else {
        return false;
    };
    let Some(light) = crate::resource::host_resource(light) else {
        return false;
    };
    static PASS: OnceLock<Option<Pass>> = OnceLock::new();
    let Some(pass) = PASS
        .get_or_init(|| {
            make(&host.device)
                .inspect_err(|e| tracing::warn!(error = %e, "no ambient light"))
                .ok()
        })
        .as_ref()
    else {
        return false;
    };
    // SAFETY: a live resource.
    let d = unsafe { light.GetDesc() };
    static ONCE: std::sync::Once = std::sync::Once::new();
    if !matches!(d.Format, DXGI_FORMAT_R32_UINT | DXGI_FORMAT_R32_TYPELESS) {
        ONCE.call_once(|| tracing::warn!(format = d.Format.0, "no ambient light: another format"));
        return false;
    }
    if d.Width == 0 || d.Height == 0 {
        return false;
    }
    ONCE.call_once(|| {
        tracing::info!(level, width = d.Width, height = d.Height, "ambient light");
    });
    let slot = pass.next.fetch_add(1, Ordering::Relaxed) % SLOTS;
    list.keep(light);
    let l = &list.list;
    uav_barrier(l, light);
    // SAFETY: an open list; a live texture the list keeps, in the
    // unordered-access state as the program before left it; a view in a slot
    // no list in flight reads.
    unsafe {
        let cpu = pass.views.GetCPUDescriptorHandleForHeapStart();
        let gpu = pass.views.GetGPUDescriptorHandleForHeapStart();
        let at = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: cpu.ptr + (slot * pass.step) as usize,
        };
        let view = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        host.device
            .CreateUnorderedAccessView(light, None::<&ID3D12Resource>, Some(&view), at);
        l.SetDescriptorHeaps(&[Some(pass.views.clone())]);
        l.SetComputeRootSignature(&pass.root);
        l.SetPipelineState(&pass.pso);
        let constants = [level, 1.0f32, 0.98, 0.94];
        l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        l.SetComputeRootDescriptorTable(
            1,
            D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: gpu.ptr + u64::from(slot * pass.step),
            },
        );
        l.Dispatch((d.Width as u32).div_ceil(8), d.Height.div_ceil(8), 1);
    }
    uav_barrier(l, light);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_level_defaults_overrides_and_disables() {
        assert_eq!(parse_level(None), Some(DEFAULT));
        assert_eq!(parse_level(Some("")), Some(DEFAULT));
        assert_eq!(parse_level(Some("   ")), Some(DEFAULT));
        assert_eq!(parse_level(Some("off")), None);
        assert_eq!(parse_level(Some(" OFF ")), None);
        assert_eq!(parse_level(Some("0")), Some(0.0));
        assert_eq!(parse_level(Some("0.0")), Some(0.0));
        assert_eq!(parse_level(Some("0.6")), Some(0.6));
        assert_eq!(parse_level(Some(" 1.25 ")), Some(1.25));
        assert_eq!(parse_level(Some("-0.1")), Some(DEFAULT));
        assert_eq!(parse_level(Some("NaN")), Some(DEFAULT));
        assert_eq!(parse_level(Some("inf")), Some(DEFAULT));
        assert_eq!(parse_level(Some("invalid")), Some(DEFAULT));
    }

    /// The pass compiles with the system's compiler and the host takes it.
    #[test]
    fn ambient_pass_builds() {
        let Some(host) = crate::host::host() else {
            return;
        };
        make(&host.device).expect("the ambient pass");
    }
}
