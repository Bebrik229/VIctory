//! End-to-end tests for immediate samplers chosen by uniform control flow,
//! including multi-way (flat) and nested immediate sampler phis, and mixing
//! regular and comparison sampling on the same immediate sampler words.
//! Synthetic GFX10 programs lowered to DXIL, validated by `dxil.dll` when
//! present, and run on the host GPU and WARP.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::resource::{Role, Source};
use crate::{dxbc, io, ir, resource, structure};

mod comparison {
    use std::ffi::c_void;
    use std::mem::ManuallyDrop;

    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::{
        DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32_TYPELESS, DXGI_SAMPLE_DESC,
    };
    use windows::core::{Interface, Result};

    const ONE_SAMPLE: DXGI_SAMPLE_DESC = DXGI_SAMPLE_DESC {
        Count: 1,
        Quality: 0,
    };

    // The general test host's RGBA texture does not support comparison on
    // all devices. Use an R32_FLOAT SRV with the host's compute pipeline.
    pub(super) fn dispatch(
        device: &ID3D12Device,
        root: &ID3D12RootSignature,
        pso: &ID3D12PipelineState,
        user_data: &[u32],
    ) -> Result<Vec<u32>> {
        let direct = D3D12_COMMAND_LIST_TYPE_DIRECT;
        // SAFETY: `device`, `root`, and `pso` are valid live D3D12 objects; all
        // mapped/placed descriptors and command-list resources stay alive until
        // the fence wait completes.
        unsafe {
            let queue_desc = D3D12_COMMAND_QUEUE_DESC {
                Type: direct,
                ..Default::default()
            };
            let queue: ID3D12CommandQueue = device.CreateCommandQueue(&queue_desc)?;
            let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(direct)?;
            let list: ID3D12GraphicsCommandList =
                device.CreateCommandList(0, direct, &allocator, pso)?;
            let texture = resource(
                device,
                D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                    Width: 2,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    Format: DXGI_FORMAT_R32_FLOAT,
                    SampleDesc: ONE_SAMPLE,
                    ..Default::default()
                },
            )?;
            let texels = [0.25f32, 0.75];
            let row_pitch = D3D12_TEXTURE_DATA_PITCH_ALIGNMENT;
            let upload = resource(
                device,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                buffer(u64::from(row_pitch)),
            )?;
            let mut mapped = std::ptr::null_mut::<c_void>();
            upload.Map(0, None, Some(&mut mapped))?;
            mapped.cast::<[f32; 2]>().write(texels);
            upload.Unmap(0, None);
            let mut src = D3D12_TEXTURE_COPY_LOCATION {
                pResource: ManuallyDrop::new(Some(upload.clone())),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: DXGI_FORMAT_R32_FLOAT,
                            Width: 2,
                            Height: 1,
                            Depth: 1,
                            RowPitch: row_pitch,
                        },
                    },
                },
            };
            let mut dst = D3D12_TEXTURE_COPY_LOCATION {
                pResource: ManuallyDrop::new(Some(texture.clone())),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            ManuallyDrop::drop(&mut src.pResource);
            ManuallyDrop::drop(&mut dst.pResource);
            transition(
                &list,
                &texture,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            );

            let output = resource(
                device,
                D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_DESC {
                    Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    ..buffer(8)
                },
            )?;
            let readback = resource(
                device,
                D3D12_HEAP_TYPE_READBACK,
                D3D12_RESOURCE_STATE_COPY_DEST,
                buffer(8),
            )?;
            let heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 2,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            };
            let heap: ID3D12DescriptorHeap = device.CreateDescriptorHeap(&heap_desc)?;
            let srv = heap.GetCPUDescriptorHandleForHeapStart();
            let stride = device.GetDescriptorHandleIncrementSize(heap_desc.Type) as usize;
            let uav = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: srv.ptr + stride,
            };
            device.CreateShaderResourceView(&texture, None, srv);
            device.CreateUnorderedAccessView(
                &output,
                None,
                Some(&D3D12_UNORDERED_ACCESS_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                    Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_UAV {
                            NumElements: 2,
                            Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                            ..Default::default()
                        },
                    },
                }),
                uav,
            );
            list.SetDescriptorHeaps(&[Some(heap.clone())]);
            list.SetComputeRootSignature(root);
            list.SetComputeRoot32BitConstants(
                0,
                user_data.len() as u32,
                user_data.as_ptr().cast(),
                0,
            );
            list.Dispatch(1, 1, 1);
            transition(
                &list,
                &output,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            );
            list.CopyResource(&readback, &output);
            list.Close()?;
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
            queue.ExecuteCommandLists(&[Some(list.cast()?)]);
            queue.Signal(&fence, 1)?;
            fence.SetEventOnCompletion(1, Default::default())?;
            readback.Map(0, None, Some(&mut mapped))?;
            let words = mapped.cast::<[u32; 2]>().read().to_vec();
            readback.Unmap(0, None);
            Ok(words)
        }
    }

    fn buffer(width: u64) -> D3D12_RESOURCE_DESC {
        D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: width,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: ONE_SAMPLE,
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        }
    }

    fn resource(
        device: &ID3D12Device,
        kind: D3D12_HEAP_TYPE,
        state: D3D12_RESOURCE_STATES,
        desc: D3D12_RESOURCE_DESC,
    ) -> Result<ID3D12Resource> {
        let mut resource = None;
        // SAFETY: `device` is a valid D3D12 device and `desc` describes a valid committed resource.
        unsafe {
            device.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: kind,
                    CreationNodeMask: 1,
                    VisibleNodeMask: 1,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &desc,
                state,
                None,
                &mut resource,
            )?;
        }
        Ok(resource.expect("committed resource"))
    }

    fn transition(
        list: &ID3D12GraphicsCommandList,
        resource: &ID3D12Resource,
        before: D3D12_RESOURCE_STATES,
        after: D3D12_RESOURCE_STATES,
    ) {
        let mut barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: ManuallyDrop::new(Some(resource.clone())),
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: before,
                    StateAfter: after,
                }),
            },
        };
        // SAFETY: `list` and `resource` are valid D3D12 objects and `barrier` holds a Transition variant.
        unsafe {
            list.ResourceBarrier(std::slice::from_ref(&barrier));
            ManuallyDrop::drop(&mut (*barrier.Anonymous.Transition).pResource);
        }
    }
}

/// User data: s0 a texture SRV table, s1 a root constant, s2 a raw UAV table.
const SAMPLER_TEST_LOADS_WAVE32: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF40C_0202,
    0xFA00_0000, // s_load_dwordx8 s[8:15], s[4:5], 0x0
    0xBE86_0302, // s_mov_b32 s6, s2
    0xBE87_0380, // s_mov_b32 s7, 0
    0xF408_0403,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[6:7], 0x0
];

/// The root constant picks clamp (non-zero) or wrap (zero).
const TWO_SAMPLER_PHI_WAVE32: &[u32] = &[
    0xBF06_8001, // s_cmp_eq_u32 s1, 0
    0xBF85_0006, // s_cbranch_scc1 -> 0x40
    0xBE94_03FF,
    0x0000_0092, // s_mov_b32 s20, 0x92 (clamp x and y)
    0xBE95_0380, // s_mov_b32 s21, 0
    0xBE96_0380, // s_mov_b32 s22, 0
    0xBE97_0380, // s_mov_b32 s23, 0
    0xBF82_0004, // s_branch -> 0x50
    0xBE94_0380, // 0x40: s_mov_b32 s20, 0 (wrap)
    0xBE95_0380, // s_mov_b32 s21, 0
    0xBE96_0380, // s_mov_b32 s22, 0
    0xBE97_0380, // s_mov_b32 s23, 0
];

/// Sample at u = 1.25 and store red to dword 0 of the UAV.
const SAMPLE_AND_STORE_WAVE32: &[u32] = &[
    0x7E00_02FF,
    0x3FA0_0000, // 0x50: v_mov_b32 v0, 1.25
    0x7E02_02F0, // v_mov_b32 v1, 0.5
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xF09C_0108,
    0x00A2_0200, // image_sample_lz v2, v[0:1], s[8:15], s[20:23]
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0x7E06_0280, // v_mov_b32 v3, 0
    0xE070_1000,
    0x8004_0203, // buffer_store_dword v2, v3, s[16:19], 0 offen
    0xBF81_0000, // s_endpgm
];

/// The same resources and sample as above, with three edges into the join.
const FLAT_SAMPLER_PHI_WAVE32: &[u32] = &[
    0xBF06_8001, // s_cmp_eq_u32 s1, 0
    0xBF85_000E, // s_cbranch_scc1 wrap
    0xBF06_8101, // s_cmp_eq_u32 s1, 1
    0xBF85_0006, // s_cbranch_scc1 clamp
    0xBE94_03FF,
    0x0000_0036, // border: s_mov_b32 s20, 0x36 (black border x and y)
    0xBE95_0380, // s_mov_b32 s21, 0
    0xBE96_0380, // s_mov_b32 s22, 0
    0xBE97_0380, // s_mov_b32 s23, 0
    0xBF82_000A, // s_branch join
    0xBE94_03FF,
    0x0000_0092, // clamp: s_mov_b32 s20, 0x92
    0xBE95_0380, // s_mov_b32 s21, 0
    0xBE96_0380, // s_mov_b32 s22, 0
    0xBE97_0380, // s_mov_b32 s23, 0
    0xBF82_0004, // s_branch join
    0xBE94_0380, // wrap: s_mov_b32 s20, 0
    0xBE95_0380, // s_mov_b32 s21, 0
    0xBE96_0380, // s_mov_b32 s22, 0
    0xBE97_0380, // s_mov_b32 s23, 0
];

/// The inner branch joins before the outer branch. Both samples use the
/// same descriptor words, with different sampler kinds.
const NESTED_SAMPLER_PHI_WAVE32: &[u32] = &[
    0xBF06_8001, // s_cmp_eq_u32 s1, 0
    0xBF85_000F, // s_cbranch_scc1 wrap
    0xBF06_8101, // s_cmp_eq_u32 s1, 1
    0xBF85_0006, // s_cbranch_scc1 clamp
    0xBE94_03FF,
    0x0000_4036, // border: s_mov_b32 s20, 0x4036 (greater)
    0xBE95_0380, // s_mov_b32 s21, 0
    0xBE96_0380, // s_mov_b32 s22, 0
    0xBE97_0380, // s_mov_b32 s23, 0
    0xBF82_0005, // s_branch inner_join
    0xBE94_03FF,
    0x0000_1092, // clamp: s_mov_b32 s20, 0x1092 (less)
    0xBE95_0380, // s_mov_b32 s21, 0
    0xBE96_0380, // s_mov_b32 s22, 0
    0xBE97_0380, // s_mov_b32 s23, 0
    0xBF8C_C07F, // inner_join: s_waitcnt lgkmcnt(0)
    0xBF82_0005, // s_branch outer_join
    0xBE94_03FF,
    0x0000_1000, // wrap: s_mov_b32 s20, 0x1000 (less)
    0xBE95_0380, // s_mov_b32 s21, 0
    0xBE96_0380, // s_mov_b32 s22, 0
    0xBE97_0380, // s_mov_b32 s23, 0
    0x7E00_02F0, // outer_join: v_mov_b32 v0, 0.5 (comparison reference)
    0x7E02_02FF,
    0x3FA0_0000, // v_mov_b32 v1, 1.25
    0x7E04_02F0, // v_mov_b32 v2, 0.5
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xF09C_0108,
    0x00A2_0301, // image_sample_lz v3, v[1:2], s[8:15], s[20:23]
    0xF0BC_0108,
    0x00A2_0400, // image_sample_c_lz v4, v[0:2], s[8:15], s[20:23]
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0x7E0A_0280, // v_mov_b32 v5, 0
    0xE074_1000,
    0x8004_0305, // buffer_store_dwordx2 v[3:4], v5, s[16:19], 0 offen
    0xBF81_0000, // s_endpgm
];

fn param(root: RootType, user_data: u8) -> RootParam {
    RootParam {
        root,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data,
        slot: user_data,
    }
}

fn entry(kind: ResourceType, dwords: u8, user_data: u8) -> UsedResource {
    UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords,
        user_data,
        slot: user_data,
    }
}

fn build_checked(
    words: &[u32],
    layout: Vec<RootParam>,
    used: Vec<UsedResource>,
    threads: u32,
    check_sources: impl FnOnce(&resource::Resources),
) -> HostShader {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout,
        used,
        io: StageIo::Cs(ComputeIo {
            rsrc2: 3 << 1,
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    check_sources(&res);
    let io = io::raise(&p, &native).expect("io::raise");
    let bytes = dxbc::build(&[]);
    let c = dxbc::Container::parse(&bytes).expect("container");
    let shader = super::emit(&c, &native, &p, &st, &io, &res).expect("emit");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    shader
}

fn build_sampler_test(words: &[u32], check_sources: impl FnOnce(&[&Source])) -> HostShader {
    build_checked(
        words,
        vec![
            param(RootType::Table, 0),
            param(RootType::Constants, 1),
            param(RootType::Table, 2),
        ],
        vec![
            entry(ResourceType::Srv, 8, 0),
            entry(ResourceType::Uav, 4, 2),
        ],
        1,
        |res| {
            let sources: Vec<_> = res
                .descriptors
                .iter()
                .filter(|d| d.role == Role::Sampler)
                .map(|d| &d.source)
                .collect();
            check_sources(&sources);
        },
    )
}

fn run_sampler_test(shader: &HostShader, cases: &[(u32, &[f32])], comparison: bool) {
    // At u = 1.25, wrap reads the first texel and clamp reads the second.
    let texels = [[0.25, 0.0, 0.0, 1.0], [0.75, 0.0, 0.0, 1.0]];
    let views = [
        View::Texture(2, 1, &texels),
        View::RawUav(cases[0].1.len() as u32),
    ];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        for &(choice, want) in cases {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Table, 0) => 0,
                    (RootType::Constants, 1) => choice,
                    // The UAV at heap index 1: 32 bytes in.
                    (RootType::Table, 2) => 32,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let words = if comparison {
                comparison::dispatch(&host.device, &root, &pso, &ud)
                    .unwrap_or_else(|e| panic!("{label}: comparison dispatch failed: {e}"))
            } else {
                host.dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &views,
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"))
                .swap_remove(0)
            };
            let got: Vec<_> = words.into_iter().map(f32::from_bits).collect();
            assert_eq!(got, want, "{label}, choice {choice}");
        }
        ran += 1;
    }
    assert!(ran > 0, "no SM 6.6 device");
}

#[test]
fn a_sampler_chosen_by_a_uniform_branch_samples_as_chosen() {
    let words = [
        SAMPLER_TEST_LOADS_WAVE32,
        TWO_SAMPLER_PHI_WAVE32,
        SAMPLE_AND_STORE_WAVE32,
    ]
    .concat();
    let shader = build_sampler_test(&words, |_| {});
    assert_eq!(shader.samplers.len(), 2);
    run_sampler_test(&shader, &[(0, &[0.25]), (1, &[0.75])], false);
}

#[test]
fn a_flat_sampler_phi_selects_among_three_static_samplers() {
    let words = [
        SAMPLER_TEST_LOADS_WAVE32,
        FLAT_SAMPLER_PHI_WAVE32,
        SAMPLE_AND_STORE_WAVE32,
    ]
    .concat();
    let shader = build_sampler_test(&words, |sources| {
        let [Source::Phi { args, .. }] = sources else {
            panic!("expected one sampler phi");
        };
        assert_eq!(args.len(), 3);
        assert!(args.iter().all(|arg| matches!(arg, Source::Constant(_))));
    });
    assert_eq!(shader.samplers.len(), 3);
    run_sampler_test(&shader, &[(0, &[0.25]), (1, &[0.75]), (2, &[0.0])], false);
}

#[test]
fn a_nested_sampler_phi_selects_regular_and_comparison_samplers() {
    let words = [SAMPLER_TEST_LOADS_WAVE32, NESTED_SAMPLER_PHI_WAVE32].concat();
    let shader = build_sampler_test(&words, |sources| {
        assert_eq!(sources.len(), 2);
        for source in sources {
            let Source::Phi { args, .. } = source else {
                panic!("expected an outer sampler phi");
            };
            assert_eq!(args.len(), 2);
            assert_eq!(
                args.iter()
                    .filter(|arg| matches!(arg, Source::Constant(_)))
                    .count(),
                1
            );
            let inner = args.iter().find_map(|arg| match arg {
                Source::Phi { args, .. } => Some(&args[..]),
                _ => None,
            });
            assert!(matches!(
                inner,
                Some([Source::Constant(_), Source::Constant(_)])
            ));
        }
    });
    assert_eq!(shader.static_samplers.len(), 6);
    for word in [0x1000, 0x1092, 0x4036] {
        let comparison: Vec<_> = shader
            .samplers
            .iter()
            .zip(&shader.static_samplers)
            .filter(|&(&words, _)| words == [word, 0, 0, 0])
            .map(|(_, sampler)| sampler.filter & 0x80 != 0)
            .collect();
        assert_eq!(comparison, [false, true]);
    }
    run_sampler_test(
        &shader,
        &[(0, &[0.25, 0.0]), (1, &[0.75, 1.0]), (2, &[0.0, 1.0])],
        true,
    );
}
