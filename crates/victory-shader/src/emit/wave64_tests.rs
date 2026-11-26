//! A console wave64 compute program on a host of 32-lane waves: a wave reduction inside a
//! per-lane branch, with that branch taken wave-wide ([`crate::structure::structure_wide`])
//! so that the two host waves can exchange their halves inside it. Lowered, validated by
//! `dxil.dll` when present, and run on the host GPU when it takes waves of 32.

use std::collections::HashSet;

use super::host::{Dispatch, Host, Root, Stride, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// 64 threads in one console wave. Each thread t holds the bit `1 << (t >> 1)`; the threads
/// below s0 (a root constant) OR theirs over the wave inside `if (t < s0) { ... }` and store
/// the result to dword 0 of the raw UAV.
/// `skip`: the branch past the body when no thread is below s0: `s_cbranch_execz`, or
/// `s_cbranch_scc0` on the SCC `s_and_saveexec` sets ("some lane is left").
fn program(skip: u32) -> NativeProgram {
    let mut words = vec![
        0x7E0A_0281, // v_mov_b32 v5, 1
        0x2C0C_0081, // v_lshrrev_b32 v6, 1, v0
        0x340E_0B06, // v_lshlrev_b32 v7, v6, v5
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        skip | 23,   // the branch to the restore of EXEC
    ];
    words.extend_from_slice(&super::wave::WAVE_OR_V7_TO_S7);
    words.extend_from_slice(&[
        0x7E00_0207, // v_mov_b32 v0, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
    ]);
    native(&words)
}

/// A wave64 compute program of 64 threads (one console wave; the thread index in v0) with 4
/// root constants in s0..s3 and a raw UAV's descriptor in s[8:11], then `body`.
fn native(body: &[u32]) -> NativeProgram {
    native_of(body, true, 64)
}

/// The same for a console of 32-lane waves when `wave64` is not set, with `threads` threads.
fn native_of(body: &[u32], wave64: bool, threads: u32) -> NativeProgram {
    let mut words = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0x3D << 26 | 0x02 << 18 | 8 << 6 | 2,
        125 << 25, // s_load_dwordx4 s[8:11], s[4:5], 0
    ];
    words.extend_from_slice(body);
    words.push(S_ENDPGM);
    NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 16,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// `program()` for a host of 32-lane waves, with the branches closing `wide` taken wave-wide.
fn lowered(native: &NativeProgram, p: &ir::Program, wide: &HashSet<usize>) -> Result<HostShader, ShaderError> {
    lowered_on(native, p, wide, 32)
}

/// [`lowered`] for a host whose waves have at most `host_wave` lanes.
fn lowered_on(
    native: &NativeProgram,
    p: &ir::Program,
    wide: &HashSet<usize>,
    host_wave: u32,
) -> Result<HostShader, ShaderError> {
    let st = structure::structure_wide(p, wide).expect("structure");
    let res = resource::track(p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(p, native).expect("io::raise");
    let was = lower::TEST_HOST_WAVE.replace(Some(host_wave));
    let lowered = lower::lower(&lower::Parts {
        native,
        p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        native16: false,
    });
    lower::TEST_HOST_WAVE.set(was);
    let lowered = lowered?;
    Ok(HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: res.root_indexed_views,
    })
}

#[test]
fn a_sum_of_halves_over_the_wave_is_the_consoles_tree_bit_for_bit() {
    // thread t holds the 16-bit float with bits 0x3C00 + 37 t (1.0 and up, with odd
    // mantissas, so that additions round); all of them store the wave's sum
    let mut words = vec![
        0x340C_0085, // v_lshlrev_b32 v6, 5, v0
        0x340E_0082, // v_lshlrev_b32 v7, 2, v0
        0x4A0E_0F06, // v_add_nc_u32 v7, v6, v7
        0x4A0E_0F00, // v_add_nc_u32 v7, v0, v7
        0x4A0E_0EFF,
        0x0000_3C00, // v_add_nc_u32 v7, 0x3c00, v7
    ];
    // (the wave OR's words with v_add_f16 for v_or_b32: opcode 50 for 28)
    words.extend(super::wave::WAVE_OR_V7_TO_S7.iter().map(|&w| if w >> 25 == 28 { w & 0x01FF_FFFF | 50 << 25 } else { w }));
    words.extend_from_slice(&[
        0x7E00_0207, // v_mov_b32 v0, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ]);
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    assert_eq!(p.blocks.iter().flat_map(|b| &b.insts).filter(|i| i.name() == "v_add_f16").count(), 6);
    let shader = lowered(&native, &p, &HashSet::new()).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    // the console's sum: lanes 2k and 2k+1, then pairs, ... each addition rounded to 16 bits
    // (to nearest, ties to even); values here are normal numbers
    let wide = |h: u16| f32::from_bits((u32::from(h >> 10) + 112) << 23 | u32::from(h & 0x3FF) << 13);
    let narrow = |f: f32| {
        let b = f.to_bits();
        let h = ((b >> 23) - 112) << 10 | (b >> 13 & 0x3FF);
        let rest = b & 0x1FFF;
        (h + u32::from(rest > 0x1000 || rest == 0x1000 && h & 1 == 1)) as u16
    };
    let add = |a: u16, b: u16| narrow(wide(a) + wide(b));
    let terms: Vec<u16> = (0..64u16).map(|t| 0x3C00 + 37 * t).collect();
    let mut tree = terms.clone();
    while tree.len() > 1 {
        tree = tree.chunks(2).map(|c| add(c[0], c[1])).collect();
    }
    let in_a_row = terms.iter().copied().reduce(add).unwrap();
    assert_ne!(tree[0], in_a_row, "these terms tell the tree from another order");
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let (root, pso) = match host.compute_pipeline(&shader) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("the hardware device does not take this program (waves of 32?): {e}; not run");
            return;
        }
    };
    let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(1)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    // (the high half is lane 0's term's: zero)
    assert_eq!(out[0][0], u32::from(tree[0]), "tree {:#x}, in a row {in_a_row:#x}", tree[0]);
    eprintln!("ran on the hardware device: the sum {:#x} of the console's tree", tree[0]);
}

/// The console's 16-bit wave sum where the lanes at or past s0 have left the program (a
/// thread past the end of the data): on the console they add +0, and after each step of the
/// tree they hold their run's partial sum all the same. On a host of 64-lane waves (AMD) the
/// lanes that are here read each partner run's sum from a lane of it that is here.
#[test]
fn a_sum_of_halves_where_lanes_have_left_is_the_consoles_tree_bit_for_bit() {
    let mut words = vec![
        0x7DA8_0000, // v_cmpx_gt_u32 exec, s0, v0: the lanes below s0 stay
        0, // s_cbranch_execz to the end (filled in below)
        0x340C_0085, // v_lshlrev_b32 v6, 5, v0
        0x340E_0082, // v_lshlrev_b32 v7, 2, v0
        0x4A0E_0F06, // v_add_nc_u32 v7, v6, v7
        0x4A0E_0F00, // v_add_nc_u32 v7, v0, v7
        0x4A0E_0EFF,
        0x0000_3C00, // v_add_nc_u32 v7, 0x3c00, v7
    ];
    words.extend(super::wave::WAVE_OR_V7_TO_S7.iter().map(|&w| if w >> 25 == 28 { w & 0x01FF_FFFF | 50 << 25 } else { w }));
    words.extend_from_slice(&[
        0x7E00_0207, // v_mov_b32 v0, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ]);
    // past every word after it, to the program's end
    words[1] = 0xBF88_0000 | (words.len() as u32 - 2);
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let shader = lowered_on(&native, &p, &HashSet::new(), 64).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let wide = |h: u16| {
        if h == 0 {
            0.0
        } else {
            f32::from_bits((u32::from(h >> 10) + 112) << 23 | u32::from(h & 0x3FF) << 13)
        }
    };
    let narrow = |f: f32| {
        if f == 0.0 {
            return 0;
        }
        let b = f.to_bits();
        let h = ((b >> 23) - 112) << 10 | (b >> 13 & 0x3FF);
        let rest = b & 0x1FFF;
        (h + u32::from(rest > 0x1000 || rest == 0x1000 && h & 1 == 1)) as u16
    };
    let add = |a: u16, b: u16| narrow(wide(a) + wide(b));
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = false;
    for stay in [64u32, 63, 40, 33, 32, 17, 5, 1] {
        let terms: Vec<u16> = (0..64u32)
            .map(|t| if t < stay { (0x3C00 + 37 * t) as u16 } else { 0 })
            .collect();
        let mut tree = terms.clone();
        while tree.len() > 1 {
            tree = tree.chunks(2).map(|c| add(c[0], c[1])).collect();
        }
        let Some(got) = on_the_card(&shader, [stay, 0]) else {
            return;
        };
        ran = true;
        // (the high half is lane 0's term's: zero)
        assert_eq!(got, u32::from(tree[0]), "{stay} lanes stay: got {got:#x}, the console's tree {:#x}", tree[0]);
    }
    if ran {
        eprintln!("ran on the hardware device: the console's tree with lanes gone, in every case");
    }
}

/// One `v_add_f16` whose exact result lies between two halves: 1 + 3/1024 (0x3C03) plus 2
/// (0x4000) is 3 + 1.5 steps of that range, which the console rounds up to 0x4202. The
/// host's `f32tof16` may give 0x4201 (toward zero); with `VICTORY_F16_NEAREST=1` the
/// translation gives the console's whatever the host does.
#[test]
fn a_16_bit_sum_rounds_to_nearest_when_asked_to() {
    let words = [
        0x7E0E_02FF,
        0x0000_3C03, // v_mov_b32 v7, 0x3c03
        0x7E0C_02FF,
        0x0000_4000, // v_mov_b32 v6, 0x4000
        0x6400_0D07, // v_add_f16 v0, v7, v6
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ];
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    assert!(p.blocks.iter().flat_map(|b| &b.insts).any(|i| i.name() == "v_add_f16"));
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for nearest in [false, true] {
        let was = lower::TEST_F16_NEAREST.replace(Some(nearest));
        let shader = lowered(&native, &p, &HashSet::new());
        lower::TEST_F16_NEAREST.set(was);
        let shader = shader.expect("lower");
        if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
            let mut copy = shader.dxil.clone();
            v.validate(&mut copy).expect("DXIL validation failed");
        }
        let Ok(host) = Host::new(false) else {
            eprintln!("no SM 6.6 hardware device; not run");
            return;
        };
        let Ok((root, pso)) = host.compute_pipeline(&shader) else {
            eprintln!("the hardware device does not take this program; not run");
            return;
        };
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(1)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
        let got = out[0][0] & 0xFFFF;
        eprintln!("v_add_f16 0x3c03 + 0x4000 on the hardware device, nearest {nearest}: {got:#x}");
        if nearest {
            assert_eq!(got, 0x4202);
        } else {
            assert!(got == 0x4201 || got == 0x4202, "{got:#x}");
        }
    }
}

#[test]
fn a_wave_or_inside_a_lane_branch_needs_the_branch_wave_wide() {
    for (skip, name) in [(0xBF88_0000, "s_cbranch_execz"), (0xBF84_0000, "s_cbranch_scc0")] {
        a_wave_or_inside(skip, name);
    }
}

fn a_wave_or_inside(skip: u32, name: &str) {
    let native = program(skip);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let branch = (0..p.blocks.len())
        .find(|&b| p.blocks[b].insts.last().is_some_and(|i| i.name() == name))
        .expect("the branch");
    // per lane, the two host waves cannot meet inside the branch
    lower::take_needs();
    let per_lane = lowered(&native, &p, &HashSet::new());
    assert!(
        matches!(&per_lane, Err(ShaderError::Unsupported { what, .. }) if *what == "a wave reduction where not every lane runs"),
        "{:?}",
        per_lane.err()
    );
    // ... and the lowering names the block, around which that branch is the one to widen
    let needs = lower::take_needs();
    let st = structure::structure(&p).expect("structure");
    assert_eq!(lower::branches_around(&st, &needs), Some(vec![branch]));
    // (taken, past the body, when every lane's bit is clear)
    assert_eq!(st.wave_any[branch], Some(false));
    // wave-wide, every lane runs the reduction (EXEC stays a guard on what it does there)
    let wide = HashSet::from([branch]);
    let st = structure::structure_wide(&p, &wide).expect("structure");
    let inside = match p.blocks[branch].exit {
        ir::Exit::Branch { not_taken, .. } => not_taken,
        _ => unreachable!(),
    };
    assert!(st.guards[inside].iter().any(Option::is_some));
    let shader = lowered(&native, &p, &wide).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    // On a GPU that takes waves of 32 (two host waves for the console's one): the OR is of
    // both halves, whichever half's lanes take the branch.
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let (root, pso) = match host.compute_pipeline(&shader) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("the hardware device does not take this program (waves of 32?): {e}; not run");
            return;
        }
    };
    // (threads below s0 take the branch; thread t gives bit t >> 1)
    for (below, expected) in [(64u32, u32::MAX), (40, 0x000F_FFFF), (33, 0x0001_FFFF), (10, 0x1F), (1, 1)] {
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            if let (RootType::Constants, 0, 0) = (u.root, u.slot, u.dword) {
                ud[u.index as usize] = below;
            }
        }
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(1)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
        assert_eq!(out[0][0], expected, "{name}: {below} threads in the branch");
    }
    eprintln!("ran on the hardware device ({name}): the OR of both halves in every case");
}

/// Runs `shader` on the hardware device with the root constants `s0`, `s1` and gives dword 0
/// of the UAV; None (and a line) without a device that takes it.
fn on_the_card(shader: &HostShader, constants: [u32; 2]) -> Option<u32> {
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return None;
    };
    let Ok((root, pso)) = host.compute_pipeline(shader) else {
        eprintln!("the hardware device does not take this program; not run");
        return None;
    };
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        if let (RootType::Constants, 0, d @ 0..=1) = (u.root, u.slot, u.dword) {
            ud[u.index as usize] = constants[d as usize];
        }
    }
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(1)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    Some(out[0][0])
}

/// A 32-lane console wave. A register holds a lane mask on one path (the lanes below s0)
/// and a number on the other, and is read as a number after the join, as the trace programs
/// keep their lanes' masks and their node numbers on one stack
/// (`deferred_tiled:CS_LaserLightShadowRaytracing`, s46 at 0x104, 0x75c and 0x188). The
/// mask arrives as the ballot of its lanes.
#[test]
fn a_lane_mask_joined_with_a_number_is_read_as_the_number_it_is() {
    let words = [
        0x7D88_0000, // v_cmp_gt_u32 vcc_lo, s0, v0
        0xBE94_036A, // s_mov_b32 s20, vcc_lo
        0xBF06_8001, // s_cmp_eq_u32 s1, 0
        0xBF85_0001, // s_cbranch_scc1 over the next
        0xBE94_0385, // s_mov_b32 s20, 5
        0x7E00_0214, // v_mov_b32 v0, s20
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ];
    let native = native_of(&words, false, 32);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let shader = lowered(&native, &p, &HashSet::new()).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (below, other, expected) in [(32u32, 0u32, u32::MAX), (10, 0, 0x3FF), (0, 0, 0), (10, 1, 5)] {
        let Some(got) = on_the_card(&shader, [below, other]) else { return };
        assert_eq!(got, expected, "{below} lanes in the mask, s1 = {other}");
    }
    eprintln!("ran on the hardware device: the mask's bits, or the number");
}

/// A 32-lane console wave. `if (t < s0) { ... SCC = (s1 == 7) }` and a branch on SCC after
/// the join: the skip brings the SCC of `s_and_saveexec` ("some lane is left") along
/// (`rtdirectionalshadows:CS_RTShadowTrace32HQ_TiledFarOnly`, 0x360-0x39c). On the console
/// the skip is taken by the wave, so with some lane inside every lane sees the inner test.
/// A lane that skipped on its own would see "some lane is left" instead: refused per lane,
/// right with the branch taken wave-wide. Lane 31 stores what it saw.
#[test]
fn scc_of_a_mask_operation_joined_with_a_scalar_test_needs_the_skip_wave_wide() {
    let words = [
        0x7D88_0000, // v_cmp_gt_u32 vcc_lo, s0, v0
        0xBE8D_3C6A, // s_and_saveexec_b32 s13, vcc_lo
        0xBF88_0001, // s_cbranch_execz over the next
        0xBF06_8701, // s_cmp_eq_u32 s1, 7
        0xBEFE_030D, // s_mov_b32 exec_lo, s13
        0xBF85_0002, // s_cbranch_scc1 to "seven"
        0x7E0A_0281, // v_mov_b32 v5, 1
        0xBF82_0001, // s_branch over the next
        0x7E0A_0282, // seven: v_mov_b32 v5, 2
        0x7DA4_009F, // v_cmpx_eq_u32 exec_lo, 31, v0
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16 | 5 << 8, // buffer_store_dword v5, off, s[8:11], 0
    ];
    let native = native_of(&words, false, 32);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let branch = (0..p.blocks.len())
        .find(|&b| p.blocks[b].insts.last().is_some_and(|i| i.name() == "s_cbranch_execz"))
        .expect("the skip");
    lower::take_needs();
    let per_lane = lowered(&native, &p, &HashSet::new());
    assert!(
        matches!(&per_lane, Err(ShaderError::Unsupported { what, .. }) if *what == "a lane mask read as data"),
        "{:?}",
        per_lane.err()
    );
    let st = structure::structure(&p).expect("structure");
    assert_eq!(lower::branches_around(&st, &lower::take_needs()), Some(vec![branch]));
    let shader = lowered(&native, &p, &HashSet::from([branch])).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    // (lanes below s0 inside, s1) -> what lane 31 stores: 2 if SCC was set at the branch
    for (below, s1, expected) in [(0u32, 7u32, 1u32), (10, 7, 2), (10, 3, 1), (32, 3, 1), (32, 7, 2)] {
        let Some(got) = on_the_card(&shader, [below, s1]) else { return };
        assert_eq!(got, expected, "{below} lanes inside, s1 = {s1}");
    }
    eprintln!("ran on the hardware device: the inner test's SCC for every lane once some lane is inside");
}

/// Lowers `native` for a host of 32-lane waves and returns both the [`HostShader`] and
/// the lowering diagnostics (`flat_skips`, `lds_meetings`).
fn lowered_with_stats(
    native: &NativeProgram,
    p: &ir::Program,
    wide: &HashSet<usize>,
) -> Result<(HostShader, usize, u32), ShaderError> {
    let st = structure::structure_wide(p, wide).expect("structure");
    let res = resource::track(p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(p, native).expect("io::raise");
    let was = lower::TEST_HOST_WAVE.replace(Some(32));
    let l = lower::lower(&lower::Parts {
        native,
        p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        native16: false,
    });
    lower::TEST_HOST_WAVE.set(was);
    let l = l?;
    let flat_skips = l.flat_skips;
    let lds_meetings = l.lds_meetings;
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: l.dxil,
        user_data: l.user_data,
        samplers: l.samplers,
        static_samplers: Vec::new(),
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        root_indexed_views: res.root_indexed_views,
    };
    Ok((shader, flat_skips, lds_meetings))
}

/// Multi-wave (`group = 128 > 64`) console wave64 shader with a wave-varying scalar skip
/// (`s_bcnt1_i32_b64 -> s_cbranch_scc1`) around a pure single-block body that itself uses
/// wave exchange (`ShallowWaterSim:CS_AssembleComputeList` pattern). Flattening the skip
/// keeps the workgroup-wide exchange barriers uniform across both console waves while
/// preserving per-wave results via `select` at the join.
#[test]
fn multi_wave_pure_scalar_skip_around_exchange_is_flattened() {
    let words = [
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE82_106A, // s_bcnt1_i32_b64 s2, vcc
        0x7E08_0281, // v_mov_b32 v4, 1
        0xBF85_0004, // s_cbranch_scc1 +4 (skip pure body when any lane in this wave has v0 < s0)
        // Pure single-block body with wave exchange:
        0x7D84_0001, // v_cmp_eq_u32 vcc, s1, v0
        0xBE82_106A, // s_bcnt1_i32_b64 s2, vcc
        0x8502_8082, // s_cselect_b32 s2, 2, 0
        0x7E08_0202, // v_mov_b32 v4, s2
        // Join: each console wave writes its v4 to UAV[wave_id]
        0x2C0A_0086, // v_lshrrev_b32 v5, 6, v0
        0x340A_0A82, // v_lshlrev_b32 v5, 2, v5
        0xE070_1000,
        0x8002_0405, // buffer_store_dword v4, v5, s[8:11], 0 offen
    ];
    let native = native_of(&words, true, 128);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let (shader, flat_skips, _) =
        lowered_with_stats(&native, &p, &HashSet::new()).expect("lower");
    assert_eq!(flat_skips, 1, "wave-varying pure scalar skip must be flattened");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let Ok((root, pso)) = host.compute_pipeline(&shader) else {
        eprintln!("the hardware device does not take this program; not run");
        return;
    };
    // Wave 0 = lanes 0..63, Wave 1 = lanes 64..127.
    // Divergent case (s0=1, s1=100): Wave 0 skips body -> 1; Wave 1 enters body -> 2.
    for (s0, s1, expected) in [
        (1u32, 100u32, [1u32, 2u32]),
        (1, 999, [1, 0]),
        (0, 100, [0, 2]),
        (65, 100, [1, 1]),
    ] {
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            if let (RootType::Constants, 0, d @ 0..=1) = (u.root, u.slot, u.dword) {
                ud[u.index as usize] = [s0, s1][d as usize];
            }
        }
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(2)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
        assert_eq!([out[0][0], out[0][1]], expected, "s0={s0}, s1={s1}");
    }
    eprintln!("ran on the hardware device: flattened multi-wave scalar skip around exchange");
}

/// A multi-wave (`group = 128 > 64`) wave-varying scalar skip whose body has a side effect
/// (`buffer_store_dword`) cannot be flattened, so lowering must refuse placing a workgroup-wide
/// exchange barrier inside it rather than emitting a divergent barrier.
#[test]
fn multi_wave_side_effect_scalar_skip_around_exchange_refuses_divergent_barrier() {
    let words = [
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE82_106A, // s_bcnt1_i32_b64 s2, vcc
        0xBF85_0006, // s_cbranch_scc1 +6
        // Side-effectful body with wave exchange:
        0x7D84_0001, // v_cmp_eq_u32 vcc, s1, v0
        0xBE82_106A, // s_bcnt1_i32_b64 s2, vcc
        0x8502_8082, // s_cselect_b32 s2, 2, 0
        0x7E08_0202, // v_mov_b32 v4, s2
        0xE070_0000,
        0x8002_0400, // buffer_store_dword v4, off, s[8:11], 0
    ];
    let native = native_of(&words, true, 128);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let res = lowered(&native, &p, &HashSet::new());
    assert!(
        matches!(
            &res,
            Err(ShaderError::Unsupported { what, .. })
                if *what == "a lane mask exchanged where not every lane runs"
        ),
        "expected divergent multi-wave exchange refusal, got {:?}",
        res.err()
    );
}

/// Native single-wave Wave32 compute group (`threads = [32, 1, 1]`) sharing LDS across lanes
/// with no `s_barrier` (`ShallowWaterSim:CS_ApplyDisturb` pattern). Lowering must insert a
/// TGSM barrier on the write-to-read transition so cross-lane reads see sibling writes under
/// DXIL and independent thread scheduling.
#[test]
fn single_wave32_shared_lds_syncs_write_to_read() {
    let words = [
        0x4A02_0000, // v_add_nc_u32 v1, s0, v0
        0x3404_0082, // v_lshlrev_b32 v2, 2, v0
        0x36 << 26 | 0x0D << 18,
        1 << 8 | 2, // ds_write_b32 v2, v1 (no s_barrier after)
        0x4A06_0091, // v_add_nc_u32 v3, 17, v0
        0x3606_069F, // v_and_b32 v3, 31, v3
        0x3406_0682, // v_lshlrev_b32 v3, 2, v3
        0x36 << 26 | 0x36 << 18,
        4 << 24 | 3, // ds_read_b32 v4, v3
        0xE070_1000,
        0x8002_0402, // buffer_store_dword v4, v2, s[8:11], 0 offen
    ];
    let mut native = native_of(&words, false, 32);
    if let StageIo::Cs(cs) = &mut native.io {
        cs.rsrc2 |= 1 << 15; // 512 B LDS
    }
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let (shader, _, lds_meetings) =
        lowered_with_stats(&native, &p, &HashSet::new()).expect("lower");
    assert_eq!(lds_meetings, 1, "write->read transition in single-wave Wave32 LDS must emit 1 TGSM barrier");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let Ok((root, pso)) = host.compute_pipeline(&shader) else {
        eprintln!("the hardware device does not take this program; not run");
        return;
    };
    let base = 0x1234_0000u32;
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        if let (RootType::Constants, 0, 0) = (u.root, u.slot, u.dword) {
            ud[u.index as usize] = base;
        }
    }
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(32)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    for t in 0..32u32 {
        let expected = base + ((t + 17) & 31);
        assert_eq!(out[0][t as usize], expected, "lane {t}");
    }
    eprintln!("ran on the hardware device: single-wave Wave32 LDS cross-lane sync across all 32 lanes");
}

/// The real `ShallowWaterSim:CS_GenerateDisturbOverlappingPairs` container
/// (#25328ea424d08c00) end to end: two overlapping disturbers, one tile. The console
/// shader compacts the overlapping indices through wave scans and an atomic base and writes
/// the pairs buffer, a per-tile record, and the global counter. Any mistranslation of the
/// scan primitives corrupts the pairs list the disturber pass reads.
#[test]
fn generate_pairs_real_container_compacts_two_disturbers() {
    let bin = std::fs::read(
        r"C:\Users\Admen\.gemini\antigravity\brain\88a63322-5dbd-45fa-a99f-6948f2bcb3be\scratch\water_shaders\ShallowWaterSim_CS_GenerateDisturbOverlappingPairs_9b8019bf_Wrapped__0xd4b2119.bin",
    ).expect("the extracted shader container");
    let c = crate::dxbc::Container::parse(&bin).expect("container");
    let dxil = c.part(crate::dxbc::FourCC::DXIL).expect("DXIL part");
    let native = crate::native_programs(dxil.data)
        .expect("native programs")
        .into_iter()
        .find(|p| p.stage == HwStage::Cs)
        .expect("a compute program");
    eprintln!("wave64={} vgprs={} layout={} params", native.wave64, native.vgprs, native.layout.len());
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let shader = lowered(&native, &p, &HashSet::new()).expect("lower");
    for u in &shader.user_data {
        eprintln!("user_data index={} slot={} root={:?} dword={}", u.index, u.slot, u.root, u.dword);
    }
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let Ok((root, pso)) = host.compute_pipeline(&shader) else {
        eprintln!("the hardware device does not take this program; not run");
        return;
    };
    // table order in the heap: bbox SRV, pairs UAV, tile-record UAV, counter UAV, count CBV
    let f = |x: f32| x.to_bits();
    // every entry overlaps everything: whatever the self index register holds, the overlap
    // test passes for every i != self
    let bboxes: Vec<u32> = (0..4096)
        .flat_map(|_| [f(0.0), f(0.0), f(10.0), f(10.0)])
        .collect();
    let count_cbv: Vec<u32> = vec![8, 0, 0, 0];
    let views = [
        View::Buffer { data: &bboxes, stride: Stride::Raw },
        View::RawUav(160), // pairs
        View::RawUav(8),   // tile records
        View::RawUav(4),   // counter
        View::Cbv(&count_cbv),
    ];
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        if u.root == RootType::Table {
            ud[u.index as usize] = 0; // the table starts at heap index 0 (value = byte offset)
        }
    }
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &views,
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    eprintln!("counter: {:08x?}", &out[2][..4]);
    eprintln!("tile record: {:08x?}", &out[1][..8]);
    eprintln!("pairs[0..16]: {:08x?}", &out[0][..16]);
    // self is disturber 0, all of 1..7 overlap it: the wave compacts seven ids in order
    for k in 0..7u32 {
        assert_eq!(out[0][k as usize], k + 1, "pairs[{k}]");
    }
    assert_eq!(out[2][0], 7, "the counter after the wave adds its total");
    assert_eq!(out[1][0], 1, "the record's overlap flag");
    assert_eq!(out[1][2], 7, "the record's pair count");
}

/// Input-layout probe for `ShallowWaterSim:CS_ApplyDisturb` (#128424b3011dae43):
/// parses the real container, lowers it, and prints the whole user_data wiring so the
/// end-to-end harness can fill the root constants correctly.
#[test]
fn apply_disturb_input_layout_probe() {
    let bin = std::fs::read(
        r"C:\Users\Admen\.gemini\antigravity\brain\88a63322-5dbd-45fa-a99f-6948f2bcb3be\scratch\water_shaders\ShallowWaterSim_CS_ApplyDisturb_1cb9a44e_Wrapped__0xd4b8637.bin",
    ).expect("the extracted shader container");
    let c = crate::dxbc::Container::parse(&bin).expect("container");
    let dxil = c.part(crate::dxbc::FourCC::DXIL).expect("DXIL part");
    let native = crate::native_programs(dxil.data)
        .expect("native programs")
        .into_iter()
        .find(|p| p.stage == HwStage::Cs)
        .expect("a compute program");
    eprintln!("wave64={} vgprs={} layout_len={}", native.wave64, native.vgprs, native.layout.len());
    for e in &native.layout {
        eprintln!("layout: {:?}", e);
    }
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let shader = lowered(&native, &p, &HashSet::new()).expect("lower");
    eprintln!("user_data_dwords={}", shader.user_data_dwords());
    for u in &shader.user_data {
        eprintln!("user_data index={} slot={} root={:?} dword={}", u.index, u.slot, u.root, u.dword);
    }
}

/// Half-local cross-lane operations (`ds_bpermute_b32`, `ds_swizzle_b32` group-of-32,
/// `v_permlanex16_b32`, DPP8, and DPP16 rows including asymmetric `row_mask`) stay inside
/// their own 32-lane half, so a wave64 program lowers to 32-lane host waves and matches
/// RDNA2 lane-for-lane across all 64 lanes.
#[test]
fn half_local_cross_lane_ops_of_wave64_match_rdna2_on_a_wave32_host() {
    let lane_v1 = |l: u32| 0x1000_0001 + l;
    let lane_v2_old = |l: u32| 0xDEAD_0000 + l;
    let ref_wave64 = |src: fn(u32) -> u32| -> [u32; 64] {
        std::array::from_fn(|l| lane_v1(src(l as u32)))
    };
    let ref_wave64_even = |src: fn(u32) -> u32, inactive: fn(u32) -> u32| -> [u32; 64] {
        std::array::from_fn(|l| {
            let l = l as u32;
            if l & 1 != 0 {
                return lane_v2_old(l);
            }
            let s = src(l);
            if s & 1 == 0 { lane_v1(s) } else { inactive(l) }
        })
    };
    let dpp16 = |vdst: u8, vsrc0: u8, ctrl: u16, row_mask: u8, fi: bool| -> [u32; 2] {
        [
            0x7E00_02FA | (u32::from(vdst) << 17),
            (u32::from(row_mask) << 28)
                | (0xF << 24)
                | (u32::from(fi) << 18)
                | (u32::from(ctrl) << 8)
                | u32::from(vsrc0),
        ]
    };
    let dpp8_rev = |vdst: u8, vsrc0: u8| -> [u32; 2] {
        let sel: u32 = [7u32, 6, 5, 4, 3, 2, 1, 0]
            .iter()
            .enumerate()
            .fold(0, |a, (i, &x)| a | (x << (3 * i)));
        [
            0x7E00_02E9 | (u32::from(vdst) << 17),
            (sel << 8) | u32::from(vsrc0),
        ]
    };
    let ds_swizzle = |vdst: u8, vaddr: u8, offset: u16| -> [u32; 2] {
        [
            (0x36 << 26) | (0x35 << 18) | u32::from(offset),
            (u32::from(vdst) << 24) | u32::from(vaddr),
        ]
    };
    let ds_bpermute = |vdst: u8, vaddr: u8, vdata: u8, offset: u16| -> [u32; 2] {
        [
            (0x36 << 26) | (0xB3 << 18) | u32::from(offset),
            (u32::from(vdst) << 24) | (u32::from(vdata) << 8) | u32::from(vaddr),
        ]
    };
    let build = |op: &[u32], even: bool| -> HostShader {
        let mut w = vec![
            0x4A02_00FF,
            0x1000_0001, // v_add_nc_u32 v1, 0x1000_0001, v0
            0x4A04_00FF,
            0xDEAD_0000, // v_add_nc_u32 v2, 0xDEAD_0000, v0
            0x3406_0082, // v_lshlrev_b32 v3, 2, v0
        ];
        if even {
            w.extend_from_slice(&[
                0x3608_0081, // v_and_b32 v4, 1, v0
                0xBE8C_047E, // s_mov_b64 s[12:13], exec
                0x7DA4_0880, // v_cmpx_eq_u32 exec, 0, v4
            ]);
            w.extend_from_slice(op);
            w.push(0xBEFE_040C); // s_mov_b64 exec, s[12:13]
        } else {
            w.extend_from_slice(op);
        }
        w.extend_from_slice(&[
            0x340A_0082, // v_lshlrev_b32 v5, 2, v0
            0xE070_1000,
            0x8002_0205, // buffer_store_dword v2, v5, s[8:11], 0 offen
        ]);
        let native = native(&w);
        let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
        let shader = lowered_on(&native, &p, &HashSet::new(), 32).expect("lower");
        if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
            v.validate(&mut shader.dxil.clone())
                .expect("DXIL validation failed");
        }
        shader
    };

    let mut row0_only: [u32; 64] = std::array::from_fn(|l| lane_v2_old(l as u32));
    for l in 0..16 {
        row0_only[l] = if l == 0 { lane_v2_old(0) } else { lane_v1(l as u32 - 1) };
    }
    let mut row0_and_2: [u32; 64] = std::array::from_fn(|l| lane_v2_old(l as u32));
    for base in [0usize, 32] {
        for l in 0..16 {
            row0_and_2[base + l] = if l == 0 {
                lane_v2_old((base + l) as u32)
            } else {
                lane_v1((base + l - 1) as u32)
            };
        }
    }

    let items: [(&str, HostShader, [u32; 64]); 9] = [
        (
            "dpp16 row_ror:4",
            build(&dpp16(2, 1, 0x124, 0xF, true), false),
            ref_wave64(|l| (l & !15) | ((l + 12) & 15)),
        ),
        (
            "dpp16 row_mirror",
            build(&dpp16(2, 1, 0x140, 0xF, true), false),
            ref_wave64(|l| l ^ 15),
        ),
        (
            "dpp16 row_share:3",
            build(&dpp16(2, 1, 0x153, 0xF, true), false),
            ref_wave64(|l| (l & !15) | 3),
        ),
        (
            "dpp16 row_shr:1 row_mask:0x5",
            build(&dpp16(2, 1, 0x111, 0x5, false), false),
            row0_and_2,
        ),
        (
            "dpp16 row_shr:1 row_mask:0x1",
            build(&dpp16(2, 1, 0x111, 0x1, false), false),
            row0_only,
        ),
        (
            "dpp8 [7,6,5,4,3,2,1,0]",
            build(&dpp8_rev(2, 1), false),
            ref_wave64(|l| l ^ 7),
        ),
        (
            "ds_swizzle_b32 bitmode SWAP(16)",
            build(&ds_swizzle(2, 1, 0x401F), false),
            ref_wave64(|l| l ^ 16),
        ),
        (
            "ds_bpermute_b32 lane + 1 within the half",
            build(&ds_bpermute(2, 3, 1, 4), false),
            ref_wave64(|l| (l & 32) | ((l + 1) & 31)),
        ),
        (
            "ds_bpermute_b32 lane + 1, even lanes active (odd sources read 0)",
            build(&ds_bpermute(2, 3, 1, 4), true),
            ref_wave64_even(|l| (l & 32) | ((l + 1) & 31), |_| 0),
        ),
    ];

    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    for (label, shader, expected) in &items {
        let Ok((root, pso)) = host.compute_pipeline(shader) else {
            eprintln!("the hardware device does not take {label}; not run");
            return;
        };
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(64)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        assert_eq!(out[0][..64], expected[..], "{label}");
        eprintln!("[PASS] {label}");
    }
}
