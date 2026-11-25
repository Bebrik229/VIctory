//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So active records are copied and clamped
//! on the GPU into a device-local scratch buffer, just before execution, by a
//! compute shader of ours (HLSL compiled by the system's compiler; a 5.1
//! shader with its root signature inside), leaving the title's argument and
//! count buffers untouched.
//!
//! When a count buffer is given, threads at or past `min(max, count)` return
//! immediately without charging the list's budget or reading uninitialized
//! tail slots. Each list has a budget (2^27 vertex instances or threads,
//! groups counted as 64 threads, mesh groups as 128): active records past it
//! run empty, saturating without 32-bit wraparound so a list of thousands of
//! garbage records cannot run for seconds either.
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension and 2^20 per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use crate::host::{HostList, host, transition};

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=7, b0), SRV(t0), SRV(t1), UAV(u0), UAV(u1)"
cbuffer Record : register(b0) {
    uint kind;
    uint src_stride;
    uint records;
    uint args_offset;
    // Bit 0: a count buffer bounds the records.
    // Bit 1: the records are a window clamped ahead of their draws: one that
    //        asks for more than can run is left empty and costs nothing.
    // Bit 2: record diagnostic group stats into budget[1..5].
    // Bit 3: gathered records; record i reads its source byte offset from
    //        count_buf[count_offset + i * 4].
    uint has_count;
    uint count_offset;
    uint copy_bytes;
};
ByteAddressBuffer src_args : register(t0);
ByteAddressBuffer count_buf : register(t1);
RWByteAddressBuffer dst_args : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
static const uint CAP = BUDGET + 1u;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
groupshared uint s_cost[64];
groupshared uint s_base;
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint gtid : SV_GroupIndex) {
    uint eff_count = records;
    bool ahead = (has_count & 2u) != 0;
    bool stats = (has_count & 4u) != 0;
    if ((has_count & 1u) != 0) {
        eff_count = min(eff_count, count_buf.Load(count_offset));
    }
    if (gid.x * 64u >= eff_count) return;
    bool active = id.x < eff_count;
    uint src_at = args_offset + id.x * src_stride;
    // Branch avoids reading count_buf when not gathering (it may hold src itself).
    [branch]
    if ((has_count & 8u) != 0) {
        src_at = count_buf.Load(count_offset + id.x * 4u);
    }
    uint dst_at = id.x * copy_bytes;
    uint cost = 0u;
    uint2 draw_counts = uint2(0, 0);
    uint3 groups = uint3(0, 0, 0);
    if (active) {
        if (kind <= 1) {
            // Draws: vertex or index count, then instances.
            uint2 asked = src_args.Load2(src_at);
            if (ahead && (asked.x > (1u << 24) || asked.y > (1u << 24))) {
                asked = uint2(0, 0);
            }
            uint count = min(asked.x, 1u << 24);
            uint instances = min(asked.y, (1u << 26) / max(count, 1u));
            cost = count * instances;
            draw_counts = uint2(count, instances);
        } else if (kind == 2) {
            uint3 asked = src_args.Load3(src_at);
            if (stats && asked.x <= (1u << 24) && asked.y <= 64u && asked.z <= 64u) {
                uint noted;
                budget.InterlockedMax(12, asked.x, noted);
                if (asked.x > 65535u) {
                    budget.InterlockedAdd(16, 1u, noted);
                }
            }
            uint3 g = min(asked, uint3(65535u, 65535u, 65535u));
            // x * y fits 32 bits; the whole count may not.
            if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
                g = uint3(0, 0, 0);
            }
            cost = g.x * g.y * g.z * 64u;
            groups = g;
        } else {
            // Mesh dispatches: the console's are one-dimensional.
            uint3 g = src_args.Load3(src_at);
            // Ahead of its draw, a record over the most a dispatch runs is a
            // slot the title has not filled (the title leaves all
            // ones there): empty, and not counted below either.
            if (ahead && g.x > 65535u) {
                g = uint3(0, 0, 0);
            }
            if (stats && g.x > 0u) {
                uint was;
                budget.InterlockedMax(4, g.x, was);
                budget.InterlockedAdd(8, min(g.x, 1u << 24), was);
            }
            g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
            cost = g.x * g.y * g.z * 128u;
            groups = g;
        }
    }
    // Saturating inclusive prefix sum over the 64-thread group:
    // collapses up to 64 serialized CAS loops into 1 atomic per group.
    s_cost[gtid] = min(cost, CAP);
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint d = 1u; d < 64u; d <<= 1u) {
        uint prev = (gtid >= d) ? s_cost[gtid - d] : 0u;
        GroupMemoryBarrierWithGroupSync();
        s_cost[gtid] = min(s_cost[gtid] + prev, CAP);
        GroupMemoryBarrierWithGroupSync();
    }
    uint incl = s_cost[gtid];
    if (gtid == 0u) {
        uint total = min(s_cost[63], BUDGET);
        uint before = 0u;
        if (total > 0u) {
            budget.InterlockedOr(0, 0, before);
            [allow_uav_condition]
            while (before < BUDGET) {
                uint next = (BUDGET - before <= total) ? BUDGET : (before + total);
                uint prev;
                budget.InterlockedCompareExchange(0, before, next, prev);
                if (prev == before) break;
                before = prev;
            }
        }
        s_base = before;
    }
    GroupMemoryBarrierWithGroupSync();
    if (!active) return;
    bool over = (cost > 0u) && (s_base >= BUDGET || (BUDGET - s_base) < incl);
    if (kind <= 1) {
        if (over) {
            draw_counts = uint2(0, 0);
        }
        dst_args.Store2(dst_at, draw_counts);
        for (uint off = 8; off < copy_bytes; off += 4) {
            dst_args.Store(dst_at + off, src_args.Load(src_at + off));
        }
    } else {
        if (over) {
            groups = uint3(0, 0, 0);
        }
        dst_args.Store3(dst_at, groups);
        for (uint off = 12; off < copy_bytes; off += 4) {
            dst_args.Store(dst_at + off, src_args.Load(src_at + off));
        }
    }
}
"#;

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

fn make() -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    make_for(device)
}

fn make_for(device: &ID3D12Device) -> windows::core::Result<Clamp> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            s!("clamp"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                )
            };
            tracing::warn!("argument clamp: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes =
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Clamp { root, pso })
    }
}

/// Minimum record byte stride for a supported PC indirect argument `kind`.
fn min_stride(kind: u32) -> Option<u32> {
    match kind {
        0 => Some(16),
        1 => Some(20),
        2 | 10 => Some(12),
        _ => None,
    }
}

/// Validates record alignment and returns `(src_bytes, dst_bytes)`.
fn validate_layout_packed(
    (kind, copy_bytes, src_stride, records): (u32, u32, u32, u32),
    src_offset: u64,
    count_offset: Option<u64>,
) -> Option<(u64, u64)> {
    let min = min_stride(kind)?;
    if copy_bytes < min
        || src_stride < copy_bytes
        || !copy_bytes.is_multiple_of(4)
        || !src_stride.is_multiple_of(4)
        || !src_offset.is_multiple_of(4)
    {
        return None;
    }
    if count_offset.is_some_and(|off| !off.is_multiple_of(4)) {
        return None;
    }
    let src_bytes = if records == 0 {
        0
    } else {
        u64::from(records - 1)
            .checked_mul(u64::from(src_stride))?
            .checked_add(u64::from(copy_bytes))?
    };
    let dst_bytes = u64::from(copy_bytes).checked_mul(u64::from(records))?;
    (src_bytes <= u64::from(u32::MAX) && dst_bytes <= u64::from(u32::MAX))
        .then_some((src_bytes, dst_bytes))
}

#[cfg(test)]
fn validate_layout(
    (kind, stride, records): (u32, u32, u32),
    src_offset: u64,
    count_offset: Option<u64>,
) -> Option<u64> {
    validate_layout_packed((kind, stride, stride, records), src_offset, count_offset).map(|(_, d)| d)
}

/// Records, in `list`, the non-destructive clamp of up to `records` records of
/// `kind` (PC argument numbering) `stride` bytes apart from `src` (and bounded
/// by the 32-bit count in `count` when given) into a device-local scratch
/// buffer, leaving that scratch buffer in `INDIRECT_ARGUMENT` at offset `0`.
/// Replaces the list's compute root signature and pipeline (callers bind
/// theirs after). `None` if the clamp, alignment or buffer bounds fail.
pub(crate) fn bound(
    list: &HostList,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32),
) -> Option<ID3D12Resource> {
    let (kind, stride, records) = spec;
    let (_, dst_bytes) = validate_layout_packed((kind, stride, stride, records), src.1, count.map(|(_, off)| off))?;
    let (c, (budget_res, budget), (dst, was_in_indirect)) =
        (clamp()?, list.budget()?, list.indirect_scratch(dst_bytes)?);
    let ok = record_clamp_full(
        c,
        &list.list,
        &dst,
        was_in_indirect,
        Some(&budget_res),
        budget,
        src,
        count,
        (kind, stride, stride, records),
        false,
        HostList::indirect_stats(),
        false,
    );
    if ok {
        list.mark_indirect_ready();
        Some(dst)
    } else {
        None
    }
}

/// [`bound`] for a window of records clamped ahead of the draws that will
/// read them (`crate::draw`'s windows; no count buffer). Source records lie
/// `src_stride` bytes apart; each record's `copy_bytes` are packed tightly
/// into the scratch buffer (`dst_at = id.x * copy_bytes`). When `chained` is
/// true (advancing directly from a prior window with no intervening compute or
/// storing draw), uses a targeted UAV barrier on `budget` instead of a global
/// pipeline drain.
pub(crate) fn bound_ahead(
    list: &HostList,
    src: (&ID3D12Resource, u64),
    spec: (u32, u32, u32, u32),
    chained: bool,
) -> Option<ID3D12Resource> {
    let (_, dst_bytes) = validate_layout_packed(spec, src.1, None)?;
    let (c, (budget_res, budget), (dst, was_in_indirect)) =
        (clamp()?, list.budget()?, list.indirect_scratch(dst_bytes)?);
    let ok = record_clamp_full(
        c,
        &list.list,
        &dst,
        was_in_indirect,
        Some(&budget_res),
        budget,
        src,
        None,
        spec,
        true,
        HostList::indirect_stats(),
        chained,
    );
    if ok {
        list.mark_indirect_ready();
        Some(dst)
    } else {
        None
    }
}

#[cfg(test)]
fn record_clamp(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    dst: &ID3D12Resource,
    budget: u64,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32),
) -> bool {
    record_clamp_as(c, l, dst, budget, src, count, spec, false)
}

/// Test entry point preserving full-stride copy and diagnostic stats.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn record_clamp_as(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    dst: &ID3D12Resource,
    budget: u64,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    (kind, stride, records): (u32, u32, u32),
    ahead: bool,
) -> bool {
    record_clamp_full(
        c,
        l,
        dst,
        false,
        None,
        budget,
        src,
        count,
        (kind, stride, stride, records),
        ahead,
        true,
        false,
    )
}

struct ClampBindings {
    dst_bytes: u64,
    src_gpu: u64,
    count_gpu: u64,
    records: u32,
    constants: [u32; 7],
}

/// Validates layout alignment, source/count buffer widths, and 32-bit byte
/// offsets (rebasing root SRV GPU VAs above 4 GiB) before binding the clamp
/// compute shader.
fn prepare_clamp_bindings(
    (src, src_offset): (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    (kind, copy_bytes, src_stride, records): (u32, u32, u32, u32),
    ahead: bool,
    stats: bool,
) -> Option<ClampBindings> {
    let (src_bytes, dst_bytes) = validate_layout_packed(
        (kind, copy_bytes, src_stride, records),
        src_offset,
        count.map(|(_, off)| off),
    )?;
    // SAFETY: live D3D12 buffer resources. Root descriptors have no hardware
    // bounds check in D3D12, so verify actual buffer widths before binding.
    let (src_width, src_base) = unsafe { (src.GetDesc().Width, src.GetGPUVirtualAddress()) };
    if !src_base.is_multiple_of(4) || src_offset.checked_add(src_bytes).is_none_or(|end| end > src_width) {
        return None;
    }
    let (src_gpu, args_offset) = match src_offset
        .checked_add(src_bytes)
        .filter(|&end| end <= u64::from(u32::MAX))
    {
        Some(_) => (src_base, src_offset as u32),
        None => (src_base.checked_add(src_offset)?, 0),
    };
    let (count_gpu, has_count, count_offset) = match count {
        Some((buf, off)) => {
            // SAFETY: a live D3D12 buffer resource.
            let (count_width, base) = unsafe { (buf.GetDesc().Width, buf.GetGPUVirtualAddress()) };
            if !base.is_multiple_of(4) || off.checked_add(4).is_none_or(|end| end > count_width) {
                return None;
            }
            match off.checked_add(4).filter(|&end| end <= u64::from(u32::MAX)) {
                Some(_) => (base, 1u32, off as u32),
                None => (base.checked_add(off)?, 1u32, 0u32),
            }
        }
        None => (src_gpu, 0u32, 0u32),
    };
    if !src_gpu.is_multiple_of(4) || !count_gpu.is_multiple_of(4) {
        return None;
    }
    let flags = has_count | (u32::from(ahead) << 1) | (u32::from(stats) << 2);
    let constants = [kind, src_stride, records, args_offset, flags, count_offset, copy_bytes];
    Some(ClampBindings {
        dst_bytes,
        src_gpu,
        count_gpu,
        records,
        constants,
    })
}

#[allow(clippy::too_many_arguments)]
fn record_clamp_full(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    dst: &ID3D12Resource,
    was_in_indirect: bool,
    budget_res: Option<&ID3D12Resource>,
    budget: u64,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32, u32),
    ahead: bool,
    stats: bool,
    chained: bool,
) -> bool {
    let Some(b) = prepare_clamp_bindings(src, count, spec, ahead, stats) else {
        return false;
    };
    // SAFETY: live D3D12 buffer resource.
    let (dst_width, dst_gpu) = unsafe { (dst.GetDesc().Width, dst.GetGPUVirtualAddress()) };
    if !dst_gpu.is_multiple_of(4) || !budget.is_multiple_of(4) || b.dst_bytes > dst_width {
        return false;
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::IndirectBounded, 1);
    #[cfg(feature = "diagnostics")]
    let waits = crate::copystat::barrier_test() & 2 == 0;
    #[cfg(not(feature = "diagnostics"))]
    let waits = true;
    let before_state = if was_in_indirect {
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT
    } else {
        D3D12_RESOURCE_STATE_COMMON
    };
    let uav_wait = if chained {
        uav_barrier_for(budget_res)
    } else {
        uav_barrier()
    };
    let first = [
        uav_wait,
        transition(
            dst,
            before_state,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        ),
    ];
    let second = [transition(
        dst,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
    )];
    let skip = usize::from(!waits);
    unsafe {
        // Earlier shader writes (or just `budget` on chained windows) first.
        l.ResourceBarrier(&first[skip..]);
        if b.records > 0 {
            l.SetComputeRootSignature(&c.root);
            l.SetPipelineState(&c.pso);
            l.SetComputeRoot32BitConstants(0, 7, b.constants.as_ptr().cast(), 0);
            l.SetComputeRootShaderResourceView(1, b.src_gpu);
            l.SetComputeRootShaderResourceView(2, b.count_gpu);
            l.SetComputeRootUnorderedAccessView(3, dst_gpu);
            l.SetComputeRootUnorderedAccessView(4, budget);
            l.Dispatch(b.records.div_ceil(64), 1, 1);
        }
        // `UAV -> INDIRECT_ARGUMENT` already orders `dst` for ExecuteIndirect.
        l.ResourceBarrier(&second);
    }
    true
}

/// Maximum scattered records gathered per argument buffer in one list (multiple of 64).
pub(crate) const GATHER_MOST: u32 = 4096;
/// Maximum byte stride supported for gathered indirect records.
pub(crate) const GATHER_STRIDE: u32 = 32;

/// Immutable snapshot of one [`GatherSet`] handed to `PlannedPrelude::record`.
#[derive(Clone)]
pub(crate) struct GatherPlan {
    pub(crate) src: ID3D12Resource,
    pub(crate) scratch: ID3D12Resource,
    pub(crate) table_gpu: u64,
    pub(crate) kind: u32,
    pub(crate) stride: u32,
    pub(crate) records: u32,
}

// SAFETY: D3D12 resources and immutable scalars are safe to send to a prelude worker.
unsafe impl Send for GatherPlan {}

/// Scattered indirect records of one `(src_id, kind, stride)` collected during list recording
/// and clamped together in a single compute dispatch inside the list's prelude.
pub(crate) struct GatherSet {
    pub(crate) src: ID3D12Resource,
    pub(crate) src_id: u64,
    pub(crate) kind: u32,
    pub(crate) stride: u32,
    pub(crate) scratch: ID3D12Resource,
    pub(crate) table_gpu: u64,
    table: usize,
    slots: std::collections::HashMap<u32, u32>,
}

// SAFETY: D3D12 objects are free-threaded; `table` points into the owning list's upload scratch.
unsafe impl Send for GatherSet {}

impl GatherSet {
    pub(crate) fn new(
        src: ID3D12Resource,
        src_id: u64,
        (kind, stride): (u32, u32),
        scratch: ID3D12Resource,
        (table_gpu, table): (u64, *mut u8),
    ) -> Self {
        Self {
            src,
            src_id,
            kind,
            stride,
            scratch,
            table_gpu,
            table: table as usize,
            slots: std::collections::HashMap::new(),
        }
    }

    pub(crate) fn records(&self) -> u32 {
        self.slots.len() as u32
    }

    pub(crate) fn slot(&mut self, offset: u32) -> Option<u32> {
        if let Some(&slot) = self.slots.get(&offset) {
            return Some(slot);
        }
        let slot = self.slots.len() as u32;
        if slot >= GATHER_MOST {
            return None;
        }
        // SAFETY: `table` points to `GATHER_MOST` mapped `u32` entries in list upload memory.
        unsafe { (self.table as *mut u32).add(slot as usize).write_volatile(offset) };
        self.slots.insert(offset, slot);
        Some(slot)
    }

    pub(crate) fn into_scratch(self) -> ID3D12Resource {
        self.scratch
    }

    pub(crate) fn plan(&self) -> GatherPlan {
        GatherPlan {
            src: self.src.clone(),
            scratch: self.scratch.clone(),
            table_gpu: self.table_gpu,
            kind: self.kind,
            stride: self.stride,
            records: self.records(),
        }
    }
}

/// Validates whether a record of `(kind, stride)` at `offset` in a buffer at `src_gpu`
/// fits the scattered-gather clamp constraints.
pub(crate) fn gatherable(src_gpu: u64, offset: u64, (kind, stride): (u32, u32)) -> Option<u32> {
    let min = min_stride(kind)?;
    if stride < min
        || stride > GATHER_STRIDE
        || !stride.is_multiple_of(4)
        || !offset.is_multiple_of(4)
        || !src_gpu.is_multiple_of(4)
    {
        return None;
    }
    u32::try_from(offset).ok().filter(|o| o.checked_add(stride).is_some())
}

/// Records batched gather-clamp dispatches into a prelude list `l`, charging `budget`
/// and leaving every populated `GatherPlan::scratch` in `INDIRECT_ARGUMENT` state.
pub(crate) fn record_gathers(
    l: &ID3D12GraphicsCommandList,
    plans: &[GatherPlan],
    budget: &ID3D12Resource,
) -> bool {
    use windows::core::Interface;
    let (Some(c), Ok(l2)) = (clamp(), l.cast::<ID3D12GraphicsCommandList2>()) else {
        return false;
    };
    record_gathers_with(c, l, &l2, plans, budget)
}

fn record_gathers_with(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    l2: &ID3D12GraphicsCommandList2,
    plans: &[GatherPlan],
    budget: &ID3D12Resource,
) -> bool {
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_WRITEBUFFERIMMEDIATE_PARAMETER,
    };
    // SAFETY: live D3D12 buffer resource.
    let budget_gpu = unsafe { budget.GetGPUVirtualAddress() };
    if !budget_gpu.is_multiple_of(4) {
        return false;
    }
    let zeros = [0u64, 4, 8, 12, 16].map(|off| D3D12_WRITEBUFFERIMMEDIATE_PARAMETER {
        Dest: budget_gpu + off,
        Value: 0,
    });
    // SAFETY: open command list; `budget` is `COMMON` between list runs.
    unsafe {
        l.ResourceBarrier(&[transition(
            budget,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )]);
        l2.WriteBufferImmediate(zeros.len() as u32, zeros.as_ptr(), None);
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(
                budget,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            ),
        ]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRootUnorderedAccessView(4, budget_gpu);
    }
    let mut all = true;
    let mut ran_any = false;
    for plan in plans.iter().filter(|p| p.records > 0) {
        // SAFETY: live D3D12 buffer resources.
        let (src_gpu, dst_gpu) = unsafe {
            (plan.src.GetGPUVirtualAddress(), plan.scratch.GetGPUVirtualAddress())
        };
        if gatherable(src_gpu, 0, (plan.kind, plan.stride)).is_none()
            || plan.records > GATHER_MOST
            || !dst_gpu.is_multiple_of(4)
            || !plan.table_gpu.is_multiple_of(4)
        {
            all = false;
            continue;
        }
        let constants = [plan.kind, plan.stride, plan.records, 0u32, 8u32, 0u32, plan.stride];
        let pre = [
            if ran_any {
                uav_barrier_for(Some(budget))
            } else {
                uav_barrier_for(None)
            },
            transition(
                &plan.scratch,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            ),
        ];
        let post = [transition(
            &plan.scratch,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        )];
        // SAFETY: open command list; validated root SRVs/UAVs and constants.
        unsafe {
            let start = usize::from(!ran_any);
            l.ResourceBarrier(&pre[start..]);
            l.SetComputeRoot32BitConstants(0, 7, constants.as_ptr().cast(), 0);
            l.SetComputeRootShaderResourceView(1, src_gpu);
            l.SetComputeRootShaderResourceView(2, plan.table_gpu);
            l.SetComputeRootUnorderedAccessView(3, dst_gpu);
            l.Dispatch(plan.records.div_ceil(64), 1, 1);
            l.ResourceBarrier(&post);
        }
        ran_any = true;
    }
    // SAFETY: return `budget` to `COMMON` before the main list starts.
    unsafe {
        l.ResourceBarrier(&[transition(
            budget,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        )]);
    }
    all
}

/// Parses `VICTORY_HOIST_CLAMPS` (`0..=7`), defaulting to mode `7`.
pub(crate) fn parse_hoist_mode(raw: Option<&str>) -> u8 {
    raw.and_then(|v| v.trim().parse::<u8>().ok())
        .filter(|&n| n < 8)
        .unwrap_or(7)
}

/// Whether hoist `mode` (`0..=7`) hoists indirect clamps for `kind`
/// (`0` draw, `1` indexed draw, `2` compute dispatch, `10` mesh dispatch).
pub(crate) fn hoist_allows(mode: u8, kind: u32) -> bool {
    const DISPATCH: u32 = 2;
    const MESH: u32 = 10;
    kind != DISPATCH
        && match mode {
            0 | 6 => false,
            3 => kind == MESH,
            4 => kind != MESH,
            _ => true,
        }
}

/// How clamps of indirect draws are recorded:
/// - `0`: each inline before its draw (no list cut);
/// - `1`: hoisted into a pre-list before the draw list, without barriers
///   between clamps;
/// - `2`: as `1`, with a UAV barrier before each hoisted draw in the draw list;
/// - `3`: as `1`, for mesh dispatches only;
/// - `4`: as `1`, for non-mesh draws only;
/// - `5`: as `1`, always allocating a new list for clamps;
/// - `6`: lists cut as for `1`, with each clamp still inline before its draw;
/// - `7`: as `1`, with a UAV barrier on `budget` between hoisted clamps so
///   records past the list's budget stay deterministic in draw order (the
///   default; prevents mesh object flicker).
///
/// `VICTORY_HOIST_CLAMPS=<n>` sets it (default `7`; `0` disables hoisting);
/// diagnostics builds also poll `VICTORY_HOIST_FILE` every 500 ms if set.
pub(crate) fn hoist_mode() -> u8 {
    use std::sync::atomic::{AtomicU8, Ordering};
    static MODE: AtomicU8 = AtomicU8::new(u8::MAX);
    let mode = MODE.load(Ordering::Relaxed);
    if mode == u8::MAX {
        let set = parse_hoist_mode(std::env::var("VICTORY_HOIST_CLAMPS").ok().as_deref());
        if MODE
            .compare_exchange(u8::MAX, set, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
            && cfg!(feature = "diagnostics")
            && let Some(file) = std::env::var_os("VICTORY_HOIST_FILE")
        {
            let _ = std::thread::Builder::new()
                .name("victory-hoist".into())
                .spawn(move || {
                    loop {
                        std::thread::sleep(std::time::Duration::from_millis(500));
                        let read = std::fs::read_to_string(&file)
                            .ok()
                            .and_then(|t| t.trim().parse::<u8>().ok());
                        let cuts = CUTS.each_ref().map(|c| c.swap(0, Ordering::Relaxed));
                        if cuts != [0, 0] {
                            tracing::info!(
                                cuts = cuts[0],
                                queries = cuts[1],
                                "lists cut for hoisted clamps"
                            );
                        }
                        if let Some(n) = read.filter(|&n| n < 8 && n != MODE.load(Ordering::Relaxed)) {
                            MODE.store(n, Ordering::Relaxed);
                            tracing::info!(mode = n, "clamps of indirect draws: mode changed");
                        }
                    }
                });
        }
        return MODE.load(Ordering::Relaxed);
    }
    mode
}

/// Cuts made for hoisted clamps, and those skipped for open queries.
pub(crate) static CUTS: [std::sync::atomic::AtomicU64; 2] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// Whether clamps of indirect draws go to a pre-list before the draw list.
pub(crate) fn hoisting() -> bool {
    hoist_mode() != 0
}

/// As [`bound`] / [`bound_ahead`], recorded in `clamps`, a pre-list that runs
/// before the list executing the indirect draws. Returns `(chunk_buffer, offset)`
/// readable as indirect arguments once `clamps.clamp_end()` has run.
///
/// `spec` is `(kind, copy_bytes, src_stride, records)`. When `ahead` is true
/// (`VICTORY_INDIRECT_WINDOW` composing with hoisted clamps), unfilled garbage
/// slots are left empty and charge nothing, and records `src_stride` bytes
/// apart are packed tightly at `copy_bytes` intervals starting at `offset`.
pub(crate) fn bound_before(
    clamps: &HostList,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32, u32),
    ahead: bool,
) -> Option<(ID3D12Resource, u64)> {
    if spec.3 == 0 {
        return None;
    }
    let b = prepare_clamp_bindings(src, count, spec, ahead, HostList::indirect_stats())?;
    let (c, (budget_res, budget)) = (clamp()?, clamps.budget()?);
    if !budget.is_multiple_of(4) {
        return None;
    }
    let (dst, at) = clamps.clamp_room(b.dst_bytes)?;
    // SAFETY: live D3D12 buffer resource.
    let (dst_width, dst_base) = unsafe { (dst.GetDesc().Width, dst.GetGPUVirtualAddress()) };
    let dst_gpu = dst_base.checked_add(at)?;
    if !dst_gpu.is_multiple_of(4) || at.checked_add(b.dst_bytes).is_none_or(|end| end > dst_width) {
        return None;
    }
    record_hoisted_clamp(
        c,
        &clamps.list,
        &b,
        dst_gpu,
        &budget_res,
        budget,
        hoist_mode() == 7,
    );
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::IndirectBounded, 1);
    Some((dst, at))
}

fn record_hoisted_clamp(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    b: &ClampBindings,
    dst_gpu: u64,
    budget_res: &ID3D12Resource,
    budget: u64,
    ordered: bool,
) {
    // SAFETY: an open list; our root signature and pipeline; live source,
    // count, destination (writable: `clamp_room`) and budget buffers.
    unsafe {
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 7, b.constants.as_ptr().cast(), 0);
        l.SetComputeRootShaderResourceView(1, b.src_gpu);
        l.SetComputeRootShaderResourceView(2, b.count_gpu);
        l.SetComputeRootUnorderedAccessView(3, dst_gpu);
        l.SetComputeRootUnorderedAccessView(4, budget);
        l.Dispatch(b.records.div_ceil(64), 1, 1);
        // Mode 7: serialize clamps on `budget` so records past the list's
        // budget are charged deterministically in draw order (no flicker).
        if ordered {
            l.ResourceBarrier(&[uav_barrier_for(Some(budget_res))]);
        }
    }
}

/// Returns the scratch buffer to `COMMON` if it is currently in
/// `INDIRECT_ARGUMENT` (used by unit tests; production lists transition once
/// at `HostList::close()`).
#[cfg(test)]
pub(crate) fn done(list: &HostList, scratch: &ID3D12Resource) {
    if list.take_indirect_ready() {
        done_on(&list.list, scratch);
    }
}

#[cfg(test)]
fn done_on(list: &ID3D12GraphicsCommandList, scratch: &ID3D12Resource) {
    // SAFETY: an open list and a live scratch buffer in INDIRECT_ARGUMENT.
    unsafe {
        list.ResourceBarrier(&[transition(
            scratch,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_COMMON,
        )])
    };
}

pub(crate) fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    uav_barrier_for(None)
}

fn uav_barrier_for(resource: Option<&ID3D12Resource>) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                // SAFETY: borrowed COM pointer; the barrier never releases it.
                pResource: match resource {
                    Some(r) => unsafe { std::mem::transmute_copy(r) },
                    None => std::mem::ManuallyDrop::new(None),
                },
            }),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::Win32::Graphics::Dxgi::*;
    use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
    use windows::core::Interface;

    const WORKER_CS: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer hits : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint prev;
    hits.InterlockedAdd(0, 1, prev);
}
"#;

    struct WarpHarness {
        device: ID3D12Device,
        queue: ID3D12CommandQueue,
        allocator: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
        fence: ID3D12Fence,
        submitted: u64,
        event: HANDLE,
        clamp: Clamp,
        worker: Clamp,
    }

    impl WarpHarness {
        fn new() -> Option<Self> {
            // SAFETY: plain DXGI WARP query and D3D12 test harness setup.
            unsafe {
                let factory: IDXGIFactory4 =
                    CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
                let adapter: IDXGIAdapter = factory.EnumWarpAdapter().ok()?;
                let desc = adapter.cast::<IDXGIAdapter1>().ok()?.GetDesc1().ok()?;
                if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
                    return None;
                }
                let mut device: Option<ID3D12Device> = None;
                D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
                let device = device?;
                let queue: ID3D12CommandQueue = device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })
                    .ok()?;
                let allocator: ID3D12CommandAllocator = device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .ok()?;
                let list: ID3D12GraphicsCommandList = device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .ok()?;
                let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
                let event = CreateEventW(None, false, false, None).ok()?;
                let clamp = make_for(&device).expect("clamp shader must compile on WARP");
                let worker = compile_cs(&device, WORKER_CS).expect("worker shader must compile");
                Some(Self {
                    device,
                    queue,
                    allocator,
                    list,
                    fence,
                    submitted: 0,
                    event,
                    clamp,
                    worker,
                })
            }
        }

        /// CPU-mapped L0 write-back buffer matching `Shadow`'s heap type.
        fn shared_buffer(&self, size: u64) -> (ID3D12Resource, *mut u8, u64) {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size.max(4),
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let mut buffer: Option<ID3D12Resource> = None;
            let mut data = std::ptr::null_mut();
            // SAFETY: plain buffer creation and persistent CPU mapping.
            unsafe {
                self.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COMMON,
                        None,
                        &mut buffer,
                    )
                    .unwrap();
                let buffer = buffer.unwrap();
                buffer.Map(0, None, Some(&mut data)).unwrap();
                let gpu = buffer.GetGPUVirtualAddress();
                (buffer, data.cast(), gpu)
            }
        }

        /// Device-local `DEFAULT`-heap UAV scratch buffer.
        fn default_uav(&self, size: u64) -> ID3D12Resource {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size.max(64 << 10),
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let mut buffer: Option<ID3D12Resource> = None;
            // SAFETY: plain device-local buffer creation.
            unsafe {
                self.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COMMON,
                        None,
                        &mut buffer,
                    )
                    .unwrap();
            }
            buffer.unwrap()
        }

        fn signature(&self, kind: u32, stride: u32) -> ID3D12CommandSignature {
            let arg = D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
                ..Default::default()
            };
            let desc = D3D12_COMMAND_SIGNATURE_DESC {
                ByteStride: stride,
                NumArgumentDescs: 1,
                pArgumentDescs: &arg,
                NodeMask: 0,
            };
            let mut sig: Option<ID3D12CommandSignature> = None;
            // SAFETY: valid single-argument command signature description.
            unsafe {
                self.device
                    .CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut sig)
                    .unwrap();
            }
            sig.unwrap()
        }

        fn submit_and_wait(&mut self) {
            self.submitted += 1;
            // SAFETY: closing, executing, waiting on fence, and resetting our list.
            unsafe {
                self.list.Close().unwrap();
                let cmd: ID3D12CommandList = self.list.cast().unwrap();
                self.queue.ExecuteCommandLists(&[Some(cmd)]);
                self.queue.Signal(&self.fence, self.submitted).unwrap();
                self.fence
                    .SetEventOnCompletion(self.submitted, self.event)
                    .unwrap();
                WaitForSingleObject(self.event, INFINITE);
                self.allocator.Reset().unwrap();
                self.list.Reset(&self.allocator, None).unwrap();
            }
        }
    }

    fn compile_cs(device: &ID3D12Device, source: &str) -> windows::core::Result<Clamp> {
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: source and entry names outlive `D3DCompile`.
        unsafe {
            D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                s!("worker"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )?;
            let code = code.unwrap();
            let bytes = std::slice::from_raw_parts(
                code.GetBufferPointer().cast::<u8>(),
                code.GetBufferSize(),
            );
            let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
            let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            })?;
            Ok(Clamp { root, pso })
        }
    }

    /// The compare of copies in the card's memory (`crate::host`), on the
    /// software adapter. Told 1 it flags the pieces whose bytes are not
    /// what was hashed last and leaves the hashes: a piece a shader changed
    /// and a later one changed back then looks untouched, which is right
    /// only if the piece was fetched after both. Told 2 (submissions sent
    /// ahead with the relay, whose pieces are fetched flight by flight) it
    /// stores the hash it made: each compare flags what changed since the
    /// compare before it, also a change back.
    #[test]
    fn warp_card_compare_stores_the_hash_it_made_when_told() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let compare = compile_cs(&warp.device, crate::host::CARD_COMPARE).expect("the card compare must compile on WARP");
        const PIECE: usize = 64 << 10;
        const PIECES: u32 = 3;
        let size = PIECE * PIECES as usize;
        let (_now, now_ptr, now_gpu) = warp.shared_buffer(size as u64);
        let (_hashes, _, hashes_gpu) = warp.shared_buffer(16 * u64::from(PIECES));
        let (_differs, differs_ptr, differs_gpu) = warp.shared_buffer(64);
        let content: Vec<u8> = (0..size).map(|i| (i * 31 + i / 251) as u8).collect();
        // SAFETY: the mapping holds `size` bytes.
        unsafe { std::ptr::copy_nonoverlapping(content.as_ptr(), now_ptr, size) };
        // One pass over the three pieces; what it left in `differs` (7s
        // where it stored nothing).
        let run = |warp: &mut WarpHarness, mode: u32| -> [u32; 3] {
            // SAFETY: our open list, live buffers, the compare's own root
            // signature; the mappings hold what is read and written.
            unsafe {
                for piece in 0..PIECES as usize {
                    differs_ptr.cast::<u32>().add(piece).write_volatile(7);
                }
                warp.list.SetComputeRootSignature(&compare.root);
                warp.list.SetPipelineState(&compare.pso);
                let constants = [0u32, 0, size as u32, PIECES, 0, mode];
                warp.list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                warp.list.SetComputeRootShaderResourceView(1, now_gpu);
                warp.list.SetComputeRootUnorderedAccessView(2, hashes_gpu);
                warp.list.SetComputeRootUnorderedAccessView(3, differs_gpu);
                warp.list.Dispatch(PIECES.div_ceil(64), 1, 1);
            }
            warp.submit_and_wait();
            // SAFETY: as above.
            [0, 1, 2].map(|piece| unsafe { differs_ptr.cast::<u32>().add(piece).read_volatile() })
        };
        let at = PIECE + 1234;
        // SAFETY (each): inside the mapping.
        let set = |value: u8| unsafe { now_ptr.add(at).write_volatile(value) };
        // 0: the hashes are stored, nothing is flagged.
        assert_eq!(run(&mut warp, 0), [7, 7, 7]);
        assert_eq!(run(&mut warp, 1), [0, 0, 0]);
        // A change in the second piece: flagged, and with 1 flagged again
        // (the hash stays the old one).
        set(content[at] ^ 0x40);
        assert_eq!(run(&mut warp, 1), [0, 1, 0]);
        assert_eq!(run(&mut warp, 1), [0, 1, 0]);
        // Changed back: with 1 the piece looks untouched.
        set(content[at]);
        assert_eq!(run(&mut warp, 1), [0, 0, 0]);
        // With 2 the change is flagged once, and the hash is the new one.
        set(content[at] ^ 0x40);
        assert_eq!(run(&mut warp, 2), [0, 1, 0]);
        assert_eq!(run(&mut warp, 2), [0, 0, 0]);
        // Changed back: flagged, where 1 saw nothing.
        set(content[at]);
        assert_eq!(run(&mut warp, 2), [0, 1, 0]);
        assert_eq!(run(&mut warp, 2), [0, 0, 0]);
    }

    /// The fetch inside flights (`crate::host::CARD_FETCH`), on the software
    /// adapter, in one list behind the compare as `Host::card_before`
    /// records it: each flagged piece takes a ticket and, while it is
    /// within the slots of the lowest ticket in the air, its bytes go to
    /// its slot; an unflagged piece's entry says so; a ticket past the
    /// slots is written but copies nothing.
    #[test]
    fn warp_card_fetch_copies_flagged_pieces_to_their_slots() {
        use crate::host::{CARD_COMPARE, CARD_FETCH, INFLIGHT_NONE, INFLIGHT_SLOT};
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let compare = compile_cs(&warp.device, CARD_COMPARE).expect("the card compare must compile on WARP");
        let fetch = compile_cs(&warp.device, CARD_FETCH).expect("the card fetch must compile on WARP");
        const PIECE: usize = 64 << 10;
        const PIECES: u32 = 3;
        // The copy starts 40 bytes into its piece 0: piece 0 is short, the
        // others straddle title pieces and start off a multiple of 16.
        const SKEW: u32 = 40;
        let size = PIECE * PIECES as usize;
        let (_now, now_ptr, now_gpu) = warp.shared_buffer(size as u64);
        let (_hashes, _, hashes_gpu) = warp.shared_buffer(16 * u64::from(PIECES + 1));
        let (_differs, _, differs_gpu) = warp.shared_buffer(64);
        let (_table, table_ptr, table_gpu) = warp.shared_buffer(4 + 64);
        let slots = 1u32;
        let (_fetched, fetched_ptr, fetched_gpu) = warp.shared_buffer(INFLIGHT_SLOT * u64::from(slots));
        let content: Vec<u8> = (0..size).map(|i| (i * 31 + i / 251) as u8).collect();
        // SAFETY: the mappings hold what is written.
        unsafe {
            std::ptr::copy_nonoverlapping(content.as_ptr(), now_ptr, size);
            table_ptr.cast::<u32>().write_volatile(0);
        }
        let pieces = PIECES + 1;
        let run = |warp: &mut WarpHarness, mode: u32, landed: u32| -> Vec<u32> {
            // SAFETY: our open list, live buffers, the shaders' own root
            // signatures; the mappings hold what is read and written.
            unsafe {
                warp.list.SetComputeRootSignature(&compare.root);
                warp.list.SetPipelineState(&compare.pso);
                let constants = [0u32, SKEW, size as u32, pieces, 0, mode];
                warp.list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                warp.list.SetComputeRootShaderResourceView(1, now_gpu);
                warp.list.SetComputeRootUnorderedAccessView(2, hashes_gpu);
                warp.list.SetComputeRootUnorderedAccessView(3, differs_gpu);
                warp.list.Dispatch(1, 1, 1);
                if mode == 2 {
                    warp.list.ResourceBarrier(&[uav_barrier()]);
                    warp.list.SetComputeRootSignature(&fetch.root);
                    warp.list.SetPipelineState(&fetch.pso);
                    // Two groups for four pieces: the loop over pieces.
                    let constants = [0u32, SKEW, size as u32, pieces, 0, landed, slots, 2];
                    warp.list.SetComputeRoot32BitConstants(0, 8, constants.as_ptr().cast(), 0);
                    warp.list.SetComputeRootShaderResourceView(1, now_gpu);
                    warp.list.SetComputeRootUnorderedAccessView(2, differs_gpu);
                    warp.list.SetComputeRootUnorderedAccessView(3, table_gpu);
                    warp.list.SetComputeRootUnorderedAccessView(4, fetched_gpu);
                    warp.list.Dispatch(2, 1, 1);
                }
            }
            warp.submit_and_wait();
            // SAFETY: as above.
            (0..=pieces as usize).map(|i| unsafe { table_ptr.cast::<u32>().add(i).read_volatile() }).collect()
        };
        // SAFETY (each): inside the mapping of the copy.
        let set = |at: usize, value: u8| unsafe { now_ptr.add(at).write_volatile(value) };
        // SAFETY (each): the slot's mapping holds `INFLIGHT_SLOT` bytes.
        let slot = |len: usize| unsafe { std::slice::from_raw_parts(fetched_ptr, len).to_vec() };
        run(&mut warp, 0, 0);
        // Piece 2 of the copy (title piece 2: bytes 2 * 64 KiB - 40 on), its
        // slot from the multiple of 16 below to the one above its end.
        let from = 2 * PIECE - SKEW as usize;
        let widened = from & !15;
        assert_ne!(widened, from);
        set(from + 100, content[from + 100] ^ 0x40);
        let table = run(&mut warp, 2, 0);
        assert_eq!(table, [1, INFLIGHT_NONE, INFLIGHT_NONE, 0, INFLIGHT_NONE]);
        let mut wanted = content[widened..widened + INFLIGHT_SLOT as usize].to_vec();
        wanted[from + 100 - widened] ^= 0x40;
        assert_eq!(slot(INFLIGHT_SLOT as usize), wanted, "piece 2 in slot 0");
        // Two pieces flagged with one slot, ticket 0 not landed: tickets 1
        // and 2 get none and the slot keeps what it held.
        set(5, content[5] ^ 1);
        set(PIECE + 7, content[PIECE + 7] ^ 1);
        let table = run(&mut warp, 2, 0);
        assert_eq!(table[0], 3);
        let mut tickets = [table[1], table[2]];
        tickets.sort_unstable();
        assert_eq!(tickets, [1, 2]);
        assert_eq!(&table[3..], [INFLIGHT_NONE, INFLIGHT_NONE]);
        assert_eq!(slot(INFLIGHT_SLOT as usize), wanted);
        // Every ticket up to 2 landed: the next one, 3, gets slot 0. Piece
        // 0 starts at the copy's start.
        set(9, content[9] ^ 2);
        let table = run(&mut warp, 2, 3);
        assert_eq!(table, [4, 3, INFLIGHT_NONE, INFLIGHT_NONE, INFLIGHT_NONE]);
        let mut piece0 = content[..PIECE - SKEW as usize].to_vec();
        piece0[5] ^= 1;
        piece0[9] ^= 2;
        assert_eq!(slot(PIECE - SKEW as usize), piece0);
    }

    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
    }

    #[test]
    fn rejects_unaligned_or_undersized_layouts() {
        assert_eq!(validate_layout((0, 16, 4), 0, None), Some(64));
        assert_eq!(validate_layout((1, 20, 4), 64, Some(128)), Some(80));
        assert_eq!(validate_layout((2, 12, 0), 0, Some(0)), Some(0));
        assert_eq!(validate_layout((10, 16, 2), 8, None), Some(32));
        // Undersized strides for each kind.
        assert_eq!(validate_layout((0, 12, 1), 0, None), None);
        assert_eq!(validate_layout((1, 16, 1), 0, None), None);
        assert_eq!(validate_layout((2, 8, 1), 0, None), None);
        assert_eq!(validate_layout((10, 8, 1), 0, None), None);
        // Unknown kind.
        assert_eq!(validate_layout((3, 16, 1), 0, None), None);
        // Unaligned stride, source offset, or count offset.
        assert_eq!(validate_layout((0, 18, 1), 0, None), None);
        assert_eq!(validate_layout((0, 16, 1), 2, None), None);
        assert_eq!(validate_layout((0, 16, 1), 0, Some(6)), None);
        // Total byte size exceeding 32-bit ByteAddressBuffer indexing.
        assert_eq!(validate_layout((1, 20, u32::MAX), 0, None), None);
    }

    /// WARP D3D12 execution test proving:
    /// (a) `countBuffer` with a small count (`3`) and `4,093` garbage tail slots
    ///     (`max = 4096`) does NOT blow `BUDGET` or zero valid dispatches in
    ///     this or subsequent `ExecuteIndirect` calls in the same list,
    /// (b) `src_args` (and `count_buf`) are NOT mutated in-place, and
    /// (c) full `stride` bytes (`stride = 24` > `12`) are copied for active
    ///     records into `dst_args` at offset `0`.
    #[test]
    fn warp_count_buffer_ignores_garbage_tail_and_preserves_source() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_DISPATCH: u32 = 2;
        const STRIDE: u32 = 24; // 6 dwords: 3 dispatch group counts + 3 extra payload dwords
        const MAX: u32 = 4096;
        const ARGS_OFFSET: usize = 64; // non-zero offset with sentinel prefix
        const COUNT_OFFSET_1: usize = 128;
        const COUNT_OFFSET_2: usize = 192;

        let src_bytes = ARGS_OFFSET + (STRIDE as usize) * (MAX as usize) + 64;
        let (src_buf, src_ptr, _) = warp.shared_buffer(src_bytes as u64);
        let (count_buf, count_ptr, _) = warp.shared_buffer(512);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (_hits_buf, hits_ptr, hits_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer((STRIDE * 8) as u64);
        let dst_buf = warp.default_uav(u64::from(STRIDE) * u64::from(MAX));
        let sig = warp.signature(KIND_DISPATCH, STRIDE);

        // Fill the entire source buffer with non-zero sentinel bytes, then
        // populate 3 valid records at `ARGS_OFFSET` and 4,093 huge garbage
        // records in `3..MAX` that would exhaust `BUDGET` ~2,000x over if read.
        let mut initial_src = vec![0xA5u8; src_bytes];
        let write_record = |buf: &mut [u8], idx: usize, words: [u32; 6]| {
            let base = ARGS_OFFSET + idx * (STRIDE as usize);
            for (w, val) in words.into_iter().enumerate() {
                buf[base + w * 4..base + (w + 1) * 4].copy_from_slice(&val.to_le_bytes());
            }
        };
        write_record(
            &mut initial_src,
            0,
            [3, 2, 1, 0x1111_2222, 0x3333_4444, 0x5555_6666],
        );
        write_record(
            &mut initial_src,
            1,
            [5, 1, 1, 0x7777_8888, 0x9999_AAAA, 0xBBBB_CCCC],
        );
        // Record 2 exceeds 65,535 in X -> clamped to (65535, 1, 1).
        write_record(
            &mut initial_src,
            2,
            [100_000, 1, 1, 0xDDDD_EEEE, 0xFFFF_0001, 0x0123_4567],
        );
        for slot in 3..MAX as usize {
            // Each garbage slot would clamp to 65535 * 16 * 1 = 1,048,560 groups
            // (`cost = 67,107,840 ~= BUDGET / 2`), exhausting `BUDGET` immediately!
            write_record(
                &mut initial_src,
                slot,
                [0xFFFF_FFFF, 16, 1, 0xDEAD_BEEF, 0xBAD0_C0DE, slot as u32],
            );
        }

        // Second indirect call's source records (at offset 0 of a second buffer,
        // or later in `count_buf`): put 2 active records + garbage tail in a
        // second range to run back-to-back in the SAME command list.
        let mut initial_count = vec![0x5Au8; 512];
        initial_count[COUNT_OFFSET_1..COUNT_OFFSET_1 + 4].copy_from_slice(&3u32.to_le_bytes());
        initial_count[COUNT_OFFSET_2..COUNT_OFFSET_2 + 4].copy_from_slice(&2u32.to_le_bytes());

        // SAFETY: buffers are mapped with exact capacities above.
        unsafe {
            std::ptr::copy_nonoverlapping(initial_src.as_ptr(), src_ptr, src_bytes);
            std::ptr::copy_nonoverlapping(initial_count.as_ptr(), count_ptr, 512);
            (budget_ptr as *mut u32).write_volatile(0);
            (hits_ptr as *mut u32).write_volatile(0);
            std::ptr::write_bytes(readback_ptr, 0, (STRIDE * 8) as usize);
        }

        // Record Call #1 (`eff_count = 3` out of `MAX = 4096`), copy the first
        // 5 records of `dst_buf` out for inspection, then record Call #2
        // (`eff_count = 2` out of `MAX = 4096`) reusing `dst_buf` in the same list!
        assert!(record_clamp(
            &warp.clamp,
            &warp.list,
            &dst_buf,
            budget_gpu,
            (&src_buf, ARGS_OFFSET as u64),
            Some((&count_buf, COUNT_OFFSET_1 as u64)),
            (KIND_DISPATCH, STRIDE, MAX),
        ));
        // SAFETY: open command list, valid worker PSO/root and buffers.
        unsafe {
            warp.list.SetComputeRootSignature(&warp.worker.root);
            warp.list.SetPipelineState(&warp.worker.pso);
            warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            warp.list.ExecuteIndirect(
                &sig,
                MAX,
                &dst_buf,
                0,
                Some(&count_buf),
                COUNT_OFFSET_1 as u64,
            );
        }
        done_on(&warp.list, &dst_buf);

        // Snapshot `dst_buf` after Call #1 (from `COMMON`, copy 5 * STRIDE bytes,
        // then return `dst_buf` to `COMMON` before Call #2).
        // SAFETY: open command list; `dst_buf` returned to `COMMON` by `done_on`.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list
                .CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, u64::from(STRIDE * 5));
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        }

        // Call #2 in the SAME command list (`eff_count = 2` -> records 0 and 1:
        // 6 + 5 = 11 groups). Would run empty if Call #1's tail blew `BUDGET`!
        assert!(record_clamp(
            &warp.clamp,
            &warp.list,
            &dst_buf,
            budget_gpu,
            (&src_buf, ARGS_OFFSET as u64),
            Some((&count_buf, COUNT_OFFSET_2 as u64)),
            (KIND_DISPATCH, STRIDE, MAX),
        ));
        // SAFETY: open command list, valid worker PSO/root and buffers.
        unsafe {
            warp.list.SetComputeRootSignature(&warp.worker.root);
            warp.list.SetPipelineState(&warp.worker.pso);
            warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            warp.list.ExecuteIndirect(
                &sig,
                MAX,
                &dst_buf,
                0,
                Some(&count_buf),
                COUNT_OFFSET_2 as u64,
            );
        }
        done_on(&warp.list, &dst_buf);

        warp.submit_and_wait();

        // SAFETY: GPU execution finished; read mapped host results.
        let (hits, spent, after_src, after_count, clamped_words) = unsafe {
            let hits = (hits_ptr as *const u32).read_volatile();
            let spent = (budget_ptr as *const u32).read_volatile();
            let after_src = std::slice::from_raw_parts(src_ptr.cast_const(), src_bytes).to_vec();
            let after_count = std::slice::from_raw_parts(count_ptr.cast_const(), 512).to_vec();
            let words = std::slice::from_raw_parts(
                readback_ptr.cast::<u32>().cast_const(),
                (STRIDE as usize / 4) * 5,
            )
            .to_vec();
            (hits, spent, after_src, after_count, words)
        };

        // Call #1: (3*2*1) + (5*1*1) + (65535*1*1) = 65,546 groups.
        // Call #2: (3*2*1) + (5*1*1) = 11 groups.
        let expected_groups = (6 + 5 + 65_535) + (6 + 5);
        assert_eq!(
            hits, expected_groups,
            "valid indirect dispatches must all execute"
        );
        assert_eq!(
            spent,
            expected_groups * 64,
            "budget must reflect only active records below countBuffer"
        );
        assert_eq!(
            after_src, initial_src,
            "src_args must never be mutated in-place"
        );
        assert_eq!(
            after_count, initial_count,
            "count_buf must never be mutated"
        );

        // Check `dst_args` records 0..3: clamped group counts + preserved extra stride words.
        assert_eq!(
            &clamped_words[0..6],
            &[3, 2, 1, 0x1111_2222, 0x3333_4444, 0x5555_6666]
        );
        assert_eq!(
            &clamped_words[6..12],
            &[5, 1, 1, 0x7777_8888, 0x9999_AAAA, 0xBBBB_CCCC]
        );
        assert_eq!(
            &clamped_words[12..18],
            &[65_535, 1, 1, 0xDDDD_EEEE, 0xFFFF_0001, 0x0123_4567]
        );
        // Inactive slot 3 was never copied from `src_args`.
        assert_ne!(clamped_words[21], 0xDEAD_BEEF);
    }

    /// WARP D3D12 test proving `BUDGET` saturates without 32-bit unsigned
    /// wraparound across >64 huge indexed-draw records (`count = None`), while
    /// preserving `StartIndexLocation`, `BaseVertexLocation`,
    /// `StartInstanceLocation`, and custom stride padding, and leaving
    /// `src_args` untouched.
    /// A window of mesh records clamped ahead: the filled ones come out as
    /// they are and are charged, the slots of all ones come out empty and
    /// charge nothing; clamped as single records the same slots are clamped
    /// to the most and spend the budget.
    #[test]
    fn warp_window_leaves_unfilled_slots_empty_and_uncharged() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_MESH: u32 = 10;
        const STRIDE: u32 = 32;
        const RECORDS: u32 = 16;
        let filled = [5u32, 0, 519];
        let total = (STRIDE * RECORDS) as usize;
        let mut src = vec![0xffu8; total];
        for (i, groups) in filled.into_iter().enumerate() {
            let words = [groups, 1, 1, 0xAB00 + i as u32];
            for (w, word) in words.into_iter().enumerate() {
                let at = i * STRIDE as usize + w * 4;
                src[at..at + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
        for ahead in [true, false] {
            let (src_buf, src_ptr, _) = warp.shared_buffer(total as u64);
            let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
            let (readback_buf, readback_ptr, _) = warp.shared_buffer(total as u64);
            let dst_buf = warp.default_uav(total as u64);
            // SAFETY: mapped buffers of these sizes.
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), src_ptr, total);
                std::ptr::write_bytes(budget_ptr, 0, 256);
                std::ptr::write_bytes(readback_ptr, 0, total);
            }
            assert!(record_clamp_as(
                &warp.clamp,
                &warp.list,
                &dst_buf,
                budget_gpu,
                (&src_buf, 0),
                None,
                (KIND_MESH, STRIDE, RECORDS),
                ahead,
            ));
            done_on(&warp.list, &dst_buf);
            // SAFETY: an open list; the scratch buffer is copied for the check.
            unsafe {
                warp.list.ResourceBarrier(&[transition(
                    &dst_buf,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                warp.list.CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, total as u64);
                warp.list.ResourceBarrier(&[transition(
                    &dst_buf,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                )]);
            }
            warp.submit_and_wait();
            // SAFETY: the work is done; mapped buffers.
            let (spent, largest, out) = unsafe {
                (
                    (budget_ptr as *const u32).read_volatile(),
                    (budget_ptr as *const u32).add(1).read_volatile(),
                    std::slice::from_raw_parts(readback_ptr.cast::<u32>().cast_const(), total / 4).to_vec(),
                )
            };
            for (i, groups) in filled.into_iter().enumerate() {
                let rec = &out[i * 8..i * 8 + 4];
                assert_eq!(rec, [groups, u32::from(groups > 0).max(1).min(1), 1, 0xAB00 + i as u32], "filled record {i}, ahead {ahead}");
            }
            let charged_filled = (5 + 519) * 128;
            if ahead {
                for i in filled.len()..RECORDS as usize {
                    assert_eq!(out[i * 8..i * 8 + 3], [0, 0, 0], "unfilled slot {i} must come out empty");
                    assert_eq!(out[i * 8 + 3], u32::MAX, "the words after the record are copied as they are");
                }
                assert_eq!(spent, charged_filled, "unfilled slots charge nothing");
                assert_eq!(largest, 519, "and are not counted as asked for");
            } else {
                // Thirteen slots at 65,535 groups: sixteen would be all of it.
                assert_eq!(spent, charged_filled + 13 * 65535 * 128, "clamped one by one, each slot is charged the most");
                assert_eq!(largest, u32::MAX);
            }
        }
    }

    #[test]
    fn warp_budget_saturates_without_wraparound_and_preserves_draw_fields() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_INDEXED: u32 = 1;
        const STRIDE: u32 = 24; // 5 indexed-draw words + 1 trailing custom word
        const RECORDS: u32 = 128; // 2 waves of 64: old InterlockedAdd wrapped 2^32 at 64!
        const BUDGET_CAP: u32 = 1 << 27;

        let total_bytes = (STRIDE * RECORDS) as usize;
        let (src_buf, src_ptr, _) = warp.shared_buffer(total_bytes as u64);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer(total_bytes as u64);
        let dst_buf = warp.default_uav(total_bytes as u64);

        let mut initial_src = vec![0u8; total_bytes];
        for i in 0..RECORDS as usize {
            let words: [u32; 6] = [
                u32::MAX,                    // IndexCount -> clamps to 1 << 24
                u32::MAX,                    // InstanceCount -> clamps to 4 (cost = 1 << 26)
                1000 + i as u32,             // StartIndexLocation
                (-200i32 + i as i32) as u32, // BaseVertexLocation
                5000 + i as u32,             // StartInstanceLocation
                0xCAFE_0000 | (i as u32),    // Extra stride word
            ];
            let base = i * (STRIDE as usize);
            for (w, val) in words.into_iter().enumerate() {
                initial_src[base + w * 4..base + (w + 1) * 4].copy_from_slice(&val.to_le_bytes());
            }
        }

        // SAFETY: mapped buffers have sufficient capacity.
        unsafe {
            std::ptr::copy_nonoverlapping(initial_src.as_ptr(), src_ptr, total_bytes);
            (budget_ptr as *mut u32).write_volatile(0);
            std::ptr::write_bytes(readback_ptr, 0, total_bytes);
        }

        assert!(record_clamp(
            &warp.clamp,
            &warp.list,
            &dst_buf,
            budget_gpu,
            (&src_buf, 0),
            None,
            (KIND_INDEXED, STRIDE, RECORDS),
        ));
        done_on(&warp.list, &dst_buf);

        // SAFETY: open command list; copy clamped scratch buffer for verification.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list
                .CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, total_bytes as u64);
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        }
        warp.submit_and_wait();

        // SAFETY: GPU work completed.
        let (spent, after_src, out_words) = unsafe {
            let spent = (budget_ptr as *const u32).read_volatile();
            let after_src = std::slice::from_raw_parts(src_ptr.cast_const(), total_bytes).to_vec();
            let out_words = std::slice::from_raw_parts(
                readback_ptr.cast::<u32>().cast_const(),
                total_bytes / 4,
            )
            .to_vec();
            (spent, after_src, out_words)
        };

        assert_eq!(
            spent, BUDGET_CAP,
            "budget must saturate at BUDGET without wrapping"
        );
        assert_eq!(after_src, initial_src, "src_args must remain untouched");

        let mut kept = 0usize;
        let mut zeroed = 0usize;
        for i in 0..RECORDS as usize {
            let rec = &out_words[i * 6..(i + 1) * 6];
            if rec[0] == (1 << 24) && rec[1] == 4 {
                kept += 1;
            } else {
                assert_eq!((rec[0], rec[1]), (0, 0), "excess draw must be zeroed");
                zeroed += 1;
            }
            assert_eq!(rec[2], 1000 + i as u32);
            assert_eq!(rec[3], (-200i32 + i as i32) as u32);
            assert_eq!(rec[4], 5000 + i as u32);
            assert_eq!(rec[5], 0xCAFE_0000 | (i as u32));
        }
        assert_eq!((kept, zeroed), (2, (RECORDS as usize) - 2));
    }

    /// Exercises `HostList::indirect_scratch`, `bound`, `done`, `Host::execute`,
    /// and `HostList::reset` end-to-end on the host device.
    #[test]
    fn host_list_reuses_and_grows_scratch_across_submissions() {
        let Some(host) = super::host() else {
            return;
        };
        let Ok(list) = host.new_list() else {
            return;
        };
        let Some((src_buf, src_ptr)) = host.shared_buffer(256 << 10) else {
            return;
        };
        let Some((count_buf, count_ptr)) = host.shared_buffer(256) else {
            return;
        };
        // SAFETY: mapped host buffers.
        unsafe {
            std::ptr::write_bytes(src_ptr, 0x7F, 256 << 10);
            (count_ptr as *mut u32).write_volatile(2);
        }

        // First two calls fit inside the initial 64 KiB scratch buffer and
        // must reuse the exact same `ID3D12Resource`.
        let b1 = bound(&list, (&src_buf, 0), Some((&count_buf, 0)), (0, 16, 64)).unwrap();
        done(&list, &b1);
        let b2 = bound(&list, (&src_buf, 64), Some((&count_buf, 0)), (1, 20, 128)).unwrap();
        done(&list, &b2);
        assert_eq!(
            b1.as_raw(),
            b2.as_raw(),
            "scratch buffer within capacity must be reused"
        );

        // A call needing >64 KiB grows the scratch buffer mid-list while keeping
        // `b1` alive until reset.
        let b3 = bound(&list, (&src_buf, 0), Some((&count_buf, 0)), (1, 20, 5000)).unwrap();
        done(&list, &b3);
        assert_ne!(
            b3.as_raw(),
            b1.as_raw(),
            "larger request must grow scratch buffer"
        );

        list.close();
        host.execute(&[&list]);
        list.reset();

        // After reset, requests up to the grown high-water mark reuse `b3` with
        // zero new allocations.
        let b4 = bound(&list, (&src_buf, 0), Some((&count_buf, 0)), (1, 20, 4096)).unwrap();
        done(&list, &b4);
        assert_eq!(
            b4.as_raw(),
            b3.as_raw(),
            "grown scratch buffer must persist across reset"
        );
        list.close();
        host.execute(&[&list]);
        list.reset();
    }

    /// Reproduces the indirect-window Root-SRV OOB page fault: a buffer whose
    /// tail holds only 16 bytes (< 32-byte step, >= 12-byte mesh stride) must
    /// reject a 32-byte `copy_bytes` read past `Width`, while a packed
    /// `copy_bytes = 12, src_stride = 32` clamp reads only `[0..12]` per
    /// record and stays strictly inside `Width`.
    #[test]
    fn warp_window_rejects_oob_src_and_packs_mesh_records() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_MESH: u32 = 10;
        // Exact buffer width = 32 + 16 = 48 bytes: record 0 at offset 0,
        // record 1 at offset 32 with only 16 bytes left (< step 32, >= 12).
        let (src_buf, src_ptr, _) = warp.shared_buffer(48);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer(64);
        let dst_buf = warp.default_uav(64);
        unsafe {
            std::ptr::write_bytes(src_ptr, 0, 48);
            std::ptr::write_bytes(budget_ptr, 0, 256);
            std::ptr::write_bytes(readback_ptr, 0, 64);
            let words = src_ptr.cast::<u32>();
            // Record 0 at byte 0: (7, 1, 1)
            words.add(0).write(7);
            words.add(1).write(1);
            words.add(2).write(1);
            // Record 1 at byte 32 (dword 8): (19, 1, 1)
            words.add(8).write(19);
            words.add(9).write(1);
            words.add(10).write(1);
        }
        // Old bug: stride=32 at offset=32 on a 48-byte buffer reads [32..64] -> MUST be rejected!
        assert!(!record_clamp_full(
            &warp.clamp,
            &warp.list,
            &dst_buf,
            false,
            None,
            budget_gpu,
            (&src_buf, 32),
            None,
            (KIND_MESH, 32, 32, 1),
            false,
            false,
            false,
        ));
        // Packed window: copy_bytes=12, src_stride=32, records=2 -> reads [0..12] and [32..44] <= 48!
        assert!(record_clamp_full(
            &warp.clamp,
            &warp.list,
            &dst_buf,
            false,
            None,
            budget_gpu,
            (&src_buf, 0),
            None,
            (KIND_MESH, 12, 32, 2),
            true,
            false,
            false,
        ));
        done_on(&warp.list, &dst_buf);
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list.CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, 24);
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        }
        warp.submit_and_wait();
        let out = unsafe { std::slice::from_raw_parts(readback_ptr.cast::<u32>().cast_const(), 6) };
        assert_eq!(out, &[7, 1, 1, 19, 1, 1]);
    }

    /// Verifies our fused 64-thread coalesced `CARD_COMPARE_FETCH` shader:
    /// mode 0 hashes, mode 3 detects a single-bit mutation, stores updated
    /// hash, allocates a ticket, and copies the exact widened piece into
    /// `fetched` in one pass with no intermediate barrier.
    #[test]
    fn warp_card_compare_fetch_fused_hashes_and_fetches_in_one_pass() {
        use crate::host::{CARD_COMPARE_FETCH, INFLIGHT_NONE, INFLIGHT_SLOT};
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let fused = compile_cs(&warp.device, CARD_COMPARE_FETCH)
            .expect("CARD_COMPARE_FETCH must compile on WARP");
        const PIECE: usize = 64 << 10;
        const PIECES: u32 = 3;
        const SKEW: u32 = 40;
        let size = PIECE * PIECES as usize;
        let (_now, now_ptr, now_gpu) = warp.shared_buffer(size as u64);
        let (_hashes, _, hashes_gpu) = warp.shared_buffer(16 * u64::from(PIECES + 1));
        let (_differs, differs_ptr, differs_gpu) = warp.shared_buffer(64);
        let (_table, table_ptr, table_gpu) = warp.shared_buffer(4 + 64);
        let slots = 2u32;
        let (_fetched, fetched_ptr, fetched_gpu) = warp.shared_buffer(INFLIGHT_SLOT * u64::from(slots));
        let content: Vec<u8> = (0..size).map(|i| (i * 31 + i / 251) as u8).collect();
        unsafe {
            std::ptr::copy_nonoverlapping(content.as_ptr(), now_ptr, size);
            table_ptr.cast::<u32>().write_volatile(0);
        }
        let pieces = PIECES + 1;
        let run = |warp: &mut WarpHarness, mode: u32, landed: u32| -> (Vec<u32>, Vec<u32>) {
            unsafe {
                warp.list.SetComputeRootSignature(&fused.root);
                warp.list.SetPipelineState(&fused.pso);
                let constants = [0u32, SKEW, size as u32, pieces, 0, mode, landed, slots, pieces];
                warp.list.SetComputeRoot32BitConstants(0, 9, constants.as_ptr().cast(), 0);
                warp.list.SetComputeRootShaderResourceView(1, now_gpu);
                warp.list.SetComputeRootUnorderedAccessView(2, hashes_gpu);
                warp.list.SetComputeRootUnorderedAccessView(3, differs_gpu);
                warp.list.SetComputeRootUnorderedAccessView(4, table_gpu);
                warp.list.SetComputeRootUnorderedAccessView(5, fetched_gpu);
                warp.list.Dispatch(pieces, 1, 1);
            }
            warp.submit_and_wait();
            let diffs = (0..pieces as usize)
                .map(|i| unsafe { differs_ptr.cast::<u32>().add(i).read_volatile() })
                .collect();
            let tbl = (0..=pieces as usize)
                .map(|i| unsafe { table_ptr.cast::<u32>().add(i).read_volatile() })
                .collect();
            (diffs, tbl)
        };
        // Mode 0: seed hashes.
        run(&mut warp, 0, 0);
        // Mode 3 with no changes: all 0 diffs, all INFLIGHT_NONE, counter stays 0.
        let (d0, t0) = run(&mut warp, 3, 0);
        assert_eq!(d0, vec![0, 0, 0, 0]);
        assert_eq!(t0, vec![0, INFLIGHT_NONE, INFLIGHT_NONE, INFLIGHT_NONE, INFLIGHT_NONE]);
        // Mutate 1 byte inside piece 2 and run mode 3 (fused compare + hash update + fetch).
        let from = 2 * PIECE - SKEW as usize;
        let widened = from & !15;
        unsafe { now_ptr.add(from + 100).write_volatile(content[from + 100] ^ 0x40) };
        let (d1, t1) = run(&mut warp, 3, 0);
        assert_eq!(d1, vec![0, 0, 1, 0]);
        assert_eq!(t1, vec![1, INFLIGHT_NONE, INFLIGHT_NONE, 0, INFLIGHT_NONE]);
        let mut wanted = content[widened..widened + INFLIGHT_SLOT as usize].to_vec();
        wanted[from + 100 - widened] ^= 0x40;
        let got = unsafe { std::slice::from_raw_parts(fetched_ptr, INFLIGHT_SLOT as usize).to_vec() };
        assert_eq!(got, wanted, "fused pass copied mutated piece 2 into slot 0");
        // Second run with no new changes: hash was updated, so 0 diffs now.
        let (d2, t2) = run(&mut warp, 3, 1);
        assert_eq!(d2, vec![0, 0, 0, 0]);
        assert_eq!(t2, vec![1, INFLIGHT_NONE, INFLIGHT_NONE, INFLIGHT_NONE, INFLIGHT_NONE]);
    }

    /// Verifies `VICTORY_HOIST_CLAMPS` mode parsing (default = 7, `0` = off,
    /// `1..=7` modes, invalid/out-of-range fallback to 7) and per-kind filters.
    #[test]
    fn hoist_mode_parsing_and_kind_filters() {
        assert_eq!(parse_hoist_mode(None), 7);
        assert_eq!(parse_hoist_mode(Some("0")), 0);
        for m in 1..=7u8 {
            assert_eq!(parse_hoist_mode(Some(&m.to_string())), m);
        }
        assert_eq!(parse_hoist_mode(Some(" 3 \n")), 3);
        assert_eq!(parse_hoist_mode(Some("8")), 7);
        assert_eq!(parse_hoist_mode(Some("99")), 7);
        assert_eq!(parse_hoist_mode(Some("-1")), 7);
        assert_eq!(parse_hoist_mode(Some("off")), 7);
        assert_eq!(parse_hoist_mode(Some("")), 7);

        // Kinds: 0 DRAW, 1 INDEXED, 2 DISPATCH, 10 MESH.
        for m in 0..=7u8 {
            assert!(!hoist_allows(m, 2), "DISPATCH is never hoisted (mode {m})");
        }
        for k in [0, 1, 10] {
            assert!(!hoist_allows(0, k));
            assert!(!hoist_allows(6, k));
            for m in [1, 2, 5, 7] {
                assert!(hoist_allows(m, k), "mode {m} must hoist kind {k}");
            }
        }
        // Mode 3: mesh only; Mode 4: non-mesh draws only.
        assert!(!hoist_allows(3, 0));
        assert!(!hoist_allows(3, 1));
        assert!(hoist_allows(3, 10));
        assert!(hoist_allows(4, 0));
        assert!(hoist_allows(4, 1));
        assert!(!hoist_allows(4, 10));
    }

    /// Exercises hoisted `bound_before` + `HostList::clamp_room` / `clamp_end` /
    /// `reset` across single-record clamps, packed `VICTORY_INDIRECT_WINDOW`
    /// ahead-clamps, 256 KiB chunk spillover, and post-reset chunk reuse.
    #[test]
    fn host_list_hoisted_clamps_pack_offsets_spill_chunks_and_reset_cleanly() {
        let Some(host) = super::host() else {
            return;
        };
        let Ok(clamps) = host.new_list() else {
            return;
        };
        let Some((src_buf, src_ptr)) = host.shared_buffer(512 << 10) else {
            return;
        };
        // SAFETY: mapped host buffer.
        unsafe { std::ptr::write_bytes(src_ptr, 0, 512 << 10) };

        // 1. Single mesh record (12B -> rounded to 16B) at offset 0.
        let (b0, at0) = bound_before(&clamps, (&src_buf, 0), None, (10, 12, 12, 1), false)
            .expect("single hoisted clamp");
        assert_eq!(at0, 0);

        // 2. Packed VICTORY_INDIRECT_WINDOW ahead-clamp (16 mesh records,
        //    copy_bytes=12, src_stride=32 -> 192B) packs right behind at offset 16.
        let (b1, at1) = bound_before(&clamps, (&src_buf, 32), None, (10, 12, 32, 16), true)
            .expect("windowed hoisted clamp");
        assert_eq!(b1.as_raw(), b0.as_raw(), "second clamp must share first 256 KiB chunk");
        assert_eq!(at1, 16);

        // 3. Third clamp follows at 16 + 192 = 208 (16-byte aligned).
        let (b2, at2) = bound_before(&clamps, (&src_buf, 0), None, (1, 20, 20, 1), false)
            .expect("indexed draw hoisted clamp");
        assert_eq!(b2.as_raw(), b0.as_raw());
        assert_eq!(at2, 208);

        // 4. Request exceeding remaining space in chunk 0 spills into chunk 1 at offset 0.
        //    Chunk 0 is 256 KiB (262,144B) with 240B already used; 13,100 indexed
        //    records * 20B = 262,000B <= 262,144B, > 261,904B remaining.
        let (b3, at3) = bound_before(&clamps, (&src_buf, 0), None, (1, 20, 20, 13_100), false)
            .expect("chunk spillover must succeed");
        assert_ne!(b3.as_raw(), b0.as_raw(), "spillover must allocate a second chunk");
        assert_eq!(at3, 0);

        // 5. Request exceeding a single 256 KiB chunk or 0 records returns None (falls back inline).
        assert!(bound_before(&clamps, (&src_buf, 0), None, (1, 20, 20, 0), false).is_none());
        assert!(bound_before(&clamps, (&src_buf, 0), None, (1, 20, 20, 20_000), false).is_none());

        clamps.clamp_end();
        clamps.close();
        host.execute(&[&clamps]);
        clamps.reset();

        // After reset, chunk 0 is reused from offset 0 with clean COMMON -> UAV transition.
        let (b4, at4) = bound_before(&clamps, (&src_buf, 0), None, (10, 12, 12, 1), false)
            .expect("post-reset hoisted clamp");
        assert_eq!(b4.as_raw(), b0.as_raw(), "chunk 0 must be reused after reset");
        assert_eq!(at4, 0);
        clamps.clamp_end();
        clamps.close();
        host.execute(&[&clamps]);
        clamps.reset();
    }

    /// WARP D3D12 test verifying hoisted clamp dispatches (mode 7 serialized
    /// on `budget` + packed `VICTORY_INDIRECT_WINDOW` ahead-clamp) write exact
    /// clamped records at disjoint offsets in one chunk buffer and charge
    /// `BUDGET` deterministically in dispatch order.
    #[test]
    fn warp_hoisted_clamps_deterministic_budget_and_window_packing() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_INDEXED: u32 = 1;
        const KIND_MESH: u32 = 10;
        const BUDGET_CAP: u32 = 1 << 27;

        let (src_buf, src_ptr, _) = warp.shared_buffer(4096);
        let (budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer(512);
        let chunk_buf = warp.default_uav(64 << 10);
        // SAFETY: live D3D12 buffer.
        let chunk_gpu = unsafe { chunk_buf.GetGPUVirtualAddress() };

        let mut src = vec![0xffu8; 4096];
        // Mesh window at src offset 0 (src_stride=32, copy_bytes=12, 4 records):
        // records 0 and 1 filled (groups 7 and 11), records 2 and 3 unfilled (0xffffffff).
        let write_u32 = |buf: &mut [u8], off: usize, vals: &[u32]| {
            for (i, v) in vals.iter().enumerate() {
                buf[off + i * 4..off + (i + 1) * 4].copy_from_slice(&v.to_le_bytes());
            }
        };
        write_u32(&mut src, 0, &[7, 1, 1]);
        write_u32(&mut src, 32, &[11, 1, 1]);

        // Three huge indexed draws at src offsets 256, 320, 384 (each asks for
        // 1<<24 * 4 = 1<<26 = BUDGET/2): with mode-7 serialization, draw A (1<<26)
        // fits, draw B (1<<26) saturates BUDGET (actually mesh window already
        // spent (7+11)*128 = 2304, so draw A fits and draw B exceeds remaining
        // BUDGET and is deterministically zeroed, as is draw C!).
        // Make draw A ask for (1<<24, 2) = 1<<25, draw B ask for (1<<24, 4) = 1<<26,
        // draw C ask for (1<<24, 4) = 1<<26 -> A + B = 3<<25 + 2304 < 2^27 (BOTH
        // KEEP!), C pushes past 2^27 (DETERMINISTICALLY ZEROED!).
        write_u32(&mut src, 256, &[1 << 24, 2, 101, 102, 103]);
        write_u32(&mut src, 320, &[1 << 24, 4, 201, 202, 203]);
        write_u32(&mut src, 384, &[1 << 24, 4, 301, 302, 303]);

        // SAFETY: mapped buffers of sufficient capacity.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), src_ptr, 4096);
            std::ptr::write_bytes(budget_ptr, 0, 256);
            std::ptr::write_bytes(readback_ptr, 0, 512);
        }

        // Transition chunk COMMON -> UAV once (as `HostList::clamp_room` does).
        // SAFETY: open list, live buffer.
        unsafe {
            warp.list.ResourceBarrier(&[
                uav_barrier(),
                transition(
                    &chunk_buf,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                ),
            ]);
        }

        // Dispatch 1: packed mesh window at chunk offset 0 (4 * 12 = 48B).
        let b_mesh = prepare_clamp_bindings((&src_buf, 0), None, (KIND_MESH, 12, 32, 4), true, false)
            .expect("mesh window bindings");
        record_hoisted_clamp(&warp.clamp, &warp.list, &b_mesh, chunk_gpu, &budget_buf, budget_gpu, true);

        // Dispatches 2, 3, 4: indexed draws A, B, C at chunk offsets 64, 96, 128.
        for (src_off, dst_off) in [(256u64, 64u64), (320, 96), (384, 128)] {
            let b = prepare_clamp_bindings((&src_buf, src_off), None, (KIND_INDEXED, 20, 20, 1), false, false)
                .expect("indexed draw bindings");
            record_hoisted_clamp(
                &warp.clamp,
                &warp.list,
                &b,
                chunk_gpu + dst_off,
                &budget_buf,
                budget_gpu,
                true,
            );
        }

        // One batched `clamp_end` transition UAV -> INDIRECT_ARGUMENT, then readback.
        // SAFETY: open list, live buffers.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &chunk_buf,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            )]);
            warp.list.ResourceBarrier(&[transition(
                &chunk_buf,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list.CopyBufferRegion(&readback_buf, 0, &chunk_buf, 0, 160);
            warp.list.ResourceBarrier(&[transition(
                &chunk_buf,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        }
        warp.submit_and_wait();

        // SAFETY: GPU execution completed; mapped readback & budget buffers.
        let (spent, words) = unsafe {
            (
                (budget_ptr as *const u32).read_volatile(),
                std::slice::from_raw_parts(readback_ptr.cast::<u32>().cast_const(), 40).to_vec(),
            )
        };
        // Mesh window packed tightly at chunk offset 0 (dwords 0..12):
        assert_eq!(&words[0..3], &[7, 1, 1]);
        assert_eq!(&words[3..6], &[11, 1, 1]);
        assert_eq!(&words[6..9], &[0, 0, 0], "unfilled mesh tail 2 zeroed");
        assert_eq!(&words[9..12], &[0, 0, 0], "unfilled mesh tail 3 zeroed");
        // Draw A at offset 64 (dword 16): kept.
        assert_eq!(&words[16..21], &[1 << 24, 2, 101, 102, 103]);
        // Draw B at offset 96 (dword 24): kept.
        assert_eq!(&words[24..29], &[1 << 24, 4, 201, 202, 203]);
        // Draw C at offset 128 (dword 32): deterministically zeroed past BUDGET.
        assert_eq!(&words[32..37], &[0, 0, 301, 302, 303]);
        assert_eq!(spent, BUDGET_CAP, "budget saturated cleanly without wraparound");
    }

    #[test]
    fn gathered_records_must_fit_the_clamp() {
        assert_eq!(gatherable(0x1000, 0x6d71550, (10, 12)), Some(0x6d71550));
        assert_eq!(gatherable(0x1000, 0, (1, 20)), Some(0));
        assert_eq!(gatherable(0x1000, 0x6d71552, (10, 12)), None);
        assert_eq!(gatherable(0x1002, 0x40, (10, 12)), None);
        assert_eq!(gatherable(0x1000, 0x40, (10, 8)), None);
        assert_eq!(gatherable(0x1000, 0x40, (10, GATHER_STRIDE + 4)), None);
        assert_eq!(gatherable(0x1000, 0x40, (7, 12)), None);
        assert_eq!(gatherable(0x1000, 1 << 32, (10, 12)), None);
        assert_eq!(gatherable(0x1000, u64::from(u32::MAX) - 3, (10, 12)), None);
    }

    #[test]
    fn warp_gathered_records_match_single_clamps() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_MESH: u32 = 10;
        const STRIDE: u32 = 12;
        let (src_buf, src_ptr, _) = warp.shared_buffer(4096);
        let (table_buf, table_ptr, table_gpu) = warp.shared_buffer(u64::from(GATHER_MOST) * 4);
        let (budget_buf, budget_ptr, _) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer(512);
        let scratch = warp.default_uav(64 << 10);

        let write_u32 = |ptr: *mut u8, byte_off: usize, vals: &[u32]| unsafe {
            for (i, &v) in vals.iter().enumerate() {
                (ptr.add(byte_off) as *mut u32).add(i).write_volatile(v);
            }
        };
        unsafe {
            std::ptr::write_bytes(src_ptr, 0, 4096);
            std::ptr::write_bytes(budget_ptr, 0, 256);
            std::ptr::write_bytes(readback_ptr, 0, 512);
        }
        // Scattered records in `src_buf`: offsets 64, 512, 1024.
        write_u32(src_ptr, 64, &[9, 1, 1]);
        write_u32(src_ptr, 512, &[70000, 4, 2]); // clamped to 65535 x 1 x 1
        write_u32(src_ptr, 1024, &[0xffffffff, 1, 1]); // garbage 0xffffffff clamped to 65535 x 1 x 1

        let mut set = GatherSet::new(
            src_buf.clone(),
            1,
            (KIND_MESH, STRIDE),
            scratch.clone(),
            (table_gpu, table_ptr),
        );
        assert_eq!(set.slot(64), Some(0));
        assert_eq!(set.slot(512), Some(1));
        assert_eq!(set.slot(64), Some(0), "duplicate offset reuses existing slot");
        assert_eq!(set.slot(1024), Some(2));
        let _keep_table = table_buf;

        let l2 = warp.list.cast::<ID3D12GraphicsCommandList2>().unwrap();
        assert!(record_gathers_with(
            &warp.clamp,
            &warp.list,
            &l2,
            &[set.plan()],
            &budget_buf,
        ));
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &scratch,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list.CopyBufferRegion(&readback_buf, 0, &scratch, 0, 3 * u64::from(STRIDE));
            warp.list.ResourceBarrier(&[transition(
                &scratch,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        }
        warp.submit_and_wait();

        let words = unsafe { std::slice::from_raw_parts(readback_ptr.cast::<u32>().cast_const(), 9).to_vec() };
        assert_eq!(&words[0..3], &[9, 1, 1]);
        assert_eq!(&words[3..6], &[65535, 1, 1]);
        assert_eq!(&words[6..9], &[65535, 1, 1]);
    }
}

/// What one small draw costs this card, bare and with one thing changed
/// between every two draws: a measurement, not a check (it asserts only
/// that the card answered). The timing run of the lit scene (run 187) has
/// some 11,000 small commands a frame at 5 to 16 microseconds each, and no
/// log says whether that is what we put around a command or what the card
/// takes for one. Each variant here is 10,000 one-triangle draws into a
/// 2560x1440 target, timed on the card by one pair of timestamps around
/// the whole batch (not one per command, which would time the stamps).
///
/// Run by name, no game needed:
/// `cargo test -p victory-gpu --release -- --ignored bare_command_cost --nocapture`
#[cfg(test)]
mod bare_cost {
    use windows::Win32::Foundation::{HANDLE, RECT};
    use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_12_0, D3D_PRIMITIVE_TOPOLOGY};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::*;
    use windows::Win32::Graphics::Dxgi::*;
    use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
    use windows::core::{Interface, s};

    use crate::host::{compile, transition};

    /// Draws in a batch.
    const DRAWS: u32 = 10_000;
    const WIDTH: u32 = 2560;
    const HEIGHT: u32 = 1440;

    /// One small triangle in the middle of the target (about 180 pixels),
    /// and two pixel programs that read one texel of a texture through a
    /// table, so that the view heap set is in use.
    const DRAW: &str = r#"
#define RS "DescriptorTable(SRV(t0))"
Texture2D<float4> image : register(t0);
[RootSignature(RS)]
float4 corner(uint i : SV_VertexID) : SV_Position {
    float2 p = float2((i << 1) & 2, i & 2);
    return float4(p * 0.02 - 0.01, 0.5, 1.0);
}
[RootSignature(RS)]
float4 first(float4 at : SV_Position) : SV_Target {
    return image.Load(int3(0, 0, 0));
}
[RootSignature(RS)]
float4 second(float4 at : SV_Position) : SV_Target {
    return 1.0 - image.Load(int3(0, 0, 0));
}
"#;

    /// A compute pass of next to nothing, as our clamp of an indirect
    /// record is: one group of one thread that writes one word.
    const WORK: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer words : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void work() {
    words.Store(0, words.Load(0) + 1);
}
"#;

    /// What changes between every two draws of a batch.
    #[derive(Clone, Copy, PartialEq)]
    enum Between {
        /// Nothing.
        Nothing,
        /// The pipeline: two alternate.
        Pipeline,
        /// A second texture goes from target to shader read and back.
        BarrierPair,
        /// The depth texture bound for testing goes from write to read or
        /// back.
        DepthMove,
        /// The pair of descriptor heaps: two pairs alternate.
        Heaps,
        /// The colour target: two alternate.
        Targets,
        /// A compute pass of ours of one group, and the pipeline back.
        Compute,
        /// The same between two waits for every earlier shader write, as
        /// the clamp records them.
        ComputeWithWaits,
    }

    struct Rig {
        queue: ID3D12CommandQueue,
        allocator: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
        fence: ID3D12Fence,
        submitted: u64,
        event: HANDLE,
        root: ID3D12RootSignature,
        pipelines: [ID3D12PipelineState; 2],
        work_root: ID3D12RootSignature,
        work: ID3D12PipelineState,
        words: ID3D12Resource,
        other: ID3D12Resource,
        depth: ID3D12Resource,
        /// Kept alive: the views below name them.
        _kept: Vec<ID3D12Resource>,
        _target_heaps: [ID3D12DescriptorHeap; 2],
        heap_pairs: [[Option<ID3D12DescriptorHeap>; 2]; 2],
        tables: [D3D12_GPU_DESCRIPTOR_HANDLE; 2],
        targets: [D3D12_CPU_DESCRIPTOR_HANDLE; 2],
        depth_written: D3D12_CPU_DESCRIPTOR_HANDLE,
        depth_tested: D3D12_CPU_DESCRIPTOR_HANDLE,
        stamps: ID3D12QueryHeap,
        back: ID3D12Resource,
        card: String,
    }

    fn texture(
        device: &ID3D12Device,
        (width, height): (u32, u32),
        format: DXGI_FORMAT,
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
    ) -> ID3D12Resource {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(width),
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: flags,
            ..Default::default()
        };
        let mut out: Option<ID3D12Resource> = None;
        // SAFETY: plain texture creation.
        unsafe {
            device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut out)
                .unwrap();
        }
        out.unwrap()
    }

    fn buffer(
        device: &ID3D12Device,
        size: u64,
        kind: D3D12_HEAP_TYPE,
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
    ) -> ID3D12Resource {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: kind,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: flags,
            ..Default::default()
        };
        let mut out: Option<ID3D12Resource> = None;
        // SAFETY: plain buffer creation.
        unsafe {
            device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut out)
                .unwrap();
        }
        out.unwrap()
    }

    fn heap(device: &ID3D12Device, kind: D3D12_DESCRIPTOR_HEAP_TYPE, count: u32, visible: bool) -> ID3D12DescriptorHeap {
        let flags = if visible {
            D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE
        } else {
            D3D12_DESCRIPTOR_HEAP_FLAG_NONE
        };
        // SAFETY: plain heap creation.
        unsafe {
            device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: count,
                    Flags: flags,
                    NodeMask: 0,
                })
                .unwrap()
        }
    }

    fn pipeline(device: &ID3D12Device, root: &ID3D12RootSignature, vs: &[u8], ps: &[u8]) -> ID3D12PipelineState {
        let code = |bytes: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes.as_ptr().cast(),
            BytecodeLength: bytes.len(),
        };
        let mut blend = D3D12_BLEND_DESC::default();
        blend.RenderTarget[0].RenderTargetWriteMask = 0xf;
        let mut formats = [DXGI_FORMAT(0); 8];
        formats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        // SAFETY: the description points at live data and never releases
        // its pointer to the root signature.
        unsafe {
            let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(root),
                VS: code(vs),
                PS: code(ps),
                BlendState: blend,
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                // Depth is tested and never written, so the texture may be
                // in the write state or in the read state under one view.
                DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                    DepthEnable: true.into(),
                    DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
                    DepthFunc: D3D12_COMPARISON_FUNC_LESS_EQUAL,
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                RTVFormats: formats,
                DSVFormat: DXGI_FORMAT_D32_FLOAT,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            device.CreateGraphicsPipelineState(&desc).unwrap()
        }
    }

    impl Rig {
        /// The default adapter, as the layer takes it; `None` without one
        /// or on a software adapter.
        fn new() -> Option<Self> {
            let vs = compile(DRAW, s!("corner"), s!("vs_5_1")).ok()?;
            let ps = [
                compile(DRAW, s!("first"), s!("ps_5_1")).ok()?,
                compile(DRAW, s!("second"), s!("ps_5_1")).ok()?,
            ];
            let cs = compile(WORK, s!("work"), s!("cs_5_1")).ok()?;
            // SAFETY: plain object creation on a device of our own; the
            // views are made in heaps that live as long as the rig.
            unsafe {
                let mut device: Option<ID3D12Device> = None;
                D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
                let device = device?;
                let desc = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
                    .and_then(|factory| factory.EnumAdapterByLuid::<IDXGIAdapter3>(device.GetAdapterLuid()))
                    .ok()?
                    .cast::<IDXGIAdapter1>()
                    .ok()?
                    .GetDesc1()
                    .ok()?;
                if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
                    return None;
                }
                let card = String::from_utf16_lossy(&desc.Description)
                    .trim_end_matches('\0')
                    .to_owned();
                let queue: ID3D12CommandQueue = device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })
                    .ok()?;
                let allocator: ID3D12CommandAllocator =
                    device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).ok()?;
                let list: ID3D12GraphicsCommandList = device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .ok()?;
                let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
                let event = CreateEventW(None, false, false, None).ok()?;

                let root: ID3D12RootSignature = device.CreateRootSignature(0, &vs[..]).ok()?;
                let pipelines = [
                    pipeline(&device, &root, &vs, &ps[0]),
                    pipeline(&device, &root, &vs, &ps[1]),
                ];
                let work_root: ID3D12RootSignature = device.CreateRootSignature(0, &cs[..]).ok()?;
                let work: ID3D12PipelineState = device
                    .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&work_root),
                        CS: D3D12_SHADER_BYTECODE {
                            pShaderBytecode: cs.as_ptr().cast(),
                            BytecodeLength: cs.len(),
                        },
                        ..Default::default()
                    })
                    .ok()?;

                let size = (WIDTH, HEIGHT);
                let colour = |state| texture(&device, size, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, state);
                let target = [
                    colour(D3D12_RESOURCE_STATE_RENDER_TARGET),
                    colour(D3D12_RESOURCE_STATE_RENDER_TARGET),
                ];
                let other = colour(D3D12_RESOURCE_STATE_RENDER_TARGET);
                let depth = texture(
                    &device,
                    size,
                    DXGI_FORMAT_D32_FLOAT,
                    D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                    D3D12_RESOURCE_STATE_DEPTH_WRITE,
                );
                let image = texture(
                    &device,
                    (4, 4),
                    DXGI_FORMAT_R8G8B8A8_UNORM,
                    D3D12_RESOURCE_FLAG_NONE,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                );
                let words = buffer(
                    &device,
                    64 << 10,
                    D3D12_HEAP_TYPE_DEFAULT,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                );
                let back = buffer(
                    &device,
                    16,
                    D3D12_HEAP_TYPE_READBACK,
                    D3D12_RESOURCE_FLAG_NONE,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                );
                let mut stamps: Option<ID3D12QueryHeap> = None;
                device
                    .CreateQueryHeap(
                        &D3D12_QUERY_HEAP_DESC {
                            Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                            Count: 2,
                            NodeMask: 0,
                        },
                        &mut stamps,
                    )
                    .ok()?;
                let stamps = stamps?;

                // Target and depth views.
                let rtvs = heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2, false);
                let rtv_step = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) as usize;
                let mut targets = [rtvs.GetCPUDescriptorHandleForHeapStart(); 2];
                targets[1].ptr += rtv_step;
                device.CreateRenderTargetView(&target[0], None, targets[0]);
                device.CreateRenderTargetView(&target[1], None, targets[1]);
                let dsvs = heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 2, false);
                let dsv_step = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV) as usize;
                let depth_written = dsvs.GetCPUDescriptorHandleForHeapStart();
                let mut depth_tested = depth_written;
                depth_tested.ptr += dsv_step;
                device.CreateDepthStencilView(&depth, None, depth_written);
                let tested = D3D12_DEPTH_STENCIL_VIEW_DESC {
                    Format: DXGI_FORMAT_D32_FLOAT,
                    ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
                    Flags: D3D12_DSV_FLAG_READ_ONLY_DEPTH,
                    ..Default::default()
                };
                device.CreateDepthStencilView(&depth, Some(&tested), depth_tested);

                // Two pairs of shader-visible heaps: views (one view of the
                // small texture each) and samplers (none used).
                let view_heaps = [
                    heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, true),
                    heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, true),
                ];
                let sampler_heaps = [
                    heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 1, true),
                    heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 1, true),
                ];
                for views in &view_heaps {
                    device.CreateShaderResourceView(&image, None, views.GetCPUDescriptorHandleForHeapStart());
                }
                let tables = [
                    view_heaps[0].GetGPUDescriptorHandleForHeapStart(),
                    view_heaps[1].GetGPUDescriptorHandleForHeapStart(),
                ];
                let heap_pairs = [
                    [Some(view_heaps[0].clone()), Some(sampler_heaps[0].clone())],
                    [Some(view_heaps[1].clone()), Some(sampler_heaps[1].clone())],
                ];

                // The first list: both targets and the depth texture cleared.
                list.ClearRenderTargetView(targets[0], &[0.0; 4], None);
                list.ClearRenderTargetView(targets[1], &[0.0; 4], None);
                list.ClearDepthStencilView(depth_written, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, None);
                let [first_target, second_target] = target;
                let mut rig = Rig {
                    queue,
                    allocator,
                    list,
                    fence,
                    submitted: 0,
                    event,
                    root,
                    pipelines,
                    work_root,
                    work,
                    words,
                    other,
                    depth,
                    _kept: vec![first_target, second_target, image],
                    _target_heaps: [rtvs, dsvs],
                    heap_pairs,
                    tables,
                    targets,
                    depth_written,
                    depth_tested,
                    stamps,
                    back,
                    card,
                };
                rig.finish();
                Some(rig)
            }
        }

        /// Closes the open list, runs it, waits for the card and opens the
        /// list again.
        fn finish(&mut self) {
            self.submitted += 1;
            // SAFETY: our list, queue and fence; the allocator is reset
            // only after the card finished everything recorded from it.
            unsafe {
                self.list.Close().unwrap();
                let run: ID3D12CommandList = self.list.cast().unwrap();
                self.queue.ExecuteCommandLists(&[Some(run)]);
                self.queue.Signal(&self.fence, self.submitted).unwrap();
                self.fence.SetEventOnCompletion(self.submitted, self.event).unwrap();
                WaitForSingleObject(self.event, INFINITE);
                self.allocator.Reset().unwrap();
                self.list.Reset(&self.allocator, None).unwrap();
            }
        }

        /// What every list starts with: a new list has nothing set.
        fn begin(&self) {
            let l = &self.list;
            // SAFETY: an open list and live objects and views.
            unsafe {
                l.SetDescriptorHeaps(&self.heap_pairs[0]);
                l.SetGraphicsRootSignature(&self.root);
                l.SetPipelineState(&self.pipelines[0]);
                l.SetGraphicsRootDescriptorTable(0, self.tables[0]);
                l.OMSetRenderTargets(1, Some(&self.targets[0]), false, Some(&self.depth_tested));
                l.RSSetViewports(&[D3D12_VIEWPORT {
                    TopLeftX: 0.0,
                    TopLeftY: 0.0,
                    Width: WIDTH as f32,
                    Height: HEIGHT as f32,
                    MinDepth: 0.0,
                    MaxDepth: 1.0,
                }]);
                l.RSSetScissorRects(&[RECT {
                    left: 0,
                    top: 0,
                    right: WIDTH as i32,
                    bottom: HEIGHT as i32,
                }]);
                // A triangle list.
                l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(4));
            }
        }

        /// One batch of [`DRAWS`] draws in `lists` lists run one by one,
        /// with `between` recorded before every draw: the microseconds a
        /// draw took on the card, and the milliseconds recording took.
        fn batch(&mut self, between: Between, lists: u32) -> (f64, f64) {
            let (target, read, written, tested) = (
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_DEPTH_WRITE,
                D3D12_RESOURCE_STATE_DEPTH_READ,
            );
            let started = std::time::Instant::now();
            let mut drawn = 0u32;
            for part in 0..lists {
                let l = &self.list;
                // SAFETY: an open list; our query heap; live objects.
                unsafe {
                    if part == 0 {
                        l.EndQuery(&self.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0);
                    }
                }
                self.begin();
                // SAFETY: an open list with everything set; live objects.
                // Every state moved is moved back within the batch (the
                // draws of a list are an even number).
                unsafe {
                    for _ in 0..DRAWS / lists {
                        let odd = (drawn % 2) as usize;
                        match between {
                            Between::Nothing => {}
                            Between::Pipeline => l.SetPipelineState(&self.pipelines[odd]),
                            Between::BarrierPair => {
                                l.ResourceBarrier(&[transition(&self.other, target, read)]);
                                l.ResourceBarrier(&[transition(&self.other, read, target)]);
                            }
                            Between::DepthMove => {
                                let (from, to) = if odd == 0 { (written, tested) } else { (tested, written) };
                                l.ResourceBarrier(&[transition(&self.depth, from, to)]);
                            }
                            Between::Heaps => {
                                l.SetDescriptorHeaps(&self.heap_pairs[odd]);
                                l.SetGraphicsRootDescriptorTable(0, self.tables[odd]);
                            }
                            Between::Targets => {
                                l.OMSetRenderTargets(1, Some(&self.targets[odd]), false, Some(&self.depth_tested));
                            }
                            Between::Compute | Between::ComputeWithWaits => {
                                let waits = between == Between::ComputeWithWaits;
                                if waits {
                                    l.ResourceBarrier(&[super::uav_barrier()]);
                                }
                                l.SetComputeRootSignature(&self.work_root);
                                l.SetPipelineState(&self.work);
                                l.SetComputeRootUnorderedAccessView(0, self.words.GetGPUVirtualAddress());
                                l.Dispatch(1, 1, 1);
                                if waits {
                                    l.ResourceBarrier(&[super::uav_barrier()]);
                                }
                                l.SetPipelineState(&self.pipelines[0]);
                            }
                        }
                        l.DrawInstanced(3, 1, 0, 0);
                        drawn += 1;
                    }
                    if part + 1 < lists {
                        // The next list of the batch: this one is sent on
                        // its own and the list is opened again at once
                        // (the allocator is not reset before the card is
                        // done with all of them).
                        l.Close().unwrap();
                        let run: ID3D12CommandList = l.cast().unwrap();
                        self.queue.ExecuteCommandLists(&[Some(run)]);
                        l.Reset(&self.allocator, None).unwrap();
                    } else {
                        l.EndQuery(&self.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 1);
                        l.ResolveQueryData(&self.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &self.back, 0);
                    }
                }
            }
            let recording = started.elapsed().as_secs_f64() * 1e3;
            self.finish();
            let mut data = std::ptr::null_mut();
            let range = D3D12_RANGE { Begin: 0, End: 16 };
            // SAFETY: a read-back buffer the finished list resolved two
            // timestamps into.
            let ticks = unsafe {
                self.back.Map(0, Some(&range), Some(&mut data)).unwrap();
                let stamps = std::slice::from_raw_parts(data.cast::<u64>(), 2).to_vec();
                self.back.Unmap(0, None);
                stamps[1].saturating_sub(stamps[0])
            };
            // SAFETY: a plain query of our queue.
            let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1).max(1);
            (ticks as f64 * 1e6 / frequency as f64 / f64::from(drawn.max(1)), recording)
        }
    }

    #[test]
    #[ignore = "a measurement on the card: run it by name with --ignored --nocapture"]
    fn bare_command_cost() {
        let Some(mut rig) = Rig::new() else {
            eprintln!("no hardware adapter, or our shaders did not compile; skipped");
            return;
        };
        eprintln!("bare command cost on {}: {DRAWS} one-triangle draws into {WIDTH}x{HEIGHT}, three batches each", rig.card);
        // The first batch pays for whatever the driver does at first use.
        rig.batch(Between::Nothing, 1);
        let variants = [
            ("(a) nothing changes between draws", Between::Nothing, 1),
            ("(b) another pipeline every draw", Between::Pipeline, 1),
            ("(c) a barrier pair on a second texture every draw", Between::BarrierPair, 1),
            ("(d) the tested depth texture moved between write and read every draw", Between::DepthMove, 1),
            ("(e) another pair of descriptor heaps every draw", Between::Heaps, 1),
            ("(f) another colour target every draw", Between::Targets, 1),
            ("(g) nothing changes, in 100 lists run one by one", Between::Nothing, 100),
            ("(h) a compute pass of ours of one group before every draw", Between::Compute, 1),
            ("(i) the same pass between two waits for earlier shader writes", Between::ComputeWithWaits, 1),
        ];
        let mut least = Vec::new();
        for (name, between, lists) in variants {
            let mut runs: Vec<(f64, f64)> = (0..3).map(|_| rig.batch(between, lists)).collect();
            runs.sort_by(|a, b| a.0.total_cmp(&b.0));
            eprintln!(
                "{name}: {:.2} microseconds a draw on the card (the three: {:.2}, {:.2}, {:.2}); recording took {:.1} ms",
                runs[0].0, runs[0].0, runs[1].0, runs[2].0, runs[0].1
            );
            least.push(runs[0].0);
        }
        assert!(least.iter().all(|us| us.is_finite() && *us > 0.0), "the card gave no time for a batch");
    }
}
