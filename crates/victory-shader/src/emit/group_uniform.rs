//! Lane- and workgroup-uniformity analysis for barrier placement in compute lowering.
//!
//! Separated from `lower.rs` to keep the emitter within the per-file size limit.

use std::collections::{HashMap, HashSet};

use crate::gfx10::Fields;
use crate::ir::{EXEC, Program, Reg, VCC, Value};
use crate::structure::{Atom, Bool, LaneExit, Structured};

use super::reads_mask_as_data;

/// Per IR block: every lane of the wave that started the program runs it – no per-lane branch
/// separates it from the entry (it lies in no lane branch's region: the blocks reachable from
/// the branch that do not post-dominate it). None with an indirect exit (not followed).
pub(super) fn all_lane_blocks(st: &Structured) -> Option<Vec<bool>> {
    let n = st.exits.len();
    let mut all = vec![true; n];
    for (_, region) in lane_regions(st)? {
        for b in (0..n).filter(|&b| region[b]) {
            all[b] = false;
        }
    }
    Some(all)
}

/// Every per-lane branch (by its block) with its region: the blocks reachable from it that do
/// not post-dominate it, which only some lanes run. Not the branches the wave takes together
/// ([`Structured::wide`]), nor those on scalar conditions. None with an indirect exit.
pub(super) fn lane_regions(st: &Structured) -> Option<Vec<(usize, Vec<bool>)>> {
    branch_regions(st, |d, when| {
        st.wide.contains(&d) || st.bools.support(when).iter().all(|a| matches!(a, Atom::Uniform(_)))
    })
}

/// Branches not skipped by `uniform_branch`, each with its region (the blocks reachable from
/// the branch that do not post-dominate it). `None` with an indirect exit.
fn branch_regions(
    st: &Structured,
    uniform_branch: impl Fn(usize, Bool) -> bool,
) -> Option<Vec<(usize, Vec<bool>)>> {
    let n = st.exits.len();
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (b, e) in st.exits.iter().enumerate() {
        match *e {
            LaneExit::Jump(t) => succ[b].push(t),
            LaneExit::Branch { taken, not_taken, .. } => {
                succ[b].push(taken);
                succ[b].push(not_taken);
            }
            LaneExit::End => {}
            LaneExit::Indirect => return None,
        }
    }
    // post-dominators (bitsets; a virtual exit after every End)
    let words = n.div_ceil(64);
    let full = vec![u64::MAX; words];
    let mut pdom: Vec<Vec<u64>> = (0..n)
        .map(|b| {
            if succ[b].is_empty() {
                let mut v = vec![0; words];
                v[b / 64] |= 1 << (b % 64);
                v
            } else {
                full.clone()
            }
        })
        .collect();
    let mut changed = true;
    while changed {
        changed = false;
        for b in (0..n).rev() {
            if succ[b].is_empty() {
                continue;
            }
            let mut v = full.clone();
            for &t in &succ[b] {
                for w in 0..words {
                    v[w] &= pdom[t][w];
                }
            }
            v[b / 64] |= 1 << (b % 64);
            if v != pdom[b] {
                pdom[b] = v;
                changed = true;
            }
        }
    }
    let mut regions = Vec::new();
    for (d, e) in st.exits.iter().enumerate() {
        let LaneExit::Branch { when, .. } = *e else { continue };
        if uniform_branch(d, when) {
            continue;
        }
        // blocks reachable from the branch that do not post-dominate it
        let mut seen = vec![false; n];
        let mut stack: Vec<usize> = succ[d].clone();
        while let Some(b) = stack.pop() {
            if seen[b] || pdom[d][b / 64] >> (b % 64) & 1 == 1 {
                continue;
            }
            seen[b] = true;
            stack.extend(succ[b].iter().copied());
        }
        regions.push((d, seen));
    }
    Some(regions)
}

/// Whether `r` can carry a workgroup-uniform scalar value (SGPR other than `EXEC`/`VCC`, or `SCC`).
fn is_group_scalar_reg(r: Reg) -> bool {
    match r {
        Reg::Scc => true,
        Reg::S(n) => {
            n != EXEC
                && n != EXEC + 1
                && n != VCC
                && n != VCC + 1
                && n < crate::ir::PARKED
        }
        Reg::V(_) => false,
    }
}

/// Scalar values guaranteed equal across every console wave in a compute threadgroup:
/// entry SGPR inputs (user data, group IDs), pure scalar/SMEM-load results of uniform inputs,
/// and uniform scalar phis outside non-uniform branch regions and joins.
fn group_uniform_values(
    p: &Program,
    st: &Structured,
    ignored_branches: &HashSet<usize>,
) -> Option<(HashSet<Value>, Vec<(usize, Vec<bool>)>)> {
    let mut uniform: HashSet<Value> = HashSet::new();
    for &(r, v) in &p.inputs {
        if matches!(r, Reg::S(_)) && is_group_scalar_reg(r) {
            uniform.insert(v);
        }
    }
    for blk in &p.blocks {
        for phi in &blk.phis {
            if is_group_scalar_reg(phi.reg) {
                uniform.insert(phi.value);
            }
        }
        for inst in &blk.insts {
            let scalar_ok = match &inst.fields {
                Fields::Sop { .. } => {
                    let n = inst.name();
                    !(n.starts_with("s_getreg")
                        || n.starts_with("s_memtime")
                        || n.starts_with("s_memrealtime"))
                }
                Fields::Smem { .. } => inst.name().contains("load"),
                _ => false,
            };
            if scalar_ok {
                for &(r, v) in &inst.writes {
                    if is_group_scalar_reg(r) {
                        uniform.insert(v);
                    }
                }
            }
        }
    }
    loop {
        let regions = branch_regions(st, |d, when| {
            ignored_branches.contains(&d)
                || (!st.wide.contains(&d)
                    && st
                        .bools
                        .support(when)
                        .iter()
                        .all(|a| matches!(a, Atom::Uniform(v) if uniform.contains(v))))
        })?;
        let mut changed = false;
        for (b, blk) in p.blocks.iter().enumerate() {
            for phi in &blk.phis {
                if !uniform.contains(&phi.value) {
                    continue;
                }
                let all_same = phi.args.first().is_some_and(|a| phi.args.iter().all(|x| x == a));
                let args_ok = phi.args.iter().all(|a| uniform.contains(a));
                let control_ok = all_same
                    || !regions.iter().any(|(_, reg)| {
                        reg.get(b).copied().unwrap_or(false)
                            || blk.preds.iter().any(|&q| reg.get(q).copied().unwrap_or(false))
                    });
                if !(args_ok && control_ok) {
                    uniform.remove(&phi.value);
                    changed = true;
                }
            }
            for inst in &blk.insts {
                if inst.writes.iter().all(|&(_, v)| !uniform.contains(&v)) {
                    continue;
                }
                let inputs_ok = inst
                    .reads
                    .iter()
                    .all(|&(r, v)| is_group_scalar_reg(r) && uniform.contains(&v));
                if !inputs_ok {
                    for &(_, v) in &inst.writes {
                        if uniform.remove(&v) {
                            changed = true;
                        }
                    }
                }
            }
        }
        if !changed {
            return Some((uniform, regions));
        }
    }
}

/// For multi-wave (`group > 64`) compute groups using workgroup-wide barriers (`exchange_halves`,
/// `split_lds`): blocks guaranteed to be executed by every console wave in the workgroup, plus
/// any pure wave-varying scalar forward-skips flattened into unconditional fallthrough + `select`
/// at their join (e.g. `if (!WaveActiveAnyTrue(a)) active = WaveActiveAnyTrue(b)`).
pub(super) fn all_group_blocks(
    p: &Program,
    st: &Structured,
    allow_flatten: bool,
) -> Option<(Vec<bool>, HashMap<usize, (usize, usize)>)> {
    let n = st.exits.len();
    let order = crate::ir::reverse_postorder(&p.blocks);
    let mut rpo = vec![usize::MAX; n];
    for (idx, &b) in order.iter().enumerate() {
        rpo[b] = idx;
    }
    let is_pure_body = |b: usize| -> bool {
        p.blocks[b].insts.iter().all(|inst| match &inst.fields {
            Fields::Valu(_) => true,
            Fields::Sop { .. } => {
                let name = inst.name();
                !(name.starts_with("s_setpc")
                    || name.starts_with("s_swappc")
                    || name.starts_with("s_call")
                    || name.starts_with("s_rfe"))
            }
            Fields::Sopp { .. } => {
                let name = inst.name();
                !matches!(name, "s_barrier" | "s_endpgm" | "s_sendmsg" | "s_trap")
            }
            _ => false,
        })
    };
    let needs_exchange = |b: usize| -> bool {
        p.blocks[b].insts.iter().any(|inst| {
            let name = inst.name();
            reads_mask_as_data(st, inst)
                || name.starts_with("v_mbcnt")
                || matches!(name, "v_readlane_b32" | "v_writelane_b32" | "v_readfirstlane_b32")
                || inst.writes.iter().any(|&(_, v)| st.scc_some.contains_key(&v))
        })
    };

    let (uniform, initial_regions) = group_uniform_values(p, st, &HashSet::new())?;
    let mut flat_skips: HashMap<usize, (usize, usize)> = HashMap::new();
    let mut ignored: HashSet<usize> = HashSet::new();
    if allow_flatten {
        let mut initial_all = vec![true; n];
        for (_, reg) in &initial_regions {
            for b in (0..n).filter(|&b| reg[b]) {
                initial_all[b] = false;
            }
        }
        let mut used_joins: HashSet<usize> = HashSet::new();
        for &(d, _) in &initial_regions {
            if !initial_all[d] || st.wide.contains(&d) {
                continue;
            }
            let LaneExit::Branch { when, taken, not_taken } = st.exits[d] else {
                continue;
            };
            if !st.bools.support(when).iter().all(|a| matches!(a, Atom::Uniform(_))) {
                continue;
            }
            if st.bools.support(when).iter().all(|a| matches!(a, Atom::Uniform(v) if uniform.contains(v))) {
                continue;
            }
            // Pure single-block triangle skip: `d` -> `body` -> `join` and `d` -> `join`.
            let pair = if st.exits.get(not_taken) == Some(&LaneExit::Jump(taken)) {
                Some((not_taken, taken))
            } else if st.exits.get(taken) == Some(&LaneExit::Jump(not_taken)) {
                Some((taken, not_taken))
            } else {
                None
            };
            let Some((body, join)) = pair else { continue };
            if d == body || body == join || d == join {
                continue;
            }
            if rpo[d] >= rpo[join] || rpo[body] >= rpo[join] {
                continue;
            }
            if p.blocks[body].preds != [d] {
                continue;
            }
            let jpreds = &p.blocks[join].preds;
            if jpreds.len() != 2 || !jpreds.contains(&d) || !jpreds.contains(&body) {
                continue;
            }
            if used_joins.contains(&join) || !is_pure_body(body) || !needs_exchange(body) {
                continue;
            }
            used_joins.insert(join);
            flat_skips.insert(d, (body, join));
            ignored.insert(d);
        }
    }
    let (_, regions) = if ignored.is_empty() {
        (uniform, initial_regions)
    } else {
        group_uniform_values(p, st, &ignored)?
    };
    let mut all = vec![true; n];
    for (_, reg) in &regions {
        for b in (0..n).filter(|&b| reg[b]) {
            all[b] = false;
        }
    }
    flat_skips.retain(|&d, _| all[d]);
    Some((all, flat_skips))
}

/// LDS accesses of a console wave run as two host waves, where some sit in blocks that only
/// some lanes run (`all` is [`all_lane_blocks`]): per IR block, whether the host waves meet (a
/// group barrier) at its start and before its exit. `None` if barriers cannot keep the
/// console's order.
///
/// The console runs the wave's lanes in step, so an LDS write is seen by every later read. Two
/// host waves keep that order if they meet between any two accesses of which one writes. A
/// barrier must be run by every thread, so it cannot stand in a lane branch's region; but
/// every way into and out of such a region passes a block all lanes run, and the waves meet
/// there: before the exit of an all-lane block into a region with LDS accesses, and at the
/// start of an all-lane block entered from one. Accesses in all-lane blocks get their own
/// barrier as before. What that cannot order is a write and a read inside one region (blocks
/// joined without an all-lane block between them): refused.
/// shortcut: two writes in one region are taken not to hit the same dword from both halves.
pub(super) fn lds_meetings(p: &Program, st: &Structured, all: &[bool]) -> Option<Vec<(bool, bool)>> {
    let n = st.exits.len();
    let succ = |b: usize| -> Vec<usize> {
        match st.exits[b] {
            LaneExit::Jump(t) => vec![t],
            LaneExit::Branch { taken, not_taken, .. } => vec![taken, not_taken],
            LaneExit::End | LaneExit::Indirect => Vec::new(),
        }
    };
    // regions: blocks not every lane runs, joined by the edges between them
    let mut region: Vec<usize> = (0..n).collect();
    fn find(region: &mut [usize], mut b: usize) -> usize {
        while region[b] != b {
            region[b] = region[region[b]];
            b = region[b];
        }
        b
    }
    for b in 0..n {
        for t in succ(b) {
            if !all[b] && !all[t] {
                let (x, y) = (find(&mut region, b), find(&mut region, t));
                region[x] = y;
            }
        }
    }
    // per region: LDS reads, LDS writes
    let mut reads = vec![false; n];
    let mut writes = vec![false; n];
    for (b, block) in p.blocks.iter().enumerate().take(n) {
        if all[b] {
            continue;
        }
        let r = find(&mut region, b);
        for i in &block.insts {
            let name = i.name();
            if !matches!(i.fields, Fields::Ds { .. }) || name.starts_with("ds_swizzle") || name.contains("permute") {
                continue;
            }
            if name.starts_with("ds_read") {
                reads[r] = true;
            } else {
                writes[r] = true;
            }
        }
    }
    if (0..n).any(|r| reads[r] && writes[r]) {
        return None;
    }
    let mut meet = vec![(false, false); n];
    for b in 0..n {
        for t in succ(b) {
            if all[b] && !all[t] {
                let r = find(&mut region, t);
                meet[b].1 |= reads[r] || writes[r];
            } else if !all[b] && all[t] {
                let r = find(&mut region, b);
                meet[t].0 |= reads[r] || writes[r];
            }
        }
    }
    Some(meet)
}
