//! Which title memory the title wrote since a host copy took it (decision
//! 0007, "Copies of title memory"). The runtime reserves title GPU memory
//! with write watching (`victory-hle` xmem), so the kernel keeps a dirty bit
//! per 4 KiB page; asking for a range's written pages also clears them. Host
//! copies of the same memory can overlap (nested buffers, textures in
//! buffers), so what one asks is kept here for the others: a journal of
//! each page's last write, numbered by the asking.
//!
//! Physical-page views (xmem's sections) cannot be write-watched. There a
//! copy leaves each 64 KiB block it took read-only, and our vectored handler
//! takes the first write to it: it notes the block as written and gives the
//! write access back. The runtime tells us when it maps or unmaps views
//! ([`title_memory_remapped`]), which resets their protection. Blocks found
//! wholly uncommitted are skipped while they remain so; one query can
//! confirm a whole region, including ordinary reservations between views.
//!
//! Writes into physical pages through another view are seen too: the
//! runtime tells us the views of the pages it wrote itself ([`writing`],
//! xmem's `write_physical` and DirectStorage's page destinations), and a
//! write by the title through any view of a guarded page faults like one
//! at the guarded address (the guard makes every view of the page
//! read-only, [`open`] gives them all back; not a page of more than
//! [`MANY_VIEWS`] views). A
//! system call writing into a guarded block fails (none is known to).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Mutex, RwLock};

/// The unit tracked and copied: 64 KiB, the console's page (the kernel's
/// 4 KiB pages fold into it). Finer units made the checks of the world's
/// big buffers cost more than the copies they saved.
pub(crate) const PAGE: usize = 64 << 10;
/// The kernel's page, the unit of its answers.
const KERNEL_PAGE: usize = 4096;

/// A title-memory range whose changed pages another subsystem needs to read.
pub(crate) struct Tracked {
    start: usize,
    end: usize,
    seen: Mutex<BTreeMap<usize, u64>>,
}

/// Tracks a runtime GPU-memory range, including physical-page views when
/// `views` is true. Other process memory is not tracked.
pub(crate) fn track(start: usize, len: usize, views: bool) -> Option<Tracked> {
    let end = start.checked_add(len)?;
    if len == 0 || writes(start, end, views).is_none() {
        return None;
    }
    Some(Tracked {
        start,
        end,
        seen: Mutex::default(),
    })
}

struct Journal {
    /// Numbers the askings; 0 is "never".
    asked: u64,
    /// Page number -> the asking that last found it written.
    written: BTreeMap<usize, u64>,
    /// Write-watched page ranges asked ahead for a submission's refreshes
    /// ([`ask_ahead`], [`ask_once`]), kept sorted and non-overlapping:
    /// `writes` answers for a range inside one from the journal without
    /// another kernel call.
    ahead: Vec<(usize, usize)>,
    /// Guarded physical-page view ranges asked ahead ([`ask_ahead`],
    /// [`ask_once`]), kept sorted and non-overlapping. Separate from `ahead`
    /// so `writes(..., false)` never returns `Some` for guarded-only memory.
    ahead_guarded: Vec<(usize, usize)>,
    ahead_asked: u64,
    /// The ahead asking stands for a whole submission ([`ask_once`]): no
    /// other ahead asking is made, and `ahead`/`ahead_guarded` stay, until
    /// [`ask_once_end`].
    kept: bool,
}

impl Journal {
    fn covers_ahead(&self, first: usize, last: usize, guarded: bool) -> bool {
        covers(&self.ahead, first, last) || (guarded && covers(&self.ahead_guarded, first, last))
    }
}

static JOURNAL: RwLock<Journal> = RwLock::new(Journal {
    asked: 0,
    written: BTreeMap::new(),
    ahead: Vec::new(),
    ahead_guarded: Vec::new(),
    ahead_asked: 0,
    kept: false,
});

/// Serializes kernel write-watch / guard miss queries and journal commits
/// without holding `JOURNAL` across syscalls, so `ahead`-hit readers
/// (`asked`, `writes` fast path) run concurrently under `JOURNAL.read()`.
///
/// Lock order (never inverted):
/// - `WATCH_ASK` -> `JOURNAL` (never acquires `GUARDS` while holding `JOURNAL`)
/// - `WATCH_ASK` -> `GUARDS` -> `inplace` (never acquires `JOURNAL` while holding `GUARDS`)
/// - `on_write` (VEH) touches ONLY `READ_RECOVER_RIP` and `GUARDS` -> `inplace`.
static WATCH_ASK: Mutex<()> = Mutex::new(());

/// Monotonic revision incremented under `GUARDS` whenever guard state is
/// invalidated by `writing`, `title_memory_written`, `views_mapped`, or
/// `title_memory_remapped`. Paired with `SeqCst` checks under `JOURNAL.write()`
/// so `ask_once`/`ask_ahead` never publish stale `ahead_guarded` entries
/// across a concurrent invalidation.
static GUARD_REV: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

thread_local! {
    /// Per-thread scratch buffer for `GetWriteWatch` output addresses so
    /// queries do not mutate a shared buffer inside `Journal`.
    static FOUND: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
static KERNEL_ASKS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn journal_read() -> std::sync::RwLockReadGuard<'static, Journal> {
    JOURNAL.read().unwrap_or_else(|e| e.into_inner())
}

fn journal_write() -> std::sync::RwLockWriteGuard<'static, Journal> {
    JOURNAL.write().unwrap_or_else(|e| e.into_inner())
}

fn watch_ask() -> std::sync::MutexGuard<'static, ()> {
    WATCH_ASK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether `[first, last)` lies inside one interval of a sorted,
/// non-overlapping, coalesced slice.
fn covers(ranges: &[(usize, usize)], first: usize, last: usize) -> bool {
    let idx = ranges.partition_point(|&(a, _)| a <= first);
    idx > 0 && ranges[idx - 1].1 >= last
}

/// Whether `[first, last)` overlaps any interval of a sorted,
/// non-overlapping slice.
fn overlaps_any(ranges: &[(usize, usize)], first: usize, last: usize) -> bool {
    if first >= last || ranges.is_empty() {
        return false;
    }
    let idx = ranges.partition_point(|&(_, b)| b <= first);
    idx < ranges.len() && ranges[idx].0 < last
}

/// Sorts `ranges` and merges overlapping or adjacent intervals (`next.0 <= cur.1`)
/// without bridging gaps.
fn coalesce_ranges(ranges: &mut Vec<(usize, usize)>) {
    if ranges.len() <= 1 {
        return;
    }
    ranges.sort_unstable();
    let mut write = 0;
    for read in 1..ranges.len() {
        let (a, b) = ranges[read];
        if a <= ranges[write].1 {
            ranges[write].1 = ranges[write].1.max(b);
        } else {
            write += 1;
            ranges[write] = (a, b);
        }
    }
    ranges.truncate(write + 1);
}

/// Inserts `[first, last)` into a sorted, non-overlapping interval list,
/// merging overlapping or adjacent entries.
fn insert_coalesced(ranges: &mut Vec<(usize, usize)>, first: usize, last: usize) {
    if first >= last {
        return;
    }
    let start = ranges.partition_point(|&(_, b)| b < first);
    let mut merged_first = first;
    let mut merged_last = last;
    let mut end = start;
    while end < ranges.len() && ranges[end].0 <= merged_last {
        merged_first = merged_first.min(ranges[end].0);
        merged_last = merged_last.max(ranges[end].1);
        end += 1;
    }
    ranges.splice(start..end, std::iter::once((merged_first, merged_last)));
}

/// Removes `[first, last)` from a sorted, non-overlapping interval list,
/// splitting any interval that extends beyond it.
fn evict_range(ranges: &mut Vec<(usize, usize)>, first: usize, last: usize) {
    if first >= last || ranges.is_empty() {
        return;
    }
    let start = ranges.partition_point(|&(_, b)| b <= first);
    if start >= ranges.len() || ranges[start].0 >= last {
        return;
    }
    let mut end = start;
    while end < ranges.len() && ranges[end].0 < last {
        end += 1;
    }
    let left = (ranges[start].0 < first).then_some((ranges[start].0, first));
    let right = (ranges[end - 1].1 > last).then_some((last, ranges[end - 1].1));
    let mut replacement = [(0usize, 0usize); 2];
    let mut n = 0;
    if let Some(r) = left {
        replacement[n] = r;
        n += 1;
    }
    if let Some(r) = right {
        replacement[n] = r;
        n += 1;
    }
    ranges.splice(start..end, replacement[..n].iter().copied());
}

/// Folds kernel 4 KiB addresses into 64 KiB page numbers, collapsing
/// consecutive duplicates before they touch `Journal::written`.
fn dedup_pages(addrs: &[usize], out: &mut Vec<usize>) {
    let mut last_page = usize::MAX;
    for &at in addrs {
        let page = at / PAGE;
        if page != last_page {
            out.push(page);
            last_page = page;
        }
    }
}

/// Records `pages` as written at asking `now`, skipping consecutive duplicates.
fn insert_dedup(written: &mut BTreeMap<usize, u64>, pages: &[usize], now: u64) {
    let mut last_page = usize::MAX;
    for &page in pages {
        if page != last_page {
            written.insert(page, now);
            last_page = page;
        }
    }
}

fn overlaps_ahead_guarded(first: usize, last: usize) -> bool {
    overlaps_any(&journal_read().ahead_guarded, first, last)
}

/// Evicts `[first, last)` and any individual `extra_pages` from
/// `Journal::ahead_guarded`. Fast-paths out under `JOURNAL.read()` when
/// nothing overlaps. Must never be called while holding `GUARDS`.
fn evict_ahead_guarded(first: usize, last: usize, extra_pages: &[usize]) {
    let needs_write = {
        let j = journal_read();
        !j.ahead_guarded.is_empty()
            && (overlaps_any(&j.ahead_guarded, first, last)
                || extra_pages
                    .iter()
                    .any(|&p| overlaps_any(&j.ahead_guarded, p, p + 1)))
    };
    if !needs_write {
        return;
    }
    let mut j = journal_write();
    evict_range(&mut j.ahead_guarded, first, last);
    for &p in extra_pages {
        evict_range(&mut j.ahead_guarded, p, p + 1);
    }
}

/// Other views' 64 KiB page numbers aliased to physical pages in `[first, last)`.
fn alias_pages_of_range(first: usize, last: usize) -> Vec<usize> {
    if !view_guard() || first >= last {
        return Vec::new();
    }
    let mut out = Vec::new();
    for (base, bytes) in
        crate::inplace::views_between(first.saturating_mul(PAGE), last.saturating_mul(PAGE))
    {
        let v_first = (base / PAGE).max(first);
        let v_last = base.saturating_add(bytes).div_ceil(PAGE).min(last);
        for page in v_first..v_last {
            let shown = crate::inplace::pages_of(page * PAGE);
            if shown.len() >= 2 && shown.len() <= MANY_VIEWS {
                for alias in shown {
                    let ap = alias / PAGE;
                    if ap < first || ap >= last {
                        out.push(ap);
                    }
                }
            }
        }
    }
    out
}

/// Splits valid `(start, end)` byte ranges into write-watched candidates and
/// already-known physical-page views in a single `GUARDS` lock acquisition.
fn partition_pieces(ranges: &[(usize, usize)]) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
    let mut ww = Vec::new();
    let mut views = Vec::new();
    if force_view_ask() {
        for &(start, end) in ranges {
            if start < end {
                ww.push((start / PAGE, end.div_ceil(PAGE)));
            }
        }
        return (ww, views);
    }
    let g = guards();
    for &(start, end) in ranges {
        if start < end {
            let piece = (start / PAGE, end.div_ceil(PAGE));
            if known_view_in(&g, piece.0 * PAGE) {
                views.push(piece);
            } else {
                ww.push(piece);
            }
        }
    }
    (ww, views)
}

/// Ranges of one submission closer than this are asked about together:
/// walking the pages between them costs less than another kernel call (a
/// call that resets written pages costs 30-70 microseconds whatever the
/// range).
const AHEAD_GAP: usize = (8 << 20) / PAGE;
/// The most one asking covers (its answer needs 8 bytes per 4 KiB page).
const AHEAD_MAX: usize = (1 << 30) / PAGE;

/// Sorts `ww_pieces`, groups nearby ranges within `AHEAD_GAP`/`AHEAD_MAX`,
/// queries them via [`ask_run`], and coalesces the resulting `ahead` ranges.
fn ask_grouped_runs(
    ww_pieces: &mut [(usize, usize)],
    ahead: &mut Vec<(usize, usize)>,
    dirty: &mut Vec<usize>,
) {
    if ww_pieces.len() < 2 {
        return;
    }
    ww_pieces.sort_unstable();
    let mut i = 0;
    while i < ww_pieces.len() {
        let first = ww_pieces[i].0;
        let mut last = ww_pieces[i].1;
        let mut k = i + 1;
        while k < ww_pieces.len()
            && ww_pieces[k].0 <= last + AHEAD_GAP
            && last.max(ww_pieces[k].1) - first <= AHEAD_MAX
        {
            last = last.max(ww_pieces[k].1);
            k += 1;
        }
        ask_run(&ww_pieces[i..k], ahead, dirty);
        i = k;
    }
    coalesce_ranges(ahead);
}

/// Guards coalesced known-view page ranges, appending dirty pages to `dirty`
/// and covered intervals to `ahead_guarded`.
fn guard_view_pieces(
    view_pieces: &mut Vec<(usize, usize)>,
    ahead_guarded: &mut Vec<(usize, usize)>,
    dirty: &mut Vec<usize>,
) -> u64 {
    coalesce_ranges(view_pieces);
    let rev_before = GUARD_REV.load(std::sync::atomic::Ordering::SeqCst);
    for &(first, last) in view_pieces.iter() {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::WatchKnownView, 1);
        #[cfg(feature = "diagnostics")]
        let guarding = std::time::Instant::now();
        let written = guard(first.saturating_mul(PAGE), last.saturating_mul(PAGE));
        #[cfg(feature = "diagnostics")]
        let us = guarding.elapsed().as_micros() as u64;
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::GuardUs, us);
        if let Some(pages) = written {
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                add(C::Watches, 1);
                add(C::WatchUs, us);
                slow(us);
            }
            dirty.extend(pages);
            ahead_guarded.push((first, last));
        }
    }
    rev_before
}

/// Asks the kernel about many ranges `(start, end)` at once, before the
/// [`writes`] of each: ranges close together are one asking, and `writes`
/// then answers from the journal until [`ask_ahead_done`]. The world's
/// submissions refresh thousands of host copies (6,664 askings a frame,
/// 456 ms, in run 14). Known physical-page views are filtered out before
/// grouping so write-watched runs do not bisect across them, and guarded
/// into `ahead_guarded`; lone non-view ranges are left to `writes`.
pub(crate) fn ask_ahead(ranges: &[(usize, usize)]) {
    // Fast path: the submission's one asking stands (`ask_once`).
    if journal_read().kept {
        return;
    }
    if ranges.iter().filter(|r| r.0 < r.1).take(2).count() < 2 {
        return;
    }
    let _ask = watch_ask();
    let need_clear = {
        let j = journal_read();
        if j.kept {
            return;
        }
        !j.ahead.is_empty() || !j.ahead_guarded.is_empty()
    };
    if need_clear {
        let mut j = journal_write();
        if j.kept {
            return;
        }
        j.ahead.clear();
        j.ahead_guarded.clear();
    }
    let (mut ww_pieces, mut view_pieces) = partition_pieces(ranges);
    let mut dirty = Vec::new();
    let mut new_ahead = Vec::new();
    ask_grouped_runs(&mut ww_pieces, &mut new_ahead, &mut dirty);
    let mut new_ahead_guarded = Vec::new();
    let rev_before = guard_view_pieces(&mut view_pieces, &mut new_ahead_guarded, &mut dirty);
    let mut j = journal_write();
    if j.kept {
        if !dirty.is_empty() {
            j.asked += 1;
            let now = j.asked;
            insert_dedup(&mut j.written, &dirty, now);
        }
        return;
    }
    j.asked += 1;
    let now = j.asked;
    j.ahead_asked = now;
    insert_dedup(&mut j.written, &dirty, now);
    j.ahead = new_ahead;
    if GUARD_REV.load(std::sync::atomic::Ordering::SeqCst) == rev_before {
        j.ahead_guarded = new_ahead_guarded;
    } else {
        j.ahead_guarded.clear();
    }
}

/// Since the last frame line: the kernel askings that took a millisecond
/// or more, their microseconds, and the longest asking's. An asking walks
/// page tables, some ten microseconds for most ranges; in run 75 the 2,644
/// askings of a frame took 219 ms in the lines where the pass over the
/// heap's views ran long, 25-50 ms in the others. These numbers say
/// whether a few askings wait for something or all of them are slow.
#[cfg(feature = "diagnostics")]
static SLOW: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

#[cfg(feature = "diagnostics")]
fn slow(us: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    if us >= 1000 {
        SLOW[0].fetch_add(1, Relaxed);
        SLOW[1].fetch_add(us, Relaxed);
    }
    SLOW[2].fetch_max(us, Relaxed);
}

/// [`SLOW`], taken.
#[cfg(feature = "diagnostics")]
pub(crate) fn slow_asks() -> [u64; 3] {
    [0, 1, 2].map(|i| SLOW[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}

/// The submission's refreshes are done: `writes` asks the kernel again.
pub(crate) fn ask_ahead_done() {
    {
        let j = journal_read();
        if j.kept || (j.ahead.is_empty() && j.ahead_guarded.is_empty()) {
            return;
        }
    }
    let _ask = watch_ask();
    let mut j = journal_write();
    if !j.kept {
        j.ahead.clear();
        j.ahead_guarded.clear();
    }
}

/// Whether a submission asks the kernel once, before its first refresh,
/// about every range its lists refresh (`VICTORY_ASK_ONCE=1`; off by
/// default).
///
/// A submission's refreshes are done in groups, between the title's own
/// buffer copies (each copy flushes the refreshes recorded before it), and
/// every group asked the kernel again: 2,644 askings a frame, 70 a
/// submission. With the switch the journal answers all of them from one
/// asking made when the submission starts.
///
/// What changes: memory the title's other threads write while the
/// submission is being prepared is seen by the next submission's asking,
/// not by this one's later groups. What this submission's lists read was
/// written before they were submitted, so before the asking. Our own
/// writes into title memory while preparing are told to the journal
/// ([`wrote`]) or end the asking ([`ask_once_end`]).
pub(crate) fn once() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default since run 81 (with card buffers: 375-450 ms a frame
    // against 573-647, the picture unchanged over four minutes);
    // `VICTORY_ASK_ONCE=0` asks per range, as before.
    *ON.get_or_init(|| std::env::var_os("VICTORY_ASK_ONCE").is_none_or(|v| v != "0"))
}

/// With [`once`]: asks about `ranges` (everything a submission's lists
/// refresh) as [`ask_ahead`] does, and lets that asking stand until
/// [`ask_once_end`]. Known physical-page views skip `GetWriteWatch`, are
/// guarded once for the submission, and are recorded in `ahead_guarded`
/// (invalidated by `writing`, `wrote`, `title_memory_written`,
/// `views_mapped`, and `title_memory_remapped`).
pub(crate) fn ask_once(ranges: &[(usize, usize)]) {
    if !once() {
        return;
    }
    let _ask = watch_ask();
    let need_clear = {
        let j = journal_read();
        j.kept || !j.ahead.is_empty() || !j.ahead_guarded.is_empty()
    };
    if need_clear {
        let mut j = journal_write();
        j.kept = false;
        j.ahead.clear();
        j.ahead_guarded.clear();
    }
    let (mut ww_pieces, mut view_pieces) = partition_pieces(ranges);
    let mut dirty = Vec::new();
    let mut new_ahead = Vec::new();
    ask_grouped_runs(&mut ww_pieces, &mut new_ahead, &mut dirty);

    // `ask_run` leaves a range that lies alone to `writes`, which would ask
    // once per flush (70x/submission). Ask each uncovered write-watched
    // candidate once here; known views are already separated out above.
    ww_pieces.dedup();
    for &(first, last) in &ww_pieces {
        if !covers(&new_ahead, first, last) && ask_kernel(first, last, &mut dirty) {
            insert_coalesced(&mut new_ahead, first, last);
        }
    }

    let mut new_ahead_guarded = Vec::new();
    let rev_before = guard_view_pieces(&mut view_pieces, &mut new_ahead_guarded, &mut dirty);

    if dirty.is_empty() && new_ahead.is_empty() && new_ahead_guarded.is_empty() {
        return;
    }
    let mut j = journal_write();
    j.asked += 1;
    let now = j.asked;
    j.ahead_asked = now;
    insert_dedup(&mut j.written, &dirty, now);
    j.ahead = new_ahead;
    if GUARD_REV.load(std::sync::atomic::Ordering::SeqCst) == rev_before {
        j.ahead_guarded = new_ahead_guarded;
    } else {
        j.ahead_guarded.clear();
    }
    j.kept = !j.ahead.is_empty() || !j.ahead_guarded.is_empty();
}

/// The submission's one asking stands no longer: its refreshes are done,
/// or title memory was written in a way the journal was not told.
pub(crate) fn ask_once_end() {
    if !journal_read().kept {
        return;
    }
    let _ask = watch_ask();
    let mut j = journal_write();
    if j.kept {
        j.kept = false;
        j.ahead.clear();
        j.ahead_guarded.clear();
    }
}

/// We wrote title memory `[at, at + len)` while a submission's one asking
/// stands (the title's buffer copies, done at submission): its pages count
/// as written after that asking, so the refreshes that follow copy them.
pub(crate) fn wrote(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let has_guarded = {
        let j = journal_read();
        if !j.kept {
            return;
        }
        !j.ahead_guarded.is_empty()
    };
    let first = at / PAGE;
    let last = at.saturating_add(len).div_ceil(PAGE);
    let extra_pages = if has_guarded {
        alias_pages_of_range(first, last)
    } else {
        Vec::new()
    };
    let _ask = watch_ask();
    let mut j = journal_write();
    if !j.kept {
        return;
    }
    j.asked += 1;
    let now = j.asked;
    // The asking that stands moves with it: `asked` and `writes` give it as
    // "now", and a page noted written after "now" stays stale for every
    // copy that takes it (copied at `now`, written at `now + 1`): copied
    // again at every refresh, and a reason to settle flights
    // (fxceless/victory-fork#410).
    j.ahead_asked = now;
    for page in first..last {
        j.written.insert(page, now);
    }
    for &page in &extra_pages {
        j.written.insert(page, now);
    }
    if !j.ahead_guarded.is_empty() {
        evict_range(&mut j.ahead_guarded, first, last);
        for &page in &extra_pages {
            evict_range(&mut j.ahead_guarded, page, page + 1);
        }
    }
}

/// One kernel asking for `pieces` (page ranges in order); by halves if it
/// fails, as a run across two reservations or into memory that is not
/// write-watched does. A piece alone is left to [`writes`] (or `ask_once`).
fn ask_run(
    pieces: &[(usize, usize)],
    ahead: &mut Vec<(usize, usize)>,
    dirty: &mut Vec<usize>,
) {
    if pieces.len() < 2 {
        return;
    }
    let first = pieces[0].0;
    let last = pieces.iter().map(|p| p.1).max().unwrap_or(first);
    if ask_kernel(first, last, dirty) {
        ahead.push((first, last));
        return;
    }
    let (a, b) = pieces.split_at(pieces.len() / 2);
    ask_run(a, ahead, dirty);
    ask_run(b, ahead, dirty);
}

/// Calls `GetWriteWatch(WRITE_WATCH_FLAG_RESET)` on pages `[first, last)` into
/// the calling thread's scratch buffer, appending deduplicated 64 KiB page
/// numbers into `dirty`. Returns false if the range is not write-watched
/// memory of one reservation.
fn query_kernel_raw(first: usize, last: usize, dirty: &mut Vec<usize>) -> bool {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    #[cfg(test)]
    KERNEL_ASKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    FOUND.with(|cell| {
        let mut found = cell.borrow_mut();
        let mut count = (last - first) * PAGE / KERNEL_PAGE;
        if found.len() < count {
            found.resize(count, 0);
        }
        let mut granularity = 0;
        // SAFETY: `found` holds at least `count` entries.
        let failed = unsafe {
            GetWriteWatch(
                WRITE_WATCH_FLAG_RESET,
                (first * PAGE) as *const _,
                (last - first) * PAGE,
                found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        } != 0;
        if failed {
            return false;
        }
        dedup_pages(&found[..count], dirty);
        true
    })
}

/// Asks the kernel for the written pages of pages `[first, last)` and appends
/// deduplicated 64 KiB page numbers into `dirty`; false (and nothing changed)
/// if the range is not write-watched memory of one reservation.
fn ask_kernel(first: usize, last: usize, dirty: &mut Vec<usize>) -> bool {
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    if !query_kernel_raw(first, last, dirty) {
        return false;
    }
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        let elapsed = started.elapsed().as_micros() as u64;
        add(C::Watches, 1);
        add(C::WatchUs, elapsed);
        slow(elapsed);
    }
    true
}

/// What [`writes`] would say of `[start, end)` from the journal alone, if
/// the range lies in one asked ahead ([`ask_ahead`], [`ask_once`]): the
/// pages written since tracking began and the asking that last found each
/// written, with the number of the asking that stands (as [`writes`] gives
/// it). `None` if the kernel or guards would have to be asked.
pub(crate) fn asked(start: usize, end: usize) -> Option<(u64, Vec<(usize, u64)>)> {
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let j = journal_read();
    j.covers_ahead(first, last, true).then(|| {
        (
            j.ahead_asked,
            j.written
                .range(first..last)
                .map(|(&p, &w)| (p, w))
                .collect(),
        )
    })
}

/// The pages of `[start, end)` (whole pages) as last written, for pages
/// written since tracking began: `(page number, asking)` in page order, and
/// this asking's number, which is higher than any in the list but those the
/// kernel just reported. `None` if the memory is not write-watched (nor,
/// with `guarded`, view memory, which is then guarded).
pub(crate) fn writes(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    // Fast path: shared read lock only so concurrent refresh workers never
    // serialize against each other or against miss-path syscalls.
    {
        let j = journal_read();
        if j.covers_ahead(first, last, guarded) {
            return Some((
                j.ahead_asked,
                j.written
                    .range(first..last)
                    .map(|(&p, &w)| (p, w))
                    .collect(),
            ));
        }
    }
    // View memory the guards already hold cannot be write-watched: reject
    // `guarded = false` callers immediately without taking `WATCH_ASK` or
    // issuing a doomed `GetWriteWatch` syscall.
    if !guarded && known_view(first * PAGE) {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::WatchKnownView, 1);
        return None;
    }
    // Miss path: hold `WATCH_ASK` across both the syscall and the brief
    // `JOURNAL.write()` commit so overlapping misses cannot race on
    // `WRITE_WATCH_FLAG_RESET` or `guard()`, while `JOURNAL` remains unlocked
    // during the syscall so ahead-hit readers never stall.
    let _ask = watch_ask();
    {
        let j = journal_read();
        if j.covers_ahead(first, last, guarded) {
            return Some((
                j.ahead_asked,
                j.written
                    .range(first..last)
                    .map(|(&p, &w)| (p, w))
                    .collect(),
            ));
        }
    }
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // View memory the guards already hold cannot be write-watched: the
    // kernel asking would only fail. Copies of it are flushed many times a
    // frame, each time paying that failed call before the guards answer.
    let is_known_view = known_view(first * PAGE);
    let mut dirty = Vec::new();
    let failed = is_known_view || !query_kernel_raw(first, last, &mut dirty);
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(
        if is_known_view {
            crate::copystat::C::WatchKnownView
        } else {
            crate::copystat::C::WatchFailed
        },
        failed as u64,
    );
    if failed {
        if !guarded {
            return None;
        }
        #[cfg(feature = "diagnostics")]
        let guarding = std::time::Instant::now();
        dirty = guard(first.saturating_mul(PAGE), last.saturating_mul(PAGE))?;
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::GuardUs, guarding.elapsed().as_micros() as u64);
    }
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        let elapsed = started.elapsed().as_micros() as u64;
        add(C::Watches, 1);
        add(C::WatchUs, elapsed);
        slow(elapsed);
    }
    let mut j = journal_write();
    j.asked += 1;
    let now = j.asked;
    insert_dedup(&mut j.written, &dirty, now);
    Some((
        now,
        j.written
            .range(first..last)
            .map(|(&p, &w)| (p, w))
            .collect(),
    ))
}

impl Tracked {
    /// Takes changed pages of `[from, to)`, cut to the tracked range. A
    /// partial page stays changed so a later take can read its other bytes.
    pub(crate) fn take(&self, from: usize, to: usize) -> Vec<(usize, usize)> {
        let (from, to) = (from.max(self.start), to.min(self.end));
        if from >= to {
            return Vec::new();
        }
        let Some((_, pages)) = writes(from, to, true) else {
            return Vec::new();
        };
        let mut seen = self.seen.lock().unwrap_or_else(|e| e.into_inner());
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for (page, write) in pages {
            if seen.get(&page).is_some_and(|&last| last >= write) {
                continue;
            }
            let (start, end) = ((page * PAGE).max(from), ((page + 1) * PAGE).min(to));
            if start >= end {
                continue;
            }
            // Keep partial pages dirty: the caller has not taken the rest.
            if (page * PAGE).max(self.start) >= from
                && ((page + 1) * PAGE).min(self.end) <= to
            {
                seen.insert(page, write);
            }
            match ranges.last_mut() {
                Some((_, last)) if *last == start => *last = end,
                _ => ranges.push((start, end)),
            }
        }
        ranges
    }
}

/// Guarded blocks of view memory.
const BLOCK: usize = PAGE;

/// Whether env vars force asking the kernel even for blocks already held by
/// the guards (`VICTORY_VIEW_ASK=1` or `VICTORY_GUARD_ASK=1`).
fn force_view_ask() -> bool {
    static ASK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ASK.get_or_init(|| {
        ["VICTORY_VIEW_ASK", "VICTORY_GUARD_ASK"]
            .iter()
            .any(|n| std::env::var_os(n).is_some_and(|v| v == "1"))
    })
}

fn known_view_in(g: &Guards, start: usize) -> bool {
    let first = start / BLOCK;
    g.protected.contains(&first) || g.absent.contains(&first)
}

/// Whether the block holding `start` is view memory the guards already
/// hold (read-only by our hand, or noted absent): asking the kernel's write
/// watch about it can only fail. `VICTORY_VIEW_ASK=1` (or
/// `VICTORY_GUARD_ASK=1`) asks the kernel anyway, as before.
fn known_view(start: usize) -> bool {
    if force_view_ask() {
        return false;
    }
    known_view_in(&guards(), start)
}

struct Guards {
    /// Blocks we made read-only.
    protected: BTreeSet<usize>,
    /// Of those, the ones not written since a copy took them.
    clean: BTreeSet<usize>,
    /// Blocks already copied as zeros. While they remain wholly uncommitted,
    /// the copy can skip hundreds of MB of failing kernel reads per world frame.
    absent: BTreeSet<usize>,
    /// Blocks whose physical page was guarded together with its other views
    /// (those views' blocks are in it too): a write to one opens them all.
    /// A write range can be huge (`fread`), so only these are looked up.
    aliased: BTreeSet<usize>,
}

static GUARDS: Mutex<Guards> = Mutex::new(Guards {
    protected: BTreeSet::new(),
    clean: BTreeSet::new(),
    absent: BTreeSet::new(),
    aliased: BTreeSet::new(),
});

fn guards() -> std::sync::MutexGuard<'static, Guards> {
    GUARDS.lock().unwrap_or_else(|e| e.into_inner())
}

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded (all of a block's pages), guarding them again
/// before the caller copies them. `None` if the memory is not a view.
fn guard(start: usize, end: usize) -> Option<Vec<usize>> {
    // A range whose first block the guards hold already (read-only by our
    // hand, or noted absent) is a view: no question to Windows. The
    // question walks the page tables to the end of the pages that share the
    // first one's state, and a guarded view is one long run of read-only
    // pages: 0.6 ms for the title's 174 MB heap, 16 times a frame, and most
    // of the 68 ms a frame of "guards" (run 58).
    // `VICTORY_GUARD_ASK=1` asks every time, as before.
    static ASK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let known = !*ASK.get_or_init(|| std::env::var_os("VICTORY_GUARD_ASK").is_some_and(|v| v == "1")) && {
        let g = guards();
        let first = start / BLOCK;
        g.protected.contains(&first) || g.absent.contains(&first)
    };
    if !known && !is_view(start, end) {
        return None;
    }
    install();
    let mut g = guards();
    let mut written = Vec::new();
    let mut run: Option<usize> = None;
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    // The range's clean blocks, in order: read off the set in one pass,
    // not looked up one block at a time. A frame asks about some 10 GB of
    // view memory, 150,000 blocks, nearly all clean: the lookups were most
    // of the 60-90 ms a frame of "guards" (run 54). What the loop adds to
    // the set meanwhile is the block it stands at or ones before it.
    let clean: Vec<usize> = g.clean.range(first..last).copied().collect();
    if clean.len() == last - first {
        // Nothing written, mapped or new since the range was last taken.
        return Some(written);
    }
    let mut next_clean = 0;
    let mut uncommitted_until = first;
    for block in first..=last {
        while next_clean < clean.len() && clean[next_clean] < block {
            next_clean += 1;
        }
        let is_clean = next_clean < clean.len() && clean[next_clean] == block;
        let mut open = block < last && !is_clean;
        if open && g.absent.contains(&block) {
            // Ordinary commits need no runtime mapping notification. Reuse
            // this asking's query for the following blocks in the same region.
            if block >= uncommitted_until {
                uncommitted_until = uncommitted_end(block);
            }
            open = block >= uncommitted_until;
            if open {
                g.absent.remove(&block);
            }
        }
        if open {
            written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
            if g.protected.contains(&block) {
                // Written through the runtime, still read-only.
                g.clean.insert(block);
            } else {
                run.get_or_insert(block);
                continue;
            }
        }
        if let Some(from) = run.take() {
            protect(&mut g, from, block);
        }
    }
    Some(written)
}

/// Whether `[start, end)` is view memory. The title's views lie in
/// placeholders, mapped and unmapped block by block as it streams, so a
/// range can begin in a block that is not mapped right now: the first
/// pages of the range that are there decide. A range with nothing there
/// counts as a view (its blocks are noted absent, and looked at again when
/// the runtime maps them).
///
/// Looking at the first address only took such a range for "not a view":
/// its copy was marked not watched for good and copied whole at every
/// refresh, around its unmapped pages (run 54: 300 such ranges, 570 MB
/// copied a frame for them, 255 copies a frame reading around holes).
/// `VICTORY_GUARD_HOLES=0` looks at the first address only, as before.
fn is_view(start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    static HOLES: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let holes = *HOLES.get_or_init(|| std::env::var_os("VICTORY_GUARD_HOLES").is_none_or(|v| v != "0"));
    let mut at = start;
    loop {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        if n == 0 {
            return false;
        }
        if info.Type == MEM_MAPPED {
            return true;
        }
        // Committed memory that is no view (the title's heaps, ours), or
        // the answer as it was before.
        if info.State == MEM_COMMIT || !holes {
            return false;
        }
        // A hole (a placeholder, or free): on to what follows it.
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if next <= at || next >= end {
            return true;
        }
        at = next;
    }
}

/// Makes blocks `[from, to)` read-only and notes them guarded and clean;
/// block by block if they span views, leaving out any that fail, and
/// noting those not committed (unmapped) absent. The same physical pages
/// at every other view of them are made read-only too: a write through
/// one of those is this memory written, and only a fault can say so.
/// `VICTORY_VIEW_GUARD=0` leaves the other views as they are, as before.
fn protect(g: &mut Guards, from: usize, to: usize) {
    protect_here(g, from, to);
    if !view_guard() {
        return;
    }
    for (base, bytes) in crate::inplace::views_between(from.saturating_mul(BLOCK), to.saturating_mul(BLOCK)) {
        let first = (base / BLOCK).max(from);
        let last = (base.saturating_add(bytes).div_ceil(BLOCK)).min(to);
        for block in first..last {
            let shown = crate::inplace::pages_of(block * BLOCK);
            // One view (nothing else to guard), or the page the title shows
            // wherever it has nothing (thousands of views): each view of
            // that one is guarded by itself.
            if shown.len() < 2 || shown.len() > MANY_VIEWS {
                continue;
            }
            #[cfg(feature = "diagnostics")]
            ALIASES[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            for alias in shown {
                let alias = alias / BLOCK;
                g.aliased.insert(alias);
                if alias != block {
                    protect_alias(g, alias);
                }
            }
        }
    }
}

/// Views of one page past which its views are not guarded together.
pub(crate) const MANY_VIEWS: usize = 64;

/// Whether all the views of a guarded page are guarded together
/// (`VICTORY_VIEW_GUARD`).
fn view_guard() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_GUARD").is_none_or(|v| v != "0"))
}

/// Diagnostics: blocks guarded together with their other views, blocks
/// opened because another view of theirs was written, and blocks opened
/// because a view of theirs was mapped; since the start.
#[cfg(feature = "diagnostics")]
pub(crate) static ALIASES: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Makes `block` read-only: another view of pages a copy has just taken
/// through its own view, so that a write through this one faults too. It is
/// not noted clean. The copies of title memory at this view have not taken
/// the pages: noted clean, the block would not be reported written at their
/// next check and they would keep the bytes from before the write (`guard`
/// reports a block that is read-only and not clean as written, then notes it
/// clean). A block that is clean stays so: had the pages been written since
/// its copies took them, `open` would have opened it.
fn protect_alias(g: &mut Guards, block: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    if g.protected.contains(&block) {
        return;
    }
    let mut old = 0;
    // SAFETY: title view memory; only its protection changes.
    if unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READONLY, &mut old) } != 0 {
        g.protected.insert(block);
    }
}

/// [`protect`] for one range, without the other views of its pages.
fn protect_here(g: &mut Guards, from: usize, to: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    let set = |from: usize, to: usize| {
        let mut old = 0;
        // SAFETY: title view memory; only its protection changes.
        unsafe {
            VirtualProtect(
                (from * BLOCK) as *const _,
                (to - from) * BLOCK,
                PAGE_READONLY,
                &mut old,
            ) != 0
        }
    };
    let note = |g: &mut Guards, from: usize, to: usize| {
        g.protected.extend(from..to);
        g.clean.extend(from..to);
    };
    if set(from, to) {
        note(g, from, to);
        return;
    }
    for block in from..to {
        if set(block, block + 1) {
            note(g, block, block + 1);
        } else if block < uncommitted_end(block) {
            g.absent.insert(block);
        }
    }
}

/// The end (exclusive) of the whole uncommitted blocks starting at `block`.
fn uncommitted_end(block: usize) -> usize {
    use windows_sys::Win32::System::Memory::{
        MEM_FREE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery((block * BLOCK) as *const _, &mut info, size_of_val(&info)) };
    if n != 0 && matches!(info.State, MEM_RESERVE | MEM_FREE) {
        (info.BaseAddress as usize).saturating_add(info.RegionSize) / BLOCK
    } else {
        block
    }
}

/// Gives write access back to the guarded blocks of `[start, end)` and
/// notes them written; true if any block in `[start, end)` was guarded.
/// The same physical pages at every other view of them are opened too: the
/// write that faulted (or is ours, `crate::host::copy_mapped`) goes through
/// one view, and every guard over the pages must copy them again.
fn open(g: &mut Guards, start: usize, end: usize) -> bool {
    open_collecting(g, start, end).0
}

/// [`open`], also returning any other views' block numbers opened via aliases
/// so callers outside the VEH can evict them from `Journal::ahead_guarded`.
fn open_collecting(g: &mut Guards, start: usize, end: usize) -> (bool, Vec<usize>) {
    let any = open_here(g, start, end);
    if start >= end || g.aliased.is_empty() {
        return (any, Vec::new());
    }
    let mut opened_aliases = Vec::new();
    let aliased: Vec<usize> = g.aliased.range(start / BLOCK..end.div_ceil(BLOCK)).copied().collect();
    for block in aliased {
        g.aliased.remove(&block);
        for alias in crate::inplace::pages_of(block * BLOCK) {
            let alias_block = alias / BLOCK;
            if alias_block != block && open_here(g, alias, alias.saturating_add(BLOCK)) {
                #[cfg(feature = "diagnostics")]
                ALIASES[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                opened_aliases.push(alias_block);
            }
            g.aliased.remove(&alias_block);
        }
    }
    (any, opened_aliases)
}

/// [`open`] for one range, without the other views of its pages.
fn open_here(g: &mut Guards, start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    let blocks: Vec<usize> = g
        .protected
        .range(start / BLOCK..end.div_ceil(BLOCK))
        .copied()
        .collect();
    for &block in &blocks {
        g.protected.remove(&block);
        g.clean.remove(&block);
        let mut old = 0;
        // SAFETY: a block we made read-only.
        unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old) };
    }
    !blocks.is_empty()
}

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler): opens its guarded blocks.
/// The runtime calls it (as `kernel_will_write`) before the title's kernel
/// reads into its own memory.
pub fn writing(at: usize, len: usize) {
    // The length is the title's (`fread`'s size times count, saturated).
    let end = at.saturating_add(len);
    let (any, mut aliases) = {
        let mut g = guards();
        let res = open_collecting(&mut g, at, end);
        if res.0 || !res.1.is_empty() {
            GUARD_REV.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
        res
    };
    if len > 0 {
        let (first, last) = (at / PAGE, end.div_ceil(PAGE));
        let opened = any || !aliases.is_empty();
        if opened || overlaps_ahead_guarded(first, last) {
            if !opened {
                // Blocks may already have been opened by `on_write`; bump
                // `GUARD_REV` and gather view aliases before evicting.
                GUARD_REV.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                aliases = alias_pages_of_range(first, last);
            }
            evict_ahead_guarded(first, last, &aliases);
        }
    }
    // A placed texture there changes with no descriptor written.
    crate::descriptor::host::remapped(at, len);
}

/// What `VICTORY_PAGE_WRITES` asks to be done with a write the guards do
/// not see once its bytes are there ([`title_memory_written`]), as bits:
/// 1, the guarded blocks under it are opened and count as written (host
/// copies of buffers and textures read through tables take them again);
/// 2, the placed textures picked by index that lie under it are due again;
/// 4, both also for reads to plain addresses when they have finished.
/// Unset or `guards`: bit 1 alone. `0`: nothing, counted only (the default
/// until run 199). `1`: bits 1 and 2. `slots`: bit 2 alone. `all`:
/// everything.
/// (Bit 1 by default: in every run 10,000 to 50,000 guarded blocks of host
/// copies lay under such writes and kept the bytes from before; runs 197
/// and 198 opened them, 9,766 and 10,796 blocks, with nothing worse for
/// it. Bit 2 makes texture slots due and is the one that flooded the
/// queue in an earlier form: still asked for by name.)
pub(crate) fn page_writes() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_PAGE_WRITES").as_deref().map(str::trim) {
        Ok("0") => 0,
        Ok("1") => 3,
        Ok("slots") => 2,
        Ok("all") => 7,
        _ => 1,
    })
}

/// Since the start: writes into physical pages through the runtime's own
/// views that landed at addresses the title has mapped (one for each run
/// of addresses); their bytes; guarded blocks that lay under them (host
/// copies that would otherwise keep the bytes from before); reads to
/// plain addresses that finished.
pub(crate) static WRITTEN: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// The bytes of a write the guards do not see are there: title memory
/// `[at, at + len)`. `pages`: DirectStorage wrote them into physical pages
/// through a view of the runtime's own, and this is an address where the
/// title has those pages mapped (no title address faulted, nothing was
/// mapped or unmapped; the shortcut this file's header names). Otherwise a
/// read to a plain address has finished, of which [`writing`] was told
/// before it began.
///
/// Upstream's c75fdba does the same for its trackers (its `Writing` marks
/// the pages once more when dropped; page destinations are reported at
/// each address showing them). Here it is counted always and acted on as
/// [`page_writes`] says. Called without the runtime's page-table lock;
/// takes the guards' lock, then, alone, the lock of the ranges kept for
/// the texture pass.
pub fn title_memory_written(at: usize, len: usize, pages: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    if len == 0 {
        return;
    }
    let mode = page_writes();
    let end = at.saturating_add(len);
    if pages {
        WRITTEN[0].fetch_add(1, Relaxed);
        WRITTEN[1].fetch_add(len as u64, Relaxed);
    } else {
        WRITTEN[3].fetch_add(1, Relaxed);
        if mode & 4 == 0 {
            return;
        }
    }
    let opened = {
        let mut g = guards();
        let guarded = g.protected.range(at / BLOCK..end.div_ceil(BLOCK)).count() as u64;
        if pages {
            WRITTEN[2].fetch_add(guarded, Relaxed);
        }
        if mode & 1 != 0 && guarded != 0 {
            let res = open_collecting(&mut g, at, end);
            if res.0 || !res.1.is_empty() {
                GUARD_REV.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
            Some(res.1)
        } else {
            None
        }
    };
    if mode & 1 != 0 {
        let (first, last) = (at / PAGE, end.div_ceil(PAGE));
        let aliases = if let Some(aliases) = opened {
            aliases
        } else if overlaps_ahead_guarded(first, last) {
            GUARD_REV.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            alias_pages_of_range(first, last)
        } else {
            Vec::new()
        };
        // The runtime writes through a view of its own, which neither the
        // kernel's write watch nor our guards may see (for example, pages in
        // write-watched memory or views not yet guarded): note the pages
        // written in the journal here, or host copies made before the file
        // arrived keep stale bytes for good. Acquires `JOURNAL.write()` only
        // after `GUARDS` is released (maintaining lock order).
        let mut j = journal_write();
        j.asked += 1;
        let now = j.asked;
        if !j.ahead.is_empty() || !j.ahead_guarded.is_empty() {
            j.ahead_asked = now;
        }
        for page in first..last {
            j.written.insert(page, now);
        }
        for &page in &aliases {
            j.written.insert(page, now);
        }
        if !j.ahead_guarded.is_empty() {
            evict_range(&mut j.ahead_guarded, first, last);
            for &page in &aliases {
                evict_range(&mut j.ahead_guarded, page, page + 1);
            }
        }
    }
    // A placed texture there changed with no descriptor written.
    crate::descriptor::host::page_written(at, len);
}

/// The runtime mapped a view of the physical bytes `[offset, offset + len)`
/// of `section` (`crate::inplace::title_memory_view`): the view is
/// writable, so the same bytes guarded at another view must be checked
/// again — and are, the pages noted written, which also has the next check
/// guard the new view (`protect`).
pub(crate) fn views_mapped(section: usize, offset: u64, len: usize) {
    if !view_guard() {
        return;
    }
    let ranges = crate::inplace::views_of_section(section, offset, len);
    if ranges.is_empty() {
        return;
    }
    let mut opened = Vec::new();
    {
        let mut g = guards();
        for &(at, n) in &ranges {
            if open_here(&mut g, at, at.saturating_add(n)) {
                #[cfg(feature = "diagnostics")]
                ALIASES[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                opened.push((at / PAGE, at.saturating_add(n).div_ceil(PAGE)));
            }
        }
        if !opened.is_empty() {
            GUARD_REV.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }
    for (first, last) in opened {
        evict_ahead_guarded(first, last, &[]);
    }
}

/// Held shared by a read of title memory in place, and alone by the runtime
/// while it changes what title memory shows.
///
/// The runtime does not change a mapping in one step. To unmap one page of
/// a view it unmaps the whole view and maps what stays again
/// (`xmem.rs` `unmap`), and a view it maps is a reservation until the next
/// call commits it (`map`, then `commit`). For that moment pages the title
/// never gave up are "reserved, never committed", and a thread reading
/// them in place takes an access violation: run 71 in `Layout::detile`,
/// run 74 in the hash before it, both under `Staging::refresh_now`, which
/// had asked whether the texture was all mapped and been told yes. Asking
/// cannot help; not changing the mapping while such a read is under way
/// does.
static MAPPING: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// What the runtime holds while it changes what title memory shows
/// ([`title_memory_remapping`]).
pub struct Remapping(#[allow(dead_code)] std::sync::RwLockWriteGuard<'static, ()>);

/// The runtime is about to map, unmap or move physical pages of title
/// memory: until what it returns is dropped, no read of title memory in
/// place is under way and none begins ([`title_memory_reading`]). Taken
/// around the whole change, the unmapping and the mapping again.
pub fn title_memory_remapping() -> Remapping {
    Remapping(MAPPING.write().unwrap_or_else(|e| e.into_inner()))
}

thread_local! {
    /// Active [`Reading`] guards held by the current thread, so fault-safe
    /// fast reads nested inside an outer [`title_memory_reading`] (e.g.
    /// texture refreshes) skip a recursive `SRWLOCK` acquisition.
    static READING_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// What a read of title memory in place holds ([`title_memory_reading`]).
pub struct Reading(#[allow(dead_code)] std::sync::RwLockReadGuard<'static, ()>);

impl Drop for Reading {
    fn drop(&mut self) {
        READING_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    }
}

/// For reading title memory in place: until what it returns is dropped,
/// the runtime changes no mapping, so memory found mapped stays mapped.
/// Not to be taken twice by one thread, nor held across anything that
/// maps title memory.
pub fn title_memory_reading() -> Reading {
    let guard = MAPPING.read().unwrap_or_else(|e| e.into_inner());
    READING_DEPTH.with(|d| d.set(d.get() + 1));
    Reading(guard)
}

/// [`title_memory_reading`] if it can be had at once: `None` while the
/// runtime changes a mapping (or waits to), for a caller that has another
/// way to read and would rather not wait.
pub fn title_memory_reading_now() -> Option<Reading> {
    let guard = match MAPPING.try_read() {
        Ok(guard) => guard,
        Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
        Err(std::sync::TryLockError::WouldBlock) => return None,
    };
    READING_DEPTH.with(|d| d.set(d.get() + 1));
    Some(Reading(guard))
}

/// Non-blocking read guard against concurrent title-memory remapping, safe
/// even if this thread already holds an outer [`Reading`].
#[cfg(target_arch = "x86_64")]
fn acquire_reading_now() -> Option<Option<Reading>> {
    if READING_DEPTH.with(std::cell::Cell::get) > 0 {
        Some(None)
    } else {
        title_memory_reading_now().map(Some)
    }
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new and read-write, so it is no longer guarded (nor known
/// unmapped).
pub fn title_memory_remapped(at: usize, len: usize) {
    // A placed texture there arrives, goes or changes with no descriptor
    // written.
    crate::descriptor::host::remapped(at, len);
    // The views the runtime told us of there are no longer what it shows.
    crate::inplace::views_gone(at, len);
    // And nothing stays in place over it.
    crate::inplace::title_memory_freeing(at, len);
    let end = at.saturating_add(len);
    {
        let mut g = guards();
        let range = at / BLOCK..end.div_ceil(BLOCK);
        let absent: Vec<usize> = g.absent.range(range.clone()).copied().collect();
        for block in absent {
            g.absent.remove(&block);
        }
        let blocks: Vec<usize> = g.protected.range(range.clone()).copied().collect();
        for block in blocks {
            g.protected.remove(&block);
            g.clean.remove(&block);
        }
        let aliased: Vec<usize> = g.aliased.range(range).copied().collect();
        for block in aliased {
            g.aliased.remove(&block);
        }
        if len > 0 {
            GUARD_REV.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }
    if len > 0 {
        evict_ahead_guarded(at / PAGE, end.div_ceil(PAGE), &[]);
    }
}

#[cfg(target_arch = "x86_64")]
thread_local! {
    /// Recovery instruction pointer for fault-safe reads of title memory
    /// ([`try_copy_fast`], [`try_diff_bounds_fast`]). Non-zero only across the
    /// inline-assembly loop that touches title memory; cleared on normal exit
    /// or by [`install`]'s vectored handler on a read access violation.
    static READ_RECOVER_RIP: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Whether title-memory copies and echo-diffs read title memory directly under
/// [`install`]'s vectored read-fault handler instead of `ReadProcessMemory`
/// (`VICTORY_FAST_COPY=0` disables it).
#[cfg(target_arch = "x86_64")]
fn fast_copy() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FAST_COPY").is_none_or(|v| v != "0"))
}

/// Registers the handler that takes writes to guarded blocks (and read faults
/// inside [`try_copy_fast`] / [`try_diff_bounds_fast`]), once, ahead of the
/// crash reporter's.
fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        #[cfg(target_arch = "x86_64")]
        {
            const READ: usize = 0;
            if r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION as NTSTATUS
                && r.NumberParameters >= 2
                && r.ExceptionInformation[0] == READ
            {
                let recover = READ_RECOVER_RIP.with(|slot| slot.replace(0));
                if recover != 0 {
                    // SAFETY: `ContextRecord` belongs to the faulting thread;
                    // `recover` is the in-block label of the active `asm!` read
                    // loop on this thread (`rsp`, callee-saved registers, and
                    // `DF=0` remain intact).
                    unsafe { (*(*info).ContextRecord).Rip = recover as u64 };
                    return EXCEPTION_CONTINUE_EXECUTION;
                }
            }
        }
        const WRITE: usize = 1;
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let at = r.ExceptionInformation[1];
        // Held across both checks: another thread may have opened the block
        // since this one faulted, and a copy may have guarded it again since
        // (then it is ours again, not read-only for some other reason).
        let mut g = guards();
        if open(&mut g, at, at + 1) {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::GuardFaults, 1);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if writable(at) {
            EXCEPTION_CONTINUE_EXECUTION
        } else {
            EXCEPTION_CONTINUE_SEARCH
        }
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our guard state, TLS recovery slot, and
    // page protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

/// Copies `len` bytes from title memory `src` into writable host memory `dst`
/// without a kernel syscall, using [`install`]'s vectored handler to recover
/// cleanly if `src` touches an uncommitted or inaccessible page.
///
/// Returns `false` if fast copy is disabled (`VICTORY_FAST_COPY=0`), if title
/// memory is currently being remapped (non-blocking), or if a read fault occurs
/// (in which case `dst` may have been partially written and the caller must
/// fall back).
#[cfg(target_arch = "x86_64")]
pub(crate) fn try_copy_fast(dst: *mut u8, src: *const u8, len: usize) -> bool {
    if len == 0 {
        return true;
    }
    if !fast_copy() || (src as usize).checked_add(len).is_none() {
        return false;
    }
    install();
    let Some(_reading) = acquire_reading_now() else {
        return false;
    };
    let mut remaining = len;
    let s = src;
    let d = dst;
    READ_RECOVER_RIP.with(|slot| {
        let slot_ptr = slot.as_ptr();
        // SAFETY: `dst` is writable host memory of `len` bytes, non-overlapping
        // with `src`; `_reading` prevents concurrent title-memory unmapping;
        // `slot_ptr` points to this thread's `READ_RECOVER_RIP` slot; on a read
        // fault during `rep movsb`, `on_write` clears the slot and redirects
        // `Rip` to `2:` with `rcx > 0`, `rsp`/callee-saved registers intact,
        // and `DF = 0`.
        unsafe {
            core::arch::asm!(
                "lea {tmp}, [rip + 2f]",
                "mov qword ptr [{slot}], {tmp}",
                "cld",
                "rep movsb",
                "2:",
                "mov qword ptr [{slot}], 0",
                slot = in(reg) slot_ptr,
                tmp = out(reg) _,
                inout("rdi") d => _,
                inout("rsi") s => _,
                inout("rcx") remaining,
                options(nostack),
            );
        }
    });
    remaining == 0
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn try_copy_fast(_dst: *mut u8, _src: *const u8, len: usize) -> bool {
    len == 0
}

/// Fault-safe, zero-copy comparison of title memory `[src, src + cur.len())`
/// against host buffer `cur`, returning the tightest inclusive `(first, last)`
/// byte offsets where they differ.
///
/// Returns:
/// - `Some(None)` if every byte matches,
/// - `Some(Some((first, last)))` if differing bytes were found (`first <= last < cur.len()`),
/// - `None` if fast copy is disabled (`VICTORY_FAST_COPY=0`), if title memory
///   is currently being remapped, or if a read fault occurred while scanning `src`.
#[cfg(target_arch = "x86_64")]
pub(crate) fn try_diff_bounds_fast(src: *const u8, cur: &[u8]) -> Option<Option<(usize, usize)>> {
    let len = cur.len();
    if len == 0 {
        return Some(None);
    }
    if !fast_copy() || (src as usize).checked_add(len).is_none() {
        return None;
    }
    install();
    let _reading = acquire_reading_now()?;
    let cur_ptr = cur.as_ptr();
    let end16 = len & !15;
    let (first, faulted) = READ_RECOVER_RIP.with(|slot| {
        let slot_ptr = slot.as_ptr();
        let idx: usize;
        let faulted: usize;
        // SAFETY: `cur_ptr` is valid for `len` bytes; `_reading` prevents
        // concurrent title-memory remapping; all loads stay within `[0, len)`;
        // `DF` is never set; on a read fault from `src`, `on_write` clears
        // `READ_RECOVER_RIP` and redirects `Rip` to `8:` with `faulted == 1`.
        unsafe {
            core::arch::asm!(
                "xor {idx}, {idx}",
                "mov {faulted}, 1",
                "lea {tmp}, [rip + 8f]",
                "mov qword ptr [{slot}], {tmp}",
                "3:",
                "cmp {idx}, {end16}",
                "jae 4f",
                "movdqu {x0}, xmmword ptr [{src} + {idx}]",
                "movdqu {x1}, xmmword ptr [{cur} + {idx}]",
                "pcmpeqb {x0}, {x1}",
                "pmovmskb {tmp:e}, {x0}",
                "xor {tmp:e}, 0xffff",
                "jnz 5f",
                "add {idx}, 16",
                "jmp 3b",
                "5:",
                "bsf {tmp:e}, {tmp:e}",
                "add {idx}, {tmp}",
                "jmp 7f",
                "4:",
                "cmp {idx}, {len}",
                "jae 7f",
                "movzx {tmp:e}, byte ptr [{src} + {idx}]",
                "cmp {tmp:l}, byte ptr [{cur} + {idx}]",
                "jne 7f",
                "inc {idx}",
                "jmp 4b",
                "7:",
                "xor {faulted}, {faulted}",
                "8:",
                "mov qword ptr [{slot}], 0",
                src = in(reg) src,
                cur = in(reg) cur_ptr,
                len = in(reg) len,
                end16 = in(reg) end16,
                slot = in(reg) slot_ptr,
                idx = out(reg) idx,
                faulted = out(reg) faulted,
                tmp = out(reg) _,
                x0 = out(xmm_reg) _,
                x1 = out(xmm_reg) _,
                options(nostack),
            );
        }
        (idx, faulted)
    });
    if faulted != 0 {
        return None;
    }
    if first >= len {
        return Some(None);
    }
    if first == len - 1 {
        return Some(Some((first, first)));
    }
    // Scan backward over `[first, len)` with DF=0 at all times (never `std`),
    // rebasing pointers to `first` so relative offsets stay in `0..len - first`.
    let src_tail = src.wrapping_add(first);
    let cur_tail = cur_ptr.wrapping_add(first);
    let (rel_last, faulted) = READ_RECOVER_RIP.with(|slot| {
        let slot_ptr = slot.as_ptr();
        let mut rem = len - first;
        let faulted: usize;
        // SAFETY: `cur_tail` is valid for `len - first` bytes; all loads stay
        // strictly inside `[0, len - first)`; `DF` stays 0 at all times; on a
        // read fault from `src_tail`, `on_write` clears `READ_RECOVER_RIP` and
        // redirects `Rip` to `8:` with `faulted == 1`.
        unsafe {
            core::arch::asm!(
                "mov {faulted}, 1",
                "lea {tmp}, [rip + 8f]",
                "mov qword ptr [{slot}], {tmp}",
                "3:",
                "cmp {rem}, 16",
                "jb 4f",
                "sub {rem}, 16",
                "movdqu {x0}, xmmword ptr [{src} + {rem}]",
                "movdqu {x1}, xmmword ptr [{cur} + {rem}]",
                "pcmpeqb {x0}, {x1}",
                "pmovmskb {tmp:e}, {x0}",
                "xor {tmp:e}, 0xffff",
                "jnz 5f",
                "jmp 3b",
                "5:",
                "bsr {tmp:e}, {tmp:e}",
                "add {rem}, {tmp}",
                "jmp 7f",
                "4:",
                "test {rem}, {rem}",
                "jz 7f",
                "dec {rem}",
                "movzx {tmp:e}, byte ptr [{src} + {rem}]",
                "cmp {tmp:l}, byte ptr [{cur} + {rem}]",
                "jne 7f",
                "jmp 4b",
                "7:",
                "xor {faulted}, {faulted}",
                "8:",
                "mov qword ptr [{slot}], 0",
                src = in(reg) src_tail,
                cur = in(reg) cur_tail,
                slot = in(reg) slot_ptr,
                rem = inout(reg) rem,
                faulted = out(reg) faulted,
                tmp = out(reg) _,
                x0 = out(xmm_reg) _,
                x1 = out(xmm_reg) _,
                options(nostack),
            );
        }
        (rem, faulted)
    });
    if faulted != 0 {
        return None;
    }
    Some(Some((first, first + rel_last)))
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn try_diff_bounds_fast(_src: *const u8, cur: &[u8]) -> Option<Option<(usize, usize)>> {
    cur.is_empty().then_some(None)
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    #[test]
    fn written_pages_are_reported_to_every_asker() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 4 * PAGE);
        let (_, before) = writes(start, end, false).expect("watched");
        assert!(before.is_empty());
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write(1) };
        let (now, after) = writes(start, end, false).expect("watched");
        assert_eq!(after, [(p / PAGE + 1, now)]);
        // Another asker after the kernel's bit was cleared still sees it.
        let (later, again) = writes(p + PAGE, p + 2 * PAGE, false).expect("watched");
        assert!(later > now);
        assert_eq!(again, [(p / PAGE + 1, now)]);
        let mut plain = vec![0u8; PAGE];
        assert!(
            writes(
                plain.as_mut_ptr() as usize,
                plain.as_ptr() as usize + PAGE,
                true
            )
            .is_none()
        );
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as written.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + 2 * BLOCK).is_some()); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(
            !read(&mut file),
            "the kernel does not fault into our handler"
        );
        writing(view + 100, 13);
        assert!(read(&mut file));
        // SAFETY: the bytes just read.
        let landed = unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) };
        assert_eq!(landed, b"from the disk");
        assert_eq!(
            guard(view, view + 2 * BLOCK),
            Some((view / PAGE..view / PAGE + 1).collect())
        );
        // SAFETY: our view.
        unsafe {
            UnmapViewOfFile(
                windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                },
            )
        };
        title_memory_remapped(view, 2 * BLOCK);
    }

    #[test]
    fn writes_in_partly_committed_view_blocks_are_reported() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            SEC_RESERVE, UnmapViewOfFile,
        };

        const SIZE: usize = 3 * BLOCK;
        struct View(MEMORY_MAPPED_VIEW_ADDRESS);
        impl Drop for View {
            fn drop(&mut self) {
                title_memory_remapped(self.0.Value as usize, SIZE);
                // SAFETY: our view is still mapped.
                unsafe { UnmapViewOfFile(self.0) };
            }
        }
        // SAFETY: a new reserved pagefile section. The view holds its own
        // reference after we close the handle, including on assertion failure.
        let view = unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_RESERVE,
                0,
                SIZE as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE);
            CloseHandle(section);
            assert!(!view.Value.is_null());
            View(view)
        };
        let start = view.0.Value as usize;
        let commit = |at: usize, len| {
            // SAFETY: these pages belong to our reserved view.
            let committed = unsafe { VirtualAlloc(at as _, len, MEM_COMMIT, PAGE_READWRITE) };
            assert_eq!(committed as usize, at);
        };
        let early = start + BLOCK + 3 * KERNEL_PAGE;
        let late = start + 2 * BLOCK + 3 * KERNEL_PAGE;
        commit(start, BLOCK);
        commit(early, KERNEL_PAGE);
        let refresh = || guard(start, start + SIZE).unwrap();
        assert_eq!(refresh(), [start / PAGE, early / PAGE, late / PAGE]);
        // The last block is still entirely reserved.
        assert!(!refresh().contains(&(late / PAGE)));

        commit(late, KERNEL_PAGE); // Ordinary commit, without a runtime mapping notification.
        for at in [early, late] {
            assert!(writable(at));
            // SAFETY: each address is inside its committed read-write page.
            unsafe { (at as *mut u8).write_volatile(1) };
        }
        assert_eq!(refresh(), [early / PAGE, late / PAGE]);
    }

    /// A range of view memory whose first block is not mapped is a view
    /// all the same: its blocks are reported once, then the ones written
    /// or mapped since.
    #[test]
    fn a_range_that_begins_in_an_unmapped_block_is_guarded() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of three; the middle block is mapped first.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (3 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        // Nothing mapped at all: a view with every block absent.
        assert!(is_view(p, p + 3 * BLOCK));
        let middle = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        // The range begins in the placeholder, its middle block is a view.
        assert!(is_view(p, p + 3 * BLOCK));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()), "nothing written since");
        // SAFETY: inside the mapped middle block.
        unsafe { ((p + BLOCK + 8) as *mut u8).write_volatile(1) };
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        let first = map(0);
        title_memory_remapped(p, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(0).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        // Memory of ours that is no view stays none.
        let mut local = [0u8; 16];
        let at = local.as_mut_ptr() as usize;
        assert!(!is_view(at, at + local.len()));
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), middle, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(p, 3 * BLOCK);
    }

    /// Blocks of a placeholder between views are reported once, then not
    /// again (they cannot be guarded) until the runtime maps them.
    #[test]
    fn unmapped_blocks_are_reported_once_until_mapped() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of two; block 0 shows the section's first block, as the
        // runtime maps physical pages.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let first = map(0);
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(
            guard(p, p + 3 * BLOCK),
            Some(Vec::new()),
            "nothing written since"
        );
        let second = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), second, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(p, 3 * BLOCK);
    }

    /// A pagefile section mapped twice: two views of the same physical
    /// pages, the second left for the test; the section handle, the two
    /// addresses.
    fn two_views(bytes: usize) -> (usize, usize, usize) {
        use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile,
        };
        // SAFETY: a fresh pagefile section, mapped twice; the views keep
        // the section alive and the handle is closed by the caller.
        unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                bytes as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let one = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes).Value as usize;
            let two = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes).Value as usize;
            assert_ne!(one, 0);
            assert_ne!(two, 0);
            crate::inplace::title_memory_view(one, bytes, section as usize, 0);
            crate::inplace::title_memory_view(two, bytes, section as usize, 0);
            (section as usize, one, two)
        }
    }

    /// The section of [`two_views`] is closed and its views unmapped; the
    /// guard is told they are gone first.
    fn drop_views(section: usize, views: &[usize], bytes: usize) {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
        for &view in views {
            title_memory_remapped(view, bytes);
            // SAFETY: a view of the section mapped by `two_views`.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                })
            };
        }
        // SAFETY: the section's last reference.
        unsafe { CloseHandle(section as *mut std::ffi::c_void) };
    }

    /// The title wrote the pages of a guarded texture through another view
    /// of them: the fault at that view is the guard's to see, and the
    /// guarded view's next check reports the pages written.
    #[test]
    fn a_write_through_another_view_is_reported_at_the_guarded_one() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        // The first check takes it in (and every view of its pages).
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        // Writing through the other view faults into the handler, which
        // opens that view and the guarded one: the next check sees it.
        // SAFETY: the second view, read-write once the handler has opened it.
        unsafe { ((other + BLOCK + 8) as *mut u8).write(1) };
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// A view of guarded pages is mapped anew: it is writable, so the
    /// guarded view checks its pages again (and guards the new view).
    #[test]
    fn a_view_mapped_beside_a_guarded_one_makes_it_check_again() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        title_memory_remapped(other, SIZE);
        crate::inplace::title_memory_view(other, SIZE, section, 0);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// A write of the runtime's own, told at one view of the pages
    /// (`writing`): the other views' next checks report the pages written.
    #[test]
    fn a_write_told_at_one_view_is_reported_at_the_others() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        writing(other + BLOCK, BLOCK);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        // Half a page and the next: both blocks, not just the first's.
        writing(other + BLOCK / 2, BLOCK);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// Both views of the pages are checked, each for a copy of its own
    /// (the title shows one physical page at several addresses, and each
    /// address has its host copy): a write through either is reported at
    /// both, whichever is checked first. Guarding the pages for the first
    /// view's check must not note the second view clean, or its copy would
    /// keep the bytes from before the write.
    #[test]
    fn a_write_is_reported_at_every_checked_view() {
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let pages = |view: usize, block: usize| (view + block * BLOCK) / PAGE;
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 0), pages(one, 1)]));
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 0), pages(two, 1)]), "the second view's first check");
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()), "nothing written yet");
        assert_eq!(guard(two, two + SIZE), Some(Vec::new()), "nothing written yet");
        // SAFETY: the second view, read-write once the handler has opened it.
        unsafe { ((two + BLOCK + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 1)]));
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 1)]), "written through this view, checked second");
        // SAFETY: the first view, as above.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 0)]));
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 0)]), "written through this view, checked second");
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()));
        assert_eq!(guard(two, two + SIZE), Some(Vec::new()));
        drop_views(section, &[one, two], SIZE);
    }

    /// A page shown in more than `MANY_VIEWS` views is guarded at each by
    /// itself: the others stay writable, and nothing looks through them all.
    #[test]
    fn a_page_of_many_views_is_guarded_one_view_at_a_time() {
        let (section, one, two) = two_views(BLOCK);
        let filler = one + (1 << 45);
        for n in 0..MANY_VIEWS {
            crate::inplace::title_memory_view(filler + n * BLOCK, BLOCK, section, 0);
        }
        assert_eq!(guard(one, one + BLOCK), Some(vec![one / PAGE]));
        assert!(writable(two), "another view of it is left alone");
        // SAFETY: the guarded view, opened by the handler.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + BLOCK), Some(vec![one / PAGE]));
        title_memory_remapped(filler, MANY_VIEWS * BLOCK);
        drop_views(section, &[one, two], BLOCK);
    }

    /// A write or mapping that runs past the end of the address space (a
    /// title's `fread` with a huge count) covers up to the end; the sum used
    /// to overflow. Up there no test has guarded blocks.
    #[test]
    fn ranges_past_the_address_space_end_do_not_overflow() {
        let top = usize::MAX - 16;
        writing(top, PAGE);
        title_memory_remapped(top, usize::MAX);
    }

    /// Helper: clears test pages from `Journal::written` after releasing a test
    /// allocation so Windows VA reuse across `--test-threads=1` tests sees a
    /// clean journal.
    fn forget_written(start: usize, len: usize) {
        let mut j = journal_write();
        for page in start / PAGE..start.saturating_add(len).div_ceil(PAGE) {
            j.written.remove(&page);
        }
    }

    /// Helper: allocates a fresh pagefile-backed view of `bytes` and returns
    /// `(section_handle, view_base)`.
    fn one_view(bytes: usize) -> (usize, usize) {
        use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile,
        };
        // SAFETY: pagefile-backed section mapped once; caller frees with `drop_views`.
        unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                bytes as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes).Value as usize;
            assert_ne!(view, 0);
            (section as usize, view)
        }
    }

    /// Consecutive 4 KiB kernel pages inside the same 64 KiB `PAGE` collapse
    /// into a single `BTreeMap` entry before updating `Journal::written`.
    #[test]
    fn dedup_folds_consecutive_4k_pages_into_64k_pages() {
        let base = 0x1000 * PAGE;
        let mut addrs = Vec::new();
        for sub in 0..(PAGE / KERNEL_PAGE) {
            addrs.push(base + sub * KERNEL_PAGE);
        }
        for sub in 0..(PAGE / KERNEL_PAGE) {
            addrs.push(base + 2 * PAGE + sub * KERNEL_PAGE);
        }
        let mut out = Vec::new();
        dedup_pages(&addrs, &mut out);
        assert_eq!(out, vec![base / PAGE, base / PAGE + 2]);

        // End-to-end against live `GetWriteWatch`: dirty all 16 4 KiB sub-pages
        // in page 0 and page 2 of a 3-page write-watched allocation.
        // SAFETY: fresh write-watched allocation, released below.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                3 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        assert!(writes(p, p + 3 * PAGE, false).is_some());
        for sub in 0..(PAGE / KERNEL_PAGE) {
            // SAFETY: committed write-watched sub-pages of pages 0 and 2.
            unsafe {
                ((p + sub * KERNEL_PAGE) as *mut u8).write_volatile(0x11);
                ((p + 2 * PAGE + sub * KERNEL_PAGE) as *mut u8).write_volatile(0x22);
            }
        }
        let (now, dirty) = writes(p, p + 3 * PAGE, false).expect("watched");
        assert_eq!(dirty, vec![(p / PAGE, now), (p / PAGE + 2, now)]);
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
        forget_written(p, 3 * PAGE);
    }

    /// Interval coalescing, insertion, covering, and split-eviction invariants.
    #[test]
    fn ahead_interval_helpers_keep_sorted_disjoint_invariant() {
        let mut v = vec![(30, 40), (10, 15), (15, 20), (12, 18), (50, 60)];
        coalesce_ranges(&mut v);
        assert_eq!(v, vec![(10, 20), (30, 40), (50, 60)]);
        assert!(covers(&v, 10, 20));
        assert!(covers(&v, 12, 18));
        assert!(!covers(&v, 15, 25));
        assert!(!covers(&v, 0, 5));

        insert_coalesced(&mut v, 20, 30);
        assert_eq!(v, vec![(10, 40), (50, 60)]);

        evict_range(&mut v, 15, 25);
        assert_eq!(v, vec![(10, 15), (25, 40), (50, 60)]);
        assert!(covers(&v, 10, 15));
        assert!(!covers(&v, 14, 16));
        assert!(covers(&v, 25, 40));

        evict_range(&mut v, 12, 55);
        assert_eq!(v, vec![(10, 12), (55, 60)]);
    }

    /// `ahead` and `ahead_guarded` stay separate so `writes(..., false)` never
    /// returns `Some` for a guarded-only physical-page view, while `writes(..., true)`
    /// and `asked(...)` hit `ahead_guarded` without re-running `guard()`, and
    /// `writing` / `wrote` (with aliases) / `title_memory_remapped` evict cleanly.
    #[test]
    fn ahead_guarded_respects_guarded_false_and_evicts_on_invalidation() {
        const VIEW_SIZE: usize = 3 * BLOCK;
        let (section, view) = one_view(VIEW_SIZE);
        // SAFETY: fresh 2-page write-watched allocation, freed below.
        let ww = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                2 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(ww, 0);

        // Prime `view` so it enters `GUARDS` and becomes a `known_view`.
        assert!(writes(view, view + VIEW_SIZE, true).is_some());
        assert!(known_view(view));

        ask_once(&[(ww, ww + 2 * PAGE), (view, view + VIEW_SIZE)]);
        let (asked_before, ahead_ww, ahead_view) = {
            let j = journal_read();
            (j.asked, j.ahead.clone(), j.ahead_guarded.clone())
        };
        assert!(covers(&ahead_ww, ww / PAGE, (ww + 2 * PAGE) / PAGE));
        assert!(!covers(&ahead_ww, view / PAGE, (view + VIEW_SIZE) / PAGE));
        assert!(covers(&ahead_view, view / PAGE, (view + VIEW_SIZE) / PAGE));
        assert!(!covers(&ahead_view, ww / PAGE, (ww + 2 * PAGE) / PAGE));

        // `guarded = false` MUST return `None` for the guarded-only view.
        assert!(writes(view, view + VIEW_SIZE, false).is_none());
        // `guarded = true` and `asked` hit `ahead_guarded` without advancing `j.asked`.
        let (now_view, _) = writes(view, view + VIEW_SIZE, true).expect("guarded hit");
        assert_eq!(now_view, asked_before);
        let (now_asked, _) = asked(view, view + VIEW_SIZE).expect("asked hit");
        assert_eq!(now_asked, asked_before);
        // Write-watched range hits `ahead` for both `guarded = false` and `guarded = true`.
        let (now_ww_f, _) = writes(ww, ww + 2 * PAGE, false).expect("ww hit false");
        let (now_ww_t, _) = writes(ww, ww + 2 * PAGE, true).expect("ww hit true");
        assert_eq!(now_ww_f, asked_before);
        assert_eq!(now_ww_t, asked_before);
        assert_eq!(journal_read().asked, asked_before);

        // `writing` on the middle block splits `ahead_guarded`, keeping block 0 and 2 cached.
        writing(view + BLOCK, BLOCK);
        {
            let j = journal_read();
            assert!(covers(&j.ahead_guarded, view / PAGE, view / PAGE + 1));
            assert!(!covers(&j.ahead_guarded, view / PAGE + 1, view / PAGE + 2));
            assert!(covers(&j.ahead_guarded, view / PAGE + 2, view / PAGE + 3));
        }

        // `title_memory_remapped` on block 2 evicts block 2 while keeping block 0 cached.
        title_memory_remapped(view + 2 * BLOCK, BLOCK);
        {
            let j = journal_read();
            assert!(covers(&j.ahead_guarded, view / PAGE, view / PAGE + 1));
            assert!(!covers(&j.ahead_guarded, view / PAGE + 2, view / PAGE + 3));
        }
        ask_once_end();
        drop_views(section, &[view], VIEW_SIZE);
        forget_written(view, VIEW_SIZE);
        // SAFETY: our allocation.
        unsafe { VirtualFree(ww as *mut _, 0, MEM_RELEASE) };
        forget_written(ww, 2 * PAGE);

        // Alias invalidation under `ask_once` + `writing` + `wrote`.
        let (alias_sec, one, two) = two_views(2 * BLOCK);
        assert!(writes(one, one + 2 * BLOCK, true).is_some());
        assert!(writes(two, two + 2 * BLOCK, true).is_some());
        ask_once(&[(one, one + 2 * BLOCK), (two, two + 2 * BLOCK)]);
        let snap = journal_read().ahead_asked;
        writing(one, BLOCK);
        // SAFETY: block 0 of `one` was opened by `writing`.
        unsafe { ((one + 16) as *mut u8).write_volatile(0x7f) };
        wrote(one, BLOCK);
        {
            let j = journal_read();
            assert!(!covers(&j.ahead_guarded, one / PAGE, one / PAGE + 1));
            assert!(!covers(&j.ahead_guarded, two / PAGE, two / PAGE + 1));
            assert!(covers(&j.ahead_guarded, one / PAGE + 1, one / PAGE + 2));
            assert!(covers(&j.ahead_guarded, two / PAGE + 1, two / PAGE + 2));
        }
        let (after, pages_two) = writes(two, two + 2 * BLOCK, true).expect("alias view");
        assert!(after > snap);
        assert!(
            pages_two.iter().any(|&(p, w)| p == two / PAGE && w > snap),
            "write via view `one` must be reported at aliased view `two`"
        );
        ask_once_end();
        drop_views(alias_sec, &[one, two], 2 * BLOCK);
        forget_written(one, 2 * BLOCK);
        forget_written(two, 2 * BLOCK);
    }

    /// `known_view` ranges skip `GetWriteWatch` in `ask_once`, `ask_ahead`,
    /// and `writes(..., false)`, and do not force bisection of write-watched
    /// ranges passed alongside them.
    #[test]
    fn known_view_skips_kernel_asks_in_ahead_and_once() {
        use std::sync::atomic::Ordering::Relaxed;
        const VIEW_SIZE: usize = 2 * BLOCK;
        let (section, view) = one_view(VIEW_SIZE);
        // SAFETY: fresh 4-page write-watched allocation, freed below.
        let ww = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(ww, 0);

        // Prime `view` into `GUARDS`.
        assert!(writes(view, view + VIEW_SIZE, true).is_some());
        assert!(known_view(view));

        let k0 = KERNEL_ASKS.load(Relaxed);
        ask_once(&[(view, view + VIEW_SIZE)]);
        assert_eq!(KERNEL_ASKS.load(Relaxed) - k0, 0, "ask_once must skip GetWriteWatch for known views");
        ask_once_end();

        let k1 = KERNEL_ASKS.load(Relaxed);
        ask_ahead(&[(view, view + BLOCK), (view + BLOCK, view + VIEW_SIZE)]);
        assert_eq!(KERNEL_ASKS.load(Relaxed) - k1, 0, "ask_ahead must skip GetWriteWatch for known views");
        ask_ahead_done();

        let k2 = KERNEL_ASKS.load(Relaxed);
        assert!(writes(view, view + VIEW_SIZE, false).is_none());
        assert_eq!(KERNEL_ASKS.load(Relaxed) - k2, 0, "writes(..., false) must skip GetWriteWatch for known views");

        // Passing a known view alongside two pieces of one write-watched range
        // must filter out the view before grouping, issuing exactly 1 kernel ask.
        let k3 = KERNEL_ASKS.load(Relaxed);
        ask_once(&[
            (ww, ww + PAGE),
            (view, view + VIEW_SIZE),
            (ww + 2 * PAGE, ww + 3 * PAGE),
        ]);
        assert_eq!(KERNEL_ASKS.load(Relaxed) - k3, 1, "known view must not trigger bisection");
        {
            let j = journal_read();
            assert!(covers(&j.ahead, ww / PAGE, (ww + 3 * PAGE) / PAGE));
            assert!(covers(&j.ahead_guarded, view / PAGE, (view + VIEW_SIZE) / PAGE));
        }
        ask_once_end();

        drop_views(section, &[view], VIEW_SIZE);
        forget_written(view, VIEW_SIZE);
        // SAFETY: our allocation.
        unsafe { VirtualFree(ww as *mut _, 0, MEM_RELEASE) };
        forget_written(ww, 4 * PAGE);
    }

    /// `asked()` and `writes()` ahead-hits acquire only `JOURNAL.read()`, so
    /// 8 concurrent worker threads complete even while `WATCH_ASK` is held by
    /// an in-flight miss syscall, and overlapping concurrent misses never drop
    /// dirty pages.
    #[test]
    fn concurrent_ahead_reads_do_not_block_on_watch_ask_and_misses_never_drop_pages() {
        use std::sync::{Arc, Barrier};
        const VIEW_SIZE: usize = 2 * BLOCK;
        let (section, view) = one_view(VIEW_SIZE);
        // SAFETY: fresh 4-page write-watched allocation, freed below.
        let ww = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(ww, 0);

        assert!(writes(view, view + VIEW_SIZE, true).is_some());
        ask_once(&[(ww, ww + 4 * PAGE), (view, view + VIEW_SIZE)]);
        let expected_asked = journal_read().ahead_asked;

        // Hold `WATCH_ASK` to prove ahead-hit readers never touch it or take a write lock.
        let ask_guard = watch_ask();
        let barrier = Arc::new(Barrier::new(8));
        let mut handles = Vec::with_capacity(8);
        for _ in 0..8 {
            let b = barrier.clone();
            handles.push(std::thread::spawn(move || {
                b.wait();
                for _ in 0..64 {
                    let (a1, _) = asked(ww, ww + 4 * PAGE).expect("ww asked");
                    let (a2, _) = asked(view, view + VIEW_SIZE).expect("view asked");
                    let (w1, _) = writes(ww, ww + 4 * PAGE, false).expect("ww writes");
                    let (w2, _) = writes(view, view + VIEW_SIZE, true).expect("view writes");
                    assert_eq!(a1, expected_asked);
                    assert_eq!(a2, expected_asked);
                    assert_eq!(w1, expected_asked);
                    assert_eq!(w2, expected_asked);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        drop(ask_guard);
        ask_once_end();

        // Overlapping miss-path stress: 4 threads concurrently query overlapping
        // write-watched ranges while a writer dirties pages; every write must be
        // observed in `Journal::written` with an asking strictly greater than before.
        let (baseline, _) = writes(ww, ww + 4 * PAGE, false).expect("baseline");
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut readers = Vec::with_capacity(4);
        for tid in 0..4 {
            let done = done.clone();
            readers.push(std::thread::spawn(move || {
                while !done.load(std::sync::atomic::Ordering::Acquire) {
                    let lo = ww + (tid % 2) * PAGE;
                    let hi = ww + (2 + tid % 3) * PAGE;
                    let res = writes(lo, hi, false);
                    assert!(res.is_some());
                }
            }));
        }
        for round in 0..32u8 {
            for page in 0..4 {
                // SAFETY: committed write-watched page owned by this test.
                unsafe { ((ww + page * PAGE + 8) as *mut u8).write_volatile(round) };
            }
            std::thread::yield_now();
        }
        done.store(true, std::sync::atomic::Ordering::Release);
        for r in readers {
            r.join().unwrap();
        }
        let (_final_now, pages) = writes(ww, ww + 4 * PAGE, false).expect("final");
        for page in 0..4 {
            let p = ww / PAGE + page;
            let seen = pages.iter().find(|&&(pg, _)| pg == p).map(|&(_, w)| w);
            assert!(
                seen.is_some_and(|w| w > baseline),
                "dirty page {page} lost across concurrent overlapping writes()"
            );
        }

        drop_views(section, &[view], VIEW_SIZE);
        forget_written(view, VIEW_SIZE);
        // SAFETY: our allocation.
        unsafe { VirtualFree(ww as *mut _, 0, MEM_RELEASE) };
        forget_written(ww, 4 * PAGE);
    }

    /// Disk-filled writes delivered through a separate runtime view are
    /// invisible to both `GetWriteWatch` and page guards: `title_memory_written`
    /// must record those pages in `JOURNAL`, advance `ahead_asked` when an
    /// ahead snapshot stands, and evict `ahead_guarded` cleanly.
    #[test]
    fn title_memory_written_marks_journal_pages_without_guards_or_write_watch() {
        // SAFETY: fresh 4-page write-watched allocation, released below.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 4 * PAGE);
        let (baseline, before) = writes(start, end, false).expect("watched");
        assert!(before.is_empty());

        // Simulate runtime writing pages 1 and 2 via a separate view without
        // touching `p` directly (so `GetWriteWatch` sees nothing).
        title_memory_written(p + PAGE + 128, PAGE, true);
        let (now, dirty) = writes(start, end, false).expect("watched");
        assert!(now > baseline);
        let p1 = p / PAGE + 1;
        let p2 = p / PAGE + 2;
        assert_eq!(dirty.len(), 2, "only pages 1 and 2 must be marked: {dirty:?}");
        assert_eq!(dirty[0].0, p1);
        assert_eq!(dirty[1].0, p2);
        assert_eq!(dirty[0].1, dirty[1].1);
        assert!(dirty[0].1 > baseline && dirty[0].1 < now);

        // Under active `ask_once` (both write-watched `ahead` and guarded
        // `ahead_guarded`): `title_memory_written` marks page 0 written,
        // advances `ahead_asked` so `written <= ahead_asked`, and evicts the
        // written block from `ahead_guarded`.
        const VIEW_SIZE: usize = 2 * BLOCK;
        let (section, view) = one_view(VIEW_SIZE);
        assert!(writes(view, view + VIEW_SIZE, true).is_some());
        ask_once(&[(start, end), (view, view + VIEW_SIZE)]);
        let snap = journal_read().ahead_asked;
        title_memory_written(p, 64, true);
        let (after_ww, pages_ww) = asked(start, end).expect("still covered by ahead");
        assert!(after_ww > snap);
        assert!(
            pages_ww.iter().any(|&(pg, w)| pg == p / PAGE && w == after_ww),
            "page 0 must be marked at updated ahead_asked={after_ww}: {pages_ww:?}"
        );
        title_memory_written(view, BLOCK, true);
        {
            let j = journal_read();
            assert!(!covers(&j.ahead_guarded, view / PAGE, view / PAGE + 1));
            assert!(covers(&j.ahead_guarded, view / PAGE + 1, view / PAGE + 2));
            assert!(j.written.get(&(view / PAGE)).is_some_and(|&w| w > after_ww));
        }
        ask_once_end();

        // Overflow near the top of the address space must saturate cleanly.
        title_memory_written(usize::MAX - 16, PAGE, true);

        drop_views(section, &[view], VIEW_SIZE);
        forget_written(view, VIEW_SIZE);
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
        forget_written(p, 4 * PAGE);
        forget_written(usize::MAX - 16, PAGE);
    }
}

#[cfg(test)]
mod mapping_tests {
    use super::{title_memory_reading, title_memory_remapping};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    /// A change of mapping waits for a read in place that is under way, and
    /// a read in place waits for a change of mapping.
    #[test]
    fn reads_in_place_and_remappings_wait_for_each_other() {
        // A reader holds; the change does not begin until it lets go.
        let changed = Arc::new(AtomicBool::new(false));
        let reading = title_memory_reading();
        let (began, wait) = mpsc::channel();
        let change = {
            let changed = changed.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _changing = title_memory_remapping();
                changed.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert!(!changed.load(Ordering::SeqCst), "the mapping changed under a read in place");
        drop(reading);
        change.join().unwrap();
        assert!(changed.load(Ordering::SeqCst));
        // A change holds; the read does not begin until it is done.
        let read = Arc::new(AtomicBool::new(false));
        let changing = title_memory_remapping();
        let (began, wait) = mpsc::channel();
        let reader = {
            let read = read.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _reading = title_memory_reading();
                read.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert!(!read.load(Ordering::SeqCst), "a read in place began during a change of mapping");
        drop(changing);
        reader.join().unwrap();
        assert!(read.load(Ordering::SeqCst));
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod fast_copy_tests {
    use super::{
        KERNEL_PAGE, PAGE, title_memory_reading, title_memory_remapping, try_copy_fast,
        try_diff_bounds_fast,
    };
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE,
        VirtualAlloc, VirtualFree, VirtualProtect,
    };

    #[test]
    fn try_copy_fast_handles_zero_unaligned_and_large_buffers() {
        let mut sentinel = [0xaa_u8; 8];
        // Zero-length copy succeeds even with a null source without touching memory.
        assert!(try_copy_fast(sentinel.as_mut_ptr(), std::ptr::null(), 0));
        assert_eq!(sentinel, [0xaa; 8]);

        // Unaligned small and medium sizes across 16-byte / page boundaries.
        let src: Vec<u8> = (0..(2 * PAGE + 317)).map(|i| (i.wrapping_mul(31) ^ (i >> 5)) as u8).collect();
        for &(src_off, dst_off, len) in &[
            (0, 0, 1),
            (1, 3, 7),
            (3, 1, 15),
            (5, 7, 16),
            (7, 2, 17),
            (13, 9, 63),
            (15, 11, 64),
            (19, 5, 4097),
            (27, 13, PAGE + 123),
            (3, 17, 2 * PAGE + 257),
        ] {
            let mut dst = vec![0xee_u8; dst_off + len + 19];
            // SAFETY: both slices cover `len` bytes from their respective offsets.
            let ok = unsafe { try_copy_fast(dst.as_mut_ptr().add(dst_off), src.as_ptr().add(src_off), len) };
            assert!(ok, "copy failed for ({src_off}, {dst_off}, {len})");
            assert_eq!(&dst[dst_off..dst_off + len], &src[src_off..src_off + len]);
            assert!(dst[..dst_off].iter().all(|&b| b == 0xee));
            assert!(dst[dst_off + len..].iter().all(|&b| b == 0xee));
        }
    }

    #[test]
    fn try_copy_fast_recovers_from_uncommitted_and_noaccess_pages() {
        let mut dst = vec![0x55_u8; 2 * KERNEL_PAGE];
        // SAFETY: reserves two pages without committing; freed before returning.
        unsafe {
            let reserved = VirtualAlloc(std::ptr::null(), 2 * KERNEL_PAGE, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!reserved.is_null());
            assert!(!try_copy_fast(dst.as_mut_ptr(), reserved, KERNEL_PAGE));

            // Commit the first page as PAGE_NOACCESS and verify recovery as well.
            let committed = VirtualAlloc(reserved.cast(), KERNEL_PAGE, MEM_COMMIT, PAGE_NOACCESS);
            assert_eq!(committed, reserved.cast());
            assert!(!try_copy_fast(dst.as_mut_ptr(), reserved, 64));

            // Make it PAGE_READONLY (as guarded views are) and verify fast copy succeeds.
            let mut old = 0;
            assert_ne!(VirtualProtect(reserved.cast(), KERNEL_PAGE, PAGE_READWRITE, &mut old), 0);
            reserved.write_bytes(0x3c, KERNEL_PAGE);
            assert_ne!(VirtualProtect(reserved.cast(), KERNEL_PAGE, PAGE_READONLY, &mut old), 0);
            dst.fill(0);
            assert!(try_copy_fast(dst.as_mut_ptr(), reserved, KERNEL_PAGE));
            assert!(dst[..KERNEL_PAGE].iter().all(|&b| b == 0x3c));

            assert_ne!(VirtualFree(reserved.cast(), 0, MEM_RELEASE), 0);
        }
        // Null pointer also recovers cleanly without crashing.
        assert!(!try_copy_fast(dst.as_mut_ptr(), std::ptr::null(), 16));
    }

    #[test]
    fn try_copy_fast_straddling_committed_and_uncommitted_recovers_cleanly() {
        // SAFETY: allocates a 2-page reservation with only page 0 committed; freed below.
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), 2 * KERNEL_PAGE, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            assert_eq!(VirtualAlloc(base.cast(), KERNEL_PAGE, MEM_COMMIT, PAGE_READWRITE), base.cast());
            base.write_bytes(0x7a, KERNEL_PAGE);

            let mut dst = vec![0u8; 2 * KERNEL_PAGE];
            // Unaligned copy crossing from committed page 0 into uncommitted page 1.
            assert!(!try_copy_fast(dst.as_mut_ptr(), base.add(37), 2 * KERNEL_PAGE - 37));
            // A subsequent valid copy within page 0 must still succeed (slot cleanly reset).
            let mut ok_dst = vec![0u8; KERNEL_PAGE - 37];
            assert!(try_copy_fast(ok_dst.as_mut_ptr(), base.add(37), KERNEL_PAGE - 37));
            assert!(ok_dst.iter().all(|&b| b == 0x7a));

            assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
        }
    }

    #[test]
    fn try_diff_bounds_fast_matches_bytewise_across_sizes_and_offsets() {
        // Empty slice never touches source.
        assert_eq!(try_diff_bounds_fast(std::ptr::null(), &[]), Some(None));

        let base: Vec<u8> = (0..120u32).map(|i| (i * 13 + 5) as u8).collect();
        for len in [1, 2, 7, 8, 15, 16, 17, 31, 32, 33, 47, 48, 49, 65] {
            let cur = &base[3..3 + len];
            let same = cur.to_vec();
            assert_eq!(try_diff_bounds_fast(same.as_ptr(), cur), Some(None), "identical len={len}");
            for first in 0..len {
                for last in first..len {
                    let mut src = cur.to_vec();
                    src[first] ^= 0x80;
                    src[last] ^= 0x01;
                    let expected_first = src.iter().zip(cur).position(|(a, b)| a != b).unwrap();
                    let expected_last = src.iter().zip(cur).rposition(|(a, b)| a != b).unwrap();
                    assert_eq!(
                        try_diff_bounds_fast(src.as_ptr(), cur),
                        Some(Some((expected_first, expected_last))),
                        "len={len} first={first} last={last}"
                    );
                }
            }
        }

        // Large 64 KiB+ buffer: identical, start, middle, end, and two-point diffs.
        let big_len = PAGE + 37;
        let cur: Vec<u8> = (0..big_len).map(|i| (i.wrapping_mul(29) >> 2) as u8).collect();
        let mut src = cur.clone();
        assert_eq!(try_diff_bounds_fast(src.as_ptr(), &cur), Some(None));
        for &pos in &[0, 1, 15, 16, 17, 4095, 4096, PAGE / 2, big_len - 17, big_len - 16, big_len - 2, big_len - 1] {
            src[pos] ^= 0xff;
            assert_eq!(try_diff_bounds_fast(src.as_ptr(), &cur), Some(Some((pos, pos))), "pos={pos}");
            src[pos] ^= 0xff;
        }
        src[19] ^= 0x11;
        src[big_len - 23] ^= 0x22;
        assert_eq!(try_diff_bounds_fast(src.as_ptr(), &cur), Some(Some((19, big_len - 23))));
    }

    #[test]
    fn try_diff_bounds_fast_recovers_from_uncommitted_pages() {
        // SAFETY: allocates a 3-page reservation with page 0 committed, page 1 uncommitted,
        // and page 2 committed; freed before returning.
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), 3 * KERNEL_PAGE, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            assert_eq!(VirtualAlloc(base.cast(), KERNEL_PAGE, MEM_COMMIT, PAGE_READWRITE), base.cast());
            base.write_bytes(0x42, KERNEL_PAGE);

            // 1. Wholly uncommitted page -> forward scan faults on byte 0.
            let cur1 = vec![0x42_u8; KERNEL_PAGE];
            assert_eq!(try_diff_bounds_fast(base.add(KERNEL_PAGE), &cur1), None);

            // 2. Identical committed page 0 + uncommitted page 1 -> forward scan faults at page 1.
            let cur2 = vec![0x42_u8; 2 * KERNEL_PAGE];
            assert_eq!(try_diff_bounds_fast(base, &cur2), None);

            // 3. Diff at byte 0 on committed page 0 + uncommitted page 1 -> backward scan faults at page 1!
            let mut cur3 = vec![0x42_u8; 2 * KERNEL_PAGE];
            cur3[0] = 0x00;
            assert_eq!(try_diff_bounds_fast(base, &cur3), None);

            // 4. Unaligned tail fault: committed page 0 identical + 5 bytes into uncommitted page 1.
            let cur4 = vec![0x42_u8; KERNEL_PAGE + 5];
            assert_eq!(try_diff_bounds_fast(base, &cur4), None);

            // 5. Committed page 0 + uncommitted page 1 + committed page 2 with diffs at both ends:
            // diff bounds find (0, 3*KERNEL_PAGE - 1), and then try_copy_fast faults cleanly on page 1.
            assert_eq!(
                VirtualAlloc(base.add(2 * KERNEL_PAGE).cast(), KERNEL_PAGE, MEM_COMMIT, PAGE_READWRITE),
                base.add(2 * KERNEL_PAGE).cast()
            );
            base.add(2 * KERNEL_PAGE).write_bytes(0x42, KERNEL_PAGE);
            let mut cur5 = vec![0x42_u8; 3 * KERNEL_PAGE];
            cur5[0] = 0x99;
            cur5[3 * KERNEL_PAGE - 1] = 0x88;
            assert_eq!(
                try_diff_bounds_fast(base, &cur5),
                Some(Some((0, 3 * KERNEL_PAGE - 1)))
            );
            assert!(!try_copy_fast(cur5.as_mut_ptr(), base, cur5.len()));

            // Subsequent valid diff on committed page 0 still works.
            let mut cur_ok = vec![0x42_u8; KERNEL_PAGE];
            assert_eq!(try_diff_bounds_fast(base, &cur_ok), Some(None));
            cur_ok[123] = 0;
            assert_eq!(try_diff_bounds_fast(base, &cur_ok), Some(Some((123, 123))));

            assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
        }
    }

    #[test]
    fn fast_paths_yield_to_remapping_and_allow_nested_reading() {
        let src = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17];
        let mut dst = [0u8; 17];

        // While a remapping holds the write lock, fast paths immediately return false / None.
        let remapping = title_memory_remapping();
        assert!(!try_copy_fast(dst.as_mut_ptr(), src.as_ptr(), src.len()));
        assert_eq!(try_diff_bounds_fast(src.as_ptr(), &dst), None);
        drop(remapping);

        // While this thread already holds `Reading`, nested fast paths succeed without deadlocking.
        let reading = title_memory_reading();
        assert_eq!(try_diff_bounds_fast(src.as_ptr(), &dst), Some(Some((0, 16))));
        assert!(try_copy_fast(dst.as_mut_ptr(), src.as_ptr(), src.len()));
        assert_eq!(dst, src);
        assert_eq!(try_diff_bounds_fast(src.as_ptr(), &dst), Some(None));
        drop(reading);
    }
}
