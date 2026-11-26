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
//!
//! With `VICTORY_LIGHT_REFRESH` a block that is read-only by our hand,
//! unwritten since, and whose earlier writes are all in the journal has a
//! bit ([`PUBLISHED`]): a range whose blocks all have theirs is answered
//! from the journal alone, with no lock of the guards and no kernel call
//! ([`light`]).

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
    base_page: usize,
    seen: Mutex<Vec<u64>>,
}

/// Tracks a runtime GPU-memory range, including physical-page views when
/// `views` is true. Other process memory is not tracked.
pub(crate) fn track(start: usize, len: usize, views: bool) -> Option<Tracked> {
    let end = start.checked_add(len)?;
    if len == 0 || writes(start, end, views).is_none() {
        return None;
    }
    let base_page = start / PAGE;
    let pages = end.div_ceil(PAGE).saturating_sub(base_page);
    Some(Tracked {
        start,
        end,
        base_page,
        seen: Mutex::new(vec![0; pages]),
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

static ASK_ONCE_KEPT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Lock-free check whether [`ask_once`] currently holds an active asking for
/// the submission in flight.
pub(crate) fn ask_once_active() -> bool {
    ASK_ONCE_KEPT.load(std::sync::atomic::Ordering::Acquire)
}

impl Journal {
    fn set_kept(&mut self, value: bool) {
        self.kept = value;
        ASK_ONCE_KEPT.store(value, std::sync::atomic::Ordering::Release);
    }

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
    crate::inplace::for_each_alias_in_range(
        first.saturating_mul(PAGE),
        last.saturating_mul(PAGE),
        MANY_VIEWS,
        |_, alias| {
            let ap = alias / PAGE;
            if ap < first || ap >= last {
                out.push(ap);
            }
        },
    );
    out
}

/// Splits valid `(start, end)` byte ranges into write-watched candidates and
/// already-known physical-page views in a single `GUARDS` lock acquisition
/// followed by one `inplace` views lookup for any unguarded view ranges.
fn partition_pieces(ranges: &[(usize, usize)]) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
    partition_pieces_with(ranges, light())
}

fn partition_pieces_with(
    ranges: &[(usize, usize)],
    light_on: bool,
) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
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
    let mut unchecked = Vec::new();
    let mut unknown = Vec::new();
    for &(start, end) in ranges {
        if start < end {
            let piece = (start / PAGE, end.div_ceil(PAGE));
            if light_on && PUBLISHED.all(start / BLOCK, (start / BLOCK) + 1) {
                views.push(piece);
            } else {
                unchecked.push((start, piece));
            }
        }
    }
    if !unchecked.is_empty() {
        let g = guards();
        for (start, piece) in unchecked {
            if known_view_in(&g, piece.0 * PAGE) {
                views.push(piece);
            } else {
                let _ = start;
                unknown.push(piece);
            }
        }
    }
    if !unknown.is_empty() {
        crate::inplace::with_view_lookup(|in_view| {
            for piece in unknown {
                if in_view(piece.0 * PAGE) {
                    views.push(piece);
                } else {
                    ww.push(piece);
                }
            }
        });
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
/// and non-pinned covered intervals to `ahead_guarded`.
fn guard_view_pieces(
    view_pieces: &mut Vec<(usize, usize)>,
    ahead_guarded: &mut Vec<(usize, usize)>,
    dirty: &mut Vec<usize>,
) -> u64 {
    guard_view_pieces_with(view_pieces, ahead_guarded, dirty, light())
}

fn guard_view_pieces_with(
    view_pieces: &mut Vec<(usize, usize)>,
    ahead_guarded: &mut Vec<(usize, usize)>,
    dirty: &mut Vec<usize>,
    light_on: bool,
) -> u64 {
    coalesce_ranges(view_pieces);
    let rev_before = GUARD_REV.load(std::sync::atomic::Ordering::SeqCst);
    let mut bump = true;
    for &(first, last) in view_pieces.iter() {
        // `PAGE == BLOCK` (64 KiB), so each coalesced page interval aligns with block bits.
        if light_on && PUBLISHED.all(first, last) {
            LIGHT_ASKS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            ahead_guarded.push((first, last));
            continue;
        }
        if light_on {
            LIGHT_ASKS[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::WatchKnownView, 1);
        #[cfg(feature = "diagnostics")]
        let guarding = std::time::Instant::now();
        let written = guard_inner(first.saturating_mul(PAGE), last.saturating_mul(PAGE), bump);
        #[cfg(feature = "diagnostics")]
        let us = guarding.elapsed().as_micros() as u64;
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::GuardUs, us);
        if let Some((pages, any_pinned)) = written {
            bump = false;
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                add(C::Watches, 1);
                add(C::WatchUs, us);
                slow(us);
            }
            dirty.extend(pages);
            if !any_pinned {
                ahead_guarded.push((first, last));
            }
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
    if ask_once_active() {
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
    if light_kept() && !view_pieces.is_empty() {
        publish_ranges(&j, &dirty, &view_pieces, told());
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
    if ask_once_active() {
        return;
    }
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
    // The light refresh: a few of the holes known by their bits are asked
    // of Windows again, once per frame ([`look_at_holes`]).
    if light_kept() {
        static LAST_HOLE_FRAME: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(u64::MAX);
        let frame = crate::frame::current();
        if LAST_HOLE_FRAME.swap(frame, std::sync::atomic::Ordering::Relaxed) != frame {
            look_at_holes(HOLES_A_LOOK);
        }
    }
    let _ask = watch_ask();
    let need_clear = {
        let j = journal_read();
        j.kept || !j.ahead.is_empty() || !j.ahead_guarded.is_empty()
    };
    if need_clear {
        let mut j = journal_write();
        j.set_kept(false);
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
    let kept = !j.ahead.is_empty() || !j.ahead_guarded.is_empty();
    j.set_kept(kept);
    if light_kept() && !view_pieces.is_empty() {
        publish_ranges(&j, &dirty, &view_pieces, told());
    }
}

/// The submission's one asking stands no longer: its refreshes are done,
/// or title memory was written in a way the journal was not told.
pub(crate) fn ask_once_end() {
    if !ask_once_active() || !journal_read().kept {
        return;
    }
    let _ask = watch_ask();
    let mut j = journal_write();
    if j.kept {
        j.set_kept(false);
        j.ahead.clear();
        j.ahead_guarded.clear();
    }
}

/// We wrote title memory `[at, at + len)` while a submission's one asking
/// stands (the title's buffer copies, done at submission): its pages count
/// as written after that asking, so the refreshes that follow copy them.
pub(crate) fn wrote(at: usize, len: usize) {
    if len == 0 || !ask_once_active() {
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

/// We stored shaders' bytes into title view memory `page` through a view of
/// our own ([`crate::inplace::back_own`]), without opening any guard; `others`
/// are the other title views of the same page. Returns the asking index at
/// which `page` was last found written (`0`: never) and the new one.
pub(crate) fn wrote_own(page: usize, others: &[usize]) -> (u64, u64) {
    let _ask = watch_ask();
    let mut j = journal_write();
    j.asked += 1;
    let now = j.asked;
    if j.kept || !j.ahead_guarded.is_empty() {
        j.ahead_asked = now;
    }
    let before = j.written.insert(page, now).unwrap_or(0);
    if !j.ahead_guarded.is_empty() {
        evict_range(&mut j.ahead_guarded, page, page + 1);
    }
    for &other in others {
        if other != page {
            j.written.insert(other, now);
            if !j.ahead_guarded.is_empty() {
                evict_range(&mut j.ahead_guarded, other, other + 1);
            }
        }
    }
    (before, now)
}

/// Whether the block `page` is read-only by our hand and unwritten since a
/// copy took it (diagnostics of `VICTORY_BACK_OWN_CHECK`).
pub(crate) fn clean(page: usize) -> bool {
    guards().clean.contains(&page)
}

/// `VICTORY_LIGHT_REFRESH`: 0 off, 1 on, 2 in turn ([`light`]).
fn light_mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_LIGHT_REFRESH").as_deref() {
        Ok("1") => 1,
        Ok("alt") => 2,
        _ => 0,
    })
}

/// Whether the bits of the light refresh are kept ([`PUBLISHED`]): with
/// the switch on, and in both phases of `alt`, so that the phase that asks
/// them finds them true whatever was written in the phase before.
fn light_kept() -> bool {
    light_mode() != 0
}

/// Whether an asking about view memory is answered from the bits of its
/// blocks where they can answer (`VICTORY_LIGHT_REFRESH=1`; off by
/// default).
///
/// A frame asks the guards about view memory 2,000 to 3,700 times, 4.5 to
/// 9.4 GB of it, to hear for 98-99 % that nothing was written; every such
/// asking takes the guards' lock, which the title's threads wait for in
/// their write faults, and reads the range's clean blocks off the set:
/// 22.8 ms a frame of the submitting thread in the rooms, 44.5 ms outdoors
/// (run 275). With the switch a range whose blocks all have their bit
/// ([`PUBLISHED`]) is answered from the journal alone ([`writes`],
/// [`nothing_new`]), and its first block is known for view memory without
/// the lock ([`known_view`]).
///
/// What changes in the answers: nothing. A range is known by its bits
/// exactly while the guards would find it all clean, or clean but for
/// holes noted for good ([`told`]); only who is asked changes.
///
/// `VICTORY_LIGHT_REFRESH=alt`: twenty seconds with, twenty without, in
/// turn, each change said in the log with the counts of the phase before.
/// The bits are kept in both phases ([`light_kept`]) and asked in one.
/// (After avatarmanz/VIctory-7900XTX 2aa8134b and 74045e46, by hand.)
pub(crate) fn light() -> bool {
    match light_mode() {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), std::sync::atomic::Ordering::Relaxed) != u8::from(on) {
                let [bits, locked, filled] = light_asks();
                tracing::info!(
                    "refreshes of host copies go the light way from now (VICTORY_LIGHT_REFRESH=alt): {}; since the change before: {bits} askings of view memory answered by the bits of their blocks, {locked} that took the guards' lock, {filled} holes found filled with nobody telling",
                    if on { "ON" } else { "off, as usual (the bits are kept)" }
                );
            }
            on
        }
    }
}

/// Since they were last taken ([`light_asks`]), with the light refresh
/// kept: askings of view memory answered by the bits of their blocks,
/// askings that went to the guards, and holes known by their bits that
/// Windows said were filled with nobody telling ([`look_at_holes`]).
static LIGHT_ASKS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// [`LIGHT_ASKS`], taken.
fn light_asks() -> [u64; 3] {
    [0, 1, 2].map(|i| LIGHT_ASKS[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}

/// The words of a piece of [`Bits`]: the bits of 4 GiB.
const BITS_WORDS: usize = 1 << 10;
/// The pieces of [`Bits`]: 128 TiB, what a process has.
const BITS_PIECES: usize = 1 << 15;

/// A bit per 64 KiB block of the address space, read and cleared with no
/// lock; in pieces made when first needed and never let go.
struct Bits([std::sync::atomic::AtomicPtr<BitsPiece>; BITS_PIECES]);

/// The bits of 4 GiB.
type BitsPiece = [std::sync::atomic::AtomicU64; BITS_WORDS];

impl Bits {
    const fn new() -> Bits {
        Bits([const { std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()) }; BITS_PIECES])
    }

    /// The piece that holds word `word`, if it was ever made.
    fn piece(&self, word: usize) -> Option<&BitsPiece> {
        let piece = self.0.get(word / BITS_WORDS)?.load(std::sync::atomic::Ordering::Acquire);
        // SAFETY: null, or a piece made by `set`, which is never freed.
        unsafe { piece.as_ref() }
    }

    /// Sets `block`'s bit (nothing past the address space).
    fn set(&self, block: usize) {
        use std::sync::atomic::Ordering::{AcqRel, Acquire, SeqCst};
        let word = block / 64;
        let Some(place) = self.0.get(word / BITS_WORDS) else {
            return;
        };
        if place.load(Acquire).is_null() {
            let made: *mut BitsPiece = Box::into_raw(Box::new([const { std::sync::atomic::AtomicU64::new(0) }; BITS_WORDS]));
            if place.compare_exchange(std::ptr::null_mut(), made, AcqRel, Acquire).is_err() {
                // SAFETY: made above and shown to nobody: another thread's
                // piece is in the place.
                drop(unsafe { Box::from_raw(made) });
            }
        }
        if let Some(piece) = self.piece(word) {
            piece[word % BITS_WORDS].fetch_or(1u64 << (block % 64), SeqCst);
        }
    }

    /// Clears `block`'s bit: ahead of what makes the block writable, and
    /// seen by whoever looks after it.
    fn clear(&self, block: usize) {
        let word = block / 64;
        if let Some(piece) = self.piece(word) {
            piece[word % BITS_WORDS].fetch_and(!(1u64 << (block % 64)), std::sync::atomic::Ordering::SeqCst);
        }
    }

    /// Whether the bits of all of `[first, last)` are set (not so of none).
    fn all(&self, first: usize, last: usize) -> bool {
        let mut block = first;
        while block < last {
            let word = block / 64;
            let until = ((word + 1) * 64).min(last);
            let mask = (u64::MAX >> (64 - (until - block))) << (block % 64);
            let Some(piece) = self.piece(word) else {
                return false;
            };
            if piece[word % BITS_WORDS].load(std::sync::atomic::Ordering::SeqCst) & mask != mask {
                return false;
            }
            block = until;
        }
        first < last
    }
}

/// The blocks an asking can answer for from the journal alone. A block has
/// its bit only while both of these hold: it is read-only by our hand and
/// noted clean (`Guards::clean`), or it is a hole noted for good that lies
/// in a placeholder reservation (`Guards::holes`); and the journal holds
/// every write to it that an asking found.
///
/// Why that is so. A bit is set in one place, [`publish`]: by the asking
/// that found the block written and had it made clean, after it noted that
/// in the journal and with the journal's lock still held, under the
/// guards' lock, and only if the block is clean at that moment. It is
/// cleared under the guards' lock before the block's write access is given
/// back (`open_here`, which the write fault's handler, [`writing`],
/// [`title_memory_written`] and [`views_mapped`] all go through), and when
/// the runtime tells that what the block shows is new
/// ([`title_memory_remapped`], [`title_memory_committed`]). Clearing does
/// nothing where no bit was ever set, so it is not behind the switch.
///
/// Who asks the bits holds the journal's lock from before the look until
/// it has read the journal ([`published`]), and nobody sets a bit without
/// that lock: a block opened and guarded again between the look and the
/// reading cannot have its bit back in time to be taken for unwritten. A
/// write that comes while the bits are looked at is no different from one
/// that comes just after an asking: its block's bit is gone, and the next
/// asking goes to the guards and finds the block open.
///
/// Our own stores through a view of our own (`crate::inplace::write_own`)
/// open no guard and take no bit: they are noted in the journal
/// ([`wrote_own`]), which is where an asking answered by the bits looks.
static PUBLISHED: Bits = Bits::new();

/// An asking of view memory `[start, end)` found the pages `written` (page
/// numbers) written, had the guards make their blocks read-only and clean
/// ([`guard`]) and has noted them in the journal, whose lock it holds
/// still (`_noted`): the blocks of them that are clean now are known by
/// their bits from here on ([`PUBLISHED`]). With `holes` ([`told`]), so
/// are the blocks of the range noted uncommitted for good that lie in a
/// placeholder reservation; Windows is asked that once for each
/// ([`placeholder`]).
///
/// The blocks an asking leaves clean that have no bit yet are the ones it
/// reports written: every other clean block was left clean by an asking
/// before it, which did this.
fn publish(_noted: &Journal, written: &[usize], start: usize, end: usize, holes: bool) {
    publish_ranges(_noted, written, &[(start / BLOCK, end.div_ceil(BLOCK))], holes);
}

fn publish_ranges(_noted: &Journal, written: &[usize], ranges: &[(usize, usize)], holes: bool) {
    let need_holes = holes
        && ranges
            .iter()
            .any(|&(first, last)| first < last && !PUBLISHED.all(first, last));
    if written.is_empty() && !need_holes {
        return;
    }
    let mut g = guards();
    let g = &mut *g;
    for &page in written {
        let block = page * PAGE / BLOCK;
        // (Not one that a write has opened since the guards were asked,
        // nor one that could not be made read-only.)
        if g.clean.contains(&block) {
            PUBLISHED.set(block);
        }
    }
    if !need_holes {
        return;
    }
    for &(first, last) in ranges {
        if first >= last || PUBLISHED.all(first, last) {
            continue;
        }
        // The notes of the range not looked at yet. Both sets are read in
        // block order, and every block of `holes` has a note.
        let mut fresh: Vec<usize> = {
            let mut looked = g.holes.range(first..last).copied().peekable();
            g.verified
                .range(first..last)
                .map(|(&block, _)| block)
                .filter(|&block| {
                    while looked.next_if(|&hole| hole < block).is_some() {}
                    looked.next_if_eq(&block).is_none()
                })
                .collect()
        };
        for block in g.absent_static.range(first..last) {
            if !g.holes.contains(&block) {
                fresh.push(block);
            }
        }
        for block in fresh {
            g.holes.insert(block);
            // (Asked now, not when the note was made: a view the runtime is
            // mapping there, and has not told of yet, is no placeholder.)
            if (g.absent.contains(&block) || g.absent_static.contains(&block)) && placeholder(block) {
                PUBLISHED.set(block);
            }
        }
    }
}

/// Whether every block of `[start, end)` has its bit ([`PUBLISHED`]): what
/// [`writes`] would answer of the range is in the journal. `_held`: the
/// journal's lock, which the caller holds from before this look until it
/// has read the journal (no bit is set meanwhile: [`publish`]).
fn published(_held: &Journal, start: usize, end: usize) -> bool {
    PUBLISHED.all(start / BLOCK, end.div_ceil(BLOCK))
}

/// Whether a block noted uncommitted stays noted until the runtime tells
/// of it (`VICTORY_ABSENT_ONCE=1` with `VICTORY_ABSENT_MS=told`): then, and
/// only then, can such a block be known by its bit ([`publish`]). With
/// notes that grow old, or none, every asking of a range that holds a hole
/// goes to the guards, which put the question to Windows as they do.
fn told() -> bool {
    absent_mode() == 1 && absent_keep_ms() == u64::MAX
}

#[link(name = "ntdll", kind = "raw-dylib")]
unsafe extern "system" {
    /// The kernel's question about a process's memory, `(process, address,
    /// class of the answer, answer out, its room, bytes answered out)`: an
    /// NTSTATUS, negative when it cannot answer. It changes nothing.
    fn NtQueryVirtualMemory(
        process: *mut std::ffi::c_void,
        at: *const std::ffi::c_void,
        class: u32,
        out: *mut std::ffi::c_void,
        len: usize,
        returned: *mut usize,
    ) -> i32;
}

/// Whether `block` lies in a placeholder reservation: memory in which pages
/// come to be only by mapping a view, or making an allocation, in its
/// place, which the runtime does and tells of ([`title_memory_remapped`],
/// [`title_memory_committed`]). An ordinary reservation can be committed
/// by anybody, and free memory taken by anybody, with no word to us:
/// `VirtualQuery` does not tell the three apart, this question does.
fn placeholder(block: usize) -> bool {
    /// `MEMORY_REGION_INFORMATION` (ntddk); only the kinds are read.
    #[repr(C)]
    #[derive(Default)]
    #[allow(dead_code)]
    struct Region {
        allocation_base: usize,
        allocation_protect: u32,
        region_type: u32,
        region_size: usize,
        commit_size: usize,
        partition_id: usize,
        node_preference: usize,
    }
    // (`MemoryRegionInformationEx`; class 3 answers with the kind of memory
    // in that word, not the flags.)
    const MEMORY_REGION_INFORMATION_EX: u32 = 7;
    const PLACEHOLDER_RESERVATION: u32 = 1 << 8;
    let mut region = Region::default();
    // SAFETY: the answer's room and its size; the kernel checks the address.
    let status = unsafe {
        NtQueryVirtualMemory(
            windows_sys::Win32::System::Threading::GetCurrentProcess(),
            (block * BLOCK) as *const _,
            MEMORY_REGION_INFORMATION_EX,
            (&raw mut region).cast(),
            size_of::<Region>(),
            std::ptr::null_mut(),
        )
    };
    status >= 0 && region.region_type & PLACEHOLDER_RESERVATION != 0
}

/// The note that `block` is uncommitted is gone, or is true no longer: the
/// block is not known by its bit any more ([`publish`]).
fn forget_hole(g: &mut Guards, block: usize) {
    if g.holes.remove(&block) {
        PUBLISHED.clear(block);
    }
}

/// How long a note kept for good lives for a hole that lies in no
/// placeholder reservation ([`note_life`]), in milliseconds.
const UNSURE_MS: u64 = 100;

/// How long the note that `block` is uncommitted lives, of `keep`
/// ([`absent_once`]). With the light refresh a note kept for good is kept
/// so only where pages cannot come to be without the runtime's word: in a
/// placeholder reservation, whose holes are known by their bits. A hole
/// that [`publish`] looked at and found in an ordinary reservation or in
/// free memory, which anybody can commit or take with no word to us, is
/// asked of Windows again when its note is [`UNSURE_MS`] old. (Without the
/// light refresh no hole is looked at, and `keep` stands.)
fn note_life(g: &Guards, block: usize, keep: u64) -> u64 {
    if keep == u64::MAX && g.holes.contains(&block) && !PUBLISHED.all(block, block + 1) {
        UNSURE_MS
    } else {
        keep
    }
}

/// Holes asked of Windows again at each submission ([`look_at_holes`]): a
/// question is a microsecond or so, and the submitting thread is the
/// frame's limit.
const HOLES_A_LOOK: usize = 16;

/// With the light refresh: `most` of the holes known by their bits are
/// asked of Windows again, the next ones each time and round again, in
/// case one was filled with nobody telling ([`title_memory_remapped`],
/// [`title_memory_committed`]), which this title is not known to do. One
/// that was loses its bit and its note, so that the next asking of it goes
/// to the guards, which find it; counted, and said once. One whose
/// reservation is a placeholder no longer (let go, or made anew) loses its
/// bit and keeps its note.
///
/// Not while the runtime changes a mapping: a view it has mapped and not
/// told of yet is no hole filled behind our back, and forgotten here it
/// could be made read-only before the runtime's word, which forgets the
/// guard and leaves the protection.
fn look_at_holes(most: usize) {
    static FROM: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let Some(_still) = title_memory_reading_now() else {
        return;
    };
    let part: Vec<usize> = {
        let g = guards();
        if g.holes.is_empty() {
            return;
        }
        let from = FROM.load(std::sync::atomic::Ordering::Relaxed);
        let part: Vec<usize> = g.holes.range(from..).take(most).copied().collect();
        FROM.store(part.last().map_or(0, |last| last + 1), std::sync::atomic::Ordering::Relaxed);
        part
    };
    for block in part {
        // (Not the ones looked at that lie in no placeholder: they have no
        // bit, and their notes are the guards' to keep.)
        if !PUBLISHED.all(block, block + 1) {
            continue;
        }
        let filled = block >= uncommitted_info(block).0;
        if !filled && placeholder(block) {
            continue;
        }
        let evicted = {
            let mut g = guards();
            // (Told of since the part was taken: forgotten already.)
            if !g.holes.contains(&block) {
                continue;
            }
            PUBLISHED.clear(block);
            if filled {
                g.holes.remove(&block);
                g.verified.remove(&block);
                g.absent.remove(&block);
                g.absent_static.remove(&block);
                GUARD_REV.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                LIGHT_ASKS[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                static SAID: std::sync::Once = std::sync::Once::new();
                SAID.call_once(|| {
                    tracing::warn!(
                        address = format_args!("{:#x}", block * BLOCK),
                        "a hole of view memory was filled with nobody telling: host copies of it held old bytes until now (VICTORY_LIGHT_REFRESH; said once, counted with the light refresh's askings)"
                    );
                });
                true
            } else {
                false
            }
        };
        if evicted {
            evict_ahead_guarded(block, block + 1, &[]);
        }
    }
}

/// Whether nothing of `[start, end)` was written that a copy does not
/// hold, known with nothing asked of the guards or the kernel: `copied` is
/// the copy's note, for each page from page `base` on, of the asking after
/// which it last took the page (0: never), as `crate::shadow` keeps it.
/// True only if [`writes`] would answer from the journal (the range was
/// asked ahead, or its blocks are known by their bits) and the answer would
/// have the copy take no page. `false` if something would have to be
/// asked, and always without the light refresh ([`light`]).
pub(crate) fn nothing_new(start: usize, end: usize, base: usize, copied: &[u64]) -> bool {
    light() && nothing_noted(start, end, base, copied)
}

/// [`nothing_new`], whatever the switch says.
fn nothing_noted(start: usize, end: usize, base: usize, copied: &[u64]) -> bool {
    if start >= end {
        return false;
    }
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let Some(taken) = first.checked_sub(base).and_then(|from| copied.get(from..last - base)) else {
        return false;
    };
    // A page never taken is taken at the next refresh (so is, at every
    // refresh, a page the copy's refreshes only ever hold in part).
    if taken.contains(&0) {
        return false;
    }
    let j = journal_read();
    // The bits first, then the journal, both under its lock ([`published`]).
    let by_bits = !j.covers_ahead(first, last, true);
    if by_bits && !published(&j, first * PAGE, last * PAGE) {
        return false;
    }
    let same = j.written.range(first..last).all(|(&page, &write)| write <= taken[page - first]);
    if same && by_bits {
        LIGHT_ASKS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    same
}

/// `VICTORY_ABSENT_ONCE`: 0 off, 1 on, 2 in turn ([`absent_once`]).
fn absent_mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_ABSENT_ONCE").as_deref() {
        Ok("1") => 1,
        Ok("alt") => 2,
        _ => 0,
    })
}

/// Whether a block of view memory that Windows said is uncommitted is not
/// asked about again for [`absent_keep_ms`] (`VICTORY_ABSENT_ONCE=1`; off
/// by default).
///
/// `VICTORY_ABSENT_ONCE=alt`: twenty seconds with, twenty without, in
/// turn, each change said in the log with the counts of the phase before.
fn absent_once() -> bool {
    match absent_mode() {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), std::sync::atomic::Ordering::Relaxed) != u8::from(on) {
                let [asked, kept] = absent_looks();
                tracing::info!(
                    "blocks of view memory found uncommitted are asked of Windows again only after {} ms from now (VICTORY_ABSENT_ONCE=alt): {}; since the change before: {asked} questions to Windows about such blocks, {kept} blocks answered from their note",
                    absent_keep_ms(),
                    if on { "ON" } else { "off, at every asking as usual" }
                );
            }
            on
        }
    }
}

/// How long a block found uncommitted is not asked about again
/// ([`absent_once`]): `VICTORY_ABSENT_MS`, 100 by default (about a frame),
/// 1 to 1000.
fn absent_keep_ms() -> u64 {
    static MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MS.get_or_init(|| match std::env::var("VICTORY_ABSENT_MS").as_deref().map(str::trim) {
        // Until the runtime tells of the block ([`title_memory_remapped`],
        // [`title_memory_committed`]): no question to Windows at all for a
        // block noted uncommitted. Run 279 with 100 ms: 600,000 to 690,000
        // questions in twenty seconds without the notes, and still 357,000
        // to 604,000 with them, because a frame is as long as a note lives.
        Ok("told") => u64::MAX,
        Ok(v) => v.parse::<u64>().map_or(100, |n| n.clamp(1, 1000)),
        Err(_) => 100,
    })
}

/// Since they were last taken ([`absent_looks`]): questions to Windows
/// about blocks noted absent, and such blocks answered from their note
/// ([`absent_once`]). Counted with the switch off too.
static ABSENT_LOOKS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// [`ABSENT_LOOKS`], taken.
fn absent_looks() -> [u64; 2] {
    [0, 1].map(|i| ABSENT_LOOKS[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}

/// Milliseconds since the guards' first asking: the clock of
/// [`absent_once`]'s notes.
fn guard_ms() -> u64 {
    static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    EPOCH.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64
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

fn query_writes<T>(
    start: usize,
    end: usize,
    guarded: bool,
    collect: impl FnOnce(&Journal, usize, usize) -> T,
) -> Option<(u64, T)> {
    // The light refresh: its bits kept, and asked or not, as the switch
    // says now.
    let light = if light_kept() { 1 + u8::from(light()) } else { 0 };
    // Always on: the counts every 65,536 askings that allow for view
    // memory (in turn they are said at each change).
    if guarded && light_mode() == 1 {
        static CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if (CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1).is_multiple_of(65_536) {
            let [bits, locked, filled] = light_asks();
            tracing::info!(
                "refreshes of host copies go the light way (VICTORY_LIGHT_REFRESH=1): of the last 65536 askings that allowed for view memory {bits} were answered by the bits of their blocks and {locked} took the guards' lock (the others were of memory the kernel watches); {filled} holes found filled with nobody telling"
            );
        }
    }
    query_writes_as(start, end, guarded, light, collect)
}

/// [`query_writes`] with the light refresh as `light` says ([`light`]): 0, as
/// without the switch, no bit is set or asked; 1, the blocks the asking
/// leaves clean get their bits and none is asked (the phase of `alt`
/// without); 2, a range whose blocks all have their bit is answered from
/// the journal.
fn query_writes_as<T>(
    start: usize,
    end: usize,
    guarded: bool,
    light: u8,
    collect: impl FnOnce(&Journal, usize, usize) -> T,
) -> Option<(u64, T)> {
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    // Fast path: shared read lock only so concurrent refresh workers never
    // serialize against each other or against miss-path syscalls.
    let hit_published = {
        let j = journal_read();
        if j.covers_ahead(first, last, guarded) {
            return Some((j.ahead_asked, collect(&j, first, last)));
        }
        guarded && light == 2 && published(&j, first * PAGE, last * PAGE)
    };
    // The light refresh: every block of the range is read-only by our hand
    // and unwritten since an asking noted it, or a hole noted for good
    // ([`PUBLISHED`]). The guards would find nothing written and this
    // asking would note nothing: its number and the journal as it is are
    // the answer, with the guards' lock not taken and no kernel call.
    if hit_published {
        let mut j = journal_write();
        if j.covers_ahead(first, last, guarded) {
            return Some((j.ahead_asked, collect(&j, first, last)));
        }
        if published(&j, first * PAGE, last * PAGE) {
            LIGHT_ASKS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            j.asked += 1;
            return Some((j.asked, collect(&j, first, last)));
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
            return Some((j.ahead_asked, collect(&j, first, last)));
        }
        if guarded && light == 2 && published(&j, first * PAGE, last * PAGE) {
            drop(j);
            let mut j = journal_write();
            if j.covers_ahead(first, last, guarded) {
                return Some((j.ahead_asked, collect(&j, first, last)));
            }
            if published(&j, first * PAGE, last * PAGE) {
                LIGHT_ASKS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                j.asked += 1;
                return Some((j.asked, collect(&j, first, last)));
            }
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
    // The light refresh: the guards were asked, and what they found is
    // noted. The blocks they made clean are known by their bits from now
    // on (the journal's lock is still held: no other asking came between).
    if failed && light != 0 {
        LIGHT_ASKS[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        publish(&j, &dirty, first * PAGE, last * PAGE, told());
    }
    Some((now, collect(&j, first, last)))
}

#[cfg(test)]
fn writes_as(start: usize, end: usize, guarded: bool, light: u8) -> Option<(u64, Vec<(usize, u64)>)> {
    query_writes_as(start, end, guarded, light, |j, first, last| {
        j.written.range(first..last).map(|(&p, &w)| (p, w)).collect()
    })
}

/// The pages of `[start, end)` (whole pages) as last written, for pages
/// written since tracking began: `(page number, asking)` in page order, and
/// this asking's number, which is higher than any in the list but those the
/// kernel just reported. `None` if the memory is not write-watched (nor,
/// with `guarded`, view memory, which is then guarded).
pub(crate) fn writes(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    query_writes(start, end, guarded, |j, first, last| {
        j.written.range(first..last).map(|(&p, &w)| (p, w)).collect()
    })
}

/// Returns `(asked, unchanged)` for `[start, end)` without allocating:
/// `unchanged` is true iff `since != 0` and every written page in the range
/// was last written at or before `since`. Follows the same tracking semantics
/// and `guarded` / `known_view` rules as [`writes`].
pub(crate) fn writes_unchanged_since(
    start: usize,
    end: usize,
    since: u64,
    guarded: bool,
) -> Option<(u64, bool)> {
    query_writes(start, end, guarded, |j, first, last| {
        since != 0 && j.written.range(first..last).all(|(_, &w)| w <= since)
    })
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
            let idx = page.saturating_sub(self.base_page);
            if seen.get(idx).is_some_and(|&last| last >= write) {
                continue;
            }
            let (start, end) = ((page * PAGE).max(from), ((page + 1) * PAGE).min(to));
            if start >= end {
                continue;
            }
            // Keep partial pages dirty: the caller has not taken the rest.
            if (page * PAGE).max(self.start) >= from
                && ((page + 1) * PAGE).min(self.end) <= to
                && let Some(slot) = seen.get_mut(idx)
            {
                *slot = write;
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

/// Block numbers as bits, in chunks of 4096 blocks (256 MiB of address
/// space). The guards' checks look at every block of a range under the
/// guards' lock, which every copy's check waits for: a bit is several times
/// quicker to look at than a tree's node.
struct BlockSet {
    chunks: std::collections::HashMap<usize, Box<[u64; 64]>, std::hash::BuildHasherDefault<ChunkHash>>,
}

/// Chunk numbers are spread enough by a multiplication.
#[derive(Default)]
struct ChunkHash(u64);

impl std::hash::Hasher for ChunkHash {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
        }
    }
    fn write_usize(&mut self, n: usize) {
        self.0 = (n as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
}

impl BlockSet {
    const SHIFT: usize = 12;

    const fn new() -> Self {
        Self { chunks: std::collections::HashMap::with_hasher(std::hash::BuildHasherDefault::new()) }
    }

    fn at(block: usize) -> (usize, usize, u64) {
        let bit = block & ((1 << Self::SHIFT) - 1);
        (block >> Self::SHIFT, bit / 64, 1u64 << (bit % 64))
    }

    fn contains(&self, block: &usize) -> bool {
        let (chunk, word, bit) = Self::at(*block);
        self.chunks.get(&chunk).is_some_and(|c| c[word] & bit != 0)
    }

    /// Whether it was not in the set.
    fn insert(&mut self, block: usize) -> bool {
        let (chunk, word, bit) = Self::at(block);
        let c = self.chunks.entry(chunk).or_insert_with(|| Box::new([0; 64]));
        let new = c[word] & bit == 0;
        c[word] |= bit;
        new
    }

    /// Whether it was in the set.
    fn remove(&mut self, block: &usize) -> bool {
        let (chunk, word, bit) = Self::at(*block);
        let Some(c) = self.chunks.get_mut(&chunk) else { return false };
        let had = c[word] & bit != 0;
        c[word] &= !bit;
        had
    }

    /// Calls `f` for each `(word_index, mask)` of `chunk` overlapping `first..last`.
    fn chunk_words<F: FnMut(usize, u64) -> bool>(chunk: usize, first: usize, last: usize, mut f: F) -> bool {
        let base = chunk << Self::SHIFT;
        let c_start = first.saturating_sub(base);
        let c_end = (last - base).min(1 << Self::SHIFT);
        let w_first = c_start / 64;
        let w_last = (c_end - 1) / 64;
        for w in w_first..=w_last {
            let lo = if w == w_first { !0u64 << (c_start % 64) } else { !0u64 };
            let hi = if w == w_last && c_end % 64 != 0 {
                !0u64 >> (64 - (c_end % 64))
            } else {
                !0u64
            };
            if !f(w, lo & hi) {
                return false;
            }
        }
        true
    }

    /// Whether every block in `first..last` is in the set, without allocating.
    #[cfg(test)]
    fn all_in_range(&self, first: usize, last: usize) -> bool {
        if first >= last {
            return true;
        }
        let (first_chunk, last_chunk) = (first >> Self::SHIFT, (last - 1) >> Self::SHIFT);
        for chunk in first_chunk..=last_chunk {
            let Some(bits) = self.chunks.get(&chunk) else {
                return false;
            };
            if !Self::chunk_words(chunk, first, last, |w, mask| (bits[w] & mask) == mask) {
                return false;
            }
        }
        true
    }

    /// Number of blocks of `range` in the set, without allocating.
    fn count_range(&self, range: std::ops::Range<usize>) -> usize {
        if range.start >= range.end {
            return 0;
        }
        let (first_chunk, last_chunk) = (range.start >> Self::SHIFT, (range.end - 1) >> Self::SHIFT);
        let mut count = 0;
        let mut count_chunk = |chunk: usize, bits: &[u64; 64]| {
            Self::chunk_words(chunk, range.start, range.end, |w, mask| {
                count += (bits[w] & mask).count_ones() as usize;
                true
            });
        };
        if last_chunk - first_chunk < self.chunks.len() {
            for chunk in first_chunk..=last_chunk {
                if let Some(bits) = self.chunks.get(&chunk) {
                    count_chunk(chunk, bits);
                }
            }
        } else {
            for (&chunk, bits) in &self.chunks {
                if (first_chunk..=last_chunk).contains(&chunk) {
                    count_chunk(chunk, bits);
                }
            }
        }
        count
    }

    /// Whether every block in `first..last` is in `self | other`, without allocating.
    fn all_covered_by_two(&self, other: &BlockSet, first: usize, last: usize) -> bool {
        if first >= last {
            return true;
        }
        let (first_chunk, last_chunk) = (first >> Self::SHIFT, (last - 1) >> Self::SHIFT);
        for chunk in first_chunk..=last_chunk {
            let a = self.chunks.get(&chunk);
            let b = other.chunks.get(&chunk);
            if a.is_none() && b.is_none() {
                return false;
            }
            if !Self::chunk_words(chunk, first, last, |w, mask| {
                let covered = a.map_or(0, |c| c[w]) | b.map_or(0, |c| c[w]);
                (covered & mask) == mask
            }) {
                return false;
            }
        }
        true
    }

    /// Calls `f(block)` for every block in `first..last` in neither `self` nor `other`,
    /// in ascending order, without allocating.
    fn for_each_missing_in_two(&self, other: &BlockSet, first: usize, last: usize, mut f: impl FnMut(usize)) {
        if first >= last {
            return;
        }
        let (first_chunk, last_chunk) = (first >> Self::SHIFT, (last - 1) >> Self::SHIFT);
        for chunk in first_chunk..=last_chunk {
            let a = self.chunks.get(&chunk);
            let b = other.chunks.get(&chunk);
            let base = chunk << Self::SHIFT;
            Self::chunk_words(chunk, first, last, |w, mask| {
                let covered = a.map_or(0, |c| c[w]) | b.map_or(0, |c| c[w]);
                let mut missing = (!covered) & mask;
                while missing != 0 {
                    let bit = missing.trailing_zeros() as usize;
                    missing &= missing - 1;
                    f(base + w * 64 + bit);
                }
                true
            });
        }
    }

    /// The blocks of `range` in the set, in order.
    fn range(&self, range: std::ops::Range<usize>) -> std::vec::IntoIter<usize> {
        let mut found = Vec::new();
        if range.start < range.end {
            let (first, last) = (range.start >> Self::SHIFT, (range.end - 1) >> Self::SHIFT);
            let mut chunks: Vec<usize> = if last - first < self.chunks.len() {
                (first..=last).filter(|c| self.chunks.contains_key(c)).collect()
            } else {
                let mut all: Vec<usize> =
                    self.chunks.keys().copied().filter(|c| (first..=last).contains(c)).collect();
                all.sort_unstable();
                all
            };
            chunks.dedup();
            for chunk in chunks {
                let bits = &self.chunks[&chunk];
                for (w, &word) in bits.iter().enumerate() {
                    let mut word = word;
                    while word != 0 {
                        let block = (chunk << Self::SHIFT) + w * 64 + word.trailing_zeros() as usize;
                        word &= word - 1;
                        if range.contains(&block) {
                            found.push(block);
                        }
                    }
                }
            }
        }
        found.into_iter()
    }
}

impl Extend<usize> for BlockSet {
    fn extend<I: IntoIterator<Item = usize>>(&mut self, blocks: I) {
        for block in blocks {
            self.insert(block);
        }
    }
}

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

type BlockMap<V> =
    std::collections::HashMap<usize, V, std::hash::BuildHasherDefault<ChunkHash>>;

fn known_view_in(g: &Guards, start: usize) -> bool {
    let first = start / BLOCK;
    g.protected.contains(&first) || g.absent.contains(&first) || g.absent_static.contains(&first)
}

/// Whether the block holding `start` is view memory the guards already
/// hold (read-only by our hand, or noted absent), or a registered runtime
/// physical-page view: asking the kernel's write watch about it can only fail.
/// `VICTORY_VIEW_ASK=1` (or `VICTORY_GUARD_ASK=1`) asks the kernel anyway, as before.
fn known_view(start: usize) -> bool {
    if force_view_ask() {
        return false;
    }
    let first = start / BLOCK;
    // The light refresh: a block known by its bit is read-only by our hand
    // or a hole the guards noted, which is what they are asked below.
    if light() && PUBLISHED.all(first, first + 1) {
        return true;
    }
    known_view_in(&guards(), start) || crate::inplace::is_view_at(start)
}

/// Whether repeated VEH write faults temporarily pin hot view blocks writable
/// (`VICTORY_HOT_GUARD=0` disables).
fn hot_guard_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HOT_GUARD").is_none_or(|v| v != "0"))
}

const HOT_FAULT_WINDOW: u64 = 32;
const HOT_FAULT_THRESHOLD: u16 = 2;
const HOT_PIN_EPOCHS: u64 = 128;

static HOT_GUARD_STATS: [std::sync::atomic::AtomicU64; 2] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 2];

pub(crate) fn hot_guard_stats() -> [u64; 2] {
    use std::sync::atomic::Ordering::Relaxed;
    [
        HOT_GUARD_STATS[0].swap(0, Relaxed),
        HOT_GUARD_STATS[1].swap(0, Relaxed),
    ]
}

struct Guards {
    /// Blocks we made read-only.
    protected: BlockSet,
    /// Of those, the ones not written since a copy took them.
    clean: BlockSet,
    /// Uncommitted `MEM_RESERVE | MEM_MAPPED` view blocks whose ordinary commit
    /// can arrive without a runtime remapping notification.
    absent: BlockSet,
    /// Uncommitted `MEM_FREE` or non-`MEM_MAPPED` placeholder blocks that remain
    /// uncommitted until `title_memory_remapped`.
    absent_static: BlockSet,
    /// When `absent` blocks were last found uncommitted (milliseconds of
    /// [`guard_ms`]), for `VICTORY_ABSENT_ONCE`.
    verified: BTreeMap<usize, u64>,
    /// With the light refresh and notes kept for good ([`told`]): the
    /// blocks of `verified` / `absent_static` that [`publish`] has looked
    /// at. Those of them that lie in a placeholder reservation
    /// ([`placeholder`]) are known by their bits ([`PUBLISHED`]) until the
    /// runtime tells of them; the others are answered from their notes,
    /// which then grow old ([`note_life`]).
    holes: BTreeSet<usize>,
    /// Blocks whose physical page was guarded together with its other views
    /// (those views' blocks are in it too): a write to one opens them all.
    /// A write range can be huge (`fread`), so only these are looked up.
    aliased: BTreeSet<usize>,
    /// Blocks the kernel is about to write for the runtime ([`writing`]),
    /// from, to, and until when they are left writable: an asynchronous read
    /// can land after the call returns, and the kernel fails rather than
    /// faulting into our handler. Until expiry they stay writable and every
    /// check reports them dirty.
    pins: Vec<(usize, usize, std::time::Instant)>,
    /// Incremented once per `guard_view_pieces` batch or standalone `guard()` check.
    guard_epoch: u64,
    /// Per-block `(last_epoch, consecutive_veh_faults)` for hot-block detection.
    fault_counts: BlockMap<(u64, u16)>,
    /// Blocks temporarily pinned writable until `guard_epoch` reaches the stored limit.
    hot_pins: BlockMap<u64>,
}

/// How long blocks stay writable after [`writing`].
const PIN: std::time::Duration = std::time::Duration::from_secs(2);

impl Guards {
    /// Whether `block` is left writable for an in-flight kernel write.
    fn pinned(&self, block: usize) -> bool {
        self.pins.iter().any(|&(from, to, _)| from <= block && block < to)
    }

    /// Removes `[first, last)` from `pins`, splitting any pin that extends
    /// beyond it (called on remap/unmap so reused virtual addresses never
    /// inherit stale pins).
    fn unpin(&mut self, first: usize, last: usize) {
        if first >= last || self.pins.is_empty() {
            return;
        }
        let mut next = Vec::with_capacity(self.pins.len());
        for &(from, to, until) in &self.pins {
            if to <= first || from >= last {
                next.push((from, to, until));
            } else {
                if from < first {
                    next.push((from, first, until));
                }
                if to > last {
                    next.push((last, to, until));
                }
            }
        }
        self.pins = next;
    }
}

static GUARDS: Mutex<Guards> = Mutex::new(Guards {
    protected: BlockSet::new(),
    clean: BlockSet::new(),
    absent: BlockSet::new(),
    absent_static: BlockSet::new(),
    verified: BTreeMap::new(),
    holes: BTreeSet::new(),
    aliased: BTreeSet::new(),
    pins: Vec::new(),
    guard_epoch: 0,
    fault_counts: std::collections::HashMap::with_hasher(std::hash::BuildHasherDefault::new()),
    hot_pins: std::collections::HashMap::with_hasher(std::hash::BuildHasherDefault::new()),
});

fn guards() -> std::sync::MutexGuard<'static, Guards> {
    GUARDS.lock().unwrap_or_else(|e| e.into_inner())
}

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded (all of a block's pages), guarding them again
/// before the caller copies them. `None` if the memory is not a view.
fn guard(start: usize, end: usize) -> Option<Vec<usize>> {
    guard_inner(start, end, true).map(|(written, _)| written)
}

/// Same as [`guard`], also returning whether any block in the range was kept
/// writable (`pinned` or `hot_pins`) and advancing `guard_epoch` once when
/// `bump_epoch` is true.
fn guard_inner(start: usize, end: usize, bump_epoch: bool) -> Option<(Vec<usize>, bool)> {
    let keep = if absent_once() { absent_keep_ms() } else { 0 };
    let now = if keep != 0 { guard_ms() } else { 0 };
    if bump_epoch && absent_mode() == 1 {
        static CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if (CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1).is_multiple_of(65_536) {
            let [asked, kept] = absent_looks();
            tracing::info!(
                "blocks of view memory found uncommitted are asked of Windows again only after {} ms (VICTORY_ABSENT_ONCE=1): in the last 65536 askings of the guards {asked} questions to Windows about such blocks, {kept} blocks answered from their note",
                absent_keep_ms()
            );
        }
    }
    guard_at(start, end, bump_epoch, now, keep)
}

fn guard_at(start: usize, end: usize, bump_epoch: bool, now: u64, keep: u64) -> Option<(Vec<usize>, bool)> {
    install();
    static ASK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let ask = *ASK.get_or_init(|| std::env::var_os("VICTORY_GUARD_ASK").is_some_and(|v| v == "1"));
    let mut g = guards();
    let known = !ask && known_view_in(&g, start);
    if !known {
        drop(g);
        if !crate::inplace::is_view_at(start) && !is_view(start, end) {
            return None;
        }
        g = guards();
    }
    if bump_epoch {
        g.guard_epoch = g.guard_epoch.wrapping_add(1);
    }
    if !g.pins.is_empty() {
        let now = std::time::Instant::now();
        g.pins.retain(|&(_, _, until)| until > now);
    }
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    // Fast zero-alloc check: blocks in `clean | absent_static` require no inspection.
    if g.clean.all_covered_by_two(&g.absent_static, first, last) {
        return Some((Vec::new(), false));
    }
    thread_local! {
        static MISSING: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    MISSING.with(|cell| {
        let mut missing = cell.borrow_mut();
        missing.clear();
        g.clean
            .for_each_missing_in_two(&g.absent_static, first, last, |b| missing.push(b));
        let mut written = Vec::new();
        let mut run: Option<(usize, usize)> = None;
        let mut any_pinned = false;
        let mut uncommitted_until = first;
        let epoch = g.guard_epoch;
        let hot_on = hot_guard_on();
        for &block in missing.iter() {
            let mut open = true;
            if g.absent.contains(&block) {
                // (With the light refresh, kept for good only in a placeholder
                // reservation: [`note_life`].)
                let life = note_life(&g, block, keep);
                let noted = life != 0 && g.verified.get(&block).is_some_and(|&at| now < at.saturating_add(life));
                if noted {
                    ABSENT_LOOKS[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    open = false;
                } else {
                    if block >= uncommitted_until {
                        ABSENT_LOOKS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        uncommitted_until = uncommitted_info(block).0;
                    }
                    open = block >= uncommitted_until;
                    if open {
                        g.absent.remove(&block);
                        g.verified.remove(&block);
                        forget_hole(&mut g, block);
                    } else if keep != 0 {
                        g.verified.insert(block, now);
                    }
                }
            }
            if !open {
                if let Some((from, to)) = run.take()
                    && protect(&mut g, from, to)
                {
                    any_pinned = true;
                }
                continue;
            }
            let is_hot = if hot_on {
                match g.hot_pins.get(&block).copied() {
                    Some(until) if until > epoch => true,
                    Some(_) => {
                        g.hot_pins.remove(&block);
                        false
                    }
                    None => false,
                }
            } else {
                false
            };
            written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
            if g.pinned(block) || is_hot {
                any_pinned = true;
                if is_hot {
                    HOT_GUARD_STATS[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                if let Some((from, to)) = run.take()
                    && protect(&mut g, from, to)
                {
                    any_pinned = true;
                }
            } else if g.protected.contains(&block) {
                if let Some((from, to)) = run.take()
                    && protect(&mut g, from, to)
                {
                    any_pinned = true;
                }
                // Written through the runtime, still read-only.
                g.clean.insert(block);
            } else {
                match &mut run {
                    Some((_, to)) if *to == block => *to = block + 1,
                    Some(_) => {
                        let (from, to) = run.replace((block, block + 1)).unwrap();
                        if protect(&mut g, from, to) {
                            any_pinned = true;
                        }
                    }
                    None => run = Some((block, block + 1)),
                }
            }
        }
        if let Some((from, to)) = run.take()
            && protect(&mut g, from, to)
        {
            any_pinned = true;
        }
        Some((written, any_pinned))
    })
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
///
/// Returns `true` if any block in `[from, to)` has a mapped alias that could
/// not be guarded; those blocks are opened and left dirty so writes through
/// the unguarded alias are never missed.
fn protect(g: &mut Guards, from: usize, to: usize) -> bool {
    protect_here(g, from, to);
    if !view_guard() {
        return false;
    }
    let mut last_blk = usize::MAX;
    let mut unguarded = Vec::new();
    crate::inplace::for_each_alias_in_range(
        from.saturating_mul(BLOCK),
        to.saturating_mul(BLOCK),
        MANY_VIEWS,
        |block_addr, alias| {
            let block = block_addr / BLOCK;
            if block != last_blk {
                last_blk = block;
                #[cfg(feature = "diagnostics")]
                ALIASES[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                g.aliased.insert(block);
            }
            let alias = alias / BLOCK;
            g.aliased.insert(alias);
            if !protect_alias(g, alias) && unguarded.last().copied() != Some(block) {
                unguarded.push(block);
            }
        },
    );
    if unguarded.is_empty() {
        return false;
    }
    for block in unguarded {
        open_here(g, block * BLOCK, (block + 1) * BLOCK);
    }
    true
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
///
/// Returns `false` when `block` is mapped/writable and could not be guarded.
fn protect_alias(g: &mut Guards, block: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    if g.pinned(block)
        || (hot_guard_on()
            && g.hot_pins
                .get(&block)
                .is_some_and(|&until| until > g.guard_epoch))
    {
        return false;
    }
    if g.protected.contains(&block) {
        return true;
    }
    let mut old = 0;
    // SAFETY: title view memory; only its protection changes.
    if unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READONLY, &mut old) } != 0 {
        g.protected.insert(block);
        true
    } else {
        g.absent_static.contains(&block) || uncommitted_info(block).0 > block
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
        } else {
            let (end_block, is_mapped_reserve) = uncommitted_info(block);
            if block < end_block {
                if is_mapped_reserve {
                    g.absent.insert(block);
                } else {
                    g.absent_static.insert(block);
                }
            }
        }
    }
}

/// The end (exclusive) of the whole uncommitted blocks starting at `block`,
/// and whether the range is a `MEM_RESERVE | MEM_MAPPED` view that may commit
/// without a runtime remapping notification.
fn uncommitted_info(block: usize) -> (usize, bool) {
    use windows_sys::Win32::System::Memory::{
        MEM_FREE, MEM_MAPPED, MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery((block * BLOCK) as *const _, &mut info, size_of_val(&info)) };
    if n != 0 && matches!(info.State, MEM_RESERVE | MEM_FREE) {
        let end_block = (info.BaseAddress as usize).saturating_add(info.RegionSize) / BLOCK;
        let is_mapped_reserve = info.State == MEM_RESERVE && info.Type == MEM_MAPPED;
        (end_block, is_mapped_reserve)
    } else {
        (block, false)
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
    // The light refresh: no view of these pages is known by its bit once
    // one of them can be written (who asks the bits takes no lock of ours;
    // after avatarmanz/VIctory-7900XTX 689df313).
    if light_kept() && start < end && !g.aliased.is_empty() {
        for &block in g.aliased.range(start / BLOCK..end.div_ceil(BLOCK)) {
            for alias in crate::inplace::pages_of(block * BLOCK) {
                PUBLISHED.clear(alias / BLOCK);
            }
        }
    }
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
        .collect();
    for &block in &blocks {
        g.protected.remove(&block);
        g.clean.remove(&block);
        // Before the block can be written ([`PUBLISHED`]; nothing to do
        // where no bit was ever set).
        PUBLISHED.clear(block);
        let mut old = 0;
        // SAFETY: a block we made read-only.
        unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old) };
    }
    !blocks.is_empty()
}

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler): opens its guarded blocks
/// and pins them writable for [`PIN`] (including other views of the same
/// physical pages) so asynchronous kernel reads cannot fail against a guard.
/// The runtime calls it (as `kernel_will_write`) before the title's kernel
/// reads into its own memory.
pub fn writing(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    // The length is the title's (`fread`'s size times count, saturated).
    let end = at.saturating_add(len);
    let (first, last) = (at / BLOCK, end.div_ceil(BLOCK));
    let until = std::time::Instant::now() + PIN;
    let aliases = {
        let mut g = guards();
        if !g.pins.is_empty() {
            let now = std::time::Instant::now();
            g.pins.retain(|&(_, _, t)| t > now);
        }
        g.pins.push((first, last, until));
        let (_, mut aliases) = open_collecting(&mut g, at, end);
        if view_guard() {
            crate::inplace::for_each_alias_in_range(
                first.saturating_mul(BLOCK),
                last.saturating_mul(BLOCK),
                MANY_VIEWS,
                |_, alias| {
                    let alias_block = alias / BLOCK;
                    g.pins.push((alias_block, alias_block + 1, until));
                    PUBLISHED.clear(alias_block);
                    open_here(&mut g, alias, alias.saturating_add(BLOCK));
                    aliases.push(alias_block);
                },
            );
        }
        GUARD_REV.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        aliases
    };
    evict_ahead_guarded(first, last, &aliases);
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
        let guarded = g.protected.count_range(at / BLOCK..end.div_ceil(BLOCK)) as u64;
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
        // The light refresh: all of them told as written before the first can
        // be written (as in [`open`]).
        if light_kept() {
            for &(at, n) in &ranges {
                for block in g.protected.range(at / BLOCK..at.saturating_add(n).div_ceil(BLOCK)) {
                    PUBLISHED.clear(block);
                }
            }
        }
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
struct Mapping {
    changing: Option<std::ops::Range<usize>>,
    readers: Vec<std::ops::Range<usize>>,
}

static MAPPING: Mutex<Mapping> = Mutex::new(Mapping {
    changing: None,
    readers: Vec::new(),
});
static WRITER: Mutex<()> = Mutex::new(());
static MAPPING_CHANGED: std::sync::Condvar = std::sync::Condvar::new();
/// Non-zero while any `title_memory_remapping*` guard is active or entering.
static REMAPPING_ACTIVE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// Active lock-free fast-copy / fast-diff readers.
static FAST_READERS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// `[0]` fast copy succeeded, `[1]` yielded to active remap, `[2]` faulted on unmapped/guarded page.
static FAST_COPY_STATS: [std::sync::atomic::AtomicU64; 3] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 3];
const ALL_MEMORY: std::ops::Range<usize> = 0..usize::MAX;

pub(crate) fn fast_copy_stats() -> [u64; 3] {
    use std::sync::atomic::Ordering::Relaxed;
    [
        FAST_COPY_STATS[0].swap(0, Relaxed),
        FAST_COPY_STATS[1].swap(0, Relaxed),
        FAST_COPY_STATS[2].swap(0, Relaxed),
    ]
}

fn overlaps(a: &std::ops::Range<usize>, b: &std::ops::Range<usize>) -> bool {
    a.start.max(b.start) < a.end.min(b.end)
}

impl Mapping {
    fn changing(&self, range: &std::ops::Range<usize>) -> bool {
        self.changing.as_ref().is_some_and(|w| overlaps(w, range))
    }

    fn pin(&mut self, range: std::ops::Range<usize>) -> Reading {
        self.readers.push(range.clone());
        READING_RANGES.with(|r| r.borrow_mut().push(range.clone()));
        Reading(range, std::marker::PhantomData)
    }
}

/// What the runtime holds while it changes what title memory shows
/// ([`title_memory_remapping`]).
pub struct Remapping(#[allow(dead_code)] std::sync::MutexGuard<'static, ()>);

impl Drop for Remapping {
    fn drop(&mut self) {
        let mut state = MAPPING.lock().unwrap_or_else(|e| e.into_inner());
        state.changing = None;
        REMAPPING_ACTIVE.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        MAPPING_CHANGED.notify_all();
    }
}

/// The runtime is about to map, unmap or move physical pages of title
/// memory: until what it returns is dropped, no read of title memory in
/// place is under way and none begins ([`title_memory_reading`]). Taken
/// around the whole change, the unmapping and the mapping again.
pub fn title_memory_remapping() -> Remapping {
    title_memory_remapping_range(ALL_MEMORY)
}

/// Like [`title_memory_remapping`], scoped to `range`: reads of disjoint
/// ranges continue without waiting, while reads overlapping `range` wait
/// until the returned guard is dropped.
pub fn title_memory_remapping_range(range: std::ops::Range<usize>) -> Remapping {
    let writer = WRITER.lock().unwrap_or_else(|e| e.into_inner());
    REMAPPING_ACTIVE.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let mut state = MAPPING.lock().unwrap_or_else(|e| e.into_inner());
    state.changing = Some(range.clone());
    let state = MAPPING_CHANGED
        .wait_while(state, |s| {
            FAST_READERS.load(std::sync::atomic::Ordering::SeqCst) > 0
                || s.readers.iter().any(|r| overlaps(r, &range))
        })
        .unwrap_or_else(|e| e.into_inner());
    drop(state);
    Remapping(writer)
}

thread_local! {
    /// Active [`Reading`] ranges held by the current thread, so fault-safe
    /// fast reads nested inside an outer [`title_memory_reading`] (e.g.
    /// texture refreshes) skip a recursive lock acquisition.
    static READING_RANGES: std::cell::RefCell<Vec<std::ops::Range<usize>>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// What a read of title memory in place holds ([`title_memory_reading`]).
pub struct Reading(std::ops::Range<usize>, std::marker::PhantomData<*const ()>);

impl Drop for Reading {
    fn drop(&mut self) {
        READING_RANGES.with(|r| {
            let mut ranges = r.borrow_mut();
            if let Some(i) = ranges.iter().rposition(|range| *range == self.0) {
                ranges.swap_remove(i);
            }
        });
        let mut state = MAPPING.lock().unwrap_or_else(|e| e.into_inner());
        let i = state
            .readers
            .iter()
            .position(|r| *r == self.0)
            .expect("active mapping reader");
        state.readers.swap_remove(i);
        if state.changing(&self.0) {
            MAPPING_CHANGED.notify_all();
        }
    }
}

/// For reading title memory in place: until what it returns is dropped,
/// the runtime changes no mapping, so memory found mapped stays mapped.
/// Not to be taken twice by one thread, nor held across anything that
/// maps title memory.
pub fn title_memory_reading() -> Reading {
    title_memory_reading_range(ALL_MEMORY)
}

/// Like [`title_memory_reading`], scoped to `range`: remappings of disjoint
/// ranges continue without waiting, while remappings overlapping `range`
/// wait until the returned guard is dropped.
pub fn title_memory_reading_range(range: std::ops::Range<usize>) -> Reading {
    let state = MAPPING.lock().unwrap_or_else(|e| e.into_inner());
    MAPPING_CHANGED
        .wait_while(state, |s| s.changing(&range))
        .unwrap_or_else(|e| e.into_inner())
        .pin(range)
}

/// [`title_memory_reading`] if it can be had at once: `None` while the
/// runtime changes a mapping (or waits to), for a caller that has another
/// way to read and would rather not wait.
#[allow(dead_code)]
pub fn title_memory_reading_now() -> Option<Reading> {
    title_memory_reading_now_range(ALL_MEMORY)
}

/// [`title_memory_reading_range`] if it can be had at once: `None` while
/// the runtime changes an overlapping mapping (or `MAPPING` is contended).
pub(crate) fn title_memory_reading_now_range(range: std::ops::Range<usize>) -> Option<Reading> {
    let mut state = match MAPPING.try_lock() {
        Ok(state) => state,
        Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
        Err(std::sync::TryLockError::WouldBlock) => return None,
    };
    (!state.changing(&range)).then(|| state.pin(range))
}

#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
enum FastReadGuard {
    Nested,
    Fast,
    Scoped(#[allow(dead_code)] Reading),
}

impl Drop for FastReadGuard {
    fn drop(&mut self) {
        if matches!(self, FastReadGuard::Fast)
            && FAST_READERS.fetch_sub(1, std::sync::atomic::Ordering::SeqCst) == 1
            && REMAPPING_ACTIVE.load(std::sync::atomic::Ordering::SeqCst) != 0
        {
            let _m = MAPPING.lock().unwrap_or_else(|e| e.into_inner());
            MAPPING_CHANGED.notify_all();
        }
    }
}

/// Non-blocking read guard against concurrent title-memory remapping, safe
/// even if this thread already holds an outer [`Reading`] covering `range`.
/// Uses lock-free atomic refcounting when no remapping is active.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
fn acquire_reading_now(range: std::ops::Range<usize>) -> Option<FastReadGuard> {
    if READING_RANGES.with(|r| {
        r.borrow()
            .iter()
            .any(|held| held.start <= range.start && range.end <= held.end)
    }) {
        return Some(FastReadGuard::Nested);
    }
    if REMAPPING_ACTIVE.load(std::sync::atomic::Ordering::Acquire) == 0 {
        FAST_READERS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if REMAPPING_ACTIVE.load(std::sync::atomic::Ordering::SeqCst) == 0 {
            return Some(FastReadGuard::Fast);
        }
        if FAST_READERS.fetch_sub(1, std::sync::atomic::Ordering::SeqCst) == 1 {
            let _m = MAPPING.lock().unwrap_or_else(|e| e.into_inner());
            MAPPING_CHANGED.notify_all();
        }
    }
    title_memory_reading_now_range(range).map(FastReadGuard::Scoped)
}

/// The runtime committed title memory `[at, at + len)` for the title
/// (`XMemVirtualAlloc` with a commit): what [`absent_once`] noted about its
/// blocks is forgotten, so the next asking puts the question to Windows
/// again, as every asking did before the notes. With `VICTORY_ABSENT_MS=told`
/// this and [`title_memory_remapped`] are the only ways a block noted
/// uncommitted is asked about again.
pub fn title_memory_committed(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let (first, last) = (at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
    let evicted = {
        let mut g = guards();
        let mut changed = false;
        if absent_mode() != 0 && !g.verified.is_empty() {
            let noted: Vec<usize> = g.verified.range(first..last).map(|(&block, _)| block).collect();
            for block in noted {
                g.verified.remove(&block);
                forget_hole(&mut g, block);
                changed = true;
            }
        }
        if !g.absent_static.chunks.is_empty() {
            let static_blocks: Vec<usize> = g.absent_static.range(first..last).collect();
            for block in static_blocks {
                g.absent_static.remove(&block);
                forget_hole(&mut g, block);
                changed = true;
            }
        }
        if !g.holes.is_empty() {
            let holes: Vec<usize> = g.holes.range(first..last).copied().collect();
            for block in holes {
                forget_hole(&mut g, block);
                changed = true;
            }
        }
        if changed {
            GUARD_REV.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
        changed
    };
    if evicted {
        evict_ahead_guarded(at / PAGE, at.saturating_add(len).div_ceil(PAGE), &[]);
    }
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new and read-write, so it is no longer guarded (nor known
/// unmapped).
pub fn title_memory_remapped(at: usize, len: usize) {
    crate::texback::remapped(at, len);
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
        let absent: Vec<usize> = g.absent.range(range.clone()).collect();
        for block in absent {
            g.absent.remove(&block);
            forget_hole(&mut g, block);
        }
        let absent_static: Vec<usize> = g.absent_static.range(range.clone()).collect();
        for block in absent_static {
            g.absent_static.remove(&block);
            forget_hole(&mut g, block);
        }
        let verified: Vec<usize> = g.verified.range(range.clone()).map(|(&block, _)| block).collect();
        for block in verified {
            g.verified.remove(&block);
            forget_hole(&mut g, block);
        }
        let holes: Vec<usize> = g.holes.range(range.clone()).copied().collect();
        for block in holes {
            forget_hole(&mut g, block);
        }
        let blocks: Vec<usize> = g.protected.range(range.clone()).collect();
        for block in blocks {
            g.protected.remove(&block);
            g.clean.remove(&block);
            // What it shows is new ([`PUBLISHED`]).
            PUBLISHED.clear(block);
            // A view mapped moments ago may have already been protected as an
            // alias of a guarded page between `views_mapped` and this call:
            // give write access back (fails harmlessly on unmapped memory).
            let mut old = 0;
            // SAFETY: only the protection of title view memory changes.
            unsafe {
                windows_sys::Win32::System::Memory::VirtualProtect(
                    (block * BLOCK) as *const _,
                    BLOCK,
                    windows_sys::Win32::System::Memory::PAGE_READWRITE,
                    &mut old,
                )
            };
        }
        let aliased: Vec<usize> = g.aliased.range(range.clone()).copied().collect();
        for block in aliased {
            g.aliased.remove(&block);
        }
        if !g.hot_pins.is_empty() {
            g.hot_pins.retain(|&b, _| !(range.start..range.end).contains(&b));
        }
        if !g.fault_counts.is_empty() {
            g.fault_counts.retain(|&b, _| !(range.start..range.end).contains(&b));
        }
        g.unpin(range.start, range.end);
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
            HOT_GUARD_STATS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if hot_guard_on() {
                let block = at / BLOCK;
                let epoch = g.guard_epoch;
                let entry = g.fault_counts.entry(block).or_insert((epoch, 0));
                if epoch.saturating_sub(entry.0) <= HOT_FAULT_WINDOW {
                    entry.1 = entry.1.saturating_add(1);
                } else {
                    entry.1 = 1;
                }
                entry.0 = epoch;
                if entry.1 >= HOT_FAULT_THRESHOLD {
                    entry.1 = 0;
                    g.hot_pins.insert(block, epoch.saturating_add(HOT_PIN_EPOCHS));
                }
                if g.fault_counts.len() > 16384 {
                    g.fault_counts
                        .retain(|_, &mut (e, _)| epoch.saturating_sub(e) <= HOT_FAULT_WINDOW * 2);
                    g.hot_pins.retain(|_, &mut until| until > epoch);
                }
            }
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if writable(at) {
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        // Read-only, in a physical-page view we know of, and not noted
        // guarded: only we make title views read-only, so restore write access.
        if view_guard() && !crate::inplace::pages_of(at - at % BLOCK).is_empty() {
            let block = at - at % BLOCK;
            let mut old = 0;
            // SAFETY: only the protection of title view memory changes.
            let opened = unsafe {
                windows_sys::Win32::System::Memory::VirtualProtect(
                    block as *const _,
                    BLOCK,
                    windows_sys::Win32::System::Memory::PAGE_READWRITE,
                    &mut old,
                )
            } != 0;
            if opened {
                GUARD_REV.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                UNTRACKED_VIEW_FAULTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return EXCEPTION_CONTINUE_EXECUTION;
            }
        }
        EXCEPTION_CONTINUE_SEARCH
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // The switch is read here, not first in the handler ([`open`] asks
    // it): reading the environment allocates.
    light_kept();
    // SAFETY: the handler only touches our guard state, TLS recovery slot, and
    // page protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

/// Count of write faults on read-only title view blocks not tracked in
/// `Guards::protected` that the vectored handler restored to read-write.
static UNTRACKED_VIEW_FAULTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

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
    let start = src as usize;
    let Some(_reading) = acquire_reading_now(start..start + len) else {
        FAST_COPY_STATS[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
    if remaining == 0 {
        FAST_COPY_STATS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        true
    } else {
        FAST_COPY_STATS[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        false
    }
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
    let start = src as usize;
    let _reading = acquire_reading_now(start..start + len)?;
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

    /// The bit set answers as a tree does, ranges across chunks included.
    #[test]
    fn block_set_matches_a_tree() {
        let mut set = BlockSet::new();
        let mut tree = BTreeSet::new();
        let mut x = 12345u64;
        for _ in 0..20_000 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let block = ((x >> 33) % 30_000) as usize + if x & 1 == 0 { 0 } else { 1 << 30 };
            if x & 6 == 0 {
                assert_eq!(set.remove(&block), tree.remove(&block));
            } else {
                assert_eq!(set.insert(block), tree.insert(block));
            }
            assert_eq!(set.contains(&block), tree.contains(&block));
        }
        set.extend(5000..9000);
        tree.extend(5000..9000);
        for r in [0..30_000, 4095..4097, 8191..12_289, (1 << 30)..(1 << 30) + 30_000, 0..usize::MAX / 2] {
            let ours: Vec<usize> = set.range(r.clone()).collect();
            let theirs: Vec<usize> = tree.range(r).copied().collect();
            assert_eq!(ours, theirs);
        }
    }

    /// `all_in_range` and `count_range` match `BTreeSet` across empty,
    /// single-bit, sub-word, multi-word, cross-chunk, and large sparse ranges.
    #[test]
    fn block_set_all_in_range_and_count_range_match_tree() {
        let check_against_tree = |set: &BlockSet, tree: &BTreeSet<usize>, ranges: &[std::ops::Range<usize>]| {
            for r in ranges {
                let expected_count = if r.start < r.end {
                    tree.range(r.clone()).count()
                } else {
                    0
                };
                let expected_all = r.start >= r.end || expected_count == (r.end - r.start);
                assert_eq!(set.count_range(r.clone()), expected_count, "count_range mismatch for {r:?}");
                assert_eq!(
                    set.all_in_range(r.start, r.end),
                    expected_all,
                    "all_in_range mismatch for {r:?}"
                );
            }
        };

        // 1. Contiguous filled ranges: full and with 1 bit removed at start, middle, or end.
        let filled_ranges = [
            0..1,
            63..64,
            64..65,
            4095..4096,
            4096..4097,
            1..62,
            65..127,
            4097..4150,
            0..130,
            60..200,
            4000..4096,
            4095..4097,
            8191..12_289,
            (1 << 30) - 5..(1 << 30) + 130,
        ];
        #[allow(clippy::reversed_empty_ranges)]
        let empty_and_edge_probes = [
            0..0,
            10..10,
            5000..4000,
            0..1,
            63..64,
            64..65,
            4095..4096,
            4096..4097,
            1..62,
            65..127,
            4097..4150,
            0..130,
            60..200,
            4000..4096,
            4095..4097,
            8191..12_289,
            (1 << 30) - 5..(1 << 30) + 130,
            0..usize::MAX / 2,
        ];

        // Empty set check.
        check_against_tree(&BlockSet::new(), &BTreeSet::new(), &empty_and_edge_probes);

        for target in filled_ranges {
            let mut set = BlockSet::new();
            let mut tree = BTreeSet::new();
            set.extend(target.clone());
            tree.extend(target.clone());
            assert!(set.all_in_range(target.start, target.end));
            assert_eq!(set.count_range(target.clone()), target.end - target.start);
            check_against_tree(&set, &tree, &empty_and_edge_probes);

            let gaps = [
                target.start,
                target.start + (target.end - target.start) / 2,
                target.end - 1,
            ];
            for gap in gaps {
                let mut g_set = BlockSet::new();
                let mut g_tree = BTreeSet::new();
                g_set.extend(target.clone());
                g_tree.extend(target.clone());
                assert!(g_set.remove(&gap));
                assert!(g_tree.remove(&gap));
                assert!(!g_set.all_in_range(target.start, target.end));
                assert_eq!(g_set.count_range(target.clone()), target.end - target.start - 1);
                check_against_tree(&g_set, &g_tree, &empty_and_edge_probes);
            }
        }

        // 2. Mixed random + dense chunks against BTreeSet.
        let mut set = BlockSet::new();
        let mut tree = BTreeSet::new();
        let mut x = 67890u64;
        for _ in 0..20_000 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let block = ((x >> 33) % 30_000) as usize + if x & 1 == 0 { 0 } else { 1 << 30 };
            if x & 6 == 0 {
                set.remove(&block);
                tree.remove(&block);
            } else {
                set.insert(block);
                tree.insert(block);
            }
        }
        set.extend(4090..12_300);
        tree.extend(4090..12_300);
        check_against_tree(&set, &tree, &empty_and_edge_probes);
    }
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
    pub(super) fn two_views(bytes: usize) -> (usize, usize, usize) {
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
    pub(super) fn drop_views(section: usize, views: &[usize], bytes: usize) {
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
        // Block 1 stays pinned writable & dirty across views until the kernel
        // write pin expires or is released.
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        guards().unpin(guarded / BLOCK + 1, guarded / BLOCK + 2);
        guards().unpin(other / BLOCK + 1, other / BLOCK + 2);
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

    #[test]
    fn unguarded_inplace_view_skips_kernel_asks_in_partition_and_known_view() {
        use std::sync::atomic::Ordering::Relaxed;
        const VIEW_SIZE: usize = 2 * BLOCK;
        let (section, view) = one_view(VIEW_SIZE);
        crate::inplace::title_memory_section(section, VIEW_SIZE as u64, false);
        crate::inplace::title_memory_view(view, VIEW_SIZE, section, 0);
        // Fresh `view` is NOT yet in `GUARDS`, only registered in `inplace::VIEWS`.
        assert!(known_view(view));
        let (ww, views) = partition_pieces(&[(view, view + VIEW_SIZE)]);
        assert!(ww.is_empty());
        assert_eq!(views.len(), 1);
        let k0 = KERNEL_ASKS.load(Relaxed);
        ask_once(&[(view, view + VIEW_SIZE)]);
        assert_eq!(KERNEL_ASKS.load(Relaxed) - k0, 0);
        ask_once_end();
        drop_views(section, &[view], VIEW_SIZE);
        forget_written(view, VIEW_SIZE);
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

    #[test]
    fn writes_unchanged_since_matches_writes_on_watched_and_guarded_ranges() {
        // 1. Write-watched range.
        // SAFETY: fresh write-watched allocation, freed below.
        let ww = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                2 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(ww, 0);
        let (ww_start, ww_end) = (ww, ww + 2 * PAGE);

        let (now0, unchanged0) =
            writes_unchanged_since(ww_start, ww_end, 0, false).expect("watched initial");
        assert!(!unchanged0);
        let (_, pages0) = writes(ww_start, ww_end, false).expect("writes initial");
        assert!(pages0.is_empty());

        let (now1, unchanged1) =
            writes_unchanged_since(ww_start, ww_end, now0, false).expect("watched untouched");
        assert!(unchanged1);
        let (_, pages1) = writes(ww_start, ww_end, false).expect("writes untouched");
        assert!(pages1.iter().all(|&(_, w)| w <= now0));

        // SAFETY: committed write-watched page of ours.
        unsafe { ((ww + PAGE + 8) as *mut u8).write(1) };
        let (now2, unchanged2) =
            writes_unchanged_since(ww_start, ww_end, now1, false).expect("watched after write");
        assert!(now2 > now1);
        assert!(!unchanged2);
        let (_, pages2) = writes(ww_start, ww_end, false).expect("writes after write");
        assert!(!pages2.iter().all(|&(_, w)| w <= now1));

        let (_, unchanged3) =
            writes_unchanged_since(ww_start, ww_end, now2, false).expect("watched clean again");
        assert!(unchanged3);

        // SAFETY: our allocation.
        unsafe { VirtualFree(ww as *mut _, 0, MEM_RELEASE) };
        forget_written(ww_start, 2 * PAGE);

        // 2. Guarded physical-page view range.
        const VIEW_SIZE: usize = 2 * BLOCK;
        let (section, view) = one_view(VIEW_SIZE);
        let (view_start, view_end) = (view, view + VIEW_SIZE);

        let (v0, v_unchanged0) =
            writes_unchanged_since(view_start, view_end, 0, true).expect("guarded initial");
        assert!(!v_unchanged0);
        assert!(writes_unchanged_since(view_start, view_end, 0, false).is_none());
        assert!(writes(view_start, view_end, false).is_none());

        let (v1, v_unchanged1) =
            writes_unchanged_since(view_start, view_end, v0, true).expect("guarded untouched");
        assert!(v_unchanged1);
        let (_, v_pages1) = writes(view_start, view_end, true).expect("guarded writes untouched");
        assert!(v_pages1.iter().all(|&(_, w)| w <= v0));

        // SAFETY: guarded block 1 of `view`; VEH fault opens it and marks it written.
        unsafe { ((view + BLOCK + 8) as *mut u8).write_volatile(0x5a) };
        let (v2, v_unchanged2) =
            writes_unchanged_since(view_start, view_end, v1, true).expect("guarded after write");
        assert!(v2 > v1);
        assert!(!v_unchanged2);
        let (_, v_pages2) = writes(view_start, view_end, true).expect("guarded writes after write");
        assert!(!v_pages2.iter().all(|&(_, w)| w <= v1));

        let (_, v_unchanged3) =
            writes_unchanged_since(view_start, view_end, v2, true).expect("guarded clean again");
        assert!(v_unchanged3);

        drop_views(section, &[view], VIEW_SIZE);
        forget_written(view_start, VIEW_SIZE);
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

    /// The tests of the light refresh take turns: they read its counts,
    /// which no other test moves (the switch is not set in tests, so
    /// [`writes`] itself keeps no bits and asks none).
    fn light_turn() -> std::sync::MutexGuard<'static, ()> {
        static TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());
        TURN.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Askings answered by the bits, and askings that went to the guards.
    fn light_counts() -> [u64; 2] {
        [0, 1].map(|i| LIGHT_ASKS[i].load(std::sync::atomic::Ordering::Relaxed))
    }

    /// Whether `[start, end)` is known by its bits.
    fn known(start: usize, end: usize) -> bool {
        published(&journal_read(), start, end)
    }

    /// The pages of an answer of [`writes`] written after the asking `since`.
    fn since(since: u64, written: Vec<(usize, u64)>) -> Vec<usize> {
        written.into_iter().filter(|w| w.1 > since).map(|w| w.0).collect()
    }

    /// The asking that makes blocks clean gives them their bits; the next
    /// one is answered by them, and the guards are not asked. A write takes
    /// the written block's bit, through the guarded view or through another
    /// view of the same pages, and the next asking reports it.
    #[test]
    fn a_range_is_known_by_its_bits_until_a_write() {
        let _turn = light_turn();
        const SIZE: usize = 3 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let pages = |block: usize| (one + block * BLOCK) / PAGE;
        assert!(!known(one, one + SIZE), "nothing asked yet");
        // The first asking takes every block in, read-only and clean.
        let (first, all) = writes_as(one, one + SIZE, true, 2).expect("a view");
        assert_eq!(all, [(pages(0), first), (pages(1), first), (pages(2), first)]);
        assert!(known(one, one + SIZE), "made clean and noted");
        // The second is answered by the bits.
        let before = light_counts();
        let (second, written) = writes_as(one, one + SIZE, true, 2).expect("a view");
        assert!(second > first);
        assert!(since(first, written).is_empty(), "nothing written");
        assert_eq!(light_counts(), [before[0] + 1, before[1]], "the guards were not asked");
        // A write to the guarded view faults: the handler takes the bit.
        // SAFETY: the first view, read-write once the handler has opened it.
        unsafe { ((one + BLOCK + 8) as *mut u8).write_volatile(1) };
        assert!(!known(one, one + SIZE));
        assert!(known(one, one + BLOCK) && known(one + 2 * BLOCK, one + SIZE), "the blocks beside it keep theirs");
        let before = light_counts();
        let (third, written) = writes_as(one, one + SIZE, true, 2).expect("a view");
        assert_eq!(since(second, written), [pages(1)]);
        assert_eq!(light_counts(), [before[0], before[1] + 1], "the guards were asked");
        assert!(known(one, one + SIZE), "guarded again and noted");
        // A write through the other view of the same pages takes it too.
        // SAFETY: the second view, as above.
        unsafe { ((two + 2 * BLOCK + 8) as *mut u8).write_volatile(1) };
        assert!(!known(one + 2 * BLOCK, one + SIZE), "written through another view");
        assert!(known(one, one + 2 * BLOCK));
        let (fourth, written) = writes_as(one, one + SIZE, true, 2).expect("a view");
        assert_eq!(since(third, written), [pages(2)]);
        // A write of ours that is told of (`writing`) does as well.
        writing(one + 8, 8);
        {
            let mut g = guards();
            g.unpin(one / BLOCK, one / BLOCK + 1);
            g.unpin(two / BLOCK, two / BLOCK + 1);
        }
        assert!(!known(one, one + BLOCK));
        let (_, written) = writes_as(one, one + SIZE, true, 2).expect("a view");
        assert_eq!(since(fourth, written), [pages(0)]);
        assert!(known(one, one + SIZE));
        // What the runtime maps anew is not known.
        title_memory_remapped(one + BLOCK, BLOCK);
        assert!(!known(one + BLOCK, one + 2 * BLOCK));
        drop_views(section, &[one, two], SIZE);
        forget_written(one, SIZE);
        forget_written(two, SIZE);
    }

    /// `VICTORY_LIGHT_REFRESH=alt`: in the phase without, the bits are not
    /// asked, and they are kept: what is written in that phase, asked about
    /// in it or not, is reported by the first asking of the phase with.
    /// With the switch off no bit is ever set.
    #[test]
    fn bits_kept_through_a_phase_that_does_not_ask_them_are_true_in_the_next() {
        let _turn = light_turn();
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let pages = |block: usize| (one + block * BLOCK) / PAGE;
        // With: taken in, then known by its bits.
        let (first, _) = writes_as(one, one + SIZE, true, 2).expect("a view");
        assert!(known(one, one + SIZE));
        // Without: the bits are there and are not asked.
        let before = light_counts();
        let (second, written) = writes_as(one, one + SIZE, true, 1).expect("a view");
        assert!(since(first, written).is_empty());
        assert_eq!(light_counts(), [before[0], before[1] + 1], "without: the guards answer");
        // Written in the phase without: the fault takes the bit all the
        // same, and that phase's asking gives it back.
        // SAFETY: the first view, read-write once the handler has opened it.
        unsafe { ((one + 8) as *mut u8).write_volatile(1) };
        assert!(!known(one, one + BLOCK));
        let (third, written) = writes_as(one, one + SIZE, true, 1).expect("a view");
        assert_eq!(since(second, written), [pages(0)]);
        assert!(known(one, one + SIZE));
        // Written again in the phase without, through the other view, and
        // not asked about until the phase with: its first asking goes to
        // the guards and reports it.
        // SAFETY: the second view, as above.
        unsafe { ((two + BLOCK + 8) as *mut u8).write_volatile(1) };
        let before = light_counts();
        let (fourth, written) = writes_as(one, one + SIZE, true, 2).expect("a view");
        assert_eq!(since(third, written), [pages(1)], "a bit from before the phase without was trusted");
        assert_eq!(light_counts(), [before[0], before[1] + 1]);
        let before = light_counts();
        let (fifth, written) = writes_as(one, one + SIZE, true, 2).expect("a view");
        assert!(since(fourth, written).is_empty());
        assert_eq!(light_counts(), [before[0] + 1, before[1]]);
        // The switch off: a write still takes the bit, the asking sets none
        // and counts nothing.
        // SAFETY: the first view, as above.
        unsafe { ((one + 8) as *mut u8).write_volatile(2) };
        let before = light_counts();
        let (_, written) = writes_as(one, one + SIZE, true, 0).expect("a view");
        assert_eq!(since(fifth, written), [pages(0)]);
        assert_eq!(light_counts(), before);
        assert!(!known(one, one + BLOCK), "off: no bit is set");
        drop_views(section, &[one, two], SIZE);
        forget_written(one, SIZE);
        forget_written(two, SIZE);
    }

    /// A range with unmapped blocks is not known by its bits while its
    /// holes are asked about at every asking. Noted for good, a hole that
    /// lies in a placeholder is; then nothing is asked until the runtime
    /// tells of the block, or the look at the holes finds it filled. A
    /// hole in an ordinary reservation is never known by a bit, and its
    /// note grows old.
    #[test]
    fn holes_are_known_by_their_bits_only_when_noted_for_good() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEM_PRESERVE_PLACEHOLDER,
            MEM_REPLACE_PLACEHOLDER, MEM_RESERVE_PLACEHOLDER, MEMORY_MAPPED_VIEW_ADDRESS,
            MapViewOfFile, MapViewOfFile3, PAGE_NOACCESS, SEC_RESERVE, UnmapViewOfFile,
            UnmapViewOfFile2, VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let _turn = light_turn();
        // SAFETY: a fresh placeholder of four blocks and a pagefile
        // section of four, as the runtime maps physical pages.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                4 * BLOCK,
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
                (4 * BLOCK) as u32,
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
        let pages = |block: usize| (p + block * BLOCK) / PAGE;
        let (from, to) = (p, p + 4 * BLOCK);
        let first = map(0);
        assert!(placeholder(p / BLOCK + 1) && placeholder(p / BLOCK + 3), "what is left of the placeholder");
        assert!(!placeholder(p / BLOCK), "a view is none");
        // The first asking takes every block; blocks 1 to 3 are unmapped.
        let (one, all) = writes_as(from, to, true, 2).expect("a view");
        assert_eq!(since(one - 1, all), [pages(0), pages(1), pages(2), pages(3)]);
        assert!(known(p, p + BLOCK) && !known(from, to));
        // Without `told`, placeholder holes do not yet have bits.
        assert!(!known(from, to), "a hole not noted for good has no bit");
        // Noted for good, as with `VICTORY_ABSENT_MS=told`, and looked at
        // as the asking after the notes does.
        assert_eq!(guard_at(from, to, false, 1, u64::MAX), Some((Vec::new(), false)));
        publish(&journal_read(), &[], from, to, true);
        assert!(known(from, to), "holes in a placeholder, noted for good");
        let before = light_counts();
        let (three, written) = writes_as(from, to, true, 2).expect("a view");
        assert!(since(one, written).is_empty());
        assert_eq!(light_counts(), [before[0] + 1, before[1]], "answered by the bits, holes and all");
        // Mapped without a word: not seen, as with the notes alone ...
        let second = map(1);
        let (four, written) = writes_as(from, to, true, 2).expect("a view");
        assert!(since(three, written).is_empty(), "nothing is asked about a hole known by its bit");
        // ... until the look at the holes comes by (it goes round, and
        // waits its turn while another test changes a mapping).
        for _ in 0..2000 {
            if !known(p + BLOCK, p + 2 * BLOCK) {
                break;
            }
            look_at_holes(usize::MAX);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(!known(p + BLOCK, p + 2 * BLOCK), "found filled");
        assert!(known(p + 2 * BLOCK, to), "the holes beside it are holes still");
        let (five, written) = writes_as(from, to, true, 2).expect("a view");
        assert_eq!(since(four, written), [pages(1)]);
        assert!(known(from, to));
        // Mapped and told, as the runtime does: the bit goes at once.
        let third = map(2);
        title_memory_remapped(p + 2 * BLOCK, BLOCK);
        assert!(!known(p + 2 * BLOCK, p + 3 * BLOCK));
        let (_, written) = writes_as(from, to, true, 2).expect("a view");
        assert_eq!(since(five, written), [pages(2)]);
        // A hole in an ordinary reservation (`MEM_RESERVE | MEM_MAPPED`),
        // noted for good all the same: anybody can commit it with no word
        // to us, so it gets no bit.
        // SAFETY: a fresh reserved section view of one block, released below.
        let (res_sec, reserved) = unsafe {
            let sec = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_RESERVE,
                0,
                BLOCK as u32,
                std::ptr::null(),
            );
            assert!(!sec.is_null());
            let view = MapViewOfFile(sec, FILE_MAP_WRITE, 0, 0, BLOCK).Value as usize;
            assert_ne!(view, 0);
            (sec, view)
        };
        assert!(!placeholder(reserved / BLOCK));
        assert_eq!(
            guard_at(reserved, reserved + BLOCK, false, 1, u64::MAX),
            Some((vec![reserved / PAGE], false))
        );
        assert_eq!(
            guard_at(reserved, reserved + BLOCK, false, 2, u64::MAX),
            Some((Vec::new(), false)),
            "noted"
        );
        publish(&journal_read(), &[], reserved, reserved + BLOCK, true);
        assert!(!known(reserved, reserved + BLOCK));
        // Nor does its note live for good: committed with no word, it is
        // found when the note is old.
        // SAFETY: the reserved view made above.
        let committed = unsafe { VirtualAlloc(reserved as *const _, BLOCK, MEM_COMMIT, PAGE_READWRITE) };
        assert_eq!(committed as usize, reserved);
        assert_eq!(
            guard_at(reserved, reserved + BLOCK, false, 50, u64::MAX),
            Some((Vec::new(), false)),
            "the note is 48 ms old"
        );
        assert_eq!(
            guard_at(reserved, reserved + BLOCK, false, 2 + UNSURE_MS, u64::MAX),
            Some((vec![reserved / PAGE], false)),
            "the note is old"
        );
        title_memory_remapped(reserved, BLOCK);
        // SAFETY: our views, placeholder and reservation.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: reserved as *mut _,
            });
            CloseHandle(res_sec);
            for view in [first, second, third] {
                UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER);
            }
            for block in 0..4 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            CloseHandle(section);
        }
        title_memory_remapped(p, 4 * BLOCK);
        forget_written(reserved, BLOCK);
        forget_written(p, 4 * BLOCK);
    }

    /// A copy has nothing new to take of a range known by its bits when it
    /// took every page of it after the page's last noted write, and that is
    /// known with nothing asked.
    #[test]
    fn a_copy_that_holds_every_noted_write_has_nothing_new() {
        let _turn = light_turn();
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let base = one / PAGE;
        assert!(!nothing_noted(one, one + SIZE, base, &[1, 1]), "never asked about: something would have to be asked");
        let (first, _) = writes_as(one, one + SIZE, true, 2).expect("a view");
        let before = light_counts();
        assert!(nothing_noted(one, one + SIZE, base, &[first, first]), "both pages taken at the asking that found them");
        assert_eq!(light_counts(), [before[0] + 1, before[1]]);
        assert!(!nothing_noted(one, one + SIZE, base, &[first, first - 1]), "a page taken before its write was found");
        assert!(!nothing_noted(one, one + SIZE, base, &[first, 0]), "a page never taken");
        assert!(!nothing_noted(one, one + SIZE, base + 1, &[first, first]), "a range the copy's notes do not hold");
        assert!(!nothing_noted(one, one + SIZE, base, &[first]));
        assert!(!nothing_noted(one, one, base, &[first, first]), "no range");
        // Written: the block's bit is gone, and something must be asked.
        // SAFETY: the first view, read-write once the handler has opened it.
        unsafe { ((one + BLOCK + 8) as *mut u8).write_volatile(1) };
        assert!(!nothing_noted(one, one + SIZE, base, &[first, first]));
        assert!(nothing_noted(one, one + BLOCK, base, &[first, first]), "the block beside it is as it was");
        let (second, _) = writes_as(one, one + SIZE, true, 2).expect("a view");
        assert!(!nothing_noted(one, one + SIZE, base, &[first, first]), "the journal has the write");
        assert!(nothing_noted(one, one + SIZE, base, &[first, second]));
        drop_views(section, &[one, two], SIZE);
        forget_written(one, SIZE);
        forget_written(two, SIZE);
    }

    /// A range's bits are all of its blocks', whatever lies beside them.
    #[test]
    fn bits_of_ranges() {
        static BITS: Bits = Bits::new();
        let blocks = [0usize, 1, 63, 64, 65, 127, 128, 200, 65535, 65536, 65537, 1 << 20, (1 << 31) - 1];
        for &b in &blocks {
            assert!(!BITS.all(b, b + 1));
            BITS.set(b);
            assert!(BITS.all(b, b + 1));
        }
        assert!(BITS.all(0, 2) && BITS.all(63, 66) && BITS.all(127, 129) && BITS.all(65535, 65538));
        assert!(!BITS.all(0, 3) && !BITS.all(62, 64) && !BITS.all(65534, 65536) && !BITS.all(5, 5));
        for b in 300..700 {
            BITS.set(b);
        }
        assert!(BITS.all(300, 700) && !BITS.all(299, 700) && !BITS.all(300, 701));
        BITS.clear(511);
        assert!(!BITS.all(300, 700) && BITS.all(300, 511) && BITS.all(512, 700));
        BITS.set(1 << 40);
        BITS.clear(1 << 40);
        assert!(!BITS.all(1 << 40, (1 << 40) + 1), "outside the address space");
    }

    #[test]
    fn ask_once_skips_guards_lock_and_guard_inner_when_published() {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            PAGE_READWRITE, UnmapViewOfFile,
        };

        let (section, base) = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                BLOCK as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let base = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, BLOCK).Value as usize;
            assert_ne!(base, 0);
            (section, base)
        };
        let block = base / BLOCK;
        let first_page = base / PAGE;
        let last_page = first_page + 1;
        assert!(guard(base, base + BLOCK).is_some());

        let j = journal_write();
        super::publish_ranges(&j, &[first_page], &[(block, block + 1)], false);
        drop(j);
        assert!(super::PUBLISHED.all(block, block + 1));

        // 1) `partition_pieces_with(..., true)` routes published blocks into `views` lock-free.
        let (ww, views) = super::partition_pieces_with(&[(base, base + BLOCK)], true);
        assert!(ww.is_empty());
        assert_eq!(views, vec![(first_page, last_page)]);

        // 2) `guard_view_pieces_with(..., true)` skips `guard_inner` when `PUBLISHED` is set,
        //    preserving `ahead_guarded` and incrementing `LIGHT_ASKS[0]`.
        let before = super::LIGHT_ASKS[0].load(std::sync::atomic::Ordering::Relaxed);
        let mut view_pieces = vec![(first_page, last_page)];
        let mut ahead_guarded = Vec::new();
        let mut dirty = Vec::new();
        let _ = super::guard_view_pieces_with(&mut view_pieces, &mut ahead_guarded, &mut dirty, true);
        let after = super::LIGHT_ASKS[0].load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(ahead_guarded, vec![(first_page, last_page)]);
        assert!(dirty.is_empty());
        assert_eq!(after - before, 1);

        super::PUBLISHED.clear(block);
        title_memory_remapped(base, BLOCK);
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: base as *mut _ });
            CloseHandle(section);
        }
    }

    #[test]
    fn ask_once_active_tracks_ask_once_lifecycle() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        let mem = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                PAGE * 2,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        };
        assert!(!mem.is_null());
        super::ask_once_end();
        assert!(!super::ask_once_active());
        let addr = mem as usize;
        super::ask_once(&[(addr, addr + PAGE), (addr + PAGE, addr + PAGE * 2)]);
        assert!(super::ask_once_active());
        super::ask_once_end();
        assert!(!super::ask_once_active());
        unsafe { VirtualFree(mem, 0, MEM_RELEASE) };
    }

    #[test]
    fn tracked_take_reports_writes_once_and_re_reports_new_writes() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        let bytes = PAGE * 3;
        let mem = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                bytes,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as *mut u8;
        assert!(!mem.is_null());
        let addr = mem as usize;
        let tracked = super::track(addr, bytes, false).expect("tracked allocation");
        // Initial allocation has all pages dirty or written once.
        let _ = tracked.take(addr, addr + bytes);
        // Without new writes, take returns empty.
        assert!(tracked.take(addr, addr + bytes).is_empty());
        // Write only page 1 (middle page) and verify only `[addr + PAGE, addr + 2 * PAGE)` is returned once.
        unsafe { std::ptr::write_volatile(mem.add(PAGE + 8), 0x42) };
        assert_eq!(tracked.take(addr, addr + bytes), vec![(addr + PAGE, addr + 2 * PAGE)]);
        assert!(tracked.take(addr, addr + bytes).is_empty());
        unsafe { VirtualFree(mem.cast(), 0, MEM_RELEASE) };
    }
}

#[cfg(test)]
mod mapping_tests {
    use super::{
        title_memory_reading, title_memory_reading_now_range, title_memory_reading_range,
        title_memory_remapping, title_memory_remapping_range,
    };
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

    #[test]
    fn disjoint_ranges_proceed_while_overlapping_ranges_wait() {
        // Remapping `[0x1000, 0x2000)` lets a disjoint read proceed at once,
        // while an overlapping read waits until the remapping drops.
        let changing = title_memory_remapping_range(0x1000..0x2000);
        assert!(title_memory_reading_now_range(0x2000..0x3000).is_some());
        assert!(title_memory_reading_now_range(0x1800..0x2800).is_none());
        let disjoint = std::thread::spawn(|| {
            let _r = title_memory_reading_range(0x2000..0x3000);
        });
        disjoint.join().unwrap();

        let overlap_read = Arc::new(AtomicBool::new(false));
        let (began, wait) = mpsc::channel();
        let overlapping = {
            let overlap_read = overlap_read.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _r = title_memory_reading_range(0x1800..0x2800);
                overlap_read.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert!(
            !overlap_read.load(Ordering::SeqCst),
            "overlapping read began during a range remapping"
        );
        drop(changing);
        overlapping.join().unwrap();
        assert!(overlap_read.load(Ordering::SeqCst));

        // Holding a read on `[0x2000, 0x3000)` lets a disjoint remapping proceed,
        // while an overlapping remapping waits until the read drops.
        let reading = title_memory_reading_range(0x2000..0x3000);
        let disjoint_remap = std::thread::spawn(|| {
            let _w = title_memory_remapping_range(0x1000..0x2000);
        });
        disjoint_remap.join().unwrap();

        let overlap_changed = Arc::new(AtomicBool::new(false));
        let (began, wait) = mpsc::channel();
        let overlapping_remap = {
            let overlap_changed = overlap_changed.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _w = title_memory_remapping_range(0x2500..0x3500);
                overlap_changed.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert!(
            !overlap_changed.load(Ordering::SeqCst),
            "overlapping remapping began under an active range read"
        );
        drop(reading);
        overlapping_remap.join().unwrap();
        assert!(overlap_changed.load(Ordering::SeqCst));
    }

    #[test]
    fn writing_pins_guarded_view_writable_and_dirty_without_reprotecting() {
        use super::{BLOCK, PAGE, guard, guards, title_memory_remapped, writable, writing};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            PAGE_READWRITE, UnmapViewOfFile,
        };
        let (section, base) = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let base = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize;
            assert_ne!(base, 0);
            (section, base)
        };
        let p0 = base / PAGE;
        let p1 = (base + BLOCK) / PAGE;
        assert_eq!(guard(base, base + 2 * BLOCK), Some(vec![p0, p1]));
        assert_eq!(guard(base, base + 2 * BLOCK), Some(Vec::new()));

        // Pin block 0 for a kernel write: block 0 stays writable and dirty on repeated checks,
        // while block 1 stays clean and guarded.
        writing(base, BLOCK);
        assert!(writable(base));
        assert_eq!(guard(base, base + 2 * BLOCK), Some(vec![p0]));
        assert!(writable(base), "pinned block must stay writable after guard()");
        assert_eq!(guard(base, base + 2 * BLOCK), Some(vec![p0]));

        // Clean up pin and view.
        {
            let mut g = guards();
            g.pins.retain(|&(from, to, _)| !(from == base / BLOCK && to == base / BLOCK + 1));
        }
        title_memory_remapped(base, 2 * BLOCK);
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: base as *mut _ });
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
    }

    #[test]
    fn block_set_two_set_word_scan_matches_reference() {
        use super::{BTreeSet, BlockSet};
        let mut a = BlockSet::new();
        let mut b = BlockSet::new();
        let mut ref_a = BTreeSet::new();
        let mut ref_b = BTreeSet::new();
        let mut x = 98765u64;
        for _ in 0..10_000 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let blk = ((x >> 33) % 15_000) as usize + if x & 1 == 0 { 0 } else { 1 << 29 };
            if x & 2 == 0 {
                a.insert(blk);
                ref_a.insert(blk);
            } else {
                b.insert(blk);
                ref_b.insert(blk);
            }
        }
        a.extend(4000..4200);
        ref_a.extend(4000..4200);
        b.extend(4150..8300);
        ref_b.extend(4150..8300);

        for r in [
            0..0,
            4000..8300,
            3990..4050,
            8190..8310,
            0..10_000,
            (1 << 29)..(1 << 29) + 15_000,
        ] {
            let expected_missing: Vec<usize> = r
                .clone()
                .filter(|blk| !ref_a.contains(blk) && !ref_b.contains(blk))
                .collect();
            let mut got = Vec::new();
            a.for_each_missing_in_two(&b, r.start, r.end, |blk| got.push(blk));
            assert_eq!(got, expected_missing, "range {r:?}");
            assert_eq!(
                a.all_covered_by_two(&b, r.start, r.end),
                expected_missing.is_empty(),
                "all_covered_by_two {r:?}"
            );
        }
    }

    #[test]
    fn ahead_guarded_keeps_clean_ranges_when_another_range_is_pinned() {
        use super::{
            BLOCK, GUARD_REV, PAGE, guard, guard_view_pieces, guards, title_memory_remapped,
            writing,
        };
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            PAGE_READWRITE, UnmapViewOfFile,
        };
        let section = unsafe {
            CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (4 * BLOCK) as u32,
                std::ptr::null(),
            )
        };
        assert!(!section.is_null());
        let base = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 4 * BLOCK) }.Value as usize;
        assert_ne!(base, 0);

        let clean_range = (base, base + BLOCK);
        let pinned_range = (base + 2 * BLOCK, base + 4 * BLOCK);
        assert!(guard(clean_range.0, clean_range.1).is_some());
        assert!(guard(pinned_range.0, pinned_range.1).is_some());
        writing(pinned_range.0, BLOCK);

        let mut view_pieces = vec![
            (clean_range.0 / PAGE, clean_range.1.div_ceil(PAGE)),
            (pinned_range.0 / PAGE, pinned_range.1.div_ceil(PAGE)),
        ];
        let rev_before;
        let mut ahead_guarded = Vec::new();
        let mut dirty = Vec::new();
        rev_before = guard_view_pieces(&mut view_pieces, &mut ahead_guarded, &mut dirty);

        assert_eq!(
            GUARD_REV.load(std::sync::atomic::Ordering::SeqCst),
            rev_before,
            "pinned block must not bump GUARD_REV during guard_view_pieces"
        );
        assert_eq!(
            ahead_guarded,
            vec![(clean_range.0 / PAGE, clean_range.1.div_ceil(PAGE))],
            "clean piece stays in ahead_guarded while pinned piece is excluded"
        );

        {
            let mut g = guards();
            g.pins.retain(|&(from, to, _)| !(from == pinned_range.0 / BLOCK && to == pinned_range.0 / BLOCK + 1));
        }
        title_memory_remapped(base, 4 * BLOCK);
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: base as *mut _ });
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
    }

    #[test]
    fn hot_guard_skips_reprotect_after_repeated_veh_faults_and_expires() {
        use super::{
            BLOCK, HOT_FAULT_THRESHOLD, HOT_PIN_EPOCHS, PAGE, guard, hot_guard_on,
            title_memory_remapped, writable,
        };
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            PAGE_READWRITE, UnmapViewOfFile,
        };
        if !hot_guard_on() {
            return;
        }
        let section = unsafe {
            CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                BLOCK as u32,
                std::ptr::null(),
            )
        };
        assert!(!section.is_null());
        let base = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, BLOCK) }.Value as usize;
        assert_ne!(base, 0);
        let p0 = base / PAGE;

        assert_eq!(guard(base, base + BLOCK), Some(vec![p0]));
        assert!(!writable(base));

        // Repeated VEH write faults within window trigger a hot pin.
        for _ in 0..HOT_FAULT_THRESHOLD {
            unsafe { (base as *mut u8).write_volatile(7) };
            let got = guard(base, base + BLOCK);
            assert_eq!(got, Some(vec![p0]));
        }
        assert!(writable(base), "threshold VEH fault within window must hot-pin block writable");
        assert_eq!(guard(base, base + BLOCK), Some(vec![p0]));
        assert!(writable(base), "hot-pinned block stays writable while active");

        // Advance guard_epoch past the hot-pin TTL and verify re-arm.
        for _ in 0..HOT_PIN_EPOCHS {
            let _ = guard(base, base + BLOCK);
        }
        assert!(!writable(base), "expired hot-pin must re-protect block READONLY");
        assert_eq!(guard(base, base + BLOCK), Some(Vec::new()));

        title_memory_remapped(base, BLOCK);
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: base as *mut _ });
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
    }

    #[test]
    fn unguardable_alias_leaves_block_dirty_and_not_ahead_guarded() {
        use super::tests::{drop_views, two_views};
        use super::{BLOCK, PAGE, guard, guard_inner, guards, writable, writing};
        const SIZE: usize = BLOCK;
        let (section, one, two) = two_views(SIZE);
        let p_one = one / PAGE;
        // Pin `two` (keeping `one` unpinned) so `two` cannot be protected as an alias.
        writing(two, BLOCK);
        guards().unpin(one / BLOCK, one / BLOCK + 1);
        assert!(writable(two));

        let (dirty, any_pinned) = guard_inner(one, one + SIZE, true).expect("view");
        assert_eq!(dirty, vec![p_one]);
        assert!(any_pinned, "unguardable alias must mark range non-cacheable");
        assert!(writable(two), "pinned alias stays writable");
        assert!(
            !guards().clean.contains(&(one / BLOCK)),
            "block with unguardable alias must not be marked clean"
        );
        // Subsequent check still reports `one` dirty while `two` remains pinned.
        assert_eq!(guard(one, one + SIZE), Some(vec![p_one]));

        // Unpin `two`; next check guards both and following check is clean.
        guards().unpin(two / BLOCK, two / BLOCK + 1);
        assert_eq!(guard(one, one + SIZE), Some(vec![p_one]));
        assert!(!writable(two));
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()));

        drop_views(section, &[one, two], SIZE);
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod fast_copy_tests {
    use super::{
        KERNEL_PAGE, PAGE, title_memory_reading, title_memory_remapping,
        title_memory_remapping_range, try_copy_fast, try_diff_bounds_fast,
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

        // While an unrelated range is remapping, fast paths on `src` still succeed.
        let remapping = title_memory_remapping_range(0x1000..0x2000);
        assert_eq!(try_diff_bounds_fast(src.as_ptr(), &dst), Some(Some((0, 16))));
        assert!(try_copy_fast(dst.as_mut_ptr(), src.as_ptr(), src.len()));
        assert_eq!(dst, src);
        drop(remapping);
        dst.fill(0);

        // While an overlapping range is remapping, fast paths yield.
        let start = src.as_ptr() as usize;
        let remapping = title_memory_remapping_range(start..start + 1);
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

    #[test]
    fn fast_read_guard_blocks_remap_until_dropped_and_allows_concurrent_readers() {
        let g1 = super::acquire_reading_now(0x1000..0x2000).expect("fast reader 1");
        let g2 = super::acquire_reading_now(0x3000..0x4000).expect("fast reader 2");
        assert!(matches!(g1, super::FastReadGuard::Fast));
        assert!(matches!(g2, super::FastReadGuard::Fast));

        let blocked = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let entered = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let b2 = blocked.clone();
        let e2 = entered.clone();
        let d2 = done.clone();
        let h = std::thread::spawn(move || {
            b2.store(true, std::sync::atomic::Ordering::Release);
            let _remap = title_memory_remapping_range(0x5000..0x6000);
            e2.store(true, std::sync::atomic::Ordering::Release);
            d2.store(true, std::sync::atomic::Ordering::Release);
        });
        while !blocked.load(std::sync::atomic::Ordering::Acquire) {
            std::thread::yield_now();
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(
            !entered.load(std::sync::atomic::Ordering::Acquire),
            "remapping must wait for active fast readers"
        );
        drop(g1);
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert!(
            !entered.load(std::sync::atomic::Ordering::Acquire),
            "remapping must still wait while second fast reader is held"
        );
        drop(g2);
        h.join().unwrap();
        assert!(done.load(std::sync::atomic::Ordering::Acquire));
    }
}
