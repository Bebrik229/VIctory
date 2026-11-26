//! Host descriptor heaps behind the title's shader-visible heaps (decision
//! 0007, step 3). Translated shaders index the heaps directly at the
//! title's descriptor positions, so each host heap keeps the title heap's
//! order: slot `i` is made from what the title wrote at slot `i`, when it
//! writes it (view creation and descriptor copies) and again when a draw
//! reads it through a table and finds the slot changed. A view heap the
//! host caps below the title's slots keeps that order only for an identity
//! window of them; past it each title slot that holds a view is made in a
//! pool slot, which translated shaders look up in a heap map ([`MapLayout`]).

use std::sync::Mutex;
use std::sync::atomic::AtomicU64;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::slot_map::SlotMap;
use super::{VIEW_TAG, View, view_at};
use crate::host::host;

/// Number of buckets partitioning [`HostHeap::buffers`] by `index % BUFFER_BUCKETS`,
/// matching `draw.rs`'s 16-pass amortized buffer sweep so each pass clones only its
/// `1 / 16` slice without scanning the whole set.
pub(crate) const BUFFER_BUCKETS: usize = 16;

/// The locks [`HostHeap::kept`] is spread over: slot `i` is behind lock
/// `i % KEPT_PIECES`, at place `i / KEPT_PIECES`. Neighbouring slots are
/// behind different locks on purpose: the recording threads take their
/// tables' dynamic slots from one ring, next to one another.
const KEPT_PIECES: u32 = 64;

/// Whether the recording threads are spared the heap's kept-resource lock
/// (`VICTORY_HEAP_LOCKS=1`; after fork pull request #1146, part 1; OFF
/// unless set; `alt`: twenty seconds on, twenty off, in turn).
fn heap_locks() -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    /// 0 off, 1 on, 2 in turn.
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    match *MODE.get_or_init(|| match std::env::var("VICTORY_HEAP_LOCKS").as_deref() {
        Ok("1") => 1,
        Ok("alt") => 2,
        _ => 0,
    }) {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), Relaxed) != u8::from(on) {
                tracing::info!(
                    "the host heap's kept resources are locked per piece without the one line, from now (VICTORY_HEAP_LOCKS=alt): {}",
                    if on { "ON" } else { "off, one lock as usual" }
                );
            }
            on
        }
    }
}

/// Diagnostics, since the last line (`crate::draw::tables_line`): why
/// [`HostHeap::dynamic`] refused a table. 0: fewer slots were free than it
/// asked (the ring is full: nothing to search); 1: enough were free but a
/// search for a table no longer had failed since the last release, so none
/// was made (`too_long`); 2: a search found no run of free slots that long
/// (the free slots lie apart). 3: tables that got a run lying across the
/// place the search starts from, which the search missed before it went on
/// past that place.
#[cfg(feature = "diagnostics")]
pub(crate) static DYNAMIC_WHY: [std::sync::atomic::AtomicU64; 4] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Diagnostics: the fewest free dynamic slots [`HostHeap::dynamic`] saw
/// since the last line (`u32::MAX`: it was not called).
#[cfg(feature = "diagnostics")]
pub(crate) static DYNAMIC_LOW: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(u32::MAX);

pub(crate) struct HostHeap {
    pub(crate) heap: ID3D12DescriptorHeap,
    cpu: usize,
    gpu: u64,
    step: u32,
    /// Slots mirroring the title's at their own index (in a view heap with
    /// a heap map, the identity window); view heaps have `dynamic` more.
    count: u32,
    /// Dynamic slots from `dynamic_first`: [`DYNAMIC`] in view heaps, none
    /// in sampler heaps. (Upstream falls back to 2^16 on a capped host heap;
    /// ours keeps 2^18 there: 2^16 and 2^17 both ran out in this title.)
    dynamic: u32,
    /// The first dynamic slot: `count`, or in a view heap with a heap map
    /// the one after the map's slot ([`MapLayout`]).
    dynamic_first: u32,
    /// The title's own slot count (`count` may be fewer).
    title_count: u32,
    sampler: bool,
    /// The record (view index and generation) each host slot of the
    /// title's slots was last made from, and the heap map's bookkeeping.
    made: Mutex<Made>,
    /// Lock-free atomic mirror of `made.keys` for identity-window slots
    /// (`0..count`; ~98% cache-hit path in [`HostHeap::mirror`]).
    made_fast: Vec<std::sync::atomic::AtomicU64>,
    /// Lock-free last-seen 12-byte view key per title slot for
    /// [`HostHeap::mirror_copied`] (0: empty or non-view; allocated on first
    /// use). Avoids per-slot mutex locking during `sync_heaps`.
    copied: std::sync::OnceLock<Box<[AtomicU64]>>,
    /// The heap map of a view heap capped below the title's slots
    /// ([`MapLayout`]); `None` for the others.
    map: Option<HeapMap>,
    /// The title heap's first descriptor, as the first slot mirrored past
    /// the window gives it (0 until then): for [`HostHeap::reclaim`].
    title_start: AtomicU64,
    /// When [`HostHeap::reclaim`] last ran (`crate::host::now_ms`;
    /// `u64::MAX`: never).
    reclaimed: AtomicU64,
    /// The host resource each slot views, held while the slot does: shaders
    /// index heaps directly, so a slot the title has not rewritten since it
    /// released the resource may still be read, and a freed host resource
    /// read there is a GPU page fault (the console reads stale memory).
    /// In [`KEPT_PIECES`] pieces, each behind its own lock.
    kept: Vec<Mutex<Vec<Option<ID3D12Resource>>>>,
    /// Without [`heap_locks`]: the one line every `keep` stands in, as
    /// when `kept` was behind one lock.
    kept_one: Mutex<()>,
    /// Where the search for free dynamic slots starts, and which are taken
    /// (by lists not yet reset).
    next: Mutex<(u32, Vec<bool>)>,
    /// How many dynamic slots are free (changed under `next`'s lock): with
    /// fewer than a table needs, the search is not made. Once the world
    /// had taken them all, every later table of the frame searched every
    /// slot before it was refused.
    free: std::sync::atomic::AtomicU32,
    /// The shortest table that found no run of free slots since slots were
    /// last given back (changed under `next`'s lock; `u32::MAX`: none).
    /// Until something is given back the free runs only shrink, so a table
    /// of that length or longer is refused without the search. Enough
    /// slots were free, but in pieces, and every such table walked the
    /// whole ring before it was refused: 0.33 ms each with 131,072 slots,
    /// 1.0 s of the recording threads in a frame that refused 3,143 (run
    /// 45), under the lock all of them need. (A free run that lies across
    /// the place the search starts from is seen by neither.)
    too_long: std::sync::atomic::AtomicU32,
    /// View heaps: the bytes per element of the buffer view at each title
    /// slot, for shaders that pick a buffer by an index they compute
    /// (`victory_shader::emit::SLOT_STRIDES`): an upload buffer of
    /// `SLOT_STRIDES_COUNT` dwords, its GPU address and its mapping.
    strides: Option<(ID3D12Resource, u64, usize)>,
    /// Title slots holding buffer views, partitioned into [`BUFFER_BUCKETS`]
    /// buckets by `index % BUFFER_BUCKETS`. Such a view gets its host slot
    /// (and its memory a host copy) only when something reads it: a draw
    /// through a table, or [`HostHeap::buffer_slots_bucket`] for shaders that
    /// pick buffers from the whole heap.
    buffers: [Mutex<std::collections::BTreeSet<u32>>; BUFFER_BUCKETS],
    /// Of those and of the slots that left the set: the ones written since
    /// [`HostHeap::buffer_changes`] last took them.
    changed: Mutex<std::collections::HashSet<u32>>,
    /// Title slots holding a view of a texture the title may fill itself (a
    /// placed texture, a texture object), with the frame from which they
    /// are due: such a texture is uploaded when a draw reads it through a
    /// table; one a shader picks by a computed index is brought up to date
    /// from here ([`HostHeap::texture_slots_due`]).
    textures: Mutex<TextureQueue>,
    /// Per title slot, the frame (plus one; 0: never) in which a draw last
    /// read it through a table, or made a table it lies in (with
    /// [`TABLED_READ`] set if the draw read it). The draw that
    /// reads such a slot brings its texture up to date itself and the slot
    /// then leaves `textures`; a write of a slot that was a table's within
    /// the last [`TABLED_FRAMES`] frames is due only [`TABLED_GRACE`]
    /// frames later, by when its draw has read it. A title that builds a
    /// table per draw writes thousands of texture views a frame into slots
    /// only tables read (the title: some 2,500 a frame in its
    /// opening cutscene, a few hundred textures over and over), and every
    /// one of them stood in the queue of textures picked by index, ahead
    /// of or behind the ones shaders do pick by index.
    tabled: Vec<std::sync::atomic::AtomicU32>,
    /// Per `PAGE_TABLED_SLOTS`-aligned 64 KiB view-heap page: oldest non-zero
    /// `tabled` stamp seen across an all-tabled page scan (`0` if unverified or
    /// containing any untabled slot). Enables O(1) page-level skips in `sync_heaps`.
    page_tabled_min: Vec<std::sync::atomic::AtomicU32>,
    /// The title memory each placed texture view of this heap describes,
    /// by slot and by address ([`Spans`]): which slots a newer view of
    /// other memory there makes stale, and which to look at again when the
    /// runtime maps, unmaps or writes memory ([`remapped`]).
    placed: std::sync::Arc<Mutex<Spans>>,
    /// A capped heap's views of the title's slots past `count`, for fresh
    /// tables to copy from ([`HostHeap::copy_overflow`]).
    overflow: Option<Overflow>,
    /// Per-heap buffer sweep cache (16-bucket amortized view ranges + merged
    /// shadow spans) and per-heap unwatched-buffer last-refresh frame map,
    /// released automatically when this [`HostHeap`] is dropped.
    pub(crate) buffer_sweep: crate::draw::BufferSweep,
}

#[cfg(test)]
thread_local! {
    /// Tests: the title slots a view heap this thread makes from now on
    /// keeps, fewer than the title's (0: all, as the host takes). Per
    /// thread, so tests running meanwhile make their heaps as before.
    pub(crate) static TEST_TITLE_SLOTS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Whether a capped heap keeps views of the title's slots past its own
/// (`VICTORY_OVERFLOW_VIEWS`, on unless `0`); else fresh tables make every
/// view each time, as before.
fn overflow_views() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| overflow_views_from(std::env::var_os("VICTORY_OVERFLOW_VIEWS").as_deref()))
}

/// [`overflow_views`] for a value of the switch.
fn overflow_views_from(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_none_or(|v| v != "0")
}

/// Views of the title's slots past a capped heap's title slots, in a
/// CPU-only heap (not under the host's limit on shader-visible ones), one
/// per title slot: made from the title's record where a fresh table uses
/// the slot, and again only when the slot's record changes, as title slots
/// are made when the title writes them. Fresh tables copy them. Each keeps
/// what its view needs alive, as a title slot does.
struct Overflow {
    /// Held for its slots: only their CPU handles are used.
    _heap: ID3D12DescriptorHeap,
    cpu: usize,
    step: u32,
    len: u32,
    /// The slots, [`SHARD`] to a lock: lists recording at once take
    /// different ones.
    shards: Vec<Mutex<Vec<Over>>>,
}

/// Overflow slots per lock.
const SHARD: u32 = 1024;

/// An overflow slot: the record it was made from ([`Over::NULL`] for a
/// null view, 0 for none made), and what it keeps alive.
#[derive(Clone, Default)]
struct Over {
    key: u64,
    kept: Kept,
}

impl Over {
    const NULL: u64 = u64::MAX;
}

impl Overflow {
    fn new(device: &ID3D12Device, len: u32, step: u32) -> Option<Self> {
        // SAFETY: plain heap creation.
        let made = unsafe {
            device.CreateDescriptorHeap::<ID3D12DescriptorHeap>(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: len,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                NodeMask: 0,
            })
        };
        let heap = match made {
            Ok(heap) => heap,
            Err(e) => {
                tracing::warn!(error = %e, len, "no overflow view heap: fresh tables make their views each time");
                return None;
            }
        };
        tracing::info!(
            slots = len,
            descriptor_bytes = step,
            mb = f64::from(len) * f64::from(step) / 1e6,
            "overflow view heap (CPU memory) for the title's slots past the capped heap's (VICTORY_OVERFLOW_VIEWS)"
        );
        // SAFETY: a live heap.
        let cpu = unsafe { heap.GetCPUDescriptorHandleForHeapStart().ptr };
        let shards = (0..len.div_ceil(SHARD))
            .map(|s| Mutex::new(vec![Over::default(); SHARD.min(len - s * SHARD) as usize]))
            .collect();
        Some(Overflow { _heap: heap, cpu, step, len, shards })
    }

    fn cpu(&self, at: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE { ptr: self.cpu + at as usize * self.step as usize }
    }
}

/// The spans of every view heap made ([`HostHeap::placed`]), for the
/// texture code, which has no heap at hand ([`next_start`],
/// [`first_viewed`]).
static PLACED_SPANS: Mutex<Vec<std::sync::Weak<Mutex<Spans>>>> = Mutex::new(Vec::new());

/// Where the next texture placed inside `[start, end)` starts, past the
/// first bytes of the one at `start` ([`HEAD`]): what is there of the
/// texture at `start` ends there, whatever its full size. `None`: no view
/// in a slot shaders pick by index describes memory in that range.
pub(crate) fn next_start(start: u64, end: u64) -> Option<u64> {
    let heaps = PLACED_SPANS.lock().unwrap_or_else(|e| e.into_inner());
    heaps
        .iter()
        .filter_map(std::sync::Weak::upgrade)
        .filter_map(|spans| spans.lock().unwrap_or_else(|e| e.into_inner()).next_start(start, end))
        .min()
}

/// The lowest level that a view of the placed texture at `start` shows,
/// over the slots shaders pick by index that hold one now. The title
/// views a texture from the first level it holds. `None`: no such slot.
pub(crate) fn first_viewed(start: u64) -> Option<u32> {
    let heaps = PLACED_SPANS.lock().unwrap_or_else(|e| e.into_inner());
    heaps
        .iter()
        .filter_map(std::sync::Weak::upgrade)
        .filter_map(|spans| spans.lock().unwrap_or_else(|e| e.into_inner()).first_viewed(start))
        .min()
}

/// Levels on demand (`crate::tiles`): where the placed textures start
/// whose first level with memory changed since a texture pass last took
/// them ([`HostHeap::texture_views_remade`]).
static RESHOWN: Mutex<Vec<u64>> = Mutex::new(Vec::new());

/// The first level with memory of the placed texture at `start` changed:
/// its slots are made again at the next texture pass.
pub(crate) fn levels_changed(start: u64) {
    RESHOWN.lock().unwrap_or_else(|e| e.into_inner()).push(start);
}

/// Levels on demand (`crate::tiles`): where the placed textures start
/// whose last upload left a held level without tiles (the card had no
/// room, or no heap could be had), and when (`crate::host::moment`). A
/// texture pass makes their slots due again once the card has room
/// ([`HostHeap::texture_slots_waiting`]).
static WAITING: Mutex<std::collections::BTreeMap<u64, u64>> = Mutex::new(std::collections::BTreeMap::new());

/// The last upload of the placed texture at `start` left a held level
/// without tiles. A texture picked by index is uploaded once and then
/// only when its memory is found changed, and its staging buffer goes
/// with the upload: without this the level stayed empty (the texture
/// blurred, or black if no level had tiles) for as long as the title left
/// its memory alone, where a texture refused for want of room is tried
/// again.
pub(crate) fn levels_waiting(start: u64) {
    WAITING.lock().unwrap_or_else(|e| e.into_inner()).insert(start, crate::host::moment());
}

/// Host-slot state behind one lock: every write of a host slot for a title
/// slot takes it, so a pool slot is taken, made and published (or given
/// back) at once.
struct Made {
    /// The record (view index and generation) each host slot of the title's
    /// (`count`, and in a heap with a heap map its pool's) was last made
    /// from; 0 for none.
    keys: Vec<u64>,
    /// Which pool slot each title slot past the identity window holds (a
    /// view heap with a heap map).
    map: Option<SlotMap>,
}

impl Made {
    /// The host slot title slot `title` is made in, if it has one (none is
    /// taken here): its own below the window, its pool slot past it.
    fn host(&self, title: u32) -> Option<u32> {
        match &self.map {
            Some(map) => map.host(title),
            None => ((title as usize) < self.keys.len()).then_some(title),
        }
    }
}

/// The heap map of a view heap with a [`MapLayout`]: an upload buffer of one
/// entry per title slot from the identity window on ([`slot_map::entry`]:
/// the bitwise not of its host slot, 0 for none), persistently mapped, and
/// read by translated shaders through a raw view at the layout's map slot
/// (`victory_shader::emit::set_view_heap_map`).
///
/// [`slot_map::entry`]: super::slot_map::entry
struct HeapMap {
    buffer: ID3D12Resource,
    /// Where the entries are mapped.
    at: usize,
}

/// The title's view slots in a view heap whose host heap is capped below
/// them, with a heap map: title slots `0..window` at their own index, as
/// tables made in place need; `pool` slots for the title slots past the
/// window that hold views (looked up in the map by translated shaders);
/// the map's raw view at `slot`; then the fork's [`DYNAMIC`] dynamic slots
/// and its null guard slot last, where they were without a map.
///
/// On a capped host this layout uses 1,000,000 shader-visible views
/// ([`PORTABLE_VIEWS`]); the title's has 1,245,184 slots, some
/// 650,000-700,000 of them holding views, about 335,000 of those past the
/// 737,855 title slots a capped heap kept in place (their bindless reads
/// gave the null slot).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MapLayout {
    window: u32,
    pool: u32,
    slot: u32,
}

impl MapLayout {
    /// The layout for `title_count` title slots, an identity window of up
    /// to `window` slots and the map at `slot`.
    fn new(title_count: u32, window: u32, slot: u32) -> Self {
        let window = window.min(slot).min(title_count);
        MapLayout {
            window,
            pool: (slot - window).min(title_count - window),
            slot,
        }
    }

    /// The first dynamic slot.
    fn dynamic_first(self) -> u32 {
        self.slot + 1
    }

    /// The null guard slot, the heap's last: the limit translated shaders
    /// clamp view indices to.
    fn guard(self) -> u32 {
        self.dynamic_first() + DYNAMIC
    }

    /// The heap's slots.
    fn size(self) -> u32 {
        self.guard() + 1
    }
}

/// Bytes per title view descriptor.
const TITLE_SLOT_BYTES: u64 = 32;

/// The identity window of a heap map without `VICTORY_HEAP_WINDOW`: the
/// title's tables and its first two 64Ki runs of views lie below it, its
/// first empty run (110,940..327,680) spans it.
const WINDOW: u32 = 1 << 17;

/// The heap map's slot in a capped view heap: right before the dynamic
/// slots, so that those and the guard slot stay where they were
/// (`PORTABLE_VIEWS - DYNAMIC - 1` on, the guard at `PORTABLE_VIEWS - 1`).
/// No view index reaches it: one below the window is its own, one past it
/// is a pool slot or the guard, and tables made in place lie below the
/// window.
const MAP_SLOT: u32 = PORTABLE_VIEWS - DYNAMIC - 2;

/// The identity window: [`WINDOW`], or `VICTORY_HEAP_WINDOW` (a smaller one
/// maps more slots, a larger one fewer; at most [`MAP_SLOT`]).
fn window() -> u32 {
    std::env::var("VICTORY_HEAP_WINDOW")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(WINDOW)
        .min(MAP_SLOT)
}

/// Whether a capped view heap gets a heap map; `VICTORY_HEAP_MAP=0` keeps
/// the fork's cap (views past it read the null slot).
fn heap_map_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_MAP").is_none_or(|v| v != "0"))
}

/// The heap map translated shaders read: the identity window and the
/// map's slot ([`probe_view_limit`] sets it before the first translation).
fn heap_map() -> Option<(u32, u32)> {
    // Tests: only the thread's own map. `HostHeap::new` probes the host in
    // tests too, which on a capped card sets the translator's map for the
    // whole process.
    #[cfg(test)]
    return TEST_MAP.get();
    #[cfg(not(test))]
    victory_shader::emit::view_heap_map()
}

/// VIcky (diagnostics): the heap map's pool (title slots mapped, pool
/// slots free and waiting, refusals), each time the title slots it holds
/// pass another 64Ki.
#[cfg(feature = "diagnostics")]
fn note_pool(made: &Made) {
    use std::sync::atomic::Ordering::Relaxed;
    static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let Some(stats) = made.map.as_ref().map(SlotMap::stats) else {
        return;
    };
    let step = stats.mapped >> 16;
    if step > LOGGED.load(Relaxed) {
        LOGGED.store(step, Relaxed);
        tracing::info!(?stats, "heap map pool");
    }
}

#[cfg(test)]
thread_local! {
    /// Tests: the heap map (window, slot) of the view heaps this thread
    /// makes, as [`probe_view_limit`] sets one on a capped host, without
    /// reaching other tests' heaps.
    pub(crate) static TEST_MAP: std::cell::Cell<Option<(u32, u32)>> = const { std::cell::Cell::new(None) };
}

/// Title memory by the placed texture views that describe it: per span of
/// memory (its start, its end at full size) the title slots holding a
/// view of a texture that takes exactly that span. A console title need
/// not clear a descriptor when it streams a texture out, so a newer view
/// that cannot be there together with an older one is all that says the
/// older one is stale ([`Rule`] says when that is). Views of one span with
/// different descriptions (the formats of one family, say) are one
/// texture's.
///
/// Spans may overlap: a texture's smallest levels lie at the start of its
/// memory, and a title that holds only those puts the next texture where
/// the larger ones would be (the title: three 512x256 textures
/// of 0x2c000 bytes each whose starts are 0x2000 and 0x4000 apart, all
/// three read by the same frame's draws, run 188). What is there of a
/// texture ends where the next one starts.
/// No host heap is needed, so it is tested without one.
#[derive(Default)]
pub(crate) struct Spans {
    /// When a newer view makes an older one stale.
    rule: Rule,
    by_start: std::collections::BTreeMap<u64, (u64, std::collections::BTreeSet<u32>)>,
    by_slot: std::collections::HashMap<u32, u64>,
    /// The level each slot's view starts at ([`Spans::shows`]).
    first: std::collections::HashMap<u32, u32>,
    /// The bytes a placed texture takes, by resource description: worked
    /// out once for each (a title writes the views of a few thousand
    /// descriptions, thousands of times a frame).
    sizes: std::collections::HashMap<[u32; 14], Option<u64>>,
}

/// When a view the title writes makes an older view of other memory stale.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Rule {
    /// When it starts where the older one starts: at the same address
    /// with another length, or within the older one's first [`HEAD`]
    /// bytes (or the older within its own). No two textures share the
    /// block of their smallest levels. A view that only reaches into the
    /// address range of an older texture's larger levels says nothing.
    #[default]
    Start,
    /// Whenever their spans at full size overlap (the rule until run
    /// 188, where it made null the slots of textures packed as above:
    /// each view written took its neighbours out of the queue and off the
    /// card, for good). `VICTORY_STALE_TEXTURES=overlap`.
    Overlap,
    /// Never; a view that would be stale under [`Rule::Overlap`] is left
    /// out of the spans instead. `VICTORY_STALE_TEXTURES=0`, as before.
    Never,
}

/// The first bytes of a placed texture that are its own under
/// [`Rule::Start`]: the 4 KiB block its smallest levels share. (Of 129
/// pairs of overlapping neighbours in the listings of runs 188 and 189,
/// none starts closer than 0x2000 to the one before it.)
const HEAD: u64 = 4096;

impl Spans {
    /// The bytes of title memory a placed texture with the resource
    /// description `desc` takes (`crate::resource::placed_bytes`).
    fn bytes(&mut self, desc: &[u32; 14]) -> Option<u64> {
        if let Some(&known) = self.sizes.get(desc) {
            return known;
        }
        let bytes = crate::resource::placed_bytes(desc);
        self.sizes.insert(*desc, bytes);
        bytes
    }

    /// The starts of the spans that overlap `[start, end)`, each taken to
    /// end where the next one starts: the last span that starts before
    /// `start`, if it reaches past it at full size, and every span that
    /// starts inside.
    fn overlapping(&self, start: u64, end: u64) -> Vec<u64> {
        let before = self
            .by_start
            .range(..start)
            .next_back()
            .filter(|(_, span)| span.0 > start)
            .map(|(at, _)| *at);
        before.into_iter().chain(self.by_start.range(start..end).map(|(at, _)| *at)).collect()
    }

    /// Slot `index` holds a view of a texture in `[start, end)`. `newer`:
    /// the title wrote it now, so the slots whose views cannot be there
    /// with it ([`Rule`]) are stale: they are returned, and forgotten.
    /// Otherwise (a draw made the slot from what it held), and under
    /// [`Rule::Never`], a view that clashes with other spans is left out.
    fn note(&mut self, index: u32, start: u64, end: u64, newer: bool) -> Vec<u32> {
        let same = |span: Option<&(u64, std::collections::BTreeSet<u32>)>| span.is_some_and(|s| s.0 == end);
        if self.by_slot.get(&index) == Some(&start) && same(self.by_start.get(&start)) {
            return Vec::new();
        }
        self.forget(index);
        if end <= start {
            return Vec::new();
        }
        let rule = self.rule;
        // The spans the view cannot be there with.
        let clashing: Vec<u64> = match rule {
            Rule::Start => self
                .by_start
                .range(start.saturating_sub(HEAD - 1)..start.saturating_add(HEAD))
                .map(|(at, _)| *at)
                .collect(),
            Rule::Overlap | Rule::Never => self.overlapping(start, end),
        };
        let mut stale = Vec::new();
        for at in clashing {
            if at == start && same(self.by_start.get(&at)) {
                continue;
            }
            if !newer || rule == Rule::Never {
                return stale;
            }
            if let Some((old_end, slots)) = self.by_start.remove(&at) {
                // The first of them are said with both spans, to tell a
                // replacement from a neighbour in one launch's log.
                static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 48 {
                    tracing::info!(
                        slot = index,
                        start = format_args!("{start:#x}"),
                        end = format_args!("{end:#x}"),
                        stale_start = format_args!("{at:#x}"),
                        stale_end = format_args!("{old_end:#x}"),
                        stale_slots = slots.len(),
                        rule = ?rule,
                        "a placed texture view makes older views of other memory stale"
                    );
                }
                for slot in slots {
                    self.by_slot.remove(&slot);
                    stale.push(slot);
                }
            }
        }
        self.by_start.entry(start).or_insert_with(|| (end, Default::default())).1.insert(index);
        self.by_slot.insert(index, start);
        stale
    }

    /// The view in slot `index` starts at level `first` of its texture.
    fn shows(&mut self, index: u32, first: u32) {
        self.first.insert(index, first);
    }

    /// The lowest level a view of the texture at `start` starts at.
    fn first_viewed(&self, start: u64) -> Option<u32> {
        self.by_start.get(&start)?.1.iter().filter_map(|slot| self.first.get(slot)).min().copied()
    }

    /// The first span that starts inside `[start, end)` past the first
    /// [`HEAD`] bytes of it.
    fn next_start(&self, start: u64, end: u64) -> Option<u64> {
        self.by_start.range(start.saturating_add(HEAD)..end).next().map(|(at, _)| *at)
    }

    /// The slots that hold a view of the texture at `start`.
    fn slots_at(&self, start: u64) -> Vec<u32> {
        self.by_start.get(&start).map(|span| span.1.iter().copied().collect()).unwrap_or_default()
    }

    /// Slot `index` holds no placed texture view any more.
    fn forget(&mut self, index: u32) {
        self.first.remove(&index);
        let Some(start) = self.by_slot.remove(&index) else {
            return;
        };
        if let Some(span) = self.by_start.get_mut(&start) {
            span.1.remove(&index);
            if span.1.is_empty() {
                self.by_start.remove(&start);
            }
        }
    }

    /// The slots whose texture's memory overlaps `[start, end)`.
    fn slots_in(&self, start: u64, end: u64) -> Vec<u32> {
        self.overlapping(start, end)
            .iter()
            .filter_map(|at| self.by_start.get(at))
            .flat_map(|span| span.1.iter().copied())
            .collect()
    }

    /// The slots of the textures that hold bytes of `[start, end)`, for a
    /// write there: as [`Spans::slots_in`], but when a texture starts at
    /// `start` itself the one before it is left out (it ends where that
    /// one starts, and a request that fills a texture begins at its first
    /// byte: `slots_in` would name its neighbour with every such write).
    fn slots_holding(&self, start: u64, end: u64) -> Vec<u32> {
        if end <= start {
            return Vec::new();
        }
        let before = if self.by_start.contains_key(&start) {
            None
        } else {
            self.by_start.range(..start).next_back().filter(|(_, span)| span.0 > start).map(|(_, span)| span)
        };
        before
            .into_iter()
            .chain(self.by_start.range(start..end).map(|(_, span)| span))
            .flat_map(|span| span.1.iter().copied())
            .collect()
    }
}

/// Title memory written behind the guards' back whose bytes are there
/// (DirectStorage's reads into physical pages, at the addresses the title
/// has them mapped; `crate::watch::title_memory_written`), since a texture
/// pass last took the ranges ([`page_written`]). Kept apart from
/// [`REMAPPED`] so that what such writes do is counted by itself.
static PAGE_WRITTEN: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());

/// Since the start: such ranges kept for a pass; ranges dropped because no
/// pass took them; slots of placed textures that held bytes of them when a
/// pass took them; of those, the slots whose texture had been uploaded
/// before the write (out of the queue, or in it after an upload): they are
/// the ones that kept the bytes from before, and the ones made due when
/// `VICTORY_PAGE_WRITES` asks for it.
pub(crate) static PAGE_WRITES: [std::sync::atomic::AtomicU64; 4] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Frames from the pass that takes a written range to the look at its
/// slots: the levels of one texture arrive in a few requests, and a look
/// two frames on takes those of the next frame with it.
const PAGE_WRITE_AFTER: u64 = 2;

/// The bytes of a write the guards do not see are in title memory `[at,
/// at + len)`: the placed textures there changed with no descriptor
/// written and nothing mapped. Kept for the next texture pass
/// ([`HostHeap::texture_slots_page_written`]), as [`remapped`] keeps its
/// ranges; locks one mutex and calls nothing.
pub(crate) fn page_written(at: usize, len: usize) {
    use std::sync::atomic::Ordering::Relaxed;
    /// Ranges kept until a pass takes them.
    const MOST: usize = 1 << 15;
    let (start, end) = (at as u64, at.saturating_add(len) as u64);
    if end <= SPANNED[0].load(Relaxed) || start >= SPANNED[1].load(Relaxed) {
        return;
    }
    let mut ranges = PAGE_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    PAGE_WRITES[0].fetch_add(1, Relaxed);
    if let Some(last) = ranges.last_mut()
        && last.1 == start
    {
        last.1 = end;
        return;
    }
    if ranges.len() >= MOST {
        PAGE_WRITES[1].fetch_add(ranges.len() as u64, Relaxed);
        ranges.clear();
    }
    ranges.push((start, end));
}

/// Title memory the runtime mapped, unmapped or is about to write since a
/// texture pass last took the ranges ([`remapped`]).
static REMAPPED: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());

/// The lowest start and the highest end of any span of [`Spans`] so far
/// (the start is `u64::MAX` until there is one): what the runtime does to
/// memory outside is none of the texture pass's business.
static SPANNED: [std::sync::atomic::AtomicU64; 2] = [
    std::sync::atomic::AtomicU64::new(u64::MAX),
    std::sync::atomic::AtomicU64::new(0),
];

/// Since the start: slots whose view a newer view of other memory there
/// made stale; ranges of memory noted as mapped, unmapped or written;
/// ranges dropped because no pass took them; slots made due by such
/// ranges; and made slots whose memory was found unmapped (made null).
pub(crate) static STALE: [std::sync::atomic::AtomicU64; 5] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// When a view the title writes makes older views of other memory stale
/// ([`Rule`]): where it starts in the block of their smallest levels, by
/// default; `VICTORY_STALE_TEXTURES=0`: never, as before;
/// `VICTORY_STALE_TEXTURES=overlap`: wherever their spans overlap, the rule
/// until run 188.
fn stale_rule() -> Rule {
    static RULE: std::sync::OnceLock<Rule> = std::sync::OnceLock::new();
    *RULE.get_or_init(|| match std::env::var("VICTORY_STALE_TEXTURES").as_deref() {
        Ok("0") => Rule::Never,
        Ok("overlap") => Rule::Overlap,
        _ => Rule::Start,
    })
}

/// Keep the producer cheap while adjacent pages or repeated writes arrive.
/// A full queue falls back to checking all tracked views once, so completed
/// uploads are still invalidated when a remap arrives during a long pause.
fn record_remap(ranges: &mut Vec<(u64, u64)>, start: u64, end: u64, limit: usize) -> bool {
    if start >= end {
        return false;
    }
    if let Some(last) = ranges.last_mut()
        && last.0 <= end
        && start <= last.1
    {
        last.0 = last.0.min(start);
        last.1 = last.1.max(end);
    } else if ranges.len() >= limit {
        ranges.clear();
        ranges.push((0, u64::MAX));
        return true;
    } else {
        ranges.push((start, end));
    }
    false
}

/// Before looking up descriptor spans, merge notifications from interleaved
/// producers. The consumer cares only which addresses changed in this batch.
fn merge_remaps(ranges: &mut Vec<(u64, u64)>) {
    ranges.sort_unstable();
    ranges.dedup_by(|next, previous| {
        if next.0 > previous.1 {
            return false;
        }
        previous.1 = previous.1.max(next.1);
        true
    });
}

/// The runtime mapped or unmapped title memory `[at, at + len)`, or is
/// about to write it through the kernel (`crate::watch`): the placed
/// textures there may have arrived, gone, or changed with no descriptor
/// written. The range is kept for the next texture pass
/// ([`HostHeap::texture_slots_remapped`]). Called with the runtime's own
/// locks held: this locks one mutex of its own and calls nothing.
pub(crate) fn remapped(at: usize, len: usize) {
    use std::sync::atomic::Ordering::Relaxed;
    /// Ranges kept until a pass takes them.
    const MOST: usize = 1 << 15;
    let (start, end) = (at as u64, at.saturating_add(len) as u64);
    if end <= SPANNED[0].load(Relaxed) || start >= SPANNED[1].load(Relaxed) {
        return;
    }
    let mut ranges = REMAPPED.lock().unwrap_or_else(|e| e.into_inner());
    STALE[1].fetch_add(1, Relaxed);
    if record_remap(&mut ranges, start, end, MOST) {
        STALE[2].fetch_add(1, Relaxed);
    }
}

/// Frames for which a slot a draw read through a table counts as a table's
/// slot ([`HostHeap::tabled`]).
const TABLED_FRAMES: u64 = 240;
/// Number of 32-byte view descriptors per 64 KiB write-watch page.
const PAGE_TABLED_SLOTS: u32 = (crate::watch::PAGE / 32) as u32;

/// Frames a write of a table's slot waits in the queue for the draw that
/// reads it, before the slot is taken for one shaders pick by index.
/// Frames as `crate::frame::current` counts them: in its opening cutscene
/// The title begins two or three of them per frame it presents
/// (435 begun in the 353 s of run 52's, for some 140 presented).
const TABLED_GRACE: u64 = 8;

/// In [`HostHeap::tabled`]: a draw read the slot in that frame.
const TABLED_READ: u32 = 1 << 31;

/// Since the start: writes of texture views into slots that count as a
/// table's (due later), and slots that left the queue because a draw read
/// them through a table.
pub(crate) static TABLED_WRITES: [std::sync::atomic::AtomicU64; 2] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Whether slots draws read through tables leave the queue of textures
/// picked by index; `VICTORY_TABLED_TEXTURES=1` keeps them in it as
/// before.
fn tabled_out() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TABLED_TEXTURES").is_none_or(|v| v != "1"))
}

/// What the queue of textures picked by index knows of a title slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Waiting {
    /// The frame from which the slot is due.
    due: u64,
    /// The frame it was written in.
    written: u64,
    /// Which write of the queue put it here: the pass hands it back with
    /// what it did, so a slot written again meanwhile stays as written.
    write: u64,
    /// Looks in a row that found nothing to upload (its memory not mapped,
    /// or no more of it than at the last upload).
    looks: u8,
    /// Bytes of its memory that were mapped when it was last uploaded
    /// without all of it there; 0: not uploaded yet.
    mapped: u64,
    /// For a slot uploaded whole and looked at again
    /// ([`TextureQueue::recheck`]): where its memory is (address, bytes),
    /// the fingerprint it had at the upload, how many looks found it
    /// unchanged, and the frame of the upload. A fingerprint of 0: none.
    checked: (u64, u64, u64, u8, u64),
}

thread_local! {
    /// This thread is inside [`HostHeap::publish`].
    static PUBLISHING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The queue of title slots whose texture view waits to be brought up to
/// date for shaders that pick textures by index: which are due, and in
/// which order (no host heap needed, so it is tested without one).
#[derive(Default)]
pub(crate) struct TextureQueue {
    slots: std::collections::BTreeMap<u32, Waiting>,
    writes: u64,
    /// Slots held back because the card had no room for the host copy
    /// they need ([`TextureQueue::hold`]), each with the write it was
    /// held as.
    held: std::collections::BTreeMap<u32, u64>,
}

impl TextureQueue {
    /// A texture view was written into slot `index` in `frame`, due from
    /// frame `due`. `seen`: it was not written, a draw or the pass made
    /// the host slot from a view the queue may hold already, which then
    /// stays as it is.
    fn written(&mut self, index: u32, frame: u64, due: u64, seen: bool) {
        if seen && self.slots.contains_key(&index) {
            return;
        }
        self.writes += 1;
        let write = self.writes;
        self.slots.insert(index, Waiting { due, written: frame, write, looks: 0, mapped: 0, checked: (0, 0, 0, 0, 0) });
    }

    fn remove(&mut self, index: u32) {
        self.slots.remove(&index);
    }

    /// Up to `max` slots due at `frame`, each with its write number, in
    /// the order to serve them:
    /// - first the ones looked at before (their memory was not mapped, or
    ///   only part of it): the look is a query, and memory that has
    ///   arrived since is what the title streamed in last;
    /// - then the others, the newest write first and the oldest next, in
    ///   turns: slot order served the low slots of a backlog of
    ///   25,000-40,000 while the textures of the shot on screen waited
    ///   (run 52: 98% of the drawn pixels had no colour texture in the
    ///   first 3D frames).
    fn due(&self, frame: u64, max: usize) -> Vec<(u32, u64)> {
        let mut again: Vec<(u64, u32, u64)> = Vec::new();
        let mut fresh: Vec<(u64, u32, u64)> = Vec::new();
        for (&index, w) in self.slots.iter().filter(|(_, w)| w.due <= frame) {
            let class = if w.looks > 0 || w.mapped > 0 { &mut again } else { &mut fresh };
            class.push((w.written, index, w.write));
        }
        // By the number of the write, counted over the run.
        let newest_first = |a: &(u64, u32, u64), b: &(u64, u32, u64)| b.2.cmp(&a.2);
        again.sort_unstable_by(newest_first);
        fresh.sort_unstable_by(newest_first);
        let mut order: Vec<(u32, u64)> = again.iter().map(|&(_, index, write)| (index, write)).collect();
        let (mut new, mut old) = (0, fresh.len());
        while new < old && order.len() < max {
            order.push((fresh[new].1, fresh[new].2));
            new += 1;
            if new < old && order.len() < max {
                old -= 1;
                order.push((fresh[old].1, fresh[old].2));
            }
        }
        order.truncate(max);
        order
    }

    /// The entry of `index` if it is still the write the pass was handed.
    fn same(&mut self, index: u32, write: u64) -> Option<&mut Waiting> {
        self.slots.get_mut(&index).filter(|w| w.write == write)
    }

    /// Slot `index` is up to date: out of the queue until written again.
    fn done(&mut self, index: u32, write: u64) {
        if self.same(index, write).is_some() {
            self.slots.remove(&index);
        }
    }

    /// Nothing to upload at slot `index` now: looked at again after 1, 2,
    /// 4, ... frames, at most `most`.
    fn later(&mut self, index: u32, write: u64, frame: u64, most: u64) {
        if let Some(w) = self.same(index, write) {
            w.due = frame + (1u64 << w.looks.min(16)).min(most.max(1));
            w.looks = w.looks.saturating_add(1);
        }
    }

    /// The card has no room for the host copy slot `index` needs: it is
    /// not handed out again before frame `due`, or before the card has
    /// room again ([`TextureQueue::room`]). Nothing else of it changes: it
    /// is not a slot "waiting for its memory", and when it is due it has
    /// its old place among the slots never looked at. (It stayed due, so
    /// every pass on a full card looked at the same few hundred slots
    /// again and stopped there, and the slots behind them were never
    /// reached: the ones whose host copy exists and only wants bringing up
    /// to date, and the ones whose memory went, which is what lets host
    /// copies go and makes room.)
    fn hold(&mut self, index: u32, write: u64, due: u64) {
        if let Some(w) = self.same(index, write) {
            w.due = w.due.max(due);
            self.held.insert(index, write);
        }
    }

    /// The card has room again at `frame`: the slots held back for lack of
    /// it are due now, as if they had never been held. Returns how many.
    fn room(&mut self, frame: u64) -> usize {
        let held = std::mem::take(&mut self.held);
        let mut back = 0;
        for (index, write) in held {
            if let Some(w) = self.same(index, write) {
                w.due = w.due.min(frame);
                back += 1;
            }
        }
        back
    }

    /// Slot `index` was uploaded with `mapped` bytes of its memory there,
    /// not all of it: looked at again from frame `due`.
    fn partly(&mut self, index: u32, write: u64, due: u64, mapped: u64) {
        if let Some(w) = self.same(index, write) {
            (w.due, w.looks, w.mapped) = (due, 0, mapped.max(1));
        }
    }

    /// Slot `index` was uploaded whole from memory `at` (address, bytes)
    /// with fingerprint `print`: looked at again from frame `due`, for a
    /// title that fills a texture's memory after it wrote the view (the
    /// `looks`-th such look; `uploaded` is the upload's frame).
    #[allow(clippy::too_many_arguments)]
    fn recheck(&mut self, index: u32, write: u64, due: u64, at: (u64, u64), print: u64, looks: u8, uploaded: u64) {
        if let Some(w) = self.same(index, write) {
            (w.due, w.looks, w.mapped) = (due, 1, 0);
            w.checked = (at.0, at.1, print, looks, uploaded);
        }
    }

    /// What [`TextureQueue::recheck`] noted of slot `index`, if it is
    /// waiting for such a look.
    fn checked(&self, index: u32) -> Option<(u64, u64, u64, u8, u64)> {
        self.slots.get(&index).map(|w| w.checked).filter(|c| c.2 != 0)
    }

    /// The memory slot `index`'s texture lies in was mapped, unmapped or
    /// written in `frame`: the slot is looked at from frame `due` as one
    /// written now, whatever was known of it (a pass at work on it does
    /// not take it out).
    fn changed(&mut self, index: u32, frame: u64, due: u64) {
        self.writes += 1;
        let write = self.writes;
        let due = self.slots.get(&index).map_or(due, |w| w.due.min(due));
        self.slots.insert(index, Waiting { due, written: frame, write, looks: 0, mapped: 0, checked: (0, 0, 0, 0, 0) });
    }

    /// Bytes of slot `index`'s memory that were mapped at its last upload
    /// (0: not uploaded, or all of it was).
    fn mapped(&self, index: u32) -> u64 {
        self.slots.get(&index).map_or(0, |w| w.mapped)
    }

    /// For the frame line: slots in the queue; of them due at `frame`;
    /// waiting for their memory (looked at, never uploaded); uploaded in
    /// part; and the frames the oldest due one has waited since its write.
    fn census(&self, frame: u64) -> [u64; 5] {
        let mut c = [self.slots.len() as u64, 0, 0, 0, 0];
        for w in self.slots.values() {
            if w.mapped > 0 {
                c[3] += 1;
            } else if w.looks > 0 {
                c[2] += 1;
            }
            if w.due <= frame {
                c[1] += 1;
                c[4] = c[4].max(frame.saturating_sub(w.written));
            }
        }
        c
    }
}

/// Whether `view` is of a texture the title may fill itself: a texture it
/// placed in memory, or a texture object.
pub(crate) fn is_texture_view(view: &View) -> bool {
    match view {
        View::Placed(false, resource, _) => resource[0] != 1,
        View::Shader(r, _) => crate::resource::buffer_bytes(*r as crate::object::This).is_none(),
        _ => false,
    }
}

/// Buffer views left null because the host would refuse them
/// ([`buffer_view_ok`]), since the start.
pub(crate) static REFUSED_VIEWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether the host device is gone, and why (the first time it is asked
/// after that): for passes that make many views, to name the one that did
/// it.
pub(crate) fn device_removed() -> Option<String> {
    let host = host()?;
    // SAFETY: a plain query.
    let reason = unsafe { host.device.GetDeviceRemovedReason() };
    reason.err().map(|e| e.to_string())
}

/// Writes of title view-heap slots with a placed texture view and with a
/// view of a texture object, since the start (diagnostic counts).
pub(crate) static PLACED_TEXTURE_SLOTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(crate) static OBJECT_TEXTURE_SLOTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Bytes per element of a buffer view, as the console's descriptor holds
/// them for indexed reads (structured: the stride; raw: 1, the index is the
/// byte; typed: the format's); `None` if `view` is not a buffer view.
fn view_stride(view: &View) -> Option<u32> {
    const BUFFER: u32 = 1;
    let (words, unordered): (&[u32; 10], bool) = match view {
        View::Shader(_, Some(d)) if d[1] == BUFFER => (d, false),
        View::Unordered(_, _, Some(d)) if d[1] == BUFFER => (d, true),
        View::Placed(unordered, resource, d) if resource[0] == BUFFER => {
            (d[..10].try_into().ok()?, *unordered)
        }
        _ => return None,
    };
    // SAFETY: buffer views' union members.
    let (stride, raw, format) = unsafe {
        if unordered {
            let d = uav_desc(words);
            let b = d.Anonymous.Buffer;
            (b.StructureByteStride, b.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0, d.Format)
        } else {
            let d = srv_desc(words);
            let b = d.Anonymous.Buffer;
            (b.StructureByteStride, b.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0, d.Format)
        }
    };
    Some(if stride > 0 {
        stride
    } else if raw {
        1
    } else {
        crate::resource::element(format.0 as u32).map_or(0, |(bytes, _)| bytes as u32)
    })
}

/// Host heaps' GPU ranges (start, bytes, bytes per slot, sampler), for
/// device-removed reports: a shader indexing a heap past its end faults.
static GPU_RANGES: Mutex<Vec<(u64, u64, u64, bool)>> = Mutex::new(Vec::new());

/// For device-removed reports: where host GPU address `va` lies against
/// the host descriptor heaps.
pub(crate) fn near_heaps(va: u64) -> String {
    let ranges = GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let mut text = String::new();
    for &(start, bytes, step, sampler) in ranges.iter() {
        let kind = if sampler { "sampler" } else { "view" };
        let place = if va < start {
            format!("{:#x} below", start - va)
        } else if va < start + bytes {
            format!("inside, slot {}", (va - start) / step)
        } else {
            format!("{:#x} past its end (slot {})", va - start - bytes, (va - start) / step)
        };
        text.push_str(&format!("; {kind} heap {start:#x}+{bytes:#x}: {place}"));
    }
    text
}

/// The host's limit for shader-visible view heaps on resource binding
/// tiers 1 and 2; tier 3 hardware may take more.
const PORTABLE_VIEWS: u32 = 1_000_000;

/// The size of every host view heap on a host that takes more than
/// [`PORTABLE_VIEWS`] (AMD): the title's heap (a little over 1,000,000
/// views in the retail game) plus [`DYNAMIC`] fits with room to spare, and
/// the last slot is the null slot every view index at or past it reads.
const UNCAPPED_VIEWS: u32 = 1 << 21;

/// Picks, once, the null slot at the end of every host view heap and tells
/// the translator now (`set_view_heap_limit`) instead of when the title's
/// heap is made: a shader keeps the limit it was translated with, and the
/// title makes 25 pipelines (their shaders translated at once) before it
/// makes its view heap. Called before a shader is translated and before a
/// view heap is made.
///
/// Every host clamps. A host that caps view heaps at [`PORTABLE_VIEWS`]
/// (NVIDIA) hung on a view index past the heap; one that does not (AMD
/// RDNA3, RX 7900 XT) took a page fault and removed the device on entering
/// the world: a compute program read view slot 16,777,088 (0xffff80) of a
/// heap of 1,507,328, past its end. Unclamped, that index is undefined on
/// any host.
///
/// A capped host also gets a heap map (VIcky; unless `VICTORY_HEAP_MAP=0`):
/// the translator looks bindless views past the identity window up in it
/// (`set_view_heap_map`), and from then on every view heap of more title
/// slots than the window is laid out for it ([`MapLayout`], whatever its
/// size: the map's slot is fixed in the translations; its guard slot is
/// this null slot). The probe asks for [`UNCAPPED_VIEWS`] (2^21, the
/// translator's whole index range), so a host that takes 1,000,001 views
/// but not the title's heap (an RTX 4060 Laptop) gets the map too.
pub(crate) fn probe_view_limit() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let Some(host) = host() else { return };
        let fits = |count: u32| {
            // SAFETY: plain heap creation; the heap is released at once.
            let made: windows::core::Result<ID3D12DescriptorHeap> = unsafe {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                        NumDescriptors: count,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
            };
            made.is_ok()
        };
        let views = if fits(UNCAPPED_VIEWS) {
            UNCAPPED_VIEWS
        } else {
            PORTABLE_VIEWS
        };
        tracing::info!(
            views,
            "host view heaps hold {views} views; view indices are clamped to the null slot at the end from the first shader on"
        );
        victory_shader::emit::set_view_heap_limit(views - 1);
        if views == PORTABLE_VIEWS {
            set_capped_heap_map();
        }
    });
}

/// Sets the translator's heap map for a host view heap capped at
/// [`PORTABLE_VIEWS`], unless `VICTORY_HEAP_MAP=0`.
fn set_capped_heap_map() {
    if !heap_map_on() {
        return;
    }
    let window = window();
    victory_shader::emit::set_view_heap_map(Some((window, MAP_SLOT)));
    tracing::info!(
        window,
        map_slot = MAP_SLOT,
        "heap map: bindless views past the identity window are looked up (VICTORY_HEAP_MAP=0: off)"
    );
}

/// The offline warm-up (`victory translate --warm`, no device): translated
/// with the clamp of a capped host (`VICTORY_VIEW_LIMIT=999999`, which the
/// run's log gives), shaders get the heap map a run on that host has too,
/// so that the run finds them under its own cache key. Does nothing for
/// any other limit, or once a map is set.
pub(crate) fn offline_heap_map() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if victory_shader::emit::view_heap_limit() == PORTABLE_VIEWS - 1
            && victory_shader::emit::view_heap_map().is_none()
        {
            set_capped_heap_map();
        }
    });
}

/// Slots past the title's in a view heap, for tables a draw needs made
/// afresh (constant buffers copied per draw). A list holds the slots it
/// took until it is reset (its submissions have run) or released.
/// 2^16 ran out in the world ("no free dynamic descriptor slots", run 11;
/// upstream PR #336 takes 2^18). On a host heap capped at 1,000,000 views
/// (NVIDIA) these come out of the title's slots, whose highest seen is
/// about 720,000: 2^17 left 868,000 of them, and ran out in the opening
/// cutscene (a frame asks for up to 16,000 tables while the frame before
/// still holds its own; a draw refused its table read slots never made
/// for it, and the card stopped within frames). 2^18 leaves 737,000; a
/// table past them is made afresh in these slots like any other.
pub(crate) const DYNAMIC: u32 = 1 << 18;

// The heap map leaves the fork's dynamic slots and guard slot in place.
const _: () = assert!(MAP_SLOT + 1 + DYNAMIC == PORTABLE_VIEWS - 1);

impl HostHeap {
    pub(crate) fn new(sampler: bool, count: u32) -> Option<Self> {
        let host = host()?;
        let kind = if sampler {
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
        } else {
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
        };
        let make = |count: u32| -> windows::core::Result<ID3D12DescriptorHeap> {
            // SAFETY: plain heap creation.
            unsafe {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: count,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
            }
        };
        let extra = if sampler { 0 } else { DYNAMIC };
        let title_count = count;
        // The title's slot count, before the host's cap or test limit may leave fewer.
        let asked = count;
        // A view heap ends at the null slot shaders clamp view indices to
        // (`probe_view_limit`), whatever the title's own count: the title's
        // slots, then the dynamic ones, then null up to and with that slot.
        // Also whether the translator looks bindless views up in a heap map,
        // as it does from the first translation on.
        if !sampler {
            probe_view_limit();
        }
        let null = if sampler {
            0
        } else {
            victory_shader::emit::view_heap_limit()
        };
        let layout = (!sampler)
            .then(heap_map)
            .flatten()
            .filter(|&(window, _)| count > window)
            .map(|(window, slot)| MapLayout::new(count, window, slot));
        let (heap, count, size) = if let Some(l) = layout {
            // The map is set on a capped host only, whose null slot is the
            // layout's guard (`MAP_SLOT + 1 + DYNAMIC`).
            #[cfg(not(test))]
            debug_assert_eq!(l.guard(), null, "the heap map's guard is the clamp's null slot");
            tracing::warn!(
                count,
                window = l.window,
                pool = l.pool,
                map_slot = l.slot,
                "host view heap with a heap map: title slots past the window that hold views get pool slots"
            );
            match make(l.size()) {
                Ok(heap) => (heap, l.window, l.size()),
                Err(e) => {
                    tracing::warn!(error = %e, count, "no host view heap with a heap map");
                    return None;
                }
            }
        } else if null == 0 {
            match make(count + extra) {
                Ok(heap) => (heap, count, count + extra),
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        } else {
            let size = null + 1;
            let fitted = if count + extra <= null {
                count
            } else {
                // shortcut: descriptors past the host's limit are left out; a title that reaches
                // them needs remapping. They read the null slot instead.
                tracing::warn!(count, "host view heap capped at {size}; views past it read a null slot");
                null - extra
            };
            match make(size) {
                Ok(heap) => (heap, fitted, size),
                Err(e) => {
                    tracing::warn!(error = %e, count, size, "no host view heap");
                    return None;
                }
            }
        };
        // SAFETY: a live heap.
        let (cpu, step) = unsafe {
            (
                heap.GetCPUDescriptorHandleForHeapStart().ptr,
                host.device.GetDescriptorHandleIncrementSize(kind),
            )
        };
        tracing::debug!(count, sampler, "host descriptor heap");
        // Tests: a heap that keeps fewer title slots than the title has, as a
        // capped one does (its slots past them are dynamic ones).
        #[cfg(test)]
        let count = match TEST_TITLE_SLOTS.get() {
            0 => count,
            kept if !sampler => kept.min(count),
            _ => count,
        };
        let overflow = (!sampler && asked > count && overflow_views())
            .then(|| Overflow::new(&host.device, asked - count, step))
            .flatten();
        // SAFETY: a live shader-visible heap.
        let gpu = unsafe { heap.GetGPUDescriptorHandleForHeapStart().ptr };
        GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner()).push((
            gpu,
            u64::from(size) * u64::from(step),
            u64::from(step),
            sampler,
        ));
        // The heap map: an entry per title slot past the window, zeroed (no
        // slot: the guard).
        let map = layout.and_then(|l| {
            let bytes = u64::from((title_count - l.window).max(1)) * 4;
            let Some((buffer, at)) = host.upload_buffer(bytes) else {
                tracing::warn!(bytes, "no heap map: title slots past the window read the null slot");
                return None;
            };
            // SAFETY: the buffer's whole mapping.
            unsafe { std::ptr::write_bytes(at, 0, bytes as usize) };
            crate::host::name(&buffer, || "heap map".to_owned());
            Some(HeapMap { buffer, at: at as usize })
        });
        let strides = (!sampler)
            .then(|| {
                let bytes = u64::from(victory_shader::emit::SLOT_STRIDES_COUNT) * 4;
                let (buffer, data) = host.upload_buffer(bytes)?;
                // SAFETY: the buffer's whole mapping; a live buffer.
                let gpu = unsafe {
                    std::ptr::write_bytes(data, 0, bytes as usize);
                    buffer.GetGPUVirtualAddress()
                };
                crate::host::name(&buffer, || "element strides by view heap slot".to_owned());
                Some((buffer, gpu, data as usize))
            })
            .flatten();
        // Title-keyed state covers the title's slots past the window too.
        let (pool, dynamic_first, titled) = match layout {
            Some(l) => (l.pool, l.dynamic_first(), title_count),
            None => (0, count, count),
        };
        let h = HostHeap {
            heap,
            cpu,
            gpu,
            step,
            count,
            dynamic: extra,
            dynamic_first,
            title_count,
            sampler,
            made: Mutex::new(Made {
                keys: vec![0; (count + pool) as usize],
                map: map
                    .is_some()
                    .then(|| SlotMap::new(count, title_count, count, pool)),
            }),
            made_fast: (0..count)
                .map(|_| std::sync::atomic::AtomicU64::new(0))
                .collect(),
            copied: std::sync::OnceLock::new(),
            map,
            title_start: AtomicU64::new(0),
            reclaimed: AtomicU64::new(u64::MAX),
            kept: (0..KEPT_PIECES)
                .map(|_| {
                    Mutex::new(vec![
                        None;
                        (dynamic_first + extra).div_ceil(KEPT_PIECES) as usize
                    ])
                })
                .collect(),
            kept_one: Mutex::new(()),
            next: Mutex::new((0, vec![false; DYNAMIC as usize])),
            free: std::sync::atomic::AtomicU32::new(DYNAMIC),
            too_long: std::sync::atomic::AtomicU32::new(u32::MAX),
            strides,
            buffers: std::array::from_fn(|_| Mutex::default()),
            changed: Mutex::default(),
            textures: Mutex::default(),
            tabled: (0..if sampler { 0 } else { titled })
                .map(|_| std::sync::atomic::AtomicU32::new(0))
                .collect(),
            page_tabled_min: (0..if sampler {
                0
            } else {
                titled.div_ceil(PAGE_TABLED_SLOTS)
            })
            .map(|_| std::sync::atomic::AtomicU32::new(0))
            .collect(),
            placed: {
                let spans = std::sync::Arc::<Mutex<Spans>>::default();
                if !sampler {
                    PLACED_SPANS
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .push(std::sync::Arc::downgrade(&spans));
                }
                spans
            },
            overflow,
            buffer_sweep: crate::draw::BufferSweep::default(),
        };
        // A slot never written is undefined: shaders reading one (a slot
        // the title wrote in a way we do not know, or a table entry no
        // used-resource entry names) read a wild address. Every slot starts
        // null instead.
        let started = std::time::Instant::now();
        if let Err(e) = h.fill_null(kind, size) {
            tracing::warn!(error = %e, "host descriptor heap not made null");
        }
        tracing::debug!(ms = started.elapsed().as_millis(), "host descriptor heap made null");
        if let Some(l) = layout {
            // The map's raw view (`raw_at` holds the buffer), or one that
            // reads zeros: every lookup then gives the guard slot.
            match &h.map {
                Some(m) => h.raw_at(l.slot, &m.buffer, 0, u64::from(title_count - l.window) * 4),
                None => h.null_raw_at(l.slot),
            }
        }
        Some(h)
    }

    /// Makes the first `total` slots null: one null slot in a CPU-side heap,
    /// doubled by copies to a block, the block copied over the heap.
    fn fill_null(&self, kind: D3D12_DESCRIPTOR_HEAP_TYPE, total: u32) -> windows::core::Result<()> {
        const BLOCK: u32 = 1 << 16;
        let Some(host) = host() else { return Ok(()) };
        // SAFETY: plain heap creation and copies inside live heaps.
        unsafe {
            let block: ID3D12DescriptorHeap =
                host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: BLOCK,
                    ..Default::default()
                })?;
            let start = block.GetCPUDescriptorHandleForHeapStart();
            let at = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: start.ptr + (i * self.step) as usize,
            };
            if self.sampler {
                null_sampler(start);
            } else {
                null_view(start);
            }
            let mut filled = 1;
            while filled < BLOCK {
                let n = filled.min(BLOCK - filled);
                host.device.CopyDescriptorsSimple(n, at(filled), start, kind);
                filled += n;
            }
            let mut done = 0;
            while done < total {
                let n = BLOCK.min(total - done);
                host.device.CopyDescriptorsSimple(n, self.cpu(done), start, kind);
                done += n;
            }
        }
        Ok(())
    }

    /// A plain point sampler at slot `index`, for sampler slots without a
    /// view.
    fn null_sampler(&self, index: u32) {
        null_sampler(self.cpu(index));
    }
}

/// A plain point sampler at `at`.
fn null_sampler(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        ..Default::default()
    };
    // SAFETY: a slot of a sampler heap of ours.
    unsafe { host.device.CreateSampler(&desc, at) };
}

impl HostHeap {
    /// `n` consecutive free dynamic slots (view heaps only), taken until
    /// [`HostHeap::release`]: the first's index.
    pub(crate) fn dynamic(&self, n: u32) -> Option<u32> {
        if self.sampler || n == 0 || n > self.dynamic {
            return None;
        }
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::DynamicAsked, 1);
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let (next, taken) = &mut *guard;
        let free = self.free.load(std::sync::atomic::Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        DYNAMIC_LOW.fetch_min(free, std::sync::atomic::Ordering::Relaxed);
        let room = free >= n;
        let enough = room && n < self.too_long.load(std::sync::atomic::Ordering::Relaxed);
        let n = n as usize;
        // Search from `next` on, then from the start (extending past cursor by
        // n - 1 so contiguous runs crossing the cursor are not missed).
        let cursor = *next as usize;
        let end = self.dynamic as usize;
        let second_end = cursor.saturating_add(n.saturating_sub(1)).min(end);
        let starts = (cursor..end).chain(0..second_end);
        let mut run = 0;
        let mut found = None;
        let mut last = usize::MAX;
        // Not capped at `DYNAMIC`: that would stop the scan at the cursor
        // again and miss the runs crossing it.
        for (step, at) in starts.enumerate().take(if enough { usize::MAX } else { 0 }) {
            // A run cannot wrap past the end.
            if at != last.wrapping_add(1) {
                run = 0;
            }
            last = at;
            run = if taken[at] { 0 } else { run + 1 };
            if run == n {
                found = Some(at + 1 - n);
                // Past `next` in the second part: a run across it.
                #[cfg(feature = "diagnostics")]
                if step >= end {
                    DYNAMIC_WHY[3].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                #[cfg(not(feature = "diagnostics"))]
                let _ = step;
                break;
            }
        }
        let Some(first) = found else {
            if enough {
                self.too_long.fetch_min(n as u32, std::sync::atomic::Ordering::Relaxed);
            }
            #[cfg(feature = "diagnostics")]
            DYNAMIC_WHY[if !room {
                0
            } else if !enough {
                1
            } else {
                2
            }]
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(n, "no free dynamic descriptor slots: a table is not made"));
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                add(C::DynamicRefused, 1);
                add(C::DynamicNs, started.elapsed().as_nanos() as u64);
            }
            return None;
        };
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::DynamicNs, started.elapsed().as_nanos() as u64);
        taken[first..first + n].fill(true);
        self.free.fetch_sub(n as u32, std::sync::atomic::Ordering::Relaxed);
        *next = ((first + n) % DYNAMIC as usize) as u32;
        Some(self.dynamic_first + first as u32)
    }

    /// The title slots this heap mirrors at their own index (tables read in
    /// place reach only these: one that reaches past them is made afresh);
    /// fewer than the title's when the host caps its size, and with a heap
    /// map the identity window.
    pub(crate) fn count(&self) -> u32 {
        self.count
    }

    /// The title slots this heap mirrors: `title_count` when a heap map
    /// covers slots past the identity window, `count` otherwise.
    pub(crate) fn mirrored_count(&self) -> u32 {
        if self.map.is_some() { self.title_count } else { self.count }
    }

    /// The host slot title slot `title` is made in (`made` locked): its own
    /// index below `count`; with a heap map past it, its pool slot, taken
    /// now if it has none (then `true`: [`HostHeap::publish_map`] writes its
    /// map entry once the slot is made). `None` past the title's slots, or
    /// with the pool used up even after [`HostHeap::reclaim`] (the title
    /// slot then reads the guard slot).
    fn host_slot(&self, made: &mut Made, title: u32) -> Option<(u32, bool)> {
        if title < self.count {
            return Some((title, false));
        }
        if title >= self.title_count {
            return None;
        }
        {
            let map = made.map.as_mut()?;
            if let Some(host) = map.host(title) {
                return Some((host, false));
            }
            if let Some(host) = map.assign(title, crate::host::submissions().1) {
                return Some((host, true));
            }
        }
        if self.reclaim(made) > 0
            && let Some(host) = made.map.as_mut()?.assign(title, crate::host::submissions().1)
        {
            return Some((host, true));
        }
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::warn!(title, "heap map pool used up: title slots past the window read the null slot");
        });
        None
    }

    /// Writes title slot `title`'s heap map entry (`made` locked), after
    /// its pool slot was made.
    fn publish_map(&self, made: &Made, title: u32) {
        let (Some(map), Some(m)) = (&made.map, &self.map) else {
            return;
        };
        let Some(i) = title.checked_sub(map.window()).filter(|&i| i < map.len()) else {
            return;
        };
        // The descriptor first, then the entry naming it.
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        // SAFETY: entry `i` of the mapped heap map.
        unsafe { (m.at as *mut u32).add(i as usize).write_volatile(map.entry(title)) };
    }

    /// Gives back title slot `title`'s pool slot (`made` locked), if it has
    /// one: its map entry is zeroed (the guard slot), and the slot is
    /// handed out again once the submissions that may have read the old
    /// entry are done. Its view stays, held, until it is made again.
    fn unmap(&self, made: &mut Made, title: u32) {
        let (Some(map), Some(m)) = (made.map.as_mut(), &self.map) else {
            return;
        };
        let Some(host) = map.host(title).filter(|_| title >= map.window()) else {
            return;
        };
        let i = title - map.window();
        // SAFETY: entry `i` of the mapped heap map (`title` is mapped).
        unsafe { (m.at as *mut u32).add(i as usize).write_volatile(0) };
        // The entry is out before the submissions running are counted.
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        map.free(title, crate::host::submissions().0);
        made.keys[host as usize] = 0;
        if let Some(fast) = self.made_fast.get(host as usize) {
            fast.store(0, std::sync::atomic::Ordering::Release);
        }
    }

    /// Title slot `title` holds no view (`made` locked): its host slot reads
    /// nothing and leaves the sets of the slots shaders pick by index (as
    /// the fork did), and a pool slot is given back. `for_draw`: the slot
    /// leaves the sets even if its host slot was never made.
    fn empty_slot(&self, made: &mut Made, title: u32, for_draw: bool, mark_changed: bool) {
        match made.host(title) {
            Some(host) if made.keys[host as usize] != 0 => {
                made.keys[host as usize] = 0;
                if let Some(fast) = self.made_fast.get(host as usize) {
                    fast.store(0, std::sync::atomic::Ordering::Release);
                }
                self.make_null(host);
                let _ = self.note_buffer_in(title, None, false, mark_changed);
            }
            // Left for later and stale since: out of the sets and the stride table.
            _ if for_draw => {
                let _ = self.note_buffer_in(title, None, false, mark_changed);
            }
            _ => {}
        }
        self.unmap(made, title);
    }

    /// With the heap map's pool used up (`made` locked): gives back the pool
    /// slots of title slots that hold no live view any more (a view of a
    /// released resource, or no view), which nothing else finds: the fork
    /// learns of a slot's view only from the title's descriptor writes and
    /// copies, and a title that releases a resource need not write its
    /// slots again. At most once every two seconds; the slots given back
    /// are handed out once the submissions running now are done. Returns
    /// how many.
    fn reclaim(&self, made: &mut Made) -> usize {
        use std::sync::atomic::Ordering::Relaxed;
        const EVERY_MS: u64 = 2000;
        let now = crate::host::now_ms();
        let last = self.reclaimed.load(Relaxed);
        if last != u64::MAX && now < last.saturating_add(EVERY_MS) {
            return 0;
        }
        self.reclaimed.store(now, Relaxed);
        let start = self.title_start.load(Relaxed);
        let Some(map) = made.map.as_ref().filter(|_| start != 0) else {
            return 0;
        };
        let started = std::time::Instant::now();
        let dead: Vec<u32> = map
            .mapped()
            .map(|(title, _)| title)
            .filter(|&title| {
                let handle = start + u64::from(title) * TITLE_SLOT_BYTES;
                // SAFETY: `title < title_count`, a slot of the live title heap.
                let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
                tag != VIEW_TAG || view_at(i, generation).is_none()
            })
            .collect();
        for &title in &dead {
            self.empty_slot(made, title, true, true);
        }
        // The first passes at warn level, the rest at debug.
        static PASSES: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let (given_back, ms) = (dead.len(), started.elapsed().as_millis());
        if PASSES.fetch_add(1, Relaxed) < 16 {
            tracing::warn!(given_back, ms, "heap map pool used up: pool slots of title slots without a live view given back");
        } else {
            tracing::debug!(given_back, ms, "heap map pool used up: pool slots of title slots without a live view given back");
        }
        dead.len()
    }

    /// The GPU address of the table of element strides by title slot (view
    /// heaps).
    pub(crate) fn strides_gpu(&self) -> Option<u64> {
        self.strides.as_ref().map(|s| s.1)
    }

    /// The table's buffer, for a list to hold.
    pub(crate) fn strides_buffer(&self) -> Option<&ID3D12Resource> {
        self.strides.as_ref().map(|s| &s.0)
    }

    /// Up to `max` title slots whose texture view is due at `frame`, each
    /// with its write number, in the order to serve them
    /// ([`TextureQueue::due`]), and the queue's counts
    /// ([`TextureQueue::census`]).
    pub(crate) fn texture_slots_due(&self, frame: u64, max: usize) -> (Vec<(u32, u64)>, [u64; 5]) {
        let textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        (if max == 0 { Vec::new() } else { textures.due(frame, max) }, textures.census(frame))
    }

    /// Every title slot of the queue that is due at `frame`, and whether
    /// it was uploaded before (in part, or whole and waiting for another
    /// look): for the card accounting line
    /// (`crate::resource::card_accounting`).
    pub(crate) fn texture_slots_all_due(&self, frame: u64) -> Vec<(u32, bool)> {
        let textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        textures
            .slots
            .iter()
            .filter(|(_, w)| w.due <= frame)
            .map(|(&index, w)| (index, w.mapped > 0 || w.checked.2 != 0))
            .collect()
    }

    /// The texture view of title slot `index` (as handed out with `write`)
    /// was brought up to date: due again from frame `again` (uploaded with
    /// that many bytes of its memory there, not all), or not until the
    /// slot is written again.
    pub(crate) fn texture_slot_done(&self, index: u32, write: u64, again: Option<(u64, u64)>) {
        let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        match again {
            Some((frame, mapped)) => textures.partly(index, write, frame, mapped),
            None => textures.done(index, write),
        }
    }

    /// Title slot `index` was uploaded whole: see [`TextureQueue::recheck`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn texture_slot_recheck(
        &self,
        index: u32,
        write: u64,
        due: u64,
        at: (u64, u64),
        print: u64,
        looks: u8,
        uploaded: u64,
    ) {
        let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        textures.recheck(index, write, due, at, print, looks, uploaded);
    }

    /// See [`TextureQueue::checked`].
    pub(crate) fn texture_slot_checked(&self, index: u32) -> Option<(u64, u64, u64, u8, u64)> {
        self.textures.lock().unwrap_or_else(|e| e.into_inner()).checked(index)
    }

    /// Nothing to upload at title slot `index` now (its memory is not
    /// mapped, or no more of it than before): looked at again after 1, 2,
    /// 4, ... frames, at most `most`.
    pub(crate) fn texture_slot_later(&self, index: u32, write: u64, frame: u64, most: u64) {
        let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        textures.later(index, write, frame, most);
    }

    /// See [`TextureQueue::hold`].
    pub(crate) fn texture_slot_held(&self, index: u32, write: u64, due: u64) {
        self.textures.lock().unwrap_or_else(|e| e.into_inner()).hold(index, write, due);
    }

    /// See [`TextureQueue::room`].
    pub(crate) fn texture_slots_room(&self, frame: u64) -> usize {
        self.textures.lock().unwrap_or_else(|e| e.into_inner()).room(frame)
    }

    /// Bytes of title slot `index`'s memory that were mapped when it was
    /// last uploaded in part (0: not uploaded, or whole).
    pub(crate) fn texture_slot_mapped(&self, index: u32) -> u64 {
        self.textures.lock().unwrap_or_else(|e| e.into_inner()).mapped(index)
    }

    /// Takes the ranges of title memory the runtime mapped, unmapped or
    /// wrote since the last call ([`remapped`]) and makes the slots whose
    /// placed texture lies there due from the next frame (by when a write
    /// under way has landed): memory that arrived is uploaded, memory that
    /// went makes the slot null ([`HostHeap::texture_slot_gone`]), and a
    /// texture streamed into the place of another of the same description
    /// is uploaded although no descriptor says so. Slots tables read are
    /// left to their draws. Returns how many slots it made due.
    /// shortcut: the ranges are kept in one place for all heaps; the heap
    /// whose pass runs first takes them (a title has one view heap).
    pub(crate) fn texture_slots_remapped(&self, frame: u64) -> usize {
        use std::sync::atomic::Ordering::Relaxed;
        self.texture_slots_page_written(frame);
        let mut ranges = std::mem::take(&mut *REMAPPED.lock().unwrap_or_else(|e| e.into_inner()));
        if ranges.is_empty() {
            return 0;
        }
        merge_remaps(&mut ranges);
        let mut slots: Vec<u32> = {
            let placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
            ranges.iter().flat_map(|&(start, end)| placed.slots_in(start, end)).collect()
        };
        slots.sort_unstable();
        slots.dedup();
        slots.retain(|&slot| !self.tabled_lately(slot, frame));
        let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        for &slot in &slots {
            textures.changed(slot, frame, frame + 1);
        }
        STALE[3].fetch_add(slots.len() as u64, Relaxed);
        slots.len()
    }

    /// Takes the ranges of title memory written behind the guards' back
    /// since the last call ([`page_written`]) and finds the slots whose
    /// placed texture holds bytes of them ([`Spans::slots_holding`]). A
    /// slot whose texture was uploaded before the write shows the bytes
    /// from before: with `VICTORY_PAGE_WRITES` (`crate::watch::page_writes`,
    /// bit 2) it is due again [`PAGE_WRITE_AFTER`] frames on, as memory
    /// mapped under it would make it ([`HostHeap::texture_slots_remapped`]);
    /// its host copy exists, so the upload needs no room on the card. A
    /// slot still waiting for its first upload is left as it is: that
    /// upload takes the bytes that are there by then. Slots tables read
    /// are left to their draws. Counted either way; returns how many slots
    /// it made due.
    fn texture_slots_page_written(&self, frame: u64) -> usize {
        let ranges = std::mem::take(&mut *PAGE_WRITTEN.lock().unwrap_or_else(|e| e.into_inner()));
        if ranges.is_empty() {
            return 0;
        }
        self.slots_written_under(&ranges, frame, crate::watch::page_writes() & 2 != 0)[2]
    }

    /// [`HostHeap::texture_slots_page_written`] for `ranges` (start, end):
    /// the slots that held bytes of them, those of them uploaded before,
    /// and those made due (the latter, if `due`).
    fn slots_written_under(&self, ranges: &[(u64, u64)], frame: u64, due: bool) -> [usize; 3] {
        use std::sync::atomic::Ordering::Relaxed;
        let mut slots: Vec<u32> = {
            let placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
            ranges.iter().flat_map(|&(start, end)| placed.slots_holding(start, end)).collect()
        };
        slots.sort_unstable();
        slots.dedup();
        slots.retain(|&slot| !self.tabled_lately(slot, frame));
        let held = slots.len();
        let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        // Uploaded before: out of the queue, or in it after an upload (in
        // part, or whole and being looked at again).
        slots.retain(|slot| textures.slots.get(slot).is_none_or(|w| w.mapped > 0 || w.checked.2 != 0));
        PAGE_WRITES[2].fetch_add(held as u64, Relaxed);
        PAGE_WRITES[3].fetch_add(slots.len() as u64, Relaxed);
        if !due {
            return [held, slots.len(), 0];
        }
        for &slot in &slots {
            textures.changed(slot, frame, frame + PAGE_WRITE_AFTER);
        }
        [held, slots.len(), slots.len()]
    }

    /// Whether host slot `index` is made from the very record the title
    /// has at `handle` now. Its host texture then exists (a slot keeps what
    /// it views), so bringing that texture up to date takes no memory of
    /// the card.
    pub(crate) fn texture_slot_made_from(&self, index: u32, handle: u64) -> bool {
        // SAFETY: the handle is a descriptor slot in title memory.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        let key = u64::from(generation) << 32 | u64::from(i);
        if tag != VIEW_TAG {
            return false;
        }
        // And the slot holds a host resource: a slot is also noted as made
        // from a view the host could not make (`make` gave nothing, the
        // slot is null and keeps nothing). The host copy of such a
        // texture is held by no slot, may have been let go, and would be
        // made anew here, with the card full. `made` first, then `kept`,
        // as `mirror` takes them.
        // (With a heap map, title slot `index` past the window is made in
        // its pool slot, if it holds one.)
        let made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        let Some(host) = made.host(index) else {
            return false;
        };
        made.keys.get(host as usize).is_some_and(|&made| made != 0 && made == key)
            && self.kept[(host % KEPT_PIECES) as usize]
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get((host / KEPT_PIECES) as usize)
                .is_some_and(Option::is_some)
    }

    /// Levels on demand (`crate::tiles`): makes again the host slots of the
    /// placed textures whose first level with memory changed since the
    /// last call ([`levels_changed`]), each from the view it was made
    /// from, so that it shows the texture from that level ([`make`]). Only
    /// slots that are made; a slot draws read through a table is made by
    /// its draw. Returns how many it made.
    /// shortcut: the starts are kept in one place for all heaps; the heap
    /// whose pass runs first takes them (a title has one view heap).
    /// Levels on demand (`crate::tiles`): the card has room, so the slots
    /// of up to [`Self::WAITING_A_PASS`] placed textures whose upload left
    /// a held level without tiles at least a second ago
    /// ([`levels_waiting`]) are due again from the next frame, as memory
    /// mapped under them would make them
    /// ([`HostHeap::texture_slots_remapped`]): the upload then gives the
    /// level its tiles. To be called only while the card has room: without
    /// it the upload would give nothing again. Slots tables read are left
    /// to their draws (`texture.rs` `upload` asks again for those). Returns
    /// how many slots it made due.
    /// shortcut: kept in one place for all heaps, as the starts below.
    pub(crate) fn texture_slots_waiting(&self, frame: u64) -> usize {
        if self.sampler {
            return 0;
        }
        let starts: Vec<u64> = {
            let mut waiting = WAITING.lock().unwrap_or_else(|e| e.into_inner());
            if waiting.is_empty() {
                return 0;
            }
            let starts: Vec<u64> = waiting
                .iter()
                .filter(|&(_, &since)| crate::host::long_ago(since, 1000, 2))
                .map(|(&start, _)| start)
                .take(Self::WAITING_A_PASS)
                .collect();
            for start in &starts {
                waiting.remove(start);
            }
            starts
        };
        if starts.is_empty() {
            return 0;
        }
        let mut slots: Vec<u32> = {
            let placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
            starts.iter().flat_map(|&start| placed.slots_at(start)).collect()
        };
        slots.sort_unstable();
        slots.dedup();
        slots.retain(|&slot| !self.tabled_lately(slot, frame));
        let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        for &slot in &slots {
            textures.changed(slot, frame, frame + 1);
        }
        crate::tiles::STATS[10].fetch_add(slots.len() as u64, std::sync::atomic::Ordering::Relaxed);
        slots.len()
    }

    /// Textures a pass asks again for their levels
    /// ([`HostHeap::texture_slots_waiting`]): the first ones given tiles
    /// may take the room the others were asked for, and an upload that
    /// gives nothing is a whole texture staged and copied for nothing.
    const WAITING_A_PASS: usize = 128;

    pub(crate) fn texture_views_remade(&self) -> usize {
        if self.sampler {
            return 0;
        }
        let starts = std::mem::take(&mut *RESHOWN.lock().unwrap_or_else(|e| e.into_inner()));
        if starts.is_empty() {
            return 0;
        }
        let slots: Vec<u32> = {
            let placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
            starts.iter().flat_map(|&start| placed.slots_at(start)).collect()
        };
        let made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        let mut remade = 0;
        for slot in slots {
            // Title slot `slot`'s host slot (with a heap map, past the
            // window its pool slot, if it holds one).
            let Some(host) = made.host(slot) else {
                continue;
            };
            let Some(&key) = made.keys.get(host as usize) else {
                continue;
            };
            if key == 0 {
                continue;
            }
            let Some(view) = view_at(key as u32, (key >> 32) as u32) else {
                continue;
            };
            if let Some(kept) = make(&view, self.cpu(host)) {
                self.keep(host, kept);
                remade += 1;
            }
        }
        remade
    }

    /// The memory of the texture title slot `index` views is not mapped
    /// (any more): a host slot made from it becomes null again, so that
    /// the host copy can be let go (`crate::resource::let_go_idle`; a slot
    /// keeps what it views). True if the slot was made.
    /// (With a heap map, a title slot past the window keeps its pool slot:
    /// it still holds the view, made again once its memory is mapped.)
    pub(crate) fn texture_slot_gone(&self, index: u32) -> bool {
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        if made
            .host(index)
            .and_then(|host| made.keys.get(host as usize))
            .is_none_or(|&key| key == 0)
        {
            return false;
        }
        self.unmake(&mut made, index);
        STALE[4].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        true
    }

    /// Makes title slot `index`'s host slot null again, as it is before its
    /// view is made: a null view of the kind it held.
    fn unmake(&self, made: &mut Made, index: u32) {
        let Some(host) = made.host(index) else {
            return;
        };
        let Some(key) = made.keys.get_mut(host as usize).map(std::mem::take) else {
            return;
        };
        if key == 0 {
            return;
        }
        if let Some(fast) = self.made_fast.get(host as usize) {
            fast.store(0, std::sync::atomic::Ordering::Release);
        }
        match view_at(key as u32, (key >> 32) as u32) {
            Some(view) => null_for(&view, self.cpu(host)),
            None => null_view(self.cpu(host)),
        }
        self.keep(host, None);
    }

    /// Title slot `index` is a table's ([`HostHeap::tabled`]). `read`: a
    /// draw read it through the table and brought what it views up to
    /// date, so the slot leaves the queue of textures picked by index; the
    /// caller makes the host slot in place ([`HostHeap::mirror`]), as the
    /// pass would have, for a shader that picks it by index all the same.
    pub(crate) fn texture_slot_tabled(&self, index: u32, read: bool) {
        self.tabled_in(index, read, crate::frame::current());
    }

    /// [`HostHeap::texture_slot_tabled`] in frame `frame`.
    fn tabled_in(&self, index: u32, read: bool, frame: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        let Some(stamp) = self.tabled.get(index as usize) else {
            return;
        };
        let now = (frame as u32).wrapping_add(1) & !TABLED_READ;
        let old = stamp.load(Relaxed);
        if !read {
            if old & !TABLED_READ != now {
                stamp.store(now, Relaxed);
            }
            return;
        }
        // Once per slot and frame: tables are read by every draw.
        if old != now | TABLED_READ {
            stamp.store(now | TABLED_READ, Relaxed);
            if tabled_out() {
                let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
                if textures.slots.remove(&index).is_some() {
                    TABLED_WRITES[1].fetch_add(1, Relaxed);
                }
            }
        }
    }

    /// Whether title slot `index` was a table's within the last
    /// [`TABLED_FRAMES`] frames before `frame` (never, if such slots are
    /// to be queued as any other).
    fn tabled_lately(&self, index: u32, frame: u64) -> bool {
        let stamp = self.tabled_stamp(index) & !TABLED_READ;
        let now = (frame as u32).wrapping_add(1) & !TABLED_READ;
        tabled_out() && stamp != 0 && u64::from(now.wrapping_sub(stamp)) <= TABLED_FRAMES
    }

    /// What `tabled` holds for title slot `index`; 0: never a table's.
    fn tabled_stamp(&self, index: u32) -> u32 {
        self.tabled
            .get(index as usize)
            .map_or(0, |s| s.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// The queue's part of a write of title slot `index` in `frame`:
    /// `texture`, whether it now holds a view of a texture the title may
    /// fill; `seen` as for [`HostHeap::note_buffer`].
    fn queue_texture(&self, index: u32, texture: bool, seen: bool, frame: u64) {
        if PUBLISHING.get() {
            return;
        }
        let stamp = if tabled_out() { self.tabled_stamp(index) } else { 0 };
        let now = (frame as u32).wrapping_add(1) & !TABLED_READ;
        // A table's slot lately: the draw that reads it uploads its
        // texture, so its write waits for that draw.
        let lately = stamp != 0 && u64::from(now.wrapping_sub(stamp & !TABLED_READ)) <= TABLED_FRAMES;
        // Made now by a draw of this frame that read it through a table:
        // up to date.
        let read_now = seen && stamp == now | TABLED_READ;
        let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        if texture && !read_now {
            let due = if lately && !seen {
                TABLED_WRITES[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                frame + TABLED_GRACE
            } else {
                0
            };
            textures.written(index, frame, due, seen);
        } else {
            textures.remove(index);
        }
    }

    /// The title slots holding buffer views in bucket `bucket % BUFFER_BUCKETS`,
    /// returned in strictly ascending order (clones only `1 / BUFFER_BUCKETS`
    /// of the heap's buffer slots).
    pub(crate) fn buffer_slots_bucket(&self, bucket: usize) -> Vec<u32> {
        let slots = self.buffers[bucket % BUFFER_BUCKETS]
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        slots.iter().copied().collect()
    }

    /// All title slots holding buffer views across every bucket (tests).
    #[cfg(test)]
    pub(crate) fn buffer_slots(&self) -> Vec<u32> {
        let mut all = Vec::new();
        for bucket in &self.buffers {
            let slots = bucket.lock().unwrap_or_else(|e| e.into_inner());
            all.extend(slots.iter().copied());
        }
        all.sort_unstable();
        all
    }

    /// The title slots whose buffer view was written, or went, since the
    /// last call.
    pub(crate) fn buffer_changes(&self) -> Vec<u32> {
        let mut changed = self.changed.lock().unwrap_or_else(|e| e.into_inner());
        std::mem::take(&mut *changed).into_iter().collect()
    }

    /// Records what title slot `index` now holds for shaders that pick
    /// buffers from the heap: its element stride in the table, and whether
    /// it is a buffer view at all.
    /// `seen`: the slot was not written now, a draw or a pass made its
    /// host slot from what it held.
    /// Returns the title slots whose placed texture view the write made
    /// stale ([`Spans`]): out of the queue of textures already, their host
    /// slots are the caller's to make null (it holds `made`).
    #[cfg(test)]
    fn note_buffer(&self, index: u32, view: Option<&View>, seen: bool) -> Vec<u32> {
        self.note_buffer_in(index, view, seen, true)
    }

    /// [`HostHeap::note_buffer`] with `mark_changed` controlling whether a
    /// buffer slot update is queued into `self.changed` (`false` when called
    /// from `State::heap_buffers`, which updates its own `ranges` cache
    /// immediately and would otherwise re-process every new buffer slot on
    /// the following frame).
    fn note_buffer_in(&self, index: u32, view: Option<&View>, seen: bool, mark_changed: bool) -> Vec<u32> {
        if self.sampler {
            return Vec::new();
        }
        let frame = crate::frame::current();
        // Measured for the texture side of the same gap (textures a shader
        // picks by a computed index are neither made, if placed, nor
        // uploaded): slots written with a placed texture view, and with a
        // view of a texture object.
        let texture = match view {
            Some(View::Placed(false, resource, _)) if resource[0] != 1 => {
                PLACED_TEXTURE_SLOTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                true
            }
            Some(View::Shader(r, _)) if crate::resource::buffer_bytes(*r as crate::object::This).is_none() => {
                OBJECT_TEXTURE_SLOTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                true
            }
            _ => false,
        };
        self.queue_texture(index, texture, seen, frame);
        // The memory a placed texture view describes. Not for a slot
        // tables read (thousands of writes a frame, each read by its
        // draw): no pass looks at it, and the same texture's view in a
        // slot shaders pick by index says what it says.
        let placed_view = match view {
            Some(View::Placed(false, resource, desc))
                if resource[0] != 1 && !self.tabled_lately(index, frame) =>
            {
                // (And the level the view starts at: word 4 of a texture
                // view's description, after format, dimension, mapping
                // and a pad.)
                Some((resource, u64::from(desc[10]) | u64::from(desc[11]) << 32, desc[4]))
            }
            _ => None,
        };
        let stale = {
            use std::sync::atomic::Ordering::Relaxed;
            let mut placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
            let span = placed_view.and_then(|(resource, start, first)| {
                Some((start, start.saturating_add(placed.bytes(resource)?), first))
            });
            match span {
                Some((start, end, first)) => {
                    if start < SPANNED[0].load(Relaxed) {
                        SPANNED[0].fetch_min(start, Relaxed);
                    }
                    if end > SPANNED[1].load(Relaxed) {
                        SPANNED[1].fetch_max(end, Relaxed);
                    }
                    placed.rule = stale_rule();
                    let stale = placed.note(index, start, end, !seen);
                    placed.shows(index, first);
                    stale
                }
                None => {
                    placed.forget(index);
                    Vec::new()
                }
            }
        };
        if !stale.is_empty() {
            STALE[0].fetch_add(stale.len() as u64, std::sync::atomic::Ordering::Relaxed);
            let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
            for &slot in &stale {
                textures.remove(slot);
            }
        }
        let stride = view.and_then(view_stride);
        if let Some((_, _, data)) = &self.strides
            && index < victory_shader::emit::SLOT_STRIDES_COUNT
        {
            // SAFETY: inside the mapping; shaders read it while we write (a
            // dword, either value is one the slot had).
            unsafe { (*data as *mut u32).add(index as usize).write_volatile(stride.unwrap_or(0)) };
        }
        let was = {
            let mut bucket = self.buffers[(index as usize) % BUFFER_BUCKETS]
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if stride.is_some() {
                bucket.insert(index);
                true
            } else {
                bucket.remove(&index)
            }
        };
        if was && mark_changed {
            self.changed
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(index);
        }
        stale
    }

    /// Gives back `n` dynamic slots from `first` (an index [`dynamic`]
    /// returned).
    ///
    /// [`dynamic`]: HostHeap::dynamic
    pub(crate) fn release(&self, first: u32, n: u32) {
        let Some(at) = first.checked_sub(self.dynamic_first) else {
            return;
        };
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let end = (at + n).min(self.dynamic) as usize;
        if let Some(slots) = guard.1.get_mut(at as usize..end) {
            let given = slots.iter().filter(|taken| **taken).count() as u32;
            self.free.fetch_add(given, std::sync::atomic::Ordering::Relaxed);
            self.too_long.store(u32::MAX, std::sync::atomic::Ordering::Relaxed);
            slots.fill(false);
        }
    }

    /// Makes dynamic slots `first..first + n` what the title's table at
    /// `table`, its slots `index..index + n` past this heap's title slots,
    /// holds: overflow slots whose record changed since they were made are
    /// made again, then all `n` are copied in one go, and each dynamic slot
    /// keeps as [`HostHeap::make_at`] would. Views of textures the title
    /// placed itself whose host copy moved are made again; views the host
    /// could not make are tried again at the next use. Returns how many
    /// views it made, and the table's views (for the caller's uses); `None`
    /// without an overflow heap or past its end (the caller makes them one
    /// by one). Locks are taken once per table: the views', the overflow
    /// shards the table spans, the kept resources' shards.
    pub(crate) fn copy_overflow(&self, first: u32, index: u32, n: u32, table: u64) -> Option<(u32, Vec<Option<View>>)> {
        let o = self.overflow.as_ref()?;
        let from = index.checked_sub(self.count)?;
        if n == 0
            || u64::from(from) + u64::from(n) > u64::from(o.len)
            || first < self.dynamic_first
            || first.checked_add(n).is_none_or(|end| end > self.dynamic_first + self.dynamic)
        {
            return None;
        }
        // The table's records, and their views in one look-up.
        let records: Vec<[u32; 3]> = (0..n)
            // SAFETY: descriptor slots in title memory.
            .map(|k| unsafe { ((table + u64::from(k) * 32) as *const [u32; 3]).read_unaligned() })
            .collect();
        let views = super::views_of(&records);
        // The shards the table spans, locked in order: what is made there
        // and copied from stays as it is meanwhile.
        let (s0, s1) = (from / SHARD, (from + n - 1) / SHARD);
        let mut shards: Vec<_> = (s0..=s1)
            .map(|s| o.shards[s as usize].lock().unwrap_or_else(|e| e.into_inner()))
            .collect();
        let (mut made, mut retired) = (0, Vec::new());
        let mut kept = Vec::with_capacity(n as usize);
        for (k, (view, &[_, i, generation])) in views.iter().zip(&records).enumerate() {
            let at = from + k as u32;
            let slot = &mut shards[(at / SHARD - s0) as usize][(at % SHARD) as usize];
            // Generations start at 1 (and inline kinds use bits 30..32), so a
            // made view's key is never 0 or Over::NULL.
            let key = match view {
                Some(_) => u64::from(generation) << 32 | u64::from(i),
                None => Over::NULL,
            };
            // A placed texture's slot is made again if its host texture
            // changed under the same record, as `make` upgrades placed textures.
            let moved = || {
                view.as_ref().is_some_and(|v| {
                    let held = slot.kept.as_ref().map(windows::core::Interface::as_raw);
                    is_placed_texture(v) && held.is_none_or(|held| placed_texture_moved(v, held))
                })
            };
            if slot.key != key || moved() {
                let (made_kept, key) = match view {
                    Some(view) => {
                        let made_kept = make(view, o.cpu(at));
                        if made_kept.is_none() {
                            null_for(view, o.cpu(at));
                        }
                        // A view the host could not make is tried again at
                        // the next use.
                        let key = if made_kept.is_none() { 0 } else { key };
                        (made_kept.flatten(), key)
                    }
                    None => {
                        null_view(o.cpu(at));
                        (None, key)
                    }
                };
                // What the old view kept goes once the work in flight has
                // run, as for a title slot ([`HostHeap::keep`]).
                if let Some(old) = std::mem::replace(slot, Over { key, kept: made_kept }).kept {
                    retired.push(old);
                }
                made += 1;
            }
            kept.push(slot.kept.clone());
        }
        if let Some(host) = host() {
            // SAFETY: `n` slots inside both heaps; the source is CPU-only.
            unsafe {
                host.device
                    .CopyDescriptorsSimple(n, self.cpu(first), o.cpu(from), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
            };
        }
        drop(shards);
        crate::host::retire_resources(retired);
        self.keep_run(first, kept);
        Some((made, views))
    }

    /// [`HostHeap::keep`] of slots `first..` in one go across [`KEPT_PIECES`]
    /// shards: what they kept goes once the work in flight has run (or at
    /// once when `slot_rewritten` allows).
    fn keep_run(&self, first: u32, mut resources: Vec<Kept>) {
        if resources.is_empty() {
            return;
        }
        let mut retired = Vec::new();
        let mut dropped = Vec::new();
        {
            let _one = (!heap_locks()).then(|| self.kept_one.lock().unwrap_or_else(|e| e.into_inner()));
            let touch = resources.len().min(KEPT_PIECES as usize);
            for offset in 0..touch {
                let piece_idx = ((first + offset as u32) % KEPT_PIECES) as usize;
                let mut kept = self.kept[piece_idx].lock().unwrap_or_else(|e| e.into_inner());
                let mut k = offset;
                while k < resources.len() {
                    let dst = first + k as u32;
                    let at = (dst / KEPT_PIECES) as usize;
                    if let Some(slot) = kept.get_mut(at) {
                        let resource = resources[k].take();
                        let lean = slot
                            .as_ref()
                            .is_some_and(|old| crate::host::slot_rewritten(old, resource.as_ref()));
                        if let Some(old) = std::mem::replace(slot, resource) {
                            if lean {
                                dropped.push(old);
                            } else {
                                retired.push(old);
                            }
                        }
                    }
                    k += KEPT_PIECES as usize;
                }
            }
        }
        drop(dropped);
        crate::host::retire_resources(retired);
    }

    /// Lets go of descriptors viewing any host texture in `textures` (when
    /// evicted), walking overflow slots and retiring kept resources after
    /// dropping shard locks.
    #[allow(dead_code)]
    pub(crate) fn evict(&self, textures: &std::collections::HashSet<usize>) {
        self.evict_overflow(textures);
    }

    /// [`HostHeap::evict`] of the overflow slots: one viewing a texture given
    /// up lets go of it and is made again at its next use (its key cleared),
    /// of the texture made again then. Its descriptor is never read by the
    /// GPU (a CPU-only heap), so it is left as it is until then. What it held
    /// goes once the work in flight has run, after the shard's lock.
    /// Lock order: under `SHADER_HEAPS` (`descriptor::evict`), then a shard;
    /// `copy_overflow` takes shards without `SHADER_HEAPS`.
    fn evict_overflow(&self, textures: &std::collections::HashSet<usize>) {
        let Some(o) = &self.overflow else { return };
        let mut retired = Vec::new();
        for shard in &o.shards {
            let mut slots = shard.lock().unwrap_or_else(|e| e.into_inner());
            for slot in slots.iter_mut() {
                let given_up = slot
                    .kept
                    .as_ref()
                    .is_some_and(|r| textures.contains(&(windows::core::Interface::as_raw(r) as usize)));
                if given_up {
                    if let Some(old) = slot.kept.take() {
                        retired.push(old);
                    }
                    slot.key = 0;
                }
            }
        }
        crate::host::retire_resources(retired);
    }

    /// Makes slot `index` (a dynamic one) from `view`; a null view if the
    /// host cannot.
    pub(crate) fn make_at(&self, index: u32, view: &View) {
        let kept = make(view, self.cpu(index));
        if kept.is_none() {
            null_for(view, self.cpu(index));
        }
        self.keep(index, kept.flatten());
    }

    /// Makes slot `index` read nothing: a null view, or a plain sampler.
    pub(crate) fn make_null(&self, index: u32) {
        if self.sampler {
            self.null_sampler(index);
        } else {
            null_view(self.cpu(index));
        }
        self.keep(index, None);
    }

    /// Slot `index` now views `resource` (or nothing of the host's).
    fn keep(&self, index: u32, resource: Option<ID3D12Resource>) {
        let piece = &self.kept[(index % KEPT_PIECES) as usize];
        let at = (index / KEPT_PIECES) as usize;
        if heap_locks() {
            let (old, lean) = {
                let mut kept = piece.lock().unwrap_or_else(|e| e.into_inner());
                match kept.get_mut(at) {
                    Some(slot) => {
                        let lean = slot
                            .as_ref()
                            .is_some_and(|old| crate::host::slot_rewritten(old, resource.as_ref()));
                        (std::mem::replace(slot, resource), lean)
                    }
                    None => (None, false),
                }
            };
            // A running submission may still read the old resource through
            // this slot. (Outside the lock: letting go takes another lock,
            // or frees a host resource.) The same object written again is
            // held by the slot's new reference: the old one goes now.
            if let Some(old) = old {
                if lean {
                    drop(old);
                } else {
                    crate::host::retire_resource(old);
                }
            }
            return;
        }
        // As before: one line for the whole heap, held across the letting go.
        let _one = self.kept_one.lock().unwrap_or_else(|e| e.into_inner());
        let mut kept = piece.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = kept.get_mut(at) {
            let lean = slot
                .as_ref()
                .is_some_and(|old| crate::host::slot_rewritten(old, resource.as_ref()));
            // A running submission may still read the old resource through
            // this slot (the same object written again: held by the new
            // reference, the old one goes now).
            if let Some(old) = std::mem::replace(slot, resource) {
                if lean {
                    drop(old);
                } else {
                    crate::host::retire_resource(old);
                }
            }
        }
    }

    /// Makes slot `index` (a dynamic one) a raw buffer view of `len` bytes
    /// of `buffer` from byte `offset` (a multiple of 16): a root SRV with a
    /// size, whose reads past it return zeros instead of reaching whatever
    /// the host placed after the buffer.
    pub(crate) fn raw_at(&self, index: u32, buffer: &ID3D12Resource, offset: u64, len: u64) {
        let Some(host) = host() else { return };
        let elements = (len / 4).clamp(1, u64::from(u32::MAX)) as u32;
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: offset / 4,
                    NumElements: elements,
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                },
            },
        };
        // SAFETY: a slot of our heap and a live buffer covering the range.
        unsafe {
            host.device
                .CreateShaderResourceView(buffer, Some(&desc), self.cpu(index))
        };
        self.keep(index, Some(buffer.clone()));
    }

    /// Makes slot `index` (a dynamic one) a raw buffer view that reads
    /// zeros.
    pub(crate) fn null_raw_at(&self, index: u32) {
        null_buffer_view(self.cpu(index));
        self.keep(index, None);
    }

    /// Makes slot `index` (a dynamic one) a raw unordered access view of
    /// `len` bytes of `buffer` from byte `offset` (a multiple of 16): a
    /// root UAV with a size, whose accesses past it do nothing instead of
    /// reaching whatever the host placed after the buffer.
    pub(crate) fn raw_uav_at(&self, index: u32, buffer: &ID3D12Resource, offset: u64, len: u64) {
        let Some(host) = host() else { return };
        let elements = (len / 4).clamp(1, u64::from(u32::MAX)) as u32;
        let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    FirstElement: offset / 4,
                    NumElements: elements,
                    StructureByteStride: 0,
                    CounterOffsetInBytes: 0,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                },
            },
        };
        // SAFETY: a slot of our heap and a live buffer covering the range.
        unsafe {
            host.device.CreateUnorderedAccessView(
                buffer,
                None::<&ID3D12Resource>,
                Some(&desc),
                self.cpu(index),
            )
        };
        self.keep(index, Some(buffer.clone()));
    }

    /// Makes slot `index` (a dynamic one) a raw unordered access view of
    /// nothing: reads give zeros, writes are dropped.
    pub(crate) fn null_raw_uav_at(&self, index: u32) {
        self.keep(index, None);
        let Some(host) = host() else { return };
        let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    NumElements: 1,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                    ..Default::default()
                },
            },
        };
        // SAFETY: a slot of our heap; a null resource is allowed with a desc.
        unsafe {
            host.device.CreateUnorderedAccessView(
                None::<&ID3D12Resource>,
                None::<&ID3D12Resource>,
                Some(&desc),
                self.cpu(index),
            )
        };
    }

    /// The shader-visible handle of slot `index`.
    pub(crate) fn gpu_handle(&self, index: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: self.gpu + u64::from(index) * u64::from(self.step),
        }
    }

    /// Makes slot `index` (a dynamic one) a constant buffer view of host
    /// memory (the list's own, which it holds).
    pub(crate) fn constant_at(&self, index: u32, gpu: u64, size: u32) {
        self.constant_at_kept(index, gpu, size, None);
    }

    /// Makes slot `index` (a dynamic one) a constant buffer view of `resource`
    /// at `gpu`, holding `resource` alive while the slot views it.
    pub(crate) fn constant_at_kept(
        &self,
        index: u32,
        gpu: u64,
        size: u32,
        resource: Option<ID3D12Resource>,
    ) {
        self.keep(index, resource);
        let Some(host) = host() else { return };
        let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
            BufferLocation: gpu,
            SizeInBytes: constant_view_size(size),
        };
        // SAFETY: a slot of our heap; the memory is kept alive by `self.keep` or the caller.
        unsafe {
            host.device
                .CreateConstantBufferView(Some(&desc), self.cpu(index))
        };
    }

    fn cpu(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.cpu + index as usize * self.step as usize,
        }
    }

    /// O(1) check whether a `PAGE_TABLED_SLOTS`-aligned page `first..last`
    /// was previously proven 100% tabled and its oldest stamp is still within
    /// `TABLED_FRAMES` of `frame`.
    pub(crate) fn page_all_tabled(&self, first: u32, last: u32, frame: u64) -> bool {
        use std::sync::atomic::Ordering::Relaxed;
        if sync_tables() || !tabled_out() || self.page_tabled_min.is_empty() {
            return false;
        }
        let mirrored_limit = self.mirrored_count();
        let first = first.min(mirrored_limit);
        let last = last.min(mirrored_limit);
        if first >= last || !first.is_multiple_of(PAGE_TABLED_SLOTS) {
            return false;
        }
        let page_idx = (first / PAGE_TABLED_SLOTS) as usize;
        let expected_end = ((page_idx as u32 + 1) * PAGE_TABLED_SLOTS).min(mirrored_limit);
        if last != expected_end {
            return false;
        }
        let Some(slot) = self.page_tabled_min.get(page_idx) else {
            return false;
        };
        let min_stamp = slot.load(Relaxed);
        if min_stamp == 0 {
            return false;
        }
        let now = (frame as u32).wrapping_add(1) & !TABLED_READ;
        u64::from(now.wrapping_sub(min_stamp)) <= TABLED_FRAMES
    }

    /// Builds contiguous `[range_from, range_end)` byte spans inside
    /// `[start, looked_end)` skipping leading/trailing `page_all_tabled` pages
    /// and internal all-tabled gaps of at least `MIN_TABLED_GAP_PAGES` (16)
    /// pages so `GetWriteWatch` does not scan/reset massive all-tabled tails.
    pub(crate) fn untabled_watch_spans(
        &self,
        start: usize,
        looked_end: usize,
        step: u64,
        mirrored_count: u32,
        frame: u64,
    ) -> Vec<(usize, usize)> {
        const MIN_TABLED_GAP_PAGES: usize = 16;
        if looked_end <= start {
            return Vec::new();
        }
        if sync_tables()
            || !tabled_out()
            || self.page_tabled_min.is_empty()
            || step != 32
            || !start.is_multiple_of(crate::watch::PAGE)
        {
            return vec![(start, looked_end)];
        }
        let page_size = crate::watch::PAGE;
        let total_pages = (looked_end - start).div_ceil(page_size);
        let mut spans: Vec<(usize, usize)> = Vec::new();
        let mut p = 0usize;
        while p < total_pages {
            let page_from = start + p * page_size;
            let page_to = (page_from + page_size).min(looked_end);
            let s_first = (p * PAGE_TABLED_SLOTS as usize) as u32;
            let s_last = (((page_to - start) / step as usize) as u32).min(mirrored_count);
            if s_first < s_last && self.page_all_tabled(s_first, s_last, frame) {
                p += 1;
                continue;
            }
            let span_start = page_from;
            let mut span_end = page_to;
            p += 1;
            while p < total_pages {
                let pf = start + p * page_size;
                let pt = (pf + page_size).min(looked_end);
                let sf = (p * PAGE_TABLED_SLOTS as usize) as u32;
                let sl = (((pt - start) / step as usize) as u32).min(mirrored_count);
                if sf < sl && self.page_all_tabled(sf, sl, frame) {
                    let mut gap = 1usize;
                    while gap < MIN_TABLED_GAP_PAGES && p + gap < total_pages {
                        let gf = start + (p + gap) * page_size;
                        let gt = (gf + page_size).min(looked_end);
                        let gsf = ((p + gap) * PAGE_TABLED_SLOTS as usize) as u32;
                        let gsl = (((gt - start) / step as usize) as u32).min(mirrored_count);
                        if !(gsf < gsl && self.page_all_tabled(gsf, gsl, frame)) {
                            break;
                        }
                        gap += 1;
                    }
                    if gap >= MIN_TABLED_GAP_PAGES || p + gap == total_pages {
                        p += gap;
                        break;
                    }
                    span_end = (start + (p + gap) * page_size).min(looked_end);
                    p += gap;
                } else {
                    span_end = pt;
                    p += 1;
                }
            }
            spans.push((span_start, span_end));
        }
        spans
    }

    /// Scans complete title slots `first..last` (each `step` bytes, starting at
    /// `start`) on one changed heap page and mirrors only untabled slots whose
    /// 12-byte `[tag, i, generation]` record key changed since last seen.
    ///
    /// Trims leading/trailing tabled runs BEFORE copying title memory (so pages
    /// filled wholly by per-draw tables skip `read_into` / `try_copy_fast` and
    /// `self.copied` entirely) and returns `(page_ok, looked, mirrored, tabled)`
    /// so the caller flushes `COPIED_TABLED` once per `sync_heaps` pass instead
    /// of doing a per-slot atomic `fetch_add`.
    pub(crate) fn sync_copied_page(
        &self,
        first: u32,
        last: u32,
        start: u64,
        step: u64,
        frame: u64,
        page_bytes: &mut [u8],
    ) -> (bool, u64, u64, u64) {
        use std::sync::atomic::Ordering::Relaxed;
        let mirrored_limit = self.mirrored_count();
        let first = first.min(mirrored_limit);
        let last = last.min(mirrored_limit);
        if first >= last || step < 12 {
            return (false, 0, 0, 0);
        }
        let total = u64::from(last - first);
        if self.page_all_tabled(first, last, frame) {
            return (true, total, 0, total);
        }
        let skip_tabled = !sync_tables() && tabled_out() && !self.tabled.is_empty();
        let now = (frame as u32).wrapping_add(1) & !TABLED_READ;
        let is_tabled_raw = |raw: u32| -> bool {
            let stamp = raw & !TABLED_READ;
            stamp != 0 && u64::from(now.wrapping_sub(stamp)) <= TABLED_FRAMES
        };

        let (scan_first, scan_last, mut tabled) = if skip_tabled {
            let tabled_end = (last as usize).min(self.tabled.len());
            let mut f = first as usize;
            let mut max_age = 0u32;
            let mut oldest_stamp = 0u32;
            while f < tabled_end {
                let stamp = self.tabled[f].load(Relaxed) & !TABLED_READ;
                if stamp == 0 {
                    break;
                }
                let age = now.wrapping_sub(stamp);
                if u64::from(age) > TABLED_FRAMES {
                    break;
                }
                if oldest_stamp == 0 || age >= max_age {
                    max_age = age;
                    oldest_stamp = stamp;
                }
                f += 1;
            }
            if f == last as usize {
                if oldest_stamp != 0
                    && first.is_multiple_of(PAGE_TABLED_SLOTS)
                    && !self.page_tabled_min.is_empty()
                {
                    let page_idx = (first / PAGE_TABLED_SLOTS) as usize;
                    let expected_end =
                        ((page_idx as u32 + 1) * PAGE_TABLED_SLOTS).min(mirrored_limit);
                    if last == expected_end
                        && let Some(cell) = self.page_tabled_min.get(page_idx)
                    {
                        cell.store(oldest_stamp, Relaxed);
                    }
                }
                return (true, total, 0, total);
            }
            let mut l = last as usize;
            while l > f && l <= tabled_end && is_tabled_raw(self.tabled[l - 1].load(Relaxed)) {
                l -= 1;
            }
            let trimmed = (f - first as usize) + (last as usize - l);
            (f as u32, l as u32, trimmed as u64)
        } else {
            (first, last, 0u64)
        };

        let copy_from = start + u64::from(scan_first) * step;
        let copy_len = (scan_last - scan_first) as usize * step as usize;
        if copy_len > page_bytes.len() {
            return (false, 0, 0, 0);
        }
        let slice = &mut page_bytes[..copy_len];
        // `read_into` uses VEH-guarded `try_copy_fast` first and falls back to
        // `ReadProcessMemory` so an unmapped heap page fails cleanly without
        // turning into zeros.
        if !crate::host::read_into(copy_from as *const u8, slice) {
            return (false, 0, 0, 0);
        }

        let (mirrored, inner_tabled) = self.mirror_copied_slice(
            scan_first,
            scan_last,
            start,
            step,
            slice,
            skip_tabled,
            now,
        );
        tabled += inner_tabled;
        (true, total, mirrored, tabled)
    }

    /// Initializes `self.copied` before [`sync_heaps`] fans out across worker
    /// threads so workers never contend on `OnceLock` initialization.
    pub(crate) fn copied_begin(&self) -> &[AtomicU64] {
        let mirrored_limit = self.mirrored_count();
        self.copied
            .get_or_init(|| (0..mirrored_limit as usize).map(|_| AtomicU64::new(0)).collect())
    }

    /// Whether slot `index` was made from no record (holds no host view):
    /// empty/non-view records in such slots need no [`HostHeap::mirror`].
    pub(crate) fn shows_none(&self, index: u32) -> bool {
        if let Some(fast) = self.made_fast.get(index as usize) {
            return fast.load(std::sync::atomic::Ordering::Acquire) == 0;
        }
        self.made
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .host(index)
            .is_none()
    }

    /// Inner hot loop for [`HostHeap::sync_copied_page`]: walks `slice` holding
    /// `step`-byte records for title slots `first..last`, skips tabled slots,
    /// compares each untabled slot's 12-byte key against `self.copied`, and
    /// mirrors only changed keys (skipping empty records whose slot already shows none).
    /// Returns `(mirrored, tabled_skipped)`.
    fn mirror_copied_slice(
        &self,
        first: u32,
        last: u32,
        start: u64,
        step: u64,
        slice: &[u8],
        skip_tabled: bool,
        now: u32,
    ) -> (u64, u64) {
        let copied = self.copied_begin();
        let copied_slice = &copied[first as usize..last as usize];
        let tabled_slice = if skip_tabled {
            let end = (last as usize).min(self.tabled.len());
            if (first as usize) < end {
                &self.tabled[first as usize..end]
            } else {
                &[]
            }
        } else {
            &[]
        };

        match step {
            32 => self.scan_copied_chunks::<32>(first, start, slice, copied_slice, tabled_slice, now),
            16 => self.scan_copied_chunks::<16>(first, start, slice, copied_slice, tabled_slice, now),
            _ => self.scan_copied_dynamic(first, start, step, slice, copied_slice, tabled_slice, now),
        }
    }

    /// Constant-stride inner scan for view (`32`) and sampler (`16`) heaps:
    /// hoists `OnceLock`/`frame` checks out of the loop, eliminates per-slot
    /// bounds checks and atomic RMWs, and keeps key extraction branchless.
    fn scan_copied_chunks<const STEP: usize>(
        &self,
        first: u32,
        start: u64,
        slice: &[u8],
        copied_slice: &[AtomicU64],
        tabled_slice: &[std::sync::atomic::AtomicU32],
        now: u32,
    ) -> (u64, u64) {
        use std::sync::atomic::Ordering::Relaxed;
        const { assert!(STEP >= 12) };
        let step_u64 = STEP as u64;
        let mut mirrored = 0u64;
        let mut tabled = 0u64;

        if tabled_slice.len() >= copied_slice.len() {
            for (off, ((record, seen_slot), tabled_slot)) in slice
                .chunks_exact(STEP)
                .zip(copied_slice.iter())
                .zip(tabled_slice.iter())
                .enumerate()
            {
                let stamp = tabled_slot.load(Relaxed) & !TABLED_READ;
                if stamp != 0 && u64::from(now.wrapping_sub(stamp)) <= TABLED_FRAMES {
                    tabled += 1;
                    continue;
                }
                let tag = u32::from_le_bytes([record[0], record[1], record[2], record[3]]);
                let key = if tag == VIEW_TAG {
                    u64::from_le_bytes([
                        record[4], record[5], record[6], record[7],
                        record[8], record[9], record[10], record[11],
                    ])
                } else {
                    0
                };
                if seen_slot.load(Relaxed) == key {
                    continue;
                }
                seen_slot.store(key, Relaxed);
                let index = first + off as u32;
                if key == 0 && self.shows_none(index) {
                    continue;
                }
                let handle = start + u64::from(index) * step_u64;
                self.mirror_in_pass(index, handle);
                mirrored += 1;
            }
        } else {
            for (off, (record, seen_slot)) in slice
                .chunks_exact(STEP)
                .zip(copied_slice.iter())
                .enumerate()
            {
                if let Some(tabled_slot) = tabled_slice.get(off) {
                    let stamp = tabled_slot.load(Relaxed) & !TABLED_READ;
                    if stamp != 0 && u64::from(now.wrapping_sub(stamp)) <= TABLED_FRAMES {
                        tabled += 1;
                        continue;
                    }
                }
                let tag = u32::from_le_bytes([record[0], record[1], record[2], record[3]]);
                let key = if tag == VIEW_TAG {
                    u64::from_le_bytes([
                        record[4], record[5], record[6], record[7],
                        record[8], record[9], record[10], record[11],
                    ])
                } else {
                    0
                };
                if seen_slot.load(Relaxed) == key {
                    continue;
                }
                seen_slot.store(key, Relaxed);
                let index = first + off as u32;
                if key == 0 && self.shows_none(index) {
                    continue;
                }
                let handle = start + u64::from(index) * step_u64;
                self.mirror_in_pass(index, handle);
                mirrored += 1;
            }
        }
        (mirrored, tabled)
    }

    /// Fallback for non-16/32 descriptor strides.
    #[allow(clippy::too_many_arguments)]
    fn scan_copied_dynamic(
        &self,
        first: u32,
        start: u64,
        step: u64,
        slice: &[u8],
        copied_slice: &[AtomicU64],
        tabled_slice: &[std::sync::atomic::AtomicU32],
        now: u32,
    ) -> (u64, u64) {
        use std::sync::atomic::Ordering::Relaxed;
        let mut mirrored = 0u64;
        let mut tabled = 0u64;
        for (off, (record, seen_slot)) in slice
            .chunks_exact(step as usize)
            .zip(copied_slice.iter())
            .enumerate()
        {
            if let Some(tabled_slot) = tabled_slice.get(off) {
                let stamp = tabled_slot.load(Relaxed) & !TABLED_READ;
                if stamp != 0 && u64::from(now.wrapping_sub(stamp)) <= TABLED_FRAMES {
                    tabled += 1;
                    continue;
                }
            }
            if record.len() < 12 {
                continue;
            }
            let tag = u32::from_le_bytes([record[0], record[1], record[2], record[3]]);
            let key = if tag == VIEW_TAG {
                u64::from_le_bytes([
                    record[4], record[5], record[6], record[7],
                    record[8], record[9], record[10], record[11],
                ])
            } else {
                0
            };
            if seen_slot.load(Relaxed) == key {
                continue;
            }
            seen_slot.store(key, Relaxed);
            let index = first + off as u32;
            if key == 0 && self.shows_none(index) {
                continue;
            }
            let handle = start + u64::from(index) * step;
            self.mirror_in_pass(index, handle);
            mirrored += 1;
        }
        (mirrored, tabled)
    }

    /// Calls [`HostHeap::mirror`] from the view-heap pass, timing the outcome
    /// breakdown when `diagnostics` is enabled and adding zero overhead otherwise.
    #[inline(always)]
    fn mirror_in_pass(&self, index: u32, handle: u64) {
        #[cfg(feature = "diagnostics")]
        {
            MIRROR_HOW.with(|how| how.set(MIRROR_UNSET));
            let one = std::time::Instant::now();
            self.mirror(index, handle, false);
            mirror_counted(one.elapsed().as_micros() as u64);
        }
        #[cfg(not(feature = "diagnostics"))]
        {
            self.mirror(index, handle, false);
        }
    }

    /// Checks, in bounded chunks under `self.made`, which `(index, expected_key)`
    /// candidates still have their host slot made from `expected_key` (and
    /// mapped to a pool slot if past the identity window). Appends any slot
    /// that does not match to `mismatched`.
    pub(crate) fn collect_unmade_slots(&self, candidates: &[(u32, u64)], mismatched: &mut Vec<u32>) {
        const CHUNK: usize = 512;
        for chunk in candidates.chunks(CHUNK) {
            let made = self.made.lock().unwrap_or_else(|e| e.into_inner());
            for &(index, key) in chunk {
                let matched = key != 0
                    && made
                        .host(index)
                        .and_then(|host| made.keys.get(host as usize))
                        .is_some_and(|&made_key| made_key == key);
                if !matched {
                    mismatched.push(index);
                }
            }
        }
    }

    /// Makes host slot `index` from the record at `handle` if it is still
    /// the one with `key`, whose texture's upload is in the queue now
    /// (`crate::heap_worker`). The slot's place in the queue of textures
    /// picked by index was settled when that upload was recorded, and stays
    /// as it is. False if the title wrote the slot again.
    pub(crate) fn publish(&self, index: u32, handle: u64, key: u64) -> bool {
        // SAFETY: the handle is a descriptor slot in title memory.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        if tag != VIEW_TAG || u64::from(generation) << 32 | u64::from(i) != key {
            return false;
        }
        PUBLISHING.set(true);
        let shown = self.mirror(index, handle, true).is_some();
        PUBLISHING.set(false);
        shown
    }

    /// Makes title slot `index`'s host slot (its own, or with a heap map
    /// past the identity window its pool slot) from the record the title
    /// has at `handle` if it changed since, and returns the view recorded
    /// there. Buffer views are made only for draws (`for_draw`), which also
    /// copy their data: a host copy of a buffer is made on first use.
    pub(crate) fn mirror(&self, index: u32, handle: u64, for_draw: bool) -> Option<View> {
        self.mirror_inner(index, handle, for_draw, true).map(|(view, _)| view)
    }

    /// [`HostHeap::mirror`] for `State::heap_buffers` (`for_draw = true`):
    /// returns the view paired with the exact 64-bit key mirrored so the
    /// caller can cache it without re-reading title memory, and avoids
    /// re-inserting `index` into `self.changed` since the caller updates its
    /// own cached range immediately.
    pub(crate) fn mirror_for_buffers(&self, index: u32, handle: u64) -> Option<(View, u64)> {
        self.mirror_inner(index, handle, true, false)
    }

    fn mirror_inner(
        &self,
        index: u32,
        handle: u64,
        for_draw: bool,
        mark_changed: bool,
    ) -> Option<(View, u64)> {
        use std::sync::atomic::Ordering::Relaxed;
        let mirrored = self.mirrored_count();
        if index >= mirrored {
            // Diagnostic (NVIDIA heap cap): the highest title slot past the
            // host heap seen so far.
            static HIGHEST: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if index > HIGHEST.load(Relaxed) && HIGHEST.fetch_max(index, Relaxed) < index {
                tracing::warn!(
                    index,
                    count = self.count,
                    for_draw,
                    "title slot past the host heap"
                );
            }
            return None;
        }
        if self.map.is_some()
            && index >= self.count
            && self.title_start.load(Relaxed) == 0
        {
            // Where the title's descriptors start, for `reclaim`.
            let start = handle.wrapping_sub(u64::from(index) * TITLE_SLOT_BYTES);
            let _ = self
                .title_start
                .compare_exchange(0, start, Relaxed, Relaxed);
        }
        // SAFETY: the handle is a descriptor slot in title memory.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        // Generations start at 1, so a made slot's key is never 0.
        let key = u64::from(generation) << 32 | u64::from(i);
        if let Some(copied) = self.copied.get()
            && let Some(slot) = copied.get(index as usize)
        {
            let seen_key = if tag == VIEW_TAG { key } else { 0 };
            if slot.load(Relaxed) != seen_key {
                slot.store(seen_key, Relaxed);
            }
        }
        // Lock-free fast path for identity-window slots (`index < self.count`):
        // skip `self.made` mutex when this slot is already up to date.
        if tag == VIEW_TAG
            && key != 0
            && index < self.count
            && self.made_fast[index as usize].load(std::sync::atomic::Ordering::Acquire) == key
            && let Some(view) = view_at(i, generation)
        {
            #[cfg(feature = "diagnostics")]
            mirror_how(MIRROR_FAST);
            return Some((view, key));
        }
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        let view = (tag == VIEW_TAG).then(|| view_at(i, generation)).flatten();
        let Some(view) = view else {
            // Not a view of ours (never written, or of a resource the
            // title released): nothing to read.
            #[cfg(feature = "diagnostics")]
            mirror_how(MIRROR_NOT_VIEW);
            self.empty_slot(&mut made, index, for_draw, mark_changed);
            return None;
        };
        let Some((host, taken)) = self.host_slot(&mut made, index) else {
            // Past the window with the pool used up: the slot reads the
            // guard. What the title wrote is noted all the same (the sets
            // of slots shaders pick by index retry it).
            if !for_draw {
                for stale in self.note_buffer_in(index, Some(&view), false, mark_changed) {
                    if stale != index {
                        self.unmake(&mut made, stale);
                    }
                }
            }
            return Some((view, key));
        };
        if taken || made.keys[host as usize] != key {
            // Being made again: not "made" to a lock-free reader until it is
            // made and kept (stored after `self.keep` below).
            if let Some(fast) = self.made_fast.get(host as usize) {
                fast.store(0, std::sync::atomic::Ordering::Release);
            }
            // Views of other textures in the memory this one describes
            // are stale: their host slots read nothing from here on, and
            // let go of the host copies they held.
            #[cfg(feature = "diagnostics")]
            let noting = std::time::Instant::now();
            for stale in self.note_buffer_in(index, Some(&view), for_draw, mark_changed) {
                if stale != index {
                    self.unmake(&mut made, stale);
                }
            }
            #[cfg(feature = "diagnostics")]
            if mirror_timing() {
                MIRROR_US[0].fetch_add(noting.elapsed().as_micros() as u64, Relaxed);
            }
            // A view of a texture object that has no host copy yet waits
            // for it, as buffer views wait for a draw: the slot is made
            // when the texture is first used (`super::make_waiting`), or
            // here when a draw reads the slot through a table.
            let waiting = if for_draw { None } else { view.waiting_texture() };
            if let Some(texture) = waiting {
                super::wait_for(texture, handle);
            }
            let later = !for_draw && (view.is_buffer() || waiting.is_some());
            // Another view of the placed texture the slot shows already (a
            // title that streams a texture in writes its view again with
            // more mips each time some arrive): the slot goes on showing
            // the old view until a draw or the texture pass makes the new
            // one and uploads what it shows. Made null here, the texture
            // read as nothing for as long as it waited in the queue, every
            // time it got sharper.
            let old = made.keys[host as usize];
            if later
                && keep_views()
                && view_at(old as u32, (old >> 32) as u32)
                    .is_some_and(|old| same_placed_texture(&old, &view))
            {
                KEPT_VIEWS.fetch_add(1, Relaxed);
                #[cfg(feature = "diagnostics")]
                mirror_how(MIRROR_KEPT);
                if let Some(fast) = self.made_fast.get(host as usize) {
                    fast.store(old, std::sync::atomic::Ordering::Release);
                }
                return Some((view, key));
            }
            let next_key = if later { 0 } else { key };
            made.keys[host as usize] = next_key;
            #[cfg(feature = "diagnostics")]
            mirror_how(if later { MIRROR_LATER } else { MIRROR_MADE });
            #[cfg(feature = "diagnostics")]
            let creating = std::time::Instant::now();
            let kept = if later {
                None
            } else {
                make(&view, self.cpu(host))
            };
            if kept.is_none() && !self.sampler {
                null_for(&view, self.cpu(host));
            }
            #[cfg(feature = "diagnostics")]
            let keeping = std::time::Instant::now();
            self.keep(host, kept.flatten());
            #[cfg(feature = "diagnostics")]
            if mirror_timing() {
                MIRROR_US[1].fetch_add((keeping - creating).as_micros() as u64, Relaxed);
                MIRROR_US[2].fetch_add(keeping.elapsed().as_micros() as u64, Relaxed);
            }
            if let Some(fast) = self.made_fast.get(host as usize) {
                fast.store(next_key, std::sync::atomic::Ordering::Release);
            }
            if taken {
                self.publish_map(&made, index);
                #[cfg(feature = "diagnostics")]
                note_pool(&made);
            }
        }
        #[cfg(feature = "diagnostics")]
        mirror_how(MIRROR_SAME);
        Some((view, key))
    }

    /// Whether title slot `index` is backed by a made host descriptor (tests).
    #[cfg(test)]
    pub(crate) fn made(&self, index: u32) -> bool {
        let made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        made.host(index)
            .and_then(|host| made.keys.get(host as usize))
            .is_some_and(|&key| key != 0)
    }

    /// Whether title slot `index`'s host slot shows `buffer`; `None` if it
    /// shows no resource (diagnostics: a buffer view's slot must show the
    /// host copy that is brought up to date for it).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn shows(&self, index: u32, buffer: &ID3D12Resource) -> Option<bool> {
        let host = self.made.lock().unwrap_or_else(|e| e.into_inner()).host(index)?;
        let kept = self.kept[(host % KEPT_PIECES) as usize]
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let shown = kept.get((host / KEPT_PIECES) as usize)?.as_ref()?;
        Some(windows::core::Interface::as_raw(shown) == windows::core::Interface::as_raw(buffer))
    }

    /// What the queue of textures picked by index knows of title slot
    /// `index` at `frame`, in a few words (diagnostics).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn texture_slot_state(&self, index: u32, frame: u64) -> String {
        let textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        match textures.slots.get(&index) {
            None => "not queued".into(),
            Some(w) => format!(
                "queued due={} written@{} looks={} part_mapped={} recheck={}",
                if w.due <= frame { "now".to_string() } else { format!("@{}", w.due) },
                w.written,
                w.looks,
                w.mapped,
                w.checked.2 != 0
            ),
        }
    }

    /// The view host slot `index` was made from, in a few words, with the
    /// title and host addresses of a buffer view (diagnostics). "null" for
    /// a slot never made or left null.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn describe_slot(&self, index: u32) -> Option<String> {
        let key = *self.made.lock().unwrap_or_else(|e| e.into_inner()).keys.get(index as usize)?;
        if key == 0 {
            return Some("null".into());
        }
        let view = view_at(key as u32, (key >> 32) as u32)?;
        let mut text: String = format!("{view:?}").chars().take(120).collect();
        if let Some((address, len)) = buffer_range(&view) {
            let host = crate::shadow::find(address, len.max(1)).map(|s| s.gpu(address));
            text.push_str(&format!(" title {address:#x}+{len:#x} host {host:#x?}"));
        }
        Some(text)
    }
}

/// Logs `what` the first time it happens.
macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A view description from its words, with the format as the host has it.
fn srv_desc(words: &[u32; 10]) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_SHADER_RESOURCE_VIEW_DESC>()
            .read_unaligned()
    }
}

fn uav_desc(words: &[u32; 10]) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_UNORDERED_ACCESS_VIEW_DESC>()
            .read_unaligned()
    }
}

/// The format the host makes a typed buffer view in. Translated shaders
/// declare every typed buffer load and store as 32-bit float
/// (`victory_shader` `emit/lower.rs`), and a view in a 32-bit integer format
/// converts between float and integer values on the way: a store of an
/// integer's bits as a float is written as the integer the float rounds to
/// (zero for most of them), where the console moves the bits. The float
/// format of the same width moves them unchanged, and matches what the
/// shader declares. (The meshlet culling programs append packed meshlet
/// numbers to an `R32_UINT` list this way.)
fn bit_exact_format(format: DXGI_FORMAT) -> DXGI_FORMAT {
    match format {
        DXGI_FORMAT_R32G32B32A32_UINT | DXGI_FORMAT_R32G32B32A32_SINT => {
            DXGI_FORMAT_R32G32B32A32_FLOAT
        }
        DXGI_FORMAT_R32G32B32_UINT | DXGI_FORMAT_R32G32B32_SINT => DXGI_FORMAT_R32G32B32_FLOAT,
        DXGI_FORMAT_R32G32_UINT | DXGI_FORMAT_R32G32_SINT => DXGI_FORMAT_R32G32_FLOAT,
        DXGI_FORMAT_R32_UINT | DXGI_FORMAT_R32_SINT => DXGI_FORMAT_R32_FLOAT,
        other => other,
    }
}

/// A buffer view of title memory at `address` on its host copy: the first
/// element moves by the copy's offset.
/// Bytes per element of a buffer view.
fn element_size(desc: &D3D12_SHADER_RESOURCE_VIEW_DESC) -> u64 {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    }
}

/// The title memory a buffer shader or unordered access view reads: its
/// address and length.
pub(crate) fn buffer_range(view: &View) -> Option<(u64, u64)> {
    let (base, words): (u64, &[u32; 10]) = match view {
        View::Shader(r, Some(desc)) => (
            crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64,
            desc,
        ),
        View::Placed(false, resource, desc) if resource[0] == 1 => (
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            desc[..10].try_into().ok()?,
        ),
        View::Unordered(r, _, Some(desc)) => {
            let base = crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64;
            return unordered_range(base, desc);
        }
        View::Placed(true, resource, desc) if resource[0] == 1 => {
            let base = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            return unordered_range(base, desc[..10].try_into().ok()?);
        }
        _ => return None,
    };
    let desc = srv_desc(words);
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// The title memory the host slot of buffer view `view` is made on
/// (address, bytes): the whole buffer the view is of, which is what
/// [`make`] finds the host copy by. [`buffer_range`] is the part of it the
/// view reads.
pub(crate) fn buffer_extent(view: &View) -> Option<(u64, u64)> {
    match view {
        View::Shader(r, Some(_)) | View::Unordered(r, _, Some(_)) => {
            let (data, len) = crate::resource::buffer_bytes(*r as crate::object::This)?;
            Some((data as u64, len))
        }
        View::Placed(_, resource, desc) if resource[0] == 1 => Some((
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            u64::from(resource[4]) | u64::from(resource[5]) << 32,
        )),
        _ => None,
    }
}

/// Whether a buffer view is brought up to date in the host copy its slot
/// shows ([`buffer_copy`]). On by default; `VICTORY_VIEW_EXTENT=0` looks the
/// copy up by the view's range alone, as before.
fn view_extent() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_EXTENT").is_none_or(|v| v != "0"))
}

/// The host copy buffer view `view` reads, and the title range (address,
/// bytes) it reads there; `None` for other views and empty ranges.
///
/// The copy is the one the view's slot shows: [`make`] finds it by the
/// whole buffer the view is of ([`buffer_extent`]). Looked up by the range
/// the view reads, as it was, the copy is another one wherever the two
/// lookups differ: for memory outside buffer objects, a view of a part of a
/// buffer that crosses a window line (the buffer's window is two cells, the
/// part's one), and for buffer objects, a part that also lies in a smaller
/// buffer object. The slot then showed a copy nothing brought up to date,
/// and what shaders wrote through it was looked for in the other copy.
pub(crate) fn buffer_copy(view: &View) -> Option<(std::sync::Arc<crate::shadow::Shadow>, u64, u64)> {
    let (address, len) = buffer_range(view)?;
    if len == 0 {
        return None;
    }
    // Single `buffer_extent` + `shadow::find(base, width)` shared by both
    // `own_bytes` and `view_extent`, avoiding a duplicate global shadow lookup.
    if let Some((base, width)) = buffer_extent(view) {
        let fields = own_copy_on().then(|| buffer_fields(view)).flatten();
        let use_extent = view_extent() && address >= base && address - base < width.max(1);
        if (fields.is_some() || use_extent)
            && let Some(whole) = crate::shadow::find(base, width)
        {
            if let Some((first, count, element, stride, unknown)) = fields
                && let Some((start, bytes)) = own_bytes(whole.offset(base), base, first, count, element, stride, unknown)
                && let Some(own) = crate::shadow::exact(start, bytes)
            {
                return Some((own, start, bytes));
            }
            if use_extent {
                return Some((whole, address, len));
            }
        }
    }
    crate::shadow::find(address, len).map(|shadow| (shadow, address, len))
}

/// [`buffer_range`] of an unordered access view of memory at `base`.
fn unordered_range(base: u64, words: &[u32; 10]) -> Option<(u64, u64)> {
    let desc = uav_desc(words);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32)?.0
    };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// The host resource a slot made by [`make`] views, if any.
type Kept = Option<ID3D12Resource>;

/// Whether the pass over records copied in as bytes also mirrors slots
/// that draws read through tables lately (`VICTORY_HEAP_SYNC_TABLES=1`;
/// off by default: [`HostHeap::mirror_copied`]).
fn sync_tables() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_SYNC_TABLES").is_some_and(|v| v == "1"))
}

/// Since the last frame line: slots on changed heap pages that
/// [`HostHeap::mirror_copied`] left to their draws (tables' slots).
pub(crate) static COPIED_TABLED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

// What the view-heap pass's "make" does with each record it mirrors
// (`HostHeap::scan_copied_chunks` / `HostHeap::scan_copied_dynamic`; fps41
// round 12), by outcome of `HostHeap::mirror`: made already (the lock-free
// look), not a view of ours, the same view as made (under the lock), the old
// view kept, left for later (a buffer or a texture with no host copy: a null
// view made), made (a host view created). Diagnostics only.
#[cfg(feature = "diagnostics")]
const MIRROR_UNSET: usize = 6;
/// Not inside the pass's timed `mirror` (draws and other passes mirror too).
#[cfg(feature = "diagnostics")]
const MIRROR_IDLE: usize = 7;
#[cfg(feature = "diagnostics")]
const MIRROR_FAST: usize = 0;
#[cfg(feature = "diagnostics")]
const MIRROR_NOT_VIEW: usize = 1;
#[cfg(feature = "diagnostics")]
const MIRROR_SAME: usize = 2;
#[cfg(feature = "diagnostics")]
const MIRROR_KEPT: usize = 3;
#[cfg(feature = "diagnostics")]
const MIRROR_LATER: usize = 4;
#[cfg(feature = "diagnostics")]
const MIRROR_MADE: usize = 5;
#[cfg(feature = "diagnostics")]
thread_local! {
    /// The outcome of the `mirror` call being timed on this thread.
    static MIRROR_HOW: std::cell::Cell<usize> = const { std::cell::Cell::new(MIRROR_IDLE) };
}
/// By outcome (the six above and "other"): records, and their microseconds.
#[cfg(feature = "diagnostics")]
pub(crate) static MIRRORED: [std::sync::atomic::AtomicU64; 14] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 14];
/// Inside `mirror`, summed over the records: the buffer notes and the
/// unmaking of stale slots, creating the view (or the null one), keeping
/// the slot's resource (the old one retired).
#[cfg(feature = "diagnostics")]
pub(crate) static MIRROR_US: [std::sync::atomic::AtomicU64; 3] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 3];

#[cfg(feature = "diagnostics")]
fn mirror_how(how: usize) {
    MIRROR_HOW.with(|cell| {
        if cell.get() == MIRROR_UNSET {
            cell.set(how);
        }
    });
}

/// Whether this thread is inside the pass's timed `mirror`.
#[cfg(feature = "diagnostics")]
fn mirror_timing() -> bool {
    MIRROR_HOW.with(|cell| cell.get() != MIRROR_IDLE)
}

#[cfg(feature = "diagnostics")]
fn mirror_counted(us: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    let how = MIRROR_HOW.with(|cell| cell.replace(MIRROR_IDLE)).min(MIRROR_UNSET);
    MIRRORED[how].fetch_add(1, Relaxed);
    MIRRORED[7 + how].fetch_add(us, Relaxed);
}

/// A 64-bit fingerprint of a descriptor record's bytes (16 or 32 of them):
/// equal records give equal prints, and two different records the same
/// print about once in 2^64. Not a hash anyone can attack: the records
/// are our own.
#[cfg(test)]
fn record_print(record: &[u8]) -> u64 {
    const K: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut h = K ^ record.len() as u64;
    let mut words = record.chunks_exact(8);
    for word in &mut words {
        let w = u64::from_le_bytes(word.try_into().unwrap_or([0; 8]));
        h = (h ^ w).wrapping_mul(K);
        h ^= h >> 29;
    }
    for &b in words.remainder() {
        h = (h ^ u64::from(b)).wrapping_mul(K);
    }
    h ^ (h >> 32)
}

/// Whether a buffer view off its buffer's elements gets a copy of its own
/// bytes (`VICTORY_VIEW_OWN_COPY=0`: it is refused and reads zeros, as
/// before).
fn own_copy_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_OWN_COPY").is_none_or(|v| v != "0"))
}

/// The title memory (start, bytes) of a buffer view that cannot be made on
/// the host copy of its buffer: `count` elements of `element` bytes from
/// element `first` of the buffer at `base`, which lies `offset` bytes into
/// that copy, where `offset` is not a whole number of elements and the view
/// cannot be a raw view of the same bytes either ([`raw_window`]:
/// `unknown`, the view has no format, and `stride` is its structure's).
/// The host counts a view's first element from its buffer's start, so such
/// a view has no host view there; it is made on a copy of its own bytes
/// (`crate::shadow::exact`), where it starts at element 0. `None`: the
/// view is made on its buffer's copy, as every other.
fn own_bytes(offset: u64, base: u64, first: u64, count: u32, element: u64, stride: u32, unknown: bool) -> Option<(u64, u64)> {
    own_bytes_if(own_copy_on(), offset, base, first, count, element, stride, unknown)
}

#[allow(clippy::too_many_arguments)]
fn own_bytes_if(
    on: bool,
    offset: u64,
    base: u64,
    first: u64,
    count: u32,
    element: u64,
    stride: u32,
    unknown: bool,
) -> Option<(u64, u64)> {
    if !on || element == 0 || count == 0 || offset.is_multiple_of(element) {
        return None;
    }
    if unknown && raw_window(offset, first, count, stride).is_some() {
        return None;
    }
    let start = base.checked_add(first.checked_mul(element)?)?;
    Some((start, u64::from(count) * element))
}

/// Of a buffer view: its first element, its elements, the bytes of one (a
/// raw view's: 4), its structure's stride, and whether it has no format.
/// As [`buffer_srv`] and [`buffer_uav`] read them from the same words.
fn buffer_fields(view: &View) -> Option<(u64, u32, u64, u32, bool)> {
    const BUFFER: u32 = 1;
    let (words, unordered): (&[u32; 10], bool) = match view {
        View::Shader(_, Some(d)) if d[1] == BUFFER => (d, false),
        View::Unordered(_, _, Some(d)) if d[1] == BUFFER => (d, true),
        View::Placed(unordered, resource, d) if resource[0] == BUFFER => (d[..10].try_into().ok()?, *unordered),
        _ => return None,
    };
    if unordered {
        let d = uav_desc(words);
        // SAFETY: a buffer view's union member.
        let b = unsafe { d.Anonymous.Buffer };
        let element = if b.StructureByteStride > 0 {
            u64::from(b.StructureByteStride)
        } else if b.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
            4
        } else {
            crate::resource::element(d.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
        };
        Some((b.FirstElement, b.NumElements, element, b.StructureByteStride, d.Format == DXGI_FORMAT_UNKNOWN))
    } else {
        let d = srv_desc(words);
        // SAFETY: a buffer view's union member.
        let b = unsafe { d.Anonymous.Buffer };
        Some((b.FirstElement, b.NumElements, element_size(&d), b.StructureByteStride, d.Format == DXGI_FORMAT_UNKNOWN))
    }
}

/// A structured buffer view (`first`, `count` elements of `stride` bytes,
/// `offset` bytes into its host copy) as a raw view of the same bytes:
/// its first dword and its dwords. Translated shaders read untyped buffers
/// as raw bytes at `element * stride + offset` (the stride from the
/// used-resource entry, or from the table by heap slot for a buffer picked
/// by index), and a raw read through a structured view gives zeros on
/// NVIDIA (reported upstream with PR #193; structured reads through raw
/// views were right there). So the host's view is raw whenever it can be:
/// the bytes must start on 16 (the host's raw view alignment) and be whole
/// dwords. A view that cannot stays structured.
fn raw_window(offset: u64, first: u64, count: u32, stride: u32) -> Option<(u64, u32)> {
    if stride == 0 {
        return None;
    }
    let start = first.checked_mul(u64::from(stride))?.checked_add(offset)?;
    let bytes = u64::from(count) * u64::from(stride);
    (start % 16 == 0 && bytes % 4 == 0)
        .then(|| u32::try_from(bytes / 4).ok().map(|dwords| (start / 4, dwords)))
        .flatten()
}

/// Whether the host accepts a buffer view with these fields of a buffer of
/// `width` bytes. It removes the device for one it does not (view creation
/// returns nothing), and the console is laxer: the title's heap holds
/// views past their memory's end, of no elements, of typeless formats
/// without the raw flag, with strides the host has no structured buffers
/// for. Since every buffer view of the heap is made for shaders that pick
/// them by index, one such view ended the run (2026-11-24, run 10).
fn buffer_view_ok(format: DXGI_FORMAT, first: u64, count: u32, stride: u32, raw: bool, width: u64) -> bool {
    use crate::resource::family;
    let f = format.0 as u32;
    let element = if raw {
        // raw: R32 typeless, counted in dwords, starting on 16 bytes
        if format != DXGI_FORMAT_R32_TYPELESS || stride != 0 || !first.is_multiple_of(4) {
            return false;
        }
        4
    } else if stride > 0 {
        // structured: no format, a stride the host takes
        if format != DXGI_FORMAT_UNKNOWN || stride > 2048 || !stride.is_multiple_of(4) {
            return false;
        }
        u64::from(stride)
    } else {
        // typed: a format with a size that is not typeless
        let typeless = family(f) == f && family(f + 1) == f;
        match crate::resource::element(f) {
            Some((bytes, 1)) if f != 0 && !typeless && bytes > 0 => bytes,
            _ => return false,
        }
    };
    count > 0
        && first
            .checked_add(u64::from(count))
            .and_then(|end| end.checked_mul(element))
            .is_some_and(|end| end <= width)
}

fn buffer_srv(
    address: u64,
    len: u64,
    mut desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    let shadow = crate::shadow::find(address, len)?;
    let offset = shadow.offset(address);
    // Off the elements of that copy: on a copy of its own bytes, from its
    // element 0 (`own_bytes`).
    let (shadow, offset) = {
        // SAFETY: a buffer view's union member.
        let b = unsafe { desc.Anonymous.Buffer };
        let unknown = desc.Format == DXGI_FORMAT_UNKNOWN;
        match own_bytes(offset, address, b.FirstElement, b.NumElements, element_size(&desc), b.StructureByteStride, unknown)
            .and_then(|(start, bytes)| crate::shadow::exact(start, bytes))
        {
            Some(own) => {
                desc.Anonymous.Buffer.FirstElement = 0;
                (own, 0)
            }
            None => (shadow, offset),
        }
    };
    // A structured view as a raw view of the same bytes (see `raw_window`).
    // SAFETY: a buffer view's union member.
    let structured = unsafe { desc.Anonymous.Buffer };
    let window = (desc.Format == DXGI_FORMAT_UNKNOWN)
        .then(|| raw_window(offset, structured.FirstElement, structured.NumElements, structured.StructureByteStride))
        .flatten();
    if let Some((first, count)) = window {
        desc.Format = DXGI_FORMAT_R32_TYPELESS;
        desc.Anonymous.Buffer = D3D12_BUFFER_SRV {
            FirstElement: first,
            NumElements: count,
            StructureByteStride: 0,
            Flags: D3D12_BUFFER_SRV_FLAG_RAW,
        };
    }
    desc.Format = bit_exact_format(desc.Format);
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    if window.is_none() {
        if element == 0 || offset % element != 0 {
            once!(
                offset,
                element,
                "buffer view not on an element of its host copy"
            );
            return None;
        }
        buffer.FirstElement += offset / element;
    }
    let (first, count, stride) = (buffer.FirstElement, buffer.NumElements, buffer.StructureByteStride);
    let raw = buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0;
    // SAFETY: a live buffer.
    let width = unsafe { shadow.buffer.GetDesc() }.Width;
    if !buffer_view_ok(desc.Format, first, count, stride, raw, width) {
        REFUSED_VIEWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        once!(
            format = desc.Format.0,
            first,
            count,
            stride,
            raw,
            width,
            "buffer view the host would refuse left null"
        );
        return None;
    }
    // The console has no component mapping; one the host does not know is the default's.
    if desc.Shader4ComponentMapping != D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING {
        desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    }
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateShaderResourceView(&shadow.buffer, Some(&desc), at)
    };
    Some(Some(shadow.buffer.clone()))
}

/// An unordered access view of title buffer memory at `address` (`len`
/// bytes) on its host copy, as [`buffer_srv`]; false if the copy cannot be
/// written by shaders (upload memory).
/// shortcut: what shaders write stays in the host copy; title memory (and
/// so the CPU, and copies run on the CPU) does not see it.
fn buffer_uav(
    address: u64,
    len: u64,
    mut desc: D3D12_UNORDERED_ACCESS_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    let shadow = crate::shadow::find(address, len)?;
    // Off the elements of that copy: on a copy of its own bytes, from its
    // element 0 (`own_bytes`).
    let (shadow, own) = {
        // SAFETY: a buffer view's union member.
        let b = unsafe { desc.Anonymous.Buffer };
        let element = if b.StructureByteStride > 0 {
            u64::from(b.StructureByteStride)
        } else if b.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
            4
        } else {
            crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
        };
        let unknown = desc.Format == DXGI_FORMAT_UNKNOWN;
        match own_bytes(shadow.offset(address), address, b.FirstElement, b.NumElements, element, b.StructureByteStride, unknown)
            .and_then(|(start, bytes)| crate::shadow::exact(start, bytes))
        {
            Some(own) => {
                desc.Anonymous.Buffer.FirstElement = 0;
                (own, true)
            }
            None => (shadow, false),
        }
    };
    // SAFETY: a live buffer.
    let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
    if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
        once!("unordered access to a buffer whose host copy shaders cannot write");
        return None;
    }
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    };
    let offset = if own { 0 } else { shadow.offset(address) };
    // A structured view as a raw view of the same bytes (see `raw_window`).
    let window = (desc.Format == DXGI_FORMAT_UNKNOWN)
        .then(|| raw_window(offset, buffer.FirstElement, buffer.NumElements, buffer.StructureByteStride))
        .flatten();
    if let Some((first, count)) = window {
        buffer.FirstElement = first;
        buffer.NumElements = count;
        buffer.StructureByteStride = 0;
        buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    } else {
        if element == 0 || offset % element != 0 {
            once!(offset, element, "unordered buffer view not on an element of its host copy");
            return None;
        }
        buffer.FirstElement += offset / element;
    }
    buffer.CounterOffsetInBytes = 0;
    let (first, count, stride) = (buffer.FirstElement, buffer.NumElements, buffer.StructureByteStride);
    let raw = buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0;
    desc.Format = bit_exact_format(desc.Format);
    if window.is_some() {
        // (Left out at first: every structured unordered access view was then
        // refused by `buffer_view_ok` and left null, runs 15 and 16.)
        desc.Format = DXGI_FORMAT_R32_TYPELESS;
    }
    // SAFETY: a live buffer.
    let width = unsafe { shadow.buffer.GetDesc() }.Width;
    if !buffer_view_ok(desc.Format, first, count, stride, raw, width) {
        REFUSED_VIEWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        once!(
            format = desc.Format.0,
            first,
            count,
            stride,
            raw,
            width,
            "unordered buffer view the host would refuse left null"
        );
        return None;
    }
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateUnorderedAccessView(&shadow.buffer, None::<&ID3D12Resource>, Some(&desc), at)
    };
    Some(Some(shadow.buffer.clone()))
}

/// Whether the host can make a shader (or unordered access) view of
/// `texture` in `format` (none: the texture's own). The console allows any
/// view; the host refuses (and removes the device for) views without the
/// resource's permission, of typed depth, and in another format than a
/// typed resource's. A view of a typed texture in another format of its
/// family takes the texture's format.
/// shortcut: such a view (sRGB of UNORM, say) reads without its
/// conversion; cast with castable formats if colors come out wrong.
fn fits(texture: &ID3D12Resource, mut format: Option<&mut DXGI_FORMAT>, unordered: bool) -> bool {
    use crate::resource::{DEPTH_FORMATS, family};
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let allowed = if unordered {
        d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
    } else {
        d.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 == 0
    };
    let own = d.Format.0 as u32;
    if !allowed || DEPTH_FORMATS.iter().any(|&(typed, _)| typed == own) {
        if unordered {
            crate::resource::DEPTH_UNORDERED[3].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        once!(
            unordered,
            format = own,
            flags = d.Flags.0,
            "view the host does not allow"
        );
        return false;
    }
    // Every family has more than one format, its typeless one first.
    let typeless = family(own) == own && family(own + 1) == own;
    if let Some(format) = format.as_deref_mut()
        && unordered
        && let Some(&(_, linear)) = crate::resource::SRGB
            .iter()
            .find(|&&(s, _)| s == format.0 as u32)
    {
        format.0 = linear as i32;
    }
    let view = format.as_ref().map_or(0, |f| f.0 as u32);
    // A typed unordered view in a format the host cannot store to
    // (B5G5R5A1_UNORM on NVIDIA) removes the device when made, in the
    // texture's own format too: the slot stays null, its writes dropped.
    // The format checked is the one the view is made in: a typed
    // texture's own unless the texture is typeless.
    let typed = if view == 0 || !typeless { own } else { view };
    if unordered && !(view == 0 && typeless) && !typed_store(typed) {
        once!(view = typed, own, "an unordered view in a format the host cannot store");
        return false;
    }
    if view == 0 {
        return !typeless;
    }
    // A view of a target whose format the host refuses, in the title's
    // format: in the stand-in the host copy has.
    if view != own && crate::resource::target_format(view) == own {
        if let Some(format) = format {
            *format = d.Format;
        }
        return true;
    }
    if family(view) != family(own) {
        once!(view, own, "view format of another family");
        return false;
    }
    if let Some(format) = format
        && !typeless
    {
        *format = d.Format;
    }
    true
}

/// Whether the host stores to typed unordered views of `format`.
fn typed_store(format: u32) -> bool {
    static KNOWN: Mutex<Vec<(u32, bool)>> = Mutex::new(Vec::new());
    let mut known = KNOWN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(&(_, ok)) = known.iter().find(|(f, _)| *f == format) {
        return ok;
    }
    let mut support = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
        Format: DXGI_FORMAT(format as i32),
        ..Default::default()
    };
    let ok = host().is_some_and(|h| {
        // SAFETY: a plain query into a struct of its size.
        unsafe {
            h.device.CheckFeatureSupport(
                D3D12_FEATURE_FORMAT_SUPPORT,
                (&raw mut support).cast(),
                size_of_val(&support) as u32,
            )
        }
        .is_ok()
            && support.Support1.0 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW.0 != 0
    });
    known.push((format, ok));
    ok
}

/// The title's unordered access view of a depth texture, as a view of the
/// texture's colour twin `twin` ([`crate::resource::Twin`]): the same mip
/// and slices, in the colour format of the twin's family for a depth or
/// typeless format (R16_UNORM for 16-bit depth, R32_FLOAT for 32-bit; the
/// bits are the same). Without a description: mip 0 of every slice.
/// The twin of a stencil plane (R8) stands for the views of that plane
/// only, which become R8_UINT views of its one plane; `None` for a view of
/// the depth plane, which the twin does not stand for. The depth plane has
/// a twin of its own, viewed as a depth texture's without stencil: a view
/// in the format of the depth plane of a depth-stencil format (21, 46)
/// takes the twin's colour format too.
fn twin_uav(
    twin: &ID3D12Resource,
    desc: Option<&D3D12_UNORDERED_ACCESS_VIEW_DESC>,
) -> Option<D3D12_UNORDERED_ACCESS_VIEW_DESC> {
    // SAFETY: a live resource.
    let d = unsafe { twin.GetDesc() };
    let mut desc = desc.cloned().unwrap_or_else(|| {
        let mut all = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        if d.DepthOrArraySize > 1 {
            all.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            all.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                ArraySize: u32::from(d.DepthOrArraySize),
                ..Default::default()
            };
        }
        all
    });
    // A stencil plane's twin (R8 typeless, 60): the title's view is of the
    // stencil plane if it names plane 1 or a stencil view format (22, 47).
    if d.Format.0 == 60 {
        let mut plane = 0;
        match desc.ViewDimension {
            D3D12_UAV_DIMENSION_TEXTURE2D => {
                // SAFETY: the member the dimension names.
                let mut view = unsafe { desc.Anonymous.Texture2D };
                plane = std::mem::take(&mut view.PlaneSlice);
                desc.Anonymous.Texture2D = view;
            }
            D3D12_UAV_DIMENSION_TEXTURE2DARRAY => {
                // SAFETY: the member the dimension names.
                let mut view = unsafe { desc.Anonymous.Texture2DArray };
                plane = std::mem::take(&mut view.PlaneSlice);
                desc.Anonymous.Texture2DArray = view;
            }
            _ => {}
        }
        if plane != 1 && !matches!(desc.Format.0, 22 | 47) {
            return None;
        }
        desc.Format.0 = 62;
        return Some(desc);
    }
    // None, typeless and depth formats (R16: 0, 53, 55; R32: 0, 39, 40),
    // and those of a depth-stencil format's depth plane (19 to 21, 44 to
    // 46), for the twin of such a plane.
    desc.Format.0 = match (desc.Format.0, d.Format.0) {
        (0 | 53 | 55 | 19..=21 | 44..=46, 53) => 56,
        (0 | 39 | 40 | 19..=21 | 44..=46, 39) => 41,
        (other, _) => other,
    };
    Some(desc)
}

/// The size of a constant buffer view the host takes: 256-byte steps, and
/// at most 4,096 vectors (65,536 bytes, `D3D12_REQ_CONSTANT_BUFFER_ELEMENT_COUNT`).
/// The title names constant views of more (one of 1,306,368 bytes, in the
/// world); `CreateConstantBufferView` with such a size is an invalid call,
/// and NVIDIA removes the device for it (`DXGI_ERROR_INVALID_CALL`), in
/// the frame the view is made. A shader reads at most 4,096 vectors through
/// a constant view, so nothing it can read is lost.
fn constant_view_size(size: u32) -> u32 {
    const MOST: u32 = 4096 * 16;
    // The cap before the rounding (fork pull request #1206): a size within
    // 255 of the largest number would round past it, to 0 in a release
    // build, and a view of no bytes be made with no line said.
    if size > MOST {
        once!(size, "a constant buffer view over 64 KiB, made 64 KiB (the host takes no more)");
        return MOST;
    }
    // A view of no bytes stays one: a shader reads zeros through it, as on the
    // console. Rounded up from 1 it became 256 bytes of whatever lies at its
    // address in the host copy (the ring of per-draw constants among it), other
    // bytes every frame: a floor in the first rooms flickered.
    size.next_multiple_of(256)
}

/// Makes the host descriptor for `view` at `at`: the host resource it
/// views, or `None` if it cannot.
fn make(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<Kept> {
    let host = host()?;
    let device = &host.device;
    match view {
        View::NullConstant => {
            null_for(view, at);
            Some(None)
        }
        &View::Constant(address, size) => {
            let shadow = crate::shadow::find(address, u64::from(size))?;
            let gpu = shadow.gpu(address);
            if gpu % 256 != 0 {
                once!(
                    address = format_args!("{address:#x}"),
                    "constant buffer not 256-byte aligned on the host"
                );
                return None;
            }
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
                BufferLocation: gpu,
                SizeInBytes: constant_view_size(size),
            };
            // SAFETY: a slot of our heap; the range lies in the host copy.
            unsafe { device.CreateConstantBufferView(Some(&desc), at) };
            Some(Some(shadow.buffer.clone()))
        }
        View::Shader(resource, desc) => {
            let resource = *resource as crate::object::This;
            if let Some(texture) = crate::resource::host_resource(resource) {
                // The title's views of textures with a stencil plane, as it
                // describes them (format, dimension, component mapping, a
                // pad, then first level, levels, plane): each distinct one
                // once. They say how its shaders read the stencil plane
                // (the lighting takes each pixel's material class and its
                // "inside a room" bit from it).
                #[cfg(feature = "diagnostics")]
                {
                    // SAFETY: a live resource.
                    let own = unsafe { texture.GetDesc() }.Format.0;
                    if matches!(own, 19..=22 | 44..=47) {
                        static SEEN: Mutex<Vec<(i32, Option<[u32; 10]>)>> = Mutex::new(Vec::new());
                        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
                        if seen.len() < 64 && !seen.contains(&(own, *desc)) {
                            seen.push((own, *desc));
                            tracing::info!(
                                ?resource,
                                texture_format = own,
                                view = ?desc,
                                "shader view of a depth-stencil texture"
                            );
                        }
                    }
                }
                let mut desc = desc.as_ref().map(srv_desc);
                if !fits(texture, desc.as_mut().map(|d| &mut d.Format), false) {
                    return None;
                }
                // Levels on demand (`crate::tiles`): a texture made level by
                // level is shown from the first level that is the title's
                // and has memory, if the title's view starts before it (a
                // level without memory reads zeros: a black texture). The
                // title's own view mostly starts there already.
                if let Some(first) = crate::resource::first_shown(resource)
                    && first > 0
                {
                    // SAFETY: a live resource.
                    let own = unsafe { texture.GetDesc() };
                    // The texture's own view (the title gave none): all of it.
                    let whole = D3D12_SHADER_RESOURCE_VIEW_DESC {
                        Format: own.Format,
                        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                        Shader4ComponentMapping: 5768,
                        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                            Texture2D: D3D12_TEX2D_SRV {
                                MostDetailedMip: 0,
                                MipLevels: u32::MAX,
                                PlaneSlice: 0,
                                ResourceMinLODClamp: 0.0,
                            },
                        },
                    };
                    crate::tiles::show_from(desc.get_or_insert(whole), first, u32::from(own.MipLevels));
                }
                // SAFETY: a live texture and a slot of our heap.
                unsafe {
                    device.CreateShaderResourceView(
                        texture,
                        desc.as_ref().map(std::ptr::from_ref),
                        at,
                    )
                };
                Some(Some(texture.clone()))
            } else if let (Some((data, len)), Some(desc)) =
                (crate::resource::buffer_bytes(resource), desc)
            {
                buffer_srv(data as u64, len, srv_desc(desc), at)
            } else {
                None
            }
        }
        View::Unordered(resource, _counter, desc) => {
            let resource = *resource as crate::object::This;
            let Some(texture) = crate::resource::host_resource(resource) else {
                return match (crate::resource::buffer_bytes(resource), desc) {
                    (Some((data, len)), Some(desc)) => {
                        buffer_uav(data as u64, len, uav_desc(desc), at)
                    }
                    _ => None,
                };
            };
            // The first such views of depth textures are logged as the
            // title describes them (format, dimension, then mip, slices and
            // plane): they say which planes shaders write, and as what.
            if crate::resource::depth_and_unordered(resource) {
                static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 512 {
                    tracing::info!(
                        ?resource,
                        view = ?desc,
                        "unordered access view of a depth texture"
                    );
                }
            }
            let mut desc = desc.as_ref().map(uav_desc);
            // A depth texture's unordered access is its colour twin's: the
            // host gives a depth texture none.
            let twin = crate::resource::twin_resource(resource)
                .and_then(|twin| Some((twin, twin_uav(twin, desc.as_ref())?)))
                // A view of the depth plane of a texture with a stencil
                // plane: of that plane's own twin.
                .or_else(|| {
                    let twin = crate::resource::twin_plane_resource(resource)?;
                    Some((twin, twin_uav(twin, desc.as_ref())?))
                });
            let texture = match twin {
                Some((twin, view)) => {
                    desc = Some(view);
                    twin
                }
                None => texture,
            };
            if !fits(texture, desc.as_mut().map(|d| &mut d.Format), true) {
                return None;
            }
            // SAFETY: a live texture and a slot of our heap.
            unsafe {
                device.CreateUnorderedAccessView(
                    texture,
                    None::<&ID3D12Resource>,
                    desc.as_ref().map(std::ptr::from_ref),
                    at,
                )
            };
            Some(Some(texture.clone()))
        }
        View::Sampler(words) => {
            // SAFETY: 13 words in the host's layout.
            let desc = unsafe {
                std::ptr::from_ref(words)
                    .cast::<D3D12_SAMPLER_DESC>()
                    .read_unaligned()
            };
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::ConsoleSampler(words) => {
            let desc = crate::sampler::console(words);
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::Placed(unordered, resource, desc) => {
            if let Some(view) = placed_texture(view) {
                return make(&view, at);
            }
            const BUFFER: u32 = 1;
            if resource[0] != BUFFER {
                // shortcut: unordered access to a texture the title placed
                // itself is not mirrored; give it a resource as shader
                // views of such textures get when a dispatch writes one.
                once!(dimension = resource[0], "placed unordered texture view not mirrored");
                return None;
            }
            // The resource description's width, after the alignment.
            let len = u64::from(resource[4]) | u64::from(resource[5]) << 32;
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let words: &[u32; 10] = desc[..10].try_into().unwrap_or(&[0; 10]);
            // An empty title buffer: nothing to view, and no host copy to
            // look for (#70: such views were in flight at device removals).
            if len == 0 {
                return None;
            }
            if *unordered {
                buffer_uav(address, len, uav_desc(words), at)
            } else {
                buffer_srv(address, len, srv_desc(words), at)
            }
        }
    }
}

/// Writes of a slot with another view of the placed texture it showed,
/// since the start: the slot kept showing the old view.
pub(crate) static KEPT_VIEWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether a slot written with another view of the placed texture it
/// shows keeps showing the old one until the new is made;
/// `VICTORY_KEEP_VIEWS=0` makes it null at once, as before.
fn keep_views() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_KEEP_VIEWS").is_none_or(|v| v != "0"))
}

/// Whether both are shader views of one texture the title placed in
/// memory: the same address and the same resource description, whatever
/// part of it each shows.
fn same_placed_texture(a: &View, b: &View) -> bool {
    match (a, b) {
        (View::Placed(false, resource, desc), View::Placed(false, other, other_desc)) => {
            resource[0] != 1 && resource == other && desc[10..12] == other_desc[10..12]
        }
        _ => false,
    }
}

/// Whether `view` is a view of a texture the title placed in memory itself
/// whose host texture (or a plane of its twin) is no longer `held`, the
/// resource a slot made from it keeps: a later unordered or renderable view
/// upgraded the memory's host texture.
fn placed_texture_moved(view: &View, held: *mut std::ffi::c_void) -> bool {
    is_placed_texture(view)
        && placed_texture(view).is_some_and(|p| {
            let (View::Shader(r, _) | View::Unordered(r, ..)) = p else {
                return false;
            };
            let r = r as crate::object::This;
            let same_main = crate::resource::host_copy(r)
                .is_some_and(|copy| windows::core::Interface::as_raw(&copy.texture) == held);
            let same_twin = crate::resource::twin(r).is_some_and(|t| {
                windows::core::Interface::as_raw(&t.colour.texture) == held
                    || t.plane
                        .as_ref()
                        .is_some_and(|p| windows::core::Interface::as_raw(&p.texture) == held)
            });
            !(same_main || same_twin)
        })
}

/// Whether `view` is a view of a texture the title placed in memory itself.
fn is_placed_texture(view: &View) -> bool {
    matches!(view, View::Placed(_, resource, _) if resource[0] != 1)
}

/// A shader view of a texture the title placed in memory itself, as a view
/// of the resource made for that memory (on first use).
pub(crate) fn placed_texture(view: &View) -> Option<View> {
    const BUFFER: u32 = 1;
    let View::Placed(unordered, resource, desc) = view else {
        return None;
    };
    if resource[0] == BUFFER {
        return None;
    }
    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
    let texture = crate::resource::texture_at(address, resource, *unordered)?;
    let words = desc[..10].try_into().ok();
    if *unordered {
        // Unordered access goes through the mirror as its own resource:
        // state transitions and twin use in `readable` apply, and reads
        // and writes share the one copy.
        Some(View::Unordered(texture as usize, 0, words))
    } else {
        Some(View::Shader(texture as usize, words))
    }
}

/// A null view of the kind `view` is, for a slot whose view the host
/// cannot make: unordered access (to a buffer or a 2D texture) or a shader
/// view. Writes to it are dropped, reads give zeros.
fn null_for(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let is_buffer = |r: &usize| crate::resource::buffer_bytes(*r as crate::object::This).is_some();
    // Unordered views: whether a buffer, and the title's view dimension.
    let (buffer, dimension) = match view {
        View::Unordered(r, _, desc) => (
            is_buffer(r) || desc.is_some_and(|d| d[1] == 1),
            desc.map(|d| d[1]),
        ),
        View::Placed(true, resource, desc) => (resource[0] == 1, Some(desc[1])),
        // Shader views of buffers read nothing through a null buffer view:
        // a null 2D view where the shader expects a buffer is a mismatch
        // the hardware does not survive in general (GPU-based validation
        // reports it).
        View::Shader(r, desc) if is_buffer(r) || desc.is_some_and(|d| d[1] == 1) => {
            return null_buffer_view(at);
        }
        View::Placed(false, resource, _) if resource[0] == 1 => return null_buffer_view(at),
        // Textures: a null view of the title's dimension (1D, arrays, 3D,
        // cubes), not always 2D.
        View::Shader(_, Some(desc)) => return null_like(desc, at),
        View::Placed(false, _, desc) => {
            return null_like(desc[..10].try_into().unwrap_or(&[0; 10]), at);
        }
        View::NullConstant | View::Constant(..) => {
            // SAFETY: a live host heap slot; a null description creates an unbound CBV.
            return unsafe { host.device.CreateConstantBufferView(None::<*const D3D12_CONSTANT_BUFFER_VIEW_DESC>, at) };
        }
        _ => return null_view(at),
    };
    let desc = if buffer {
        D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            ..Default::default()
        }
    } else {
        null_uav_desc(dimension)
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device.CreateUnorderedAccessView(
            None::<&ID3D12Resource>,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
}

/// A null texture UAV of the title's view dimension (1D, 1D or 2D arrays,
/// 3D), 2D if it names none: a 2D null view where the shader writes a 3D
/// texture is a mismatch GPU-based validation reports (the world's wind
/// field, CS_CopyWindFieldBeforeUpdateWithSmooth, 2026-11-24).
fn null_uav_desc(dimension: Option<u32>) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_UINT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        ..Default::default()
    };
    match dimension.map(|d| D3D12_UAV_DIMENSION(d as i32)) {
        Some(D3D12_UAV_DIMENSION_TEXTURE1D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE1DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
            desc.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE2DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE3D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            desc.Anonymous.Texture3D = D3D12_TEX3D_UAV {
                WSize: u32::MAX,
                ..Default::default()
            };
        }
        _ => {}
    }
    desc
}

/// A buffer view that reads zeros (raw, as translated shaders mostly read
/// buffers).
fn null_buffer_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                NumElements: 1,
                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A texture view that reads zeros with the dimension (and mip and array
/// counts) of the title's description `words`; 2D if it names none.
fn null_like(words: &[u32; 10], at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    const TEXTURES: std::ops::RangeInclusive<u32> = 2..=10;
    if !TEXTURES.contains(&words[1]) {
        return null_view(at);
    }
    let Some(host) = host() else { return };
    // One mip, one slice or cube, plane 0: fields a null view still checks.
    let mut w = [0u32; 10];
    w[1] = words[1];
    match words[1] {
        6 => {}
        7 => w[5] = 1,
        3 | 5 | 10 => (w[5], w[7]) = (1, 1),
        _ => w[5] = 1,
    }
    let mut desc = srv_desc(&w);
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A view that reads zeros, for slots whose view the host cannot make.
/// shortcut: slots with no view at all (kind unknown) get this null 2D
/// view; null buffer views there did not stop the world's low-address
/// page faults (2026-11-24, run w10), so they are not the cause.
fn null_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MipLevels: 1,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Overflow views are off with `VICTORY_OVERFLOW_VIEWS=0`, on otherwise.
    #[test]
    fn overflow_views_are_on_unless_switched_off() {
        let on = |v: Option<&str>| overflow_views_from(v.map(std::ffi::OsStr::new));
        assert!(on(None) && on(Some("1")) && !on(Some("0")));
    }

    /// Typed buffer views of 32-bit integer formats are made in the float
    /// format of the same width (the shaders' float loads and stores then
    /// move the bits); every other format is the title's.
    #[test]
    fn typed_buffer_views_of_32_bit_integers_move_bits() {
        for (title, host) in [
            (DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R32_FLOAT),
            (DXGI_FORMAT_R32_SINT, DXGI_FORMAT_R32_FLOAT),
            (DXGI_FORMAT_R32G32_UINT, DXGI_FORMAT_R32G32_FLOAT),
            (DXGI_FORMAT_R32G32B32_SINT, DXGI_FORMAT_R32G32B32_FLOAT),
            (DXGI_FORMAT_R32G32B32A32_UINT, DXGI_FORMAT_R32G32B32A32_FLOAT),
        ] {
            assert_eq!(bit_exact_format(title), host);
            assert_eq!(
                crate::resource::element(title.0 as u32),
                crate::resource::element(host.0 as u32)
            );
        }
        for kept in [
            DXGI_FORMAT_R32_FLOAT,
            DXGI_FORMAT_R32_TYPELESS,
            DXGI_FORMAT_UNKNOWN,
            DXGI_FORMAT_R16G16_UINT,
            DXGI_FORMAT_R8G8B8A8_UNORM,
        ] {
            assert_eq!(bit_exact_format(kept), kept);
        }
    }

    /// `HostHeap::evict_overflow` clears only overflow slots holding an evicted
    /// texture, resetting their key to 0 so the next `copy_overflow` remakes
    /// them, and drops the retired COM references after releasing the shard lock.
    #[test]
    fn evict_overflow_clears_only_given_up_textures() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        TEST_TITLE_SLOTS.set(4);
        let heap = HostHeap::new(false, 8);
        TEST_TITLE_SLOTS.set(0);
        let heap = heap.expect("view heap");
        let o = heap.overflow.as_ref().expect("overflow heap");
        let (buf_a, _) = host.shared_buffer(256).expect("buffer a");
        let (buf_b, _) = host.shared_buffer(256).expect("buffer b");
        let raw_a = windows::core::Interface::as_raw(&buf_a) as usize;
        {
            let mut shard = o.shards[0].lock().unwrap();
            shard[0] = Over { key: 11, kept: Some(buf_a) };
            shard[1] = Over { key: 22, kept: Some(buf_b) };
        }
        let mut evicted = std::collections::HashSet::new();
        evicted.insert(raw_a);
        heap.evict(&evicted);
        let shard = o.shards[0].lock().unwrap();
        assert_eq!(shard[0].key, 0);
        assert!(shard[0].kept.is_none());
        assert_eq!(shard[1].key, 22);
        assert!(shard[1].kept.is_some());
    }

    /// A constant view is made in 256-byte steps and at most 64 KiB, the
    /// most the host takes (a larger size removes the device on NVIDIA); a
    /// 0-byte view stays 0.
    #[test]
    fn constant_views_are_at_most_64_kib() {
        assert_eq!(constant_view_size(0), 0);
        assert_eq!(constant_view_size(1), 256);
        assert_eq!(constant_view_size(16), 256);
        assert_eq!(constant_view_size(256), 256);
        assert_eq!(constant_view_size(65_536), 65_536);
        assert_eq!(constant_view_size(1_306_368), 65_536);
        assert_eq!(constant_view_size(65_537), 65_536);
        assert_eq!(constant_view_size(u32::MAX), 65_536);
        assert_eq!(constant_view_size(u32::MAX - 100), 65_536);
    }

    #[test]
    fn a_free_run_across_the_place_the_search_starts_from_is_found() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        // Slots 0..8 and 8..16 of the ring: the search now starts at 16.
        assert_eq!((heap.dynamic(8), heap.dynamic(8)), (Some(4), Some(12)));
        // Everything after them, so that only what is given back is free.
        let rest = heap.dynamic - 16;
        assert_eq!(heap.dynamic(rest), Some(4 + 16));
        // The search starts at 0 again. Slots 0..16 back, then 0..12 taken:
        // it starts at 12, with 12..16 free.
        heap.release(4, 16);
        assert_eq!(heap.dynamic(12), Some(4));
        // 10..12 back: the free run 10..16 lies across 12. Six slots fit
        // only there.
        heap.release(4 + 10, 2);
        assert_eq!(heap.dynamic(6), Some(4 + 10));
        // And nothing is found where nothing is.
        assert_eq!(heap.dynamic(1), None);
    }

    /// A slot whose texture was uploaded ahead shows it when it is
    /// published, and does not go back into the queue of textures picked by
    /// index for that; one the title wrote again is left alone.
    #[test]
    fn a_published_slot_stays_out_of_the_queue() {
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        // 64x64 BC7, one mip, at an address no test and no title uses.
        let view = |at: u64| {
            let mut desc = [0u32; 16];
            (desc[0], desc[1], desc[2], desc[5]) = (98, 4, 5768, 1);
            (desc[10], desc[11]) = (at as u32, (at >> 32) as u32);
            View::Placed(false, [3, 0, 0, 0, 64, 0, 64, 0x1_0001, 98, 1, 0, 0x105, 0, 0], desc)
        };
        let title = [0u32; 8];
        let handle = title.as_ptr() as u64;
        crate::descriptor::record(handle, view(0x7fec_0000_0000));
        let (first, key) = crate::descriptor::view_and_key(handle).unwrap();
        assert!(heap.note_buffer(1, Some(&first), false).is_empty());
        let (index, write) = heap.texture_slots_due(u64::MAX, 8).0[0];
        assert_eq!(index, 1);
        // Its upload was recorded: out of the queue, the slot not made yet.
        heap.texture_slot_done(1, write, None);
        assert!(!heap.made(1));
        assert!(heap.publish(1, handle, key));
        assert!(heap.made(1));
        assert!(heap.texture_slots_due(u64::MAX, 8).0.is_empty());
        // Written again since the upload was recorded.
        crate::descriptor::record(handle, view(0x7feb_0000_0000));
        assert!(!heap.publish(1, handle, key));
        assert_ne!(crate::descriptor::view_and_key(handle).unwrap().1, key);
    }

    /// A null texture UAV keeps the title's dimension (3D for the world's
    /// wind field), and is 2D when the title names none.
    #[test]
    fn null_uavs_keep_the_titles_dimension() {
        let d = null_uav_desc(Some(8));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE3D);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture3D.WSize }, u32::MAX);
        let d = null_uav_desc(Some(5));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE2DARRAY);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture2DArray.ArraySize }, 1);
        let dimension = |d| null_uav_desc(d).ViewDimension;
        assert_eq!(dimension(Some(2)), D3D12_UAV_DIMENSION_TEXTURE1D);
        assert_eq!(dimension(Some(4)), D3D12_UAV_DIMENSION_TEXTURE2D);
        assert_eq!(dimension(None), D3D12_UAV_DIMENSION_TEXTURE2D);
        // Multisampled UAVs need a cap; they stay 2D.
        assert_eq!(dimension(Some(6)), D3D12_UAV_DIMENSION_TEXTURE2D);
    }

    /// What the texture code asks of the spans (levels on demand): where
    /// the next texture starts inside a span, the lowest level a view of a
    /// texture shows, and a texture's slots.
    #[test]
    fn spans_say_the_next_start_and_the_lowest_level_viewed() {
        let mut spans = Spans::default();
        // A 512x512 texture of 0x56000 bytes viewed from levels 3 and 1,
        // and the next one 0x6000 after it (run 188).
        spans.note(1, 0x1_0000, 0x1_0000 + 0x5_6000, true);
        spans.shows(1, 3);
        spans.note(2, 0x1_0000, 0x1_0000 + 0x5_6000, true);
        spans.shows(2, 1);
        spans.note(3, 0x1_6000, 0x1_6000 + 0x5_6000, true);
        assert_eq!(spans.next_start(0x1_0000, 0x1_0000 + 0x5_6000), Some(0x1_6000));
        assert_eq!(spans.next_start(0x1_6000, 0x1_6000 + 0x5_6000), None);
        assert_eq!(spans.first_viewed(0x1_0000), Some(1));
        assert_eq!(spans.first_viewed(0x1_6000), None);
        assert_eq!(spans.slots_at(0x1_0000), vec![1, 2]);
        // The view from level 1 goes: the texture is viewed from level 3.
        spans.forget(2);
        assert_eq!(spans.first_viewed(0x1_0000), Some(3));
    }

    /// Views of one placed texture are told from views of others by the
    /// address and the resource description, not by what they show.
    #[test]
    fn views_of_one_placed_texture_are_told_by_address_and_description() {
        let texture = [3, 0, 0, 0, 512, 0, 512, 0x8_0001, 98, 1, 0, 0x105, 0, 0];
        let view = |resource: [u32; 14], first_mip: u32, address: u32| {
            let mut desc = [0u32; 16];
            (desc[0], desc[1], desc[4], desc[5], desc[10], desc[11]) = (98, 4, first_mip, 8 - first_mip, address, 0xb);
            View::Placed(false, resource, desc)
        };
        let low = view(texture, 3, 0x1000_0000);
        // more mips of the same texture
        assert!(same_placed_texture(&low, &view(texture, 0, 0x1000_0000)));
        // another address, another description, a buffer, an unordered view
        assert!(!same_placed_texture(&low, &view(texture, 3, 0x2000_0000)));
        let mut wider = texture;
        wider[4] = 1024;
        assert!(!same_placed_texture(&low, &view(wider, 3, 0x1000_0000)));
        let mut buffer = texture;
        buffer[0] = 1;
        assert!(!same_placed_texture(&view(buffer, 3, 0x1000_0000), &view(buffer, 3, 0x1000_0000)));
        let View::Placed(_, resource, desc) = low.clone() else { unreachable!() };
        assert!(!same_placed_texture(&low, &View::Placed(true, resource, desc)));
        assert!(!same_placed_texture(&low, &View::Shader(0, None)));
    }

    /// A null 3D UAV made on the real device leaves it alive.
    #[test]
    fn null_3d_uav_is_accepted_by_the_device() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let view = View::Unordered(0, 0, Some([28, 8, 0, 0, 0, 0, 0, 0, 0, 0]));
        null_for(&view, heap.cpu(0));
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// Dynamic slots stay taken until given back, and runs do not wrap.
    #[test]
    fn a_structured_view_made_raw_passes_the_host_check() {
        // 100 elements of 20 bytes from element 8, in a copy that starts 32
        // bytes before the view's memory: bytes 192..2192.
        let (first, count) = raw_window(32, 8, 100, 20).unwrap();
        assert_eq!((first, count), (48, 500));
        assert!(buffer_view_ok(DXGI_FORMAT_R32_TYPELESS, first, count, 0, true, 4096));
        // the format a raw view must name; the structured one's is refused
        assert!(!buffer_view_ok(DXGI_FORMAT_UNKNOWN, first, count, 0, true, 4096));
        // past the buffer, not on 16 bytes, no elements
        assert!(!buffer_view_ok(DXGI_FORMAT_R32_TYPELESS, first, count, 0, true, 2000));
        assert!(raw_window(4, 0, 100, 20).is_none());
        assert!(!buffer_view_ok(DXGI_FORMAT_R32_TYPELESS, 0, 0, 0, true, 4096));
        // a structured view the host takes as it is
        assert!(buffer_view_ok(DXGI_FORMAT_UNKNOWN, 1, 10, 12, false, 4096));
    }

    /// The queue of textures picked by index serves the slots looked at
    /// before first, then the newest write and the oldest in turns; a
    /// slot whose memory is not there is looked at again after
    /// 1, 2, 4 frames up to the limit; and what the pass reports about a
    /// slot written again meanwhile is dropped.
    #[test]
    fn the_texture_queue_serves_the_newest_writes_first() {
        let mut q = TextureQueue::default();
        // Slots 10..18 written in frames 1..9, slot 5 last.
        for i in 0..8u32 {
            q.written(10 + i, u64::from(i) + 1, 0, false);
        }
        q.written(5, 9, 0, false);
        let order: Vec<u32> = q.due(9, 100).into_iter().map(|d| d.0).collect();
        // newest, oldest, newest, oldest, and so on
        assert_eq!(order, [5, 10, 17, 11, 16, 12, 15, 13, 14]);
        assert_eq!(q.due(9, 2).len(), 2);
        assert_eq!(q.census(9), [9, 9, 0, 0, 8]);
        // Slot 17's memory is not mapped: again next frame, then in 2, 4, 4.
        let write = |q: &TextureQueue, index: u32| q.due(u64::MAX, 100).iter().find(|d| d.0 == index).unwrap().1;
        let w17 = write(&q, 17);
        let mut frame = 9;
        for wait in [1, 2, 4, 4] {
            q.later(17, w17, frame, 4);
            assert!(!q.due(frame + wait - 1, 100).iter().any(|d| d.0 == 17));
            frame += wait;
            // Due again, and ahead of every slot never looked at.
            assert_eq!(q.due(frame, 100)[0].0, 17);
        }
        assert_eq!(q.census(9)[2], 1);
        q.done(17, w17);
        assert!(!q.due(u64::MAX, 100).iter().any(|d| d.0 == 17));
        // Uploaded with part of its memory there: kept, with what was there.
        let w16 = write(&q, 16);
        q.partly(16, w16, 12, 4096);
        assert_eq!(q.mapped(16), 4096);
        assert_eq!(q.due(12, 100)[0].0, 16);
        // The title writes slot 16 again while the pass works on it: what
        // the pass then says about the old write changes nothing.
        q.written(16, 13, 0, false);
        q.done(16, w16);
        q.later(16, w16, 13, 4);
        assert_eq!(q.mapped(16), 0);
        assert_eq!(q.due(13, 100)[0].0, 16);
        // A draw that makes the host slot (`seen`) keeps what the queue knows,
        let w15 = write(&q, 15);
        q.partly(15, w15, 20, 8192);
        q.written(15, 14, 0, true);
        assert_eq!(q.mapped(15), 8192);
        // and queues a slot the queue did not hold.
        q.written(99, 14, 0, true);
        assert!(q.due(14, 100).iter().any(|d| d.0 == 99));
        // A write due later (a table's slot) is not handed out before.
        q.written(98, 14, 16, false);
        assert!(!q.due(15, 100).iter().any(|d| d.0 == 98));
        assert!(q.due(16, 100).iter().any(|d| d.0 == 98));
    }

    /// Slots held back for lack of room on the card are not handed out
    /// until their frame, keep their place, are not counted as waiting
    /// for their memory, and all come back at once when the card has room;
    /// a slot written again meanwhile is the title's new write.
    #[test]
    fn slots_held_for_lack_of_room_come_back_when_there_is_room() {
        let mut q = TextureQueue::default();
        for i in 0..4u32 {
            q.written(20 + i, 1, 0, false);
        }
        let due = q.due(1, 100);
        assert_eq!(due.len(), 4);
        // The card is full: slots 20 and 21 are held back until frame 101.
        for &(index, write) in due.iter().filter(|d| d.0 < 22) {
            q.hold(index, write, 101);
        }
        let now: Vec<u32> = q.due(2, 100).into_iter().map(|d| d.0).collect();
        assert!(!now.contains(&20) && !now.contains(&21));
        assert!(now.contains(&22) && now.contains(&23));
        assert_eq!(q.census(2)[2], 0, "held slots do not wait for their memory");
        // Due again at their frame, in the order they had.
        assert_eq!(q.due(101, 100), due);
        // Slot 21 is written again while held: due as written.
        q.written(21, 3, 0, false);
        assert!(q.due(3, 100).iter().any(|d| d.0 == 21));
        // Room again at frame 5: slot 20 is due at once, and only it was
        // still held as it was.
        assert_eq!(q.room(5), 1);
        assert!(q.due(5, 100).iter().any(|d| d.0 == 20));
        assert_eq!(q.room(6), 0);
    }

    /// Textures of which only the smallest levels are held, packed into
    /// one another's address range (the three of run 188: 0x2c000 bytes
    /// each at full size, their starts 0x2000 and 0x4000 apart): writing
    /// any of them, in any order and again, leaves the others.
    #[test]
    fn packed_neighbours_stay() {
        let mut spans = Spans::default();
        assert_eq!(spans.rule, Rule::Start);
        let at = 0xb_86bf_8000u64;
        let (a, b, c) = (at, at + 0x2000, at + 0x6000);
        assert!(spans.note(105_998, a, a + 0x2c000, true).is_empty());
        assert!(spans.note(106_004, b, b + 0x2c000, true).is_empty());
        assert!(spans.note(108_521, c, c + 0x2c000, true).is_empty());
        // The first written again (as a title does when levels arrive),
        // from another slot too: its span reaches over the other two.
        assert!(spans.note(7, a, a + 0x2c000, true).is_empty());
        assert!(spans.note(106_004, b, b + 0x2c000, true).is_empty());
        let mut all = spans.slots_in(0, u64::MAX);
        all.sort_unstable();
        assert_eq!(all, [7, 105_998, 106_004, 108_521]);
        // A draw makes a slot of the second texture: it is one of them.
        assert!(spans.note(9, b, b + 0x2c000, false).is_empty());
        assert_eq!(spans.by_slot.get(&9), Some(&b));
    }

    /// A texture written where another starts replaces it: at the same
    /// address with another length, or inside the block of its smallest
    /// levels. The packed neighbours of the one replaced stay.
    #[test]
    fn a_texture_written_at_the_start_of_another_replaces_it() {
        let mut spans = Spans::default();
        let at = 0x7_0000_0000u64;
        assert!(spans.note(1, at, at + 0x2c000, true).is_empty());
        assert!(spans.note(2, at + 0x2000, at + 0x2e000, true).is_empty());
        // Another length from the same start: slot 1 is stale, slot 2 not.
        assert_eq!(spans.note(3, at, at + 0xb000, true), [1]);
        // A start inside the first 4 KiB of slot 3's texture: stale too.
        assert_eq!(spans.note(4, at + 0x800, at + 0x1800, true), [3]);
        // And the other way: a texture that starts just before one.
        assert_eq!(spans.note(5, at + 0x1800, at + 0x4800, true), [2]);
        let mut left = spans.slots_in(0, u64::MAX);
        left.sort_unstable();
        assert_eq!(left, [4, 5]);
        // Exactly 4 KiB apart: both there.
        assert!(spans.note(6, at + 0x2800, at + 0x3800, true).is_empty());
        // A draw's view that clashes is left out, and nothing is stale.
        assert!(spans.note(8, at + 0x900, at + 0x2900, false).is_empty());
        assert_eq!(spans.by_slot.get(&8), None);
        assert_eq!(spans.by_slot.len(), 3);
    }

    /// A slot written with another texture leaves its old span; the same
    /// view written again changes nothing.
    #[test]
    fn a_slot_rewritten_moves_to_its_new_span() {
        let mut spans = Spans::default();
        assert!(spans.note(1, 0x10000, 0x3c000, true).is_empty());
        assert!(spans.note(2, 0x12000, 0x3e000, true).is_empty());
        assert!(spans.note(1, 0x10000, 0x3c000, true).is_empty());
        assert_eq!(spans.slots_in(0x10000, 0x10001), [1]);
        // Slot 1 now views a texture far away.
        assert!(spans.note(1, 0x90000, 0x91000, true).is_empty());
        assert_eq!(spans.slots_in(0x10000, 0x10001), [] as [u32; 0]);
        assert_eq!(spans.slots_in(0x12000, 0x12001), [2]);
        assert_eq!(spans.slots_in(0x90000, 0x90001), [1]);
        spans.forget(1);
        spans.forget(2);
        assert!(spans.by_start.is_empty() && spans.by_slot.is_empty());
    }

    /// Memory the runtime maps, unmaps or writes names the textures that
    /// are there: among packed neighbours the one whose held levels the
    /// range lies in (it ends where the next one starts), and the ones
    /// that start inside the range.
    #[test]
    fn memory_changed_under_packed_neighbours_names_the_ones_there() {
        let mut spans = Spans::default();
        let at = 0x7_0000_0000u64;
        assert!(spans.note(1, at, at + 0x2c000, true).is_empty());
        assert!(spans.note(2, at + 0x2000, at + 0x2e000, true).is_empty());
        assert!(spans.note(3, at + 0x6000, at + 0x32000, true).is_empty());
        // Inside the first one's held levels.
        assert_eq!(spans.slots_in(at + 0x1000, at + 0x2000), [1]);
        // Inside the second's: not the first, which ends where it starts.
        assert_eq!(spans.slots_in(at + 0x3000, at + 0x4000), [2]);
        // Past the last start: the last one.
        assert_eq!(spans.slots_in(at + 0x20000, at + 0x21000), [3]);
        // A range over all of them.
        assert_eq!(spans.slots_in(at, at + 0x8000), [1, 2, 3]);
        // Past the end of all: none.
        assert_eq!(spans.slots_in(at + 0x32000, at + 0x33000), [] as [u32; 0]);
    }

    /// A write names the textures that hold its bytes: the one that
    /// starts where the write starts, not the neighbour before it, whose
    /// held levels end there.
    #[test]
    fn a_write_at_the_start_of_a_packed_texture_names_it_alone() {
        let mut spans = Spans::default();
        let at = 0x7_0000_0000u64;
        assert!(spans.note(1, at, at + 0x2c000, true).is_empty());
        assert!(spans.note(2, at + 0x2000, at + 0x2e000, true).is_empty());
        assert!(spans.note(3, at + 0x6000, at + 0x32000, true).is_empty());
        // A request that fills the second texture's held levels.
        assert_eq!(spans.slots_in(at + 0x2000, at + 0x6000), [1, 2]);
        assert_eq!(spans.slots_holding(at + 0x2000, at + 0x6000), [2]);
        // From inside the first one's held levels on: both.
        assert_eq!(spans.slots_holding(at + 0x1000, at + 0x3000), [1, 2]);
        // A whole 64 KiB page: every texture that starts in it.
        assert_eq!(spans.slots_holding(at, at + 0x10000), [1, 2, 3]);
        // Past the last start, the last one; past all, or no bytes, none.
        assert_eq!(spans.slots_holding(at + 0x20000, at + 0x21000), [3]);
        assert_eq!(spans.slots_holding(at + 0x32000, at + 0x33000), [] as [u32; 0]);
        assert_eq!(spans.slots_holding(at + 0x2000, at + 0x2000), [] as [u32; 0]);
    }

    /// A write behind the guards' back under a placed texture: a slot
    /// waiting for its first upload is left alone, one uploaded before is
    /// counted, and made due two frames on when that is asked for. (The
    /// ranges are handed over here, not through the list all heaps share.)
    #[test]
    fn a_page_write_makes_an_uploaded_texture_slot_due_when_asked() {
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        let slots = |at: u64| -> Vec<u32> { heap.texture_slots_due(at, 8).0.into_iter().map(|d| d.0).collect() };
        // 64x64 BC7, one mip, at an address no test and no title uses.
        let at = 0x7fed_0000_0000u64;
        let mut desc = [0u32; 16];
        (desc[10], desc[11]) = (at as u32, (at >> 32) as u32);
        let view = View::Placed(false, [3, 0, 0, 0, 64, 0, 64, 0x1_0001, 98, 1, 0, 0x105, 0, 0], desc);
        assert!(heap.note_buffer(1, Some(&view), false).is_empty());
        let written = [(at, at + 0x1000)];
        // Waiting for its first upload: held, not uploaded, nothing done.
        assert_eq!(heap.slots_written_under(&written, 50, true), [1, 0, 0]);
        let (queued, write) = heap.texture_slots_due(u64::MAX, 8).0[0];
        assert_eq!(queued, 1);
        heap.texture_slot_done(1, write, None);
        assert_eq!(slots(u64::MAX), []);
        // Uploaded and out of the queue. Memory elsewhere: nothing.
        assert_eq!(heap.slots_written_under(&[(at + 0x10_0000, at + 0x10_1000)], 60, true), [0, 0, 0]);
        // Under it, counted only: it stays out of the queue.
        assert_eq!(heap.slots_written_under(&written, 60, false), [1, 1, 0]);
        assert_eq!(slots(u64::MAX), []);
        // Under it, asked for: due two frames on, not before.
        assert_eq!(heap.slots_written_under(&written, 60, true), [1, 1, 1]);
        assert_eq!(slots(61), []);
        assert_eq!(slots(62), [1]);
    }

    /// `VICTORY_STALE_TEXTURES=0`: nothing is ever stale, and a view that
    /// overlaps others is left out of the spans, as before.
    #[test]
    fn never_stale_leaves_overlapping_views_out() {
        let mut spans = Spans { rule: Rule::Never, ..Default::default() };
        assert!(spans.note(1, 0x1000, 0x3000, true).is_empty());
        assert!(spans.note(2, 0x2000, 0x4000, true).is_empty());
        assert!(spans.note(3, 0x1000, 0x2000, true).is_empty());
        assert_eq!(spans.slots_in(0, u64::MAX), [1]);
    }

    /// The rule until run 188 (`VICTORY_STALE_TEXTURES=overlap`): a view
    /// written over memory that older views describe otherwise makes
    /// those stale; views of the same span are one texture's.
    #[test]
    fn a_newer_view_of_other_memory_makes_the_older_stale() {
        let mut spans = Spans { rule: Rule::Overlap, ..Default::default() };
        // Three textures side by side, the first in two slots.
        assert!(spans.note(1, 0x1000, 0x3000, true).is_empty());
        assert!(spans.note(2, 0x1000, 0x3000, true).is_empty());
        assert!(spans.note(3, 0x3000, 0x4000, true).is_empty());
        assert!(spans.note(4, 0x8000, 0x9000, true).is_empty());
        // The same slot written with the same span again: nothing changes.
        assert!(spans.note(2, 0x1000, 0x3000, true).is_empty());
        assert_eq!(spans.slots_in(0x2000, 0x3800), [1, 2, 3]);
        assert_eq!(spans.slots_in(0x4000, 0x8000), [] as [u32; 0]);
        // A draw makes a slot whose view overlaps them: left out, nothing stale.
        assert!(spans.note(9, 0x2000, 0x3800, false).is_empty());
        assert_eq!(spans.slots_in(0x2000, 0x3800), [1, 2, 3]);
        // The title writes a texture over the end of the first and all of
        // the second: their three slots are stale.
        let mut stale = spans.note(5, 0x2000, 0x4000, true);
        stale.sort_unstable();
        assert_eq!(stale, [1, 2, 3]);
        assert_eq!(spans.slots_in(0, u64::MAX), [5, 4]);
        // Another start in the same memory: stale in turn. Another length
        // from the same start too.
        assert_eq!(spans.note(6, 0x2800, 0x3000, true), [5]);
        assert_eq!(spans.note(7, 0x2800, 0x2c00, true), [6]);
        // A slot that moves to another texture leaves its span behind.
        assert!(spans.note(7, 0x8000, 0x9000, true).is_empty());
        assert_eq!(spans.slots_in(0x2800, 0x2c00), [] as [u32; 0]);
        assert_eq!(spans.slots_in(0x8000, 0x8001), [4, 7]);
        spans.forget(4);
        spans.forget(7);
        assert!(spans.by_start.is_empty() && spans.by_slot.is_empty());
    }

    /// Memory the runtime maps under a slot's texture makes the slot due
    /// again from the next frame; a stale slot leaves the queue.
    #[test]
    fn remapped_memory_makes_its_texture_slots_due() {
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        let slots = |at: u64| -> Vec<u32> { heap.texture_slots_due(at, 8).0.into_iter().map(|d| d.0).collect() };
        // 64x64 BC7, one mip, at an address no test and no title uses.
        let at = 0x7fee_0000_0000u64;
        let view = |address: u64, width: u32| {
            let mut desc = [0u32; 16];
            (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
            View::Placed(false, [3, 0, 0, 0, width, 0, 64, 0x1_0001, 98, 1, 0, 0x105, 0, 0], desc)
        };
        let len = crate::resource::placed_bytes(&[3, 0, 0, 0, 64, 0, 64, 0x1_0001, 98, 1, 0, 0x105, 0, 0]).unwrap();
        assert!(heap.note_buffer(1, Some(&view(at, 64)), false).is_empty());
        let (queued, write) = heap.texture_slots_due(u64::MAX, 8).0[0];
        assert_eq!(queued, 1);
        heap.texture_slot_done(1, write, None);
        assert_eq!(slots(u64::MAX), []);
        // Memory elsewhere: nothing. Under the texture: due from the next frame.
        remapped((at + 2 * len) as usize, 0x1000);
        remapped(at as usize, 0x1000);
        assert_eq!(heap.texture_slots_remapped(50), 1);
        assert_eq!(slots(50), []);
        assert_eq!(slots(51), [1]);
        // A wider texture written at the same address: slot 1 is stale.
        assert_eq!(heap.note_buffer(2, Some(&view(at, 128)), false), [1]);
        assert_eq!(slots(u64::MAX), [2]);
    }

    /// A slot a draw read through a table leaves the queue; the title's
    /// next write of it waits for its draw, and is due after the grace
    /// frames if none reads it.
    #[test]
    fn slots_read_through_tables_leave_the_texture_queue() {
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        let frame = 100;
        let slots = |at: u64| -> Vec<u32> { heap.texture_slots_due(at, 8).0.into_iter().map(|d| d.0).collect() };
        let texture = View::Placed(false, [3, 0, 0, 0, 4, 0, 4, 0x1_0001, 28, 1, 0, 0x105, 0, 0], [0; 16]);
        assert!(is_texture_view(&texture));
        heap.queue_texture(2, true, false, frame);
        heap.queue_texture(3, true, false, frame);
        assert_eq!(slots(frame), [3, 2]);
        heap.tabled_in(3, true, frame);
        assert_eq!(slots(frame), [2]);
        // Made in place by the draw that read it: still out.
        heap.queue_texture(3, true, true, frame);
        assert_eq!(slots(frame), [2]);
        // Written again in the next frame: due once the grace frames have
        // passed without a draw that reads it.
        heap.queue_texture(3, true, false, frame + 1);
        assert_eq!(slots(frame + TABLED_GRACE), [2]);
        assert_eq!(slots(frame + 1 + TABLED_GRACE), [3, 2]);
        // The draw comes after all: out again.
        heap.tabled_in(3, true, frame + 1);
        assert_eq!(slots(u64::MAX), [2]);
        // A slot of a table no draw read stays queued; its next write waits.
        heap.tabled_in(2, false, frame + 1);
        assert_eq!(slots(frame + 1), [2]);
        heap.queue_texture(2, true, false, frame + 2);
        assert_eq!(slots(frame + 2), []);
        // Long after the last table, a write is due at once.
        heap.queue_texture(2, true, false, frame + 3 + TABLED_FRAMES);
        assert_eq!(slots(frame + 3 + TABLED_FRAMES), [2]);
    }

    #[test]
    fn dynamic_search_can_find_a_run_across_the_cursor() {
        let Some(heap) = HostHeap::new(false, 4) else {
            eprintln!("no host descriptor heap; skipped");
            return;
        };
        let first = heap.dynamic(1).unwrap();
        let left = heap.dynamic(2).unwrap();
        let right = heap.dynamic(2).unwrap();
        let rest = heap.dynamic(heap.dynamic - 5).unwrap();
        assert_eq!((first, left, right, rest), (4, 5, 7, 9));

        // Reallocating the left half puts the cursor back in the middle.
        heap.release(left, 2);
        assert_eq!(heap.dynamic(2), Some(left));
        heap.release(left, 2);
        heap.release(right, 2);
        assert_eq!(heap.dynamic(4), Some(left));
        assert_eq!(heap.dynamic(1), None);

        // A free slot at each physical end does not form a two-slot range.
        heap.release(first, 1);
        let last = first + heap.dynamic - 1;
        heap.release(last, 1);
        assert_eq!(heap.dynamic(2), None);
    }

    #[test]
    fn a_null_cbv_drops_the_slots_previous_resource() {
        let Some(host) = host() else { return };
        let heap = HostHeap::new(false, 4).expect("view heap");
        let (buffer, _) = host.shared_buffer(256).expect("buffer");
        heap.keep(0, Some(buffer));
        let mut slot = [0u64; 4];
        let handle = slot.as_mut_ptr() as u64;
        crate::descriptor::create_cbv(std::ptr::null(), std::ptr::null(), handle);
        assert_eq!(heap.mirror(0, handle, false), Some(View::NullConstant));
        assert!(heap.kept[0].lock().unwrap()[0].is_none());
        // SAFETY: the device is live.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    #[test]
    fn dynamic_slots_are_held_until_released() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        let a = heap.dynamic(heap.dynamic - 100).unwrap();
        let b = heap.dynamic(60).unwrap();
        assert_eq!((a, b), (4, 4 + heap.dynamic - 100));
        // 40 left at the end: 50 do not fit until something is given back.
        assert_eq!(heap.dynamic(50), None);
        heap.release(a, 50);
        assert_eq!(heap.dynamic(50), Some(4));
        assert_eq!(heap.dynamic(40), Some(4 + heap.dynamic - 40));
    }

    /// The title's view heap on NVIDIA's 1,000,000 views with a
    /// heap map: title slots below 131,072 at their own index, 606,782 pool
    /// slots, the map at 737,854, and the fork's dynamic slots (from
    /// 737,855) and guard slot (999,999, the translator's limit) where they
    /// were. A window past the room is cut to it; the pool holds no more
    /// slots than the title has past the window.
    #[test]
    fn the_titles_view_heap_is_laid_out_in_a_million_slots() {
        let l = MapLayout::new(1_245_184, WINDOW, MAP_SLOT);
        assert_eq!((l.window, l.pool, l.slot), (131_072, 606_782, 737_854));
        assert_eq!(l.dynamic_first(), PORTABLE_VIEWS - DYNAMIC - 1, "the fork's first dynamic slot");
        assert_eq!(l.guard(), PORTABLE_VIEWS - 1, "the fork's guard slot");
        assert_eq!(l.size(), PORTABLE_VIEWS);
        let wide = MapLayout::new(1_245_184, 900_000, MAP_SLOT);
        assert_eq!((wide.window, wide.pool), (MAP_SLOT, 0));
        let small = MapLayout::new(20, 16, 24);
        assert_eq!((small.window, small.pool, small.dynamic_first()), (16, 4, 25));
        assert_eq!(small.size(), 25 + DYNAMIC + 1);
    }

    /// A view heap made while the translator has a heap map (as a capped
    /// host's `probe_view_limit` sets it) is laid out as [`MapLayout`] says,
    /// its map's raw view at the map slot; one no larger than the window
    /// keeps the fork's layout.
    #[test]
    fn a_view_heap_with_a_heap_map_keeps_the_dynamic_slots_after_the_map() {
        if host().is_none() {
            return;
        }
        TEST_MAP.set(Some((16, 24)));
        let mapped = HostHeap::new(false, 64);
        let fits = HostHeap::new(false, 8);
        TEST_MAP.set(None);
        let (mapped, fits) = (mapped.unwrap(), fits.unwrap());
        assert_eq!((mapped.count(), mapped.title_count, mapped.dynamic_first), (16, 64, 25));
        assert!(mapped.map.is_some());
        {
            let made = mapped.made.lock().unwrap();
            assert_eq!(made.keys.len(), 16 + 8, "the window's and the pool's");
            assert_eq!(made.map.as_ref().map(SlotMap::len), Some(48));
        }
        assert_eq!(mapped.tabled.len(), 64, "by title slot");
        let total_kept: usize = mapped.kept.iter().map(|p| p.lock().unwrap().len()).sum();
        assert!(total_kept >= (25 + DYNAMIC) as usize);
        assert!(
            mapped.kept[(24 % KEPT_PIECES) as usize].lock().unwrap()[(24 / KEPT_PIECES) as usize]
                .is_some(),
            "the map's raw view holds it"
        );
        assert_eq!(mapped.dynamic(DYNAMIC), Some(25));
        assert_eq!(mapped.dynamic(1), None);
        assert_eq!((fits.count(), fits.dynamic_first), (8, 8));
        assert!(fits.map.is_none() && fits.made.lock().unwrap().map.is_none());
        assert_eq!(fits.dynamic(1), Some(8));
    }

    /// A title view heap of `count` slots and its first descriptor, made
    /// as the title makes them (its host heap is made at first use).
    fn title_heap(count: u32) -> (crate::object::This, u64) {
        #[repr(C)]
        struct Extended(super::super::HeapDesc, super::super::Placement);
        let desc = super::super::HeapDesc {
            kind: 0,
            count,
            flags: 1,
            nodes: 0,
        };
        let none = super::super::Placement { address: 0, size: 0 };
        let extended = Extended(desc, none);
        let mut out = std::ptr::null();
        let made = super::super::create_heap(
            std::ptr::null(),
            &extended.0,
            &super::super::HEAP_IIDS[0],
            &mut out,
        );
        assert_eq!(made, crate::object::S_OK);
        (out, super::super::heap(out).start)
    }

    /// Writes a constant buffer view of `address` into title slot `slot`.
    fn title_cbv(start: u64, slot: u32, address: u64) {
        super::super::create_cbv(
            std::ptr::null(),
            &[address as u32, (address >> 32) as u32, 256, 0],
            start + u64::from(slot) * TITLE_SLOT_BYTES,
        );
    }

    fn release(object: crate::object::This) -> u32 {
        crate::object::slot::<extern "system" fn(crate::object::This) -> u32>(object, 0x10)(object)
    }

    /// The heap map entry of title slot `title`, and its pool slot.
    fn entry(heap: &HostHeap, title: u32) -> (u32, Option<u32>) {
        let window = heap.count();
        let at = heap.map.as_ref().unwrap().at;
        // SAFETY: an entry of the mapped heap map.
        let entry = unsafe { (at as *const u32).add((title - window) as usize).read_volatile() };
        let pool = heap.made.lock().unwrap().map.as_ref().unwrap().host(title);
        (entry, pool)
    }

    /// Title slots past the identity window that the title writes take
    /// pool slots, published in the heap map (entries hold the slot's
    /// bitwise not) once made; slots below it keep their index. A slot the
    /// title writes again keeps its pool slot; one it empties gives it back
    /// (entry zeroed: the guard), quarantined while submissions run.
    #[test]
    fn title_slots_past_the_window_take_pool_slots() {
        if host().is_none() {
            return;
        }
        let (title, start) = title_heap(64);
        title_cbv(start, 3, 0x1_0000);
        title_cbv(start, 40, 0x2_0000);
        TEST_MAP.set(Some((16, 24)));
        let heap = super::super::heap(title).host().expect("a host heap");
        TEST_MAP.set(None);
        assert_eq!(heap.count(), 16);
        let (e40, pooled) = entry(heap, 40);
        let pooled = pooled.expect("a pool slot");
        assert!((16..24).contains(&pooled), "{pooled}");
        assert_eq!(e40, !pooled);
        assert_eq!(entry(heap, 41), (0, None));
        assert_eq!(heap.made.lock().unwrap().host(3), Some(3), "below the window: its own");
        assert_eq!(heap.title_start.load(std::sync::atomic::Ordering::Relaxed), start);

        // Written again: the same pool slot. Another slot: another one.
        title_cbv(start, 40, 0x3_0000);
        assert_eq!(entry(heap, 40), (!pooled, Some(pooled)));
        title_cbv(start, 63, 0x3_0000);
        let other = entry(heap, 63).1.expect("a pool slot");
        assert_ne!(other, pooled);

        // Emptied (as a write of no view finds it): the entry reads the
        // guard and the pool slot waits.
        let handle = start + 40 * TITLE_SLOT_BYTES;
        // SAFETY: a descriptor slot of the title heap.
        unsafe { (handle as *mut u8).write_bytes(0, 32) };
        assert!(heap.mirror(40, handle, false).is_none());
        assert_eq!(entry(heap, 40), (0, None));
        let stats = heap.made.lock().unwrap().map.as_ref().unwrap().stats();
        assert_eq!((stats.mapped, stats.quarantined), (1, 1));
        assert_eq!(release(title), 0);
    }

    /// With the pool used up, the pool slots of title slots that hold no
    /// live view any more (overwritten as bytes: no descriptor write says
    /// so) are given back, and the slot that asked takes one.
    #[test]
    fn a_used_up_pool_gives_back_slots_without_views() {
        if host().is_none() {
            return;
        }
        let (title, start) = title_heap(64);
        TEST_MAP.set(Some((16, 20)));
        let heap = super::super::heap(title).host().expect("a host heap");
        TEST_MAP.set(None);
        for slot in 16..20 {
            title_cbv(start, slot, 0x1_0000 + u64::from(slot) * 0x100);
        }
        let freed = entry(heap, 17).1.expect("a pool slot");
        // SAFETY: a descriptor slot of the title heap.
        unsafe { ((start + 17 * TITLE_SLOT_BYTES) as *mut u8).write_bytes(0, 32) };
        title_cbv(start, 30, 0x9_0000);
        assert_eq!(entry(heap, 17), (0, None), "given back");
        assert!(entry(heap, 16).1.is_some() && entry(heap, 19).1.is_some());
        // Handed out once the submissions running when it was given back
        // have finished (other tests' may be).
        let (running, _) = crate::host::submissions();
        let waited = std::time::Instant::now();
        while entry(heap, 30).1.is_none() && waited.elapsed() < std::time::Duration::from_secs(20) {
            if crate::host::submissions().1 >= running {
                let _ = heap.mirror(30, start + 30 * TITLE_SLOT_BYTES, false);
            } else {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }
        assert_eq!(entry(heap, 30), (!freed, Some(freed)));
        assert_eq!(release(title), 0);
    }

    /// `HostHeap::buffers` is partitioned into `BUFFER_BUCKETS` (16) buckets
    /// by `index % 16`: each bucket returns only its sorted `1 / 16` slice,
    /// and `mark_changed = false` updates buckets without re-queuing `changed`.
    #[test]
    fn buffer_slots_are_partitioned_into_16_sorted_buckets() {
        let Some(heap) = HostHeap::new(false, 256) else {
            return; // no host GPU
        };
        let mut raw_desc = [0u32; 16];
        // Buffer SRV (dimension = 1, raw flag = 1, R32_TYPELESS = 39).
        raw_desc[0] = 39;
        raw_desc[1] = 1;
        raw_desc[6] = 16;
        raw_desc[8] = 1;
        let buf_view = View::Placed(false, [1, 0, 0, 0, 64, 0, 1, 1, 0, 1, 0, 0, 0, 0], raw_desc);
        let inserted = [0u32, 1, 15, 16, 17, 31, 32, 48, 63, 128, 255];
        for &slot in &inserted {
            assert!(heap.note_buffer_in(slot, Some(&buf_view), false, true).is_empty());
        }
        let mut changes = heap.buffer_changes();
        changes.sort_unstable();
        assert_eq!(changes, inserted);
        assert!(heap.buffer_changes().is_empty());

        let mut union = Vec::new();
        for bucket in 0..BUFFER_BUCKETS {
            let slice = heap.buffer_slots_bucket(bucket);
            assert!(slice.windows(2).all(|w| w[0] < w[1]), "bucket {bucket} must be strictly sorted");
            for &slot in &slice {
                assert_eq!((slot as usize) % BUFFER_BUCKETS, bucket);
            }
            union.extend(slice);
        }
        union.sort_unstable();
        assert_eq!(union, inserted);
        assert_eq!(heap.buffer_slots(), inserted);

        // `mark_changed = false` (used by `mirror_for_buffers`) removes slot 16
        // from bucket 0 without re-inserting it into `changed`.
        assert!(heap.note_buffer_in(16, None, false, false).is_empty());
        assert_eq!(heap.buffer_slots_bucket(0), vec![0, 32, 48, 128]);
        assert!(heap.buffer_changes().is_empty());

        // `mark_changed = true` removes slot 32 and queues it in `changed`.
        assert!(heap.note_buffer_in(32, None, false, true).is_empty());
        assert_eq!(heap.buffer_slots_bucket(0), vec![0, 48, 128]);
        assert_eq!(heap.buffer_changes(), vec![32]);
    }

    /// `sync_copied_page` skips reading title memory on all-tabled pages, trims
    /// leading/trailing tabled runs on mixed pages, skips unchanged untabled
    /// slots, detects zeroing of previously mirrored slots, and batches the
    /// tabled count in its return value instead of incrementing `COPIED_TABLED`
    /// per slot.
    #[test]
    fn sync_copied_page_skips_tabled_and_unchanged_slots() {
        let Some(heap) = HostHeap::new(false, 64) else {
            return; // no host GPU
        };
        let frame = 500u64;
        let mut page_bytes = vec![0u8; crate::watch::PAGE];

        // Mark slots 0..16 as tabled in `frame`. Even with an unreadable `start = 0`,
        // an all-tabled page must return `(true, 16, 0, 16)` without touching memory.
        for slot in 0..16u32 {
            heap.tabled_in(slot, true, frame);
        }
        let tabled_before = COPIED_TABLED.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            heap.sync_copied_page(0, 16, 0, TITLE_SLOT_BYTES, frame, &mut page_bytes),
            (true, 16, 0, 16)
        );
        assert_eq!(
            COPIED_TABLED.load(std::sync::atomic::Ordering::Relaxed),
            tabled_before,
            "sync_copied_page must batch tabled count instead of touching COPIED_TABLED per slot"
        );

        // Allocate a real 64-slot title heap buffer: mark leading (0..4), middle (6),
        // and trailing (8..16) slots as tabled; leave slots 4, 5, 7 untabled.
        let mut backing = vec![0u8; 64 * TITLE_SLOT_BYTES as usize];
        let start = backing.as_mut_ptr() as u64;
        let slot_handle = |s: u32| start + u64::from(s) * TITLE_SLOT_BYTES;
        // Write NullConstant views (mirrored immediately with `for_draw = false`) at slots 5 and 7,
        // and also at tabled slot 2 (which must be ignored!).
        super::super::create_cbv(std::ptr::null(), std::ptr::null(), slot_handle(2));
        super::super::create_cbv(std::ptr::null(), std::ptr::null(), slot_handle(5));
        super::super::create_cbv(std::ptr::null(), std::ptr::null(), slot_handle(7));

        let Some(mixed) = HostHeap::new(false, 64) else {
            return;
        };
        for s in (0..4u32).chain(std::iter::once(6)).chain(8..16u32) {
            mixed.tabled_in(s, true, frame);
        }
        // First pass: 16 looked, 2 untabled non-zero slots mirrored (5, 7), 13 tabled skipped.
        assert_eq!(
            mixed.sync_copied_page(0, 16, start, TITLE_SLOT_BYTES, frame, &mut page_bytes),
            (true, 16, 2, 13)
        );
        assert!(!mixed.made(2), "tabled slot 2 must not be mirrored");
        assert!(!mixed.made(4), "empty untabled slot 4 stays unmade");
        assert!(mixed.made(5), "untabled slot 5 mirrored");
        assert!(mixed.made(7), "untabled slot 7 mirrored");

        // Second pass with nothing changed: 0 mirrored, 13 tabled skipped.
        assert_eq!(
            mixed.sync_copied_page(0, 16, start, TITLE_SLOT_BYTES, frame, &mut page_bytes),
            (true, 16, 0, 13)
        );

        // Verify `collect_unmade_slots` matches live key at slot 5 and flags slot 4 / wrong key.
        let (k5, _) = super::super::record_key(slot_handle(5)).expect("key at slot 5");
        let mut mismatched = Vec::new();
        mixed.collect_unmade_slots(&[(5, k5), (4, k5), (5, k5 ^ 1), (7, 0)], &mut mismatched);
        assert_eq!(mismatched, vec![4, 5, 7]);

        // Zero slot 5 in memory: next sync must detect `key = 0 != old_key` and clear slot 5.
        let off5 = 5 * TITLE_SLOT_BYTES as usize;
        backing[off5..off5 + TITLE_SLOT_BYTES as usize].fill(0);
        assert_eq!(
            mixed.sync_copied_page(0, 16, start, TITLE_SLOT_BYTES, frame, &mut page_bytes),
            (true, 16, 1, 13)
        );
        assert!(!mixed.made(5), "zeroed slot 5 must be cleared");
        assert!(mixed.made(7), "unchanged slot 7 stays made");

        // Empty-slot fast-path: if `copied[4]` had a non-zero key (e.g. a deferred
        // buffer view written with `for_draw = false`) while `shows_none(4)` is
        // true, a subsequent zeroing updates `copied[4]` to 0 without mirroring.
        assert!(mixed.shows_none(4));
        mixed.copied_begin()[4].store(0x1234_5678, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            mixed.sync_copied_page(0, 16, start, TITLE_SLOT_BYTES, frame, &mut page_bytes),
            (true, 16, 0, 13)
        );
        assert_eq!(mixed.copied_begin()[4].load(std::sync::atomic::Ordering::Relaxed), 0);
        assert!(!mixed.made(4));
    }

    #[test]
    fn page_all_tabled_caches_and_expires_full_tabled_page() {
        let Some(heap) = HostHeap::new(false, PAGE_TABLED_SLOTS * 2) else {
            return; // no host GPU
        };
        let frame = 1_000u64;
        assert!(!heap.page_all_tabled(0, PAGE_TABLED_SLOTS, frame));
        for slot in 0..PAGE_TABLED_SLOTS {
            heap.tabled_in(slot, true, frame);
        }
        let mut page_bytes = vec![0u8; crate::watch::PAGE];
        assert_eq!(
            heap.sync_copied_page(0, PAGE_TABLED_SLOTS, 0, TITLE_SLOT_BYTES, frame, &mut page_bytes),
            (true, u64::from(PAGE_TABLED_SLOTS), 0, u64::from(PAGE_TABLED_SLOTS))
        );
        assert!(heap.page_all_tabled(0, PAGE_TABLED_SLOTS, frame));
        assert!(heap.page_all_tabled(0, PAGE_TABLED_SLOTS, frame + TABLED_FRAMES));
        assert!(!heap.page_all_tabled(0, PAGE_TABLED_SLOTS, frame + TABLED_FRAMES + 1));
    }

    #[test]
    fn untabled_watch_spans_trims_tails_and_large_gaps_only() {
        const PAGES: usize = 30;
        let slots = (PAGES as u32) * PAGE_TABLED_SLOTS;
        let Some(heap) = HostHeap::new(false, slots) else {
            return; // no host GPU
        };
        let frame = 2_000u64;
        let start = crate::watch::PAGE * 64;
        let looked_end = start + PAGES * crate::watch::PAGE;

        // 1) All untabled -> single full span.
        assert_eq!(
            heap.untabled_watch_spans(start, looked_end, TITLE_SLOT_BYTES, slots, frame),
            vec![(start, looked_end)]
        );

        // 2) Mark pages 0..2, 4..22 (18-page gap), and 25..30 (trailing tail) all-tabled.
        // Page 2..4 (len 2), small gap at page 6? Let's leave page 3 and pages 22..25 untabled,
        // plus test a small 3-page gap by having page 26 untabled instead of 25..30 trailing:
        // Wait: mark pages 0..2 (leading), 3..6? No, mark 0..2 (leading), 3..5?
        // Pages:
        // 0..2   tabled (leading trimmed)
        // 2      untabled
        // 3..6   tabled (3-page gap < 16 -> bridged)
        // 6      untabled
        // 7..24  tabled (17-page gap >= 16 -> split)
        // 24..26 untabled
        // 26..30 tabled (trailing trimmed)
        let mut page_bytes = vec![0u8; crate::watch::PAGE];
        for p in (0..2usize).chain(3..6).chain(7..24).chain(26..30) {
            let s0 = (p as u32) * PAGE_TABLED_SLOTS;
            let s1 = s0 + PAGE_TABLED_SLOTS;
            for s in s0..s1 {
                heap.tabled_in(s, true, frame);
            }
            assert_eq!(
                heap.sync_copied_page(s0, s1, 0, TITLE_SLOT_BYTES, frame, &mut page_bytes),
                (true, u64::from(PAGE_TABLED_SLOTS), 0, u64::from(PAGE_TABLED_SLOTS))
            );
        }
        assert_eq!(
            heap.untabled_watch_spans(start, looked_end, TITLE_SLOT_BYTES, slots, frame),
            vec![
                (start + 2 * crate::watch::PAGE, start + 7 * crate::watch::PAGE),
                (start + 24 * crate::watch::PAGE, start + 26 * crate::watch::PAGE),
            ]
        );

        // 3) All pages tabled -> empty spans (zero GetWriteWatch calls).
        for p in [2usize, 6, 24, 25] {
            let s0 = (p as u32) * PAGE_TABLED_SLOTS;
            let s1 = s0 + PAGE_TABLED_SLOTS;
            for s in s0..s1 {
                heap.tabled_in(s, true, frame);
            }
            heap.sync_copied_page(s0, s1, 0, TITLE_SLOT_BYTES, frame, &mut page_bytes);
        }
        assert!(heap.untabled_watch_spans(start, looked_end, TITLE_SLOT_BYTES, slots, frame).is_empty());
    }

    #[test]
    fn texture_remaps_visit_each_changed_region_once() {
        let mut ranges = Vec::new();
        for (from, to) in [(40, 50), (30, 40), (35, 40), (10, 20), (45, 60), (60, 61), (1, 1)] {
            assert!(!record_remap(&mut ranges, from, to, 3));
        }
        merge_remaps(&mut ranges);
        assert_eq!(ranges, [(10, 20), (30, 61)]);
        assert!(record_remap(&mut ranges, 90, 99, 2));
        assert!(!record_remap(&mut ranges, 120, 160, 2));
        assert_eq!(ranges, [(0, u64::MAX)]);
    }
}

#[cfg(test)]
mod record_print_tests {
    use super::record_print;

    /// Equal records print alike; a record that differs in any one byte,
    /// or in its length, prints otherwise.
    #[test]
    fn a_record_that_changed_prints_otherwise() {
        let record: Vec<u8> = (0..32u8).map(|i| i.wrapping_mul(37).wrapping_add(5)).collect();
        let print = record_print(&record);
        assert_eq!(print, record_print(&record.clone()));
        for at in 0..32 {
            for bit in 0..8 {
                let mut other = record.clone();
                other[at] ^= 1 << bit;
                assert_ne!(record_print(&other), print, "byte {at} bit {bit}");
            }
        }
        assert_ne!(record_print(&record[..16]), print);
        assert_ne!(record_print(&[0u8; 32]), record_print(&[0u8; 16]));
        // A record of zeros is told from "never looked at" by the caller's `| 1`.
        assert_ne!(record_print(&[0u8; 32]) | 1, 0);
    }
}

#[cfg(test)]
mod own_copy_tests {
    use super::own_bytes_if;

    /// A buffer view gets a copy of its own bytes only when it has no host
    /// view on its buffer's copy: off that copy's elements, and not a
    /// structured view that can be a raw view of the same bytes.
    #[test]
    fn a_view_off_its_copys_elements_gets_its_own_bytes() {
        let base = 0xd_9000_0000u64;
        // On an element: the buffer's copy (the two cases of every run's
        // log are the ones below).
        assert_eq!(own_bytes_if(true, 24, base, 3, 10, 12, 0, false), None);
        assert_eq!(own_bytes_if(true, 0, base, 0, 10, 12, 0, false), None);
        // A typed view of 12-byte elements, 5,222,240 bytes into its copy
        // (8 past an element): its own bytes, from its first element.
        assert_eq!(own_bytes_if(true, 5_222_240, base, 3, 10, 12, 0, false), Some((base + 36, 120)));
        // A structured view of 32-byte elements 8 bytes in: not on 16, so
        // no raw view of the copy either.
        assert_eq!(own_bytes_if(true, 8, base, 2, 4, 32, 32, true), Some((base + 64, 128)));
        // The same 16 bytes in: off its elements, but a raw view of the
        // copy's bytes does (as before).
        assert_eq!(own_bytes_if(true, 16, base, 2, 4, 32, 32, true), None);
        // No elements, no size, or the switch off: as before.
        assert_eq!(own_bytes_if(true, 8, base, 2, 0, 32, 32, true), None);
        assert_eq!(own_bytes_if(true, 8, base, 2, 4, 0, 0, false), None);
        assert_eq!(own_bytes_if(false, 8, base, 2, 4, 32, 32, true), None);
    }
}
