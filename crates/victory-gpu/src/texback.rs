//! Textures shaders write that the title's own code then reads on the CPU.
//! On the console the CPU and the GPU share memory, so what a compute pass
//! stores into a texture is in title memory for the CPU to read. Here it stays
//! in the texture's host copy. The title's ocean is one: its wave passes write
//! a 256x256 height map (R32_FLOAT) and a displacement map (R32G32_FLOAT), and
//! its CPU code samples both, which is what boats float on; it read memory
//! nothing had written (run rw1, `VICTORY_READ_WATCH=1`: about 30 places in
//! the title read those two kinds), and boats sank.
//!
//! For textures of the kinds in `VICTORY_TEXTURE_BACK` (comma-separated
//! `<width>x<height>f<format>`; `1` for the four water formats found by
//! the upstream trace), a list in which shaders write one copies it
//! into a read-back buffer of ours when it closes, and once the list has run
//! the texels are tiled into the texture's title memory in the console's
//! layout, as the console's GPU would have left them.
//!
//! shortcut: by kind, not by texture: textures of those kinds the title never
//! reads are copied back too (about a megabyte per such list). The copy shows
//! what the list's last write left; a title read in between sees the frame
//! before, as it would before the console's GPU has run.

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use windows::Win32::Graphics::Direct3D12::*;

use crate::object::This;

const MAX_POOL: usize = 64;
static POOL: Mutex<Vec<(u64, ID3D12Resource)>> = Mutex::new(Vec::new());

thread_local! {
    static TILED: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static PREV: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// Returns the narrowest byte range `[lo, hi)` where `before` and `after` differ,
/// or `None` when identical.
fn changed_span(before: &[u8], after: &[u8]) -> Option<(usize, usize)> {
    debug_assert_eq!(before.len(), after.len());
    let len = before.len().min(after.len());
    if len == 0 {
        return None;
    }
    let words = len / 8;
    // Fast word scan from front and back to find differing 8-byte blocks.
    let mut lo_word = 0usize;
    while lo_word < words {
        let off = lo_word * 8;
        if before[off..off + 8] != after[off..off + 8] {
            break;
        }
        lo_word += 1;
    }
    if lo_word == words {
        let mut lo = words * 8;
        while lo < len && before[lo] == after[lo] {
            lo += 1;
        }
        if lo == len {
            return None;
        }
        let mut hi = len;
        while hi > lo + 1 && before[hi - 1] == after[hi - 1] {
            hi -= 1;
        }
        return Some((lo, hi));
    }
    let mut lo = lo_word * 8;
    while lo < len && before[lo] == after[lo] {
        lo += 1;
    }
    let mut hi = len;
    let tail_start = words * 8;
    while hi > tail_start && before[hi - 1] == after[hi - 1] {
        hi -= 1;
    }
    if hi == tail_start {
        let mut hi_word = words;
        while hi_word > lo_word + 1 {
            let off = (hi_word - 1) * 8;
            if before[off..off + 8] != after[off..off + 8] {
                break;
            }
            hi_word -= 1;
        }
        hi = hi_word * 8;
        while hi > lo + 1 && before[hi - 1] == after[hi - 1] {
            hi -= 1;
        }
    }
    Some((lo, hi))
}

/// Picks the smallest pooled readback buffer with `cap >= needed` and at most
/// ~2x `needed` (allowing +64 KiB slack for small textures).
fn pick_pool_index(caps: &[u64], needed: u64) -> Option<usize> {
    let max_cap = needed.saturating_mul(2).max(needed.saturating_add(65536));
    caps.iter()
        .enumerate()
        .filter(|&(_, &cap)| cap >= needed && cap <= max_cap)
        .min_by_key(|&(_, &cap)| cap)
        .map(|(idx, _)| idx)
}

fn acquire_buffer(host: &crate::host::Host, needed: u64) -> Option<(u64, ID3D12Resource)> {
    let needed = needed.max(1);
    {
        let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
        let caps: Vec<u64> = pool.iter().map(|(c, _)| *c).collect();
        if let Some(idx) = pick_pool_index(&caps, needed) {
            return Some(pool.swap_remove(idx));
        }
    }
    host.read_back_buffer(needed).map(|buf| (needed, buf))
}

fn release_buffer(cap: u64, buf: ID3D12Resource) {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    if pool.len() < MAX_POOL {
        pool.push((cap, buf));
    }
}

/// The kinds copied back: (width, height, format).
fn kinds() -> &'static [(u64, u32, u32)] {
    static KINDS: OnceLock<Vec<(u64, u32, u32)>> = OnceLock::new();
    KINDS.get_or_init(|| {
        const DEFAULT: &str = "256x256f41,256x256f16,64x64f16,128x128f36";
        let list = match std::env::var("VICTORY_TEXTURE_BACK").as_deref() {
            Ok("1") => DEFAULT.to_owned(),
            Ok(spec) => spec.to_owned(),
            Err(_) => String::new(),
        };
        let kinds: Vec<(u64, u32, u32)> = list
            .split(',')
            .filter_map(|k| {
                let (size, format) = k.trim().split_once('f')?;
                let (w, h) = size.split_once('x')?;
                Some((w.parse().ok()?, h.parse().ok()?, format.parse().ok()?))
            })
            .collect();
        if !kinds.is_empty() {
            tracing::info!(
                ?kinds,
                "textures of these kinds that shaders write are copied back into title memory (VICTORY_TEXTURE_BACK)"
            );
        }
        kinds
    })
}

/// Whether title texture `resource` is of a kind copied back.
pub(crate) fn wanted(resource: This) -> bool {
    let kinds = kinds();
    !kinds.is_empty()
        && crate::resource::texture_kind(resource).is_some_and(|k| kinds.contains(&k))
}

/// An outstanding write to a specific allocation. Mapping and resource
/// changes invalidate only the writes that could touch the replaced memory.
struct Ticket {
    owner: usize,
    at: usize,
    end: usize,
    valid: AtomicBool,
}

static TICKETS: Mutex<Vec<Weak<Ticket>>> = Mutex::new(Vec::new());
static TRACKING: AtomicBool = AtomicBool::new(false);

fn track(owner: usize, at: usize, end: usize) -> Arc<Ticket> {
    let ticket = Arc::new(Ticket {
        owner,
        at,
        end,
        valid: AtomicBool::new(true),
    });
    let mut tickets = TICKETS.lock().unwrap_or_else(|e| e.into_inner());
    tickets.retain(|weak| weak.strong_count() != 0);
    tickets.push(Arc::downgrade(&ticket));
    TRACKING.store(true, Ordering::Relaxed);
    ticket
}

fn invalidate(matches: impl Fn(&Ticket) -> bool) {
    if !TRACKING.load(Ordering::Relaxed) {
        return;
    }
    TICKETS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .retain(|weak| {
            let Some(ticket) = weak.upgrade() else {
                return false;
            };
            if matches(&ticket) {
                ticket.valid.store(false, Ordering::Release);
            }
            true
        });
}

pub(crate) fn released(owner: usize) {
    invalidate(|ticket| ticket.owner == owner);
}

pub(crate) fn replaced(at: usize) {
    invalidate(|ticket| ticket.at == at);
}

pub(crate) fn remapped(at: usize, len: usize) {
    let end = at.saturating_add(len);
    invalidate(|ticket| at < ticket.end && ticket.at < end);
}

pub(crate) struct Target {
    pub(crate) copy: Arc<crate::resource::HostTexture>,
    layout: victory_formats::texture::Layout,
    ticket: Arc<Ticket>,
}

impl Target {
    pub(crate) fn new(resource: This) -> Option<Self> {
        // Keep the host copy independently of the title's resource lifetime.
        let copy = crate::resource::host_copy(resource)?;
        let _mapping = crate::watch::title_memory_reading();
        let (layout, at, size) = crate::resource::title_texture(resource)?;
        let end = at.checked_add(size).filter(|&end| at != 0 && end > at)?;
        let ticket = track(resource as usize, at, end);
        Some(Self {
            copy,
            layout,
            ticket,
        })
    }

    pub(crate) fn owner(&self) -> usize {
        self.ticket.owner
    }
}

/// The readback resources and the allocation the pixels belong to.
pub(crate) struct Back {
    buffer: Option<ID3D12Resource>,
    capacity: u64,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    mips: u32,
    layout: victory_formats::texture::Layout,
    ticket: Arc<Ticket>,
}

impl Drop for Back {
    fn drop(&mut self) {
        if let Some(buf) = self.buffer.take() {
            release_buffer(self.capacity, buf);
        }
    }
}

impl Back {
    pub(crate) fn range(&self) -> (usize, usize) {
        (self.ticket.at, self.ticket.end)
    }
}

/// Records a host copy from a texture already in COPY_SOURCE.
pub(crate) fn record(list: &ID3D12GraphicsCommandList, target: Target) -> Option<Back> {
    if !target.ticket.valid.load(Ordering::Acquire) {
        return None;
    }
    let host = crate::host::host()?;
    let texture = &target.copy.texture;
    // SAFETY: a live texture kept through the command list's state tracking.
    let desc = unsafe { texture.GetDesc() };
    let mips = u32::from(desc.MipLevels.max(1));
    let subresources = mips * u32::from(desc.DepthOrArraySize.max(1));
    let n = subresources as usize;
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n];
    let mut rows = vec![0u32; n];
    let mut total = 0u64;
    // SAFETY: out arrays of `n` entries.
    unsafe {
        host.device.GetCopyableFootprints(
            &desc,
            0,
            subresources,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        )
    };
    // The last footprint's advertised extent can omit row-end padding.
    // The tiler reads a pitched slice, so the mapped buffer must include it.
    for (footprint, &rows) in footprints.iter().zip(&rows) {
        let format = footprint.Footprint;
        let pitched = u64::from(format.RowPitch)
            .checked_mul(u64::from(rows))?
            .checked_mul(u64::from(format.Depth))?;
        total = total.max(footprint.Offset.checked_add(pitched)?);
    }
    let (capacity, buffer) = acquire_buffer(host, total)?;
    for (i, footprint) in footprints.iter().enumerate() {
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(&buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *footprint,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: i as u32,
            },
        };
        // SAFETY: an open list; the texture is in the copy-source state.
        unsafe { list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    Some(Back {
        buffer: Some(buffer),
        capacity,
        footprints,
        rows,
        mips,
        layout: target.layout,
        ticket: target.ticket,
    })
}

/// Once the list has run, tile its texels back into the same title allocation.
pub(crate) fn land(back: &Back) {
    static COPIED: AtomicU64 = AtomicU64::new(0);
    static SKIPPED: AtomicU64 = AtomicU64::new(0);
    static STALE: AtomicU64 = AtomicU64::new(0);
    let Some(buffer) = back.buffer.as_ref() else {
        return;
    };
    let _mapping = crate::watch::title_memory_reading();
    let _ticket_guard = TICKETS.lock().unwrap_or_else(|e| e.into_inner());
    if !back.ticket.valid.load(Ordering::Acquire) {
        let skipped = STALE.fetch_add(1, Ordering::Relaxed) + 1;
        if skipped.is_power_of_two() {
            tracing::info!(skipped, "skipping texture readbacks whose title allocation changed");
        }
        return;
    }
    let (at, end) = back.range();
    let size = end - at;
    TILED.with_borrow_mut(|tiled| {
        tiled.resize(size, 0);
        // Keep title bytes the subresources do not cover.
        if !crate::host::read_into(at as *const u8, tiled) {
            return;
        }
        PREV.with_borrow_mut(|prev| {
            prev.clear();
            prev.extend_from_slice(tiled);
            let mut data = std::ptr::null_mut();
            // SAFETY: our read-back buffer, written by the list that has run.
            unsafe {
                if buffer.Map(0, None, Some(&mut data)).is_ok() && !data.is_null() {
                    for (i, (footprint, &rows)) in back.footprints.iter().zip(&back.rows).enumerate() {
                        let (mip, slice) = (i as u32 % back.mips, i as u32 / back.mips);
                        let f = footprint.Footprint;
                        let pitch = f.RowPitch as usize;
                        let len = pitch * rows as usize * f.Depth as usize;
                        // SAFETY: the subresource's part of the mapped buffer.
                        let linear = std::slice::from_raw_parts(
                            data.cast::<u8>().add(footprint.Offset as usize),
                            len,
                        );
                        if let Err(e) =
                            back.layout
                                .tile(tiled, mip, slice, linear, pitch, pitch * rows as usize)
                        {
                            static ONCE: std::sync::Once = std::sync::Once::new();
                            ONCE.call_once(|| {
                                tracing::warn!(?e, mip, slice, "a texture copied back could not be tiled")
                            });
                            buffer.Unmap(0, None);
                            return;
                        }
                    }
                    buffer.Unmap(0, None);
                } else {
                    return;
                }
            }
            let Some((lo, hi)) = changed_span(prev, tiled) else {
                SKIPPED.fetch_add(1, Ordering::Relaxed);
                return;
            };
            crate::host::copy_mapped((at + lo) as *mut u8, tiled[lo..hi].as_ptr(), hi - lo, true);
            crate::watch::wrote(at + lo, hi - lo);
            let copied = COPIED.fetch_add(1, Ordering::Relaxed) + 1;
            if copied == 1 || copied.is_multiple_of(128) {
                tracing::info!(copied, bytes = hi - lo, total = size, "texture readback landed");
            }
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pick_pool_index_chooses_smallest_fitting_within_cap_limit() {
        assert_eq!(pick_pool_index(&[1024, 65536, 131072, 262144], 70000), Some(2));
        assert_eq!(pick_pool_index(&[1024, 16384], 32768), None);
        assert_eq!(pick_pool_index(&[512 * 1024], 65536), None);
        assert_eq!(pick_pool_index(&[65536], 16384), Some(0));
    }

    #[test]
    fn only_readbacks_for_changed_allocations_are_invalidated() {
        let first = track(91, 0x1000, 0x2000);
        let next = track(92, 0x2000, 0x3000);
        let far = track(93, 0x4000, 0x5000);

        remapped(0x0f00, 0x200);
        assert!(!first.valid.load(Ordering::Acquire));
        assert!(next.valid.load(Ordering::Acquire));
        replaced(0x2000);
        assert!(!next.valid.load(Ordering::Acquire));
        assert!(far.valid.load(Ordering::Acquire));

        released(93);
        assert!(!far.valid.load(Ordering::Acquire));
        assert!(track(94, 0x2000, 0x3000).valid.load(Ordering::Acquire));
    }

    #[test]
    fn changed_span_finds_narrowest_dirty_bounds_or_none() {
        assert_eq!(changed_span(&[], &[]), None);
        assert_eq!(changed_span(&[1, 2, 3], &[1, 2, 3]), None);
        let zeros = vec![0u8; 64];
        assert_eq!(changed_span(&zeros, &zeros), None);

        let mut mod_tail = vec![0u8; 19];
        let orig_tail = mod_tail.clone();
        mod_tail[17] = 9;
        assert_eq!(changed_span(&orig_tail, &mod_tail), Some((17, 18)));

        let mut mod_start = vec![0u8; 32];
        let orig_start = mod_start.clone();
        mod_start[0] = 1;
        assert_eq!(changed_span(&orig_start, &mod_start), Some((0, 1)));

        let mut mod_mid = vec![0u8; 40];
        let orig_mid = mod_mid.clone();
        mod_mid[9] = 4;
        mod_mid[27] = 7;
        assert_eq!(changed_span(&orig_mid, &mod_mid), Some((9, 28)));

        let ones = vec![1u8; 37];
        let twos = vec![2u8; 37];
        assert_eq!(changed_span(&ones, &twos), Some((0, 37)));
    }
}
