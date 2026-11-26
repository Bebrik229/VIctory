//! Compiled host pipelines kept on disk, and the title pipelines worth
//! starting early (decision 0007, "Pipeline cache").
//!
//! 1. Every host pipeline goes into the host's pipeline library, named by
//!    a hash of all that its host description holds (the translated
//!    shaders' code, which changes when a translation does, our root
//!    signature, the graphics state). The library is saved under the cache
//!    folder in [`SHARDS`] files, one per hash of the name, so a save writes
//!    only the files that gained pipelines (the host serialises a whole
//!    library and nothing less); the host refuses a library of another
//!    adapter or driver, and an entry whose description differs, so neither
//!    is ever used.
//! 2. The title pipelines a run drew or dispatched with are listed by a key
//!    known when the title creates them (their shader containers and
//!    graphics state); the next run hands those to the workers at
//!    creation, during loading.
//! 3. The pre-build (`victory translate --cache --pipelines`) lists the
//!    title pipelines it has put in the library, by the same key, so a
//!    second pre-build skips them without translating or loading anything.
//!    The list holds only for the library bytes it was saved with and for
//!    what names the pipelines (the translator, this module and `host`).
//!
//! Library calls overlap: the host documents the library as safe to use
//! from several threads ("The pipeline library is thread-safe to use, and
//! will internally synchronize as necessary", `CreatePipelineLibrary`,
//! Thread Safety), with one exception kept here: two loads of the same
//! pipeline are made one after the other ([`same_name`]). Loads take no
//! state lock at all ([`Shared`]), since a load adds nothing to the
//! library: a save holds the shard's state lock only while taking a
//! snapshot, and loads go on without waiting.
//! `VICTORY_LIBRARY_LOCK=1`: one call at a time, as before.
//!
//! The library's files are mapped, not read into memory ([`Blob`]): the host
//! keeps pointing into the bytes they were made from for as long as they
//! live, and read into a buffer they were 1,352 MB of the process's own
//! memory for the whole run (run 60), all of it in physical memory. Mapped,
//! the pages are the file's: only those the host reads are in memory, and
//! the system can drop them without writing them anywhere. A save writes
//! the new shard through a mapping of the staging file in the same way
//! (a buffer of the shard's size was made, filled and written out at each
//! save). `VICTORY_LIBRARY_MAP=0`: read and written through buffers, as
//! before.
//!
//! The mapped file's handle stays open for as long as the mapping lives
//! ([`holds`]). A save renames the new library over the file this run has
//! mapped, and with the handle closed Windows then gives up the old file's
//! contents: every page of the mapping that is not in memory at that
//! moment, or leaves memory later (the trim below, or Windows taking pages
//! back), reads as zeros from then on. The host's library reads such pages
//! as stored pipelines; the driver, handed a pipeline of zeros, removes the
//! device with DXGI_ERROR_DRIVER_INTERNAL_ERROR (runs 72, 73 and 82, each
//! within 5 ms of the first library call after a save and a trim; and the
//! library run 77 saved from such pages was refused by run 78). Measured
//! with a 64 MB file written past the system's cache, mapped and replaced:
//! 16,352 of 16,384 pages read zeros with the handle closed, none with it
//! open (the test below). `VICTORY_LIBRARY_HANDLE=0`: closed, as before.

use std::collections::HashSet;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{HSTRING, Interface, PCWSTR};

static DIR: OnceLock<PathBuf> = OnceLock::new();

/// Keeps the pipeline library and the early list in `dir` from now on.
pub fn use_pipeline_cache(dir: PathBuf) {
    sweep_staging(&dir, std::process::id(), STAGING_KEPT);
    crate::shader_meta::init(&dir);
    let _ = DIR.set(dir);
}

/// The folder holding the pipeline library, if one was set.
pub(super) fn folder() -> Option<&'static PathBuf> {
    DIR.get()
}

/// How long another process's staging file is left alone: the folder is
/// shared, and a running instance may be writing it.
const STAGING_KEPT: std::time::Duration = std::time::Duration::from_secs(10 * 60);

const STAGING_PREFIXES: &[&str] = &[
    "host-pipelines.bin.tmp-",
    "host-pipelines.tmp-",
    "early-pipelines.txt.tmp-",
    "early-pipelines.tmp-",
    "prebuilt-pipelines.txt.tmp-",
    "shader-meta.bin.tmp-",
];

const LEGACY_STAGING_FILES: &[&str] = &["host-pipelines.tmp", "early-pipelines.tmp"];

/// Removes the staging files of cut writes (`write_atomically` / `save`)
/// from `dir`: those of process `pid` now (no write of this run has started
/// yet), other processes' (and legacy `.tmp` files without a pid) once
/// untouched for `kept`.
fn sweep_staging(dir: &std::path::Path, pid: u32, kept: std::time::Duration) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name_str) = name.to_str() else { continue };
        let writer = staging_pid(name_str);
        let legacy_tmp = writer.is_none() && LEGACY_STAGING_FILES.contains(&name_str);
        if writer.is_none() && !legacy_tmp {
            continue;
        }
        let Ok(meta) = entry.metadata() else { continue };
        if !meta.is_file() {
            continue;
        }
        let age = meta.modified().ok().and_then(|t| t.elapsed().ok());
        if writer == Some(pid) || age.is_some_and(|age| age >= kept) {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// The `<pid>-<n>` suffix of a sharded staging file (`host-pipelines.XX.bin.tmp-...`,
/// `host-pipelines.XX.tmp-...`, or `prebuilt-pipelines.XX.txt.tmp-...`).
fn shard_staging_rest(name: &str) -> Option<&str> {
    if let Some(after) = name.strip_prefix("host-pipelines.") {
        let (tag, rest) = after.split_once('.')?;
        if tag.len() == 2 && tag.bytes().all(|b| b.is_ascii_hexdigit()) {
            return rest.strip_prefix("bin.tmp-").or_else(|| rest.strip_prefix("tmp-"));
        }
    }
    if let Some(after) = name.strip_prefix("prebuilt-pipelines.") {
        let (tag, rest) = after.split_once('.')?;
        if tag.len() == 2 && tag.bytes().all(|b| b.is_ascii_hexdigit()) {
            return rest.strip_prefix("txt.tmp-");
        }
    }
    None
}

/// The process whose staging file `name` is (`<file>.tmp-<pid>-<n>` or
/// `<stem>.tmp-<pid>-<n>` of a cache file), if it is one.
fn staging_pid(name: &str) -> Option<u32> {
    let rest = STAGING_PREFIXES
        .iter()
        .find_map(|prefix| name.strip_prefix(prefix))
        .or_else(|| shard_staging_rest(name))?;
    let (pid, n) = rest.split_once('-')?;
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    if !digits(pid) || !digits(n) {
        return None;
    }
    pid.parse().ok()
}

/// Unique staging path beside `path` (`<path>.tmp-<pid>-<n>`), so concurrent
/// processes or threads never collide on a shared `.tmp` filename.
fn staging_path(path: &std::path::Path) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let pid = std::process::id();
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let mut name = path.as_os_str().to_os_string();
    name.push(format!(".tmp-{pid}-{n}"));
    PathBuf::from(name)
}

/// Host pipelines created, loaded from the library, and title pipelines
/// started at creation, since the start.
pub(crate) static MADE: AtomicU64 = AtomicU64::new(0);
pub(crate) static LOADED: AtomicU64 = AtomicU64::new(0);
pub(crate) static EARLY: AtomicU64 = AtomicU64::new(0);
/// For the memory lines: the bytes of the library's file, kept in memory
/// while the library lives, and the bytes the last save wrote (a buffer of
/// that size is made, filled and written every time pipelines were added,
/// at most every 20 s).
pub(crate) static BLOB: AtomicU64 = AtomicU64::new(0);
pub(crate) static SAVED: AtomicU64 = AtomicU64::new(0);

/// Whether the library's file is mapped ([`Blob`]) and saved through a
/// mapping. On by default; `VICTORY_LIBRARY_MAP=0` uses buffers, as before.
fn mapped() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LIBRARY_MAP").is_none_or(|v| v != "0"))
}

/// Whether the mapped library file's handle is kept open while it is
/// mapped (see the module's notes: without it a save in the same run turns
/// the mapping's pages into zeros as they leave memory). On by default;
/// `VICTORY_LIBRARY_HANDLE=0` closes it after mapping, as before.
fn holds() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LIBRARY_HANDLE").is_none_or(|v| v != "0"))
}

/// Whether the mapped library file's pages are prefetched in the background
/// when opened (`VICTORY_LIBRARY_PREFETCH=0` disables it).
fn prefetch() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LIBRARY_PREFETCH").is_none_or(|v| v != "0"))
}

fn prefetch_mapped(at: usize, len: usize) {
    #[repr(C)]
    struct Win32MemoryRangeEntry {
        virtual_address: *mut std::ffi::c_void,
        number_of_bytes: usize,
    }
    type PrefetchVirtualMemoryFn = unsafe extern "system" fn(
        windows::Win32::Foundation::HANDLE,
        usize,
        *const Win32MemoryRangeEntry,
        u32,
    ) -> i32;

    let work = move || {
        let started = std::time::Instant::now();
        // SAFETY: `kernel32.dll` is loaded in the process; `PrefetchVirtualMemory`
        // (resolved dynamically for older Windows / Wine compatibility) accepts the
        // current process handle and the mapped view range kept alive by `Library`.
        let ok = unsafe {
            let Ok(kernel32) = windows::Win32::System::LibraryLoader::GetModuleHandleW(
                windows::core::w!("kernel32.dll"),
            ) else {
                return;
            };
            let Some(proc) = windows::Win32::System::LibraryLoader::GetProcAddress(
                kernel32,
                windows::core::PCSTR(c"PrefetchVirtualMemory".as_ptr().cast()),
            ) else {
                return;
            };
            let prefetch_vm: PrefetchVirtualMemoryFn = std::mem::transmute(proc);
            let entry = Win32MemoryRangeEntry {
                virtual_address: at as *mut std::ffi::c_void,
                number_of_bytes: len,
            };
            prefetch_vm(
                windows::Win32::System::Threading::GetCurrentProcess(),
                1,
                &entry,
                0,
            ) != 0
        };
        let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
        tracing::debug!(len, ok, elapsed_ms, "prefetched mapped host pipeline library");
        crate::host::trail(|| {
            format!("pipeline library: prefetched {len} bytes in {elapsed_ms:.1} ms (ok={ok})")
        });
    };
    let _ = std::thread::Builder::new()
        .name("victory-library-prefetch".into())
        .spawn(work);
}

/// A whole file mapped into memory: read-only (the library's file) or
/// writable (the file a save fills). Unmapped when dropped.
struct Mapped {
    at: windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS,
    len: usize,
    /// The file, for a mapping made by [`Mapped::open`] ([`holds`]): open
    /// until after the view is unmapped, so that the file a rename took
    /// the name from keeps its contents for the view.
    _file: Option<std::fs::File>,
}

impl Mapped {
    /// Maps `file` (`len` bytes, not zero) for reading, or for reading and
    /// writing.
    fn map(file: &std::fs::File, len: usize, write: bool) -> Option<Mapped> {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_READ, FILE_MAP_WRITE, MapViewOfFile, PAGE_READONLY, PAGE_READWRITE,
        };
        if len == 0 {
            return None;
        }
        // SAFETY: a file we hold open; the view keeps the file's pages
        // after the mapping object and the file are closed.
        let at = unsafe {
            let mapping = CreateFileMappingW(
                file.as_raw_handle(),
                std::ptr::null(),
                if write { PAGE_READWRITE } else { PAGE_READONLY },
                0,
                0,
                std::ptr::null(),
            );
            if mapping.is_null() {
                return None;
            }
            let at = MapViewOfFile(mapping, if write { FILE_MAP_WRITE } else { FILE_MAP_READ }, 0, 0, 0);
            CloseHandle(mapping);
            at
        };
        (!at.Value.is_null()).then_some(Mapped { at, len, _file: None })
    }

    /// The file at `path`, mapped for reading; `None` if it is not there,
    /// is empty or cannot be mapped.
    fn open(path: &std::path::Path) -> Option<Mapped> {
        Mapped::open_holding(path, holds())
    }

    /// As [`Mapped::open`]; `hold` says whether the file stays open with
    /// the mapping.
    fn open_holding(path: &std::path::Path, hold: bool) -> Option<Mapped> {
        // Opened as the standard library opens a file to read: others may
        // read, write, rename and delete it meanwhile, so a save still
        // renames the new library over this one.
        let file = std::fs::File::open(path).ok()?;
        let len = usize::try_from(file.metadata().ok()?.len()).ok()?;
        let mut mapped = Mapped::map(&file, len, false)?;
        if hold {
            mapped._file = Some(file);
        }
        Some(mapped)
    }

    /// A new file of `len` bytes at `path` (replacing what is there),
    /// mapped for writing.
    fn create(path: &std::path::Path, len: usize) -> Option<Mapped> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .ok()?;
        file.set_len(len as u64).ok()?;
        Mapped::map(&file, len, true)
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: the view holds `len` readable bytes until dropped.
        unsafe { std::slice::from_raw_parts(self.at.Value.cast(), self.len) }
    }

    /// Only for a view made by [`Mapped::create`].
    fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: a writable view of `len` bytes, ours alone.
        unsafe { std::slice::from_raw_parts_mut(self.at.Value.cast(), self.len) }
    }
}

impl Drop for Mapped {
    fn drop(&mut self) {
        // SAFETY: our view.
        unsafe { windows_sys::Win32::System::Memory::UnmapViewOfFile(self.at) };
    }
}

/// The bytes a library is made from: the host reads them for as long as
/// the library lives, so they are kept beside it.
enum Blob {
    /// No file (an empty library).
    None,
    /// The file read into memory (`VICTORY_LIBRARY_MAP=0`, tests, or a file
    /// that could not be mapped).
    Held(Vec<u8>),
    /// The file itself, mapped.
    Mapped(Mapped),
}

impl Blob {
    /// The library file in `dir`: a library a save left beside it
    /// ([`NEXT_FILE`]) takes its place first.
    fn of(dir: &std::path::Path) -> Blob {
        Blob::of_files(&dir.join(LIBRARY_FILE), &dir.join(NEXT_FILE))
    }

    /// The library file at `path`: a library a save left beside it (`next`)
    /// takes its place first.
    fn of_files(path: &std::path::Path, next: &std::path::Path) -> Blob {
        if next.exists() && std::fs::rename(next, path).is_err() {
            tracing::warn!("the host pipeline library saved last could not take the old one's place");
        }
        if mapped()
            && let Some(file) = Mapped::open(path)
        {
            return Blob::Mapped(file);
        }
        match std::fs::read(path) {
            Ok(bytes) if !bytes.is_empty() => Blob::Held(bytes),
            _ => Blob::None,
        }
    }

    fn bytes(&self) -> &[u8] {
        match self {
            Blob::None => &[],
            Blob::Held(bytes) => bytes,
            Blob::Mapped(file) => file.bytes(),
        }
    }

    fn is_empty(&self) -> bool {
        self.bytes().is_empty()
    }
}

/// The mapped library files' views `(address, bytes)` and their total byte
/// count (`0`: none mapped), when a pipeline was last loaded from the
/// library (`crate::host::now_ms`), and whether the views' pages were let go
/// since ([`trim`]).
static VIEWS: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
static VIEW_BYTES: AtomicU64 = AtomicU64::new(0);
static LOADED_AT: AtomicU64 = AtomicU64::new(0);
static TRIMMED: AtomicBool = AtomicBool::new(false);
/// The bytes of the views let go so far, for the memory line.
pub(crate) static TRIMMED_BYTES: AtomicU64 = AtomicU64::new(0);
/// When the views' pages were last let go (`crate::host::now_ms`), and how
/// often and when a save of this run last renamed a new shard over a
/// mapped file: for [`removal_note`].
static TRIMMED_AT: AtomicU64 = AtomicU64::new(0);
static REPLACED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// The library calls running now: thread, what (load, make, store,
/// serialise), the pipeline, and since when (`crate::host::now_us`).
static CALLS: Mutex<Vec<(u32, &'static str, String, u64)>> = Mutex::new(Vec::new());

thread_local! {
    /// The pipeline this thread is making (its shaders), for [`CALLS`].
    static LABEL: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) };
}

/// Names the pipeline this thread makes next.
pub(crate) fn label(text: String) {
    LABEL.with(|l| *l.borrow_mut() = text);
}

/// A library call of this thread, listed in [`CALLS`] until dropped.
struct Call(u32, u64);

fn call(what: &'static str, name: &str) -> Call {
    let (thread, since) = (crate::host::thread_id(), crate::host::now_us());
    let label = LABEL.with(|l| l.borrow().clone());
    CALLS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push((thread, what, format!("{name} {label}"), since));
    Call(thread, since)
}

impl Drop for Call {
    fn drop(&mut self) {
        CALLS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|c| (c.0, c.3) != (self.0, self.1));
    }
}

/// For the report when the host device is removed: the library calls
/// running at that moment, whether this run replaced a mapped file and
/// let its pages go, and how many pages of the mappings read as zeros now.
/// A library holds no page of zeros (the 259 MB one of run 83 has no run of
/// 2,048 zero bytes): each such page is one whose contents were lost (see
/// the module's notes), and a pipeline stored there is what the driver was
/// handed.
pub(crate) fn removal_note() -> String {
    let now = crate::host::now_us();
    let mut text = String::from("pipeline library: ");
    {
        let calls = CALLS.lock().unwrap_or_else(|e| e.into_inner());
        if calls.is_empty() {
            text.push_str("no call running");
        }
        for (i, (thread, what, name, since)) in calls.iter().enumerate() {
            let _ = write!(
                text,
                "{}{what} of {name} running for {:.1} ms on thread {thread}",
                if i == 0 { "" } else { ", " },
                now.saturating_sub(*since) as f64 / 1000.0
            );
        }
    }
    let views = VIEWS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    if views.is_empty() {
        text.push_str("; its files are not mapped");
        return text;
    }
    let len: usize = views.iter().map(|&(_, len)| len).sum();
    let ago = |ms: u64| crate::host::now_ms().saturating_sub(ms) as f64 / 1000.0;
    let _ = write!(
        text,
        "; {} mapped files ({len} bytes), the handle {}",
        views.len(),
        if holds() { "kept open" } else { "closed (VICTORY_LIBRARY_HANDLE=0)" }
    );
    match REPLACED[0].load(Ordering::Relaxed) {
        0 => text.push_str("; this run has not replaced a file"),
        n => {
            let _ = write!(
                text,
                "; this run replaced a file under the mapping {n} times, last {:.1} s ago",
                ago(REPLACED[1].load(Ordering::Relaxed))
            );
        }
    }
    if TRIMMED_BYTES.load(Ordering::Relaxed) > 0 {
        let _ = write!(text, "; its pages were last let go {:.1} s ago", ago(TRIMMED_AT.load(Ordering::Relaxed)));
    }
    // Within two seconds: pages not in memory come from the disk.
    let started = std::time::Instant::now();
    let (mut zero, mut looked, mut first) = (0u64, 0u64, None);
    let mut page_index = 0u64;
    'views: for (at, view_len) in views {
        // SAFETY: each view lives as long as its library, which is never dropped.
        let bytes = unsafe { std::slice::from_raw_parts(at as *const u8, view_len) };
        for page in bytes.chunks(4096) {
            if page_index % 4096 == 0 && started.elapsed() > std::time::Duration::from_secs(2) {
                break 'views;
            }
            looked += 1;
            if page.iter().all(|&b| b == 0) {
                zero += 1;
                if first.is_none() {
                    first = Some(page_index);
                }
            }
            page_index += 1;
        }
    }
    let _ = write!(text, "; {zero} of the {looked} pages looked at read as zeros now");
    if let Some(first) = first {
        let _ = write!(text, " (the first is page {first}): the library's contents were lost there");
    }
    text
}

/// Lets the mapped library files' pages go from the process's memory once
/// no pipeline has been loaded for ten seconds. The host reads nearly the
/// whole file while the early pipelines load (run 65: all 1,490 MB of it
/// were in the process's memory a minute in, and 1,300 MB still at the
/// end, on a PC with 2 GB free); after that it reads the file again only
/// when a new pipeline loads. Pages let go stay in Windows' file cache
/// while nothing else wants the memory, and come back on the next read.
/// Called once a second by the sweep; does its work on a thread of its own
/// (hundreds of thousands of pages), once after each burst of loads.
/// `VICTORY_LIBRARY_TRIM=0`: the pages stay until Windows takes them.
pub(crate) fn trim() {
    static ON: OnceLock<bool> = OnceLock::new();
    if VIEW_BYTES.load(Ordering::Relaxed) == 0
        || TRIMMED.load(Ordering::Relaxed)
        || !*ON.get_or_init(|| std::env::var_os("VICTORY_LIBRARY_TRIM").is_none_or(|v| v != "0"))
    {
        return;
    }
    let quiet_ms = crate::host::ram_idle_ms(crate::host::available_physical_ram());
    if crate::host::now_ms() < LOADED_AT.load(Ordering::Relaxed) + quiet_ms
        || TRIMMED.swap(true, Ordering::Relaxed)
    {
        return;
    }
    let views = VIEWS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    if views.is_empty() {
        TRIMMED.store(false, Ordering::Relaxed);
        return;
    }
    let work = move || {
        let mut total = 0u64;
        for (at, len) in views {
            // SAFETY: each view lives as long as its library, which is never
            // dropped. Unlocking pages that are not locked fails, as expected,
            // and takes them out of the working set: the documented way to let
            // pages of a mapped file go.
            unsafe { windows_sys::Win32::System::Memory::VirtualUnlock(at as *const _, len) };
            total += len as u64;
        }
        TRIMMED_BYTES.fetch_add(total, Ordering::Relaxed);
        TRIMMED_AT.store(crate::host::now_ms(), Ordering::Relaxed);
        crate::host::trail(|| format!("pipeline library: the mapped files' pages let go from memory ({total} bytes)"));
    };
    if std::thread::Builder::new().name("victory-library-trim".into()).spawn(work).is_err() {
        TRIMMED.store(false, Ordering::Relaxed);
    }
}

struct Library {
    library: ID3D12PipelineLibrary1,
    /// The blob the library was made from: it must outlive the library.
    _blob: Blob,
    /// Pipelines stored since the last save.
    added: u64,
    /// Title pipelines the pre-build put in the library or found there, by
    /// key, and whether that list changed since the last save.
    prebuilt: HashSet<String>,
    prebuilt_changed: bool,
}

impl Library {
    /// Whether it gained pipelines or pre-build keys since the last save.
    fn dirty(&self) -> bool {
        self.added > 0 || self.prebuilt_changed
    }
}

// SAFETY: D3D12 objects are free-threaded, and the library synchronises
// its own calls (see the module's notes); what it does not, two loads of
// one name, is serialised by `same_name`.
unsafe impl Send for Library {}

/// The library for loads, beside the one stores and saves lock.
struct Shared {
    /// The same library as `state`'s, loaded from without its lock.
    loads: ID3D12PipelineLibrary1,
    state: Mutex<Library>,
}

// SAFETY: as for `Library`.
unsafe impl Send for Shared {}
// SAFETY: the library is free-threaded but for loads of one name, which
// `same_name` keeps apart; the rest is behind `state`'s lock.
unsafe impl Sync for Shared {}

/// Held by a load of the pipeline named `name`: loads of the same name
/// wait for each other (the host's one exception to the library's thread
/// safety: such a load may change that pipeline's entry). Names share 64
/// locks, so two different names seldom wait.
fn same_name(name: &str) -> MutexGuard<'static, ()> {
    static LOADING: [Mutex<()>; 64] = [const { Mutex::new(()) }; 64];
    let at = name.bytes().fold(0usize, |h, b| h.wrapping_mul(31).wrapping_add(usize::from(b)));
    LOADING[at % LOADING.len()].lock().unwrap_or_else(|e| e.into_inner())
}

/// With `VICTORY_LIBRARY_LOCK=1`: held by every library call, as before.
fn one_call() -> Option<MutexGuard<'static, ()>> {
    static ON: OnceLock<bool> = OnceLock::new();
    static CALLING: Mutex<()> = Mutex::new(());
    ON.get_or_init(|| std::env::var_os("VICTORY_LIBRARY_LOCK").is_some_and(|v| v == "1"))
        .then(|| CALLING.lock().unwrap_or_else(|e| e.into_inner()))
}

const LIBRARY_FILE: &str = "host-pipelines.bin";
/// A library the driver refused, kept as it was ([`open_at`]).
const REJECTED_FILE: &str = "host-pipelines.rejected.bin";
/// A saved library that could not replace [`LIBRARY_FILE`] while this run
/// had it mapped (a system that refuses to replace a mapped file): the next
/// start renames it over the library first ([`Blob::of`]).
const NEXT_FILE: &str = "host-pipelines.next";
const EARLY_FILE: &str = "early-pipelines.txt";
const PREBUILT_FILE: &str = "prebuilt-pipelines.txt";
/// Title pipelines listed for an early start, at most.
const EARLY_MAX: usize = 8192;

/// How many files the library is kept in. The host only serialises a whole
/// library, so a run that made a handful of pipelines wrote all of it (926 MB
/// here, every 20 s, and closer to 1.7 GB on NVIDIA); with the names spread
/// over shards, a save writes the shards that gained pipelines.
const SHARDS: usize = 16;

/// The file shard `shard` is kept in.
fn shard_file(shard: usize) -> String {
    format!("host-pipelines.{shard:02}.bin")
}

/// A rejected shard, kept as it was.
fn shard_rejected_file(shard: usize) -> String {
    format!("host-pipelines.{shard:02}.rejected.bin")
}

/// A saved shard that could not replace [`shard_file`] while mapped.
fn shard_next_file(shard: usize) -> String {
    format!("host-pipelines.{shard:02}.next")
}

/// The pre-build's list for shard `shard`.
fn prebuilt_file(shard: usize) -> String {
    format!("prebuilt-pipelines.{shard:02}.txt")
}

/// The shard a pipeline named `name` belongs to. The name is a hash, so this
/// is spread and stable across runs: a pipeline always lands in the same
/// file, which is what lets a save write only the files that changed.
fn shard_of(name: &str) -> usize {
    let mut h = 0u32;
    for b in name.bytes() {
        h = h.wrapping_mul(31).wrapping_add(u32::from(b));
    }
    h as usize % SHARDS
}

/// `blob` as a library with its pre-build list, if there is one.
fn shard(
    device: &ID3D12Device1,
    blob: Blob,
    list: Option<Vec<u8>>,
    rejected_rename: Option<(&std::path::Path, &std::path::Path)>,
) -> Option<Shared> {
    let tag = (!blob.is_empty() && list.is_some()).then(|| tag(blob.bytes()));
    let mut library = open_at(device, blob, rejected_rename)?;
    // The pre-build's list, if it was saved with these bytes.
    if let (Some(list), Some(tag)) = (list, tag)
        && !library._blob.is_empty()
    {
        library.prebuilt = listed(&list, &tag);
    }
    Some(Shared { loads: library.library.clone(), state: Mutex::new(library) })
}

fn libraries() -> Option<&'static [Shared; SHARDS]> {
    static LIBRARIES: OnceLock<Option<[Shared; SHARDS]>> = OnceLock::new();
    LIBRARIES
        .get_or_init(|| {
            let device = crate::host::host()?.device.cast::<ID3D12Device1>().ok()?;
            let started = std::time::Instant::now();
            let mut shards: Vec<Shared> = Vec::with_capacity(SHARDS);
            let mut files = 0usize;
            let mut bytes = 0usize;
            let mut mapped_files = 0usize;
            for at in 0..SHARDS {
                let (blob, list, rejected_rename) = match DIR.get() {
                    Some(d) => {
                        let path = d.join(shard_file(at));
                        let next = d.join(shard_next_file(at));
                        let rejected = d.join(shard_rejected_file(at));
                        let blob = Blob::of_files(&path, &next);
                        let list = std::fs::read(d.join(prebuilt_file(at))).ok();
                        (blob, list, Some((path, rejected)))
                    }
                    None => (Blob::None, None, None),
                };
                if !blob.is_empty() {
                    files += 1;
                    bytes += blob.bytes().len();
                    if matches!(blob, Blob::Mapped(_)) {
                        mapped_files += 1;
                    }
                }
                let rename_refs = rejected_rename.as_ref().map(|(p, r)| (p.as_path(), r.as_path()));
                let shared = shard(&device, blob, list, rename_refs)?;
                shards.push(shared);
            }
            if files == 0 {
                tracing::info!("no host pipeline library yet; starting an empty one");
            } else {
                tracing::info!(
                    files,
                    mapped_files,
                    bytes,
                    ms = started.elapsed().as_millis() as u64,
                    "host pipeline library loaded"
                );
            }
            shards.try_into().ok()
        })
        .as_ref()
}

/// The one-file library of earlier runs, read (never written) so a cache made
/// before the shards is not lost: a pipeline found there is stored into its
/// shard, so a warm run fills them and the file can then be deleted.
fn legacy() -> Option<&'static Shared> {
    static LEGACY: OnceLock<Option<Shared>> = OnceLock::new();
    LEGACY
        .get_or_init(|| {
            let device = crate::host::host()?.device.cast::<ID3D12Device1>().ok()?;
            let dir = DIR.get()?;
            let path = dir.join(LIBRARY_FILE);
            let next = dir.join(NEXT_FILE);
            let blob = Blob::of_files(&path, &next);
            if blob.is_empty() {
                return None;
            }
            let bytes = blob.bytes().len();
            let list = std::fs::read(dir.join(PREBUILT_FILE)).ok();
            let rejected = dir.join(REJECTED_FILE);
            let shared = shard(&device, blob, list, Some((&path, &rejected)))?;
            if shared.state.lock().unwrap_or_else(|e| e.into_inner())._blob.is_empty() {
                return None;
            }
            tracing::info!(
                bytes,
                file = LIBRARY_FILE,
                "the one-file host pipeline library is still read: what it holds is stored into the shards as it is used"
            );
            Some(shared)
        })
        .as_ref()
}

/// The library in `blob`, or an empty one if the host rejects the blob (an
/// adapter or driver change, damage).
#[cfg_attr(not(test), allow(dead_code))]
fn open(device: &ID3D12Device1, blob: Blob) -> Option<Library> {
    open_at(device, blob, None)
}

fn open_at(
    device: &ID3D12Device1,
    blob: Blob,
    rejected_rename: Option<(&std::path::Path, &std::path::Path)>,
) -> Option<Library> {
    let make = |bytes: &[u8]| -> windows::core::Result<ID3D12PipelineLibrary1> {
        // SAFETY: the bytes outlive the library (kept beside it).
        unsafe {
            device.CreatePipelineLibrary::<ID3D12PipelineLibrary>(bytes)?.cast()
        }
    };
    let mut rejected = false;
    if !blob.is_empty() {
        match make(blob.bytes()) {
            Ok(library) => {
                let bytes = blob.bytes().len();
                BLOB.fetch_add(bytes as u64, Ordering::Relaxed);
                if let Blob::Mapped(file) = &blob {
                    VIEWS
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .push((file.at.Value as usize, file.len));
                    VIEW_BYTES.fetch_add(file.len as u64, Ordering::Relaxed);
                    LOADED_AT.store(crate::host::now_ms(), Ordering::Relaxed);
                    if prefetch() {
                        prefetch_mapped(file.at.Value as usize, file.len);
                    }
                }
                return Some(Library {
                    library,
                    _blob: blob,
                    added: 0,
                    prebuilt: HashSet::new(),
                    prebuilt_changed: false,
                });
            }
            Err(e) => {
                rejected = true;
                let kept_as = rejected_rename
                    .and_then(|(_, r)| r.file_name())
                    .and_then(|n| n.to_str())
                    .unwrap_or(REJECTED_FILE);
                tracing::warn!(error = %e, "host pipeline library rejected; starting an empty one (the rejected file is kept as {kept_as})");
            }
        }
    }
    // The rejected bytes are let go (a mapped file is unmapped) before the
    // empty library is made: nothing reads them any more.
    drop(blob);
    // And the file is kept under another name, not written over by this
    // run's first save: a refusal may be this device's alone.
    if rejected
        && let Some((from, to)) = rejected_rename
        && let Err(e) = std::fs::rename(from, to)
    {
        tracing::warn!(error = %e, "the rejected host pipeline library could not be kept under its other name");
    }
    let library = make(&[]).ok()?;
    Some(Library {
        library,
        _blob: Blob::None,
        added: 0,
        prebuilt: HashSet::new(),
        prebuilt_changed: false,
    })
}

/// What the pre-build's list holds for: the library's bytes, and what
/// names its pipelines (the translation, as the shader cache keys it; the
/// host descriptions).
fn tag(blob: &[u8]) -> String {
    let limit = victory_shader::emit::loop_limit().to_le_bytes();
    let translator = victory_shader::emit::translation_fingerprint();
    let mut h = blake3::Hasher::new();
    for part in [
        blob,
        &translator[..],
        include_bytes!("../shader_cache.rs"),
        &limit,
        include_bytes!("host.rs"),
        include_bytes!("library.rs"),
    ] {
        h.update(&(part.len() as u64).to_le_bytes());
        h.update(part);
    }
    h.finalize().to_hex().to_string()
}

/// The pre-build's list as saved: the tag it holds for, then its keys.
fn list(tag: &str, keys: &HashSet<String>) -> String {
    let keys = keys.iter().map(String::as_str);
    std::iter::once(tag).chain(keys).collect::<Vec<_>>().join("\n")
}

/// The keys of a saved list, if it holds for `tag` (else none).
fn listed(list: &[u8], tag: &str) -> HashSet<String> {
    let list = String::from_utf8_lossy(list);
    let mut lines = list.lines();
    match lines.next() == Some(tag) {
        true => lines.map(str::to_owned).collect(),
        false => HashSet::new(),
    }
}

/// Whether the pre-build has the title pipeline `key` in the library.
#[allow(dead_code)]
pub(crate) fn prebuilt(key: &str) -> bool {
    fn holds(l: &'static Shared, key: &str) -> bool {
        l.state.lock().unwrap_or_else(|e| e.into_inner()).prebuilt.contains(key)
    }
    libraries().is_some_and(|ls| holds(&ls[shard_of(key)], key)) || legacy().is_some_and(|l| holds(l, key))
}

/// The pre-build has the title pipeline `key` in the library now.
#[allow(dead_code)]
pub(crate) fn note_prebuilt(key: String) {
    if let Some(libraries) = libraries() {
        let mut shard = libraries[shard_of(&key)].state.lock().unwrap_or_else(|e| e.into_inner());
        shard.prebuilt_changed |= shard.prebuilt.insert(key);
    }
}

/// A pipeline's name in the library: the hash of what its host
/// description holds.
pub(crate) fn name(kind: &str, root: &[u8], code: &[&[u8]], state: &str) -> String {
    name_with(kind, root, code, |out| out.write_str(state))
}

/// A pipeline's name in the library, streaming its state description into the
/// hasher with no intermediate `String` allocation while keeping the exact
/// length-prefixed hash layout of [`name`].
pub(crate) fn name_with(
    kind: &str,
    root: &[u8],
    code: &[&[u8]],
    write_state: impl Fn(&mut dyn std::fmt::Write) -> std::fmt::Result,
) -> String {
    struct Len(usize);
    impl std::fmt::Write for Len {
        fn write_str(&mut self, s: &str) -> std::fmt::Result {
            self.0 += s.len();
            Ok(())
        }
    }

    let mut h = blake3::Hasher::new();
    for part in [kind.as_bytes(), root]
        .into_iter()
        .chain(code.iter().copied())
    {
        h.update(&(part.len() as u64).to_le_bytes());
        h.update(part);
    }
    let mut len = Len(0);
    let _ = write_state(&mut len);
    h.update(&(len.0 as u64).to_le_bytes());
    let _ = write_state(&mut super::Blake3Writer(&mut h));
    h.finalize().to_hex()[..32].to_owned()
}

/// The pipeline named `name`: loaded from the library by `load` if it is
/// there with the same description, else made by `create` and stored.
pub(crate) fn pipeline(
    name: &str,
    load: impl Fn(&ID3D12PipelineLibrary1, PCWSTR) -> windows::core::Result<ID3D12PipelineState>,
    create: impl FnOnce() -> windows::core::Result<ID3D12PipelineState>,
) -> windows::core::Result<ID3D12PipelineState> {
    let wide = HSTRING::from(name);
    let Some(libraries) = libraries() else {
        MADE.fetch_add(1, Ordering::Relaxed);
        return create();
    };
    let library = &libraries[shard_of(name)];
    let began = std::time::Instant::now();
    // Outside `library.state`'s lock: loads take up to seconds, and a save
    // holds the state lock while taking its snapshot. The library's blob lives
    // as long as it does, which is never dropped.
    let (loaded, from_legacy) = {
        let _all = one_call();
        let _one = same_name(name);
        let _call = call("load", name);
        match load(&library.loads, PCWSTR(wide.as_ptr())) {
            Ok(pso) => (Ok(pso), false),
            Err(e) => match legacy().and_then(|old| load(&old.loads, PCWSTR(wide.as_ptr())).ok()) {
                Some(pso) => (Ok(pso), true),
                None => (Err(e), false),
            },
        }
    };
    // The load read the library's files (see `trim`).
    LOADED_AT.store(crate::host::now_ms(), Ordering::Relaxed);
    TRIMMED.store(false, Ordering::Relaxed);
    let refused = match loaded {
        Ok(pso) => {
            LOADED.fetch_add(1, Ordering::Relaxed);
            if from_legacy {
                let _all = one_call();
                let _call = call("store", name);
                let mut l = library.state.lock().unwrap_or_else(|e| e.into_inner());
                // SAFETY: a live pipeline; a name already stored is refused, which
                // leaves the library as it was.
                if unsafe { l.library.StorePipeline(PCWSTR(wide.as_ptr()), &pso) }.is_ok() {
                    l.added += 1;
                }
            } else {
                crate::host::trail(|| {
                    let label = LABEL.with(|l| l.borrow().clone());
                    format!("pipeline library: loaded {label} in {:.1} ms", began.elapsed().as_secs_f64() * 1e3)
                });
            }
            return Ok(pso);
        }
        Err(e) => e,
    };
    // Compiled outside any lock: compiles take up to seconds.
    let pso = {
        let _call = call("make", name);
        create()?
    };
    MADE.fetch_add(1, Ordering::Relaxed);
    crate::host::trail(|| {
        let label = LABEL.with(|l| l.borrow().clone());
        format!(
            "pipeline library: made {label} in {:.1} ms (the library did not load it: {refused})",
            began.elapsed().as_secs_f64() * 1e3
        )
    });
    let _all = one_call();
    let _call = call("store", name);
    let mut l = library.state.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: a live pipeline; a name already stored (a description that
    // differs) is refused, which leaves the library as it was.
    if unsafe { l.library.StorePipeline(PCWSTR(wide.as_ptr()), &pso) }.is_ok() {
        l.added += 1;
    }
    Ok(pso)
}

/// Writes the shards that gained pipelines since the last save, and the
/// early list if it changed, when `SAVE_EVERY` has passed (or `now`).
pub(crate) fn save(now: bool) {
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    const SAVE_EVERY: std::time::Duration = std::time::Duration::from_secs(20);
    let (Some(dir), Some(libraries)) = (DIR.get(), libraries()) else { return };
    // Serialize saves through publication and acknowledgement. A periodic
    // save can skip a busy writer. A shutdown save waits and takes a new snapshot.
    let mut last = if now {
        LAST.lock().unwrap_or_else(|e| e.into_inner())
    } else {
        match LAST.try_lock() {
            Ok(last) => last,
            Err(std::sync::TryLockError::WouldBlock) => return,
            Err(std::sync::TryLockError::Poisoned(last)) => last.into_inner(),
        }
    };
    if !now && last.is_some_and(|t| t.elapsed() < SAVE_EVERY) {
        return;
    }
    *last = Some(std::time::Instant::now());
    if let Err(e) = std::fs::create_dir_all(dir) {
        tracing::warn!(error = %e, "host pipeline library not saved");
        return;
    }
    crate::shader_meta::save(dir);
    let early_changed = USED_CHANGED.swap(false, Ordering::Relaxed);
    let mut files = 0usize;
    let mut bytes = 0usize;
    let mut any_in_place = false;
    let mut any_deferred = false;
    for (at, shared) in libraries.iter().enumerate() {
        if let Some((size, in_place)) = save_shard_mapped(dir, at, shared) {
            files += 1;
            bytes += size;
            any_in_place |= in_place;
            any_deferred |= !in_place;
        }
    }
    if files > 0 {
        SAVED.store(bytes as u64, Ordering::Relaxed);
        crate::host::trail(|| {
            let where_saved = match (any_in_place, any_deferred) {
                (true, false) => "renamed over the shard files",
                (false, true) => "left beside the shard files for the next start",
                _ => "updated across the shard files",
            };
            format!("pipeline library: saved {files} shards ({bytes} bytes), {where_saved}")
        });
    }
    let used = {
        let before = early_before();
        let now = USED_NOW.lock().unwrap_or_else(|e| e.into_inner());
        match now.as_ref() {
            Some(now) => early_list(before, now),
            None => early_list(before, &KeySet::default()),
        }
    };
    let mut early_saved = false;
    if early_changed || now {
        match write_atomically(&dir.join(EARLY_FILE), used.join("\n").as_bytes()) {
            Ok(()) => early_saved = true,
            Err(e) => {
                if early_changed {
                    USED_CHANGED.store(true, Ordering::Relaxed);
                }
                tracing::warn!(error = %e, "the early list not saved");
            }
        }
    }
    if files == 0 && !early_saved {
        return;
    }
    tracing::info!(
        files,
        bytes,
        early = used.len(),
        made = MADE.load(Ordering::Relaxed),
        loaded = LOADED.load(Ordering::Relaxed),
        started_early = EARLY.load(Ordering::Relaxed),
        "host pipeline library saved"
    );
}

#[cfg_attr(not(test), allow(dead_code))]
fn save_shard(
    dir: &std::path::Path,
    at: usize,
    shared: &Shared,
    write: impl Fn(&std::path::Path, &[u8]) -> std::io::Result<()>,
) -> Option<usize> {
    save_shard_with(dir, at, shared, None, |path, bytes| write(path, bytes).map(|()| true))
        .map(|(size, _)| size)
}

fn save_shard_mapped(dir: &std::path::Path, at: usize, shared: &Shared) -> Option<(usize, bool)> {
    save_shard_with(dir, at, shared, Some((dir, at)), |path, bytes| {
        if path.file_name().and_then(|n| n.to_str()) == Some(shard_file(at).as_str()) {
            let tmp = staging_path(path);
            let res = std::fs::write(&tmp, bytes).and_then(|()| replace_shard(dir, at, &tmp));
            if res.is_err() {
                let _ = std::fs::remove_file(&tmp);
            }
            res
        } else {
            write_atomically(path, bytes).map(|()| true)
        }
    })
}

fn save_shard_with(
    dir: &std::path::Path,
    at: usize,
    shared: &Shared,
    stage_to: Option<(&std::path::Path, usize)>,
    publish: impl Fn(&std::path::Path, &[u8]) -> std::io::Result<bool>,
) -> Option<(usize, bool)> {
    // Take a snapshot with the state locked, then release it for file I/O.
    let ShardSnapshot { blob, prebuilt, added, prebuilt_count } = serialise(shared, stage_to)?;
    let (blob_written, size) = match blob {
        ShardBlob::Staged { tmp, len } => {
            let res = replace_shard(dir, at, &tmp);
            if res.is_err() {
                let _ = std::fs::remove_file(&tmp);
            }
            (res, len)
        }
        ShardBlob::Bytes(bytes) => {
            let len = bytes.len();
            (publish(&dir.join(shard_file(at)), &bytes), len)
        }
    };
    let (in_place, written) = match blob_written {
        Ok(in_place) => match &prebuilt {
            Some(list) => (in_place, publish(&dir.join(prebuilt_file(at)), list.as_bytes()).map(|_| ())),
            None => (in_place, Ok(())),
        },
        Err(e) => (false, Err(e)),
    };
    if let Err(e) = written {
        // Left dirty: the next save writes it.
        tracing::warn!(error = %e, shard = at, "a host pipeline library shard not saved");
        return None;
    }
    if in_place && VIEW_BYTES.load(Ordering::Relaxed) > 0 {
        REPLACED[1].store(crate::host::now_ms(), Ordering::Relaxed);
        REPLACED[0].fetch_add(1, Ordering::Relaxed);
    }
    // Acknowledge the snapshot only after both files have been published.
    let mut l = shared.state.lock().unwrap_or_else(|e| e.into_inner());
    l.added -= added;
    l.prebuilt_changed = l.prebuilt.len() != prebuilt_count;
    Some((size, in_place))
}

enum ShardBlob {
    /// Already written into a staging file through a mapping, ready to rename.
    Staged { tmp: PathBuf, len: usize },
    /// Serialised into a buffer.
    Bytes(Vec<u8>),
}

struct ShardSnapshot {
    blob: ShardBlob,
    prebuilt: Option<String>,
    added: u64,
    prebuilt_count: usize,
}

/// `shared`'s library as bytes with the pre-build's list for them: `None` if
/// the shard is clean or the host cannot serialise it now (it stays dirty
/// for the next save).
fn serialise(shared: &Shared, stage_to: Option<(&std::path::Path, usize)>) -> Option<ShardSnapshot> {
    let _all = one_call();
    let l = shared.state.lock().unwrap_or_else(|e| e.into_inner());
    if !l.dirty() {
        return None;
    }
    // SAFETY: a plain size query on a live library.
    let size = unsafe { l.library.GetSerializedSize() };
    let _call = call("serialise", "a library shard");
    if let Some((dir, at)) = stage_to
        && mapped()
    {
        let tmp = staging_path(&dir.join(shard_file(at)));
        if let Some(mut file) = Mapped::create(&tmp, size) {
            // SAFETY: a buffer of the size the library asks for.
            if let Err(e) = unsafe { l.library.Serialize(file.bytes_mut()) } {
                drop(file);
                let _ = std::fs::remove_file(&tmp);
                tracing::warn!(error = %e, shard = at, "a host pipeline library shard not serialised");
                return None;
            }
            let prebuilt = (!l.prebuilt.is_empty()).then(|| list(&tag(file.bytes()), &l.prebuilt));
            drop(file);
            return Some(ShardSnapshot {
                blob: ShardBlob::Staged { tmp, len: size },
                prebuilt,
                added: l.added,
                prebuilt_count: l.prebuilt.len(),
            });
        }
    }
    let Some(mut bytes) = zeroed(size) else {
        tracing::warn!(bytes = size, "no memory to serialise a library shard; saved later");
        return None;
    };
    // SAFETY: a buffer of the size the library asks for.
    if let Err(e) = unsafe { l.library.Serialize(&mut bytes) } {
        tracing::warn!(error = %e, "a host pipeline library shard not serialised");
        return None;
    }
    let prebuilt = (!l.prebuilt.is_empty()).then(|| list(&tag(&bytes), &l.prebuilt));
    Some(ShardSnapshot {
        blob: ShardBlob::Bytes(bytes),
        prebuilt,
        added: l.added,
        prebuilt_count: l.prebuilt.len(),
    })
}

/// `n` zero bytes, or `None` when the host has no memory for them. The
/// library is about 1.7 GB on NVIDIA (each pipeline's machine code), and an
/// infallible allocation of that size aborted the whole process mid-run
/// ("memory allocation of 2650101862 bytes failed", RTX 4050 Laptop).
fn zeroed(n: usize) -> Option<Vec<u8>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).ok()?;
    v.resize(n, 0);
    Some(v)
}

/// Renames the written shard `tmp` over the shard's file in `dir`, returning
/// `true` when replaced in place or `false` when deferred as `.next`.
fn replace_shard(dir: &std::path::Path, at: usize, tmp: &std::path::Path) -> std::io::Result<bool> {
    match std::fs::rename(tmp, dir.join(shard_file(at))) {
        Ok(()) => Ok(true),
        Err(_) => std::fs::rename(tmp, dir.join(shard_next_file(at))).map(|()| false),
    }
}

/// Renames the written library `tmp` over the legacy library's file in `dir`.
#[cfg(test)]
fn replace_library(dir: &std::path::Path, tmp: &std::path::Path) -> std::io::Result<()> {
    std::fs::rename(tmp, dir.join(LIBRARY_FILE)).or_else(|_| std::fs::rename(tmp, dir.join(NEXT_FILE)))
}

/// Writes `bytes` to a temporary file beside `path`, then renames it over
/// `path`: a reader never sees a part. Serialised so concurrent writes do not
/// contend on the target rename.
fn write_atomically(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    static WRITING: Mutex<()> = Mutex::new(());
    let _guard = WRITING.lock().unwrap_or_else(|e| e.into_inner());
    let tmp = staging_path(path);
    let res = std::fs::write(&tmp, bytes).and_then(|()| std::fs::rename(&tmp, path));
    if res.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    res
}

/// Parses a canonical 32-character lowercase hexadecimal pipeline key as `u128`.
/// Non-canonical keys (different length, uppercase, sign, non-hex) return `None`
/// and fall back to exact string storage so semantics and round-tripping stay identical.
fn parse_key_u128(key: &str) -> Option<u128> {
    (key.len() == 32 && key.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')))
        .then(|| u128::from_str_radix(key, 16).ok())
        .flatten()
}

/// Compact set of pipeline keys: canonical 32-hex keys live in `HashSet<u128>`
/// with zero per-key heap `String` allocations; any non-32-hex line falls back
/// to `other`.
#[derive(Default)]
struct KeySet {
    hex: HashSet<u128>,
    other: Vec<String>,
}

impl KeySet {
    fn from_lines(text: &str) -> Self {
        let mut set = Self::default();
        for line in text.lines().filter(|l| !l.is_empty()) {
            set.insert(line);
        }
        set
    }

    fn insert(&mut self, key: &str) -> bool {
        if let Some(id) = parse_key_u128(key) {
            self.hex.insert(id)
        } else if self.other.iter().any(|s| s == key) {
            false
        } else {
            self.other.push(key.to_owned());
            true
        }
    }

    fn contains(&self, key: &str) -> bool {
        if let Some(id) = parse_key_u128(key) {
            self.hex.contains(&id)
        } else {
            self.other.iter().any(|s| s == key)
        }
    }

    fn len(&self) -> usize {
        self.hex.len() + self.other.len()
    }
}

/// The list to keep: this run's keys, then earlier runs' others, at most
/// `EARLY_MAX` (the oldest go once it is full).
fn early_list(before: &KeySet, now: &KeySet) -> Vec<String> {
    let mut out = Vec::with_capacity((now.len() + before.len()).min(EARLY_MAX));
    for &id in &now.hex {
        if out.len() == EARLY_MAX {
            return out;
        }
        out.push(format!("{id:032x}"));
    }
    for key in &now.other {
        if out.len() == EARLY_MAX {
            return out;
        }
        out.push(key.clone());
    }
    for &id in &before.hex {
        if out.len() == EARLY_MAX {
            return out;
        }
        if !now.hex.contains(&id) {
            out.push(format!("{id:032x}"));
        }
    }
    for key in &before.other {
        if out.len() == EARLY_MAX {
            return out;
        }
        if !now.other.iter().any(|k| k == key) {
            out.push(key.clone());
        }
    }
    out
}

/// Earlier runs' early-start keys, loaded once so [`wanted_early`] is lock-free.
static EARLY_BEFORE: OnceLock<KeySet> = OnceLock::new();
/// Title pipeline keys drawn or dispatched with in this run.
static USED_NOW: Mutex<Option<KeySet>> = Mutex::new(None);
/// Whether [`USED_NOW`] holds at least one key recorded in this run.
static HAS_USED_NOW: AtomicBool = AtomicBool::new(false);
static USED_CHANGED: AtomicBool = AtomicBool::new(false);

fn load_early_before(dir: Option<&std::path::Path>) -> KeySet {
    let listed = dir
        .and_then(|d| std::fs::read_to_string(d.join(EARLY_FILE)).ok())
        .unwrap_or_default();
    let before = KeySet::from_lines(&listed);
    tracing::info!(pipelines = before.len(), "title pipelines to start early");
    before
}

fn early_before() -> &'static KeySet {
    EARLY_BEFORE.get_or_init(|| load_early_before(DIR.get().map(PathBuf::as_path)))
}

/// Whether the title pipeline with `key` was requested in this or an earlier run.
pub(crate) fn wanted_early(key: &str) -> bool {
    if early_before().contains(key) {
        return true;
    }
    if !HAS_USED_NOW.load(Ordering::Acquire) {
        return false;
    }
    USED_NOW
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|s| s.contains(key))
}

/// A draw or dispatch requested the title pipeline with `key`.
pub(crate) fn note_used(key: &str) {
    let _ = early_before();
    let mut now = USED_NOW.lock().unwrap_or_else(|e| e.into_inner());
    let set = now.get_or_insert_with(KeySet::default);
    if set.len() < EARLY_MAX && set.insert(key) {
        HAS_USED_NOW.store(true, Ordering::Release);
        USED_CHANGED.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_change_with_what_goes_into_the_pipeline() {
        let a = name("graphics", b"root", &[b"vs", b"ps"], "state");
        assert_eq!(a, name("graphics", b"root", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps2"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps"], "state2"));
        assert_ne!(a, name("graphics", b"root2", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("mesh", b"root", &[b"vs", b"ps"], "state"));
        // Parts do not run into each other.
        assert_ne!(a, name("graphics", b"root", &[b"vsp", b"s"], "state"));
    }

    #[test]
    fn a_rejected_blob_gives_an_empty_library() {
        let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok()) else {
            return; // no host GPU here
        };
        let library = open(&device, Blob::Held(b"not a pipeline library".repeat(16))).expect("an empty library");
        // SAFETY: a plain query on a live library.
        assert!(unsafe { library.library.GetSerializedSize() } > 0);
        assert!(matches!(library._blob, Blob::None));
    }

    /// A mapped file reads as the file's bytes, a file written through a
    /// mapping holds what was written, and the library's file can be
    /// replaced while it is mapped (or the new one waits as the next).
    #[test]
    fn the_library_file_is_mapped_and_replaced() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(LIBRARY_FILE);
        assert!(matches!(Blob::of(dir.path()), Blob::None), "no file yet");
        std::fs::write(&path, b"").unwrap();
        assert!(matches!(Blob::of(dir.path()), Blob::None), "an empty file");
        let old: Vec<u8> = (0..200_000u32).map(|i| i as u8).collect();
        std::fs::write(&path, &old).unwrap();
        let blob = Blob::of(dir.path());
        assert!(matches!(blob, Blob::Mapped(_)) || !mapped());
        assert_eq!(blob.bytes(), &old[..]);
        // A save while the old file is mapped.
        let tmp = path.with_extension("tmp");
        let new: Vec<u8> = (0..300_000u32).map(|i| (i * 7) as u8).collect();
        {
            let mut file = Mapped::create(&tmp, new.len()).expect("a new file, mapped");
            file.bytes_mut().copy_from_slice(&new);
        }
        assert_eq!(std::fs::read(&tmp).unwrap(), new);
        replace_library(dir.path(), &tmp).unwrap();
        assert!(!tmp.exists());
        // The mapping still reads the old library.
        assert_eq!(blob.bytes(), &old[..]);
        drop(blob);
        // The next start reads the new one, wherever the save left it.
        assert_eq!(Blob::of(dir.path()).bytes(), &new[..]);
        assert!(!dir.path().join(NEXT_FILE).exists());
    }

    /// The bytes at `at` of the pattern the test below writes.
    fn pattern(at: usize, out: &mut [u8]) {
        for (i, word) in out.chunks_exact_mut(4).enumerate() {
            let v = ((at / 4 + i) as u32).wrapping_mul(2_654_435_761).wrapping_add(1);
            word.copy_from_slice(&v.to_le_bytes());
        }
    }

    /// A file of `len` bytes (whole MiB) of [`pattern`], written past the
    /// system's cache: none of its pages is in memory afterwards. False
    /// where the folder's file system does not take such writes.
    fn write_uncached(path: &std::path::Path, len: usize) -> bool {
        use std::io::Write;
        use std::os::windows::fs::OpenOptionsExt;
        // FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH
        const UNCACHED: u32 = 0x2000_0000 | 0x8000_0000;
        const CHUNK: usize = 1 << 20;
        let Ok(mut file) = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .custom_flags(UNCACHED)
            .open(path)
        else {
            return false;
        };
        // Such writes want a buffer on a sector boundary.
        let mut store = vec![0u8; CHUNK + 4096];
        let start = store.as_ptr().align_offset(4096);
        for at in (0..len).step_by(CHUNK) {
            let chunk = &mut store[start..start + CHUNK];
            pattern(at, chunk);
            if file.write_all(chunk).is_err() {
                return false;
            }
        }
        true
    }

    /// The device loss of runs 72, 73 and 82: the library's file is
    /// replaced while it is mapped, and pages of the mapping that were not
    /// in memory then are read afterwards (in the runs: pages the trim let
    /// go and Windows gave to something else). With the file's handle kept
    /// open they still read as the old file. (With it closed, as before,
    /// all but the first few read as zeros on the PC this was found on.)
    #[test]
    fn a_replaced_library_file_still_reads_from_the_disk() {
        const LEN: usize = 8 << 20;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(LIBRARY_FILE);
        if !write_uncached(&path, LEN) {
            return; // no uncached writes here: nothing to show
        }
        let mapped = Mapped::open_holding(&path, true).expect("the file, mapped");
        // A save: the new library, renamed over the mapped one.
        let tmp = path.with_extension("tmp");
        {
            let mut file = Mapped::create(&tmp, LEN).expect("a new file, mapped");
            file.bytes_mut().fill(0xEE);
        }
        replace_library(dir.path(), &tmp).unwrap();
        // Every page of the old mapping, read for the first time now.
        let mut expected = vec![0u8; 4096];
        for (page, bytes) in mapped.bytes().chunks(4096).enumerate() {
            pattern(page * 4096, &mut expected);
            assert!(bytes == &expected[..], "page {page} of the replaced library no longer reads as the old file");
        }
    }

    #[test]
    fn name_with_matches_name() {
        let chunks = ["alpha ", "beta\n", "gamma \"quoted\" ", "δ"];
        let full: String = chunks.concat();
        assert_eq!(
            name_with("graphics", b"root-blob", &[b"vs-code", b"ps-code"], |out| {
                for chunk in chunks {
                    out.write_str(chunk)?;
                }
                Ok(())
            }),
            name("graphics", b"root-blob", &[b"vs-code", b"ps-code"], &full),
        );
        assert_eq!(
            name_with("compute", b"", &[b""], |_| Ok(())),
            name("compute", b"", &[b""], ""),
        );
    }

    /// This run's keys stay in the list; earlier runs' fill it up to the cap.
    #[test]
    fn the_early_list_keeps_this_run_first() {
        let keys = |r: std::ops::Range<usize>| {
            let mut set = KeySet::default();
            for i in r {
                set.insert(&format!("{i:032x}"));
            }
            set
        };
        let before = keys(0..EARLY_MAX);
        let now = keys(EARLY_MAX - 10..EARLY_MAX + 10);
        let list = early_list(&before, &now);
        assert_eq!(list.len(), EARLY_MAX);
        assert!((EARLY_MAX - 10..EARLY_MAX + 10).all(|i| list.contains(&format!("{i:032x}"))));
        assert_eq!(list.iter().collect::<HashSet<_>>().len(), EARLY_MAX, "no key twice");
    }

    #[test]
    fn parse_key_u128_accepts_only_canonical_32_hex() {
        assert_eq!(
            parse_key_u128("00000000000000000000000000000000"),
            Some(0)
        );
        assert_eq!(
            parse_key_u128("ffffffffffffffffffffffffffffffff"),
            Some(u128::MAX)
        );
        assert_eq!(
            parse_key_u128("0123456789abcdef0123456789abcdef"),
            Some(0x0123_4567_89ab_cdef_0123_4567_89ab_cdef)
        );
        for non_canonical in [
            "",
            "deadbeef",
            "0123456789abcdef0123456789abcde",   // 31 chars
            "0123456789abcdef0123456789abcdef0", // 33 chars
            "0123456789ABCDEF0123456789ABCDEF",  // uppercase
            "+0123456789abcdef0123456789abcde",  // leading +
            "-0123456789abcdef0123456789abcde",  // leading -
            " 0123456789abcdef0123456789abcde",  // leading space
            "0123456789abcdef0123456789abcdeg",  // non-hex char
        ] {
            assert_eq!(parse_key_u128(non_canonical), None, "{non_canonical:?}");
        }
    }

    #[test]
    fn early_before_lock_free_lookup_handles_hex_and_non_hex() {
        let dir = tempfile::tempdir().unwrap();
        let hex_a = "00000000000000000000000000000001";
        let hex_b = "abcdef0123456789abcdef0123456789";
        let hex_missing = "abcdef0123456789abcdef0123456780";
        let upper = "ABCDEF0123456789ABCDEF0123456789";
        let plus = "+0000000000000000000000000000001";
        let non_hex = "custom-pipeline-key";

        let content = format!(
            "\n{hex_a}\n{hex_b}\n{hex_a}\n\n{upper}\n{plus}\n{non_hex}\n{non_hex}\n"
        );
        std::fs::write(dir.path().join(EARLY_FILE), content).unwrap();

        let cell: OnceLock<KeySet> = OnceLock::new();
        let loaded = cell.get_or_init(|| load_early_before(Some(dir.path())));
        assert_eq!(loaded.hex.len(), 2);
        assert_eq!(loaded.other.len(), 3);
        assert_eq!(loaded.len(), 5);

        std::thread::scope(|s| {
            for _ in 0..4 {
                s.spawn(|| {
                    let before = cell.get().unwrap();
                    assert!(before.contains(hex_a));
                    assert!(before.contains(hex_b));
                    assert!(!before.contains(hex_missing));
                    assert!(before.contains(upper));
                    assert!(before.contains(plus));
                    assert!(before.contains(non_hex));
                    assert!(!before.contains("00000000000000000000000000000002"));
                    assert!(!before.contains("other-missing"));
                });
            }
        });

        // Missing directory / file loads cleanly as empty.
        assert_eq!(load_early_before(None).len(), 0);
        let empty_dir = tempfile::tempdir().unwrap();
        assert_eq!(load_early_before(Some(empty_dir.path())).len(), 0);

        // `early_list` merges both hex and fallback keys with `now` first and no duplicates.
        let mut now = KeySet::default();
        now.insert(hex_b);
        now.insert(hex_missing);
        now.insert(non_hex);
        now.insert("new-custom");
        let merged = early_list(loaded, &now);
        assert_eq!(merged.len(), 7);
        assert_eq!(merged.iter().collect::<HashSet<_>>().len(), 7);
        for expected in [hex_a, hex_b, hex_missing, upper, plus, non_hex, "new-custom"] {
            assert!(merged.iter().any(|k| k == expected), "missing {expected}");
        }
    }

    /// Staging files of cut writes go: this process's at once, others' and
    /// legacy `.tmp` once old; the files themselves and anything else stay.
    #[test]
    fn stale_staging_files_are_swept() {
        let dir = tempfile::tempdir().unwrap();
        let old = std::time::SystemTime::now() - 2 * STAGING_KEPT;
        let file = |d: &std::path::Path, name: &str, modified: Option<std::time::SystemTime>| {
            let f = std::fs::File::create(d.join(name)).unwrap();
            if let Some(t) = modified {
                f.set_modified(t).unwrap();
            }
        };
        let gone = [
            "host-pipelines.bin.tmp-7-0",
            "host-pipelines.00.bin.tmp-7-0",
            "host-pipelines.0f.bin.tmp-7-1",
            "host-pipelines.tmp-7-1",
            "early-pipelines.txt.tmp-7-3",
            "early-pipelines.tmp-7-4",
            "shader-meta.bin.tmp-7-5",
            "prebuilt-pipelines.03.txt.tmp-7-6",
            "host-pipelines.bin.tmp-8-1",
            "host-pipelines.15.bin.tmp-8-1",
            "host-pipelines.tmp-8-1",
            "prebuilt-pipelines.txt.tmp-9-12",
            "prebuilt-pipelines.15.txt.tmp-9-13",
            "host-pipelines.tmp",
            "early-pipelines.tmp",
        ];
        let kept = [
            "host-pipelines.bin",
            "host-pipelines.00.bin",
            "host-pipelines.00.next",
            "host-pipelines.00.rejected.bin",
            "early-pipelines.txt",
            "prebuilt-pipelines.00.txt",
            "host-pipelines.bin.tmp-8-2",    // another process's, recent
            "host-pipelines.00.bin.tmp-8-2", // another process's shard, recent
            "host-pipelines.tmp-8-2",        // another process's stem-tmp, recent
            "host-pipelines.bin.tmp-7-x",    // not a staging name
            "host-pipelines.00.bin.tmp-7-x", // not a staging name
            "host-pipelines.bin.tmp-7",      // not a staging name
            "other.bin.tmp-7-0",             // not a file of ours
        ];
        for (i, name) in gone.iter().enumerate() {
            file(dir.path(), name, (i >= 8).then_some(old));
        }
        for name in kept {
            file(dir.path(), name, (!name.contains("tmp-8-2")).then_some(old));
        }
        std::fs::create_dir(dir.path().join("host-pipelines.bin.tmp-9-0")).unwrap();
        sweep_staging(dir.path(), 7, STAGING_KEPT);
        for name in gone {
            assert!(!dir.path().join(name).exists(), "{name} swept");
        }
        for name in kept.into_iter().chain(["host-pipelines.bin.tmp-9-0"]) {
            assert!(dir.path().join(name).exists(), "{name} kept");
        }

        // Legacy `.tmp` files without a pid are kept while recent.
        let recent_dir = tempfile::tempdir().unwrap();
        file(recent_dir.path(), "host-pipelines.tmp", None);
        file(recent_dir.path(), "early-pipelines.tmp", None);
        sweep_staging(recent_dir.path(), 7, STAGING_KEPT);
        assert!(recent_dir.path().join("host-pipelines.tmp").exists());
        assert!(recent_dir.path().join("early-pipelines.tmp").exists());
    }

    /// A size the host cannot give is refused, not fatal; others are zeros.
    #[test]
    fn a_save_buffer_the_host_cannot_give_is_refused() {
        assert!(zeroed(usize::MAX / 2).is_none());
        assert_eq!(zeroed(100).as_deref(), Some(&[0u8; 100][..]));
    }

    /// The pre-build's list holds only for the library bytes it was
    /// saved with.
    #[test]
    fn the_prebuilt_list_holds_for_its_library_only() {
        let keys: HashSet<String> = ["a".repeat(32), "b".repeat(32)].into();
        let saved = list(&tag(b"library"), &keys);
        assert_eq!(listed(saved.as_bytes(), &tag(b"library")), keys);
        assert!(listed(saved.as_bytes(), &tag(b"library2")).is_empty());
        assert!(listed(b"", &tag(b"library")).is_empty());
    }

    /// A pipeline's shard is stable across runs and spread over all of them,
    /// which is what makes a save write only what changed.
    #[test]
    fn names_spread_over_the_shards_and_keep_their_shard() {
        let mut seen = std::collections::HashSet::new();
        for i in 0..10_000u32 {
            let name = format!("{i:032x}");
            let shard = shard_of(&name);
            assert!(shard < SHARDS);
            assert_eq!(shard, shard_of(&name), "the same name, the same shard");
            seen.insert(shard);
        }
        assert_eq!(seen.len(), SHARDS, "every shard gets names");
    }

    #[test]
    fn concurrent_write_atomically_never_fails_or_truncates() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("host-pipelines.00.bin");
        let threads: Vec<_> = (0..4)
            .map(|t| {
                let target = target.clone();
                std::thread::spawn(move || {
                    let payload = vec![b'a' + t as u8; 4096];
                    for _ in 0..30 {
                        write_atomically(&target, &payload).unwrap();
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        let final_bytes = std::fs::read(&target).unwrap();
        assert_eq!(final_bytes.len(), 4096);
        assert!(final_bytes.iter().all(|&b| b == final_bytes[0]));
    }

    /// A stored pipeline loads while a save holds the shard's state lock.
    #[test]
    fn a_pipeline_loads_while_a_save_holds_the_library() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::core::s;
        const NAME: &str = "test: a pipeline loads while a save holds the library";
        let (Some(device), Some(libraries)) = (
            crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok()),
            libraries(),
        ) else {
            return; // no host GPU here
        };
        // No resources, and the root signature made apart: a compiler need
        // not keep one the source names (Wine's does not).
        let source = "[numthreads(1, 1, 1)] void main() {}";
        let mut blob = None;
        // SAFETY: the source outlives the call; the blob is read within it.
        let code = unsafe {
            if D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                None,
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut blob,
                None,
            )
            .is_err()
            {
                return; // no shader compiler here
            }
            let blob = blob.unwrap();
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
                .to_vec()
        };
        // SAFETY: an empty description; the blob is read while it lives.
        let root: ID3D12RootSignature = unsafe {
            let mut blob = None;
            D3D12SerializeRootSignature(
                &D3D12_ROOT_SIGNATURE_DESC::default(),
                D3D_ROOT_SIGNATURE_VERSION_1,
                &mut blob,
                None,
            )
            .unwrap();
            let blob = blob.unwrap();
            let bytes = std::slice::from_raw_parts(
                blob.GetBufferPointer().cast::<u8>(),
                blob.GetBufferSize(),
            );
            device.CreateRootSignature(0, bytes).unwrap()
        };
        struct Made(ID3D12Device1, ID3D12RootSignature, Vec<u8>);
        // SAFETY: D3D12 objects are free-threaded.
        unsafe impl Sync for Made {}
        impl Made {
            fn get(&self, made: bool) -> windows::core::Result<ID3D12PipelineState> {
                let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    // SAFETY: a borrowed pointer the description never releases.
                    pRootSignature: unsafe { std::mem::transmute_copy(&self.1) },
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: self.2.as_ptr().cast(),
                        BytecodeLength: self.2.len(),
                    },
                    ..Default::default()
                };
                pipeline(
                    NAME,
                    // SAFETY: the description points at live data.
                    |library, name| unsafe { library.LoadComputePipeline(name, &desc) },
                    || match made {
                        // SAFETY: as above.
                        true => unsafe { self.0.CreateComputePipelineState(&desc) },
                        false => panic!("made again, not loaded"),
                    },
                )
            }
        }
        let made = Made(device, root, code);
        made.get(true).expect("made and stored");
        let saving = libraries[shard_of(NAME)].state.lock().unwrap_or_else(|e| e.into_inner());
        let (done, loaded) = std::sync::mpsc::channel();
        std::thread::scope(|s| {
            let made = &made;
            s.spawn(move || done.send(made.get(false).is_ok()));
            let loaded = loaded.recv_timeout(std::time::Duration::from_secs(30));
            drop(saving);
            assert_eq!(loaded, Ok(true), "the load waited for the save");
        });
    }

    #[test]
    fn pipelines_and_prebuilt_keys_added_during_publication_are_saved_next_time() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::core::s;

        let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok())
        else {
            return;
        };
        let source = br#"
[RootSignature("UAV(u0)")]
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {}
"#;
        let mut code = None;
        // SAFETY: the source and entry point live across the compiler call.
        unsafe {
            D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                None,
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )
        }
        .expect("compile the synthetic compute shader");
        let code = code.unwrap();
        // SAFETY: the compiled blob lives through root-signature and PSO creation.
        let (root, bytecode) = unsafe {
            let bytes =
                std::slice::from_raw_parts(code.GetBufferPointer().cast(), code.GetBufferSize());
            (
                device
                    .CreateRootSignature::<ID3D12RootSignature>(0, bytes)
                    .unwrap(),
                bytes,
            )
        };
        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            // SAFETY: `root` outlives the description's users.
            pRootSignature: unsafe { std::mem::transmute_copy(&root) },
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytecode.as_ptr().cast(),
                BytecodeLength: bytecode.len(),
            },
            ..Default::default()
        };
        // SAFETY: the description holds a live signature and shader blob.
        let pipeline: ID3D12PipelineState =
            unsafe { device.CreateComputePipelineState(&desc) }.unwrap();
        let dir = tempfile::tempdir().unwrap();
        let shared = shard(&device, Blob::None, None, None).expect("an empty native library");
        let at = 0;
        let blob_path = dir.path().join(shard_file(at));
        let sidecar_path = dir.path().join(prebuilt_file(at));
        let store = |key: &str| {
            let mut l = shared.state.lock().unwrap();
            // SAFETY: the pipeline and unique name are live.
            unsafe { l.library.StorePipeline(&HSTRING::from(key), &pipeline) }.unwrap();
            l.added += 1;
            l.prebuilt_changed |= l.prebuilt.insert(key.to_owned());
        };
        let published = || {
            let blob = std::fs::read(&blob_path).unwrap();
            let sidecar = std::fs::read(&sidecar_path).unwrap();
            let reopened =
                shard(&device, Blob::Held(blob), Some(sidecar), None).expect("published shard");
            assert!(
                !reopened.state.lock().unwrap()._blob.is_empty(),
                "host accepted the published blob"
            );
            let loaded = |key| {
                // SAFETY: the library and matching description are live.
                unsafe {
                    reopened
                        .loads
                        .LoadComputePipeline::<_, ID3D12PipelineState>(&HSTRING::from(key), &desc)
                }
                .is_ok()
            };
            let keys = reopened.state.lock().unwrap().prebuilt.clone();
            (loaded("before"), loaded("during"), keys)
        };

        store("before");
        assert!(
            save_shard(dir.path(), at, &shared, |path, bytes| {
                if path == blob_path {
                    // The writer has captured its snapshot and released the state lock.
                    store("during");
                }
                write_atomically(path, bytes)
            })
            .is_some()
        );
        let first = published();
        assert_eq!(first, (true, false, HashSet::from(["before".to_owned()])));
        let pending = {
            let l = shared.state.lock().unwrap();
            (l.added, l.prebuilt_changed)
        };
        println!(
            "first save: loaded before={}, during={}; pending stores={}, prebuilt dirty={}",
            first.0, first.1, pending.0, pending.1
        );

        let second = save_shard(dir.path(), at, &shared, write_atomically);
        println!("second save wrote shard: {}", second.is_some());
        assert!(
            second.is_some(),
            "a pipeline stored during publication still needs saving"
        );
        assert_eq!(pending, (1, true));
        let second = published();
        assert_eq!(
            second,
            (
                true,
                true,
                HashSet::from(["before".to_owned(), "during".to_owned()])
            )
        );
        println!(
            "reopened second save: loaded before={}, during={}, sidecar contains both keys",
            second.0, second.1
        );
        assert!(save_shard(dir.path(), at, &shared, write_atomically).is_none());
    }

    #[test]
    fn shard_write_failure_leaves_shard_dirty() {
        let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok())
        else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let shared = shard(&device, Blob::None, None, None).expect("an empty native library");
        {
            let mut l = shared.state.lock().unwrap();
            l.prebuilt_changed |= l.prebuilt.insert("pending-key".to_owned());
            assert!(l.dirty());
        }
        let saved = save_shard(dir.path(), 0, &shared, |_, _| {
            Err(std::io::Error::other("simulated disk failure"))
        });
        assert!(saved.is_none());
        let l = shared.state.lock().unwrap();
        assert!(l.dirty(), "failed write must leave shard dirty");
        assert!(l.prebuilt_changed);
    }
}
