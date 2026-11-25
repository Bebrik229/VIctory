//! Compiled host pipelines kept on disk, and the title pipelines worth
//! starting early (decision 0007, "Pipeline cache").
//!
//! 1. Every host pipeline goes into the host's pipeline library, named by
//!    a hash of all that its host description holds (the translated
//!    shaders' code, which changes when a translation does, our root
//!    signature, the graphics state). The library is saved under the cache
//!    folder; the host refuses a library of another adapter or driver, and
//!    an entry whose description differs, so neither is ever used.
//! 2. The title pipelines a run drew or dispatched with are listed by a key
//!    known when the title creates them (their shader containers and
//!    graphics state); the next run hands those to the workers at
//!    creation, during loading.
//!
//! Library calls overlap: the host documents the library as safe to use
//! from several threads ("The pipeline library is thread-safe to use, and
//! will internally synchronize as necessary", `CreatePipelineLibrary`,
//! Thread Safety), with one exception kept here: two loads of the same
//! pipeline are made one after the other ([`same_name`]). Under one lock of
//! ours for every call, the eight workers made pipelines one at a time:
//! 87 to 96 of 110 looks at each worker found it waiting for that lock
//! (run 49), and an earlier run's 2,300 pipelines, all in the library, were
//! still being loaded 46 s into the opening cutscene.
//! `VICTORY_LIBRARY_LOCK=1`: one call at a time, as before.
//!
//! The library's file is mapped, not read into memory ([`Blob`]): the host
//! keeps pointing into the bytes it was made from for as long as it lives,
//! and read into a buffer they were 1,352 MB of the process's own memory
//! for the whole run (run 60), all of it in physical memory. Mapped, the
//! pages are the file's: only those the host reads are in memory, and the
//! system can drop them without writing them anywhere. A save writes the
//! new library through a mapping of the new file in the same way (a buffer
//! of the library's size was made, filled and written out at each save).
//! `VICTORY_LIBRARY_MAP=0`: read and written through buffers, as before.
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
use std::sync::{Mutex, MutexGuard, OnceLock, RwLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{HSTRING, Interface, PCWSTR};

static DIR: OnceLock<PathBuf> = OnceLock::new();

/// Keeps the pipeline library and the early list in `dir` from now on.
pub fn use_pipeline_cache(dir: PathBuf) {
    sweep_staging(&dir, std::process::id(), STAGING_KEPT);
    crate::shader_meta::init(&dir);
    let _ = DIR.set(dir);
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

/// The process whose staging file `name` is (`<file>.tmp-<pid>-<n>` or
/// `<stem>.tmp-<pid>-<n>` of a cache file), if it is one.
fn staging_pid(name: &str) -> Option<u32> {
    let rest = STAGING_PREFIXES
        .iter()
        .find_map(|prefix| name.strip_prefix(prefix))?;
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
        let path = dir.join(LIBRARY_FILE);
        let next = dir.join(NEXT_FILE);
        if next.exists() && std::fs::rename(&next, &path).is_err() {
            tracing::warn!("the host pipeline library saved last could not take the old one's place");
        }
        if mapped()
            && let Some(file) = Mapped::open(&path)
        {
            return Blob::Mapped(file);
        }
        match std::fs::read(&path) {
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
}

/// The mapped library file's view (address, bytes; 0: not mapped), when a
/// pipeline was last loaded from the library (`crate::host::now_ms`), and
/// whether the view's pages were let go since ([`trim`]).
static VIEW: [std::sync::atomic::AtomicUsize; 2] = [const { std::sync::atomic::AtomicUsize::new(0) }; 2];
static LOADED_AT: AtomicU64 = AtomicU64::new(0);
static TRIMMED: AtomicBool = AtomicBool::new(false);
/// The bytes of the view let go so far, for the memory line.
pub(crate) static TRIMMED_BYTES: AtomicU64 = AtomicU64::new(0);
/// When the view's pages were last let go (`crate::host::now_ms`), and how
/// often and when a save of this run last renamed a new library over the
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
/// running at that moment, whether this run replaced the mapped file and
/// let its pages go, and how many pages of the mapping read as zeros now.
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
    let (at, len) = (VIEW[0].load(Ordering::Relaxed), VIEW[1].load(Ordering::Relaxed));
    if at == 0 {
        text.push_str("; its file is not mapped");
        return text;
    }
    let ago = |ms: u64| crate::host::now_ms().saturating_sub(ms) as f64 / 1000.0;
    let _ = write!(
        text,
        "; its file is mapped ({len} bytes), the handle {}",
        if holds() { "kept open" } else { "closed (VICTORY_LIBRARY_HANDLE=0)" }
    );
    match REPLACED[0].load(Ordering::Relaxed) {
        0 => text.push_str("; this run has not replaced the file"),
        n => {
            let _ = write!(
                text,
                "; this run replaced the file under the mapping {n} times, last {:.1} s ago",
                ago(REPLACED[1].load(Ordering::Relaxed))
            );
        }
    }
    if TRIMMED_BYTES.load(Ordering::Relaxed) > 0 {
        let _ = write!(text, "; its pages were last let go {:.1} s ago", ago(TRIMMED_AT.load(Ordering::Relaxed)));
    }
    // Within two seconds: pages not in memory come from the disk.
    let started = std::time::Instant::now();
    // SAFETY: the view lives as long as the library, which is never dropped.
    let bytes = unsafe { std::slice::from_raw_parts(at as *const u8, len) };
    let (mut zero, mut looked, mut first) = (0u64, 0u64, None);
    for (n, page) in bytes.chunks(4096).enumerate() {
        if n % 4096 == 0 && started.elapsed() > std::time::Duration::from_secs(2) {
            break;
        }
        looked += 1;
        if page.iter().all(|&b| b == 0) {
            zero += 1;
            if first.is_none() {
                first = Some(n);
            }
        }
    }
    let _ = write!(text, "; {zero} of the {looked} pages looked at read as zeros now");
    if let Some(first) = first {
        let _ = write!(text, " (the first is page {first}): the library's contents were lost there");
    }
    text
}

/// Lets the mapped library file's pages go from the process's memory once
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
    let (at, len) = (VIEW[0].load(Ordering::Relaxed), VIEW[1].load(Ordering::Relaxed));
    if at == 0
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
    let work = move || {
        // SAFETY: the view lives as long as the library, which is never
        // dropped. Unlocking pages that are not locked fails, as expected,
        // and takes them out of the working set: the documented way to let
        // pages of a mapped file go.
        unsafe { windows_sys::Win32::System::Memory::VirtualUnlock(at as *const _, len) };
        TRIMMED_BYTES.fetch_add(len as u64, Ordering::Relaxed);
        TRIMMED_AT.store(crate::host::now_ms(), Ordering::Relaxed);
        crate::host::trail(|| format!("pipeline library: the mapped file's pages let go from memory ({len} bytes)"));
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
    added: AtomicU64,
    /// Stores share it and a save takes it alone: the size the library
    /// asks for must still be its size when it is written out.
    saving: RwLock<()>,
}

// SAFETY: D3D12 objects are free-threaded, and the library synchronises
// its own calls (see the module's notes); what it does not, two loads of
// one name, is serialised by `same_name`.
unsafe impl Send for Library {}
// SAFETY: as above; the other fields are an atomic and a lock.
unsafe impl Sync for Library {}

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
/// A library the driver refused, kept as it was ([`open`]).
const REJECTED_FILE: &str = "host-pipelines.rejected.bin";
/// A saved library that could not replace [`LIBRARY_FILE`] while this run
/// had it mapped (a system that refuses to replace a mapped file): the next
/// start renames it over the library first ([`Blob::of`]).
const NEXT_FILE: &str = "host-pipelines.next";
const EARLY_FILE: &str = "early-pipelines.txt";
/// Title pipelines listed for an early start, at most.
const EARLY_MAX: usize = 8192;

fn library() -> Option<&'static Library> {
    static LIBRARY: OnceLock<Option<Library>> = OnceLock::new();
    LIBRARY
        .get_or_init(|| {
            let device = crate::host::host()?.device.cast::<ID3D12Device1>().ok()?;
            let blob = DIR.get().map_or(Blob::None, |d| Blob::of(d));
            let library = open(&device, blob);
            if library.is_none() {
                tracing::warn!("no host pipeline library: pipelines are not kept");
            }
            library
        })
        .as_ref()
}

/// The library in `blob`, or an empty one if the host rejects the blob (an
/// adapter or driver change, damage).
fn open(device: &ID3D12Device1, blob: Blob) -> Option<Library> {
    let make = |bytes: &[u8]| -> windows::core::Result<ID3D12PipelineLibrary1> {
        // SAFETY: the bytes outlive the library (kept beside it).
        unsafe {
            device.CreatePipelineLibrary::<ID3D12PipelineLibrary>(bytes)?.cast()
        }
    };
    let mut rejected = false;
    if !blob.bytes().is_empty() {
        match make(blob.bytes()) {
            Ok(library) => {
                let bytes = blob.bytes().len();
                BLOB.store(bytes as u64, Ordering::Relaxed);
                tracing::info!(bytes, mapped = matches!(blob, Blob::Mapped(_)), "host pipeline library loaded");
                if let Blob::Mapped(file) = &blob {
                    VIEW[1].store(file.len, Ordering::Relaxed);
                    VIEW[0].store(file.at.Value as usize, Ordering::Relaxed);
                    LOADED_AT.store(crate::host::now_ms(), Ordering::Relaxed);
                    if prefetch() {
                        prefetch_mapped(file.at.Value as usize, file.len);
                    }
                }
                return Some(Library {
                    library,
                    _blob: blob,
                    added: AtomicU64::new(0),
                    saving: RwLock::new(()),
                });
            }
            Err(e) => {
                rejected = true;
                tracing::warn!(error = %e, "host pipeline library rejected; starting an empty one (the rejected file is kept as {REJECTED_FILE})");
            }
        }
    } else {
        tracing::info!("no host pipeline library yet; starting an empty one");
    }
    // The rejected bytes are let go (a mapped file is unmapped) before the
    // empty library is made: nothing reads them any more.
    drop(blob);
    // And the file is kept under another name, not written over by this
    // run's first save: a refusal may be this device's alone.
    if rejected
        && let Some(dir) = DIR.get()
        && let Err(e) = std::fs::rename(dir.join(LIBRARY_FILE), dir.join(REJECTED_FILE))
    {
        tracing::warn!(error = %e, "the rejected host pipeline library could not be kept under its other name");
    }
    let library = make(&[]).ok()?;
    Some(Library {
        library,
        _blob: Blob::None,
        added: AtomicU64::new(0),
        saving: RwLock::new(()),
    })
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
    load: impl FnOnce(&ID3D12PipelineLibrary1, PCWSTR) -> windows::core::Result<ID3D12PipelineState>,
    create: impl FnOnce() -> windows::core::Result<ID3D12PipelineState>,
) -> windows::core::Result<ID3D12PipelineState> {
    let wide = HSTRING::from(name);
    let Some(library) = library() else {
        MADE.fetch_add(1, Ordering::Relaxed);
        return create();
    };
    let began = std::time::Instant::now();
    let loaded = {
        let _all = one_call();
        let _one = same_name(name);
        let _call = call("load", name);
        load(&library.library, PCWSTR(wide.as_ptr()))
    };
    // The load read the library's file (see `trim`).
    LOADED_AT.store(crate::host::now_ms(), Ordering::Relaxed);
    TRIMMED.store(false, Ordering::Relaxed);
    let refused = match loaded {
        Ok(pso) => {
            LOADED.fetch_add(1, Ordering::Relaxed);
            crate::host::trail(|| {
                let label = LABEL.with(|l| l.borrow().clone());
                format!("pipeline library: loaded {label} in {:.1} ms", began.elapsed().as_secs_f64() * 1e3)
            });
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
    let _stores = library.saving.read().unwrap_or_else(|e| e.into_inner());
    let _call = call("store", name);
    // SAFETY: a live pipeline; a name already stored (a description that
    // differs) is refused, which leaves the library as it was.
    if unsafe { library.library.StorePipeline(PCWSTR(wide.as_ptr()), &pso) }.is_ok() {
        library.added.fetch_add(1, Ordering::Relaxed);
    }
    Ok(pso)
}

/// Writes the library and the early list if pipelines were added since the
/// last save and `SAVE_EVERY` has passed (or `now`).
pub(crate) fn save(now: bool) {
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    const SAVE_EVERY: std::time::Duration = std::time::Duration::from_secs(20);
    let (Some(dir), Some(library)) = (DIR.get(), library()) else { return };
    {
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        if !now && last.is_some_and(|t| t.elapsed() < SAVE_EVERY) {
            return;
        }
        *last = Some(std::time::Instant::now());
    }
    if let Err(e) = std::fs::create_dir_all(dir) {
        tracing::warn!(error = %e, "host pipeline library not saved");
        return;
    }
    crate::shader_meta::save(dir);
    let tmp = staging_path(&dir.join(LIBRARY_FILE));
    // The bytes of the library written to `tmp`; 0 when no pipeline was
    // added: the library on disk is then what it was, and only the early
    // list is written (a title pipeline first drawn with wrote all 1,352 MB
    // of the library again: run 60's second save).
    let bytes = {
        let _all = one_call();
        // No store between asking the size and writing; loads go on.
        let _alone = library.saving.write().unwrap_or_else(|e| e.into_inner());
        let added = library.added.load(Ordering::Relaxed);
        if !USED_CHANGED.swap(false, Ordering::Relaxed) && added == 0 {
            return;
        }
        if added == 0 {
            0
        } else {
            // SAFETY: a plain query.
            let size = unsafe { library.library.GetSerializedSize() };
            let _call = call("serialise", "the whole library");
            if let Err(e) = serialise(&library.library, &tmp, size) {
                tracing::warn!(error = %e, "host pipeline library not serialised");
                let _ = std::fs::remove_file(&tmp);
                return;
            }
            library.added.store(0, Ordering::Relaxed);
            SAVED.store(size as u64, Ordering::Relaxed);
            size
        }
    };
    let used = {
        let before = early_before();
        let now = USED_NOW.lock().unwrap_or_else(|e| e.into_inner());
        match now.as_ref() {
            Some(now) => early_list(before, now),
            None => early_list(before, &KeySet::default()),
        }
    };
    let library_written = if bytes > 0 { replace_library(dir, &tmp) } else { Ok(()) };
    if bytes > 0 && library_written.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    if bytes > 0 && library_written.is_ok() {
        // In place unless the system refused and the new one waits beside it.
        let in_place = !dir.join(NEXT_FILE).exists();
        if in_place && VIEW[0].load(Ordering::Relaxed) != 0 {
            REPLACED[1].store(crate::host::now_ms(), Ordering::Relaxed);
            REPLACED[0].fetch_add(1, Ordering::Relaxed);
        }
        crate::host::trail(|| {
            format!(
                "pipeline library: saved {bytes} bytes, {}",
                if in_place { "renamed over the library's file" } else { "left beside the library's file for the next start" }
            )
        });
    }
    let written = library_written
        .and_then(|()| write_atomically(&dir.join(EARLY_FILE), used.join("\n").as_bytes()));
    match written {
        Ok(()) => tracing::info!(
            bytes,
            early = used.len(),
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            "host pipeline library saved"
        ),
        Err(e) => tracing::warn!(error = %e, "host pipeline library not saved"),
    }
}

/// Writes the library's `size` serialised bytes to the file `tmp`: through
/// a mapping of that file, so that the bytes are the file's pages and not
/// 1.3 GB more of the process's own memory for the seconds a save takes;
/// through a buffer where the file cannot be mapped (and with
/// `VICTORY_LIBRARY_MAP=0`).
fn serialise(library: &ID3D12PipelineLibrary1, tmp: &std::path::Path, size: usize) -> std::io::Result<()> {
    let failed = |e: windows::core::Error| std::io::Error::other(e.to_string());
    if mapped()
        && let Some(mut file) = Mapped::create(tmp, size)
    {
        // SAFETY: a buffer of the size the library asks for.
        return unsafe { library.Serialize(file.bytes_mut()) }.map_err(failed);
    }
    let mut bytes = vec![0u8; size];
    // SAFETY: a buffer of the size the library asks for.
    unsafe { library.Serialize(&mut bytes) }.map_err(failed)?;
    std::fs::write(tmp, &bytes)
}

/// Renames the written library `tmp` over the library's file in `dir`. This
/// run may have that file mapped: Windows 11 replaces a mapped file, and
/// the mapping goes on reading the old one for as long as a handle on it
/// is open ([`holds`]; with none, only the pages in memory at that moment
/// stay: the rest read as zeros). Where a system refuses, the new library
/// is left beside it as [`NEXT_FILE`] for the next start.
fn replace_library(dir: &std::path::Path, tmp: &std::path::Path) -> std::io::Result<()> {
    std::fs::rename(tmp, dir.join(LIBRARY_FILE)).or_else(|_| std::fs::rename(tmp, dir.join(NEXT_FILE)))
}

/// Writes `bytes` to a temporary file beside `path`, then renames it over
/// `path`: a reader never sees a part.
fn write_atomically(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
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
            "host-pipelines.tmp-7-1",
            "early-pipelines.txt.tmp-7-3",
            "early-pipelines.tmp-7-4",
            "shader-meta.bin.tmp-7-5",
            "host-pipelines.bin.tmp-8-1",
            "host-pipelines.tmp-8-1",
            "prebuilt-pipelines.txt.tmp-9-12",
            "host-pipelines.tmp",
            "early-pipelines.tmp",
        ];
        let kept = [
            "host-pipelines.bin",
            "early-pipelines.txt",
            "host-pipelines.bin.tmp-8-2", // another process's, recent
            "host-pipelines.tmp-8-2",     // another process's stem-tmp, recent
            "host-pipelines.bin.tmp-7-x", // not a staging name
            "host-pipelines.bin.tmp-7",   // not a staging name
            "other.bin.tmp-7-0",          // not a file of ours
        ];
        for (i, name) in gone.iter().enumerate() {
            file(dir.path(), name, (i >= 5).then_some(old));
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
}
