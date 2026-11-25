//! Process-wide state the implementations need.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

static MAIN_IMAGE_BASE: AtomicUsize = AtomicUsize::new(0);

/// Records where the title's main image is mapped. The title expects
/// `GetModuleHandle(NULL)` and friends to return this, not our own exe.
pub fn set_main_image(base: usize) {
    MAIN_IMAGE_BASE.store(base, Ordering::Release);
}

pub fn main_image() -> Option<usize> {
    match MAIN_IMAGE_BASE.load(Ordering::Acquire) {
        0 => None,
        base => Some(base),
    }
}

/// Original file of each title image we mapped, by base address.
static IMAGE_PATHS: Mutex<Vec<(usize, PathBuf)>> = Mutex::new(Vec::new());

/// Records the file a title image was mapped from: the title asks for its
/// own path (`GetModuleFileName`) to find its data, and must not get our
/// cache copy.
pub fn set_image_path(base: usize, path: &Path) {
    let path = std::path::absolute(path).unwrap_or_else(|_| path.to_owned());
    let mut paths = IMAGE_PATHS.lock().unwrap_or_else(|e| e.into_inner());
    paths.retain(|(b, _)| *b != base);
    paths.push((base, path));
}

/// Forgets a title image that was unmapped again.
pub fn forget_image(base: usize) {
    let mut paths = IMAGE_PATHS.lock().unwrap_or_else(|e| e.into_inner());
    paths.retain(|(b, _)| *b != base);
}

pub fn image_path(base: usize) -> Option<PathBuf> {
    let paths = IMAGE_PATHS.lock().unwrap_or_else(|e| e.into_inner());
    paths
        .iter()
        .find(|(b, _)| *b == base)
        .map(|(_, p)| p.clone())
}

/// The title image whose original file is called `file` (any case).
pub fn image_named(file: &str) -> Option<usize> {
    let paths = IMAGE_PATHS.lock().unwrap_or_else(|e| e.into_inner());
    paths
        .iter()
        .find(|(_, p)| p.file_name().is_some_and(|n| n.eq_ignore_ascii_case(file)))
        .map(|(b, _)| *b)
}

/// Maps a title DLL the title loads itself (`LoadLibrary`) and returns its
/// base; `Ok(None)` if `file` is not one of the title's DLLs.
pub type TitleDllLoader = Box<dyn Fn(&str) -> Result<Option<usize>, String> + Send + Sync>;

static TITLE_DLL_LOADER: std::sync::OnceLock<TitleDllLoader> = std::sync::OnceLock::new();

/// Set once, by whoever mapped the title (it knows the folders and the
/// resolver); later calls are ignored.
pub fn set_title_dll_loader(loader: TitleDllLoader) {
    let _ = TITLE_DLL_LOADER.set(loader);
}

pub fn load_title_dll(file: &str) -> Result<Option<usize>, String> {
    TITLE_DLL_LOADER.get().map_or(Ok(None), |load| load(file))
}

/// What the title's manifest (`MicrosoftGame.config` beside its executable)
/// says about it, and where its persistent local storage lives.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Title {
    pub id: u32,
    /// The store's product id (empty if the manifest has none).
    pub store_id: String,
    /// Persistent local storage: our folder for it and the title's quota.
    /// Game saves live beside it ([`Title::saves`]).
    pub storage: PathBuf,
    pub storage_bytes: u64,
}

impl Title {
    /// Id, store id and storage quota from the manifest (zero or empty
    /// where it has none); the storage folder is left for the caller.
    pub fn from_manifest(text: &str) -> Title {
        let elements = elements(text);
        let get = |path: &str| {
            elements
                .iter()
                .find(|(p, _)| p == path)
                .map(|(_, text)| text.as_str())
        };
        let id = get("Game/TitleId").and_then(|v| u32::from_str_radix(v, 16).ok());
        let storage_mb = get("Game/PersistentLocalStorage/SizeMB").and_then(|v| v.parse().ok());
        Title {
            id: id.unwrap_or(0),
            store_id: get("Game/StoreId").unwrap_or_default().to_owned(),
            storage: PathBuf::new(),
            storage_bytes: storage_mb.unwrap_or(0) << 20,
        }
    }

    /// Where game saves live: next to persistent local storage, under the
    /// data directory, never the cache.
    pub fn saves(&self) -> PathBuf {
        self.storage.with_file_name("saves")
    }
}

/// Each element of an XML text as its path from the root (`Game/TitleId`)
/// and its trimmed text up to the first child, the predefined entities
/// decoded; comments, CDATA sections, declarations and attributes are
/// skipped.
fn elements(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut path: Vec<&str> = Vec::new();
    let mut rest = text;
    while let Some(open) = rest.find('<') {
        rest = &rest[open..];
        let end = if rest.starts_with("<!--") {
            rest.find("-->").map(|e| e + 3)
        } else if rest.starts_with("<![CDATA[") {
            rest.find("]]>").map(|e| e + 3)
        } else {
            rest.find('>').map(|e| e + 1)
        };
        let Some(end) = end else { break };
        let tag = &rest[1..end - 1];
        rest = &rest[end..];
        if tag.starts_with(['!', '?']) {
            continue;
        }
        if let Some(name) = tag.strip_prefix('/') {
            if let Some(at) = path.iter().rposition(|n| *n == name.trim()) {
                path.truncate(at);
            }
            continue;
        }
        let closed = tag.ends_with('/');
        let name = tag
            .trim_end_matches('/')
            .split_whitespace()
            .next()
            .unwrap_or("");
        path.push(name);
        let body = rest[..rest.find('<').unwrap_or(rest.len())].trim();
        let body = [("&lt;", "<"), ("&gt;", ">"), ("&quot;", "\""), ("&apos;", "'"), ("&amp;", "&")]
            .iter()
            .fold(body.to_owned(), |b, (entity, c)| b.replace(entity, c));
        out.push((path.join("/"), body));
        if closed {
            path.pop();
        }
    }
    out
}

static TITLE: Mutex<Option<Title>> = Mutex::new(None);

pub fn set_title(title: Title) {
    *TITLE.lock().unwrap_or_else(|e| e.into_inner()) = Some(title);
}

/// The running title; the default (id 0, no storage) until set.
pub fn title() -> Title {
    TITLE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
        .unwrap_or_default()
}

/// Gives the process the console's millisecond timers. Titles pace
/// themselves with short waits (the title's audio mixer sleeps
/// about 4 ms per 5.3 ms block) and cannot tune the timer on the console;
/// a desktop process waits in 15.6 ms ticks unless it asks for finer ones,
/// and Windows 11 drops that request while the window is hidden unless the
/// process opts out. Process-wide, for the whole run.
#[cfg(windows)]
fn apply_priority_and_throttling(execution: u32, target: Option<u32>) {
    use windows_sys::Win32::System::Threading::{
        ABOVE_NORMAL_PRIORITY_CLASS, BELOW_NORMAL_PRIORITY_CLASS, GetCurrentProcess,
        GetPriorityClass, HIGH_PRIORITY_CLASS, IDLE_PRIORITY_CLASS, NORMAL_PRIORITY_CLASS,
        PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION, PROCESS_POWER_THROTTLING_STATE,
        ProcessPowerThrottling, SetPriorityClass, SetProcessInformation,
    };
    let keep = PROCESS_POWER_THROTTLING_STATE {
        Version: PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        ControlMask: PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION | execution,
        StateMask: 0,
    };
    // SAFETY: plain process settings; `keep` outlives the call.
    unsafe {
        SetProcessInformation(
            GetCurrentProcess(),
            ProcessPowerThrottling,
            (&raw const keep).cast(),
            size_of_val(&keep) as u32,
        );
    }
    if let Some(target) = target {
        // SAFETY: `GetCurrentProcess()` is a valid pseudo-handle and `target` is a
        // valid priority class constant.
        unsafe {
            let current = GetPriorityClass(GetCurrentProcess());
            if current == 0
                || current == IDLE_PRIORITY_CLASS
                || current == BELOW_NORMAL_PRIORITY_CLASS
                || current == NORMAL_PRIORITY_CLASS
                || (target == HIGH_PRIORITY_CLASS && current == ABOVE_NORMAL_PRIORITY_CLASS)
            {
                SetPriorityClass(GetCurrentProcess(), target);
            }
        }
    }
}

#[cfg(windows)]
pub fn use_console_timers() {
    use windows_sys::Win32::Media::timeBeginPeriod;
    use windows_sys::Win32::System::Threading::{
        ABOVE_NORMAL_PRIORITY_CLASS, HIGH_PRIORITY_CLASS, PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
    };
    // Also opt out of Windows efficiency mode (reduced clocks / slow cores when
    // unfocused); `VICTORY_EFFICIENCY_MODE=1` leaves the system its choice.
    let execution = if std::env::var_os("VICTORY_EFFICIENCY_MODE").is_some_and(|v| v == "1") {
        0
    } else {
        PROCESS_POWER_THROTTLING_EXECUTION_SPEED
    };
    // Raise CPU scheduling priority so background tasks do not preempt the
    // title's frame pacing or audio mixer; `VICTORY_LOW_PRIORITY=1` skips it,
    // `VICTORY_HIGH_PRIORITY=1` bumps to `HIGH_PRIORITY_CLASS`.
    let target = if std::env::var_os("VICTORY_LOW_PRIORITY").is_some_and(|v| v == "1") {
        None
    } else if std::env::var_os("VICTORY_HIGH_PRIORITY").is_some_and(|v| v == "1") {
        Some(HIGH_PRIORITY_CLASS)
    } else {
        Some(ABOVE_NORMAL_PRIORITY_CLASS)
    };
    // SAFETY: valid process-wide timer resolution request.
    unsafe { timeBeginPeriod(1) };
    apply_priority_and_throttling(execution, target);

    static GUARD: std::sync::Once = std::sync::Once::new();
    GUARD.call_once(|| {
        let _ = std::thread::Builder::new()
            .name("victory-prio-guard".to_string())
            .spawn(move || loop {
                std::thread::sleep(std::time::Duration::from_secs(5));
                apply_priority_and_throttling(execution, target);
            });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_title_id_store_id_and_storage_quota() {
        let text = "\u{feff}<?xml version=\"1.0\"?>\n<Game configVersion=\"1\">\
                    <!-- <TitleId>00000000</TitleId> -->\
                    <Identity Name=\"x\"/>\n  <TitleId>639AA8DD</TitleId>\n\
                    <![CDATA[ <TitleId>1</TitleId> ]]>                    <StoreId>9&amp;C</StoreId><Other><SizeMB>7</SizeMB></Other>\
                    <PersistentLocalStorage>\n  <SizeMB>1024</SizeMB> <!-- c -->\n\
                    </PersistentLocalStorage></Game>";
        let t = Title::from_manifest(text);
        assert_eq!((t.id, t.storage_bytes), (0x639A_A8DD, 1024 << 20));
        assert_eq!(t.store_id, "9&C");
        assert_eq!(Title::from_manifest("").storage_bytes, 0);
    }
}
