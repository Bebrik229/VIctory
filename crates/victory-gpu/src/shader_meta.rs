//! Persistent per-shader metadata cache (`shader-meta.bin`).
//!
//! Stores `(used, stores)` keyed by the 32-byte BLAKE3 hash of the console
//! shader container so warm pipeline-cache loads skip `Container::parse`, zlib
//! decompression (`native_programs`), and RDNA2 instruction scanning.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{OnceLock, RwLock};

use victory_shader::{ResourceType, RootType, UsedResource};

const FILE_NAME: &str = "shader-meta.bin";
const MAGIC: &[u8; 4] = b"VSM1";
const VERSION: u32 = 1;

type MetaEntry = (Vec<UsedResource>, bool);
type MetaMap = HashMap<[u8; 32], MetaEntry>;

struct MetaStore {
    map: MetaMap,
    dirty: bool,
}

impl MetaStore {
    fn empty() -> Self {
        Self {
            map: HashMap::new(),
            dirty: false,
        }
    }

    fn load(dir: &Path) -> Self {
        Self {
            map: load_from(&dir.join(FILE_NAME)).unwrap_or_default(),
            dirty: false,
        }
    }
}

static STORE: OnceLock<RwLock<MetaStore>> = OnceLock::new();
static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

fn global_store() -> &'static RwLock<MetaStore> {
    STORE.get_or_init(|| RwLock::new(MetaStore::empty()))
}

/// Loads `dir/shader-meta.bin` into the global metadata store.
///
/// Missing or corrupt files silently initialize an empty store.
pub(crate) fn init(dir: &Path) {
    let loaded = MetaStore::load(dir);
    if let Err(lock) = STORE.set(RwLock::new(loaded)) {
        let mut guard = global_store().write().unwrap_or_else(|e| e.into_inner());
        *guard = lock.into_inner().unwrap_or_else(|e| e.into_inner());
    }
}

/// Returns cached `(used, stores)` for `hash`, or computes and inserts it on miss.
pub(crate) fn get_or_compute<F>(
    hash: &[u8; 32],
    compute: F,
) -> Result<(Vec<UsedResource>, bool), String>
where
    F: FnOnce() -> Result<(Vec<UsedResource>, bool), String>,
{
    get_or_compute_in(global_store(), hash, compute)
}

/// Atomically flushes dirty metadata entries to `dir/shader-meta.bin`.
pub(crate) fn save(dir: &Path) {
    save_in(global_store(), dir);
}

fn get_or_compute_in<F>(
    store: &RwLock<MetaStore>,
    hash: &[u8; 32],
    compute: F,
) -> Result<MetaEntry, String>
where
    F: FnOnce() -> Result<MetaEntry, String>,
{
    {
        let guard = store.read().unwrap_or_else(|e| e.into_inner());
        if let Some(entry) = guard.map.get(hash) {
            return Ok(entry.clone());
        }
    }

    let val = compute()?;

    {
        let mut guard = store.write().unwrap_or_else(|e| e.into_inner());
        let mut inserted = false;
        guard.map.entry(*hash).or_insert_with(|| {
            inserted = true;
            val.clone()
        });
        if inserted {
            guard.dirty = true;
        }
    }

    Ok(val)
}

fn save_in(store: &RwLock<MetaStore>, dir: &Path) {
    {
        let guard = store.read().unwrap_or_else(|e| e.into_inner());
        if !guard.dirty {
            return;
        }
    }

    let bytes = {
        let mut guard = store.write().unwrap_or_else(|e| e.into_inner());
        if !guard.dirty {
            return;
        }
        guard.dirty = false;
        encode(&guard.map)
    };

    let n = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let tmp = dir.join(format!("shader-meta.bin.tmp-{}-{n}", std::process::id()));
    let target = dir.join(FILE_NAME);
    if let Err(e) = write_atomic(&tmp, &target, &bytes) {
        let _ = std::fs::remove_file(&tmp);
        let mut guard = store.write().unwrap_or_else(|e| e.into_inner());
        guard.dirty = true;
        tracing::warn!(path = %target.display(), error = %e, "failed to save shader metadata cache");
    }
}

fn write_atomic(tmp: &Path, target: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(tmp, bytes)?;
    std::fs::rename(tmp, target)?;
    Ok(())
}

fn load_from(path: &Path) -> Option<MetaMap> {
    let bytes = std::fs::read(path).ok()?;
    decode(&bytes)
}

fn encode_resource_type(out: &mut Vec<u8>, kind: ResourceType) {
    match kind {
        ResourceType::Srv => out.push(0),
        ResourceType::Sampler => out.push(1),
        ResourceType::Cbv => out.push(2),
        ResourceType::Uav => out.push(3),
        ResourceType::Other(x) => {
            out.push(4);
            out.push(x);
        }
    }
}

fn encode_root_type(out: &mut Vec<u8>, root: RootType) {
    match root {
        RootType::Constants => out.push(0),
        RootType::Descriptor => out.push(1),
        RootType::Table => out.push(2),
        RootType::ImmediateSampler => out.push(3),
        RootType::ResourceHeap => out.push(4),
        RootType::SamplerHeap => out.push(5),
        RootType::Other(x) => {
            out.push(6);
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
}

fn encode(map: &MetaMap) -> Vec<u8> {
    let mut out = Vec::with_capacity(12 + map.len() * 64);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(map.len() as u32).to_le_bytes());
    for (hash, (used, stores)) in map {
        out.extend_from_slice(hash);
        out.push(u8::from(*stores));
        out.extend_from_slice(&(used.len() as u32).to_le_bytes());
        for res in used {
            encode_resource_type(&mut out, res.kind);
            encode_root_type(&mut out, res.root);
            out.extend_from_slice(&res.table_offset_dwords.to_le_bytes());
            out.extend_from_slice(&res.lower_bound.to_le_bytes());
            out.extend_from_slice(&res.stride.to_le_bytes());
            out.push(res.dwords);
            out.push(res.user_data);
            out.push(res.slot);
        }
    }
    out
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let (head, rest) = self.0.split_at_checked(n)?;
        self.0 = rest;
        Some(head)
    }

    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn hash(&mut self) -> Option<[u8; 32]> {
        self.take(32)?.try_into().ok()
    }

    fn resource_type(&mut self) -> Option<ResourceType> {
        match self.u8()? {
            0 => Some(ResourceType::Srv),
            1 => Some(ResourceType::Sampler),
            2 => Some(ResourceType::Cbv),
            3 => Some(ResourceType::Uav),
            4 => Some(ResourceType::Other(self.u8()?)),
            _ => None,
        }
    }

    fn root_type(&mut self) -> Option<RootType> {
        match self.u8()? {
            0 => Some(RootType::Constants),
            1 => Some(RootType::Descriptor),
            2 => Some(RootType::Table),
            3 => Some(RootType::ImmediateSampler),
            4 => Some(RootType::ResourceHeap),
            5 => Some(RootType::SamplerHeap),
            6 => Some(RootType::Other(self.u32()?)),
            _ => None,
        }
    }
}

fn decode(bytes: &[u8]) -> Option<MetaMap> {
    const MIN_ENTRY_BYTES: usize = 32 + 1 + 4;
    const MIN_RESOURCE_BYTES: usize = 1 + 1 + 4 + 4 + 4 + 1 + 1 + 1;

    let mut r = Reader(bytes);
    if r.take(4)? != MAGIC {
        return None;
    }
    if r.u32()? != VERSION {
        return None;
    }

    let count = r.u32()? as usize;
    if count.checked_mul(MIN_ENTRY_BYTES)? > r.0.len() {
        return None;
    }

    let mut map = HashMap::with_capacity(count.min(256));
    for _ in 0..count {
        let hash = r.hash()?;
        let stores = match r.u8()? {
            0 => false,
            1 => true,
            _ => return None,
        };
        let used_len = r.u32()? as usize;
        if used_len.checked_mul(MIN_RESOURCE_BYTES)? > r.0.len() {
            return None;
        }
        let mut used = Vec::with_capacity(used_len.min(256));
        for _ in 0..used_len {
            used.push(UsedResource {
                kind: r.resource_type()?,
                root: r.root_type()?,
                table_offset_dwords: r.u32()?,
                lower_bound: r.u32()?,
                stride: r.u32()?,
                dwords: r.u8()?,
                user_data: r.u8()?,
                slot: r.u8()?,
            });
        }
        map.insert(hash, (used, stores));
    }

    r.0.is_empty().then_some(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_resources() -> Vec<UsedResource> {
        let kinds = [
            ResourceType::Srv,
            ResourceType::Sampler,
            ResourceType::Cbv,
            ResourceType::Uav,
            ResourceType::Other(0),
            ResourceType::Other(255),
        ];
        let roots = [
            RootType::Constants,
            RootType::Descriptor,
            RootType::Table,
            RootType::ImmediateSampler,
            RootType::ResourceHeap,
            RootType::SamplerHeap,
            RootType::Other(0),
            RootType::Other(u32::MAX),
        ];
        let mut out = Vec::new();
        for (i, (&kind, &root)) in kinds.iter().cycle().zip(roots.iter().cycle()).take(8).enumerate() {
            out.push(UsedResource {
                kind,
                root,
                table_offset_dwords: i as u32 * 4,
                lower_bound: 10 + i as u32,
                stride: 32 + i as u32,
                dwords: (4 + i) as u8,
                user_data: i as u8,
                slot: (i + 1) as u8,
            });
        }
        out
    }

    #[test]
    fn encode_decode_roundtrip_and_corruption() {
        let mut map = HashMap::new();
        let h1 = [0x11u8; 32];
        let h2 = [0x22u8; 32];
        map.insert(h1, (sample_resources(), true));
        map.insert(h2, (Vec::new(), false));

        let bytes = encode(&map);
        let decoded = decode(&bytes).expect("valid encoded map must decode");
        assert_eq!(decoded, map);

        // Wrong magic and version.
        let mut bad_magic = bytes.clone();
        bad_magic[3] = b'0';
        assert_eq!(decode(&bad_magic), None);

        let mut bad_ver = bytes.clone();
        bad_ver[4..8].copy_from_slice(&2u32.to_le_bytes());
        assert_eq!(decode(&bad_ver), None);

        // Truncated at any point and trailing garbage.
        for cut in 0..bytes.len() {
            assert_eq!(decode(&bytes[..cut]), None, "truncated at {cut}");
        }
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert_eq!(decode(&trailing), None);

        // Invalid stores byte on a single-entry payload.
        let mut single = HashMap::new();
        single.insert(h2, (Vec::new(), true));
        let mut bad_stores = encode(&single);
        bad_stores[12 + 32] = 2;
        assert_eq!(decode(&bad_stores), None);
    }

    #[test]
    fn store_persists_to_disk_and_skips_recompute() {
        let dir = tempfile::tempdir().unwrap();
        let store = RwLock::new(MetaStore::load(dir.path()));
        assert!(!store.read().unwrap().dirty);

        let hash = [0xABu8; 32];
        let expected = (sample_resources(), true);
        let runs = AtomicU64::new(0);

        // Miss computes once and marks store dirty.
        let got = get_or_compute_in(&store, &hash, || {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(expected.clone())
        })
        .unwrap();
        assert_eq!(got, expected);
        assert_eq!(runs.load(Ordering::Relaxed), 1);
        assert!(store.read().unwrap().dirty);

        // Second lookup on the same store hits without recomputing.
        let got2 = get_or_compute_in(&store, &hash, || {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok((Vec::new(), false))
        })
        .unwrap();
        assert_eq!(got2, expected);
        assert_eq!(runs.load(Ordering::Relaxed), 1);

        // Failed compute does not cache or dirty.
        let err_hash = [0xCDu8; 32];
        assert!(get_or_compute_in(&store, &err_hash, || Err("boom".to_owned())).is_err());

        // Save writes shader-meta.bin and clears dirty.
        save_in(&store, dir.path());
        assert!(dir.path().join(FILE_NAME).is_file());
        assert!(!store.read().unwrap().dirty);

        // Clean save is a no-op and leaves dirty false.
        save_in(&store, dir.path());
        assert!(!store.read().unwrap().dirty);

        // Fresh MetaStore loaded from disk hits without calling compute.
        let reloaded = RwLock::new(MetaStore::load(dir.path()));
        assert!(!reloaded.read().unwrap().dirty);
        let got3 = get_or_compute_in(&reloaded, &hash, || {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok((Vec::new(), false))
        })
        .unwrap();
        assert_eq!(got3, expected);
        assert_eq!(runs.load(Ordering::Relaxed), 1);
    }
}
