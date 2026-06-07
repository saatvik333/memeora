//! Per-tag **profiles**: a compact, cached summary of a container's memories.
//!
//! A profile splits a container into two parts (matching how memory works):
//! - **static** — stable facts and preferences, strongest first;
//! - **dynamic** — recent episodes, newest first.
//!
//! Rebuilding from scratch scans every latest memory in the tag, so reads are
//! served from a [`ProfileCache`] keyed by tag and invalidated on write. Since
//! the daemon is the sole writer, it calls [`ProfileCache::invalidate`] after each
//! upsert — keeping cached reads ~free while never serving a stale profile.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::Result;
use crate::store::{Memory, MemoryKind, VectorStore, now_unix};

/// Tuning for [`build_profile`].
#[derive(Debug, Clone)]
pub struct ProfileParams {
    /// Max static (fact/preference) memories to keep.
    pub max_static: usize,
    /// Max dynamic (episode) memories to keep.
    pub max_dynamic: usize,
    /// Upper bound on latest memories scanned per tag when building.
    pub scan_cap: usize,
}

impl Default for ProfileParams {
    fn default() -> Self {
        ProfileParams {
            max_static: 20,
            max_dynamic: 10,
            scan_cap: 4096,
        }
    }
}

/// A compact view of a container: stable knowledge plus recent activity.
#[derive(Debug, Clone, Default)]
pub struct Profile {
    /// Facts and preferences, strongest (then newest) first.
    pub statics: Vec<Memory>,
    /// Episodes, newest first.
    pub dynamics: Vec<Memory>,
}

impl Profile {
    /// Total memories across both sections.
    pub fn len(&self) -> usize {
        self.statics.len() + self.dynamics.len()
    }

    /// Whether the profile holds no memories.
    pub fn is_empty(&self) -> bool {
        self.statics.is_empty() && self.dynamics.is_empty()
    }
}

/// Build a profile for `container_tag` from the store's latest, non-expired memories.
pub fn build_profile(
    store: &dyn VectorStore,
    container_tag: &str,
    params: &ProfileParams,
) -> Result<Profile> {
    let now = now_unix();
    let latest = store.list_latest(container_tag, params.scan_cap)?;

    let mut statics = Vec::new();
    let mut dynamics = Vec::new();
    for memory in latest {
        if memory.is_expired(now) {
            continue;
        }
        match memory.kind {
            MemoryKind::Fact | MemoryKind::Preference => statics.push(memory),
            MemoryKind::Episode => dynamics.push(memory),
        }
    }

    // Static: strongest first, ties broken by recency.
    statics.sort_by(|a, b| {
        b.strength
            .partial_cmp(&a.strength)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.created_at.cmp(&a.created_at))
    });
    statics.truncate(params.max_static);

    // Dynamic: newest first (list_latest is already ordered, but be explicit).
    dynamics.sort_by_key(|m| std::cmp::Reverse(m.created_at));
    dynamics.truncate(params.max_dynamic);

    Ok(Profile { statics, dynamics })
}

/// A per-tag cache of [`Profile`]s.
///
/// Profiles are returned as `Arc<Profile>` for cheap sharing across readers.
/// Writers must [`invalidate`](ProfileCache::invalidate) the affected tag.
pub struct ProfileCache {
    params: ProfileParams,
    cache: Mutex<HashMap<String, Arc<Profile>>>,
}

impl ProfileCache {
    /// Build a cache with the given parameters.
    pub fn new(params: ProfileParams) -> Self {
        ProfileCache {
            params,
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Build a cache with default [`ProfileParams`].
    pub fn with_defaults() -> Self {
        Self::new(ProfileParams::default())
    }

    /// Recover the lock guard even if a previous holder panicked — a poisoned
    /// profile cache is at worst stale, never unsafe.
    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, Arc<Profile>>> {
        self.cache.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Return the cached profile for `container_tag`, building and caching it on a miss.
    pub fn get_or_build(
        &self,
        store: &dyn VectorStore,
        container_tag: &str,
    ) -> Result<Arc<Profile>> {
        if let Some(profile) = self.lock().get(container_tag).cloned() {
            return Ok(profile);
        }
        // Build outside the lock (the store query may be slow); a concurrent
        // double-build is harmless since the result is identical.
        let profile = Arc::new(build_profile(store, container_tag, &self.params)?);
        self.lock()
            .insert(container_tag.to_string(), profile.clone());
        Ok(profile)
    }

    /// Drop the cached profile for `container_tag` (call after writing to it).
    pub fn invalidate(&self, container_tag: &str) {
        self.lock().remove(container_tag);
    }

    /// Drop all cached profiles.
    pub fn clear(&self) {
        self.lock().clear();
    }

    /// Number of cached tags.
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SqliteStore;

    fn mem(
        id: &str,
        kind: MemoryKind,
        strength: f32,
        created_at: i64,
        expires: Option<i64>,
    ) -> Memory {
        let mut m = Memory::new(id, id, kind, "tag", vec![1.0, 0.0]);
        m.strength = strength;
        m.created_at = created_at;
        m.expires_at = expires;
        m
    }

    fn store_with(memories: &[Memory]) -> SqliteStore {
        let mut s = SqliteStore::open_in_memory(2).unwrap();
        for m in memories {
            s.upsert(m).unwrap();
        }
        s
    }

    #[test]
    fn partitions_by_kind() {
        let s = store_with(&[
            mem("f", MemoryKind::Fact, 1.0, 100, None),
            mem("p", MemoryKind::Preference, 1.0, 101, None),
            mem("e", MemoryKind::Episode, 1.0, 102, None),
        ]);
        let p = build_profile(&s, "tag", &ProfileParams::default()).unwrap();
        assert_eq!(p.statics.len(), 2);
        assert_eq!(p.dynamics.len(), 1);
        assert_eq!(p.dynamics[0].id, "e");
        assert_eq!(p.len(), 3);
    }

    #[test]
    fn statics_ordered_by_strength() {
        let s = store_with(&[
            mem("weak", MemoryKind::Fact, 0.2, 100, None),
            mem("strong", MemoryKind::Fact, 0.9, 50, None),
        ]);
        let p = build_profile(&s, "tag", &ProfileParams::default()).unwrap();
        assert_eq!(p.statics[0].id, "strong");
        assert_eq!(p.statics[1].id, "weak");
    }

    #[test]
    fn dynamics_ordered_by_recency() {
        let s = store_with(&[
            mem("old", MemoryKind::Episode, 1.0, 100, None),
            mem("new", MemoryKind::Episode, 1.0, 200, None),
        ]);
        let p = build_profile(&s, "tag", &ProfileParams::default()).unwrap();
        assert_eq!(p.dynamics[0].id, "new");
        assert_eq!(p.dynamics[1].id, "old");
    }

    #[test]
    fn respects_section_limits() {
        let s = store_with(&[
            mem("f1", MemoryKind::Fact, 0.9, 100, None),
            mem("f2", MemoryKind::Fact, 0.8, 101, None),
            mem("f3", MemoryKind::Fact, 0.7, 102, None),
            mem("e1", MemoryKind::Episode, 1.0, 100, None),
            mem("e2", MemoryKind::Episode, 1.0, 101, None),
        ]);
        let params = ProfileParams {
            max_static: 2,
            max_dynamic: 1,
            scan_cap: 4096,
        };
        let p = build_profile(&s, "tag", &params).unwrap();
        assert_eq!(p.statics.len(), 2);
        assert_eq!(p.statics[0].id, "f1"); // strongest kept
        assert_eq!(p.dynamics.len(), 1);
    }

    #[test]
    fn excludes_expired() {
        let s = store_with(&[
            mem("live", MemoryKind::Fact, 1.0, 100, None),
            mem("dead", MemoryKind::Fact, 1.0, 100, Some(1)),
        ]);
        let p = build_profile(&s, "tag", &ProfileParams::default()).unwrap();
        assert_eq!(p.statics.len(), 1);
        assert_eq!(p.statics[0].id, "live");
    }

    #[test]
    fn cache_serves_then_invalidates() {
        let mut s = store_with(&[mem("f1", MemoryKind::Fact, 1.0, 100, None)]);
        let cache = ProfileCache::with_defaults();

        let first = cache.get_or_build(&s, "tag").unwrap();
        assert_eq!(first.len(), 1);
        assert_eq!(cache.len(), 1);

        // Write a new memory; the cache still serves the stale profile...
        s.upsert(&mem("f2", MemoryKind::Fact, 1.0, 101, None))
            .unwrap();
        let cached = cache.get_or_build(&s, "tag").unwrap();
        assert_eq!(cached.len(), 1, "should still be the cached profile");

        // ...until the tag is invalidated, then it rebuilds.
        cache.invalidate("tag");
        let rebuilt = cache.get_or_build(&s, "tag").unwrap();
        assert_eq!(rebuilt.len(), 2);
    }
}
