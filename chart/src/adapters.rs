// Credits to `xacrimon` for sending me this link which served as the original basis for this
// module: https://github.com/xacrimon/conc-map-bench/commit/fbed218dce28fa9f5976a3d50617d43024d05bfd.
//
// No license present at the time of access.

use bustle::*;
use chashmap::CHashMap;
use contrie::ConMap;
use dashmap::DashMap;
use fxhash::FxBuildHasher;
use sharded::Map;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct ShardTable<K>(Arc<Map<K, u32>>);

impl<K> Collection for ShardTable<K>
where
    K: Send + Sync + From<u64> + Copy + 'static + std::hash::Hash + Eq,
{
    type Handle = Self;
    fn with_capacity(capacity: usize) -> Self {
        Self(Arc::new(Map::with_capacity(capacity)))
    }

    fn pin(&self) -> Self::Handle {
        self.clone()
    }
}

impl<K> CollectionHandle for ShardTable<K>
where
    K: Send + From<u64> + Copy + 'static + std::hash::Hash + Eq,
{
    type Key = K;

    fn get(&mut self, key: &Self::Key) -> bool {
        let guard = self.0.read::<Self::Key>(key);
        (*guard).get(key).is_some()
    }

    fn insert(&mut self, key: &Self::Key) -> bool {
        self.0.write(key).insert(*key, 0).is_none()
    }

    fn remove(&mut self, key: &Self::Key) -> bool {
        self.0.write(key).remove(key).is_some()
    }

    fn update(&mut self, key: &Self::Key) -> bool {
        let mut map = self.0.write(key);
        map.get_mut(key).map(|v| *v += 1).is_some()
    }
}

#[derive(Clone)]
pub struct CHashMapTable<K>(Arc<CHashMap<K, u32>>);

impl<K> Collection for CHashMapTable<K>
where
    K: Send + Sync + From<u64> + Copy + 'static + std::hash::Hash + Eq + std::fmt::Debug,
{
    type Handle = Self;
    fn with_capacity(capacity: usize) -> Self {
        Self(Arc::new(CHashMap::with_capacity(capacity)))
    }

    fn pin(&self) -> Self::Handle {
        self.clone()
    }
}

impl<K> CollectionHandle for CHashMapTable<K>
where
    K: Send + Sync + From<u64> + Copy + 'static + std::hash::Hash + Eq + std::fmt::Debug,
{
    type Key = K;

    fn get(&mut self, key: &Self::Key) -> bool {
        self.0.get(key).is_some()
    }

    fn insert(&mut self, key: &Self::Key) -> bool {
        self.0.insert(*key, 0).is_none()
    }

    fn remove(&mut self, key: &Self::Key) -> bool {
        self.0.remove(key).is_some()
    }

    fn update(&mut self, key: &Self::Key) -> bool {
        self.0
            .get_mut(key)
            .map(|mut r| {
                *r += 1;
            })
            .is_some()
    }
}

#[derive(Clone)]
pub struct ContrieTable<K: Eq + std::hash::Hash + 'static>(Arc<ConMap<K, Mutex<u32>>>);

impl<K> Collection for ContrieTable<K>
where
    K: Send + Sync + From<u64> + Copy + 'static + std::hash::Hash + Eq + std::fmt::Debug,
{
    type Handle = Self;
    fn with_capacity(_: usize) -> Self {
        Self(Arc::new(ConMap::new()))
    }

    fn pin(&self) -> Self::Handle {
        self.clone()
    }
}

impl<K> CollectionHandle for ContrieTable<K>
where
    K: Send + Sync + From<u64> + Copy + 'static + std::hash::Hash + Eq + std::fmt::Debug,
{
    type Key = K;

    fn get(&mut self, key: &Self::Key) -> bool {
        self.0.get(key).is_some()
    }

    fn insert(&mut self, key: &Self::Key) -> bool {
        self.0.insert(*key, Mutex::new(0)).is_none()
    }

    fn remove(&mut self, key: &Self::Key) -> bool {
        self.0.remove(key).is_some()
    }

    fn update(&mut self, key: &Self::Key) -> bool {
        self.0
            .get(key)
            .map(|e| {
                *e.value().lock().unwrap() += 1;
            })
            .is_some()
    }
}

#[derive(Clone)]
pub struct FlurryTable(Arc<flurry::HashMap<u64, u32, FxBuildHasher>>);

pub struct FlurryHandle(flurry::HashMapRef<'static, u64, u32, FxBuildHasher>);

impl Collection for FlurryTable {
    type Handle = FlurryHandle;

    fn with_capacity(capacity: usize) -> Self {
        Self(Arc::new(flurry::HashMap::with_capacity_and_hasher(
            capacity,
            FxBuildHasher::default(),
        )))
    }

    fn pin(&self) -> Self::Handle {
        unsafe { std::mem::transmute(self.0.pin()) }
    }
}

impl CollectionHandle for FlurryHandle {
    type Key = u64;

    fn get(&mut self, key: &Self::Key) -> bool {
        self.0.get(key).is_some()
    }

    fn insert(&mut self, key: &Self::Key) -> bool {
        self.0.insert(*key, 0).is_none()
    }

    fn remove(&mut self, key: &Self::Key) -> bool {
        self.0.remove(key).is_some()
    }

    fn update(&mut self, key: &Self::Key) -> bool {
        self.0.compute_if_present(key, |_, v| Some(v + 1)).is_some()
    }
}

#[derive(Clone)]
pub struct DashMapTable<K>(Arc<DashMap<K, u32>>);

impl<K> Collection for DashMapTable<K>
where
    K: Send + Sync + From<u64> + Copy + 'static + std::hash::Hash + Eq + std::fmt::Debug,
{
    type Handle = Self;
    fn with_capacity(capacity: usize) -> Self {
        Self(Arc::new(DashMap::with_capacity(capacity)))
    }

    fn pin(&self) -> Self::Handle {
        self.clone()
    }
}

impl<K> CollectionHandle for DashMapTable<K>
where
    K: Send + Sync + From<u64> + Copy + 'static + std::hash::Hash + Eq + std::fmt::Debug,
{
    type Key = K;

    fn get(&mut self, key: &Self::Key) -> bool {
        self.0.get(key).is_some()
    }

    fn insert(&mut self, key: &Self::Key) -> bool {
        self.0.insert(*key, 0).is_none()
    }

    fn remove(&mut self, key: &Self::Key) -> bool {
        self.0.remove(key).is_some()
    }

    fn update(&mut self, key: &Self::Key) -> bool {
        self.0.get_mut(key).map(|mut v| *v += 1).is_some()
    }
}
