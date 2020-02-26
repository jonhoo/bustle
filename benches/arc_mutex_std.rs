use bustle::*;
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct Std<K>(Arc<Mutex<HashMap<K, ()>>>);

impl<K> BenchmarkTarget for Std<K>
where
    K: Send + From<u64> + Copy + 'static + Hash + Eq + std::fmt::Debug,
{
    type Key = K;

    fn with_capacity(capacity: usize) -> Self {
        Self(Arc::new(Mutex::new(HashMap::with_capacity(capacity))))
    }

    fn get(&mut self, key: &Self::Key) -> bool {
        self.0.lock().unwrap().get(key).is_some()
    }

    fn insert(&mut self, key: &Self::Key) -> bool {
        self.0.lock().unwrap().insert(*key, ()).is_some()
    }

    fn remove(&mut self, key: &Self::Key) -> bool {
        self.0.lock().unwrap().remove(key).is_some()
    }

    fn update(&mut self, key: &Self::Key) -> bool {
        use std::collections::hash_map::Entry;
        let mut map = self.0.lock().unwrap();
        if let Entry::Occupied(mut e) = map.entry(*key) {
            e.insert(());
            true
        } else {
            false
        }
    }
}

fn main() {
    tracing_subscriber::fmt::init();
    Workload::new(1, Mix::read_heavy()).run::<Std<u64>>();
}
