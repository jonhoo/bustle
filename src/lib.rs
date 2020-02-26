//! A benchmarking harness for concurrent key-value collections.
//!
//! Say you have a concurrent collection (like a `HashMap`) and you want to measure how well it
//! performs across different workloads. Does it collapse when there are many writes? Or when there
//! are many threads? Or if there are concurrent removals? This crate tries to give you answers.
//!
//! `bustle` runs a concurrent mix of operations (a "workload") against your collection, measuring
//! statistics as it goes, and gives you a report at the end about how you did. There are many
//! parameters to tweak, but hopefully the documentation for each element will help you decide. You
//! probably want to measure your collection against many different workloads, rather than just a
//! single one.
//!
//! To run the benchmark, just implement [`BenchmarkTarget`] for your collection, build a
//! [`Workload`], and call [`Workload::run`] parameterized by your collection type. You may want to
//! look at the benchmarks for lock-protected collections from the standard library in `benches/`
//! for inspiration.
//!
//! The crate is, at the time of writing, a pretty direct port of the [Universal Benchmark] from
//! `libcuckoo`, though that may change over time.
//!
//!   [Universal Benchmark]: https://github.com/efficient/libcuckoo/tree/master/tests/universal-benchmark
#![deny(missing_docs)]
#![warn(
    rust_2018_idioms,
    missing_debug_implementations,
    unreachable_pub,
    intra_doc_link_resolution_failure
)]

use rand::prelude::*;
use tracing::{debug, info, info_span};

/// A workload mix configration.
///
/// The sum of the fields must add to 100.
#[derive(Clone, Copy, Debug)]
pub struct Mix {
    /// The percentage of operations in the mix that are reads.
    pub read: u8,
    /// The percentage of operations in the mix that are inserts.
    pub insert: u8,
    /// The percentage of operations in the mix that are removals.
    pub remove: u8,
    /// The percentage of operations in the mix that are updates.
    pub update: u8,
    /// The percentage of operations in the mix that are update-or-inserts.
    pub upsert: u8,
}

impl Mix {
    /// Constructs a very read-heavy workload (~95%), with limited concurrent modifications.
    pub fn read_heavy() -> Self {
        Self {
            read: 94,
            insert: 2,
            update: 3,
            remove: 1,
            upsert: 0,
        }
    }

    /// Constructs a very insert-heavy workload (~80%), with some reads and updates.
    pub fn insert_heavy() -> Self {
        Self {
            read: 10,
            insert: 80,
            update: 10,
            remove: 0,
            upsert: 0,
        }
    }

    /// Constructs a very update-heavy workload (~50%), with some other modifications and the rest
    /// reads.
    pub fn update_heavy() -> Self {
        Self {
            read: 35,
            insert: 5,
            update: 50,
            remove: 5,
            upsert: 5,
        }
    }

    /// Constructs a workload where all operations occur with equal probability.
    pub fn uniform() -> Self {
        Self {
            read: 20,
            insert: 20,
            update: 20,
            remove: 20,
            upsert: 20,
        }
    }
}

/// A benchmark workload builder.
#[derive(Clone, Copy, Debug)]
pub struct Workload {
    /// The mix of operations to run.
    mix: Mix,

    /// The initial capacity of the table, specified as a power of 2.
    ///
    /// Defaults to 21 (so `2^21 ~= 2M`).
    initial_cap_e: u8,

    /// The percentage of the initial table capacity should we fill the table to before running the
    /// benchmark.
    ///
    /// Defaults to 0%.
    prefill_f: f64,

    /// Total number of operations we are running, specified as a percentage of the initial
    /// capacity.
    ///
    /// This value can exceed 100.
    ///
    /// Defaults to 75%.
    ops_f: f64,

    /// Number of threads to run the benchmark with.
    threads: usize,

    /// Random seed to randomize the workload.
    ///
    /// If `None`, the seed is picked randomly.
    /// If `Some`, the workload is deterministic if `threads == 1`.
    seed: Option<[u8; 16]>,
}

/// A collection that can be benchmarked by bustle.
///
/// Note that for all these methods, the benchmarker does not dictate what the values are. Feel
/// free to use the same value for all operations, or use distinct ones and check that your
/// retrievals indeed return the right results.
pub trait BenchmarkTarget: Clone + Send + 'static {
    /// The `u64` seeds used to construct `Key` (through `From<u64>`) are distinct.
    /// The returned keys must be as well.
    ///
    /// The keys are required to be `Eq + Hash` so that the check for distinctness is relatively
    /// efficient.
    type Key: Send + From<u64> + std::fmt::Debug + Eq + std::hash::Hash;

    /// Allocate a new instance of the benchmark target with the given capacity.
    fn with_capacity(capacity: usize) -> Self;

    /// Perform a lookup for `key`.
    ///
    /// Should return `true` if the key is found.
    fn get(&mut self, key: &Self::Key) -> bool;

    /// Insert `key` into the collection.
    ///
    /// Should return `true` if a value was replaced by the operation.
    fn insert(&mut self, key: &Self::Key) -> bool;

    /// Remove `key` from the collection.
    ///
    /// Should return `true` if the key existed and was removed.
    fn remove(&mut self, key: &Self::Key) -> bool;

    /// Update the value for `key` in the collection, if it exists.
    ///
    /// Should return `true` if the key existed and was updated.
    ///
    /// Should **not** insert the key if it did not exist.
    fn update(&mut self, key: &Self::Key) -> bool;
}

impl Workload {
    /// Start building a new benchmark workload.
    pub fn new(threads: usize, mix: Mix) -> Self {
        Self {
            mix,
            initial_cap_e: 21,
            prefill_f: 0.0,
            ops_f: 75.0,
            threads,
            seed: None,
        }
    }

    /// Execute this workload against the collection type given by `T`.
    pub fn run<T: BenchmarkTarget>(&self) {
        assert_eq!(
            self.mix.read + self.mix.insert + self.mix.remove + self.mix.update + self.mix.upsert,
            100,
            "mix fractions do not add up to 100%"
        );
        assert!(self.prefill_f <= 100.0);

        let initial_capacity = 1 << self.initial_cap_e;
        let total_ops = (initial_capacity as f64 * self.ops_f) as usize;

        let seed = self.seed.unwrap_or_else(rand::random);
        let mut rng: rand::rngs::SmallRng = rand::SeedableRng::from_seed(seed);

        // NOTE: it'd be nice to include std::intrinsics::type_name::<T> here
        let span = info_span!("benchmark", mix = ?self.mix, threads = self.threads);
        let _guard = span.enter();
        debug!(initial_capacity, total_ops, ?seed, "workload parameters");

        info!("generating operation mix");
        let mut op_mix = Vec::with_capacity(100);
        op_mix.append(&mut vec![Operation::Read; usize::from(self.mix.read)]);
        op_mix.append(&mut vec![Operation::Insert; usize::from(self.mix.insert)]);
        op_mix.append(&mut vec![Operation::Remove; usize::from(self.mix.remove)]);
        op_mix.append(&mut vec![Operation::Update; usize::from(self.mix.update)]);
        op_mix.append(&mut vec![Operation::Upsert; usize::from(self.mix.upsert)]);
        op_mix.shuffle(&mut rng);

        info!("generating key space");
        let prefill = (initial_capacity as f64 * self.prefill_f) as usize;
        // We won't be running through `op_mix` more than ceil(total_ops / 100), so calculate that
        // ceiling and multiply by the number of inserts and upserts to get an upper bound on how
        // many elements we'll be inserting.
        let max_insert_ops =
            (total_ops + 99) / 100 * usize::from(self.mix.insert + self.mix.upsert);
        let insert_keys = std::cmp::max(initial_capacity, max_insert_ops) + prefill;
        // Round this quantity up to a power of 2, so that we can use an LCG to cycle over the
        // array "randomly".
        let insert_keys_per_thread = insert_keys.next_power_of_two();
        let mut generators = Vec::new();
        for _ in 0..self.threads {
            let mut thread_seed = [0u8; 16];
            rng.fill_bytes(&mut thread_seed[..]);
            generators.push(std::thread::spawn(move || {
                let mut rng: rand::rngs::SmallRng = rand::SeedableRng::from_seed(thread_seed);
                let mut keys: Vec<T::Key> = Vec::with_capacity(insert_keys_per_thread);
                keys.extend((0..insert_keys_per_thread).map(|_| rng.next_u64().into()));
                keys
            }));
        }
        let keys: Vec<_> = generators
            .into_iter()
            .map(|jh| jh.join().unwrap())
            .collect();

        // Check that the seeds are indeed distinct.
        // This _should_ be the case, but we double-check in case of broken From<u64> impls.
        // Otherwise, the user will just see really weird errors in the benchmarking phase.
        {
            debug!("checking that key space is distinct");
            let mut distinct = std::collections::HashSet::with_capacity(keys.len());
            for key in &keys {
                assert!(distinct.insert(key));
            }
        }

        info!("constructing initial table");
        let table = T::with_capacity(initial_capacity);
        let tables: Vec<_> = std::iter::repeat(table).take(self.threads).collect();

        // And fill it
        let prefill_per_thread = prefill / self.threads;
        let mut prefillers = Vec::new();
        for (mut table, keys) in tables.into_iter().zip(keys) {
            prefillers.push(std::thread::spawn(move || {
                for i in 0..prefill_per_thread {
                    let inserted = table.insert(&keys[i]);
                    assert!(inserted);
                }
                (table, keys)
            }));
        }
        let thread_state: Vec<_> = prefillers
            .into_iter()
            .map(|jh| jh.join().unwrap())
            .collect();

        info!("start workload mix");
        let ops_per_thread = total_ops / self.threads;
        let op_mix: &'static [_] = Box::leak(op_mix.into_boxed_slice());
        let start = std::time::Instant::now();
        let mut mix_threads = Vec::with_capacity(self.threads);
        for (mut table, keys) in thread_state {
            mix_threads.push(std::thread::spawn(move || {
                mix(
                    &mut table,
                    &keys,
                    op_mix,
                    ops_per_thread,
                    prefill_per_thread,
                )
            }));
        }
        let _samples: Vec<_> = mix_threads
            .into_iter()
            .map(|jh| jh.join().unwrap())
            .collect();
        let took = start.elapsed();

        info!(?took, ops = total_ops, "workload mix finished");

        // TODO: do more with this information
        // TODO: collect statistics per operation type
        eprintln!("{} operations in {:?}", total_ops, took);
        eprintln!("avg: {:?}", took / total_ops as u32);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Operation {
    Read,
    Insert,
    Remove,
    Update,
    Upsert,
}

fn mix<T: BenchmarkTarget>(
    tbl: &mut T,
    keys: &[T::Key],
    op_mix: &'static [Operation],
    ops: usize,
    prefilled: usize,
) {
    // Invariant: erase_seq <= insert_seq
    // Invariant: insert_seq < numkeys
    let nkeys = keys.len();
    let mut erase_seq = 0;
    let mut insert_seq = prefilled;
    let mut find_seq = 0;

    // We're going to use a very simple LCG to pick random keys.
    // We want it to be _super_ fast so it doesn't add any overhead.
    assert!(nkeys.is_power_of_two());
    assert!(nkeys > 4);
    assert_eq!(op_mix.len(), 100);
    let a = nkeys / 2 + 1;
    let c = nkeys / 4 - 1;
    let find_seq_mask = nkeys - 1;

    for (i, op) in (0..(ops / op_mix.len()))
        .flat_map(|_| op_mix.iter())
        .enumerate()
    {
        if i == ops {
            break;
        }

        match op {
            Operation::Read => {
                let should_find = find_seq >= erase_seq && find_seq < insert_seq;
                let found = tbl.get(&keys[find_seq]);
                if find_seq >= erase_seq {
                    assert_eq!(
                        should_find, found,
                        "get({:?}) {} {} {}",
                        &keys[find_seq], find_seq, erase_seq, insert_seq
                    );
                } else {
                    // due to upserts, we may _or may not_ find the key
                }

                // Twist the LCG since we used find_seq
                find_seq = (a * find_seq + c) & find_seq_mask;
            }
            Operation::Insert => {
                let replaced = tbl.insert(&keys[insert_seq]);
                assert!(
                    !replaced,
                    "insert({:?}) should insert a new value",
                    &keys[insert_seq]
                );
                insert_seq += 1;
            }
            Operation::Remove => {
                if erase_seq == insert_seq {
                    // If `erase_seq` == `insert_eq`, the table should be empty.
                    let removed = tbl.remove(&keys[find_seq]);
                    assert!(
                        !removed,
                        "remove({:?}) succeeded on empty table",
                        &keys[find_seq]
                    );

                    // Twist the LCG since we used find_seq
                    find_seq = (a * find_seq + c) & find_seq_mask;
                } else {
                    let removed = tbl.remove(&keys[erase_seq]);
                    assert!(removed, "remove({:?}) should succeed", &keys[erase_seq]);
                    erase_seq += 1;
                }
            }
            Operation::Update => {
                // Same as find, except we update to the same default value
                let should_exist = find_seq >= erase_seq && find_seq < insert_seq;
                let updated = tbl.update(&keys[find_seq]);
                if find_seq >= erase_seq {
                    assert_eq!(should_exist, updated, "update({:?})", &keys[find_seq]);
                } else {
                    // due to upserts, we may or may not have updated an existing key
                }

                // Twist the LCG since we used find_seq
                find_seq = (a * find_seq + c) & find_seq_mask;
            }
            Operation::Upsert => {
                // Pick a number from the full distribution, but cap it to the insert_seq, so we
                // don't insert a number greater than insert_seq.
                let n = std::cmp::min(find_seq, insert_seq);

                // Twist the LCG since we used find_seq
                find_seq = (a * find_seq + c) & find_seq_mask;

                let _inserted = !tbl.insert(&keys[n]);
                if n == insert_seq {
                    insert_seq += 1;
                }
            }
        }
    }
}
