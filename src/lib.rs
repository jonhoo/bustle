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
//! To run the benchmark, just implement [`Collection`] for your collection (`CollectionHandle` may
//! end up just being a call to `clone`), build a [`Workload`], and call [`Workload::run`]
//! parameterized by your collection type. You may want to look at the benchmarks for
//! lock-protected collections from the standard library in `benches/` for inspiration.
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
    rustdoc::broken_intra_doc_links
)]

use std::{sync::Arc, sync::Barrier, time::Duration};

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
    initial_cap_log2: u8,

    /// The fraction of the initial table capacity should we populate before running the benchmark.
    prefill_f: f64,

    /// Total number of operations as a multiple of the initial capacity.
    ops_f: f64,

    /// Number of threads to run the benchmark with.
    threads: usize,

    /// Random seed to randomize the workload.
    ///
    /// If `None`, the seed is picked randomly.
    /// If `Some`, the workload is deterministic if `threads == 1`.
    seed: Option<[u8; 32]>,
}

/// A collection that can be benchmarked by bustle.
///
/// Any thread that performs operations on the collection will first call `pin` and then perform
/// collection operations on the `Handle` that is returned. `pin` will not be called in the hot
/// loop of the benchmark.
pub trait Collection: Send + Sync + 'static {
    /// A thread-local handle to the concurrent collection under test.
    type Handle: CollectionHandle;

    /// Allocate a new instance of the benchmark target with the given capacity.
    fn with_capacity(capacity: usize) -> Self;

    /// Pin a thread-local handle to the concurrent collection under test.
    fn pin(&self) -> Self::Handle;
}

/// A handle to a key-value collection.
///
/// Note that for all these methods, the benchmarker does not dictate what the values are. Feel
/// free to use the same value for all operations, or use distinct ones and check that your
/// retrievals indeed return the right results.
pub trait CollectionHandle {
    /// The `u64` seeds used to construct `Key` (through `From<u64>`) are distinct.
    /// The returned keys must be as well.
    type Key: From<u64>;

    /// Perform a lookup for `key`.
    ///
    /// Should return `true` if the key is found.
    fn get(&mut self, key: &Self::Key) -> bool;

    /// Insert `key` into the collection.
    ///
    /// Should return `true` if no value previously existed for the key.
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

/// Information about a measurement.
#[derive(Debug, Clone)]
pub struct Measurement {
    /// A seed used for the run.
    pub seed: [u8; 32],
    /// A total number of operations.
    pub total_ops: u64,
    /// Spent time.
    pub spent: Duration,
    /// A number of operations per second.
    pub throughput: f64,
    /// An average value of latency.
    pub latency: Duration,
}

impl Workload {
    /// Start building a new benchmark workload.
    pub fn new(threads: usize, mix: Mix) -> Self {
        Self {
            mix,
            initial_cap_log2: 25,
            prefill_f: 0.0,
            ops_f: 0.75,
            threads,
            seed: None,
        }
    }

    /// Set the initial capacity for the map.
    ///
    /// Note that the capacity will be `2^` the given capacity!
    ///
    /// The number of operations and the number of pre-filled keys are determined based on the
    /// computed initial capacity, so keep that in mind if you change this parameter.
    ///
    /// Defaults to 25 (so `2^25 ~= 34M`).
    pub fn initial_capacity_log2(&mut self, capacity: u8) -> &mut Self {
        self.initial_cap_log2 = capacity;
        self
    }

    /// Set the fraction of the initial table capacity we should populate before running the
    /// benchmark.
    ///
    /// Defaults to 0%.
    pub fn prefill_fraction(&mut self, fraction: f64) -> &mut Self {
        assert!(fraction >= 0.0);
        assert!(fraction <= 1.0);
        self.prefill_f = fraction;
        self
    }

    /// Set the number of operations to run as a multiple of the initial capacity.
    ///
    /// This value can exceed 1.0.
    ///
    /// Defaults to 0.75 (75%).
    pub fn operations(&mut self, multiple: f64) -> &mut Self {
        assert!(multiple >= 0.0);
        self.ops_f = multiple;
        self
    }

    /// Set the seed used to randomize the workload.
    ///
    /// The seed does _not_ dictate thread interleaving, so you will only observe the exact same
    /// workload if you run the benchmark with `nthreads == 1`.
    pub fn seed(&mut self, seed: [u8; 32]) -> &mut Self {
        self.seed = Some(seed);
        self
    }

    /// Execute this workload against the collection type given by `T`.
    ///
    /// The key type must be `Send` since we generate the keys on a different thread than the one
    /// we do the benchmarks on.
    ///
    /// The key type must be `Debug` so that we can print meaningful errors if an assertion is
    /// violated during the benchmark.
    ///
    /// Returns the seed used for the run.
    pub fn run<T: Collection>(&self) -> [u8; 32]
    where
        <T::Handle as CollectionHandle>::Key: Send + std::fmt::Debug,
    {
        let m = self.run_silently::<T>();

        // TODO: do more with this information
        // TODO: collect statistics per operation type
        eprintln!(
            "{} operations across {} thread(s) in {:?}; time/op = {:?}",
            m.total_ops,
            self.threads,
            m.spent,
            m.spent / m.total_ops as u32,
        );

        m.seed
    }

    /// Execute this workload against the collection type given by `T`.
    ///
    /// The key type must be `Send` since we generate the keys on a different thread than the one
    /// we do the benchmarks on.
    ///
    /// The key type must be `Debug` so that we can print meaningful errors if an assertion is
    /// violated during the benchmark.
    #[allow(clippy::cognitive_complexity)]
    pub fn run_silently<T: Collection>(&self) -> Measurement
    where
        <T::Handle as CollectionHandle>::Key: Send + std::fmt::Debug,
    {
        assert_eq!(
            self.mix.read + self.mix.insert + self.mix.remove + self.mix.update + self.mix.upsert,
            100,
            "mix fractions do not add up to 100%"
        );

        let initial_capacity = 1 << self.initial_cap_log2;
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
        let insert_keys_per_thread =
            ((insert_keys + self.threads - 1) / self.threads).next_power_of_two();
        let mut generators = Vec::new();
        for _ in 0..self.threads {
            let mut thread_seed = [0u8; 32];
            rng.fill_bytes(&mut thread_seed[..]);
            generators.push(std::thread::spawn(move || {
                let mut rng: rand::rngs::SmallRng = rand::SeedableRng::from_seed(thread_seed);
                let mut keys: Vec<<T::Handle as CollectionHandle>::Key> =
                    Vec::with_capacity(insert_keys_per_thread);
                keys.extend((0..insert_keys_per_thread).map(|_| rng.next_u64().into()));
                keys
            }));
        }
        let keys: Vec<_> = generators
            .into_iter()
            .map(|jh| jh.join().unwrap())
            .collect();

        info!("constructing initial table");
        let table = Arc::new(T::with_capacity(initial_capacity));

        // And fill it
        let prefill_per_thread = prefill / self.threads;
        let mut prefillers = Vec::new();
        for keys in keys {
            let table = Arc::clone(&table);
            prefillers.push(std::thread::spawn(move || {
                let mut table = table.pin();
                for key in &keys[0..prefill_per_thread] {
                    let inserted = table.insert(key);
                    assert!(inserted);
                }
                keys
            }));
        }
        let keys: Vec<_> = prefillers
            .into_iter()
            .map(|jh| jh.join().unwrap())
            .collect();

        info!("start workload mix");
        let ops_per_thread = total_ops / self.threads;
        let op_mix = Arc::new(op_mix.into_boxed_slice());
        let barrier = Arc::new(Barrier::new(self.threads + 1));
        let mut mix_threads = Vec::with_capacity(self.threads);
        for keys in keys {
            let table = Arc::clone(&table);
            let op_mix = Arc::clone(&op_mix);
            let barrier = Arc::clone(&barrier);
            mix_threads.push(std::thread::spawn(move || {
                let mut table = table.pin();
                mix(
                    &mut table,
                    &keys,
                    &op_mix,
                    ops_per_thread,
                    prefill_per_thread,
                    barrier,
                )
            }));
        }

        barrier.wait();
        let start = std::time::Instant::now();
        barrier.wait();
        let spent = start.elapsed();

        let _samples: Vec<_> = mix_threads
            .into_iter()
            .map(|jh| jh.join().unwrap())
            .collect();

        let avg = spent / total_ops as u32;
        info!(?spent, ops = total_ops, ?avg, "workload mix finished");

        let total_ops = total_ops as u64;
        let threads = self.threads as u32;

        Measurement {
            seed,
            total_ops,
            spent,
            throughput: total_ops as f64 / spent.as_secs_f64(),
            latency: Duration::from_nanos((spent * threads).as_nanos() as u64 / total_ops),
        }
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

fn mix<H: CollectionHandle>(
    tbl: &mut H,
    keys: &[H::Key],
    op_mix: &[Operation],
    ops: usize,
    prefilled: usize,
    barrier: Arc<Barrier>,
) where
    H::Key: std::fmt::Debug,
{
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

    // The elapsed time is measured by the lifetime of `workload_scope`.
    let workload_scope = scopeguard::guard(barrier, |barrier| {
        barrier.wait();
    });
    workload_scope.wait();

    for (i, op) in (0..((ops + op_mix.len() - 1) / op_mix.len()))
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
                let new_key = tbl.insert(&keys[insert_seq]);
                assert!(
                    new_key,
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

                let _inserted = tbl.insert(&keys[n]);
                if n == insert_seq {
                    insert_seq += 1;
                }
            }
        }
    }
}
