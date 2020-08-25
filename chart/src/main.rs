mod adapters;

use adapters::{ContrieTable, DashMapTable, FlurryTable, ShardTable};
use bustle::*;
use std::thread::sleep;
use std::time::Duration;

#[macro_use]
extern crate tracing;
extern crate tracing_serde;

fn gc_cycle() {
    sleep(Duration::from_millis(20000));
    let mut new_guard = crossbeam_epoch::pin();
    new_guard.flush();
    for _ in 0..32 {
        new_guard.repin();
    }
    let mut old_guard = crossbeam_epoch_old::pin();
    old_guard.flush();

    for _ in 0..32 {
        old_guard.repin();
    }
}

fn bench<W, T: Collection>(work: W, kind: &'static str)
where
    <T::Handle as CollectionHandle>::Key: Send + std::fmt::Debug,
    W: Fn(usize) -> Workload,
{
    // we could probably just pass a friendly name in
    // instead
    for n in 1..num_cpus::get() {
        let span = info_span!("kind", kind = kind);
        let _guard = span.enter();

        let _res = work(n).run::<T>();
        gc_cycle();
    }
}

// Currently this will run four concurrent HashMap implementations through
// Workload::run, 5x each over each Mix preset, for 0..num_cpus.
// For a 10 thread CPU that's 4 * 5 * 4 * 10 = 800 runs, I think.
// The output is a hacked JSON formatted trace which can be used for analysis.
fn main() {
    tracing_subscriber::fmt().json().flatten_event(true).init();

    const NUM_SEEDS: usize = 5;

    let workloads = vec![
        ("read_heavy", Mix::read_heavy()),
        ("write_heavy", Mix::insert_heavy()),
        ("update_heavy", Mix::update_heavy()),
        ("uniform", Mix::uniform()),
    ];

    for (task, mix) in workloads {
        let span = info_span!("task", task = task);
        let _guard = span.enter();

        let work = |n: usize| -> Workload { Workload::new(n, mix) };

        // random seed is used in each run
        // not sure what impact it has, but it probably doesn't hurt to
        // run this a handful of times..
        for trial_num in 0..NUM_SEEDS {
            let span = info_span!("trial_num", trial_num = trial_num);
            let _guard = span.enter();
            bench::<_, ContrieTable<u64>>(work, "contrie");
            bench::<_, DashMapTable<u64>>(work, "dashmap");
            bench::<_, ShardTable<u64>>(work, "sharded");
            bench::<_, FlurryTable>(work, "flurry");
            // seems like these are slow outliers
            // bench::<_, CHashMapTable<u64>>(work);
            // bench::<_, MutexStdTable<u64>>(work);
        }
    }
}
