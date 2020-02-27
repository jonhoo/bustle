[![Crates.io](https://img.shields.io/crates/v/bustle.svg)](https://crates.io/crates/bustle)
[![Documentation](https://docs.rs/bustle/badge.svg)](https://docs.rs/bustle/)
[![Build Status](https://dev.azure.com/jonhoo/jonhoo/_apis/build/status/bustle?branchName=master)](https://dev.azure.com/jonhoo/jonhoo/_build/latest?definitionId=18&branchName=master)

Bustle is a benchmarking harness for concurrent key-value collections.

Say you have a concurrent collection (like a `HashMap`) and you want to measure how well it
performs across different workloads. Does it collapse when there are many writes? Or when there
are many threads? Or if there are concurrent removals? This crate tries to give you answers.

`bustle` runs a concurrent mix of operations (a "workload") against your collection, measuring
statistics as it goes, and gives you a report at the end about how you did. There are many
parameters to tweak, but hopefully the documentation for each element will help you decide. You
probably want to measure your collection against many different workloads, rather than just a
single one.

See [the documentation](https://docs.rs/bustle) for details.

The crate is, at the time of writing, a pretty direct port of the [Universal Benchmark] from
`libcuckoo`, though that may change over time.

If you have a concurrent key-value collection you would like to run
through this benchmark, feel free to send a PR to add it as a benchmark
to this repository!

## License

Licensed under Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
licensed as above, without any additional terms or conditions.

  [Universal Benchmark]: https://github.com/efficient/libcuckoo/tree/master/tests/universal-benchmark
