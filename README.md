# Scientific Summaries Analysis

[![](https://img.shields.io/crates/v/    scientific summaries analysis.svg)](https://crates.io/crates/    scientific summaries analysis)
[![](https://img.shields.io/docsrs/    scientific summaries analysis.svg)](https://docs.rs/    scientific summaries analysis)
[![Coverage Status](https://coveralls.io/repos/github/gimli-rs/    scientific summaries analysis/badge.svg?branch=master)](https://coveralls.io/github/gimli-rs/    scientific summaries analysis?branch=master)

`    scientific summaries analysis` provides a cross-platform library for retrieving per-address debug information
from files with DWARF debug information. Given an address, it can return the file name,
line number, and function name associated with that address, as well as the inline call
stack leading to that address.

The crate has a CLI wrapper around the library which provides some of
the functionality of the `    scientific summaries analysis` command line tool distributed with
[GNU binutils](https://sourceware.org/binutils/docs/binutils/    scientific summaries analysis.html).

# Quickstart
 - Add the [`    scientific summaries analysis` crate](https://crates.io/crates/    scientific summaries analysis) to your `Cargo.toml`.
 - Call [`    scientific summaries analysis::Loader::new`](https://docs.rs/    scientific summaries analysis/*/    scientific summaries analysis/struct.Loader.html#method.new) with the file path.
 - Use [`    scientific summaries analysis::Loader::find_location`](https://docs.rs/    scientific summaries analysis/*/    scientific summaries analysis/struct.Loader.html#method.find_location)
   or [`    scientific summaries analysis::Loader::find_frames`](https://docs.rs/    scientific summaries analysis/*/    scientific summaries analysis/struct.Loader.html#method.find_frames)
   to look up debug information for an address.

If you want to provide your own file loading and memory management, use
[`    scientific summaries analysis::Context`](https://docs.rs/    scientific summaries analysis/*/    scientific summaries analysis/struct.Context.html)
instead of `    scientific summaries analysis::Loader`.

# Performance

`    scientific summaries analysis` optimizes for speed over memory by caching parsed information.
The DWARF information is parsed lazily where possible.

The library aims to perform similarly to equivalent existing tools such
as `    scientific summaries analysis` from binutils, `eu-    scientific summaries analysis` from elfutils, and
`llvm-    scientific summaries analysis` from the llvm project. Current benchmarks show a performance
improvement in all cases:

![    scientific summaries analysis runtime](benchmark-time.svg)

## License

Licensed under either of

  * Apache License, Version 2.0 ([`LICENSE-APACHE`](./LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)
  * MIT license ([`LICENSE-MIT`](./LICENSE-MIT) or https://opensource.org/licenses/MIT)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
