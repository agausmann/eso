# eso

Random esoteric languages which I decide are funny enough to implement.

## Usage

The list of supported languages is provided below. The general convention is
that the language's input/output are connected to stdin/stdout, and you are
expected to write the script contents to a file and pass the file path as a
command-line argument.

- [PUBERTY](https://esolangs.org/wiki/PUBERTY) - `cargo run puberty SCRIPT-FILE`

## Goals

- Zero dependencies - I write everything myself. Some exceptions are made for
  QoL improvements, including `thiserror`/`anyhow` for error handling and
`rand` for RNG for probabilistic languages.

- Over all else, have fun!
