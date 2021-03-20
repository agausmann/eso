use eso::puberty::Program;
use std::io::{stdin, stdout, Read, Write};
use std::process::exit;
use std::{env, fs};

fn usage<R>() -> R {
    eprintln!(
        "Usage: {} SCRIPT-FILE",
        env::args().next().unwrap_or("puberty".into())
    );
    exit(1);
}

fn error<E: std::fmt::Display, R>(error: E) -> R {
    eprintln!("{}", error);
    exit(1);
}

fn main() {
    let script_path = env::args_os().nth(1).unwrap_or_else(usage);
    let script = fs::read_to_string(script_path).unwrap_or_else(error);
    let program: Program = script.parse().unwrap_or_else(error);
    let input = || stdin().bytes().next().transpose().unwrap_or_else(error);
    let output = |byte| stdout().write_all(&[byte]).unwrap_or_else(error);
    program.run(input, output).unwrap_or_else(error);
}
