use std::collections::HashMap;
use std::num::Wrapping;
use std::ops::RangeBounds;
use thiserror::Error;

#[derive(Debug)]
pub struct Program {
    code: Code,
    initial_registers: PrivateRegisters,
}

impl Program {
    pub fn parse(code: &str) -> Result<Self, ParseError> {
        Parser::new(code).parse()
    }

    pub fn spawn<I, O>(&self, input: I, output: O) -> Runtime<I, O>
    where
        I: FnMut() -> Option<u8>,
        O: FnMut(u8),
    {
        Runtime::new(self, input, output)
    }

    pub fn run<I, O>(&self, input: I, output: O) -> Result<(), RuntimeError>
    where
        I: FnMut() -> Option<u8>,
        O: FnMut(u8),
    {
        self.spawn(input, output).run()
    }
}

impl std::str::FromStr for Program {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

#[derive(Debug, Error)]
#[error("{line}:{column} Syntax error: {message}")]
pub struct ParseError {
    line: usize,
    column: usize,
    message: String,
}

impl ParseError {
    fn new(message: String, parser: &Parser) -> Self {
        let (line, column) = parser.location();
        Self {
            line,
            column,
            message,
        }
    }
    fn expected(expected: &[&str], parser: &Parser) -> Self {
        if let &[expected] = expected {
            Self::new(
                format!("Expected {:?} near {:?}", expected, parser.context()),
                parser,
            )
        } else {
            Self::new(
                format!("Expected one of {:?} near {:?}", expected, parser.context()),
                parser,
            )
        }
    }
    fn pronoun_mismatch(expected: &str, got: &str, parser: &Parser) -> Self {
        Self::new(
            format!(
                "What are you, a bigot? (Expected {:?}, got {:?})",
                expected, got
            ),
            parser,
        )
    }

    fn no_kinks(parser: &Parser) -> Self {
        Self::new(format!("Expected at least one kink"), parser)
    }

    fn kink_plurality(plural: bool, kinks: usize, parser: &Parser) -> Self {
        if plural {
            Self::new(
                format!(
                    "Expected more than one kink (due to \"kinks are\"), got {}",
                    kinks
                ),
                parser,
            )
        } else {
            Self::new(
                format!(
                    "Expected exactly one kink (due to \"kink is\"), got {}",
                    kinks
                ),
                parser,
            )
        }
    }

    fn unclosed_comment(parser: &Parser) -> Self {
        //TODO more context
        Self::new(format!("Unclosed comment block"), parser)
    }

    fn loop_mismatch(parser: &Parser) -> Self {
        //TODO more context
        Self::new(format!("Loop mismatch"), parser)
    }

    fn invalid_command(word: &str, parser: &Parser) -> Self {
        Self::new(format!("Invalid command {:?}", word), parser)
    }

    fn expected_whitespace(parser: &Parser) -> Self {
        Self::new(
            format!("Expected whitespace near {:?}", parser.context()),
            parser,
        )
    }

    fn eof(parser: &Parser) -> Self {
        Self::new(format!("Unexpected EOF"), parser)
    }

    fn out_of_range<R>(num: u64, range: R, parser: &Parser) -> Self
    where
        R: RangeBounds<u64> + std::fmt::Debug,
    {
        Self::new(
            format!("Value {} outside expected range {:?}", num, range),
            parser,
        )
    }
}

#[derive(Debug, Error)]
#[error("Runtime error: {message}")]
pub struct RuntimeError {
    message: String,
    registers: Registers,
}

impl RuntimeError {
    fn new<I, O>(message: String, runtime: &Runtime<I, O>) -> Self
    where
        I: FnMut() -> Option<u8>,
        O: FnMut(u8),
    {
        Self {
            message,
            registers: runtime.registers(),
        }
    }

    fn halted<I, O>(runtime: &Runtime<I, O>) -> Self
    where
        I: FnMut() -> Option<u8>,
        O: FnMut(u8),
    {
        Self::new(format!("Tried to run while halted"), runtime)
    }

    fn cooldown<I, O>(runtime: &Runtime<I, O>) -> Self
    where
        I: FnMut() -> Option<u8>,
        O: FnMut(u8),
    {
        Self::new(format!("Tried to use kink during cooldown"), runtime)
    }

    pub fn registers(&self) -> &Registers {
        &self.registers
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub struct Registers {
    pub pc: usize,
    pub regptr: usize,
    pub a: u8,
    pub b: u8,
    pub c: u8,
    pub d: u8,
    pub e: u8,
    pub f: u8,
}

pub struct Runtime<'a, I, O> {
    program: &'a Program,
    pc: usize,
    registers: PrivateRegisters,
    regptr: usize,
    cooldowns: Vec<usize>,
    halted: bool,
    input: I,
    output: O,
}

impl<'a, I, O> Runtime<'a, I, O>
where
    I: FnMut() -> Option<u8>,
    O: FnMut(u8),
{
    pub fn new(program: &'a Program, input: I, output: O) -> Self {
        Self {
            program,
            pc: 0,
            registers: program.initial_registers,
            regptr: 0,
            cooldowns: Vec::new(),
            input,
            output,
            halted: false,
        }
    }

    pub fn halted(&self) -> bool {
        self.halted
    }

    pub fn program(&self) -> &'a Program {
        self.program
    }

    pub fn registers(&self) -> Registers {
        Registers {
            pc: self.pc,
            regptr: self.regptr,
            a: self.registers[0].0,
            b: self.registers[1].0,
            c: self.registers[2].0,
            d: self.registers[3].0,
            e: self.registers[4].0,
            f: self.registers[5].0,
        }
    }

    pub fn step(&mut self) -> Result<(), RuntimeError> {
        if self.halted {
            return Err(RuntimeError::halted(self));
        }

        if self.pc >= self.program.code.len() {
            self.halt();
            return Ok(());
        }

        let command = self.program.code[self.pc];
        self.pc += 1;
        match command {
            Command::Fap => {
                self.registers[self.regptr] += Wrapping(1);
                for cooldown in &mut self.cooldowns {
                    if *cooldown > 0 {
                        *cooldown -= 1;
                    }
                }
            }
            Command::Ugh => {
                self.registers[self.regptr] = Wrapping(0);
            }
            Command::Fuck(kink) => {
                let cooldown = self.cooldown(kink);
                if *cooldown == 0 {
                    *cooldown = 2usize.pow(kink - 2);
                    self.registers[self.regptr] += Wrapping(kink as u8);
                } else {
                    self.halt();
                    return Err(RuntimeError::cooldown(self));
                }
            }
            Command::Hnng(kink) => {
                let cooldown = self.cooldown(kink);
                if *cooldown == 0 {
                    *cooldown = 2usize.pow(kink - 1);
                    self.registers[self.regptr] += Wrapping(2u8.pow(kink));
                } else {
                    self.halt();
                    return Err(RuntimeError::cooldown(self));
                }
            }
            Command::Yeah => {
                self.regptr = (self.regptr + 1) % 6;
            }
            Command::Yes => {
                (self.output)(self.registers[self.regptr].0);
            }
            Command::Oh => match (self.input)() {
                Some(input) => {
                    self.registers[self.regptr] = Wrapping(input);
                }
                None => {
                    self.halt();
                    return Ok(());
                }
            },
            Command::Sigh | Command::OMGMOMGETOUTTAHERE => {
                self.halt();
            }
            Command::Squirt => {
                for i in 0..6 {
                    (self.output)(self.registers[(self.regptr + i) % 6].0);
                }
            }
            Command::Hrg(mmf) => {
                if self.registers[self.regptr].0 == 0 {
                    self.pc = mmf + 1;
                }
            }
            Command::Mmf(hrg) => {
                self.pc = hrg;
            }
        }
        Ok(())
    }

    pub fn run(&mut self) -> Result<(), RuntimeError> {
        while !self.halted {
            self.step()?;
        }
        Ok(())
    }

    fn cooldown(&mut self, kink: u32) -> &mut usize {
        let index = (kink - 2) as usize;
        if index >= self.cooldowns.len() {
            self.cooldowns.resize(index + 1, 0);
        }
        &mut self.cooldowns[index]
    }

    fn halt(&mut self) {
        self.halted = true;
    }
}

type Code = Vec<Command>;
type PrivateRegisters = [Wrapping<u8>; 6];
struct Introduction {
    kink_values: HashMap<String, u32>,
    initial_registers: PrivateRegisters,
}

#[derive(Debug, Clone, Copy)]
enum Command {
    Fap,
    Ugh,
    Fuck(u32),
    Hnng(u32),
    Yeah,
    Yes,
    Oh,
    Sigh,
    OMGMOMGETOUTTAHERE,
    Squirt,
    Hrg(usize),
    Mmf(usize),
}

/// Recursive descent parser
struct Parser<'a> {
    full_input: &'a str,
    input: &'a str,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            full_input: input,
            input,
        }
    }

    fn parse(mut self) -> Result<Program, ParseError> {
        let Introduction {
            kink_values,
            initial_registers,
        } = self.introduction()?;
        let code = self.code(&kink_values)?;
        Ok(Program {
            code,
            initial_registers,
        })
    }

    fn introduction(&mut self) -> Result<Introduction, ParseError> {
        // It is August 15, 2018, 04:32:06 PM.
        self.trim();
        self.words("It is")?;
        let month = match self.word()? {
            "January" => 1,
            "February" => 2,
            "March" => 3,
            "April" => 4,
            "May" => 5,
            "June" => 6,
            "July" => 7,
            "August" => 8,
            "September" => 9,
            "October" => 10,
            "November" => 11,
            "December" => 12,
            _ => {
                return Err(ParseError::expected(
                    &[
                        "January",
                        "February",
                        "March",
                        "April",
                        "May",
                        "June",
                        "July",
                        "August",
                        "September",
                        "October",
                        "November",
                        "December",
                    ],
                    self,
                ))
            }
        };
        let day = self.int(1..=MONTH_DAYS_MAX[(month - 1) as usize])?;
        let day_suffix = self
            .word()?
            .strip_suffix(",")
            .ok_or_else(|| ParseError::expected(&[","], self))?;
        match day_suffix {
            "st" | "nd" | "rd" | "th" | "" => {}
            _ => return Err(ParseError::expected(&["st", "nd", "rd", "th", ""], self)),
        }
        let year = self.int(..)?;
        self.words(",")?;
        let hour_12h = self.int(0..=12)?;
        self.take(":")?;
        let minute = self.int(0..=59)?;
        self.take(":")?;
        let second = self.int(0..=59)?;
        self.trim();
        let hour_24h = match self.word()? {
            "AM." => hour_12h,
            "PM." => hour_12h + 12,
            _ => return Err(ParseError::expected(&["AM.", "PM."], self)),
        };

        // Izu is in his bed, bored.
        let character = self.word()?;
        self.words("is in")?;
        let pronoun = self.possessive(None)?;
        self.words("bed, bored.")?;

        // His secret kinks are fatfurs, inflation, growth and kitsunes.
        self.cap_possessive(Some(pronoun))?;
        self.words("secret")?;
        let kink = self.word()?;
        let is = self.word()?;
        let kink_plurality = match (kink, is) {
            ("kink", "is") => false,
            ("kinks", "are") => true,
            _ => return Err(ParseError::expected(&["kink is", "kinks are"], self)),
        };

        let kinks: Vec<&'a str> = self
            .until(".")?
            .split("and")
            .flat_map(|ands| ands.split(","))
            .map(|kink| kink.trim())
            .collect();
        self.trim();

        if kinks.is_empty() {
            return Err(ParseError::no_kinks(self));
        }
        if kink_plurality ^ (kinks.len() > 1) {
            return Err(ParseError::kink_plurality(
                kink_plurality,
                kinks.len(),
                self,
            ));
        }

        let kink_values: HashMap<String, u32> = kinks
            .iter()
            .enumerate()
            .map(|(i, &kink)| (kink.to_lowercase(), (kinks.len() + 1 - i) as u32))
            .collect();

        // Suddenly he spots fatfurs.
        let spotted = match self.word()? {
            "Suddenly" | "Then" => {
                if self.try_parse(|this| this.pronoun(Some(pronoun))).is_ok() {
                    match pronoun {
                        Pronoun::He | Pronoun::She => self.words("spots")?,
                        Pronoun::They => self.words("spot")?,
                    }

                    let kink = self
                        .word()?
                        .strip_suffix(".")
                        .ok_or_else::<ParseError, _>(|| ParseError::expected(&["."], self))?;

                    // Start of next sentence
                    match self.word()? {
                        "Suddenly" | "Then" | "Soon" => {}
                        _ => return Err(ParseError::expected(&["Suddenly", "Then", "Soon"], self)),
                    }

                    Some(kink)
                } else {
                    None
                }
            }
            "Soon" => None,
            _ => return Err(ParseError::expected(&["Suddenly", "Then", "Soon"], self)),
        };

        // (Then) the following sounds become audible.
        self.words("the following sounds become audible.")?;

        let a = match spotted {
            Some(kink) => 2u8.pow(kink_values[kink] as u32),
            None => 0,
        };
        let b = 0;
        let c = character.len() as u8;
        let d = unix_seconds(year, month, day, hour_24h, minute, second) as u8;
        let e = 0;
        let f = 0;

        let initial_registers = [
            Wrapping(a),
            Wrapping(b),
            Wrapping(c),
            Wrapping(d),
            Wrapping(e),
            Wrapping(f),
        ];

        Ok(Introduction {
            kink_values,
            initial_registers,
        })
    }

    fn pronoun(&mut self, expected: Option<Pronoun>) -> Result<Pronoun, ParseError> {
        let word = self.word()?;
        match word {
            "he" => Ok(Pronoun::He),
            "she" => Ok(Pronoun::She),
            "they" => Ok(Pronoun::They),
            _ => {
                if let Some(expected) = expected {
                    Err(ParseError::pronoun_mismatch(
                        expected.personal(),
                        word,
                        self,
                    ))
                } else {
                    Err(ParseError::expected(&["he", "she", "they"], self))
                }
            }
        }
    }

    fn possessive(&mut self, expected: Option<Pronoun>) -> Result<Pronoun, ParseError> {
        let word = self.word()?;
        match word {
            "his" => Ok(Pronoun::He),
            "her" => Ok(Pronoun::She),
            "their" => Ok(Pronoun::They),
            _ => {
                if let Some(expected) = expected {
                    Err(ParseError::pronoun_mismatch(
                        expected.possessive(),
                        word,
                        self,
                    ))
                } else {
                    Err(ParseError::expected(&["his", "her", "their"], self))
                }
            }
        }
    }

    fn cap_possessive(&mut self, expected: Option<Pronoun>) -> Result<Pronoun, ParseError> {
        let word = self.word()?;
        match word {
            "His" => Ok(Pronoun::He),
            "Her" => Ok(Pronoun::She),
            "Their" => Ok(Pronoun::They),
            _ => {
                if let Some(expected) = expected {
                    Err(ParseError::pronoun_mismatch(
                        expected.cap_possessive(),
                        word,
                        self,
                    ))
                } else {
                    Err(ParseError::expected(&["His", "Her", "Their"], self))
                }
            }
        }
    }

    fn code(&mut self, kink_values: &HashMap<String, u32>) -> Result<Code, ParseError> {
        let mut code = Vec::new();
        let mut hrgs = Vec::new();
        let mut words = self.input.split_whitespace();

        while let Some(word) = words.next() {
            //XXX lots of allocations here, maybe case-convert the entire input all at once
            // if this becomes a problem, but this is cleaner.
            let lower_word = word.to_lowercase();

            match lower_word.as_str() {
                "fap" => code.push(Command::Fap),
                "ugh" => code.push(Command::Ugh),
                "yeah" => code.push(Command::Yeah),
                "yes" => code.push(Command::Yes),
                "oh" => code.push(Command::Oh),
                "sigh" => code.push(Command::Sigh),
                "squirt" => code.push(Command::Squirt),
                "ngh" => {
                    // Skip comment contents
                    loop {
                        match words.next() {
                            Some("hhh") => break,
                            None => {
                                return Err(ParseError::unclosed_comment(self));
                            }
                            _ => {}
                        }
                    }
                }
                "hrg" => {
                    hrgs.push(code.len());
                    // Zero is a placeholder for the pointer to mmf -
                    // it is populated when the matching mmf is found.
                    code.push(Command::Hrg(0));
                }
                "mmf" => match hrgs.pop() {
                    Some(hrg) => {
                        code[hrg] = Command::Hrg(code.len());
                        code.push(Command::Mmf(hrg));
                    }
                    None => return Err(ParseError::loop_mismatch(self)),
                },
                _ => {
                    if let Some(kink) = lower_word.strip_suffix(",fuck") {
                        code.push(Command::Fuck(kink_values[kink]));
                    } else if let Some(kink) = lower_word.strip_suffix(",hnng") {
                        code.push(Command::Hnng(kink_values[kink]));
                    } else if word == "OMGMOMGETOUTTAHERE" {
                        // special case - OMGMOMGETOUTTAHERE is case-sensitive
                        code.push(Command::OMGMOMGETOUTTAHERE);
                    } else {
                        return Err(ParseError::invalid_command(word, self));
                    }
                }
            }
        }

        if !hrgs.is_empty() {
            return Err(ParseError::loop_mismatch(self));
        }

        //TODO update state as you go - doing it like this means that errors only generate with the
        // line/column of the start of code.
        self.input = "";
        Ok(code)
    }

    fn try_parse<F, R>(&mut self, func: F) -> Result<R, ParseError>
    where
        F: FnOnce(&mut Self) -> Result<R, ParseError>,
    {
        let saved_input = self.input;
        let result = func(self);
        if result.is_err() {
            self.input = saved_input;
        }
        result
    }

    fn trim(&mut self) {
        self.input = self.input.trim_start();
    }

    fn trim_force(&mut self) -> Result<(), ParseError> {
        match self.input.chars().next() {
            // Input is empty
            None => Ok(()),

            // Input starts with whitespace
            Some(c) if c.is_whitespace() => {
                self.trim();
                Ok(())
            }

            _ => Err(ParseError::expected_whitespace(self)),
        }
    }

    fn take(&mut self, pattern: &str) -> Result<(), ParseError> {
        if let Some(after) = self.input.strip_prefix(pattern) {
            self.input = after;
            Ok(())
        } else {
            Err(ParseError::expected(&[pattern], self))
        }
    }

    fn until(&mut self, pattern: &str) -> Result<&'a str, ParseError> {
        if let Some((before, after)) = self.input.split_once(pattern) {
            self.input = after;
            Ok(before)
        } else {
            Err(ParseError::expected(&[pattern], self))
        }
    }

    fn words(&mut self, pattern: &str) -> Result<(), ParseError> {
        self.try_parse(|this| {
            for word in pattern.split_whitespace() {
                this.take(word)?;
                this.trim_force()?;
            }
            Ok(())
        })
    }

    fn word(&mut self) -> Result<&'a str, ParseError> {
        if self.input.is_empty() {
            Err(ParseError::eof(self))
        } else {
            let word_end = self
                .input
                .char_indices()
                .filter_map(|(i, c)| c.is_whitespace().then(|| i))
                .next()
                .unwrap_or(self.input.len());
            let word = &self.input[..word_end];
            self.input = &self.input[word_end..];
            self.trim();
            Ok(word)
        }
    }

    fn int<R>(&mut self, range: R) -> Result<u64, ParseError>
    where
        R: RangeBounds<u64> + std::fmt::Debug,
    {
        if self.input.is_empty() {
            Err(ParseError::eof(self))
        } else {
            let int_end = self
                .input
                .char_indices()
                .filter_map(|(i, c)| (!c.is_ascii_digit()).then(|| i))
                .next()
                .unwrap_or(self.input.len());
            let num = self.input[..int_end].parse().unwrap();
            if range.contains(&num) {
                self.input = &self.input[int_end..];
                Ok(num)
            } else {
                Err(ParseError::out_of_range(num, range, self))
            }
        }
    }

    fn context(&self) -> &'a str {
        self.input.split_whitespace().next().unwrap_or("")
    }

    fn location(&self) -> (usize, usize) {
        let mut line = 1;
        let mut column = 1;
        let byte = self.full_input.len() - self.input.len();
        for (i, c) in self.full_input.char_indices() {
            if c == '\n' {
                line += 1;
                column = 1;
            } else {
                column += c.len_utf8();
            }
            if i == byte {
                break;
            }
        }
        (line, column)
    }
}

#[derive(Copy, Clone, PartialEq)]
enum Pronoun {
    He,
    She,
    They,
}

impl Pronoun {
    fn personal(self) -> &'static str {
        match self {
            Self::He => "he",
            Self::She => "she",
            Self::They => "they",
        }
    }

    fn possessive(self) -> &'static str {
        match self {
            Self::He => "his",
            Self::She => "her",
            Self::They => "their",
        }
    }

    fn cap_possessive(self) -> &'static str {
        match self {
            Self::He => "His",
            Self::She => "Her",
            Self::They => "Their",
        }
    }
}

const MONTH_DAYS_MIN: [u64; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
const MONTH_DAYS_MAX: [u64; 12] = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

fn unix_seconds(year: u64, month: u64, day: u64, hour: u64, minute: u64, second: u64) -> u64 {
    // Convert day/month to zero-indexed.
    let day = day - 1;
    let month = month - 1;

    fn is_leap_year(year: u64) -> bool {
        year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
    }

    let previous_leap_years = (1970..year).filter(|&yr| is_leap_year(yr)).count() as u64;
    let year_offset_days = 365 * (year - 1970) + previous_leap_years;

    let leap_day_passed = if is_leap_year(year) && month > 1 {
        1
    } else {
        0
    };
    let month_offset_days = MONTH_DAYS_MIN[..month as usize]
        .iter()
        .copied()
        .sum::<u64>()
        + leap_day_passed;

    second + 60 * (minute + 60 * (hour + 24 * (day + month_offset_days + year_offset_days)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_program(code: &str, input: &[u8], expected_output: &[u8]) {
        let program: Program = code.parse().unwrap();
        let mut input = input.iter().copied();
        let mut output = Vec::new();
        program
            .spawn(|| input.next(), |b| output.push(b))
            .run()
            .unwrap();
        assert_eq!(input.next(), None, "Input overflow");
        assert_eq!(output, expected_output, "Output mismatch");
    }

    #[test]
    fn unix_epoch() {
        assert_eq!(unix_seconds(1970, 1, 1, 0, 0, 0), 0);
    }

    #[test]
    fn today() {
        assert_eq!(unix_seconds(2021, 3, 18, 20, 12, 57), 1616098377);
    }

    #[test]
    fn hello_world() {
        assert_program(include_str!("helloworld.fap"), b"", b"Hello world\n");
    }

    #[test]
    fn cat() {
        assert_program(include_str!("cat.fap"), b"foo bar baz", b"foo bar baz");
    }

    #[test]
    fn truthmachine() {
        assert_program(include_str!("truthmachine.fap"), b"0", b"0");
    }
}
