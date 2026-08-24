//! Shared text formatting for pbz commands.

use std::io::{self, Write};

/// C `printf("%g")`: `sig` significant digits, fixed or scientific notation
/// by magnitude, trailing zeros trimmed. Matches UCSC bigWigToBedGraph
/// output. Built on std's correctly rounded `{:.*e}` formatting.
pub(crate) fn write_g(out: &mut impl Write, value: f64, sig: usize) -> io::Result<()> {
    if value == 0.0 || !value.is_finite() {
        return write!(out, "{value}");
    }
    let sci = format!("{:.*e}", sig - 1, value);
    let (mantissa, exp) = sci.split_once('e').expect("std e-format");
    let exp: i32 = exp.parse().expect("std e-format exponent");
    let (sign, mantissa) = match mantissa.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", mantissa),
    };
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let digits = digits.trim_end_matches('0');
    let digits = if digits.is_empty() { "0" } else { digits };
    if exp >= -4 && exp < sig as i32 {
        if exp >= 0 {
            let split = (exp as usize) + 1;
            if split >= digits.len() {
                write!(out, "{sign}{digits}{}", "0".repeat(split - digits.len()))
            } else {
                write!(out, "{sign}{}.{}", &digits[..split], &digits[split..])
            }
        } else {
            write!(out, "{sign}0.{}{digits}", "0".repeat((-exp - 1) as usize))
        }
    } else {
        let (first, rest) = digits.split_at(1);
        if rest.is_empty() {
            write!(out, "{sign}{first}e{exp:+03}")
        } else {
            write!(out, "{sign}{first}.{rest}e{exp:+03}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g(value: f64, sig: usize) -> String {
        let mut buf = Vec::new();
        write_g(&mut buf, value, sig).unwrap();
        String::from_utf8(buf).unwrap()
    }

    #[test]
    fn write_g_matches_c_percent_g() {
        assert_eq!(g(0.018554688, 6), "0.0185547");
        assert_eq!(g(1.0, 6), "1");
        assert_eq!(g(-0.5, 6), "-0.5");
        assert_eq!(g(0.25, 6), "0.25");
        assert_eq!(g(1234567.0, 6), "1.23457e+06");
        assert_eq!(g(0.00001, 6), "1e-05");
        assert_eq!(g(123.456, 6), "123.456");
        assert_eq!(g(f64::NAN, 6), "NaN");
        assert_eq!(g(0.0, 6), "0");
    }
}
