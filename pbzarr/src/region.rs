use std::fmt;
use std::str::FromStr;

use crate::error::{PbzError, Result};

/// A 0-based, half-open genomic region.
///
/// Represents a query against a PBZ store. All coordinates follow BED conventions:
/// positions are 0-based and intervals are half-open `[start, end)`.
///
/// Can be parsed from strings via [`FromStr`] / [`.parse()`](str::parse):
///
/// ```
/// use pbzarr::Region;
///
/// let r: Region = "chr1:1000-2000".parse().unwrap();
/// assert_eq!(r.contig, "chr1");
/// assert_eq!(r.start, Some(1000));
/// assert_eq!(r.end, Some(2000));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Region {
    pub contig: String,
    /// Start position (0-based, inclusive). `None` means start of contig.
    pub start: Option<u64>,
    /// End position (0-based, exclusive). `None` means end of contig.
    pub end: Option<u64>,
}

impl FromStr for Region {
    type Err = PbzError;

    /// Parse a region from a string.
    ///
    /// Accepted forms:
    /// - `"chr1"` — entire contig
    /// - `"chr1:1000-2000"` — half-open range `[1000, 2000)`
    /// - `"chr1:1000"` — single position
    ///
    /// Commas in numeric values are stripped (e.g. `chr1:1,000,000-2,000,000`).
    /// Whitespace is trimmed.
    fn from_str(s: &str) -> Result<Self> {
        let s = s.trim();

        if s.is_empty() {
            return Err(PbzError::InvalidRegion {
                message: "empty region string".to_string(),
            });
        }

        let (contig, coords) = match s.rsplit_once(':') {
            Some((c, coords)) => {
                if c.is_empty() {
                    return Err(PbzError::InvalidRegion {
                        message: "missing contig name".to_string(),
                    });
                }
                (c, Some(coords))
            }
            None => (s, None),
        };

        let contig = contig.to_string();

        let Some(coords) = coords else {
            return Ok(Region {
                contig,
                start: None,
                end: None,
            });
        };

        // Strip commas from coordinate part for thousands-separator support
        let coords = coords.replace(',', "");

        if coords.is_empty() {
            return Ok(Region {
                contig,
                start: None,
                end: None,
            });
        }

        match coords.split_once('-') {
            Some((start_str, end_str)) => {
                let start = parse_position(start_str, "start")?;
                let end = parse_position(end_str, "end")?;

                if start >= end {
                    return Err(PbzError::InvalidRegion {
                        message: format!("start ({start}) must be less than end ({end})"),
                    });
                }

                Ok(Region {
                    contig,
                    start: Some(start),
                    end: Some(end),
                })
            }
            None => {
                let pos = parse_position(&coords, "position")?;
                Ok(Region {
                    contig,
                    start: Some(pos),
                    end: None,
                })
            }
        }
    }
}

impl fmt::Display for Region {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.start, self.end) {
            (Some(s), Some(e)) => write!(f, "{}:{}-{}", self.contig, s, e),
            (Some(s), None) => write!(f, "{}:{}", self.contig, s),
            _ => write!(f, "{}", self.contig),
        }
    }
}

/// Parse a region string into a [`Region`].
///
/// Convenience wrapper around [`Region::from_str`]. See its documentation for
/// accepted formats.
pub fn parse_region(region: &str) -> Result<Region> {
    region.parse()
}

fn parse_position(s: &str, label: &str) -> Result<u64> {
    s.parse::<u64>().map_err(|_| PbzError::InvalidRegion {
        message: format!("invalid {label}: {s:?}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contig_only() {
        let r = parse_region("chr1").unwrap();
        assert_eq!(r.contig, "chr1");
        assert_eq!(r.start, None);
        assert_eq!(r.end, None);
    }

    #[test]
    fn contig_with_trailing_colon() {
        let r = parse_region("chr1:").unwrap();
        assert_eq!(r.contig, "chr1");
        assert_eq!(r.start, None);
        assert_eq!(r.end, None);
    }

    #[test]
    fn range() {
        let r = parse_region("chr1:1000-2000").unwrap();
        assert_eq!(r.contig, "chr1");
        assert_eq!(r.start, Some(1000));
        assert_eq!(r.end, Some(2000));
    }

    #[test]
    fn single_position() {
        let r = parse_region("chr1:5000").unwrap();
        assert_eq!(r.contig, "chr1");
        assert_eq!(r.start, Some(5000));
        assert_eq!(r.end, None);
    }

    #[test]
    fn comma_separated_numbers() {
        let r = parse_region("chr1:1,000,000-2,000,000").unwrap();
        assert_eq!(r.contig, "chr1");
        assert_eq!(r.start, Some(1_000_000));
        assert_eq!(r.end, Some(2_000_000));
    }

    #[test]
    fn whitespace_trimming() {
        let r = parse_region("  chr1:100-200  ").unwrap();
        assert_eq!(r.contig, "chr1");
        assert_eq!(r.start, Some(100));
        assert_eq!(r.end, Some(200));
    }

    #[test]
    fn zero_start() {
        let r = parse_region("chr1:0-1000").unwrap();
        assert_eq!(r.contig, "chr1");
        assert_eq!(r.start, Some(0));
        assert_eq!(r.end, Some(1000));
    }

    #[test]
    fn err_empty() {
        assert!(parse_region("").is_err());
        assert!(parse_region("   ").is_err());
    }

    #[test]
    fn err_missing_contig() {
        assert!(parse_region(":1000-2000").is_err());
    }

    #[test]
    fn err_invalid_number() {
        assert!(parse_region("chr1:abc-2000").is_err());
        assert!(parse_region("chr1:1000-xyz").is_err());
        assert!(parse_region("chr1:abc").is_err());
    }

    #[test]
    fn err_inverted_range() {
        assert!(parse_region("chr1:2000-1000").is_err());
    }

    #[test]
    fn err_equal_start_end() {
        assert!(parse_region("chr1:1000-1000").is_err());
    }

    #[test]
    fn from_str_trait() {
        let r: Region = "chr1:100-200".parse().unwrap();
        assert_eq!(r.contig, "chr1");
        assert_eq!(r.start, Some(100));
        assert_eq!(r.end, Some(200));
    }

    #[test]
    fn display_range() {
        let r = Region {
            contig: "chr1".into(),
            start: Some(1000),
            end: Some(2000),
        };
        assert_eq!(r.to_string(), "chr1:1000-2000");
    }

    #[test]
    fn display_single_position() {
        let r = Region {
            contig: "chr1".into(),
            start: Some(5000),
            end: None,
        };
        assert_eq!(r.to_string(), "chr1:5000");
    }

    #[test]
    fn display_contig_only() {
        let r = Region {
            contig: "chr1".into(),
            start: None,
            end: None,
        };
        assert_eq!(r.to_string(), "chr1");
    }

    #[test]
    fn display_round_trip() {
        let cases = ["chr1", "chr1:1000", "chr1:1000-2000"];
        for input in cases {
            let r: Region = input.parse().unwrap();
            assert_eq!(r.to_string(), input);
        }
    }
}
