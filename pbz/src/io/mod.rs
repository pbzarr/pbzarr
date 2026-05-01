pub mod bed_error;
pub mod bed_reader;
pub mod bed_writer;
pub mod bedgraph_writer;
pub mod d4_reader;
pub mod interval_to_per_base;
pub mod tsv_writer;
pub mod value_reader;

use color_eyre::Result;
use color_eyre::eyre::eyre;
use pbzarr::Track;
use std::path::Path;

use crate::cli::ExportFormat;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Format {
    Tsv,
    Bedgraph,
    Bed,
}

pub fn pick_format(explicit: ExportFormat, output: Option<&Path>, track: &Track) -> Result<Format> {
    let chosen = match explicit {
        ExportFormat::Tsv => Format::Tsv,
        ExportFormat::Bedgraph => Format::Bedgraph,
        ExportFormat::Bed => Format::Bed,
        ExportFormat::Auto => match output.and_then(infer_from_extension) {
            Some(f) => f,
            None => Format::Tsv,
        },
    };
    validate_compatible(chosen, track)?;
    Ok(chosen)
}

pub fn infer_from_extension(p: &Path) -> Option<Format> {
    let name = p.file_name()?.to_str()?.to_ascii_lowercase();
    if name.ends_with(".bg") || name.ends_with(".bedgraph") {
        Some(Format::Bedgraph)
    } else if name.ends_with(".bed") {
        Some(Format::Bed)
    } else if name.ends_with(".tsv") || name.ends_with(".txt") {
        Some(Format::Tsv)
    } else {
        None
    }
}

pub fn validate_compatible(fmt: Format, track: &Track) -> Result<()> {
    match fmt {
        Format::Bedgraph => {
            if track.dtype() == "bool" {
                return Err(eyre!(
                    "bedgraph requires a numeric track; '{}' is bool. Try --format bed.",
                    track.name()
                ));
            }
            if track.has_samples() && track.samples().map(|c| c.len()).unwrap_or(0) > 1 {
                return Err(eyre!(
                    "bedgraph requires a single-column track; '{}' has {} columns. Try --format tsv or --column NAME.",
                    track.name(),
                    track.samples().map(|c| c.len()).unwrap_or(0)
                ));
            }
        }
        Format::Bed => {
            if track.dtype() != "bool" {
                return Err(eyre!(
                    "BED requires dtype=bool; '{}' is {}. Try --format bedgraph or --format tsv.",
                    track.name(),
                    track.dtype()
                ));
            }
        }
        Format::Tsv => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn extension_inference() {
        assert_eq!(
            infer_from_extension(&PathBuf::from("a.bg")),
            Some(Format::Bedgraph)
        );
        assert_eq!(
            infer_from_extension(&PathBuf::from("a.bedgraph")),
            Some(Format::Bedgraph)
        );
        assert_eq!(
            infer_from_extension(&PathBuf::from("a.bed")),
            Some(Format::Bed)
        );
        assert_eq!(
            infer_from_extension(&PathBuf::from("a.tsv")),
            Some(Format::Tsv)
        );
        assert_eq!(
            infer_from_extension(&PathBuf::from("a.txt")),
            Some(Format::Tsv)
        );
        assert_eq!(infer_from_extension(&PathBuf::from("a.unknown")), None);
        assert_eq!(infer_from_extension(&PathBuf::from("noext")), None);
    }
}
