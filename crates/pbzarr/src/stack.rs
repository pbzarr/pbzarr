//! Batch stack: combine N single-sample stores into one cohort store.
//!
//! Each shared scalar track becomes a `(ΣL, N)` cohort track. Implemented by
//! reading each source's scalar track through a [`PbzTrackReader`] and driving
//! them into the cohort track with the existing [`run_pipeline`], one track at
//! a time. Growing an existing cohort (incremental sample append) is a separate
//! concern and not handled here.

use std::marker::PhantomData;
use std::sync::Arc;

use ndarray::ArrayViewMut2;

use crate::genome::{Genome, Region};
use crate::import::{Config, Report, run_pipeline};
use crate::io::error::Result as IoResult;
use crate::io::{Dtype, Numeric, ReaderError, ValueReader};
use crate::{PbzError, PbzStore, Result, Track, TrackConfig};

/// A `ValueReader` over one scalar track of an already-written pbz store. Used
/// as a stack source: each `read_into` reads the track's region and copies it
/// into the caller's single-column buffer.
pub struct PbzTrackReader<T> {
    store: Arc<PbzStore>,
    track: String,
    genome: Arc<Genome>,
    _marker: PhantomData<T>,
}

impl<T: Numeric> PbzTrackReader<T> {
    fn new(store: Arc<PbzStore>, track: &str) -> Result<Self> {
        let genome = {
            let t = store.track(track).ok_or_else(|| {
                PbzError::Store(format!("stack: track {track:?} missing in source"))
            })?;
            Arc::clone(t.genome())
        };
        Ok(Self {
            store,
            track: track.to_owned(),
            genome,
            _marker: PhantomData,
        })
    }
}

impl<T: Numeric> ValueReader for PbzTrackReader<T> {
    type Item = T;

    fn contigs(&self) -> &Genome {
        &self.genome
    }

    fn n_fields(&self) -> usize {
        1
    }

    fn read_into(
        &self,
        contig_name: &str,
        start: u64,
        end: u64,
        mut dst: ArrayViewMut2<'_, Self::Item>,
    ) -> IoResult<()> {
        if end <= start {
            return Ok(());
        }
        let cid = self.genome.id(contig_name).ok_or_else(|| {
            ReaderError::Other(anyhow::anyhow!(
                "stack: contig {contig_name:?} not in source track {:?}",
                self.track
            ))
        })?;
        let track = self.store.track(&self.track).ok_or_else(|| {
            ReaderError::Other(anyhow::anyhow!(
                "stack: source track {:?} vanished",
                self.track
            ))
        })?;
        let region = Region {
            contig: cid,
            start,
            end,
        };
        let vals = track
            .read_region::<T>(&region)
            .map_err(|e| ReaderError::Other(anyhow::anyhow!("{e}")))?;
        let vals = vals
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|e| ReaderError::Other(anyhow::anyhow!("stack: source rank: {e}")))?;
        dst.slice_mut(ndarray::s![.., 0]).assign(&vals);
        Ok(())
    }

    fn fork(&self) -> IoResult<Self> {
        Ok(Self {
            store: Arc::clone(&self.store),
            track: self.track.clone(),
            genome: Arc::clone(&self.genome),
            _marker: PhantomData,
        })
    }
}

/// Configuration for [`stack`].
pub struct StackConfig {
    /// Tracks to stack. `None` stacks every track of the first source; each
    /// selected track must exist (scalar, same dtype, same genome) in all sources.
    pub tracks: Option<Vec<String>>,
    /// Cohort column-axis name (default `"sample"`).
    pub column_dim: Option<String>,
    /// Sample-axis chunk width (default: full width `N`).
    pub column_chunk_size: Option<usize>,
    /// Worker threads.
    pub workers: usize,
}

impl Default for StackConfig {
    fn default() -> Self {
        Self {
            tracks: None,
            column_dim: None,
            column_chunk_size: None,
            workers: 4,
        }
    }
}

/// Combine `sources` (opened single-sample stores + labels) into a fresh cohort
/// store `out`. Every stacked track must be scalar (rank 1), share a dtype, and
/// share a genome (`genome_checksum`) across all sources. Each becomes a
/// `(ΣL, N)` cohort track whose column labels are the source labels.
pub fn stack(
    sources: Vec<(PbzStore, String)>,
    out: &mut PbzStore,
    config: StackConfig,
) -> Result<Report> {
    if sources.is_empty() {
        return Err(PbzError::Metadata("stack: no sources".into()));
    }
    let n = sources.len();
    let mut stores: Vec<Arc<PbzStore>> = Vec::with_capacity(n);
    let mut labels: Vec<String> = Vec::with_capacity(n);
    for (store, label) in sources {
        stores.push(Arc::new(store));
        labels.push(label);
    }

    let track_names: Vec<String> = match &config.tracks {
        Some(ts) => ts.clone(),
        None => stores[0].track_names().map(|s| s.to_owned()).collect(),
    };
    if track_names.is_empty() {
        return Err(PbzError::Metadata("stack: no tracks to stack".into()));
    }

    let dim = config.column_dim.clone().unwrap_or_else(|| "sample".into());
    let column_chunk = config.column_chunk_size.unwrap_or(n);

    let mut bytes_written = 0u64;
    let mut tasks_completed = 0usize;
    let mut contigs_written = 0usize;

    for tname in &track_names {
        let t0 = stores[0].track(tname).ok_or_else(|| {
            PbzError::Metadata(format!(
                "stack: source {:?} has no track {tname:?}",
                labels[0]
            ))
        })?;
        if t0.rank() != 1 {
            return Err(PbzError::Metadata(format!(
                "stack: track {tname:?} in {:?} is not scalar (rank {})",
                labels[0],
                t0.rank()
            )));
        }
        let dtype = t0.dtype();
        let checksum = t0.genome().checksum();
        let genome = t0.genome().as_ref().clone();
        let pos_chunk = t0.chunk_size()?;

        for i in 1..n {
            let ti = stores[i].track(tname).ok_or_else(|| {
                PbzError::Metadata(format!(
                    "stack: source {:?} has no track {tname:?}",
                    labels[i]
                ))
            })?;
            if ti.rank() != 1 {
                return Err(PbzError::Metadata(format!(
                    "stack: track {tname:?} in {:?} is not scalar",
                    labels[i]
                )));
            }
            if ti.dtype() != dtype {
                return Err(PbzError::Metadata(format!(
                    "stack: track {tname:?} dtype {} in {:?} != {} in {:?}",
                    ti.dtype(),
                    labels[i],
                    dtype,
                    labels[0]
                )));
            }
            if ti.genome().checksum() != checksum {
                return Err(PbzError::Metadata(format!(
                    "stack: track {tname:?} genome in {:?} differs from {:?}",
                    labels[i], labels[0]
                )));
            }
        }

        let cfg = TrackConfig::new(dtype)
            .columns(labels.clone())
            .column_dim(dim.clone())
            .column_chunk_size(column_chunk)
            .chunk_size(pos_chunk);
        let pcfg = Config {
            workers: config.workers,
            ..Config::default()
        };
        let report = out.create_tracks_with(vec![(tname.clone(), genome, cfg)], |tracks| {
            dispatch(dtype, &stores, tname, tracks[0], &pcfg)
        })?;
        bytes_written += report.bytes_written;
        tasks_completed += report.tasks_completed;
        contigs_written = report.contigs_written;
    }

    Ok(Report {
        contigs_written,
        bytes_written,
        tasks_completed,
    })
}

fn dispatch(
    dtype: Dtype,
    stores: &[Arc<PbzStore>],
    track: &str,
    cohort: &Track,
    cfg: &Config,
) -> Result<Report> {
    match dtype {
        Dtype::U8 => stack_one::<u8>(stores, track, cohort, cfg),
        Dtype::U16 => stack_one::<u16>(stores, track, cohort, cfg),
        Dtype::U32 => stack_one::<u32>(stores, track, cohort, cfg),
        Dtype::I8 => stack_one::<i8>(stores, track, cohort, cfg),
        Dtype::I16 => stack_one::<i16>(stores, track, cohort, cfg),
        Dtype::I32 => stack_one::<i32>(stores, track, cohort, cfg),
        Dtype::F32 => stack_one::<f32>(stores, track, cohort, cfg),
        Dtype::F64 => stack_one::<f64>(stores, track, cohort, cfg),
        Dtype::Bool => stack_one::<bool>(stores, track, cohort, cfg),
    }
}

fn stack_one<T: Numeric>(
    stores: &[Arc<PbzStore>],
    track: &str,
    cohort: &Track,
    cfg: &Config,
) -> Result<Report> {
    let readers: Vec<PbzTrackReader<T>> = stores
        .iter()
        .map(|s| PbzTrackReader::<T>::new(Arc::clone(s), track))
        .collect::<Result<_>>()?;
    run_pipeline::<T, _>(cohort, readers, cfg)
}
