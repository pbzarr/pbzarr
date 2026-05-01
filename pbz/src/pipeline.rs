use crate::io::value_reader::{ValueChunk, ValueDtype, ValueReader};
use color_eyre::Result;
use color_eyre::eyre::eyre;
use crossbeam_channel::bounded;
use indicatif::ProgressBar;
use ndarray::{Array1, Array2};
use pbzarr::Track;
use std::sync::{Arc, Mutex};
use std::thread;

pub struct ImportPipeline {
    pub readers: Vec<Box<dyn ValueReader>>,
    pub track: Arc<Track>,
    pub contigs: Vec<(String, u64)>,
    pub chunk_size: u64,
    pub reader_workers: usize,
    pub writer_workers: usize,
    pub progress: Option<ProgressBar>,
    pub dtype: ValueDtype,
    pub has_samples: bool,
}

#[derive(Clone, Debug)]
struct ChunkTask {
    contig: String,
    contig_length: u64,
    chunk_idx: u64,
}

struct WriteTask {
    contig: String,
    chunk_idx: u64,
    payload: ChunkPayload,
}

// `Bool_1D` and `U32_1D` cover scalar tracks that the v1 D4 import doesn't
// produce (it's always columnar uint32). Kept here so future ingest paths
// (BED masks, methylation) can plug into the same pipeline without reshaping it.
#[allow(non_camel_case_types, dead_code)]
enum ChunkPayload {
    U32_2D(Array2<u32>),
    U32_1D(Array1<u32>),
    Bool_1D(Array1<bool>),
}

impl ImportPipeline {
    pub fn run(self) -> Result<()> {
        let total_chunks: u64 = self
            .contigs
            .iter()
            .map(|(_, len)| len.div_ceil(self.chunk_size))
            .sum();
        let mut tasks: Vec<ChunkTask> = Vec::with_capacity(total_chunks as usize);
        for (c, len) in &self.contigs {
            let n_chunks = len.div_ceil(self.chunk_size);
            for i in 0..n_chunks {
                tasks.push(ChunkTask {
                    contig: c.clone(),
                    contig_length: *len,
                    chunk_idx: i,
                });
            }
        }

        let task_cap = (self.reader_workers * 2).max(1);
        let write_cap = (self.writer_workers * 2).max(1);
        let (task_tx, task_rx) = bounded::<ChunkTask>(task_cap);
        let (write_tx, write_rx) = bounded::<WriteTask>(write_cap);

        // Wrap readers in Arc<Mutex> so multiple worker threads can lock them when reading.
        // Each reader holds its own D4 view state; one mutex per reader avoids global contention
        // across N inputs.
        let readers_locks: Vec<Arc<Mutex<Box<dyn ValueReader>>>> = self
            .readers
            .into_iter()
            .map(|r| Arc::new(Mutex::new(r)))
            .collect();

        let producer = {
            let task_tx = task_tx.clone();
            thread::spawn(move || -> Result<()> {
                for t in tasks {
                    if task_tx.send(t).is_err() {
                        break;
                    }
                }
                Ok(())
            })
        };
        drop(task_tx);

        let chunk_size = self.chunk_size;
        let dtype = self.dtype;
        let has_samples = self.has_samples;

        let mut reader_handles = Vec::with_capacity(self.reader_workers);
        for _ in 0..self.reader_workers {
            let task_rx = task_rx.clone();
            let write_tx = write_tx.clone();
            let readers_locks = readers_locks.clone();
            reader_handles.push(thread::spawn(move || -> Result<()> {
                while let Ok(task) = task_rx.recv() {
                    let chunk_start = task.chunk_idx * chunk_size;
                    let chunk_end = (chunk_start + chunk_size).min(task.contig_length);
                    let len = (chunk_end - chunk_start) as usize;

                    let payload = match dtype {
                        ValueDtype::U32 => {
                            if has_samples {
                                let n = readers_locks.len();
                                let mut data = ndarray::Array2::<u32>::zeros((len, n));
                                for (col_idx, reader) in readers_locks.iter().enumerate() {
                                    let mut r = reader.lock().unwrap();
                                    let chunk =
                                        r.read_chunk(&task.contig, chunk_start, chunk_end)?;
                                    let arr = match chunk {
                                        ValueChunk::U32(a) => a,
                                        other => {
                                            return Err(eyre!(
                                                "expected U32 from reader, got {:?}",
                                                std::mem::discriminant(&other)
                                            ));
                                        }
                                    };
                                    if arr.len() != len {
                                        return Err(eyre!(
                                            "reader returned {} positions, expected {}",
                                            arr.len(),
                                            len
                                        ));
                                    }
                                    for (i, v) in arr.iter().enumerate() {
                                        data[(i, col_idx)] = *v;
                                    }
                                }
                                ChunkPayload::U32_2D(data)
                            } else {
                                // Single-reader scalar path (not used by D4 import — D4 always columnar)
                                let mut r = readers_locks[0].lock().unwrap();
                                let chunk = r.read_chunk(&task.contig, chunk_start, chunk_end)?;
                                match chunk {
                                    ValueChunk::U32(a) => ChunkPayload::U32_1D(a),
                                    other => {
                                        return Err(eyre!(
                                            "expected U32 chunk, got {:?}",
                                            std::mem::discriminant(&other)
                                        ));
                                    }
                                }
                            }
                        }
                        other => return Err(eyre!("unsupported import dtype: {other}")),
                    };

                    let wt = WriteTask {
                        contig: task.contig,
                        chunk_idx: task.chunk_idx,
                        payload,
                    };
                    if write_tx.send(wt).is_err() {
                        break;
                    }
                }
                Ok(())
            }));
        }
        drop(task_rx);
        drop(write_tx);

        let mut writer_handles = Vec::with_capacity(self.writer_workers);
        for _ in 0..self.writer_workers {
            let write_rx = write_rx.clone();
            let track = self.track.clone();
            let progress = self.progress.clone();
            writer_handles.push(thread::spawn(move || -> Result<()> {
                while let Ok(wt) = write_rx.recv() {
                    match wt.payload {
                        ChunkPayload::U32_2D(arr) => {
                            track
                                .write_chunk::<u32>(&wt.contig, wt.chunk_idx, arr)
                                .map_err(|e| {
                                    eyre!(
                                        "write_chunk failed at {}:{}: {e}",
                                        wt.contig,
                                        wt.chunk_idx
                                    )
                                })?;
                        }
                        ChunkPayload::U32_1D(arr) => {
                            track
                                .write_chunk_1d::<u32>(&wt.contig, wt.chunk_idx, arr)
                                .map_err(|e| {
                                    eyre!(
                                        "write_chunk_1d failed at {}:{}: {e}",
                                        wt.contig,
                                        wt.chunk_idx
                                    )
                                })?;
                        }
                        ChunkPayload::Bool_1D(arr) => {
                            track
                                .write_chunk_1d::<bool>(&wt.contig, wt.chunk_idx, arr)
                                .map_err(|e| {
                                    eyre!(
                                        "write_chunk_1d failed at {}:{}: {e}",
                                        wt.contig,
                                        wt.chunk_idx
                                    )
                                })?;
                        }
                    }
                    if let Some(ref p) = progress {
                        p.inc(1);
                    }
                }
                Ok(())
            }));
        }
        drop(write_rx);

        let mut errors: Vec<color_eyre::Report> = Vec::new();
        match producer.join() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => errors.push(e),
            Err(_) => errors.push(eyre!("producer thread panicked")),
        }
        for h in reader_handles {
            match h.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => errors.push(e),
                Err(_) => errors.push(eyre!("reader thread panicked")),
            }
        }
        for h in writer_handles {
            match h.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => errors.push(e),
                Err(_) => errors.push(eyre!("writer thread panicked")),
            }
        }
        if let Some(p) = self.progress.as_ref() {
            p.finish();
        }
        if !errors.is_empty() {
            // Return the first error; subsequent errors get logged via tracing.
            if errors.len() > 1 {
                for e in &errors[1..] {
                    tracing::warn!("additional pipeline error: {e}");
                }
            }
            return Err(errors.into_iter().next().unwrap());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use pbzarr::{PbzStore, TrackConfig};

    /// Mock reader that returns a constant value across all positions.
    struct MockReader {
        contigs: Vec<(String, u64)>,
        value: u32,
    }

    impl ValueReader for MockReader {
        fn dtype(&self) -> ValueDtype {
            ValueDtype::U32
        }
        fn contigs(&self) -> &[(String, u64)] {
            &self.contigs
        }
        fn read_chunk(&mut self, _contig: &str, start: u64, end: u64) -> Result<ValueChunk> {
            let len = (end - start) as usize;
            Ok(ValueChunk::U32(Array1::from_elem(len, self.value)))
        }
    }

    #[test]
    fn pipeline_round_trips_uint32_two_samples() {
        let dir = tempfile::TempDir::new().unwrap();
        let store_path = dir.path().join("s.pbz.zarr");
        let store = PbzStore::create(&store_path, &["chr1".to_string()], &[300u64]).unwrap();
        let cfg = TrackConfig {
            dtype: "uint32".into(),
            samples: Some(vec!["a".into(), "b".into()]),
            chunk_size: 100,
            sample_chunk_size: 2,
            description: None,
            source: None,
            extra: serde_json::Map::new(),
        };
        let track = Arc::new(store.create_track("depths", cfg).unwrap());

        let readers: Vec<Box<dyn ValueReader>> = vec![
            Box::new(MockReader {
                contigs: vec![("chr1".into(), 300)],
                value: 5,
            }),
            Box::new(MockReader {
                contigs: vec![("chr1".into(), 300)],
                value: 7,
            }),
        ];

        let pipeline = ImportPipeline {
            readers,
            track: track.clone(),
            contigs: vec![("chr1".into(), 300u64)],
            chunk_size: 100,
            reader_workers: 2,
            writer_workers: 2,
            progress: None,
            dtype: ValueDtype::U32,
            has_samples: true,
        };
        pipeline.run().unwrap();

        for chunk_idx in 0..3 {
            let arr = track.read_chunk::<u32>("chr1", chunk_idx).unwrap();
            assert_eq!(arr.shape(), &[100, 2]);
            for row in 0..100 {
                assert_eq!(arr[(row, 0)], 5);
                assert_eq!(arr[(row, 1)], 7);
            }
        }
    }
}
