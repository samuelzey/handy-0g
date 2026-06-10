use anyhow::Result;
use hound::{WavReader, WavSpec, WavWriter};
use log::debug;
use std::io::Cursor;
use std::path::Path;

/// Mono / 16 kHz / 16-bit signed PCM — the spec [`save_wav_file`] writes and
/// the format Whisper variants (local and cloud) consume natively. Exposed so
/// callers that need the same shape in memory (cloud ASR upload) can stay in
/// sync without re-declaring it.
pub const STANDARD_WAV_SPEC: WavSpec = WavSpec {
    channels: 1,
    sample_rate: 16000,
    bits_per_sample: 16,
    sample_format: hound::SampleFormat::Int,
};

/// Read a WAV file and return normalised f32 samples.
pub fn read_wav_samples<P: AsRef<Path>>(file_path: P) -> Result<Vec<f32>> {
    let reader = WavReader::open(file_path.as_ref())?;
    let samples = reader
        .into_samples::<i16>()
        .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
        .collect::<Result<Vec<f32>, _>>()?;
    Ok(samples)
}

/// Verify a WAV file by reading it back and checking the sample count.
pub fn verify_wav_file<P: AsRef<Path>>(file_path: P, expected_samples: usize) -> Result<()> {
    let reader = WavReader::open(file_path.as_ref())?;
    let actual_samples = reader.len() as usize;
    if actual_samples != expected_samples {
        anyhow::bail!(
            "WAV sample count mismatch: expected {}, got {}",
            expected_samples,
            actual_samples
        );
    }
    Ok(())
}

/// Save audio samples as a WAV file
pub fn save_wav_file<P: AsRef<Path>>(file_path: P, samples: &[f32]) -> Result<()> {
    let mut writer = WavWriter::create(file_path.as_ref(), STANDARD_WAV_SPEC)?;

    // Convert f32 samples to i16 for WAV
    for sample in samples {
        let sample_i16 = (sample * i16::MAX as f32) as i16;
        writer.write_sample(sample_i16)?;
    }

    writer.finalize()?;
    debug!("Saved WAV file: {:?}", file_path.as_ref());
    Ok(())
}

/// Encode f32 PCM samples as an in-memory mono/16k/16-bit WAV byte buffer.
///
/// Identical wire format to [`save_wav_file`]; provided so cloud ASR can
/// upload the captured audio without an intermediate file write. Callers
/// pass the same `samples` they would feed to a local Whisper engine; the
/// returned `Vec<u8>` is a complete WAV (RIFF header + PCM data) ready for
/// `multipart/form-data`.
pub fn encode_wav_bytes(samples: &[f32]) -> Result<Vec<u8>> {
    let mut buf: Vec<u8> = Vec::with_capacity(44 + samples.len() * 2);
    {
        let cursor = Cursor::new(&mut buf);
        let mut writer = WavWriter::new(cursor, STANDARD_WAV_SPEC)?;
        for sample in samples {
            let sample_i16 = (sample * i16::MAX as f32) as i16;
            writer.write_sample(sample_i16)?;
        }
        writer.finalize()?;
    }
    Ok(buf)
}
