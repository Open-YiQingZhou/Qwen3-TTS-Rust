use hound;
use std::path::Path;

pub struct AudioSample {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

impl AudioSample {
    pub fn load_wav(path: impl AsRef<Path>) -> Result<Self, String> {
        let mut reader = hound::WavReader::open(path).map_err(|e| e.to_string())?;
        let spec = reader.spec();
        let samples: Vec<f32> = reader
            .samples::<i16>()
            .map(|s| s.unwrap_or(0) as f32 / 32768.0) // Normalize i16 to f32
            .collect();

        Ok(Self {
            samples,
            sample_rate: spec.sample_rate,
            channels: spec.channels,
        })
    }

    pub fn save_wav(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let spec = hound::WavSpec {
            channels: self.channels,
            sample_rate: self.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).map_err(|e| e.to_string())?;

        for &sample in &self.samples {
            let amp = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
            writer.write_sample(amp).map_err(|e| e.to_string())?;
        }
        writer.finalize().map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn duration(&self) -> f32 {
        self.samples.len() as f32 / self.sample_rate as f32
    }
}
