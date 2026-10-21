use anyhow::{Context, Result, bail};
use ptts::flow_lm::{NormalRng, Rng, StepInput};
use ptts::loader::{load_voice_emb, load_weights};
use ptts::plan::EosPolicy;
use ptts::preprocess::Normalize;
use ptts::tok::Tok;
use ptts::tts_model::{MAX_TOKENS_PER_CHUNK, TTSConfig, TTSModel};
use std::io::Write;
use std::path::{Path, PathBuf};
use xn::Tensor;

struct Recorder {
    inner: NormalRng,
    log: Vec<f32>,
}

impl Rng for Recorder {
    fn sample(&mut self) -> f32 {
        let v = self.inner.sample();
        self.log.push(v);
        v
    }
}

type Q = xn::Unquantized<f32, xn::CpuDevice>;

fn english() -> Result<Normalize> {
    "en".parse().map_err(|e| anyhow::anyhow!("{e:?}"))
}

fn write_f32(path: &Path, values: &[f32]) -> Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    for x in values {
        f.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}

fn lines(path: &str) -> Result<Vec<String>> {
    Ok(std::fs::read_to_string(path)?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(str::to_owned)
        .collect())
}

fn plan(tokenizer: &str, input: &str) -> Result<()> {
    let tok = Tok::open(Path::new(tokenizer))?;
    let normalize = english()?;
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    for line in lines(input)? {
        let normalized = normalize.apply(&line);
        let planned = (|| -> Result<_> {
            let c = ptts::plan::chunks(&tok, &line, normalize, MAX_TOKENS_PER_CHUNK, 12.5)?;
            Ok(ptts::plan::fit_or_error(c, ptts::plan::MAX_FIT_TOKENS, &tok, 12.5)?)
        })();
        let doc = match planned {
            Ok(chunks) => {
                let js: Vec<_> = chunks
                    .iter()
                    .map(|c| {
                        serde_json::json!({
                            "text": c.text,
                            "tokens": c.tokens,
                            "frames_after_eos": c.frames_after_eos,
                            "frame_budget": c.frame_budget,
                        })
                    })
                    .collect();
                serde_json::json!({"input": line, "normalized": normalized, "chunks": js})
            }
            Err(e) => serde_json::json!({
                "input": line,
                "normalized": normalized,
                "chunks": [],
                "error": e.to_string(),
            }),
        };
        writeln!(out, "{doc}")?;
    }
    Ok(())
}

fn load(dir: &Path) -> Result<(TTSConfig, TTSModel<Q>)> {
    let cfg: TTSConfig = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let tokenizer = Tok::open(&dir.join("tokenizer.json"))?;
    let vb = load_weights::<Q>(&dir.join("model.safetensors"), &xn::CPU)?;
    let conditions = std::collections::HashMap::new();
    let model = TTSModel::load(&vb, Box::new(tokenizer), &cfg, &conditions)?;
    Ok((cfg, model))
}

fn voice(cfg: &TTSConfig, model: &TTSModel<Q>, path: &Path) -> Result<Tensor<f32, xn::CpuDevice>> {
    Ok(load_voice_emb(path, cfg.embeddings_ext.as_deref(), model.speaker_proj(), &xn::CPU)?)
}

fn voices(dir: &str, out: &str, files: &[String]) -> Result<()> {
    let (cfg, model) = load(Path::new(dir))?;
    std::fs::create_dir_all(out)?;
    for file in files {
        let path = PathBuf::from(file);
        let emb = voice(&cfg, &model, &path)?;
        let stem = path.file_stem().context("voice file name")?.to_string_lossy().into_owned();
        let v: Vec<f32> = emb.flatten_all()?.to_vec1()?;
        write_f32(&Path::new(out).join(format!("{stem}.f32")), &v)?;
        println!("{stem} {:?}", emb.dims());
    }
    Ok(())
}

fn speak(dir: &str, voice_file: &str, temperature: f32, seed: u64, input: &str, out: &str) -> Result<()> {
    let (cfg, model) = load(Path::new(dir))?;
    let out = PathBuf::from(out);
    std::fs::create_dir_all(&out)?;
    let emb = voice(&cfg, &model, Path::new(voice_file))?;
    let v: Vec<f32> = emb.flatten_all()?.to_vec1()?;
    write_f32(&out.join("voice_emb.f32"), &v)?;
    let tok = model.flow_lm.conditioner.tokenizer.as_deref().context("tokenizer")?;
    let normalize = english()?;
    for (n, line) in lines(input)?.iter().enumerate() {
        let chunks = ptts::plan::chunks(tok, line, normalize, MAX_TOKENS_PER_CHUNK, cfg.mimi.frame_rate)?;
        let chunks = ptts::plan::fit_or_error(chunks, ptts::plan::MAX_FIT_TOKENS, tok, cfg.mimi.frame_rate)?;
        let voice_len = emb.dim(1usize)?;
        let budget = chunks
            .iter()
            .map(|c| voice_len + c.tokens.len() + c.frame_budget)
            .max()
            .unwrap_or(voice_len);
        let mut base = model.init_flow_lm_state(1, budget)?;
        model.prompt_audio(&mut base, &emb)?;
        let mut rng = Recorder { inner: NormalRng::new(temperature, seed)?, log: vec![] };
        let mut js = vec![];
        let mut pcm_all: Vec<f32> = vec![];
        for chunk in &chunks {
            let mut state = base.clone();
            model.prompt_text(&mut state, &chunk.tokens)?;
            let mut mimi = model.init_mimi_state(1)?;
            let mut prev: Option<Tensor<f32, xn::CpuDevice>> = None;
            let mut eos = EosPolicy::new(chunk.frames_after_eos);
            let mut frames = vec![];
            for _ in 0..chunk.frame_budget {
                let input = match &prev {
                    None => StepInput::Bos { batch: 1 },
                    Some(t) => StepInput::Latent(t),
                };
                rng.log.clear();
                let (next, eos_logit) = model.generate_step_parts(&mut state, input, &mut rng)?;
                let eos_v: Vec<f32> = eos_logit.to_vec()?;
                let is_eos = model.eos_from_logit(&eos_v);
                let pcm: Vec<f32> = model.decode_latent(&next, &mut mimi)?.to_vec()?;
                let lat: Vec<f32> = next.flatten_all()?.to_vec1()?;
                frames.push(serde_json::json!({
                    "noise": rng.log.clone(),
                    "latent": lat,
                    "eos": eos_v[0],
                    "samples": pcm.len(),
                }));
                pcm_all.extend_from_slice(&pcm);
                if eos.should_stop(is_eos) {
                    break;
                }
                prev = Some(next);
            }
            js.push(serde_json::json!({
                "text": chunk.text,
                "tokens": chunk.tokens,
                "frames_after_eos": chunk.frames_after_eos,
                "frame_budget": chunk.frame_budget,
                "frames": frames,
            }));
        }
        let doc = serde_json::json!({
            "input": line,
            "normalized": normalize.apply(line),
            "temperature": temperature,
            "seed": seed,
            "chunks": js,
        });
        std::fs::write(out.join(format!("{n:02}.json")), serde_json::to_string(&doc)?)?;
        write_f32(&out.join(format!("{n:02}.f32")), &pcm_all)?;
        println!("{n:02}: {} chunk(s), {} samples", chunks.len(), pcm_all.len());
    }
    Ok(())
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(String::as_str) {
        Some("plan") if a.len() == 4 => plan(&a[2], &a[3]),
        Some("voices") if a.len() >= 5 => voices(&a[2], &a[3], &a[4..]),
        Some("speak") if a.len() == 8 => speak(&a[2], &a[3], a[4].parse()?, a[5].parse()?, &a[6], &a[7]),
        _ => bail!(
            "usage: ptts-ref plan TOKENIZER LINES | voices MODEL OUT VOICE... | speak MODEL VOICE TEMPERATURE SEED LINES OUT"
        ),
    }
}
