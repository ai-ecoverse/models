"""Reference client for the export: the mel frontend and greedy TDT loop a JS engine has to reproduce.

mel() is transformers' ParakeetFeatureExtractor for one utterance, in NumPy (matches it to ~1e-4; the gap is
float64 FFT vs torch.stft in float32). greedy_tdt() reproduces ParakeetForTDT.generate() token for token.
"""
import numpy as np


def mel_filters(pp):
    import librosa
    return librosa.filters.mel(sr=pp["sampling_rate"], n_fft=pp["n_fft"], n_mels=pp["feature_size"],
                               fmin=pp["mel_fmin"], fmax=pp["mel_fmax"], norm="slaney", htk=False).astype(np.float32)


def lead(wave, pp):
    """The evaluated setting: leading_silence_seconds of zeros before the audio."""
    return np.concatenate([np.zeros(int(round(pp["leading_silence_seconds"] * pp["sampling_rate"])), np.float32),
                           wave.astype(np.float32)])


def mel(wave, pp, filters):
    """wave: f32 mono 16 kHz, leading silence already added. -> input_features [1,T,128] f32, attention_mask [1,T] i64."""
    n_fft, hop, win = pp["n_fft"], pp["hop_length"], pp["win_length"]
    x = wave.astype(np.float32)
    L = len(x)
    x = np.concatenate([x[:1], x[1:] - np.float32(pp["preemphasis"]) * x[:-1]]).astype(np.float32)
    xp = np.pad(x, (n_fft // 2, n_fft // 2))                    # center=True, zero (constant) padding
    T = 1 + L // hop
    w = np.zeros(n_fft, np.float32)                             # symmetric Hann(400), centred in 512
    off = (n_fft - win) // 2
    w[off:off + win] = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(win) / (win - 1))).astype(np.float32)
    frames = xp[np.arange(n_fft)[None, :] + hop * np.arange(T)[:, None]] * w
    spec = np.fft.rfft(frames.astype(np.float64), n=n_fft, axis=-1)
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)          # [T, 257], power 2
    m = np.log(power @ filters.T + np.float32(pp["log_zero_guard_value"]))  # [T, 128]
    valid = L // hop                                            # (L + 2*(n_fft//2) - n_fft) // hop
    mean = m[:valid].mean(0)
    std = np.sqrt(((m[:valid] - mean) ** 2).sum(0) / (valid - 1))         # unbiased, valid frames only
    m = (m - mean) / (std + np.float32(pp["normalize_epsilon"]))          # eps added to std
    m[valid:] = 0.0
    return m[None].astype(np.float32), (np.arange(T) < valid).astype(np.int64)[None]


def greedy_tdt(dj, enc, enc_len, gen):
    """enc [1,T',640] from encoder_model. Returns (tokens incl. blanks, durations): generate()'s sequences and
    durations without the leading decoder_start blank."""
    V, blank, durs = gen["vocab_size"], gen["blank_token_id"], gen["durations"]
    H, Ld = gen["decoder_hidden_size"], gen["num_decoder_layers"]
    max_steps = gen["max_symbols_per_step"] * enc.shape[1] - 1   # generate()'s max_length, minus the start token
    h = np.zeros((Ld, 1, H), np.float32)
    c = np.zeros_like(h)
    last, t, toks, ds = blank, 0, [], []
    while t < enc_len and len(toks) < max_steps:
        tl, dl, h2, c2 = dj.run(None, {"encoder_outputs": enc[:, t:t + 1], "input_ids": np.array([[last]], np.int64),
                                       "state_h": h, "state_c": c})
        tok, d = int(tl[0].argmax()), durs[int(dl[0].argmax())]
        if tok == blank:
            d = max(d, 1)                     # a blank always advances
        else:
            last, h, c = tok, h2, c2          # prediction network advances only on non-blank
        toks.append(tok)
        ds.append(d)
        t += d
        if tok == gen["eos_token_id"]:
            break
    return toks, ds


def detokenize(ids, pieces, blank):
    """Concatenate pieces, '▁' -> ' ', strip. Blank and <special> pieces dropped. No CTC-style grouping."""
    s = "".join(pieces[i] for i in ids if i != blank and not (pieces[i].startswith("<") and pieces[i].endswith(">")))
    return s.replace("\u2581", " ").strip()
