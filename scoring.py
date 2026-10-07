#!/usr/bin/env python3
"""
scoring.py — Goodness of Pronunciation (GOP) scorer

Model:  bookbot/wav2vec2-ljspeech-gruut  (wav2vec2-base, 94M params, ~376MB)
        Fine-tuned on LJSpeech with gruut IPA labels.
G2P:    gruut en-us  (pure Python, no system dependencies)

Alphabet match: both use gruut's en-us IPA inventory. Verified at startup
by comparing gruut phone output against the model tokenizer vocabulary.

Usage:
    python scoring.py <audio_file> "<target phrase>"
"""

import sys
import json
import numpy as np
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import gruut

MODEL_ID = "bookbot/wav2vec2-ljspeech-gruut"
SAMPLE_RATE = 16000

_processor: Wav2Vec2Processor | None = None
_model: Wav2Vec2ForCTC | None = None


def _load_model():
    global _processor, _model
    if _processor is None:
        print("[scoring] Loading model (first run downloads ~376 MB)…", file=sys.stderr)
        _processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
        _model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
        _model.eval()
    return _processor, _model


_STRESS = str.maketrans("", "", "ˈˌ")   # strip primary/secondary IPA stress marks


def g2p(text: str) -> list[str]:
    """
    Convert text to gruut IPA phonemes, normalised to the model's alphabet.

    gruut marks stress inside vowel tokens (e.g. 'ˈʌ', 'ˈɪ').
    bookbot/wav2vec2-ljspeech-gruut was trained without stress markers,
    so we strip U+02C8 (ˈ) and U+02CC (ˌ) before returning.
    """
    phones = []
    for sentence in gruut.sentences(text, lang="en-us"):
        for word in sentence:
            if word.phonemes:
                phones.extend(word.phonemes)
    return [p.translate(_STRESS) for p in phones]


def verify_alphabet(text: str = "bus ticket") -> bool:
    """
    Print both alphabets and confirm every G2P phone is in the model vocab.
    Returns True if they match, False if any G2P phone is missing.
    Call this before running scored batches.
    """
    processor, _ = _load_model()
    vocab = processor.tokenizer.get_vocab()
    special = {processor.tokenizer.pad_token, processor.tokenizer.unk_token, "|"}
    model_phones = {k for k in vocab if k not in special}

    g2p_phones = set(g2p(text))

    print(f"\n--- Alphabet verification for '{text}' ---")
    print(f"G2P phones : {sorted(g2p_phones)}")
    print(f"Model vocab: {sorted(model_phones)}")
    oov = g2p_phones - model_phones
    if oov:
        print(f"WARNING: G2P phones NOT in model vocab: {sorted(oov)}")
        return False
    print("All G2P phones are in the model vocabulary. ✓")
    return True


def _get_log_probs(audio_path: str):
    """Return (T, vocab_size) log-softmax array and the loaded processor."""
    processor, model = _load_model()
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits  # (1, T, V)
    log_probs = torch.log_softmax(logits.squeeze(0), dim=-1).numpy()  # (T, V)
    return log_probs, processor


def _viterbi_align(
    log_probs: np.ndarray,
    ref_ids: list[int],
    blank_id: int,
) -> list[tuple[int, int, int]]:
    """
    CTC Viterbi forced alignment.

    Returns a list of (phone_position, start_frame, end_frame) tuples,
    one entry per phoneme in ref_ids.  Frames where the aligner chose
    the blank label are not included — only frames labelled with the
    actual phone contribute.
    """
    T, _ = log_probs.shape
    N = len(ref_ids)
    if N == 0 or T < N:
        return []

    # Build extended label sequence: blank p0 blank p1 blank … pN blank
    ext = [blank_id]
    for pid in ref_ids:
        ext.append(pid)
        ext.append(blank_id)
    S = len(ext)  # 2N + 1

    NEG_INF = -1e30
    alpha = np.full((T, S), NEG_INF, dtype=np.float64)
    backptr = np.zeros((T, S), dtype=np.int32)

    alpha[0, 0] = log_probs[0, ext[0]]
    if S > 1:
        alpha[0, 1] = log_probs[0, ext[1]]

    for t in range(1, T):
        for s in range(S):
            candidates = [s]
            if s > 0:
                candidates.append(s - 1)
            # Skip over a blank if current and s-2 are different non-blank phones
            if (s >= 2
                    and ext[s] != blank_id
                    and ext[s - 2] != blank_id
                    and ext[s] != ext[s - 2]):
                candidates.append(s - 2)

            best_prev = max(candidates, key=lambda c: alpha[t - 1, c])
            alpha[t, s] = alpha[t - 1, best_prev] + log_probs[t, ext[s]]
            backptr[t, s] = best_prev

    # Traceback from the better of the two final states
    last = S - 1 if alpha[T - 1, S - 1] >= alpha[T - 1, S - 2] else S - 2
    path = np.empty(T, dtype=np.int32)
    path[T - 1] = last
    for t in range(T - 2, -1, -1):
        path[t] = backptr[t + 1, path[t + 1]]

    # Collect frame ranges for each phone position (odd indices in ext)
    segments = []
    for phone_pos, s_idx in enumerate(range(1, S, 2)):
        frames = np.where(path == s_idx)[0]
        if len(frames) > 0:
            segments.append((phone_pos, int(frames[0]), int(frames[-1])))
    return segments


def score_utterance(audio_path: str, target_phrase: str) -> dict:
    """
    Score pronunciation of audio against target_phrase.

    Returns:
        {
            "overall": int 0-100,
            "phones": [{"phone": str, "score": int 0-100}, ...],
            "target_phrase": str,
            "status": "ok"
        }
    On failure:
        {"status": "error", "reason": str, "overall": None, "phones": None}
    """
    try:
        ref_phones = g2p(target_phrase)
    except Exception as exc:
        return _err(f"G2P failed: {exc}")

    if not ref_phones:
        return _err("G2P produced an empty phoneme sequence")

    try:
        log_probs, processor = _get_log_probs(audio_path)
    except Exception as exc:
        return _err(f"Model inference failed: {exc}")

    vocab = processor.tokenizer.get_vocab()
    blank_id = processor.tokenizer.pad_token_id

    # Map each phone to its vocab ID; skip any phone missing from the vocab
    ref_ids = []
    ref_phones_clean = []
    for phone in ref_phones:
        pid = vocab.get(phone)
        if pid is not None and pid != blank_id:
            ref_ids.append(pid)
            ref_phones_clean.append(phone)

    if not ref_ids:
        return _err(
            f"None of the G2P phones are in the model vocab. "
            f"G2P output: {ref_phones}"
        )

    segments = _viterbi_align(log_probs, ref_ids, blank_id)
    if not segments:
        return _err("Viterbi alignment returned no segments")

    probs = np.exp(log_probs)  # (T, V), row-wise softmax posteriors

    phone_scores = []
    for phone_pos, start, end in segments:
        if phone_pos >= len(ref_phones_clean):
            continue
        phone = ref_phones_clean[phone_pos]
        pid = ref_ids[phone_pos]
        frame_slice = probs[start : end + 1]  # (n_frames, V)
        if frame_slice.shape[0] == 0:
            continue
        target_post = frame_slice[:, pid]               # posterior of intended phone
        max_post = frame_slice.max(axis=1)              # best-competing posterior
        # GOP ratio per frame: how dominant is the intended phone?
        frame_gop = np.clip(target_post / (max_post + 1e-9), 0.0, 1.0)
        phone_score = round(float(np.mean(frame_gop)) * 100)
        phone_scores.append({"phone": phone, "score": phone_score})

    if not phone_scores:
        return _err("No per-phone scores could be computed")

    overall = round(sum(p["score"] for p in phone_scores) / len(phone_scores))
    return {
        "overall": overall,
        "phones": phone_scores,
        "target_phrase": target_phrase,
        "status": "ok",
    }


def _err(reason: str) -> dict:
    return {"status": "error", "reason": reason, "overall": None, "phones": None}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scoring.py <audio_file> \"<target phrase>\"")
        sys.exit(1)

    audio_file = sys.argv[1]
    phrase = sys.argv[2]

    ok = verify_alphabet(phrase)
    if not ok:
        print(
            "\nERROR: alphabet mismatch — stopping. Fix the G2P/model pair before scoring.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\n[Scoring '{phrase}' against {audio_file}]")
    result = score_utterance(audio_file, phrase)
    print(json.dumps(result, indent=2, ensure_ascii=False))
