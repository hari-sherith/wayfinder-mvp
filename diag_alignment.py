#!/usr/bin/env python3
"""
diag_alignment.py — Viterbi alignment diagnostic dump.

Usage:
    python diag_alignment.py <audio_file> "<target phrase>"

Prints:
  1. CTC argmax per frame (what the model thinks is in each frame)
  2. Which extended-label-sequence state each frame was assigned to by Viterbi
  3. Frame counts per phone position and for blank states
  4. Per-phone log-prob mean at the assigned frames
"""

import sys
import numpy as np

# Reuse the loading + G2P code from scoring.py
sys.path.insert(0, ".")
from scoring import _get_log_probs, g2p


def run_diagnostic(audio_path: str, target_phrase: str):
    print(f"\n=== Diagnostic: '{target_phrase}' against {audio_path} ===\n")

    ref_phones = g2p(target_phrase)
    print(f"G2P output ({len(ref_phones)} phones): {ref_phones}\n")

    log_probs_raw, processor = _get_log_probs(audio_path)
    T, V = log_probs_raw.shape
    print(f"CTC frames: T={T}  vocab_size={V}\n")

    vocab = processor.tokenizer.get_vocab()
    id2tok = {v: k for k, v in vocab.items()}
    blank_id = processor.tokenizer.pad_token_id
    print(f"Blank token id: {blank_id}  ('{id2tok[blank_id]}')\n")

    # Apply same "|" mask as score_utterance so diagnostics reflect actual scoring
    word_boundary_id = vocab.get("|")
    log_probs = log_probs_raw.copy()
    if word_boundary_id is not None:
        log_probs[:, word_boundary_id] = -1e30
        print(f"Word-boundary token '|' id={word_boundary_id} masked out (set to -inf)\n")

    # Map phones to IDs, same filtering as score_utterance
    ref_ids = []
    ref_phones_clean = []
    for phone in ref_phones:
        pid = vocab.get(phone)
        if pid is not None and pid != blank_id:
            ref_ids.append(pid)
            ref_phones_clean.append(phone)

    print(f"Reference phone sequence ({len(ref_phones_clean)} phones):")
    for i, (ph, pid) in enumerate(zip(ref_phones_clean, ref_ids)):
        print(f"  [{i}] '{ph}' -> id={pid}")
    print()

    # Build extended label sequence
    ext = [blank_id]
    for pid in ref_ids:
        ext.append(pid)
        ext.append(blank_id)
    S = len(ext)
    print(f"Extended label sequence (S={S}, format: state_idx: token_id 'token'):")
    for s, e in enumerate(ext):
        marker = "<phone>" if e != blank_id else "<blank>"
        print(f"  s={s}: id={e} '{id2tok[e]}' {marker}")
    print()

    # CTC argmax per frame
    argmax_frames = np.argmax(log_probs, axis=1)
    print(f"CTC argmax per frame (T={T}):")
    prev_tok = None
    run_start = 0
    runs = []
    for t, tok_id in enumerate(argmax_frames):
        if tok_id != prev_tok:
            if prev_tok is not None:
                runs.append((prev_tok, run_start, t - 1))
            prev_tok = tok_id
            run_start = t
    runs.append((prev_tok, run_start, T - 1))

    blank_argmax_frames = sum(1 for tok_id in argmax_frames if tok_id == blank_id)
    print(f"  Blank frames (argmax): {blank_argmax_frames}/{T} ({100*blank_argmax_frames//T}%)")
    print(f"  Run-length encoding of argmax path:")
    for tok_id, start, end in runs:
        n = end - start + 1
        print(f"    frames {start:4d}-{end:4d} ({n:3d} frames): id={tok_id} '{id2tok[tok_id]}'")
    print()

    # Viterbi forced alignment
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
            if (s >= 2
                    and ext[s] != blank_id
                    and ext[s - 2] != blank_id
                    and ext[s] != ext[s - 2]):
                candidates.append(s - 2)
            best_prev = max(candidates, key=lambda c: alpha[t - 1, c])
            alpha[t, s] = alpha[t - 1, best_prev] + log_probs[t, ext[s]]
            backptr[t, s] = best_prev

    last = S - 1 if alpha[T - 1, S - 1] >= alpha[T - 1, S - 2] else S - 2
    path = np.empty(T, dtype=np.int32)
    path[T - 1] = last
    for t in range(T - 2, -1, -1):
        path[t] = backptr[t + 1, path[t + 1]]

    # Summarise the path
    print("Viterbi forced-alignment path (state per frame → phone or blank):")
    prev_state = None
    run_start = 0
    state_runs = []
    for t, s in enumerate(path):
        if s != prev_state:
            if prev_state is not None:
                state_runs.append((prev_state, run_start, t - 1))
            prev_state = s
            run_start = t
    state_runs.append((prev_state, run_start, T - 1))

    blank_path_frames = sum(1 for s in path if ext[s] == blank_id)
    print(f"  Blank frames in Viterbi path: {blank_path_frames}/{T} ({100*blank_path_frames//T}%)")
    print(f"  Run-length encoding of Viterbi path:")
    for state, start, end in state_runs:
        n = end - start + 1
        tok_id = ext[state]
        marker = "<phone>" if tok_id != blank_id else "<blank>"
        print(f"    frames {start:4d}-{end:4d} ({n:3d} frames): s={state} id={tok_id} '{id2tok[tok_id]}' {marker}")
    print()

    # Per-phone segment summary
    probs = np.exp(log_probs)
    print("Per-phone segment summary:")
    for phone_pos, s_idx in enumerate(range(1, S, 2)):
        frames = np.where(path == s_idx)[0]
        phone = ref_phones_clean[phone_pos]
        pid = ref_ids[phone_pos]
        if len(frames) == 0:
            print(f"  [{phone_pos}] '{phone}': NO FRAMES ASSIGNED")
            continue
        frame_slice = probs[frames]
        target_post = frame_slice[:, pid]
        max_post = frame_slice.max(axis=1)
        gop = np.clip(target_post / (max_post + 1e-9), 0.0, 1.0)
        score = round(float(np.mean(gop)) * 100)
        mean_target = float(np.mean(target_post))
        mean_max = float(np.mean(max_post))
        print(
            f"  [{phone_pos}] '{phone}' (id={pid}): "
            f"{len(frames)} frames [{frames[0]}-{frames[-1]}]  "
            f"target_post_mean={mean_target:.4f}  max_post_mean={mean_max:.4f}  "
            f"GOP={score}"
        )

    print()
    print("=== End diagnostic ===")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python diag_alignment.py <audio_file> \"<target phrase>\"")
        sys.exit(1)
    run_diagnostic(sys.argv[1], sys.argv[2])
