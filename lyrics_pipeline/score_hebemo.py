"""Score matched lyrics with HebEMO (8 emotions) + heBERT sentiment.

Differences from the 2026-02 Colab run:
  * long songs are scored in overlapping 512-token windows and averaged (weighted by
    window length) instead of being silently truncated (49 songs were cut);
  * section labels ('פזמון:', 'x2', '[...]') are removed before scoring;
  * token counts and window counts are recorded per song.

Needs network access to huggingface.co the first time (≈4 GB of model weights).
Runs on CPU (≈20-40 min for ~900 songs) or GPU.

Usage: python score_hebemo.py            -> out/song_emotions_v2.csv
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
OUT = HERE / "out"
EMOTIONS = ["joy", "sadness", "anger", "fear", "trust", "disgust", "surprise", "anticipation"]
SECTION = re.compile(r"^\s*(פזמון|בית|גשר|מעבר|chorus|verse|bridge|intro|outro)\s*\d*\s*:?\s*$", re.I | re.M)
REPEAT = re.compile(r"\(?\s*[xX×]\s*\d+\s*\)?|\(?\s*\d+\s*[xX×]\s*\)?")


def clean_lyrics(t: str) -> str:
    t = SECTION.sub("", t)
    t = REPEAT.sub("", t)
    t = re.sub(r"\[[^\]]*\]", "", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def windows(ids: list[int], size: int = 510, stride: int = 255) -> list[list[int]]:
    if len(ids) <= size:
        return [ids]
    out, start = [], 0
    while True:
        out.append(ids[start:start + size])
        if start + size >= len(ids):
            return out
        start += stride


def main():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained("avichr/heBERT")
    models = {e: AutoModelForSequenceClassification.from_pretrained(f"avichr/hebEMO_{e}").to(dev).eval()
              for e in EMOTIONS}
    sent = AutoModelForSequenceClassification.from_pretrained("avichr/heBERT_sentiment_analysis").to(dev).eval()

    m = pd.read_csv(OUT / "matches.csv")
    m = m[m.lyrics_file.notna()]
    rows, skipped = [], []
    for i, r in enumerate(m.itertuples()):
        text = clean_lyrics((OUT / "lyrics" / r.lyrics_file).read_text(encoding="utf-8"))
        letters = re.findall(r"[^\W\d_]", text)
        heb_share = sum("\u0590" <= ch <= "\u05FF" for ch in letters) / max(1, len(letters))
        if heb_share < 0.5:  # HebEMO is Hebrew-only: English/French lyrics are excluded, not mis-scored
            skipped.append({"uri": r.uri, "hebrew_share": round(heb_share, 2)})
            continue
        ids = tok(text, add_special_tokens=False)["input_ids"]
        wins = windows(ids)
        batch = [[tok.cls_token_id] + w + [tok.sep_token_id] for w in wins]
        maxlen = max(map(len, batch))
        input_ids = torch.tensor([b + [tok.pad_token_id] * (maxlen - len(b)) for b in batch], device=dev)
        attn = (input_ids != tok.pad_token_id).long()
        weights = torch.tensor([len(w) for w in wins], dtype=torch.float, device=dev)
        weights = weights / weights.sum()
        res = {"uri": r.uri, "n_tokens": len(ids), "n_windows": len(wins)}
        with torch.no_grad():
            for e, mod in models.items():
                p = torch.softmax(mod(input_ids=input_ids, attention_mask=attn).logits, dim=1)[:, 1]
                res[e] = float((p * weights).sum())
            p = torch.softmax(sent(input_ids=input_ids, attention_mask=attn).logits, dim=1)
            for j, lab in enumerate(["sentiment_neutral", "sentiment_positive", "sentiment_negative"]):
                res[lab] = float((p[:, j] * weights).sum())
        rows.append(res)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(m)}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "song_emotions_v2.csv", index=False)
    pd.DataFrame(skipped).to_csv(OUT / "skipped_non_hebrew.csv", index=False)
    print(f"scored {len(rows)} songs -> {OUT/'song_emotions_v2.csv'}; skipped {len(skipped)} non-Hebrew lyrics")


if __name__ == "__main__":
    main()
