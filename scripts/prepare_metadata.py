"""
One-off data preparation: normalizes the user's raw metadata CSV into the
schema expected by the pipeline (see CONFIG/README). The original raw CSV
is never modified -- a copy is written to data/raw/metadata.csv.

Transformations applied:
  - image_path  : from `file_path`               (relative to data/raw/images/)
  - location    : from `location_label`
  - floor       : string -> int   ground -> 0, second -> 2
  - session     : derived from (location, capture_timestamp) capture bursts,
                  e.g. "2026-09-09-morning-A". Keeps every frame burst from
                  the same location inside one session (leakage-avoidance).
  - time_of_day : derived from capture_timestamp hour (<12 -> morning)
  - latitude / longitude : passed through unchanged
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT.parent / "metadata.csv"          # original user CSV, untouched
OUT_CSV = ROOT / "data" / "raw" / "metadata.csv"

FLOOR_MAP = {"ground": 0, "first": 1, "second": 2, "third": 3}


def time_of_day(ts: pd.Timestamp) -> str:
    return "morning" if ts.hour < 12 else "afternoon"


def main():
    df = pd.read_csv(RAW_CSV)
    df["capture_timestamp"] = pd.to_datetime(df["capture_timestamp"])

    def rel_to_images(fp: str) -> str:
        # Original file_path is relative to the repo root and includes an
        # "images/" prefix; the schema wants image_path relative to the
        # images_dir itself, so strip any leading directory.
        parts = Path(fp).parts
        if parts and parts[0] == "images":
            parts = parts[1:]
        return str(Path(*parts)) if parts else fp

    out = pd.DataFrame()
    out["image_path"] = df["file_path"].map(rel_to_images)
    out["location"] = df["location_label"]
    out["floor"] = df["floor"].map(FLOOR_MAP)
    out["session"] = ""
    out["time_of_day"] = df["capture_timestamp"].map(time_of_day)
    out["latitude"] = df["latitude"]
    out["longitude"] = df["longitude"]

    groups = sorted(df.groupby(["location_label", "capture_timestamp"]).groups.keys())
    LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    session_ids = {}
    for i, (loc, ts) in enumerate(groups):
        letters = LETTERS[i // 26] + LETTERS[i % 26] if i >= 26 else LETTERS[i % 26]
        session_ids[(loc, ts)] = f"{ts.date()}--{time_of_day(ts)}-{letters}"

    session_col = df.apply(lambda r: session_ids[(r["location_label"], r["capture_timestamp"])], axis=1)
    out["session"] = session_col

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    print(f"Wrote {len(out)} rows to {OUT_CSV}")
    print(f"Floors present: {sorted(out['floor'].dropna().unique().tolist())}")
    print(f"Distinct sessions: {out['session'].nunique()}")
    print(f"Distinct locations: {out['location'].nunique()}")


if __name__ == "__main__":
    main()