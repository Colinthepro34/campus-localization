"""
Produces data/processed/splits.json: {"train": [...], "val": [...], "test": [...]}
of image_path values. Prefers splitting whole sessions into whole splits, so
that no two images from the same session land in different splits (this is
the leakage-avoidance rule from the project spec). Falls back to stratified
random split per-location for locations with only one session.
"""
import argparse
import json
import random
from pathlib import Path

import pandas as pd
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    random.seed(cfg["split"]["random_seed"])

    df = pd.read_csv(cfg["data"]["metadata_csv"])
    test_frac = cfg["split"]["test_size_fraction"]
    val_frac = cfg["split"]["val_size_fraction"]

    train, val, test = [], [], []

    for location, loc_df in df.groupby("location"):
        sessions = loc_df["session"].unique().tolist()
        random.shuffle(sessions)

        if len(sessions) >= 3:
            n_test = max(1, round(len(sessions) * test_frac))
            n_val = max(1, round(len(sessions) * val_frac))
            test_sessions = sessions[:n_test]
            val_sessions = sessions[n_test:n_test + n_val]
            train_sessions = sessions[n_test + n_val:]

            train += loc_df[loc_df["session"].isin(train_sessions)]["image_path"].tolist()
            val += loc_df[loc_df["session"].isin(val_sessions)]["image_path"].tolist()
            test += loc_df[loc_df["session"].isin(test_sessions)]["image_path"].tolist()
        else:
            # Fallback: not enough distinct sessions for a clean session split.
            # Stratified random split within this location, logged as a warning.
            paths = loc_df["image_path"].tolist()
            random.shuffle(paths)
            n = len(paths)
            n_test = max(1, round(n * test_frac))
            n_val = max(1, round(n * val_frac))
            test += paths[:n_test]
            val += paths[n_test:n_test + n_val]
            train += paths[n_test + n_val:]
            print(
                f"WARNING: location '{location}' has only {len(sessions)} session(s); "
                f"used a random fallback split instead of session-based split."
            )

    splits = {"train": train, "val": val, "test": test}
    out_path = Path(cfg["data"]["processed_dir"]) / "splits.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(splits, indent=2))

    print(f"train={len(train)}  val={len(val)}  test={len(test)}")
    print(f"Splits written to {out_path}")

    overlap_tv = set(train) & set(val)
    overlap_tt = set(train) & set(test)
    overlap_vt = set(val) & set(test)
    assert not overlap_tv, f"Leakage: {len(overlap_tv)} images in both train and val"
    assert not overlap_tt, f"Leakage: {len(overlap_tt)} images in both train and test"
    assert not overlap_vt, f"Leakage: {len(overlap_vt)} images in both val and test"
    print("OK: no overlap between splits")


if __name__ == "__main__":
    main()