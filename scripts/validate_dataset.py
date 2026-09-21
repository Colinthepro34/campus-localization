"""
Validates data/raw/metadata.csv against data/raw/images/.
Checks: missing files, missing columns, class balance, near-duplicate
images (perceptual hash), and whether sessions exist per location.
Writes a report to outputs/reports/dataset_validation.md and exits with
a non-zero status code if any CRITICAL issue is found.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import imagehash
import pandas as pd
import yaml
from PIL import Image


REQUIRED_COLUMNS = ["image_path", "location", "floor", "session", "time_of_day"]


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def check_columns(df: pd.DataFrame, report: list) -> bool:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        report.append(f"- **CRITICAL**: missing required columns: {missing}")
        return False
    report.append("- OK: all required columns present")
    return True


def check_files_exist(df: pd.DataFrame, images_dir: Path, report: list) -> pd.DataFrame:
    missing_rows = []
    for idx, row in df.iterrows():
        p = images_dir / row["image_path"]
        if not p.exists():
            missing_rows.append(row["image_path"])
    if missing_rows:
        report.append(
            f"- **CRITICAL**: {len(missing_rows)} image files listed in metadata "
            f"but not found on disk. First 10: {missing_rows[:10]}"
        )
        df = df[~df["image_path"].isin(missing_rows)]
    else:
        report.append("- OK: every row's image_path exists on disk")
    return df


def check_class_balance(df: pd.DataFrame, report: list):
    counts = df.groupby(["location", "floor"]).size().sort_values()
    report.append("\n### Images per (location, floor)\n")
    report.append("| location | floor | count |")
    report.append("|---|---|---|")
    for (loc, floor), c in counts.items():
        flag = " ⚠️ low count" if c < 8 else ""
        report.append(f"| {loc} | {floor} | {c}{flag} |")
    low = counts[counts < 8]
    if len(low) > 0:
        report.append(
            f"\n- **WARNING**: {len(low)} (location, floor) pairs have fewer than "
            f"8 images. These are unlikely to generalize well."
        )


def check_sessions(df: pd.DataFrame, report: list):
    report.append("\n### Sessions per location\n")
    session_counts = df.groupby("location")["session"].nunique().sort_values()
    report.append("| location | distinct sessions |")
    report.append("|---|---|")
    single_session_locations = []
    for loc, n in session_counts.items():
        report.append(f"| {loc} | {n} |")
        if n < 2:
            single_session_locations.append(loc)
    if single_session_locations:
        report.append(
            f"\n- **WARNING**: these locations have only 1 capture session, so a "
            f"clean session-based test split is not possible for them: "
            f"{single_session_locations}. They will fall back to a stratified "
            f"random split in Task 3, which is weaker evidence of "
            f"generalization — flag this in the final report."
        )


def check_near_duplicates(df: pd.DataFrame, images_dir: Path, report: list, hash_threshold: int = 3):
    report.append("\n### Near-duplicate detection (perceptual hash)\n")
    hashes = {}
    duplicate_groups = defaultdict(list)
    for _, row in df.iterrows():
        p = images_dir / row["image_path"]
        try:
            h = imagehash.phash(Image.open(p).convert("RGB"))
        except Exception as e:
            report.append(f"- **WARNING**: could not hash {row['image_path']}: {e}")
            continue
        hashes[row["image_path"]] = h

    items = list(hashes.items())
    seen = set()
    for i in range(len(items)):
        path_i, hash_i = items[i]
        if path_i in seen:
            continue
        group = [path_i]
        for j in range(i + 1, len(items)):
            path_j, hash_j = items[j]
            if path_j in seen:
                continue
            if hash_i - hash_j <= hash_threshold:
                group.append(path_j)
                seen.add(path_j)
        if len(group) > 1:
            duplicate_groups[path_i] = group
            seen.add(path_i)

    if duplicate_groups:
        report.append(
            f"- **WARNING**: found {len(duplicate_groups)} groups of near-duplicate "
            f"images (hash distance <= {hash_threshold}). These are strong leakage "
            f"candidates if any pair ends up split across train/test. Sample groups:"
        )
        for k, (rep, group) in enumerate(list(duplicate_groups.items())[:5]):
            report.append(f"  - Group {k+1}: {group}")
    else:
        report.append("- OK: no near-duplicate images detected above threshold")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    images_dir = Path(cfg["data"]["images_dir"])
    metadata_csv = Path(cfg["data"]["metadata_csv"])
    report = ["# Dataset validation report\n"]

    if not metadata_csv.exists():
        print(f"CRITICAL: {metadata_csv} does not exist. Cannot continue.")
        sys.exit(1)

    df = pd.read_csv(metadata_csv)
    report.append(f"- Loaded {len(df)} rows from {metadata_csv}")

    ok = check_columns(df, report)
    if not ok:
        Path("outputs/reports").mkdir(parents=True, exist_ok=True)
        Path("outputs/reports/dataset_validation.md").write_text("\n".join(report))
        print("CRITICAL issues found. See outputs/reports/dataset_validation.md")
        sys.exit(1)

    df = check_files_exist(df, images_dir, report)
    check_class_balance(df, report)
    check_sessions(df, report)
    check_near_duplicates(df, images_dir, report)

    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    out_path = Path("outputs/reports/dataset_validation.md")
    out_path.write_text("\n".join(report))
    print(f"Report written to {out_path}")

    critical = any("CRITICAL" in line for line in report)
    if critical:
        print("Result: CRITICAL issues present. Fix before continuing.")
        sys.exit(1)
    print("Result: no CRITICAL issues. Warnings (if any) are in the report.")
    sys.exit(0)


if __name__ == "__main__":
    main()