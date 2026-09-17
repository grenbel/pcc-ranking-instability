"""Arrange metric-emitter outputs into the layout of the supplementary archive.

The archive-relative drivers in this directory (instability_summary.py, validpt_sensitivity.py,
upsample_sensitivity.py, mixed_pilot.py) read ``<archive>/metrics*/<Baseline>/<op>_s<sev>.json``.
The metric emitter writes ``<model>_<op>_s<sev>_per_sample.json``. This helper copies the
emitter's files of one protocol into an unpacked archive (or any target directory) using the
archive conventions:

    protocol   archive subdirectory   file name change            row change
    zero_pad   metrics/               none                        model_name SnowFlakeNet -> SnowflakeNet
    validpt    metrics_validpt/       "_validpt" op suffix dropped model_name SnowFlakeNet -> SnowflakeNet
    upsample   metrics_upsample/      none                        none (byte-identical copy)
    composed   metrics_mixed/         none                        none (byte-identical copy)

Baseline directories always use the archive spelling (``SnowflakeNet``); the ``op`` field
inside the rows is never changed.

Usage:
    python analysis/arrange_metrics.py --protocol zero_pad --src logs/zero_pad --archive /path/to/supplementary_archive
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# protocol -> (archive subdirectory, respell model_name in rows, op suffix to drop from file names)
PROTOCOLS = {
    "zero_pad": ("metrics", True, ""),
    "validpt": ("metrics_validpt", True, "_validpt"),
    "upsample": ("metrics_upsample", False, ""),
    "composed": ("metrics_mixed", False, ""),
}
# registry spelling (rows, npz stamps) -> archive spelling (directories, respelled rows)
ARCHIVE_NAME = {"PoinTr": "PoinTr", "AdaPoinTr": "AdaPoinTr", "SnowFlakeNet": "SnowflakeNet",
                "SeedFormer": "SeedFormer"}
SUFFIX = "_per_sample.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--protocol", required=True, choices=sorted(PROTOCOLS))
    ap.add_argument("--src", required=True, help="metric_emitter.py --output-dir of that protocol")
    ap.add_argument("--archive", required=True,
                    help="root of an unpacked supplementary archive (or any directory to populate)")
    ap.add_argument("--overwrite", action="store_true", help="replace files that already exist")
    args = ap.parse_args()
    subdir, respell, strip = PROTOCOLS[args.protocol]
    src, dst_root = Path(args.src), Path(args.archive) / subdir
    files = sorted(src.glob(f"*{SUFFIX}"))
    if not files:
        sys.exit(f"[arrange] no *{SUFFIX} files in {src}")
    n = 0
    for p in files:
        stem = p.name[:-len(SUFFIX)]
        model = next((m for m in ARCHIVE_NAME if stem.startswith(m + "_")), None)
        if model is None:
            sys.exit(f"[arrange] {p.name}: unknown model prefix (expected one of {list(ARCHIVE_NAME)})")
        op, sep, sev = stem[len(model) + 1:].rpartition("_s")
        if not sep or not sev.isdigit():
            sys.exit(f"[arrange] {p.name}: cannot parse '<op>_s<sev>'")
        if strip:
            if not op.endswith(strip):
                sys.exit(f"[arrange] {p.name}: op '{op}' lacks the expected '{strip}' suffix")
            op = op[:-len(strip)]
        dst = dst_root / ARCHIVE_NAME[model] / f"{op}_s{sev}.json"
        if dst.exists() and not args.overwrite:
            sys.exit(f"[arrange] {dst} exists (pass --overwrite to replace)")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if respell:
            with open(p, encoding="utf-8") as f:
                rows = json.load(f)
            for i, r in enumerate(rows):
                if r.get("model_name") != model:
                    sys.exit(f"[arrange] {p.name} row {i}: model_name {r.get('model_name')!r} != {model!r}")
                r["model_name"] = ARCHIVE_NAME[model]
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2, allow_nan=False)
        else:
            shutil.copyfile(p, dst)
        n += 1
    print(f"[arrange] {n} files written under {dst_root}")


if __name__ == "__main__":
    main()
