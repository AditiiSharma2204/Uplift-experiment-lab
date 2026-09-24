"""Download, load and cache the Criteo uplift v2.1 data; build the stratified dev sample.

The gzipped CSV (~300 MB) is read directly by pyarrow with explicit narrow dtypes,
so the ~3 GB uncompressed CSV never touches disk. A parquet cache makes reloads fast.
"""
from __future__ import annotations

import hashlib
import shutil
import time
import urllib.request

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq

from .common import FEATURES, OUTCOMES, PROCESSED_DIR, RAW_DIR, SEED, TREATMENT

# Official Criteo AI Lab copy on Hugging Face (the scikit-uplift S3 mirror now returns 403).
CRITEO_URL = "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz"
RAW_GZ = RAW_DIR / "criteo-research-uplift-v2.1.csv.gz"
RAW_MD5 = "d2236769ef69e9be52556110102911ec"  # same hash scikit-uplift pins for the full file
FULL_PARQUET = PROCESSED_DIR / "criteo_full.parquet"
DEV_PARQUET = PROCESSED_DIR / "criteo_dev.parquet"

INT_COLS = [TREATMENT, *OUTCOMES, "exposure"]
COLUMN_TYPES = {**{f: pa.float32() for f in FEATURES}, **{c: pa.int8() for c in INT_COLS}}


def download(url: str = CRITEO_URL, dest=RAW_GZ) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"Downloading {url} -> {dest}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.rename(dest)


def verify_md5(path=RAW_GZ, expected: str = RAW_MD5) -> None:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    assert h.hexdigest() == expected, f"Checksum mismatch for {path}: {h.hexdigest()}"


def build_parquet_cache() -> None:
    if FULL_PARQUET.exists():
        return
    download()
    verify_md5()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    table = pv.read_csv(RAW_GZ, convert_options=pv.ConvertOptions(column_types=COLUMN_TYPES))
    missing = set(COLUMN_TYPES) - set(table.column_names)
    assert not missing, f"Unexpected schema, missing columns: {missing}"
    pq.write_table(table, FULL_PARQUET, compression="zstd")
    print(f"Parsed {table.num_rows:,} rows in {time.time() - t0:.0f}s -> {FULL_PARQUET}")


def load_full(columns: list[str] | None = None) -> pd.DataFrame:
    build_parquet_cache()
    return pd.read_parquet(FULL_PARQUET, columns=columns)


def memory_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum() / 2**20)


def stratified_sample(df: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    """Sample ~n rows keeping the joint treatment x visit x conversion shares exact (to rounding)."""
    strata = [TREATMENT, *OUTCOMES]
    frac = n / len(df)
    rng = np.random.default_rng(seed)
    parts = []
    for _, idx in df.groupby(strata, observed=True).indices.items():
        k = int(round(len(idx) * frac))
        parts.append(rng.choice(idx, size=k, replace=False))
    take = np.sort(np.concatenate(parts))
    return df.iloc[take].reset_index(drop=True)


def build_dev_sample(n: int = 1_000_000, seed: int = SEED) -> pd.DataFrame:
    if DEV_PARQUET.exists():
        return pd.read_parquet(DEV_PARQUET)
    dev = stratified_sample(load_full(), n, seed)
    dev.to_parquet(DEV_PARQUET, compression="zstd", index=False)
    return dev


def load_dev() -> pd.DataFrame:
    return build_dev_sample()
