"""Fetch VerSe'20 files without downloading whole zips (HTTP range requests).

The restructured VerSe'20 zips are 11-14 GB each. For development we only need
the small derivatives (masks + centroid JSONs) and a few CTs.

Examples
--------
    python scripts/fetch_verse.py --split test --part derivatives
    python scripts/fetch_verse.py --split training --part ct --n-random 3 --seed 42
    python scripts/fetch_verse.py --split training --part all --subjects sub-verse500
"""

from __future__ import annotations

import argparse
import random
import struct
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from remotezip import RemoteZip

BASE_URL = "https://s3.bonescreen.de/public/VerSe-complete"
SPLIT_DIR = {"training": "01_training", "validation": "02_validation", "test": "03_test"}


def _keep(name: str) -> bool:
    return not (name.startswith("__MACOSX") or name.endswith((".DS_Store", "/")))


def _subject(name: str) -> str:
    # dataset-01training/<rawdata|derivatives>/<subject>/<file>
    return name.split("/")[2]


def _part_ok(name: str, part: str) -> bool:
    if part == "all":
        return True
    if part == "ct":
        return "/rawdata/" in name
    return "/derivatives/" in name and name.endswith(("_msk.nii.gz", "_ctd.json"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", choices=SPLIT_DIR, required=True)
    ap.add_argument("--part", choices=["derivatives", "ct", "all"], default="derivatives")
    ap.add_argument("--subjects", nargs="*", help="subject ids, e.g. sub-verse500")
    ap.add_argument("--n-random", type=int, help="pick N random subjects (seeded)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/verse20")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    url = f"{BASE_URL}/dataset-verse20{args.split}.zip"
    out_root = Path(args.out) / SPLIT_DIR[args.split]
    with RemoteZip(url) as z:  # only used to read the central directory
        infos = [i for i in z.infolist() if _keep(i.filename) and i.filename.count("/") >= 3]
    subjects = sorted({_subject(i.filename) for i in infos})
    if args.subjects:
        chosen = set(args.subjects)
    elif args.n_random:
        chosen = set(random.Random(args.seed).sample(subjects, args.n_random))
    else:
        chosen = set(subjects)
    todo = [i for i in infos if _subject(i.filename) in chosen and _part_ok(i.filename, args.part)]
    total = sum(i.file_size for i in todo)
    print(f"{args.split}: {len(chosen)} subjects, {len(todo)} files, {total / 1e6:.0f} MB")

    def job(info: zipfile.ZipInfo) -> None:
        dst = out_root / Path(*info.filename.split("/")[1:])  # drop 'dataset-0Xsplit'
        if dst.exists() and dst.stat().st_size == info.file_size:
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")
        tmp.write_bytes(_fetch_member(url, info))
        tmp.replace(dst)

    with ThreadPoolExecutor(args.workers) as ex:
        for n, _ in enumerate(ex.map(job, todo), 1):
            if n % 25 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}", flush=True)
    print("chosen:", " ".join(sorted(chosen)))


CHUNK = 8 << 20  # large members are fetched in 8 MB ranges (long single requests stall)


def _get_range(url: str, start: int, end: int, retries: int = 5) -> bytes:
    """GET bytes [start, end] (inclusive), retrying on timeouts / short reads."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                buf = r.read()
            if len(buf) == end - start + 1:
                return buf
        except OSError:
            if attempt == retries - 1:
                raise
    raise OSError(f"short read for range {start}-{end}")


def _fetch_member(url: str, info: zipfile.ZipInfo) -> bytes:
    """Download one zip member via range requests and check its CRC."""
    off = info.header_offset
    # local header: 30 fixed bytes + name + extra (extra length may differ from the
    # central directory, so read the real lengths from the local header)
    head = _get_range(url, off, off + 29)
    if head[:4] != b"PK\x03\x04":
        raise OSError(f"bad local header for {info.filename}")
    n_name, n_extra = struct.unpack("<HH", head[26:30])
    start = off + 30 + n_name + n_extra
    stop = start + info.compress_size  # exclusive
    raw = b"".join(_get_range(url, s, min(s + CHUNK, stop) - 1) for s in range(start, stop, CHUNK))
    if info.compress_type == zipfile.ZIP_STORED:
        data = raw
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(raw, -15)
    else:
        raise OSError(f"unsupported compression {info.compress_type} for {info.filename}")
    if zlib.crc32(data) != info.CRC:
        raise OSError(f"CRC mismatch for {info.filename}")
    return data


if __name__ == "__main__":
    main()
