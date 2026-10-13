"""Bounded acquisition of four CC BY 4.0 Neutral BVHs, with source receipts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path

SOURCE = "https://www.ianxmason.com/100style/"
FILES = {
    name: "https://drive.usercontent.google.com/download?export=download&id=" + file_id
    for name, file_id in {
        "Neutral_ID.bvh": "1Vnz5wnFmOztNzaSBHjSt9v6y0pl0FK8X",
        "Neutral_FW.bvh": "1-TqZdIJvpr-QVvAmYJREUZkuAgpAQSfd",
        "Neutral_BW.bvh": "1pmJ3GAceoi04kCmXsHvC56TJzKDEHh02",
        "Neutral_SW.bvh": "1HzUccloKCjQgpObQ0ZXTEDW7-RllNwHX",
        "Frame_Cuts.csv": "1d0VM8k4UjA4dDmaviuZMjAf-WQNLUwnZ",
    }.items()
}
MAX_FILE_BYTES = 32 * 1024 * 1024
ARCHIVE_URL = "https://zenodo.org/records/8127870/files/100STYLE.zip?download=1"
ARCHIVE_BYTES = 1468515354


def check_download(name, payload):
    if not 0 < len(payload) <= MAX_FILE_BYTES:
        raise ValueError("100STYLE file is empty or exceeds the trial size budget")
    text = payload.decode("utf-8-sig")
    if name.endswith(".bvh"):
        match = re.search(r"Frames:\s*(\d+)\s+Frame Time:\s*([\d.]+)", text)
        if (
            not text.lstrip().startswith("HIERARCHY")
            or not match
            or int(match[1]) < 9
            or abs(float(match[2]) - 1 / 60) > 1e-5
            or "ROOT Hips" not in text
        ):
            raise ValueError("100STYLE response is not the expected Xsens 60Hz BVH")
        hierarchy, motion = text.split("MOTION", 1)
        header = re.match(r"\s*Frames:\s*(\d+)\s+Frame Time:\s*([\d.]+)\s*", motion)
        channels = sum(map(int, re.findall(r"CHANNELS\s+(\d+)", hierarchy)))
        rows = [row.split() for row in motion[header.end() :].splitlines() if row.strip()]
        if len(rows) != int(match[1]) or not channels or any(len(row) != channels for row in rows):
            raise ValueError("100STYLE BVH does not contain complete declared motion frames")
        import math

        try:
            finite = all(math.isfinite(float(value)) for row in rows for value in row)
        except ValueError:
            finite = False
        if not finite:
            raise ValueError("100STYLE BVH channels are not complete finite numbers")
        return {"frames": int(match[1]), "fps": 60, "raw_duration_s": int(match[1]) / 60}
    rows = list(csv.reader(io.StringIO(text)))
    if not rows or not any("Neutral" in cell for row in rows for cell in row):
        raise ValueError("100STYLE frame-cut response is invalid or lacks Neutral")
    return {"rows": len(rows)}


class ArchiveRanges:
    """Seekable fixed-release ZIP, refusing servers that ignore byte ranges."""

    def __init__(self):
        self.position = 0

    def seek(self, offset, whence=0):
        self.position = offset + (0 if whence == 0 else self.position if whence == 1 else ARCHIVE_BYTES)
        if not 0 <= self.position <= ARCHIVE_BYTES:
            raise ValueError("100STYLE ZIP seek exceeds the fixed release")
        return self.position

    def tell(self):
        return self.position

    def seekable(self):
        return True

    def read(self, size=-1):
        size = min(ARCHIVE_BYTES - self.position, size if size >= 0 else ARCHIVE_BYTES - self.position)
        if size > MAX_FILE_BYTES:
            raise ValueError("100STYLE ZIP range exceeds the trial budget")
        if size == 0:
            return b""
        first, last = self.position, self.position + size - 1
        request = urllib.request.Request(ARCHIVE_URL, headers={"Range": f"bytes={first}-{last}"})
        with urllib.request.urlopen(request, timeout=60) as response:
            if (
                response.status != 206
                or response.headers.get("Content-Range") != f"bytes {first}-{last}/{ARCHIVE_BYTES}"
            ):
                raise ValueError("100STYLE archive server did not honor the bounded byte range")
            payload = response.read(size + 1)
        if len(payload) != size:
            raise ValueError("100STYLE archive range was truncated")
        self.position += size
        return payload


def archive_file(name):
    with zipfile.ZipFile(ArchiveRanges()) as archive:
        members = [item for item in archive.infolist() if Path(item.filename).name == name]
        if len(members) != 1 or members[0].file_size > MAX_FILE_BYTES:
            raise ValueError("100STYLE archive member is missing, ambiguous or too large")
        item = members[0]
        return archive.read(item), {
            "archive_member": item.filename,
            "zip_crc32": f"{item.CRC:08x}",
            "archive_md5_verified": False,
        }


def acquire(output):
    if output.exists():
        raise ValueError("100STYLE acquisition refuses to overwrite an existing trial")
    staging = output.with_name(output.name + ".staging")
    staging.mkdir(parents=True, exist_ok=False)
    receipts = []
    for name, url in FILES.items():
        request = urllib.request.Request(url, headers={"User-Agent": "EmbodiedBein/0.1 100STYLE-data-probe"})
        with urllib.request.urlopen(request, timeout=40) as response:
            if not response.url.startswith("https://drive.usercontent.google.com/"):
                raise ValueError("100STYLE download redirected outside the pinned source")
            payload = response.read(MAX_FILE_BYTES + 1)
        extra = {}
        try:
            stats = check_download(name, payload)
        except ValueError as exc:
            print(
                json.dumps({"rejected_source": url, "reason": str(exc), "fallback": ARCHIVE_URL}), flush=True
            )
            payload, extra = archive_file(name)
            stats = check_download(name, payload)
            url = ARCHIVE_URL
        (staging / name).write_bytes(payload)
        item = {
            "file": name,
            "uri": url,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            **stats,
            **extra,
        }
        receipts.append(item)
        print(json.dumps(item), flush=True)
    manifest = {
        "schema_version": "eb/100style-source-trial/v1",
        "source": SOURCE,
        "release": "zenodo:8127870/v1",
        "license": "CC-BY-4.0",
        "attribution": "The 100STYLE Dataset - Ian Mason; Mason, Starke and Komura (2022)",
        "files": receipts,
        "training_ready": False,
        "unchecked": ["RPO retarget calibration", "dynamic replay", "independent evaluation"],
    }
    (staging / "source-manifest.json").write_text(json.dumps(manifest, indent=2))
    staging.rename(output)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    acquire(args.output)
