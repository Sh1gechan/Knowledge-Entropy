"""
Prepare data/dolma_1B/first_1k.json from Dolma v1.6-sample.

This script streams the first file of Dolma v1.6-sample (json.gz format)
and saves the first N=2048 documents as data/dolma_1B/first_1k.json,
matching the format expected by analysis/entropy.py:
  [{"id": <int>, "text": <str>}, ...]

Usage:
    cd /home/ishida/Knowledge-Entropy
    source .venv/bin/activate
    python3 scripts/prepare_dolma_sample.py
"""

import gzip
import json
import os
import urllib.request
from pathlib import Path

N_DOCS = 2048
OUTPUT_PATH = Path("data/dolma_1B/first_1k.json")
URL_LIST = "https://huggingface.co/datasets/allenai/dolma/resolve/main/urls/v1_6-sample.txt"


def get_file_urls():
    with urllib.request.urlopen(URL_LIST, timeout=30) as r:
        return r.read().decode().strip().split("\n")


def stream_docs(url, max_docs):
    """Stream documents from a gzipped jsonl file at the given URL."""
    import subprocess
    docs = []
    # Use curl to handle redirects (-L) and download to stdout
    result = subprocess.run(
        ["curl", "-sL", "--max-time", "120", url],
        capture_output=True,
    )
    if result.returncode != 0 or len(result.stdout) < 100:
        raise RuntimeError(f"curl failed (returncode={result.returncode}, size={len(result.stdout)})")
    import io
    with gzip.open(io.BytesIO(result.stdout), "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            if text:
                docs.append(text)
            if len(docs) >= max_docs:
                break
    return docs


def main():
    if OUTPUT_PATH.exists():
        print(f"Already exists: {OUTPUT_PATH}. Skipping.")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Fetching Dolma v1.6-sample URL list...")
    urls = get_file_urls()
    print(f"Total files: {len(urls)}")

    docs = []
    for i, url in enumerate(urls):
        remaining = N_DOCS - len(docs)
        if remaining <= 0:
            break
        print(f"[{i+1}/{len(urls)}] Streaming {url} (need {remaining} more docs)...")
        try:
            new_docs = stream_docs(url, remaining)
            docs.extend(new_docs)
            print(f"  Got {len(new_docs)} docs. Total so far: {len(docs)}")
        except Exception as e:
            print(f"  WARNING: Failed to stream {url}: {e}. Trying next file...")
            continue

    if not docs:
        print("ERROR: Could not retrieve any documents.")
        return

    to_save = [{"id": i, "text": text} for i, text in enumerate(docs)]
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(to_save, f, indent=4, ensure_ascii=False)

    print(f"\nSaved {len(to_save)} documents to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
