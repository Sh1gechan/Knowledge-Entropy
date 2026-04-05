"""
Prepare PubMed training dataset for Knowledge Entropy experiments.

Following the paper:
  "We randomly selected 204,800 instances from the PubMed dataset,
   and matched the sequence length to 1,024 tokens by concatenating instances.
   This resulted in a training dataset consisting of roughly 210 million tokens."

Source: NCBI FTP baseline (https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/)
  - Downloads xml.gz files directly, parses AbstractText fields
  - No dependency on HuggingFace dataset scripts

Output: data/pubmed_tokenized/  (HuggingFace Dataset saved with save_to_disk)
  - Each sample has "input_ids" field of length max_seq_len (1024)
  - Loaded via HFDataset.load_from_disk() in olmo/data/__init__.py

Usage:
    cd /home/ishida/Knowledge-Entropy
    source .venv/bin/activate
    python3 scripts/prepare_pubmed.py [--n_samples 204800] [--seq_len 1024] [--output data/pubmed_tokenized]
"""

import argparse
import gzip
import io
import random
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

NCBI_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"


def list_ncbi_files():
    """List all xml.gz files from NCBI PubMed baseline FTP."""
    import urllib.request
    with urllib.request.urlopen(NCBI_BASE_URL, timeout=30) as r:
        html = r.read().decode()
    files = []
    for line in html.split("\n"):
        if "xml.gz" in line and 'href="' in line:
            start = line.index('href="') + 6
            end = line.index('"', start)
            fname = line[start:end]
            if fname.endswith(".xml.gz"):
                files.append(NCBI_BASE_URL + fname)
    return sorted(files)


def fetch_and_parse(url):
    """Download a gzipped PubMed XML file and extract abstracts."""
    result = subprocess.run(
        ["curl", "-sL", "--max-time", "120", url],
        capture_output=True,
    )
    if result.returncode != 0 or len(result.stdout) < 100:
        raise RuntimeError(f"curl failed for {url}")
    xml_bytes = gzip.decompress(result.stdout)
    root = ET.fromstring(xml_bytes)
    texts = []
    for article in root.findall(".//PubmedArticle"):
        abstract_el = article.find(".//AbstractText")
        if abstract_el is not None and abstract_el.text:
            text = abstract_el.text.strip()
            if len(text) > 50:
                texts.append(text)
    return texts


def main(args):
    from datasets import Dataset
    from transformers import AutoTokenizer

    random.seed(42)

    output_path = Path(args.output)
    if output_path.exists():
        print(f"Already exists: {output_path}. Skipping.")
        print("Delete the directory to regenerate.")
        return

    print("Listing NCBI PubMed baseline files...")
    all_urls = list_ncbi_files()
    print(f"Found {len(all_urls)} files.")

    # Collect until we have enough tokens to produce n_chunks chunks of seq_len tokens.
    # PubMed abstracts average ~175 tokens, so we need far more abstracts than chunks.
    target_tokens = args.n_chunks * args.seq_len
    print(f"Collecting abstracts (target: {args.n_chunks} chunks = {target_tokens:,} tokens)...")
    texts = []
    total_approx_tokens = 0
    for i, url in enumerate(all_urls):
        print(f"  [{i+1}/{len(all_urls)}] {url.split('/')[-1]} (abstracts: {len(texts)}, approx tokens: {total_approx_tokens:,})")
        try:
            new_texts = fetch_and_parse(url)
            texts.extend(new_texts)
            total_approx_tokens = sum(len(t.split()) * 4 // 3 for t in texts)  # rough token estimate
            print(f"    -> {len(new_texts)} abstracts, total: {len(texts)}, approx tokens: {total_approx_tokens:,}")
        except Exception as e:
            print(f"    WARNING: {e}")
            continue
        if total_approx_tokens >= target_tokens:
            break

    print(f"\nCollected {len(texts)} abstracts total.")
    random.shuffle(texts)

    print("Loading OLMo tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-hf")

    print(f"Tokenizing and packing into seq_len={args.seq_len} chunks...")
    token_buffer = []
    chunks = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        token_buffer.extend(ids)
        while len(token_buffer) >= args.seq_len:
            chunks.append({"input_ids": token_buffer[:args.seq_len]})
            token_buffer = token_buffer[args.seq_len:]

    print(f"Created {len(chunks)} chunks of length {args.seq_len}.")
    print(f"Total tokens: {len(chunks) * args.seq_len:,} (~{len(chunks) * args.seq_len / 1e6:.1f}M)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving dataset to {output_path} ...")
    hf_dataset = Dataset.from_list(chunks)
    hf_dataset.save_to_disk(str(output_path))
    print(f"Done. Saved {len(hf_dataset)} samples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_chunks", type=int, default=204800,
                        help="Number of seq_len-token chunks to produce (paper uses 204800 = ~210M tokens)")
    parser.add_argument("--seq_len", type=int, default=1024,
                        help="Sequence length for tokenization (paper uses 1024)")
    parser.add_argument("--output", type=str, default="data/pubmed_tokenized",
                        help="Output directory for the HuggingFace dataset")
    args = parser.parse_args()
    main(args)
