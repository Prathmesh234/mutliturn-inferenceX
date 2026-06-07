#!/usr/bin/env python3
"""Upload a batch parquet (+ optional README card) to a HF dataset repo, then verify."""
import argparse
from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--path-in-repo", required=True, help="e.g. data/batch_0.parquet")
    ap.add_argument("--readme")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    api = HfApi()
    api.upload_file(path_or_fileobj=a.parquet, path_in_repo=a.path_in_repo,
                    repo_id=a.repo, repo_type="dataset",
                    commit_message=f"Update {a.path_in_repo}")
    print("uploaded", a.path_in_repo)
    if a.readme:
        api.upload_file(path_or_fileobj=a.readme, path_in_repo="README.md",
                        repo_id=a.repo, repo_type="dataset",
                        commit_message="Update dataset card")
        print("uploaded README.md")

    info = api.repo_info(repo_id=a.repo, repo_type="dataset")
    print("visibility private =", info.private)

    if a.verify:
        from datasets import load_dataset
        ds = load_dataset(a.repo, split="train", download_mode="force_redownload")
        print("HUB rows:", len(ds), "cols:", len(ds.column_names))
        t = ds[0]["turns"]
        print("turns[0] keys:", list(t[0].keys()))
        print("turn[1].seconds_since_prev:", t[1]["seconds_since_prev"] if len(t) > 1 else "n/a")
        tc = ds[0]["tool_calls"]
        print("tool_calls:", len(tc), "| keys:", list(tc[0].keys()) if tc else [])
        print("metadata.duration_ms:", ds[0]["metadata"].get("duration_ms"))


if __name__ == "__main__":
    main()
