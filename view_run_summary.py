#!/usr/bin/env python3
"""
View summary of a completed multi-run pipeline execution.

Usage:
    python view_run_summary.py <output_directory>

Example:
    python view_run_summary.py ./models_output/test_multirun_minimal/multirun_50runs_30cf_64-64-64anf_1files_160226_1234/
"""
import sys
import json
from pathlib import Path


def view_summary(output_dir):
    """Display summary from a completed run."""
    output_path = Path(output_dir)

    # Check for summary files
    json_path = output_path / "run_summary.json"
    txt_path = output_path / "run_summary.txt"

    if not json_path.exists():
        print(f"Error: No run_summary.json found in {output_dir}")
        print("Make sure this directory contains completed pipeline results.")
        return

    # Display text summary if available
    if txt_path.exists():
        print("\n" + "="*70)
        print("SUMMARY FROM:", output_path.name)
        print("="*70 + "\n")

        with open(txt_path, 'r') as f:
            print(f.read())

    # Load JSON for detailed access
    with open(json_path, 'r') as f:
        data = json.load(f)

    print("\nJSON data available with keys:", list(data.keys()))
    print("Load with: json.load(open('run_summary.json'))")

    # Quick stats
    print("\n" + "="*70)
    print("QUICK REFERENCE:")
    print("="*70)
    print(f"Total runs: {data['configuration']['num_runs']}")
    print(f"Workers: {data['configuration']['num_workers']}")
    print(f"Wall time: {data['timing']['wall_time_minutes']:.2f} min")
    print(f"Peak memory: {data['memory']['peak_memory_gb']:.3f} GB")
    print(f"Avg runtime/run: {data['timing']['average_runtime_per_run']:.2f} sec")
    print("="*70 + "\n")


def list_recent_runs(base_dir="./models_output"):
    """List recent multi-run directories."""
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"Directory {base_dir} not found")
        return

    # Find directories with run_summary.json
    run_dirs = []
    for d in base_path.rglob("run_summary.json"):
        run_dirs.append(d.parent)

    if not run_dirs:
        print(f"No completed runs found in {base_dir}")
        return

    # Sort by modification time (most recent first)
    run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    print("\nRecent completed runs:")
    print("="*70)
    for i, d in enumerate(run_dirs[:10], 1):  # Show top 10
        rel_path = d.relative_to(base_path)
        print(f"{i}. {rel_path}")
    print("="*70)

    if run_dirs:
        print(f"\nTo view a summary, run:")
        print(f"  python view_run_summary.py {run_dirs[0]}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("View summary of a completed multi-run pipeline.\n")
        print("Usage:")
        print(f"  {sys.argv[0]} <output_directory>\n")
        list_recent_runs()
    else:
        view_summary(sys.argv[1])
