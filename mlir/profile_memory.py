"""Collect matched IREE queue allocations from native executions. TSV/CSV to stdout."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("bin", type=Path)
parser.add_argument("parameters", type=Path)
parser.add_argument("modules", nargs="+", type=Path)
args = parser.parse_args()
out = csv.writer(sys.stdout)
out.writerow(
    (
        "batch",
        "sequence",
        "round",
        "module",
        "queue_requests",
        "queue_requested_bytes",
        "queue_live_peak_bytes",
        "pool_committed_peak_bytes",
    )
)
with TemporaryDirectory() as scratch:
    bundle = Path(scratch) / "memory.ireeprof"
    for batch, seq in ((1, 16), (1, 64), (1, 128), (3, 97), (1, 256)):
        for trial in range(3):
            order = (
                args.modules[trial:] + args.modules[:trial]
                if trial < len(args.modules)
                else args.modules[::-1]
            )
            for module in order:
                prefix = f"--input={batch}x{seq}xi64="
                bundle.unlink(missing_ok=True)
                subprocess.run(
                    [
                        args.bin / "iree-run-module",
                        f"--module={module}",
                        f"--parameters=model={args.parameters}",
                        "--device=local-task",
                        "--function=main_graph",
                        prefix + "1",
                        prefix + "1",
                        prefix + "0",
                        "--output=@/dev/null",
                        "--device_profiling_mode=memory-events",
                        f"--device_profiling_output={bundle}",
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
                report = subprocess.check_output(
                    [args.bin / "iree-profile", "memory", "--format=jsonl", bundle],
                    text=True,
                )
                rows = [json.loads(line) for line in report.splitlines()]
                (summary,) = (r for r in rows if r["type"] == "memory_summary")
                assert summary["dropped_records"] == summary["truncated_chunks"] == 0
                (pool,) = (
                    r
                    for r in rows
                    if r["type"] == "memory_pool" and r["kind"] == "queue_allocation"
                )
                (backing,) = (
                    r
                    for r in rows
                    if r["type"] == "memory_pool" and r["kind"] == "pool_reservation"
                )
                out.writerow(
                    (
                        batch,
                        seq,
                        trial + 1,
                        module.stem,
                        pool["lifecycle_total_open_count"],
                        pool["lifecycle_total_open_bytes"],
                        pool["lifecycle_high_water_bytes"],
                        backing["pool_bytes_committed_high_water"],
                    )
                )
