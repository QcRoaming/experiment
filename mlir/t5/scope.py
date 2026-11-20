"""Report temporary capacity from measured queues and the fixed-schedule live bound."""

import csv
import json
from pathlib import Path
from statistics import mean
import sys

out = Path(sys.argv[1])
cases = {
    (c["target"], c["source"], c["pack"]): c
    for c in json.loads((out / "packs.json").read_text())["cases"]
}
capacities = {}
with (out / "tree-capacities.csv").open() as f:
    for r in csv.DictReader(f):
        plan = int(r["hint_target"]), int(r["hint_source"]), r["priority"]
        shape = int(r["target"]), int(r["source"])
        capacities.setdefault(plan, {})[shape] = int(r["capacity"])
hints = sorted({p[:2] for p in capacities})


def single_tree_score(plan):
    ratios = [capacities[plan][t, s] / cases[t, s, 1]["live_lower_bound_bytes"]
              for t, s in hints]
    return max(ratios), sum(ratios), plan


one_tree = min(capacities, key=single_tree_score)
with (out / "three/results.csv").open() as f:
    queues = {}
    for r in csv.DictReader(f):
        key = int(r["target"]), int(r["source"])
        assert r["queue_requests"] == "2" and r["bitwise_equal"] == "True"
        queues.setdefault(key, {})[r["module"]] = int(r["queue_live_peak_bytes"])

rows = []
savings = []
single_tree_savings = []
for (target, source), measured in queues.items():
    outputs = cases[target, source, 0]
    output_size = sum(outputs["aligned_sizes"])
    # All output slices coexist and use the same layout in the two variants.
    assert output_size == outputs["live_lower_bound_bytes"]
    lower = cases[target, source, 1]["live_lower_bound_bytes"]
    single = measured["single"] - output_size
    two = measured["tree2"] - output_size
    assert min(single, two) >= lower
    saving = 100 * (two - single) / two
    remaining = 100 * (two - lower) / measured["tree2"]
    tree_capacity = capacities[one_tree][target, source]
    rows.append((target, source, output_size, lower, single, two,
                 f"{saving:.5f}", f"{remaining:.5f}", tree_capacity))
    savings.append(saving)
    single_tree_savings.append(100 * (tree_capacity - single) / tree_capacity)

with (out / "scope.csv").open("w") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(("target", "source", "output_pack_bytes", "temp_lower_bound",
                     "temp_single", "temp_two_trees", "single_saving_percent",
                     "whole_queue_remaining_upper_bound_percent", "single_tree_replay_bytes"))
    writer.writerows(rows)
print(f"Mean temporary-capacity saving against the two trees: {mean(savings):.4f}%")
print(f"Mean saving against single tree {one_tree} (replay): {mean(single_tree_savings):.4f}%")
