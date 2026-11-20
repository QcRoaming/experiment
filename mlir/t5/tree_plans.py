"""Use raw TorchInductor AllocationPools and select two plans using only hint shapes."""

import argparse
import csv
from itertools import combinations
import json
from pathlib import Path

import sympy as sp
import torch
from torch._inductor import config
from torch._inductor.codegen import memory_planning as mp

HINTS = [(32, 32), (8, 128), (128, 8)]


class Buffer:
    def __init__(self, index):
        self.index = index

    def get_device(self):
        return torch.device("cpu")

    def get_name(self):
        return str(self.index)


def expression(tree, names, offset, out):
    if isinstance(tree, mp.Allocation):
        i = tree.node.index
        out[str(i)] = offset
        return names[i]
    if isinstance(tree, mp.Empty):
        return 0
    if isinstance(tree, mp.TemporalSplit):
        return {
            "op": "max",
            "args": [expression(x, names, offset, out) for x in tree.allocations],
        }
    if isinstance(tree, mp.SpatialSplit):
        left = expression(tree.left, names, offset, out)
        right = expression(
            tree.right, names, {"op": "add", "args": [offset, left]}, out
        )
        return {"op": "add", "args": [left, right]}
    raise ValueError(type(tree))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packs", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    assert torch.__version__.split("+")[0] == "2.5.1"
    cases = {
        (c["target"], c["source"]): c
        for c in json.loads(args.packs.read_text())["cases"]
        if c["pack"] == 1
    }
    plans, capacity, rows = {}, {}, []
    for hint in HINTS:
        h = cases[hint]
        names = h["size_ssa"]
        symbols = {
            n: sp.Symbol("size" + n[1:], positive=True, integer=True) for n in names
        }
        for priority in ("area-stable", "default"):
            key = (*hint, priority)
            blocks = [
                mp.Allocation(
                    Buffer(i),
                    mp.LiveRange(a, b + 1),
                    h["aligned_sizes"][i],
                    symbols[names[i]],
                )
                for i, (a, b) in enumerate(h["intervals"])
            ]
            with config.patch(memory_pool="intermediates"):
                pools = mp.AllocationPools()

                def rank(b):
                    if priority == "default":
                        return -b.size_hint, -len(b.live_range)
                    return (-b.size_hint * len(b.live_range),)

                for b in sorted(blocks, key=rank):
                    pools.allocate(b)
                pools.finalize()
            (pool,) = [p for ps in pools.device_to_pools.values() for p in ps]
            expr = pool.root.get_symbolic_size()
            native_offsets = [sp.sympify(b.offset) for b in blocks]
            capacity[key] = {}
            for shape, case in cases.items():
                sizes = case["aligned_sizes"]
                env = {symbols[n]: v for n, v in zip(names, sizes)}
                cap = int(expr.xreplace(env))
                offsets = [int(o.xreplace(env)) for o in native_offsets]
                for i, (a, b) in enumerate(h["intervals"]):
                    assert offsets[i] % 64 == 0 and 0 <= offsets[i] <= cap - sizes[i]
                    for j, (c, d) in enumerate(h["intervals"][:i]):
                        if max(a, c) <= min(b, d):
                            assert (
                                offsets[i] + sizes[i] <= offsets[j]
                                or offsets[j] + sizes[j] <= offsets[i]
                            )
                capacity[key][shape] = cap
                rows.append((*key, *shape, cap, case["live_lower_bound_bytes"]))
            offsets = {}
            total = expression(pool.root, names, 0, offsets)
            plans[key] = {
                "hint": hint,
                "priority": priority,
                "capacity_ast": total,
                "offset_asts": offsets,
            }

    def score(pair):
        ratios = [
            min(capacity[p][s] for p in pair) / cases[s]["live_lower_bound_bytes"]
            for s in HINTS
        ]
        return max(ratios), sum(ratios), pair

    chosen = min(combinations(sorted(plans), 2), key=score)
    output = {
        "ssa_by_slice": names,
        "intervals": h["intervals"],
        "plans": [plans[key] for key in chosen],
    }
    (args.out / "trees.json").write_text(json.dumps(output, separators=(",", ":")))
    with (args.out / "tree-capacities.csv").open("w") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(
            (
                "hint_target",
                "hint_source",
                "priority",
                "target",
                "source",
                "capacity",
                "live_lower_bound",
            )
        )
        w.writerows(rows)
    print("Selected trees:", chosen)


if __name__ == "__main__":
    main()
