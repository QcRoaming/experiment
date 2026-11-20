"""Evaluate actual IREE Before/After LayoutSlicesPass dumps for dynamic BERT."""

import argparse
import csv
import operator
import re
import sys
from itertools import product
from pathlib import Path

PACK = re.compile(
    r"  (%\w+):\d+ = stream\.resource\.pack [^\n]*slices\(\{(.*?)\}\) : index", re.S
)
SLICE = re.compile(r"\[(\d+), (\d+)\] = (%\w+)")
ALLOCATION = re.compile(
    r"stream\.resource\.alloca [^\n]*?=> !stream\.resource<\w+>\{(%\w+(?:#\d+)?)\}"
)
OPS = {
    "muli": operator.mul,
    "addi": operator.add,
    "maxui": max,
    "minsi": min,
    "divsi": operator.floordiv,
}


def model_passes(dump):
    sections = re.split(
        r"^// -----// IR Dump (Before|After) [^\n]*\n", dump, flags=re.M
    )
    pairs = [(sections[i], sections[i + 1]) for i in range(1, len(sections), 2)]
    candidates = [
        (a, pairs[i + 1])
        for i, (stage, a) in enumerate(pairs)
        if stage == "Before"
        and "@main_graph$async(" in a.splitlines()[0]
        and PACK.search(a)
    ]
    assert len(candidates) == 1 and candidates[0][1][0] == "After"
    before, (_, after) = candidates[0]
    return before, after


def values(body, batch, sequence):
    env = {}
    for line in body.splitlines():
        match = re.match(r"  (%\w+) = (.+)", line)
        if not match:
            continue
        var, rhs = match.groups()
        if m := re.match(r"arith.constant (\d+) :", rhs):
            env[var] = int(m[1])
        elif m := re.match(
            r"hal.buffer_view.dim<%arg[012] : !hal.buffer_view>\[([01])\]", rhs
        ):
            env[var] = (batch, sequence)[int(m[1])]
        elif m := re.match(r"(?:arith\.(\w+)|util\.(align)) (%\w+), (%\w+)\b", rhs):
            op, align, left, right = m.groups()
            if left not in env or right not in env:
                continue
            x, y = env[left], env[right]
            if align:
                env[var] = (x + y - 1) // y * y
            elif op in OPS:
                assert x >= 0 and y >= 0
                env[var] = OPS[op](x, y)
    return env


def fixed_slots(intervals):
    # First free slot in start-time order; neither sizes nor input shape consulted.
    groups = []
    for i, (start, _) in sorted(enumerate(intervals), key=lambda item: item[1][0]):
        slot = next((g for g in groups if intervals[g[-1]][1] < start), None)
        if slot is None:
            slot = []
            groups.append(slot)
        slot.append(i)
    return groups


def screen(dump):
    before, after = model_passes(dump)
    before_allocs, after_allocs = ALLOCATION.findall(before), ALLOCATION.findall(after)
    assert len(before_allocs) == len(after_allocs)
    packs = []
    for match in PACK.finditer(before):
        entries = SLICE.findall(match[2])
        assert len(entries) == match[2].count("=")
        intervals = [(int(a), int(b)) for a, b, _ in entries]
        name = match[1] + "#0"
        assert before_allocs.count(name) == 1
        total = after_allocs[before_allocs.index(name)]
        packs.append((intervals, [s for _, _, s in entries], total))
    for batch, seq in product((1, 2, 4), (16, 64, 128)):
        initial, lowered = values(before, batch, seq), values(after, batch, seq)
        for i, (intervals, names, total) in enumerate(packs):
            # This CPU configuration aligns offsets and slab lengths to 64.
            sizes = [((initial[n] + 63) // 64) * 64 for n in names]
            lb = max(
                sum(
                    size
                    for (start, end), size in zip(intervals, sizes)
                    if start <= t <= end
                )
                for t, _ in intervals
            )
            slots = fixed_slots(intervals)
            fixed = sum(max(sizes[j] for j in slot) for slot in slots)
            assert lb <= min(fixed, lowered[total])
            yield batch, seq, i, len(names), lowered[total], fixed, lb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", type=Path)
    args = parser.parse_args()
    out = csv.writer(sys.stdout)
    out.writerow(
        (
            "batch",
            "sequence",
            "pack",
            "slices",
            "iree_bytes",
            "first_free_slot_bytes",
            "live_lower_bound_bytes",
        )
    )
    out.writerows(screen(args.dump.read_text()))
