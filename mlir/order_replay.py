"""Frozen B1S128 size*lifetime first-fit order, lowered to symbolic predecessor maxima."""

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from itertools import product
import numpy as np

import arena

p = argparse.ArgumentParser(description=__doc__)
for x in ["dump", "pre-layout", "original", "parameters", "iree-bin", "out"]:
    p.add_argument("--" + x, type=Path, required=True)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)

before, _ = arena.model_passes(a.dump.read_text())
(refpack,) = arena.PACK.finditer(before)
refentries = arena.SLICE.findall(refpack[2])
spans = [(int(l), int(r)) for l, r, _ in refentries]
assert len(spans) == 27
conflict = [[max(l, x) <= min(r, y) for x, y in spans] for l, r in spans]
env = arena.values(before, 1, 128)
ref = [(env[n] + 63) // 64 * 64 for _, _, n in refentries]
order = sorted(
    range(27), key=lambda i: (-ref[i] * (spans[i][1] - spans[i][0] + 1), -ref[i], i)
)
offsets = {}
for i in order:
    pos = 0
    for lo, hi in sorted((v, v + ref[j]) for j, v in offsets.items() if conflict[i][j]):
        if pos + ref[i] <= lo:
            break
        pos = max(pos, hi)
    offsets[i] = pos
physical = sorted(order, key=lambda i: (offsets[i], i))
pred = {
    i: [j for j in physical[: physical.index(i)] if conflict[i][j]] for i in physical
}
src = a.pre_layout.read_text()
(m,) = arena.PACK.finditer(src)
entries = arena.SLICE.findall(m[2])
assert entries == refentries
lines = []
serial = 0


def emit(rhs):
    global serial
    name = f"%order_{serial}"
    serial += 1
    lines.append(f"    {name} = {rhs}")
    return name


c64 = emit("arith.constant 64 : index")
zero = emit("arith.constant 0 : index")
sizes = [emit(f"util.align {n}, {c64} : index") for _, _, n in entries]
starts = {}
ends = {}


def maximum(names):
    result = zero
    for n in names:
        result = emit(f"arith.maxui {result}, {n} : index")
    return result


for i in physical:
    starts[i] = maximum([ends[j] for j in pred[i]])
    ends[i] = emit(f"arith.addi {starts[i]}, {sizes[i]} : index")
total = maximum(list(ends.values()))
replacements = {m[1] + "#0": total}
replacements.update({m[1] + f"#{i + 1}": starts[i] for i in range(27)})
src = src[: m.start()] + "\n".join(lines) + src[m.end() :]
for old, new in sorted(replacements.items(), key=lambda kv: -len(kv[0])):
    src = re.sub(re.escape(old) + r"(?!\d)", lambda _: new, src)
assert "stream.resource.pack" not in src
(a.out / "ordered.mlir").write_text(src)
(a.out / "order.json").write_text(
    json.dumps(
        {
            "hint": [1, 128],
            "key": "(-aligned_bytes*inclusive_lifetime,-aligned_bytes,index)",
            "physical": physical,
            "hint_offsets": offsets,
            "predecessors": pred,
        },
        indent=2,
    )
)


def run(tool, *args):
    cmd = [str(a.iree_bin / tool), *map(str, args)]
    r = subprocess.run(cmd, text=True, capture_output=True)
    with (a.out / "commands.log").open("a") as f:
        f.write(repr(cmd) + "\n" + r.stdout + r.stderr)
    if r.returncode:
        raise RuntimeError(r.stderr)


pipe = "builtin.module(util.func(iree-stream-layout-slices,canonicalize),iree-util-propagate-subranges,iree-stream-cleanup-pipeline,iree-stream-automatic-reference-counting,iree-stream-annotate-constant-transient-size,iree-stream-verify-lowering-to-cmd,iree-stream-optimization-pipeline,iree-stream-cleanup-pipeline,symbol-dce)"
for name, path in [("stock", a.pre_layout), ("ordered", a.out / "ordered.mlir")]:
    run(
        "iree-opt", path, "--pass-pipeline=" + pipe, "-o", a.out / f"{name}.stream.mlir"
    )
    run(
        "iree-compile",
        a.out / f"{name}.stream.mlir",
        "--compile-from=stream",
        "--iree-opt-level=O2",
        "--iree-hal-target-backends=llvm-cpu",
        "--iree-llvmcpu-target-cpu=host",
        "--iree-stream-partitioning-favor=min-peak-memory",
        "-o",
        a.out / f"{name}.vmfb",
    )
    print("VMFB", a.out / f"{name}.vmfb", flush=True)


def byte_equal(left, right):
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.tobytes() == right.tobytes()
    )


rows = []
for b, s in [*product((1, 2, 4), (16, 64, 128)), (3, 97), (1, 256)]:
    pos = np.arange(b * s, dtype=np.int64).reshape(b, s)
    inputs = []
    for i, x in enumerate(
        [(pos * 37 + 11) % 30522, (pos % 7 != 3).astype(np.int64), pos % 2]
    ):
        path = a.out / f"in-{b}-{s}-{i}.npy"
        np.save(path, x)
        inputs.append("--input=@" + str(path))
    outputs = []
    for name, module in [
        ("original", a.original),
        ("stock", a.out / "stock.vmfb"),
        ("ordered", a.out / "ordered.vmfb"),
    ]:
        path = a.out / f"{name}-{b}-{s}.npy"
        run(
            "iree-run-module",
            "--module=" + str(module),
            "--parameters=model=" + str(a.parameters),
            "--device=local-task",
            "--function=main_graph",
            *inputs,
            "--output=@" + str(path),
        )
        outputs.append(np.load(path))
    orig, stock, new = outputs
    env = arena.values(before, b, s)
    sz = [(env[n] + 63) // 64 * 64 for _, _, n in refentries]
    finish = {}
    for i in physical:
        finish[i] = max((finish[j] for j in pred[i]), default=0) + sz[i]
    lower = max(sum(v for (l, r), v in zip(spans, sz) if l <= t <= r) for t, _ in spans)
    row = {
        "batch": b,
        "sequence": s,
        "shape_group": "extended" if (b, s) in [(3, 97), (1, 256)] else "grid",
        "original_stock_equal": byte_equal(orig, stock),
        "original_ordered_equal": byte_equal(orig, new),
        "stock_ordered_equal": byte_equal(stock, new),
        "max_abs": float(np.max(np.abs(orig - new))),
        "ordered_bytes": max(finish.values()),
        "live_lower_bound": lower,
    }
    rows.append(row)
    print(row, flush=True)
with (a.out / "results.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=rows[0])
    w.writeheader()
    w.writerows(rows)
assert all(r["original_stock_equal"] and r["original_ordered_equal"] for r in rows)
