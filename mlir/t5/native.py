"""T5 random ONNX fixture: fixed hint area order, all-output checks, queue events."""

import argparse, csv, json, re, subprocess
import shape_eval as ev
from pathlib import Path
from time import perf_counter
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
for key in ["pre-layout", "iree-bin", "parameters", "out"]:
    p.add_argument("--" + key, type=Path, required=True)
p.add_argument("--trees-json", type=Path)
p.add_argument("--stock-vmfb", type=Path)
p.add_argument("--single-vmfb", type=Path)
p.add_argument("--shapes", default="1,32,32;1,17,95;2,65,31")
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
src = a.pre_layout.read_text()
start = re.search(r"^  util.func[^\n]*@main_graph\$async\(", src, re.M).start()
body = src[start:]
body = body[: body.index("\n  }") + 4]
body = "\n".join(
    line[2:] if line.startswith("  ") else line for line in body.splitlines()
)
env = ev.values(body, 1, 32, 32)
witnesses = []
replace = {}
trees = json.loads(a.trees_json.read_text()) if a.trees_json else None
if trees:
    trees["trees"] = [
        {
            "capacity": plan["capacity_ast"],
            "offsets": [
                plan["offset_asts"][str(i)] for i in range(len(trees["ssa_by_slice"]))
            ],
        }
        for plan in trees["plans"]
    ]


def rewrite(m):
    k = len(witnesses)
    entries = ev.SLICE.findall(m[2])
    spans = [(int(x), int(y)) for x, y, _ in entries]
    n = len(spans)
    sizes = [(env[name] + 63) // 64 * 64 for _, _, name in entries]
    conflict = [[max(l, x) <= min(r, y) for x, y in spans] for l, r in spans]
    order = sorted(
        range(n),
        key=lambda i: (-sizes[i] * (spans[i][1] - spans[i][0] + 1), -sizes[i], i),
    )
    offset = {}
    for i in order:
        pos = 0
        for lo, hi in sorted(
            (v, v + sizes[j]) for j, v in offset.items() if conflict[i][j]
        ):
            if pos + sizes[i] <= lo:
                break
            pos = max(pos, hi)
        offset[i] = pos
    physical = sorted(order, key=lambda i: (offset[i], i))
    pred = {
        i: [j for j in physical[: physical.index(i)] if conflict[i][j]]
        for i in physical
    }
    lines = []
    serial = 0

    def emit(rhs):
        nonlocal serial
        name = f"%native{k}_{serial}"
        serial += 1
        lines.append(f"    {name} = {rhs}")
        return name

    c64 = emit("arith.constant 64 : index")
    zero = emit("arith.constant 0 : index")
    runtime = [emit(f"util.align {name}, {c64} : index") for _, _, name in entries]
    starts = {}
    ends = {}

    def maximum(names):
        if not names:
            return zero
        v = names[0]
        for x in names[1:]:
            v = emit(f"arith.maxui {v}, {x} : index")
        return v

    if trees and k == 1:
        assert [n for _, _, n in entries] == trees["ssa_by_slice"]
        assert [list(x) for x in spans] == trees["intervals"]
        memo = {}

        def lower(expr):
            key = json.dumps(expr, sort_keys=True)
            if key in memo:
                return memo[key]
            if isinstance(expr, int):
                v = zero if expr == 0 else emit(f"arith.constant {expr} : index")
            elif isinstance(expr, str):
                name = expr if expr.startswith("%") else "%" + expr[1:]
                v = runtime[[n for _, _, n in entries].index(name)]
            else:
                args = [lower(x) for x in expr["args"]]
                if expr["op"] == "max":
                    v = maximum(args)
                elif expr["op"] == "add":
                    v = args[0]
                    for x in args[1:]:
                        v = emit(f"arith.addi {v}, {x} : index")
                elif expr["op"] == "mul":
                    v = args[0]
                    for x in args[1:]:
                        v = emit(f"arith.muli {v}, {x} : index")
                else:
                    raise ValueError(expr)
            memo[key] = v
            return v

        assert len(trees["trees"]) == 2
        capacities = [lower(tree["capacity"]) for tree in trees["trees"]]
        alternatives = [[lower(x) for x in tree["offsets"]] for tree in trees["trees"]]
        assert all(len(x) == n for x in alternatives)
        choose = emit(f"arith.cmpi ule, {capacities[0]}, {capacities[1]} : index")
        total = emit(f"arith.select {choose}, {capacities[0]}, {capacities[1]} : index")
        starts = {
            i: emit(
                f"arith.select {choose}, {alternatives[0][i]}, {alternatives[1][i]} : index"
            )
            for i in range(n)
        }
        replace[m[1] + "#0"] = total
        replace.update({m[1] + f"#{i + 1}": starts[i] for i in range(n)})
        witnesses.append(
            {
                "pack": k,
                "slices": n,
                "source": "native Torch Inductor two frozen trees",
                "hints": [[8, 128], [32, 32]],
                "offset_alignment": 64,
                "range_alignment": 64,
            }
        )
        return "\n".join(lines)
    for i in physical:
        starts[i] = maximum([ends[j] for j in pred[i]])
        ends[i] = emit(f"arith.addi {starts[i]}, {runtime[i]} : index")
    total = maximum(list(ends.values()))
    replace[m[1] + "#0"] = total
    replace.update({m[1] + f"#{i + 1}": starts[i] for i in range(n)})
    witnesses.append(
        {
            "pack": k,
            "slices": n,
            "hint": [1, 32, 32],
            "physical": physical,
            "predecessors": pred,
            "intervals": spans,
            "size_ssa": [name for _, _, name in entries],
            "offset_alignment": 64,
            "range_alignment": 64,
        }
    )
    return "\n".join(lines)


modified = ev.PACK.sub(rewrite, src)
for old, new in sorted(replace.items(), key=lambda kv: -len(kv[0])):
    modified = re.sub(re.escape(old) + r"(?!\d)", lambda _: new, modified)
assert len(witnesses) == 2 and "stream.resource.pack" not in modified
variant = "tree2" if trees else "ordered"
(a.out / f"{variant}.mlir").write_text(modified)
(a.out / "order.json").write_text(json.dumps(witnesses, indent=2))


def run(tool, *args):
    cmd = [str(a.iree_bin / tool), *map(str, args)]
    start = perf_counter()
    r = subprocess.run(cmd, text=True, capture_output=True)
    with (a.out / "tool-times.csv").open("a") as f:
        writer = csv.writer(f, lineterminator="\n")
        if f.tell() == 0:
            writer.writerow(("tool", "wall_seconds"))
        writer.writerow((tool, f"{perf_counter() - start:.6f}"))
    with (a.out / "commands.log").open("a") as f:
        f.write(repr(cmd) + "\n" + r.stdout + r.stderr)
    if r.returncode:
        raise RuntimeError(r.stderr)
    return r.stdout


pipe = "builtin.module(util.func(iree-stream-layout-slices,canonicalize),iree-util-propagate-subranges,iree-stream-cleanup-pipeline,iree-stream-automatic-reference-counting,iree-stream-annotate-constant-transient-size,iree-stream-verify-lowering-to-cmd,iree-stream-optimization-pipeline,iree-stream-cleanup-pipeline,symbol-dce)"
modules = {"stock": a.stock_vmfb} if a.stock_vmfb else {}
if trees:
    assert a.single_vmfb is not None
    modules["single"] = a.single_vmfb
build = [(variant, a.out / f"{variant}.mlir")]
if not a.stock_vmfb:
    build.insert(0, ("stock", a.pre_layout))
for name, path in build:
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
    modules[name] = a.out / f"{name}.vmfb"
    print("VMFB", modules[name], flush=True)
rows = []
for b, t, s in [tuple(map(int, x.split(","))) for x in a.shapes.split(";")]:
    arrays = [
        ((np.arange(b * s).reshape(b, s) % 7) != 3).astype(np.int64),
        (np.arange(b * t, dtype=np.int64).reshape(b, t) * 37 + 11) % 32100,
        (
            np.sin(np.arange(b * s * 32, dtype=np.float32) * np.float32(0.03125)) * 0.25
        ).reshape(b, s, 32),
    ]
    inputs = []
    for i, x in enumerate(arrays):
        path = a.out / f"input-{b}-{t}-{s}-{i}.npy"
        np.save(path, x)
        inputs.append("--input=@" + str(path))
    all_outputs = {}
    for name, module in modules.items():
        prefix = a.out / f"{name}-{b}-{t}-{s}"
        outputs = ["--output=@" + str(prefix) + f"-{i}.npy" for i in range(21)]
        bundle = Path(str(prefix) + ".ireeprof")
        run(
            "iree-run-module",
            "--module=" + str(module),
            "--parameters=model=" + str(a.parameters),
            "--device=local-task",
            "--function=main_graph",
            *inputs,
            *outputs,
            "--device_profiling_mode=memory-events",
            "--device_profiling_output=" + str(bundle),
        )
        tensors = [np.load(str(prefix) + f"-{i}.npy") for i in range(21)]
        assert all(np.all(np.isfinite(x)) for x in tensors)
        all_outputs[name] = tensors
        report = run("iree-profile", "memory", "--format=jsonl", bundle)
        Path(str(prefix) + ".jsonl").write_text(report)
        records = [json.loads(x) for x in report.splitlines()]
        (summary,) = [x for x in records if x["type"] == "memory_summary"]
        assert summary["dropped_records"] == summary["truncated_chunks"] == 0
        (pool,) = [
            x
            for x in records
            if x["type"] == "memory_pool" and x["kind"] == "queue_allocation"
        ]
        (backing,) = [
            x
            for x in records
            if x["type"] == "memory_pool" and x["kind"] == "pool_reservation"
        ]
        row = {
            "batch": b,
            "target": t,
            "source": s,
            "module": name,
            "queue_requests": pool["lifecycle_total_open_count"],
            "queue_requested_bytes": pool["lifecycle_total_open_bytes"],
            "queue_live_peak_bytes": pool["lifecycle_high_water_bytes"],
            "pool_committed_peak_bytes": backing["pool_bytes_committed_high_water"],
        }
        rows.append(row)
    pairs = [
        (x, y)
        for name, values in all_outputs.items()
        for x, y in zip(all_outputs["stock"], values)
    ]
    equal = all(
        x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
        for x, y in pairs
    )
    diff = max(float(np.max(np.abs(x - y))) for x, y in pairs)
    for r in rows[-len(modules) :]:
        r.update(outputs=21, bitwise_equal=equal, max_abs=diff)
    print(rows[-len(modules) :], flush=True)
    assert equal
with (a.out / "results.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=rows[0])
    w.writeheader()
    w.writerows(rows)
