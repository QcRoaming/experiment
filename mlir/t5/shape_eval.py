import re, json, argparse
from pathlib import Path
from itertools import product

PACK = re.compile(
    r"  (%\w+):\d+ = stream\.resource\.pack [^\n]*slices\(\{(.*?)\}\) : index", re.S
)
SLICE = re.compile(r"\[(\d+), (\d+)\] = (%[\w-]+)")


def values(body, batch, target, source):
    env = {}
    unknown = {}
    for line in body.splitlines():
        m = re.match(r"  (%[\w-]+) = (.+)", line)
        if not m:
            continue
        var, rhs = m.groups()
        refs = re.findall(r"%[\w-]+", rhs)
        if m := re.match(r"arith.constant (-?\d+) :", rhs):
            env[var] = int(m[1])
        elif m := re.match(
            r"hal.buffer_view.dim<%arg([012]) : !hal.buffer_view>\[([01])\]", rhs
        ):
            arg, dim = map(int, m.groups())
            env[var] = batch if dim == 0 else target if arg == 1 else source
        elif rhs.startswith("arith.") or rhs.startswith("util.align"):
            if any(x not in env for x in refs):
                unknown[var] = rhs
                continue
            x = [env[r] for r in refs]
            op = rhs.split()[0]
            if op == "arith.index_cast":
                v = x[0]
            elif op == "arith.muli":
                v = x[0] * x[1]
            elif op == "arith.addi":
                v = x[0] + x[1]
            elif op in ["arith.divui", "arith.divsi"]:
                v = abs(x[0]) // abs(x[1]) * (1 if x[0] * x[1] >= 0 else -1)
            elif op in ["arith.maxui", "arith.maxsi"]:
                v = max(x)
            elif op in ["arith.minui", "arith.minsi"]:
                v = min(x)
            elif op == "arith.cmpi":
                pred = rhs.split()[1].rstrip(",")
                v = {
                    "slt": x[0] < x[1],
                    "ult": x[0] < x[1],
                    "eq": x[0] == x[1],
                    "sle": x[0] <= x[1],
                }[pred]
            elif op == "arith.select":
                v = x[1] if x[0] else x[2]
            elif op == "util.align":
                v = (x[0] + x[1] - 1) // x[1] * x[1]
            else:
                unknown[var] = rhs
                continue
            env[var] = v
        else:
            unknown[var] = rhs
    env["_unknown"] = unknown
    return env


def packs(body):
    return [
        {
            "name": m[1],
            "intervals": [[int(a), int(b)] for a, b, _ in SLICE.findall(m[2])],
            "size_ssa": [n for _, _, n in SLICE.findall(m[2])],
        }
        for m in PACK.finditer(body)
    ]


def cases(body):
    ps = packs(body)
    out = []
    shapes = list(product([8, 32, 128], repeat=2)) + [
        (17, 95),
        (95, 17),
        (63, 65),
        (65, 63),
    ]
    for t, s in dict.fromkeys(shapes):
        env = values(body, 1, t, s)
        for i, p in enumerate(ps):
            missing = [n for n in p["size_ssa"] if n not in env]
            assert not missing, [(n, env["_unknown"].get(n)) for n in missing]
            raw = [env[n] for n in p["size_ssa"]]
            sizes = [(v + 63) // 64 * 64 for v in raw]
            lb = max(
                sum(z for z, (a, b) in zip(sizes, p["intervals"]) if a <= time <= b)
                for time, _ in p["intervals"]
            )
            out.append(
                dict(
                    batch=1,
                    target=t,
                    source=s,
                    pack=i,
                    **p,
                    raw_sizes=raw,
                    aligned_sizes=sizes,
                    alignment=64,
                    interval_end="inclusive",
                    live_lower_bound_bytes=lb,
                )
            )
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("body", type=Path)
    parser.add_argument("output", type=Path)
    a = parser.parse_args()
    a.output.write_text(
        json.dumps(
            {
                "axis_semantics": {
                    "arg0": ["batch", "source"],
                    "arg1": ["batch", "target"],
                    "arg2": ["batch", "source", 32],
                },
                "cases": cases(a.body.read_text()),
            },
            indent=2,
        )
    )
