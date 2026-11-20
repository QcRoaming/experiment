import sys, re, csv, json, hashlib, argparse
import arena
from pathlib import Path
import torch, sympy as sp
from torch._inductor import config
import torch._inductor.codegen.memory_planning as mp

parser = argparse.ArgumentParser()
parser.add_argument("--dump", type=Path, required=True)
parser.add_argument("--hint-batch", type=int, default=1)
parser.add_argument("--hint-sequence", type=int, default=64)
parser.add_argument(
    "--priority", choices=("default", "area-stable", "area-bytes"), default="default"
)
args = parser.parse_args()
B, S = sp.symbols("B S", integer=True, positive=True)
before, after = arena.model_passes(args.dump.read_text())


def symvalues(body):
    env = {}
    for line in body.splitlines():
        m = re.match(r"  (%\w+) = (.+)", line)
        if not m:
            continue
        var, rhs = m.groups()
        if m := re.match(r"arith.constant (\d+) :", rhs):
            env[var] = sp.Integer(m[1])
        elif m := re.match(
            r"hal.buffer_view.dim<%arg[012] : !hal.buffer_view>\[([01])\]", rhs
        ):
            env[var] = (B, S)[int(m[1])]
        elif m := re.match(r"(?:arith\.(\w+)|util\.(align)) (%\w+), (%\w+)\b", rhs):
            op, al, l, r = m.groups()
            if l not in env or r not in env:
                continue
            x, y = env[l], env[r]
            if al:
                env[var] = sp.ceiling(x / y) * y
            elif op == "muli":
                env[var] = x * y
            elif op == "addi":
                env[var] = x + y
            elif op == "maxui":
                env[var] = sp.Max(x, y)
            elif op == "minsi":
                env[var] = sp.Min(x, y)
            elif op == "divsi":
                env[var] = sp.floor(x / y)
    return env


class Node:
    def __init__(self, name):
        self.name = name

    def get_device(self):
        return torch.device("cpu")

    def get_name(self):
        return self.name


sym = symvalues(before)
hint = {B: args.hint_batch, S: args.hint_sequence}
packs = []
with config.patch(memory_pool="intermediates"):
    for match in arena.PACK.finditer(before):
        entries = arena.SLICE.findall(match[2])
        blocks = []
        for j, (a, b, n) in enumerate(entries):
            size = sp.ceiling(sym[n] / 64) * 64
            blocks.append(
                mp.Allocation(
                    Node(n + "_" + str(j)),
                    mp.LiveRange(int(a), int(b) + 1),
                    int(size.subs(hint)),
                    size,
                )
            )
        pool = mp.AllocationPools()

        def priority(b):
            if args.priority == "default":
                return -b.size_hint, -len(b.live_range)
            return -b.size_hint * len(b.live_range), (
                -b.size_hint if args.priority == "area-bytes" else 0
            )

        for block in sorted(blocks, key=priority):
            pool.allocate(block)
        pool.finalize()
        packs.append((blocks, pool))
print(
    json.dumps(
        {
            "torch_version": torch.__version__,
            "git_version": torch.version.git_version,
            "source": mp.__file__,
            "sha256": hashlib.sha256(Path(mp.__file__).read_bytes()).hexdigest(),
            "shim": False,
            "hint": f"B{args.hint_batch}S{args.hint_sequence}",
            "alignment": 64,
            "inclusive_to_exclusive": True,
        }
    ),
    file=sys.stderr,
)
out = csv.writer(sys.stdout)
out.writerow(
    [
        "batch",
        "sequence",
        "pack",
        "slices",
        "iree_bytes",
        "first_free_slot_bytes",
        "live_lower_bound_bytes",
        "inductor_bytes",
        "pools",
    ]
)
for b, s, p, n, iree, fixed, lb in arena.screen(args.dump.read_text()):
    blocks, pp = packs[p]
    env = {B: b, S: s}
    capacity = sum(
        int(pool.root.get_symbolic_size().subs(env))
        for pools in pp.device_to_pools.values()
        for pool in pools
    )
    for i, x in enumerate(blocks):
        xo = int(sp.sympify(x.offset).subs(env))
        xs = int(x.symbolic_size.subs(env))
        assert xo % 64 == 0
        for y in blocks[i + 1 :]:
            if (
                x.live_range.begin < y.live_range.end
                and y.live_range.begin < x.live_range.end
            ):
                yo = int(sp.sympify(y.offset).subs(env))
                ys = int(y.symbolic_size.subs(env))
                assert x.pool is not y.pool or xo + xs <= yo or yo + ys <= xo
    assert capacity >= lb
    out.writerow(
        [
            b,
            s,
            p,
            n,
            iree,
            fixed,
            lb,
            capacity,
            sum(map(len, pp.device_to_pools.values())),
        ]
    )
