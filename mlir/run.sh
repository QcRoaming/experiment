#!/usr/bin/env bash
# Reproduce the BERT pilot. All generated files and weights stay under OUT.
set -eu -o pipefail
src=$(cd "$(dirname "$0")" && pwd)
bin=$(cd "${IREE_BIN:-$(dirname "$(command -v iree-compile)")}" && pwd)
export PYTHONDONTWRITEBYTECODE=1
out=${1:?pass a new output directory}
mkdir -p "$out"
cd "$out"
curl -fsSL --retry 1 --max-time 90 -o bert.onnx 'https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-onnx/resolve/da60a9ed87aeb7e57d9b15be5d021b28d153f584/onnx/model.onnx'
echo 'c26b1b3e2f210b1e49e7a1a5067f6451c7c9039ac5a91848c5d455372fd5d618  bert.onnx' | sha256sum -c -
"$bin/iree-import-onnx" bert.onnx --externalize-params -o bert.mlir
cpu=(--iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=host)
mkdir -p release/phases
flags=(--iree-opt-level=O2 --iree-stream-partitioning-favor=min-peak-memory)
"$bin/iree-compile" bert.mlir "${cpu[@]}" "${flags[@]}" --compile-to=stream --mlir-disable-threading \
  --dump-compilation-phases-to="release/phases" --mlir-print-ir-before=iree-stream-layout-slices \
  --mlir-print-ir-after=iree-stream-layout-slices -o "release/stream.mlir" 2>"release/passes.log"
"$bin/iree-compile" "release/stream.mlir" "${cpu[@]}" "${flags[@]}" --compile-from=stream -o "release/base.vmfb"
"$bin/iree-compile" bert.mlir "${cpu[@]}" "${flags[@]}" --compile-to=stream --mlir-disable-threading \
  --mlir-print-ir-module-scope --mlir-print-ir-before=iree-stream-layout-slices \
  --mlir-elide-elementsattrs-if-larger=1000000000 --mlir-elide-resource-strings-if-larger=1000000000 \
  -o /dev/null 2>"release/module.log"
python3 - release <<'PY'
from pathlib import Path
import sys
folder = Path(sys.argv[1])
items = (folder / "module.log").read_text().split("// -----// IR Dump Before ")
block = [s for s in items if s and "@main_graph$async" in s.splitlines()[0]][-1]
module = block.split("\n", 1)[1]
assert "..." not in module and "stream.resource.pack" in module
(folder / "pre.mlir").write_text(module)
PY
"$bin/python" "$src/order_replay.py" --dump "release/passes.log" --pre-layout "release/pre.mlir" \
  --original "release/base.vmfb" --parameters bert_params.irpa --iree-bin "$bin" --out "release/out"
python3 "$src/profile_memory.py" "$bin" bert_params.irpa release/out/{stock,ordered}.vmfb >release/ordered-memory.csv
