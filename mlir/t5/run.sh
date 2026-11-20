#!/usr/bin/env bash
set -euo pipefail
src=$(cd "$(dirname "$0")" && pwd)
compiler=${IREE_BIN:+$IREE_BIN/iree-compile}
compiler=${compiler:-$(command -v iree-compile)}
bin=$(dirname "$(realpath "$compiler")")
out=$(realpath -m "${1:?Pass an empty output directory}")
model=${T5_ONNX:+$(realpath "$T5_ONNX")}
mkdir -p "$out"
if [[ -n $(ls -A "$out") ]]; then
  echo "Choose an empty output directory: $out" >&2
  exit 2
fi
cd "$out"
export PYTHONDONTWRITEBYTECODE=1
if [[ -n $model ]]; then cp "$model" model.onnx; else
  curl -fsSL --retry 1 --max-time 90 'https://huggingface.co/onnx-internal-testing/tiny-random-T5ForConditionalGeneration-ONNX/resolve/5d8961085d0e950b1030551a431128f8a2437f4d/onnx/decoder_model.onnx' -o model.onnx
fi
echo '6160a119770043ac0b35e4c1c27e98d01016ea3fa11c46d3416debd23e33ab99  model.onnx' | sha256sum -c -
echo 'stage,wall_seconds' > stages.csv
stage() {
  local label=$1
  shift
  /usr/bin/time -a -o stages.csv -f "$label,%e" "$@"
}
stage import "$bin/iree-import-onnx" model.onnx --externalize-params -o model.mlir
stage capture bash "$src/capture.sh" "$bin" model.mlir "$out"
python3 - "$src" "$out" <<'PY'
from pathlib import Path
import json,sys
sys.path.insert(0,sys.argv[1])
from shape_eval import cases
out=Path(sys.argv[2])
s=(out/'pre-layout.mlir').read_text().split('  util.func public @main_graph$async(')[1].split('\n  }')[0]
body='\n'.join(line[2:] for line in s.splitlines()[1:])
(out/'packs.json').write_text(json.dumps({'cases':cases(body)}))
PY
stage tree_planning_and_offline_checks "${TORCH_PYTHON:-python3}" "$src/tree_plans.py" "$out/packs.json" "$out"
shapes='1,17,95;1,95,17;1,63,65;1,65,63'
common=(--pre-layout "$out/pre-layout.mlir" --iree-bin "$bin" --parameters "$out/model_params.irpa" --shapes "$shapes")
"$bin/python" "$src/native.py" "${common[@]}" --out "$out/single"
"$bin/python" "$src/native.py" "${common[@]}" --out "$out/three" --trees-json "$out/trees.json" --stock-vmfb "$out/single/stock.vmfb" --single-vmfb "$out/single/ordered.vmfb"
python3 "$src/scope.py" "$out"
