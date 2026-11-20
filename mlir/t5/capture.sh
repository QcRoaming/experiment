#!/usr/bin/env bash
set -euo pipefail
# Usage: capture.sh IREE_BIN MODEL_MLIR OUTPUT_DIRECTORY
IREE_BIN=$1
MODEL_MLIR=$2
OUTPUT_DIRECTORY=$3
mkdir -p "$OUTPUT_DIRECTORY"
"$IREE_BIN/iree-compile" "$MODEL_MLIR" --iree-opt-level=O2 --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=host --iree-stream-partitioning-favor=min-peak-memory --compile-to=stream --mlir-disable-threading --mlir-print-ir-before=iree-stream-layout-slices --mlir-print-ir-module-scope --mlir-elide-elementsattrs-if-larger=1000000000 --mlir-elide-resource-strings-if-larger=1000000000 -o "$OUTPUT_DIRECTORY/stock-stream.mlir" 2>"$OUTPUT_DIRECTORY/pre.log"
python3 - "$OUTPUT_DIRECTORY" <<'PY'
from pathlib import Path
import sys
out=Path(sys.argv[1])
parts=(out/'pre.log').read_text().split('// -----// IR Dump Before ')
s=[x for x in parts if x and '@main_graph$async' in x.splitlines()[0]][-1]
s=s[s.index('\n')+1:]
assert s.count('stream.resource.pack')==2 and 'cf.assert' not in s and '...' not in s
(out/'pre-layout.mlir').write_text(s)
PY
