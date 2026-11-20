#!/usr/bin/env bash
# Run the pinned compiler's Tensor-to-Flow conversion on host functions.
set -euo pipefail
bin=$1
input=$2
out=$3
opts=(--iree-opt-level=O2 --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=host)
mkdir -p "$out"
"$bin/iree-compile" "$input" "${opts[@]}" --compile-to=flow -o "$out/flow.mlir"
"$bin/iree-opt" "$out/flow.mlir" \
  --pass-pipeline='builtin.module(util.func(iree-flow-convert-to-flow))' \
  -o "$out/host-flow.mlir"
"$bin/iree-compile" "$out/host-flow.mlir" "${opts[@]}" --compile-from=flow -o "$out/host-reflow.vmfb"
