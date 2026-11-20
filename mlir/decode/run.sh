#!/usr/bin/env bash
# Requires IREE compiler/runtime CLI, and Python with numpy and onnxruntime.
# Optional BART_ONNX/T5_ONNX point to cached original public ONNX files.
set -euo pipefail
src=$(cd "$(dirname "$0")" && pwd)
: "${IREE_BIN:=$(dirname "$(command -v iree-compile)")}"
bin=$(cd "$IREE_BIN" && pwd)
python=${PYTHON:-python3}
mkdir -p "${1:?Pass an output directory}"
out=$(realpath "$1")
if [[ -n $(ls -A "$out") ]]; then
  echo "Choose an empty output directory: $out" >&2
  exit 2
fi
export PYTHONDONTWRITEBYTECODE=1

for model in bart t5; do
  work="$out/$model"
  mkdir -p "$work"
  if [[ $model == bart ]]; then
    repo=onnx-internal-testing/tiny-random-BartForConditionalGeneration-ONNX
    revision=1b79f810bb092a83539dfea358459e82dfed197e
    sha=c73e86bd6a799d62987f6678256d636e1c920b27f493b02d6a4ff0a12b5b8fa8
    cached=${BART_ONNX:-}
  else
    repo=onnx-internal-testing/tiny-random-T5ForConditionalGeneration-ONNX
    revision=5d8961085d0e950b1030551a431128f8a2437f4d
    sha=17fb0c797bb755e0f4728a60628d639d9627abee5e79342cdbcc8b82673c2c31
    cached=${T5_ONNX:-}
  fi
  url="https://huggingface.co/$repo/resolve/$revision/onnx/decoder_with_past_model.onnx"
  if [[ -n $cached ]]; then
    cp "$cached" "$work/model.onnx"
  elif [[ ! -f $work/model.onnx ]]; then
    curl -fsSL --retry 1 --max-time 90 "$url" -o "$work/model.onnx"
  fi
  echo "$sha  $work/model.onnx" | sha256sum -c -
  printf '%s\n' "$url" "$sha" > "$work/model-source.txt"
  "$bin/iree-compile" --version > "$work/compiler-version.txt"
  (cd "$work" && "$bin/iree-import-onnx" model.onnx --externalize-params -o model.mlir)
  : > "$work/compiles.tsv"
  printf 'setting\tsuccess\tseconds\n' >> "$work/compiles.tsv"
  modules=()
  for setting in O0 O0-strip O2 O2-global-O0; do
    case "$setting" in
      O0) flags=(--iree-opt-level=O0) ;;
      O0-strip) flags=(--iree-opt-level=O0 --iree-opt-strip-assertions=true) ;;
      O2) flags=(--iree-opt-level=O2) ;;
      O2-global-O0) flags=(--iree-opt-level=O2 --iree-global-optimization-opt-level=O0) ;;
    esac
    start=$SECONDS
    if "$bin/iree-compile" "$work/model.mlir" "${flags[@]}" \
        --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=host \
        -o "$work/$setting.vmfb" > "$work/$setting.log" 2>&1; then
      printf '%s\ttrue\t%s\n' "$setting" "$((SECONDS-start))" >> "$work/compiles.tsv"
      modules+=(--module "$setting=$work/$setting.vmfb")
    else
      printf '%s\tfalse\t%s\n' "$setting" "$((SECONDS-start))" >> "$work/compiles.tsv"
    fi
  done
  if [[ $model == t5 ]]; then
    start=$SECONDS
    if bash "$src/reflow.sh" "$bin" "$work/model.mlir" "$work/reflow" > "$work/reflow.log" 2>&1; then
      printf 'host-reflow\ttrue\t%s\n' "$((SECONDS-start))" >> "$work/compiles.tsv"
      modules+=(--module "host-reflow=$work/reflow/host-reflow.vmfb")
    else
      printf 'host-reflow\tfalse\t%s\n' "$((SECONDS-start))" >> "$work/compiles.tsv"
    fi
  fi
  if [[ ${#modules[@]} == 0 ]]; then
    echo "No successful $model compilation; inspect $work/*.log" >&2
    exit 1
  fi
  "$python" "$src/check_outputs.py" --model "$work/model.onnx" \
    --parameters "$work/model_params.irpa" --iree-bin "$bin" \
    --out "$work/check" --shapes '1,32,64;2,17,95' "${modules[@]}"
done
