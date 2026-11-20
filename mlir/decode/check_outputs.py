"""Check every single-token decoder output against ONNX Runtime CPU.

Input dimensions follow the public BART/T5 decoder-with-past ONNX signatures.
B=batch, P=past decoder length, S=encoder length; the current token length is 1.
"""
import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import onnxruntime as ort


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "parameters", "iree-bin", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--module", action="append", required=True,
                        help="label=/path/to/model.vmfb; may be repeated")
    parser.add_argument("--shapes", default="1,32,64;2,17,95", help="B,P,S tuples")
    return parser.parse_args()


def make_inputs(session, batch, past, source):
    rng = np.random.default_rng(1120 + batch + 3 * past + 7 * source)
    dimensions = {"batch_size": batch, "past_decoder_sequence_length": past,
                  "encoder_sequence_length": source, "encoder_sequence_length_out": source}
    vocabulary = session.get_outputs()[0].shape[-1]
    assert isinstance(vocabulary, int)
    inputs = {}
    for info in session.get_inputs():
        shape = tuple(dimensions[d] if isinstance(d, str) else d for d in info.shape)
        if info.name == "input_ids":
            value = rng.integers(0, vocabulary, size=shape, dtype=np.int64)
        elif info.name == "encoder_attention_mask":
            value = rng.integers(0, 2, size=shape, dtype=np.int64)
            value[:, 0] = 1  # At least one unmasked encoder token.
        else:
            value = (rng.standard_normal(shape) * 0.1).astype(np.float32)
        inputs[info.name] = value
    return inputs


def main():
    args = arguments()
    args.out.mkdir(parents=True, exist_ok=True)
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    output_info = session.get_outputs()
    rows, commands = [], []
    for shape in args.shapes.split(";"):
        batch, past, source = map(int, shape.split(","))
        key = f"b{batch}p{past}s{source}"
        inputs = make_inputs(session, batch, past, source)
        input_flags = []
        for index, info in enumerate(session.get_inputs()):
            path = args.out / f"{key}-input{index}.npy"
            np.save(path, inputs[info.name])
            input_flags.append("--input=@" + str(path))
        reference = session.run(None, inputs)
        for index, value in enumerate(reference):
            np.save(args.out / f"{key}-ort{index}.npy", value)
        for module in args.module:
            label, module_path = module.split("=", 1)
            prefix = args.out / f"{key}-{label}"
            command = [str(args.iree_bin / "iree-run-module"), "--module=" + module_path,
                       "--parameters=model=" + str(args.parameters), "--device=local-task",
                       "--function=main_graph", *input_flags,
                       *[f"--output=@{prefix}-out{i}.npy" for i in range(len(reference))]]
            result = subprocess.run(command, text=True, capture_output=True)
            commands.append({"command": command, "returncode": result.returncode})
            Path(str(prefix) + ".log").write_text(result.stdout + result.stderr)
            if result.returncode:
                rows.append(dict(batch=batch, past=past, source=source, module=label,
                                 output=-1, name="runtime_failure", ok=False,
                                 max_abs="", max_rel="", shape=""))
                continue
            module_rows = []
            for index, (expected, info) in enumerate(zip(reference, output_info)):
                actual = np.load(f"{prefix}-out{index}.npy")
                same_shape = actual.shape == expected.shape
                delta = np.abs(actual - expected) if same_shape else np.array([np.inf])
                max_abs = float(np.max(delta))
                max_rel = (float(np.max(delta / np.maximum(np.abs(expected), 1e-6)))
                           if same_shape else float("inf"))
                ok = (same_shape and actual.dtype == expected.dtype
                      and bool(np.all(np.isfinite(actual)))
                      and bool(np.allclose(actual, expected, rtol=1e-4, atol=1e-5)))
                module_rows.append(dict(batch=batch, past=past, source=source, module=label,
                                        output=index, name=info.name, ok=ok,
                                        max_abs=max_abs, max_rel=max_rel, shape=str(actual.shape)))
            rows.extend(module_rows)
            print(key, label, "outputs", len(module_rows),
                  "passed", all(r["ok"] for r in module_rows),
                  "max_abs", max(r["max_abs"] for r in module_rows), flush=True)
    with (args.out / "results.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    metadata = {"reference": "ONNX Runtime CPUExecutionProvider", "onnxruntime": ort.__version__,
                "numpy": np.__version__, "rtol": 1e-4, "atol": 1e-5,
                "seed": "1120+B+3*P+7*S", "shapes": args.shapes, "modules": args.module,
                "input_order": [x.name for x in session.get_inputs()],
                "output_order": [x.name for x in output_info], "commands": commands}
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if not all(row["ok"] for row in rows):
        raise SystemExit("Output/reference mismatch or runtime failure: inspect results.csv")


if __name__ == "__main__":
    main()
