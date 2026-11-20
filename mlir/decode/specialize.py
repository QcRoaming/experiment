"""Optionally bind B=1, P=32, and S=64 on ONNX graph inputs/outputs."""
import argparse

import onnx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Input model.onnx")
    parser.add_argument("axes", choices=("B", "P", "S", "BP", "BS", "PS", "BPS"))
    parser.add_argument("output", help="Output model.onnx")
    args = parser.parse_args()
    bindings = {
        "B": {"batch_size": 1},
        "P": {"past_decoder_sequence_length": 32,
              "past_decoder_sequence_length + 1": 33},
        "S": {"encoder_sequence_length": 64,
              "encoder_sequence_length_out": 64},
    }
    dimensions = {name: value for axis in args.axes
                  for name, value in bindings[axis].items()}
    model = onnx.load(args.model)
    for value in [*model.graph.input, *model.graph.output]:
        for dimension in value.type.tensor_type.shape.dim:
            if dimension.dim_param in dimensions:
                dimension.dim_value = dimensions[dimension.dim_param]
    onnx.checker.check_model(model)
    onnx.save(model, args.output)


if __name__ == "__main__":
    main()
