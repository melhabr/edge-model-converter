import argparse
from packaging import version

import tensorflow as tf

import edgetpu_converter
import tensorrt_converter
import openvino_converter
import converter_util

# TODO: Organize argument grouping
def setup_args(parser):

    # Common arguments
    parser.add_argument("--input", "-i", help="Path to input file", required=True, type=str)
    parser.add_argument("--input_dims", "-id", help="Dimensions of input tensor", type=int, nargs='+')
    parser.add_argument("--output_dir", "-o", help="Output dir and filename.", default="./converted_model")

    parser.add_argument("--edgetpu", "-c", help="Perform conversion for Coral EdgeTPU.", action='store_true')
    parser.add_argument("--tensorrt", "-t", help="Perform Tensorrt conversion.", action='store_true')
    parser.add_argument("--openvino", "-ov", help="Perform OpenVINO IR conversion", action='store_true')

    edgetpu_converter.add_edgetpu_args(parser)
    tensorrt_converter.add_tensorrt_arguments(parser)
    openvino_converter.add_openvino_args(parser)


if __name__ == '__main__':

    if version.parse(tf.__version__) >= version.parse("2.0.0"):
        print("ERROR: This script is only compatible with tensorflow 1")
        exit(1)

    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args()

    # Create graph
    with tf.compat.v1.gfile.GFile(args.input, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Get graph data
    graph_chars = converter_util.GraphCharacteristics(graph_def)

    # Set input dimensions
    input_dims = converter_util.get_input_dims(args, graph_chars.input_nodes[0])  # TODO: Only supports one input tensor

    if args.edgetpu:
        edgetpu_converter.convert_to_edgetpu(args, input_dims, graph_chars=graph_chars)

    if args.tensorrt:
        tensorrt_converter.convert_to_tensorrt(args, input_dims, graph_chars=graph_chars)

    if args.openvino:
        openvino_converter.convert_to_openvino(args, input_dims, graph_chars=graph_chars)