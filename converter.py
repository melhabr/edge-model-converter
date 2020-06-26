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

    # EdgeTPU arguments
    parser.add_argument("--edgetpu", "-c", help="Perform conversion for Coral EdgeTPU.", action='store_true')
    # TODO: Allow additions of several mean-std pairs
    parser.add_argument("--q_mean", help="Mean of training data for quantization (if model is quantized)",
                        default=128, type=int)
    parser.add_argument("--q_std", help="STD of training data for quantization (if model is quantized)",
                        default=128, type=int)
    # TensorRT arguments
    parser.add_argument("--tensorrt", "-t", help="Perform Tensorrt conversion.", action='store_true')
    parser.add_argument("--no_cuda", help="Disables script components that require the CUDA runtime.", action='store_true')

    # OpenVINO arguments
    parser.add_argument("--openvino", "-ov", help="Perform OpenVINO IR conversion", action='store_true')
    # TODO: Check that openvino install dir is valid
    parser.add_argument("--openvino_dir", "-ovdir", help="Directory of openvino installation", default="/opt/intel/openvino")
    parser.add_argument("--transformations_config", "-tc", help="Directory of openvino config", required=True)
    parser.add_argument("--pipeline_config", "-pc", help="Tensorflow pipeline config")
    parser.add_argument("--channel_order", "-co", help="Order of input channels", choices=["RGB", "BRG"], default="RGB")

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
        if args.transformations_config is None:
            print("Error: --openvino_dir and --transformations_config args are required for openvino conversion")
            exit(1)
        openvino_converter.convert_to_openvino(args, input_dims, graph_chars=graph_chars)