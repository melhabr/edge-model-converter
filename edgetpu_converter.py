import argparse
import tensorflow as tf
import subprocess

import converter_util


def setup_args(parser):
    parser.add_argument("--input", "-i", help="Path to input file", required=True, type=str)
    parser.add_argument("--input_dims", "-id", help="Dimensions of input tensor", type=int, nargs='+')
    parser.add_argument("--output_dir", "-o", help="Output dir and filename.", default="./converted_model")

def add_edgetpu_args(parser):

    # TODO: Allow addition of several mean-std pairs
    parser.add_argument("--q_mean", help="Mean of training data for quantization (if model is quantized)",
                        default=128, type=int)
    parser.add_argument("--q_std", help="STD of training data for quantization (if model is quantized)",
                        default=128, type=int)


# Accepts a tensorflow frozen graph and produces a edgetpu-compiled graph, as well as an intermediate tflite flatbuffer
# Arguments:
# args: program arguments
# graph_chars: GraphCharacteristics object
def convert_to_edgetpu(args, input_dims, graph_chars=None):
    if graph_chars is None:
        with tf.compat.v1.gfile.GFile(args.input, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        graph_chars = converter_util.GraphCharacteristics(graph_def)

    output_nodes = []

    # Correct for multiple output dimensions.
    if "TFLite_Detection_PostProcess" in graph_chars.output_node_names:
        output_nodes = ["TFLite_Detection_PostProcess", "TFLite_Detection_PostProcess:1",
                        "TFLite_Detection_PostProcess:2", "TFLite_Detection_PostProcess:3"]
    elif any([node.attr['_output_types'] is not None for node in graph_chars.output_nodes]):
        for node in graph_chars.output_nodes:
            num_out = len(node.attr['_output_types'].list.type)
            if num_out > 0:
                print("Node {} has {} output dimensions".format(node.name, num_out))
                output_nodes = [node.name + ':' + str(i) for i in range(num_out)]
            output_nodes[0] = node.name
    else:
        output_nodes = graph_chars.output_nodes

    print("Corrected output names: ", output_nodes)

    # Check for quantization
    quantized = False
    for node in output_nodes:
        if graph_chars.nodes_by_name[node.split(':')[0]].attr['_output_quantized'].b:
            print("Quantization detected. Using quantized conversion with mean and STD ({}, {})"
                  .format(args.q_mean, args.q_std))
            quantized = True
            break

    # Convert and save model
    converter = tf.lite.TFLiteConverter.from_frozen_graph(args.input, graph_chars.input_node_names, output_nodes,
                                                          input_shapes={graph_chars.input_node_names[0]: input_dims})
    converter.allow_custom_ops = True
    if quantized:
        converter.inference_type = tf.compat.v1.lite.constants.QUANTIZED_UINT8  # TODO: Fix assumption that quantization is 8-bit
        converter.inference_input_type = tf.compat.v1.lite.constants.QUANTIZED_UINT8
        q_stats = {}
        for node in graph_chars.input_node_names:
            q_stats.update({node: (args.q_mean, args.q_std)})
        converter.quantized_input_stats = q_stats

    tflite_model = converter.convert()
    open(args.output_dir + ".tflite", "wb").write(tflite_model)
    print("Model successfully converted to tflite flatbuffer")

    # Compile the flatbuffer for edge TPU
    subprocess.run(["edgetpu_compiler", args.output_dir + ".tflite"], check=True)
    print("Model successfully compiled")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    setup_args(parser)
    add_edgetpu_args(parser)
    args = parser.parse_args()

    # Create graph
    with tf.compat.v1.gfile.GFile(args.input, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Get graph data
    graph_chars = converter_util.GraphCharacteristics(graph_def)

    # Set input dimensions
    input_dims = converter_util.get_input_dims(args, graph_chars.input_nodes[0])  # TODO: Only supports one input tensor

    convert_to_edgetpu(args, input_dims, graph_chars=graph_chars)
