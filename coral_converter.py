import argparse
import tensorflow as tf
import subprocess

import converter_util

def setup_args(parser):
    parser.add_argument("--input", "-i", help="Path to input file", required=True, type=str)
    parser.add_argument("--input_dims", "-id", help="Dimensions of input tensor", type=int, nargs='+')
    # TODO: Allow additions of several mean-std pairs
    parser.add_argument("--output_dir", "-o", help="Output dir and filename.", default="./converted_model")
    parser.add_argument("--q_mean", help="Mean of training data for quantization (if model is quantized)",
                        default=128, type=int)
    parser.add_argument("--q_std", help="STD of training data for quantization (if model is quantized)",
                        default=128, type=int)


def convert_to_coral(args, node_name_dict, input_nodes, input_dims, output_nodes):

    # Correct for multiple output dimensions. TODO: Find more general solution (or verify that this solution works generally
    for node in output_nodes:
        num_out = len(node_name_dict[node].attr['_output_types'].list.type)
        if num_out > 0:
            print("Node {} has {} output dimensions".format(node, num_out))
            output_nodes = [node + ':' + str(i) for i in range(num_out)]
        output_nodes[0] = node
    print("Corrected output names: ", output_nodes)


    # Check for quantization
    quantized = False
    for node in output_nodes:
        if node_name_dict[node.split(':')[0]].attr['_output_quantized'].b:
            print("Quantization detected. Using quantized conversion with mean and STD ({}, {})"
                  .format(args.q_mean, args.q_std))
            quantized = True
            break

    # Convert and save model
    # TODO: Should use v2 version of converter
    converter = tf.lite.TFLiteConverter.from_frozen_graph(args.input, input_nodes, output_nodes,
                                                          input_shapes={input_nodes[0]: input_dims})
    converter.allow_custom_ops = True
    if quantized:
        converter.inference_type = tf.compat.v1.lite.constants.QUANTIZED_UINT8  # TODO: Fix assumption that quantization is 8-bit
        converter.inference_input_type = tf.compat.v1.lite.constants.QUANTIZED_UINT8
        q_stats = {}
        for node in input_nodes:
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
    args = parser.parse_args()

    # Create graph
    with tf.compat.v1.gfile.GFile(args.input, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Get graph data
    node_name_dict, node_output_dict, input_nodes, output_nodes = converter_util.analyze_graph(graph_def)

    # Set input dimensions
    input_dims = converter_util.get_input_dims(args, node_name_dict[input_nodes[0]]) # TODO: Only supports one input tensor

    convert_to_coral(args, node_name_dict, input_nodes, input_dims, output_nodes)