import argparse
import sys
import glob
import os

import tensorflow as tf

import converter_util


def setup_args(parser):
    parser.add_argument("--input", "-i", help="Path to input file", required=True, type=str)
    parser.add_argument("--input_dims", "-id", help="Dimensions of input tensor", type=int, nargs='+')
    parser.add_argument("--output_dir", "-o", help="Output dir and filename.", default="./converted_model")


def add_openvino_args(parser):
    parser.add_argument("--openvino_dir", "-ovdir", help="Directory of openvino installation")
    parser.add_argument("--transformations_config", "-tc", help="Directory of openvino config")
    parser.add_argument("--pipeline_config", "-pc", help="Tensorflow pipeline config")
    parser.add_argument("--channel_order", "-co", help="Order of input channels", choices=["RGB", "BRG"], default="RGB")


def convert_to_openvino(args, input_dims, graph_chars):
    if args.transformations_config is None:
        print("Error:--transformations_config args are required for openvino conversion")
        return

    if args.openvino_dir is None:
        openvino_dir = os.getenv("INTEL_OPENVINO_DIR")
        if openvino_dir is None:
            print("Could not find an OpenVINO installation. Assuming location in /opt/intel/openvino, but check that"
                  "OpenVINO is installed")
            openvino_dir = "/opt/intel/openvino"
    else:
        openvino_dir = args.openvino_dir

    sys.path.insert(1, openvino_dir + "/deployment_tools/model_optimizer")
    from mo.main import main
    from mo.utils.cli_parser import get_tf_cli_parser

    sys.argv = ['']

    # Set input model
    sys.argv.append("--input_model")
    sys.argv.append(args.input)

    # Set transformation config
    sys.argv.append("--transformations_config")
    sys.argv.append(args.transformations_config)

    # Set pipeline
    if args.pipeline_config is None:
        sp = args.input.rsplit('/', 1)
        if len(sp) == 1:
            localdir = './'
        else:
            localdir = sp[0] + '/'
        pipelines = glob.glob(localdir + '*.config')
        if len(pipelines) != 1:
            print("Error: No clear pipeline file")
            exit(1)
        args.pipeline_config = pipelines[0]
    sys.argv.append("--tensorflow_object_detection_api_pipeline_config")
    sys.argv.append(args.pipeline_config)

    # Set input dimensions
    sys.argv.append("--input_shape")
    sys.argv.append(str(input_dims))

    # Check reversal
    if args.channel_order == "RGB":
        sys.argv.append("--reverse_input_channels")

    # Set output dir
    sys.argv.append("--output_dir")
    sys.argv.append(args.output_dir.rsplit('/', 1)[0] + '/')

    # Set output nodes
    sys.argv.append("--output")
    sys.argv.append(','.join([node.name for node in graph_chars.output_nodes]))

    main(get_tf_cli_parser(), 'tf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    setup_args(parser)
    add_openvino_args(parser)
    args = parser.parse_args()

    with tf.compat.v1.gfile.GFile(args.input, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Get graph data
    graph_chars = converter_util.GraphCharacteristics(graph_def)

    # Set input dimensions
    input_dims = converter_util.get_input_dims(args, graph_chars.input_nodes[0])

    convert_to_openvino(args, input_dims, graph_chars=graph_chars)
