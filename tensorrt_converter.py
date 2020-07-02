"""

This code substantially adapted from JK Jung [https://github.com/jkjung-avt/tensorrt_demos]
License:

MIT License

Copyright (c) 2019 JK Jung

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import argparse
import tensorflow as tf
import tensorrt as trt
import uff
import graphsurgeon as gs

import converter_util


def setup_args(parser):
    parser.add_argument("--input", "-i", help="Path to input file", required=True, type=str)
    parser.add_argument("--input_dims", "-id", help="Dimensions of input tensor", type=int, nargs='+')
    parser.add_argument("--debug", "-d", help="Debug flag", action='store_true')
    parser.add_argument("--no_cuda", help="Disables script components that require the CUDA runtime.", action='store_true')
    parser.add_argument("--output_dir", "-o", help="Output dir and filename.", default="./converted_model")

def add_plugin(graph, input_dims, graph_chars=None):
    graph_def = graph.as_graph_def()

    if graph_chars is None:
        graph_chars = converter_util.GraphCharacteristics(graph_def)

    num_classes = converter_util.get_num_classes(graph_def, graph_chars=graph_chars)
    input_order = converter_util.get_NMS_input_order(graph_def, "Postprocessor", graph_chars=graph_chars)

    if args.debug:
        print("Detected number of classes: ", num_classes)
        print("Detected NMS input order: ", input_order)


    assert_nodes = graph.find_nodes_by_op("Assert")
    graph.remove(assert_nodes, remove_exclusive_dependencies=True)

    identity_nodes = graph.find_nodes_by_op("Identity")
    graph.forward_inputs(identity_nodes)

    Input = gs.create_plugin_node(
        name="Input",
        op="Placeholder",
        shape=(1,) + input_dims
    )

    # TODO: Consider automation of parameters
    PriorBox = gs.create_plugin_node(
        name="MultipleGridAnchorGenerator",
        op="GridAnchor_TRT",
        minSize=0.2,
        maxSize=0.95,
        aspectRatios=[1.0, 2.0, 0.5, 3.0, 0.33],
        variance=[0.1, 0.1, 0.2, 0.2],
        featureMapShapes=[19, 10, 5, 3, 2, 1],
        numLayers=6
    )

    NMS = gs.create_plugin_node(
        name="NMS",
        op="NMS_TRT",
        shareLocation=1,
        varianceEncodedInTarget=0,
        backgroundLabelId=0,
        confidenceThreshold=0.3,
        nmsThreshold=0.6,
        topK=100,
        keepTopK=100,
        numClasses=num_classes,
        inputOrder=input_order,
        confSigmoid=1,
        isNormalized=1
    )

    concat_box_loc = gs.create_plugin_node(
        "concat_box_loc",
        op="FlattenConcat_TRT",
        axis=1,
        ignoreBatch=0
    )

    concat_box_conf = gs.create_plugin_node(
        "concat_box_conf",
        op="FlattenConcat_TRT",
        axis=1,
        ignoreBatch=0
    )

    concat_priorbox = gs.create_node(
        "concat_priorbox",
        op="ConcatV2",
        axis=2
    )

    namespace_map = {
        "MultipleGridAnchorGenerator": PriorBox,
        "Preprocessor": Input,
        "ToFloat": Input,
        "Cast": Input,
        "image_tensor": Input,
        "Postprocessor": NMS,
        "concat": concat_box_loc,
        "concat_1": concat_box_conf,
        "Concatenate": concat_priorbox,
        "MultipleGridAnchorGenerator/Concatenate": concat_priorbox
    }
    graph.collapse_namespaces(namespace_map)

    graph.remove(graph.graph_outputs, remove_exclusive_dependencies=False)
    if "Input" in graph.find_nodes_by_op("NMS_TRT")[0].input:
        graph.find_nodes_by_op("NMS_TRT")[0].input.remove("Input")
    if graph.find_nodes_by_name("Input")[0].input:
        graph.find_nodes_by_name("Input")[0].input.remove("image_tensor:0")

    return graph


def convert_to_tensorrt(args, input_dims, graph_chars=None):
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')

    input_dims_corrected = (input_dims[3], input_dims[1], input_dims[2])

    graph = add_plugin(gs.DynamicGraph(args.input), input_dims_corrected, graph_chars=graph_chars)

    try:
        uff.from_tensorflow(
            graph.as_graph_def(),
            output_nodes=['NMS'],
            output_filename=(args.output_dir + ".uff"))
    except TypeError as e:
        if e.__str__() == "Cannot convert value 0 to a TensorFlow DType.":
            raise EnvironmentError("Please modify your graphsurgeon package according to the following:\n"
                                   "https://github.com/AastaNV/TRT_object_detection#update-graphsurgeon-converter")

    if args.no_cuda:
        exit(0)

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        builder.max_workspace_size = 1 << 28
        builder.max_batch_size = 1
        builder.fp16_mode = True

        parser.register_input('Input', input_dims_corrected)
        parser.register_output('MarkOutput_0')
        parser.parse(args.output_dir + ".uff", network)
        engine = builder.build_cuda_engine(network)

        buf = engine.serialize()
        with open(args.output_dir + '_tensorrt.bin', 'wb') as f:
            f.write(buf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args()

    with tf.compat.v1.gfile.GFile(args.input, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Get graph data
    graph_chars = converter_util.GraphCharacteristics(graph_def)

    # Set input dimensions
    input_dims = converter_util.get_input_dims(args, graph_chars.input_nodes[0])

    convert_to_tensorrt(args, input_dims, graph_chars=graph_chars)
