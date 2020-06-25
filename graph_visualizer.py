import tensorflow as tf
import argparse
import subprocess
import webbrowser

def visualize_graph(graph_def):

    with tf.compat.v1.Session() as sess:
        tf.import_graph_def(graph_def, name='')
        writer = tf.compat.v1.summary.FileWriter("./graphs", sess.graph)

    subprocess.run(["tensorboard", "--logdir", "./graphs"])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", help="Path to input file", required=True, type=str)
    args = parser.parse_args()

    with tf.compat.v1.gfile.GFile(args.input, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    visualize_graph(graph_def)