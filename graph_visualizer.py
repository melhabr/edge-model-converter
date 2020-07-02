import tensorflow as tf
import argparse
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", help="Path to input file", type=str)
    parser.add_argument("--port", "-p", help="Port to use", default=6006, type=int)
    args = parser.parse_args()

    with tf.compat.v1.gfile.GFile(args.input, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.compat.v1.Session() as sess:
        tf.import_graph_def(graph_def, name='')
        writer = tf.compat.v1.summary.FileWriter("./graphs", sess.graph)

    subprocess.run(["tensorboard", "--logdir", "./graphs", "--port", args.port])