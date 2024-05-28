import tensorflow as tf
import pandas as pd


def make_parser():
    """Make the argument parser"""
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # required arguments
    parser.add_argument(
        "--input_file",
        required=True,
        help="Path to hdf file to be converted",
        dest="input_file",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Path to which the TFRecords is saved",
        dest="output_file"
    )

    return parser

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def main(options):
    print(options)


if __name__ == "__main__":
    # parse arguments
    parser = make_parser()
    args = parser.parse_args()
    options = vars(args)

    main(options)

