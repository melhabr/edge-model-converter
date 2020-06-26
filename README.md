# EDGE Model Converter

The edge-model-converter is a set of scripts and tools designed to assist with the conversion of machine learning models
to edge-computing compatible formats. Currently, the converter supports the conversion of frozen Tensorflow protobuffer
graphs to formats compatible with [Google's Edge TPU](https://cloud.google.com/edge-tpu), [Nvidia's TensorRT](https://github.com/NVIDIA/TensorRT),
and [Intel's OpenVINO](https://docs.openvinotoolkit.org/) platform. This converter currently only support SSD models.

## Requirements

See the [requirements file](REQUIREMENTS.md) for details on required packages and installations.

## Usage

Basic converter usage for all three filetypes would look like this:

```
$ python3 converter.py -i [path/to/input.pb] --edgetpu --tensorrt --openvino [other conversion-specific options]
```

Note that each conversion script can be run as a standalone script, e.g.
```
$ python3 tensorrt_converter.py -i [path/to/input.pb]
```
would work fine. 

All converter scripts accept the common command line arguments, and the converter-specific arguments are accepted by 
both the respective converter and the general converter file (converter.py).

### Common Command Line Arguments

These arguments are common to all conversion scripts. 

**Required Arguments**

`--input (-i)` - Path to input file (a frozen tensorflow protobuffer file, usually with a .pb extension)

**Optional Arguments**

`--input_dims (-id)` - Dimensions of input tensor. The conversion script will attempt to automatically identify the
input dimensions based on the graph, but they are sometimes left unspecified (this is frequently true for batch size or
image dimensions). If user specification of input dimensions is required and `--input_dims` is not specified, the script
will prompt the user. If the provided dimensions conflict with existing dimension, the script will raise an error.

`--output_dir (-o)` - Output directory and filename. Defaults to `./converted_model`. The correct file extension is
automatically added. 

### Edge TPU Command Line Arguments

**Optional Arguments**

`--q_mean` - Quantization mean to use, if model is quantized. Defaults to 128.

`--q_std` - Quantization standard deviation to use, if model is quantized. Defaults to 128.

### TensorRT Command Line Arguments

**Optional Arguments**

`--no_cuda` - Disables components of the TensorRT that require the CUDA runtime. This will result in the TensorRT
converter only performing the uff conversion. The resulting uff model will need to be compiled on the corresponding
CUDA-enabled machine. 

### OpenVINO Command Line Arguments

**Required Arguments**

`--transformations_config (-tc)` - The path to a .json file containing the transformation config corresponding to the
input model. If you are converting one of the standard models, a configuration file is likely provided by intel, and can
be found in `${openvino_intall_dir}/deployment_tools/model_optimizer/extensions/front/tf`.

**Optional Arguments**

`--openvino_dir (-ovdir)` - Path to OpenVINO install directory. Defaults to `/opt/intel/openvino`. _While this argument is
optional, care should be taken to ensure it is set properly._

`--pipeline_config (-pc)` - Path to a pipeline configuration file for the model. Defaults to any *.config file in the
same directory as the model. 

`--channel_order (-co)` - Order of color channels. Accepts RGB or BGR. Defaults to RGB. 
