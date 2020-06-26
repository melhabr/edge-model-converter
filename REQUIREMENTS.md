# Converter Requirements and Installation

### Common Requirements

These scripts were tested using Python 3.6. They will likely work with other versions of Python 3, and may also work 
with versions of Python 2, but such compatibility is not guaranteed. 

All scripts require Tensorflow. These scripts have been tested using versions of Tensorflow 1.15. They may also work
with earlier versions of Tensorflow. **They will not work with Tensorflow 2, of any version.**

A compatible version of Tensorflow can be installed using the following command:
```
$ pip3 install tensorflow==1.15.3
```

### EdgeTPU Requirements

Conversion to a tflite flatbuffer file does not require any additional software (beyond Tensorflow). Compilation of
the tflite flatbuffer into an edgetpu-compatible model requires the `edgetpu-compiler` program. Download and
installation can be found [here](https://coral.ai/docs/edgetpu/compiler/#system-requirements).

### TensorRT Requirements

Conversion to a UFF model requires certain CUDA-related packages, but does not actually require a working CUDA runtime 
(which means you can create the .uff file without an NVIDIA GPU or platform). The important modules for .uff conversion
are the `uff` module and the `graphsurgeon` module, and their prerequsites. Download and installation details for those
modules can be found [here](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar). The
installation file will claim that CUDA is a requirement, but the `uff` and `graphsurgeon` modules will run with only the
CUDA toolkit installed (no CUDA driver).

Conversion to output binary, however, requires a working CUDA runtime. I suspect this is because that when the .uff file
is compiled, it is compiled with respect to the CUDA runtime on the system. Download and installation instructions can
be found [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

### OpenVINO Requirements

Use of the OpenVINO converter requires a working OpenVINO installation. Download and installation instructions can be
found [here](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html). By default,
OpenVINO is installed to `/opt/intel/openvino`, but if you install it somewhere else, keep track of the directory and be
sure to pass the path to the OpenVINO converter using the `--openvino_dir` argument.  