# Compatibility

This file details converter success/issues with existing models. Where a model has been successfully converted, the 
result can be found the converted_models folder.

Mobilenet models are all 300x300 px. 

TensorRT models are converted on a Jetson Nano TX2. The resulting binary may not be compatible with other Nvidia 
hardware.

Model | EdgeTPU | TensorRT | OpenVINO IR
--- | --- | ---- | ----------------|
ssd_mobilenet_v1_coco | NO|YES|YES |
ssd_mobilenet_v1_quantized_coco     |YES|NO|NO 
ssd_mobilenet_v1_0.75_depth_coco    |NO|NO|NO
ssd_mobilenet_v2_coco               |NO|YES|YES
ssd_mobilenet_v2_quantized_coco     |YES|NO|NO
ssd_inception_v2_coco               |NO | YES|YES
