import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.keras.applications.resnet50 import ResNet50

# Load model0
model = ResNet50(weights='imagenet')
model.save('resnet50_saved_model') 

# Optimize with tftrt
converter = trt.TrtGraphConverterV2(input_saved_model_dir='resnet50_saved_model', precision_mode=trt.TrtPrecisionMode.FP32, max_workspace_size_bytes=8000000000)
converter.convert()

# Save the model
converter.save(output_saved_model_dir='resnet50_saved_model_TFTRT_FP32')