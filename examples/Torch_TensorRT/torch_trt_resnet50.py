import torch
import torch_tensorrt
torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

# load model; We are going to use a pretrained resnet model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True).eval().to("cuda")

# Compile with Torch TensorRT; 
# documentation: https://nvidia.github.io/Torch-TensorRT/py_api/torch_tensorrt.html#functions
trt_model = torch_tensorrt.compile(model, 
    inputs= [torch_tensorrt.Input((1, 3, 224, 224))],
    enabled_precisions= { torch.half} # Run with FP32
)

# Save the model
torch.jit.save(trt_model, "model.pt")