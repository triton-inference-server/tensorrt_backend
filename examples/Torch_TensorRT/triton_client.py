import numpy as np
from torchvision import transforms
from PIL import Image
import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype

def rn50_preprocess(img_path="img1.jpg"):
    img = Image.open()
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(img).numpy()

transoformed_img = rn50_preprocess()

# Setting up client
triton_client = httpclient.InferenceServerClient(url="localhost:8000")

test_input = httpclient.InferInput("input", transoformed_img.shape, datatype="FP32")
test_input.set_data_from_numpy(transoformed_img, binary_data=True)

test_output = httpclient.InferRequestedOutput("output", binary_data=True, class_count=1000)

# Quering the server
results = triton_client.infer(model_name="resnet50", inputs=[test_input], outputs=[test_output])
test_output_fin = results.as_numpy('output')
print(test_output_fin[:5])