from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow as tf

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype

def process_image(image_path="img1.jpg"):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)

transoformed_img = process_image()

# Setting up client
triton_client = httpclient.InferenceServerClient(url="localhost:8000")

test_input = httpclient.InferInput("input_1", transoformed_img.shape, datatype="FP32")
test_input.set_data_from_numpy(transoformed_img, binary_data=True)

test_output = httpclient.InferRequestedOutput("predictions", binary_data=True, class_count=1000)

# Quering the server
results = triton_client.infer(model_name="resnet50", inputs=[test_input], outputs=[test_output])

test_output_fin = results.as_numpy('predictions')
print(test_output_fin)