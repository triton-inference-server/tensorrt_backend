# Tensorflow TensorRT to Triton

This document contains barebone instructions explaining how to deploy a model accelerated by using Tensorflow TensorRT on NVIDIA Triton Inference Server. For an indepth explaination, refer this [blog](https://TODO_add_blog_link). This README and the other files we provide showcase how to deploy a simple resnet model.

## Step 1: Optimize your model wtih Torch TensorRT

If you are unfamiliar with Tensorflow TensorRT please refer this [video](https://www.youtube.com/watch?v=w7871kMiAs8&ab_channel=NVIDIADeveloper). The first step in this pipeline is to accelerate your model. If you are using PyTorch as you framework of choice for training, you can either user TensorRT or Torch-TensorRT depending on your model's operations.

For using Tensorflow TensorRT, let's first pull our PyTorch docker container which comes installed with both TensorRT and Tensorflow TensorRT. You may need to create an account and get the API key from [here](https://ngc.nvidia.com/setup/). Sign up and login with your key (follow the instructions [here](https://ngc.nvidia.com/setup/api-key) after siging up).

```
# <xx.xx> is the yy:mm for the publishing tag for NVIDIA's Tensorflow 
# container; eg. 21.12

docker run -it --gpus all -v /path/to/this/folder:/resnet50_eg nvcr.io/nvidia/tensorflow:<xx.xx>-tf2-py3
```

We have already made a short script `tf_trt_resnet50.py` as a sample to use Torch TensorRT. For more examples visit our [Github Repository](https://github.com/NVIDIA/Torch-TensorRT/).

```
python tf_trt_resnet50.py

# you can exit out of this container now
exit
```

## Step 2: Set Up Triton Inference Server

If you are new to the Triton Inference Server and want to learn more, we hightly recommend to check out our [Github Repository](https://developer.nvidia.com/nvidia-triton-inference-server).

To use Triton, we need to make a model repository. The structure of the repository should look something like this:
```
model_repository
|
+-- resnet50
    |
    +-- config.pbtxt
    +-- 1
        |
        +-- model.savedmodel
            |
            +-- saved_model.pb
            +-- variables
                |
                +-- variables.data-00000-of-00001
                +-- variables.index
```

As might be apparent from the model structure above, each model requires a configuration file to spin up the server. We provide a sample of a `config.pbtxt`, which you can use for this specific example. If you are new to Triton, we highly encorage you to checkout out this [section of our documentation](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md) for more. 

Once you have the model repository setup, it is time to launch the triton server! You can do that with the docker command below.
```
docker run --gpus=1 --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v /full/path/to/docs/examples/model_repository:/models nvcr.io/nvidia/tritonserver:<xx.yy>-tf2-python-py3 tritonserver --model-repository=/models --backend-config=tensorflow,version=2
```

## Step 3: Using a Triton Client to Query the Server

Download the image.

```
wget  -O img1.jpg "https://www.hakaimagazine.com/wp-content/uploads/header-gulf-birds.jpg"
```

Install dependencies.
```
pip install --upgrade tensorflow
pip install pillow
pip install nvidia-pyindex
pip install tritonclient[all]
```

Run client
```
python3 triton_client.py
```