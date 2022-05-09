# TensorRT to Triton

This README showcase how to deploy a simple resnet model accelerated by using TensorRT on NVIDIA Triton Inference Server. For an indepth explaination, refer this [blog](https://TODO_add_blog_link).

## Step 1: Optimize your model wtih Torch TensorRT

If you are unfamiliar with TensorRT please refer this [video](https://youtu.be/rK-jxPPY9V4). The first step in this pipeline is to accelerate your model with TensorRT. For the purposes of this demonstration, we are going to assume that you have your trained model in ONNX format. 

(Optional) If you don't have an ONNX model handy and just want to follow along, feel free to use this script:
```
# <xx.xx> is the yy:mm for the publishing tag for NVIDIA's TensorRT 
# container; eg. 21.12

docker run -it --gpus all -v /path/to/this/folder:/resnet50_eg nvcr.io/nvidia/pytorch:<xx.xx>-py3

python export_resnet_to_onnx.py
exit
```

You may need to create an account and get the API key from [here](https://ngc.nvidia.com/setup/). Sign up and login with your key (follow the instructions [here](https://ngc.nvidia.com/setup/api-key) after siging up).

Now that we have an ONNX model, we can use TensorRT to optimize your model. These optimizations are stored in the for of a TensorRT Engine. You might see references to TensorRT "plan files". For simplicity sake, you can assume these terms to be refering to the same entity.

While there are several ways of installing TensorRT, the easiest way is to simply get our pre-built docker container!

```
docker run -it --gpus all -v /path/to/this/folder:/trt_optimize nvcr.io/nvidia/tensorrt:<xx:yy>-py3
```
There are several ways to build a TensorRT Engine; for this demonsration, we will simply use the `trtexec` [CLI Tool](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec).

```
trtexec --onnx=resnet50.onnx \
        --saveEngine=resnet50.engine \
        --explicitBatch \
        --useCudaGraph
```

Before we proceed to the next step, it is important that we know the names of the "input" and "output" layers of your network, as these would be required by Triton. One easy way is to use `polygraphy` which comes packaged with the TensorRT container. If you want to learn more about Polygrpahy and its usage, visit [this](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy) repository. You can checkout a plethora of [examples](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy/examples/cli/inspect) demonstrating the utility of Polygraphy to inspect models.

```
polygraphy inspect model resnet50.engine --mode=basic
```
With this, we are ready to proceed to the next step; setting up the Triton Inference Server!

## Step 2: Set Up Triton Inference Server

If you are new to the Triton Inference Server and want to learn more, we hightly recommend to check out our [Github Repository](https://github.com/triton-inference-server).

To use Triton, we need to make a model repository. The structure of the repository should look something like this:
```
model_repository
|
+-- resnet50
    |
    +-- config.pbxt
    +-- 1
        |
        +-- model.plan
```
By now you should already have the model from Step 1. To get `model.plan` file, just rename the `resnet50.engine` file that was generated in the previous step.
```
mv resnet50.engine model_repository/resnet50/1/model.plan
```

As might be apparent from the model structure above, each model requires a configuration file to spin up the server. We provide a sample of a `config.pbtxt`, which you can use for this specific example. If you are new to Triton, we highly encorage you to checkout out this [section of our documentation](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md) for more. 

Once you have the model repository setup, it is time to launch the triton server! You can do that with the docker command below.
```
docker run --gpus=1 --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v /full/path/to/docs/examples/model_repository:/models nvcr.io/nvidia/tritonserver:<xx.yy>-py3 tritonserver --model-repository=/models
```

## Step 3: Using a Triton Client to Query the Server

Download the image.

```
wget  -O img1.jpg "https://www.hakaimagazine.com/wp-content/uploads/header-gulf-birds.jpg"
```

Install dependencies.
```
pip install torchvision
pip install attrdict
pip install nvidia-pyindex
pip install tritonclient[all]
```

Run client
```
python3 triton_client.py
```
