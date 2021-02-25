# Traffic Scene Perception from Top-Views – With Drones

## Object Detection

We use [darknet](https://github.com/AlexeyAB/darknet) to detect traffic participants in drone images.

The simplest solution is to build a docker image and use that wherever you want.
Here, you only need to install the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) if using NVIDIA GPUs.

You can either build a docker image yourself or use a [prebuilt image](https://hub.docker.com/repository/docker/wsascha/traptop-drone) with the weights noted below.
To run the object detection, call
```bash
docker run -t --runtime=nvidia -v <path/to/inputs>:/data/inputs -v <path/to/outputs>:/data/outputs wsascha/traptop-drone:yolo-v4-visdrone <more optional args>
```
Here, `<more optional args>` stands for arguments documented [here](docker/yolo-v4-visdrone/object_detection.py).

[Here](https://mega.nz/file/6lB3mIIB#B5xBRIBvNkAAP3YRh57YwYkAVmq2FjWkPvvC2iRA6wI) you can find weights for [yolo v4](https://arxiv.org/abs/2004.10934) trained on the [VisDrone](http://aiskyeye.com/) data set.
