# CNTK 2.2 with Azure Machine Learning Workbench

The repo takes the [FasterRCNN](https://github.com/Microsoft/CNTK/tree/master/Examples/Image/Detection) example from the [CNTK](https://github.com/Microsoft/CNTK) samples GitHub repo and makes it run inside a docker container both locally and within a remote NC series data science VM in Azure complete with GPU support. 

- Derive a new docker container 
- Configure Conda dependencies

## Videos

Some videos of this process:

- [Remote Compute with AML Workbench Part A - Create a Data Science VM in Azure](https://youtu.be/ID55vVDv0R4)
- [Remote Compute with AML Workbench Part B - Create a Custom Docker Base Image](https://youtu.be/WR0QBL4O61o)
- [Remote Compute with AML Workbench Part C - Link a Remote GPU based VM and Run an Experiment](https://youtu.be/rdV1sgF77Is)

## Getting Started

In order to start create a new project in Azure Machine Learning Workbench and copy the files in this repo over to the new project directory. 

## Train

This project contains source images (created with [VoTT](https://github.com/Microsoft/VoTT)) and code to train a model that can identify fluffy animals. 

<img src="https://user-images.githubusercontent.com/5225782/33003163-06fa7916-ce0d-11e7-836e-5aeb6cd704aa.png" width="500"/>

<img src="https://user-images.githubusercontent.com/5225782/33003165-085e058e-ce0d-11e7-80e6-be77fbb9396b.png" width="500"/>

To train this model you can use environments created from the pre-made Docker and Conda configs in this repo. 

## Create a new Docker base image

The base docker image that is used by default (`microsoft/mmlspark:plus-0.7.91`) when creating new AML Workbench projects does not have all the dependencies installed that are required by CNTK 2.2 (it's actually OpenCV that is the problem).

Luckily it's possible to derive a new container from `microsoft/mmlspark:plus-0.7.91` and add the dependencies required. 

The supplied [docker\dockerfile](https://github.com/jakkaj/CNTK_AMLWorkbench/blob/master/docker/dockerfile) runs the required dependencies during build. 

You can see the file is running the extra dependencies. Note some are commented out in the dockerfile - these are extra dependencies listed by OpenCV that were not required for this project. 

```yaml
RUN apt-get update
RUN apt-get -y install libpng-dev
RUN apt-get -y install libjasper-dev
```

To build your new container, run the following command from the Docker folder. 

```docker build -t reponame/imagename .```

Once that is built edit `docker.compute` and update the baseDocker image.

 `baseDockerImage: "reponame/imagename"` 

### Docker image for remote data science VM with support for GPU

See [Running a script on a remote Docker](https://docs.microsoft.com/en-us/azure/machine-learning/preview/experimentation-service-configuration#running-a-script-on-a-remote-docker) for instructions in linking a remote Docker environment before continuing. 

There is a different `dockerfile` and Conda file for GPU based machines. 

Derive from the [docker_gpu\dockerfile](https://github.com/jakkaj/CNTK_AMLWorkbench/blob/master/docker_gpu/dockerfile) Make sure the `.compute` file reflects this new container name. Also make sure you use the `conda_dependencies_gpu.yml` file in the `.runconfig` file. 

```docker build -t reponame/imagename .```

Once you've built the container, you'll have to push it up to [Docker Hub](https://hub.docker.com/) so the remote data science VM can see it. 

Get an account at [Docker Hub](https://hub.docker.com/) then run the following commands:

`docker login`
`docker push reponame/imagename`

*Note:* Your "reponame" is the name of your Docker Hub account. 

Importantly make sure you add the following line to the `.compute` file:

`nvidiaDocker: true`

Now set up the remote environment by using the following command (where removeenvname is the name you used when you ran the `az ml computetarget attach` command. 

`az ml experiment prepare -c remoteenvname`

Once complete you will be able to run your experiements by calling 

`az ml experiment submit -c remoteenvname`



### Pre-made containers

If you don't want to build and publish your own containers, you may use the premade ones:

- `jakkaj/ml` for non-GPU
- `jakkaj/mlgpu` for GPU

## Train the model

Run `az ml experiment submit -c docker` or `-c remoteenvname` to run the experiment.

A nice way to run experiments in different environments is to use the [Visual Studio Code Tools for AI Plugin](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.vscode-ai). 

The experiment will save the created model in the "[output](https://docs.microsoft.com/en-us/azure/machine-learning/preview/how-to-read-write-files)" folder. The system will download the AlexNet pre-trained model the first time it is run.  
