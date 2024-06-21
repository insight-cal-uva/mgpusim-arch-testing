# OpenCL vs Cuda Demo

This project's goal is to gain experience with Open Compute Langauge and explore the differences between Open CL and Cuda in the programming model, performance and simulation pipelines.

A large goal of this project is to explore other simulators such as mgpusim for simulating workloads on AMD gpus.

## Project Structure

The project is a MNIST handwritten digit classifier which is implemented in RAW Cuda and OpenCl for **inference**. However, training is done in a PyTorch Jupyter notebook and the weights and biases are saved sequentially in a binary format.

Cross validation between the computer numerical outputs of **both** the OpenCL version and the Cuda version will take place.

The file structure is as follows:

- `train` contains a Jupyter notebook in which training of the initial model with PyTorch is conducted
- `opencl` contains the Open CL implementation of neural network
- `cuda` contains the cuda implementation of the neural network

## Neural Network Design

The neural network uses will be a basic feed-forward neural network with use of the ReLu activation function for each layer.

## Running the Project

The project uses a common Makefile for **all** implementations of the neural network for ease of interaction.

## Building the Open CL Kernels

The OpenCl kernels are compiled into the kernel bytecode for NVIDIA using RocM 3.8 through a dockerfile. To build a kernel, run:

```bash
docker build -t rocm-cl-compiler .
```

To build the container and then use:

```bash
docker run --rm -v $(pwd):/workspace rocm-cl-compiler
```

To build `kernels.cl` to `kernels.hsaco`. At this point, you should be done! The compiled kernels are in the correct format to work with `v3` of mgpusim. 

## Performance Considerations

For this project, there are a few main performance considerations:

- Latency
- Throughput

Latency will be measured by the time to produce an inference on a random input after warmup with the weights.

Throughput will be measured by increasing the number of batched inputs per layer until asymtotic performance is reached.

## Unrealistic Design Considerations

MNIST images are quite small (28x28). Given modern hardware (this software was developed on a 2080ti mobile card), the largest network pratical without overfitting is likely able to fit in cache in modern hardware. 

Therefore, the network is made too large for the problem size such that it can be more representative of a workload which **does not** fit in the cache of a modern GPU and represents a greater challenge from a hardware perspective.

## Analysis

Both correlation and roofile analysis will be ran on these programs to produce the correct output.

## Future Work

In the future, tools like HIP which aim to provide a common compute langauge between AMD and NVIDIA GPUs by allowing kernel translation between the two may be tested. However, this may be through the use of tools such as Hippify which use the llvm-generated abstract syntax trees to automatically convert a CUDA codebase to Hip.

