# Use ROCm base image
FROM rocm/dev-ubuntu-18.04:3.8

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages
# RUN apt-get update 

# installing build tools
# RUN apt-get install -y \
#     cmake \
#     build-essential

RUN apt-get install -y \
    rocm-dev \
    rocm-opencl-dev \
    rocm-clang-ocl

# Create a working directory
WORKDIR /workspace

# Copy the OpenCL kernel file into the container
COPY ./nnet/kernels.cl .

# Run the bash script (apparently Docker does not like more than one command)
CMD ["/bin/bash", "./nnet/create-files.sh"]
