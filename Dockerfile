# Use ROCm base image
FROM rocm/dev-ubuntu-22.04:5.4-complete

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    rocm-opencl \
    rocm-opencl-dev \
    rocm-clang-ocl

# Create a working directory
WORKDIR /workspace

# Copy the OpenCL kernel file into the container
COPY kernel.cl .

# Command to compile the .cl file into .hsaco
CMD ["clang-ocl", "-mcpu=fiji", "-o", "kernels.hsaco", "kernels.cl"]
