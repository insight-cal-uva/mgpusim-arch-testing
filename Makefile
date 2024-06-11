.PHONY: clean train cuda-eval hip-eval

CC = nvcc
CFLAGS = 
LDFLAGS = 


HIP_PATH?= $(wildcard /opt/rocm)
HIPCC=$(HIP_PATH)/bin/hipcc

train: 
	jupyter nbconvert --to notebook --inplace --execute train/train.ipynb

cuda-eval: model cuda/a.out

hip-eval: hip/a.out
	./hip/a.out

opencl-eval: opencl/kernels.hsaco

opencl/kernels.hsaco: opencl/kernels.cl
	clang-ocl -mcpu=fiji -o opencl/kernels.hsaco opencl/kernels.cl

cuda/a.out: cuda/main.cu
	nvcc cuda/main.cu

hip/a.out: hip/main.cpp
	$(HIPCC) hip/main.cpp -o hip/a.out


model: train

clean:
	rm *.bin model


