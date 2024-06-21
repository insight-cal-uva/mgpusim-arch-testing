#!/bin/bash

# Creates both assembled and disassembled binaries for both
# - fiji
# - gfx803
#
# AMD Architecture GPUs

# create .hsasco files
/opt/rocm/bin/clang-ocl -mcpu=gfx803 nnet/kernels.cl -o nnet/kernels-gfx803.hsaco
/opt/rocm/bin/clang-ocl -mcpu=fiji nnet/kernels.cl -o nnet/kernels-fiji.hsaco

# Disassemble the .hsaco file

/opt/rocm/llvm/bin/llvm-objdump --mcpu=gfx803 --disassemble nnet/kernels-gfx803.hsaco > nnet/kernels-gfx803.disasm
/opt/rocm/llvm/bin/llvm-objdump --mcpu=fiji --disassemble nnet/kernels-fiji.hsaco > nnet/kernels-fiji.disasm