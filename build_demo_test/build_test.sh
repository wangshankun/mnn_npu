#!/bin/sh
cmake .. && make -j8
cd ./demo_sub_models/src && ./build.sh
export XPU_EXE_PATH=./demo_sub_models/src

cd -
./MNNV2Basic.out shufflenet-v2_t31_fuse.mnn 1 0 0 1 1x3x224x224



