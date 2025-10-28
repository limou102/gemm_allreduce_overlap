set -ex

/opt/rocm/lib/llvm/bin/clang++ \
    -D__HIP_PLATFORM_AMD__ \
    -xhip xgmi_test.cpp -o xgmi_test \
    --offload-arch=native \
    -O2 -std=c++17 \
    -I/opt/rocm/include/ \
    -L/opt/rocm/lib/ \
    -lhipblaslt \
    -lamdhip64 \
