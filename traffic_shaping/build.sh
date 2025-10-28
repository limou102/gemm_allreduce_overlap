set -ex

/opt/rocm-7.0.0/lib/llvm/bin/clang++ call_hipblaslt.cpp -o call_hipblaslt -g -O0 -std=c++17 -I/opt/rocm/include/ -D__HIP_PLATFORM_AMD__ -L/opt/rocm/lib/ -lhipblaslt -lamdhip64
#/opt/rocm-7.0.0/lib/llvm/bin/clang++ -xhip xgmi_test.cpp -o xgmi_test --offload-arch=native -O2 -std=c++17 -I/opt/rocm/include/
