set -ex

/opt/rocm-7.0.0/lib/llvm/bin/clang++ -xhip call_hipblaslt.cpp -o call_hipblaslt --offload-arch=native -O2 -std=c++17 -I/opt/rocm/include/
