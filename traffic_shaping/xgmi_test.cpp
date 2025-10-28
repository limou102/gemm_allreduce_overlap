#include "gemm_runner.hpp"

#include <thread>
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <inttypes.h>
#include <assert.h>
#include <string.h>
#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>

#define _CHECK(condition)                                                                   \
    {                                                                                       \
        const hipError_t error = condition;                                                 \
        if(error != hipSuccess)                                                             \
        {                                                                                   \
            std::cerr << "An error encountered: \"" << hipGetErrorString(error) << "\" at " \
                      << __FILE__ << ':' << __LINE__ << std::endl;                          \
            exit(1);                                                                  \
        }                                                                                   \
    }

constexpr int _MaxDevices = 8;
constexpr int _MaxLinks = _MaxDevices - 1;

size_t _target_transfer_size = 64ull * 1024 * 1024; //64MB
bool _local_read = true;
int _comm_iters = 20;

int _device_count = 0;
int _links_per_device = 0;
void** _src = {};
void** _dst = {};

std::vector<hipStream_t> _comm_streams;
std::vector<hipStream_t> _gemm_streams;
std::vector<GemmRunner> _gemm_runners;
int _gemm_iters = 20;

typedef void (*kernel_t)(void** __restrict, void** __restrict, int, int, int, int, int, int);

template<int Unroll, class T>
__global__ void test_kernel(void** __restrict src, void** __restrict dst, int dev, int links_per_device, int block_size, int link_stride_items, int iters, int active_threads)
{
    int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_thread >= active_threads)
    {
        return;
    }

    int block_worker = global_thread / block_size;
    int block_thread = global_thread % block_size;
    size_t iter_items = block_size * Unroll;
    int link = block_worker % links_per_device;
    int block_worker_in_link = block_worker / links_per_device;

    size_t offs = block_worker_in_link * iter_items + block_thread;
    for (int i = 0; i < iters; i++)
    {
        const T* psrc = (const T*)(src[dev * _MaxLinks + link]) + offs;
        T* pdst = (T*)(dst[dev * _MaxLinks + link]) + offs;
        T v[Unroll];
        #pragma unroll Unroll
        for (uint32_t j = 0; j < Unroll; j++)
        {
            v[j] = __builtin_nontemporal_load(psrc);
            psrc += block_size;
        }
        #pragma unroll Unroll
        for (uint32_t j = 0; j < Unroll; j++)
        {
            __builtin_nontemporal_store(v[j], pdst);
            pdst += block_size;
        }
        offs += link_stride_items;
    }
}

void* AllocMemory(int dev, size_t size)
{
    void* p = nullptr;
    _CHECK(hipSetDevice(dev));
#if 1
    _CHECK(hipExtMallocWithFlags(&p, size, hipDeviceMallocUncached));
#else
    _CHECK(hipMalloc(&p, size));
#endif
    return p;
}

void SetupTransfers()
{
    _CHECK(hipHostMalloc(&_src, sizeof(void*) * _MaxDevices * _MaxLinks, 0));
    _CHECK(hipHostMalloc(&_dst, sizeof(void*) * _MaxDevices * _MaxLinks, 0));

    for (int dev = 0; dev < _device_count; dev++)
    {
        for (int peer = 0; peer < _device_count; peer++)
        {
            if (dev != peer)
            {
                int peer_status;
                _CHECK(hipDeviceCanAccessPeer(&peer_status, dev, peer));
                if (peer_status == 0)
                {
                    printf("Cannot enable peer access between %d and %d.\n", dev, peer);
                    exit(-1);
                }
                _CHECK(hipSetDevice(dev));
                _CHECK(hipDeviceEnablePeerAccess(peer, 0));
                int link = (peer < dev) ? peer : peer - 1;

                if (_local_read)
                {
                    _src[dev * _MaxLinks + link] = AllocMemory(dev, _target_transfer_size);
                    _dst[dev * _MaxLinks + link] = AllocMemory(peer, _target_transfer_size);
                }
                else
                {
                    _src[dev * _MaxLinks + link] = AllocMemory(peer, _target_transfer_size);
                    _dst[dev * _MaxLinks + link] = AllocMemory(dev, _target_transfer_size);
                }
            }
        }
    }
}

void DestroyTransfers()
{
    for (int dev = 0; dev < _device_count; dev++)
    {
        for (int link = 0; link < _device_count - 1; link++)
        {
            if (_src[dev * _MaxLinks + link] != nullptr)
            {
                _CHECK(hipFree(_src[dev * _MaxLinks + link]));
            }
            if (_dst[dev * _MaxLinks + link] != nullptr)
            {
                _CHECK(hipFree(_dst[dev * _MaxLinks + link]));
            }
        }
    }
    _CHECK(hipHostFree(_src));
    _CHECK(hipHostFree(_dst));
}

template<class T>
void RunTest(int unroll, int wg_size, int cu, int block_size)
{
    int threads = cu * wg_size;
    int block_workers = (cu * wg_size) / (block_size * _links_per_device);
    if (block_workers == 0)
    {
        printf("Insufficient number of threads %d to serve block size of %d for %d links\n", threads, block_size, _links_per_device);
        return;
    }

    int active_threads = block_workers * block_size * _links_per_device;
    float used_cu = static_cast<float>(active_threads) / wg_size;
    int link_stride_items = active_threads * unroll / _links_per_device;
    size_t total_iter_size = sizeof(T) * link_stride_items;
    int iters = static_cast<int>(_target_transfer_size / total_iter_size);
    size_t actual_transfer_size = total_iter_size * iters;

    kernel_t kernel;
    switch (unroll)
    {
        case 1: kernel = test_kernel<1, T>; break;
        case 2: kernel = test_kernel<2, T>; break;
        case 3: kernel = test_kernel<3, T>; break;
        case 4: kernel = test_kernel<4, T>; break;
        case 5: kernel = test_kernel<5, T>; break;
        case 6: kernel = test_kernel<6, T>; break;
        case 7: kernel = test_kernel<7, T>; break;
        case 8: kernel = test_kernel<8, T>; break;
        default:
            printf("Invalid unroll %d\n", unroll);
            exit(-1);
    }

    const dim3 grid_dim(cu, 1, 1);
    const dim3 block_dim(wg_size, 1, 1);
    for (int i = 0; i < _comm_iters; i++)
    {
        for (int dev = 0; dev < _device_count; dev++)
        {
            _CHECK(hipSetDevice(dev));
            kernel<<<grid_dim, block_dim, 0, _comm_streams[dev]>>>(_src, _dst, dev, _links_per_device, block_size, link_stride_items, iters, active_threads);
        }
    }
}

void SetDefaults()
{
    _comm_streams.resize(_device_count);
    _gemm_streams.resize(_device_count);
    _gemm_runners.resize(_device_count);
    for (int dev = 0; dev < _device_count; dev++) {
        _CHECK(hipSetDevice(dev));
        _CHECK(hipStreamCreateWithFlags(&_comm_streams[dev], hipStreamNonBlocking));
        _CHECK(hipStreamCreateWithFlags(&_gemm_streams[dev], hipStreamNonBlocking));
        _gemm_runners[dev].Init(static_cast<uint64_t>(1)<<31, 8, 4096, 2048, 4096);
        _gemm_runners[dev].RunSelf(_gemm_streams[dev]);
    }
}

void SyncAllDevices() {
    for(int dev = 0; dev < _device_count; dev++)
    {
        _CHECK(hipSetDevice(dev));
        _CHECK(hipDeviceSynchronize());
    }
}

void LaunchGemm()
{
    for(int i=0; i<_gemm_iters; i++) {
        for(int dev = 0; dev < _device_count; dev++) {
            _CHECK(hipSetDevice(dev));
            _gemm_runners[dev].RunSelf(_gemm_streams[dev]);
        }
    }
}

int main(int argc, char **argv)
{
    _CHECK(hipGetDeviceCount(&_device_count));
    if (_device_count < 2)
    {
        printf("Insufficient number of GPU devices.\n");
        exit(-1);
    }
    else if (_device_count > _MaxDevices)
    {
        printf("Too many devices found, only %d will be used.\n", _MaxDevices);
        _device_count = _MaxDevices;
    }
    _links_per_device = _device_count - 1;

    SetDefaults();
    SetupTransfers();
    SyncAllDevices();

    // TODO : use conditional event for barrier
    std::this_thread::sleep_for(std::chrono::milliseconds(5000));
    
    // run gemm alone
    LaunchGemm();
    SyncAllDevices();

    // run communication alone
    // unroll=4, wg=256, cu=21
    // These parameters can achieve a bandwidth of 320 GB/s on the MI300X, which is close to saturation
    RunTest<__uint128_t>(4, 256, 21, 256);
    SyncAllDevices();

    // run gemm + communication overlap
    LaunchGemm();
    RunTest<__uint128_t>(4, 256, 21, 256);
    SyncAllDevices();

    DestroyTransfers();
    return 0;
}

