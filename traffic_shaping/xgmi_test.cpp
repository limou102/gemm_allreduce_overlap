#include <iostream>
#include <cstdlib>
//#include <set>
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

size_t _target_transfer_size = 256ull * 1024 * 1024;
bool _local_read = true;
int _block_size = -1;
int _prewarm_iters = 100;
int _test_iters = 100;
bool _report_metric = true;
int _cu_min = 1;
int _cu_max = 1;
int _cu_step = 1;
int _dw_count = 4;

int _device_count = 0;
int _links_per_device = 0;
void** _src = {};
void** _dst = {};

std::vector<int> _wg_size_list;
std::vector<int> _unroll_list;

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

    hipEvent_t start[_MaxDevices];
    hipEvent_t stop[_MaxDevices];
    for (int dev = 0; dev < _device_count; dev++)
    {
        _CHECK(hipSetDevice(dev));
        _CHECK(hipEventCreate(&start[dev]));
        _CHECK(hipEventCreate(&stop[dev]));
    }

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
    for (int i = 0; i < _prewarm_iters; i++)
    {
        for (int dev = 0; dev < _device_count; dev++)
        {
            _CHECK(hipSetDevice(dev));
            kernel<<<grid_dim, block_dim, 0, 0>>>(_src, _dst, dev, _links_per_device, block_size, link_stride_items, iters, active_threads);
        }
    }
    for (int dev = 0; dev < _device_count; dev++)
    {
        _CHECK(hipSetDevice(dev));
        _CHECK(hipDeviceSynchronize());
    }

    std::vector<double> min_bw(_MaxDevices, __FLT_MAX__);
    std::vector<double> max_bw(_MaxDevices, 0.0f);
    std::vector<double> sum_bw(_MaxDevices, 0.0f);

    for (int i = 0; i < _test_iters; i++)
    {
        for (int dev = 0; dev < _device_count; dev++)
        {
            _CHECK(hipSetDevice(dev));
            hipExtLaunchKernelGGL(kernel, grid_dim, block_dim, 0, 0, start[dev], stop[dev], 0,
                _src, _dst, dev, _links_per_device, block_size, link_stride_items, iters, active_threads);
        }

        for (int dev = 0; dev < _device_count; dev++)
        {
            _CHECK(hipSetDevice(dev));
            _CHECK(hipEventSynchronize(stop[dev]));
            float t_ms;
            _CHECK(hipEventElapsedTime(&t_ms, start[dev], stop[dev]));
            double bw = (double)actual_transfer_size * _links_per_device / (t_ms  / 1000.0) / 1000000000;
            if (!_report_metric)
            {
                bw /= 1.024 * 1.024 * 1.024;
            }

            min_bw[dev] = std::min(min_bw[dev], bw);
            max_bw[dev] = std::max(max_bw[dev], bw);
            sum_bw[dev] += bw;
        }
    }

    for (int dev = 0; dev < _device_count; dev++)
    {
        double avg_bw = sum_bw[dev] / _test_iters;
        printf("%s,%d,%d,%d,%d,%.2f,%d,%.1f,%.1f,%.1f\n", _local_read ? "Y" : "N", wg_size, block_size, unroll, cu, used_cu, dev, min_bw[dev], max_bw[dev], avg_bw);
    }

    for (int dev = 0; dev < _device_count; dev++)
    {
        _CHECK(hipEventDestroy(start[dev]));
        _CHECK(hipEventDestroy(stop[dev]));
    }
}

void SetDefaults()
{
    for (int wg_size = 256; wg_size <= 1024; wg_size += 64)
    {
        _wg_size_list.push_back(wg_size);
    }
    for (int unroll = 1; unroll <= 8; unroll++)
    {
        _unroll_list.push_back(unroll);
    }
}

void LoadValList(std::vector<int>& d, const char* s)
{
    d.clear();

    char b[100];
    strncpy(b, s, 100);
    char* t = strtok(b, ",");
    while (t != NULL)
    {
        d.push_back(atoi(t));
        t = strtok(NULL, ",");
    }
}

void ParseParams(char** argv)
{
    int i = 1;
    while (argv[i] != NULL)
    {
        if (strcmp(argv[i], "-s") == 0)
        {
            if (argv[i + 1] == NULL)
            {
                printf("Missing per-link transfer size in MiB\n");
                exit(1);
            }
            _target_transfer_size = atoi(argv[i + 1]) * 1024 * 1024;
            i += 2;
        }
        else if (strcmp(argv[i], "-i") == 0)
        {
            if (argv[i + 1] == NULL)
            {
                printf("Missing test iterations\n");
                exit(1);
            }
            _test_iters = atoi(argv[i + 1]);
            i += 2;
        }
        else if (strcmp(argv[i], "-p") == 0)
        {
            if (argv[i + 1] == NULL)
            {
                printf("Missing prewarm iterations\n");
                exit(1);
            }
            _prewarm_iters = atoi(argv[i + 1]);
            i += 2;
        }
        else if (strcmp(argv[i], "-lr") == 0)
        {
            if (argv[i + 1] == NULL)
            {
                printf("Missing local read flag value\n");
                exit(1);
            }
            _local_read = atoi(argv[i + 1]) != 0;
            i += 2;
        }
        else if (strcmp(argv[i], "-m") == 0)
        {
            if (argv[i + 1] == NULL)
            {
                printf("Missing reported metric unit flag value\n");
                exit(1);
            }
            _report_metric = atoi(argv[i + 1]) != 0;
            i += 2;
        }
        else if (strcmp(argv[i], "-wg") == 0)
        {
            if (argv[i + 1] == NULL)
            {
                printf("Missing workgroup size\n");
                exit(1);
            }
            LoadValList(_wg_size_list, argv[i + 1]);
            i += 2;
        }
        else if (strcmp(argv[i], "-u") == 0)
        {
            if (argv[i + 1] == NULL)
            {
                printf("Missing unroll count\n");
                exit(1);
            }
            LoadValList(_unroll_list, argv[i + 1]);
            i += 2;
        }
        else if (strcmp(argv[i], "-cu") == 0)
        {
            if (argv[i + 1] == NULL)
            {
                printf("Missing CU count cpecification using {min,max,step} format\n");
                exit(1);
            }
            std::vector<int> cu_params;
            LoadValList(cu_params, argv[i + 1]);
            i += 2;

            if (cu_params.size() != 3)
            {
                printf("Incorrect CU count specification. Expected {min,max,step} format\n");
                exit(1);
            }
            _cu_min = cu_params[0];
            _cu_max = cu_params[1];
            _cu_step = cu_params[2];
        }
        else if (strcmp(argv[i], "-b") == 0)
        {
            if (argv[i + 1] == NULL)
            {
                printf("Missing block size\n");
                exit(1);
            }
            _block_size = atoi(argv[i + 1]);
            i += 2;
        }
        else if (strcmp(argv[i], "-d") == 0)
        {
            if (argv[i + 1] == NULL)
            {
                printf("Missing data size in DWORDs\n");
                exit(1);
            }
            _dw_count = atoi(argv[i + 1]);
            i += 2;
        }
        else
        {
            printf("Unknown parameter %s\n", argv[i]);
            exit(1);
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
    _cu_min = _links_per_device;
    _cu_max = _links_per_device * 16;
    _cu_step = _links_per_device;

    SetDefaults();
    if (argc > 1)
    {
        ParseParams(argv);
    }

    SetupTransfers();
    printf("local_read,wg_size,block_size,unroll,cu_target,cu_used,dev,min_bw,max_bw,avg_bw\n");
    for (auto wg_size : _wg_size_list)
    {
        for (auto unroll : _unroll_list)
        {
            for (int cu = _cu_min; cu <= _cu_max; cu += _cu_step)
            {
                int block_size = (_block_size == -1) ? wg_size : _block_size;
                switch (_dw_count)
                {
                    case 1: RunTest<uint32_t>(unroll, wg_size, cu, block_size); break;
                    case 2: RunTest<uint64_t>(unroll, wg_size, cu, block_size); break;
                    case 4: RunTest<__uint128_t>(unroll, wg_size, cu, block_size); break;
                    default: printf("Wrong data size\n");
                }
            }
        }
    }
    DestroyTransfers();

    return 0;
}

