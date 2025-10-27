from itertools import product
import json
import os
from datetime import datetime
import argparse
from contextlib import nullcontext
from statistics import mean

import torch
import torch.distributed as dist
import torch.profiler

from primus_turbo.pytorch.core import TurboStream

def benchmark(
    matmul_size,
    comm_size,
    matmul_stream,
    comm_stream,
    world_size,
    warmup_steps=10,
    benchmark_steps=30,
):
    torch.cuda.empty_cache()
    with torch.device("cuda"):
        A = torch.randn(*matmul_size[0], dtype=torch.bfloat16)
        B = torch.randn(*matmul_size[1], dtype=torch.bfloat16)
        comm_tensor = torch.randn(*comm_size, dtype=torch.float32)

    # warmup matmul
    for _ in range(warmup_steps):
        with torch.cuda.stream(matmul_stream):
            torch.matmul(A, B)
    torch.cuda.synchronize()

    # benchmark matmul
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record(matmul_stream)
    for _ in range(benchmark_steps):
        with torch.cuda.stream(matmul_stream):
            torch.matmul(A, B)
    end_event.record(matmul_stream)
    torch.cuda.synchronize()

    matmul_time = start_event.elapsed_time(end_event) / benchmark_steps

    # warmup comm (all_reduce)
    for _ in range(warmup_steps):
        with torch.cuda.stream(comm_stream):
            dist.all_reduce(comm_tensor)
    torch.cuda.synchronize()

    # benchmark comm
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record(comm_stream)
    for _ in range(benchmark_steps):
        with torch.cuda.stream(comm_stream):
            dist.all_reduce(comm_tensor)
    end_event.record(comm_stream)
    torch.cuda.synchronize()

    comm_time = start_event.elapsed_time(end_event) / benchmark_steps

    # warmup matmul-comm
    for _ in range(warmup_steps):
        with torch.cuda.stream(matmul_stream):
            torch.matmul(A, B)
        with torch.cuda.stream(comm_stream):
            dist.all_reduce(comm_tensor)
    torch.cuda.synchronize()

    # benchmark matmul-comm
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_matmul_events = [torch.cuda.Event(enable_timing=True) for _ in range(benchmark_steps)] 
    end_matmul_events = [torch.cuda.Event(enable_timing=True) for _ in range(benchmark_steps)] 
    start_comm_events = [torch.cuda.Event(enable_timing=True) for _ in range(benchmark_steps)] 
    end_comm_events = [torch.cuda.Event(enable_timing=True) for _ in range(benchmark_steps)] 

    start_event.record()

    for i in range(benchmark_steps):
        start_matmul_events[i].record(matmul_stream)
        with torch.cuda.stream(matmul_stream):
            torch.matmul(A, B)
        end_matmul_events[i].record(matmul_stream)

        start_comm_events[i].record(comm_stream)
        with torch.cuda.stream(comm_stream):
            dist.all_reduce(comm_tensor)
        end_comm_events[i].record(comm_stream)
        torch.cuda.synchronize()

    torch.cuda.current_stream().wait_stream(comm_stream)
    end_event.record()

    torch.cuda.synchronize()

    matmul_comm_time = start_event.elapsed_time(end_event) / benchmark_steps
    overlapped_matmul_time = mean([start_matmul_events[i].elapsed_time(end_matmul_events[i]) for i in range(benchmark_steps)])
    overlapped_comm_time = mean([start_comm_events[i].elapsed_time(end_comm_events[i]) for i in range(benchmark_steps)])

    return matmul_time, comm_time, matmul_comm_time, overlapped_matmul_time, overlapped_comm_time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse matrix dimensions and configuration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=8192, help="Number of rows in matrix A (GEMM)")
    parser.add_argument("-n", type=int, default=4608, help="Number of columns in matrix B (GEMM)")
    parser.add_argument("-k", type=int, default=36864, help="Common dimension between matrices A and B (GEMM)")
    parser.add_argument(
        "--comm-size", type=int, default=None, help="Number of float32 elements for communication tensor", required=True)
    parser.add_argument("--num-comm-cu", type=int, default=None, help="number of CU for communication stream, default None means use all compute units")
    parser.add_argument("-p", "--profile", action="store_true", help="Enable PyTorch profiler to generate chrome trace")
    return parser.parse_args()

def get_cu_masks(num_cu : int, num_cu_enable : int, enable_from_left : bool):
    num_pad = (32 - num_cu % 32) % 32
    if enable_from_left:
        bits = '1' * num_cu_enable + '0' * (num_cu - num_cu_enable) + '0' * num_pad
    else:
        bits = '0' * num_pad + '0' * (num_cu - num_cu_enable) + '1' * num_cu_enable

    masks = []
    for i in range(0, len(bits), 32):
        chunk = bits[i:i+32]
        masks.append(int(chunk, 2))
    return masks

if __name__ == "__main__":
    args = parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    num_cu = torch.cuda.get_device_properties(0).multi_processor_count
    if args.num_comm_cu and (args.num_comm_cu >= num_cu or args.num_comm_cu <= 0):
        args.num_comm_cu = None

    matmul_stream = torch.cuda.Stream()
    #matmul_stream = TurboStream(device=f"cuda:{local_rank}",
    #    cu_masks = get_cu_masks(num_cu, num_cu-args.num_comm_cu, False))

    if args.num_comm_cu is None:
        comm_stream = torch.cuda.Stream()
    else:
        comm_stream = TurboStream(device=f"cuda:{local_rank}",
            cu_masks = get_cu_masks(num_cu, args.num_comm_cu, True))
        if rank == 0:
            print(f"limit {args.num_comm_cu} CUs for communication stream")

    matmul_sizes = [((args.m, args.k), (args.k, args.n))]

    comm_sizes = [(args.comm_size,)]

    if rank == 0:
        print(f"Using RCCL all_reduce with world_size={world_size}")
        print(f"bfloat16, Matmul: A @ B where A={matmul_sizes[0][0]}, B={matmul_sizes[0][1]}")
        print(f"float32, Comm size: {comm_sizes[0]}")

    profiler_context = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        with_stack=True,
    ) if args.profile else nullcontext()
    with profiler_context as prof:
        matmul_size = matmul_sizes[0]
        comm_size = comm_sizes[0]
        matmul_time, comm_time, matmul_comm_time, overlapped_matmul_time, overlapped_comm_time = benchmark(
            matmul_size,
            comm_size,
            matmul_stream.torch_stream if hasattr(matmul_stream, "torch_stream") else matmul_stream,
            comm_stream.torch_stream if hasattr(comm_stream, "torch_stream") else comm_stream,
            world_size,
        )

        size_A, size_B = matmul_size
        if rank == 0:
            print("-" * 60)
            print(
                f"A: {size_A[0]}x{size_A[1]} @ B: {size_B[0]}x{size_B[1]}, comm: {comm_size[0]}",
            )
            print(f"  matmul alone:         {matmul_time:.4f} ms")
            print("      tflops:  {:.4f}".format(2.0*args.m*args.n*args.k/matmul_time*1000/(2**40)))
            print(f"  comm alone:           {comm_time:.4f} ms")
            print("      busbw:  {:.4f} GB/s".format(args.m*args.n*4.0/comm_time*1000*2*(world_size-1)/world_size/1e9))
            print(f"  matmul + comm:        {matmul_comm_time:.4f} ms")
            print(f"  overlapped matmul:    {overlapped_matmul_time:.4f} ms")
            print("      tflops: {:.4f}".format(2.0*args.m*args.n*args.k/overlapped_matmul_time*1000/(2**40)))
            print(f"  overlapped comm:      {overlapped_comm_time:.4f} ms")
            print("      busbw:  {:.4f} GB/s".format(args.m*args.n*4.0/overlapped_comm_time*1000*2*(world_size-1)/world_size/1e9))
            print("-" * 60)

    if rank == 0 and args.profile:
        prof.export_chrome_trace(f"rccl_allreduce_trace_rank{rank}.json")
        print(f"Profiler trace saved to rccl_allreduce_trace_rank{rank}.json")

    dist.destroy_process_group()

