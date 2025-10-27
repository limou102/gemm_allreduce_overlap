set -e

export NCCL_SOCKET_IFNAME=lo

# from chen,yuankai, the optimal value for compute-communication overlap
#export GPU_MAX_HW_QUEUES=8

# m,n,k for gemm
gemm_shapes=(
#"3840 3840 4352 6553600"
#"256 3840 3840 6553600"
#"3840 3840 21760 6553600"
"4352 3840 3840 6553600"
#"21760 3840 3840"
#"2304 3840 3840"
#"5120 5120 5120"
)

common_command='torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint="localhost:0"'


for shape in "${gemm_shapes[@]}"; do
    read -r m n k comm_size<<< "$shape"
    launch_command_rccl="${common_command} benchmark_rccl_allreduce_hipblaslt_cumask.py -m $m -n $n -k $k --comm-size $comm_size --num-comm-cu -1 2>/dev/null"
    echo "launch with command : $launch_command_rccl"
    eval $launch_command_rccl

    echo ""
done

