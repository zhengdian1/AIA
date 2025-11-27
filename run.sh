# !/bin/bash

set -e

export TORCH_CPP_LOG_LEVEL=ERROR
export NCCL_DEBUG=WARN
ulimit -n 80000

cluster_spec="${AFO_ENV_CLUSTER_SPEC}"
role=$(jq -r .role <<< "$cluster_spec")
if [ "$role" != "worker" ]; then
    echo "Error: $role vs worker" >&2
    exit 1
fi

node_rank=$(jq -r .index <<< "$cluster_spec")
nnodes=$(jq -r '.worker | length' <<< "$cluster_spec")
nproc_per_node=$(nvidia-smi --list-gpus | wc -l | tr -d ' ')

master=$(jq -r '.worker[0]' <<< "$cluster_spec")
IFS=":" read -r master_addr ports <<< "$master"
IFS="," read -ra master_ports <<< "$ports"
master_port=${master_ports[0]}

echo "master=$master_addr, master_port=$master_port, nproc_per_node=$nproc_per_node, nnodes=$nnodes, node_rank=$node_rank"

export PYTHONPATH=$(pwd)

torchrun \
    --nproc_per_node=$nproc_per_node \
    --nnodes=$nnodes \
    --node_rank=$node_rank \
    --master_addr=$master_addr \
    --master_port=$master_port \
    train.py \
    --backend=nccl \
    --yml_path=configs/t2i_generation.yml