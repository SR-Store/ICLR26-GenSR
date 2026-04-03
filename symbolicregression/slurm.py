
from logging import getLogger
import os
import sys
import torch
import socket
import signal
import subprocess


logger = getLogger()


def sig_handler(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    prod_id = int(os.environ["SLURM_PROCID"])
    logger.warning("Host: %s - Global rank: %i" % (socket.gethostname(), prod_id))
    if prod_id == 0:
        logger.warning("Requeuing job " + os.environ["SLURM_JOB_ID"])
        os.system("scontrol requeue " + os.environ["SLURM_JOB_ID"])
    else:
        logger.warning("Not the master process, no need to requeue.")
    sys.exit(-1)


def term_handler(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    logger.warning("Bypassing SIGTERM.")


def init_signal_handler():
    signal.signal(signal.SIGUSR1, sig_handler)
    signal.signal(signal.SIGTERM, term_handler)
    logger.warning("Signal handler installed.")


def init_distributed_mode(params):
    params.is_slurm_job = "SLURM_JOB_ID" in os.environ and not params.debug_slurm
    print("SLURM job: %s" % str(params.is_slurm_job))

    if params.is_slurm_job:

        assert params.local_rank == -1

        SLURM_VARIABLES = [
            "SLURM_JOB_ID",
            "SLURM_JOB_NODELIST",
            "SLURM_JOB_NUM_NODES",
            "SLURM_NTASKS",
            "SLURM_TASKS_PER_NODE",
            "SLURM_MEM_PER_NODE",
            "SLURM_MEM_PER_CPU",
            "SLURM_NODEID",
            "SLURM_PROCID",
            "SLURM_LOCALID",
            "SLURM_TASK_PID",
        ]

        PREFIX = "%i - " % int(os.environ["SLURM_PROCID"])
        for name in SLURM_VARIABLES:
            value = os.environ.get(name, None)
            print(PREFIX + "%s: %s" % (name, str(value)))


        params.n_nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
        params.node_id = int(os.environ["SLURM_NODEID"])

        params.local_rank = int(os.environ["SLURM_LOCALID"])
        params.global_rank = int(os.environ["SLURM_PROCID"])

        params.world_size = int(os.environ["SLURM_NTASKS"])
        params.n_gpu_per_node = params.world_size // params.n_nodes

        hostnames = subprocess.check_output(
            ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]]
        )
        params.master_addr = hostnames.split()[0].decode("utf-8")
        print(PREFIX + "Master address: %s" % params.master_addr)
        print(PREFIX + "Master port   : %i" % params.master_port)

        os.environ["MASTER_ADDR"] = params.master_addr
        os.environ["MASTER_PORT"] = str(params.master_port)
        os.environ["WORLD_SIZE"] = str(params.world_size)
        os.environ["RANK"] = str(params.global_rank)

    elif params.local_rank != -1 or "LOCAL_RANK" in os.environ:

        assert params.master_port == -1

        if params.local_rank == -1:
            params.local_rank = int(os.environ["LOCAL_RANK"])
        params.global_rank = int(os.environ["RANK"])
        params.world_size = int(os.environ["WORLD_SIZE"])
        params.n_gpu_per_node = int(os.environ.get("NGPU", os.environ.get("LOCAL_WORLD_SIZE", 1)))

        params.n_nodes = params.world_size // params.n_gpu_per_node
        params.node_id = params.global_rank // params.n_gpu_per_node

    else:
        assert params.local_rank == -1
        assert params.master_port == -1
        params.n_nodes = 1
        params.node_id = 0
        params.local_rank = 0
        params.global_rank = 0
        params.world_size = 1
        params.n_gpu_per_node = 1

    assert params.n_nodes >= 1
    assert 0 <= params.node_id < params.n_nodes
    assert 0 <= params.local_rank <= params.global_rank < params.world_size
    assert params.world_size == params.n_nodes * params.n_gpu_per_node

    params.is_master = params.node_id == 0 and params.local_rank == 0
    params.multi_node = params.n_nodes > 1
    params.multi_gpu = params.world_size > 1

    PREFIX = "%i - " % params.global_rank
    print(PREFIX + "Number of nodes: %i" % params.n_nodes)
    print(PREFIX + "Node ID        : %i" % params.node_id)
    print(PREFIX + "Local rank     : %i" % params.local_rank)
    print(PREFIX + "Global rank    : %i" % params.global_rank)
    print(PREFIX + "World size     : %i" % params.world_size)
    print(PREFIX + "GPUs per node  : %i" % params.n_gpu_per_node)
    print(PREFIX + "Master         : %s" % str(params.is_master))
    print(PREFIX + "Multi-node     : %s" % str(params.multi_node))
    print(PREFIX + "Multi-GPU      : %s" % str(params.multi_gpu))
    print(PREFIX + "Hostname       : %s" % socket.gethostname())

    if not params.cpu:
        torch.cuda.set_device(params.local_rank)

    if params.multi_gpu:


        print("Initializing PyTorch distributed ...")
        torch.distributed.init_process_group(
            init_method="env://", backend="nccl",
        )
        print(f"PyTorch distributed initialized successfully (rank {params.global_rank})")

