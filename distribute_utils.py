import os
import torch
from utils import parse_args


class prioritize_thread:
    """
    Make sure the selected process first execute the code,
    code preprocessing
    """
    def __init__(self, local_rank, selected_rank):
        self.local_rank = local_rank
        self.selected_rank = selected_rank
    
    def __enter__(self):
        # if not the selected process, block the execution
        if self.local_rank != self.selected_rank:
            torch.distributed.barrier()
    
    def __exit__(self, type, value, traceback):
        # after the code block executed, sync 
        if self.local_rank == self.selected_rank:
            torch.distributed.barrier()  

class DistributedManager:
    def ___init__(self, args):
        self.local_rank = args.local_rank
    
    def rank(self):
        return self.local_rank

if __name__ == "__main__":
    args = parse_args()

    torch.distributed.init_process_group(backend='nccl')

    if args.local_rank == 0:
        print("world size", torch.distributed.get_world_size())

    torch.distributed.barrier()

    with prioritize_thread(args.local_rank, 0):
        print(args.local_rank)