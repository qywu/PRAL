import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import logging

try:
    from apex import amp
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex to run this example."
    )

logger = logging.getLogger(__name__)


class DistributedManager:
    """
    Set Up Distributed Training
    Should be Distributed Agnostic (Works both distributed and non-distributed)
    """
    def __init__(self, args):
        self.args = args
        self.rank = args.local_rank
        self.main_rank = 0

        if args.local_rank != -1:
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            torch.distributed.init_process_group(backend='nccl')
            self.args.device = device
            # get the world size
            self.args.n_gpu = torch.distributed.get_world_size()
        else:
            # no distributed traing
            # set default cuda device
            self.args.device = torch.device("cuda")
            self.args.n_gpu = 1

    def init_training(self, models, optimizers, num_losses=1):
        """
        num_losses: for GAN training
        """
        # send to device
        if isinstance(models, list):
            models = [model.to(self.args.device) for model in models]
        else:
            models = models.to(self.args.device)

        if self.args.fp16:
            models, optimizers = amp.initialize(
                models,
                optimizers,
                opt_level=self.args.fp16_opt_level,
                num_losses=num_losses
            )

        # Distributed training (should be after apex fp16 initialization)
        if self.rank != -1:
            if isinstance(models, list):
                models = [
                    DistributedDataParallel(
                        model,
                        device_ids=[self.rank],
                        output_device=self.rank,
                        find_unused_parameters=True
                    ) for model in models
                ]
            else:
                models = DistributedDataParallel(
                    models,
                    device_ids=[self.rank],
                    output_device=self.rank,
                    find_unused_parameters=True
                )

        # send to device

        logger.warning(
            f"Process rank: {self.args.local_rank}, "
            f"device: {self.args.device}, "
            f"n_gpu: {self.args.n_gpu}, "
            f"distributed training: {self.args.local_rank != -1}, "
            f"16-bits training: {self.args.fp16}"
        )

        return models, optimizers

    def is_main_rank(self):
        if self.rank in [-1, self.main_rank]:
            return True
        else:
            return False

    def backward_loss(self, loss, model, optimizer, loss_id=0):
        if self.args.fp16:
            with amp.scale_loss(
                loss, optimizer, loss_id=loss_id
            ) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

    def clip_grad_norm(self, model, optimizer):
        if self.args.max_grad_norm > 0.0:
            if self.args.fp16:
                torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), self.args.max_grad_norm
                )
            else:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.args.max_grad_norm
                )

    # def acquire_rank(self, rank=0):
    #     if self.args.local_rank not in [-1, rank]:
    #         torch.distributed.barrier()

    # def release_rank(self, rank=0):
    #     if self.args.local_rank == rank:
    #         torch.distributed.barrier()
