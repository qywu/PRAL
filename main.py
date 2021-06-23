import os
import json
import numpy as np
import random
import time
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from allennlp.training.checkpointer import Checkpointer
import logging

from gpt_model import GPT2SimpleLM
from transformers import GPT2Tokenizer
from torchfly.common.download import get_pretrained_weights 
from transformers import AdamW, get_linear_schedule_with_warmup

from dialog_utils import DialogFragmentSampler
from distributed_utils import DistributedManager
from utils import parse_args, freeze_model, get_transformer_optim_params
from model import ARDM

logger = logging.getLogger(__name__)


class DialogCorpusDataset(Dataset):
    def __init__(self, data, tokenizer):
        # only interested in the values
        self.data = list(data.values())
        self.tokenizer = tokenizer
        self.tokenizer.max_len = 4096
        self.turn_ending = tokenizer.encode("\n\n\n")
        self.sampler = DialogFragmentSampler(max_tokens=800)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # get data
        sample = self.data[index]
        dialog = sample
        dialog_fragment = self.sampler(dialog)
        return dialog_fragment["token_ids"]

    def collate(self, inputs):
        """
        Return:
            A dict of pytorch tensors
        """
        # only one item in the batch

        total_len = sum([len(item) for item in inputs[0]])
        # make random positions
        start_position = random.randint(0, 1024 - total_len)

        position_ids = []
        for item in inputs[0]:
            pos = torch.arange(start_position,
                               start_position + len(item)).unsqueeze(0)
            position_ids.append(pos)
            start_position = start_position + len(item)

        inputs = [torch.LongTensor([item]) for item in inputs[0]]

        batch = {
            "input_ids": inputs,
            "position_ids": position_ids
        }

        return batch


def dialog_to_tensor(tokenzier, dialog, device=None):
    res = [torch.LongTensor([tokenizer.encode(item)]) for item in dialog]
    if device:
        res = [item.to(device) for item in res]
    return res

def batch_to_device(batch, device):
    new_batch = {}
    for key, value in batch.items():
        if isinstance(value, list):
            new_batch[key] = [tensor.to(device) for tensor in value]
        else:
            new_batch[key] = value.to(device)

    return new_batch


if __name__ == '__main__':
    from imp import reload
    reload(logging)
    LEVEL = logging.INFO

    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(format=format_str, 
                        level=LEVEL)
    args = parse_args()
    manager = DistributedManager(args)

    # define the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # construct dataset
    with open("dialog_corpus.json") as f:
        train_data = json.load(f)

    train_dataset = DialogCorpusDataset(train_data, tokenizer)

    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.batch_size,
                                  collate_fn=train_dataset.collate)

    # define the model
    model = ARDM(args)

    num_train_optimization_steps = (len(train_dataset) *
                                    args.num_train_epochs // args.batch_size //
                                    args.gradient_accumulation_steps //
                                    args.n_gpu)

    # dialog = dialog_to_tensor(tokenizer, dialog, device)
    optimizer_parameters = get_transformer_optim_params(args, model)
    optimizer = AdamW(optimizer_parameters, lr=args.learning_rate, eps=1e-06)

    if args.warmup_steps < 0:
        args.warmup_steps = int(args.warmup_ratio * len(train_dataset))

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                     num_warmup_steps=args.warmup_steps,
                                     num_training_steps=num_train_optimization_steps)

    manager.init_training(model, optimizer)

    update_count = 0
    if manager.is_main_rank():
        progress_bar = tqdm.tqdm
    else:
        progress_bar = iter

    if manager.is_main_rank():
        checkpointer = Checkpointer(
            "Checkpoint",
            keep_serialized_model_every_num_seconds=3600 * 4,
            num_serialized_models_to_keep=10)
        writer = SummaryWriter()
        start = time.time()
        update_loss = 0.0
        update_kl = 0.0

    for ep in range(args.num_train_epochs):
        pbar = progress_bar(train_dataloader)

        for batch in pbar:
            batch = batch_to_device(batch, args.device)

            loss, kl = model.train_one_step(batch)
            manager.backward_loss(loss, model, optimizer)
            update_count += 1

            if update_count % args.gradient_accumulation_steps == args.gradient_accumulation_steps - 1:
                manager.clip_grad_norm(model, optimizer)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # timer
                if manager.is_main_rank():
                    end = time.time()
                    speed = args.batch_size * args.n_gpu * args.gradient_accumulation_steps / (
                        end - start)
                    start = end
                    # show progress
                    pbar.set_postfix(loss=update_loss,
                                     kl=update_kl,
                                     speed=speed)

            # post-processing
            if manager.is_main_rank():
                pass
                if update_count % args.logging_steps == 0:
                    writer.add_scalar('loss', loss.item(), update_count)
                    writer.add_scalar('kl', kl, update_count)
                    update_loss = update_loss * 0.9 + 0.1 * loss.item()
                    update_kl = update_kl * 0.9 + 0.1 * kl

                # saving models

                if update_count % args.save_steps == 0:
                    checkpointer.save_checkpoint(update_count,
                                                 model.state_dict(),
                                                 optimizer.state_dict(),
                                                 is_best_so_far=True)
