from transformers import AutoTokenizer, LlamaForSequenceClassification
from accelerate import Accelerator
import os
import time
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.wrap import (
    _or_policy,
    transformer_auto_wrap_policy,
    lambda_auto_wrap_policy,
    enable_wrap,
    wrap,
)
import torch.optim as optim
import functools
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import torch
import mlflow
from functools import partial

import tqdm


llama_model_name = "/home/ubuntu/llama-13b-hf"
tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
model = LlamaForSequenceClassification.from_pretrained(llama_model_name)


def custom_collate(batch):
    texts = [data["text"] for data in batch]
    labels = [data["label"] for data in batch]
    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "label": labels,
    }


def train(model, rank, world_size, train_loader, optimizer, scheduler, epoch):
    model.train()
    # local_rank = int(os.environ['LOCAL_RANK'])
    local_rank = rank

    fsdp_loss = torch.zeros(2).to(local_rank)

    if rank == 0:
        inner_pbar = tqdm.tqdm(
            range(len(train_loader)), colour="blue", desc="r0 Training Epoch"
        )
    for i, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(local_rank)
        attention_mask = batch["attention_mask"].to(local_rank)
        labels = torch.tensor(batch["label"]).to(local_rank)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs["loss"]
        loss.backward()

        fsdp_loss[0] += loss.item()
        fsdp_loss[1] += len(batch)
        optimizer.step()
        scheduler.step()

        if rank == 0:
            inner_pbar.update(1)

        if i % 5 == 0:
            dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
            loss = fsdp_loss[0] / fsdp_loss[1]
            fsdp_loss = torch.zeros(2).to(local_rank)

            if rank == 0:
                print("loss: ", loss)
                step = len(train_loader) * epoch + i
                mlflow.log_metric("loss", loss, step=step, synchronous=False)
                mlflow.log_metric(
                    "learning_rate",
                    scheduler.get_last_lr()[0],
                    step=step,
                    synchronous=False,
                )

    if rank == 0:
        inner_pbar.close()


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LlamaForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer


def lambda_policy_fn(module):
    if (
        len(list(module.named_children())) == 0
        and getattr(module, "weight", None) is not None
        and module.weight.requires_grad
    ):
        return True
    return False


def fsdp_main(rank, world_size, tokenizer, model):
    setup(rank, world_size)

    batch_size = 8
    num_epochs = 2
    save_model = True

    local_rank = rank

    mlflow.login()

    mlflow.set_experiment("/chenmoney-fsdp-on-a10-cluster")
    from datasets import load_dataset

    imdb = load_dataset("imdb")
    train_dataset = imdb["train"].select(range(2560))
    val_dataset = imdb["test"].select(range(1280))

    sampler1 = DistributedSampler(
        train_dataset, rank=rank, num_replicas=world_size, shuffle=True
    )
    sampler2 = DistributedSampler(val_dataset, rank=rank, num_replicas=world_size)

    train_kwargs = {"batch_size": batch_size, "sampler": sampler1}
    test_kwargs = {"batch_size": batch_size, "sampler": sampler2}
    cuda_kwargs = {
        "num_workers": 4,
        "pin_memory": True,
        "shuffle": False,
    }
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = DataLoader(train_dataset, collate_fn=custom_collate, **train_kwargs)
    val_loader = DataLoader(
        val_dataset,
        collate_fn=partial(custom_collate, tokenizer=tokenizer, model=model),
        **test_kwargs,
    )

    lambda_policy = functools.partial(
        lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn
    )
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=(LlamaDecoderLayer,),
    )

    auto_wrap_policy = functools.partial(
        _or_policy, policies=[lambda_policy, transformer_wrap_policy]
    )

    llama_auto_wrap_policy = auto_wrap_policy
    # sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    torch.cuda.set_device(local_rank)

    # model is on CPU before input to FSDP
    model = FSDP(
        model,
        auto_wrap_policy=llama_auto_wrap_policy,
        # sharding_strategy=sharding_strategy,
        device_id=torch.cuda.current_device(),
    )

    optimizer = optim.AdamW(model.parameters(), 5e-4)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer,
        total_iters=num_epochs * len(train_loader),
    )
    with mlflow.start_run():
        for epoch in range(num_epochs):
            t0 = time.time()
            train(model, rank, world_size, train_loader, optimizer, scheduler, epoch)

            if rank == 0:
                print(f"--> epoch {epoch} completed...entering save and stats zone")

            if save_model:
                # save
                if rank == 0:
                    print(f"--> entering save model state")

                save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(
                    model, StateDictType.FULL_STATE_DICT, save_policy
                ):
                    cpu_state = model.state_dict()
                # print(f"saving process: rank {rank}  done w state_dict")

                if rank == 0:
                    print(f"--> saving model ...")
                    currEpoch = "-" + str(epoch) + ".pt"
                    print(f"--> attempting to save model prefix {currEpoch}")
                    save_name = (
                        "saved_models/opt_text_classification/opt_finetuned" + currEpoch
                    )
                    print(f"--> saving as model name {save_name}")

                    torch.save(cpu_state, save_name)

    dist.barrier()
    cleanup()


if __name__ == "__main__":
    torch.manual_seed(42)

    # llama_model_name = "/home/ubuntu/llama-13b-hf"
    # model, tokenizer = load_model_and_tokenizer(llama_model_name)

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(
        fsdp_main, args=(WORLD_SIZE, tokenizer, model), nprocs=WORLD_SIZE, join=True
    )
