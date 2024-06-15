import functools
import os
import time

import mlflow
import torch
import torch.distributed as dist
import torch.optim as optim
import tqdm
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.distributed.fsdp import BackwardPrefetch, FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (MixedPrecision, ShardingStrategy,
                                    StateDictType)
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.wrap import (enable_wrap, lambda_auto_wrap_policy,
                                         transformer_auto_wrap_policy, wrap)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (AutoTokenizer, BitsAndBytesConfig, LlamaConfig,
                          LlamaForSequenceClassification)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


def train(args, model, rank, world_size, train_loader, optimizer, scheduler, epoch):
    model.train()
    local_rank = int(os.environ["LOCAL_RANK"])

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

        if i % 1 == 0:
            dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
            loss = fsdp_loss[0] / fsdp_loss[1]
            fsdp_loss = torch.zeros(2).to(local_rank)

            if rank == 0:
                print("loss: ", loss)
                print("logits: ", outputs.logits)
                print("label: ", labels)
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


def setup():
    # initialize the process group
    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()


def load_model_and_tokenizer(
    model_name, rank, quantization_config=None, lora_config=None
):
    if rank == 0:
        model = LlamaForSequenceClassification.from_pretrained(
            model_name,
            use_cache=False,
            quantization_config=quantization_config,
        )
    else:
        llama_config = LlamaConfig.from_pretrained(
            model_name,
            quantization_config=quantization_config,
        )
        llama_config.use_cache = False
        with torch.device("meta"):
            model = LlamaForSequenceClassification(llama_config)
    model = get_peft_model(model, lora_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    return model, tokenizer


def fsdp_main(args=None):
    model_name = "/home/ubuntu/llama-7b-hf"

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_has_fp16_weight=True,
    )
    lora_config = LoraConfig(
        r=16, lora_alpha=8, lora_dropout=0.05, bias="none", task_type="SEQ_CLS"
    )

    model, tokenizer = load_model_and_tokenizer(
        model_name, rank, quantization_config=None, lora_config=lora_config
    )

    batch_size = 4
    num_epochs = 2
    save_model = True

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

    setup()

    train_kwargs = {"batch_size": batch_size, "sampler": sampler1}
    test_kwargs = {"batch_size": batch_size, "sampler": sampler2}
    cuda_kwargs = {
        "num_workers": 4,
        "pin_memory": True,
        # 'shuffle': True,
    }
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

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

    train_loader = DataLoader(train_dataset, collate_fn=custom_collate, **train_kwargs)
    val_loader = DataLoader(val_dataset, collate_fn=custom_collate, **test_kwargs)

    def lambda_fn(module: nn.Module):
        if isinstance(module, LlamaDecoderLayer):
            return True  # like transformer_auto_wrap_policy
        if isinstance(module, torch.nn.Linear) and all(
            p.requires_grad for p in module.parameters()
        ):
            return True  # wrap each trainable linear separately
        return False

    llama_auto_wrap_policy = functools.partial(
        lambda_auto_wrap_policy, lambda_fn=lambda_fn
    )

    # llama_auto_wrap_policy = functools.partial(
    #     transformer_auto_wrap_policy,
    #     transformer_layer_cls={
    #         LlamaDecoderLayer,
    #     },
    # )
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    torch.cuda.set_device(local_rank)

    # model is on CPU before input to FSDP
    model = FSDP(
        model,
        auto_wrap_policy=llama_auto_wrap_policy,
        cpu_offload=CPUOffload(offload_params=True),
        sharding_strategy=sharding_strategy,
        mixed_precision=MixedPrecision(
            param_dtype=torch.float16, cast_forward_inputs=True
        ),
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        sync_module_states=True,
        param_init_fn=lambda module: module.to_empty(
            device=torch.device("cuda"), recurse=False
        )
        if rank != 0
        else None,
    )

    optimizer = optim.AdamW(model.parameters(), 5e-4)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer,
        total_iters=num_epochs * len(train_loader),
    )
    with mlflow.start_run():
        for epoch in range(num_epochs):
            t0 = time.time()
            train(
                args, model, rank, world_size, train_loader, optimizer, scheduler, epoch
            )

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
                        "saved_models/llama_text_classification/llama_finetuned"
                        + currEpoch
                    )
                    print(f"--> saving as model name {save_name}")

                    torch.save(cpu_state, save_name)

    dist.barrier()
    cleanup()


if __name__ == "__main__":
    torch.manual_seed(42)

    fsdp_main()
