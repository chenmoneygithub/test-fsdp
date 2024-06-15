from transformers import AutoTokenizer, OPTForSequenceClassification, OPTConfig
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
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
import torch.optim as optim
import functools
from transformers.models.opt.modeling_opt import OPTDecoderLayer
import torch
import mlflow
import tqdm


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


def setup():
    # initialize the process group
    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()


def load_model_and_tokenizer(model_name, rank):
    if rank == 0:
        model = OPTForSequenceClassification.from_pretrained(model_name)
    else:
        opt_config = OPTConfig.from_pretrained(model_name)
        # opt_config.use_cache = use_cache
        with torch.device("meta"):
            model = OPTForSequenceClassification(opt_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = OPTForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer


def fsdp_main(args=None):
    opt_model_name = "facebook/opt-350m"

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    model, tokenizer = load_model_and_tokenizer(opt_model_name, rank)

    batch_size = 8
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
        "shuffle": False,
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

    opt_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            OPTDecoderLayer,
        },
    )
    sharding_strategy: ShardingStrategy = (
        ShardingStrategy.SHARD_GRAD_OP
    )  # for Zero2 and FULL_SHARD for Zero3
    torch.cuda.set_device(local_rank)

    # model is on CPU before input to FSDP
    model = FSDP(
        model,
        auto_wrap_policy=opt_auto_wrap_policy,
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
                        "saved_models/opt_text_classification/opt_finetuned" + currEpoch
                    )
                    print(f"--> saving as model name {save_name}")

                    torch.save(cpu_state, save_name)

    dist.barrier()
    cleanup()


if __name__ == "__main__":
    torch.manual_seed(42)

    fsdp_main()
