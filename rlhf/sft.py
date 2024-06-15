from trl.trainer import ConstantLengthDataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from accelerate import Accelerator
from trl import SFTTrainer
from datasets import load_dataset
from transformers import TrainingArguments
from peft import LoraConfig, TaskType
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch

max_seq_length = 512


model_name = "/home/ubuntu/llama-13b-hf"


def create_datasets(tokenizer):
    dataset = load_dataset("Anthropic/hh-rlhf")

    train_data = dataset["train"]
    valid_data = dataset["test"]
    print(
        f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}"
    )

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        infinite=True,
        seq_length=max_seq_length,
        formatting_func=lambda x: x["chosen"],
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        infinite=True,
        seq_length=max_seq_length,
        formatting_func=lambda x: x["chosen"],
    )
    return train_dataset, valid_dataset


def fsdp_main():
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = TrainingArguments(
        output_dir="./checkpoints",
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=2000,
        max_steps=10000,
        eval_steps=1000,
        logging_steps=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        gradient_accumulation_steps=1,
        bf16=True,
        weight_decay=0.05,
        run_name="rlhf-llama-sft",
        report_to="wandb",
        ddp_find_unused_parameters=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        # device_map={"": Accelerator().process_index},
    )

    import pdb

    pdb.set_trace()

    print("GEEZ!!")


# train_data, val_data = create_datasets(tokenizer)

# trainer = SFTTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_data,
#     eval_dataset=val_data,
#     peft_config=lora_config,
#     packing=True,
# )

# trainer.train()
# trainer.save_model("./saved_sft")

if __name__ == "__main__":
    fsdp_main()
