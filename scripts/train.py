import torch
from trl import SFTTrainer
from peft import LoraConfig
import argparse
from datasets import load_dataset, load_from_disk
from transformers.trainer_utils import get_last_checkpoint
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--training_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--data_field", type=str)
    args, _ = parser.parse_known_args()

    train_dataset = load_from_disk(args.training_dir)
    model_name = 'microsoft/Phi-3-mini-4k-instruct'
    llama_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name,
                                                       quantization_config=BitsAndBytesConfig(
                                                           load_in_4bit=True,
                                                           bnb_4bit_compute_dtype=getattr(torch, 'float16'),
                                                           bnb_4bit_quant_type='nf4'
                                                       ))

    llama_model.config.use_cache = False
    llama_model.config.pretraining_tp = 1

    llama_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name,
                                                    trust_remote_code=True)
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_tokenizer.padding_side = "right"

    training_args = TrainingArguments(per_device_train_batch_size=2,
                                        output_dir=args.output_dir,
                                        max_steps=100,
                                        eval_steps=5,
                                        save_strategy="steps",
                                        save_steps=5,
                                        report_to="tensorboard")

    trainer = SFTTrainer(model=llama_model,
                         args=training_args,
                         max_seq_length=256,
                         train_dataset=train_dataset,
                         tokenizer=llama_tokenizer,
                         dataset_text_field=args.data_field,
                         peft_config=LoraConfig(r=64, lora_alpha=16, lora_dropout=0.1, task_type='CAUSAL_LM', target_modules="all-linear",))

    if get_last_checkpoint(args.output_dir) is not None:
        last_checkpoint = get_last_checkpoint(args.output_dir)
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()

    trainer.train()

    trainer.save_model(args.output_dir)
    trainer.save_state()