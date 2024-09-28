import torch
from trl import SFTTrainer
from peft import LoraConfig
import argparse
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset, load_from_disk
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--dataset_dir", type=str)
    
    args, _ = parser.parse_known_args()


    model_name = 'microsoft/Phi-3-mini-4k-instruct'
    dataset_name = 'JM-Lee/Phi-3-mini-128K-instruct_instruction'
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name,
                                                    trust_remote_code=True)
    train_dataset =  load_dataset(path=dataset_name, split="train", cache_dir=args.dataset_dir)


    def merge_columns(examples):
        # Merge the 'system', 'instruction', and 'response' columns into one column
        merged_texts = [s + i + r for s, i, r in zip(examples['system'], examples['instruction'], examples['response'])]
        return {'merged_text': merged_texts}

    train_dataset = train_dataset.map(merge_columns, batched=True)
    train_dataset = train_dataset.remove_columns(['system', 'instruction', 'response'])

    def tokenize_function(examples):
        # Tokenize the new merged column
        return tokenizer(examples['merged_text'], padding="max_length", truncation=True)
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)

    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name,
                                                       quantization_config=BitsAndBytesConfig(
                                                           load_in_4bit=True,
                                                           bnb_4bit_compute_dtype=getattr(torch, 'float16'),
                                                           bnb_4bit_quant_type='nf4'
                                                       ))

    model.config.use_cache = False
    model.config.pretraining_tp = 1


    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    training_args = TrainingArguments(per_device_train_batch_size=2,
                                        output_dir=args.output_dir,
                                        max_steps=100,
                                        eval_steps=5,
                                        save_strategy="steps",
                                        save_steps=5,
                                        report_to="tensorboard")

    trainer = SFTTrainer(model=model,
                         args=training_args,
                         max_seq_length=256,
                         train_dataset=train_dataset,
                         tokenizer=tokenizer,
                         dataset_text_field='merged_text',
                         peft_config=LoraConfig(r=64, lora_alpha=16, lora_dropout=0.1, task_type='CAUSAL_LM', target_modules="all-linear",))

    if get_last_checkpoint(args.output_dir) is not None:
        last_checkpoint = get_last_checkpoint(args.output_dir)
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()

    trainer.train()

    trainer.save_model(args.output_dir)
    trainer.save_state()