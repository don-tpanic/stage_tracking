import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import time
import math
import deepspeed
import argparse
import numpy as np
import wandb 
import torch
from peft import PeftModel
from datasets import load_dataset
from torch.utils.data import DataLoader
import utils


def build_model(config, task='const-gen'):
    model, tokenizer = utils.load_model_and_tokenizer(
        model_fpath=config['model_fpath'],
        task=task,
        config=config
    )
    return model, tokenizer


def build_dataset(config):
    convo_format = config['convo_format']
    model_fpath = config['model_fpath']

    def preprocess_function(examples):
        user_tokens=[195]
        assistant_tokens=[196]
        # `examples` are the entire dataset with keys `text_bot`, `text_user`, `label`
        # For now we assume there is no situation where (text+label) in token id
        # space is greater than `max_length`.
        assistant_tokens = []
        input_ids = []
        attention_masks = []
        labels = []
        for text_bot, text_user, label in zip(examples['text_bot'], examples['text_user'], examples['label']):
            # Tokenize both bot and user text (with special tokens)
            # e.g. <a>...<u>...<a>
            input_id = assistant_tokens + tokenizer.encode(text_bot)
            input_id += user_tokens + tokenizer.encode(text_user) + assistant_tokens
            input_id_len = len(input_id)

            # Pad to max_length
            input_id = input_id + [tokenizer.pad_token_id] * (config["model_max_length"] - input_id_len)
            input_ids.append(input_id)

            # Tokenize label and left pad with -100 to the length of text,
            # and append eos_token to the end of label; right pad to max_length.
            # `(input_id_len-1)` is because the last token (<a>) in input_id should 
            # start predicting the label.
            label = tokenizer.encode(label)
            label = [-100] * (input_id_len-1) + label + [tokenizer.eos_token_id]
            label = label + [-100] * (config["model_max_length"] - len(label))
            labels.append(label)

            input_id = torch.LongTensor(input_id)
            attention_mask = input_id.ne(tokenizer.pad_token_id)
            attention_masks.append(attention_mask)
        
        examples['input_ids'] = input_ids
        examples['attention_mask'] = attention_masks
        examples['label'] = labels
        return examples        

    # Load dataset
    dataset = load_dataset(
        f"data/{convo_format}", 
        data_files={"train": "train.json", "eval": "eval.json"}
    )
    tokenizer = utils.load_model_and_tokenizer(
        model_fpath=model_fpath, 
        tokenizer_only=True, 
        task='const-gen',
        config=config
    )
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    return tokenized_dataset


def train(config, sweep_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # if not sweep, config will be 
    # the native config from yaml file
    if sweep_config is not None:
        wandb.init(config=sweep_config)
        config = wandb.config

    # Build model
    model, tokenizer = build_model(config)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config["learning_rate"], 
        weight_decay=config["weight_decay"]
    )

    # Build dataset
    tokenized_dataset = build_dataset(config)
    tokenized_dataset = tokenized_dataset.remove_columns(["text_bot"])
    tokenized_dataset = tokenized_dataset.remove_columns(["text_user"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")
    train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=1, shuffle=True)
    eval_dataloader = DataLoader(tokenized_dataset["eval"], batch_size=1, shuffle=True)

    # Training loop
    model.train()
    for epoch in range(config['num_train_epochs']):
        print(f"\nEpoch {epoch+1} training")
        for i, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / config["gradient_accumulation_steps"]
            loss.backward()
            if (i+1) % config["gradient_accumulation_steps"] == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Compute ppl
            train_ppl = math.exp(loss.item())
            wandb.log({"train_loss": loss.item(), "train_ppl": train_ppl})

        # Evaluation loop
        if (epoch+1) % 2 == 0:
            print(f"\nEpoch {epoch+1} evaluation")
            model.eval()
            total_loss = 0
            total_tokens = 0

            stage_labels = utils.label2id().keys()
            # stage_label: [0, 0], 0th=match, 1st=count
            total_match = {stage_label: [0, 1e-8] for stage_label in stage_labels}
            with torch.no_grad():
                for i, batch in enumerate(eval_dataloader):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    n_tokens = (batch["labels"] != -100).sum()
                    loss = outputs.loss
                    total_loss += loss.item() * n_tokens
                    total_tokens += n_tokens

                    # --- Visually checking generated text ---
                    # eval set originally uses right padding for loss eval,
                    # However, for generation we need to manually remove padding 
                    # ensure the last token of input_ids is not a padding token.
                    # ref - https://github.com/huggingface/transformers/issues/18388#issuecomment-1204369688
                    input_ids = batch["input_ids"]  # (bs, seq_len)
                    input_ids = input_ids[input_ids != tokenizer.pad_token_id].unsqueeze(0)
                    attention_mask = torch.ones_like(input_ids)

                    generation = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        do_sample=False,
                        temperature=0,
                        top_k=0,
                        return_dict_in_generate=True,
                        output_scores=True,
                        max_new_tokens=8,
                    )

                    # Decode generated text
                    prompt_len = input_ids.shape[1]
                    new_tokens_ids = generation.sequences[0][prompt_len:].detach().to('cpu').numpy()
                    pred_label = tokenizer.decode(new_tokens_ids)
                    true_label = tokenizer.decode(batch["labels"][batch["labels"] != -100].squeeze().tolist())
                    print('\npred_label before truncation:', pred_label)

                    pred_label = pred_label.replace('</s>', '').strip()
                    true_label = true_label.replace('</s>', '').strip()
                    true_label_len = len(true_label)
                    pred_label = pred_label[:true_label_len]
                    print(f"pred_label: {pred_label}")
                    print(f"true_label: {true_label}")

                    # Compute how many tokens are the same between pred and true label
                    # For now, we do not care about the order of the tokens
                    same_tokens = 0
                    pred_tokens = [token for token in pred_label]
                    true_tokens = [token for token in true_label]
                    for pred_token in pred_tokens:
                        if pred_token in true_tokens:
                            same_tokens += 1

                    # if same_tokens is >= 50% of the true_token, 
                    # we consider a match for now
                    print(f"pred_tokens={pred_tokens}, true_tokens={true_tokens}")
                    print(f"same_tokens={same_tokens}, true_label_len={true_label_len}")
                    if same_tokens / true_label_len >= 0.5:
                        print(f"[MATCH] same_tokens / true_label_len =", same_tokens / true_label_len)
                        total_match[true_label][0] += 1
                    total_match[true_label][1] += 1
                    print(f'total_match={total_match}')
                    # --- Visually checking generated text ---
                        
                average_val_loss = total_loss / total_tokens
                eval_ppl = math.exp(average_val_loss)
                logging_results = {
                    "eval_loss": average_val_loss,
                    "eval_ppl": eval_ppl,
                }
                for stage_label, match_count in total_match.items():
                    logging_results[f"{stage_label}"] = match_count[0] / match_count[1]
                print(f"logging_results={logging_results}")
                wandb.log(logging_results)
    
    # Clear RAM
    del model
    del optimizer
    del tokenized_dataset
    del train_dataloader
    del eval_dataloader
    torch.cuda.empty_cache()


def main(mode, config_version, hpt, sweep_count):
    config = utils.load_config(config_version)

    if mode == 'train':
        if hpt:
            sweep_config = {
                "name": config['config_version'],
                "method": "bayes",
                "metric": {"name": "eval_ppl", "goal": "minimize"},
                "parameters": {
                    # "learning_rate": {"distribution": "uniform", "min": 1e-8, "max": 1e-4},
                    "weight_decay": {"distribution": "uniform", "min": 0.0, "max": 0.1},
                },
                # TODO: Add early termination
                # "early_terminate": {
                #     "type": "hyperband",
                #     "s": 2,
                #     "eta": 3,
                #     "max_iter": 27,
                #     "metric": "eval_ppl",
                #     "goal": "minimize",
                # },
            }
            sweep_id = wandb.sweep(sweep_config)

            # Add the rest of the unchanged config to sweep_config
            for key, val in config.items():
                if key not in sweep_config["parameters"]:
                    sweep_config[key] = val

            # Run sweep agent
            wandb.agent(
                sweep_id, 
                lambda: train(config, sweep_config), 
                count=sweep_count
            )

            # TODO: Get the best run
            # best_run = wandb.runs(data_type="sweep", order_direction="desc")[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--logging', type=str, default='none')
    parser.add_argument('--config_version', type=str)
    parser.add_argument('--hpt', type=utils.str2bool, default=False)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--sweep_count', type=int, default=1)

    if parser.parse_args().logging == 'wandb':
        os.environ["WANDB_PROJECT"] = "text_classification"
        os.environ["WANDB_ENTITY"] = "neurowave-ai"

    main(
        mode=parser.parse_args().mode,
        config_version=f"config_{parser.parse_args().config_version}", 
        hpt=parser.parse_args().hpt,
        sweep_count=parser.parse_args().sweep_count
    )

    # python finetune.py --logging wandb --config_version 3 --hpt True --mode train
    # deepspeed --num_gpus=7 finetune.py --config_version 5