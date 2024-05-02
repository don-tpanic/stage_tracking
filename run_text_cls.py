import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import deepspeed
import argparse
import numpy as np
import wandb 
import torch
from peft import PeftModel
from datasets import load_dataset
from torch.utils.data import DataLoader
import transformers
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback

import utils


def compute_metrics(eval_pred):
    res = {}
    # top1 and top3 accuracy
    predictions, labels = eval_pred
    # predictions (batch_size, num_labels)
    # labels (batch_size, )
    predictions = np.argsort(predictions, axis=1)
    top1 = predictions[:, -1]
    top3 = predictions[:, -3:]
    top1_acc = np.mean(top1 == labels)
    top3_acc = np.mean(np.any(top3 == labels[:, None], axis=1))

    res["top1_acc"] = top1_acc
    res["top3_acc"] = top3_acc

    # top1 and top3 accuracy per label keys
    keys  = [f'top1_acc_stage_{i}' for i in range(len(utils.label2id()))]
    keys += [f'top3_acc_stage_{i}' for i in range(len(utils.label2id()))]

    for key in keys:
        i = int(key.split('_')[-1])
        top1_acc_per_label = np.mean(top1[labels == i] == labels[labels == i])
        top3_acc_per_label = np.mean(np.any(top3[labels == i] == labels[labels == i][:, None], axis=1))
        res[f"top1_acc_stage_{i}"] = top1_acc_per_label
        res[f"top3_acc_stage_{i}"] = top3_acc_per_label

    return res


def train(config_version, hpt):
    config = utils.load_config(config_version)
    convo_format = config['convo_format']
    model_fpath = config['model_fpath']

    def model_init():
        print("[[[ Initializing model ]]]")
        model, _ = utils.load_model_and_tokenizer(
            model_fpath=model_fpath,
            num_labels=len(utils.label2id()),
            id2label=utils.id2label(),
            label2id=utils.label2id(),
            task='seq-cls',
            config=config
        )
        return model

    def preprocess_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    def wandb_hp_space(trial):
        return {
            "name": config['config_version'],
            "method": "bayes",
            "metric": {"name": "objective", "goal": "maximize"},
            "parameters": {
                "learning_rate": {"distribution": "uniform", "min": 1e-8, "max": 1e-4},
                "weight_decay": {"distribution": "uniform", "min": 0.0, "max": 0.1},
            },
        }

    # Load dataset
    dataset = load_dataset(f"data/{convo_format}", data_files={"train": "train.json", "eval": "eval.json"})
    tokenizer = utils.load_model_and_tokenizer(model_fpath=model_fpath, tokenizer_only=True, config=config)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    print(tokenized_dataset)
    
    # Training arguments
    if config['deepspeed']:
        # https://github.com/huggingface/transformers/issues/24445
        ds_config_file = "ds_config.json"
        kwargs = dict(deepspeed=ds_config_file)
        print(
            "Using DeepSpeed ZeRO-3 with config file:",
            kwargs
        )
    else:
        kwargs = {}

    training_args = TrainingArguments(
        output_dir=f"trained_models/{config_version}",
        num_train_epochs=30,
        learning_rate=config["learning_rate"],
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=parser.parse_args().logging,
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        fp16=config["fp16_training"],
        **kwargs,
    )

    world_size = training_args.world_size
    process_index = training_args.process_index
    local_process_index = training_args.local_process_index
    print(
        f"world_size: {world_size}, process_index: {process_index}, local_process_index: {local_process_index}"
    )

    if hpt:
        print("Hyperparameter tuning")
        trainer = Trainer(
            model_init=model_init,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["eval"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )
        best_trial = trainer.hyperparameter_search(
            direction="maximize",
            backend="wandb",
            hp_space=wandb_hp_space,
            n_trials=10,
            # compute_objective=compute_objective,
        )
        print(best_trial)
        # BestRun(run_id='hi0qvrrx', objective=nan, hyperparameters={'learning_rate': 8.044073472981232e-05, 'weight_decay': 0.06049261086119254, 'assignments': {}, 'metric': 'eval/loss'}, run_summary=None)
        wandb.finish()

        # Delete trials weights except the best one from directory
        # And move the best one to `config_version` directory
        for trial in os.listdir(f"trained_models/{config_version}"):
            if trial != f"run-{best_trial.run_id}":
                print(f'Deleting {trial}, except run-{best_trial.run_id}')
                os.system(f"rm -rf trained_models/{config_version}/{trial}")
            else:
                print(f'Moving {trial} to {config_version}')
                os.system(f"mv trained_models/{config_version}/{trial}/*/* trained_models/{config_version}/")

    else:
        print("Single run")
        trainer = Trainer(
            model=model_init(),
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["eval"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )
        trainer.train()


def test(config_version):
    """
    Load best model from hpt and evaluate acc on test set.
    """
    config = utils.load_config(config_version)

    # LoRA only saves adapter weights.
    if config["trainable_weights"] == 'lora':
        model, tokenizer = utils.load_model_and_tokenizer(
            model_fpath=config["model_fpath"],
            num_labels=len(utils.label2id()),
            id2label=utils.id2label(),
            label2id=utils.label2id(),
            task='seq-cls',
            config=config
        )

        peft_model_id = f"trained_models/{config_version}"
        model = PeftModel.from_pretrained(
            model, peft_model_id
        )
    else:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            f"trained_models/{config_version}",
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            f"trained_models/{config_version}",
        )

    # Load dataset
    convo_format = config['convo_format']
    def preprocess_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    dataset = load_dataset(f"data/{convo_format}", data_files={"test": "test.json"})
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")
    test_dataloader = DataLoader(tokenized_dataset["test"], batch_size=1)

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Testing loop with metric computation
    with torch.no_grad():
        all_predictions = []
        all_labels = []
        
        for ct, batch in enumerate(test_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = outputs.logits.cpu().numpy()
            labels = batch['labels'].cpu().numpy()
            all_predictions.append(predictions)
            all_labels.append(labels)

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    metrics = compute_metrics((all_predictions, all_labels))
    for key, val in metrics.items():
        print(f"{key}: {val}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--logging', type=str, default='none')
    parser.add_argument('--config_version', type=str)
    parser.add_argument('--hpt', type=utils.str2bool, default=False)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--local_rank', type=int, default=0)

    if parser.parse_args().logging == 'wandb':
        os.environ["WANDB_PROJECT"] = "text_classification"
        os.environ["WANDB_ENTITY"] = "neurowave-ai"

    if parser.parse_args().mode == 'train':
        train(
            config_version=f"config_{parser.parse_args().config_version}", 
            hpt=parser.parse_args().hpt
        )
    
    elif parser.parse_args().mode == 'test':
        test(
            config_version=f"config_{parser.parse_args().config_version}", 
        )

    # python finetune.py --logging wandb --config_version 3 --hpt True --mode train
    # deepspeed --num_gpus=7 finetune.py --config_version 5