import os
import yaml
import argparse
import torch
import transformers
from peft import (
    get_peft_model,
    prepare_model_for_int8_training,
    LoraConfig,
)
import deepspeed


def label2id():
    pass


def id2label():
    pass


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_config(config_version):
    with open(os.path.join(f'configs', f'{config_version}.yaml')) as f:
        config = yaml.safe_load(f)
    return config


def trainable_parameters(model):
    trainable_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
    total_params = sum(p.numel() for n, p in model.named_parameters())
    print(f"Trainable params: {trainable_params}, Total params: {total_params}")


def load_model_and_tokenizer(
        model_fpath,
        num_labels=None,
        id2label=None,
        label2id=None,
        task='seq-cls',
        tokenizer_only=False,
        load_in_8bit=False,
        torch_dtype=torch.float32,
        device_map=None,
        config=None,
    ):
    if task == 'seq-cls':
        # Tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_fpath,
            use_auth_token=True,
            trust_remote_code=True,
        )
        if 'Baichuan' in model_fpath:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.model_max_length = config["model_max_length"]
            if config["load_in_8bit"]:
                load_in_8bit = True
                device_map = 'auto'
            if config["load_in_fp16"]:
                torch_dtype = torch.float16
            if config["deepspeed"]:
                device_map = None  
                # Accelerate and DeepSpeed incompatible: 
                # ref1 - https://github.com/huggingface/accelerate/issues/1815#event-10090483529
                # ref2 - https://github.com/microsoft/DeepSpeed/issues/3028#issuecomment-1676267626

        if tokenizer_only:
            return tokenizer
        
        # Model
        if 'baichuan' not in model_fpath:
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                model_fpath,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
                load_in_8bit=load_in_8bit,
                torch_dtype=torch_dtype,
                device_map=device_map,
                use_auth_token=True,
                trust_remote_code=True,
            )
        else:
            from model_zoo.Baichuan_7B.models import modeling_baichuan
            model = modeling_baichuan.BaiChuanForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=model_fpath,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
                load_in_8bit=load_in_8bit,
                torch_dtype=torch_dtype,
                device_map=device_map,
                use_auth_token=True,
            )

        if config['trainable_weights'] == 'classifier':
            print("Trainable weights: classifier")
            for name, param in model.named_parameters():
                # print(name, param.shape)
                if not ('classifier' in name or 'score' in name):
                    param.requires_grad = False
            trainable_parameters(model)
        
        elif config['trainable_weights'] == 'all':
            print("Trainable weights: all")
            trainable_parameters(model)
        
        elif config['trainable_weights'] == 'lora':
            print("Trainable weights: LoRA")
            if load_in_8bit:
                # LoRA + 8-bit training
                model = prepare_model_for_int8_training(
                    model, 
                    use_gradient_checkpointing=config["use_gradient_checkpointing"]
                )

            peft_config = LoraConfig(
                task_type="SEQ_CLS", 
                inference_mode=False, 
                r=8,
                lora_alpha=16, 
                lora_dropout=0.1,
                target_modules=config["target_modules"],
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

    elif task == 'const-gen':
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_fpath,
            use_auth_token=True,
            trust_remote_code=True,
        )
        if 'Baichuan' in model_fpath:
            print(f"pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")
            print(f"eos_token: {tokenizer.eos_token}, eos_token_id: {tokenizer.eos_token_id}")
            tokenizer.model_max_length = config["model_max_length"]
            if config["load_in_8bit"]:
                load_in_8bit = True
                device_map = 'auto'
            if config["load_in_fp16"]:
                torch_dtype = torch.float16
            if config["deepspeed"]:
                device_map = None  

        if tokenizer_only:
            return tokenizer
        
        # Model
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_fpath,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch_dtype,
            device_map=device_map,
            use_auth_token=True,
            trust_remote_code=True,
        )

        if config['trainable_weights'] == 'lora':
            print("Trainable weights: LoRA")
            if load_in_8bit:
                # LoRA + 8-bit training
                model = prepare_model_for_int8_training(
                    model, 
                    use_gradient_checkpointing=config["use_gradient_checkpointing"]
                )

            peft_config = LoraConfig(
                task_type="CAUSAL_LM", 
                inference_mode=False, 
                r=8,                        
                lora_alpha=32,              
                lora_dropout=0.1,
                target_modules=config["target_modules"],
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

    for name, param in model.named_parameters():
        print(name, param.dtype, param.shape, param.requires_grad)
    
    return model, tokenizer