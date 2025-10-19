import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"


import datasets.arrow_dataset
from tqdm import tqdm
import numpy as np
from datasets import load_dataset, load_from_disk
import copy
import sys
import random

import transformers
from utils.utils import set_random_seed
from model_merging_methods.merging_methods import MergingMethod
from torch.func import functional_call
import torch.nn.functional as F

import sys
import json
import argparse
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import time
import logging
from functools import partial
from torchmetrics import Accuracy, MeanMetric
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, GPT2Tokenizer#, GPT2ForSequenceClassification
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from utils.glue_data_loader import GLUEDataLoader
from utils.metrics import compute_metrics
from utils.customized_trainers import CustomizedTrainer
from model_merging_methods.mask_weights_utils import mask_model_weights
from model_merging_methods.task_vector import TaskVector
from utils.load_config import cache_dir

from gpt_adapters import GPT2ClassifierWithAdapter

from transformers import (
    GPT2ForSequenceClassification,
    GPT2Model,
    GPT2Tokenizer,
    default_data_collator,
    AutoConfig
)



def mrpc_tokenize_function(examples, tokenizer):
    inputs = tokenizer(
        examples['sentence1'],#, 'sentence2'],
        examples["sentence2"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs


def mnli_tokenize_function(examples, tokenizer):
    inputs = tokenizer(
        examples["premise"],
        examples["hypothesis"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs


def cola_tokenize_function(examples, tokenizer):
    inputs = tokenizer(
        examples["sentence"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs


def qnli_tokenize_function(examples, tokenizer):
    inputs = tokenizer(
        examples["question"],
        examples["sentence"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs


def qqp_tokenize_function(examples, tokenizer):
    inputs = tokenizer(
        examples["question1"],
        examples["question2"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs

class TokenizedGLUE:
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def load_dataset(
        self, name
    ):
        glue_dataset_loaders = {
            "mrpc": self.load_mrpc_dataset,
            "mnli": self.load_mnli_dataset,
            "cola": self.load_cola_dataset,
            "sst2": self.load_sst2_dataset,
            "qnli": self.load_qnli_dataset,
            "qqp": self.load_qqp_dataset,
            "rte": self.load_rte_dataset,
            # "wnli": load_wnli_dataset,
        }
        return glue_dataset_loaders[name]()


    def load_mrpc_dataset(self):
        # dataset = load_from_disk('/remote-home/yepeng2/cache/GLUE_DOWNLOAD/mrpc')
        dataset = load_dataset("glue", "mrpc", cache_dir=cache_dir)
        dataset = dataset.map(
            partial(mrpc_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=['sentence1', 'sentence2'],
        )
        return dataset


    def load_rte_dataset(self):
        # dataset = load_from_disk('/remote-home/yepeng2/cache/GLUE_DOWNLOAD/rte')
        dataset = load_dataset("glue", "rte", cache_dir=cache_dir)
        dataset = dataset.map(
            # RTE has the same format as MRPC
            partial(mrpc_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["sentence1", "sentence2"],
        )
        return dataset


    def load_wnli_dataset(self):
        dataset = load_dataset("glue", "wnli", cache_dir=cache_dir)
        # dataset = load_from_disk('/remote-home/yepeng2/cache/GLUE_DOWNLOAD/wnli')
        dataset = dataset.map(
            partial(mrpc_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["sentence1", "sentence2"],
        )
        return dataset


    def load_qqp_dataset(self):
        dataset = load_dataset("glue", "qqp", cache_dir=cache_dir)
        # dataset = load_from_disk('/remote-home/yepeng2/cache/GLUE_DOWNLOAD/qqp')
        dataset = dataset.map(
            partial(qqp_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=['question1', 'question2'],
        )
        return dataset


    def load_mnli_dataset(self):
        dataset = load_dataset("glue", "mnli",  cache_dir=cache_dir)
        # dataset = load_from_disk('/remote-home/yepeng2/cache/GLUE_DOWNLOAD/mnli')
        dataset = dataset.map(
            partial(mnli_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["premise", "hypothesis"],
        )
        return dataset


    def load_cola_dataset(self):
        dataset = load_dataset("glue", "cola", cache_dir=cache_dir)
        # dataset = load_from_disk('/remote-home/yepeng2/cache/GLUE_DOWNLOAD/cola')
        dataset = dataset.map(
            partial(cola_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["sentence"],
        )
        return dataset


    def load_sst2_dataset(self):
        dataset = load_dataset("glue", "sst2", cache_dir=cache_dir)
        # dataset = load_from_disk('/remote-home/yepeng2/cache/GLUE_DOWNLOAD/sst2')
        print(dataset.column_names)
        dataset = dataset.map(
            partial(cola_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["sentence"],
        )
        return dataset


    def load_qnli_dataset(self):
        # dataset = load_from_disk('/remote-home/yepeng2/cache/GLUE_DOWNLOAD/qnli')
        dataset = load_dataset("glue", "qnli", cache_dir=cache_dir)
        dataset = dataset.map(
            partial(qnli_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["question", "sentence"],
        )
        return dataset




class AdaMerging(torch.nn.Module):
    def __init__(self, paramslist, model, names, models, adapter_use=False, paramslist_adapters=None, names_adapters=None):
        super(AdaMerging, self).__init__()
        self.paramslist = paramslist
        self.names = names
        self.model = model
        if adapter_use == True:
            self.paramslist_adapters = paramslist_adapters
            self.names_adapters = names_adapters
        self.pretrain_lambdas = torch.nn.Parameter(torch.ones(1, 1))
        prior = -1.872
        rlambdas = torch.ones(1, len(paramslist)-1) * prior  # (1 * tasks)
        # rlambdas = torch.tensor([[-1.4789, -1.3519, -1.9089, -1.6717, -1.7518, -1.6086, -1.5178]])
        # rlambdas = torch.tensor([[-1.8273, -1.3574, -0.8384, -1.0836, -1.6311, -1.8202, -1.3461]])
        self.lambdas_raw = torch.nn.Parameter(rlambdas)
        self.adapter_use = adapter_use

        self.models = models

    def lambdas(self):
        # task_lambdas = torch.clamp(self.lambdas_raw, min=0.0, max=1.0)
        # task_lambdas = F.softmax(self.lambdas_raw, dim=1)
        task_lambdas = F.softplus(self.lambdas_raw)
        lambdass = torch.cat((self.pretrain_lambdas, task_lambdas), 1)
        return lambdass

    def collect_trainable_params(self):
        return [self.lambdas_raw]


    def forward(self, input_ids, attention_mask, dataset_name_ind, fast_weights=None):
        alph = self.lambdas()
        # print(alph)
        # print(next(self.model.parameters()).device) )
        params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[0]))) for j, p in enumerate(zip(*self.paramslist)))

        params = tuple(p for p in params)
        # load_weights(self.model, self.names, params)

        # layer_name = 'classifier_{}'.format(dataset_name)
        # if self.adapter_use == True:
        #     self.model.gpt2.config = AutoConfig.from_pretrained(
        # pretrained_model_name_or_path=args.ckpt_path+f"/gpt2_{dataset_names[dataset_name_ind]}") # load the config
        # else:
        self.model.config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=args.ckpt_path+f"/gpt2_{dataset_names[dataset_name_ind]}") # load the config


        self.model.score = self.models[dataset_name_ind].module.score
        # out = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        params_dic = {name: new_weight for name, new_weight in zip(self.names, params)}
        if fast_weights is not None:
            for name_adapter, params_adapter in zip(self.names_adapters, fast_weights[dataset_name_ind]):
                params_dic[name_adapter] = params_adapter
        else:
            for name_adapter, params_adapter in zip(self.names_adapters, self.paramslist_adapters[dataset_name_ind]):
                params_dic[name_adapter] = params_adapter
        
        out = functional_call(
            self.model,
            params_dic,
            (input_ids,),  # args
            {
                "attention_mask": attention_mask,
                "output_hidden_states": True
            }  # kwargs
        )
        # out = classification_head(feature)

        return out


def setup_logger(args):
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("datasets").setLevel(logging.ERROR)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    save_merge_log_path = f"./save_merge_logs/{args.merging_method_name}/{args.language_model_name}"
    os.makedirs(save_merge_log_path, exist_ok=True)

    ts = time.time()
    readable = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
    fh = logging.FileHandler(f"{save_merge_log_path}/{readable}.log")
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    run_start_time = time.time()
    if local_rank == 0:
        logger.info(f"********** Run starts. **********")
        logger.info(f"configuration is {args}")
    return logger, run_start_time

def train_adapters(model, train_dataloader, optimizer, scheduler, device, epochs=3):
    best_f1 = 0.0
    
    for epoch in range(epochs):
        print(f"======== Epoch {epoch+1} / {epochs} ========")
        model.train()
        total_loss = 0
        
        # 训练循环
        progress_bar = tqdm(train_dataloader, desc="Training", position=0, leave=True)
        for batch in progress_bar:
            # 将数据移到GPU
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # 参数更新
            optimizer.step()
            scheduler.step()
            
            # 更新进度条
            progress_bar.set_postfix({"loss": loss.item()})
        
        # 计算平均训练损失
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # 保存最佳模型
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save(model.state_dict(), "best_weibo_sentiment_model.pth")
            print("Saved best model!")

def evaluate(args, model, tokenizer, logger):
    for idx, dataset_name in enumerate(args.dataset_names):
        glue = TokenizedGLUE(tokenizer)
        ds = glue.load_dataset(dataset_name)

        try:
            ds_val = ds['validation']
        except:
            ds_val = ds['validation_mismatched']
        with torch.no_grad():
            accuracy = Accuracy("multiclass", num_classes=num_labels[dataset_name]).to(args.device)  # len(ds['validation'].unique('label')))#, num_classes=num_labels[dataset_name])
            sampler = DistributedSampler(ds_val)
            loader = DataLoader(
                ds_val,
                sampler=sampler,
                collate_fn=default_data_collator,
                # batch_size=args.batch_size,
                batch_size=8,
                num_workers=1,
            )
            for batch in (
                    pbar := tqdm(
                        loader, desc="Evaluating", leave=False, dynamic_ncols=True
                    )
            ):
                input_ids = batch["input_ids"].to(args.device)
                attention_mask = batch["attention_mask"].to(args.device)
                labels = batch["labels"].to(args.device)
                # print(input_ids.shape)
                # print(attention_mask.shape)
                # print(labels.shape)
                # outputs = model(input_ids, attention_mask=attention_mask)
                # print(outputs.logits)  # 分类 logits
                # sys.exit()

                model.score = models_ddp[idx].module.score
                outputs = model(input_ids, attention_mask=attention_mask)
                # outputs = adamerging_mtl_model(input_ids, attention_mask=attention_mask, dataset_name_ind=idx)
                logits = outputs.logits
                acc = accuracy(logits.detach(), labels.detach())
                pbar.set_postfix({"accuracy": acc.item()})

            acc = accuracy.compute().item()
            if local_rank == 0:
                logger.info(f"acc on {dataset_name}: {acc}")


num_labels = {
        'cola': 2,
        'sst2': 2,
        'mrpc': 2,
        'stsb': 5,
        'qqp': 2,
        'mnli': 3,
        'qnli': 2,
        'rte': 2
    }
dataset_names = ["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte"]
dataset_samples = [8551, 8000, 3668, 8000, 8000, 8000, 2490]  # number of training samples for each dataset

if __name__ == "__main__":
    # print(cache_dir)
    # sys.exit()
    parser = argparse.ArgumentParser("Interface for inference PLMs on glue")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--language_model_name", type=str, default="gpt2", help="name of the language model", choices=["gpt2"])
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--merging_method_name", type=str, default="emr_merging",
                        help="name of the method to merge models",
                        choices=["emr_merging"])
    parser.add_argument("--gpu", type=int, default=2, help="number of gpu to use")
    parser.add_argument('--ckpt_path', type=str, default='/HDDDATA/data/ckpts/gpt2',help="ckpt path")

    parser.add_argument('--epochs', type=int, default=2, help="number of epochs")

    try:
        args = parser.parse_args()
        args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    except:
        parser.print_help()
        sys.exit()
    args.dataset_names = dataset_names
    args.dataset_samples = dataset_samples

    #### set random seed
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # np.random.seed(args.seed)
    # random.seed(args.seed)

    local_rank = int(os.environ["LOCAL_RANK"])   # torchrun 会传入
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    args.device = torch.device("cuda", local_rank)
    print(f"local_rank: {local_rank}, device: {args.device}")


    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=args.ckpt_path+'/gpt2')
    tokenizer.model_max_length = 512
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        elif tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

    # glue_data_loader = GLUEDataLoader(tokenizer=tokenizer)
    pretrained_model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.ckpt_path+'/gpt2').to('cpu')
    # pretrained_model = AutoAdapterModel.from_pretrained(pretrained_model_name_or_path=args.ckpt_path+'/gpt2').to('cpu')
    pretrained_model_dic = {param_name: param_value for param_name, param_value in pretrained_model.named_parameters()}

    models = []
    loaders = []
    for dataset_name in dataset_names:
        args.dataset_name = dataset_name
        load_model_path = args.ckpt_path+f"/gpt2_{dataset_name}"
        finetuned_model = GPT2ForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=load_model_path).to('cpu')
        finetuned_model.config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=load_model_path) # load the config
        models.append(finetuned_model)

    task_vectors = [TaskVector(pretrained_model=pretrained_model, finetuned_model=ft_model, exclude_param_names_regex=[".*score.*"]) for
                                        ft_model in models]

    paramslist = []
    paramslist += [tuple(v.detach().requires_grad_().to(args.device) for _, v in pretrained_model_dic.items())] # pretrain
    paramslist += [tuple(v.detach().requires_grad_().to(args.device) for _, v in tv.task_vector_param_dict.items())  for i, tv in enumerate(task_vectors)] # task vectors
    torch.cuda.empty_cache()


    # model = pretrained_model
    model = GPT2ClassifierWithAdapter(pretrained_model_name=args.ckpt_path+'/gpt2')
    model = model.gpt2
    model.to(args.device)
    model.eval()
    # names_wo_adapters = list(pretrained_model_dic.keys())
    model_dic = {param_name: param_value for param_name, param_value in model.named_parameters()}
    names_adapters = [name for name in model_dic.keys() if ".adapter." in name]
    names_others = [name for name in model_dic.keys() if ".adapter." not in name]

    paramslist_adapters = []
    paramslist_adapters += [tuple(v.detach().requires_grad_().to(args.device) for k, v in model_dic.items() if k in names_adapters)  for i, tv in enumerate(task_vectors)] # task vectors

    models_ddp = []
    for ftmodel in models:
        ftmodel = ftmodel.to(args.device)
        ftmodel.eval()
        ftmodel = torch.nn.parallel.DistributedDataParallel(
            ftmodel,
            device_ids=[local_rank],
            output_device=local_rank,
            # find_unused_parameters=True
        )
        models_ddp.append(ftmodel)
    adamerging_mtl_model = AdaMerging(paramslist, model, names_others, models_ddp, adapter_use=True, paramslist_adapters=paramslist_adapters, names_adapters=names_adapters)
    # adamerging_mtl_model.paramslist_adapters = torch.load("./cache/adapters_trained.pt", map_location=args.device)
    adamerging_mtl_model.lambdas_raw = torch.load("./cache/lambdas_raw.pt")
    adamerging_mtl_model = adamerging_mtl_model.to(args.device)

    merged_params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, adamerging_mtl_model.lambdas()[0]))) for j, p in enumerate(zip(*adamerging_mtl_model.paramslist)))
    model.load_state_dict({name: param for name, param in zip(adamerging_mtl_model.names, merged_params)}, strict=False)

    # adamerging_mtl_model = torch.nn.parallel.DistributedDataParallel(
    #     adamerging_mtl_model,
    #     device_ids=[local_rank],
    #     output_device=local_rank,
    #     find_unused_parameters=True
    # )
    # print(adamerging_mtl_model.module.paramslist_adapters[0][0])
    # sys.exit(0)



    # set up logger
    logger, run_start_time = setup_logger(args)
    
    
    evaluate(args, model, tokenizer, logger)
    sys.exit(0)

    '''
    # training lambdas
    optimizer = torch.optim.Adam(adamerging_mtl_model.module.collect_trainable_params(), lr=1e-2, betas=(0.9, 0.999), weight_decay=0.)
    loss_fn = torch.nn.MSELoss()
    # loss_fn = torch.nn.L1Loss()


    loaders_train = []
    loaders_test = []
    for idx, dataset_name in enumerate(args.dataset_names):
        glue = TokenizedGLUE(tokenizer)
        ds = glue.load_dataset(dataset_name)
        ds_train = ds['train']
        try:
            ds_test = ds['test']
        except:
            ds_test = ds['test_mismatched']
        indices = random.sample(range(len(ds_train)), 1000)  # 随机索引
        sampled_ds_train = ds_train.select(indices)
        sampler_train = DistributedSampler(sampled_ds_train)
        loader_train = DataLoader(
            sampled_ds_train,
            sampler=sampler_train,
            collate_fn=default_data_collator,
            batch_size=args.batch_size,
            num_workers=1,
        )
        loaders_train.append(loader_train)
        indices = random.sample(range(len(ds_test)), 1000)  # 随机索引
        sampled_ds_test = ds_test.select(indices)
        sampler_test = DistributedSampler(sampled_ds_test)
        loader_test = DataLoader(
            sampled_ds_test,
            sampler=sampler_test,
            collate_fn=default_data_collator,
            batch_size=args.batch_size,
            num_workers=1,
        )
        loaders_test.append(loader_test)


    iters_train = [iter(loader) for loader in loaders_train]
    iters_test = [iter(loader) for loader in loaders_test]
    num_steps_inner = 5
    num_steps_meta = args.epochs
    # print(adamerging_mtl_model.module.paramslist_adapters[0][0])
    for meta_step in range(num_steps_meta):

        losses = 0.0
        # fast_weights = adamerging_mtl_model.module.paramslist_adapters
        fast_weights = [tuple(p.detach().clone().requires_grad_() for p in task_tuple) for task_tuple in adamerging_mtl_model.module.paramslist_adapters]
        for step in range(num_steps_inner):
            for idx, it in enumerate(iters_train):
                try:
                    batch = next(it)
                except StopIteration:
                    iters_train[idx] = iter(loaders_train[idx])
                    batch = next(iters_train[idx])

                # for idx, dataset_name in enumerate(args.dataset_names):
                    # for batch in (
                    #         pbar := tqdm(
                    #             loaders[idx], desc=f"alignment_training_{dataset_name}", leave=False, dynamic_ncols=True
                    #         )
                    # ):
                input_ids = batch["input_ids"].to(args.device)
                attention_mask = batch["attention_mask"].to(args.device)
                # labels = batch["labels"].to(args.device)

                # outputs = merged_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                outputs = adamerging_mtl_model(input_ids, attention_mask=attention_mask, dataset_name_ind=idx, fast_weights=fast_weights)
                # print(outputs.hidden_states.shape)
                with torch.no_grad():
                    outputs_ft = models_ddp[idx](input_ids, attention_mask=attention_mask, output_hidden_states=True)
                loss = loss_fn(outputs.hidden_states[-1], outputs_ft.hidden_states[-1].detach())
                grads = torch.autograd.grad(loss, fast_weights[idx])
                # if local_rank == 0:
                #     print(grads[0])
                #     print(grads[-1])
                # sys.exit()
                fast_weights[idx] = tuple(p - 0.01 * g for p, g in zip(fast_weights[idx], grads))

        for idx, it in enumerate(iters_test):
            try:
                batch = next(it)
            except StopIteration:
                iters_test[idx] = iter(loaders_test[idx])
                batch = next(iters_test[idx])
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            outputs = adamerging_mtl_model(input_ids, attention_mask=attention_mask, dataset_name_ind=idx, fast_weights=fast_weights)
            with torch.no_grad():
                outputs_ft = models_ddp[idx](input_ids, attention_mask=attention_mask, output_hidden_states=True)
            loss_meta = loss_fn(outputs.hidden_states[-1], outputs_ft.hidden_states[-1].detach())
            optimizer.zero_grad()
            loss_meta.backward()
            optimizer.step()
            losses += loss_meta.item() / len(args.dataset_names)

        if local_rank == 0:
            print(f"metastep {meta_step} finished, loss: {losses}")
            logger.info(f"metastep {meta_step} finished, loss: {losses}")
            '''



    # train adapter
    # print(len(adamerging_mtl_model.module.collect_trainable_params(train_adapter=True, dataset_name_ind=0)))
    # sys.exit()
    # optimizers = [
    #     torch.optim.Adam(adamerging_mtl_model.module.collect_trainable_params(train_adapter=True, dataset_name_ind=idx), lr=1e-2, betas=(0.9, 0.999), weight_decay=0.)
    #         for idx in range(len(args.dataset_names))
    #     ]
    loss_fn = torch.nn.MSELoss()
    
    loaders_train = []
    for idx, (dataset_name, dataset_sample) in enumerate(zip(args.dataset_names, args.dataset_samples)):
        glue = TokenizedGLUE(tokenizer)
        ds = glue.load_dataset(dataset_name)
        ds_train = ds['train']
        indices = random.sample(range(len(ds_train)), dataset_sample)  # 随机索引
        sampled_ds_train = ds_train.select(indices)
        sampler_train = DistributedSampler(sampled_ds_train)
        loader_train = DataLoader(
            sampled_ds_train,
            sampler=sampler_train,
            collate_fn=default_data_collator,
            batch_size=args.batch_size,
            num_workers=1,
        )
        loaders_train.append(loader_train)

    num_steps_train_adapters = 1
    for idx, loader in enumerate(loaders_train):
        if local_rank == 0:
            logger.info(f"Start to train adapter for {args.dataset_names[idx]}")
        # optimizer = torch.optim.Adam(adamerging_mtl_model.module.collect_trainable_params(train_adapter=True, dataset_name_ind=idx), lr=1e-2, betas=(0.9, 0.999), weight_decay=0.)
        # optimizer = torch.optim.Adam(adamerging_mtl_model.module.paramslist_adapters[idx], lr=1e-2, betas=(0.9, 0.999), weight_decay=0.)
        for step in range(num_steps_train_adapters):
            losses = 0.0
            # if local_rank == 0:
            print(adamerging_mtl_model.module.paramslist_adapters[0][0])
            for batch_idx, batch in enumerate(
                    pbar := tqdm(
                        loader, desc="AdapterTraining", leave=False, dynamic_ncols=True
                    )
            ):
                input_ids = batch["input_ids"].to(args.device)
                attention_mask = batch["attention_mask"].to(args.device)
                # labels = batch["labels"].to(args.device)

                # outputs = merged_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                outputs = adamerging_mtl_model(input_ids, attention_mask=attention_mask, dataset_name_ind=idx)
                # print(outputs.hidden_states.shape)
                with torch.no_grad():
                    outputs_ft = models_ddp[idx](input_ids, attention_mask=attention_mask, output_hidden_states=True)
                loss = loss_fn(outputs.hidden_states[-1], outputs_ft.hidden_states[-1].detach())
                # optimizer.zero_grad()
                # loss.backward()
                grads = torch.autograd.grad(loss, adamerging_mtl_model.module.paramslist_adapters[idx])
                print(grads)
                adamerging_mtl_model.module.paramslist_adapters[idx] = tuple(p - 0.01 * g for p, g in zip(adamerging_mtl_model.module.paramslist_adapters[idx], grads))
                # optimizer.step()
                losses += loss.item() / 20
                # if (batch_idx+1) % 20 == 0 and local_rank == 0:
                #     logger.info(f"Adapter training step {step} finished, loss: {losses}")
                #     losses = 0.0
                # if local_rank == 0:
                print(adamerging_mtl_model.module.paramslist_adapters[0][0])
                    # print(adamerging_mtl_model.module.paramslist_adapters[1][1].grad)
                    # print(adamerging_mtl_model.module.lambdas())
                
                sys.exit(0)
    


    # evaluate
    torch.save(adamerging_mtl_model.module.paramslist_adapters, "./cache/adapters_trained.pt")
    # torch.save(adamerging_mtl_model.module.lambdas_raw, "./cache/lambdas_raw.pt")
    evaluate(args, adamerging_mtl_model, tokenizer, logger)



    print('init lambda:')
    print(adamerging_mtl_model.module.lambdas())
    print('collect_trainable_params:')
    print(list(adamerging_mtl_model.module.collect_trainable_params()))
    logger.info(f"Final lambdas: {adamerging_mtl_model.module.lambdas()}")
    logger.info(f"Final lambdas_raw: {adamerging_mtl_model.module.lambdas_raw}")
    sys.exit()

    # performance = get_emr_merge_performance(args, models, loaders, tokenizer, logger)