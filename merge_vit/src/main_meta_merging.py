import os
os.environ["PYTORCH_SDP_BACKEND"] = "math"
import time
import numpy as np
import open_clip
import types
# form open_clip.model import ResidualAttentionBlock

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
import time
import tqdm
import sys
# sys.path.append('/remote-home/yepeng2')
from task_vectors import TaskVector
from eval import eval_single_dataset
from args import parse_arguments

from datasetsss.common import get_dataloader, maybe_dictionarize, get_dataloader_shuffle
from heads import get_classification_head
from modeling import ImageClassifier

from datasetsss.registry import get_dataset

def create_log_dir(path, filename='log.txt'):
    import logging
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path+'/'+filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def apply_vector(vector, pretrained_checkpoint):#, scaling_coef=1.0):
    """Apply a task vector to a pretrained model."""
    with torch.no_grad():
        pretrained_model = torch.load(pretrained_checkpoint)
        new_state_dict = {}
        pretrained_state_dict = pretrained_model.state_dict()
        for key in pretrained_state_dict:
            if key not in vector:
                print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                continue
            new_state_dict[key] = pretrained_state_dict[key] + vector[key]
    pretrained_model.load_state_dict(new_state_dict, strict=False)
    return pretrained_model


def emr_merge(task_vectors):
    sum_param = {}
    n2p = []
    for m in range(len(task_vectors)):
        n2p_temp = task_vectors[m].vector
        n2p.append(n2p_temp)
        for n in n2p_temp:
            if n not in sum_param:
                sum_param[n] = []
            sum_param[n].append(n2p_temp[n])
    sum_param = {k: torch.stack(v, 0).mean(0) for k, v in sum_param.items()}
    vector_unified = {}
    scales = torch.zeros(len(task_vectors))
    masks = {}
    for n in sum_param:
        masks[n] = []
        flag = (sum_param[n]>0) * 2 - 1
        param_max = torch.zeros_like(n2p[0][n])
        for m in range(len(task_vectors)):
            param = task_vectors[m].vector[n]
            mask = (param * flag) > 0
            masks[n].append(mask)
            param_abs = torch.abs(mask*param)
            param_max = torch.where(param_abs>param_max, param_abs, param_max)
            scales[m] += torch.mean(torch.abs(param))
        vector_unified[n] =  param_max * flag
    new_scales = torch.zeros(len(task_vectors))
    for m in range(len(task_vectors)):
        for n in vector_unified:
            p = vector_unified[n] * masks[n][m]
            new_scales[m] += torch.mean(torch.abs(p))
    rescalers = scales / new_scales

    return vector_unified, masks, rescalers

class AdaMerging(torch.nn.Module):
    def __init__(self, paramslist, model, names, models, adapter_use=False, paramslist_adapters=None, names_adapters=None):
        super(AdaMerging, self).__init__()
        self.paramslist = paramslist
        self.names = names
        self.model = model
        self.train_preprocess = model.train_preprocess
        self.val_preprocess = model.val_preprocess
        if adapter_use == True:
            self.paramslist_adapters = paramslist_adapters
            self.names_adapters = names_adapters
        self.pretrain_lambdas = torch.nn.Parameter(torch.ones(1, 1))
        # prior = -1.872
        prior = -2.014
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
        # task_lambdas = task_lambdas * 2
        lambdass = torch.cat((self.pretrain_lambdas, task_lambdas), 1)
        return lambdass

    def collect_trainable_params(self):
        return [self.lambdas_raw]


    def forward(self, input, dataset_name_ind, fast_weights=None):
        alph = self.lambdas()
        params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[0]))) for j, p in enumerate(zip(*self.paramslist)))
        params = tuple(p for p in params)


        # self.model.score = self.models[dataset_name_ind].module.score
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
            (input,)
        )
        # out = classification_head(feature)

        return out


class Adapter(nn.Module):
    def __init__(self, hidden_size, bottleneck_size=64):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck_size)
        self.act = nn.ReLU()
        self.up = nn.Linear(bottleneck_size, hidden_size)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.down.weight, std=1e-3)
        # nn.init.zeros_(self.down.weights)
        nn.init.zeros_(self.down.bias)
        
        nn.init.normal_(self.up.weight, std=1e-3)
        # nn.init.zeros_(self.down.weights)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x - self.up(self.act(self.down(x)))
    
def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)
    
def new_forward(self, x, attn_mask=None):
        x = self.old_forward(x, attn_mask)
        return self.adapter(x)

def add_adapters_to_vit(vit_model, bottleneck_size=64):
    for i, block in enumerate(vit_model.transformer.resblocks):
        hidden_size = block.mlp[0].in_features  
        block.adapter = Adapter(hidden_size, bottleneck_size)

        block.old_forward = block.forward
        block.forward = types.MethodType(new_forward, block)

exam_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD'] # SUN397 | Cars | RESISC45 | EuroSAT | SVHN | GTSRB | MNIST | DTD
model = 'ViT-B-32'
args = parse_arguments()
args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
args.home = '/HDDDATA/data_cj/merge_vit/src' # replace with your path of checkpoints and datasets
args.data_location = args.home + '/data'
args.model = model
args.save = args.home + '/checkpoints/' + model
args.logs_path = '../logs/' + model
# args.batch_size = 512
args.dataset_names = exam_datasets
pretrained_checkpoint = args.home + '/checkpoints/'+model+'/zeroshot.pt'

### set random seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
# random.seed(args.seed)

ts = time.time()
readable = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(ts))
# log = create_log_dir(args.logs_path, f'log_{readable}_{args.inner_steps}_{args.metalr}_{args.adalr}.txt')
log = create_log_dir(args.logs_path, f'hyperexp_{args.inner_steps}_{args.metalr}_{args.adalr}.txt')

task_vectors = [
    TaskVector(pretrained_checkpoint, args.home + '/checkpoints/'+model+'/'+dataset_name+'/finetuned.pt') for dataset_name in exam_datasets
]

pretrained_model = torch.load(pretrained_checkpoint)
pretrained_model_dic = {param_name: param_value for param_name, param_value in pretrained_model.named_parameters()}

models = []
for dataset_name in args.dataset_names:
    finetuned_model = torch.load(args.home + '/checkpoints/'+model+'/'+dataset_name+'/finetuned.pt')
    finetuned_model.to(args.device)
    finetuned_model.eval()
    models.append(finetuned_model)

paramslist = []
paramslist += [tuple(v.detach().clone().requires_grad_().to(args.device) for _, v in pretrained_model_dic.items())] # pretrain
paramslist += [tuple(v.detach().clone().requires_grad_().to(args.device) for _, v in tv.vector.items())  for i, tv in enumerate(task_vectors)] # task vectors
# torch.cuda.empty_cache()


add_adapters_to_vit(pretrained_model.model.visual, bottleneck_size=64)
model = pretrained_model
# print(type(model))
# print(type(pretrained_model))
# print(type(model.train_preprocess))
# # for name,param in model.named_parameters():
# #     print(name, param.shape)
# sys.exit(0)
model.to(args.device)
model.eval()
# names_wo_adapters = list(pretrained_model_dic.keys())
model_dic = {param_name: param_value for param_name, param_value in model.named_parameters()}
names_adapters = [name for name in model_dic.keys() if ".adapter." in name]
names_others = [name for name in model_dic.keys() if ".adapter." not in name]

paramslist_adapters = []
paramslist_adapters += [tuple(v.detach().clone().requires_grad_().to(args.device) for k, v in model_dic.items() if k in names_adapters)  for i, tv in enumerate(task_vectors)] # task vectors

# models_ddp = []
# for ftmodel in models:
#     ftmodel = ftmodel.to(args.device)
#     ftmodel.eval()
#     ftmodel = torch.nn.parallel.DistributedDataParallel(
#         ftmodel,
#         device_ids=[local_rank],
#         output_device=local_rank,
#         # find_unused_parameters=True
#     )
#     models_ddp.append(ftmodel)
adamerging_mtl_model = AdaMerging(paramslist, model, names_others, models, adapter_use=True, paramslist_adapters=paramslist_adapters, names_adapters=names_adapters)
# adamerging_mtl_model.paramslist_adapters = torch.load("/home/cj/EMR_Merging/merge_vit/src/cache/2025_09_17_02_01_41_1_0.01_0.1_3000/adapters.pt", map_location=args.device)
# adamerging_mtl_model.lambdas_raw = torch.load("/home/cj/EMR_Merging/merge_vit/src/cache/2025_09_17_02_01_41_1_0.01_0.1_3000/lambdas_raw.pt")
# adamerging_mtl_model.lambdas_raw = torch.load("./cache/bf9.9/lambdas_raw.pt")
adamerging_mtl_model = adamerging_mtl_model.to(args.device)
run_start_time = time.time()


# training lambdas
optimizer = torch.optim.Adam(adamerging_mtl_model.collect_trainable_params(), lr=args.metalr, betas=(0.9, 0.999), weight_decay=0.)
loss_fn = torch.nn.MSELoss()
# loss_fn = torch.nn.L1Loss()


loaders_train = []
loaders_test = []
for idx, dataset_name in enumerate(args.dataset_names):
    dataset = get_dataset(dataset_name, pretrained_model.val_preprocess, location=args.data_location, batch_size=args.meta_batch_size)
    dataloader = get_dataloader_shuffle(dataset)
    loaders_train.append(dataloader)

    dataset = get_dataset(
        dataset_name,
        pretrained_model.val_preprocess,
        location=args.data_location,
        batch_size=args.meta_batch_size_test
    )
    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=None)
    loaders_test.append(dataloader)


iters_train = [iter(loader) for loader in loaders_train]
iters_test = [iter(loader) for loader in loaders_test]
num_steps_inner = args.inner_steps
num_steps_meta = args.epochs
for meta_step in range(num_steps_meta):

    losses = 0.0
    fast_weights_list = []
    fast_weights = [tuple(p.detach().clone().requires_grad_() for p in task_tuple) for task_tuple in adamerging_mtl_model.paramslist_adapters]
    fast_weights_list.append(fast_weights)
    grads_list = []
    for step in range(num_steps_inner):
        grads_list_task = []
        fast_weights = []
        for idx, it in enumerate(iters_train):
            try:
                data = next(it)
            except StopIteration:
                iters_train[idx] = iter(loaders_train[idx])
                data = next(iters_train[idx])
            data = maybe_dictionarize(data)
            x = data['images'].to(args.device)
            y = data['labels'].to(args.device)

            outputs = adamerging_mtl_model(x, dataset_name_ind=idx, fast_weights=fast_weights_list[step])
            with torch.no_grad():
                outputs_ft = models[idx](x)
            loss = loss_fn(outputs, outputs_ft.detach())
            # grads = torch.autograd.grad(loss, fast_weights[idx])
            grads = torch.autograd.grad(loss, fast_weights_list[step][idx], create_graph=True)
            fast_weights.append(tuple(p - args.adalr * g for p, g in zip(fast_weights_list[step][idx], grads)))
            grads_list_task.append(grads)
            # l2_norm = torch.norm(fast_weights[idx][0], p=2)
            # grads1 = torch.autograd.grad(l2_norm, adamerging_mtl_model.lambdas_raw)
            # print(grads)
            # sys.exit(0)
        fast_weights_list.append(fast_weights)
        grads_list.append(grads_list_task)

    for idx, it in enumerate(iters_test):
        try:
            data = next(it)
        except StopIteration:
            iters_test[idx] = iter(loaders_test[idx])
            data = next(iters_test[idx])
        data = maybe_dictionarize(data)
        x = data['images'].to(args.device)
        y = data['labels'].to(args.device)
        outputs = adamerging_mtl_model(x, dataset_name_ind=idx, fast_weights=fast_weights_list[-1])
        with torch.no_grad():
            outputs_ft = models[idx](x)
        loss_meta = loss_fn(outputs, outputs_ft.detach())
        losses += loss_meta
        # print(loss_meta)
        
    optimizer.zero_grad()
    losses.backward()
    # print(adamerging_mtl_model.lambdas_raw.grad)
    # sys.exit()
    optimizer.step()

    print(f"metastep {meta_step} finished, loss: {losses.item()}, lambdas: {adamerging_mtl_model.lambdas()}")
    log.info(f"metastep {meta_step} finished, loss: {losses.item()}, lambdas: {adamerging_mtl_model.lambdas()}")
    
folder = f"./cache/{readable}_{args.inner_steps}_{args.metalr}_{args.adalr}"
os.makedirs(folder, exist_ok=True)
torch.save(adamerging_mtl_model.lambdas_raw, f"./cache/{readable}_{args.inner_steps}_{args.metalr}_{args.adalr}/lambdas_raw.pt")
torch.save(adamerging_mtl_model.paramslist_adapters, f"./cache/{readable}_{args.inner_steps}_{args.metalr}_{args.adalr}/adapters.pt")



# train adapters and evaluate
accs = []
loss_fn = torch.nn.MSELoss()
for idx, dataset_name in enumerate(exam_datasets):
    optimizer = torch.optim.Adam(adamerging_mtl_model.paramslist_adapters[idx], lr=1e-3, betas=(0.9, 0.999), weight_decay=0.)
    log.info(f"Start to train adapter for {args.dataset_names[idx]}")

    losses = 0.
    dataset = get_dataset(dataset_name, pretrained_model.val_preprocess, location=args.data_location, batch_size=args.batch_size)
    dataloader = get_dataloader_shuffle(dataset)

    for _, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data['images'].to(args.device)
            y = data['labels'].to(args.device)

            outputs = adamerging_mtl_model(x, dataset_name_ind=idx)
            with torch.no_grad():
                outputs_ft = models[idx](x)
            loss = loss_fn(outputs, outputs_ft.detach())
            # loss = softmax_entropy(outputs).mean(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
            log.info(f"loss {loss.item()}")
            # sys.exit()

    log.info(f"losses {losses}")

    classification_head = get_classification_head(args, dataset_name)
    test_model = ImageClassifier(adamerging_mtl_model, classification_head)
    test_model.to(args.device)
    test_model.eval()

    dataset = get_dataset(
        dataset_name,
        test_model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size
    )
    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=None)

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for _, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data['images'].to(args.device)
            y = data['labels'].to(args.device)

            logits = test_model(x, idx)

            pred = logits.argmax(dim=1, keepdim=True).to(args.device)

            correct += pred.eq(y.view_as(pred)).sum().item()
            
            n += y.size(0)

        top1 = correct / n

    metrics = {'top1': top1}
    print(f'Done evaluating on {dataset_name}. Accuracy: {100*top1:.2f}%')
    log.info(str(dataset_name) + ':' + str(metrics.get('top1')*100)+'%')
    accs.append(metrics.get('top1')*100)

finish_time = time.time()
log.info('Avg ACC:' + str(np.mean(accs)) + '%')
log.info(f"Final acc list: {accs}")
log.info(f"Total time cost: {finish_time - run_start_time} seconds.")
log.info(f"********** Run ends. **********\n\n")
