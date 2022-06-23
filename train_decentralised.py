#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   train.py
@Time    :   2022/06/18 16:55:20
@Author  :   Bo 
'''
import numpy as np 
import os 
import math
import matplotlib.pyplot as plt 
import torch 
import torch.nn as nn 
import data.prepare_data as pd 
import model.mlp as mlp 
import wandb
import pytorch_lightning as pl
from torchvision import transforms
from pytorch_lightning import LightningModule, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import AMPType
import config.params as params 
import pickle 


def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argsort(memory_available)


def global_batch_size_calc(batch_size, num_gpus, parallel_strategy):
    if parallel_strategy == "dp":
        global_batch_size = batch_size
    elif parallel_strategy == "ddp":
        global_batch_size = num_gpus * batch_size if num_gpus > 0 else batch_size
    else:
        global_batch_size = batch_size
    return global_batch_size



def get_lr_schedule(lr_init, epochs, num_global_iter_per_epoch, schedule="cosine"):
    iters = np.arange(num_global_iter_per_epoch * epochs)
    final_lr = 1e-5
    if schedule == "cosine":
        lr_schedule = np.array([
            final_lr + 0.5 * (lr_init - final_lr) *
            (1 + math.cos(math.pi * t / (num_global_iter_per_epoch * epochs)))
            for t in iters
        ])
    else:
        lr_schedule = np.array([lr_init for t in iters])
    return lr_schedule


class DecentralisedLightning(LightningModule):
    def __init__(self, batch_size=128, enc_lr=0.001, strategy="dp", 
                 lr_schedule="cosine", epochs=50, num_gpus=1, device_number=1, 
                 num_devices=1, num_samples=1000):
        super().__init__()
        try:
            if args.gpu_index == 10:
                free_id = get_freer_gpu()
                use_id = free_id[-args.num_gpu:]
                use_id_list = ",".join(["%d" % i for i in use_id])
            else:
                use_id_list = ",".join(["%d" % v for v in [args.gpu_index]])
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = use_id_list
        except:
            print("GPU doesn't exist")

        self.save_hyperparameters()
        self.mlp_model = mlp.MLP(784, 1)
        global_batch_size = global_batch_size_calc(batch_size, num_gpus, strategy)
        self.train_iters_per_epoch = num_samples // global_batch_size
        self.lr_schedule = get_lr_schedule(enc_lr, epochs, self.train_iters_per_epoch, lr_schedule)
        self.device_communicate_name = "device_%02d" % (device_number)
        self.num_devices = num_devices

    def configure_optimizers(self):
        params = self.parameters()
        if self.num_devices < 32:
            weight_decay_param = 5e-4 #5e-4
        else:
            weight_decay_param = 0.8
        optimizer = torch.optim.SGD(params, self.hparams.enc_lr, weight_decay=weight_decay_param)
        for name, v in self.named_parameters():
            if not v.requires_grad:
                print(name, "------does not require gradient")
        return optimizer

    def optimizer_step(self, epoch=None,
                     batch_idx=None,
                     optimizer=None,
                     optimizer_idx=None,
                     optimizer_closure=None,
                     on_tpu=None,
                     using_native_amp=None,
                     using_lbfgs=None):
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = self.lr_schedule[self.trainer.global_step]
            self.log("learning_rate_encoder", self.lr_schedule[self.trainer.global_step], on_step=True, on_epoch=False)

        if self.trainer.amp_backend == AMPType.NATIVE:
            optimizer_closure()
            self.trainer.scaler.step(optimizer)
        elif self.trainer.amp_backend == AMPType.APEX:
            optimizer_closure()
            optimizer.step()
        else:
            optimizer.step(closure=optimizer_closure)
        
    def training_step(self, batch, batch_idx):
        x, y = batch 
        pred = self.mlp_model(x.squeeze(1)).squeeze(1)
        loss = mlp.loss(pred, y)
        self.log("train_loss_%s" % self.device_communicate_name, loss, sync_dist=True)
        tr_pred = (pred.detach().cpu() >= 0.5).to(torch.float32)
        accu = (tr_pred == y.detach().cpu()).sum().div(len(x))
        self.log("train_accuracy_%s" % self.device_communicate_name, accu, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.mlp_model(x.squeeze(1)).squeeze(1)
        loss = mlp.loss(pred, y)
        self.log("validation_loss_%s" % self.device_communicate_name, loss, sync_dist=True)
        y_detach_cpu = y.detach().cpu()
        val_pred = (pred.detach().cpu() >= 0.5).to(torch.float32)
        accu = (val_pred == y_detach_cpu).sum().div(len(x))
        self.log("validation_accuracy_%s" % self.device_communicate_name, accu, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self.mlp_model(x.squeeze(1)).squeeze(1)
        loss = mlp.loss(pred, y)
        self.log("test_loss_%s" % self.device_communicate_name, loss, sync_dist=True)
        y_detach_cpu = y.detach().cpu()
        tt_pred = (pred.detach().cpu() >= 0.5).to(torch.float32)
        accu = (tt_pred == y_detach_cpu).sum().div(len(x))
        self.log("test_accuracy_%s" % self.device_communicate_name, accu, sync_dist=True)
            

def train(args, model_dir):
    create_dir(model_dir)
    project_name = "Local_SGD_Learning"
    wandb_logger = WandbLogger(project=project_name, config=args,
                               save_dir=model_dir+"/")
    transform_use = transforms.ToTensor()
    seed_use = np.random.randint(0, 10000, [args.num_communication])
    y_loc = []
    test_prediction = {}
    test_prediction["communication_rounds"] = args.num_communication
    test_prediction["number_devices"] = args.num_device
    test_prediction["epoch"] = args.epochs
    test_prediction["seed"] = seed_use
    test_prediction["tr_distribution_non_iid"] = []
    for communication_round in range(args.num_communication):    
        if args.num_device >= 32:
            lr_use = get_lr_schedule(args.enc_lr, args.num_communication, 1, "cosine")[communication_round]
        else:
            lr_use = args.enc_lr        
        print(lr_use)
        tr_data, tr_label, tt_data, tt_label = pd.GetData(args.num_device).forward(seed_use[communication_round], 
                                                                                   args.iid)  
        if len(tr_data[0]) <= args.batch_size:
            args.batch_size = len(tr_data[0])
        elif len(tr_data[0]) / args.batch_size < 2:
            args.batch_size = len(tr_data[0]) // 2
        if args.iid is False:
            tr_label_count = [np.unique(v, return_counts=True) for v in tr_label]
            test_prediction["tr_distribution_non_iid"].append(tr_label_count)
        if communication_round == 0:
            test_prediction["batch_size"] = args.batch_size
            test_prediction["num_iter_per_epochs"] = len(tr_data[0]) // args.batch_size  
            test_prediction["prediction_communication_round"] = np.zeros([args.num_communication, len(tt_data)])
            test_prediction["ground_truth"] = tt_label           
        tt_dataset = pd.ImageLoader(tt_data, tt_label, transform_use)    
        val_loader = pd.get_test_dataloader(tt_dataset, len(tt_dataset), args.workers_load_data)
        
        for s_device in range(args.num_device):
            if communication_round == 0:
                pl.seed_everything(1024)
            model_use = DecentralisedLightning(args.batch_size, lr_use, "dp", 
                                               args.lr_decay_schedule, 
                                               args.epochs, args.num_gpu, 
                                               device_number=s_device, 
                                               num_devices=args.num_device, 
                                               num_samples=len(tr_data[s_device]))
            wandb_logger.watch(log_freq=500, log_graph=False, model=model_use)
            if communication_round != 0:
                model_use.load_state_dict(model_previous_step)
            tr_dataset = pd.ImageLoader(tr_data[s_device], tr_label[s_device], transform_use)
            train_loader = pd.get_dataloader(tr_dataset, args.batch_size, args.workers_load_data)    
            print("##################################################################")
            print("Communication round", communication_round)
            print("Device number:", s_device)
            print("The number of the test data", len(val_loader))
            print("The number of training data", np.shape(tr_data[s_device]), np.unique(tr_label[s_device], return_counts=True))
            for name, p in model_use.named_parameters():
                if communication_round == 0:
                    if "bias" in name:
                        print(name, p)
                    else:
                        print(name, torch.mean(p))
                else:
                    if "bias" in name:
                        print(name, "current bias", p, "saved bias", model_previous_step[name])
                    else:
                        print(name, "current weights", torch.mean(p), "saved weight", torch.mean(model_previous_step[name]))                
            log_steps = int(len(train_loader) / args.num_gpu) if args.num_gpu > 0 else len(train_loader)
            trainer = pl.Trainer(max_epochs=args.epochs, progress_bar_refresh_rate=20,
                                gpus=args.num_gpu, logger=wandb_logger, log_every_n_steps=log_steps,
                                strategy=args.strategy, check_val_every_n_epoch=1,
                                deterministic=True,
                                gradient_clip_val=1.0)
            trainer.fit(model_use, train_loader, val_loader)
            if s_device == 0:
                model_avg = {}
                for name, p in model_use.named_parameters():
                    model_avg[name] = torch.zeros_like(p)
                    model_avg[name] += p 
            else:
                for name, p in model_use.named_parameters():
                    model_avg[name] += p                
            # if communication_round == 0:
            #     model_avg = {}
            #     for name, p in model_use.named_parameters():
            #         model_avg[name] = torch.zeros_like(p)
            #         model_avg[name] += p 
            # else:
            #     for name, p in model_use.named_parameters():
            #         model_avg[name] += p  
        for name in model_avg.keys():
            model_avg[name] = model_avg[name] / args.num_device      
        
        model_previous_step = {}
        for name in model_avg.keys():
            model_previous_step[name] = model_avg[name].clone()
        pt_path = model_dir + "/communication_round_%d_iteration_%04d_model_avg.pt" % (communication_round, 
                                                                                       (communication_round + 1) * args.epochs * (len(tr_data[0]) // args.batch_size))
        torch.save({"model_state_dict": model_previous_step, 
                    }, pt_path)
        model_centralised = DecentralisedLightning(args.batch_size, args.enc_lr, "dp", 
                                                  args.lr_decay_schedule, 
                                                  args.epochs, args.num_gpu, 
                                                  device_number=s_device, 
                                                  num_devices=args.num_device, 
                                                  num_samples=len(tr_data[0]))
        model_centralised.load_state_dict(model_previous_step)
        model_centralised.eval()
        model_centralised.to(torch.device("cuda"))
        _tt_pred_centralised = []
        for iter, (x, y) in enumerate(val_loader):
            _pred = model_centralised.mlp_model(x.squeeze(1).to(torch.device("cuda"))).squeeze(1)
            _tt_pred_centralised = (_pred.detach().cpu() > 0.5).to(torch.float32)  
            _accu = (_tt_pred_centralised == y.detach().cpu()).to(torch.float32).sum()
            _accu = _accu / len(x)
            print(len(x), len(y), _accu, len(_tt_pred_centralised), _tt_pred_centralised.sum())
        x_loc = [(v+1) * args.epochs * (len(tr_data[0]) // args.batch_size) for v in range(communication_round + 1)]
        y_loc.append(_accu.numpy())
        plt.plot(x_loc, y_loc, ls='', marker='*', color='red')        
        wandb.log({"centralised_accuracy_plot_communication_rounds_%02d_device_%02d_epochs_%02d" % (args.num_communication, 
                                                                                                    args.num_device, 
                                                                                                    args.epochs): plt})
        wandb.log({"centralised_test_accuracy": _accu})
        plt.close('all')
        test_prediction["prediction_communication_round"][communication_round] = _pred.detach().cpu().numpy()
    wandb.finish()
    with open(model_dir + "/statistics.obj", "wb") as f:
        pickle.dump(test_prediction, f)

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


if __name__ == "__main__":
    args = params.give_args()
    for arg in vars(args):
        print(arg, getattr(args, arg))
    try:
        if args.gpu_index == 10:
            free_id = get_freer_gpu()
            use_id = free_id[-args.num_gpu:]
            use_id_list = ",".join(["%d" % i for i in use_id])
        else:
            use_id_list = ",".join(["%d" % v for v in [args.gpu_index]])
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = use_id_list
    except:
        print("GPU doesn't exist")
    if args.loc == "home":
        model_mom = "../exp_data/"
    elif args.loc == "scratch":
        model_mom = "/scratch/blia/exp_data/"  
    elif args.loc == "nobackup":
        model_mom = "/nobackup/blia/exp_data/"  
    model_mom += "LocalSGD/"

    model_dir_sub = model_mom + "decentralised_training_iid_%s_alpha_%.1f/" % (args.iid, args.alpha)
    
    model_dir_sub += "enc_lr_%.3f_%04d_devices_%04d_communication_rounds_%02d_epochs/" % (args.enc_lr, args.num_device, args.num_communication, 
                                                                                          args.epochs)
    model_dir_sub += "version_%s/" % args.version
    train(args, model_dir_sub)




            
            
