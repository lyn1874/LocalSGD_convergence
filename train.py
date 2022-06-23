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


class CentralLightning(LightningModule):
    def __init__(self, batch_size=128, enc_lr=0.001, strategy="dp", 
                 lr_schedule="cosine", epochs=50, num_gpus=1, model_dir=".."):
        super().__init__()
        try:
            free_id = get_freer_gpu()
            use_id = free_id[-num_gpus:]
            use_id_list = ",".join(["%d" % i for i in use_id])
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = use_id_list
        except:
            print("GPU doesn't exist")

        self.save_hyperparameters()
        self.mlp_model = mlp.MLP(784, 1)
        num_samples = 50000 * 2
        global_batch_size = global_batch_size_calc(batch_size, num_gpus, strategy)
        self.train_iters_per_epoch = num_samples // global_batch_size
        self.lr_schedule = get_lr_schedule(enc_lr, epochs, self.train_iters_per_epoch, lr_schedule)
        self.model_dir=model_dir

    def configure_optimizers(self):
        params = self.parameters()
        weight_decay_param = 5e-4 #5e-4
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
        self.log("train_loss", loss, sync_dist=True)
        tr_pred = (pred.detach().cpu() >= 0.5).to(torch.float32)
        accu = (tr_pred == y.detach().cpu()).sum().div(len(x))
        self.log("train_accuracy", accu, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.mlp_model(x.squeeze(1)).squeeze(1)
        np.save(self.model_dir + "/tt_stat_%06d" % self.global_step, 
                [y.detach().cpu().numpy(), pred.detach().cpu().numpy()])
        loss = mlp.loss(pred, y)
        self.log("validation_loss", loss, sync_dist=True)
        y_detach_cpu = y.detach().cpu()
        val_pred = (pred.detach().cpu() >= 0.5).to(torch.float32)
        accu = (val_pred == y_detach_cpu).sum().div(len(x))
        self.log("validation_accuracy", accu, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self.mlp_model(x.squeeze(1)).squeeze(1)
        loss = mlp.loss(pred, y)
        self.log("test_loss", loss, sync_dist=True)
        y_detach_cpu = y.detach().cpu()
        tt_pred = (pred.detach().cpu() >= 0.5).to(torch.float32)
        accu = (tt_pred == y_detach_cpu).sum().div(len(x))
        self.log("test_accuracy", accu, sync_dist=True)
            

def train(args, model_dir):
    create_dir(model_dir)
    args.seed_use = int(args.seed_use)
    tr_data, tr_label, tt_data, tt_label = pd.GetData(1).forward(args.seed_use, True)
    pl.seed_everything(args.seed_use)
    model_use = CentralLightning(args.batch_size, args.enc_lr, "dp", 
                                 args.lr_decay_schedule, args.epochs, args.num_gpu, model_dir=model_dir)
    project_name = "Local_SGD_Learning"
    wandb_logger = WandbLogger(project=project_name, config=args,
                               save_dir=model_dir+"/")
    wandb_logger.watch(log_freq=500, log_graph=False, model=model_use)
    
    transform_use = transforms.ToTensor()
    
    tr_dataset = pd.ImageLoader(tr_data, tr_label, transform_use)
    tt_dataset = pd.ImageLoader(tt_data, tt_label, transform_use)
    
    train_loader = pd.get_dataloader(tr_dataset, args.batch_size, args.workers_load_data)    
    val_loader = pd.get_test_dataloader(tt_dataset, len(tt_dataset), args.workers_load_data)

    print("The length of the training loader", len(train_loader), "testing loader", len(val_loader))

    log_steps = int(len(train_loader) / args.num_gpu) if args.num_gpu > 0 else len(train_loader)
    log_steps = [30 if log_steps > 30 else log_steps][0]
    trainer = pl.Trainer(max_epochs=args.epochs, progress_bar_refresh_rate=20,
                         gpus=args.num_gpu, logger=wandb_logger, log_every_n_steps=log_steps,
                         strategy=args.strategy, check_val_every_n_epoch=1,
                         deterministic=True,
                         gradient_clip_val=1.0)

    trainer.fit(model_use, train_loader, val_loader)
    trainer.save_checkpoint(model_dir + "/model-epoch=%02d.ckpt" % args.epochs)
    wandb.finish()


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


if __name__ == "__main__":
    args = params.give_args()
    for arg in vars(args):
        print(arg, getattr(args, arg))
    try:
        free_id = get_freer_gpu()
        use_id = free_id[-args.num_gpu:]
        use_id_list = ",".join(["%d" % i for i in use_id])
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

    model_dir_sub = model_mom + "centralised_training/"

    train(args, model_dir_sub)




            
            
