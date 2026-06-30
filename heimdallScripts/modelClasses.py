#base stack
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

#nflows
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import (
    CompositeTransform,
)
from nflows.transforms.autoregressive import (
    MaskedAffineAutoregressiveTransform, MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
)

from nflows.transforms.permutations import ReversePermutation

#parallelizing
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

import numpy as np
import random
import consts

def printNormGrad(parameters : torch.tensor):
    total = 0
    for p in parameters:
        if p.grad is not None:
            total += p.grad.detach().norm(2).item()

    return total**0.5


def ddp_setup(rank : int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


class Encoder(torch.nn.Module):
    def __init__(self, 
                 input_dim : int, 
                 middle_dim : int, 
                 output_dim : int):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim,middle_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(middle_dim,middle_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(middle_dim,middle_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(middle_dim,output_dim)
        )
    def forward(self,x):
        return self.encoder(x)


class CNF(torch.nn.Module):
    def __init__(self,
                 n_features : int,
                 context_features : int,
                 n_layers : int,
                 hidden_features : int,
                 num_bins : int,
                 tails : str,
                 tail_bound : float
                 ):
        super().__init__()
            
        base_dist = StandardNormal(shape=[n_features])
        self.transforms = []

        for i in range(n_layers):
            self.transforms.append(ReversePermutation(features=n_features))
            self.transforms.append(MaskedAffineAutoregressiveTransform(features=n_features, 
                                                                  hidden_features=hidden_features, 
                                                                  context_features=context_features)) #conditioned on compressed poissonData
            
            self.transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(features = n_features,
                                                                                      hidden_features = hidden_features,
                                                                                      context_features = context_features,
                                                                                      num_bins = num_bins,   #10
                                                                                      tails = tails, #linear
                                                                                      tail_bound = tail_bound)) #3.5
            #self.transforms.append(ReversePermutation(features=n_features))

                   
        transform = CompositeTransform(self.transforms)
        self.flow = Flow(transform,base_dist)

    def forward(self,x,context):
        return self.flow.log_prob(x,context=context)

class trainingDataset(Dataset):
    def __init__(self, t_string : str, d_string : str):
        self.theta = np.load(t_string,mmap_mode='r')
        self.data = np.load(d_string,mmap_mode='r')
                  
    def __len__(self): 
        return len(self.data)


    def __getitem__(self, idx): 
        return (torch.from_numpy(self.theta[idx]),
        torch.from_numpy(self.data[idx]))

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset, shuffle = True),
        num_workers = 1,
        prefetch_factor = 2
    )


class CNF_trainer():
    def __init__(self,
                 EModel : Encoder,
                 CNFModel : CNF,
                 theta_paths: str,
                 data_paths : str,
                 val_theta_paths : str,
                 val_data_paths : str,
                 gpu_id: int,
                 batch_size : int
                 ):

        
        self.gpu_id = gpu_id
        self.EModel = EModel.to(gpu_id)
        self.CNFModel = CNFModel.to(gpu_id)
        self.EModel = DDP(self.EModel, device_ids=[gpu_id])
        self.CNFModel = DDP(self.CNFModel, device_ids=[gpu_id])
        self.cnf_losses = []
        self.val_cnf_losses = []
        self.data_paths = data_paths
        self.theta_paths = theta_paths
        self.val_data_paths = val_data_paths
        self.val_theta_paths = val_theta_paths
        self.plotDir = "plots/"

        #CNF HYPER-PARAMS TO CHANGE
        self.max_norm = None
        
        self.optimizer = torch.optim.Adam([
            {"params": self.EModel.parameters(), "lr": consts.encoder_lr}, 
            {"params": self.CNFModel.parameters(), "lr": consts.cnf_lr}
            ])

        self.batch_size = batch_size

    def _burn_in(self,epoch):

        random_samples = random.sample(range(min(4,len(self.data_paths))),4)
        total = []
        #print("Burn in Phase, calculating max_norm for gradient clipping for files : ")

        for file_idx in random_samples:
            file_name_t = self.theta_paths[file_idx]
            file_name_d = self.data_paths[file_idx]
            #print(f"{file_name_d}")
            dataset = trainingDataset(file_name_t, file_name_d)
            train_data = prepare_dataloader(dataset, self.batch_size)
            train_data.sampler.set_epoch(epoch*len(random_samples)+file_idx)
            
            for dS in train_data:
                x_input,x_cond = dS
                x_input = x_input.to(self.gpu_id, non_blocking = True)
                x_cond = x_cond.to(self.gpu_id, non_blocking = True)

                self.optimizer.zero_grad(set_to_none=True)
                z_cond = self.EModel(x_cond)
                nll = - self.CNFModel(x_input, context=z_cond)
                cnf_loss = nll.mean()
                cnf_loss.backward() 
            
                eGrad = printNormGrad(self.EModel.parameters())
                cnfGrad = printNormGrad(self.CNFModel.parameters())
                total.append((eGrad*eGrad+cnfGrad*cnfGrad)**0.5)
        
        max_norm = torch.tensor([np.percentile(np.asarray(total),99.9)], device = self.gpu_id)
        self.optimizer.zero_grad(set_to_none=True)

        return max_norm

    def _loss_validation(self,epoch):
        self.CNFModel.eval()
        self.EModel.eval()
        with torch.no_grad():
            for file_idx in range(len(self.val_data_paths)):
                file_name_t = self.val_theta_paths[file_idx]
                file_name_d = self.val_data_paths[file_idx]
                #print(f"Validation - epoch : {epoch}; {file_name_d}, {file_name_t}")
                dataset = trainingDataset(file_name_t, file_name_d)
                train_data = prepare_dataloader(dataset, self.batch_size)
                train_data.sampler.set_epoch(epoch*len(self.val_data_paths)+file_idx)
                for dS in train_data:
                    x_input,x_cond = dS
                    x_input = x_input.to(self.gpu_id, non_blocking = True)
                    x_cond = x_cond.to(self.gpu_id, non_blocking = True)
                    z_cond = self.EModel(x_cond)
                    nll = - self.CNFModel(x_input, context=z_cond)
                    cnf_loss = nll.mean()
                    self.val_cnf_losses.append(cnf_loss.detach()) #not .item() so no forced sync we will just sync at the end

        self.CNFModel.train()
        self.EModel.train()


       

    def _run_batch(self, x_input, x_cond,record_loss : bool):
        self.optimizer.zero_grad(set_to_none=True)
        z_cond = self.EModel(x_cond)
        nll = - self.CNFModel(x_input, context=z_cond)
        cnf_loss = nll.mean()
        cnf_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.CNFModel.parameters(),max_norm = self.max_norm)

        self.optimizer.step()

        if record_loss:
            return cnf_loss.item()
        else :
            return None


    def _run_epoch(self,epoch):    
        for file_idx in range(len(self.data_paths)):
            file_name_t = self.theta_paths[file_idx]
            file_name_d = self.data_paths[file_idx]
            dataset = trainingDataset(file_name_t, file_name_d)
            train_data = prepare_dataloader(dataset, self.batch_size)
            train_data.sampler.set_epoch(epoch*len(self.data_paths)+file_idx)

            print(f"[GPU{self.gpu_id}] Epoch {epoch} | file : {file_name_t},{file_name_d} | Steps: {len(train_data)}")
            for counter,dS in enumerate(train_data):
                x_batch,x_cond = dS
                #print(x_batch.shape, x_cond.shape)
                x_batch = x_batch.to(self.gpu_id, non_blocking = True)
                x_cond = x_cond.to(self.gpu_id, non_blocking = True)
                loss_rec = self._run_batch(x_batch,x_cond,record_loss = (counter % 10000 == 0))

                if loss_rec is not None:
                    self.cnf_losses.append(loss_rec)
                    print(loss_rec)

    def _train(self,max_epoch : int, PATH : str): 
        for epoch in tqdm(range(max_epoch)):
            if epoch == 0:
                max_norm = self._burn_in(epoch)   # every rank does this

                torch.distributed.all_reduce(
                    max_norm,
                    op=torch.distributed.ReduceOp.MAX
                )

                self.max_norm = max_norm.item()
                if self.gpu_id == 0:
                    print(f"max_norm : {self.max_norm}")

            self._run_epoch(epoch)
            
            if epoch % 2 == 0:
                self._loss_validation(epoch)


            if (epoch == max_epoch -1):
                self.val_cnf_losses = torch.stack(self.val_cnf_losses)

                torch.distributed.all_reduce(
                    self.val_cnf_losses,
                    op=torch.distributed.ReduceOp.SUM
                )
                
                self.val_cnf_losses = self.val_cnf_losses / torch.distributed.get_world_size()

                if self.gpu_id == 0:
                    self._save_checkpoint(PATH)

                    plt.plot(self.cnf_losses)
                    plt.savefig(self.plotDir+"CNFLoss.png")
                    plt.clf()
                    print(f"Saved CNF Loss Plot at {self.plotDir} CNFLOSS.png")

                    plt.plot(self.val_cnf_losses.cpu())
                    plt.savefig(self.plotDir+"val_CNFLoss.png")
                    plt.clf()
                    print(f"Saved CNF Loss Plot at {self.plotDir} val_CNFLOSS.png")

        

    def _save_checkpoint(self, PATH : str):
        torch.save({
        "CNF_Model": self.CNFModel.module.state_dict(),
        "CNF_Optim": self.optimizer.state_dict(),
        "E_Model": self.EModel.module.state_dict(),
        "E_Optim": self.optimizer.state_dict(),
        },  PATH + "Model_checkpoint.pt")

        print(f"Training checkpoint saved at {PATH}")










'''
self.GradNorms = []
for p in self.CNFModel.parameters():
    pNorm = p.grad.norm().item()
    self.GradNorms.append(pNorm)
normTorch = torch.tensor(self.GradNorms)
nanFlag = ~torch.isfinite(normTorch)
print(f"idx : {nanFlag}")
print(f"Non-Finites - {normTorch[nanFlag]}")
print(normTorch.nanmean())
plt.plot(self.GradNorms)
plt.savefig(self.plotDir+"CNFNorms.png")
plt.clf()
'''








        





 

     

