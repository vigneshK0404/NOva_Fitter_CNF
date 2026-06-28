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

def printNormGrad(parameters : torch.tensor):
    total = 0
    for p in parameters:
        if p is not None:
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
                 gpu_id: int,
                 batch_size : int
                 ):

        
        self.gpu_id = gpu_id
        self.EModel = EModel.to(gpu_id)
        self.CNFModel = CNFModel.to(gpu_id)
        self.EModel = DDP(self.EModel, device_ids=[gpu_id])
        self.CNFModel = DDP(self.CNFModel, device_ids=[gpu_id])
        self.cnf_losses = []
        self.data_paths = data_paths
        self.theta_paths = theta_paths
        self.plotDir = "plots/"

        #CNF HYPER-PARAMS TO CHANGE

        
        self.optimizer = torch.optim.Adam([
            {"params": self.EModel.parameters(), "lr": 1e-3}, 
            {"params": self.CNFModel.parameters(), "lr": 1e-3}
            ])

        self.batch_size = batch_size

    def _run_batch(self, x_input, x_cond,record_loss : bool):
        self.optimizer.zero_grad()
        z_cond = self.EModel(x_cond)
        nll = - self.CNFModel(x_input, context=z_cond)
        cnf_loss = nll.mean()
        cnf_loss.backward()

        """
        if self.gpu_id == 0 :
            eGrad = printNormGrad(self.EModel.parameters())
            cnfGrad = printNormGrad(self.CNFModel.parameters())
            print(f"E : {eGrad} CNF : {cnfGrad} total : {(eGrad*eGrad+cnfGrad*cnfGrad)**0.5}")
        """
        
        torch.nn.utils.clip_grad_norm_(self.CNFModel.parameters(),max_norm = 38.0)

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
            self._run_epoch(epoch)
            if (epoch == max_epoch -1) and self.gpu_id == 0:
                self._save_checkpoint(PATH)
                plt.plot(self.cnf_losses)
                plt.savefig(self.plotDir+"CNFLoss.png")
                plt.clf()
                print(f"Saved CNF Loss Plot at {self.plotDir} CNFLOSS.png")

        

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








        





 

     

