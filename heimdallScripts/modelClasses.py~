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
    MaskedAffineAutoregressiveTransform,
)
from nflows.transforms.autoregressive import (
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
)
from nflows.transforms.permutations import ReversePermutation

#parallelizing
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os



def ddp_setup(rank : int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


class autoEncoder(torch.nn.Module):
    def __init__(self, 
                 input_dim : int, 
                 middle_dim : int, 
                 output_dim : int):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim,middle_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(middle_dim,output_dim)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(output_dim,middle_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(middle_dim,input_dim),
        )
    def _encode(self,x):
        return self.encoder(x)

    def _decode(self,x):
        return self.decoder(x)

    def forward(self,x):
        eData = self._encode(x)
        dData = self._decode(eData)
        return dData


class AE_trainer:
    def __init__(
            self,
            AEModel : autoEncoder,
            train_data: DataLoader,
            gpu_id: int,
            batch_size : int
            ):
        
        self.gpu_id = gpu_id
        self.AEModel = AEModel.to(gpu_id)
        self.train_data = train_data
        self.AEModel = DDP(self.AEModel, device_ids=[gpu_id])
        self.r_losses = []
        self.plotDir = "/raid/vigneshk/plots/"

        #CAN CHANGE AE TRAINING PARAMETERS HERE

        self.loss_fn = torch.nn.MSELoss() 
        self.optimizer = torch.optim.Adam(self.AEModel.parameters(), lr = 1e-4)
        self.batch_size = batch_size

    def _run_batch(self, x_input, record_loss : bool):
        self.optimizer.zero_grad()
        y_true = self.AEModel(x_input)
        loss = self.loss_fn(y_true,x_input)
        loss.backward()
        self.optimizer.step()

        if record_loss:
            return loss.item()
        else :
            return None
        

    def _run_epoch(self,epoch):
        #b_size = len(next(iter(self.train_data))[0])        
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        counter = 0
        for x_batch in self.train_data:
            counter += 1
            x_batch = x_batch.to(self.gpu_id, non_blocking = True)
            loss_rec = self._run_batch(x_batch,record_loss = (counter % 100 == 0))

            if loss_rec is not None:
                self.r_losses.append(loss_rec)
                print(loss_rec)


    def _train(self,max_epoch : int):
        for epoch in tqdm(range(max_epoch)):
            self._run_epoch(epoch)
            if (epoch == max_epoch -1) and self.gpu_id == 0:
                self._save_checkpoint()


        plt.plot(self.r_losses)
        plt.savefig(self.plotDir+"AELoss.png")

    def _save_checkpoint(self):
        PATH = "/raid/vigneshk/Models/"
        torch.save({
        "AE_Model": self.AEModel.module.state_dict(),
        "AE_Optim": self.optimizer.state_dict(),
        },  PATH + "AE_checkpoint.pt")

        print(f"Training checkpoint saved at {PATH}")



class trainingDataSet(Dataset):
  def __init__(self,thetaData, dataPoisson_latent):
    assert len(thetaData) == len(dataPoisson_latent)
    self.thetaData = thetaData
    self.dataPoisson_latent = dataPoisson_latent

  def __len__(self):
    return len(self.thetaData)

  def __getitem__(self, idx):
    return self.thetaData[idx], self.dataPoisson_latent[idx]


class CNF(torch.nn.Module):
    def __init__(self,
                 n_features : int, #Central data features 6 Ni,mui,sigi (i1,2)
                 context_features : int, #compressed Poisson features
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


class CNF_trainer():
    def __init__(self,
                 CNFModel : CNF,
                 train_data: DataLoader,
                 gpu_id: int,
                 batch_size : int
                 ):

        
        self.gpu_id = gpu_id
        self.CNFModel = CNFModel.to(gpu_id)
        self.train_data = train_data
        self.CNFModel = DDP(self.CNFModel, device_ids=[gpu_id])
        self.cnf_losses = []
        self.plotDir = "/raid/vigneshk/plots/"

        #CNF HYPER-PARAMS TO CHANGE

        
        self.optimizer = torch.optim.Adam(self.CNFModel.parameters(), lr = 3e-4)
        self.batch_size = batch_size 

    def _run_batch(self, x_input, x_cond,record_loss : bool):
        self.optimizer.zero_grad()
        nll = - self.CNFModel(x_input, context=x_cond)
        cnf_loss = nll.mean()
        cnf_loss.backward()
    
        torch.nn.utils.clip_grad_norm_(self.CNFModel.parameters(),max_norm = 1.0)

        self.optimizer.step()

        if record_loss:
            return cnf_loss.item()
        else :
            return None


    def _run_epoch(self,epoch):    
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)

        for counter,dS in enumerate(self.train_data):
            x_batch,x_cond = dS
            x_batch = x_batch.to(self.gpu_id, non_blocking = True)
            x_cond = x_cond.to(self.gpu_id, non_blocking = True)
            loss_rec = self._run_batch(x_batch,x_cond,record_loss = (counter % 1000 == 0))

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
        },  PATH + "CNF_checkpoint.pt")

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








        





 

     

