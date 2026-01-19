#base stack
import torch
import torchinfo
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

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
                #print(loss_rec)


    def _train(self,max_epoch):
        for epoch in tqdm(range(max_epoch)):
            self._run_epoch(epoch)
            if (epoch == max_epoch -1) and self.gpu_id == 0:
                self._save_checkpoint()


        plt.plot(self.r_losses)
        plt.savefig(self.plotDir+"AELoss.png")

    def _save_checkpoint(self):
        print(self.r_losses)
        PATH = "/raid/vigneshk/Models/AE_checkpoint.pt"
        torch.save({
        "AE_Model": self.AEModel.module.state_dict(),
        "AE_Optim": self.optimizer.module.state_dict(),
        },  PATH)

        print(f"Training checkpoint saved at {PATH}")



class CNF(torch.nn.Module):
    def __init__(self,
                 n_features, #Central data features 6 Ni,mui,sigi (i1,2)
                 context_features, #compressed Poisson features
                 n_layers,
                 hidden_features
                 ):
        super().__init__()

        base_dist = StandardNormal(shape=[n_features])
        transforms = []

        for i in range(n_layers):
            transforms.append(MaskedAffineAutoregressiveTransform(features=n_features, 
                                                                  hidden_features=hidden_features, 
                                                                  context_features=context_features)) #conditioned on compressed poissonData
            transforms.append(ReversePermutation(features=n_features))

        transform = CompositeTransform(transforms)
        self.flow = Flow(transform,base_dist)


 

     

