#base stack
import torch
import torchinfo
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import tqdm
from torch.utils.data import DataSet, DataLoader

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
                 output_dim : int) -> None:
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

    def _forward(self,x):
        eData = self.encode(x)
        dData = self.decode(eData)
        return dData


class AE_trainer:
    def __init__(
            self,
            input_dim : int,
            middle_dim : int,
            output_dim : int,
            AEModel : autoEncoder,
            train_data: DataLoader,
            gpu_id: int
            batch_size : int
            ) -> None :
        
        self.gpu_id = gpu_id
        self.AEModel = AEModel(input_dim,middle_dim,output_dim).to(gpu_id)
        self.train_data = train_data
        self.AEmodel = DDP(AEmodel, device_ids=[gpu_id])
        self.r_losses = []
        self.plotDir = "/raid/vigneshk/plots/"

        #CAN CHANGE TRAINING PARAMETERS HERE

        self.loss_fn = torch.nn.MSELoss() 
        self.optimizer = torch.optim.Adam(AEModel.parameters(), lr = 1e-4)
        self.batch_size = batch_size

    def _run_batch(self, x_input, y_pred):
        self.optimizer.zero_grad()
        y_true = self.AEModel._forward(x_input)
        loss = self.loss_fn(y_true,y_pred)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_epoch(self,epoch):
        b_size = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_size} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for x_batch in train_data:
            x_batch = x_batch.to(self.gpu_id)
            self.r_losses.append(self._run_batch(x_batch))

    def _train(self,max_epoch):
        for epoch in range(max_epoch):
            self._run_epoch(epoch)
        plt.plot(self.r_losses)
        plt.savefig(self.plotDir)


class CNF():
  def __init__(self,n_features, context_features,n_layers):

    base_dist = StandardNormal(shape=[n_features])
    transforms = []

    for i in range(n_layers):
      transforms.append(MaskedAffineAutoregressiveTransform(features=n_features, hidden_features=16, context_features=context_features)) #conditioned on compressed poissonData
      transforms.append(ReversePermutation(features=n_features))

    transform = CompositeTransform(transforms)
    self.flow = Flow(transform,base_dist)

