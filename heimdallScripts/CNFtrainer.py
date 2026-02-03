from modelClasses import ddp_setup, CNF, CNF_trainer, trainingDataSet
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import torchinfo
import pickle

from validateCNF import valCNF


EPSILON = 1e-6


#thetaData = torch.tensor(np.load("/raid/vigneshk/data/theta_data.npy")).float()
dataPoisson_latent = torch.tensor(np.load("/raid/vigneshk/data/poissonEncoded.npy")).float()
thetaStandard = torch.tensor(np.load("/raid/vigneshk/data/thetaData_standard.npy")).float()

latent_mean = dataPoisson_latent.mean(axis=0)
latent_std = dataPoisson_latent.std(axis=0, correction=0)
dataPoisson_latent_Standard = (dataPoisson_latent - latent_mean) / (latent_std + EPSILON)

def prepare_Model(rank : int, hyper_params : dict):
    n_features = int(thetaStandard.shape[1])
    n_layers = 5
    hidden_features = 20
    contextF = int(dataPoisson_latent_Standard.shape[1])
    num_bins = 10
    tails = "linear"
    tail_bound = 3.5

    cnf = CNF(n_features,
              context_features= contextF,
              n_layers=n_layers,
              hidden_features=hidden_features,
              num_bins = num_bins,
              tails = tails,
              tail_bound = tail_bound)

    if rank == 0 :
        temp = {"CNFn_layers" : n_layers,
                "Layer Type" : [],
                "Hidden Width" : hidden_features,
                "spline_num_bins" : num_bins, 
                "spline_tails" : tails, 
                "spline_tail_bound" : tail_bound
                }

        uniqueLayers = int(len(cnf.transforms) / n_layers)
        #print(f"total unique: {uniqueLayers}")

        temp_idx = 0
        temp_str = str()
     
        for layer_idx in range(uniqueLayers):
            #print(f"Layer: {layer_idx}")
            temp_str = str(cnf.transforms[layer_idx])
            temp_idx = temp_str.find("(")
            temp["Layer Type"].append(temp_str[:temp_idx])
        
        hyper_params.update(temp)
        print(f"prepareModel : {hyper_params}")

    #print(f"data:{n_features}, context:{contextF}")
    return cnf

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset, shuffle = False)
    )

def main(rank: int, world_size: int, total_epochs: int, batch_size: int, base_hyper_params : dict):
    hyper_params = dict(base_hyper_params)

    ddp_setup(rank, world_size)
    CNFmodel = prepare_Model(rank, hyper_params)
    CNFmodel = CNFmodel.train()
    dataset = trainingDataSet(thetaStandard,dataPoisson_latent_Standard)
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = CNF_trainer(CNFmodel, train_data, rank, batch_size)  

    if rank == 0:
        hyper_params["Optimizer"] = str(trainer.optimizer)
        print(f"main : {hyper_params}")

        with open('/raid/vigneshk/Models/hP.bin', 'wb') as handle:
            pickle.dump(hyper_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    trainer._train(total_epochs)
    torch.distributed.barrier() #makes sure all GPUs finish before next one starts
    destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    batch_size = 4096
    total_epochs = 1 

    CNF_hyperParams = {}

    CNF_hyperParams =  {"batch_size" : batch_size,
             "Grad Clip" : True,
             "Epochs" : total_epochs}

    print(f"Before : {CNF_hyperParams}")
    
    mp.spawn(main, args=(world_size, total_epochs, batch_size, CNF_hyperParams), nprocs=world_size) 

    valCNF()






