from modelClasses import autoEncoder
from generateDataFuncs import generateTrainingData, plots
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader



def latentGen(PATH : str, data_train : np.array):

    input_dim = int(data_train.shape[1])
    middle_dim = int(input_dim / 2)
    output_dim = int(input_dim / 3)

    dnumber = 0
    device = torch.device(f"cuda:{dnumber}" if torch.cuda.is_available() else "cpu")
    print(device)

    encodeModel = autoEncoder(input_dim,middle_dim,output_dim)
    ckpt_AE  = torch.load(f"{PATH}AE_checkpoint.pt", map_location=device)
    encodeModel.load_state_dict(ckpt_AE["AE_Model"])
    encodeModel.eval()
    encodeModel = encodeModel.to(device)


    dP_tensor = torch.tensor(data_train).float()
    p_latent_list = []

    batch_size = 100000


    p_DL = DataLoader(dP_tensor, batch_size = batch_size, shuffle=False)


    with torch.no_grad() :
        for x_batch in tqdm(p_DL) :
            x_batch = x_batch.to(device)
            encodeX = encodeModel._encode(x_batch).to("cpu").numpy()
            p_latent_list.append(encodeX)


    pPATH = "/raid/vigneshk/data/dataEncoded"
    meanPath = "/raid/vigneshk/data/latentMean"
    stdPath = "/raid/vigneshk/data/latentStd"

    p_latentData = np.concatenate(p_latent_list,axis=0)
    latent_mean = np.mean(p_latentData, axis=0)
    latent_std = np.std(p_latentData, axis=0, ddof=0)

    np.save(pPATH,p_latentData)
    np.save(meanPath,latent_mean)
    np.save(stdPath,latent_std)

    print(f"Saved latentData of shape : {p_latentData.shape} at {pPATH}")



def sampleGen(encodeModel : autoEncoder, device : torch.device):
    address = "/raid/vigneshk/plots/sampleEncode.png"
    testP,_ ,testG= generateTrainingData(1,1)
    
    print(f"Poisson_raw : {testP}")
    print(testG)

    testP = torch.tensor(testP, dtype = torch.float32, device = device)
    decodePtest = encodeModel(testP).to("cpu").detach().numpy().flatten()
    testP_CPU = testP.to("cpu").detach().numpy().flatten()
    
    print(decodePtest)
    
   
    bins = np.arange(0,20.5,0.2)
    plots(testP_CPU, testG, bins, address)



if __name__ == "__main__": 
    data_train = np.load("/raid/vigneshk/data/dataTrain.npy")
    latentGen("/raid/vigneshk/Models/NOvACNF/",data_train)

