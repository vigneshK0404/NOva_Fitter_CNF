from fpdf import FPDF
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch

EPSILON = 1e-4

def findMode(thetaDist : np.array):

    iterations = thetaDist.shape[1]
    modeVals = []

    for i in range(iterations):
        data = thetaDist[:,i]
        hist, bin_edges = np.histogram(data)
        idx = np.argmax(hist)
        mode = (bin_edges[idx] + bin_edges[idx+1])/2
        modeVals.append(mode.item())

    return np.array(modeVals)


def plotHist(thetaDist : np.array , titles : list, base_PATH : str):

    iterations = thetaDist.shape[1]

    
    PATH = base_PATH + "hP.bin"
    with open(PATH, 'rb') as handle:
        hyper_params = pickle.load(handle)

    full_string = str()
   
    for key,value in hyper_params.items():
        full_string += key
        full_string += " : "
        full_string += str(value)
        full_string += "\n"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.multi_cell(w=0,h=10,txt=full_string,border=1)
    
    for i in range(iterations):
        data_plot = thetaDist[:,i]
        outOfRange = data_plot[(data_plot > 100) | (data_plot < -100)]
        print(f"{titles[i]} Out of Range : {outOfRange}")

        imagePath = base_PATH + titles[i] + ".png"
        pdf.add_page()

        plt.figure()
        _, bins, _ = plt.hist(data_plot, edgecolor = "black")
        bin_diffs = np.diff(bins)
        if np.all(bin_diffs - bin_diffs[0]) == 0 :
            print(f"uniform binning : {titles[i]} - {bin_diffs[0]}")
        else :
            print(f"non-uniform binnins : {titles[i]} - {bin_diffs}")

        plt.xlabel("relative_difference %")
        plt.ylabel("counts")
        plt.title(titles[i])
        plt.savefig(imagePath)
        plt.close()
        pdf.image(imagePath,x=10,y=60,w=200,h=170)
    
    pdf.output(base_PATH + "ThetaPlots.pdf","F")


def drawLatent():
    dataPoisson_latent = torch.tensor(np.load("/raid/vigneshk/data/poissonEncoded.npy")).float()
    latent_mean = dataPoisson_latent.mean(dim=0)
    latent_std = dataPoisson_latent.std(dim=0, correction=0)

    dP_std = (dataPoisson_latent - latent_mean) / (latent_std + EPSILON)

    print(f"latent Space : {dP_std}")
    base = "/raid/vigneshk/plots/LatentPlots/LatentPoissonBin_Std"
    
    for i in range(10):
        data = dP_std[:,i]
        plt.hist(data)
        plt.savefig(base+str(i)+".png")
        plt.clf()
    


#drawLatent()
