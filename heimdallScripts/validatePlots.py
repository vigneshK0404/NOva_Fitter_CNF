from fpdf import FPDF
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
from sklearn.cluster import MeanShift, estimate_bandwidth, DBSCAN
from matplotlib.ticker import MaxNLocator




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

def ModeDBScan(thetaDist : np.array, clusterDist : float, min_samples : int):
    db = DBSCAN(eps = clusterDist, min_samples = min_samples).fit(thetaDist)
    uniqueClusters = set(db.labels_)
    uniqueClusters.discard(-1)
    
    centers = []
    lens = []

    for i in uniqueClusters:
        mask = db.labels_ == i
        cluster = thetaDist[mask]
        lens.append(cluster.shape[0])
        clusterCentroid = np.mean(cluster,axis=0)
        centers.append(clusterCentroid)
        

    centers = np.array(centers)
    lens = np.array(lens)

    idx = np.argmax(lens)

    return centers[idx]



def ModeMeanShift(thetaDist: np.array, smoothing: float, minRatio: int):

    bandwidth = estimate_bandwidth(thetaDist, quantile=0.2, n_samples=1000) * smoothing
    min_freq = int(thetaDist.shape[0] / minRatio)

    ms = MeanShift(
        bandwidth=bandwidth,
        bin_seeding=True,
        max_iter=100,
        min_bin_freq=min_freq
    )

    ms.fit(thetaDist)
    return ms.cluster_centers_

def plot2DMarginals(truth : np.array, thetaDist : np.array, titles : list, base_PATH : str):

    iterations = thetaDist.shape[1]
    
    pdf = FPDF()
    
    for i in range(iterations):
        x = thetaDist[:,i]
        trueX = truth[i]
        titleX = titles[i]
        for j in range(i+1,iterations):
            y = thetaDist[:,j]
            trueY = truth[j]
            titleY = titles[j]
            title = f"{titleX} vs {titleY}"
            imagePath = base_PATH + title + ".png"
            plt.figure()            
            H, xeds, yeds, _ = plt.hist2d(x,y,bins=100,cmap='viridis')
            plt.axvline(x=trueX, linestyle='--',color='red')
            plt.axhline(y=trueY, linestyle='--',color='red')
            idx = np.unravel_index(np.argmax(H),H.shape)
            modeXCol = (xeds[idx[0]] + xeds[idx[0]+1])/2
            modeYRow = (yeds[idx[1]] + yeds[idx[1]+1])/2
            print("")
            print(f"idx : {idx}")
            print(f"{titleX} : [{xeds[0]}, {xeds[-1]}] -> {modeXCol}")
            print(f"{titleY} : [{yeds[0]}, {yeds[-1]}] -> {modeYRow}")
            print("")

            plt.colorbar(label='Frequency of points')
            plt.xlabel(titleX)
            plt.ylabel(titleY)
            plt.title(title)
            plt.locator_params(axis='both', nbins=10)
            plt.savefig(imagePath)
            plt.close()

            pdf.add_page()
            pdf.image(imagePath,x=10,y=60,w=200,h=170)
    pdf.output(base_PATH + "2DMarginals.pdf", "F")
        



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
