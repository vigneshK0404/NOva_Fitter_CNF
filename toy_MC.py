import numpy as np
import matplotlib.pyplot as plt

def gauss(N,mu,sig,x):
    N_t = N.reshape(-1,1)
    mu_t = mu.reshape(-1,1)
    sig_t = sig.reshape(-1,1)

    term1 = N_t / (sig_t * np.sqrt(2 * np.pi))
    term2 = np.exp(-0.5 * np.square((x - mu_t) / sig_t))
    return np.array(term1 * term2) 


def generatePrior(sampleSize):
    N1 = np.random.uniform(10,50,sampleSize)
    N2 = np.random.uniform(10,30,sampleSize)

    mu1 = np.random.uniform(1,3,sampleSize)
    mu2 = np.random.uniform(5,9,sampleSize)

    sig1 = np.random.uniform(1,3,sampleSize)
    sig2 = np.random.uniform(5,9,sampleSize)

    return N1,mu1,sig1,N2,mu2,sig2


sampleNumber = 10000000

N1,mu1,sig1,N2,mu2,sig2 = generatePrior(sampleNumber)

raw = np.arange(0.5,10,1) #startbinCenter, endBinEdge, StepSize [0.5,1.5...9.5]
gaussTotal = gauss(N1,mu1,sig1,raw) + gauss(N2,mu2,sig2,raw)

binNumber = len(gaussTotal)
dataPoisson = np.random.poisson(lam=gaussTotal,size=None)

thetaData = np.column_stack((N1,mu1,sig1,N2,mu2,sig2))

print(thetaData)
print(dataPoisson)


np.savez(
    "trainingData.npz",
    theta = thetaData,
    d = dataPoisson

        )









'''
binEdges = np.array(range(11))

for i in range(2):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    axes[0].hist(binEdges[:-1],binEdges,weights=gaussTotal[i], color='Yellow',edgecolor="black")
    axes[0].set_title('Histogram without Poisson')

    axes[1].hist(binEdges[:-1], binEdges, weights=dataPoisson[i] ,color='Pink',edgecolor="black")
    axes[1].set_title('Histogram with Poisson')

    for ax in adef produceData(N1,mu1,sig1,N2,mu2,sig2,start,end,stepSize):
    size = len(raw)
    
    dataPoisson = np.random.poisson(lam=data,size=size)

    
    return dataPoissonxes:
        ax.set_xlabel('d')
        ax.set_ylabel('Frequency')


    plt.tight_layout()

    plt.savefig(str(i))
'''
