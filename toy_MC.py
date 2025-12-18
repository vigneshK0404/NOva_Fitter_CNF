import numpy as np
import matplotlib.pyplot as plt

def gauss(N,mu,sig,x):
    term1 = N / (sig * np.sqrt(2 * np.pi))
    term2 = np.exp(-0.5 * ((x - mu) / sig)**2)
    return term1 * term2 


def produceData(N1,mu1,sig1,N2,mu2,sig2,start,end,stepSize):
    raw = np.arange(start,end,stepSize)
    x = gauss(N1,mu1,sig1,raw)
    y = gauss(N2,mu2,sig2,raw)

    data = x+y
    size = len(raw)
    binEdges = np.array(range(end+1))

    dataPoisson = np.random.poisson(lam=data,size=size)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    axes[0].hist(binEdges[:-1],binEdges,weights=data, color='Yellow',edgecolor="black")
    axes[0].set_title('Histogram without Poisson')

    axes[1].hist(binEdges[:-1], binEdges, weights=dataPoisson ,color='Pink',edgecolor="black")
    axes[1].set_title('Histogram with Poisson')

    for ax in axes:
        ax.set_xlabel('Theta')
        ax.set_ylabel('d')

    
    plt.tight_layout()

    plt.show()

    return dataPoisson

def generateParams():
    N1 = np.random.uniform(10,50,1)
    N2 = np.random.uniform(10,30,1)

    mu1 = np.random.uniform(1,3,1)
    mu2 = np.random.uniform(5,9,1)

    sig1 = np.random.uniform(1,3,1)
    sig2 = np.random.uniform(5,9,1)

    return N1,mu1,sig1,N2,mu2,sig2

N1,mu1,sig1,N2,mu2,sig2 = generateParams()
produceData(N1,mu1,sig1,N2,mu2,sig2,0.5,10,1)
#produceData(10,4,2,10,7,3.5,0.5,10,1)
    
