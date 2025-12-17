import numpy as np
import matplotlib.pyplot as plt

def gauss(N,mu,sig,x):
    term1 = N / (sig * np.sqrt(2 * np.pi))
    term2 = np.exp(-0.5 * ((x - mu) / sig)**2)
    return term1 * term2 


def produceData(N1,mu1,sig1,N2,mu2,sig2,start,end,size):
    raw = np.linspace(start,end,size)
    print(raw)

    x = gauss(N1,mu1,sig1,raw)
    y = gauss(N2,mu2,sig2,raw)

    data = x+y

    dataPoisson = np.random.poisson(lam=data,size=size)

    print(data)
    print(x)
    print(y)
    print(dataPoisson)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    axes[0].plot(raw,data, color='Yellow')
    axes[0].set_title('Histogram without Poisson')

    axes[1].plot(raw, dataPoisson, color='Pink')
    axes[1].set_title('Histogram with Poisson')

    for ax in axes:
        ax.set_xlabel('Theta')
        ax.set_ylabel('d')

    
    plt.tight_layout()

    plt.show()

    return dataPoisson

def generateParams():
    N1 = np.random.uniform(0,5,1)
    N2 = np.random.uniform(0,3,1)

    mu1 = np.random.uniform(1,3,1)
    mu2 = np.random.uniform(5,9,1)

    sig1 = np.random.uniform(1,3,1)
    sig2 = np.random.uniform(5,9,1)

    return N1,mu1,sig1,N2,mu2,sig2

N1,mu1,sig1,N2,mu2,sig2 = generateParams()
produceData(N1,mu1,sig1,N2,mu2,sig2,0,10,10)
    
