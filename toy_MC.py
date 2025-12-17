import numpy as np
import matplotlib.pyplot as plt

def gauss(N,mu,sig,x):
    return N/(sig*np.sqrt(np.pi)) * np.exp(-np.square(x-mu)/(2*np.square(sig)))


def produceData(N1,mu1,sig1,N2,mu2,sig2,start,end,size):
    raw = np.linspace(start,end,size)

    x = gauss(N1,mu1,sig1,raw)
    y = gauss(N2,mu2,sig2,raw)

    data = x+y

    dataPoisson = np.random.poisson(lam=data,size=size)

    print(data)
    print(x)
    print(y)
    print(dataPoisson)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    axes[0].hist(data, bins=10, color='Yellow', edgecolor='black')
    axes[0].set_title('Histogram without Poisson')

    axes[1].hist(dataPoisson, bins=10, color='Pink', edgecolor='black')
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



size = 1000
produceData(2,3,1.5,5,7,2,0.5,10.5,100)
    
