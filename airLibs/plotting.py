import matplotlib.pyplot as plt
import os
import numpy as np

def plotAeorCoeffs(data):

    fig, axs = plt.subplots(2, 2,figsize=(8,8))

    axs[0, 0].set_title('Cm vs AoA')
    # axs[0, 0].set_xlabel('a')
    axs[0, 0].set_ylabel('Cm')
    axs[0, 1].set_title('Cd vs AoA')
    axs[0, 1].set_xlabel('a')
    axs[0, 1].set_ylabel('Cd')
    axs[1, 0].set_title('Cl vs AoA')
    axs[1, 0].set_xlabel('a')
    axs[1, 0].set_ylabel('Cl')
    axs[1, 1].set_title('Cl vs Cd')
    axs[1, 1].set_xlabel('Cd')
    # axs[1, 1].set_ylabel('Cl')
    for dat in data:
        foo,style,label = dat    
        a,cl,cd,cm = foo.T
        axs[0, 1].plot(a,cd,style,label=label)
        axs[1, 0].plot(a,cl,style,label=label)
        axs[1, 1].plot(cd,cl,style,label=label)
        axs[0, 0].plot(a,cm,style,label=label)
    axs[0, 1].legend()
    axs[1, 0].legend()
    axs[1, 1].legend()
    axs[0, 0].legend()
    fig.tight_layout()

def plotCP(angle):
    fname = 'COEFPRE.OUT'
    folders = next(os.walk('.'))[1]
    if angle<0:
        anglef = 'm'+str(angle)[::-1].strip('-').zfill(6)[::-1]
    else:
        anglef = str(angle)[::-1].zfill(7)[::-1]
    fname =f'{anglef}/{fname}'
    data = np.loadtxt(fname).T
    c = data[0]
    p1 = data[1]
    plt.title('Pressure Coefficient')
    plt.xlabel('x/c')
    plt.ylabel('C_p')
    plt.plot(c,p1)
    plt.show()
