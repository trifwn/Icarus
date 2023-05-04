import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

from ICARUS.Software.GenuVP3.postProcess.getStripData import getStripData


def GNVPstrips3D(pln, case, NBs, category = 'Wind'):

    stripDat, data = getStripData(pln, case, NBs)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_title(f"{pln.name} {category} Data")
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(30, 150)
    ax.axis('scaled')
    ax.set_xlim(-pln.span/2, pln.span/2)
    ax.set_ylim(-pln.span/2, pln.span/2)
    ax.set_zlim(-pln.span/2, pln.span/2)
    maxValue = data[category].max()
    minValue = data[category].min()

    norm = mpl.colors.Normalize(vmin=minValue, vmax=maxValue)
    cmap = cm.get_cmap('viridis', 12)
    
    for i,wg in enumerate(pln.surfaces):
        i = i+1
        if i not in NBs:
            continue
        for j,surf in enumerate(wg.allStrips):
            stripD = data[(data['Body']== i) & (data['Strip']== j+1)]
            stripD = float(stripD[category].values)
            color = cmap(norm(stripD))
            surf.plotStrip(fig,ax,None,color)
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.2)
    plt.show()
    return stripDat

def GNVPstrips2D(pln, case, NB, category = 'Wind'):
    
    if type(NB) is not int:
        print("Only one body can be selected for 2D plots")
        return 0
    
    stripDat, data = getStripData(pln, case, [NB])
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title(f"{pln.name} {pln.surfaces[NB-1].name} {category} Data")
    ax.set_xlabel('Spanwise')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel(category)
    x = [i for i,data in enumerate(data[category])] 
    ax.scatter(x,data[category])


    return stripDat