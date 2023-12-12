import numpy as np


def xflr_eigs(filename):
    f = open(filename)
    with open(filename, encoding='utf-8') as f:
        lines = f.readlines()
    long_li = lines[105]
    lat_li = lines[116]
    eig_mat = np.zeros((5, 2))
    eig_mat[0, 0] = float(long_li[22:38].partition("+")[0])
    eig_mat[0, 1] = float(long_li[22:38].partition("+")[2].partition("i")[0])
    eig_mat[1, 0] = float(long_li[74:92].partition("+")[0])
    eig_mat[1, 1] = float(long_li[74:92].partition("+")[2].partition("i")[0])
    eig_mat[2, 0] = float(lat_li[22:38].partition("+")[0])
    eig_mat[2, 1] = float(lat_li[22:38].partition("+")[2].partition("i")[0])
    eig_mat[3, 0] = float(lat_li[46:67].partition("+")[0])
    eig_mat[3, 1] = float(lat_li[46:67].partition("+")[2].partition("i")[0])
    eig_mat[4, 0] = float(lat_li[99:119].partition("+")[0])
    eig_mat[4, 1] = float(lat_li[99:119].partition("+")[2].partition("i")[0])

    return eig_mat
