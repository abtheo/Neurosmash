import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import collections
import sys
import os
import glob
import math
import sklearn
from sklearn import metrics



result_filepaths = glob.glob(".\\Results\\*.*")
#result_filepaths = glob.glob(".\\Results\\[p-v]_[0-9]_*.*")
print(f"Found files: {result_filepaths}")

cmap = plt.cm.Blues

for i,result in enumerate(result_filepaths):
    name = result_filepaths[i][8:-4]

    R = np.load(result)

    #Scale between 0-1, meaning results are also indicative of a win percentage
    R = R / 10

    plt.subplot(math.floor(len(result_filepaths) / math.floor(math.sqrt(len(result_filepaths)))), math.ceil(len(result_filepaths) / math.floor(math.sqrt(len(result_filepaths)))), i+1)

    #Plot Moving Averages
    window_width = 25
    cumsum_vec = np.cumsum(np.insert(R, 0, 0)) 
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    X = range(len(ma_vec))
    plt.plot(X, ma_vec)

    #Graph properties
    plt.ylim([0,1])
    plt.xlim([0,250])
    plt.title(name)

    mean = np.mean(R)
    rmse = np.sqrt(((R - np.ones(len(R))) ** 2).mean())
    print(f"{name} :\n Mean = {mean}\n RMSE = {rmse} ")

    #fpr, tpr, thresholds = metrics.roc_curve(np.ones(len(R)), R)
    # auc = metrics.auc(range(len(R)), R)#(fpr, tpr)
    # print(auc)

    plt.legend([Line2D([0], [0], color=cmap(mean), lw=0.5) , Line2D([0], [0], color=cmap(rmse), lw=0.5)],
            [f"Mean: {mean}", "RMSE: %.3f" % rmse],
            loc='upper left')

    # plt.legend([Line2D([0], [0], color=cmap(C["Return"]["BackProduction"]), lw=4)],
    #          ["R : BP = %.3f" % C["Return"]["BackProduction"]],
    #          loc='upper right')




plt.subplots_adjust(left=0.13, bottom=0, right=0.9, top=0.95, wspace=0.25, hspace=0.5)
plt.show()