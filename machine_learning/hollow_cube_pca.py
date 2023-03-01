import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.model_selection import train_test_split

# from iwp_data import c_list, iso_value_list, relative_density_list
from hollow_cube_data import c_list, thickness_list

import numpy as np
import random

import seaborn as sns
import pandas as pd

from sklearn.decomposition import PCA

#----------------------- MAIN -----------------------#

def main():
    manualSeed = 42
    random.seed(manualSeed)
    np.random.seed(manualSeed)

    c_list.pop(327)
    thickness_list.pop(327)




    N = len(thickness_list)


    #----------------------- STANDARDIZE -----------------------#
 
    # x_data = np.asarray(c_list)

    x_data = np.array([np.array(inputArray) for inputArray in c_list])
    y_data = np.asarray(thickness_list)
    x_data = x_data.transpose()

    c11mean, c11std = np.mean(x_data[0]),  np.std(x_data[0])
    c12mean, c12std = np.mean(x_data[1]),  np.std(x_data[1]) 
    c44mean, c44std = np.mean(x_data[2]),  np.std(x_data[2]) 
    
    x_data[0] = ( x_data[0] - c11mean ) / c11std
    x_data[1] = ( x_data[1] - c12mean ) / c12std
    x_data[2] = ( x_data[2] - c44mean ) / c44std
    x_data = x_data.transpose()

    # x_data = np.asarray([[c11[i], c12[i], c44[i]] for i in range(len(c11))])

    y_data = ( y_data - np.mean(y_data) ) / np.std(y_data) 
    print(f"x_data.shape: {x_data.shape}")
    print(f"y_data.shape: {y_data.shape}")

    #----------------------- Principal Component Analysis -----------------------#

    pca = PCA(n_components = 3)

    principalComponents = pca.fit_transform(x_data)

    # principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1'])
    # principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2', 'principal component 3'])

    print(f"Principal components: {principalDf}")

    print(f"pca.explained_variance_ratio: {pca.explained_variance_ratio_} is the percentage of variance explained by each of the principal components")
    print(f"This shows that {round(pca.explained_variance_ratio_[0]*100, 2)} % of the information is captured by a single principal component")

    print(f"pca.components_:{pca.components_} are the principal axes in feature space, representing the directions of maximum variance in the data.") 

    #----------------------- CORRELATION PLOT -----------------------#
    c11 = np.asarray([e[0] for e in c_list])
    c12 = np.asarray([e[1] for e in c_list])
    c44 = np.asarray([e[2] for e in c_list])

    df = pd.DataFrame({"c11":c11.reshape(N,), "c12":c12.reshape(N,), "c44":c44.reshape(N,)}, index=range(0,N))  

    corr_matrix = df.corr()
    # sns.heatmap(corr_matrix)
    fig5 = plt.figure(figsize=plt.figaspect(1))
    mask = np.zeros_like(corr_matrix)
    # mask[np.triu_indices_from(mask)] = True
    mask[np.triu_indices(3,1)] = True
    covarianceMatrix = sns.heatmap((corr_matrix), mask=mask, square=True, cmap="bone", vmin = 0.9, vmax = 1, annot=True) #,linewidths=1, linecolor='black'
    # covarianceMatrix.set_title('Correlation Heatmap', fontdict={'fontsize':20}, pad=12)
    # sns.heatmap((corr_matrix), square=True, cmap="bone", vmin = 0.95, vmax = 1, annot=True)
    
    plt.show()


if __name__ == '__main__':
	main()