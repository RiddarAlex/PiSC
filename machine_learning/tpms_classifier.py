import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

from iwp_data import c_list as c_iwp
from hollow_cube_data import c_list_100 as c_hc

import numpy as np
import random

import copy

import seaborn as sns
import pandas as pd

#----------------------- CLASSES -----------------------#
class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

#----------------------- FUNCTIONS -----------------------#
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    ground_truth = torch.argmax(yb, dim=1)
    return (preds == ground_truth).float()#.mean()

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

def evaluate(model, loss_func, dl):
    model.eval()
    with torch.no_grad():
        losses, nums = zip(
            *[loss_batch(model, loss_func, xb, yb) for xb, yb in dl]
        )
    return np.sum(np.multiply(losses, nums)) / np.sum(nums)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    min_loss = 1e42
    epochsWithNoImprovement = 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        current_loss = evaluate(model, loss_func, valid_dl)

        # Early stopping
        if current_loss < min_loss:
            min_loss = current_loss
            epochsWithNoImprovement = 0
            bestModel = copy.deepcopy(model)
        else:
            epochsWithNoImprovement += 1

        if epochsWithNoImprovement > 200:
            model = bestModel
            epoch = epoch-200
            print('Early stopping!')
            break
        
        if epoch%50 == 0:
            print(f"Epoch {epoch} loss: {current_loss}")
            print(f"Current best: {min_loss}")
    
    return epoch + 1


#----------------------- MAIN -----------------------#
def main():

    manualSeed = 42
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    #----------------------- DATA PREPROCESSING -----------------------#
    standardization = True
    print(f"Hollow Cube datapoints: {len(c_hc)}")
    print(f"IWP datapoints: {len(c_iwp)}")
    c_list = c_hc + c_iwp

    y_hc = [[1, 0] for e in c_hc] #element 0: Hollow Cube
    y_iwp = [[0, 1] for e in c_iwp] #element 1: IWP
    class_ground_truth = y_hc + y_iwp


    #----------------------- STANDARDIZE -----------------------#
    x_data = np.array([np.array(inputArray) for inputArray in c_list]) #array of arrays
    y_data = np.array([np.array(outputArray) for outputArray in class_ground_truth])

    if standardization:
        x_data = x_data.transpose()

        c11mean, c11std = np.mean(x_data[0]),  np.std(x_data[0])
        c12mean, c12std = np.mean(x_data[1]),  np.std(x_data[1]) 
        c44mean, c44std = np.mean(x_data[2]),  np.std(x_data[2]) 
        
        x_data[0] = ( x_data[0] - c11mean ) / c11std
        x_data[1] = ( x_data[1] - c12mean ) / c12std
        x_data[2] = ( x_data[2] - c44mean ) / c44std
        x_data = x_data.transpose()
    else:
        scale_factor = 10**8
        x_data /= scale_factor

    x_data = torch.FloatTensor(x_data)
    y_data = torch.FloatTensor(y_data)#.unsqueeze(1)

    
    print(f"x_data.shape: {x_data.shape}")
    print(f"y_data.shape: {y_data.shape}")

    N = len(c_list)
    
    #----------------------- BUILD DATALOADER -----------------------#
    dataset = TensorDataset(x_data, y_data)
    train_ds, valid_ds, test_ds = torch.utils.data.random_split(dataset, [0.5, 0.2, 0.3], generator=torch.Generator().manual_seed(42))

    bs = 1  # batch size
    train_dl, valid_dl, test_dl = DataLoader(train_ds, batch_size=bs), DataLoader(valid_ds, batch_size=bs), DataLoader(test_ds, batch_size=bs)

    #----------------------- FFNN DESIGN -----------------------#
    #loss_func = torch.nn.MSELoss()
    loss_func = F.cross_entropy
    hiddenLayerSize = 16
    model = nn.Sequential(
        Lambda(lambda xx: xx.view(xx.size(0), -1)),    
        nn.Linear(3, hiddenLayerSize),
        nn.ReLU(),
        nn.Linear(hiddenLayerSize, hiddenLayerSize),
        nn.ReLU(),
        nn.Linear(hiddenLayerSize, hiddenLayerSize),
        nn.ReLU(),
        nn.Linear(hiddenLayerSize, hiddenLayerSize),
        nn.ReLU(),
        nn.Linear(hiddenLayerSize, hiddenLayerSize),
        nn.ReLU(),
        nn.Linear(hiddenLayerSize, 2),
    )
    
    #----------------------- TRAINING -----------------------#
    epochs = 1000  #1000  # how many epochs to train for

    # opt = optim.SGD(model.parameters(), lr=0.01)
    opt = optim.Adam(model.parameters())

    epoch = fit(epochs, model, loss_func, opt, train_dl, valid_dl)

    test_dl = DataLoader(test_ds, batch_size=len(test_ds)) ### SET TO ONE SINGLE BATCH
    print(f"After {epoch} epochs: Testdata Loss = {evaluate(model, loss_func, test_dl)}")

    #----------------------- TEST DATA PREDICTIONS -----------------------#
    x1_test = []
    x2_test = []
    x3_test = []
    y_test  = []
    predictions = []
    ground_truth = []
    accuracies = []
    
    for inputs, labels in test_dl:
        preds = model(inputs)
        predictions = preds
        ground_truth = labels  
        # predictions.append(preds)
        # ground_truth.append(labels)

    correct_predictions = accuracy(predictions, ground_truth)
    predictions = [e.item() for e in torch.argmax(predictions, dim=1)]
    ground_truth = [e.item() for e in torch.argmax(ground_truth, dim=1)]
    print(f"len(ground_truth) = {len(ground_truth)}")
    print(f"Predictions:{predictions} \nGround truth:{ground_truth}")  
    accuracies.append(correct_predictions.mean())
    
    print(f"accuracies = {accuracies}")
    print(f"Accuracy after {epoch} epochs: {sum(accuracies)/len(accuracies)}")
    
    true_HC = len([1 for i in range(len(predictions)) if ( (predictions[i] == ground_truth[i]) and (predictions[i] == 1) )])
    false_HC = len([1 for i in range(len(predictions)) if ( (predictions[i] != ground_truth[i]) and (predictions[i] == 1) )])

    true_IWP = len([1 for i in range(len(predictions)) if ( (predictions[i] == ground_truth[i]) and (predictions[i] == 0) )])
    false_IWP = len([1 for i in range(len(predictions)) if ( (predictions[i] != ground_truth[i]) and (predictions[i] == 0) )])

    # correct_pred_HC = [1 if ( (predictions[i] == ground_truth[i]) and (predictions[i] == 1) ) else 0 for i in range(len(predictions))]
    print(f"")
    # expression for item in iterable if condition == True
    print(f"true_HC = {true_HC}, false_HC = {false_HC}")
    print(f"true_IWP = {true_IWP}, false_IWP = {false_IWP}")
    print('breaknig')
    """"

    for xb, yb in test_dl:
        y = yb.squeeze(1).numpy()
        for sublist in xb:
            x1_test.append(sublist[0])
            x2_test.append(sublist[1])  
            x3_test.append(sublist[2])
        
        for element in yb:
            y_test.append(element)
        
        prediction = model(xb).detach().numpy()
        predictions.append(prediction)

    #----------------------- INVERSE TRANSFORM DATA -----------------------#
    y_test_predictions = np.asarray(predictions[0])
    y_test = np.asarray(y_test)
    x1 = np.asarray(x1_test)
    x2 = np.asarray(x2_test)
    x3 = np.asarray(x3_test)

    if standardization:
        x1 = x1 * c11std + c11mean
        x2 = x2 * c12std + c12mean
        x3 = x3 * c44std + c44mean
    else:
        x1 *= scale_factor
        x2 *= scale_factor
        x3 *= scale_factor

    #----------------------- ERROR IN COMPARABLE TERMS -----------------------#
    predictedRD = np.asarray([[e[0], e[1]] for e in y_test_predictions])
    groundTruthRD = np.asarray([[e[0]] for e in y_test])
    error = predictedRD-groundTruthRD
    print(predictedRD)
    print(groundTruthRD)
    # print(f"The mean squared error for predictions corresponding to test data is MSE = {np.mean(np.square(error))}")
    # print(f"The mean absolute error for predictions corresponding to test data is MAE = {np.mean(np.absolute(error))}")

    #----------------------- CHOOSE PLOTS -----------------------#
    c_element_plot = False
    c_vs_prediction_plot = True
    error_plot = True
    correlation_plot = False

    #----------------------- PLOT C-ELEMENTS ON THEIR OWN -----------------------#
    if c_element_plot:
        c_line = plt.figure(  )
        ax2 = c_line.add_subplot(1, 1, 1, projection='3d')  
        ax2.scatter(x1, x2, x3, cmap=cm.plasma, linewidth=0, antialiased=False)
        ax2.title.set_text('C elements')
        ax2.plot(x1, x2, x3, color='r')
        ax2.set_xlabel('C11')
        ax2.set_ylabel('C12')
        ax2.set_zlabel('C44')

    #----------------------- PLOT C-ELEMENTS vs PREDICTIONS -----------------------#
    if c_vs_prediction_plot:
        c_fig = plt.figure(  )
        plt.title('C-elements vs predicted RD', fontdict={'fontsize':14})
        plt.scatter(y_test_predictions, x1, label='C11 -> Prediction', marker='.', c="tomato")
        plt.scatter(y_test_predictions, x2, label='C12 -> Prediction', marker='.', c="steelblue")
        plt.scatter(y_test_predictions, x3, label='C44 -> Prediction', marker='.', c="seagreen")
        plt.scatter(y_test, x1, label='C11 -> Ground Truth', facecolors='none', edgecolors="coral")
        plt.scatter(y_test, x2, label='C12 -> Ground Truth', facecolors='none', edgecolors="cornflowerblue")
        plt.scatter(y_test, x3, label='C44 -> Ground Truth', facecolors='none', edgecolors="mediumseagreen")
        plt.ylabel('C elements', fontdict={'fontsize':14})
        plt.xlabel('Relative Density', fontdict={'fontsize':14}) 
        plt.legend()

    #----------------------- PLOT ERROR VS Relative Density -----------------------#
    if error_plot:
        error_fig = plt.figure( ) #figsize=plt.figaspect(2)
        plt.title('Error vs Relative Density', fontdict={'fontsize':14})
        plt.scatter(predictedRD, error, label='Prediction Error', marker='.', c="tomato")
        poly = np.polyfit(predictedRD, predictedRD-groundTruthRD, 1)
        p = np.poly1d(poly)
        plt.plot(predictedRD, p(predictedRD), label='Error Trend', c="steelblue")
        plt.ylabel('Error', fontdict={'fontsize':14})
        plt.xlabel('Ground Truth RD', fontdict={'fontsize':14})#, fontdict={'fontsize':20}) 
        # plt.legend()

    #----------------------- CORRELATION PLOT -----------------------#
    if correlation_plot:
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
        covarianceMatrix.set_title('Correlation Heatmap', fontdict={'fontsize':20}, pad=12)
        # sns.heatmap((corr_matrix), square=True, cmap="bone", vmin = 0.95, vmax = 1, annot=True)
    
    plt.show()
"""

if __name__ == '__main__':
	main()