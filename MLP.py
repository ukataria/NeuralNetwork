import pandas as pd
from tabulate import tabulate
from decimal import Decimal
import math
alpha = 0.7
beta = 0.3
#reading the data stored in the spreadsheets, storing bias in 1D lists, weights in 2D lists
b1Excel = pd.read_excel("b1.xlsx")
b1 = b1Excel.iloc[:, 0]
b1Old = []
for i in b1:
    b1Old.append(i)
b2Excel = pd.read_excel("b2.xlsx")
b2 = b2Excel.iloc[:, 0]
b2Old = []
for i in b2:
    b2Old.append(i)
w1Excel = pd.read_excel("w1.xlsx")
w1 = []
for i in range(0, w1Excel.shape[1]):
    w1.append(w1Excel.iloc[:, i])
w1Old = []
for i in w1:
    temp = []
    for j in i:
        temp.append(j)
    w1Old.append(temp)
w2Excel = pd.read_excel("w2.xlsx")
w2 = []
for i in range(0, w2Excel.shape[1]):
    w2.append(w2Excel.iloc[:, i])
w2Old = []
for i in w2:
    temp = []
    for j in i:
        temp.append(j)
    w2Old.append(temp)
cross_data = pd.read_excel("cross_data.xlsx")
training = []
for i in range(0, cross_data.shape[1]):
    training.append(cross_data.iloc[:, i])
inputNum = w1Excel.shape[1]
outputNum = w2Excel.shape[0]
hiddenNum = w1Excel.shape[0]
#training set
#change end value to len(training[0]) once program is complete
sumSquaredErrors = 0.0
for i in range(0, len(training[0])):
    input = []
    for j in range(0, inputNum):
        input.append(training[j][i])
    target = []
    for j in range(w1Excel.shape[1], cross_data.shape[1]):
        target.append(training[j][i])
    #forward propagation
    y1 = []
    for j in range(0, hiddenNum):
        v = 0.0
        for k in range(0, len(input)):
            v += w1[k][j] * input[k]
        v += b1[j]
        y1.append(1 / (1 + math.e ** (-1 * v)))
    y2 = []
    for j in range(0, outputNum):
        v = 0.0
        for k in range(0, len(y1)):
            v += w2[k][j] * y1[k]
        v += b2[j]
        y2.append(1 / (1 + math.e ** (-1 * v)))
    #calculating error
    squareError = 0.0
    for j in range(0, outputNum):
        squareError += (target[j] - y2[j]) ** 2.0
    totalError = (1 / outputNum) * squareError
    sumSquaredErrors += totalError
    #backpropagation outer layer
    for j in range(0, outputNum):
        localGradient = (target[j] - y2[j]) * y2[j] * (1 - y2[j])
        #weights
        for k in range(0, hiddenNum):
            tempW = w2[k][j]
            w2[k][j] = w2[k][j] + beta * (w2[k][j] - w2Old[k][j]) + alpha * localGradient * y1[k]
            w2Old[k][j] = tempW
        #biases
        tempB = b2[j]
        b2[j] = b2[j] + beta * (b2[j] - b2Old[j]) + alpha * localGradient * 1
        b2Old[j] = tempB
    #backpropagation hidden layer
    for j in range(0, hiddenNum):
        localGradient = y1[j] * (1 - y1[j])
        sigmaGradientWeight = 0.0
        for k in range(0, outputNum):
            sigmaGradientWeight += (target[k] - y2[k]) * y2[k] * (1 - y2[k]) * w2Old[j][k]
        localGradient *= sigmaGradientWeight
        #weights
        for k in range(0, inputNum):
            tempW = w1[k][j]
            w1[k][j] = w1[k][j] + beta * (w1[k][j] - w1Old[k][j]) + alpha * localGradient * input[k]
            w1Old[k][j] = tempW
        #biases
        tempB = b1[j]
        b1[j] = b1[j] + beta * (b1[j] - b1Old[j]) + alpha * localGradient * 1
        b1Old[j] = tempB
    #output after 1 input
    print("Epoch 1 Training (" + str(i + 1) + " / " + str(len(training[0])) + "): " + "{0:7.4f}".format(totalError))
print("--------------------------------------------------------------------------------------------------------")
#organizing w1 and w2 tables and rounding:
w1Final = []
for i in range(0, hiddenNum):
    temp = []
    for j in range(0, inputNum):
        temp.append("{0:7.4f}".format(w1[j][i]))
    w1Final.append(temp)
w2Final = []
for i in range(0, outputNum):
    temp = []
    for j in range(0, hiddenNum):
        temp.append("{0:7.4f}".format(w2[j][i]))
    w2Final.append(temp)
#output of weights, biases, and MSE after 1 epoch
print("W1 Table: " + "(from " + str(inputNum) + " inputs to " + str(hiddenNum) + " hidden nodes)")
print(tabulate(w1Final, floatfmt=(".4f")))
print("B1 Table: (" + str(hiddenNum) + " hidden nodes)")
for i in b1:
    print("{0:7.4f}".format(i))
print("W2 Table: " + "(from " + str(hiddenNum) + " hidden nodes to " + str(outputNum) + " outputs)")
print(tabulate(w2Final, floatfmt=(".4f")))
print("B2 Table: (" + str(outputNum) + " outputs)")
for i in b2:
    print("{0:7.4f}".format(i))
print("MSE after epoch 1: " + "{0:7.4f}".format(sumSquaredErrors / len(training[0]), 4))