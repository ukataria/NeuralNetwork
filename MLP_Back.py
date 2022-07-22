import pandas as pd
import math
#reading the data stored in the spreadsheets, storing bias in 1D lists, weights in 2D lists
b1Excel = pd.read_excel("b1.xlsx")
b1 = b1Excel.iloc[:, 0]
b2Excel = pd.read_excel("b2.xlsx")
b2 = b2Excel.iloc[:, 0]
w1Excel = pd.read_excel("w1.xlsx")
w1 = []
for i in range(0, w1Excel.shape[1]):
    w1.append(w1Excel.iloc[:, i])
w2Excel = pd.read_excel("w2.xlsx")
w2 = []
for i in range(0, w2Excel.shape[1]):
    w2.append(w2Excel.iloc[:, i])
cross_data = pd.read_excel("cross_data.xlsx")
training = []
for i in range(0, cross_data.shape[1]):
    training.append(cross_data.iloc[0:220, i])
inputNum = w1Excel.shape[1]
outputNum = w2Excel.shape[0]
hiddenNum = w1Excel.shape[0]
#training set
#change end value to len(training[0]) once program is complete
for i in range(0, 1):
    input = []
    for j in range(0, inputNum):
        input.append(training[j][0])
    target = []
    for j in range(w1Excel.shape[1], cross_data.shape[1]):
        target.append(training[j][0])
    #forward propagation
    y1 = []
    for j in range(0, hiddenNum):
        v = 0.0
        for k in range(0, len(input)):
            v += w1[k][j] * input[k]
        y1.append(1 / (1 + math.e ** (-1 * v)))
    y2 = []
    for j in range(0, outputNum):
        v = 0.0
        for k in range(0, len(y1)):
            v += w2[k][j] * y1[k]
        y2.append(1 / (1 + math.e ** (-1 * v)))
    #calculating error
    squareError = 0.0
    for j in range(0, outputNum):
        squareError += (target[j] - y2[j]) ** 2.0
    totalError = (1 / outputNum) * squareError
    #backpropagation
    