import numpy as np
import pandas as pd
from numpy.linalg import inv
import math


# def projection(x, y, t):
#     """where y is the vector of years corresponding to the variable x which you want to project until t """
#     y = t - y
#     X = np.vstack([np.ones(len(y)), y]).T
#     p = inv(X.T @ X) @ X.T @ x
#     'Predicting now'
#     n = np.arange(y[-1]-1, -1, -1)
#     X1 = np.vstack([np.ones(len(n)), n]).T
#     Predicted = X1 @ p
#     print(p)
#     # # R-squared
#     # r2 = 1-np.linalg.norm(x - X @ p)**2/x.shape[0]/np.var(x)
#     # print(r2)
#     return Predicted

def projection(x, y, t):
    """where y is the vector of years corresponding to the variable x which you want to project until t """
    X = np.vstack([np.ones(len(y)), y]).T
    p = inv(X.T @ X) @ X.T @ x
    'Predicting now'
    n = np.arange(y[-1]+1, t+1)
    X1 = np.vstack([np.ones(len(n)), n]).T
    Predicted = X1 @ p
    # print(p)
    # R-squared
    r2 = 1-np.linalg.norm(x - X @ p)**2/x.shape[0]/np.var(x)
    # print(r2)
    return Predicted


def linear_param(y, x1, x2):
    X = np.array([x1, x2, np.ones(len(x1))]).T
    p = inv(X.T @ X) @ X.T @ y
    return p


def linear_model(x1, x2, p):
    X = np.array([x1, x2, np.ones(len(x1))]).T
    Y = X @ p
    return Y


def mse(x, y):
    return (1 - ((np.linalg.norm(y - x) ** 2 / y.shape) / np.var(y)))


def grey_degree(X, Y, Z):
    # Normalising the data for it to be fixed between 0 and 1 and to be dimensionless
    X_norm = X / np.max(X)
    Y_norm = Y / np.max(Y)
    Z_norm = Z / np.max(Z)
    # Calculating the absolute difference between the series
    diff = np.abs(X_norm - Y_norm)
    diff1 = np.abs(X_norm - Z_norm)

    # Calculating the grey relational degree from the grey relational coefficient
    # 0.5 is the distinguishing coefficient taken from literature
    grey_degree_gdp = np.mean((np.min(diff) + 0.5 * np.max(diff)) / (diff + 0.5 * np.max(diff)))
    grey_degree_pop = np.mean((np.min(diff1) + 0.5 * np.max(diff1)) / (diff1 + 0.5 * np.max(diff1)))
    return grey_degree_gdp, grey_degree_pop


def grey_degree1(X, Y, Z, D):
    # Normalising the data for it to be fixed between 0 and 1 and to be dimensionless
    X_norm = X / np.max(X)
    Y_norm = Y / np.max(Y)
    Z_norm = Z / np.max(Z)
    D_norm = D / np.max(D)
    # Calculating the absolute difference between the series
    diff = np.abs(X_norm - Y_norm)
    diff1 = np.abs(X_norm - Z_norm)
    diff2 = np.abs(X_norm - D_norm)

    # Calculating the grey relational degree from the grey relational coefficient
    # 0.5 is the distinguishing coefficient taken from literature
    grey_degree_gdp = np.mean((np.min(diff) + 0.5 * np.max(diff)) / (diff + 0.5 * np.max(diff)))
    grey_degree_pop = np.mean((np.min(diff1) + 0.5 * np.max(diff1)) / (diff1 + 0.5 * np.max(diff1)))
    grey_degree_cost = np.mean((np.min(diff2) + 0.5 * np.max(diff2)) / (diff2 + 0.5 * np.max(diff2)))
    print(f"The grey degree between water consumption and GDP is : {round(grey_degree_gdp, 3)}")
    print(f"The grey degree between water consumption and population is : {round(grey_degree_pop, 3)}")
    print(f"The grey degree between water consumption and cost per m3 is : {round(grey_degree_cost, 3)}")
    return grey_degree_gdp, grey_degree_pop, grey_degree_cost


def grey_degree2(X, Y):
    # Normalising the data for it to be fixed between 0 and 1 and to be dimensionless
    X_norm = X / np.max(X)
    Y_norm = Y / np.max(Y)
    # Calculating the absolute difference between the series
    diff = np.abs(X_norm - Y_norm)

    # Calculating the grey relational degree from the grey relational coefficient
    # 0.5 is the distinguishing coefficient taken from literature
    grey_degree_Y = np.round(np.mean((np.min(diff) + 0.5 * np.max(diff)) / (diff + 0.5 * np.max(diff))), 2)

    return grey_degree_Y


def AGMC_12(X, Y, w):
    t = len(X)  # time series for the data
    r = len(Y)
    X_norm = X / np.max(X)
    Y_norm = Y / np.max(Y)

    # X_norm = X
    # Y_norm = Y
    # Finding the accumulated sequence
    X11 = [X_norm[0]]
    X21 = [Y_norm[0]]

    for i in range(1, t):
        X11.append((w * X_norm[i - 1]) + X_norm[i])
    for i in range(1, r):
        X21.append((w * Y_norm[i - 1]) + Y_norm[i])

    X21t = X21[:t]

    # Finding the model parameters
    Y = []
    for i in range(1, t):
        Y.append(X11[i] - X11[i - 1])

    B4 = np.ones(t - 1)
    B1 = []
    B2 = []

    for i in range(1, t):
        B1.append(-0.5 * (X11[i - 1] + X11[i]))
        B2.append(0.5 * (X21t[i - 1] + X21t[i]))
    B = (np.array([B1, B2, B4])).T

    # u is the grey control parameter and b1, b2 and b3 are the grey developmental coefficient from the grey differential equation
    [b1, b2, u] = inv(B.T @ B) @ B.T @ Y

    # print(b1, b2, u)

    def f_h(b, x, u):
        return (b * x) + u

    F_H = []
    for h in range(0, r):
        F_H.append(f_h(b2, X21[h], u))
    # Finding the first order accumulation predictive sequence
    X11Apredict = [X_norm[0]]
    for t in range(2, r + 1):
        # if t-1 < 2:
        #     X11Apredict.append(X_norm[0]*math.exp(-b1*(t-1)) + (0.5*math.exp(-b1*(t-1))*F_H[0]) + 0.5*F_H[t-1])
        # else:
        ht = []
        for h in range(2, t):
            ht.append(math.exp(-b1 * (t - h)) * F_H[h - 1])
        htn = sum(ht)
        X11Apredict.append(
            X_norm[0] * math.exp(-b1 * (t - 1)) + (0.5 * math.exp(-b1 * (t - 1)) * F_H[0]) + 0.5 * F_H[t - 1] + htn)
    X11Apredict = np.array(X11Apredict) * np.max(X)
    # Finding the predicted sequence
    X11predict = [X_norm[0]]
    for i in range(1, r):
        X11predict.append(X11Apredict[i] - (w * X11Apredict[i - 1]))
    Predicted_water_consumption = np.array(X11predict) * np.max(X)
    return Predicted_water_consumption
    # return X11Apredict


def AGMC_12n(X, Y, w):
    t = len(X)  # time series for the data
    r = len(Y)

    # Finding the accumulated sequence
    X11 = [X[0]]
    X21 = [Y[0]]

    for i in range(1, t):
        X11.append((w * X[i - 1]) + X[i])
    for i in range(1, r):
        X21.append((w * Y[i - 1]) + Y[i])

    X21t = X21[:t]

    # Finding the model parameters
    Y = []
    for i in range(1, t):
        Y.append(X11[i] - X11[i - 1])

    B4 = np.ones(t - 1)
    B1 = []
    B2 = []

    for i in range(1, t):
        B1.append(-0.5 * (X11[i - 1] + X11[i]))
        B2.append(0.5 * (X21t[i - 1] + X21t[i]))
    B = (np.array([B1, B2, B4])).T

    # u is the grey control parameter and b1, b2 and b3 are the grey developmental coefficient from the grey differential equation
    [b1, b2, u] = inv(B.T @ B) @ B.T @ Y

    # print(b1, b2, u)

    def f_h(b, x, u):
        return (b * x) + u

    F_H = []
    for h in range(0, r):
        F_H.append(f_h(b2, X21[h], u))
    # Finding the first order accumulation predictive sequence
    X11Apredict = [X[0]]
    for t in range(2, r + 1):
        # if t-1 < 2:
        #     X11Apredict.append(X_norm[0]*math.exp(-b1*(t-1)) + (0.5*math.exp(-b1*(t-1))*F_H[0]) + 0.5*F_H[t-1])
        # else:
        ht = []
        for h in range(2, t):
            ht.append(math.exp(-b1 * (t - h)) * F_H[h - 1])
        htn = sum(ht)
        X11Apredict.append(
            X[0] * math.exp(-b1 * (t - 1)) + (0.5 * math.exp(-b1 * (t - 1)) * F_H[0]) + 0.5 * F_H[t - 1] + htn)
    # Finding the predicted sequence
    X11predict = [X[0]]
    for i in range(1, r):
        X11predict.append(X11Apredict[i] - (w * X11Apredict[i - 1]))
    return X11predict


def AGMC_12a(X, Y, w):
    t = len(X)  # time series for the data
    r = len(Y)

    # Finding the accumulated sequence
    X11 = [X[0]]
    X21 = [Y[0]]

    for i in range(1, t):
        X11.append((w * X[i - 1]) + X[i])
    for i in range(1, r):
        X21.append((w * Y[i - 1]) + Y[i])

    X21t = X21[:t]

    # Finding the model parameters
    Y = []
    for i in range(1, t):
        Y.append(X11[i] - X11[i - 1])

    B4 = np.ones(t - 1)
    B1 = []
    B2 = []

    for i in range(1, t):
        B1.append(-0.5 * (X11[i - 1] + X11[i]))
        B2.append(0.5 * (X21t[i - 1] + X21t[i]))
    B = (np.array([B1, B2, B4])).T

    # u is the grey control parameter and b1, b2 and b3 are the grey developmental coefficient from the grey differential equation
    [b1, b2, u] = inv(B.T @ B) @ B.T @ Y

    # print(b1, b2, u)

    def f_h(b, x, u):
        return (b * x) + u

    F_H = []
    for h in range(0, r):
        F_H.append(f_h(b2, X21[h], u))
    # Finding the first order accumulation predictive sequenceh
    X11Apredict = [X[0]]
    for t in range(2, r + 1):
        ht = []
        for h in range(2, t):
            ht.append(math.exp(-b1 * (t - h - 0.5)) * (F_H[h - 1] +F_H[h-2])/2)
        htn = sum(ht)
        X11Apredict.append(
            X[0] * math.exp(-b1 * (t - 1)) + htn)
    # Finding the predicted sequence
    X11predict = [X[0]]
    for i in range(1, r):
        X11predict.append(X11Apredict[i] - (w * X11Apredict[i - 1]))
    return X11predict
# def NGMC_1N(X,Y,Z,w):
#     t = len(X)  # time series for the data
#     r = len(Y)
#     # Finding the accumulated sequence
#     X11 = np.cumsum(X)
#     X21 = np.cumsum(Y)
#     X31 = np.cumsum(Z)
#     X21t = X21[:t]
#     X31t = X31[:t]
#     Z11 = []
#     for i in range(t):
#         Z11.append(0.5*X11[i+1] + 0.5*X11[i])
#     H = np.array([Z11,np.ones(t-1),range(2,t+1),X21[1:t], X31[1:t])


def AGMC_13(X, Y, Z, w):
    t = len(X)  # time series for the data
    r = len(Y)
    X_norm = X / np.max(X)
    Y_norm = Y / np.max(Y)
    Z_norm = Z / np.max(Z)
    # Finding the accumulated sequence
    X11 = [X_norm[0]]
    X21 = [Y_norm[0]]
    X31 = [Z_norm[0]]
    for i in range(1, t):
        X11.append((w * X_norm[i - 1]) + X_norm[i])
    for i in range(1, r):
        X21.append((w * Y_norm[i - 1]) + Y_norm[i])
        X31.append((w * Z_norm[i - 1]) + Z_norm[i])
    X21t = X21[:t]
    X31t = X31[:t]
    # Finding the model parameters
    Y = []
    for i in range(1, t):
        Y.append(X11[i] - X11[i - 1])

    B4 = np.ones(t - 1)
    B1 = []
    B2 = []
    B3 = []
    for i in range(1, t):
        B1.append(-0.5 * (X11[i - 1] + X11[i]))
        B2.append(0.5 * (X21t[i - 1] + X21t[i]))
        B3.append(0.5 * (X31t[i - 1] + X31t[i]))
    B = (np.array([B1, B2, B3, B4])).T

    # u is the grey control parameter and b1, b2 and b3 are the grey developmental coefficient from the grey differential equation
    [b1, b2, b3, u] = inv(B.T @ B) @ B.T @ Y

    # print(b1, b2,b3, u)

    def f_h(b1, b2, x1, x2, u):
        return (b1 * x1) + (b2 * x2) + u

    F_H = []
    for h in range(0, r):
        F_H.append(f_h(b2, b3, X21[h], X31[h], u))
    # Finding the first order accumulation predictive sequence
    X11Apredict = [X_norm[0]]
    for t in range(2, r + 1):
        # if t-1 < 2:
        #     X11Apredict.append(X_norm[0]*math.exp(-b1*(t-1)) + (0.5*math.exp(-b1*(t-1))*F_H[0]) + 0.5*F_H[t-1])
        # else:
        ht = []
        for h in range(2, t):
            ht.append(math.exp(-b1 * (t - h)) * F_H[h - 1])
        htn = sum(ht)
        X11Apredict.append(
            X_norm[0] * math.exp(-b1 * (t - 1)) + (0.5 * math.exp(-b1 * (t - 1)) * F_H[0]) + 0.5 * F_H[t - 1] + htn)

    # Finding the predicted sequence
    X11predict = [X_norm[0]]
    for i in range(1, r):
        X11predict.append(X11Apredict[i] - (w * X11Apredict[i - 1]))
    Predicted_water_consumption = np.array(X11predict) * np.max(X)
    return Predicted_water_consumption


def GMC_12(X, Y):
    t = len(X)  # time series for the data
    r = len(Y)

    # Finding the accumulated sequence
    X11 = np.cumsum(X)
    X21 = np.cumsum(Y)
    X21t = X21[:t]

    # Finding the model parameters
    Y = []
    for i in range(1, t):
        Y.append(X11[i] - X11[i - 1])

    B4 = np.ones(t - 1)
    B1 = []
    B2 = []

    for i in range(1, t):
        B1.append(-0.5 * (X11[i - 1] + X11[i]))
        B2.append(0.5 * (X21t[i - 1] + X21t[i]))
    B = (np.array([B1, B2, B4])).T

    # u is the grey control parameter and b1, b2 and b3 are the grey developmental coefficient from the grey differential equation
    [b1, b2, u] = inv(B.T @ B) @ B.T @ Y

    # print(b1, b2, u)

    def f_h(b, x, u):
        return (b * x) + u

    F_H = []
    for h in range(0, r):
        F_H.append(f_h(b2, X21[h], u))
    # Finding the first order accumulation predictive sequence
    X11Apredict = [X[0]]
    for t in range(2, r + 1):
        # if t-1 < 2:
        #     X11Apredict.append(X[0]*math.exp(-b1*(t-1)) + (0.5*math.exp(-b1*(t-1))*F_H[0]) + 0.5*F_H[t-1])
        # else:
        ht = []
        for h in range(2, t):
            ht.append(math.exp(-b1 * (t - h)) * F_H[h - 1])
        htn = sum(ht)
        X11Apredict.append(
            X[0] * math.exp(-b1 * (t - 1)) + (0.5 * math.exp(-b1 * (t - 1)) * F_H[0]) + 0.5 * F_H[t - 1] + htn)

    # Finding the predicted sequence
    X11predict = [X[0]]
    for i in range(1, r):
        X11predict.append(X11Apredict[i] - (X11Apredict[i - 1]))

    return X11predict


def GMC_13(X, Y, Z):
    t = len(X)  # time series for the data
    r = len(Y)

    # Finding the accumulated sequence
    X11 = np.cumsum(X)
    X21 = np.cumsum(Y)
    X31 = np.cumsum(Z)
    X21t = X21[:t]
    X31t = X31[:t]

    # Finding the model parameters
    Y = []
    for i in range(1, t):
        Y.append(X11[i] - X11[i - 1])

    B4 = np.ones(t - 1)
    B1 = []
    B2 = []
    B3 = []
    for i in range(1, t):
        B1.append(-0.5 * (X11[i - 1] + X11[i]))
        B2.append(0.5 * (X21t[i - 1] + X21t[i]))
        B3.append(0.5 * (X31t[i - 1] + X31t[i]))
    B = (np.array([B1, B2, B3, B4])).T

    # u is the grey control parameter and b1, b2 and b3 are the grey developmental coefficient from the grey differential equation
    [b1, b2, b3, u] = inv(B.T @ B) @ B.T @ Y

    # print(b1, b2, u)

    def f_h(b1, b2, x1, x2, u):
        return (b1 * x1) + (b2 * x2) + u

    F_H = []
    for h in range(0, r):
        F_H.append(f_h(b2, b3, X21[h], X31[h], u))
    # Finding the first order accumulation predictive sequence
    X11Apredict = [X[0]]
    for t in range(2, r + 1):
        # if t-1 < 2:
        #     X11Apredict.append(X[0]*math.exp(-b1*(t-1)) + (0.5*math.exp(-b1*(t-1))*F_H[0]) + 0.5*F_H[t-1])
        # else:
        ht = []
        for h in range(2, t):
            ht.append(math.exp(-b1 * (t - h)) * F_H[h - 1])
        htn = sum(ht)
        X11Apredict.append(
            X[0] * math.exp(-b1 * (t - 1)) + (0.5 * math.exp(-b1 * (t - 1)) * F_H[0]) + 0.5 * F_H[t - 1] + htn)

    # Finding the predicted sequence
    X11predict = [X[0]]
    for i in range(1, r):
        X11predict.append(X11Apredict[i] - (X11Apredict[i - 1]))
    return X11predict


def MAPE(X1, XP):
    # Calculating the error
    r = len(XP)
    MAPE1 = []
    for i in range(0, r):
        MAPE1.append(np.abs((XP[i] - X1[i]) / X1[i]))
    return 100 * np.mean(MAPE1)


# import tensorflow as tf
# from tensorflow import keras
# import numpy as np
# from sklearn.metrics import r2_score
# from keras.models import Sequential
# from keras.layers import Dense
# from sklearn.model_selection import train_test_split
#
#
# def ANN(A, Y, Z, epochs):
#     # Normalizing the data
#     y = A / np.max(A)
#     X = np.array([Y / max(Y), Z / np.max(Z)]).T
#     X_new = X[0:len(y), :]
#     # Splitting the data into train and test
#     X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.25, random_state=42)
#
#     # Defining the model
#     model = Sequential()
#     model.add(Dense(16, input_shape=(2,), activation='relu'))
#     model.add(Dense(16, activation='relu'))
#     model.add(Dense(16, activation='relu'))
#     model.add(Dense(16, activation='relu'))
#     model.add(Dense(8, activation='relu'))
#     model.add(Dense(1))
#
#     # Compiling the model
#     model.compile(loss='mean_squared_error', optimizer='adam')
#
#     # Training the model
#     model.fit(X_train, y_train, epochs=epochs)
#
#     # from matplotlib.pyplot import figure, show
#     # data_loss = pd.DataFrame(model.history.history)
#     # axes = figure(figsize =(12, 6)).add_subplot()
#     # axes.plot(data_loss)
#     # axes.set_xlabel('Epochs')
#     # axes.set_ylabel('Loss')
#     # axes.set_title('Loss Vs Epochs')
#     # show()
#
#     # Predicting using the model
#     model_predict = (model.predict(X)) * np.max(A)
#     model_predict = model_predict.tolist()
#     return model_predict
