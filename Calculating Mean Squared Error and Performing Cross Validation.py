#David Cordero
#Data Mining Assignment 1
#MSE and Cross Validation of Data Sets

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

#import all of the data sets
train_100_10 = pd.read_csv('C:/Users/dcord/Downloads/train-100-10.csv')
train_100_100 = pd.read_csv('C:/Users/dcord/Downloads/train-100-100.csv')
train_1000_100 = pd.read_csv('C:/Users/dcord/Downloads/train-1000-100.csv')

train_50_1000_100 = pd.read_csv('C:/Users/dcord/Downloads/train-50(1000)-100.csv')
train_100_1000_100 = pd.read_csv('C:/Users/dcord/Downloads/train-100(1000)-100.csv')
train_150_1000_100 = pd.read_csv('C:/Users/dcord/Downloads/train-150(1000)-100.csv')

test_100_10 = pd.read_csv('C:/Users/dcord/Downloads/test-100-10.csv')
test_100_100 = pd.read_csv('C:/Users/dcord/Downloads/test-100-100.csv')
test_1000_100 = pd.read_csv('C:/Users/dcord/Downloads/test-1000-100.csv')


#functions!!!
def calc_coeffs(xtrain, ytrain):
    coef_list = []
    n, m = xtrain.shape
    lamdavals = np.arange(0,151,1)

    for lambda_val in  lamdavals:
        I = np.identity(xtrain.shape[1])
        coefficients = np.linalg.inv(xtrain.T @ xtrain + lambda_val * I) @ xtrain.T @ ytrain
        coef_list.append(coefficients)

    return coef_list

def calc_mse(X, Y, coef_list):
    mse_list = []

    for coefficients in coef_list:
        Y_pred = X @ coefficients
        mse = np.mean((Y_pred - Y) ** 2)
        mse_list.append(mse)

    return mse_list


#question 2, printing mse's!!!
print("Question 2 - Printing MSE per Lambda ")

#set 1
dataset1_train = np.genfromtxt("C:/Users/dcord/Downloads/train-100-10.csv", delimiter = ',', skip_header = 1, dtype = float)

xtrain1 = dataset1_train[:, :-3]
ytrain1 = dataset1_train[:, -3]

one1 = np.ones((len(xtrain1), 1))
xtrain1 = np.hstack((one1, xtrain1))
ytrain1 = ytrain1.reshape(-1, 1)

dataset1_test = np.genfromtxt("C:/Users/dcord/Downloads/test-100-10.csv", delimiter=',', skip_header=1, dtype = float)
xtest1 = dataset1_test[:, :-1]
ytest1 = dataset1_test[:,-1]

one1 = np.ones((len(xtest1), 1))
xtest1 = np.hstack((one1, xtest1))
ytest1 = ytest1.reshape(-1, 1)

coef_list1 = calc_coeffs(xtrain1, ytrain1)
train_mse1= calc_mse(xtrain1, ytrain1, coef_list1)
test_mse1 = calc_mse(xtest1, ytest1, coef_list1)

min1 = min(test_mse1)
min_pos1 = [index for index, value in enumerate(test_mse1) if value == min1]
print("100-10: Lambda value:", min_pos1, "MSE Value:", min1)


#set 2
dataset2_train = np.genfromtxt("C:/Users/dcord/Downloads/train-100-100.csv", delimiter = ',', skip_header = 1, dtype = float)

xtrain2 = dataset2_train[:, :-1]
ytrain2 = dataset2_train[:, -1]

one2 = np.ones((len(xtrain2), 1))
xtrain2 = np.hstack((one2, xtrain2))
ytrain2 = ytrain2.reshape(-1, 1)

dataset2_test = np.genfromtxt("C:/Users/dcord/Downloads/test-100-100.csv", delimiter = ',', skip_header = 1, dtype = float)
xtest2 = dataset2_test[:, :-1]
ytest2 = dataset2_test[:, -1]

one2 = np.ones((len(xtest2), 1))
xtest2 = np.hstack((one2, xtest2))
ytest2 = ytest2.reshape(-1, 1)

coef_list2 = calc_coeffs(xtrain2, ytrain2)
train_mse2= calc_mse(xtrain2, ytrain2, coef_list2)
test_mse2 = calc_mse(xtest2, ytest2, coef_list2)

min_value2 = min(test_mse2)
min_pos2 = [index for index, value in enumerate(test_mse2) if value == min_value2]
print("100-100:  Lambda value:", min_pos2, "MSE Value:", min_value2)

#set 3
dataset3_train = np.genfromtxt("C:/Users/dcord/Downloads/train-1000-100.csv", delimiter=',', skip_header=1, dtype = float)
xtrain3 = dataset3_train[:, :-1]
ytrain3 = dataset3_train[:, -1]

one3 = np.ones((len(xtrain3), 1))
xtrain3 = np.hstack((one3, xtrain3))
ytrain3 = ytrain3.reshape(-1, 1)

dataset3_test = np.genfromtxt("C:/Users/dcord/Downloads/test-1000-100.csv", delimiter = ',', skip_header = 1, dtype = float)
xtest3 = dataset3_test[:, :-1]
ytest3 = dataset3_test[:, -1]

one3 = np.ones((len(xtest3), 1))
xtest3 = np.hstack((one3, xtest3))
ytest3 = ytest3.reshape(-1, 1)

coef_list3 = calc_coeffs(xtrain3, ytrain3)
train_mse3= calc_mse(xtrain3, ytrain3, coef_list3)
test_mse3 = calc_mse(xtest3, ytest3, coef_list3)

min_value3 = min(test_mse3)
min_pos3 = [index for index, value in enumerate(test_mse3) if value == min_value3]
print("1000-100:  Lambda value:", min_pos3, "MSE Value:", min_value3)


#set 4
dataset4_train = np.genfromtxt("C:/Users/dcord/Downloads/train-50(1000)-100.csv", delimiter=',', skip_header=1, dtype = float)
xtrain4 = dataset4_train[:, :-1]
ytrain4 = dataset4_train[:, -1]

one4 = np.ones((len(xtrain4), 1))
xtrain4 = np.hstack((one4, xtrain4))
ytrain4 = ytrain4.reshape(-1, 1)

dataset4_test = np.genfromtxt("C:/Users/dcord/Downloads/test-1000-100.csv", delimiter=',',skip_header=1, dtype = float)
xtest4 = dataset4_test[:, :-1]
ytest4 = dataset4_test[:, -1]

one4 = np.ones((len(xtest4), 1))
xtest4 = np.hstack((one4, xtest4))
ytest4 = ytest4.reshape(-1, 1)

coef_list4 = calc_coeffs(xtrain4, ytrain4)
train_mse4= calc_mse(xtrain4, ytrain4, coef_list4)
test_mse4 = calc_mse(xtest4, ytest4, coef_list4)

min_value4 = min(test_mse4)
min_pos4 = [index for index, value in enumerate(test_mse4) if value == min_value4]
print("50(1000)-100:  Lambda value:", min_pos4, "MSE Value:", min_value4)


#set 5
dataset5_train = np.genfromtxt("C:/Users/dcord/Downloads/train-100(1000)-100.csv", delimiter=',', skip_header=1, dtype = float)
xtrain5 = dataset5_train[:, :-1]
ytrain5 = dataset5_train[:, -1]

one5 = np.ones((len(xtrain5), 1))
xtrain5 = np.hstack((one5, xtrain5))
ytrain5 = ytrain5.reshape(-1, 1)

dataset5_test = np.genfromtxt("C:/Users/dcord/Downloads/test-1000-100.csv", delimiter=',', skip_header=1, dtype = float)
xtest5 = dataset5_test[:, :-1]
ytest5 = dataset5_test[:, -1]

one5 = np.ones((len(xtest5), 1))
xtest5 = np.hstack((one5, xtest5))
ytest5 = ytest5.reshape(-1, 1)

coef_list5 = calc_coeffs(xtrain5, ytrain5)
train_mse5= calc_mse(xtrain5, ytrain5, coef_list5)
test_mse5 = calc_mse(xtest5, ytest5, coef_list5)

min5 = min(test_mse5)
min_pos5 = [index for index, value in enumerate(test_mse5) if value == min5]

print("100(1000)-100:  Lambda value:", min_pos5, "MSE Value:", min5)

#set 6
dataset6_train = np.genfromtxt("C:/Users/dcord/Downloads/train-150(1000)-100.csv", delimiter=',', skip_header=1, dtype = float)
xtrain6 = dataset6_train[:, :-1]
ytrain6 = dataset6_train[:, -1]

one6 = np.ones((len(xtrain6), 1))
xtrain6 = np.hstack((one6, xtrain6))
ytrain6 = ytrain6.reshape(-1, 1)

dataset6_test = np.genfromtxt("C:/Users/dcord/Downloads/test-1000-100.csv", delimiter=',', skip_header=1, dtype = float)
xtest6 = dataset6_test[:, :-1]
ytest6 = dataset6_test[:, -1]

one6 = np.ones((len(xtest6), 1))
xtest6 = np.hstack((one6, xtest6))
ytest6 = ytest6.reshape(-1, 1)

coef_list6 = calc_coeffs(xtrain6, ytrain6)
train_mse6= calc_mse(xtrain6, ytrain6, coef_list6)
test_mse6 = calc_mse(xtest6, ytest6, coef_list6)

min6 = min(test_mse6)
min_pos6 = [index for index, value in enumerate(test_mse6) if value == min6]
print("150(1000)-100:  Lambda value:", min_pos6, "MSE Value:", min6)


#question 3, cross validation!!!
print("Question 3 - Cross Validation: ")

#set 1
xtrain = xtrain1
ytrain = ytrain1
lamdavals = np.arange(0, 151)
n = 10

train_mse = []
test_mse = []

fold_size = math.floor(len(xtrain) / n)

for i in range(n):
    starting_index = i * fold_size
    ending_index = (i + 1) * fold_size if i < n - 1 else len(xtrain)
    
    xtrain_fold = np.r_[xtrain[:starting_index], xtrain[ending_index:]]
    ytrain_fold = np.r_[ytrain[:starting_index], ytrain[ending_index:]]
    
    Xval_fold = xtrain[starting_index:ending_index]
    yval_fold = ytrain[starting_index:ending_index]
    
    coef_list = calc_coeffs(xtrain_fold, ytrain_fold)
    
    train_mse_fold, val_mse_fold = calc_mse(xtrain_fold, ytrain_fold, coef_list), calc_mse(Xval_fold, yval_fold, coef_list)
    train_mse.append(train_mse_fold)
    test_mse.append(val_mse_fold)

avg_train_mse = np.mean(train_mse, axis=0)
avg_test_mse = np.mean(test_mse, axis=0)

top_lambda_i = np.argmin(avg_test_mse)
top_lambda =  lamdavals[top_lambda_i]
top_test_mse = avg_test_mse[top_lambda_i]

print("C-V 100-10:  Lambda value:", top_lambda, "MSE Value:", top_test_mse)

#set 2
xtrain = xtrain2
ytrain = ytrain2
lamdavals = np.arange(0, 151)
n = 10

train_mse = []
test_mse = []

fold_size = math.floor(len(xtrain) / n)

for i in range(n):
    starting_index = i * fold_size
    ending_index = (i + 1) * fold_size if i < n - 1 else len(xtrain)
    
    xtrain_fold = np.r_[xtrain[:starting_index], xtrain[ending_index:]]
    ytrain_fold = np.r_[ytrain[:starting_index], ytrain[ending_index:]]

    Xval_fold = xtrain[starting_index:ending_index]
    yval_fold = ytrain[starting_index:ending_index]
    
    coef_list = calc_coeffs(xtrain_fold, ytrain_fold)
    
    train_mse_fold, val_mse_fold = calc_mse(xtrain_fold, ytrain_fold, coef_list), calc_mse(Xval_fold, yval_fold, coef_list)
    train_mse.append(train_mse_fold)
    test_mse.append(val_mse_fold)

avg_train_mse = np.mean(train_mse, axis=0)
avg_test_mse = np.mean(test_mse, axis=0)

top_lambda_i = np.argmin(avg_test_mse)
top_lambda =  lamdavals[top_lambda_i]
top_test_mse = avg_test_mse[top_lambda_i]

print("C-V 100-100:  Lambda value:", top_lambda, "MSE Value:", top_test_mse)

#set 3
xtrain = xtrain3
ytrain = ytrain3
lamdavals = np.arange(0, 151)
n = 10

train_mse = []
test_mse = []

fold_size = math.floor(len(xtrain) / n)

for i in range(n):
    starting_index = i * fold_size
    ending_index = (i + 1) * fold_size if i < n - 1 else len(xtrain)
    
    xtrain_fold = np.r_[xtrain[:starting_index], xtrain[ending_index:]]
    ytrain_fold = np.r_[ytrain[:starting_index], ytrain[ending_index:]]

    Xval_fold = xtrain[starting_index:ending_index]
    yval_fold = ytrain[starting_index:ending_index]

    
    Xval_fold = xtrain[starting_index:ending_index]
    yval_fold = ytrain[starting_index:ending_index]
    
    coef_list = calc_coeffs(xtrain_fold, ytrain_fold)
    
    train_mse_fold, val_mse_fold = calc_mse(xtrain_fold, ytrain_fold, coef_list), calc_mse(Xval_fold, yval_fold, coef_list)
    train_mse.append(train_mse_fold)
    coef_list = calc_coeffs(xtrain_fold, ytrain_fold)
    
    train_mse_fold, val_mse_fold = calc_mse(xtrain_fold, ytrain_fold, coef_list), calc_mse(Xval_fold, yval_fold, coef_list)
    train_mse.append(train_mse_fold)
    test_mse.append(val_mse_fold)

avg_train_mse = np.mean(train_mse, axis=0)
avg_test_mse = np.mean(test_mse, axis=0)

top_lambda_i = np.argmin(avg_test_mse)
top_lambda =  lamdavals[top_lambda_i]
top_test_mse = avg_test_mse[top_lambda_i]

print("C-V 1000-100:  Lambda value:", top_lambda, "MSE Value:", top_test_mse)

#set 4
xtrain = xtrain4
ytrain = ytrain4
lamdavals = np.arange(0, 151)
n = 10

train_mse = []
test_mse = []

fold_size = math.floor(len(xtrain) / n)

for i in range(n):
    starting_index = i * fold_size
    ending_index = (i + 1) * fold_size if i < n - 1 else len(xtrain)
    
    xtrain_fold = np.r_[xtrain[:starting_index], xtrain[ending_index:]]
    ytrain_fold = np.r_[ytrain[:starting_index], ytrain[ending_index:]]
    
    Xval_fold = xtrain[starting_index:ending_index]
    yval_fold = ytrain[starting_index:ending_index]
    
    coef_list = calc_coeffs(xtrain_fold, ytrain_fold)
    
    train_mse_fold, val_mse_fold = calc_mse(xtrain_fold, ytrain_fold, coef_list), calc_mse(Xval_fold, yval_fold, coef_list)
    train_mse.append(train_mse_fold)
    test_mse.append(val_mse_fold)

avg_train_mse = np.mean(train_mse, axis=0)
avg_test_mse = np.mean(test_mse, axis=0)

top_lambda_i = np.argmin(avg_test_mse)
top_lambda =  lamdavals[top_lambda_i]
top_test_mse = avg_test_mse[top_lambda_i]

print("C-V 50(1000)-100:  Lambda value:", top_lambda, "MSE Value:", top_test_mse)

#set 5
xtrain = xtrain5
ytrain = ytrain5
lamdavals = np.arange(0, 151)
n = 10

train_mse = []
test_mse = []

fold_size = math.floor(len(xtrain) / n)

for i in range(n):
    starting_index = i * fold_size
    ending_index = (i + 1) * fold_size if i < n - 1 else len(xtrain)
    
    xtrain_fold = np.r_[xtrain[:starting_index], xtrain[ending_index:]]
    ytrain_fold = np.r_[ytrain[:starting_index], ytrain[ending_index:]]
    
    Xval_fold = xtrain[starting_index:ending_index]
    yval_fold = ytrain[starting_index:ending_index]
    
    coef_list = calc_coeffs(xtrain_fold, ytrain_fold)
    
    train_mse_fold, val_mse_fold = calc_mse(xtrain_fold, ytrain_fold, coef_list), calc_mse(Xval_fold, yval_fold, coef_list)
    train_mse.append(train_mse_fold)
    test_mse.append(val_mse_fold)

avg_train_mse = np.mean(train_mse, axis=0)
avg_test_mse = np.mean(test_mse, axis=0)

top_lambda_i = np.argmin(avg_test_mse)
top_lambda =  lamdavals[top_lambda_i]
top_test_mse = avg_test_mse[top_lambda_i]

print("C-V 100(1000)-100:  Lambda value:", top_lambda, "MSE Value:", top_test_mse)

#set 6
xtrain = xtrain6
ytrain = ytrain6
lamdavals = np.arange(0, 151)
n = 10

train_mse = []
test_mse = []

fold_size = math.floor(len(xtrain) / n)

for i in range(n):
    starting_index = i * fold_size
    ending_index = (i + 1) * fold_size if i < n - 1 else len(xtrain)
    
    xtrain_fold = np.r_[xtrain[:starting_index], xtrain[ending_index:]]
    ytrain_fold = np.r_[ytrain[:starting_index], ytrain[ending_index:]]
    
    Xval_fold = xtrain[starting_index:ending_index]
    yval_fold = ytrain[starting_index:ending_index]
    
    coef_list = calc_coeffs(xtrain_fold, ytrain_fold)
    
    train_mse_fold, val_mse_fold = calc_mse(xtrain_fold, ytrain_fold, coef_list), calc_mse(Xval_fold, yval_fold, coef_list)
    train_mse.append(train_mse_fold)
    test_mse.append(val_mse_fold)

avg_train_mse = np.mean(train_mse, axis=0)
avg_test_mse = np.mean(test_mse, axis=0)

top_lambda_i = np.argmin(avg_test_mse)
top_lambda =  lamdavals[top_lambda_i]
top_test_mse = avg_test_mse[top_lambda_i]

print("C-V 150(1000)-100:  Lambda value:", top_lambda, "MSE Value:", top_test_mse)

#graphs!!!
#graph1
lamdavals = list(np.arange(0,151))

x =  lamdavals
y1 = train_mse1
y2 = test_mse1

plt.plot(x, y1, label="Training", color="green")
plt.plot(x, y2, label="Testing", color="blue")

plt.xlabel("Lambda")
plt.ylabel("MSE Values")
plt.title("100-10")
plt.legend()

plt.show()

#graph2 
x2 =  lamdavals
y1_2 = train_mse2
y2_2 = test_mse2

plt.plot(x2, y1_2, label="Training", color="green")
plt.plot(x2, y2_2, label="Testing", color="blue")

plt.xlabel("Lambda")
plt.ylabel("MSE Values")
plt.title("100-100")
plt.legend()

plt.show()

#graph 3
x3 =  lamdavals
y1_3 = train_mse3
y2_3 = test_mse3

plt.plot(x3, y1_3, label="Training", color="green")
plt.plot(x3, y2_3, label="Testing", color="blue")

plt.xlabel("Lambda")
plt.ylabel("MSE Values")
plt.title("1000-100")
plt.legend()

plt.show()

#graph 4
x4 =  lamdavals
y1_4 = train_mse4
y2_4 = test_mse4

plt.plot(x4, y1_4, label="Training", color="green")
plt.plot(x4, y2_4, label="Testing", color="blue")

plt.xlabel("Lambda")
plt.ylabel("MSE Values")
plt.title("50(1000)-100")
plt.legend()

plt.show()

#graph 5
x5 =  lamdavals
y1_5 = train_mse5
y2_5 = test_mse5

plt.plot(x5, y1_5, label="Training", color="green")
plt.plot(x5, y2_5, label="Testing", color="blue")

plt.xlabel("Lambda")
plt.ylabel("MSE Values")
plt.title("100(1000)-100")
plt.legend()

plt.show()

#graph 6
x6 =  lamdavals
y1_6 = train_mse6
y2_6 = test_mse6

plt.plot(x6, y1_6, label="Training", color="green")
plt.plot(x6, y2_6, label="Testing", color="blue")

plt.xlabel("Lambda")
plt.ylabel("MSE Values")
plt.title("150(1000)-100")
plt.legend()

plt.show()

#graph 7
#set 2 wo lambda 0
lamdavals2 =  lamdavals[1:]
train_mse2 = train_mse2[1:]
test_mse2 = test_mse2[1:]

x7 =  lamdavals2
y1_7 = train_mse2
y2_7 = test_mse2

plt.plot(x7, y1_7, label="Training", color="green")
plt.plot(x7, y2_7, label="Testing", color="blue")

plt.xlabel("Lambda")
plt.ylabel("MSE Values")
plt.title("100-100 without lambda 0")
plt.legend()

plt.show()

#graph 8
#set 4 wo lambda 0
lamdavals4 =  lamdavals[1:]
train_mse4 = train_mse4[1:]
test_mse4 = test_mse4[1:]

x8 =  lamdavals4
y1_8 = train_mse4
y2_8 = test_mse4

plt.plot(x8, y1_8, label="Training", color="green")
plt.plot(x8, y2_8, label="Testing", color="blue")

plt.xlabel("Lambda")
plt.ylabel("MSE Values")
plt.title("50(1000)-100 without lambda 0")
plt.legend()

plt.show()

#graph 9
#set 5 wo lambda 0
lamdavals5 =  lamdavals[1:]

train_mse5 = train_mse5[1:]
test_mse5 = test_mse5[1:]

x9 =  lamdavals5
y1_9 = train_mse5
y2_9 = test_mse5

plt.plot(x9, y1_9, label="Training", color="green")
plt.plot(x9, y2_9, label="Testing", color="blue")

plt.xlabel("Lambda")
plt.ylabel("MSE Values")
plt.title("100(1000)-100 without lambda 0")
plt.legend()

plt.show()
