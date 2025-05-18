import numpy as np
import matplotlib.pyplot as plt
train=np.loadtxt('data.csv',delimiter=',',dtype='int',skiprows=1)
train_x=train[:,0:2]
train_y=train[:,2]

mu=train_x.mean(axis=0)
sigma=train_x.std(axis=0)
def standardize(x):
    return (x-mu)/sigma

train_z=standardize(train_x)

def to_matrix(x):
    x0 = np.ones([x.shape[0], 1])
    return np.hstack([x0, x])


X=to_matrix(train_z)

theta = np.random.rand(3)

def f(x):
    return  1/(1+np.exp(-np.dot(x,theta)))

epoch=1000
count=0
ETA=1e-3
for _ in range(epoch):
    theta=theta-ETA*np.dot(f(X)-train_y,X)
    count+=1
    print('第{}次：theta={}'.format(count,theta))

x0 = np.linspace(-2, 2, 100)
plt.plot(train_z[train_y == 1, 0], train_z[train_y == 1, 1], 'o')
plt.plot(train_z[train_y == 0, 0], train_z[train_y == 0, 1], 'x')
plt.plot(x0, -(theta[0] + theta[1] * x0) / theta[2], linestyle='dashed')
plt.show()



