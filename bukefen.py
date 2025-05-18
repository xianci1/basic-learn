import numpy as np
import matplotlib.pyplot as plt

train=np.loadtxt('class.csv',delimiter=',',skiprows=1)
train_x=train[:,0:2]
train_y=train[:,2]
mu=train_x.mean(axis=0)
sigma=train_x.std(axis=0)

def standardize(x):
    return (x-mu)/sigma

train_z=standardize(train_x)

theta=np.random.rand(4)

def f(x):
    return 1/(1+np.exp(-np.dot(x,theta)))

def to_matrix(x):
    X0=np.ones([x.shape[0],1])
    X3=x[:,0,np.newaxis]**2
    return  np.hstack([X0,x,X3])

X=to_matrix(train_z)

ETA=1e-3
epoch=5000
count=0
for _ in range(epoch):
    theta=theta-ETA*np.dot(f(X)-train_y,X)
    count+=1
    print("第{}次：theta={}".format(count,theta))



plt.plot(train_z[train_y==1,0],train_z[train_y==1,1],'x')
plt.plot(train_z[train_y==0,0],train_z[train_y==0,1],'o')
x1 = np.linspace(-2, 2, 100)
x2 = -(theta[0] + theta[1] * x1 + theta[3] * x1 ** 2) / theta[2]
plt.plot(x1,x2,linestyle='dashed')

plt.show()












