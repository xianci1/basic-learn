import numpy as np
import matplotlib.pyplot as plt

train=np.loadtxt('click.csv',dtype='int',delimiter=',',skiprows=1)
train_x=train[:,0]
train_y=train[:,1]

mu=train_x.mean()
sigma=train_x.std()
train_z=(train_x-mu)/sigma

def to_matrix(x):
    return np.vstack([np.ones(x.shape[0]),x,x**2]).T

theta=np.random.rand(3)

X=to_matrix(train_z)
def f(x):
    return np.dot(x,theta)

def E(x,y):
    return 0.5*np.sum((f(x)-y)**2)

diff=1
ETA=1e-3
error=E(X,train_y)

while diff>1e-2:
    theta=theta-ETA*np.dot(f(X)-train_y,X)
    current_error=E(X,train_y)
    diff=error-current_error
    error=current_error

plt.plot(train_z,train_y,'o')
x=np.linspace(-3,3,100)
plt.plot(x,f(to_matrix(x)))
plt.show()

