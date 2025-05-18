import  numpy as np
import matplotlib.pyplot as plt

train=np.loadtxt('click.csv',delimiter=',',dtype=int,skiprows=1)
train_x=train[:,0]
train_y=train[:,1]

average=train_x.mean()
sigma=train_x.std()

train_z=(train_x-average)/sigma

theta0=np.random.rand()
theta1=np.random.rand()
#预测函数
def f(x):
    return theta1*x+theta0


#目标函数
def E(x,y):
    return 0.5*np.sum((y-f(x))**2)

ETA=1e-3
diff=1
error=E(train_z,train_y)
count=0
while diff>1e-2:
            temp0=theta0-ETA*np.sum(f(train_z)-train_y)
            temp1=theta1-ETA*np.sum((f(train_z)-train_y)*train_z)
            theta0=temp0
            theta1=temp1
            crruenterror=E(train_z,train_y)
            diff=error-crruenterror
            error=crruenterror
            count+=1
            print('第{}次: theta0 = {:.3f}, theta1 = {:.3f}, 差值 = {:.4f}'.format(count,theta0,theta1,diff))

x=np.linspace(-3,3,100)
plt.plot(train_z,train_y,'o')
plt.plot(x,f(x))
plt.show()











