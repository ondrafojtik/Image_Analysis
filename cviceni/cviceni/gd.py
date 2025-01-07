import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import time
import math
plt.ion()
#from IPython import display
#display.set_matplotlib_formats('svg')

# gradient == derivace v ruznych smerech

def fx(x):
    return x**2 - 6*x + 1

def deriv(x):
    return 2*x - 6

x = np.linspace(-14, 20, 2000)
print(x)

localmin = 15.0 #np.random.choice(x,1)
print("localmin", localmin)

plt.plot(x, fx(x))
plt.plot(localmin, fx(localmin), 'ro')
plt.show()
plt.pause(5)

learning_rate = 0.01
training_epochs = 50

for i in range(training_epochs):
    print("---------------- epoch", i)
    grad = deriv(localmin)
    move = learning_rate * grad
    localmin = localmin - move
    print("grad", grad)
    print("localmin new", localmin)
    print("localmin fx", fx(localmin))
    plt.plot(localmin, fx(localmin), 'ro')
    plt.show()
    plt.pause(0.1)
print(localmin)

plt.savefig("out.png")
