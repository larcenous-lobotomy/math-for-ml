import numpy as np
import matplotlib.pyplot as plt

#initialise A and b
A = np.random.rand(20, 10)
b = np.random.rand(20,1)

#closed-form solution to least squares problem
x_hat = np.dot(np.linalg.pinv(A), b)

#store norms to plot convergence
norm_array = np.zeros(500)

#initialise x_k=1
x_k = np.zeros((10, 1))
#set parameter mu to 1 / ||A||^2
mu = 1 / (np.linalg.norm(A))**2

#run Richardson for 500 iterations
for i in range(500):
    x_k -= mu*np.dot(np.transpose(A), (np.dot(A, x_k) - b))
    norm_array[i] = np.linalg.norm(x_k - x_hat)

#plot ||x_k - x||
plt.figure(figsize=(30,10))
plt.plot(norm_array, 'b')
plt.plot([0 for i in range(500)], 'r')
plt.show()
