# -*- coding: utf-8 -*-

#Imports
import numpy as np
import matplotlib.pyplot as plt

#Generate random square matrix of dimension=10 with elements in (0,1)
matrix = np.random.randint(100, size=(10,10))

matrix

#obtain symmetric matrix
symmetric_matrix = (matrix + matrix.T) / 2

symmetric_matrix

#Numpy's built-in algorithm
eigenvalues, eigenvectors = np.linalg.eig(symmetric_matrix)

eigenvalues

#estimate eigenvectors
A = symmetric_matrix

#store refinement in every iteration
eigenvalues_estimate_list = []
differences = []

#pick a starting basis for the space R^len(A)
V = np.random.randint(100,size=(10,10))

#QR decomposition for V
q, r = np.linalg.qr(V)

for k in range(100):
    eigenvalues_estimate = []
    for eigenvector in q.T:
        Lambda = np.dot(np.dot(eigenvector.T, A), eigenvector) / np.dot(eigenvector.T, eigenvector)
        eigenvalues_estimate.append(Lambda)

    #differences.append(eigenvalues_estimate - eigenvalues)
    eigenvalues_estimate_list.append(eigenvalues_estimate)
    W = np.dot(A,q)
    q, r = np.linalg.qr(W)

q

order = sorted(range(len(eigenvalues_estimate_list[0])), key=lambda x: eigenvalues_estimate_list[-1][x])
sorted_eigenvalues_estimate_list = [[row[x] for x in order] for row in eigenvalues_estimate_list]

eigenvalues.sort()

differences = [[x[i] - eigenvalues[i] for i in range(len(eigenvalues))] for x in sorted_eigenvalues_estimate_list]

differences = np.transpose(differences)

plt.plot(differences[-1])

plt.plot(differences[-2])

plt.plot(differences[-3])

plt.plot(differences[-4])

plt.plot(differences[-5])
