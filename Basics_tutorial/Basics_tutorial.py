import numpy as np
import matplotlib.pyplot as plt

# 1. Matrix manipulations
data = np.loadtxt("SN_covmat.txt")

# a) Dimensions
print("Length is", len(data))
cov_matrix = np.reshape(data, (31,31))
print("Dimensions of covariance matrix:", np.shape(cov_matrix))

# b) Plotting covariance matrix
plt.imshow(cov_matrix)
plt.title("Covariance matrix")
plt.colorbar()
plt.show()

# c) Computing correlation matrix
inv_sigma_matrix = np.diag(1/np.sqrt(np.diag(cov_matrix)))
corr_matrix = inv_sigma_matrix @ cov_matrix @ inv_sigma_matrix
plt.imshow(corr_matrix)
plt.title("Correlation matrix")
plt.colorbar()
plt.show()

# d) Most and least correlated data points
print("Most correlated:", np.where(corr_matrix == np.max(corr_matrix)))
print("Least correlated:", np.where(corr_matrix == np.min(corr_matrix)))

# f) Largest error bar
max_sigma = np.max(np.diag(cov_matrix))
print(f"Largest error bar: {np.where(np.diag(cov_matrix) == max_sigma)}")

# g) Determinant of covariance matrix
det_cov_matrix = np.linalg.det(cov_matrix)
print("Determinant of covariance matrix:", det_cov_matrix)

# h) Precision matrix
precision_matrix = np.linalg.inv(cov_matrix)
plt.imshow(precision_matrix)
plt.title("Precision matrix")
plt.colorbar()
plt.show()