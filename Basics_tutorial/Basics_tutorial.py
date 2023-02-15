import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm

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

# 7. PDF of astronomical magnitudes
# Sampling the process of measuring magnitudes

# Function for fitting a Gaussian
def gauss(x, A, mu, sigma):
    return A*np.exp(-(x-mu)**2/(2*sigma**2))

# Function to return the magnitude of a star given fluxes
def m1_calculator(m2, F1, F2):
    m1 = m2 - 2.5*np.log10(F1/F2)
    return m1

# One star's magnitude is known
mu_F1 = 100
sigma_F1 = 10
F1 = np.random.normal(loc = mu_F1, scale = sigma_F1, size = 1000)
F2 = 1
m2 = 0
m1 = m1_calculator(m2, F1, F2)
counts, bin_edges = np.histogram(m1, bins = 50)
array_1 = bin_edges[:-1]
array_2 = bin_edges[1:]
bin_means = 0.5*(array_1 + array_2)
params, _ = curve_fit(gauss, xdata = bin_means, ydata = counts) 
A_1, mu_1, sigma_1 = params
plt.hist(m1, bins = 50)
plt.plot(bin_means, gauss(bin_means, A_1, mu_1, sigma_1), label = "Best fit")
plt.xlabel("$m_1$")
plt.ylabel("Counts")
plt.title("$m_2$ known but $m_1$ unknown")
plt.legend()
plt.show()

# Both magnitudes unknown
mu_F1 = 100
sigma_F1 = 10
F1 = np.random.normal(loc = mu_F1, scale = sigma_F1, size = 1000)
mu_f2 = 200
sigma_F2 = 20
F2 = np.random.normal(loc = mu_F1, scale = sigma_F1, size = 1000)
# calculating m2 by using m = 0 as the base
m2 = m1_calculator(0, F2, 1)
m1 = m1_calculator(m2, F1, F2)
counts, bin_edges = np.histogram(m1, bins = 50)
array_1 = bin_edges[:-1]
array_2 = bin_edges[1:]
bin_means = 0.5*(array_1 + array_2)
params, _ = curve_fit(gauss, xdata = bin_means, ydata = counts) 
A_1, mu_1, sigma_1 = params
plt.hist(m1, bins = 50)
plt.plot(bin_means, gauss(bin_means, A_1, mu_1, sigma_1), label = "Best fit")
plt.xlabel("$m_1$")
plt.ylabel("Counts")
plt.title("Both $m_1$ and $m_2$ unknown")
plt.legend()
plt.show()

# Both known but follow same distribution 
mu_F1 = 100
sigma_F1 = 10
F1 = np.random.normal(loc = mu_F1, scale = sigma_F1, size = 1000)
mu_f2 = 200
sigma_F2 = 10
F2 = np.random.normal(loc = mu_F1, scale = sigma_F1, size = 1000)
# calculating m2 by using m = 0 as the base
m2 = m1_calculator(0, F2, 1)
m1 = m1_calculator(m2, F1, F2)
counts, bin_edges = np.histogram(m1, bins = 50)
array_1 = bin_edges[:-1]
array_2 = bin_edges[1:]
bin_means = 0.5*(array_1 + array_2)
params, _ = curve_fit(gauss, xdata = bin_means, ydata = counts) 
A_1, mu_1, sigma_1 = params
plt.hist(m1, bins = 50)
plt.plot(bin_means, gauss(bin_means, A_1, mu_1, sigma_1), label = "Best fit")
plt.xlabel("$m_1$")
plt.ylabel("Counts")
plt.title("Both $m_1$ and $m_2$ unknown but same $\sigma$")
plt.legend()
plt.show()
