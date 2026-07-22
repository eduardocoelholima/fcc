import numpy as np
import torch
from scipy.linalg import sqrtm


# Function to calculate mean and covariance of features
def calculate_mean_covariance(features):
    # Calculate mean
    mean = torch.mean(features, dim=0).numpy()
    # Calculate covariance
    # Note: torch.cov requires a tensor of shape (n_features, n_samples)
    # We need to transpose the features tensor for this.
    # Also, it's recommended to move the tensor to CPU if it's on GPU
    # for covariance calculation if memory is a concern.
    cov = torch.cov(features.T).numpy()
    return mean, cov


# Function to calculate the Fréchet Inception Distance (FID)
def calculate_fid(mu1, sigma1, mu2, sigma2):
    # Calculate the squared difference between the means
    diff = mu1 - mu2
    sum_sq_diff = np.sum(diff**2)

    # Calculate the square root of the product of covariances
    # This requires handling potential numerical instability
    covmean = sqrtm(sigma1.dot(sigma2))

    # Check for numerical issues and correct if necessary
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Calculate the trace of the sum of covariances minus twice the square root of their product
    fid = sum_sq_diff + np.trace(sigma1 + sigma2 - 2 * covmean)

    return fid


# Wrapper function for FID calculation
def fid(images1, images2):
    # Calculate mean and covariance for 'cat' features
    mean_cat, cov_cat = calculate_mean_covariance(images1)
    # print("Mean and covariance calculated for 'cat' features.")
    # print(f"Mean shape: {mean_cat.shape}")
    # print(f"Covariance shape: {cov_cat.shape}")
    # Calculate mean and covariance for 'fox' features
    mean_fox, cov_fox = calculate_mean_covariance(images2)
    # print("Mean and covariance calculated for 'fox' features.")
    # print(f"Mean shape: {mean_fox.shape}")
    # print(f"Covariance shape: {cov_fox.shape}")
    return calculate_fid(mean_cat, cov_cat, mean_fox, cov_fox)


def main():
    print('FID routines loaded')


# main()