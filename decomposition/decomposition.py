import numpy as np
import operator

class PrincipalComponentAnalysis(object):
    def fit(self, dataset):
        # Calculate the mean of each feature
        features_mean = list(map(np.mean, dataset.T))

        # Normalize the dataset
        normalized_data = dataset - features_mean

        # Calculate covariance matrix of the features
        cov_matrix = np.cov(normalized_data.T)
        
        # Eigenvectors and eigenvalues from the covariance matrix
        # The ith eigenvalue corresponde to the ith column vector
        eigenval, eigenvecs = np.linalg.eig(cov_matrix)
        
        # Transform list of list [vals, vecs] to list of tuple (val, vec)
        eigen = list(zip(eigenval, eigenvecs.T))

        # Sort elements from biggest eigenvalue to smallest
        eigen.sort(key=lambda x: x[0], reverse=True)

        self.eigenval, self.sorted_eigenvecs = tuple(zip(*eigen))

    def transform(self, dataset, n_components):
        dataset_len = len(dataset)

        # Create transformer matrix
        transformer = self.sorted_eigenvecs[0]
        for i in range(1, n_components):
            transformer = np.vstack((transformer, self.sorted_eigenvecs[i]))

        # Project the dataset on the new system
        transformed = transformer.dot(dataset.T)

        transformed.shape = (dataset_len, n_components)
        return transformed
