import numpy as np
import operator


class Decomposition(object):
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

class PrincipalComponentAnalysis(Decomposition):
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

        _, self.sorted_eigenvecs = tuple(zip(*eigen))

class LinearDiscriminantAnalysis(Decomposition):
    def fit(self, dataset, labels):
        global_means = np.array(list(map(np.mean, dataset.T)))
        between_matrix = 0
        within_matrix = 0

        unique_labels = set(labels)
        for label in unique_labels:
            # Separate data by label and get amount of data
            indices = np.where(labels == label)[0]
            data_amount = indices.size

            # Calculate the mean of each feature for every label (between matrix step)
            means = np.array(list(map(np.mean, dataset[indices].T)))

            # Normalize by the global means (between matrix step)
            means -= global_means

            between_matrix += data_amount * means.dot(means.T)
            within_matrix += np.cov(dataset.T)

        # Calculate the inverse of the within matrix
        inverse_within = np.linalg.inv(within_matrix)

        # Eigenvectors and eigenvalues from the covariance matrix
        # The ith eigenvalue corresponde to the ith column vector
        eigenval, eigenvecs = np.linalg.eig(inverse_within.dot(between_matrix))

        # Transform list of list [vals, vecs] to list of tuple (val, vec)
        eigen = list(zip(eigenval, eigenvecs.T))

        # Sort elements from biggest eigenvalue to smallest
        eigen.sort(key=lambda x: x[0], reverse=True)

        _, self.sorted_eigenvecs = tuple(zip(*eigen))