import codecs

from decomposition import PrincipalComponentAnalysis, LinearDiscriminantAnalysis
from retriever.data_collection import DataCollection, EDataType
from plot import plot_stuff
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import numpy as np

print('Collecting data...')
data_collection = DataCollection('res/promise/')

pca = PrincipalComponentAnalysis()
lda = LinearDiscriminantAnalysis()

knn = KNeighborsClassifier(n_neighbors=9, weights='distance', algorithm='auto')
kf = KFold(n_splits=5, shuffle=True) # 80% for training, 20% for testing

results = {}
for relation in data_collection.documents:
    file_ = codecs.open('res/results/' + relation + '.txt', 'w+', 'utf-8')

    results[relation] = {}
    results[relation]['PCA'] = {}
    results[relation]['LDA'] = {}

    print('\nTesting for ' + relation + ' data set')
    relation_data, relation_labels = data_collection.get_data_label(relation)
    data_len = len(relation_data)
    features_amount = relation_data[0].size
    print('\tTotal data collected: ' + str(data_len))
    print('\tTotal of features per data: ' + str(features_amount))

    # Only numerical features
    features_types = data_collection.get_features_types(relation)

    pca.fit(relation_data)
    lda.fit(relation_data, relation_labels)

    pca_precision_mean = []
    pca_recall_mean = []
    pca_f1_mean = []

    lda_precision_mean = []
    lda_recall_mean = []
    lda_f1_mean = []
    for n_components in range(1, features_amount+1):
        pca_components = pca.transform(relation_data, n_components)
        lda_components = lda.transform(relation_data, n_components)

        pca_metrics = []
        lda_metrics = []
        for train_indexes, test_indexes in kf.split(relation_data):
            pca_train, pca_test = pca_components[train_indexes], pca_components[test_indexes]
            lda_train, lda_test = lda_components[train_indexes], lda_components[test_indexes]
            train_labels, test_labels = relation_labels[train_indexes], relation_labels[test_indexes]

            # PCA testing
            knn.fit(pca_train, train_labels)
            pred_labels = knn.predict(pca_test)
            pca_metrics.append(precision_recall_fscore_support(test_labels, pred_labels, average='weighted'))

            # LDA testing
            knn.fit(lda_train, train_labels)
            pred_labels = knn.predict(lda_test)
            lda_metrics.append(precision_recall_fscore_support(test_labels, pred_labels, average='weighted'))
        
        pca_precisions = [precision for precision, _, _, _ in pca_metrics]
        pca_recalls = [recall for _, recall, _, _ in pca_metrics]
        pca_f1s = [f1 for _, _, f1, _ in pca_metrics]

        lda_precisions = [precision for precision, _, _, _ in lda_metrics]
        lda_recalls = [recall for _, recall, _, _ in lda_metrics]
        lda_f1s = [f1 for _, _, f1, _ in lda_metrics]
        
        pca_precision_mean.append(np.mean(pca_precisions))
        pca_recall_mean.append(np.mean(pca_recalls))
        pca_f1_mean.append(np.mean(pca_f1s))

        lda_precision_mean.append(np.mean(lda_precisions))
        lda_recall_mean.append(np.mean(lda_recalls))
        lda_f1_mean.append(np.mean(lda_f1s))

        results[relation]['PCA']['precision'] = pca_precision_mean
        results[relation]['PCA']['recall'] = pca_recall_mean
        results[relation]['PCA']['f1'] = pca_f1_mean

        results[relation]['LDA']['precision'] = lda_precision_mean
        results[relation]['LDA']['recall'] = lda_recall_mean
        results[relation]['LDA']['f1'] = lda_f1_mean

    file_.write('PCA precision:\n' + str(pca_precision_mean))
    file_.write('\n\nPCA recall:\n' + str(pca_recall_mean))
    file_.write('\n\nPCA f1:\n' + str(pca_f1_mean))

    file_.write('\n\nLDA precision:\n' + str(lda_precision_mean))
    file_.write('\n\nLDA recall:\n' + str(lda_recall_mean))
    file_.write('\n\nLDA f1:\n' + str(lda_f1_mean))

    plot_stuff(results[relation], relation, list(range(1, features_amount+1)))
