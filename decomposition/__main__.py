import codecs

from decomposition import PrincipalComponentAnalysis
from retriever.data_collection import DataCollection, EDataType

from sklearn.metrics import precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import numpy as np

print('Collecting data...')
data_collection = DataCollection('res/promise/')

pca = PrincipalComponentAnalysis()
knn = KNeighborsClassifier(n_neighbors=9, weights='distance', algorithm='auto')
kf = KFold(n_splits=5, shuffle=True) # 80% for training, 20% for testing

for relation in data_collection.documents:
    file_ = codecs.open('res/results/' + relation + '.txt', 'w+', 'utf-8')

    print('\nTesting for ' + relation + ' data set')
    relation_data, relation_labels = data_collection.get_data_label(relation)
    data_len = len(relation_data)
    features_amount = relation_data[0].size
    print('\tTotal data collected: ' + str(data_len))
    print('\tTotal of features per data: ' + str(features_amount))

    # Only numerical features
    features_types = data_collection.get_features_types(relation)

    pca.fit(relation_data)

    precision_mean = []
    recall_mean = []
    f1_mean = []
    for n_components in range(1, features_amount+1):
        data_components = pca.transform(relation_data, n_components)

        metrics = []
        for train_indexes, test_indexes in kf.split(data_components):
            train_data, test_data = data_components[train_indexes], data_components[test_indexes]
            train_labels, test_labels = relation_labels[train_indexes], relation_labels[test_indexes]

            knn.fit(train_data, train_labels)
            pred_labels = knn.predict(test_data)

            metrics.append(precision_recall_fscore_support(test_labels, pred_labels, average='weighted'))
        
        precisions = [precision for precision, _, _, _ in metrics]
        recalls = [recall for _, recall, _, _ in metrics]
        f1s = [f1 for _, _, f1, _ in metrics]
        
        precision_mean.append(np.mean(precisions))
        recall_mean.append(np.mean(recalls))
        f1_mean.append(np.mean(f1s))

    file_.write('precision:\n' + str(precision_mean))
    file_.write('\n\nrecall:\n' + str(recall_mean))
    file_.write('\n\nf1:\n' + str(f1_mean))
