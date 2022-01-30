from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

from data_prepare import Data
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix

test_size = 0.2
class_names = ['edible', 'poisonous']
labels, features = Data.load_data(file_name="data/agaricus-lepiota.data")

global_labels = LabelEncoder().fit_transform(labels)

# normalize feature vector
scaler = MinMaxScaler(feature_range=(0, 1))
global_features = scaler.fit_transform(features)

# split the training and testing data
(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(
    np.array(global_features),
    np.array(global_labels),
    test_size=test_size)
print("Train data - Total {}, feature size {}".format(trainDataGlobal.shape[0], trainDataGlobal.shape[1]))
print("Test data - Total {}, feature size {}".format(testDataGlobal.shape[0], testDataGlobal.shape[1]))



# scoring = "accuracy"
# results = []
# names = []
# models = []
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('RF', RandomForestClassifier(n_estimators=100)))
# models.append(('SVM', SVC()))
#
# for name, model in models:
#     kfold = KFold(n_splits=5, random_state=1, shuffle=True)
#     cv_results = cross_val_score(model, global_features, global_labels, cv=kfold, scoring=scoring)
#     results.append(cv_results)
#     names.append(name)
#     msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#     print(msg)


clf = Perceptron(random_state=1, max_iter=30, tol=0.001)
# clf = MLPClassifier(solver='sgd', alpha=1e-5, learning_rate_init=0.1, verbose=10, hidden_layer_sizes=(15,15), random_state=1)

# fit the training data to the model
clf.fit(trainDataGlobal, trainLabelsGlobal)


predictions = clf.predict(testDataGlobal)

matrix = confusion_matrix(predictions, testLabelsGlobal)
acc_per_cls = matrix.diagonal()/matrix.sum(axis=0)

print('---------- Accuracy -----------')
for label, acc in zip(class_names, acc_per_cls):
    print('{} : {}'.format(label, acc))

