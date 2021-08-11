"""
mnist_svm.py
------------

A classifier program for recognizing 
handwritten digits from the MNIST
data set, using an SVM classifier.
"""

import mnist_loader

from sklearn import svm

def svm_baseline():
    training_data, validation_data, test_data = mnist_loader.load_data()

    X_train, y_train = training_data[0], training_data[1]
    clf = svm.SVC()
    clf.fit(X_train, y_train)

    X_test, y_test = test_data[0], test_data[1]
    predictions = [int(a) for a in clf.predict(X_test)]
    num_correct = sum(int(a == y) for a, y in zip(predictions, y_test))

    print("Baseline classifier using an SVM.")
    print("{0} of {1} values correct".format(num_correct, len(y_test)))

if __name__ == "__main__":
    svm_baseline()