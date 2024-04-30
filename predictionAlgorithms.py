import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import random
from copy import deepcopy
import csv

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


def logistic(x):
    return 1 / (1 + np.exp(-x))


def nll(Y, T, epsi=1e-10):
    return -1 * np.sum(
        [
            t * math.log(Y[idx] + epsi) + (1 - t) * math.log(1 - Y[idx] + epsi)
            for idx, t in enumerate(T)
        ]
    )


def logisticAccuracy(Y, T):
    acc = 0
    for idx, y in enumerate(Y):
        # TP
        if y >= 0.5:
            if T[idx] == 1:
                acc += 1
        # TN
        else:
            if T[idx] == 0:
                acc += 1
    return acc / len(Y)


def logisticPrecision(Y, T):
    truep = 0
    falsp = 0

    for idx, y in enumerate(Y):
        # TP
        if y >= 0.5:
            if T[idx] == 1:
                truep += 1
            # FP
            else:
                falsp += 1
    return truep / (truep + falsp) if truep + falsp != 0 else 0


def logisticRecall(Y, T):
    truep = 0
    falsn = 0

    for idx, y in enumerate(Y):
        # TP
        if y >= 0.5:
            if T[idx] == 1:
                truep += 1
            # FN
        else:
            if T[idx] == 1:
                falsn += 1
    return truep / (truep + falsn) if truep + falsn != 0 else 0


def logisticF1Score(prec, rec):
    pass


def split_dataset(X, T, train=0.8):
    N = X.shape[0]
    N_train = int(round(N * train))

    X_train, X_test = X[:N_train, :], X[N_train:, :]
    T_train, T_test = T[:N_train], T[N_train:]
    return X_train, T_train, X_test, T_test


def labLogistic(X_train, X_test, T_train, T_test, W, lr=0.01, epochs_no=100):
    train_acc, test_acc = [], []
    train_nll, test_nll = [], []
    precList, recallList, f1scoreList = [], [], []
    (
        total_acc,
        total_prec,
        total_recall,
    ) = (
        0,
        0,
        0,
    )
    tempW = W.copy()
    for i in range(epochs_no):
        Y_train = list(map(lambda x: logistic(x), np.dot(X_train, tempW)))
        # Update parameters - Gradient Descent
        tempW -= lr * np.dot(np.transpose(X_train), (Y_train - T_train))

        # Just for plotting
        Y_test = 1.0 / (1.0 + np.exp(-np.dot(X_test, tempW)))
        train_acc.append(logisticAccuracy(Y_train, T_train))
        test_acc.append(logisticAccuracy(Y_test, T_test))
        train_nll.append(nll(Y_train, T_train))
        test_nll.append(nll(Y_test, T_test))
        precList.append(logisticPrecision(Y_train, T_train))
        recallList.append(logisticRecall(Y_train, T_train))

    total_acc = train_acc[-1]
    total_prec = precList[-1]
    total_recall = recallList[-1]
    total_f1score = (
        2 * (total_prec * total_recall) / (total_prec + total_recall)
        if (total_prec + total_recall) != 0
        else 0
    )

    return (
        train_acc,
        test_acc,
        train_nll,
        test_nll,
        total_acc,
        total_prec,
        total_recall,
        total_f1score,
    )


def scikitLogistic(X_train, X_test, T_train, T_test):
    logModel = LogisticRegression()
    logModel.fit(X_train, T_train)
    predict = logModel.predict(X_test)
    # acc = accuracy_score(T_test, predict)
    raport = classification_report(T_test, predict, output_dict=True)
    prec = raport["weighted avg"]["precision"]
    rec = raport["weighted avg"]["recall"]
    f1score = raport["weighted avg"]["f1-score"]
    return prec, rec, f1score


def scikitTree(X_train, X_test, T_train, T_test):
    treeModel = DecisionTreeClassifier()
    treeModel.fit(X_train, T_train)
    predict = treeModel.predict(X_test)
    # acc = accuracy_score(T_test, predict)
    raport = classification_report(T_test, predict, output_dict=True)
    prec = raport["weighted avg"]["precision"]
    rec = raport["weighted avg"]["recall"]
    f1score = raport["weighted avg"]["f1-score"]
    return prec, rec, f1score


def getDataSet():
    """Reads a dataset

    Args:
        dataSetName (str): Name for the dataset

    Returns:
        A tuple containing (classes, attributes, examples):
        classes (set): the classes that are found in the dataset
        attributes (list of strings): the attributes for the dataset
        examples (list of dictionaries): one example contains an entry as
            (attribute name, attribute value)
    """

    dataset_file = "dataset.csv"
    f_in = open(dataset_file, "r")
    csv_reader = csv.reader(f_in, delimiter=",")

    # Read the header row
    row = next(csv_reader)

    # The last element represents the class
    attributeNames = row[:-1]

    examples = []
    classes = set()

    for row in csv_reader:
        *attributes, label = row
        classes.add(label)
        example = dict(zip(attributeNames, attributes))
        example["Revenue"] = label
        examples.append(example)

    f_in.close()
    return classes, attributeNames, examples


class Node:
    """Representation for a node from the decision tree"""

    def __init__(self, label):
        """
        for non-leafs it is the name of the attribute
        for leafs it is the class
        """
        self.label = label

        # Dictionary of (attribute value, nodes)
        self.children = {}

    def display(self, indent=""):
        print(indent + (self.label + ":" if self.children else "<" + self.label + ">"))
        indent += "   "
        if self.children:
            for key, value in self.children.items():
                print(indent + ":" + key)
                value.display(indent + "   ")


def randomTree(X, A, d):
    if d == 0 or not A:
        classes = {}
        for x in X:
            if x["Revenue"] in classes:
                classes[x["Revenue"]] = classes[x["Revenue"]] + 1
            else:
                classes[x["Revenue"]] = 1

        freq_class = max(classes, key=classes.get)
        return Node(freq_class)

    elif d > 0:
        attr = random.choice(A)
        A_new = deepcopy(A)
        A_new.remove(attr)
        node = Node(attr)

        vals = {}
        for x in X:
            crt_val = x[attr]
            if crt_val not in vals:
                vals[crt_val] = [x]
            else:
                vals[crt_val].append(x)

        for val, nodes in vals.items():
            child_node = randomTree(nodes, A_new, d - 1)
            node.children[val] = child_node

        return node


def plot_evolution(train_acc, test_acc, train_nll, test_nll):
    epochs_no = len(train_acc)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(range(epochs_no), train_acc, sns.xkcd_rgb["green"], label="Train Accuracy")
    ax1.plot(range(epochs_no), test_acc, sns.xkcd_rgb["red"], label="Test Accuracy")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="lower right", ncol=1)

    ax2.plot(range(epochs_no), train_nll, sns.xkcd_rgb["green"], label="Train NLL")
    ax2.plot(range(epochs_no), test_nll, sns.xkcd_rgb["red"], label="Test NLL")
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("NLL")
    ax2.legend(loc="upper right", ncol=1)
    return (ax1, ax2)
