import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import warnings

from predictionAlgorithms import (
    plot_evolution,
    scikitLogistic,
    scikitTree,
    labLogistic,
    split_dataset,
    randomTree,
    getDataSet,
)
from dataexploration import (
    labelToIdx,
    balanceAnalysis,
    numericPercentiles,
    catgHist,
    corrNumeric,
    corrCategoric,
    normAnalysis,
)


# suppress warnings
warnings.filterwarnings("ignore")


def evaluation(train_test_DF):
    for norm in normTypes:
        currDF = normAnalysis(train_test_DF, norm)
        X = currDF.drop(columns=currDF.columns[-1]).values
        (N, D) = X.shape
        X = np.concatenate([np.ones((N, 1)), X], axis=1)
        W = np.random.randn((D + 1))
        T = allNumDF[allNumDF.columns[-1]].values
        (
            precList_scikitLogistic,
            recallList_scikitLogistic,
            f1scoreList_scikitLogistic,
        ) = ([], [], [])
        (
            precList_labLogistic,
            recallList_labLogistic,
            f1scoreList_labLogistic,
        ) = ([], [], [])
        (
            precList_scikitTree,
            recallList_scikitTree,
            f1scoreList_scikitTree,
        ) = ([], [], [])
        for _ in range(10):
            X_train, X_test, T_train, T_test = train_test_split(X, T, test_size=0.2)
            # Scikit Logistic Regresion
            total_prec, total_recall, total_f1score = scikitLogistic(
                X_train, X_test, T_train, T_test
            )
            precList_scikitLogistic.append(total_prec)
            recallList_scikitLogistic.append(total_recall)
            f1scoreList_scikitLogistic.append(total_f1score)

            # Modified Lab Logistic Regresion
            results = labLogistic(
                X_train,
                X_test,
                T_train,
                T_test,
                W,
                lr=0.1,
                epochs_no=EPOCHS_NO,
            )

            (
                train_acc,
                test_acc,
                train_nll,
                test_nll,
                total_acc,
                total_prec,
                total_recall,
                total_f1score,
            ) = results
            precList_labLogistic.append(total_prec)
            recallList_labLogistic.append(total_recall)
            f1scoreList_labLogistic.append(total_f1score)
            # plot_evolution(precList, test_acc, train_nll, test_nll)

            # Scikit DecisionTreeClassifier
            total_prec, total_recall, total_f1score = scikitTree(
                X_train, X_test, T_train, T_test
            )
            precList_scikitTree.append(total_prec)
            recallList_scikitTree.append(total_recall)
            f1scoreList_scikitTree.append(total_f1score)

            # Modified Lab Decision Random Tree
            # TO DO
            # plt.show()
        prec_scikitLogistic = np.mean(precList_scikitLogistic)
        recall_scikitLogistic = np.mean(recallList_scikitLogistic)
        f1score_scikitLogistic = np.mean(f1scoreList_scikitLogistic)
        prec_labLogistic = np.mean(precList_labLogistic)
        recall_labLogistic = np.mean(recallList_labLogistic)
        f1score_labLogistic = np.mean(f1scoreList_labLogistic)
        prec_scikitTree = np.mean(precList_scikitTree)
        recall_scikitTree = np.mean(recallList_scikitTree)
        f1score_scikitTree = np.mean(f1scoreList_scikitTree)
        print(
            f"scikit logistic using {norm} results: precision={prec_scikitLogistic:.5f}, recall={recall_scikitLogistic:.5f}, f1_score={f1score_scikitLogistic:.5f} "
        )
        print(
            f"modifed lab logistic using {norm} results: precision={prec_labLogistic:.5f}, recall={recall_labLogistic:.5f}, f1_score={f1score_labLogistic:.5f} "
        )
        print(
            f"scikit decision tree using {norm} results: precision={prec_scikitTree:.5f}, recall={recall_scikitTree:.5f}, f1_score={f1score_scikitTree:.5f} "
        )
        print("")


EPOCHS_NO = 500
X_train, X_test = [], []
T_train, T_test = [], []
normTypes = ["minmax", "standard", "robust"]
df = pd.read_csv("dataset.csv")


roundedDF = df.round(3)
numDF = df.select_dtypes(include=["int", "float"])
catgDF = df.select_dtypes(include=["bool", "object"])
catgNoRevDF = catgDF.drop(columns=["Revenue"])
allNumDF = labelToIdx(df, catgDF)
evaluation(allNumDF)


# balanceAnalysis(roundedDF)
# numericPercentiles(numDF, df)
# catgHist(catgDF)
# corrNumeric(numDF, df)
# corrCategoric(catgNoRevDF, df)
