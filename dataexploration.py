import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import pointbiserialr, chi2_contingency
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
)


def balanceAnalysis(currDF):
    startTime = time.time()
    for label in currDF:
        labelCounts = currDF[label].value_counts()
        treshold = 10
        labelCounts = labelCounts[labelCounts >= treshold]
        tempDF = labelCounts.reset_index()
        tempDF.columns = [label, "Value"]
        tempDF.plot(
            kind="bar", x=label, y="Value", alpha=0.5, color="b", figsize=(16, 9)
        )
        plt.title(f"Label Distribution for {label}")
        plt.xlabel("Count")
        plt.ylabel("Frequency")

    finishTime = time.time()
    elapsedTime = finishTime - startTime
    print(f"Class Balance Analysis took {elapsedTime:.2f}s \n")

    plt.show()


def numericPercentiles(currDF, mainDF):
    for label in currDF:
        percentiles = np.percentile(mainDF[label], np.arange(0, 101, 10))
        plt.figure(figsize=(10, 6))
        plt.hist(mainDF[label], bins=50, alpha=0.5, color="b")
        plt.title(f"Label Distribution in percentile for {label}")
        plt.xlabel(label)
        plt.ylabel("Frequency")
        for percentile in percentiles:
            plt.axvline(percentile, color="r", linestyle="dashed", linewidth=2)
    plt.show()


def catgHist(currDF):
    for label in currDF:
        currSeries = currDF[label].value_counts()
        tempDF = currSeries.reset_index()
        tempDF.columns = [label, "Value"]
        tempDF.plot(kind="hist", bins=100, alpha=0.5, color="b", figsize=(16, 9))
        # plt.yticks([1, 10])
        plt.title(f"Label Distribution for {label}")
        plt.xlabel(label)
        plt.ylabel("Frequency")
    plt.show()


def corrNumeric(currDF, mainDF):
    correlationNumeric = pd.DataFrame(
        index=currDF.columns, columns=["Point-Biserial Correlation", "p_value"]
    )

    for idx, label in enumerate(currDF):
        corrCoef, p_value = pointbiserialr(currDF[label], mainDF["Revenue"])
        correlationNumeric.iloc[idx] = [corrCoef, p_value]

    plotPVAL(correlationNumeric, "Point-Biserial Correlation")


def corrCategoric(currDF, mainDF):
    correlationCategoric = pd.DataFrame(
        index=currDF.columns, columns=["Pearson Chi-squared", "p_value"]
    )
    for idx, label in enumerate(currDF):
        chisqt = pd.crosstab(currDF[label], mainDF["Revenue"])
        corrCoef, p_value, _, _ = chi2_contingency(chisqt)
        correlationCategoric.iloc[idx] = [corrCoef, p_value]
    print(correlationCategoric)
    plotPVAL(correlationCategoric, "Pearson Chi-squared")


def plotPVAL(currDF, coeficient):
    filterdDF = currDF[currDF["p_value"] < 0.05]
    filterdDF.plot(
        kind="bar",
        y=coeficient,
        alpha=0.5,
        color="b",
        figsize=(16, 9),
    )
    plt.title(f"{coeficient} for values with p-value <0.05")
    plt.ylabel(f"{coeficient}")
    plt.xlabel("Values")
    plt.show()


def labelToIdx(currDF, catgDF):
    encoder = LabelEncoder()
    tempDF = currDF.copy()
    for label in currDF:
        if label in catgDF:
            tempDF[label] = encoder.fit_transform(currDF[label])
    return tempDF


def normAnalysis(currDF, normType):
    if normType == "minmax":
        minMaxScaler = MinMaxScaler()
        minMaxDF = pd.DataFrame(minMaxScaler.fit_transform(currDF))
        return minMaxDF

    if normType == "standard":
        standardScaler = StandardScaler()
        standardDF = pd.DataFrame(standardScaler.fit_transform(currDF))
        return standardDF

    if normType == "robust":
        robustScaler = RobustScaler()
        robustDF = pd.DataFrame(robustScaler.fit_transform(currDF))
        return robustDF
