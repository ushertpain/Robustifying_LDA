from scipy.stats.mstats import winsorize
import numpy as np
from scipy import stats
import math
############# L-estimator ###########
def lda(data):
    z1 = []
    z2 = []
    z1sqr = []
    z2sqr = []
    z4sqr = []
    Mean = [[np.mean(data[:, 0]), np.mean(data[:, 1])]]
    Size = np.size(data, axis=0)
    m1 = Mean[0][0]
    m2 = Mean[0][1]
    for i in range(Size):
        z1.append(data[i][0] - m1)
        z2.append(data[i][1] - m2)
        z1sqr.append((z1[i] ** 2))
        z2sqr.append((z2[i] ** 2))
        z4sqr.append(abs(z1[i] * z2[i]))
    v1 = np.sum(z1sqr) / (Size - 1)
    v2 = np.sum(z2sqr) / (Size - 1)
    v4 = np.sum(z4sqr) / (Size - 1)
    v3 = v4
    Z = [[v1, v3], [v4, v2]]
    return Mean, Z


def l_median(data):
    z1 = []
    z2 = []
    z1sqr = []
    z2sqr = []
    z4sqr = []
    Mean = [[np.median(data[:, 0]), np.median(data[:, 1])]]
    Size = np.size(data, axis=0)
    m1 = Mean[0][0] # mean of group 1
    m2 = Mean[0][1] # mean of group 2
    for i in range(Size):
        z1.append(data[i][0] - m1)
        z2.append(data[i][1] - m2)
        z1sqr.append((z1[i] ** 2))
        z2sqr.append((z2[i] ** 2))
        z4sqr.append((z1[i] * z2[i]))
    v1 = sum(z1sqr) / (Size - 1)
    v2 = sum(z2sqr) / (Size - 1)
    v4 = sum(z4sqr) / (Size - 1)
    v3 = v4
    Z = [[v1, v3], [v4, v2]]
    return Mean, Z


def l_trimean(data):
    z1 = []
    z2 = []
    z1sqr = []
    z2sqr = []
    z4sqr = []
    Size = np.size(data, axis=0)
    Q1 = np.percentile(data[:, 0], [25, 50, 75])
    Q2 = np.percentile(data[:, 1], [25, 50, 75])
    m1 = Q1[0] + (2 * Q1[1]) + Q1[2] * (1 / 4)
    m2 = Q2[0] + (2 * Q2[1]) + Q2[2] * (1 / 4)
    Mean = [[m1, m2]]
    for i in range(Size):
        z1.append(data[i][0] - m1)
        z2.append(data[i][1] - m2)
        z1sqr.append((z1[i] ** 2))
        z2sqr.append((z2[i] ** 2))
        z4sqr.append((z1[i] * z2[i]))
    v1 = sum(z1sqr) / (Size - 1)
    v2 = sum(z2sqr) / (Size - 1)
    v4 = sum(z4sqr) / (Size - 1)
    v3 = v4
    Z = [[v1, v3], [v4, v2]]
    return Mean, Z


def l_truncated_mean(data):
    z1 = []
    z2 = []
    z1sqr = []
    z2sqr = []
    z4sqr = []
    Mean = [[stats.trim_mean(data[:, 0], 0.15), stats.trim_mean(data[:, 1], 0.15)]]
    Size = np.size(data, axis=0)
    m1 = Mean[0][0]  # mean of group 1
    m2 = Mean[0][1]  # mean of group 2
    for i in range(Size):
        z1.append(data[i][0] - m1)
        z2.append(data[i][1] - m2)
        z1sqr.append((z1[i] ** 2))
        z2sqr.append((z2[i] ** 2))
        z4sqr.append((z1[i] * z2[i]))
    v1 = sum(z1sqr) / (Size - 1)
    v2 = sum(z2sqr) / (Size - 1)
    v4 = sum(z4sqr) / (Size - 1)
    v3 = v4
    Z = [[v1, v3], [v4, v2]]
    return Mean, Z


def l_winsorized_mean(data):

    z1 = []
    z2 = []
    z1sqr = []
    z2sqr = []
    z4sqr = []
    Mean = [[np.mean((winsorize(data[:, 0], limits=(0.15, 0.15), inplace=True))),
            np.mean((winsorize(data[:, 1], limits=(0.15, 0.15), inplace=True)))]]
    Size = np.size(data, axis=0)
    m1 = Mean[0][0]  # mean of group 1
    m2 = Mean[0][1]  # mean of group 2
    for i in range(Size):
        z1.append(data[i][0] - m1)
        z2.append(data[i][1] - m2)
        z1sqr.append((z1[i] ** 2))
        z2sqr.append((z2[i] ** 2))
        z4sqr.append((z1[i] * z2[i]))
    v1 = sum(z1sqr) / (Size - 1)
    v2 = sum(z2sqr) / (Size - 1)
    v4 = sum(z4sqr) / (Size - 1)
    v3 = v4
    Z = [[v1, v3], [v4, v2]]
    return Mean, Z


############# S-estimator ###########
'''def s_sd(data):
    z1 = []
    z2 = []
    z1sqr = []
    z2sqr = []
    z4sqr = []
    Mean = [[np.average(data[:, 0]), np.average(data[:, 1])]]
    Size = np.size(data, axis=0)
    m1 = Mean[0][0]
    m2 = Mean[0][1]
    for i in range(Size):
        z1.append(data[i][0] - m1)
        z2.append(data[i][1] - m2)
        z1sqr.append((z1[i] ** 2))
        z2sqr.append((z2[i] ** 2))
        z4sqr.append(abs(z1[i] * z2[i]))
    v1 = np.sqrt(np.sum(z1sqr) / (Size - 1))
    v2 = np.sqrt(np.sum(z2sqr) / (Size - 1))
    v4 = np.sqrt(np.sum(z4sqr) / (Size - 1))
    v3 = v4
    Z = [[v1, v3], [v4, v2]]
    return Mean, Z'''


def s_mad(data):
    z1 = []
    z2 = []
    z1sqr = []
    z2sqr = []
    z4sqr = []
    Mean = [[np.average(data[:, 0]), np.average(data[:, 1])]]
    Size = np.size(data, axis=0)
    m1 = Mean[0][0]
    m2 = Mean[0][1]
    for i in range(Size):
        z1.append(data[i][0] - m1)
        z2.append(data[i][1] - m2)
        z1sqr.append((z1[i] ** 2))
        z2sqr.append((z2[i] ** 2))
        z4sqr.append(abs(z1[i] * z2[i]))
    v1 = (1.4826 * stats.median_absolute_deviation(data[:, 0])) ** 2
    v2 = (1.4826 * stats.median_absolute_deviation(data[:, 1])) ** 2
    v4 = np.sqrt(np.sum(z4sqr) / (Size - 1))
    v3 = v4
    Z = [[v1, v3], [v4, v2]]
    return Mean, Z


def s_Sn(data):
    med1 = []
    med2 = []
    z1 = []
    z2 = []
    z4sqr = []
    Mean = [[np.average(data[:, 0]), np.average(data[:, 1])]]
    Size = np.size(data, axis=0)
    m1 = Mean[0][0]
    m2 = Mean[0][1]
    for i in range(Size):
        z1.append(data[i][0] - m1)
        z2.append(data[i][1] - m2)
        z4sqr.append(abs(z1[i] * z2[i]))
    v4 = np.sqrt(np.sum(z4sqr) / (Size - 1))
    v3 = v4
    for i in range(Size):
        diff1 = []
        diff2 = []
        for j in range(Size):
            diff1.append(abs(data[i][0] - data[j][0]))
            diff2.append(abs(data[i][1] - data[j][1]))
        med1.append(np.median(diff1))
        med2.append(np.median(diff2))
    v1 = (1.1926 * np.median(med1)) ** 2
    v2 = (1.1926 * np.median(med2)) ** 2
    Z = [[v1, v3], [v4, v2]]
    return Mean, Z


def s_Qn(data):
    diff1 = []
    diff2 = []
    z1 = []
    z2 = []
    z4sqr = []
    Mean = [[np.average(data[:, 0]), np.average(data[:, 1])]]
    Size = np.size(data, axis=0)
    m1 = Mean[0][0]
    m2 = Mean[0][1]
    for i in range(Size):
        z1.append(data[i][0] - m1)
        z2.append(data[i][1] - m2)
        z4sqr.append(abs(z1[i] * z2[i]))
    v4 = np.sqrt(np.sum(z4sqr) / (Size - 1))
    v3 = v4
    for i in range(0, Size):
        for j in range(0, Size):
            if i < j:
                diff1.append(abs(data[i][0] - data[j][0]))
                diff2.append(abs(data[i][1] - data[j][1]))
    diff1.sort()
    diff2.sort()
    h = int(math.floor(Size / 2) + 1)
    k = int((h * (h - 1)) / 2)
    v1 = (2.2219 * diff1[k - 1]) ** 2
    v2 = (2.2219 * diff2[k - 1]) ** 2
    Z = [[v1, v3], [v4, v2]]
    return Mean, Z