import pandas as pd
import numpy as np

# EWTOPSIS （Combined Entropy Weight and Technique for Order Preference by Similarity to an Ideal Solution ）

# Standardization of reverse indicators
def normalization_1(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

# Standardization of positive indicators
def normalization_2(data):
    _range = np.max(data) - np.min(data)
    return (np.max(data) - data) / _range

#Entropy weight method for calculating weights
def entropWeight(data):
    # print(data)
    P = np.array(data.T) / np.sum(data, axis=1)
    # To calculate the entrop
    E = np.nansum(-P * np.log(P + 1e-5) / np.log(len(data.T)), axis=0)
    # To calculate weight coefficients
    return (1 - E) / (1 - E).sum()


def topsis(data, weight=None):
    weight = entropWeight(data) if weight is None else np.array(weight)  # weights

    # 1) To find positive and negative ideal solutions (assuming both are positive)
    Z_p = pd.DataFrame([(data.T * weight).T.max(axis = 1)], index=['Positive ideal solution'])
    Z_m = pd.DataFrame([(data.T * weight).T.min(axis = 1)], index=['Negative ideal solution'])
    # expand the size of array
    m = np.repeat(([Z_p.loc['Positive ideal solution', :]]), len(data.T), axis=0)
    n = np.repeat(([Z_m.loc['Negative ideal solution', :]]), len(data.T), axis=0)

    # 2) To calculate the distance from each object to positive and negative ideal solutions
    Result = pd.DataFrame(data.copy(), index=['obj_Bionic_beams_Mass', 'obj_Directional_Deformation_Reported_Frequency']).transpose()
    Result['Positive ideal solution'] = np.sqrt(((data.T - m) ** 2).sum(axis=1))
    #print(Result['Positive ideal solution'])
    Result['Negative ideal solution'] = np.sqrt(((data.T - n) ** 2).sum(axis=1))
    #print(Result['Negative ideal solution'])

    # 3) To calculate the comprehensive score indexes and sort it
    Result['Comprehensive score indexes'] = Result['Negative ideal solution'] / (Result['Negative ideal solution'] + Result['Positive ideal solution'])
    #print(Result['Comprehensive score indexes'])
    Result['Sorting'] = Result['Comprehensive score indexes'].rank(method='first',ascending=False)
    #print(Result['Sorting'])
    return Result, Z_p, weight


if __name__ == '__main__':
    data = pd.read_csv('Case III pt frontiers.csv', sep=',', header=0)
    data1 = np.array(data.copy()).T
    for i in range(0, len(data1)):
        data1[i] = normalization_2(data1[i])
    # print(data1[1])
    [result, z1, weight] = topsis(data1)
    # The score for each FP (frontier point)
    print("Score for each frontier point：",(np.array(data.copy())*weight).sum(axis=1))
    print(result.head(12))  # eight points
    # print(z1)
    print("Weights: ",weight)
