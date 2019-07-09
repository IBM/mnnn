from sklearn.cluster import KMeans
import numpy as np
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


def kmeanInfer(a):

    model = joblib.load("modelFood/100c_300000s_kmean_angle_model.pkl")
    # print("clustering", model.cluster_centers_.shape)
    # print("labels", model.labels_.shape)
    pred_X = model.predict(a)+1
    return pred_X