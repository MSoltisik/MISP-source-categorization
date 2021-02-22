import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objs as go

import os
import sys

# statistical method classifiers
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import BayesianRidge

# ML method classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# Tools for dimension reduction
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NeighborhoodComponentsAnalysis

from collections import Counter

# path to the dataset file	
FILE_PATH = 'combine.csv'
FILE_PATH_TEST = 'misp.event.846.csv' 

# UI widgets for selecting the dataset size
dataset_size = st.sidebar.slider("Dataset Size", min_value = 1, max_value = 120000, value = 5000)
test_set_ratio = st.sidebar.slider("Test Set Ratio", min_value = 0.1, max_value = 0.9, value = 0.2)

st.write("Upload own MISP event data for testing:")
uploaded_file = st.file_uploader("Upload Files",type=['csv'])

# UI selection for the ML method
selected_method = st.sidebar.selectbox("Evaluation Method", ("Linear Regression", "Lasso", "Bayesian Ridge Regression", "KNN (K-Neighbors)", "SVM (Support Vector)", "Random Forest"), index=5)

# Creating the UI widgets for manipulating algorithm parameters
def show_parameter_ui(method_name):
    params = dict()
    
    # Lasso method
    if (method_name == "Lasso"):
    	alpha = st.sidebar.slider("Alpha", min_value = 0.01, max_value = 1.0, value = 1.0)																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																					
    	params["alpha"] = alpha
    
    # Bayesian Ridge Regression method
    elif (method_name == "Bayesian Ridge Regression"):
    	lambda_init = st.sidebar.slider("Initial Lambda", min_value = 0.01, max_value = 3.0, value = 1.0)
    	alpha_init = st.sidebar.slider("Initial Alpha", min_value = 0.01, max_value = 3.0, value = 1.0)
    	params["lambda_init"] = lambda_init
    	params["alpha_init"] = alpha_init

    # KNN method
    elif (method_name == "KNN (K-Neighbors)"):
        K = st.sidebar.slider("K", min_value = 1, max_value = 15, value = 1)
        params["K"] = K

    # SVM method
    elif (method_name == "SVM (Support Vector)"):
        C = st.sidebar.slider("C", min_value = 0.01, max_value = 10.0)
        params["C"] = C

    # Random Forest method
    elif (method_name == "Random Forest"):
        max_depth = st.sidebar.slider("Max Depth", min_value = 2, max_value = 15)
        n_estimators = st.sidebar.slider("Number of Estimators", min_value = 1, max_value = 100)
        random_seed = st.sidebar.slider("Random Seed", min_value = 1, max_value = 99999)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
        params["random_seed"] = random_seed

    return params

params = show_parameter_ui(selected_method)

# Returning the classifier depending on the chosen ML method
def get_classifier(method_name, params):
    # Linear Regression
    if (method_name == "Linear Regression"):
    	clf = LinearRegression()
    	
    # Lasso
    elif (method_name == "Lasso"):
    	clf = Lasso(alpha = params["alpha"])
    	
    # Bayesian Ridge Regression
    elif (method_name == "Bayesian Ridge Regression"):
    	clf = BayesianRidge(alpha_init = params["alpha_init"], lambda_init = params["lambda_init"]	)
    	
    # KNN method
    elif (method_name == "KNN (K-Neighbors)"):
        clf = KNeighborsClassifier(n_neighbors = params["K"])

    # SVM method
    elif (method_name == "SVM (Support Vector)"):
        clf = SVC(C = params["C"])

    # Random Forest method
    else:
        clf = RandomForestClassifier(max_depth = params["max_depth"], n_estimators = params["n_estimators"], random_state = params["random_seed"])

    return clf

clf = get_classifier(selected_method, params)

# Loads n rows of the data set
def load_data(nrows, full_path):
    data = pd.read_csv(full_path, nrows=nrows)
    return data

full_path_data = os.path.join(os.path.dirname(sys.path[0]), FILE_PATH)
full_path_testing = os.path.join(os.path.dirname(sys.path[0]), FILE_PATH_TEST)

data = load_data(dataset_size, full_path_data)
data_testing = pd.read_csv(uploaded_file, nrows=dataset_size) if (uploaded_file != None) else load_data(dataset_size, full_path_testing)

# Displaying the data
st.write("Base Data: ", data.shape)
st.write(data)

st.write("Testing Data: ", data_testing.shape)
st.write(data_testing)

# getting a set of all possible tags
tags = set()
for entry in data["event_tag"]:
    entry_tags = entry.split(",")
    tags.update(entry_tags)

tags_testing = set()
for entry in data_testing["event_tag"]:
    entry_tags = entry.split(",")
    tags_testing.update(entry_tags)

tags.remove("event_tag")

tags_present_original = data["event_tag"].str.split(",").tolist()
tags_present_testing = data_testing["event_tag"].str.split(",").tolist()

for tag in tags:
    contains_tag_original = [(tag in x) for x in tags_present_original]
    contains_tag_testing = [(tag in x) for x in tags_present_testing]

    data[tag] = contains_tag_original
    data_testing[tag] = contains_tag_testing

# Encoding non-numerical data:
le = {}
for column_name in data.head(1):
    values = data[column_name].append(data_testing[column_name])

    le[column_name] = preprocessing.LabelEncoder()
    le[column_name].fit(values.astype(str))

    data[column_name] = le[column_name].transform(data[column_name].astype(str))
    data_testing[column_name] = le[column_name].transform(data_testing[column_name].astype(str))

# overall correct/incorrect guesses on the original data, used for determining overall method reliability
guesses_correct = 0
guesses_incorrect = 0

# overall positive/negative guesses for each of the tags on the testing data
guesses_testing = pd.DataFrame(columns=['Tag', 'Positive', 'Negative', 'Confidence', 'Is Tagged As'])

# doing prediction on every tag
for tag in tags:      
    # splitting labels and features
    labels_original = data[tag]
    features_original = data.drop([tag, "event_tag"], axis = 1)

    labels_testing = data_testing[tag]
    features_testing = data_testing.drop([tag, "event_tag"], axis = 1)

    # creating train and test set on the original data
    X_train, X_test, y_train, y_test = train_test_split(features_original, labels_original, test_size = test_set_ratio, random_state = 1234)

    # Training
    model = clf.fit(X_train, y_train)

    # Making predictions on both test set fromn the original, and on the testing data
    y_pred = clf.predict(X_test)
    y_pred_rounded = [round(num) for num in y_pred]

    y_pred_new = clf.predict(features_testing)
    y_pred_new_rounded = [round(num) for num in y_pred_new]

    # Decoding the data
    result_original = X_test
    for column_name in result_original.columns:
        result_original[column_name] = le[column_name].inverse_transform(result_original[column_name])

    result_testing = features_testing
    for column_name in result_testing.columns:
        result_testing[column_name] = le[column_name].inverse_transform(result_testing[column_name])

    # adding total/correct guesses on the original data for purposes of calculating overall reliability
    tag_original = le[tag].inverse_transform(y_test)
    tag_prediction = le[tag].inverse_transform(y_pred)

    guesses_correct += (tag_original == tag_prediction).sum()
    guesses_incorrect += (tag_original != tag_prediction).sum()

    # checking the positive/negative guesses for this tag on the testing data
    guesses_positive = y_pred_new.sum()
    guesses_negative = y_pred_new.size - guesses_positive
    guesses_confidence = round(guesses_positive * 100    / (guesses_positive + guesses_negative), 2)

    # adding the values to the result dataframe
    new_row = {'Tag' : tag, 'Positive' : guesses_positive, 'Negative' : guesses_negative, "Confidence" : guesses_confidence, "Is Tagged As" : tag in tags_testing}
    guesses_testing = guesses_testing.append(new_row, ignore_index = True)

guesses_testing = guesses_testing.sort_values(by = 'Confidence', ascending = False)
st.write("Tags estimated (based on model data):")
st.write(guesses_testing)

st.write("All tags in the tested data: ", tags_testing)

acc = guesses_correct / (guesses_correct + guesses_incorrect)
st.write("Method accuracy on original data: ", acc)
