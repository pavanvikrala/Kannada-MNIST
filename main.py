# Importing all the necessary modules and libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier


def evaluate_model(x_train, x_test, ytrain, user_choice):
    """This will create a required model"""
    if user_choice == "Decision Trees":
        # Creating Decision Tree Model
        model = DecisionTreeClassifier()

    elif user_choice == "Random Forests":
        model = RandomForestClassifier(n_estimators=100)
    elif user_choice == "Naive-Bayes Model":
        model = GaussianNB()
    elif user_choice == "KNN Classifier":
        model = KNeighborsClassifier(n_neighbors=5)
    elif user_choice == "Support Vector Matrix(SVM)":
        model = SVC(kernel='rbf', probability=True)
    else:
        return None

    model.fit(x_train, ytrain)
    # Make predictions on test data
    y_predict = model.predict(x_test)
    return y_predict, model


def report(y, ypred):
    """This will show us all the required evaluation metrics like Precision, recall, f1-score"""
    cls_report = classification_report(y, ypred, output_dict=True)
    df = pd.DataFrame(cls_report).transpose()
    st.success("Here's the evaluation metrics")
    st.info(f"Classification Report for the selected model👇")
    st.table(df)


def matrix(y, ypred):
    """This will show us the confusion matrix in the form of a seaborn heatmap"""
    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.set()
    cm = confusion_matrix(y, ypred)
    sns.heatmap(data=cm, annot=True, fmt='d', cmap='rocket')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    st.pyplot()


img_url = 'https://res.cloudinary.com/df7cbq3qu/image/upload/v1690433388/Science-word-start-with-k_ymeoyt.jpg'
page_config = {"page_title": 'Kannada MNIST', "page_icon": img_url}
st.set_page_config(**page_config)

st.title("Kannada MNIST - Classification Problem")
# Changing the directory and loading the data
os.chdir("C:/Users/pavan/Kannada_MNIST_datataset_paper/Kannada_MNIST_npz/Kannada_MNIST")

# Loading the data
X_train = np.load("X_kannada_MNIST_train.npz")["arr_0"]
X_test = np.load("X_kannada_MNIST_test.npz")["arr_0"]
y_train = np.load("y_kannada_MNIST_train.npz")["arr_0"]
y_test = np.load("y_kannada_MNIST_test.npz")["arr_0"]

# Preprocess the data
# Normalize the image to [0,1]

X_train_normalized = X_train.astype(np.float32)/255
X_test_normalized = X_test.astype(np.float32)/255

# Reshaping the data to 2d

X_train_reshaped = X_train_normalized.reshape(X_train_normalized.shape[0], 28*28)
X_test_reshaped = X_test_normalized.reshape(X_test_normalized.shape[0], 28*28)

values = ['Select', 10, 15, 20, 25, 30]
default_ix = values.index(10)
components = st.selectbox("Select the number of components you'd like to perform PCA", values, index=default_ix)
# Performing Principal Component Analysis(PCA) to the components
if components:
    pca = PCA(n_components=components)
    X_train_pca = pca.fit_transform(X_train_reshaped)
    X_test_pca = pca.transform(X_test_reshaped)

    st.info("Now select a machine learning model you wish to apply")
    choice = st.selectbox("Here👇", (" ", "Decision Trees", "Random Forests", "Naive-Bayes Model",
                                    "KNN Classifier", "Support Vector Matrix(SVM)"))
    if choice != " ":
        y_pred_test, selected_model = evaluate_model(x_train=X_train_pca, x_test=X_test_pca,
                                                     ytrain=y_train, user_choice=choice)
        report(y_test, y_pred_test)
        matrix(y_test, y_pred_test)

        # AUC/RUC curves
        # Performing one-hot encoding  on target labels

        y_train_encoded = pd.get_dummies(y_train).values.astype(float)
        y_test_encoded = pd.get_dummies(y_test).values.astype(float)

        model_onevrest = OneVsRestClassifier(selected_model)
        model_onevrest.fit(X_train_pca, y_train_encoded)

        # Making predictions on test data
        y_pred = model_onevrest.predict_proba(X_test_pca)

        # ROC and AUC curves for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for cls in range(10):
            fpr[cls], tpr[cls], _ = roc_curve(y_test_encoded[:, cls], y_pred[:, cls])
            roc_auc[cls] = auc(fpr[cls], tpr[cls])

        fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(15, 10))

        for i, ax in enumerate(axs.flat):
            if i < 10:
                ax.plot(fpr[i], tpr[i], color="orange", lw=2, label="ROC")
                ax.plot([0, 1], [0, 1], color="darkblue", linestyle="--", label="Guessing")
                ax.set(xlim=[-0.09, 1], ylim=[0, 1.02],
                       xlabel='False Positive Rate (fpr)',
                       ylabel='true Positive Rate (tpr)')
                ax.set_title(f'Class{i}')
                ax.legend()

        plt.tight_layout(rect=[0, 0.01, 1, 0.95])
        st.pyplot(fig)
