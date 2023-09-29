import math

import pandas as pd
import numpy as np
import streamlit as st

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as scst
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler


st.title("Логистическая регрессия")
st.markdown("---")

try:
    train = pd.read_csv(st.file_uploader("Загрузите свой файл", type = "csv"));

    if train is not None:
        st.dataframe(train)

        ss = StandardScaler()

        columns_to_normalize = st.multiselect('Выберите колонки для фичей:', train.columns)

        train[columns_to_normalize] = ss.fit_transform(train[columns_to_normalize])

        def sigmoid(x):
            return 1/(1+np.exp(-x))

        class LogReg:
            def __init__(self, learning_rate):
                self.learning_rate = learning_rate
                        
            def fit(self, X, y):
                                
                X = np.array(X)
                
                self.coef_ = np.random.normal(size=X.shape[1])
                self.intercept_ = np.random.normal()

                n_epochs = 1000
                for epoch in range(n_epochs):
                    
                    y_pred = self.intercept_ + X@self.coef_
                    predictions = sigmoid(y_pred)
                    error = (y - predictions)

                    w0_grad = -1 * error 
                    w_grad = -1 * X * error.reshape(-1, 1)

                    self.coef_ = self.coef_ - self.learning_rate * w_grad.mean(axis=0)
                    self.intercept_ = self.intercept_ - self.learning_rate * w0_grad.mean()

            def predict(self, X):
                X = np.array(X)
                y_pred = sigmoid(X@self.coef_ + self.intercept)
                class_pred = [0 if y<=0.5 else 1 for y in y_pred]
                return class_pred
            
        logreg = LogReg(0.1)

        logreg.fit(train[columns_to_normalize], train.iloc[:,-1].to_numpy())

        i = 1
        for item in logreg.coef_:
            st.write(f'Коэффициент №{i}: {item}')
            i+=1
        st.write(f'Свободный член: {logreg.intercept_}')

        st.title('Графики для двух фичей')
        fig, ax = plt.subplots()

        columns_to_draw = st.multiselect('Выберите фичи для постройки графиков:', train.columns)
     
        # Scatter plot
        st.subheader('Scatter Plot')
        plt.figure(figsize=(8, 6))
        ax.scatter(train[columns_to_draw[0]], train[columns_to_draw[1]])
        plt.xlabel(f'{columns_to_draw[0]}')
        plt.ylabel(f'{columns_to_draw[1]}')
        plt.title('Scatter Plot')
        st.pyplot(fig)

        # Bar plot
        st.subheader('Bar Plot')
        plt.figure(figsize=(8, 6))
        ax.bar(train[columns_to_draw[0]], train[columns_to_draw[1]])
        plt.xlabel(f'{columns_to_draw[0]}')
        plt.ylabel(f'{columns_to_draw[1]}')
        plt.title('Bar Plot')
        st.pyplot(fig)

        # Line plot
        st.subheader('Line Plot')
        plt.figure(figsize=(8, 6))
        ax.plot(train[columns_to_draw[0]], train[columns_to_draw[1]])
        plt.xlabel(f'{columns_to_draw[0]}')
        plt.ylabel(f'{columns_to_draw[1]}')
        plt.title('Line Plot')
        st.pyplot(fig)

except Exception as e:
    pass