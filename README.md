---

# Stock Price Prediction Using LSTM

This project demonstrates how to predict stock prices using a Long Short-Term Memory (LSTM) neural network. The dataset used is the Google stock price data.

## Table of Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Testing the Model](#testing-the-model)
- [Results](#results)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction

This project uses an LSTM neural network to predict the stock prices of Google. The model is trained on historical stock price data and aims to predict future stock prices.

## Prerequisites

- Python 3.x
- Libraries:
  - numpy
  - pandas
  - matplotlib
  - scikit-learn
  - keras

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/SUBHODIP200/stock-price-prediction-.git
    cd stock-price-prediction-
    ```

2. Install the required libraries:
    ```sh
    pip install numpy pandas matplotlib scikit-learn keras
    ```

## Dataset

The dataset used for this project is Google stock price data. Make sure you have `Google_train_data.csv` and `Google_test_data.csv` in the project directory.

## Model Architecture

The model consists of three LSTM layers with dropout regularization to prevent overfitting, and a Dense output layer to predict the stock prices.

## Training the Model

1. Load and preprocess the training data:
    ```python
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, Dropout

    data = pd.read_csv('Google_train_data.csv')
    data["Close"] = pd.to_numeric(data.Close, errors='coerce')
    data = data.dropna()
    trainData = data.iloc[:, 4:5].values

    sc = MinMaxScaler(feature_range=(0, 1))
    trainData = sc.fit_transform(trainData)

    X_train = []
    y_train = []

    for i in range(60, len(trainData)):
        X_train.append(trainData[i-60:i, 0])
        y_train.append(trainData[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    ```

2. Build and train the model:
    ```python
    model = Sequential()

    model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    hist = model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=2)
    ```

3. Plot the training loss:
    ```python
    import matplotlib.pyplot as plt

    plt.plot(hist.history['loss'])
    plt.title('Training model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()
    ```
    ![stock2](https://github.com/user-attachments/assets/65d14b21-ee7e-4bc8-b9eb-230ca49b6e32)

## Testing the Model

1. Load and preprocess the test data:
    ```python
    testData = pd.read_csv('Google_test_data.csv')
    testData["Close"] = pd.to_numeric(testData.Close, errors='coerce')
    testData = testData.dropna()
    testData = testData.iloc[:, 4:5]
    y_test = testData.iloc[60:, 0:].values

    inputClosing = testData.iloc[:, 0:].values
    inputClosing_scaled = sc.transform(inputClosing)

    X_test = []
    for i in range(60, len(testData)):
        X_test.append(inputClosing_scaled[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    ```

2. Predict and plot the results:
    ```python
    predicted_price = model.predict(X_test)
    predicted_price = sc.inverse_transform(predicted_price)

    plt.plot(y_test, color='red', label='Actual Stock Price')
    plt.plot(predicted_price, color='green', label='Predicted Stock Price')
    plt.title('Google stock price prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
    ```
    
    ![stock1](https://github.com/user-attachments/assets/cd8d2aa3-7501-4a12-bb55-acca003e0c56)

3. Calculate the R2 score:
    ```python
    from sklearn.metrics import r2_score

    r2_score(y_test, predicted_price)
    ```

## Results

The results include a plot showing the actual vs. predicted stock prices and the R2 score to evaluate the model's performance.

## Conclusion

This project demonstrates the use of LSTM networks for time series prediction of stock prices. The model can be further improved with more data and hyperparameter tuning.

## References

- [Keras Documentation](https://keras.io/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

---

