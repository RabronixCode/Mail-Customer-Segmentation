import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping


df = pd.read_csv("./data/daily-foreign-exchange-rates-31-.csv")
df["Date"] = pd.to_datetime(df["Date"])

full_date_range = pd.date_range(start=df["Date"].min(), end=df["Date"].max())
df_full = df.set_index("Date").reindex(full_date_range)
df_full.reset_index()
df_full.rename(columns={"index": "Date"}, inplace=True)


df_handled = df_full.copy()
df_handled["Exchange Rate"] = df_full["Exchange Rate"].interpolate(method='linear')

decomposition = seasonal_decompose(df_handled["Exchange Rate"], model='additive', period=365)
decomposition.plot()
#plt.show()


# Autocorrelation plot
plot_acf(df_handled['Exchange Rate'], lags=40)
#plt.show()

# Partial Autocorrelation plot
plot_pacf(df_handled['Exchange Rate'], lags=40)
#plt.show()

scaler = MinMaxScaler(feature_range=(0, 1))
df_handled['Exchange Rate'] = scaler.fit_transform(df_handled[['Exchange Rate']])

def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

time_steps = 2
X, y = create_sequences(df_handled['Exchange Rate'].values, time_steps)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape input data
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.1))
model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.1))
model.add(Dense(units=1))

optimizer = Adam(learning_rate=0.001)

# Implement Early Stopping
early_stopping = EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)

# Compile the model
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, callbacks=[early_stopping])

# Make predictions
predictions = model.predict(X_test)

# Inverse transform the predictions and the actual values
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Compare predictions and actual values
print(predictions[:5], y_test[:5])

# Plot predictions vs actual values
plt.figure(figsize=(10, 6))
plt.plot(y_test, color='blue', label='Actual Exchange Rate')
plt.plot(predictions, color='red', label='Predicted Exchange Rate')
plt.title('Exchange Rate Prediction')
plt.xlabel('Time')
plt.ylabel('Exchange Rate')
plt.legend()
plt.show()

# Get the last 'look_back' number of data points from the original series
recent_data = df_handled[-time_steps:]

recent_data = np.reshape(recent_data, (1, time_steps, 1))

# Number of future predictions to make
n_future_predictions = 5

# Store predictions
future_predictions = []

# Start with the most recent data (last 'look_back' number of data points)
input_sequence = recent_data  # Use the prepared recent_data sequence

for _ in range(n_future_predictions):
    # Predict the next value
    predicted_value = model.predict(input_sequence)
    
    # Reshape the predicted value to match the input sequence shape
    predicted_value_reshaped = np.reshape(predicted_value, (1, 1, 1))
    
    # Update the input sequence: remove the first value and add the predicted value
    input_sequence = np.append(input_sequence[:, 1:, :], predicted_value_reshaped, axis=1)
    
    # Store the prediction
    future_predictions.append(predicted_value[0][0])

# Inverse transform all predictions to get the original scale
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Print the future predictions
print(f"Future Predictions: {future_predictions.flatten()}")
