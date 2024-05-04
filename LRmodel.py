import pandas as pd # Importing the pandas library under the alias pd
from sklearn.model_selection import train_test_split # Importing the train_test_split function from the sklearn.model_selection module
from sklearn.linear_model import LinearRegression # Importing the LinearRegression class from the sklearn.linear_model module
import matplotlib.pyplot as plt # Importing the pyplot module from the matplotlib library under the alias plt

data = pd.read_csv('HousingPriceData.csv') # Reading the data from the Housing_Price_Data.csv file and storing it in the data variable

x = data[['area', 'bedrooms', 'bathrooms', 'stories']]
y = data[['price']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) # Splitting the data into training and testing sets

# Define a new linear retrogression model
# This model will be used to learn the relationship between our independent and dependent variables. 
# The model is then trained based on the training data (x_train, y_train).
model = LinearRegression() 
model.fit(x_train, y_train)

# Predicts the prices from the testing set and stores the results in y_pred.
y_pred = model.predict(x_test)

#Visualization of Model

plt.figure(figsize=(10, 6)) # Initialize a 10 by 6 graph
plt.scatter(y_test, y_pred, color='pink') # Scatter plot of the actual prices against the predicted prices
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)  # Plotting the line of best fit

plt.title('Housing Price Predictions') # Title of the graph
plt.xlabel('Actual Price') # Label for the x-axis
plt.ylabel('Predicted Price') # Label for the y-axis
plt.ticklabel_format(style='plain', axis='both') # Formatting the axis labels
plt.show() # Display the graph
