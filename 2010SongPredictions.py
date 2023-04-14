import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load the dataset into a Pandas DataFrame
df = pd.read_csv("2010.csv")

# Define the features and target variable
X = df[["bpm", "nrgy", "dnce", "dB", "live", "val", "dur", "acous", "spch"]]
y = df["pop"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate the R-squared score of the model
r2 = r2_score(y_test, y_pred)
print("R-squared score: {:.2f}".format(r2))
