import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the dataset into a Pandas DataFrame
df = pd.read_csv("2010.csv")

# Define the features and target variable
X = df[["bpm", "nrgy", "dnce", "dB", "live", "val", "dur", "acous", "spch"]]
y = df["pop"]

# Train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Define the features of the new song
new_song_features = [120, 80, 70, -6, 10, 50, 240, 20, 5]  # replace with your own values

# Make a prediction on the new song
new_song_popularity = model.predict([new_song_features])[0]

print("Predicted popularity of the new song:", new_song_popularity)
