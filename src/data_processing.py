import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the Iris dataset from the URL
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Split the dataset into training and inference sets
train, inference = train_test_split(df, test_size=0.2, random_state=42)

# Save the datasets to the 'data/' directory
train.to_csv("data/train.csv", index=False)

inference.drop(columns=['species'], inplace=True)
inference.to_csv("data/inference.csv", index=False)

print("Data processing complete! Files saved to the 'data/' directory.")