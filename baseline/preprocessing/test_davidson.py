import pandas as pd
from preprocess import preprocess_text


# Load Davidson dataset
df = pd.read_csv("baseline/data/davidson_dataset.csv")


# Test first 5 tweets
for i in range(5):

    original = df.loc[i, "tweet"]
    processed = preprocess_text(original)

    print("=" * 60)

    print("Original:")
    print(original)

    print("\nProcessed:")
    print(processed)

    print("\nLabel:")
    print(df.loc[i, "class"])