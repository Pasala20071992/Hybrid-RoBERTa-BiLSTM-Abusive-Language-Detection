import pandas as pd
from preprocess import preprocess_text


# Load original Davidson dataset
input_file = "baseline/data/davidson_dataset.csv"

df = pd.read_csv(input_file)

print("Original dataset shape:", df.shape)


# Apply preprocessing to every tweet
print("\nProcessing tweets...")

df["processed_text"] = df["tweet"].apply(preprocess_text)


# Save processed dataset
output_file = "baseline/data/davidson_processed.csv"

df.to_csv(output_file, index=False)

print("\nProcessing completed.")
print("Processed dataset shape:", df.shape)
print("Saved to:", output_file)


# Display first 5 processed rows
print("\nFirst 5 processed tweets:")

for i in range(5):

    print("\nOriginal:")
    print(df.loc[i, "tweet"])

    print("Processed:")
    print(df.loc[i, "processed_text"])

    print("Label:", df.loc[i, "class"])