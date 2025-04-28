# import pandas as pd

# df=pd.read_csv("D:\langchain_ai_bot\data\cleaned.csv")
# print(df.columns)
# print(max(df['support_length']))
# print(df['support_length'].median())

# # Filter rows where 'support_reply' is "big" (e.g., > 300 characters)
# big_reply_df = df[df['support_reply'].str.len() > 100]

# print(big_reply_df.iloc[3]['support_reply'])

# # Create a new DataFrame with only the concatenated column
# new_df = pd.DataFrame({
#     'combined': 'customer: ' + df['customer_text'] + ' support assist: ' + df['support_reply']
# })

# # Display the new DataFrame
# new_df.to_csv("D:\langchain_ai_bot\data\cleaned_combined.csv")



import pandas as pd

# Load the dataset
df = pd.read_csv(r"D:\langchain_ai_bot\data\cleaned.csv")
print(df.columns)
print(len(df))
# Calculate the support length (if not already calculated)
df['support_length'] = df['support_reply'].str.len()

# Check max and median support lengths
print(f"Max support length: {df['support_length'].max()}")
print(f"Median support length: {df['support_length'].median()}")

# Filter rows where 'support_length' > 100
big_reply_df = df[df['support_length'] > 100]

# Randomly select 20 rows from those filtered
random_20_big_reply_df = big_reply_df.sample(n=20, random_state=42)

# Keep the remaining rows where 'support_length' > 100, excluding the 20 rows sampled
remaining_big_reply_df = df.drop(random_20_big_reply_df.index)
# print(len(remaining_big_reply_df))
# print(random_20_big_reply_df)
# Create a new DataFrame with only the concatenated column
new_df = pd.DataFrame({
    'combined': 'customer: ' + remaining_big_reply_df['customer_text'] + ' support assist: ' + remaining_big_reply_df['support_reply']
})

# Display the new DataFrame
new_df.to_csv("D:\langchain_ai_bot\data\cleaned_combined.csv")
random_20_big_reply_df.to_csv("D:\langchain_ai_bot\data\sample_questions.csv")

