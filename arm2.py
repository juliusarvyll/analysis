# Install the required library if not already installed
# pip install mlxtend

import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Load the dataset
# Replace 'test_venue_ratings.csv' with your file path
file_path = "generated_ratings_data.csv"
data = pd.read_csv(file_path)

# Step 1: Convert features to binary (1 for high rating, 0 for low rating)
# Define a threshold for "high" ratings
threshold = 3
binary_data = data.applymap(lambda x: 1 if x >= threshold else 0)

# Step 2: Apply FP-Growth to find frequent itemsets
frequent_itemsets = fpgrowth(binary_data, min_support=0.2, use_colnames=True)

# Step 3: Generate association rules from the frequent itemsets
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

# Step 4: Display the results
print("Frequent Itemsets:")
print(frequent_itemsets.sort_values(by='support', ascending=False))

print("\nAssociation Rules:")
print(rules.sort_values(by='confidence', ascending=False))
