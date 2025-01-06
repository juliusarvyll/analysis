import pandas as pd
import numpy as np
from niaarm import Dataset, NiaARM
from niapy.task import Task, OptimizationType
from niapy.algorithms.basic import DifferentialEvolution
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

# Read the data
df = pd.read_csv('generated_ratings_data.csv')

# Convert ratings to categorical format
# We'll create binary features for each rating level
def create_binary_features(df):
    binary_df = pd.DataFrame()
    for column in df.columns:
        unique_values = sorted(df[column].unique())
        for value in unique_values:
            binary_df[f'{column}_{value}'] = (df[column] == value).astype(int)
    return binary_df

# Transform the data
binary_df = create_binary_features(df)

# Create NIAARM dataset
data = Dataset(binary_df)

# Initialize NIAARM problem
problem = NiaARM(
    data.dimension,
    data.features,
    data.transactions,
    metrics=('support', 'confidence'),
    logging=True
)

# Create optimization task
task = Task(
    problem=problem,
    max_iters=30,
    optimization_type=OptimizationType.MAXIMIZATION
)

# Initialize Differential Evolution algorithm
algo = DifferentialEvolution(
    population_size=50,
    differential_weight=0.5,
    crossover_probability=0.9
)

# Run the algorithm
best = algo.run(task=task)

# Sort rules by support and confidence
problem.rules.sort()  # This sorts by all metrics

# Prepare data for plotting
rules_data = []
for i, rule in enumerate(problem.rules[:10]):
    antecedent_str = ' AND '.join([str(feature) for feature in rule.antecedent])
    consequent_str = ' AND '.join([str(feature) for feature in rule.consequent])
    interpretation = (
        f"If {antecedent_str}, then {consequent_str}. "
        f"This rule appears in {rule.support * 100:.2f}% of transactions and has a "
        f"confidence of {rule.confidence * 100:.2f}% that the consequent occurs."
    )

    rules_data.append({
        'antecedent': antecedent_str,
        'consequent': consequent_str,
        'support': rule.support,
        'confidence': rule.confidence,
        'interpretation': interpretation
    })

# Create a Tkinter GUI for visualization
def show_plot():
    supports = [rule['support'] * 100 for rule in rules_data]
    confidences = [rule['confidence'] * 100 for rule in rules_data]
    rule_labels = [f"Rule {i+1}" for i in range(len(rules_data))]

    fig, ax = plt.subplots()
    ax.bar(rule_labels, supports, label='Support (%)', alpha=0.7, color='blue')
    ax.plot(rule_labels, confidences, label='Confidence (%)', marker='o', color='red')

    ax.set_xlabel('Rules')
    ax.set_ylabel('Percentage')
    ax.set_title('Top 10 Association Rules')
    ax.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Create a Tkinter app
root = tk.Tk()
root.title("Association Rules Visualization")

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Add a button to show the plot
plot_button = ttk.Button(frame, text="Show Rules Plot", command=show_plot)
plot_button.grid(row=0, column=0, padx=5, pady=5)

# Add a button to show rule interpretations
interpretation_text = "\n".join([f"Rule {i+1}: {rule['interpretation']}" for i, rule in enumerate(rules_data)])

def show_interpretations():
    messagebox.showinfo("Rule Interpretations", interpretation_text)

interpretation_button = ttk.Button(frame, text="Show Interpretations", command=show_interpretations)
interpretation_button.grid(row=1, column=0, padx=5, pady=5)

# Run the Tkinter main loop
root.mainloop()

# Export rules with interpretation to CSV
pd.DataFrame(rules_data).to_csv('association_rules_with_interpretation.csv', index=False)

print("\nAssociation rules with interpretations have been saved to 'association_rules_with_interpretation.csv'.")
