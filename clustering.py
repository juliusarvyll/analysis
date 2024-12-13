import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

# Read the CSV file
df = pd.read_csv('evaluation_data_New event today (1).csv')

# Get the question columns (excluding ID and Timestamp)
question_columns = [
    'The resource speakers were knowledgeable and delivered the topics effectively.',
    'The venue was fit for the activity.',
    'The schedule of the program/activity was favorable to the participants and speakers.',
    'The event met my expectations.',
    'The activities were engaging and relevant.',
    'I would recommend this event to others. Because it is very nice'
]

# Create a long format dataframe for easier analysis
df_long = pd.melt(df, 
                  value_vars=question_columns,
                  var_name='Question',
                  value_name='Score')

# Basic Statistical Analysis
print("=== DETAILED STATISTICAL ANALYSIS ===\n")

# 1. Overall Summary Statistics
print("1. OVERALL SUMMARY STATISTICS:")
summary_stats = df_long.groupby('Question')['Score'].agg([
    'count', 'mean', 'std', 'min', 'max',
    lambda x: x.quantile(0.25),
    lambda x: x.quantile(0.50),
    lambda x: x.quantile(0.75)
]).round(2)
summary_stats.columns = ['Count', 'Mean', 'Std Dev', 'Min', '25th', 'Median', '75th', 'Max']
print(summary_stats)

# 2. Response Distribution
print("\n2. RESPONSE DISTRIBUTION:")
response_dist = pd.crosstab(df_long['Question'], df_long['Score'], normalize='index') * 100
print("\nPercentage distribution of scores for each question:")
print(response_dist.round(2))

# 3. Correlation Analysis
print("\n3. CORRELATION ANALYSIS:")
correlation_matrix = df[question_columns].corr()
print("\nCorrelation matrix between questions:")
print(correlation_matrix.round(2))

# 4. Key Findings
print("\n4. KEY FINDINGS:")

# Calculate overall satisfaction
overall_satisfaction = df_long['Score'].mean()
print(f"\nOverall satisfaction score: {overall_satisfaction:.2f}/4")

# Find strongest and weakest aspects
question_means = df_long.groupby('Question')['Score'].mean()
strongest = question_means.idxmax()
weakest = question_means.idxmin()
print(f"\nStrongest aspect: {strongest}")
print(f"Score: {question_means.max():.2f}")
print(f"\nAspect needing most improvement: {weakest}")
print(f"Score: {question_means.min():.2f}")

# Calculate response rates
print("\n5. RESPONSE ANALYSIS:")
total_responses = len(df)
print(f"Total number of responses: {total_responses}")

# Calculate consistency scores
print("\n6. RESPONSE CONSISTENCY:")
std_devs = df_long.groupby('Question')['Score'].std()
most_consistent = std_devs.idxmin()
least_consistent = std_devs.idxmax()
print(f"\nMost consistent responses: {most_consistent}")
print(f"Standard deviation: {std_devs.min():.2f}")
print(f"\nMost varied responses: {least_consistent}")
print(f"Standard deviation: {std_devs.max():.2f}")

# 7. Statistical Tests
print("\n7. STATISTICAL TESTS:")

# One-way ANOVA to test if there are significant differences between questions
f_statistic, p_value = stats.f_oneway(*[df[col] for col in question_columns])
print("\nOne-way ANOVA test results:")
print(f"F-statistic: {f_statistic:.4f}")
print(f"p-value: {p_value:.4f}")
print("Interpretation: Significant differences between questions" if p_value < 0.05 
      else "Interpretation: No significant differences between questions")

# Generate visualizations
plt.style.use('seaborn')
fig = plt.figure(figsize=(20, 15))

# 1. Heatmap of correlation matrix
plt.subplot(2, 2, 1)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Questions')

# 2. Box plot with individual points
plt.subplot(2, 2, 2)
sns.boxplot(data=df_long, x='Score', y='Question')
sns.swarmplot(data=df_long, x='Score', y='Question', color='0.25', alpha=0.5)
plt.title('Score Distribution with Individual Responses')

# 3. Histogram of all scores
plt.subplot(2, 2, 3)
sns.histplot(data=df_long, x='Score', bins=4, discrete=True)
plt.title('Overall Distribution of Scores')

# 4. Bar plot of means with error bars
plt.subplot(2, 2, 4)
question_stats = df_long.groupby('Question')['Score'].agg(['mean', 'std']).reset_index()
plt.errorbar(question_stats['mean'], range(len(question_stats)), 
            xerr=question_stats['std'], fmt='o', capsize=5)
plt.yticks(range(len(question_stats)), question_stats['Question'])
plt.title('Mean Scores with Standard Deviation')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle('Comprehensive Event Evaluation Analysis', fontsize=16, y=0.98)
plt.savefig('evaluation_analysis_detailed.png', bbox_inches='tight', dpi=300)
plt.close()