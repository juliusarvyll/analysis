import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import tkinter.messagebox

# ============================
# Data Loading and Preprocessing
# ============================

# Load data with error handling
try:
    data = pd.read_csv(
        "expanded_evaluation_data.csv",
        on_bad_lines='warn'  # Updated parameter for recent pandas versions
    )
except FileNotFoundError:
    print("Error: The file 'evaluation_data_New event today (1).csv' was not found.")
    exit()

# Filter the data to include only rows where 'Recommend' is 3 or 4
data = data[data['Recommend'].isin([3, 4])]

# Rename columns for easier handling
data.columns = [
    "Respondent_ID", "Timestamp", "Knowledgeable_Speakers", "Venue_Fit",
    "Schedule_Favorable", "Met_Expectations", "Activities_Engaging",
    "Recommend"
]

# Correct the column name for "Recommend"
data.rename(columns={"I would recommend this event to others. Because it is very nice": "Recommend"}, inplace=True)

# Convert ratings to numeric, ignoring non-numeric columns
numeric_columns = [
    "Knowledgeable_Speakers", "Venue_Fit", "Schedule_Favorable", 
    "Met_Expectations", "Activities_Engaging", "Recommend"
]

for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Prepare data for predictive analytics
X = data[["Knowledgeable_Speakers", "Venue_Fit", "Schedule_Favorable", 
          "Met_Expectations", "Activities_Engaging"]]
y = data["Recommend"]

# Define features globally for access in all functions
features = X.columns

# Handle missing values (if any)
X.fillna(X.mean(), inplace=True)
y.fillna(y.mode()[0], inplace=True)

# Map 'Recommend' labels: 3 -> 0 (Not Recommend), 4 -> 1 (Recommend)
y = y.map({3: 0, 4: 1})

# Feature Selection: Select top k features
selector = SelectKBest(score_func=f_classif, k=3)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]

# Scale selected features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Train-test split with scaled features
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Apply SMOTE to balance the classes in the training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Define cross-validation strategy
cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# ============================
# Model Training and Evaluation
# ============================

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=cv_strategy,
    scoring='accuracy'
)
grid_search.fit(X_train_resampled, y_train_resampled)

# Best model from grid search
best_model = grid_search.best_estimator_

# Predict and evaluate with the best model
y_pred = best_model.predict(X_test)

# ============================
# Feature Recommendations Mapping
# ============================

FEATURE_RECOMMENDATIONS = {
    "Knowledgeable_Speakers": "Invest in advanced training programs for speakers to enhance their knowledge and presentation skills.",
    "Venue_Fit": "Consider selecting venues that better align with event objectives and attendee preferences to improve overall experience.",
    "Schedule_Favorable": "Optimize the event schedule to ensure it is favorable and convenient for attendees, minimizing conflicts and enhancing engagement.",
    "Met_Expectations": "Focus on accurately meeting attendee expectations through tailored content and experiences.",
    "Activities_Engaging": "Increase the number and variety of engaging activities to maintain high attendee interest and participation.",
    "Recommend": "Encourage attendees who had positive experiences to recommend the event through targeted follow-up and incentives."
    # Add more feature mappings as needed
}

# ============================
# Generate Suggestions Function
# ============================

def generate_suggestions(data):
    suggestions = []

    # Thresholds for recommendations
    threshold = {
        "Knowledgeable_Speakers": 4.0,
        "Venue_Fit": 4.0,
        "Schedule_Favorable": 4.0,
        "Met_Expectations": 4.0,
        "Activities_Engaging": 4.0,
    }

    for column, limit in threshold.items():
        mean_score = data[column].mean()
        if mean_score < limit:
            suggestion = f"Improve {column.replace('_', ' ')}: Current average score is {mean_score:.2f}, consider actionable steps to raise it."
            suggestions.append(suggestion)

    return suggestions

# ============================
# Tkinter Dashboard Setup
# ============================

# Create a Tkinter window
root = tk.Tk()
root.title("Advanced Analytics Dashboard")
root.geometry("1200x900")  # Set a suitable window size

# Create a tab control
tab_control = ttk.Notebook(root)

# Create tabs
tab1 = ttk.Frame(tab_control)
tab_control.add(tab1, text='Descriptive Analytics')

tab2 = ttk.Frame(tab_control)
tab_control.add(tab2, text='Diagnostic Analytics')

tab3 = ttk.Frame(tab_control)
tab_control.add(tab3, text='Predictive Analytics')

tab4 = ttk.Frame(tab_control)
tab_control.add(tab4, text='Prescriptive Analytics')

tab5 = ttk.Frame(tab_control)
tab_control.add(tab5, text='Numeric Data')

tab6 = ttk.Frame(tab_control)
tab_control.add(tab6, text='Suggestions')  # New Suggestions Tab

tab_control.pack(expand=1, fill='both')

# ============================
# Descriptive Analytics Tab
# ============================

def plot_descriptive():
    plt.figure(figsize=(12, 8))
    data[numeric_columns].hist(bins=10, figsize=(12, 8))
    plt.suptitle("Descriptive Analytics - Histograms", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    canvas = FigureCanvasTkAgg(plt.gcf(), master=tab1)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # Dynamic description for the descriptive analytics plot
    description = ""
    for col in numeric_columns:
        col_mean = data[col].mean()
        col_median = data[col].median()
        col_std = data[col].std()
        description += (
            f"**{col}** - Mean: {col_mean:.2f}, Median: {col_median:.2f}, "
            f"Standard Deviation: {col_std:.2f}.\n"
        )
    description_label = tk.Label(
        tab1, 
        text=description, 
        wraplength=1100, 
        justify="left", 
        padx=10, 
        pady=10, 
        font=("Arial", 12)
    )
    description_label.pack()

    # Add a data summary report below the graph
    summary_stats = data[numeric_columns].describe().loc[['mean', '50%', 'std']].rename(index={'50%': 'median'})

    summary_text = "Data Summary Report:\n"
    for stat in summary_stats.index:
        summary_text += f"\n{stat.capitalize()}\n"
        for col in numeric_columns:
            summary_text += f"{col}: {summary_stats.at[stat, col]:.2f}\n"

    summary_label = tk.Label(
        tab1,
        text=summary_text,
        wraplength=1100,
        justify="left",
        padx=10,
        pady=10,
        font=("Arial", 12)
    )
    summary_label.pack()

plot_descriptive()

# ============================
# Diagnostic Analytics Tab
# ============================

def plot_diagnostic():
    correlation_matrix = data[numeric_columns].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Diagnostic Analytics - Correlation Heatmap", fontsize=16)
    plt.tight_layout()
    
    canvas = FigureCanvasTkAgg(plt.gcf(), master=tab2)
    canvas.draw()
    canvas.get_tk_widget().pack()

plot_diagnostic()

# ============================
# Predictive Analytics Tab
# ============================

def generate_recommendations(model, feature_names):
    feature_importances = model.feature_importances_
    important_features = {
        feature: importance 
        for feature, importance in zip(feature_names, feature_importances) 
        if importance > np.mean(feature_importances)
    }
    recommendations = "### Detailed Prescriptive Recommendations:\n"
    for feature, importance in important_features.items():
        recommendation = FEATURE_RECOMMENDATIONS.get(
            feature, f"Consider investigating ways to improve {feature}."
        )
        recommendations += f"- **{feature}** has a high importance score ({importance:.2f}). {recommendation}\n"
    return recommendations

def plot_predictive():
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Predictive Analytics - Confusion Matrix (Best Model)", fontsize=16)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    
    canvas = FigureCanvasTkAgg(plt.gcf(), master=tab3)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # Generate recommendations based on feature importances
    recommendations = generate_recommendations(best_model, selected_features)
    recommendation_label = tk.Label(
        tab3, 
        text=recommendations, 
        wraplength=1100, 
        justify="left", 
        padx=10, 
        pady=10, 
        font=("Arial", 12)
    )
    recommendation_label.pack()

plot_predictive()

# ============================
# Prescriptive Analytics Tab
# ============================

def plot_prescriptive():
    if hasattr(best_model, 'feature_importances_'):
        feature_importances = best_model.feature_importances_
    else:
        feature_importances = best_model.coef_[0]
    features_to_plot = selected_features if 'selected_features' in globals() else features

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances, y=features_to_plot, palette="magma")
    plt.title("Prescriptive Analytics - Feature Importances", fontsize=16)
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.tight_layout()
    
    canvas = FigureCanvasTkAgg(plt.gcf(), master=tab4)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # Display detailed recommendations
    detailed_recommendations = generate_recommendations(best_model, features_to_plot)
    recommendation_text = tk.Text(
        tab4, 
        wrap='word',
        padx=10, 
        pady=10, 
        font=("Arial", 12),
        height=10,
        width=130
    )
    recommendation_text.insert('1.0', detailed_recommendations)
    recommendation_text.config(state="disabled")  # Make it read-only
    recommendation_text.pack(expand=True, fill='both')

plot_prescriptive()

# ============================
# Numeric Data Tab
# ============================

def plot_numeric_data():
    # Create a vertical scrollbar for the entire tab
    main_scrollbar = ttk.Scrollbar(tab5, orient='vertical')
    main_scrollbar.pack(side='right', fill='y')
    
    # Create a canvas to hold all frames
    canvas = tk.Canvas(tab5, yscrollcommand=main_scrollbar.set)
    canvas.pack(side='left', fill='both', expand=True)
    main_scrollbar.config(command=canvas.yview)
    
    # Create a frame inside the canvas
    data_frame = ttk.Frame(canvas)
    canvas.create_window((0, 0), window=data_frame, anchor='nw')
    
    # Update scrollregion after adding all widgets
    data_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    # Descriptive Analytics Section
    descriptive_frame = ttk.LabelFrame(data_frame, text="Descriptive Analytics")
    descriptive_frame.pack(fill='both', expand=True, padx=10, pady=10)
    
    # Summary Statistics Table
    stats = data[numeric_columns].describe().loc[['mean', '50%', 'std']].rename(index={'50%': 'median'})
    tree_descriptive = ttk.Treeview(descriptive_frame, columns=numeric_columns, show='headings')
    for col in numeric_columns:
        tree_descriptive.heading(col, text=col)
        tree_descriptive.column(col, width=150, anchor='center')
    
    for index, row in stats.iterrows():
        # Ensure the order of values matches numeric_columns
        row_values = [row[col] for col in numeric_columns]
        tree_descriptive.insert('', 'end', values=row_values)
    
    # Add scrollbar to Descriptive Analytics table
    scrollbar_descriptive = ttk.Scrollbar(descriptive_frame, orient='vertical', command=tree_descriptive.yview)
    tree_descriptive.configure(yscroll=scrollbar_descriptive.set)
    scrollbar_descriptive.pack(side='right', fill='y')
    tree_descriptive.pack(expand=True, fill='both')
    
    # Diagnostic Analytics Section
    diagnostic_frame = ttk.LabelFrame(data_frame, text="Diagnostic Analytics")
    diagnostic_frame.pack(fill='both', expand=True, padx=10, pady=10)
    
    # Correlation Matrix Table
    corr_matrix = data[numeric_columns].corr()
    tree_diagnostic = ttk.Treeview(diagnostic_frame, columns=numeric_columns, show='headings')
    for col in numeric_columns:
        tree_diagnostic.heading(col, text=col)
        tree_diagnostic.column(col, width=150, anchor='center')
    
    for index, row in corr_matrix.iterrows():
        row_values = [row[col] for col in numeric_columns]
        tree_diagnostic.insert('', 'end', values=row_values)
    
    # Add scrollbar to Diagnostic Analytics table
    scrollbar_diagnostic = ttk.Scrollbar(diagnostic_frame, orient='vertical', command=tree_diagnostic.yview)
    tree_diagnostic.configure(yscroll=scrollbar_diagnostic.set)
    scrollbar_diagnostic.pack(side='right', fill='y')
    tree_diagnostic.pack(expand=True, fill='both')
    
    # Predictive Analytics Section
    predictive_frame = ttk.LabelFrame(data_frame, text="Predictive Analytics")
    predictive_frame.pack(fill='both', expand=True, padx=10, pady=10)
    
    # Model Performance Metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "AUC Score": roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
    }
    
    tree_predictive = ttk.Treeview(predictive_frame, columns=["Metric", "Value"], show='headings')
    tree_predictive.heading("Metric", text="Metric")
    tree_predictive.heading("Value", text="Value")
    tree_predictive.column("Metric", width=200, anchor='center')
    tree_predictive.column("Value", width=150, anchor='center')
    
    for metric, value in metrics.items():
        tree_predictive.insert('', 'end', values=(metric, f"{value:.2f}"))
    
    # Add scrollbar to Predictive Analytics table
    scrollbar_predictive = ttk.Scrollbar(predictive_frame, orient='vertical', command=tree_predictive.yview)
    tree_predictive.configure(yscroll=scrollbar_predictive.set)
    scrollbar_predictive.pack(side='right', fill='y')
    tree_predictive.pack(expand=True, fill='both')
    
    # Prescriptive Analytics Section
    prescriptive_frame = ttk.LabelFrame(data_frame, text="Prescriptive Analytics")
    prescriptive_frame.pack(fill='both', expand=True, padx=10, pady=10)
    
    # Feature Importances Table
    if hasattr(best_model, 'feature_importances_'):
        feature_importances = best_model.feature_importances_
        features_to_show = selected_features
    else:
        feature_importances = best_model.coef_[0]
        features_to_show = features  # Fallback if using a model without feature_importances_
    
    prescriptive_data = sorted(zip(features_to_show, feature_importances), key=lambda x: x[1], reverse=True)
    tree_prescriptive = ttk.Treeview(prescriptive_frame, columns=["Feature", "Importance"], show='headings')
    tree_prescriptive.heading("Feature", text="Feature")
    tree_prescriptive.heading("Importance", text="Importance")
    tree_prescriptive.column("Feature", width=200, anchor='center')
    tree_prescriptive.column("Importance", width=150, anchor='center')
    
    for feature, importance in prescriptive_data:
        tree_prescriptive.insert('', 'end', values=(feature, f"{importance:.2f}"))
    
    # Add scrollbar to Prescriptive Analytics table
    scrollbar_prescriptive = ttk.Scrollbar(prescriptive_frame, orient='vertical', command=tree_prescriptive.yview)
    tree_prescriptive.configure(yscroll=scrollbar_prescriptive.set)
    scrollbar_prescriptive.pack(side='right', fill='y')
    tree_prescriptive.pack(expand=True, fill='both')

plot_numeric_data()

# ============================
# Suggestions Tab
# ============================

def plot_scores():
    avg_scores = data.mean()
    avg_scores = avg_scores.drop(["Respondent_ID", "Timestamp"])
    
    plt.figure(figsize=(10, 6))
    avg_scores.plot(kind="bar", color="skyblue")
    plt.title("Average Scores per Category", fontsize=16)
    plt.ylabel("Average Score")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 5)
    plt.tight_layout()

    canvas = FigureCanvasTkAgg(plt.gcf(), master=tab6)
    canvas.draw()
    canvas.get_tk_widget().pack()

def generate_prescriptive_overview():
    # Fetch suggestions based on average scores
    suggestions_list = generate_suggestions(data)
    if not suggestions_list:
        suggestions_text = "All categories have satisfactory average scores. Keep up the good work!"
    else:
        suggestions_text = "\n".join(f"- {s}" for s in suggestions_list)
    
    # Display suggestions in the Suggestions tab
    suggestions_frame = ttk.LabelFrame(tab6, text="Automated Suggestions")
    suggestions_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    suggestions_text_widget = tk.Text(
        suggestions_frame, 
        wrap="word", 
        padx=10, 
        pady=10, 
        font=("Arial", 12),
        height=10,
        width=130
    )
    suggestions_text_widget.insert("1.0", suggestions_text)
    suggestions_text_widget.config(state="disabled")
    suggestions_text_widget.pack(fill="both", expand=True)

    # Add plot button
    plot_button = ttk.Button(tab6, text="Plot Average Scores", command=plot_scores)
    plot_button.pack(pady=10)

generate_prescriptive_overview()

# ============================
# Numeric Data Tab (Optional)
# ============================

# The 'Numeric Data' tab already includes comprehensive tables for all four analytics types.
# No additional actions needed unless further customization is desired.

# ============================
# Start Tkinter Event Loop
# ============================

root.mainloop()
