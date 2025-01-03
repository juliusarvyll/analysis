import os
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
from tkinter import messagebox, filedialog
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from sklearn.decomposition import PCA
import networkx as nx

# ============================
# Data Loading and Preprocessing
# ============================

# Define global variables to store data and models
data = None
X = None
y = None
features = None
selected_features = None
best_model = None
y_pred = None
cv_strategy = None
X_test = None
y_test = None
# Add tab variables
tab1 = None
tab2 = None
tab3 = None
tab4 = None
tab5 = None
tab6 = None
tab7 = None

# Feature Recommendations Mapping
FEATURE_RECOMMENDATIONS = {
    "Knowledgeable_Speakers": "Invest in advanced training programs for speakers to enhance their knowledge and presentation skills.",
    "Venue_Fit": "Consider selecting venues that better align with event objectives and attendee preferences to improve overall experience.",
    "Schedule_Favorable": "Optimize the event schedule to ensure it is favorable and convenient for attendees, minimizing conflicts and enhancing engagement.",
    "Met_Expectations": "Focus on accurately meeting attendee expectations through tailored content and experiences.",
    "Activities_Engaging": "Increase the number and variety of engaging activities to maintain high attendee interest and participation.",
    "Recommend": "Encourage attendees who had positive experiences to recommend the event through targeted follow-up and incentives."
    # Add more feature mappings as needed
}

def load_data(file_path):
    global data, X, y, features, selected_features, best_model, y_pred, cv_strategy, X_test, y_test

    try:
        data = pd.read_csv(file_path, on_bad_lines='warn')
    except FileNotFoundError:
        messagebox.showerror("File Not Found", f"Error: The file '{file_path}' was not found.")
        return False
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while loading the file:\n{e}")
        return False

    # Define numeric columns globally
    global numeric_columns
    numeric_columns = [
        "Overall_Rating", "Objectives_Met", "Venue_Rating", 
        "Schedule_Rating", "Allowance_Rating", "Speaker_Rating",
        "Facilitator_Rating", "Participant_Rating"
    ]

    # Convert all columns to numeric
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Calculate recommend value based on average of ratings
    # Weighted average: Overall_Rating has higher weight
    data['Recommend'] = np.round(
        0.4 * data['Overall_Rating'] +  # 40% weight to overall rating
        0.6 * data[numeric_columns[1:]].mean(axis=1)  # 60% weight to average of other ratings
    ).astype(int)

    # Add Recommend to numeric columns
    numeric_columns.append('Recommend')

    # Use Overall_Rating as target variable
    X = data[numeric_columns[1:-1]]  # All columns except Overall_Rating and Recommend
    y = data["Overall_Rating"]

    # Map labels for binary classification (2,3 are high ratings)
    valid_mask = y.isin([2, 3])
    X = X[valid_mask]
    y = y[valid_mask]
    y = y.map({2: 0, 3: 1})

    # Update data to match filtered rows
    data = data[valid_mask]

    # Define features globally
    features = X.columns
    selected_features = features

    # Scale all features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

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
        'C': [0.1, 1.0, 10.0],           # Regularization strength
        'class_weight': ['balanced'],     # Handle imbalanced classes
        'solver': ['lbfgs'],             # Good for small datasets
        'max_iter': [1000]               # Increase iterations if needed
    }

    # Use LogisticRegression instead
    grid_search = GridSearchCV(
        LogisticRegression(random_state=42),
        param_grid,
        cv=3,                           # Reduced CV folds for small dataset
        scoring='balanced_accuracy'      # Better for imbalanced classes
    )
    grid_search.fit(X_train_resampled, y_train_resampled)

    # Best model from grid search
    best_model = grid_search.best_estimator_

    # Predict and evaluate with the best model
    y_pred = best_model.predict(X_test)

    return True

def generate_suggestions(data_subset):
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
        mean_score = data_subset[column].mean()
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
root.geometry("1200x900")

def select_file_and_initialize():
    # Get the current directory
    current_dir = os.getcwd()

    # Prompt user to select a CSV file
    file_path = filedialog.askopenfilename(
        title="Select CSV File",
        initialdir=current_dir,
        filetypes=[("CSV Files", "*.csv")]
    )

    if file_path:
        success = load_data(file_path)
        if success:
            # Clear existing widgets from root
            for widget in root.winfo_children():
                if not isinstance(widget, tk.Menu):  # Preserve the menu
                    widget.destroy()
            initialize_dashboard()

def create_menu():
    # Create the menubar
    menubar = tk.Menu(root)
    root.config(menu=menubar)
    
    # Create File menu
    file_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label="Open CSV", command=select_file_and_initialize)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.quit)

# Initialize empty dashboard
def initialize_empty_dashboard():
    tab_control = ttk.Notebook(root)
    tab_control.pack(expand=1, fill='both')

    # Create empty tabs
    tabs = {}
    for tab_name in ['Descriptive Analytics', 'Diagnostic Analytics', 
                     'Predictive Analytics', 'Prescriptive Analytics', 
                     'Numeric Data', 'Suggestions', 'Advanced Analysis']:
        tabs[tab_name] = ttk.Frame(tab_control)
        tab_control.add(tabs[tab_name], text=tab_name)
        
        # Add placeholder text
        label = ttk.Label(
            tabs[tab_name],
            text="Please open a CSV file from the File menu to begin analysis",
            font=("Arial", 12)
        )
        label.pack(expand=True)

def plot_descriptive():
    # Create main frame
    main_frame = ttk.Frame(tab1)
    main_frame.pack(fill='both', expand=True, padx=10, pady=5)

    # Create left frame for summary
    left_frame = ttk.Frame(main_frame)
    left_frame.pack(side='left', fill='both', expand=True, padx=5)

    # Create right frame for plot
    right_frame = ttk.Frame(main_frame)
    right_frame.pack(side='right', fill='both', expand=True, padx=5)

    # Calculate and display summary statistics
    summary_stats = data[numeric_columns].describe()
    target_column = numeric_columns[-1]
    summary_text = """Descriptive Statistics Summary:

📊 Overall Response Analysis:
• Total Responses: {}
• Recommendability Rate: {:.1f}% highly recommend

📈 Distribution Analysis:""".format(
        len(data),
        (data[target_column] == 3).mean() * 100  # Changed from 4 to 3
    )
    
    for col in numeric_columns[:-1]:  # Exclude 'Recommend' for separate treatment
        col_name = col.replace('_', ' ').title()
        mean = summary_stats.loc['mean', col]
        median = summary_stats.loc['50%', col]
        mode = data[col].mode().iloc[0]
        std = summary_stats.loc['std', col]
        min_val = summary_stats.loc['min', col]
        max_val = summary_stats.loc['max', col]
        range_val = max_val - min_val
        
        summary_text += f"\n\n• {col_name}:"
        summary_text += f"\n  Central Tendency:"
        summary_text += f"\n    - Mean: {mean:.2f}/3.0"
        summary_text += f"\n    - Median: {median:.2f}/3.0"
        summary_text += f"\n    - Mode: {mode:.0f}/3.0"
        summary_text += f"\n  Variability:"
        summary_text += f"\n    - Range: {range_val:.1f} ({min_val:.1f} to {max_val:.1f})"
        summary_text += f"\n    - Standard Deviation: ±{std:.2f}"
        
        # Add interpretation
        if mean > 2.0:  # High satisfaction threshold
            summary_text += f"\n  ✓ High satisfaction in {col_name.lower()}"
        elif mean < 1.0:  # Low satisfaction threshold
            summary_text += f"\n  ⚠ Area needs attention: {col_name.lower()}"
        else:
            summary_text += f"\n  → Moderate performance in {col_name.lower()}"
        
        # Add consistency interpretation
        if std < 0.5:
            summary_text += f"\n  📊 Very consistent ratings"
        elif std < 0.75:
            summary_text += f"\n  📊 Fairly consistent ratings"
        else:
            summary_text += f"\n  📊 Varied ratings - mixed experiences"

    summary_label = ttk.Label(
        left_frame,
        text=summary_text,
        wraplength=400,
        justify="left",
        font=("Arial", 11)
    )
    summary_label.pack(pady=10)

    # Create and display plot
    plt.figure(figsize=(8, 6))
    data[numeric_columns].hist(bins=10)
    plt.suptitle("Distribution of Ratings", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    canvas = FigureCanvasTkAgg(plt.gcf(), master=right_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)

def plot_diagnostic():
    # Create main frame
    main_frame = ttk.Frame(tab2)
    main_frame.pack(fill='both', expand=True, padx=10, pady=5)

    # Create frames for two-column layout
    left_frame = ttk.Frame(main_frame)
    right_frame = ttk.Frame(main_frame)
    
    left_frame.pack(side='left', fill='both', expand=True, padx=5)
    right_frame.pack(side='left', fill='both', expand=True, padx=5)

    # 1. Correlation Analysis (left column)
    correlation_matrix = data[numeric_columns].corr()
    correlations = []
    for i in range(len(numeric_columns)):
        for j in range(i+1, len(numeric_columns)):
            correlations.append({
                'pair': (numeric_columns[i], numeric_columns[j]),
                'correlation': correlation_matrix.iloc[i, j]
            })
    
    correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
    corr_text = """Correlation Analysis:\n\n🔍 Key Relationships:"""
    
    for corr in correlations:
        pair = corr['pair']
        value = corr['correlation']
        
        # Determine correlation strength and description
        if abs(value) > 0.7:
            strength = "Strong"
            if value > 0:
                description = "As one increases, the other strongly increases"
            else:
                description = "As one increases, the other strongly decreases"
        elif abs(value) > 0.4:
            strength = "Moderate"
            if value > 0:
                description = "As one increases, the other moderately increases"
            else:
                description = "As one increases, the other moderately decreases"
        else:
            strength = "Weak"
            if value > 0:
                description = "Slight positive relationship"
            else:
                description = "Slight negative relationship"
        
        feature1 = pair[0].replace('_', ' ').title()
        feature2 = pair[1].replace('_', ' ').title()
        
        corr_text += f"\n\n• {feature1} & {feature2}:"
        corr_text += f"\n  Correlation: {value:.2f}"
        corr_text += f"\n  {strength} {value > 0 and 'positive' or 'negative'} correlation"
        corr_text += f"\n  Interpretation: {description}"

    corr_label = ttk.Label(
        left_frame,
        text=corr_text,
        wraplength=400,
        justify="left",
        font=("Arial", 11)
    )
    corr_label.pack(pady=10)

    # 2. Correlation Heatmap (right column)
    plt.figure(figsize=(6, 5))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap", fontsize=12)
    plt.tight_layout()
    
    canvas = FigureCanvasTkAgg(plt.gcf(), master=right_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=10)

    # Add heatmap interpretation
    heatmap_text = """
Heatmap Interpretation:
• Values range from -1 to 1
• Darker red indicates stronger positive correlation
• Darker blue indicates stronger negative correlation
• Values closer to 0 (white) indicate weaker relationships

Reading the Heatmap:
• 1.0 on diagonal (self-correlation)
• Symmetric matrix - value is same for both pairs
• Focus on strongest correlations (darkest colors)"""

    heatmap_label = ttk.Label(
        right_frame,
        text=heatmap_text,
        wraplength=400,
        justify="left",
        font=("Arial", 11)
    )
    heatmap_label.pack(pady=10)

def plot_predictive():
    # Create main frame
    main_frame = ttk.Frame(tab3)
    main_frame.pack(fill='both', expand=True, padx=10, pady=5)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

    # Create summary text
    summary_text = """Predictive Model Performance Summary:

📊 Model Metrics:
• Accuracy: {:.2f} - {:.1f}% of predictions are correct
• Precision: {:.2f} - {:.1f}% of positive predictions are correct
• Recall: {:.2f} - {:.1f}% of actual positive cases were identified
• F1-Score: {:.2f} - Balance between precision and recall
• AUC-ROC: {:.2f} - Model's ability to distinguish between classes

🎯 Interpretation:""".format(
        accuracy, accuracy*100,
        precision, precision*100,
        recall, recall*100,
        f1, auc
    )

    # Add interpretation based on metrics
    if accuracy > 0.7:
        summary_text += "\n• The model shows good overall prediction accuracy"
    else:
        summary_text += "\n• The model's accuracy could be improved"

    if precision > recall:
        summary_text += "\n• The model is more conservative in making positive predictions"
    else:
        summary_text += "\n• The model favors capturing positive cases over precision"

    summary_label = ttk.Label(
        main_frame,
        text=summary_text,
        wraplength=800,
        justify="left",
        font=("Arial", 11)
    )
    summary_label.pack(pady=10)

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    
    canvas = FigureCanvasTkAgg(plt.gcf(), master=main_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=10)

    # Add confusion matrix interpretation
    cm_text = f"""
Confusion Matrix Interpretation:
• True Negatives (Top Left): {cm[0,0]} - Correctly predicted negative cases
• False Positives (Top Right): {cm[0,1]} - Incorrectly predicted as positive
• False Negatives (Bottom Left): {cm[1,0]} - Incorrectly predicted as negative
• True Positives (Bottom Right): {cm[1,1]} - Correctly predicted positive cases"""

    cm_label = ttk.Label(
        main_frame,
        text=cm_text,
        wraplength=800,
        justify="left",
        font=("Arial", 11)
    )
    cm_label.pack(pady=10)

def plot_prescriptive():
    # Create main frame
    main_frame = ttk.Frame(tab4)
    main_frame.pack(fill='both', expand=True, padx=10, pady=5)

    # Create frames for two-column layout
    left_frame = ttk.Frame(main_frame)
    right_frame = ttk.Frame(main_frame)
    
    left_frame.pack(side='left', fill='both', expand=True, padx=5)
    right_frame.pack(side='left', fill='both', expand=True, padx=5)

    # 1. Feature Impact Analysis (left column)
    feature_importances = np.abs(best_model.coef_[0])
    plot_df = pd.DataFrame({
        'Feature': selected_features,
        'Importance': feature_importances,
        'Impact': best_model.coef_[0]  # Original coefficients for direction
    }).sort_values('Importance', ascending=True)

    # Add summary text above plot
    summary_text = """Feature Impact Analysis:

🎯 Impact on Event Success (All Features):"""

    # Calculate mean ratings for context - use actual column names
    feature_columns = numeric_columns[1:]  # All columns except Overall_Rating
    mean_ratings = data[feature_columns].mean()

    # First show positive impacts
    summary_text += "\n\n✓ POSITIVE IMPACTS:"
    for idx, row in plot_df[plot_df['Impact'] > 0].sort_values('Importance', ascending=False).iterrows():
        feature_name = feature_columns[idx].replace('_', ' ').title()
        importance = row['Importance']
        impact = row['Impact']
        mean_rating = mean_ratings[feature_columns[idx]]
        
        summary_text += f"\n\n• {feature_name}:"
        summary_text += f"\n  Current Rating: {mean_rating:.2f}/3.0"
        summary_text += f"\n  Impact Strength: {importance:.3f}"
        summary_text += f"\n  Effect: +{abs(impact):.2f} points in recommendability per point increase"
        if importance > np.mean(feature_importances) + np.std(feature_importances):
            summary_text += "\n  ⭐ Critical success factor"
        
    # Then show negative impacts
    summary_text += "\n\n⚠ NEGATIVE IMPACTS:"
    for idx, row in plot_df[plot_df['Impact'] < 0].sort_values('Importance', ascending=False).iterrows():
        feature_name = feature_columns[idx].replace('_', ' ').title()
        importance = row['Importance']
        impact = row['Impact']
        mean_rating = mean_ratings[feature_columns[idx]]
        
        summary_text += f"\n\n• {feature_name}:"
        summary_text += f"\n  Current Rating: {mean_rating:.2f}/3.0"
        summary_text += f"\n  Impact Strength: {importance:.3f}"
        summary_text += f"\n  Effect: -{abs(impact):.2f} points in recommendability per point decrease"
        if importance > np.mean(feature_importances) + np.std(feature_importances):
            summary_text += "\n  ⚠ Critical risk factor"

    summary_label = ttk.Label(
        left_frame,
        text=summary_text,
        wraplength=400,
        justify="left",
        font=("Arial", 11)
    )
    summary_label.pack(pady=10)

    # Plot feature importance with impact direction
    plt.figure(figsize=(8, 6))
    colors = ['red' if x < 0 else 'green' for x in plot_df['Impact']]
    sns.barplot(data=plot_df, x='Importance', y='Feature', palette=colors)
    plt.title("Feature Impact Analysis", fontsize=12)
    plt.xlabel("Impact Strength")
    plt.ylabel("Features")
    
    # Add +/- signs to feature names
    ax = plt.gca()
    labels = [f"{label.get_text()} ({'+'if impact > 0 else '-'})" 
              for label, impact in zip(ax.get_yticklabels(), plot_df['Impact'])]
    ax.set_yticklabels(labels)
    
    plt.tight_layout()
    
    canvas = FigureCanvasTkAgg(plt.gcf(), master=left_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=10)

    # 2. Detailed Recommendations (right column)
    recommendations_text = """Action Recommendations:\n\n"""
    
    # Priority actions for negative impacts
    recommendations_text += "🚨 PRIORITY ACTIONS:\n"
    for idx, row in plot_df[plot_df['Impact'] < 0].sort_values('Importance', ascending=False).iterrows():
        feature_name = feature_columns[idx].replace('_', ' ').title()
        mean_rating = mean_ratings[feature_columns[idx]]
        
        recommendations_text += f"\n📌 {feature_name}:"
        recommendations_text += f"\n• Current: {mean_rating:.2f}/3.0"
        if feature_name.replace(' ', '_') in FEATURE_RECOMMENDATIONS:
            recommendations_text += f"\n• Action: {FEATURE_RECOMMENDATIONS[feature_name.replace(' ', '_')]}"
        recommendations_text += "\n"

    # Best practices for positive impacts
    recommendations_text += "\n✨ MAINTAIN EXCELLENCE:\n"
    for idx, row in plot_df[plot_df['Impact'] > 0].sort_values('Importance', ascending=False).iterrows():
        feature_name = feature_columns[idx].replace('_', ' ').title()
        mean_rating = mean_ratings[feature_columns[idx]]
        
        recommendations_text += f"\n📌 {feature_name}:"
        recommendations_text += f"\n• Current: {mean_rating:.2f}/3.0"
        if feature_name.replace(' ', '_') in FEATURE_RECOMMENDATIONS:
            recommendations_text += f"\n• Continue: {FEATURE_RECOMMENDATIONS[feature_name.replace(' ', '_')]}"
        recommendations_text += "\n"

    recommendations_label = ttk.Label(
        right_frame,
        text=recommendations_text,
        wraplength=400,
        justify="left",
        font=("Arial", 11)
    )
    recommendations_label.pack(pady=10)

def plot_numeric_data():
    # Create main frame
    main_frame = ttk.Frame(tab5)
    main_frame.pack(fill='both', expand=True, padx=10, pady=10)

    # Create header
    header_text = """Detailed Numeric Analysis
    
📊 This tab shows comprehensive statistical measures for all numeric variables:
• count: Number of responses
• mean: Average rating
• std: Standard deviation (spread of ratings)
• min: Minimum rating given
• 25%: First quartile (25% of ratings fall below this)
• 50%: Median rating (middle value)
• 75%: Third quartile (75% of ratings fall below this)
• max: Maximum rating given
"""
    header_label = ttk.Label(
        main_frame,
        text=header_text,
        justify="left",
        font=("Arial", 11),
        wraplength=800
    )
    header_label.pack(pady=10)

    # Calculate statistics
    stats = data[numeric_columns].describe()
    
    # Create formatted text display
    text = tk.Text(main_frame, height=20, width=100, font=("Courier", 10))
    text.pack(fill='both', expand=True, padx=10, pady=10)
    
    # Format the statistics with proper alignment
    formatted_stats = stats.round(2).to_string()
    text.insert('end', formatted_stats)
    
    # Add interpretation
    interpretation = "\n\nKey Insights:\n"
    for column in numeric_columns:
        mean = stats.loc['mean', column]
        std = stats.loc['std', column]
        interpretation += f"\n{column.replace('_', ' ').title()}:"
        interpretation += f"\n• Average rating: {mean:.2f}"
        interpretation += f"\n• Consistency: {'High' if std < 0.5 else 'Moderate' if std < 1 else 'Low'}"
        interpretation += f" (std: {std:.2f})"
        interpretation += f"\n• Range: {stats.loc['min', column]:.0f} to {stats.loc['max', column]:.0f}\n"

    text.insert('end', interpretation)
    text.config(state='disabled')

def generate_prescriptive_overview():
    # Create main frame with two columns
    main_frame = ttk.Frame(tab6)
    main_frame.pack(fill='both', expand=True, padx=10, pady=10)

    # Create left and right frames
    left_frame = ttk.Frame(main_frame)
    left_frame.pack(side='left', fill='both', expand=True, padx=5)
    
    right_frame = ttk.Frame(main_frame)
    right_frame.pack(side='right', fill='both', expand=True, padx=5)

    # Calculate average scores for each feature
    avg_scores = data[numeric_columns[:-1]].mean()  # Exclude 'Recommend' column
    
    # Define thresholds
    high_threshold = 3.5
    low_threshold = 3.0

    # Identify strengths and weaknesses
    strengths = []
    weaknesses = []
    neutral = []

    for feature, score in avg_scores.items():
        feature_name = feature.replace('_', ' ').title()
        if score >= high_threshold:
            strengths.append((feature_name, score))
        elif score <= low_threshold:
            weaknesses.append((feature_name, score))
        else:
            neutral.append((feature_name, score))

    # Create overall summary
    summary_text = """Event Performance Analysis Summary\n\n"""
    
    # Add overall recommendation rate
    recommend_rate = (data['Recommend'] == 4).mean() * 100
    summary_text += f"📈 Overall Recommendation Rate: {recommend_rate:.1f}%\n"
    summary_text += "This represents the percentage of attendees who strongly recommend the event.\n\n"

    # Add strengths section
    summary_text += "🌟 STRENGTHS:\n"
    if strengths:
        for feature, score in strengths:
            summary_text += f"• {feature}: {score:.2f}/3.0\n"
            summary_text += f"  ✓ Strong performance in {feature.lower()}\n"
            summary_text += f"  → Continue current practices for {feature.lower()}\n"
    else:
        summary_text += "• No outstanding strengths identified (scores > 3.5)\n"

    summary_label = ttk.Label(
        left_frame,
        text=summary_text,
        wraplength=400,
        justify="left",
        font=("Arial", 11)
    )
    summary_label.pack(pady=10)

    # Create detailed recommendations
    recommendations_text = """Detailed Recommendations\n\n"""
    
    recommendations_text += "❗ PRIORITY IMPROVEMENTS:\n"
    if weaknesses:
        for feature, score in weaknesses:
            recommendations_text += f"\n• {feature} ({score:.2f}/3.0):\n"
            if feature.replace(' ', '_') in FEATURE_RECOMMENDATIONS:
                recommendations_text += f"  → {FEATURE_RECOMMENDATIONS[feature.replace(' ', '_')]}\n"
            recommendations_text += f"  → Target: Increase rating above {low_threshold}\n"
    else:
        recommendations_text += "• No critical areas requiring immediate attention\n"

    recommendations_text += "\n📊 AREAS FOR OPTIMIZATION:\n"
    for feature, score in neutral:
        recommendations_text += f"\n• {feature} ({score:.2f}/3.0):\n"
        if feature.replace(' ', '_') in FEATURE_RECOMMENDATIONS:
            recommendations_text += f"  → {FEATURE_RECOMMENDATIONS[feature.replace(' ', '_')]}\n"
        recommendations_text += f"  → Target: Aim for rating above {high_threshold}\n"

    # Add correlation insights
    correlation_matrix = data[numeric_columns].corr()
    recommend_correlations = correlation_matrix['Recommend'].drop('Recommend')
    top_factor = recommend_correlations.abs().idxmax()
    top_correlation = recommend_correlations[top_factor]
    
    recommendations_text += f"\n🔍 KEY INSIGHT:\n"
    recommendations_text += f"• {top_factor.replace('_', ' ').title()} shows the strongest relationship with recommendations\n"
    recommendations_text += f"• Focus on this aspect for maximum impact on future event success\n"

    recommendations_label = ttk.Label(
        right_frame,
        text=recommendations_text,
        wraplength=400,
        justify="left",
        font=("Arial", 11)
    )
    recommendations_label.pack(pady=10)

def perform_clustering_analysis(data):
    """Perform K-means clustering on the event feedback data"""
    # Prepare data for clustering
    cluster_data = data[numeric_columns[:-1]]  # Exclude 'Recommend' column
    
    # Determine optimal number of clusters (2-4)
    inertias = []
    K = range(2, 5)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(cluster_data)
        inertias.append(kmeans.inertia_)
    
    # Use 3 clusters for interpretability
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(cluster_data)
    
    # Analyze cluster characteristics
    cluster_analysis = []
    for i in range(3):
        cluster_data_i = data[cluster_labels == i]
        cluster_means = cluster_data_i[numeric_columns[:-1]].mean()
        
        # Determine cluster characteristics
        if cluster_means.mean() > 3.5:
            satisfaction = "High Satisfaction"
        elif cluster_means.mean() < 2.5:
            satisfaction = "Low Satisfaction"
        else:
            satisfaction = "Moderate Satisfaction"
            
        # Find distinguishing features
        top_features = cluster_means.nlargest(2)
        
        cluster_analysis.append({
            'cluster_id': i,
            'size': len(cluster_data_i),
            'satisfaction': satisfaction,
            'top_features': top_features,
            'means': cluster_means
        })
    
    return cluster_analysis

def perform_association_analysis(data):
    """Perform association rule mining on event feedback"""
    # Convert ratings to binary (high/low satisfaction)
    binary_data = data[numeric_columns].copy()
    for col in binary_data.columns:
        binary_data[col] = binary_data[col] > 3.5  # Changed threshold for 5-point scale
    
    try:
        # Generate frequent itemsets with much lower thresholds
        frequent_itemsets = apriori(binary_data, 
                                  min_support=0.1,  # Lowered significantly for small dataset
                                  use_colnames=True)
        
        if len(frequent_itemsets) == 0:
            print("No frequent itemsets found")
            return []

        # Generate association rules with lower thresholds
        rules = association_rules(frequent_itemsets, 
                                metric="confidence",
                                min_threshold=0.1)  # Lowered significantly
        
        if rules.empty:
            print("No rules found with current thresholds")
            return []
            
        rules = rules.sort_values('lift', ascending=False)
        
        # Format rules for interpretation
        formatted_rules = []
        for _, rule in rules.iterrows():
            antecedents = list(rule['antecedents'])
            consequents = list(rule['consequents'])
            
            formatted_rules.append({
                'if': [col.replace('_', ' ').title() for col in antecedents],
                'then': [col.replace('_', ' ').title() for col in consequents],
                'confidence': rule['confidence'],
                'lift': rule['lift']
            })
        
        return formatted_rules[:5]  # Return top 5 rules only
    except Exception as e:
        print(f"Error in association analysis: {e}")
        print("Data shape:", binary_data.shape)
        print("Data values:\n", binary_data.head())
        return []  # Return empty list if analysis fails

def display_advanced_analysis():
    """Display clustering analysis in a row layout"""
    # Create main frame
    main_frame = ttk.Frame(tab7)
    main_frame.pack(fill='both', expand=True, padx=10, pady=5)

    # Create two frames for row layout
    viz_frame = ttk.Frame(main_frame)
    cluster_frame = ttk.Frame(main_frame)
    
    # Pack frames side by side
    viz_frame.pack(side='left', fill='both', expand=True, padx=5)
    cluster_frame.pack(side='left', fill='both', expand=True, padx=5)

    # 1. Clustering Visualization
    cluster_data = data[numeric_columns[:-1]]
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(cluster_data)
    
    # Calculate cluster means to determine satisfaction levels
    cluster_means = []
    for i in range(3):
        cluster_mean = data[cluster_labels == i][numeric_columns[:-1]].mean().mean()
        cluster_means.append((i, cluster_mean))
    
    # Sort clusters by mean satisfaction and assign labels
    cluster_means.sort(key=lambda x: x[1])
    cluster_mapping = {
        cluster_means[0][0]: 0,  # Low satisfaction
        cluster_means[1][0]: 1,  # Moderate satisfaction
        cluster_means[2][0]: 2   # High satisfaction
    }
    
    # Map original labels to ordered labels
    ordered_labels = [cluster_mapping[label] for label in cluster_labels]
    
    # Create PCA transformation
    pca = PCA(n_components=2)
    cluster_data_2d = pca.fit_transform(cluster_data)
    
    plt.figure(figsize=(8, 6))
    
    # Custom colors for three satisfaction levels - Maximum contrast
    colors = ['#FF0000',    # Pure red for Low
              '#FFA500',    # Pure orange for Moderate
              '#00FF00']    # Pure green for High
    
    # Create scatter plot with custom colors and increased size
    scatter = plt.scatter(cluster_data_2d[:, 0], cluster_data_2d[:, 1], 
                         c=[colors[label] for label in ordered_labels],
                         alpha=0.8,          # More opaque
                         s=120)              # Even larger points
    
    # Add custom legend with matching colors
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color, label=label,
                                 markersize=12,        # Larger legend markers
                                 alpha=0.8)
                      for color, label in zip(colors, ['Low Satisfaction',
                                                     'Moderate Satisfaction',
                                                     'High Satisfaction'])]
    
    plt.legend(handles=legend_elements, title='Attendee Segments',
              loc='upper right', bbox_to_anchor=(1.15, 1),
              fontsize=10)                   # Larger legend text
    
    # White background for better contrast
    plt.gca().set_facecolor('white')
    
    # Darker grid for better visibility
    plt.grid(True, alpha=0.3, color='gray')
    
    plt.title('Attendee Satisfaction Clusters', fontsize=12, pad=20)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.tight_layout()
    
    canvas = FigureCanvasTkAgg(plt.gcf(), master=viz_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=5)

    viz_label = ttk.Label(
        viz_frame,
        text="Cluster Analysis Interpretation:\n" +
             "• Each point represents one attendee\n" +
             "• Colors indicate satisfaction level\n" +
             "• Grouped points show similar rating patterns",
        wraplength=300,
        justify="left",
        font=("Arial", 10)
    )
    viz_label.pack(pady=5)

    # 2. Cluster Analysis Text
    cluster_text = """Detailed Segment Analysis:\n\n"""
    
    # Sort clusters by satisfaction level
    cluster_results = perform_clustering_analysis(data)
    sorted_clusters = sorted(cluster_results, 
                           key=lambda x: x['means'].mean(),
                           reverse=True)
    
    satisfaction_labels = ['High Satisfaction', 'Moderate Satisfaction', 'Low Satisfaction']
    for i, cluster in enumerate(sorted_clusters):
        cluster_text += f"🎯 {satisfaction_labels[i]} Group ({cluster['size']} attendees):\n"
        cluster_text += "• Key Characteristics:\n"
        for feature, value in cluster['top_features'].items():
            feature_name = feature.replace('_', ' ').title()
            cluster_text += f"  - {feature_name}: {value:.2f}/3.0\n"
        cluster_text += "\n"

    cluster_label = ttk.Label(
        cluster_frame,
        text=cluster_text,
        wraplength=300,
        justify="left",
        font=("Arial", 11)
    )
    cluster_label.pack(pady=5)

def create_scrollable_frame(parent):
    """Create a scrollable frame for a tab"""
    # Create a canvas with scrollbar
    canvas = tk.Canvas(parent)
    scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    
    # Create a frame inside the canvas
    scrollable_frame = ttk.Frame(canvas)
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    # Add the frame to the canvas
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # Configure canvas to expand
    parent.grid_rowconfigure(0, weight=1)
    parent.grid_columnconfigure(0, weight=1)
    canvas.grid(row=0, column=0, sticky="nsew")
    scrollbar.grid(row=0, column=1, sticky="ns")
    
    return scrollable_frame

def initialize_dashboard():
    global tab1, tab2, tab3, tab4, tab5, tab6, tab7
    # Create a tab control
    tab_control = ttk.Notebook(root)
    tab_control.pack(expand=1, fill='both')

    # Create tabs with scrollable frames
    tab_frames = []
    for i in range(7):
        tab = ttk.Frame(tab_control)
        tab_frames.append(tab)
        scrollable_frame = create_scrollable_frame(tab)
        tab_frames.append(scrollable_frame)

    # Assign scrollable frames to global variables
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = [
        tab_frames[i*2 + 1] for i in range(7)
    ]

    # Add tabs to notebook
    tab_control.add(tab_frames[0], text='Descriptive Analytics')
    tab_control.add(tab_frames[2], text='Diagnostic Analytics')
    tab_control.add(tab_frames[4], text='Predictive Analytics')
    tab_control.add(tab_frames[6], text='Prescriptive Analytics')
    tab_control.add(tab_frames[8], text='Numeric Data')
    tab_control.add(tab_frames[10], text='Suggestions')
    tab_control.add(tab_frames[12], text='Advanced Analysis')

    # Add your existing plot functions here
    plot_descriptive()
    plot_diagnostic()
    plot_predictive()
    plot_prescriptive()
    plot_numeric_data()
    generate_prescriptive_overview()
    display_advanced_analysis()

# Create menu and initialize empty dashboard
create_menu()
initialize_empty_dashboard()

# Run the Tkinter event loop
root.mainloop()
