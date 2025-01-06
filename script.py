import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import networkx as nx
from kneed import KneeLocator
from sklearn.decomposition import PCA
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_event_ratings(df, low_threshold, high_threshold):
    """Analyze event ratings and identify areas needing improvement and high-performing areas."""
    avg_scores = df.mean().round(2)
    low_scores = avg_scores[avg_scores < low_threshold]
    high_scores = avg_scores[avg_scores >= high_threshold]
    return avg_scores, low_scores, high_scores

def prepare_for_association_rules(df, selected_features):
    """
    Convert selected features to transactions with ratings categorized as Low/Medium/High.
    
    Parameters:
    - df (DataFrame): The event ratings data
    - selected_features (list): Selected features for analysis
    
    Returns:
    - binary_df (DataFrame): Binary representation for each rating level
    """
    def categorize_rating(x):
        if pd.isna(x):
            return 'Missing'
        elif x <= 1.5:
            return 'Low'
        elif x <= 2.5:
            return 'Medium'
        else:
            return 'High'
    
    # Create binary columns for each feature-rating combination
    binary_df = pd.DataFrame()
    for feature in selected_features:
        # Categorize ratings
        categories = df[feature].apply(categorize_rating)
        # Create binary columns for each category
        for category in ['Low', 'Medium', 'High']:
            col_name = f"{feature}_{category}"
            binary_df[col_name] = (categories == category).astype(int)
    
    return binary_df

def generate_association_rules(binary_df, min_support=0.1):
    """
    Generate association rules from binary data.
    
    Parameters:
    - binary_df (DataFrame): Binary event features.
    - min_support (float): Minimum support for the apriori algorithm.
    
    Returns:
    - rules (DataFrame): Generated association rules.
    """
    try:
        frequent_itemsets = apriori(binary_df, min_support=min_support, use_colnames=True)
        
        if frequent_itemsets.empty:
            return pd.DataFrame()
        
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)
        rules = rules.sort_values('lift', ascending=False)
        
        return rules
    except Exception as e:
        print(f"Error generating rules: {e}")
        return pd.DataFrame()

def generate_recommendations_from_rules(rules, min_lift=1.5):
    """
    Generate recommendations based on association rules analysis.
    
    Parameters:
    - rules (DataFrame): Association rules generated from the data.
    - min_lift (float): Minimum lift value to consider a rule significant.
    
    Returns:
    - recommendations (dict): Dictionary of recommendations by feature.
    """
    recommendations = {}
    
    if rules.empty:
        return recommendations
        
    # Sort rules by lift for strongest associations first
    sorted_rules = rules.sort_values('lift', ascending=False)
    
    for _, rule in sorted_rules.iterrows():
        if rule['lift'] >= min_lift:
            # Extract features and ratings from antecedents and consequents
            for antecedent in rule['antecedents']:
                feature, rating = antecedent.rsplit('_', 1)
                
                if feature not in recommendations:
                    recommendations[feature] = []
                
                # Generate recommendation based on the rule
                for consequent in rule['consequents']:
                    cons_feature, cons_rating = consequent.rsplit('_', 1)
                    recommendation = {
                        'text': f"When {feature} achieves {rating} ratings, {cons_feature} tends to be {cons_rating}",
                        'action': f"{'Maintain' if rating == 'High' else 'Improve'} {feature} to {'positively impact' if rating == 'High' else 'enhance'} {cons_feature}",
                        'support': rule['support'],
                        'confidence': rule['confidence'],
                        'lift': rule['lift']
                    }
                    
                    if recommendation not in recommendations[feature]:
                        recommendations[feature].append(recommendation)
    
    return recommendations

def generate_event_recommendations(low_scores):
    """Generate standard recommendations based on low scores."""
    base_recommendations = {
        'Overall_Rating': [
            {
                'text': "Conduct comprehensive program review",
                'action': "Implement systematic review processes",
                'support': 1.0,
                'confidence': 1.0,
                'lift': 1.0
            },
            {
                'text': "Implement regular feedback sessions",
                'action': "Schedule monthly feedback meetings",
                'support': 1.0,
                'confidence': 1.0,
                'lift': 1.0
            }
        ],
        'Objectives_Met': [
            {
                'text': "Clearly communicate objectives",
                'action': "Create detailed objective documentation",
                'support': 1.0,
                'confidence': 1.0,
                'lift': 1.0
            }
        ],
        'Venue_Rating': [
            {
                'text': "Consider alternative venues",
                'action': "Research and evaluate new venues",
                'support': 1.0,
                'confidence': 1.0,
                'lift': 1.0
            }
        ],
        'Schedule_Rating': [
            {
                'text': "Offer more flexible scheduling options",
                'action': "Implement scheduling surveys",
                'support': 1.0,
                'confidence': 1.0,
                'lift': 1.0
            }
        ],
        'Speaker_Rating': [
            {
                'text': "Invite more engaging speakers",
                'action': "Develop speaker evaluation criteria",
                'support': 1.0,
                'confidence': 1.0,
                'lift': 1.0
            }
        ]
    }
    return {k: base_recommendations.get(k, []) for k in low_scores.index if k in base_recommendations}

def generate_event_maintenance_recommendations(high_scores):
    """Generate maintenance recommendations for high-performing areas."""
    maintenance_recommendations = {
        'Overall_Rating': [
            {
                'text': "Document successful practices",
                'action': "Create best practices documentation",
                'support': 1.0,
                'confidence': 1.0,
                'lift': 1.0
            }
        ],
        'Objectives_Met': [
            {
                'text': "Maintain clear documentation",
                'action': "Conduct regular documentation reviews",
                'support': 1.0,
                'confidence': 1.0,
                'lift': 1.0
            }
        ],
        'Venue_Rating': [
            {
                'text': "Maintain venue relationships",
                'action': "Schedule regular venue reviews",
                'support': 1.0,
                'confidence': 1.0,
                'lift': 1.0
            }
        ],
        'Schedule_Rating': [
            {
                'text': "Keep consistent scheduling",
                'action': "Document scheduling processes",
                'support': 1.0,
                'confidence': 1.0,
                'lift': 1.0
            }
        ],
        'Speaker_Rating': [
            {
                'text': "Build speaker database",
                'action': "Create a speaker tracking system",
                'support': 1.0,
                'confidence': 1.0,
                'lift': 1.0
            }
        ]
    }
    return {k: maintenance_recommendations.get(k, []) for k in high_scores.index if k in maintenance_recommendations}

def interpret_event_association_rules(rules):
    """
    Generate human-readable interpretations for the association rules.
    
    Parameters:
    - rules (DataFrame): Association rules generated from the data.
    
    Returns:
    - interpretations (str): Interpretations of the association rules.
    """
    if rules.empty:
        return "No significant associations found between event features."
    
    interpretations = ["Association Rules Analysis:\n"]
    
    for _, rule in rules.head(15).iterrows():
        # Extract feature and rating from antecedents and consequents
        antecedents = list(rule['antecedents'])
        consequents = list(rule['consequents'])
        
        # Format antecedents and consequents
        ant_str = []
        for ant in antecedents:
            feature, rating = ant.rsplit('_', 1)
            ant_str.append(f"'{feature}' is {rating}")
        
        cons_str = []
        for cons in consequents:
            feature, rating = cons.rsplit('_', 1)
            cons_str.append(f"'{feature}' is {rating}")
        
        # Create rule interpretation
        rule_str = f"Rule: If {' AND '.join(ant_str)}, then {' AND '.join(cons_str)}\n"
        rule_str += f"(Support: {rule['support']:.2f}, Confidence: {rule['confidence']:.2f}, Lift: {rule['lift']:.2f})\n"
        
        interpretations.append(rule_str)
    
    return "\n".join(interpretations)

def find_optimal_clusters(data, max_clusters=10):
    """
    Find the optimal number of clusters using the elbow method.
    
    Parameters:
    - data (array-like): The data to cluster.
    - max_clusters (int): The maximum number of clusters to test.
    
    Returns:
    - elbow (int): The optimal number of clusters.
    """
    wcss = []
    K = range(1, max_clusters + 1)
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    
    # Plot the elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(K, wcss, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.title('Elbow Method for Optimal k')
    plt.savefig('elbow_curve.png')
    plt.close()
    
    # Use KneeLocator to automatically find the elbow point
    kn = KneeLocator(
        K, wcss, curve='convex', direction='decreasing'
    )
    
    return kn.elbow

def cluster_events(df, selected_features):
    """
    Cluster events using the optimal number of clusters based on selected rating features.
    
    Parameters:
    - df (DataFrame): The event ratings data.
    - selected_features (list): List of feature names selected by the user.
    
    Returns:
    - df (DataFrame): Updated DataFrame with cluster assignments.
    - kmeans (KMeans): The fitted KMeans instance.
    """
    features = selected_features  # Use the selected features directly
    X = df[features].values
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find optimal number of clusters
    optimal_clusters = find_optimal_clusters(X_scaled)
    print(f"Optimal number of clusters determined: {optimal_clusters}")
    
    # Perform clustering with optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    return df, kmeans

class AnalysisGUI:
    def __init__(self, root):
        try:
            self.root = root
            self.root.title("Event Recommendation System")
            self.root.geometry("1600x1000")
            
            # Create main containers
            self.left_frame = ttk.Frame(root, padding="10")
            self.right_frame = ttk.Frame(root, padding="10")
            self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
            
            # Create GUI elements
            self.create_control_panel()
            self.create_output_area()
            self.create_visualization_area()
            
            self.df = None
            self.selected_features = []  # Initialize as empty list
            self.clusters = None
            self.kmeans = None  # Initialize kmeans
            self.centers = None
            
            print("AnalysisGUI initialized successfully.")
        except Exception as e:
            print(f"Error in AnalysisGUI.__init__: {e}")
    
    def create_control_panel(self):
        # Control panel frame
        control_frame = ttk.LabelFrame(self.left_frame, text="Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=5)
        
        # Load data button
        self.load_btn = ttk.Button(control_frame, text="Load Data", command=self.load_data)
        self.load_btn.pack(fill=tk.X, pady=5)
        
        # Analysis button
        self.analyze_btn = ttk.Button(control_frame, text="Run Event Analysis", command=self.run_analysis)
        self.analyze_btn.pack(fill=tk.X, pady=5)
        self.analyze_btn.state(['disabled'])
        
        # Parameters frame
        param_frame = ttk.LabelFrame(self.left_frame, text="Parameters", padding="10")
        param_frame.pack(fill=tk.X, pady=5)
        
        # Thresholds
        ttk.Label(param_frame, text="Low Rating Threshold:").pack()
        self.low_threshold_var = tk.StringVar(value="1.5")
        self.low_threshold_entry = ttk.Entry(param_frame, textvariable=self.low_threshold_var)
        self.low_threshold_entry.pack(fill=tk.X, pady=5)
        
        ttk.Label(param_frame, text="High Rating Threshold:").pack()
        self.high_threshold_var = tk.StringVar(value="2.5")
        self.high_threshold_entry = ttk.Entry(param_frame, textvariable=self.high_threshold_var)
        self.high_threshold_entry.pack(fill=tk.X, pady=5)
    
    def create_output_area(self):
        # Output text area
        output_frame = ttk.LabelFrame(self.left_frame, text="Event Analysis Results", padding="10")
        output_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, height=20)
        self.output_text.pack(fill=tk.BOTH, expand=True)
    
    def create_visualization_area(self):
        # Visualization area
        viz_frame = ttk.LabelFrame(self.right_frame, text="Visualizations", padding="10")
        viz_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs for different plots
        self.tab_control = ttk.Notebook(viz_frame)
        
        self.cluster_tab = ttk.Frame(self.tab_control)
        self.rules_tab = ttk.Frame(self.tab_control)
        
        self.tab_control.add(self.cluster_tab, text='Clustering')
        self.tab_control.add(self.rules_tab, text='Association Rules')
        
        self.tab_control.pack(fill=tk.BOTH, expand=True)
    
    def load_data(self):
        print("Starting load_data method")  # Debug print
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            print(f"Selected file path: {file_path}")  # Debug print
            
            if not file_path:  # User cancelled file selection
                print("No file selected")  # Debug print
                return
            
            # Load the data, explicitly setting first row as header
            print("Loading data from CSV...")  # Debug print
            self.df = pd.read_csv(file_path, header=0)
            
            # Try to convert all columns to numeric where possible, skipping the header
            numeric_data = {}
            for col in self.df.columns:
                try:
                    # Convert column to numeric, preserving the original values if conversion fails
                    numeric_series = pd.to_numeric(self.df[col], errors='coerce')
                    # Only update if we have some valid numeric values
                    if numeric_series.notna().any():
                        numeric_data[col] = numeric_series
                    else:
                        print(f"Column {col} has no valid numeric values")
                except Exception as e:
                    print(f"Could not convert column {col}: {str(e)}")
                    continue
            
            # Update dataframe with numeric columns
            for col, series in numeric_data.items():
                self.df[col] = series
            
            print(f"Data loaded. Shape: {self.df.shape}")  # Debug print
            print(f"Columns: {self.df.columns.tolist()}")  # Debug print
            print(f"Column types: {self.df.dtypes}")  # Debug print
            
            # Check if data was loaded successfully
            if self.df is None or self.df.empty:
                print("DataFrame is empty")  # Debug print
                messagebox.showerror("Error", "No data was loaded from the file.")
                return
            
            # Enable analyze button
            self.analyze_btn.state(['!disabled'])
            
            # Show success message
            self.output_text.insert(tk.END, f"Data loaded successfully: {len(self.df)} records\n")
            self.output_text.insert(tk.END, f"Columns found: {', '.join(self.df.columns)}\n\n")
            
            # Clear any previous feature selections
            self.selected_features = []
            
            print("About to create feature selection window")  # Debug print
            # Create and show the feature selection window
            self.select_features_window()
            print("After calling select_features_window")  # Debug print
            
        except Exception as e:
            print(f"Error in load_data: {str(e)}")  # Debug print
            messagebox.showerror("Error", f"Error loading data: {str(e)}")
    
    def select_features_window(self):
        print("Starting select_features_window method")  # Debug print
        try:
            feature_window = tk.Toplevel(self.root)
            feature_window.title("Select Features")
            feature_window.geometry("400x500")
            
            # Make window modal
            feature_window.transient(self.root)
            feature_window.grab_set()
            
            print("Getting columns")  # Debug print
            # Get numeric columns (including those that can be converted to numeric)
            numeric_columns = []
            for col in self.df.columns:
                # Check if column is already numeric
                if np.issubdtype(self.df[col].dtype, np.number):
                    numeric_columns.append(col)
                    print(f"Column {col} is numeric")
                else:
                    # Try converting to numeric
                    try:
                        if pd.to_numeric(self.df[col], errors='coerce').notna().any():
                            numeric_columns.append(col)
                            print(f"Column {col} can be converted to numeric")
                    except:
                        print(f"Column {col} is not numeric")
                        continue
            
            print(f"Numeric columns found: {numeric_columns}")  # Debug print
            
            if not numeric_columns:
                messagebox.showerror("Error", "No numeric columns found in the data.")
                feature_window.destroy()
                return
            
            # Create frames for column type selection
            selection_frame = ttk.Frame(feature_window)
            selection_frame.pack(fill=tk.X, padx=10, pady=5)
            
            instructions = ttk.Label(
                feature_window, 
                text="Select numeric features to include in the analysis\n(Hold Ctrl/Cmd to select multiple)",
                wraplength=350
            )
            instructions.pack(pady=10)
            
            # Frame for listbox and scrollbar
            list_frame = ttk.Frame(feature_window)
            list_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
            
            # Add scrollbar
            scrollbar = ttk.Scrollbar(list_frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Create a listbox with multiple selection enabled
            self.features_listbox = tk.Listbox(
                list_frame,
                selectmode=tk.MULTIPLE,
                height=15,
                yscrollcommand=scrollbar.set
            )
            self.features_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            # Configure scrollbar
            scrollbar.config(command=self.features_listbox.yview)
            
            # Populate listbox
            for column in numeric_columns:
                self.features_listbox.insert(tk.END, column)
            
            print("Creating buttons")  # Debug print
            # Button frame
            button_frame = ttk.Frame(feature_window)
            button_frame.pack(pady=20, padx=10, fill=tk.X)
            
            # Add Select All and Clear All buttons
            select_all_btn = ttk.Button(
                button_frame,
                text="Select All",
                command=lambda: self.features_listbox.select_set(0, tk.END)
            )
            select_all_btn.pack(side=tk.LEFT, padx=5, expand=True)
            
            clear_all_btn = ttk.Button(
                button_frame,
                text="Clear All",
                command=lambda: self.features_listbox.selection_clear(0, tk.END)
            )
            clear_all_btn.pack(side=tk.LEFT, padx=5, expand=True)
            
            submit_btn = ttk.Button(
                feature_window,
                text="Submit",
                command=lambda: self.submit_feature_selection(feature_window)
            )
            submit_btn.pack(pady=10)
            
            print("Feature selection window created successfully")  # Debug print
            
        except Exception as e:
            print(f"Error in select_features_window: {str(e)}")  # Debug print
            messagebox.showerror("Error", f"Error creating feature selection window: {str(e)}")
    
    def submit_feature_selection(self, feature_window):
        """
        Stores the selected features for analysis and closes the selection window.
        """
        selected_indices = self.features_listbox.curselection()
        if not selected_indices:
            tk.messagebox.showerror(
                "Error",
                "Please select at least one feature for analysis."
            )
            return
        
        # Store selected features as a list
        self.selected_features = [self.features_listbox.get(i) for i in selected_indices]
        
        # Update output text
        self.output_text.insert(tk.END, "Selected features for analysis:\n")
        for feature in self.selected_features:
            self.output_text.insert(tk.END, f"- {feature}\n")
        self.output_text.insert(tk.END, "\n")
        
        # Close the feature selection window
        feature_window.destroy()
    
    def plot_clusters(self):
        """
        Plots the K-means clusters using PCA for dimensionality reduction.
        """
        if not self.selected_features:
            self.output_text.insert(tk.END, "Error: No features selected for clustering.\n")
            return

        if len(self.selected_features) < 1:
            self.output_text.insert(tk.END, "Error: At least one feature must be selected for clustering.\n")
            return

        # Apply PCA to reduce dimensions to 2 for visualization
        pca = PCA(n_components=2)
        try:
            X_pca = pca.fit_transform(self.df[self.selected_features])
        except KeyError as e:
            self.output_text.insert(tk.END, f"Error: {e}\n")
            return
        except Exception as e:
            self.output_text.insert(tk.END, f"Unexpected error during PCA: {e}\n")
            return

        fig, ax = plt.subplots(figsize=(8, 6))

        # Create scatter plot with legend
        scatter = ax.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=self.clusters,
            cmap='viridis',
            label='Events'
        )
        centers_pca = pca.transform(self.kmeans.cluster_centers_)
        ax.scatter(
            centers_pca[:, 0],
            centers_pca[:, 1],
            c='red',
            marker='x',
            s=200,
            linewidths=3,
            label='Cluster Centers'
        )

        # Add labels and title
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.set_title('K-means Clustering of Events (PCA Reduced)')

        # Add legend
        ax.legend()

        # Add colorbar with label
        cbar = plt.colorbar(scatter)
        cbar.set_label('Cluster Assignment')

        # Embed plot in GUI
        canvas = FigureCanvasTkAgg(fig, master=self.cluster_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def plot_association_rules(self, rules):
        """
        Plots the association rules as a network graph.
        """
        if rules.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, 'No significant association rules found', 
                    ha='center', va='center')
            ax.set_axis_off()
            canvas = FigureCanvasTkAgg(fig, master=self.rules_tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes and edges with weights
        edge_weights = []
        edge_pairs = []
        edge_colors = []
        
        # Take top 15 rules to show more relationships
        for _, rule in rules.head(15).iterrows():
            antecedents = list(rule['antecedents'])
            consequents = list(rule['consequents'])
            for ant in antecedents:
                for cons in consequents:
                    weight = float(rule['lift'])
                    G.add_edge(ant, cons, weight=weight)
                    edge_weights.append(weight)
                    edge_pairs.append((ant, cons))
                    
                    # Determine color based on lift value
                    if weight < 1.2:  # Weak association
                        edge_colors.append('#ffcccc')  # Light red
                    elif weight < 1.5:  # Moderate association
                        edge_colors.append('#ff6666')  # Medium red
                    else:  # Strong association
                        edge_colors.append('#cc0000')  # Dark red
        
        if not edge_weights:
            ax.text(0.5, 0.5, 'No significant associations found', 
                    ha='center', va='center')
            ax.set_axis_off()
        else:
            # Draw the graph
            pos = nx.spring_layout(G, k=1)
            
            # Draw edges with varying widths and colors
            for (u, v), weight, color in zip(edge_pairs, edge_weights, edge_colors):
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=[(u, v)],
                    width=weight / max(edge_weights) * 5,
                    edge_color=color
                )
            
            # Draw nodes
            nx.draw_networkx_nodes(
                G, pos, 
                node_color='lightblue',
                node_size=2000,
                alpha=0.7
            )
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
            
            ax.set_title('Association Rules Network\n(Darker red indicates stronger association)', pad=20)
            
            # Add legend with colored lines
            legend_elements = [
                plt.Line2D([0], [0], color='#ffcccc', linewidth=2, label='Weak (lift < 1.2)'),
                plt.Line2D([0], [0], color='#ff6666', linewidth=2, label='Moderate (lift 1.2-1.5)'),
                plt.Line2D([0], [0], color='#cc0000', linewidth=2, label='Strong (lift > 1.5)'),
                plt.scatter([0], [0], c='lightblue', s=100, label='Rating Category')
            ]
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        # Remove axis
        ax.set_axis_off()
        
        # Adjust layout
        plt.tight_layout()
        
        # Embed plot in GUI
        canvas = FigureCanvasTkAgg(fig, master=self.rules_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def run_analysis(self):
        try:
            # Clear previous output and plots
            self.output_text.delete(1.0, tk.END)
            for widget in self.cluster_tab.winfo_children():
                widget.destroy()
            for widget in self.rules_tab.winfo_children():
                widget.destroy()
            
            # Get parameters
            low_threshold = float(self.low_threshold_var.get())
            high_threshold = float(self.high_threshold_var.get())
            
            # Check if features are selected
            if not self.selected_features or len(self.selected_features) == 0:
                self.output_text.insert(tk.END, "Error: Please select at least one feature before running analysis.\n")
                return
            
            # Check if selected features are numerical
            non_numeric = self.df[self.selected_features].select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numeric:
                self.output_text.insert(tk.END, f"Error: The following selected features are non-numeric and cannot be used for clustering: {non_numeric}\n")
                return
            
            # Perform event-specific analysis
            avg_scores, low_scores, high_scores = analyze_event_ratings(
                self.df, low_threshold, high_threshold
            )
            standard_recommendations = generate_event_recommendations(low_scores)
            maintenance_recs = generate_event_maintenance_recommendations(high_scores)
            
            # Generate association rules
            binary_df = prepare_for_association_rules(self.df, self.selected_features)
            rules = generate_association_rules(binary_df)
            
            # Display association rules interpretation
            self.output_text.insert(tk.END, "\nAssociation Rules Analysis:\n")
            self.output_text.insert(tk.END, "==========================\n")
            interpretation = interpret_event_association_rules(rules)
            self.output_text.insert(tk.END, interpretation + "\n")
            
            # Generate and display recommendations from association rules
            logging.debug("Generating association-based recommendations.")
            self.output_text.insert(tk.END, "\nAssociation-Based Recommendations:\n")
            self.output_text.insert(tk.END, "================================\n")
            assoc_recommendations = generate_recommendations_from_rules(rules)
            
            logging.debug(f"Association Recommendations: {assoc_recommendations}")
            
            if assoc_recommendations:
                for feature, recs in assoc_recommendations.items():
                    self.output_text.insert(tk.END, f"\nRecommendations for {feature}:\n")
                    for i, rec in enumerate(recs, 1):
                        text = rec.get('text', 'No text provided')
                        action = rec.get('action', 'No action provided')
                        support = rec.get('support', 0)
                        confidence = rec.get('confidence', 0)
                        lift = rec.get('lift', 0)
                        
                        self.output_text.insert(tk.END, f"{i}. {text}\n")
                        self.output_text.insert(tk.END, f"   Action: {action}\n")
                        self.output_text.insert(tk.END, f"   (Support: {support:.2f}, "
                                             f"Confidence: {confidence:.2f}, "
                                             f"Lift: {lift:.2f})\n")
            else:
                self.output_text.insert(tk.END, "No significant associations found to generate recommendations.\n")
            
            # Perform clustering with optimal number of clusters
            self.df, self.kmeans = cluster_events(self.df, self.selected_features)
            self.clusters = self.df['cluster'].values
            
            # Display results
            self.output_text.insert(tk.END, "Average Event Ratings by Category:\n")
            self.output_text.insert(tk.END, str(avg_scores) + "\n\n")
            
            if not low_scores.empty:
                self.output_text.insert(tk.END, "Areas Needing Improvement:\n")
                self.output_text.insert(tk.END, str(low_scores) + "\n\n")
                
                if standard_recommendations:  # Handle standard recommendations
                    self.output_text.insert(tk.END, "Recommendations for Improvement:\n")
                    
                    # Standard Recommendations
                    for category, recs in standard_recommendations.items():
                        if category in low_scores.index:
                            self.output_text.insert(tk.END, f"\n{category} (Rating: {low_scores[category]:.2f}):\n")
                            for i, rec in enumerate(recs, 1):
                                self.output_text.insert(tk.END, f"  {i}. {rec['text']}\n")
                                self.output_text.insert(tk.END, f"     Action: {rec['action']}\n")
                                self.output_text.insert(tk.END, f"     (Support: {rec['support']:.2f}, "
                                             f"Confidence: {rec['confidence']:.2f}, "
                                             f"Lift: {rec['lift']:.2f})\n")
            
            if not high_scores.empty:
                self.output_text.insert(tk.END, "\nHigh Performing Areas:\n")
                self.output_text.insert(tk.END, str(high_scores) + "\n\n")
                
                if maintenance_recs:  # Check if there are any maintenance recommendations
                    self.output_text.insert(tk.END, "Maintenance Recommendations:\n")
                    for category, recs in maintenance_recs.items():
                        if category in high_scores.index:
                            self.output_text.insert(tk.END, f"\n{category} (Rating: {high_scores[category]:.2f}):\n")
                            for i, rec in enumerate(recs, 1):
                                self.output_text.insert(tk.END, f"  {i}. {rec['text']}\n")
                                self.output_text.insert(tk.END, f"     Action: {rec['action']}\n")
                                self.output_text.insert(tk.END, f"     (Support: {rec['support']:.2f}, "
                                             f"Confidence: {rec['confidence']:.2f}, "
                                             f"Lift: {rec['lift']:.2f})\n")
            
            # Create visualizations
            self.plot_clusters()
            self.plot_association_rules(rules)
            
        except Exception as e:
            logging.error(f"Error during analysis: {e}")
            self.output_text.insert(tk.END, f"\nError during analysis: {str(e)}\n")

def main():
    try:
        root = tk.Tk()
        app = AnalysisGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()
