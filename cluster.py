import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import tkinter as tk
from tkinter import ttk, scrolledtext
from tkinter import filedialog
import networkx as nx
from kneed import KneeLocator
from sklearn.decomposition import PCA

def analyze_event_ratings(df, low_threshold, high_threshold):
    """Analyze event ratings and identify areas needing improvement and high-performing areas"""
    avg_scores = df.mean().round(2)
    low_scores = avg_scores[avg_scores < low_threshold]
    high_scores = avg_scores[avg_scores >= high_threshold]
    return avg_scores, low_scores, high_scores

def prepare_for_association_rules(df):
    # Convert event-related features to binary
    event_features = [
        'Overall_Rating',
        'Objectives_Met',
        'Venue_Rating',
        'Schedule_Rating',
        'Allowance_Rating',
        'Speaker_Rating',
        'Facilitator_Rating',
        'Participant_Rating'
    ]
    
    # Check for missing columns
    missing_features = [feature for feature in event_features if feature not in df.columns]
    if missing_features:
        print(f"Missing columns in data: {missing_features}")
        # Optionally, handle missing columns by removing them from the list
        event_features = [feature for feature in event_features if feature in df.columns]
    
    binary_df = pd.DataFrame()
    for feature in event_features:
        binary_df[feature] = df[feature].apply(lambda x: x > 0)
    return binary_df

def generate_association_rules(binary_df, min_support=0.1):
    try:
        # Generate frequent itemsets without max_len restriction to allow all combinations
        frequent_itemsets = apriori(binary_df, 
                                  min_support=min_support, 
                                  use_colnames=True)
        
        if frequent_itemsets.empty:
            return pd.DataFrame()
        
        # Generate rules with lower thresholds to catch more relationships
        rules = association_rules(frequent_itemsets, 
                                metric="confidence", 
                                min_threshold=0.2)  # Lowered threshold
        
        # Sort rules by lift to show strongest relationships first
        rules = rules.sort_values('lift', ascending=False)
        
        return rules
    except Exception as e:
        print(f"Error generating rules: {e}")
        return pd.DataFrame()

def generate_recommendations_from_rules(rules, min_lift=1.5):
    """
    Generate recommendations based on strong association rules.
    
    Parameters:
    - rules (DataFrame): Association rules generated from the data.
    - min_lift (float): Minimum lift value to consider a rule significant.
    
    Returns:
    - recommendations (dict): Recommendations extracted from the rules.
    """
    recommendations = {}
    for _, rule in rules.iterrows():
        if rule['lift'] >= min_lift:
            antecedents = list(rule['antecedents'])
            consequents = list(rule['consequents'])
            for antecedent in antecedents:
                for consequent in consequents:
                    recommendation = f"Improve '{consequent}' as it is strongly associated with '{antecedent}'."
                    if antecedent in recommendations:
                        if recommendation not in recommendations[antecedent]:
                            recommendations[antecedent].append(recommendation)
                    else:
                        recommendations[antecedent] = [recommendation]
    return recommendations

def generate_event_recommendations(low_scores):
    recommendations = {
        'Overall_Rating': ["Conduct comprehensive program review", "Implement regular feedback sessions"],
        'Objectives_Met': ["Clearly communicate objectives", "Create measurable outcomes"],
        'Venue_Rating': ["Consider alternative venues with better facilities", "Negotiate better pricing with current venue"],
        'Schedule_Rating': ["Offer more flexible scheduling options", "Conduct surveys to find optimal event timings"],
        'Allowance_Rating': ["Review budget allocation", "Seek additional funding"],
        'Speaker_Rating': ["Invite more engaging speakers", "Provide guidelines for speaker presentations"],
        'Facilitator_Rating': ["Provide facilitator training", "Create best practices"],
        'Participant_Rating': ["Improve engagement strategies", "Create interactive formats"],
        'Marketing_Rating': ["Enhance social media marketing efforts", "Collaborate with influencers"],
        'Logistics_Rating': ["Improve registration process", "Ensure timely setup and teardown"],
        'Content_Rating': ["Diversify event content", "Incorporate interactive sessions"],
        'Networking_Rating': ["Facilitate better networking opportunities", "Organize dedicated networking sessions"],
        'Feedback_Rating': ["Implement real-time feedback mechanisms", "Act on feedback promptly"]
    }
    return {k: recommendations[k] for k in low_scores.index if k in recommendations}

def generate_event_maintenance_recommendations(high_scores):
    maintenance_recommendations = {
        'Overall_Rating': ["Document successful practices", "Share best practices"],
        'Objectives_Met': ["Maintain clear documentation", "Share methodology"],
        'Venue_Rating': ["Maintain good relationships with venue providers", "Regularly review venue contracts"],
        'Schedule_Rating': ["Keep consistent scheduling practices", "Monitor attendee preferences over time"],
        'Allowance_Rating': ["Document budget strategies", "Create contingency funds"],
        'Speaker_Rating': ["Build a strong speaker database", "Maintain ongoing relationships with past speakers"],
        'Facilitator_Rating': ["Document techniques", "Create mentorship programs"],
        'Participant_Rating': ["Document engagement strategies", "Maintain recognition programs"],
        'Marketing_Rating': ["Sustain marketing campaigns", "Analyze marketing effectiveness periodically"],
        'Logistics_Rating': ["Ensure reliable logistics partners", "Regularly evaluate logistics processes"],
        'Content_Rating': ["Update event content regularly", "Incorporate trending topics"],
        'Networking_Rating': ["Enhance networking platforms", "Organize regular networking events"],
        'Feedback_Rating': ["Continuously collect and analyze feedback", "Implement long-term improvements based on feedback"]
    }
    return {k: maintenance_recommendations[k] for k in high_scores.index if k in maintenance_recommendations}

def interpret_event_association_rules(rules):
    """Generate interpretations for the association rules specific to event recommendations"""
    if rules.empty:
        return "No significant associations found between event features."
    
    interpretations = []
    for _, rule in rules.head(15).iterrows():
        antecedents = list(rule['antecedents'])
        consequents = list(rule['consequents'])
        lift = float(rule['lift'])
        confidence = float(rule['confidence'])
        
        # Determine strength description
        if lift < 1.2:
            strength = "weak"
        elif lift < 1.5:
            strength = "moderate"
        else:
            strength = "strong"
        
        # Format the interpretation
        interpretation = f"There is a {strength} association (lift={lift:.2f}) between "
        interpretation += f"{' & '.join(antecedents)} and {' & '.join(consequents)}.\n"
        interpretation += f"When {' & '.join(antecedents)} is rated highly, "
        interpretation += f"{' & '.join(consequents)} is also rated highly "
        interpretation += f"{(confidence*100):.1f}% of the time.\n"
        
        interpretations.append(interpretation)
    
    return "\n".join(interpretations)

def find_optimal_clusters(data, max_clusters=10):
    """
    Find the optimal number of clusters using the elbow method
    """
    wcss = []
    K = range(1, max_clusters + 1)
    
    # Calculate WCSS for different numbers of clusters
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

def cluster_events(df):
    """
    Cluster events using optimal number of clusters based on rating features
    """
    # Prepare the features for clustering
    features = [
        'Overall_Rating',
        'Objectives_Met',
        'Venue_Rating',
        'Schedule_Rating',
        'Allowance_Rating',
        'Speaker_Rating',
        'Facilitator_Rating',
        'Participant_Rating'
    ]
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
        self.clusters = None
        self.kmeans = None  # Initialize kmeans
        self.centers = None
    
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
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            self.df = pd.read_csv(file_path)
            self.analyze_btn.state(['!disabled'])
            self.output_text.insert(tk.END, f"Data loaded successfully: {len(self.df)} records\n")
    
    def plot_clusters(self):
        # Apply PCA to reduce dimensions to 2 for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.df[
            [
                'Overall_Rating',
                'Objectives_Met',
                'Venue_Rating',
                'Schedule_Rating',
                'Allowance_Rating',
                'Speaker_Rating',
                'Facilitator_Rating',
                'Participant_Rating'
            ]
        ])
        
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
        
        # Take top 15 rules instead of 10 to show more relationships
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
            
            # Perform event-specific analysis
            avg_scores, low_scores, high_scores = analyze_event_ratings(
                self.df, low_threshold, high_threshold
            )
            
            # Generate association rules
            binary_df = prepare_for_association_rules(self.df)
            rules = generate_association_rules(binary_df)
            
            # Generate recommendations from association rules
            assoc_recommendations = generate_recommendations_from_rules(rules)
            
            # Generate event-specific recommendations based on low scores
            recommendations = generate_event_recommendations(low_scores)
            maintenance_recs = generate_event_maintenance_recommendations(high_scores)
            
            # Perform clustering with optimal number of clusters
            self.df, self.kmeans = cluster_events(self.df)
            self.clusters = self.df['cluster'].values
            
            # Display results
            self.output_text.insert(tk.END, "Average Event Ratings by Category:\n")
            self.output_text.insert(tk.END, str(avg_scores) + "\n\n")
            
            if not low_scores.empty:
                self.output_text.insert(tk.END, "Areas Needing Improvement:\n")
                self.output_text.insert(tk.END, str(low_scores) + "\n\n")
                
                if recommendations or assoc_recommendations:  # Include association-based recommendations
                    self.output_text.insert(tk.END, "Recommendations for Improvement:\n")
                    
                    # Standard Recommendations
                    for category, recs in recommendations.items():
                        if category in low_scores.index:  # Check if category exists in low_scores
                            self.output_text.insert(tk.END, f"\n{category} (Rating: {low_scores[category]:.2f}):\n")
                            for i, rec in enumerate(recs, 1):
                                self.output_text.insert(tk.END, f"  {i}. {rec}\n")
                    
                    # Association-Based Recommendations
                    for category, recs in assoc_recommendations.items():
                        if category in low_scores.index:
                            self.output_text.insert(tk.END, f"\nAssociation-Based Recommendations for {category}:\n")
                            for i, rec in enumerate(recs, 1):
                                self.output_text.insert(tk.END, f"  {i}. {rec}\n")
            
            if not high_scores.empty:
                self.output_text.insert(tk.END, "\nHigh Performing Areas:\n")
                self.output_text.insert(tk.END, str(high_scores) + "\n\n")
                
                if maintenance_recs:  # Check if there are any maintenance recommendations
                    self.output_text.insert(tk.END, "Maintenance Recommendations:\n")
                    for category, recs in maintenance_recs.items():
                        if category in high_scores.index:  # Check if category exists in high_scores
                            self.output_text.insert(tk.END, f"\n{category} (Rating: {high_scores[category]:.2f}):\n")
                            for i, rec in enumerate(recs, 1):
                                self.output_text.insert(tk.END, f"  {i}. {rec}\n")
            
            # Add association rules interpretation
            self.output_text.insert(tk.END, "\nAssociation Rules Interpretation:\n")
            self.output_text.insert(tk.END, "================================\n")
            interpretation = interpret_event_association_rules(rules)
            self.output_text.insert(tk.END, interpretation + "\n")
            
            # Create visualizations
            self.plot_clusters()
            self.plot_association_rules(rules)
            
        except Exception as e:
            self.output_text.insert(tk.END, f"\nError during analysis: {str(e)}\n")
            print(f"Error: {str(e)}")

def main():
    root = tk.Tk()
    app = AnalysisGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
