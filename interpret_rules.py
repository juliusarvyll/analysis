import pandas as pd
from collections import defaultdict

def get_rating_description(feature):
    """Convert feature names to descriptive text"""
    feature = feature.strip()
    if feature.endswith('_Good'):
        metric = feature.replace('_Good', '').replace('_', ' ').title()
        return f"{metric} is Good"
    return feature

def analyze_rules(csv_file='association_rules.csv'):
    # Load the rules
    rules_df = pd.read_csv(csv_file)
    
    print("=== Training Evaluation Analysis Report ===")
    print("Based on Association Rule Mining of Rating Patterns\n")
    
    # Basic statistics
    print("=== Overview Statistics ===")
    print(f"• Total patterns discovered: {len(rules_df)}")
    print(f"• Average pattern occurrence (support): {rules_df['support'].mean():.1%}")
    print(f"• Average pattern reliability (confidence): {rules_df['confidence'].mean():.1%}")
    print()

    # Show significant patterns
    print("=== Most Significant Rating Patterns ===")
    print("These patterns show the strongest relationships between ratings:\n")
    
    top_patterns = rules_df.nlargest(5, 'confidence')
    
    for idx, rule in top_patterns.iterrows():
        antecedent_parts = [get_rating_description(f) for f in rule['antecedent'].split(' AND ')]
        consequent_parts = [get_rating_description(f) for f in rule['consequent'].split(' AND ')]
        
        print(f"Pattern {idx + 1}:")
        print("When:")
        for part in antecedent_parts:
            print(f"  • {part}")
        print("Then:")
        for part in consequent_parts:
            print(f"  • {part}")
        print(f"Reliability: {rule['confidence']:.1%}")
        print(f"Occurrence: {rule['support']:.1%} of all evaluations")
        print()

    # Key insights
    print("=== Key Insights ===")
    high_conf_rules = rules_df[rules_df['confidence'] >= 0.8]
    high_sup_rules = rules_df[rules_df['support'] >= 0.5]
    
    print(f"• Found {len(high_conf_rules)} highly reliable patterns (confidence ≥ 80%)")
    print(f"• Found {len(high_sup_rules)} frequently occurring patterns (support ≥ 50%)")

if __name__ == "__main__":
    analyze_rules() 