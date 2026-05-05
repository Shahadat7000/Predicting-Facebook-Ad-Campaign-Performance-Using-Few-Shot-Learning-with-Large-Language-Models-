"""
Success Metrics Summary Module
Generates comprehensive summary of success metrics distribution
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def generate_success_metrics_report(df: pd.DataFrame, output_dir: str = "visualizations"):
    """
    Generate comprehensive report on success metrics
    
    Args:
        df: Processed dataframe with success labels
        output_dir: Directory to save visualizations
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("SUCCESS METRICS COMPREHENSIVE REPORT")
    print("="*60)
    
    # Basic statistics
    print("\n1. DATASET OVERVIEW")
    print("-"*40)
    print(f"Total number of campaigns: {len(df)}")
    print(f"Number of features: {len(df.columns)}")
    print(f"Features: {', '.join(df.columns[:10])}...")
    
    # Success distribution
    print("\n2. SUCCESS DISTRIBUTION")
    print("-"*40)
    success_count = df['is_success'].sum()
    fail_count = len(df) - success_count
    success_rate = (success_count / len(df)) * 100
    
    print(f"Successful campaigns: {success_count} ({success_rate:.2f}%)")
    print(f"Unsuccessful campaigns: {fail_count} ({100-success_rate:.2f}%)")
    
    # Demographic breakdown
    print("\n3. DEMOGRAPHIC BREAKDOWN")
    print("-"*40)
    
    print("\nGender distribution:")
    gender_dist = df['gender'].value_counts()
    for gender, count in gender_dist.items():
        pct = (count/len(df))*100
        print(f"  {gender}: {count} ({pct:.1f}%)")
    
    print("\nAge group distribution:")
    age_dist = df['age'].value_counts().sort_index()
    for age, count in age_dist.items():
        pct = (count/len(df))*100
        print(f"  {age}: {count} ({pct:.1f}%)")
    
    # Success by demographic
    print("\n4. SUCCESS RATE BY DEMOGRAPHIC")
    print("-"*40)
    
    print("\nBy Gender:")
    success_by_gender = df.groupby('gender')['is_success'].mean() * 100
    for gender, rate in success_by_gender.items():
        print(f"  {gender}: {rate:.2f}%")
    
    print("\nBy Age Group:")
    success_by_age = df.groupby('age')['is_success'].mean() * 100
    for age, rate in success_by_age.items():
        print(f"  {age}: {rate:.2f}%")
    
    # Metric statistics
    print("\n5. METRIC STATISTICS")
    print("-"*40)
    metrics = ['CTR', 'CPC', 'Conversion_Rate']
    
    for metric in metrics:
        print(f"\n{metric}:")
        print(f"  Mean: {df[metric].mean():.4f}")
        print(f"  Median: {df[metric].median():.4f}")
        print(f"  Std Dev: {df[metric].std():.4f}")
        print(f"  Min: {df[metric].min():.4f}")
        print(f"  Max: {df[metric].max():.4f}")
    
    # Generate visualizations
    print("\n6. GENERATING VISUALIZATIONS")
    print("-"*40)
    
    # Success distribution pie chart
    plt.figure(figsize=(8, 6))
    plt.pie([success_count, fail_count], 
            labels=['Successful', 'Unsuccessful'],
            autopct='%1.1f%%',
            colors=['#2ecc71', '#e74c3c'],
            startangle=90)
    plt.title('Campaign Success Distribution', fontsize=14, fontweight='bold')
    plt.savefig(f"{output_dir}/success_distribution.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir}/success_distribution.png")
    
    # Metrics distribution by success status
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        success_data = df[df['is_success'] == 1][metric].dropna()
        fail_data = df[df['is_success'] == 0][metric].dropna()
        
        ax.hist(success_data, alpha=0.7, bins=20, label='Successful', color='#2ecc71')
        ax.hist(fail_data, alpha=0.7, bins=20, label='Unsuccessful', color='#e74c3c')
        ax.set_xlabel(metric)
        ax.set_ylabel('Frequency')
        ax.set_title(f'{metric} Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_distribution.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir}/metrics_distribution.png")
    
    print("\n" + "="*60)
    print("REPORT GENERATION COMPLETE")
    print("="*60)
    
    return {
        'total_campaigns': len(df),
        'success_count': int(success_count),
        'fail_count': int(fail_count),
        'success_rate': float(success_rate),
        'gender_distribution': gender_dist.to_dict(),
        'age_distribution': age_dist.to_dict(),
        'success_by_gender': success_by_gender.to_dict(),
        'success_by_age': success_by_age.to_dict()
    }

if __name__ == "__main__":
    # Load processed data
    df = pd.read_csv("data/processed_data.csv")
    report = generate_success_metrics_report(df)