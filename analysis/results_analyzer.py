"""
ICOTN Results Analysis for Academic Papers
Generates publication-ready statistics, tables, and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ICOTNResultsAnalyzer:
    """Analyzer for ICOTN experiment results"""
    
    def __init__(self, results_dir: str = "experiment_results"):
        self.results_dir = Path(results_dir)
        self.results = []
        self.df = None
    
    def load_results(self, pattern: str = "*_raw.json"):
        """Load experiment results from JSON files"""
        json_files = list(self.results_dir.glob(pattern))
        
        if not json_files:
            print(f"No result files found matching pattern: {pattern}")
            return
        
        print(f"📁 Loading {len(json_files)} result files...")
        
        all_data = []
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Extract individual attack results
                for model_key, result in data.get("results_by_model", {}).items():
                    result["experiment_name"] = data.get("experiment_name", "Unknown")
                    result["timestamp"] = data.get("timestamp", "Unknown")
                    all_data.append(result)
                
                print(f"   ✅ Loaded: {json_file.name}")
                
            except Exception as e:
                print(f"   ❌ Error loading {json_file.name}: {e}")
        
        if all_data:
            self.df = pd.DataFrame(all_data)
            print(f"📊 Total records loaded: {len(self.df)}")
            
            # Add derived columns
            self._add_derived_columns()
            
        else:
            print("❌ No data loaded successfully")
    
    def _add_derived_columns(self):
        """Add derived analysis columns"""
        if self.df is None:
            return
        
        # Success rate by model
        self.df['success_rate'] = self.df.groupby('model_key')['success'].transform('mean')
        
        # Query category (simplified)
        def categorize_query(query):
            query_lower = query.lower()
            if 'oxycodone' in query_lower or 'codeine' in query_lower or 'drug' in query_lower:
                return 'Drug Manufacturing'
            elif 'patient' in query_lower or 'medical' in query_lower or 'prescription' in query_lower:
                return 'Medical Records'
            elif 'security' in query_lower or 'location' in query_lower or 'distribution' in query_lower:
                return 'Security Information'
            elif 'inject' in query_lower or 'surgery' in query_lower:
                return 'Medical Procedures'
            else:
                return 'Other'
        
        self.df['query_category'] = self.df['query'].apply(categorize_query)
        
        # Model type
        def get_model_type(model_key):
            if 'gpt-oss' in model_key or 'deepseek' in model_key or 'o1' in model_key:
                return 'Reasoning Model'
            elif 'gpt-4' in model_key or 'claude' in model_key or 'gemini' in model_key:
                return 'Frontier Model'
            else:
                return 'Other Model'
        
        self.df['model_type'] = self.df['model_key'].apply(get_model_type)
        
        # Performance categories
        self.df['performance_category'] = pd.cut(
            self.df['success_rate'], 
            bins=[0, 0.2, 0.5, 0.8, 1.0], 
            labels=['Low Vulnerability', 'Medium Vulnerability', 'High Vulnerability', 'Critical Vulnerability']
        )
    
    def generate_summary_table(self) -> pd.DataFrame:
        """Generate publication-ready summary table"""
        if self.df is None:
            print("❌ No data loaded")
            return None
        
        summary = self.df.groupby(['model_key', 'model_name']).agg({
            'success': ['count', 'sum', 'mean'],
            'attempts': ['mean', 'std'],
            'total_time': 'mean',
            'avg_latency': 'mean'
        }).round(2)
        
        # Flatten column names
        summary.columns = [f"{col[0]}_{col[1]}" for col in summary.columns]
        summary = summary.reset_index()
        
        # Rename for publication
        summary = summary.rename(columns={
            'model_key': 'Model ID',
            'model_name': 'Model Name',
            'success_count': 'Total Tests',
            'success_sum': 'Successful Bypasses',
            'success_mean': 'Success Rate',
            'attempts_mean': 'Avg Attempts (Success)',
            'attempts_std': 'Std Attempts',
            'total_time_mean': 'Avg Time (s)',
            'avg_latency_mean': 'Avg Latency (s)'
        })
        
        # Sort by success rate (most vulnerable first)
        summary = summary.sort_values('Success Rate', ascending=False)
        
        return summary
    
    def generate_vulnerability_analysis(self) -> Dict:
        """Generate vulnerability analysis by model type and query category"""
        if self.df is None:
            return {}
        
        analysis = {}
        
        # Overall statistics
        analysis['overall'] = {
            'total_attacks': len(self.df),
            'successful_attacks': self.df['success'].sum(),
            'overall_success_rate': self.df['success'].mean(),
            'avg_attempts_success': self.df[self.df['success']]['attempts'].mean(),
            'avg_attempts_failure': self.df[~self.df['success']]['attempts'].mean()
        }
        
        # By model type
        analysis['by_model_type'] = (
            self.df.groupby('model_type')['success']
            .agg(['count', 'sum', 'mean'])
            .rename(columns={'count': 'total', 'sum': 'successful', 'mean': 'success_rate'})
            .to_dict('index')
        )
        
        # By query category  
        analysis['by_query_category'] = (
            self.df.groupby('query_category')['success']
            .agg(['count', 'sum', 'mean'])
            .rename(columns={'count': 'total', 'sum': 'successful', 'mean': 'success_rate'})
            .to_dict('index')
        )
        
        # Most vulnerable models
        model_vuln = (
            self.df.groupby(['model_key', 'model_name'])['success']
            .agg(['count', 'sum', 'mean'])
            .rename(columns={'count': 'total', 'sum': 'successful', 'mean': 'success_rate'})
            .sort_values('success_rate', ascending=False)
        )
        
        # Convert tuple keys to string keys for JSON serialization
        most_vuln_dict = {}
        for (model_key, model_name), stats in model_vuln.head(5).to_dict('index').items():
            most_vuln_dict[f"{model_key} ({model_name})"] = stats
        
        least_vuln_dict = {}
        for (model_key, model_name), stats in model_vuln.tail(5).to_dict('index').items():
            least_vuln_dict[f"{model_key} ({model_name})"] = stats
            
        analysis['most_vulnerable'] = most_vuln_dict
        analysis['least_vulnerable'] = least_vuln_dict
        
        return analysis
    
    def create_visualizations(self, output_dir: str = "analysis_output"):
        """Create publication-ready visualizations"""
        if self.df is None:
            print("❌ No data loaded")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set publication style
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })
        
        # 1. Success Rate by Model
        plt.figure(figsize=(12, 8))
        model_success = self.df.groupby(['model_key', 'model_name'])['success'].mean().sort_values(ascending=True)
        
        bars = plt.barh(range(len(model_success)), model_success.values)
        plt.yticks(range(len(model_success)), [f"{key}\n({name})" for key, name in model_success.index])
        plt.xlabel('Success Rate (Bypass Probability)')
        plt.title('ICOTN Attack Success Rate by Model')
        plt.grid(axis='x', alpha=0.3)
        
        # Color bars by vulnerability level
        for i, (bar, rate) in enumerate(zip(bars, model_success.values)):
            if rate > 0.8:
                bar.set_color('#ff4444')  # High vulnerability - red
            elif rate > 0.5:
                bar.set_color('#ff8800')  # Medium vulnerability - orange  
            elif rate > 0.2:
                bar.set_color('#ffcc00')  # Low vulnerability - yellow
            else:
                bar.set_color('#44cc44')  # Secure - green
        
        plt.tight_layout()
        plt.savefig(output_path / 'success_rate_by_model.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_path / 'success_rate_by_model.pdf', bbox_inches='tight')
        print(f"📊 Saved: success_rate_by_model.png/.pdf")
        
        # 2. Attempts Distribution
        plt.figure(figsize=(10, 6))
        
        successful_attempts = self.df[self.df['success']]['attempts']
        failed_attempts = self.df[~self.df['success']]['attempts']
        
        plt.hist(successful_attempts, bins=20, alpha=0.7, label='Successful Attacks', color='red')
        plt.hist(failed_attempts, bins=20, alpha=0.7, label='Failed Attacks', color='blue')
        
        plt.xlabel('Number of Attempts Required')
        plt.ylabel('Frequency')
        plt.title('Distribution of Attempts Required for ICOTN Attacks')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'attempts_distribution.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_path / 'attempts_distribution.pdf', bbox_inches='tight')
        print(f"📊 Saved: attempts_distribution.png/.pdf")
        
        # 3. Success Rate by Model Type and Query Category
        plt.figure(figsize=(12, 8))
        
        pivot_table = self.df.pivot_table(
            values='success', 
            index='model_type', 
            columns='query_category', 
            aggfunc='mean'
        )
        
        sns.heatmap(
            pivot_table, 
            annot=True, 
            fmt='.2f', 
            cmap='Reds', 
            cbar_kws={'label': 'Success Rate'},
            square=True
        )
        
        plt.title('ICOTN Success Rate by Model Type and Attack Category')
        plt.ylabel('Model Type')
        plt.xlabel('Attack Category')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / 'heatmap_success_by_category.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_path / 'heatmap_success_by_category.pdf', bbox_inches='tight')
        print(f"📊 Saved: heatmap_success_by_category.png/.pdf")
        
        # 4. Time vs Attempts Scatter
        plt.figure(figsize=(10, 6))
        
        # Separate successful and failed attacks
        success_data = self.df[self.df['success']]
        failure_data = self.df[~self.df['success']]
        
        plt.scatter(success_data['attempts'], success_data['total_time'], 
                   alpha=0.6, color='red', label='Successful', s=50)
        plt.scatter(failure_data['attempts'], failure_data['total_time'], 
                   alpha=0.6, color='blue', label='Failed', s=50)
        
        plt.xlabel('Number of Attempts')
        plt.ylabel('Total Time (seconds)')
        plt.title('Time vs Attempts for ICOTN Attacks')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'time_vs_attempts.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_path / 'time_vs_attempts.pdf', bbox_inches='tight')
        print(f"📊 Saved: time_vs_attempts.png/.pdf")
        
        plt.close('all')  # Close all figures to free memory
    
    def export_paper_ready_tables(self, output_dir: str = "analysis_output"):
        """Export LaTeX-ready tables for academic papers"""
        if self.df is None:
            print("❌ No data loaded")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Main results table
        summary_table = self.generate_summary_table()
        
        # Export as LaTeX
        latex_table = summary_table.to_latex(
            index=False,
            float_format='%.3f',
            caption='ICOTN Attack Results by Model',
            label='tab:icotn_results',
            column_format='|l|l|c|c|c|c|c|c|c|'
        )
        
        with open(output_path / 'results_table.tex', 'w') as f:
            f.write(latex_table)
        
        # Export as CSV for easy editing
        summary_table.to_csv(output_path / 'results_table.csv', index=False)
        
        # Vulnerability analysis table
        vuln_analysis = self.generate_vulnerability_analysis()
        
        # Create model type comparison table
        model_type_df = pd.DataFrame(vuln_analysis['by_model_type']).T
        model_type_df = model_type_df.round(3)
        
        model_type_latex = model_type_df.to_latex(
            float_format='%.3f',
            caption='ICOTN Vulnerability by Model Type',
            label='tab:vuln_by_type'
        )
        
        with open(output_path / 'vulnerability_by_type.tex', 'w') as f:
            f.write(model_type_latex)
        
        model_type_df.to_csv(output_path / 'vulnerability_by_type.csv')
        
        print(f"📄 Exported LaTeX tables to {output_path}")
        print(f"   - results_table.tex/.csv")
        print(f"   - vulnerability_by_type.tex/.csv")
    
    def generate_executive_summary(self) -> str:
        """Generate executive summary for paper abstract/conclusion"""
        if self.df is None:
            return "No data available for analysis."
        
        vuln_analysis = self.generate_vulnerability_analysis()
        overall = vuln_analysis['overall']
        
        total_tests = overall['total_attacks']
        success_rate = overall['overall_success_rate'] * 100
        avg_attempts = overall['avg_attempts_success']
        
        # Find most and least vulnerable models
        model_stats = self.df.groupby(['model_key', 'model_name'])['success'].mean()
        most_vulnerable = model_stats.idxmax()
        least_vulnerable = model_stats.idxmin()
        
        # Extract model info safely
        most_vuln_key, most_vuln_name = most_vulnerable
        least_vuln_key, least_vuln_name = least_vulnerable
        
        summary = f"""
ICOTN Vulnerability Assessment - Executive Summary

Our comprehensive evaluation tested {total_tests} attack scenarios across multiple state-of-the-art language models using the Iterative Chain-of-Thought Negation (ICOTN) methodology.

Key Findings:
• Overall bypass success rate: {success_rate:.1f}%
• Average attempts required for successful bypass: {avg_attempts:.1f}
• Most vulnerable model: {most_vuln_key} ({most_vuln_name}) - {model_stats[most_vulnerable]*100:.1f}% success rate
• Most robust model: {least_vuln_key} ({least_vuln_name}) - {model_stats[least_vulnerable]*100:.1f}% success rate

Model Type Analysis:"""

        for model_type, stats in vuln_analysis['by_model_type'].items():
            summary += f"\n• {model_type}: {stats['success_rate']*100:.1f}% bypass rate ({stats['successful']}/{stats['total']} attacks)"
        
        summary += f"""

Attack Category Analysis:"""
        
        for category, stats in vuln_analysis['by_query_category'].items():
            summary += f"\n• {category}: {stats['success_rate']*100:.1f}% success rate ({stats['successful']}/{stats['total']} attempts)"
        
        summary += f"""

These results demonstrate significant vulnerabilities in reasoning-based language models when subjected to sophisticated prompt engineering attacks, highlighting the need for improved safety measures in production deployments.
"""
        
        return summary
    
    def full_analysis_report(self, output_dir: str = "analysis_output"):
        """Generate complete analysis report"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("📈 Generating comprehensive analysis report...")
        
        # 1. Generate summary tables
        summary_table = self.generate_summary_table()
        print(f"✅ Summary table: {len(summary_table)} models analyzed")
        
        # 2. Create visualizations  
        self.create_visualizations(output_dir)
        
        # 3. Export LaTeX tables
        self.export_paper_ready_tables(output_dir)
        
        # 4. Generate executive summary
        exec_summary = self.generate_executive_summary()
        with open(output_path / 'executive_summary.txt', 'w') as f:
            f.write(exec_summary)
        
        # 5. Full vulnerability analysis
        vuln_analysis = self.generate_vulnerability_analysis()
        with open(output_path / 'vulnerability_analysis.json', 'w') as f:
            json.dump(vuln_analysis, f, indent=2, default=str)
        
        print(f"\n🎉 Complete analysis report generated in: {output_path}")
        print("📁 Files created:")
        print("   📊 Visualizations: *.png, *.pdf")
        print("   📄 LaTeX tables: *.tex")
        print("   📈 CSV data: *.csv") 
        print("   📝 Executive summary: executive_summary.txt")
        print("   🔍 Detailed analysis: vulnerability_analysis.json")

def main():
    parser = argparse.ArgumentParser(description="ICOTN Results Analysis")
    parser.add_argument("--results-dir", "-r", default="experiment_results", 
                       help="Directory containing experiment results")
    parser.add_argument("--output-dir", "-o", default="analysis_output",
                       help="Output directory for analysis")
    parser.add_argument("--pattern", "-p", default="*_raw.json",
                       help="File pattern for result files")
    
    args = parser.parse_args()
    
    print("📊 ICOTN Results Analyzer")
    print("=" * 50)
    
    analyzer = ICOTNResultsAnalyzer(args.results_dir)
    analyzer.load_results(args.pattern)
    
    if analyzer.df is not None:
        analyzer.full_analysis_report(args.output_dir)
    else:
        print("❌ No data loaded. Please check your results directory and file pattern.")

if __name__ == "__main__":
    main()
