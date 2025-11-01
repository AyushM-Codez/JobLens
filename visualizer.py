"""
Visualization Module
Creates interactive charts showing hiring patterns and trends.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional
import os
from datetime import datetime


class JobVisualizer:
    """Creates visualizations for job market data."""
    
    def __init__(self, output_dir: str = "output"):
        """Initialize visualizer with output directory."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def plot_skill_frequency(self, skill_freq_df: pd.DataFrame, top_n: int = 20, save: bool = True):
        """Create bar chart of most frequent skills."""
        top_skills = skill_freq_df.head(top_n)
        
        fig = px.bar(
            top_skills,
            x='frequency',
            y='skill',
            orientation='h',
            title=f'Top {top_n} Most In-Demand Skills',
            labels={'frequency': 'Frequency', 'skill': 'Skill'},
            color='frequency',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=600,
            showlegend=False
        )
        
        if save:
            filename = os.path.join(self.output_dir, 'skill_frequency.html')
            fig.write_html(filename)
            print(f"Saved skill frequency chart to {filename}")
        
        return fig
    
    def plot_co_occurring_skills(self, co_occur_df: pd.DataFrame, top_n: int = 20, save: bool = True):
        """Create bar chart of co-occurring skills."""
        top_pairs = co_occur_df.head(top_n)
        
        fig = px.bar(
            top_pairs,
            x='co_occurrence_count',
            y='pair',
            orientation='h',
            title=f'Top {top_n} Co-Occurring Skill Pairs',
            labels={'co_occurrence_count': 'Co-occurrence Count', 'pair': 'Skill Pair'},
            color='co_occurrence_count',
            color_continuous_scale='plasma'
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=600,
            showlegend=False
        )
        
        if save:
            filename = os.path.join(self.output_dir, 'co_occurring_skills.html')
            fig.write_html(filename)
            print(f"Saved co-occurring skills chart to {filename}")
        
        return fig
    
    def plot_hiring_trends(self, df: pd.DataFrame, save: bool = True):
        """Plot job posting trends over time."""
        if 'posted_date' not in df.columns:
            print("Warning: No 'posted_date' column found. Skipping trend visualization.")
            return None
        
        df['posted_date'] = pd.to_datetime(df['posted_date'])
        df['date'] = df['posted_date'].dt.date
        
        daily_counts = df.groupby('date').size().reset_index(name='job_count')
        daily_counts = daily_counts.sort_values('date')
        
        fig = px.line(
            daily_counts,
            x='date',
            y='job_count',
            title='Job Postings Over Time',
            labels={'date': 'Date', 'job_count': 'Number of Job Postings'},
            markers=True
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of Job Postings",
            hovermode='x unified'
        )
        
        if save:
            filename = os.path.join(self.output_dir, 'hiring_trends.html')
            fig.write_html(filename)
            print(f"Saved hiring trends chart to {filename}")
        
        return fig
    
    def plot_location_distribution(self, df: pd.DataFrame, save: bool = True):
        """Plot distribution of jobs by location."""
        if 'location' not in df.columns:
            print("Warning: No 'location' column found. Skipping location visualization.")
            return None
        
        location_counts = df['location'].value_counts().head(15)
        
        fig = px.pie(
            values=location_counts.values,
            names=location_counts.index,
            title='Job Distribution by Location (Top 15)'
        )
        
        if save:
            filename = os.path.join(self.output_dir, 'location_distribution.html')
            fig.write_html(filename)
            print(f"Saved location distribution chart to {filename}")
        
        return fig
    
    def plot_job_title_distribution(self, df: pd.DataFrame, save: bool = True):
        """Plot distribution of jobs by title."""
        if 'title' not in df.columns:
            print("Warning: No 'title' column found. Skipping title visualization.")
            return None
        
        title_counts = df['title'].value_counts().head(15)
        
        fig = px.bar(
            x=title_counts.values,
            y=title_counts.index,
            orientation='h',
            title='Job Distribution by Title (Top 15)',
            labels={'x': 'Number of Jobs', 'y': 'Job Title'},
            color=title_counts.values,
            color_continuous_scale='blues'
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=600,
            showlegend=False
        )
        
        if save:
            filename = os.path.join(self.output_dir, 'job_title_distribution.html')
            fig.write_html(filename)
            print(f"Saved job title distribution chart to {filename}")
        
        return fig
    
    def plot_skill_heatmap(self, skill_matrix: pd.DataFrame, save: bool = True):
        """Create heatmap of skill co-occurrence matrix."""
        fig = px.imshow(
            skill_matrix,
            labels=dict(x="Skill", y="Skill", color="Co-occurrence"),
            title="Skill Co-occurrence Heatmap",
            aspect="auto",
            color_continuous_scale='YlOrRd'
        )
        
        if save:
            filename = os.path.join(self.output_dir, 'skill_heatmap.html')
            fig.write_html(filename)
            print(f"Saved skill heatmap to {filename}")
        
        return fig
    
    def plot_salary_distribution(self, df: pd.DataFrame, save: bool = True):
        """Plot salary distribution if available."""
        if 'salary' not in df.columns:
            print("Warning: No 'salary' column found. Skipping salary visualization.")
            return None
        
        # Extract numeric salary values (simplified - would need more sophisticated parsing)
        def extract_avg_salary(salary_str):
            if pd.isna(salary_str) or salary_str == 'N/A':
                return None
            import re
            numbers = re.findall(r'\d+', str(salary_str).replace(',', ''))
            if len(numbers) >= 2:
                return (int(numbers[0]) + int(numbers[1])) / 2
            elif len(numbers) == 1:
                return int(numbers[0])
            return None
        
        df['avg_salary'] = df['salary'].apply(extract_avg_salary)
        df_with_salary = df[df['avg_salary'].notna()]
        
        if len(df_with_salary) == 0:
            print("Warning: Could not extract salary data. Skipping salary visualization.")
            return None
        
        fig = px.histogram(
            df_with_salary,
            x='avg_salary',
            nbins=30,
            title='Salary Distribution',
            labels={'avg_salary': 'Average Salary ($)', 'count': 'Number of Jobs'}
        )
        
        if save:
            filename = os.path.join(self.output_dir, 'salary_distribution.html')
            fig.write_html(filename)
            print(f"Saved salary distribution chart to {filename}")
        
        return fig
    
    def create_dashboard(self, df: pd.DataFrame, analysis: Dict, save: bool = True):
        """Create a comprehensive dashboard with multiple visualizations."""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Hiring Trends', 'Top Skills', 'Job Titles', 'Locations'),
            specs=[[{"secondary_y": False}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "pie"}]]
        )
        
        # Hiring trends
        if 'posted_date' in df.columns:
            df['posted_date'] = pd.to_datetime(df['posted_date'])
            daily_counts = df.groupby(df['posted_date'].dt.date).size()
            fig.add_trace(
                go.Scatter(x=daily_counts.index, y=daily_counts.values, mode='lines+markers', name='Jobs'),
                row=1, col=1
            )
        
        # Top skills
        if 'skill_frequency' in analysis:
            top_skills = analysis['skill_frequency'].head(10)
            fig.add_trace(
                go.Bar(x=top_skills['frequency'], y=top_skills['skill'], orientation='h', name='Skills'),
                row=1, col=2
            )
        
        # Job titles
        if 'title' in df.columns:
            title_counts = df['title'].value_counts().head(10)
            fig.add_trace(
                go.Bar(x=title_counts.index, y=title_counts.values, name='Titles'),
                row=2, col=1
            )
        
        # Locations
        if 'location' in df.columns:
            location_counts = df['location'].value_counts().head(8)
            fig.add_trace(
                go.Pie(labels=location_counts.index, values=location_counts.values, name='Locations'),
                row=2, col=2
            )
        
        fig.update_layout(
            height=1000,
            title_text="Job Market Analysis Dashboard",
            showlegend=False
        )
        
        if save:
            filename = os.path.join(self.output_dir, 'dashboard.html')
            fig.write_html(filename)
            print(f"Saved dashboard to {filename}")
        
        return fig


if __name__ == "__main__":
    visualizer = JobVisualizer()
    # Example usage
    # visualizer.plot_skill_frequency(analysis['skill_frequency'])

