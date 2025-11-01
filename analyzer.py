"""
Skills Analyzer Module
Analyzes job data to find frequent and co-occurring skills.
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import json
import os


class SkillsAnalyzer:
    """Analyzes skills patterns in job listings."""
    
    def __init__(self):
        """Initialize the analyzer."""
        pass
    
    def load_jobs(self, filepath: str) -> pd.DataFrame:
        """Load jobs from CSV or JSON file."""
        # Security: Normalize path to prevent directory traversal
        filepath = os.path.normpath(filepath)
        
        # Validate file extension
        if not (filepath.endswith('.csv') or filepath.endswith('.json')):
            raise ValueError("Unsupported file format. Use CSV or JSON.")
        
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        
        return df
    
    def extract_skills(self, df: pd.DataFrame) -> List[List[str]]:
        """Extract skills from job listings into a list of lists."""
        skills_lists = []
        
        if 'skills_list' in df.columns:
            # If skills are already in list format
            skills_lists = df['skills_list'].dropna().tolist()
            # Convert to list if stored as string (safe parsing using json.loads instead of eval)
            parsed_lists = []
            for s in skills_lists:
                if isinstance(s, str) and s.startswith('['):
                    try:
                        parsed_lists.append(json.loads(s))
                    except (json.JSONDecodeError, ValueError):
                        # Fallback: treat as comma-separated string
                        parsed_lists.append([skill.strip() for skill in s.strip('[]').split(',') if skill.strip()])
                elif isinstance(s, list):
                    parsed_lists.append(s)
                else:
                    # Handle other types
                    parsed_lists.append([str(s)] if s else [])
            skills_lists = parsed_lists
        elif 'skills' in df.columns:
            # If skills are in comma-separated string format
            skills_lists = df['skills'].dropna().apply(
                lambda x: [s.strip() for s in str(x).split(',') if s.strip()]
            ).tolist()
        else:
            # Try to extract from description
            print("Warning: No skills column found. Extracting from descriptions.")
            skills_lists = []
        
        return skills_lists
    
    def get_skill_frequency(self, skills_lists: List[List[str]], top_n: int = 20) -> pd.DataFrame:
        """Calculate frequency of each skill."""
        skill_counter = Counter()
        
        for skills in skills_lists:
            if isinstance(skills, list):
                skill_counter.update([skill.lower().strip() for skill in skills])
        
        # Get top N skills
        top_skills = skill_counter.most_common(top_n)
        
        df = pd.DataFrame(top_skills, columns=['skill', 'frequency'])
        df['percentage'] = (df['frequency'] / len(skills_lists) * 100).round(2)
        
        return df
    
    def get_co_occurring_skills(self, skills_lists: List[List[str]], top_n: int = 20) -> pd.DataFrame:
        """Find skills that frequently appear together."""
        co_occurrence = defaultdict(int)
        
        for skills in skills_lists:
            if not isinstance(skills, list) or len(skills) < 2:
                continue
            
            # Normalize skills
            normalized_skills = [skill.lower().strip() for skill in skills]
            
            # Count pairs (undirected)
            for i in range(len(normalized_skills)):
                for j in range(i + 1, len(normalized_skills)):
                    pair = tuple(sorted([normalized_skills[i], normalized_skills[j]]))
                    co_occurrence[pair] += 1
        
        # Convert to DataFrame
        pairs = []
        for (skill1, skill2), count in sorted(co_occurrence.items(), key=lambda x: x[1], reverse=True)[:top_n]:
            pairs.append({
                'skill1': skill1,
                'skill2': skill2,
                'co_occurrence_count': count,
                'pair': f"{skill1} & {skill2}"
            })
        
        return pd.DataFrame(pairs)
    
    def analyze_by_title(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Analyze skills by job title."""
        results = {}
        
        if 'title' not in df.columns:
            print("Warning: No 'title' column found. Skipping title-based analysis.")
            return results
        
        titles = df['title'].unique()
        
        for title in titles:
            title_df = df[df['title'] == title]
            skills_lists = self.extract_skills(title_df)
            
            if len(skills_lists) > 0:
                freq_df = self.get_skill_frequency(skills_lists, top_n=15)
                results[title] = freq_df
        
        return results
    
    def analyze_by_location(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Analyze skills by location."""
        results = {}
        
        if 'location' not in df.columns:
            print("Warning: No 'location' column found. Skipping location-based analysis.")
            return results
        
        locations = df['location'].unique()
        
        for location in locations:
            location_df = df[df['location'] == location]
            skills_lists = self.extract_skills(location_df)
            
            if len(skills_lists) > 0:
                freq_df = self.get_skill_frequency(skills_lists, top_n=15)
                results[location] = freq_df
        
        return results
    
    def get_skill_matrix(self, skills_lists: List[List[str]], top_skills: List[str]) -> pd.DataFrame:
        """Create a skill co-occurrence matrix for top skills."""
        matrix = np.zeros((len(top_skills), len(top_skills)))
        
        for skills in skills_lists:
            if not isinstance(skills, list):
                continue
            
            normalized_skills = [skill.lower().strip() for skill in skills]
            indices = [top_skills.index(skill) for skill in normalized_skills if skill in top_skills]
            
            for i in indices:
                for j in indices:
                    if i != j:
                        matrix[i][j] += 1
        
        return pd.DataFrame(matrix, index=top_skills, columns=top_skills)
    
    def analyze_jobs(self, filepath: str) -> Dict:
        """Comprehensive analysis of job data."""
        df = self.load_jobs(filepath)
        skills_lists = self.extract_skills(df)
        
        analysis = {
            'total_jobs': len(df),
            'skill_frequency': self.get_skill_frequency(skills_lists),
            'co_occurring_skills': self.get_co_occurring_skills(skills_lists),
            'by_title': self.analyze_by_title(df),
            'by_location': self.analyze_by_location(df)
        }
        
        # Add skill matrix
        top_skills = analysis['skill_frequency']['skill'].head(10).tolist()
        analysis['skill_matrix'] = self.get_skill_matrix(skills_lists, top_skills)
        
        return analysis
    
    def save_analysis(self, analysis: Dict, output_path: str):
        """Save analysis results to JSON."""
        # Security: Normalize and validate output path
        output_path = os.path.normpath(output_path)
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Validate file extension
        if not output_path.endswith('.json'):
            raise ValueError("Output file must have .json extension")
        
        # Convert DataFrames to dictionaries for JSON serialization
        json_analysis = {
            'total_jobs': analysis['total_jobs'],
            'skill_frequency': analysis['skill_frequency'].to_dict('records'),
            'co_occurring_skills': analysis['co_occurring_skills'].to_dict('records'),
            'by_title': {
                title: df.to_dict('records')
                for title, df in analysis['by_title'].items()
            },
            'by_location': {
                location: df.to_dict('records')
                for location, df in analysis['by_location'].items()
            },
            'skill_matrix': analysis['skill_matrix'].to_dict()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_analysis, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    analyzer = SkillsAnalyzer()
    # Example usage
    # analysis = analyzer.analyze_jobs("data/raw/jobs_sample.json")
    # analyzer.save_analysis(analysis, "data/processed/analysis.json")

