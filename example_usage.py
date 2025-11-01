"""
Example usage of Joblens components
"""

from scraper import JobScraper
from analyzer import SkillsAnalyzer
from visualizer import JobVisualizer
from nlp_processor import NLPProcessor
from predictor import JobPredictor
import pandas as pd


def example_workflow():
    """Demonstrate a complete workflow."""
    print("Joblens Example Workflow\n" + "="*50)
    
    # 1. Scrape jobs
    print("\n1. Collecting job data...")
    scraper = JobScraper()
    jobs = scraper.load_sample_data(num_jobs=50)
    scraper.save_jobs(jobs, "example_jobs")
    print(f"   Collected {len(jobs)} jobs")
    
    # 2. Analyze skills
    print("\n2. Analyzing skills...")
    analyzer = SkillsAnalyzer()
    scraper.save_jobs(jobs, "temp_for_analysis")
    analysis = analyzer.analyze_jobs("data/raw/temp_for_analysis.json")
    print(f"   Top skill: {analysis['skill_frequency'].iloc[0]['skill']} "
          f"({analysis['skill_frequency'].iloc[0]['frequency']} times)")
    
    # 3. Visualize
    print("\n3. Creating visualizations...")
    visualizer = JobVisualizer()
    df = pd.DataFrame(jobs)
    visualizer.plot_skill_frequency(analysis['skill_frequency'], top_n=10)
    visualizer.plot_hiring_trends(df)
    print("   Visualizations saved to output/")
    
    # 4. NLP processing
    print("\n4. Processing descriptions with NLP...")
    nlp = NLPProcessor()
    if 'description' in df.columns:
        nlp.process_and_visualize(df)
        print("   Word clouds generated")
    
    # 5. Predictions
    print("\n5. Running predictions...")
    predictor = JobPredictor()
    if 'salary' in df.columns:
        salary_results = predictor.predict_salary(df)
        if 'error' not in salary_results:
            print(f"   Salary prediction R²: {salary_results['best_model_metrics']['r2_score']:.3f}")
    
    print("\n" + "="*50)
    print("Example workflow completed!")


if __name__ == "__main__":
    example_workflow()

