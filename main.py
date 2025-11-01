"""
Main Application
Joblens - Job Market Analysis Tool
"""

import os
import argparse
from datetime import datetime
from scraper import JobScraper
from analyzer import SkillsAnalyzer
from visualizer import JobVisualizer
from nlp_processor import NLPProcessor
from predictor import JobPredictor
import pandas as pd


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description='Joblens - Job Market Analysis Tool')
    parser.add_argument('--job-title', type=str, default='Software Engineer', 
                       help='Job title to search for')
    parser.add_argument('--location', type=str, default='', 
                       help='Location to search in')
    parser.add_argument('--num-jobs', type=int, default=100, 
                       help='Number of jobs to scrape')
    parser.add_argument('--skip-scrape', action='store_true', 
                       help='Skip scraping and use existing data')
    parser.add_argument('--data-file', type=str, 
                       help='Path to existing data file (CSV or JSON)')
    parser.add_argument('--run-predictions', action='store_true', 
                       help='Run ML predictions')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Joblens - Job Market Analysis Tool")
    print("=" * 60)
    print()
    
    # Create output directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    # Step 1: Scrape job data
    scraper = JobScraper()
    if args.skip_scrape and args.data_file:
        print(f"Loading existing data from {args.data_file}...")
        # Security: Validate and sanitize input file path
        import os
        data_file = os.path.normpath(args.data_file)
        if not os.path.exists(data_file):
            print(f"  ✗ File not found: {data_file}")
            return
        
        if data_file.endswith('.csv'):
            df = pd.read_csv(data_file)
            jobs = df.to_dict('records')
        else:
            jobs = scraper.load_jobs(data_file)
    else:
        print("Step 1: Collecting job data...")
        print(f"  Searching for: {args.job_title} in {args.location or 'All locations'}")
        print(f"  Target: {args.num_jobs} jobs")
        jobs = scraper.scrape_multiple_sources(
            args.job_title, 
            args.location, 
            args.num_jobs
        )
        
        if jobs:
            data_file = scraper.save_jobs(jobs)
            print(f"  ✓ Collected {len(jobs)} job listings")
        else:
            print("  ✗ No jobs collected. Exiting.")
            return
        df = pd.DataFrame(jobs)
    
    print()
    
    # Step 2: Analyze skills
    print("Step 2: Analyzing skills...")
    analyzer = SkillsAnalyzer()
    
    # Save to temp file for analyzer
    temp_file = "data/raw/temp_jobs.json"
    scraper.save_jobs(jobs, "temp_jobs")
    
    analysis = analyzer.analyze_jobs(temp_file)
    print(f"  ✓ Analyzed {analysis['total_jobs']} jobs")
    print(f"  ✓ Found {len(analysis['skill_frequency'])} unique skills")
    
    # Save analysis
    analysis_file = f"data/processed/analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    analyzer.save_analysis(analysis, analysis_file)
    print(f"  ✓ Saved analysis to {analysis_file}")
    print()
    
    # Step 3: Create visualizations
    print("Step 3: Creating visualizations...")
    visualizer = JobVisualizer()
    
    visualizer.plot_skill_frequency(analysis['skill_frequency'], top_n=20)
    visualizer.plot_co_occurring_skills(analysis['co_occurring_skills'], top_n=20)
    visualizer.plot_hiring_trends(df)
    visualizer.plot_location_distribution(df)
    visualizer.plot_job_title_distribution(df)
    visualizer.plot_skill_heatmap(analysis['skill_matrix'])
    
    if 'salary' in df.columns:
        visualizer.plot_salary_distribution(df)
    
    visualizer.create_dashboard(df, analysis)
    print("  ✓ Created all visualizations")
    print()
    
    # Step 4: NLP processing
    print("Step 4: Processing job descriptions with NLP...")
    nlp = NLPProcessor()
    
    if 'description' in df.columns:
        nlp_analysis = nlp.process_and_visualize(df, 'description')
        print(f"  ✓ Extracted {len(nlp_analysis['keywords'])} keywords")
        print(f"  ✓ Identified {len(nlp_analysis['skills'])} technical skills")
        print("  ✓ Generated word clouds")
    else:
        print("  ⚠ No description column found. Skipping NLP processing.")
    print()
    
    # Step 5: ML Predictions (optional)
    if args.run_predictions:
        print("Step 5: Running ML predictions...")
        predictor = JobPredictor()
        
        # Salary prediction
        if 'salary' in df.columns:
            print("  Predicting salaries...")
            salary_results = predictor.predict_salary(df)
            if 'error' not in salary_results:
                print(f"    ✓ Best model: {salary_results['model']}")
                print(f"    ✓ R² Score: {salary_results['best_model_metrics']['r2_score']:.3f}")
                print(f"    ✓ MAE: ${salary_results['best_model_metrics']['mae']:,.2f}")
            else:
                print(f"    ⚠ {salary_results['error']}")
        
        # Trend prediction
        if 'posted_date' in df.columns:
            print("  Predicting hiring trends...")
            trends = predictor.predict_hiring_trends(df, days_ahead=30)
            if not trends.empty:
                avg_predicted = trends['predicted_job_count'].mean()
                print(f"    ✓ Average predicted daily jobs (next 30 days): {avg_predicted:.1f}")
        
        # Skill demand prediction
        skills_lists = analyzer.extract_skills(df)
        if skills_lists:
            print("  Predicting skill demand...")
            skill_demand = predictor.predict_skill_demand(df, skills_lists, days_ahead=30)
            if skill_demand:
                top_skills = sorted(skill_demand.items(), 
                                  key=lambda x: x[1]['predicted_demand'], 
                                  reverse=True)[:5]
                print("    Top 5 predicted in-demand skills:")
                for skill, data in top_skills:
                    print(f"      • {skill}: {data['predicted_demand']:.1f} (growth: {data['growth_rate']*100:.1f}%)")
        print()
    
    # Summary
    print("=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"✓ Total jobs analyzed: {len(jobs)}")
    print(f"✓ Top skill: {analysis['skill_frequency'].iloc[0]['skill']} "
          f"({analysis['skill_frequency'].iloc[0]['frequency']} occurrences)")
    print(f"✓ Visualizations saved to: output/")
    print(f"✓ Analysis data saved to: data/processed/")
    print()
    print("Open the HTML files in the 'output' directory to view interactive charts!")


if __name__ == "__main__":
    main()

