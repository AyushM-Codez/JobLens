# Joblens - Job Market Analysis Tool

Joblens is a comprehensive job market analysis platform that collects, analyzes, and visualizes job market data to help job seekers and recruiters understand hiring trends.

## Features

- 🧹 **Job Data Collection**: Scrapes and organizes real-world job listings (titles, skills, locations, salaries)
- 🔍 **Skills Analysis**: Finds the most frequent and co-occurring skills across different roles
- 📊 **Trend Visualization**: Creates interactive charts showing hiring patterns over time
- 🗣️ **NLP Processing**: Uses NLP to extract keywords and generate word clouds from job descriptions
- ⚡ **Predictions**: ML models to forecast trends or salary ranges

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Joblens
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data:
```bash
python setup.py
```
Or manually:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

4. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

### Basic Usage

Run the main application:
```bash
python main.py
```

### Advanced Usage

```bash
# Custom job search
python main.py --job-title "Data Scientist" --location "San Francisco" --num-jobs 200

# Skip scraping and use existing data
python main.py --skip-scrape --data-file data/raw/jobs_20240101_120000.json

# Include ML predictions
python main.py --run-predictions --num-jobs 150

# Example workflow
python example_usage.py
```

### Programmatic Usage

```python
from scraper import JobScraper
from analyzer import SkillsAnalyzer
from visualizer import JobVisualizer
from nlp_processor import NLPProcessor
from predictor import JobPredictor

# Scrape jobs
scraper = JobScraper()
jobs = scraper.load_sample_data(num_jobs=100)
scraper.save_jobs(jobs)

# Analyze skills
analyzer = SkillsAnalyzer()
analysis = analyzer.analyze_jobs("data/raw/jobs_sample.json")

# Create visualizations
visualizer = JobVisualizer()
visualizer.plot_skill_frequency(analysis['skill_frequency'])
visualizer.create_dashboard(df, analysis)

# Process with NLP
nlp = NLPProcessor()
nlp.process_and_visualize(df)

# Run predictions
predictor = JobPredictor()
salary_results = predictor.predict_salary(df)
trends = predictor.predict_hiring_trends(df, days_ahead=30)
```

## Project Structure

```
Joblens/
├── scraper.py          # Job data scraping module
├── analyzer.py         # Skills analysis module
├── visualizer.py       # Visualization module
├── nlp_processor.py    # NLP and word cloud generation
├── predictor.py        # ML prediction models
├── main.py            # Main application
├── setup.py           # Setup script for NLTK data
├── example_usage.py   # Example workflow demonstration
├── requirements.txt   # Python dependencies
├── data/              # Data storage directory
│   ├── raw/           # Raw scraped data
│   └── processed/     # Processed data
├── output/            # Generated visualizations and reports
└── models/            # Saved ML models
```

## Output

The application generates:

- **Interactive HTML Charts**: Skill frequency, co-occurring skills, hiring trends, location distribution, etc.
- **Word Clouds**: Visual representation of keywords from job descriptions
- **Analysis Reports**: JSON files with detailed skill analysis
- **Predictions**: Salary predictions and trend forecasts

All outputs are saved in the `output/` directory.

## Requirements

- Python 3.8+
- See `requirements.txt` for full list of dependencies

