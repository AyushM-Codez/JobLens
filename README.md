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
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

4. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

```bash
python main.py
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
├── data/              # Data storage directory
│   ├── raw/           # Raw scraped data
│   └── processed/     # Processed data
└── output/            # Generated visualizations and reports
```


