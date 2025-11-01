"""
Job Scraper Module
Scrapes and organizes real-world job listings from various sources.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
from datetime import datetime
from typing import List, Dict, Optional
import os
from tqdm import tqdm


class JobScraper:
    """Scrapes job listings from various sources and organizes the data."""
    
    def __init__(self, output_dir: str = "data/raw"):
        """Initialize the scraper with output directory."""
        self.output_dir = output_dir
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        os.makedirs(output_dir, exist_ok=True)
    
    def scrape_indeed_sample(self, job_title: str, location: str = "", max_results: int = 50) -> List[Dict]:
        """
        Scrape job listings from Indeed (sample implementation).
        Note: For production, use official APIs or be mindful of rate limits and ToS.
        """
        jobs = []
        try:
            # Security: Sanitize inputs to prevent URL injection
            from urllib.parse import quote
            safe_job_title = quote(str(job_title).strip()[:100])  # Limit length
            safe_location = quote(str(location).strip()[:100])
            
            # Example: Simple scraping structure (adapt based on actual site structure)
            # This is a template - real implementation would need to adapt to actual HTML structure
            url = f"https://www.indeed.com/jobs?q={safe_job_title}&l={safe_location}"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                # Parse job listings (structure may vary)
                job_cards = soup.find_all('div', class_='job_seen_beacon')
                
                for card in job_cards[:max_results]:
                    try:
                        title_elem = card.find('h2', class_='jobTitle')
                        company_elem = card.find('span', class_='companyName')
                        location_elem = card.find('div', class_='companyLocation')
                        salary_elem = card.find('span', class_='salary-snippet')
                        
                        job = {
                            'title': title_elem.get_text(strip=True) if title_elem else 'N/A',
                            'company': company_elem.get_text(strip=True) if company_elem else 'N/A',
                            'location': location_elem.get_text(strip=True) if location_elem else location,
                            'salary': salary_elem.get_text(strip=True) if salary_elem else 'N/A',
                            'description': '',
                            'posted_date': datetime.now().strftime('%Y-%m-%d'),
                            'source': 'indeed'
                        }
                        jobs.append(job)
                    except Exception as e:
                        continue
        except Exception as e:
            print(f"Error scraping Indeed: {e}")
        
        return jobs
    
    def load_sample_data(self, num_jobs: int = 100) -> List[Dict]:
        """
        Generate sample job data for testing when scraping is not available.
        This simulates real job listings with realistic data.
        """
        import random
        
        job_titles = [
            "Software Engineer", "Data Scientist", "Machine Learning Engineer",
            "Frontend Developer", "Backend Developer", "Full Stack Developer",
            "DevOps Engineer", "Product Manager", "UX Designer", "Data Analyst",
            "Cloud Architect", "Cybersecurity Analyst", "Mobile Developer",
            "QA Engineer", "Business Analyst"
        ]
        
        companies = [
            "Tech Corp", "Data Solutions Inc", "Cloud Services Ltd",
            "Innovation Labs", "Digital Ventures", "Future Systems",
            "Smart Tech", "Global Software", "Creative Solutions",
            "Enterprise Apps", "StartupHub", "BigTech Company"
        ]
        
        locations = [
            "San Francisco, CA", "New York, NY", "Austin, TX", "Seattle, WA",
            "Boston, MA", "Chicago, IL", "Los Angeles, CA", "Denver, CO",
            "Remote", "Remote, US", "Hybrid", "London, UK", "Toronto, Canada"
        ]
        
        skills_pool = [
            "Python", "Java", "JavaScript", "React", "Node.js", "SQL",
            "AWS", "Docker", "Kubernetes", "TensorFlow", "PyTorch",
            "Git", "MongoDB", "PostgreSQL", "Redis", "Kafka", "Spark",
            "Machine Learning", "Deep Learning", "Data Analysis", "REST APIs",
            "GraphQL", "TypeScript", "Vue.js", "Angular", "CI/CD", "Linux"
        ]
        
        salaries = [
            "$80,000 - $120,000", "$100,000 - $150,000", "$120,000 - $180,000",
            "$150,000 - $200,000", "$90,000 - $130,000", "$110,000 - $160,000"
        ]
        
        jobs = []
        dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
        
        for i in range(num_jobs):
            title = random.choice(job_titles)
            num_skills = random.randint(3, 8)
            job_skills = random.sample(skills_pool, num_skills)
            
            job = {
                'id': f"job_{i+1:04d}",
                'title': title,
                'company': random.choice(companies),
                'location': random.choice(locations),
                'salary': random.choice(salaries),
                'skills': ', '.join(job_skills),
                'skills_list': job_skills,
                'description': f"Looking for an experienced {title} with skills in {', '.join(job_skills[:3])}. Must have strong problem-solving abilities and team collaboration skills.",
                'posted_date': random.choice(dates).strftime('%Y-%m-%d'),
                'source': 'sample'
            }
            jobs.append(job)
            time.sleep(0.01)  # Simulate processing time
        
        return jobs
    
    def save_jobs(self, jobs: List[Dict], filename: Optional[str] = None) -> str:
        """Save scraped jobs to JSON and CSV files."""
        if filename is None:
            filename = f"jobs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Security: Sanitize filename to prevent path traversal
        import re
        filename = re.sub(r'[^\w\-_\.]', '_', str(filename))[:100]  # Remove special chars, limit length
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save as JSON
        json_path = os.path.join(self.output_dir, f"{filename}.json")
        # Additional security: Normalize path
        json_path = os.path.normpath(json_path)
        # Ensure path is within output directory
        if not json_path.startswith(os.path.abspath(self.output_dir)):
            raise ValueError("Invalid file path detected")
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(jobs, f, indent=2, ensure_ascii=False)
        
        # Save as CSV
        df = pd.DataFrame(jobs)
        # Flatten skills_list if exists
        if 'skills_list' in df.columns:
            df['skills'] = df['skills_list'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
            df = df.drop(columns=['skills_list'], errors='ignore')
        
        csv_path = os.path.join(self.output_dir, f"{filename}.csv")
        csv_path = os.path.normpath(csv_path)
        # Ensure path is within output directory
        if not csv_path.startswith(os.path.abspath(self.output_dir)):
            raise ValueError("Invalid file path detected")
        
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        print(f"Saved {len(jobs)} jobs to {json_path} and {csv_path}")
        return json_path
    
    def load_jobs(self, filepath: str) -> List[Dict]:
        """Load jobs from JSON file."""
        # Security: Normalize and validate file path
        filepath = os.path.normpath(filepath)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if not filepath.endswith('.json'):
            raise ValueError("Only JSON files are supported for loading.")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            jobs = json.load(f)
        return jobs
    
    def scrape_multiple_sources(self, job_title: str, location: str = "", num_jobs: int = 100) -> List[Dict]:
        """Scrape from multiple sources and combine results."""
        all_jobs = []
        
        print("Generating sample job data...")
        sample_jobs = self.load_sample_data(num_jobs)
        all_jobs.extend(sample_jobs)
        
        # Add more sources here as needed
        # all_jobs.extend(self.scrape_indeed_sample(job_title, location))
        
        return all_jobs


if __name__ == "__main__":
    scraper = JobScraper()
    jobs = scraper.scrape_multiple_sources("Software Engineer", num_jobs=100)
    scraper.save_jobs(jobs)

