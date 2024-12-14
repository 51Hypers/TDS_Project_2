import os
import sys
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import httpx
import chardet
import logging

# Constants
API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
AIPROXY_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjIwMDU1OTdAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.mGtFocaNamOEpoh3Y6WUB-xoAJJzW3EQntzLwbHUSXg"

# Logging Configuration
logging.basicConfig(
    filename="autolysis.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_exception(e):
    """Log exceptions to the log file."""
    logging.error(f"Exception occurred: {e}", exc_info=True)

def load_data(file_path):
    """Load CSV data with encoding detection."""
    try:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        encoding = result['encoding']
        df = pd.read_csv(file_path, encoding=encoding)
        logging.info(f"Data loaded successfully from {file_path}")
        return df
    except Exception as e:
        log_exception(e)
        sys.exit(f"Error loading data from {file_path}: {e}")

def analyze_data(df):
    """Perform basic data analysis."""
    try:
        numeric_df = df.select_dtypes(include=['number'])  # Select only numeric columns
        analysis = {
            'summary': df.describe(include='all').to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'correlation': numeric_df.corr().to_dict()  # Compute correlation only on numeric columns
        }
        logging.info("Data analysis completed successfully.")
        return analysis
    except Exception as e:
        log_exception(e)
        sys.exit(f"Error analyzing data: {e}")

def visualize_data(df):
    """Generate and save visualizations."""
    sns.set(style="whitegrid")
    numeric_columns = df.select_dtypes(include=['number']).columns
    if not numeric_columns.any():
        logging.warning("No numeric columns found for visualization.")
        return
    for column in numeric_columns:
        try:
            plt.figure()
            sns.histplot(df[column].dropna(), kde=True)
            plt.title(f'Distribution of {column}')
            plt.savefig(f'{column}_distribution.png')
            plt.close()
            logging.info(f"Visualization for {column} saved as {column}_distribution.png")
        except Exception as e:
            log_exception(e)
            logging.warning(f"Error generating visualization for {column}: {e}")

def generate_narrative(analysis):
    """Generate narrative using LLM."""
    headers = {
        'Authorization': f'Bearer {AIPROXY_TOKEN}',
        'Content-Type': 'application/json'
    }
    prompt = f"""
    Provide a comprehensive analysis of the following dataset:
    - Summary statistics: {analysis['summary']}
    - Missing values: {analysis['missing_values']}
    - Correlation matrix: {analysis['correlation']}
    Highlight key trends, anomalies, and actionable insights.
    """
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = httpx.post(API_URL, headers=headers, json=data, timeout=30.0)
        response.raise_for_status()
        narrative = response.json()['choices'][0]['message']['content']
        logging.info("Narrative generated successfully.")
        return narrative
    except httpx.HTTPStatusError as e:
        log_exception(e)
        return "Narrative generation failed due to an HTTP error."
    except httpx.RequestError as e:
        log_exception(e)
        return "Narrative generation failed due to a request error."
    except Exception as e:
        log_exception(e)
        return "Narrative generation failed due to an unexpected error."

def save_readme(analysis, narrative):
    """Save the analysis and narrative to a README.md file."""
    try:
        with open('README.md', 'w') as f:
            f.write("# Analysis Report\n\n")
            f.write("## Summary Statistics\n")
            f.write(json.dumps(analysis['summary'], indent=2))
            f.write("\n\n## Missing Values\n")
            f.write(json.dumps(analysis['missing_values'], indent=2))
            f.write("\n\n## Correlation Matrix\n")
            f.write(json.dumps(analysis['correlation'], indent=2))
            f.write("\n\n## Narrative\n")
            f.write(narrative)
        logging.info("README.md file created successfully.")
    except Exception as e:
        log_exception(e)
        sys.exit(f"Error saving README.md: {e}")

def main(file_path):
    """Main function to process the dataset."""
    if not os.path.exists(file_path):
        sys.exit(f"Dataset file not found: {file_path}")
    df = load_data(file_path)
    analysis = analyze_data(df)
    visualize_data(df)
    narrative = generate_narrative(analysis)
    save_readme(analysis, narrative)

if _name_ == "_main_":
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)
    main(sys.argv[1])
