# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests",
#     "pandas",
#     "matplotlib",
#     "seaborn",
#     "tenacity",
#     "scikit-learn",
#     "opencv-python",
#     "Pillow"
# ]
# ///

import os
import sys
import json
import uuid
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tenacity import retry, stop_after_attempt, wait_fixed
from PIL import Image
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load sensitive information from environment variables for security
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    raise ValueError("AIPROXY_TOKEN environment variable not set.")

OPENAI_API_BASE = "https://aiproxy.sanand.workers.dev/openai/v1"
MODEL_NAME = "gpt-4o-mini"

HEADERS = {
    "Authorization": f"Bearer {AIPROXY_TOKEN}"
}

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def openai_chat(messages, functions=None, function_call=None):
    data = {
        "model": MODEL_NAME,
        "messages": messages,
    }
    if functions:
        data["functions"] = functions
    if function_call:
        data["function_call"] = function_call

    response = requests.post(f"{OPENAI_API_BASE}/chat/completions", headers=HEADERS, json=data, timeout=90)
    response.raise_for_status()
    return response.json()

def safe_str(obj):
    return str(obj)[:2000]

def summarize_df(df: pd.DataFrame, max_sample: int = 5) -> list:
    """
    Summarize the DataFrame columns with dtype, null counts, unique counts, and sample values.
    """
    desc = []
    for c in df.columns:
        col_info = {
            "name": c,
            "dtype": str(df[c].dtype),
            "num_null": int(df[c].isna().sum()),
            "num_unique": int(df[c].nunique(dropna=False)),
            "sample_values": [safe_str(x) for x in df[c].dropna().unique()[:max_sample]]
        }
        desc.append(col_info)
    return desc

def basic_stats(df: pd.DataFrame) -> dict:
    """
    Compute basic statistics of the DataFrame.
    """
    stats = {}
    try:
        stats["shape"] = (int(df.shape[0]), int(df.shape[1]))
        stats["memory_usage_mb"] = float(df.memory_usage(deep=True).sum() / (1024*1024))
        # Removed 'datetime_is_numeric=True'
        describe_dict = df.describe(include='all').to_dict()
        
        def convert_types(o):
            if isinstance(o, np.integer):
                return int(o)
            elif isinstance(o, np.floating):
                return float(o)
            return o
        
        stats["describe"] = json.loads(json.dumps(describe_dict, default=convert_types))
        null_counts = df.isna().sum().to_dict()
        stats["null_counts"] = {k: int(v) for k, v in null_counts.items()}
    except Exception as e:
        print(f"Error computing basic stats: {e}")
    return stats

def calc_correlation(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Calculate the correlation matrix for numeric columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number])
    if numeric_cols.shape[1] > 1:
        return numeric_cols.corr()
    return None

def attempt_clustering(df: pd.DataFrame, n_clusters: int = 3) -> dict | None:
    """
    Attempt K-Means clustering on numeric data.
    """
    numeric = df.select_dtypes(include=[np.number]).dropna(axis=0)
    if numeric.shape[0] > 10 and numeric.shape[1] > 1:
        X = StandardScaler().fit_transform(numeric)
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        cluster_centers = km.cluster_centers_
        return {
            "cluster_labels": labels.tolist(),
            "cluster_centers": cluster_centers.tolist(),
            "columns": numeric.columns.tolist()
        }
    return None

def generate_correlation_plot(corr: pd.DataFrame) -> str:
    """
    Generate and save a correlation heatmap.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    filename = f"correlation_{uuid.uuid4().hex[:6]}.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    return filename

def generate_distribution_plot(df: pd.DataFrame) -> str | None:
    """
    Generate and save a distribution plot for the first numeric column.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        col = numeric_cols[0]
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col].dropna(), kde=True, color='blue')
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        filename = f"distribution_{uuid.uuid4().hex[:6]}.png"
        plt.savefig(filename, dpi=150)
        plt.close()
        return filename
    return None

def generate_missing_values_plot(df: pd.DataFrame) -> str | None:
    """
    Generate and save a bar plot of missing values per column.
    """
    null_counts = df.isna().sum()
    if null_counts.sum() > 0:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=null_counts.index, y=null_counts.values, hue=null_counts.index, palette='Reds', legend=False)
        plt.title("Missing Values per Column")
        plt.xlabel("Columns")
        plt.ylabel("Number of Missing Values")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        filename = f"missing_{uuid.uuid4().hex[:6]}.png"
        plt.savefig(filename, dpi=150)
        plt.close()
        return filename
    return None

def analyze_visualizations(images: list) -> dict:
    """
    Analyze generated visualization images using computer vision techniques.
    """
    analysis = {}
    for img_path in images:
        try:
            img = cv2.imread(img_path)
            if img is None:
                analysis[img_path] = "Unable to read image."
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_count = np.sum(edges > 0)
            analysis[img_path] = {
                "shape": img.shape,
                "edge_count": int(edge_count)
            }
        except Exception as e:
            analysis[img_path] = f"Error analyzing image: {e}"
    return analysis

def agentic_decision(suggestions: str) -> dict:
    """
    Make autonomous decisions based on LLM suggestions.
    """
    decisions = {}
    if "additional analysis" in suggestions.lower():
        decisions["additional_analysis"] = True
    else:
        decisions["additional_analysis"] = False
    # Further decision logic can be added here
    return decisions

def main():
    if len(sys.argv) < 2:
        print("Usage: uv run autolysis.py dataset.csv")
        sys.exit(1)
    input_file = sys.argv[1]

    # Try multiple encodings to avoid UnicodeDecodeError
    df = None
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(input_file, encoding=enc, on_bad_lines='skip')
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error reading CSV with encoding {enc}: {e}")
            continue

    if df is None:
        print("Failed to read the CSV file with available encodings.")
        sys.exit(1)

    column_summary = summarize_df(df)
    stats_summary = basic_stats(df)
    corr = calc_correlation(df)
    cluster_info = attempt_clustering(df)

    # Convert summaries to JSON strings
    column_summary_json = json.dumps(column_summary, indent=2, default=str)
    stats_summary_json = json.dumps(stats_summary, indent=2, default=str)

    user_msg = f"""We have a dataset with shape {df.shape}. Columns:
{column_summary_json}

Basic stats:
{stats_summary_json}

We tried correlation and clustering where possible.

Suggest any further generic analysis steps or summarize insights. Keep the suggestions short.
"""
    messages = [
        {"role": "system", "content": "You are a data analyst assistant."},
        {"role": "user", "content": user_msg}
    ]
    resp = openai_chat(messages)
    llm_suggestions = resp["choices"][0]["message"]["content"].strip()

    # Agentic decision based on suggestions
    decisions = agentic_decision(llm_suggestions)

    # For the narrative, also do the same for partial dumps
    partial_col_summary_json = json.dumps(column_summary[:3], default=str)
    keys_stats = list(stats_summary.keys())
    narrative_msg = f"""We have analyzed a dataset. Here is what we know:

- Columns summary: {partial_col_summary_json}... ({len(column_summary)} columns total)
- Basic stats (like describe): keys: {keys_stats}
- Missing values: {stats_summary.get('null_counts', {})}
- Correlation matrix: {'present' if corr is not None else 'not available or not meaningful'}
- Clusters: {'found' if cluster_info is not None else 'not performed'}

We also have suggestions from the LLM:
{llm_suggestions}

Now, please write a story as a Markdown README.md describing:
1. Briefly what the data might represent (make a guess if unknown)
2. The analysis steps we performed (summary stats, missing values, correlation, clustering)
3. The insights discovered (patterns, notable correlations, any clusters)
4. The implications of these insights (what could be done)

Please integrate references to charts (we will have some PNG charts in the current directory).
For example:
- A correlation heatmap (if generated)
- A distribution plot (if generated)
- A missing values plot (if generated)

Make sure to embed images in Markdown, like ![Alt text](correlation_XXXXXX.png) etc.
Use headings, lists, and emphasis.
"""

    messages = [
        {"role": "system", "content": "You are a data storytelling expert."},
        {"role": "user", "content": narrative_msg}
    ]
    resp = openai_chat(messages)
    narrative = resp["choices"][0]["message"]["content"]

    images = []
    if corr is not None and corr.shape[0] > 1:
        cfile = generate_correlation_plot(corr)
        images.append(cfile)
    dfile = generate_distribution_plot(df)
    if dfile:
        images.append(dfile)
    mfile = generate_missing_values_plot(df)
    if mfile:
        images.append(mfile)

    # Analyze visualizations using computer vision
    image_analysis = analyze_visualizations(images)
    print("Visualization Analysis:", json.dumps(image_analysis, indent=2))

    # Optionally, make agentic decisions based on LLM suggestions
    if decisions.get("additional_analysis"):
        print("LLM suggested additional analysis. Implementing additional steps...")
        # Implement additional analysis steps here
        # For example, generating more plots or performing different statistical tests
        # This is a placeholder for further agentic actions

    with open("README.md", "w", encoding="utf-8") as f:
        f.write(narrative)
        f.write("\n\n")
        lower_narr = narrative.lower()
        for img in images:
            if os.path.basename(img).lower() not in lower_narr:
                f.write(f"![Chart]({img})\n")

if __name__ == "__main__":
    main()
