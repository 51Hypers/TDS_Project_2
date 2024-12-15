# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests",
#     "pandas",
#     "matplotlib",
#     "seaborn",
#     "tenacity",
#     "scikit-learn"
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

AIPROXY_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjEwMDI3ODNAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.PB7VDPxXTC6SpA8Ev4K-744mPxQd6Hvpz_qnFGjJ1BM"

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

    r = requests.post(f"{OPENAI_API_BASE}/chat/completions", headers=HEADERS, json=data, timeout=90)
    r.raise_for_status()
    return r.json()

def safe_str(obj):
    return str(obj)[:2000]

def summarize_df(df, max_sample=5):
    desc = []
    for c in df.columns:
        col_info = {}
        col_info["name"] = c
        col_info["dtype"] = str(df[c].dtype)
        col_info["num_null"] = int(df[c].isna().sum())  # convert to int
        col_info["num_unique"] = int(df[c].nunique(dropna=False))  # convert to int
        samples = df[c].dropna().unique()
        if len(samples) > max_sample:
            samples = samples[:max_sample]
        samples = [safe_str(x) for x in samples]
        col_info["sample_values"] = samples
        desc.append(col_info)
    return desc

def basic_stats(df):
    stats = {}
    try:
        stats["shape"] = (int(df.shape[0]), int(df.shape[1]))
        stats["memory_usage_mb"] = float(df.memory_usage(deep=True).sum() / (1024*1024))
        describe_dict = df.describe(include='all', datetime_is_numeric=True).to_dict()
        # Convert all numeric types in describe_dict to Python floats or ints
        # This ensures everything is JSON serializable
        def convert_types(o):
            if isinstance(o, np.integer):
                return int(o)
            elif isinstance(o, np.floating):
                return float(o)
            return o
        stats["describe"] = json.loads(json.dumps(describe_dict, default=convert_types))
        null_counts = df.isna().sum().to_dict()
        # Convert null_counts to int
        null_counts = {k: int(v) for k, v in null_counts.items()}
        stats["null_counts"] = null_counts
    except Exception:
        pass
    return stats

def calc_correlation(df):
    numeric_cols = df.select_dtypes(include=[np.number])
    if numeric_cols.shape[1] > 1:
        return numeric_cols.corr()
    return None

def attempt_clustering(df, n_clusters=3):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    numeric = df.select_dtypes(include=[np.number]).dropna(axis=0)
    if numeric.shape[0] > 10 and numeric.shape[1] > 1:
        X = numeric.copy()
        X = StandardScaler().fit_transform(X)
        km = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
        labels = km.fit_predict(X)
        cluster_centers = km.cluster_centers_
        return {
            "cluster_labels": [int(x) for x in labels],
            "cluster_centers": cluster_centers.tolist(),
            "columns": numeric.columns.tolist()
        }
    return None

def generate_correlation_plot(corr):
    plt.figure(figsize=(6,6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    filename = f"correlation_{uuid.uuid4().hex[:6]}.png"
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.close()
    return filename

def generate_distribution_plot(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        col = numeric_cols[0]
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], kde=True, color='blue')
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        filename = f"distribution_{uuid.uuid4().hex[:6]}.png"
        plt.savefig(filename, dpi=100)
        plt.close()
        return filename
    return None

def generate_missing_values_plot(df):
    null_counts = df.isna().sum()
    if null_counts.sum() > 0:
        plt.figure(figsize=(6,4))
        sns.barplot(x=null_counts.index, y=null_counts.values, color='red')
        plt.title("Missing Values per Column")
        plt.xticks(rotation=90)
        plt.tight_layout()
        filename = f"missing_{uuid.uuid4().hex[:6]}.png"
        plt.savefig(filename, dpi=100)
        plt.close()
        return filename
    return None

if __name__ == "__main__":
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
            pass

    if df is None:
        print("Failed to read the CSV file with available encodings.")
        sys.exit(1)

    column_summary = summarize_df(df)
    stats_summary = basic_stats(df)
    corr = calc_correlation(df)
    cluster_info = attempt_clustering(df)

    # Convert column_summary and stats_summary to JSON strings using default=str to avoid serialization issues
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

    # For the narrative, also do the same for partial dumps
    partial_col_summary_json = json.dumps(column_summary[:3], default=str)
    keys_stats = list(stats_summary.keys())
    narrative_msg = f"""We have analyzed a dataset. Here is what we know:

- Columns summary: {partial_col_summary_json}... ({len(column_summary)} columns total)
- Basic stats (like describe): keys: {keys_stats}
- Missing values: {stats_summary.get('null_counts',{})}
- Correlation matrix: {'present' if corr is not None else 'not available or not meaningful'}
- Clusters: {'found' if cluster_info is not None else 'not performed'}

We also have suggestions from the LLM:
{llm_suggestions}

Now, please write a story as a Markdown `README.md` describing:
1. Briefly what the data might represent (make a guess if unknown)
2. The analysis steps we performed (summary stats, missing values, correlation, clustering)
3. The insights discovered (patterns, notable correlations, any clusters)
4. The implications of these insights (what could be done)

Please integrate references to charts (we will have some PNG charts in the current directory).
For example:
- A correlation heatmap (if generated)
- A distribution plot (if generated)
- A missing values plot (if generated)

Make sure to embed images in Markdown, like `![Alt text](correlation_XXXXXX.png)` etc.
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

    with open("README.md", "w", encoding="utf-8") as f:
        f.write(narrative)
        f.write("\n\n")
        lower_narr = narrative.lower()
        for img in images:
            if os.path.basename(img).lower() not in lower_narr:
                f.write(f"![Chart]({img})\n")
