"""Save descriptive statistics and plots; never fit model features here."""

import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter
from src.data import ROOT
from src.preprocess import clean_text


def run() -> dict:
    df = pd.read_csv(ROOT / "data/resumes.csv")
    out = ROOT / "reports/figures"
    out.mkdir(parents=True, exist_ok=True)
    counts = df.Category.value_counts().sort_index()
    lengths = df.Resume.str.split().str.len()
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.barplot(x=counts.values, y=counts.index, ax=ax, color="#3979a8")
    ax.set(xlabel="Resumes", title="Raw dataset category distribution")
    fig.tight_layout()
    fig.savefig(out / "category_distribution.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(lengths, bins=45, ax=ax)
    ax.set(xlabel="Whitespace-separated words", title="Raw resume length distribution")
    fig.tight_layout()
    fig.savefig(out / "text_lengths.png", dpi=160)
    plt.close(fig)
    terms = {}
    fig, axes = plt.subplots(6, 4, figsize=(20, 24))
    for ax, (category, group) in zip(axes.flat, df.groupby("Category")):
        counter = Counter(" ".join(group.Resume.map(clean_text)).split())
        top = counter.most_common(8)
        terms[category] = dict(top)
        ax.barh([t for t, _ in top][::-1], [n for _, n in top][::-1], color="#3979a8")
        ax.set_title(category)
    fig.suptitle("Frequent cleaned terms per category (raw dataset)")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out / "frequent_terms.png", dpi=140)
    plt.close(fig)
    stats = {
        "category_counts": counts.to_dict(),
        "word_length_statistics": lengths.describe().to_dict(),
        "frequent_terms_per_category": terms,
        "raw_exact_duplicate_texts": int(df.Resume.duplicated().sum()),
    }
    (ROOT / "reports/eda.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    return stats


if __name__ == "__main__":
    print(json.dumps(run()["word_length_statistics"], indent=2))
