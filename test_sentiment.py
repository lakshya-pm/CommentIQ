"""
Fixed test for sentiment analysis module.

Previous version assigned analyze_comments() result directly to `df`,
but analyze_comments() returns a (DataFrame, metrics_dict) tuple.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.sentiment import analyze_comments, sentiment_summary

COMMENTS = [
    "This video is amazing!",
    "Worst tutorial ever.",
    "It is okay.",
    "Very informative and useful.",
    "I wasted my time.",
]

# analyze_comments returns a TUPLE: (df, ml_metrics)
df, ml_metrics = analyze_comments(COMMENTS)

print("=== DataFrame ===")
print(df[["original_text", "sentiment", "compound"]].to_string())

print("\n=== ML Metrics ===")
for model, metrics in ml_metrics.items():
    if isinstance(metrics, dict) and "accuracy" in metrics:
        print(f"  {model}: acc={metrics['accuracy']}%, F1={metrics['f1_score']}%")

print("\n=== Summary ===")
summary = sentiment_summary(df)
for k, v in summary.items():
    print(f"  {k}: {v}")
