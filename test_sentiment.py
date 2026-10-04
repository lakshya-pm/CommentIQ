from modules.sentiment import analyze_comments
from modules.sentiment import sentiment_summary

comments = [

    "This video is amazing!",

    "Worst tutorial ever.",

    "It is okay.",

    "Very informative and useful.",

    "I wasted my time."

]

df = analyze_comments(comments)

print(df)

print()

summary = sentiment_summary(df)

print(summary)
