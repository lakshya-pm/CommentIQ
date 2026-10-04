from modules.sentiment import analyze_comments
from modules.sentiment import sentiment_summary

from modules.visualization import *

comments = [

    "Amazing tutorial",

    "Excellent explanation",

    "Worst video",

    "Very useful",

    "Bad explanation",

    "Nice work",

    "Fantastic",

    "Terrible"

]

df = analyze_comments(comments)

summary = sentiment_summary(df)

create_pie_chart(summary)

create_wordcloud(df)

create_positive_chart(df)

create_negative_chart(df)

print("Charts Generated Successfully!")
