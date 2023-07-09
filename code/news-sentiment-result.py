katadata_sentiment_final = katadata_sentiment.groupby(['title', 'sentiment']).size().to_frame('count').reset_index()
katadata_sentiment_final = katadata_sentiment_final.groupby('title').apply(
    lambda x: x.nlargest(1, 'count')).reset_index(drop=True)
katadata_sentiment_final_size = katadata_sentiment_final['sentiment'].value_counts().reset_index()
katadata_sentiment_final_size.rename(columns={'sentiment':'count', 'index':'sentiment'}, inplace=True)

import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
colors = ['steelblue', 'lightgrey', 'salmon']
total_count = katadata_sentiment_final_size['count'].sum()
plt.bar(katadata_sentiment_final_size['sentiment'], katadata_sentiment_final_size['count'], color=colors)
for i, count in enumerate(katadata_sentiment_final_size['count']):
    label = f"{count} ({count/total_count*100:.1f}%)"
    plt.text(i, count, label, ha='center', va='bottom')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Sentiment Distribution')
plt.show()
