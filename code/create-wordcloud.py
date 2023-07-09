from wordcloud import WordCloud
import matplotlib.pyplot as plt

text = ' '.join(filtered_df['tweet_clean'].astype(str).tolist())
wordcloud = WordCloud(width=1200, height=800, 
                      background_color='white', 
                      stopwords=stopwords_indonesia['word'],
                      font_path=r'segoeuib.ttf').generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
