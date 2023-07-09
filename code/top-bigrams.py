from sklearn.feature_extraction.text import CountVectorizer

def get_top_n_gram(corpus,ngram_range,n=None):
    vec = CountVectorizer(ngram_range=ngram_range,stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

import matplotlib.pyplot as plt

top_bigrams = get_top_n_gram(filtered_df['tweet_clean'],(2,2),20)
df_bi = pd.DataFrame(top_bigrams, columns = ['Text' , 'count'])
grouped_df = df_bi.groupby('Text').sum()['count'].sort_values(ascending=True)
plt.barh(grouped_df.index, grouped_df.values, color='#ff6961', alpha=0.8)
plt.xlabel('Count')
plt.ylabel('Text')
plt.title('Top 20 Bigrams')
plt.show()
