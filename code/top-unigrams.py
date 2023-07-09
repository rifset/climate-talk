from sklearn.feature_extraction.text import CountVectorizer

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

import matplotlib.pyplot as plt

top_unigrams = get_top_n_words(filtered_df['tweet_clean'], 20)
df_uni = pd.DataFrame(top_unigrams, columns = ['Text' , 'count'])
grouped_df = df_uni.groupby('Text').sum()['count'].sort_values(ascending=True)
plt.barh(grouped_df.index, grouped_df.values, color='steelblue', alpha=0.8)
plt.xlabel('Count')
plt.ylabel('Word')
plt.title('Top 20 Unigrams')
plt.show()
