filtered_df['tweet_clean'] = filtered_df['tweet'].apply(text_preprocessing)
filtered_df.sample(5)
