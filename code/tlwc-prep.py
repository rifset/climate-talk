# merging necessary data
filtered_df['tweet_len'] = filtered_df['tweet_clean'].astype(str).apply(len)
filtered_df['tweet_word_count'] = filtered_df['tweet_clean'].apply(lambda x: len(str(x).split()))
tweet_polusiudara = filtered_df[filtered_df['tweet_clean'].str.contains('polusi udara',case=False)][[
    'tweet_clean', 'tweet_len', 'tweet_word_count']]
tweet_polusiudara['keyword'] = 'polusi udara'
tweet_perubahaniklim = filtered_df[(filtered_df['tweet_clean'].str.contains('perubahan iklim',case=False))|(
    filtered_df['tweet_clean'].str.contains('climate change',case=False))][[
    'tweet_clean', 'tweet_len', 'tweet_word_count']]
tweet_perubahaniklim['keyword'] = 'perubahan iklim'
tweet_pemanasanglobal = filtered_df[(filtered_df['tweet_clean'].str.contains('pemanasan global',case=False))|(
    filtered_df['tweet_clean'].str.contains('global warming',case=False))][[
    'tweet_clean', 'tweet_len', 'tweet_word_count']]
tweet_pemanasanglobal['keyword'] = 'pemanasan global'
tweet_tlwca = tweet_polusiudara.append(tweet_perubahaniklim).append(tweet_pemanasanglobal, ignore_index=True)
