# slang words
colloquial_indonesia = pd.read_csv("https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv")
colloquial_indonesia_dis = colloquial_indonesia[['slang', 'formal']].drop_duplicates().reset_index(drop=True)

# stop words
stopwords_indonesia = pd.read_csv("indonesian-stopwords.txt", header=None, names=['word'])
