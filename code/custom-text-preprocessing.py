import nltk
import re
import string

def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def text_preprocessing(text):
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    nopunc = clean_text(text)
    tokens = pd.DataFrame(tokenizer.tokenize(nopunc), columns=['token'])
    tokens_merged = tokens.merge(colloquial_indonesia_dis, left_on='token', right_on='slang', how='left')
    tokens_merged['formal'] = np.where(tokens_merged['formal'].isna(), tokens_merged['token'], tokens_merged['formal'])
    combined_text = ' '.join(tokens_merged['formal'])
    return combined_text
