import pandas as pd
from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForSequenceClassification

pretrained= "mdhugol/indonesia-bert-sentiment-classification"
model = AutoModelForSequenceClassification.from_pretrained(pretrained)
tokenizer = AutoTokenizer.from_pretrained(pretrained)
sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
sentiment_table = pd.DataFrame({
    'sentiment_label':['LABEL_0', 'LABEL_1', 'LABEL_2'],
    'sentiment':['POSITIVE', 'NEUTRAL', 'NEGATIVE']
})

katadata = pd.read_csv("katadata_tf_idf_top20.csv")
katadata['sentiment_label'] = katadata['word'].apply(get_sentiment)
katadata_sentiment = katadata.merge(sentiment_table, on='sentiment_label')
print(katadata_sentiment)
