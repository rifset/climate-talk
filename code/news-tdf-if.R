library(data.table)
library(tidyverse)
library(tidytext)

dt_katadata <- fread("scraped_katadata.csv")

katadata_word_freq <- dt_katadata %>% 
  unnest_tokens(word, content) %>% 
  count(title, word)
katadata_tf_idf <- katadata_word_freq %>% 
  bind_tf_idf(word, title, n)

katadata_tf_idf %>% 
  slice_max(order_by = tf_idf, n = 10) %>% 
  print()
