cor_val <- search_vs_temperature %>% 
  summarize(
    metrics = "correlation", 
    value = cor(avg_temperature, popularity)
  )
print(cor_val)
