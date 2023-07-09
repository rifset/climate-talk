library(data.table)
library(tidyverse)

climate_trend_google <- fread("climate-topic-google-trend.csv")
climate_trend_google$month = ym(climate_trend_google$month)
dt_temperature <- fread("climate_data.csv")
dt_temperature$date <- dmy(dt_temperature$date)

temperature_id <- dt_temperature %>% 
  group_by(month = floor_date(date, "months")) %>% 
  summarize(avg_temperature = mean(Tavg, na.rm = TRUE))
search_vs_temperature <- temperature_id %>% 
  left_join(climate_trend_google, by = "month")

plot_label <- c("Average Temperature (in °C)", "Climate Topic Google Trend (%)")
names(plot_label) <- c("avg_temperature", "popularity")
search_vs_temperature %>%
  pivot_longer(-month) %>%
  ggplot(aes(x = month, y = value)) +
  geom_line() +
  geom_smooth(method = "lm", se = FALSE) +
  facet_wrap(~name, scales = "free_y", labeller = labeller(name = plot_label)) +
  theme_bw() +
  labs(title = "Temperature vs Climate Topic Trend") +
  theme(axis.title = element_blank())
