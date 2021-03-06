#Clearing the environment
rm(list = ls())

# Setting the current working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#import libraries
library(dplyr)
library(quanteda)
library(quanteda.textstats)
library(quanteda.corpora)
library(quanteda.textmodels)
library(ggplot2)
library(tidyr)
library(stringr)
library(tidytext)
library(broom)
library(widyr)
library(purrr)
library(scales)
library(ggpubr)
set.seed(2017)

group.colors <- c('D' = "#0015BC", 'R' = "#FF0000")

# read in file
ukraine <- read.csv("../data/ukraine_tweets.csv") %>%
  mutate(id_str = as.character(id_str))
ukraine$text <- gsub(pattern = "'", "", ukraine$text)  # replace apostrophes

#format as date, remove Independents
ukraine <- ukraine %>% mutate(created_at = as.Date(created_at, format = "%Y-%m-%d")) %>%
  filter(grepl('D|R', Party))

#clean up text
ukraine$text <- gsub(" https.*","",ukraine$text) 
ukraine$text <- gsub("https.*","",ukraine$text) 

# Remove non ASCII characters
ukraine$text <- stringi::stri_trans_general(ukraine$text, "latin-ascii")

# Removes solitary letters
ukraine$text <- gsub(" [A-z] ", " ", ukraine$text)

#clean up text
ukraine$text <- as.character(ukraine$text)

#read in sentiment dictionaries
neg = scan("../data/negative-words.txt", what = "", sep = '\n')
pos = scan("../data/positive-words.txt", what = "", sep = '\n')
dict1 <- dictionary(list(positive = pos, negative = neg))

#tokenize and create dfm
toks <- tokens(ukraine$text)
dfm_ukraine <- dfm(tokens_lookup(toks, dict1, valuetype = "glob", verbose = TRUE))

#form sentiment scores
df_ukraine <- as.data.frame(dfm_ukraine) %>% mutate(sent_score = positive - negative)

#bring in additional columns
df_ukraine <- df_ukraine %>% mutate(created_at = ukraine$created_at, party = ukraine$Party)

#create classification and filter to tighter date range
df_ukraine <- df_ukraine %>% mutate(classification = case_when(sent_score > 0 ~ 1,
                                                               sent_score <= 0 ~ 0)) %>%
  filter(created_at >= '2022-01-06 ')

#plot sentiment scores by party
sent_score_paty <- ggplot(df_ukraine, aes(x=sent_score, fill = party)) + 
  geom_histogram(bins = 18) +
  ggtitle("Sentiment Scores by Party") + theme_bw() +
  xlab('Sentiment Score') +
  ylab('Count') + 
  scale_fill_manual(values=group.colors)

sent_score_paty

#observe how sentiment changes by party, by date
by_date <- df_ukraine %>% 
  select(classification, created_at, party) %>% 
  group_by(party, created_at = lubridate::floor_date(created_at, "week")) %>% 
  arrange(created_at) %>%
  summarize(classification = mean(classification, na.rm = FALSE)) 

#observe how sentiment changes by party, by date
by_date_plot <- ggplot(data=by_date, aes(x=created_at, 
                                         y=classification, group=party, color = party)) +
  geom_line(linetype = "dashed")+
  geom_point() +
  scale_colour_manual(values=group.colors) +
  ylab('Weekly sentiment classification') +
  xlab('Date (2022)') +
  ggtitle('Sentiment Classification by Party and Week') + 
  theme_bw() + 
  geom_vline(xintercept=as.numeric(by_date$created_at[21]), linetype=4) +
  geom_text(aes(x=by_date$created_at[21], label="First Sanctions by USA", y=.58), colour="black", angle=0)

by_date_plot

ggarrange(sent_score_paty, by_date_plot,
          ncol = 1, nrow = 2)

##### Additional Sentiment Analysis
#Code adapted from: https://www.tidytextmining.com/usenet.html

#format text
cleaned_text <- ukraine %>%
  filter(str_detect(text, "^[^>]+[A-Za-z\\d]") | text == "",
         !str_detect(text, "writes(:|\\.\\.\\.)$"),
         !str_detect(text, "^In article <"))  %>%
  filter(created_at >= '2022-01-06 ')

#extract words
words <- cleaned_text %>%
  unnest_tokens(word, text) %>%
  filter(str_detect(word, "[a-z']$"),
         !word %in% stop_words$word)

#group by party
words_by_party <- words %>%
  count(Party, word, sort = TRUE) %>%
  ungroup()

#group by state
words_by_state <- words %>%
  count(State, word, sort = TRUE) %>%
  ungroup()

#ti idf by party
tf_idf_party <- words_by_party %>%
  bind_tf_idf(word, Party, n) %>%
  arrange(desc(tf_idf))

#plot top tf idf words by party
words_party_plot <- tf_idf_party %>%
  group_by(Party) %>%
  slice_max(tf_idf, n = 12) %>%
  ungroup() %>%
  mutate(word = reorder(word, tf_idf)) %>%
  ggplot(aes(tf_idf, word, fill = Party)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ Party, scales = "free") +
  labs(x = "tf-idf weight", y = NULL) +
  ggtitle('Which Words are Most Democratic or Republican?') + 
  theme_bw() + 
  scale_fill_manual(values=group.colors)

words_party_plot

#get sentiment by party
party_sentiments <- words_by_party %>%
  inner_join(get_sentiments("afinn"), by = "word") %>%
  group_by(Party) %>%
  summarize(value = sum(value * n) / sum(n))

#plot sentiments by party
party_sentiments %>%
  mutate(Party = reorder(Party, value)) %>%
  ggplot(aes(value, Party, fill = Party)) +
  geom_col(show.legend = FALSE) +
  labs(x = "Average sentiment value", y = NULL) + 
  ggtitle("Sentiment by Party") +
  theme_bw() +
  scale_fill_manual(values=group.colors)

#which words are most positive and negative
contributions <- words %>%
  inner_join(get_sentiments("afinn"), by = "word") %>%
  group_by(word) %>%
  summarize(occurences = n(),
            contribution = sum(value))

#which words are most positive and negative by party
top_sentiment_words <- words_by_party %>%
  inner_join(get_sentiments("afinn"), by = "word") %>%
  mutate(contribution = value * n / sum(n))

#plot which words are most positive and negative by party
sent_party_plot <- top_sentiment_words %>%
  group_by(Party) %>%
  slice_max(abs(contribution), n = 12) %>%
  ungroup() %>%
  mutate(Party = reorder(Party, contribution),
         word = reorder_within(word, contribution, Party)) %>%
  ggplot(aes(contribution, word, fill = contribution > 0)) +
  geom_col(show.legend = FALSE) +
  scale_y_reordered() +
  facet_wrap(~ Party, scales = "free") +
  labs(x = "Sentiment", y = NULL) +
  ggtitle('Most Positive and Negative Words by Party') +
  theme_bw()

sent_party_plot

ggarrange(words_party_plot, sent_party_plot,
          ncol = 2, nrow = 1)

#what are the most positive and negative tweets
sentiment_messages <- words %>%
  inner_join(get_sentiments("afinn"), by = "word") %>%
  group_by(Party, id_str) %>%
  summarize(sentiment = mean(value),
            words = n()) %>%
  ungroup() %>%
  filter(words >= 5)

#most positive
sentiment_messages %>%
  arrange(desc(sentiment))

#function to print tweets
print_message <- function(group, message_id) {
  result <- cleaned_text %>%
    filter(Party == group, id_str == message_id, text != "")
  
  cat(result$text, sep = "\n")
}

#print two of them
print_message("D", "1510056227128107008")
print_message("R", "1504097394551664640")

#most negative
sentiment_messages %>%
  arrange(sentiment)

#print two of them
print_message("D", "1497006412668760064")
print_message("R", "1511717810401124352")


#### Comparing Tweets By Party
#Code adapted from: https://www.tidytextmining.com/twitter.html

#chars to remove
remove_reg <- "&amp;|&lt;|&gt;"

#clean tweets
tidy_tweets <- ukraine %>% 
  filter(!str_detect(text, "^RT")) %>%
  mutate(text = str_remove_all(text, remove_reg)) %>%
  unnest_tokens(word, text, token = "tweets") %>%
  filter(!word %in% stop_words$word,
         !word %in% str_remove_all(stop_words$word, "'"),
         str_detect(word, "[a-z]")) %>%
  filter(created_at >= '2022-01-06 ')

#get word freqs by party
frequency <- tidy_tweets %>% 
  count(Party, word, sort = TRUE) %>% 
  left_join(tidy_tweets %>% 
              count(Party, name = "total")) %>%
  mutate(freq = n/total)

#re-arrange for plotting
frequency <- frequency %>% 
  select(Party, word, freq) %>% 
  pivot_wider(names_from = Party, values_from = freq) %>%
  arrange(R, D)

#which words skew most D or R?
ggplot(frequency, aes(R, D)) +
  geom_jitter(alpha = 0.1, size = 2.5, width = 0.25, height = 0.25) +
  geom_text(aes(label = word), check_overlap = TRUE, vjust = 1.5) +
  scale_x_log10(labels = percent_format()) +
  scale_y_log10(labels = percent_format()) +
  geom_abline(color = "red") +
  theme_bw() +
  xlab('Republican') + 
  ylab('Democrat') +
  ggtitle('Which words skew most Dem. or Repub.?')

#log odds ration
word_ratios <- tidy_tweets %>%
  filter(!str_detect(word, "^@")) %>%
  count(word, Party) %>%
  group_by(word) %>%
  filter(sum(n) >= 10) %>%
  ungroup() %>%
  pivot_wider(names_from = Party, values_from = n, values_fill = 0) %>%
  mutate_if(is.numeric, list(~(. + 1) / (sum(.) + 1))) %>%
  mutate(logratio = log(D / R)) %>%
  arrange(desc(logratio))

#plot log odds
word_ratios %>%
  group_by(logratio < 0) %>%
  slice_max(abs(logratio), n = 15) %>% 
  ungroup() %>%
  mutate(word = reorder(word, logratio)) %>%
  ggplot(aes(word, logratio, fill = logratio < 0)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  ylab("log odds ratio (D/R)") +
  scale_fill_discrete(name = "", labels = c("D", "R")) +
  ggtitle("Which words are most likely to be tweeted from R’s (right) or D's (left)?")

#calculate words over time by party
words_by_time <- tidy_tweets %>%
  filter(!str_detect(word, "^@")) %>%
  mutate(time_floor = lubridate::floor_date(created_at, unit = "1 month")) %>%
  count(time_floor, Party, word) %>%
  group_by(Party, time_floor) %>%
  mutate(time_total = sum(n)) %>%
  group_by(Party, word) %>%
  mutate(word_total = sum(n)) %>%
  ungroup() %>%
  rename(count = n) %>%
  filter(word_total > 30)

#make a data frame with a list column that contains little miniature data frames for each word
nested_data <- words_by_time %>%
  nest(data = c(-word, -Party)) 

#apply modeling to each little data frames inside big data frame
nested_models <- nested_data %>%
  mutate(models = map(data, ~ glm(cbind(count, time_total) ~ time_floor, ., 
                                  family = "binomial")))

#pull out the slopes for each of these models and find the important ones
slopes <- nested_models %>%
  mutate(models = map(models, tidy)) %>%
  unnest(cols = c(models)) %>%
  filter(term == "time_floor") %>%
  mutate(adjusted.p.value = p.adjust(p.value))

#get top D slopes
top_slopes_D <- slopes %>% 
  arrange(adjusted.p.value) %>% 
  filter(Party == 'D') %>%
  head(5)

#get top R slopes
top_slopes_R <- slopes %>% 
  arrange(adjusted.p.value) %>% 
  filter(Party == 'R') %>%
  head(5)

#plot top 5 D words over time
words_by_time %>%
  inner_join(top_slopes_D, by = c("word", "Party")) %>%
  filter(Party == "D") %>%
  ggplot(aes(time_floor, count/time_total, color = word)) +
  geom_line(size = 1.3) +
  labs(x = NULL, y = "Word frequency") +
  ggtitle("Top Democratic Words Over Time")

#plot top 5 R words over time
words_by_time %>%
  inner_join(top_slopes_R, by = c("word", "Party")) %>%
  filter(Party == "R") %>%
  ggplot(aes(time_floor, count/time_total, color = word)) +
  geom_line(size = 1.3) +
  labs(x = NULL, y = "Word frequency") +
  ggtitle("Top Republican Words Over Time")

#get top 5 slopes from each party
top_slopes <- slopes %>% 
  arrange(adjusted.p.value) %>% 
  group_by_(~ Party) %>%
  slice(1:5)

#plot words over time by party
words_by_time %>%
  inner_join(top_slopes, by = c("word", "Party")) %>%
  ggplot(aes(time_floor, count/time_total, color = word)) +
  geom_line(size = 1.3) +
  facet_grid(Party ~ ., scales = "free") +
  labs(x = NULL, y = "Word frequency") +
  ggtitle("Top Words Over Time by Party")
