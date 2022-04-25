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


# read in file
ukraine <- read.csv("../data/ukraine_tweets.csv")
ukraine$text <- gsub(pattern = "'", "", ukraine$text)  # replace apostrophes

#format as date, remove Independents
ukraine <- ukraine %>% mutate(created_at = as.Date(created_at, format = "%Y-%m-%d")) %>%
  filter(grepl('D|R', Party))

#clean up text
ukraine$full_text <- gsub(" https.*","",ukraine$full_text) 
ukraine$full_text <- gsub("https.*","",ukraine$full_text) 

# Remove non ASCII characters
ukraine$full_text <- stringi::stri_trans_general(ukraine$full_text, "latin-ascii")

# Removes solitary letters
ukraine$full_text <- gsub(" [A-z] ", " ", ukraine$full_text)

#clean up text
ukraine$full_text <- as.character(ukraine$full_text)

#read in sentiment dictionaries
neg = scan("../data/negative-words.txt", what = "", sep = '\n')
pos = scan("../data/positive-words.txt", what = "", sep = '\n')
dict1 <- dictionary(list(positive = pos, negative = neg))

#tokenize and create dfm
toks <- tokens(ukraine$full_text)
dfm_ukraine <- dfm(tokens_lookup(toks, dict1, valuetype = "glob", verbose = TRUE))

#form sentiment scores
df_ukraine <- as.data.frame(dfm_ukraine) %>% mutate(sent_score = positive - negative)

#bring in additional columns
df_ukraine <- df_ukraine %>% mutate(created_at = ukraine$created_at, party = ukraine$Party)

#create classification and filter to tighter date range
df_ukraine <- df_ukraine %>% mutate(classification = case_when(sent_score > 0 ~ 1, sent_score <= 0 ~ 0)) %>%
  filter(created_at >= '2022-01-06 ')

#plot sentiment scores by party
ggplot(df_ukraine, aes(x=sent_score, fill = party)) + geom_histogram(bins = 20) +
  ggtitle("Sentiment Scores by Party") + theme_bw() +
  xlab('Sentiment Score') +
  ylab('Count')

#observe how sentiment changes by party, by date
x <- df_ukraine %>% select(classification, created_at, party) %>% 
  group_by(party, created_at = lubridate::floor_date(created_at, "week")) %>% 
  arrange(created_at) %>%
  summarize(classification = mean(classification, na.rm = FALSE)) 

#observe how sentiment changes by party, by date
ggplot(data=x, aes(x=created_at, y=classification, group=party, color = party)) +
  geom_line(linetype = "dashed")+
  geom_point() +
  ylab('Weekly mean sentiment classification (>= .5 is positive)') +
  xlab('Date (2022)') +
  ggtitle('Sentiment Classification by Party and Week') + 
  theme_bw() + 
  geom_vline(xintercept=as.numeric(x$created_at[21]), linetype=4) +
  geom_text(aes(x=x$created_at[21], label="First Sanctions by USA", y=.58), colour="black", angle=0)
