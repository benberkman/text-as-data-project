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
library(topicmodels)
library(stm)
library(ggpubr)

# read in file
ukraine <- read.csv("../data/ukraine_tweets.csv") %>% mutate(id_str = as.character(id_str))

ukraine$text <- gsub(pattern = "'", "", ukraine$text)  # replace apostrophes

#format as date, remove Independents
ukraine <- ukraine %>% mutate(created_at = as.Date(created_at, format = "%Y-%m-%d")) %>%
  filter(grepl('D|R', Party))  
  
#form df to plot tweets over time
to_plot <- ukraine %>% group_by(Party, created_at) %>%
  filter(created_at >= '2022-01-06 ') %>% count()

#plot tweets over time
ggplot(to_plot, aes(created_at, n)) + 
  geom_line() + 
  facet_grid(Party ~ .) + 
  ylab('Number of Tweets') +
  ggtitle("Tweets per day by Party") +
  theme_bw()

#clean up text
ukraine$text <- gsub(" https.*","",ukraine$text) 
ukraine$text <- gsub("https.*","",ukraine$text) 

# Remove non ASCII characters
ukraine$text <- stringi::stri_trans_general(ukraine$text, "latin-ascii")

# Removes solitary letters
ukraine$text <- gsub(" [A-z] ", " ", ukraine$text)

#create binary
ukraine <- ukraine %>% mutate(Party_binary = case_when(Party == 'D' ~ 1, Party == 'R' ~ 0))

#textProcessor to prepare for stm
temp <- textProcessor(documents=ukraine$text, metadata = ukraine)
meta<-temp$meta
vocab<-temp$vocab
docs<-temp$documents

#prepDocuments to prepare for stm
prepped <- prepDocuments(documents = docs, meta = meta, vocab = vocab, lower.thresh = 30)

docs <- prepped$documents
vocab <- prepped$vocab
meta <- prepped$meta

#build stm model
#commented out because it's already built and saved
#can just load from below
#stm_ukraine <- stm(docs, vocab, K = 6, prevalence = ~meta$Party_binary + s(meta$created_at), content = meta$Party_binary)

#save and reload model (just need to load it)
#saveRDS(stm_ukraine, file = paste0(getwd(), '/ukraine_stm.rds'))
stm_ukraine <- readRDS(paste0(getwd(), '/ukraine_stm.rds'))

#summary plots
plot(stm_ukraine, type = "summary")
plot(stm_ukraine,type="labels")
plot(stm_ukraine,type="hist")

#set vars as numeric to fit regression
meta$created_at <- as.numeric(meta$created_at)
meta$Party_binary <- as.numeric(meta$Party_binary)

#estimates a regression with topics as the dependent variable and metadata as the independent variables
prep <- estimateEffect(~Party_binary + s(created_at) , stm_ukraine, meta = meta)

#how does the content vary by party
#3 is a good one
plot(stm_ukraine, type="perspectives", topics = c(1), main = 'Topic 1 Content by Party (Republican: 0 and Democrat: 1)') 
plot(stm_ukraine, type="perspectives", topics = c(2), main = 'Topic 2 Content by Party (Republican: 0 and Democrat: 1)') 
plot(stm_ukraine, type="perspectives", topics = c(3), main = 'Topic 3 Content by Party (Republican: 0 and Democrat: 1)') 
plot(stm_ukraine, type="perspectives", topics = c(4), main = 'Topic 4 Content by Party (Republican: 0 and Democrat: 1)')
plot(stm_ukraine, type="perspectives", topics = c(5), main = 'Topic 5 Content by Party (Republican: 0 and Democrat: 1)') 
plot(stm_ukraine, type="perspectives", topics = c(6), main = 'Topic 6 Content by Party (Republican: 0 and Democrat: 1)') 


#how does the prevalence change over time
plot(prep, "created_at", stm_ukraine, topics = c(3), 
     method = "continuous", xaxt = "n", xlab = "Date", main = 'Topic 3 Prevalence Over Time')

