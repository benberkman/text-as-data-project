#Clearing the environment
rm(list = ls())

# Setting the current working directory
setwd("/Users/davidtrakhtenberg/Downloads/text-as-data-project-main/data/")

# import libraries
library(dplyr)
library(quanteda)
library(quanteda.corpora)
library(quanteda.textmodels)
library(data.table)
library(readtext)
library(caret)
library(randomForest)

# read in file
ukraine <- read.csv("ukraine_tweets.csv")
ukraine$text <- gsub(pattern = "'", "", ukraine$text)  # replace apostrophes
prop.table(table(ukraine$viral)) 
# distribution of classes = 9:1 --> not viral: viral
# should we pull more twitter data? or we can just use the appropriate eval metrics
table(ukraine$viral)

### Naive Bayes
# split sample into training & test sets
set.seed(1984L)
prop_train <- 0.9 # open to change this
# Save the indexes
ids <- 1:nrow(ukraine)

ids_train <- sample(ids, ceiling(prop_train*length(ids)), replace = FALSE)
ids_test <- ids[-ids_train]
train_set <- ukraine[ids_train,]
test_set <- ukraine[ids_test,]

# get dfm for each set
train_dfm <- dfm(train_set$text, stem = TRUE, remove_punct = TRUE, remove = stopwords("english"))
test_dfm <- dfm(test_set$text, stem = TRUE, remove_punct = TRUE, remove = stopwords("english"))

test_dfm_df <- convert(test_dfm, to = "data.frame")
train_dfm_df <- convert(train_dfm, to = "data.frame")


# match test set dfm to train set dfm features
test_dfm <- dfm_match(test_dfm, features = featnames(train_dfm))

# smoothing doesn't help
nb <- textmodel_nb(train_dfm, train_set$viral, smooth = 0, prior = "uniform")

# evaluate on test set
predicted_class <- predict(nb, newdata = test_dfm, force=TRUE)

# get confusion matrix
cmat <- table(test_set$viral, predicted_class)
acc <- sum(diag(cmat))/sum(cmat) # accuracy = (TP + TN) / (TP + FP + TN + FN)
recall <- cmat[2,2]/sum(cmat[2,]) # recall = TP / (TP + FN)
precision <- cmat[2,2]/sum(cmat[,2]) # precision = TP / (TP + FP)
f1 <- 2*(recall*precision)/(recall + precision)

# print
cat(
  # "Baseline Accuracy: ", baseline_acc, "\n",
  "Accuracy:",  acc, "\n",
  "Recall:",  recall, "\n",
  "Precision:",  precision, "\n",
  "F1-score:", f1
)

# Accuracy: 0.8009153 
# Recall: 0.2272727 
# Precision: 0.1587302 
# F1-score: 0.1869159


### SVM using caret

# create document feature matrix
ukraine_dfm <- dfm(ukraine$text, stem = TRUE, remove_punct = TRUE, remove = stopwords("english")) %>% convert("matrix")

# partition using caret
ids_train <- createDataPartition(1:nrow(ukraine_dfm), p = 0.8, list = FALSE, times = 1)
train_x <- ukraine_dfm[ids_train, ] %>% as.data.frame() # train set data
train_y <- ukraine$viral[ids_train] %>% as.factor()  # train set labels
test_x <- ukraine_dfm[-ids_train, ]  %>% as.data.frame() # test set data
test_y <- ukraine$viral[-ids_train] %>% as.factor() # test set labels

# baseline
baseline_acc <- max(prop.table(table(test_y)))

# define training options
trctrl <- trainControl(method = "none") 
#none: only fits one model to the entire training set

# svm - linear
svm_mod_linear <- train(x = train_x,
                        y = train_y,
                        method = "svmLinear",
                        trControl = trctrl)

svm_linear_pred <- predict(svm_mod_linear, newdata = test_x)
svm_linear_cmat <- confusionMatrix(svm_linear_pred, test_y)

# svm - radial
# takes longer to run
svm_mod_radial <- train(x = train_x,
                        y = train_y,
                        method = "svmRadial",
                        trControl = trctrl)

svm_radial_pred <- predict(svm_mod_radial, newdata = test_x)
svm_radial_cmat <- confusionMatrix(svm_radial_pred, test_y)

cat(
  "Baseline Accuracy: ", baseline_acc, "\n",
  "SVM-Linear Accuracy:",  svm_linear_cmat$overall[["Accuracy"]], "\n",
  "SVM-Radial Accuracy:",  svm_radial_cmat$overall[["Accuracy"]]
)

# Baseline Accuracy:  0.8991982 
# SVM-Linear Accuracy: 0.8430699 
# SVM-Radial Accuracy: 0.8991982

## TO DO
# Acc for SVM isn't very useful for our case; get recall/precision instead
# Try RF, LogReg, etc
# AUC could be good metric to use as it's invariant to base rate
# Topic modelling - STM
# unsupervised exploration on left vs right 