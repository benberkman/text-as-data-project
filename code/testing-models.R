#Clearing the environment
rm(list = ls())

# Setting the current working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# import libraries
library(dplyr)
library(quanteda)
library(quanteda.corpora)
library(quanteda.textmodels)
library(data.table)
library(readtext)
library(caret)
library(glmnet)

# read in file
ukraine <- read.csv("../data/ukraine_tweets.csv")
ukraine$text <- gsub(pattern = "'", "", ukraine$text)  # replace apostrophes

#format as date, remove Independents
ukraine <- ukraine %>% mutate(created_at = as.Date(created_at, format = "%Y-%m-%d")) %>%
  filter(grepl('D|R', Party))

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

### Word Scores

# randomly sample a test speech
set.seed(1984L)
ids <- 1:nrow(ukraine)
# exclude one tweet for test set
ids_test <- sample(ids, 1, replace = FALSE) 
ids_train <- ids[-ids_test]
train_set <- ukraine[ids_train,]
test_set <- ukraine[ids_test,]

# create DFMs
train_dfm <- dfm(train_set$text, remove_punct = TRUE, remove = stopwords("english"))
test_dfm <- dfm(test_set$text, remove_punct = TRUE, remove = stopwords("english"))

ws_sm <- textmodel_wordscores(train_dfm, 
                              y = 2*train_set$viral - 1,
                              smooth = 1
)

# Look at strongest features
strong_features_dec <- sort(ws_sm$wordscores, decreasing = TRUE)  
strong_features_dec[1:15]

strong_features_inc <- sort(ws_sm$wordscores, decreasing = FALSE)  
strong_features_inc[1:10]

# plot ws scores
plot(ws_sm$wordscores, xlab="Count (of words)", ylab="Wordscore", main="Do word scores vary between parties?")


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

confusionMatrix(svm_linear_pred, test_y, mode = "prec_recall", positive = '1')
# Precision : 0.17143        
# Recall : 0.14458        
# F1 : 0.15686    
# Baseline Accuracy:  0.8991982 
# SVM-Linear Accuracy: 0.8430699 
# SVM-Radial Accuracy: 0.8991982

# Loop through training sizes
values <- list(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
for (i in values) {
  # partition data
  ids_train <- createDataPartition(1:nrow(ukraine_dfm), p = i, list = FALSE, times = 1)
  train_x <- ukraine_dfm[ids_train, ] %>% as.data.frame() # train set data
  train_y <- ukraine$viral[ids_train] %>% as.factor() # train set labels
  
  # val set
  val_x <- ukraine_dfm[-ids_train, ] %>% as.data.frame() # val set data
  val_y <- ukraine$viral[-ids_train] %>% as.factor() # val set labels
  
  # define training options
  trctrl <- trainControl(method = "cv", number = 5)
  
  # enter training mode
  svm_mod_linear <- train(x = train_x,
                          y = train_y,
                          method = 'svmLinear',
                          trControl = trctrl)
  
  # out of sample preds
  svm_linear_pred <- predict(svm_mod_linear, newdata = val_x)
  
  # conf mat
  svm_linear_cmat <- confusionMatrix(svm_linear_pred, val_y)
  metrics <- confusionMatrix(svm_linear_pred, val_y, mode = "prec_recall", positive = '1')
  
  # accuracy
  acc_linear <- sum(diag(svm_linear_cmat$table))/sum(svm_linear_cmat$table)
  print(paste("Training size:", i, "| Metrics:", metrics, "| Accuracy", acc_linear))
}

### Training size has little to no effect

### Logistic Regression

set.seed(1984L)
prop_train <- 0.8 # open to change this
# Save the indexes
ids <- 1:nrow(ukraine)

ids_train <- sample(ids, ceiling(prop_train*length(ids)), replace = FALSE)
ids_test <- ids[-ids_train]
train_set <- ukraine[ids_train,]
test_set <- ukraine[ids_test,]

# get dfm for each set
train_dfm <- dfm(train_set$text, stem = TRUE, remove_punct = TRUE, remove = stopwords("english"))
test_dfm <- dfm(test_set$text, stem = TRUE, remove_punct = TRUE, remove = stopwords("english"))

# train model 
lr <- textmodel_lr(train_dfm, train_set$viral)

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

# Accuracy: 0.899654 
# Recall: 0.0755814 
# Precision: 0.1733333 
# F1-score: 0.1052632

lr <- cv.glmnet(
  x = train_dfm, 
  y = train_set$viral,
  family = "binomial",
  nfolds = 5,
  alpha = 1,
  type.measure = "auc",
  maxit = 10000,
  parallel = TRUE
)

# visualize regularization
plot(lr, xvar="labmda", label=TRUE)


### Random Forest 

# create partition
ids_train_rf <- createDataPartition(1:nrow(ukraine_dfm), p = 0.8, list = FALSE, times = 1)
train_x <- ukraine_dfm[ids_train_rf, ] %>% as.data.frame() # train set data
train_y <- ukraine$viral[ids_train_rf] %>% as.factor()  # train set labels

# test set
test_x <- ukraine_dfm[-ids_train_rf, ] %>% as.data.frame()
test_y <- ukraine$viral[-ids_train_rf] %>% as.factor()

# train rf
rf <- randomForest(x = train_x, y = train_y, importance = TRUE)
token_imp <- round(importance(rf, 2), 2)
head(rownames(token_imp)[order(-token_imp)], 10) # 10 most important features

# get 10 most important features
varImpPlot(rf, n.var = 10, main = "Variable Importance")

# predict and conf mat
predict_test <- predict(rf, newdata = test_x)
cmat_rf <- table(test_y, predict_test)
nb_acc_sm <- sum(diag(cmat_rf))/sum(cmat_rf) # accuracy = (TP + TN) / (TP + FP + TN + FN)
nb_recall_sm <- cmat_rf[2,2]/sum(cmat_rf[2,]) # recall = TP / (TP + FN)
nb_precision_sm <- cmat_rf[2,2]/sum(cmat_rf[,2]) # precision = TP / (TP + FP)
nb_f1_sm <- 2*(nb_recall_sm*nb_precision_sm)/(nb_recall_sm + nb_precision_sm)

# report metrics
cmat_rf
cat(
  "Accuracy:",  nb_acc_sm, "\n",
  "Recall:",  nb_recall_sm, "\n",
  "Precision:",  nb_precision_sm, "\n",
  "F1-score:", nb_f1_sm
)
