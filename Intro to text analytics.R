.libPaths()
rm(list=ls())
ls()
###set working directory
setwd("C:/Users/cliff/Desktop/R")
###install packages to use
#install.packages(c("ggplot2","e1071","caret","quanteda","irlba","randomForest"),
#                 lib="C:/Users/cliff/Documents/R/win-library/3.4")
##load and explore data
#library("lattice", "foreach", "rpart", "iterators", "quanteda", "snow")
#install.packages("caret")
spam.raw <- read.csv("spam.csv",stringsAsFactors = FALSE)
##Clean up the dataframe and view your work
spam.raw <- spam.raw[,1:2]
##rename variables
names(spam.raw) <- c("Label","Text")
##check for missing values
length(which(!complete.cases(spam.raw)))
##convert the class label ito a factor
spam.raw$Label <- as.factor(spam.raw$Label)
##exolore data through class "Label" distribution
prop.table(table(spam.raw$Label))
##now, lets get the length of messages and get a feel of their distribution
spam.raw$TextLength <- nchar(spam.raw$Text)
##summary(spam.raw$TextLength)
#well, lets visualize with ggplot2, adding the segmentation for ham/spam.
library(ggplot2)
ggplot(spam.raw, aes(x=TextLength, fill=Label))+
  theme_bw()+
  geom_histogram(binwidth=5)+
  labs(y="Text Count", x = "Length of Text",
       title = "Distribution of Text Lengths with Class Labels")
#lets split data into train and test data ucing caret
library(caret)
library(tibble)

##warnings()
##use caret to split into 70/30% stratified split
##set random seed for reproducibility
set.seed(32984)
indexes <- createDataPartition(spam.raw$Label, times=1, p = 0.7, list = FALSE)
##use "indexes" to split data with proportions of class label is maintained
train <- spam.raw[indexes,]
test <- spam.raw[-indexes,]
##verify propotions
prop.table(table(train$Label))
prop.table(table(test$Label))
train$Text[21]
##the quanteda package has a lot of functions that quickly and easily workwith text data
#install.packages("quanteda")
library(quanteda)

##Tokenize SMS text messages and remove numbers, symbols, punctuations and hyphens
train$Text<-tolower(train$Text)
train.tokens<- tokens(train$Text, what = "word",
                         remove_punct=TRUE, remove_symbols=TRUE,
                         remove_hyphens=TRUE, remove_numbers=TRUE)


#?tokens
#??selectFeatures
train.tokens[[357]]
#Make the tokens lower case
#train.tokens<-tolower(train.tokens)
#?tolower
#train.tokens[[357]]
#use quantedas inbuilt stopword listfor english
#NOTE:always inspect stopwords list for applicability to your problem/domain
train.tokens <- tokens_select(train.tokens, stopwords("english"), selection= "remove" )
train.tokens[[357]]
train.tokens<-tokens_wordstem(train.tokens, language = "english")
train.tokens[[357]]
#create our first bag of woprds model
train.tokens.dfm<-dfm(train.tokens)
#create a dfm matrix
train.tokens.matrix<-as.matrix(train.tokens.dfm)
#View(train.tokens.matrix[1:20,1:100])
#dim(train.tokens.matrix)
#investigating the effect of stemming
colnames(train.tokens.matrix[,1:50])
#we use cross validation CV as its powerfull and a best practice but requires more time
#setup a the feature dataframe with labels
train.tokens.df <- cbind(Label =train$Label,as.data.frame(train.tokens.dfm))
#often, tokenization requires some additional preprocessing
names(train.tokens.df)[c(146, 148,235,238)]
#cleanup column names
names(train.tokens.df)<-make.names(names(train.tokens.df))
names(train.tokens.df)[c(146, 148,235,238)]
#use caret to create stratified folds for 10 folds cross validation repeated 3 times
#(ie 30 random stratified samples)
set.seed(48743)
cv.folds<-createMultiFolds(train$Label, k=10, times = 3)
cv.cntrl <- trainControl(method = "repeatedcv",number = 10,
                         repeats = 3, index = cv.folds)
#doSNOW package was changed to "snow" package
#install.packages("snow")
#??snow
library(doSNOW)
#cliff correction surgery on duplicatesinvestigate and eliminate duplicate token name
names(train.tokens.df)[1]
#View(train.tokens.df)
#display the records in the combined data set
#which(names(train.tokens.df) %in% "s.i.m")
which(names(train.tokens.df) %in% "X.")
#names(train.tokens.df)[310]<-"chngd."
#names(train.tokens.df)[5642]<-"simi"
#names(train.tokens.df)[275]<-"XY"
#CHECK CHECK FOR ERRORS
#Time the code execution
#start.time <- Sys.time()
#create a cluster to work on 3 logical cores
#install rpart
#install.packages("rpart")
#cl <- makeCluster(2, type = "SOCK")
#registerDoSNOW(cl)
#rpart.cv.1<-train(Label ~ ., data = train.tokens.df, method="rpart", trControl = cv.cntrl,
#                  tuneLength = 7)
#stopCluster(cl)
#timeTaken<-start.time-Sys.time()
#timeTaken



#lets create out relative term frequency function (TF)
term.frequency<-function(row) {
  row/sum(row)
}

#lets create our inverse document frequency(IDF)
inverse.doc.freq<-function(col) {
  corpus.size<-length(col)
  doc.count<-length(which(col>0))
  log10(corpus.size/doc.count)
}

#our function for calculating TF-IDF
tf.idf<-function(tf, idf){
  tf*idf
}

#first lets normalize all documents using TF
train.tokens.df<-apply(train.tokens.matrix, 1, term.frequency)
dim(train.tokens.df)
#View(train.tokens.df[1:20, 1:100])

#second, calculate the IDF vector that we will use both for training
#data and testing data
train.tokens.idf<-apply(train.tokens.matrix,2,inverse.doc.freq)
str(train.tokens.idf)

#lastly, calculate TF-IDF for training corpus
train.tokens.tfidf<-apply(train.tokens.df, 2, tf.idf, idf=train.tokens.idf)
dim(train.tokens.tfidf)
#View(train.tokens.tfidf[1:25, 1:25])

#transpose the matrix back to original representation
train.tokens.tfidf<-t(train.tokens.tfidf)
dim(train.tokens.tfidf)
#View(train.tokens.tfidf)

#lets check for incomplete cases
incomplete.cases<-which(!complete.cases(train.tokens.tfidf))
train$Text[incomplete.cases]

#fix the incomplete cases
train.tokens.tfidf[incomplete.cases,]<-rep(0.0,ncol(train.tokens.tfidf))
dim(train.tokens.tfidf) 
sum(which(!complete.cases(train.tokens.tfidf)))

#make a clean data frame using the same process as before
train.tokens.tfidf.df<-cbind(Label = train$Label, data.frame(train.tokens.tfidf))
names(train.tokens.tfidf.df)<-make.names(names(train.tokens.tfidf.df))
names(train.tokens.tfidf.df)[1:50]

#create a cluster to work on 10 logical cores
#start.time <- Sys.time()
#cl <- makeCluster(3,type="SOCK")
#registerDoSNOW(cl)
#as our data is non trivial in size at this point, use a single direction tree algorithm as
#our first model. we will graduate to using more powerfull algorithms later when we perform
#feature extraction to shrink the data size
#rpart.cv.2<-train(Label~ ., data = train.tokens.tfidf.df, method="rpart", trControl = cv.cntrl,
#                  tuneLength = 7)

#stopCluster(cl)
#timeTaken<-start.time-Sys.time()
#timeTaken

#rpart.cv.2

#N-grams allows us to augment our document-term-frequency matrices with word ordering
#This often leads to increased accuracy for machine learning models trained with more
#than just unigrams. Lets add bigrams to our data and the TF-IDF transforms the expanded
#feature matrix to see if accuracy improves

#Add bigrams to our feature matrix
train.tokens <- tokens_ngrams(train.tokens, n = 1:2)
train.tokens[[357]]

#transform to dfm and then a matrix
train.tokens.dfm<-dfm(train.tokens, tolower = FALSE)
train.tokens.matrix<-as.matrix(train.tokens.dfm)
train.tokens.dfm
##train.tokens.matrix[1:15,1:15]

?dfm_weight
#Normalize all documents using TF''''dfm_weight'
train.tokens.tf.df <- dfm_weight(train.tokens.dfm, scheme = "logave")
train.tokens.df<-apply(train.tokens.matrix, 1, term.frequency)
dim(train.tokens.df)
#View(train.tokens.df[1:20, 1:100])

#second, calculate the IDF vector that we will use both for training
#data and testing data
train.tokens.idf<-apply(train.tokens.matrix,2,inverse.doc.freq)
str(train.tokens.idf)

#lastly, calculate TF-IDF for training corpus
train.tokens.tfidf<-apply(train.tokens.df, 2, tf.idf, idf=train.tokens.idf)
dim(train.tokens.tfidf)
#View(train.tokens.tfidf[1:25, 1:25])

#transpose the matrix back to original representation
train.tokens.tfidf<-t(train.tokens.tfidf)
dim(train.tokens.tfidf)
#View(train.tokens.tfidf)

#lets check for incomplete cases
incomplete.cases<-which(!complete.cases(train.tokens.tfidf))
#train$Text[incomplete.cases]
train.tokens.tfidf[incomplete.cases,]<-rep(0.0,ncol(train.tokens.tfidf))





