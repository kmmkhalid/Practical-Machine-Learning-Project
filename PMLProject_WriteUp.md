---
title: "WriteUp for the Practical Machine Learning Project"
author: "Karla Khalid"
date: "November 23, 2018"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)

library(caret)
library(kernlab)
library(randomForest)
library(ggplot2)
library(corrplot)
```

## Overview

In this project, data collected from six (6) participants doing dumbbell lifts correctly and incorrectly is used to predict the manner in which they did the exercise. The training and testing data come from the study described here: <http://groupware.les.inf.puc-rio.br/har>. 

In particular, the participants did the dumbbell biceps curl in five different ways, as follows:  exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E). Class A corresponds to the correct execution of the exercise, while the other 4 classes correspond to common mistakes [1]. In processing the data and using prediction algorithms, the objective is to predict "how (well)" the activity was performed. 

This report describes the development of the model, including the use of cross validation techniques and the determination of the expected out-of-sample error. The prediction model is then used to predict 20 different test cases. 

## Data Preparation and Processing

1 The training & testing data were downloaded from [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv) and [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv) into my working directory and then loaded into R, as follows:

```{r load_data, echo=TRUE}

training = read.csv("~/R/pml-training.csv", sep=",", header=TRUE, na.strings = c("NA","",'#DIV/0!'))
testing = read.csv("~/R/pml-testing.csv", sep=",", header=TRUE, na.strings = c("NA","",'#DIV/0!'))
dim(training)
dim(testing)

```


2 A cursory look at the data show that there is a need to reduce variables to remove columns with missing (NA) values, along with variables that are unnecessary for predicting the quality of execution (**X** indicating row numbers, **user_name**, **raw_time_stamp**, **cvtd_timestamp**, **new_window** which was a marker for summary variable rows, **num_window**). I also checked for correlation among the remaining variables and removed those that are highly correlated.
```{r clean data, echo=TRUE}
#Remove columns with missing values
training <- training[, (colSums(is.na(training)) == 0)]
testing <- testing[, (colSums(is.na(testing)) == 0)]

#Remove first seven columns that are not useful for prediction
training <- training[, -(1:7)]
testing <- testing[, -(1:7)]

#Check and trim variables/features that are highly correlated
cor_matrix <- cor(training[,-53]) #excluding the "classe"" variable
cor_var <- findCorrelation(cor_matrix, cutoff = 0.9)
training <- training[, -cor_var]
testing <- testing[, -cor_var]

# Verify that the column names (excluding "classe"") are identical in the training and testing sets
column_training <- colnames(training)
column_testing <- colnames(testing)
all.equal(column_training[1:length(column_training)-1], column_testing[1:length(column_training)-1])

dim(training)
dim(testing)

```

Note that for building the model, only 46 out of the original 160 variables were used.

3 Split the entire training dataset so that 70% is used for training the model and 30% is used to evaluate model accuracy.

```{r split data, echo=TRUE}
#Partition the available training data into 70% training and 30% validation
train_index <- createDataPartition(training$classe, p=0.7, list = FALSE)
train_data <- training[train_index,]
validation_data <- training[-train_index,]
dim(train_data); dim(validation_data)

```

##Model Training and Evaluation

I made two considerations in the development of the model: the model type (classification trees vs random forests) and data splitting techniques (bootstrapping, with cross-validation, with repeated cross-validation), the latter implemented using the **trainControl** function.

###Classification Trees
```{r model training CT, echo=TRUE}
#Build model using the default bootstrapping on the entire training set
set.seed(1235)
model_Trees_boot <- train(classe ~., data = train_data, method = "rpart")
#Predict on the validation set
pred_boot <- predict(model_Trees_boot, newdata = validation_data)
cf_Trees_boot <- confusionMatrix(pred_boot, validation_data$classe)

#Build model using cross-validation 
tr_params <- trainControl(method = "cv", number = 5)
model_Trees_cv <- train(classe ~., data = train_data, trControl = tr_params, method = "rpart")
#Predict on the Validation Set
pred2 <- predict(model_Trees_cv, newdata = validation_data)
cf_Trees_cv <- confusionMatrix(pred2, validation_data$classe)

#Build model using repeated cross-validation 
tr_params_rcv <- trainControl(method = "repeatedcv", number = 5, repeats = 2)
model_Trees_rcv <- train(classe ~., data = train_data, trControl = tr_params_rcv, method = "rpart")
#Predict on the Validation Set
pred3 <- predict(model_Trees_rcv, newdata = validation_data)
cf_Trees_rcv <- confusionMatrix(pred3, validation_data$classe)
```

```{r Classification Trees results, echo=FALSE}
print("Model Type: Classification Trees")
m <- matrix(c(model_Trees_boot$results[1,2],cf_Trees_boot$overall[1],model_Trees_cv$results[1,2], cf_Trees_cv$overall[1],model_Trees_rcv$results[1,2], cf_Trees_rcv$overall[1]), ncol = 2, byrow = TRUE)
colnames(m) <- c("Training Accuracy", "Validation Accuracy")
rownames(m) <- c("with Bootstrapping", "with Cross-Validation (n = 5)", "with Repeated Cross-Validation (n = 5)")
m <- as.table(m)
m
```
It can be seen that the use of Classification Trees on this particular prediction problem yielded a low accuracy of `r paste0(round(cf_Trees_rcv$overall[1] * 100, 2), "%")` or an *out-of-sample error* equal to `r paste0(round(100 - cf_Trees_rcv$overall[1] * 100, 2), "%")` on the Validation Set regardless of resampling/cross-validation techniques used.  

###Random Forests 

In an effort to achieve better prediction accuracy, another model using Random Forests was built using the train data and evaluated using the validation data as follows:

```{r model training RF, echo=TRUE}
#Build RF model using bootstrapping 
set.seed(1235)
model_RFboot <- train(classe ~., data = train_data, method = "rf")
print(model_RFboot)
#Predict using the RF model on the validation set
pred_RFboot <- predict(model_RFboot, newdata = validation_data)
cm_RF <- confusionMatrix(pred_RFboot, validation_data$classe)
print(cm_RF)
```

```{r accuracy, echo=FALSE}
accuracy_RF <- cm_RF$overall["Accuracy"]
```

From the confusion matrix, it can be seen that this model yielded a prediction accuracy of `r paste0(round(accuracy_RF * 100, 2), "%")` or consequently, an *out-of-sample error* equal to `r paste0(round(100 - accuracy_RF * 100, 2), "%")`. Since these results are way better than those obtained using the previous model using Classification Trees, the RF model will be used on the testing set.

##Testing

Finally, the RF model was used to make predictions on the provided testing set as follows:

```{r testing, echo=TRUE}
pred_testing <- predict(model_RFboot, newdata = testing)
print(pred_testing)

```


## Plots

Figure 1. Correlation Matrix

```{r correlation matrix, echo=TRUE}
corrplot(cor_matrix, method = "color")
```

Note: I couldn't get the **Rattle* package to work due to problems with the Gtk2 package so I decided to do away with the visualizations for the models.


## Reference:

[1] Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.
