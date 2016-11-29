breast cancer wisconsin (Dx)
================
Rich Chien
November 21, 2016

Get input data
--------------

``` r
set.seed(7)
library(dplyr)
library(ggplot2)
library(caret)

df <- read.csv("C:/Users/RC3258/Documents/GitRepo/ds_wisconsin-bc/data.csv")

# delete id and X columns
df <- df %>%
  select(-id, -X)
```

Setup train and test sets
-------------------------

``` r
# take random 70% to train
index <- sample(1:nrow(df),size = 0.7*nrow(df)) 
train <- df[index, ] 
test <- df[-index, ] 
```

Setup models
------------

``` r
# (optional multiple core setup)
# library(doParallel)
# registerDoParallel(cores=4)
# getDoParWorkers()


# prepare training ctrl
ctrl <- trainControl(method="repeatedcv", number=10, repeats=3)


# train linear model
m.glm <- train(diagnosis~., data=train, method="glm", trControl=ctrl)

# train LVQ model
m.lvq <- train(diagnosis~., data=train, method="lvq", trControl=ctrl)

# train GBM model
m.gbm <- train(diagnosis~., data=train, method="gbm", trControl=ctrl, verbose=F)

# train SVM model
m.svm <- train(diagnosis~., data=train, method="svmRadial", trControl=ctrl)

# train rf model
m.rf <- train(diagnosis~., data=train, method="rf", trControl=ctrl)

# collect resamples
results <- resamples(list(glm=m.glm, lvq=m.lvq, gbm=m.gbm, svm=m.svm, rf=m.rf))

# summarize 
summary(results)
```

    ## 
    ## Call:
    ## summary.resamples(object = results)
    ## 
    ## Models: glm, lvq, gbm, svm, rf 
    ## Number of resamples: 30 
    ## 
    ## Accuracy 
    ##      Min. 1st Qu. Median   Mean 3rd Qu. Max. NA's
    ## glm 0.875  0.9250 0.9487 0.9364  0.9500    1    0
    ## lvq 0.850  0.8806 0.9250 0.9213  0.9500    1    0
    ## gbm 0.875  0.9487 0.9500 0.9548  0.9750    1    0
    ## svm 0.925  0.9500 0.9750 0.9682  0.9938    1    0
    ## rf  0.875  0.9309 0.9500 0.9540  0.9750    1    0
    ## 
    ## Kappa 
    ##       Min. 1st Qu. Median   Mean 3rd Qu. Max. NA's
    ## glm 0.7297  0.8360 0.8864 0.8614  0.8915    1    0
    ## lvq 0.6591  0.7349 0.8324 0.8225  0.8864    1    0
    ## gbm 0.7207  0.8847 0.8901 0.8996  0.9459    1    0
    ## svm 0.8266  0.8886 0.9441 0.9293  0.9865    1    0
    ## rf  0.7110  0.8500 0.8901 0.8973  0.9455    1    0

``` r
bwplot(results)
```

![](main_files/figure-markdown_github/models-1.png)

``` r
dotplot(results)
```

![](main_files/figure-markdown_github/models-2.png)

``` r
# examine gbm model
ctrlgbm <- trainControl(method="repeatedcv", number=10, 
                        repeats=3, summaryFunction=twoClassSummary, classProbs=T)
# train GBM model
m2.gbm <- train(diagnosis~., data=train, method="gbm",  metric="ROC", 
               trControl=ctrlgbm, verbose=F)

plot(m2.gbm)
m2.gbm$bestTune


# predict test set using model
pred2.gbm <- predict(m2.gbm, test)

# confusion matrix
confusionMatrix(pred2.gbm, test$diagnosis)

# ROC curve
pred2.gbm.prob <- predict(m2.gbm, test, type="prob")

auc <- roc(ifelse(test[,1]=="M",1,0), pred2.gbm.prob[[2]])
print(auc$auc)

# accuracy calc
res <- table(pred.svm, test$diagnosis)
accuracy <- round(100 * sum(diag(res))/sum(res), 2)

# save model
# saveRDS(m.svm, "./m.svm.rds")
 
# load model
# m.svm <- readRDS("./m.svm.rds")
```
