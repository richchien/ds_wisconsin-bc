# breast cancer wisconsin (Dx)
Rich Chien  
November 21, 2016  




## Get input data



```r
set.seed(7)

library(dplyr)
```

```
## 
## Attaching package: 'dplyr'
```

```
## The following objects are masked from 'package:stats':
## 
##     filter, lag
```

```
## The following objects are masked from 'package:base':
## 
##     intersect, setdiff, setequal, union
```

```r
library(ggplot2)
library(caret)
```

```
## Loading required package: lattice
```

```r
df <- read.csv("./data.csv")

# delete id and X columns
df <- df %>%
  select(-id, -X)
```


## Setup train and test sets


```r
# take random 70% to train
index <- sample(1:nrow(df),size = 0.7*nrow(df)) 
train <- df[index, ] 
test <- df[-index, ] 
```

## Setup models


```r
# prepare training control
control <- trainControl(method="repeatedcv", number=10, repeats=3)

# train linear model
m.glm <- train(diagnosis~., data=train, method="glm", trControl=control)

# train LVQ model
m.lvq <- train(diagnosis~., data=train, method="lvq", trControl=control)

# train GBM model
m.gbm <- train(diagnosis~., data=train, method="gbm", trControl=control, verbose=FALSE)

# train SVM model
m.svm <- train(diagnosis~., data=train, method="svmRadial", trControl=control)

# train rf model
m.rf <- train(diagnosis~., data=train, method="rf", trControl=control)

# collect resamples
results <- resamples(list(glm=m.glm, lvq=m.lvq, gbm=m.gbm, svm=m.svm, rf=m.rf))

# summarize 
summary(results)
```

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
```

```r
bwplot(results)
```

![](main_files/figure-html/models-1.png)<!-- -->

```r
dotplot(results)
```

![](main_files/figure-html/models-2.png)<!-- -->

```r
# predict test set using model
pred.svm <- predict(m.svm, test)

# confusion matrix
confusionMatrix(pred.svm, test$diagnosis)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction  B  M
##          B 97  1
##          M  1 72
##                                           
##                Accuracy : 0.9883          
##                  95% CI : (0.9584, 0.9986)
##     No Information Rate : 0.5731          
##     P-Value [Acc > NIR] : <2e-16          
##                                           
##                   Kappa : 0.9761          
##  Mcnemar's Test P-Value : 1               
##                                           
##             Sensitivity : 0.9898          
##             Specificity : 0.9863          
##          Pos Pred Value : 0.9898          
##          Neg Pred Value : 0.9863          
##              Prevalence : 0.5731          
##          Detection Rate : 0.5673          
##    Detection Prevalence : 0.5731          
##       Balanced Accuracy : 0.9880          
##                                           
##        'Positive' Class : B               
## 
```
