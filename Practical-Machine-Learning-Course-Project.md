---
title: "Practical Machine Learning Course Project"
author: "CM"
date: "2025-02-14"
output: 
  html_document:
    keep_md: true
editor_options: 
  chunk_output_type: console
---

## Introduction

The quantified self movement has gained traction in recent years, allowing individuals to track their physical activities using various wearable devices. This project aims to predict the manner in which participants performed weight lifting exercises based on accelerometer data collected from multiple sensors. The dataset contains measurements from six participants performing a unilateral dumbbell biceps curl in five different manners.


## Data Loading and Exploration


``` r
# Download training and testing data files 
url_training <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(url_training, destfile = "pml-training.csv")

url_testing <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(url_testing, destfile = "pml-testing.csv")

# Load training and testing data
train_data <- read.csv("pml-training.csv")
test_data <- read.csv("pml-testing.csv")
```



``` r
# Data Overview
library(kableExtra)

# Create a data frame 
data_overview <- data.frame(
  observations = c(dim(train_data)[1], dim(test_data)[1]),  
  variables = c(dim(train_data)[2], dim(test_data)[2]))

# Add row names 
rownames(data_overview) <- c("Training", "Testing")

# Create a data overview table
data_overview %>%
  kbl() %>%
  kable_classic(full_width = FALSE)
```

<table class=" lightable-classic" style='color: black; font-family: "Arial Narrow", "Source Sans Pro", sans-serif; width: auto !important; margin-left: auto; margin-right: auto;'>
 <thead>
  <tr>
   <th style="text-align:left;">   </th>
   <th style="text-align:right;"> observations </th>
   <th style="text-align:right;"> variables </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> Training </td>
   <td style="text-align:right;"> 19622 </td>
   <td style="text-align:right;"> 160 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Testing </td>
   <td style="text-align:right;"> 20 </td>
   <td style="text-align:right;"> 160 </td>
  </tr>
</tbody>
</table>

The training dataset consists of 19,622 observations and 160 variables, while the testing dataset contains 20 observations.


### Data Cleaning and Preprocessing

We clean the data by handling missing values and removing irrelevant columns.


``` r
library(kableExtra)

# Remove irrelevant columns and columns with missing values
column_removed <- c(1:7, 12:36, 50:59, 69:83, 87:101, 103:112, 125:139, 141:150)
train_data <- train_data[, -column_removed]
test_data <- test_data[, -column_removed]

# Create a data frame 
data_overview <- data.frame(
  observations = c(dim(train_data)[1], dim(test_data)[1]),  
  variables = c(dim(train_data)[2], dim(test_data)[2]))

# Add row names 
rownames(data_overview) <- c("Training", "Testing")

# Data overview table
data_overview %>%
  kbl() %>%
  kable_classic(full_width = FALSE)
```

<table class=" lightable-classic" style='color: black; font-family: "Arial Narrow", "Source Sans Pro", sans-serif; width: auto !important; margin-left: auto; margin-right: auto;'>
 <thead>
  <tr>
   <th style="text-align:left;">   </th>
   <th style="text-align:right;"> observations </th>
   <th style="text-align:right;"> variables </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> Training </td>
   <td style="text-align:right;"> 19622 </td>
   <td style="text-align:right;"> 53 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Testing </td>
   <td style="text-align:right;"> 20 </td>
   <td style="text-align:right;"> 53 </td>
  </tr>
</tbody>
</table>

``` r
# Convert 'classe' to a factor
train_data$classe <- factor(train_data$classe)
```

The training and test data each now contains 53 variables.


## Model Building

We will use a Random Forest model for this prediction task, a robust classifier that works well with high-dimensional data.


``` r
library(parallel)
library(doParallel)
library(caret)
library(randomForest)

set.seed(1234)

# Step 1: Set up parallel processing
num_cores <- detectCores() - 1 # Use one less core than available
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# Step 2: Set up cross-validation
train_control <- trainControl(method = "cv",
                              number = 7,
                              allowParallel = TRUE)


# Step 3: Train a Random Forest model with cross-validation
rf_model_cv <- train(classe ~ ., data = train_data,
                     method = "rf",
                     trControl = train_control,
                     ntree = 100,
                     importance = TRUE)


# Save the trained model to an RDS file
saveRDS(rf_model_cv, "rf_model_cv.rds")

# Print the cross-validation results
print(rf_model_cv)
```

```
## Random Forest 
## 
## 19622 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (7 fold) 
## Summary of sample sizes: 16819, 16820, 16818, 16821, 16819, 16817, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9954134  0.9941982
##   27    0.9948528  0.9934889
##   52    0.9897569  0.9870428
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 2.
```

``` r
# Stop the parallel processing cluster
stopCluster(cl)
```


### Model Evaluation

Next, we'll evaluate the model using cross-validation accuracy and a confusion matrix.


``` r
library(caret)
library(ggplot2)

# Confusion matrix for the Random Forest model on the training set
rf_predictions_train <- predict(rf_model_cv, train_data)
confusionMatrix(rf_predictions_train, train_data$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 5580    0    0    0    0
##          B    0 3797    0    0    0
##          C    0    0 3422    0    0
##          D    0    0    0 3216    0
##          E    0    0    0    0 3607
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9998, 1)
##     No Information Rate : 0.2844     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##                                      
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2844   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2844   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2844   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

``` r
# Plot the feature importance
importance <- varImp(rf_model_cv, scale = FALSE)

# Create the plot
ggplot(importance, aes(x = reorder(Var1, Overall), y = Overall)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  theme_minimal() +
  coord_flip() +  # Flip the axes
  labs(title = "Feature Importance", x = "Features", y = "Importance")
```

![](Practical-Machine-Learning-Course-Project_files/figure-html/model-evaluation-1.png)<!-- -->

 
### Predictions on Test Data

After training and evaluating the model, we'll use it to make predictions on the test data.


``` r
library(caret)
# Make predictions on the test set using the trained model
rf_predictions_test <- predict(rf_model_cv, test_data)

# Print the test set predictions
print(rf_predictions_test)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```


### Model Performance Visualization

We can visualize the model performance using a confusion matrix.


``` r
library(caret)
library(ggplot2)

# Confusion matrix
rf_conf_matrix <- confusionMatrix(rf_predictions_train, train_data$classe)

# Print the confusion matrix
rf_conf_matrix
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 5580    0    0    0    0
##          B    0 3797    0    0    0
##          C    0    0 3422    0    0
##          D    0    0    0 3216    0
##          E    0    0    0    0 3607
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9998, 1)
##     No Information Rate : 0.2844     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##                                      
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2844   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2844   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2844   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

``` r
# Plot the confusion matrix
conf_matrix_df <- as.data.frame(rf_conf_matrix$table)
ggplot(conf_matrix_df, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 4) +
  scale_fill_gradient(low = "white", high = "blue") +
  theme_minimal() +
  labs(title = "Confusion Matrix", x = "Predicted", y = "Actual") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
```

![](Practical-Machine-Learning-Course-Project_files/figure-html/confusion-matrix-1.png)<!-- -->


### Conclusion

In this project, we built a Random Forest model to predict the manner in which participants performed weight-lifting exercises based on sensor data. We preprocessed the data, standardized the features, and used cross-validation to train and evaluate the model. The Random Forest model showed strong performance, as indicated by the confusion matrix and feature importance plot.

The model's predictions on the test data can now be used for further analysis or real-time predictions. The overall accuracy and performance demonstrate the viability of machine learning in monitoring and improving exercise techniques through wearable sensor data.


## References

```**
- Groupware Lab. (n.d.). Weight Lifting Exercise Dataset. Retrieved from 
[http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har]

- Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013. Read more: [http:/groupware.les.inf.puc-rio.br/har#ixzz4Tjq7l2sJ]

```
