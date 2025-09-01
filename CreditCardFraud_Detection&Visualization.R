install.packages("caret")
install.packages("MLmetrics")
install.packages("pROC")
install.packages("randomForest")
install.packages("lubridate")

library(dplyr)
library(ggplot2)
library(lubridate)
library(caret)
library(randomForest)
library(glmnet)
library(class)
library(corrplot)
library(MLmetrics)
library(pROC)

# Load the data
df <- read.csv("C:\\Users\\Dell\\Downloads\\fraud_detection\\creditcard.csv")

# Display the first few rows
head(df)

# Display column names
colnames(df)

# Display shape of the dataframe
dim(df)

# Display information about the dataframe
str(df)

# Display summary statistics
summary(df)

# Count missing values
colSums(is.na(df))

# Count duplicated rows
sum(duplicated(df))

# Convert date column and create new time-related columns
df$trans_date_trans_time <- mdy_hms(df$trans_date_trans_time)
df$hour <- hour(df$trans_date_trans_time)
df$day <- weekdays(df$trans_date_trans_time)
df$month <- month(df$trans_date_trans_time)
head(df)

# Remove "fraud_" from merchant names
df$merchant <- gsub("fraud_", "", df$merchant)
head(df[["merchant"]])

# Convert dob and calculate age
df$dob <- mdy(df$dob)
df$age_cust <- year(df$trans_date_trans_time) - year(df$dob)
df <- df %>% select(-dob)
head(df)

# Calculate distances
df <- df %>%
  mutate(lat_dist = abs(round(merch_lat - lat, 2)),
         long_dist = abs(round(merch_long - long, 2)))

# Drop unnecessary columns
df <- df %>% select(-lat, -long, -merch_lat, -merch_long, -trans_date_trans_time)
summary(df)

# Get dummy variables
df <- df %>% mutate(gender = as.factor(gender), is_fraud = as.factor(is_fraud))
df_encoded <- model.matrix(~ gender + is_fraud - 1, data = df)
df_encoded <- as.data.frame(df_encoded)
df <- df %>% select(-gender, -is_fraud) %>% bind_cols(df_encoded)

# Rename columns
colnames(df) <- gsub("genderM", "is_male", colnames(df))
colnames(df) <- gsub("is_fraud1", "is_fraud", colnames(df))
head(df)

str(df)

# Count fraudulent transactions
fraud_counts <- table(df$is_fraud)
print(fraud_counts)

# Plot fraudulent transactions by gender
ggplot(df, aes(x = factor(is_male), fill = factor(is_fraud))) +
  geom_bar(position = "dodge") +
  labs(title = "Fraudulent Transactions by Gender", x = "Gender", y = "Count of Fraudulent Transactions") +
  scale_x_discrete(labels = c("Female", "Male")) +
  theme_minimal()

# Scatter plot of transaction amount vs fraud
ggplot(df, aes(x = factor(is_fraud), y = amt)) +
  geom_point(alpha = 0.7) +
  labs(title = "Transaction Amount vs Fraud", x = "Is Fraud", y = "Transaction Amount") +
  theme_minimal()

# Top 10 merchants with most fraudulent transactions
fraud_count_merch <- df %>% group_by(merchant) %>% summarise(fraud_count = sum(is_fraud))
top10_merch <- fraud_count_merch %>% top_n(10, fraud_count)

ggplot(top10_merch, aes(x = reorder(merchant, -fraud_count), y = fraud_count)) +
  geom_bar(stat = "identity") +
  labs(title = "Top 10 Merchants with Most Fraudulent Transactions", x = "Merchant", y = "Number of Fraudulent Transactions") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Top 10 categories with most fraudulent transactions
fraud_count_cat <- df %>% group_by(category) %>% summarise(fraud_count = sum(is_fraud))
top10_cat <- fraud_count_cat %>% top_n(10, fraud_count)

ggplot(top10_cat, aes(x = reorder(category, -fraud_count), y = fraud_count)) +
  geom_bar(stat = "identity") +
  labs(title = "Top 10 Categories with Most Fraudulent Transactions", x = "Category", y = "Number of Fraudulent Transactions") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Top 10 cities with most fraudulent transactions
fraud_count_city <- df %>% group_by(city) %>% summarise(fraud_count = sum(is_fraud))
top10_city <- fraud_count_city %>% top_n(10, fraud_count)

ggplot(top10_city, aes(x = reorder(city, -fraud_count), y = fraud_count)) +
  geom_bar(stat = "identity") +
  labs(title = "Top 10 Cities with Most Fraudulent Transactions", x = "City", y = "Number of Fraudulent Transactions") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Top 10 jobs with most fraudulent transactions
fraud_count_job <- df %>% group_by(job) %>% summarise(fraud_count = sum(is_fraud))
top10_job <- fraud_count_job %>% top_n(10, fraud_count)

ggplot(top10_job, aes(x = reorder(job, -fraud_count), y = fraud_count)) +
  geom_bar(stat = "identity") +
  labs(title = "Top Jobs with Most Fraudulent Transactions", x = "Job", y = "Number of Fraudulent Transactions") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Drop unnecessary columns
df <- df %>% select(-c(street, state, first, last, trans_num, unix_time, city_pop, city, merchant, category, job, day, age_cust, is_male))
str(df)

# Correlation heatmap
correlation_matrix <- cor(df %>% select_if(is.numeric))
corrplot(correlation_matrix, method = "color", tl.cex = 0.7, tl.col = "black")


outlier_thresholds <- function(dataframe, col_name) {
  Q1 <- quantile(dataframe[[col_name]], 0.05, na.rm = TRUE)
  Q3 <- quantile(dataframe[[col_name]], 0.95, na.rm = TRUE)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  return(c(lower_bound, upper_bound))
}


check_outlier <- function(dataframe, col_name) {
  limits <- outlier_thresholds(dataframe, col_name)
  if (any(dataframe[[col_name]] > limits[2], na.rm = TRUE) | 
      any(dataframe[[col_name]] < limits[1], na.rm = TRUE)) {
    return(TRUE)
  } else {
    return(FALSE)
  }
}


check_all_columns_outliers <- function(dataframe) {
  results <- sapply(names(dataframe), function(col_name) {
    check_outlier(dataframe, col_name)
  })
  return(results)
}

check_all_columns_outliers(df)

# Boxplot for each numeric variable
numeric_cols <- df %>% select_if(is.numeric) %>% colnames()
for (col in numeric_cols) 
  {
  ggplot(df, aes(y = col)) +
    geom_boxplot() +
    labs(title = paste("Box Plot of", col), y = col) +
    theme_minimal()
}

replace_with_thresholds <- function(dataframe, variable, q1 = 0.05, q3 = 0.95) {
  limits <- outlier_thresholds(dataframe, variable, q1, q3)
  dataframe[[variable]][dataframe[[variable]] < limits[1]] <- limits[1]
  dataframe[[variable]][dataframe[[variable]] > limits[2]] <- limits[2]
}

# Process numeric columns
outlier_thresholds <- function(dataframe, variable, q1, q3) {
  # Calculate thresholds
  iqr <- q3 - q1
  lower_threshold <- q1 - 1.5 * iqr
  upper_threshold <- q3 + 1.5 * iqr
  
  return(list(lower = lower_threshold, upper = upper_threshold))



# Calculate Q1 and Q3
q1 <- quantile(dataframe$value, 0.05)
q3 <- quantile(dataframe$value, 0.95)

# Call the function
thresholds <- outlier_thresholds(dataframe, "value", q1, q3)
print(thresholds)
}
# Split the data into training and testing sets
set.seed(17)
train_index <- createDataPartition(df$is_fraud, p = 0.7, list = FALSE)
train <- df[train_index, ]
test <- df[-train_index, ]

y_train <- train$is_fraud
X_train <- train %>% select(-is_fraud)
y_test <- test$is_fraud
X_test <- test %>% select(-is_fraud)

# Checing variation in 'hour'
summary(X_train$hour)

# Preprocess excluding 'hour' if it has no variation
library(caret)
preProc <- preProcess(X_train[, -which(names(X_train) == "hour")], method = "range")
X_train_processed <- predict(preProc, X_train[, -which(names(X_train) == "hour")])


# Scale the features
scaler <- preProcess(X_train_processed, method = "range")
X_train_processed <- predict(scaler, X_train_processed)
X_test <- predict(scaler, X_test)

X_train_imputed <- apply(X_train, 2, function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))
install.packages("mice")
library(randomForest)
library(mice)

# Example data frame with missing values
# Assuming X_train_imputed is your feature matrix and y_train is your response variable

# Impute missing values using the mice package
imputed_data <- mice(X_train_imputed, m = 1, maxit = 50, method = 'pmm', seed = 500)
X_train_imputed_complete <- complete(imputed_data, 1)

# Train the random forest model
model <- randomForest(X_train_imputed_complete, y_train, random_state = 46)
print(model)

# Train and evaluate Random Forest model
rf_model <- randomForest(X_train_imputed, y_train, random_state = 46)
rf_pred <- predict(rf_model, X_test)
confusionMatrix(rf_pred, y_test)

# Train and evaluate Logistic Regression model
lr_model <- train(X_train, y_train, method = "glmnet", family = "binomial", trControl = trainControl(method = "cv", number = 10))
lr_pred <- predict(lr_model, X_test)
confusionMatrix(lr_pred, y_test)

# Train and evaluate KNN model
k <- floor(sqrt(nrow(df)))
knn_pred <- knn(train = X_train, test = X_test, cl = y_train, k = k)
confusionMatrix(knn_pred, y_test)

# Compare models
model_results <- data.frame(
  Model = c("KNN", "Random Forest", "Logistic Regression"),
  Accuracy = c(accuracy(knn_pred, y_test), accuracy(rf_pred, y_test), accuracy(lr_pred, y_test)),
  Precision = c(precision(knn_pred, y_test), precision(rf_pred, y_test), precision(lr_pred, y_test)),
  Recall = c(recall(knn_pred, y_test), recall(rf_pred, y_test), recall(lr_pred, y_test)),
  F1 = c(F1_Score(knn_pred, y_test), F1_Score(rf_pred, y_test), F1_Score(lr_pred, y_test)),
  ROC_AUC = c(AUC(y_test, as.numeric(knn_pred)), AUC(y_test, as.numeric(rf_pred)), AUC(y_test, as.numeric(lr_pred)))
)

# Melt the results for plotting
model_results_melted <- model_results %>%
  gather(key = "Metric", value = "Value", -Model)

# Plot the results
ggplot(model_results_melted, aes(x = Metric, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Model Performance Metrics", x = "Metric", y = "Value") +
  theme_minimal()

