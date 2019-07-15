install.packages(c("data.table","dplyr"))

library(data.table)
library(dplyr)
library(corrplot)
library(caret)
library(glmnet)
library(randomForest)
library(doMC)
library(ROCR)

################################################################
######## Read in tables
################################################################
customer_table<- fread("/Users/anne/Desktop/DS204/2/Project/customer_table.csv")
#customer_table_temp <- read.csv("/Usersanne/Desktop/DS204/2/Project/customer_table.csv")
order_table<- fread("/Users/anne/Desktop/DS204/2/Project/order_table.csv")
product_table <- fread("/Users/anne/Desktop/DS204/2/Project/product_table.csv")
category_table <- fread("/Users/anne/Desktop/DS204/2/Project/category_table.csv")

################################################################
######## Find the targeting population
################################################################

### Find customers who made one order and became dormant for 6 months, as of 10/01/2016
pop1 <- subset(
  order_table[
    order_date<'20161001',
    .(count=.N
      ,order_date=max(order_date)
      ,order_amount=max(order_amount)
      ,product_id=max(product_id)),by=customer_id],
  count==1&order_date<'20160401')

### For the above customers, who came back and made an order within 3 months
pop2 <- subset(
  order_table[order_date>='20161001',
    .(count=.N
      ,latest_first_date=min(order_date)),by=customer_id]
  ,count>=1&latest_first_date<'20170101')

### Merge the above two tables to get our dependent variables
setkey(pop1,customer_id)
setkey(pop2,customer_id)
merge_pop <- merge(pop1,pop2,all.x=TRUE)
customer_flag <- select(merge_pop,customer_id,order_date,order_amount,product_id,latest_first_date)
customer_flag[,buyer:=ifelse(is.na(latest_first_date),0,1)]
customer_flag[,c("order_date","latest_first_date"):=NULL]


################################################################
######## Find more attributes!
################################################################

### Remove irrelavent attributes
customer_table[,c("V1","first_visit","last_visit"):=NULL]

### get the category id of the product
setkey(customer_flag,product_id)
setkey(product_table,product_id)
merge_category <- select(merge(customer_flag,product_table,all.x=TRUE),
                    customer_id,order_amount,buyer,category_id)
merge_category[,category_id:=ifelse(is.na(category_id),'unknown',category_id)]

### get the other customer attributes
setkey(merge_category,customer_id)
setkey(customer_table,customer_id)
train_temp <- merge(merge_category,customer_table,all=FALSE)

### Bucketize more varibles
train_temp[,age_bucket:=ifelse(Age<20,'0-20',ifelse(Age<40,'21-40',ifelse(Age<60,'41-60','61+')))]
train_temp[,familysize_bucket:=ifelse(Family_Size<=1,'0-1','1+')]

### Create dummy variable
country_dummy <- model.matrix( ~ Customer_Address_Country - 1, data=train_temp)
category_dummy <- model.matrix( ~ category_id - 1, data=train_temp)
gender_dummy <- model.matrix( ~ gender - 1, data=train_temp)
familysize_dummy <- model.matrix( ~ familysize_bucket - 1, data=train_temp)
age_dummy <- model.matrix( ~ age_bucket - 1, data=train_temp)

All <- cbind(train_temp,country_dummy,category_dummy,gender_dummy,familysize_dummy,age_dummy)
All[,c("Customer_Address_Country","gender","category_id","Age","Family_Size","familysize_bucket","age_bucket","gaia_id"):=NULL]

### Clean up temp tables
rm(pop1,pop2,gender_dummy,category_dummy,familysize_dummy,age_dummy,country_dummy,merge_pop,merge_category,train_temp,customer_flag)

################################################################
######## Preprocessing
################################################################

### sapply(All,function(x) sum(is.na(x)))

### remove records with missing values
All <- subset(All,!is.na(attr3))

### Zero variance
zero_variance_variables <- Filter(function(x)(length(unique(x)) == 1),All)
Filter_All <- Filter(function(x)(length(unique(x))>1), All)

### check for correlation
corrm <- cor(Filter_All[,-c("customer_id","buyer"), with = FALSE])
corrplot(corrm, order="hclust")

### remove column with correlation > 0.85
highcorr <- findCorrelation(corrm,.85)
deleted_columns <- colnames(Filter_All[,-c("customer_id","buyer"), with = FALSE][,highcorr, with = FALSE])
FilterAll <- cbind(Filter_All$buyer,Filter_All[,-c("customer_id","buyer"), with = FALSE][,-highcorr, with = FALSE])
colnames(FilterAll)[1] <- "is_Buyer"

transform_columns <- c("attr","n_of_","order_amount")
transformed_column     <- FilterAll[,grepl(paste(transform_columns, collapse = "|"),names(FilterAll)),with = FALSE]
non_transformed_column <- FilterAll[,-grepl(paste(transform_columns, collapse = "|"),names(FilterAll)),with = FALSE]
transformed_column_processed <- predict(preProcess(transformed_column, method = c("BoxCox","scale")),transformed_column)
transformedAll <- cbind(non_transformed_column, transformed_column_processed)

### Clean up temp tables
rm(transform_columns,transformed_column,non_transformed_column,FilterAll)

################################################################
######## Modeling
################################################################

set.seed(1003)
percent_of_traning <- 0.7

train_index <- createDataPartition(transformedAll$is_Buyer, p = percent_of_traning, list = FALSE, times = 1)
train_data <- transformedAll[train_index,]
test_data <- transformedAll[-train_index,]

train_x <- data.frame(subset(train_data, select = -c(is_Buyer)))
train_y <- as.factor(apply(subset(train_data, select = c(is_Buyer)), 2, as.factor))
train_y_categorial <- ifelse(train_y == 1, "YES", "NO")

test_x <- subset(test_data, select = -c(is_Buyer))
test_y <- as.factor(apply(subset(test_data, select = c(is_Buyer)), 2, as.factor))

registerDoMC(cores=8)

# logistics regression
model_logit <- glm(formula = is_Buyer ~. ,family=binomial(link='logit'),data=transformedAll)

# random forest
model_rf <- foreach(ntree=rep(100, 8), .combine=combine) %dopar% randomForest(train_x, train_y, ntree=ntree)

# lasso regression
model_glm_cv_lasso <- cv.glmnet(as.matrix(train_x),train_y_categorial,alpha = 1,family="binomial",type.measure="auc",parallel=TRUE)

################################################################
######## Modeling evaluation
################################################################

# prediction
rf_predict <- predict(model_rf, data.frame(test_x),type='prob')
rf_pred <- prediction(rf_predict[,2],test_y)

lasso_predict <- predict(model_glm_cv_lasso, as.matrix(test_x),type='response',s = "lambda.1se")
lasso_pred <- prediction(lasso_predict,test_y)

logit_predict <- predict(model_logit, test_x,type='response')
logit_pred <- prediction(logit_predict,test_y)

# auc score
rf_perf_auc <- performance(rf_pred,"auc")
lasso_perf_auc <- performance(lasso_pred,"auc")
logit_perf_auc <- performance(logit_pred,"auc")

# roc curve
rf_perf_roc <- performance(rf_pred,"tpr","fpr")
lasso_perf_roc <- performance(lasso_pred,"tpr","fpr")
logit_perf_roc <- performance(logit_pred,"tpr","fpr")

# precision-recall curve
rf_perf_pr <- performance(rf_pred,"prec","rec")
lasso_perf_pr <- performance(lasso_pred,"prec","rec")
logit_perf_pr <- performance(logit_pred,"prec","rec")
