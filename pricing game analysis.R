#read required libraries
library(xgboost)
library(Matrix)

#read data
data <- read.csv('pricing competition data.csv', encoding = 'UTF-8')

#data intro
head(data)
str(data)
plot(data$RMSE, data$Average.Profit)

#set training - test - validation batches
{
  set.seed(1)
  test_batch <- 5
  validation_batch <- 4
  data$test_data=sample.int(5, nrow(data), replace=TRUE)
}

#lm
lm0 = lm(data=data[data$test_data!=test_batch,], Average.Profit ~
           ifelse(RMSE<500.3,500.3,ifelse(RMSE>504.3,504.3,RMSE))
         +ifelse(RMSE>500.3,500.3,RMSE)
         *ifelse(Market.Share>.70,.70,Market.Share)
         +ifelse(Market.Share<.70,.70,Market.Share)
)
lm1 = lm(data=data[data$test_data!=test_batch,], Average.Profit ~
           ifelse(RMSE<500.3,500.3,ifelse(RMSE>504.3,504.3,RMSE))
         +I(ifelse(RMSE>500.3,500.3,RMSE)^2)
         *ifelse(Market.Share>.70,.70,Market.Share)
         +ifelse(Market.Share<.70,.70,Market.Share)
)

AIC(lm0, lm1)
summary(lm0)
summary(lm1)

#predict for out of sample lm
data$predicted_values_lm <- predict(lm1, data, type = 'response')

#glm
profit_shift <- 200000

glm0 <- glm(
  data = data[data$test_data!=test_batch,]
  ,family = Gamma(link='log')
  ,formula = profit_shift - Average.Profit ~
    log(ifelse(RMSE<500.3,500.3,ifelse(RMSE>504.3,504.3,RMSE)))
  +ifelse(RMSE>500.3,500.3,RMSE)
  *ifelse(Market.Share>.75,.75,Market.Share)
  +ifelse(Market.Share<.75,.75,ifelse(Market.Share>0.96,0.96,Market.Share))
  +(Market.Share==0)
)

glm1 <- glm(
  data = data[data$test_data!=test_batch,]
  ,family = Gamma(link='log')
  ,formula = profit_shift - Average.Profit ~
    log(ifelse(RMSE<500.3,500.3,ifelse(RMSE>504.3,504.3,RMSE)))
  +ifelse(RMSE>500.3,500.3,RMSE)
  *ifelse(Market.Share>.75,.75,Market.Share)
  +ifelse(Market.Share<.75,.75,ifelse(Market.Share>0.96,0.96,Market.Share))
  +(Market.Share==0)
)

AIC(glm0, glm1)
summary(glm0)
summary(glm1)

#predict for out of sample glm
data$predicted_values_glm <- profit_shift-predict(glm0, data, type = 'response')


#xgboost
#prepare data
Data_matrix<-sparse.model.matrix(Average.Profit ~ RMSE+test_data+Market.Share, data = data)
xgb.data.train <- xgb.DMatrix(data = Data_matrix[!(Data_matrix[,"test_data"] %in% c(test_batch,validation_batch)),],
                              label = data[!(data$test_data %in% c(test_batch,validation_batch)),]$Average.Profit)

xgb.data.test <- xgb.DMatrix(data = Data_matrix[Data_matrix[,"test_data"] == validation_batch,],
                             label = data[data$test_data == validation_batch,]$Average.Profit)


watchlist <- list(train = xgb.data.train, test = xgb.data.test)

#run xgboost
xgb <- xgb.train(
  data = xgb.data.train,
  nrounds=2000,
  early_stopping_rounds = 50,
  watchlist = watchlist,
  tree_method='hist',
  verbose = 1,
  eta = 0.25,
  max_depth = 2,
  min_child_weight = 3,
  gamma = 0
)

#predict for out of sample xgboost
data$prediction_xgboost <- predict(xgb, Data_matrix, reshape=T)

write.csv(data, file = 'predictions.csv')


#visualise outputs
{
  smooth_data_RMSE <- data.frame(RMSE=rep(499.4,100))
  smooth_data_RMSE$Market.Share <- (1:100)/100
  smooth_data_RMSE$Average.Profit <- 0
  smooth_data_RMSE$test_data <- 0
  Data_matrix2<-sparse.model.matrix(Average.Profit ~ RMSE+test_data+Market.Share, data = smooth_data_RMSE)
  smooth_data_RMSE$prediction_xgboost <- predict(xgb, Data_matrix2, reshape=T)
  smooth_data_RMSE$predicted_values_lm <- predict(lm1, smooth_data_RMSE, type = 'response')
  plot(y=smooth_data_RMSE$predicted_values_lm, x= smooth_data_RMSE$Market.Share, type="l")
  #lines(y=smooth_data_RMSE$prediction_xgboost, x= smooth_data_RMSE$Market.Share, type="l")
}

{
  smooth_data_Market_Share <- data.frame(Market.Share=rep(.2,100))
  smooth_data_Market_Share$RMSE <- 499.4+(1:100)/25
  smooth_data_Market_Share$Average.Profit <- 0
  smooth_data_Market_Share$test_data <- 0
  Data_matrix2<-sparse.model.matrix(Average.Profit ~ RMSE+test_data+Market.Share, data = smooth_data_Market_Share)
  smooth_data_Market_Share$prediction_xgboost <- predict(xgb, Data_matrix2, reshape=T)
  smooth_data_Market_Share$predicted_values_lm <- predict(lm1, smooth_data_Market_Share, type = 'response')
  #plot(y=smooth_data_Market_Share$prediction_xgboost, x= smooth_data_Market_Share$RMSE, type="l")
  lines(y=smooth_data_Market_Share$prediction_xgboost, x= smooth_data_Market_Share$RMSE, type="l")
}
