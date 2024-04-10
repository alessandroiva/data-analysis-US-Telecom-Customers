#Telecom Churn project, Giulia Macis, Giulia Formiconi, Alessandro Alessandro Ivashkevich (GAG group)



#Firstly, we load all the libraries that we will need to read and manipulate the data, to visualize the plots, to build the model and to get metric scores.
library(naniar)
library(plot.matrix)
library(tidyverse)
library(RColorBrewer)
library(ggpubr)
library(caret)
library(skimr)
library(Hmisc)
library(ggsci)
library(dplyr)
library(gridExtra)
library(corrplot)
library(rpart)
library(rpart.plot)
library(rattle)
library(xgboost)
library(pscl)
library(pROC)
library(magrittr)
library(class)
library(nnet)
library(grid)
library(plotly)
library(glmnet)
library(randomForest)
library(ISLR2)
library(plotly)
library(MASS)
library(mvtnorm)
library(tidyverse)
library(factoextra)
library(cluster)
library(ggpubr)
library(mclust)
library(GGally)
library(ggplot2)
library(dendextend)

#The objective of this project is predict behavior to retain customers. Analyzing all relevant customer data can help to develop focused customer retention programs. 
#The dataset contains information about the State, Area, Account, Type of plan of the customers, how much they spend on charges, the schedule they do more calls and others. 
#Here these variables are explored in order to get insights about customer behavior, and Machine Learning Algorithms are used to predict if a customer will continue with its plan or not.

#Tha dataset is loaded.
TChurn <- read.csv('C:/Users/giulia macis/Desktop/data analysis project/TelecomChurn.csv',
                   header= T,
                   sep = ",",
                   stringsAsFactors = T)

# Check for null values in the entire dataframe
is.null <- any(is.na(TChurn)) #no Na values

# Check for duplicates in the entire dataframe
has_duplicates <- any(duplicated(TChurn)) #no duplicates


# There are 3333 objects of 20 variables. All variables are numeric/int except State, International.plan, Voice.mail.plan and Churn. This latter will be our target variable.
str(TChurn)

#The summary() command will provide a statistical summary of our data. 
#It gives the output as the largest value in data, the least value, mean, median, 1st quartile, and 3rd quartile. 
#Basically, we explore the structure of our data from a statistical point of view, in order to get an idea how vairbales distribution is.
summary(TChurn)


#We have to transform the Churn=true and NotChurn=false values in the target variable 'Churn' respectively into a 1 and 0 (true=1 and false=0). 
TChurn$Churn <- as.integer(as.logical(TChurn$Churn))
TChurn$Churn <- as.factor(TChurn$Churn)

#Firstly, we take into account all the quantitative variables and the corresponding distributions.
#We take the columns that are numeric, in order to create a new data frame called "TChurn_num" of all numeric variables.
TChurn_num = select_if(TChurn,is.numeric) 

#Compute mean median a standard deviation for each numeric variable and then we plot then to visualize their distributions.
#It can be seen that the most of the variables have the mean and the median equal, meaning that they follow a symmetric distribution. 
#Basically, the data are not skewed, but on the contrary they are balanced. 
means_TC <- colMeans(TChurn_num)
(median_TC = apply (X = TChurn_num, MARGIN = 2, FUN = median))
(sd_TC = apply(X = TChurn_num, MARGIN = 2, FUN = sd))
par(mfrow = c(4,2), mar = c(1,2,2,1))
for(i in 1:ncol(TChurn_num)){
  hist(TChurn_num[,i], freq = F, main = names(TChurn_num)[i],col = rgb(.7,.7,.7), border = "white", xlab = "")
  abline(v = means_TC[i], lwd = 2)
  abline(v = median_TC[i], lwd = 2, col = rgb(.7,0,0))
  legend("topright", c("Mean", "Median"), lwd = 2, col = c(1, rgb(.7,0,0)),cex = .8, bty = "n")
}

#Data visualization analysis is done considering the target variable and the most important features, in order to get useful insights.
#Firstly, we check the churn rate.
#The pie chart shows that the Churned-False are the 85.51%, whereas the Churn-True are only the 14.49%.
#This means that the dataset is imbalanced, posing some challenges in the predictive modeling, as the model might be biased towards the majority class.

#Churn Rate
churnrate <- table(TChurn$Churn) / nrow(TChurn)
churnrate

#Pie Chart: CHURN RATE
tab <- as.data.frame(table(TChurn$Churn))
slices <- c(tab[1,2], tab[2,2]) 
lbls <- c("Churned-False", "Churned-True")
pct <- round(slices/sum(slices)*100,digits = 2)  # calculating % rounded to 2 digits
lbls <- paste(lbls, pct)                         # add percents to labels 
lbls <- paste(lbls,"%",sep="")                   # ad % to labels 
pie(slices,labels = lbls, col=rainbow(length(lbls)),angle = 90,
    main="Percentage of Customer Churned")

#Table: International.plan and Churn
table(TChurn$International.plan, TChurn$Churn)
plot(table(TChurn$International.plan, TChurn$Churn), col = c("blue", "yellow"), main = "International Plan vs Churn")

#Table: Voice.mail.plan and Churn
table(TChurn$Voice.mail.plan, TChurn$Churn)
plot(table(TChurn$Voice.mail.plan, TChurn$Churn), col = c("blue", "yellow"), main = "Voice Mail Plan vs Churn")

#Table: Customer.service.calls and Churn
table(TChurn$Customer.service.calls, TChurn$Churn)

#Histogram of Account.length
t_lenght <-- (TChurn$Account.length)
par(cex=0.7, mai=c(0.1,0.1,0.2,0.1))   #make labels and margins smaller
par(fig=c(0.1,0.7,0.3,0.9)) # define area for the histogram
hist(t_lenght)
par(fig=c(0.8,1,0,1), new=TRUE)  # define area for the boxplot
boxplot(t_lenght)

#Histogram: International plan and voice mail plan 
p1 = TChurn %>% group_by(Churn,International.plan) %>% tally() %>% mutate(prop=n/sum(n)) %>% ggplot(aes(x=International.plan, y=prop,fill=Churn)) + geom_col(position="dodge") + scale_fill_jama() + labs(y="proportion") + theme_light() + theme(legend.position="bottom")
p2 = TChurn %>% group_by(Churn,Voice.mail.plan) %>% tally() %>% mutate(prop=n/sum(n)) %>% ggplot(aes(x=Voice.mail.plan, y=prop,fill=Churn)) + geom_col(position="dodge") + scale_fill_jama() + labs(y="proportion") + theme_light() + theme(legend.position="bottom")
grid.arrange(p1,p2,ncol=2,nrow=1)

#Density plot: TOTAL DAY MINUTES and TOTAL NIGHT MINUTES
p7 = TChurn %>% ggplot(aes(x=Total.day.minutes, fill=Churn)) + geom_density(alpha=0.6) + scale_fill_jama() + theme_light() 
p8 = TChurn %>% ggplot(aes(x=Total.night.minutes, fill=Churn)) + geom_density(alpha=0.6) + scale_fill_jama() + theme_light()
grid.arrange(p7,p8,ncol=1,nrow=2)

#Density plot: TOTAL EVE MINUTES and TOTAL INITIAL MINUTES
p9 = TChurn %>% ggplot(aes(x=Total.eve.minutes, fill=Churn)) + geom_density(alpha=0.6) + scale_fill_jama() + theme_light()
p10 = TChurn %>% ggplot(aes(x=Total.intl.minutes, fill=Churn)) + geom_density(alpha=0.6) + scale_fill_jama() + theme_light()
grid.arrange(p9,p10,ncol=1,nrow=2)

#Density plot: ACCOUNT LENGHT and CUSTOMER SERVICE CALLS 
p11 = TChurn %>% ggplot(aes(x=Customer.service.calls, fill=Churn)) + geom_density(alpha=0.6) + scale_fill_jama() + theme_light()
p12 = TChurn %>% ggplot(aes(x=Account.length, fill=Churn)) + geom_density(alpha=0.6) + scale_fill_jama() + theme_light()
grid.arrange(p11,p12,ncol=1,nrow=2)

dev.off()

#Now, we focus our attention on box plot in order to get some insights about the outliers and decide how to deal with them. 

#Box plot: TOTAL DAY/EVE/NIGHT/INITIAL MINUTES for Churned / Not Churned Customers
#The Q1,Q3, IQR & the area of the box plot for Total Day mins,Total Evening mins & Total International mins for Churned Customer is higher than Not Churned Customers.
#Hence, usage pattern of Chruned customers is high as compared to Not Churned.  
#There is no significant difference between the usuage of Total Night Mins for Churned and Not Churned Customers.  
#Outliers are present in all the cases especially significant in Total Int mins, Total Evening Mins & Total Night mins.

boxplot(TChurn$Total.day.minutes ~ TChurn$Churn, data = TChurn, col = "red",
        xlab = "Customer Churned",ylab = "Total Day mins")
boxplot(TChurn$Total.eve.minutes ~ TChurn$Churn, data = TChurn, col = "red",
        xlab = "Customer Churned",ylab = "Total Evening mins")
boxplot(TChurn$Total.night.minutes ~ TChurn$Churn, data = TChurn, col = "red",
        xlab = "Customer Churned",ylab = "Total Night mins")
boxplot(TChurn$Total.intl.minutes ~ TChurn$Churn, data = TChurn, col = "red",
        xlab = "Customer Churned",ylab = "Total Intl mins")
dev.off()

# Box plot: TOTAL DAY/EVE/NIGHT/NIGHT CALLS for Churned / Not Churned Customers
#As such no significant pattern difference for Churned and Not Churned Customers across the diff scenarios except Total Night Calls where spread of IQR for Churned Customers is relatively higher. 
#The Q1, Q3 are higher Total Initial Calls for Not Churned Customers is relatively higher than the Churned Customers. 
#So, can we say that Customers Churned call less relatively but the duration of their calls are relatively high (mins used from the previous box plot)
#Outliers are present in all the cases are significantly high.

boxplot(TChurn$Total.day.calls ~ TChurn$Churn, data = TChurn, col = "red",
        xlab = "Customer Churned",ylab = "Total Day Calls")      
boxplot(TChurn$Total.eve.calls ~ TChurn$Churn, data = TChurn, col = "red",
        xlab = "Customer Churned",ylab = "Total Evening Calls")
boxplot(TChurn$Total.night.calls ~ TChurn$Churn, data = TChurn, col = "red",
        xlab = "Customer Churned",ylab = "Total Night Calls")
boxplot(TChurn$Total.intl.calls~ TChurn$Churn, data = TChurn, col = "red",
        xlab = "Customer Churned",ylab = "Total Int. Calls")   
dev.off()

# Box plot: TOTAL DAY/EVE/NIGHT/NIGHT CHARGE for Churned / Not Churned Customers
#The Q1,Q3, IQR & the area of the box for Total Day Charges for Churned Customer is significantly higher than Not Churned Customers.And relatively higher for Churned Customers for all other cases as well.
#Outliers are present in all the cases are significantly high.
boxplot(TChurn$Total.day.charge ~ TChurn$Churn, data = TChurn, col = "red",
        xlab = "Customer Churned",ylab = "Total Day Charges")
boxplot(TChurn$Total.eve.charge ~ TChurn$Churn, data = TChurn, col = "red",
        xlab = "Customer Churned",ylab = "Total Evening Charges")
boxplot(TChurn$Total.night.charge~ TChurn$Churn, data = TChurn, col = "red",
        xlab = "Customer Churned",ylab = "Total Night Charges")
boxplot(TChurn$Total.intl.charge ~ TChurn$Churn, data = TChurn, col = "red",
        xlab = "Customer Churned",ylab = "Total Int. Charges")
dev.off()

# Box plot: CUSTOMER SERVICE CALLS made by Churned / Not Churned Customers
#  One can see from the Box Plot that the spread for Customer Calls made by Churned Customer is significantly more than tha Not churned ones. 
#It seems Customers going to be churned call Customer Service a lot with their issues reg service.
boxplot(TChurn$Customer.service.calls ~ TChurn$Churn, data = TChurn, col = "red",
        xlab = "Customer Churned",ylab = "Customer Service Calls")
dev.off()


# Box plot: ACCOUNT LENGHT of Churned / Not Churned Customers
#  One can see from the above Box plot for the Length of the Service, we are not able to find out as such anything meaningful.So, lets try to plot any other kind of plots rather than box plots for analysing the behaviour of Length of the Service.
boxplot(TChurn$Account.length ~ TChurn$Churn, data = TChurn, col = "red",
        xlab = "Customer Churned",ylab = "Length of the Service(in days)")
dev.off()

#CORRELATION MAP 
#The correlation map is simply a table which displays the correlation coefficients for different numerical variables.
#In our case we can notice that the following variables are highly correlated: 
#total.night.charge and total.night. minutes
#total.day.charge and total.day.minutes
#total.intl.charge and total.intl.minutes
TChurn_num = select_if(TChurn,is.numeric)
TChurn_num = data.frame(lapply(TChurn_num, function(x) as.numeric(as.character(x))))
res2=cor(TChurn_num)
corrplot(res2, type="lower", tl.col="#636363",tl.cex=0.5 )



#The categorical variables' values (Yes and No) are transformed in the variables in '1'and '0'.
TChurn$International.plan<-ifelse(TChurn$International.plan=="Yes",1,0)
TChurn$Voice.mail.plan<-ifelse(TChurn$Voice.mail.plan=="Yes",1,0)

# Since these variables are highly correlated, it is appropriate to delete one of them to reduce the redundancy.
# List of variable indices to be removed
variables_to_remove <- c(3, 9, 12, 15, 18)

# Create reduced TChurn dataframe by removing specified variables
new_churn<- TChurn[, -variables_to_remove]
#str(new_churn)


#TASK 4 
#KNN classifier using 10-fold cross validation
#Our goal is to develop a churn prediction model that is both accurate and generalizable, allowing you to make informed decisions and take appropriate actions to mitigate customer churn.
#We have divided the train dataset into two parts: a new train data and a validation data.
set.seed(1)  #make it reproducible
validationIndex <- createDataPartition(new_churn$Churn, p=0.70, list=FALSE)
train <- new_churn[validationIndex,] # 70% of data to training
test <- new_churn[-validationIndex,] # remaining 30% for test
# Run the algorithm using 10-fold cross validation: 
set.seed
fit.knn <- train(Churn ~ Total.day.minutes+Customer.service.calls,  #choosing the response variable and the predictor variables
                 data=train, 
                 method="knn",
                 tuneGrid   = data.frame(k = 1:50), #the grid of hyperparameter values to be evaluated
                 metric="Accuracy" , # the evaluation metric to optimize during model training
                 trControl = trainControl(method="cv", number=10)) #specifies the cross-validation configuration
knn.k1 <- fit.knn$bestTune # keep this Initial k for testing with knn() function in next section
print(fit.knn)
plot(fit.knn) #the best k is 17 with accuracy around 90%

set.seed(7)
prediction <- predict(fit.knn, newdata = test) #run predictions using KNN on the test set
cf <- confusionMatrix(prediction, test$Churn) #print out the confusion matrix
cf
#With initial k = 17, the model correctly predict 86.49% target variable in test dataset
roc <- pROC::roc(test$Churn,
                 as.numeric(prediction),
                 plot = TRUE,
                 col = "midnightblue",
                 lwd = 3,
                 auc.polygon = T,
                 auc.polygon.col = "lightblue",
                 print.auc = T,
                 main = "ROC Curve kNN") #the AUC score for the model is 0.5774

#TASK 5 
#We build our classification model in order to predict the customer churn:
#-linear models: Logistic Regression, BIC and AIC step-selection
#-penalized approach: LASSO
#-non-linear models: Decision Tree, Random Forest and XGBoost 
 

#LOGISTIC REGRESSION
#what threshold should we use?
logit.model = glm(Churn ~ ., family = "binomial", data = train)
best_t = 0
best_score = 0     #the score is the sum of the specificity, accuracy and sensitivity
for(t in seq(0.01,0.98, by = 0.01)){
  probs = predict(logit.model,new_churn, type = "response")    #predict the logit.model on new_churn set 
  contrasts(new_churn$Churn)
  logit.pred = rep (0 ,length(new_churn$Churn))    
  logit.pred[probs > t] =  1
  tab <- table(logit.pred, new_churn$Churn)
  dt <-dim(tab)
  mean(logit.pred == new_churn$Churn)
  sensitivity <- tab[1, 1] / (tab[1, 1] + tab[1, 2])
  specificity <- tab[2, 2] / (tab[2, 2] + tab[2, 1])
  
  sum_values = mean(logit.pred == new_churn$Churn)+sensitivity(tab)+specificity(tab) 
  if(sum_values > best_score){
    best_score = sum_values
    best_t = t
  }
}
print(best_t) #The best threshold found is 0.18

logit.model = glm(Churn ~ ., family = "binomial", data = train)
probs = predict(logit.model,test, type = "response")          # predict the logit.model on TChurn_validation dataset 

coefficients <- coef(logit.model)
#contrasts(test$Churn)
logit.pred = rep (0 ,length(test$Churn))       # replacing all values of logit.pred with default as False
logit.pred[probs > 0.18] =  1                           # best_t = 0.18
t <- table(logit.pred, test$Churn)                     # Confusion Matrix  
t

mean(logit.pred == test$Churn) #Accuracy
sensitivity(t)
specificity(t) 
# ROC curves
roc <- pROC::roc(train$Churn,
                 logit.model$fitted.values,
                 plot = TRUE,
                 col = "midnightblue",
                 lwd = 3,
                 auc.polygon = T,
                 auc.polygon.col = "lightblue",
                 print.auc = T,
                 main = "ROC Curve logistic regression")



#AIC AND BIC
#logistic regression with all variables
logit_fit1 <- glm(Churn ~ .,
                  family = "binomial",
                  data = new_churn )
summary(logit_fit1)

#AIC AND BIC
# Forward
logit_fit1 <- glm(Churn ~ .,
                  family = "binomial",
                  data = train )
summary(logit_fit1)

logit_fit_aic1 <- step(glm(Churn ~ 1,
                           family = "binomial",
                           data = train),
                       scope = formula(logit_fit1),
                       direction = "forward")

# Backward
logit_fit_aic2 <- step(logit_fit1,
                       direction = "backward") 

# Both directions
logit_fit_aic3 <- step(logit_fit1,
                       direction = "both")

sort(coefficients(logit_fit_aic1))
excluded_vars <- setdiff(names(logit_fit1$coefficients), names(logit_fit_aic1$coefficients))
excluded_vars
sort(coefficients(logit_fit_aic2))
excluded_vars <- setdiff(names(logit_fit1$coefficients), names(logit_fit_aic2$coefficients))
excluded_vars
sort(coefficients(logit_fit_aic3))
excluded_vars <- setdiff(names(logit_fit1$coefficients), names(logit_fit_aic3$coefficients))
excluded_vars

#anova test
anova(logit_fit1, logit_fit_aic1, test = "Chisq") 

# Forward
logit_fit_bic1 <- step(glm(Churn~ 1,
                           family = "binomial",
                           data = train ),
                       scope = formula(logit_fit1),
                       direction = "forward",
                       k = log(nrow(train)))

# Backward
logit_fit_bic2 <- step(logit_fit1,
                       direction = "backward",
                       k = log(nrow(train ))) 

# Both directions
logit_fit_bic3 <- step(logit_fit1,
                       direction = "both",
                       k = log(nrow(train)))

sort(coefficients(logit_fit_bic1))
sort(coefficients(logit_fit_bic2))
sort(coefficients(logit_fit_bic3))

excluded_vars <- setdiff(names(logit_fit1$coefficients), names(logit_fit_bic1$coefficients))
excluded_vars
excluded_vars <- setdiff(names(logit_fit1$coefficients), names(logit_fit_bic2$coefficients))
excluded_vars
excluded_vars <- setdiff(names(logit_fit1$coefficients), names(logit_fit_bic3$coefficients))
excluded_vars

anova(logit_fit1, logit_fit_bic1, test = "Chisq") 

tt <- 0.5

pred_aic <- as.factor(ifelse(logit_fit_aic1$fitted.values > tt, 1, 0))
pred_bic <- as.factor(ifelse(logit_fit_bic1$fitted.values > tt, 1, 0))

# Confusion matrix
table(pred_aic, train$Churn)
table(pred_bic, train$Churn)

# Accuracy and misclassification error
(aic_acc <- mean(pred_aic ==  train$Churn))
(aic_misc <- 1 - aic_acc)
(bic_acc <- mean(pred_bic ==  train$Churn))
(bic_misc <- 1 - bic_acc)


# Predictions for the observations in the test set
set.seed(123)
prob_out_aic <- predict(logit_fit_aic1,
                        newdata = test,
                        type = "response")
pred_out_aic <- as.factor(ifelse(prob_out_aic > tt, 1,0))
prob_out_bic <- predict(logit_fit_bic1,
                        newdata = test,
                        type = "response")
pred_out_bic <- as.factor(ifelse(prob_out_bic > tt, 1,0 ))


# Confusion matrix
table(pred_out_aic, test$Churn)
table(pred_out_bic, test$Churn)

# Accuracy and misclassification error
(aic_acc <- mean(pred_out_aic == test$Churn))
(aic_misc <- 1 - aic_acc)
(bic_acc <- mean(pred_out_bic == test$Churn))
(bic_misc <- 1 - bic_acc)


# Confusion matrix
table(pred_out_aic, test$Churn)
table(pred_out_bic, test$Churn)

# Accuracy and misclassification error
(aic_acc <- mean(pred_out_aic == test$Churn))
(aic_misc <- 1 - aic_acc)
(bic_acc <- mean(pred_out_bic == test$Churn))
(bic_misc <- 1 - bic_acc)


# ROC curves
roc_aic <- pROC::roc(test$Churn,
                     prob_out_aic,
                     plot = TRUE,
                     col = "midnightblue",
                     lwd = 3,
                     auc.polygon = TRUE,
                     auc.polygon.col = "lightblue",
                     print.auc = TRUE,
                     main = "ROC Curve (AIC)")

roc_bic <- pROC::roc(test$Churn,
                     prob_out_bic,
                     plot = TRUE,
                     col = "midnightblue",
                     lwd = 3,
                     auc.polygon = TRUE,
                     auc.polygon.col = "lightblue", 
                     print.auc = TRUE,
                     main = "ROC Curve (BIC)")
# AUC scores
roc_aic$auc
roc_bic$auc


##LASSO##
Test_<-test
Train<-train
str(Train)
str(Test_)

#as.integer function transform the level 1 of the factor into the integer 1 and the level 2 as integer 2
Train$Churn<- as.integer(Train$Churn)

#Transform 1 into 0 and 2 into 1
Train$Churn <- ifelse(Train$Churn == 1, 0, ifelse(Train$Churn == 2, 1, Train$Churn))

#remove the first variable of train and test because we won't use it for our model
Train<-Train[,-1]
Test_<-Test_[,-1]
str(Test_)

#define predictor and response variables in training set
train_x <- data.matrix(Train[,-14])
train_y = Train[,14]

#define predictor and response variables in testing set
test_x <- data.matrix(Test_[,-14])
test_y = Test_[,14]

##perform k-fold cross-validation to find optimal lambda value
cv_model <- cv.glmnet(train_x, train_y, alpha = 1) #cv.glmnet() automatically performs k-fold cross validation using k = 10 folds.

#find optimal lambda value that minimizes test MSE
best_lambda <- cv_model$lambda.min
best_lambda

#produce plot of test MSE by lambda value
plot(cv_model)

best_model <- glmnet(train_x, train_y, alpha = 1, lambda = best_lambda)
y_predictions <- predict(best_model,newx = test_x)


roc <- pROC::roc(test_y,
                 y_predictions,
                 plot = TRUE,
                 col = "midnightblue", 
                 lwd = 3,
                 auc.polygon = T,
                 auc.polygon.col = "lightblue",
                 print.auc = T,
                 main = "ROC curve Lasso")
# AUC scores
roc$auc




#DECISION TREE
library(tree)
#The output tells that that R used 7 variables in the decision tree and that we have a total of 16 terminal nodes. 
#We also see that the residual mean deviance (RMD) is 0.3239 an we have a ~5% misclassification rate. 
#RMD is a metric that indicates how well the tree fits the data.
#Generally, a lower value indicates a better fitted decision tree. The classification error rate on the other hand, is an indicator of model accuracy. 
#We see that out of 3,333 observation, the tree misclassified 197 (~5%) observations.
treemodel <- tree(Churn ~.-State, data = train)
summary(treemodel) 

plot(treemodel)
text(treemodel,pretty=0)

#predictions for observations in the test set
predict_tree <- predict(treemodel, test,type = "class")
mean(predict_tree == test$Churn)
pt = confusionMatrix(predict_tree,test$Churn)   #Accuracy 0.94  
pt

Specificity = 842/(842 + 13)  #Specificity 0.98                       
Specificity
Sensitivity = 101 / (101 + 43)  #Sensitivity 0.70                 
Sensitivity

#AUC score is 0.843 
t$tp1=predict_tree
roc_t= roc(response= test$Churn, predictor = factor(t$tp1, ordered=TRUE), plot=TRUE, print.auc=TRUE)

## Improvising the model further by using various techniques 
## Like Pruning, so that it does not overfit the train set.
## Lets try to apply the prune method and see if we can further improve the model or not.
set.seed(1000)
tree_validate <- cv.tree(object = treemodel,FUN = prune.misclass )  # Use of cv.tree function to calculate the determine the optimal no. of tree levels
tree_validate                                                       # Took 7 tree levels into consideration
plot(x=tree_validate$size, y=tree_validate$dev, type="b")

## From the above plot one can see that tree_validate$dev diff is same from for tree levels 11-14, so can we assume the best tree level size to be 12 (rather than original number 15) at the cost of some bias. 

tree_model_prun <- prune.misclass(treemodel, best = 12)
plot(tree_model_prun)
text(tree_model_prun, pretty=0)
summary(tree_model_prun)

#prediction of observations in test set
predict_tree_prun <- predict(tree_model_prun, test,type = "class")
mean(predict_tree_prun == test$Churn) #Accuracy 0.93
confusionMatrix(predict_tree_prun,test$Churn)
Specificity = 733 /(733+21)     #Specificity 0.97                          
Specificity
Sensitivity = 92 / (92 + 36)    #Sensitivity 0.72                         
Sensitivity

t$tp2= predict_tree_prun
roc_t <- pROC::roc(response = test$Churn,
                   predictor = factor(t$tp2, ordered = TRUE),
                   plot = TRUE,
                   col = "midnightblue",
                   lwd = 3,
                   auc.polygon = TRUE,
                   auc.polygon.col = "lightblue", 
                   print.auc = TRUE,
                   main = "ROC Curve decision tree")


#RANDOM FOREST
tr_cont = trainControl(method="cv", number=10)
random_forest_fit <- randomForest(train$Churn ~ ., trControl  = tr_cont, data = train, importance=TRUE, keep.forest=TRUE)
importance(random_forest_fit)
importance(random_forest_fit, type=1, scale=FALSE)
importance_sorted <- importance(random_forest_fit)[order(importance(random_forest_fit)[, "MeanDecreaseGini"], decreasing = TRUE), ]
print(importance_sorted)
# Customize the variable importance plot
barplot(importance_sorted[, "MeanDecreaseGini"],
        horiz = TRUE,
        names.arg = rownames(importance_sorted),
        main = "Variable Importance Plot",
        xlab = "Mean Decrease Gini",
        ylab = "Variables",
        col = "blue",
        las = 1,
        cex.names = 0.3)

pred=predict(random_forest_fit, test, type = "prob")
oob_error <- random_forest_fit$err.rate[, "OOB"]
oob_error_rate <- mean(random_forest_fit$err.rate[, "OOB"])
print(oob_error_rate)


random_forest_fit_2 <- randomForest(train$Churn ~ Total.day.minutes+State+Customer.service.calls+Total.eve.minutes+International.plan+Total.intl.minutes+Total.intl.calls+Total.night.minutes+Total.day.calls+Account.length+Total.night.calls, trControl  = tr_cont, data = train, importance=TRUE)
oob_error_rate_2 <- mean(random_forest_fit_2$err.rate[, "OOB"])
print(oob_error_rate_2)
pred_2=predict(random_forest_fit_2, test, type = "prob")
#Plot the ROC curve
roc_rf = roc(test$Churn, pred[,2])
(ROC_rf_auc <- auc(roc_rf))
plot(roc_rf, col = "midnightblue", main = "ROC curve Random Forest", auc.polygon = T, auc.polygon.col = "lightblue", print.auc = T)

roc_rf_2 = roc(test$Churn, pred_2[,2])
(ROC_rf_auc <- auc(roc_rf_2))
plot(roc_rf, col = "midnightblue", main = "ROC curve Random Forest with selected features", auc.polygon = T, auc.polygon.col = "lightblue", print.auc = T)

#XG BOOST
#To implement this model our binary variable we need to predict, must be numeric, not factor.
#However we didn't use as.integer(as.logical(train$Churn)) because it yields NA values
#in order to not change permanently our train and test dataset we will copy them, and use these copies for this model

#XGBoost is a scalable and highly accurate implementation of gradient boosting that pushes the limits of computing power for boosted tree algorithms
#For the second one the motivation is more or less the same in fact the AUC curve give as a result of 0.923 in the validation set
#Trying also other approaches we found out that XGboost is the one performing better.
#XGBoost is a supervised machine learning method which is based on decision trees and improves on other methods such as random forest and gradient boost.
#XGboost is an algorithm which has scarse interpretability but performs well as we can notice from the AUC that is 0.988 in the validation set.

set.seed(3)

#Note that the xgboost package also uses matrix data, so we’ll use the data.matrix() 
#function to hold our predictor variables.
xgb_train = xgb.DMatrix(data = train_x, label = train_y)
xgb_test = xgb.DMatrix(data = test_x, label = test_y)
watchlist = list(train=xgb_train, test=xgb_test)

#fit XGBoost model and display training and testing data at each round
#Done in order to find the max number of boosting iterations used to build the model later on. 
#It finds all the RMSE and we pick the smallest in the test.
model = xgb.train(data = xgb_train, max.depth = 3, watchlist=watchlist, nrounds = 200)

#we build our model using a max.depth of 3 and 183 number of rounds.
final = xgboost(data = xgb_train, max.depth = 3, nrounds = 183, verbose = 0, objective = "binary:logistic")

#AUC score is 0.923
pred_y_train = predict(final, xgb_train, type = 'response')
pred_y_test= predict(final, xgb_test, type = 'response')
par(mfrow = c(1 ,1))
roc_test <- pROC::roc(test_y,
                      pred_y_test,
                      plot = TRUE,
                      col = "midnightblue",
                      lwd = 3,
                      auc.polygon = T,
                      auc.polygon.col = "lightblue",
                      print.auc = T,
                      main = "ROC Curve xgboost")



###clustering

# List of variable indices to be removed
variables_to_remove <- c(1,20)

# Create reduced_Tchurn dataframe by removing specified variables
reduced_Tchurn <- TChurn[, -variables_to_remove]
#reduced_Tchurn <- TChurn[,-1]
#let's see the distribution of our new dataset
#ggpairs(reduced_Tchurn)


reduced_Tchurn_scaled <- scale(reduced_Tchurn)
reduced_Tchurn_scaled <- data.frame(reduced_Tchurn_scaled)
#ggpairs(reduced_Tchurn_scaled)

#finding the optimal number of cluster with cross validation using the wss method
nFolds = 6
folds = sample(rep(1:nFolds, length.out = nrow(reduced_Tchurn_scaled)))
X = reduced_Tchurn_scaled[which(folds == 1), ]
print(fviz_nbclust(X, kmeans, method = 'wss'))

X = reduced_Tchurn_scaled[which(folds == 3), ]
print(fviz_nbclust(X, kmeans, method = 'wss'))

X = reduced_Tchurn_scaled[which(folds == 6), ]
print(fviz_nbclust(X, kmeans, method = 'wss'))

#seems between 2 or 3
#now let's use the 'silhouette' method
for (i in 1:nFolds) {
  X <- reduced_Tchurn_scaled[which(folds == i), ]
  print(fviz_nbclust(X, kmeans, method = 'silhouette'))
}

#the graphs indicate that 2 is the optimal number of cluster

#At this point we can try a normal kmeans and see the clusters that it makes.

km = kmeans(reduced_Tchurn_scaled, 2, nstart = 100, iter.max = 1e2)
clusters_km = km$cluster

fviz_cluster(km, reduced_Tchurn_scaled, geom = c("point"), ellipse.type = "norm", pointsize = 1) +
  theme_minimal() +
  scale_fill_brewer(palette = "Paired") +
  labs(x = NULL, y = NULL)  # Rimuovi le etichette per gli assi x e y
barplot(table(clusters_km), col="blue")


#the 2 clusters seems pretty clear, as the total within-cluster sum of squares (WSS),which is pretty good for our clustering

#let's see how churn is distributes among the two clusters
TChurn$cluster <- factor(km$cluster)   #we add the cluster variable to our initial dataframe
#plot the graph
ggplot(TChurn, aes(fill=Churn, x = cluster)) +
  geom_bar(position="dodge")+
  ggtitle("How churn is distributed among clusters")+
  xlab("Churn?")+
  ylab("Number of consumers") +
  theme_classic()

table(TChurn$cluster, TChurn$Churn)



#Now we can try computing hierarchical clustering, and see what best suits our data
#distance based method, distance chosen euclidean
dist_eucl <- factoextra::get_dist(reduced_Tchurn_scaled,
                                  method = "euclidean")

# Within Sum of Squares
factoextra::fviz_nbclust(x = reduced_Tchurn_scaled, 
                         FUNcluster = factoextra::hcut, #hierarchical clustering
                         diss = dist_eucl,              #Euclidean distance       
                         method = "wss",                #WSS
                         k.max = 20)                    #maximum value for k
#from the plot we deduce that the elbow point is 2
# A more reliable approach is using the Silhouette index.

# Silhouette
factoextra::fviz_nbclust(x = reduced_Tchurn_scaled, 
                         FUNcluster = factoextra::hcut,   
                         diss = dist_eucl,          
                         method = "silhouette",         #only difference
                         k.max = 20)   
# The optimal number of groups is k = 3.

#distance based method, distance chosen euclidean

hc_complete <- factoextra::hcut(x = dist_eucl, 
                                k = 3,
                                hc_method = "complete")
hc_average <- factoextra::hcut(x = dist_eucl, 
                               k = 3,
                               hc_method = "average")
hc_ward <- factoextra::hcut(x = dist_eucl, 
                            k = 3,
                            hc_method = "ward.D2")

#running the dendograms takes too much time 
#h1=factoextra::fviz_dend(x = hc_complete) 
#h2=factoextra::fviz_dend(x = hc_average)
#h3=factoextra::fviz_dend(x = hc_ward)
#ggarrange(h1,h2, ncol = 4, nrow=1)

# Visualization of clustering results
#store the clusters in our dataset
TChurn$hc_complete = as.integer(cutree(hc_complete,k = 3))
TChurn$hc_average= as.integer(cutree(hc_average, k=3))
TChurn$hc_ward= as.integer(cutree(hc_ward, k=3))


#plotting the sizes of our clusters
ggplot(TChurn, aes(x = hc_complete)) + geom_bar(fill = "#00bfff") +
  xlab("clusters") +
  ggtitle("What is the clusters size?") +
  ylab("number of consumers") +
  theme_classic()


#see that they are quite un-homogeneous with the complete linkage

ggplot(TChurn, aes(x = hc_average)) + geom_bar(fill = "#00bfff") +
  xlab("clusters") +
  ggtitle("What is the clusters size?") +
  ylab("number of consumers") +
  theme_classic()
#the average linkage method is not suitable for our data since it creates one single cluster 
ggplot(TChurn, aes(x = hc_ward)) + geom_bar(fill = "#00bfff") +
  xlab("clusters") +
  ggtitle("What is the clusters size?") +
  ylab("number of consumers") +
  theme_classic()
#the ward linkage method seems to yield more homogeneous clusters

#let's see how churn is distributed among clusters
ggplot(TChurn, aes(fill=Churn, x = hc_complete)) +
  geom_bar(position="dodge")+
  ggtitle("How churn is distributed among clusters")+
  xlab("Churn?")+
  ylab("Number of consumers") +
  theme_classic()
table(TChurn$hc_complete, TChurn$Churn)

#for ward method
ggplot(TChurn, aes(fill=Churn, x = hc_ward)) +
  geom_bar(position="dodge")+
  ggtitle("How churn is distributed among clusters")+
  xlab("Churn?")+
  ylab("Number of consumers") +
  theme_classic()

table(TChurn$hc_ward, TChurn$Churn)

#Now we can do the same thing using the Pearson correlation 
dist_pear <- factoextra::get_dist(reduced_Tchurn_scaled,
                                  method = "pearson")
# Within Sum of Squares
factoextra::fviz_nbclust(x = reduced_Tchurn_scaled, 
                         FUNcluster = factoextra::hcut, #hierarchical clustering
                         diss = dist_pear,              #Euclidean distance       
                         method = "wss",                #WSS
                         k.max = 20)                    #maximum value for k
#from the plot we see that the elbow point could fall between 2 and 4, let's see using the Silhouette index.

# Silhouette
factoextra::fviz_nbclust(x = reduced_Tchurn_scaled, 
                         FUNcluster = factoextra::hcut,   
                         diss = dist_pear,          
                         method = "silhouette",        
                         k.max = 20)   
# The optimal number of groups is k = 2
# Hierarchical clustering using the optimal number of groups chosen via silhouette
hier_per_ward <- factoextra::hcut(x = dist_pear, 
                                  k = 2,
                                  hc_method = "ward.D2")   #agglomeration method
# Ward.D2 is the most statistically valid method (but it may be worth to try other methods)
hier_per_complete <- factoextra::hcut(x = dist_pear, 
                                      k = 2,
                                      hc_method = "complete")
hier_per_single <- factoextra::hcut(x = dist_pear, 
                                    k = 2,
                                    hc_method = "single")
# Dendrogram
#h_1=factoextra::fviz_dend(x = hier_per_ward) 
#h_2=factoextra::fviz_dend(x = hier_per_complete) 
#h_3=factoextra::fviz_dend(x = hier_per_single)
#ggarrange(h_1,h_2,h_3, ncol = 4, nrow=1)

#now the resulting clusters are stored in new columns of the TChurn data frame. 

TChurn$hier_per_complete = as.integer(cutree(hier_per_complete,k = 2))
TChurn$hier_per_ward= as.integer(cutree(hier_per_ward, k=2))
TChurn$hier_per_single = as.integer(cutree(hier_per_single, k=2))

#plotting the sizes of our clusters
ggplot(TChurn, aes(x = hier_per_complete)) + geom_bar(fill = "#00bfff") +
  xlab("clusters") +
  ggtitle("clusters size of complete method") +
  ylab("number of consumers") +
  theme_classic()

ggplot(TChurn, aes(x = hier_per_ward)) + geom_bar(fill = "#00bfff") +
  xlab("clusters") +
  ggtitle(" clusters size ward method") +
  ylab("number of consumers") +
  theme_classic()

ggplot(TChurn, aes(x = hier_per_single)) + geom_bar(fill = "#00bfff") +
  xlab("clusters") +
  ggtitle("clusters size single method") +
  ylab("number of consumers") +
  theme_classic()
#the single method is not suitable for our data since it creates only one cluster

#let's see how churn is distributed among clusters
ggplot(TChurn, aes(fill=Churn, x = hier_per_complete)) +
  geom_bar(position="dodge")+
  ggtitle("How churn is distributed among clusters (complete method)")+
  xlab("Churn?")+
  ylab("Number of consumers") +
  theme_classic()
table(TChurn$hier_per_complete, TChurn$Churn)
#for ward method
ggplot(TChurn, aes(fill=Churn, x = hier_per_ward)) +
  geom_bar(position="dodge")+
  ggtitle("How churn is distributed among clusters (ward method)")+
  xlab("Churn?")+
  ylab("Number of consumers") +
  theme_classic()
table(TChurn$hier_per_ward, TChurn$Churn)


#what is the most accurate clustering technique?
#Calculate the silhouette scores
sil_scores <- silhouette(clusters_km, dist(reduced_Tchurn_scaled))
# Get the average silhouette score
avg_sil_score <- mean(sil_scores[, 3])


#for hierarchical distance based
sil_scores <- silhouette(TChurn$hc_complete , dist(reduced_Tchurn_scaled))
avg_sil_score_hc_complete  <- mean(sil_scores[, 3])


sil_scores <- silhouette(TChurn$hc_ward , dist(reduced_Tchurn_scaled))
avg_sil_score_hc_ward <- mean(sil_scores[, 3])

#for correlation based hierarchical clustering
sil_scores <- silhouette(TChurn$hier_per_complete , dist(reduced_Tchurn_scaled))
avg_sil_score_hier_per_complete <- mean(sil_scores[, 3])


sil_scores <- silhouette(TChurn$hier_per_ward , dist(reduced_Tchurn_scaled))
avg_sil_score_hier_per_ward <- mean(sil_scores[, 3])


#print and compare the outcome

cat("Average silhouette score of kmeans:", avg_sil_score)
cat("Average silhouette score distance based hierarchical complete :", avg_sil_score_hc_complete)
cat("Average silhouette score of distance based hierarchical ward:", avg_sil_score_hc_ward)
cat("Average silhouette score of correlation based hierarchical complete:", avg_sil_score_hier_per_complete)
cat("Average silhouette score of correlation based hierarchical ward:", avg_sil_score_hier_per_ward)

#the clusters with the highest average silhouette score in the one performed with the distance based hierarchical clustering using the ward method.
#how churn is distributed?
#we already seen that churn and not churn is not much related to our clusters, because in the most populated clusters we see that we have the highest rate of both churn an not churn.
adjustedRandIndex(TChurn$hc_ward, TChurn$Churn)


