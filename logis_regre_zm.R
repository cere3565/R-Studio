# logistic function
x <- seq(-6, 6, length = 500)
y <- exp(x)/(1+exp(x))
plot(x, exp(x)/(1+exp(x)))

# example 7.7 of section 7.2.1 
# Using logistic regression : Understanding logistic regression 
# Title: Loading the CDC data 

# Step 2 – exploring and preparing the data
cre_raw <- read.csv("credit-1.csv")
# 按照父母的種族區分

# 詳見 CDC-natality-data-file-UserGuide-2010.pdf
train <- cre_raw[cre_raw$years_at_residence<=2,]
test <- cre_raw[cre_raw$years_at_residence>2,]

# slide 7-12
str(cre_raw)
defaultProp1 <- table(cre_raw$default)
prop.table(defaultProp1)     #閥值設定

# Explore the data 
str(train)
summary(train$default)
defaultProp2 <- table(train$default)
prop.table (defaultProp2)

str(test)
defaultProp3 <- table(test$default)
prop.table (defaultProp3)

plot(train$age, train$default)

# Title: Building the model formula 
# Step 3 – training a model on the data

riskfactors <- c("months_loan_duration","credit_history","purpose",
                 "amount","employment_duration","age",
                 "existing_loans_count","job")
y <- "default"
x <- c(riskfactors)
# Concatenate Strings
fmla <- paste(y, paste(x, collapse="+"), sep="~")

# example 7.9 of section 7.2.2 
# Title: Fitting the logistic regression model 
print(fmla)
model <- glm(fmla, data=train, family=binomial(link="logit"))
str(model)

# Step 4 – evaluating model performance
summary(model)

# Title: Calculating the pseudo R-squared 
# Calculate rate of positive examples in dataset. 
pnull <- mean(as.numeric(train$default))   
loglikelihood <- function(y, py) {                                  
  sum(y * log(py) + (1-y)*log(1 - py)) }
# Calculate null deviance.                    
null.dev <- -2*loglikelihood(as.numeric(train$default), pnull)       
# Create vector of predictions for training data. 
pred <- predict(model, newdata=train, type="response")  
#   Calculate deviance of model for training data. 
resid.dev <- -2*loglikelihood(as.numeric(train$default), pred)     
pr2 <- 1-(resid.dev/null.dev)
print(pr2)

# example 7.10 of section 7.2.3 
# Title: Applying the logistic regression model 
train$pred <- predict(model, newdata=train, type="response")
test$pred <- predict(model, newdata=test, type="response")

train2 <- predict(model, newdata=train)
head(train2)
exp(train2[1:5]) / (1+exp(train2[1:5]))

# example 7.11 of section 7.2.3 
# Title: Plotting distribution of prediction score grouped by known outcome 
library(ggplot2)
ggplot(train, aes(x=pred, color=default, linetype=default)) +
  geom_density()
install.packages("ROCR") 
library(ROCR)
eval <- prediction(train$pred, train$default)

plot(performance(eval, "tpr"), main = " sensitivity (敏感度)" )
plot(performance(eval, "tnr"), main = " specificity (特異性)" )
?performance


plot(performance(eval, "tpr", "fpr"),
     print.cutoffs.at=c(0, 0.02, 0.1),
     text.adj=c(1,1), main ="ROC curve")
     

library(gmodels)
train_pred <- predict(model, newdata=train, type="response")
# train_pred_TF <- ifelse(train_pred >= 0.02, 1, 0)
# str(train_pred_TF)

summary(train$pred)
plot(train$pred)
head(train$pred)
train$pred[1:5]
sum(train$pred)
sum(train$default)
summary(train$default)

# Build confusion matrix
ctab.train <- table(default=train$default, pred=train$pred>0.3)  
ctab.train                                                      
true_positive_rate <- ctab.train[2,2]/sum(ctab.train[2,])
true_positive_rate
true_negative_rate <- ctab.train[1,1]/sum(ctab.train[1,])
true_negative_rate

ctab.test <- table(default=test$default, pred=test$pred>0.3)  

ctab.test                                                      
true_positive_rate <- ctab.test[2,2]/sum(ctab.test[2,])
true_positive_rate
true_negative_rate <- ctab.test[1,1]/sum(ctab.test[1,])
true_negative_rate

##########################
### 以下沒有教
##########################
test_pred <- predict(model, newdata=test, type="response")
test_pred_TF <- ifelse(test_pred >= 0.02, 1, 0)
str(test_pred_TF)
CrossTable(test$atRisk, test_pred_TF, 
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('actual', 'predicted'))


train2 <- predict(model, newdata=train)
head(train2)
exp(train2[1:5]) / (1+exp(train2[1:5]))


# example 7.12 of section 7.2.3 
# Making predictions 
# Title: Exploring modeling trade-offs 

#   Load ROCR library. 
library(ROCR)         
# Load grid library for the nplot function below.                              
library(grid)                                      	

# Create ROCR prediction object
predObj <- prediction(train$pred, train$atRisk)    
# Create ROCR object to calculate precision as a function of threshold.  	
precObj <- performance(predObj, measure="prec")  
# Create ROCR object to calculate recall as a function of threshold.   	
recObj <- performance(predObj, measure="rec")      

# at ( @ ) symbol@ (at) symbolROCR objects are what R calls S4 objects; 
# the slots (or fields) of an S4 object are stored as lists within the object. 
# You extract the slots from an S4 object using @ notation. 
precision <- (precObj@y.values)[[1]]   
# The x values (thresholds) are the same in 
# both predObj and recObj, so you only need to extract them once.              	
prec.x <- (precObj@x.values)[[1]]                   	
recall <- (recObj@y.values)[[1]]

# Build data frame with thresholds, precision, and recall. 
rocFrame <- data.frame(threshold=prec.x, precision=precision,
                       recall=recall)               	

# Function to plot multiple plots on one page (stacked). 
nplot <- function(plist) {                          	
  n <- length(plist)
  grid.newpage()
  pushViewport(viewport(layout=grid.layout(n,1)))
  vplayout=function(x,y) {viewport(layout.pos.row=x, layout.pos.col=y)}
  for(i in 1:n) {
    print(plist[[i]], vp=vplayout(i,1))
  }
}

# Calculate rate of at-risk births in the training set. 
pnull <- mean(as.numeric(train$atRisk))             

#   Plot enrichment rate as a function of threshold. 
p1 <- ggplot(rocFrame, aes(x=threshold)) +          	
  geom_line(aes(y=precision/pnull)) +
  coord_cartesian(xlim = c(0,0.05), ylim=c(0,10) )  
 
#  Plot recall as a function of threshold. 
p2 <- ggplot(rocFrame, aes(x=threshold)) +          	
  geom_line(aes(y=recall)) +
  coord_cartesian(xlim = c(0,0.05) )

#   Show both plots simultaneously. 
nplot(list(p1, p2))                                	

# example 7.13 of section 7.2.3 
# Making predictions 
# Title: Evaluating our chosen model 

#   Build confusion matrix. 
ctab.test <- table(pred=test$pred>0.02, atRisk=test$atRisk)  
# Rows contain predicted negatives and positives; 
# columns contain actual negatives and positives.  
ctab.test                                                      
precision <- ctab.test[2,2]/sum(ctab.test[2,])
precision
recall <- ctab.test[2,2]/sum(ctab.test[,2])
recall
enrich <- precision/mean(as.numeric(test$atRisk))
enrich

# example 7.14 of section 7.2.4 
# Finding relations and extracting advice from logistic models 
# Title: The model coefficients 
coefficients(model)

# example 7.15 of section 7.2.5 
# Title: The model summary 
summary(model)

# example 7.16 of section 7.2.5 
# Title: Calculating deviance residuals 

# Create vector of predictions for training data. 
pred <- predict(model, newdata=train, type="response")  
# Function to return the log likelihoods for each data point. 
# Argument y is the true outcome (as a numeric variable, 0/1); 
# argument py is the predicted probability. 
llcomponents <- function(y, py) 
  { y*log(py) + (1-y)*log(1-py) }

# Calculate deviance residuals. 
edev <- sign(as.numeric(train$atRisk) - pred) * 
  sqrt(-2*llcomponents(as.numeric(train$atRisk), pred))

summary(edev)

# example 7.17 of section 7.2.5 
# Title: Computing deviance 

# Function to calculate the log likelihood of a dataset.
# Variable y is the outcome in numeric form (1 for positive examples, 
# 0 for negative). 
# Variable py is the predicted probability that y==1. 
loglikelihood <- function(y, py) {                                  
  sum(y * log(py) + (1-y)*log(1 - py)) }

# Calculate rate of positive examples in dataset. 
pnull <- mean(as.numeric(train$atRisk))            
# Calculate null deviance.                 	 
null.dev <- -2*loglikelihood(as.numeric(train$atRisk), pnull)       

pnull
null.dev
# For training data, the null deviance is stored in the slot model$null.deviance. 
model$null.deviance                                                 

# Predict probabilities for training data. 
pred <- predict(model, newdata=train, type="response") 

summary(pred)
str(pred)
plot(pred)

#   Calculate deviance of model for training data. 
resid.dev <- -2*loglikelihood(as.numeric(train$atRisk), pred)     

resid.dev
# For training data, model deviance is stored in model$deviance. 
model$deviance                                                      

#   Calculate null deviance and residual deviance for test data. 
testy <- as.numeric(test$atRisk)                                   	
testpred <- predict(model, newdata=test,
                    type="response")
pnull.test <- mean(testy)
null.dev.test <- -2*loglikelihood(testy, pnull.test)
resid.dev.test <- -2*loglikelihood(testy, testpred)

pnull.test
null.dev.test
resid.dev.test

# example 7.18 of section 7.2.5 
# Reading the model summary and characterizing coefficients 
# Title: Calculating the significance of the observed fit 

#   Null model has (number of data points - 1) degrees of freedom. 
df.null <- dim(train)[[1]] - 1                              
#  Fitted model has (number of data points - number of coefficients) degrees of freedom. 
df.model <- dim(train)[[1]] - length(model$coefficients)  	

df.null
df.model

# Compute difference in deviances and difference in degrees of freedom. 
delDev <- null.dev - resid.dev                            	
deldf <- df.null - df.model
# Estimate probability of seeing the observed difference in deviances under null model (the 
# p-value) using chi-squared distribution. 
p <- pchisq(delDev, deldf, lower.tail=F)                  	# Note: 4 

delDev
deldf
p

# example 7.19 of section 7.2.5 
# Reading the model summary and characterizing coefficients 
# Title: Calculating the pseudo R-squared 

pr2 <- 1-(resid.dev/null.dev)

print(pr2)
pr2.test <- 1-(resid.dev.test/null.dev.test)
print(pr2.test)


