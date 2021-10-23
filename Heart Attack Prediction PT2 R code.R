library(caret)


data<- read.csv("heart.csv")

#See summary of data
summary(data)

#Convert numeric variables into categoricals
data$sex<-as.factor(data$sex)
data$cp<-as.factor(data$cp)
data$fbs<-as.factor(data$fbs)
data$restecg<-as.factor(data$restecg)
data$exng<-as.factor(data$exng)
data$oldpeak<-as.factor(data$oldpeak)
data$slp<-as.factor(data$slp)
data$caa<-as.factor(data$caa)
data$thall<-as.factor(data$thall)

#Output will be class variable for the following experiment:convert to factor
data$output<-as.factor(data$output)

#Remove unwanted variables
data2<-data[,-c(3,6,7,11,12,13)]
summary(data2)


#Partition the data into training and testing using the hold out method
#First, we need to set the random seed for repeatability
set.seed(1234)
#Create an index variable to perform a 70/30 split 
trainIndex <- createDataPartition(data2$output, p=.7, list=FALSE, times = 1)
data2_train <- data2[trainIndex,]
data2_test <- data2[-trainIndex,]



#Logistic Regression set the training control using repeated 10-fold cross validation with 5 repeats
trControl <- trainControl(method = 'repeatedcv',
                          number = 10,
                          repeats =  5)

logitFit <- train(output ~ ., data = data2_train, 
                  method = 'glm',
                  trControl = trControl,
                  family = 'binomial' )

summary(logitFit)

logitPredClass <- predict(logitFit,data2_test)
logitPredProbs <- predict(logitFit,data2_test,'prob')

#Now evaluate the classifier using the confusionMatrix() function
confusionMatrix(logitPredClass, data2_test$output, mode="everything")



#Support Vector Machine set the training control using repeated 10-fold cross validation with 5 repeats
trControl <- trainControl(method = 'repeatedcv',
                          number = 10,
                          repeats =  5)

svmFit <- train(output ~ ., data = data2_train, 
                method = 'svmRadial',
                trControl = trControl,
                preProcess = c("center","scale"))



svmPredict <- predict(svmFit,data2_test)

#Now evaluate the classifier using the confusionMatrix() function
confusionMatrix(svmPredict, data2_test$output, mode="everything")




#### Neural Networks
#Use Cross Validation to optimize network parameters
trControl <- trainControl(method = 'cv', number = 10)

nnetFit <- train(output ~ ., data = data2_train, method = 'nnet', preProcess = c("center","scale"), trControl = trControl)

#Examine the result of the cross validation
plot(nnetFit)

nnetPredClass <- predict(nnetFit,data2_test)

#Now evaluate the classifier using the confusionMatrix() function
confusionMatrix(nnetPredClass, data2_test$output, mode="everything")

#Plot the neural network with the NeuralNetTools package
library(NeuralNetTools)
plotnet(nnetFit, alpha = 0.6)







