#libraries
library(data.table)
library(rpart)
library(rpart.plot)
library(ROCR)
library(ggplot2)
library(glmnet)
library(crossval)

set.seed(123)

#load data
devel1 = as.data.frame(fread("devel1.csv"))
devel2 = as.data.frame(fread("devel2.csv"))
test.blind = as.data.frame(fread("test.blind.csv"))
#factorize
devel1$active = as.factor(devel1$active)
devel2$active = as.factor(devel2$active)
devel = rbind(devel1, devel2)
target.values = devel$active
devel$active = NULL

####################################
# 1a)
####################################

table(devel1$active)
table(devel2$active)

####################################
# 1b)
####################################

#find discrete features
apply(devel, 2, function(x) length(unique(x)))
continuous.features = c(1,2,3,6,7,8,9,10,11,12,13,15,16,17,18,35)
discrete = devel[,-continuous.features]

####################################
# 1c)
####################################

#remove constant features
constant.names = names(discrete[,sapply(discrete, function(x) 1==length(unique(x)))])
discrete.nonconstant = discrete[, -which(names(discrete) %in% constant.names)]

####################################
# 1d)
####################################

#create table and barplot with numbers of values
counts = sapply(discrete, function(x) length(unique(x)))
tbl = table(counts)
barplot(counts, ylab="number of values") #TODO DODELAT
barplot(tbl, ylim=c(0,30), cex.names = 0.8, cex.axis=0.8, ylab="Number of values", xlab="Number of features")
tbl

####################################
# 1e)
####################################

binary.names = names(discrete[,sapply(discrete, function(x) 2==length(unique(x)))])
binary = discrete[, which(names(discrete) %in% binary.names)]
ftr = 4
fr = function(x) { return(length(x[x > 0]))}
under.threshold.names = names(which(sapply(binary,function(x) ftr>min(fr(x), nrow(binary) - fr(x))) == TRUE))
discrete.filtered = discrete.nonconstant[, -which(names(discrete.nonconstant) %in% under.threshold.names)]

####################################
# filter features
####################################

devel1.filtered = devel1
devel1.filtered = devel1.filtered[, -which(names(devel1.filtered) %in% constant.names)]
devel1.filtered = devel1.filtered[, -which(names(devel1.filtered) %in% under.threshold.names)]
devel2.filtered = devel2
devel2.filtered = devel2.filtered[, -which(names(devel2.filtered) %in% constant.names)]
devel2.filtered = devel2.filtered[, -which(names(devel2.filtered) %in% under.threshold.names)]
test.blind.filtered = test.blind
test.blind.filtered = test.blind.filtered[, -which(names(test.blind.filtered) %in% constant.names)]
test.blind.filtered = test.blind.filtered[, -which(names(test.blind.filtered) %in% under.threshold.names)]

####################################
# 1f)
####################################

#functions for computing mutual information
entropy = function(x){
       p <- table(x) / NROW(x)
       return( -sum(p * log2(p)) )
  }
entropy.cond <- function(x, y){
     N <- NROW(x)
     p.y <- table(y) / N 
     p.joint <- as.vector(table(y, x)) / N 
     p.cond <- p.joint / rep(p.y, NROW(table(x))) 
     H.cond <- - sum( p.joint[p.joint > 0] * log2(p.cond[p.cond > 0]) ) 
     return( H.cond )
 }
mutual.information = function(x, y) {
     return(entropy(x)-entropy.cond(x,y))
}

# compute and plot information gain
information.gain = sort(sapply(discrete.filtered, function(x) mutual.information(x,target.values)), decreasing = TRUE)
op = par(mar=c(14,4,4,2))
plot(information.gain, ylab="Information gain", main="Information gain of discrete features", xlab="", xaxt="n")
axis(1, at=1:92, labels=names(discrete.filtered), las=2, cex=0.3)
rm(op)

####################################
# 2b)
####################################

# divide to positive and negative examples
positives = devel1.filtered[devel1.filtered$active == 1,]
negatives = devel1.filtered[devel1.filtered$active == 0,]

# get random indeces for positive and negative examples
sample.positive = sample(1:nrow(positives))
zeros = 10 - nrow(positives) %% 10
if (zeros < 10) sample.positive = c(sample.positive, rep(0, zeros))
index.positive = matrix(data=sample.positive, nrow=10)

sample.negative = sample(1:nrow(negatives))
zeros = 10 - nrow(negatives) %% 10
if (zeros < 10) sample.negative = c(sample.negt.tesative, rep(0, zeros))
index.negative = matrix(data=sample.negative, nrow=10)

#do 10-fold cross-validation
auc.results = numeric(0)
for(i in 1:10) {
  #prepare test and train sets
  crossval.train.positive = positives[ - index.positive[i,][index.positive[i,] > 0], ]
  crossval.test.positive  = positives[ index.positive[i,][index.positive[i,] > 0], ]
  crossval.train.negative = negatives[ - index.negative[i,][index.negative[i,] > 0], ]
  crossval.test.negative  = negatives[ index.negative[i,][index.negative[i,] > 0], ]
  crossval.train = rbind(crossval.train.positive, crossval.train.negative)
  crossval.test = rbind(crossval.test.positive, crossval.test.negative)
  #build decision tree
  dectree.model = rpart(active ~ ., data=crossval.train)
  #get predictions
  dectree.pred.prob = predict(dectree.model, crossval.test[,-109], type="prob")[,2]
  dectree.predict = prediction(dectree.pred.prob, crossval.test[,109])
  #compute auc
  dectree.auc.perf = performance(dectree.predict, measure="auc", fpr.stop=0.1)
  dectree.auc = round(dectree.auc.perf@y.values[[1]], 3)
  #add auc value to the vector
  auc.results = c(auc.results, dectree.auc)
}
#report mean, standard deviation and confidence interval
mean(auc.results)
sd(auc.results)
test = t.test(auc.results, conf.level=0.95)
test$conf.int[1]
test$conf.int[2]

####################################
# test default model on D2

dectree.model = rpart(active ~ ., data=devel1.filtered)
dectree.pred.prob = predict(dectree.model, devel2.filtered[,-109], type="prob")[,2]
dectree.predict = prediction(dectree.pred.prob, devel2.filtered[,109])
dectree.auc.perf = performance(dectree.predict, measure="auc", fpr.stop=0.1)
dectree.auc = dectree.auc.perf@y.values[[1]]
dectree.auc

####################################
# 2c)
####################################

#cp.values = c(0.1, 0.075, 0.05, 0.025, 0.01, 0.001, 0.0001)
cp.values = seq(from=0.23, to=0.00001, length.out=50)
dectree.results = data.frame(numeric(0),
                     numeric(0),
                     numeric(0),
                     numeric(0),
                     numeric(0),
                     numeric(0))
dectree.results.names = c("cp", "mean", "standard.deviation", "standard.error", "conf.interval.left", "conf.interval.right")
names(dectree.results)=dectree.results.names
for (cp.val in cp.values) {
  dectree.auc.results = numeric(0)
  for(i in 1:10) {
    crossval.train.positive = positives[ - index.positive[i,][index.positive[i,] > 0], ]
    crossval.test.positive  = positives[ index.positive[i,][index.positive[i,] > 0], ]
    crossval.train.negative = negatives[ - index.negative[i,][index.negative[i,] > 0], ]
    crossval.test.negative  = negatives[ index.negative[i,][index.negative[i,] > 0], ]
    crossval.train = rbind(crossval.train.positive, crossval.train.negative)
    crossval.test = rbind(crossval.test.positive, crossval.test.negative)
    #build decision tree
    dectree.model = rpart(active ~ ., data=crossval.train, cp=cp.val)
    #do prediction
    dectree.pred.prob = predict(dectree.model, crossval.test[,-109], type="prob")[,2]
    dectree.predict = prediction(dectree.pred.prob, crossval.test[,109])
    #compute auc
    dectree.auc.perf = performance(dectree.predict, measure="auc", fpr.stop=0.1)
    dectree.auc = round(dectree.auc.perf@y.values[[1]], 3)
    dectree.auc.results = c(dectree.auc.results, dectree.auc)
  }
  test = t.test(dectree.auc.results, conf.level=0.95)
  se = sd(dectree.auc.results)/sqrt(length(dectree.auc.results))
  cp.results = data.frame(cp.val, mean(dectree.auc.results), sd(dectree.auc.results), se, test$conf.int[1], test$conf.int[2])
  names(cp.results)=dectree.results.names
  dectree.results=rbind(dectree.results, cp.results)
}

qplot(dectree.results$cp, dectree.results$mean,  xlab = "CP", ylab = "Mean of AUC0.1", ylim = c(0, 0.1), xlim = rev(range(dectree.results$cp)))+geom_errorbar(aes(x=dectree.results$cp, ymin=dectree.results$mean-dectree.results$standard.error, ymax=dectree.results$mean+dectree.results$standard.error), width=0.25)+theme(axis.text=element_text(size=12), axis.title=element_text(size=15))

####################################
# 2d)
####################################
best.cp = 0.023
dectree.model = rpart(active ~ ., data=devel1.filtered, cp=best.cp)
dectree.pred.prob = predict(dectree.model, devel2.filtered[,-109], type="prob")[,2]
dectree.predict = prediction(dectree.pred.prob, devel2.filtered[,109])
dectree.auc.perf = performance(dectree.predict, measure="auc", fpr.stop=0.1)
dectree.auc = dectree.auc.perf@y.values[[1]]
print("cp = 0.023 \n")
dectree.auc

best.cp = 0.014
dectree.model = rpart(active ~ ., data=devel1.filtered, cp=best.cp)
dectree.pred.prob = predict(dectree.model, devel2.filtered[,-109], type="prob")[,2]
dectree.predict = prediction(dectree.pred.prob, devel2.filtered[,109])
dectree.auc.perf = performance(dectree.predict, measure="auc", fpr.stop=0.1)
dectree.auc = dectree.auc.perf@y.values[[1]]
print("cp = 0.014 \n")
dectree.auc

best.cp = 0.05
dectree.model = rpart(active ~ ., data=devel1.filtered, cp=best.cp)
dectree.pred.prob = predict(dectree.model, devel2.filtered[,-109], type="prob")[,2]
dectree.predict = prediction(dectree.pred.prob, devel2.filtered[,109])
dectree.auc.perf = performance(dectree.predict, measure="auc", fpr.stop=0.1)
dectree.auc = dectree.auc.perf@y.values[[1]]
print("cp = 0.05 \n")
dectree.auc


####################################
# 3) model without regularization - 10-fold cross-validation
####################################

#prepare samples for 10-fold cross validation
sample.positive = sample(1:nrow(positives))
zeros = 10 - nrow(positives) %% 10
if (zeros < 10) sample.positive = c(sample.positive, rep(0, zeros))
index.positive = matrix(data=sample.positive, nrow=10)
sample.negative = sample(1:nrow(negatives))
zeros = 10 - nrow(negatives) %% 10
if (zeros < 10) sample.negative = c(sample.negative, rep(0, zeros))
index.negative = matrix(data=sample.negative, nrow=10)

#do 10-fold cross-validation
logR.auc.results = numeric(0)
for(i in 1:10) {
  #prepare test and train sets
  crossval.train.positive = positives[ - index.positive[i,][index.positive[i,] > 0], ]
  crossval.test.positive  = positives[ index.positive[i,][index.positive[i,] > 0], ]
  crossval.train.negative = negatives[ - index.negative[i,][index.negative[i,] > 0], ]
  crossval.test.negative  = negatives[ index.negative[i,][index.negative[i,] > 0], ]
  crossval.train = rbind(crossval.train.positive, crossval.train.negative)
  crossval.test = rbind(crossval.test.positive, crossval.test.negative)
  #build the model
  logR.model = glm(formula = active ~ .,
                 family = binomial,
                 data = crossval.train)
  #do predictions
  logR.pred.prob = predict(logR.model,
                       crossval.test[,-109],
                       type = "response")
  logR.predict = prediction(logR.pred.prob, crossval.test[,109])
  #compute auc
  logR.auc.perf = performance(logR.predict, measure="auc", fpr.stop=0.1)
  logR.auc = round(logR.auc.perf@y.values[[1]], 3)
  logR.auc.results = c(auc.results, logR.auc)
}

mean(logR.auc.results)
sd(logR.auc.results)
test = t.test(logR.auc.results, conf.level=0.95)
test$conf.int

#####################################
# 3) model without regularization trained on D1 and tested on D2
#####################################

#build the model
logR.model = glm(formula = active ~ .,
                 family = binomial,
                 data = devel1.filtered)
#do predictions
logR.pred.prob = predict(logR.model,
                         devel2.filtered[,-109],
                         type = "response")
logR.predict = prediction(logR.pred.prob, devel2.filtered[,109])
#compute auc
logR.auc.perf = performance(logR.predict, measure="auc", fpr.stop=0.1)
logR.auc = round(logR.auc.perf@y.values[[1]], 3)
logR.auc

###############################################
#3) logistic regresion with elastic regularization
###############################################

alpha.values = seq(0, 1, 0.2)
logR.elastic.results = data.frame(numeric(0),
                     numeric(0),
                     numeric(0),
                     numeric(0),
                     numeric(0))

results.names = c("alpha", "mean", "standard.deviation", "conf.interval.left", "conf.interval.right")
names(logR.elastic.results)=results.names
for (alpha.val in alpha.values) {
    auc.results = numeric(0)
    for(i in 1:10) {
      cat(".") #TODO smazat
      crossval.train.positive = positives[ - index.positive[i,][index.positive[i,] > 0], ]
      crossval.test.positive  = positives[ index.positive[i,][index.positive[i,] > 0], ]
      crossval.train.negative = negatives[ - index.negative[i,][index.negative[i,] > 0], ]
      crossval.test.negative  = negatives[ index.negative[i,][index.negative[i,] > 0], ]
      crossval.train = rbind(crossval.train.positive, crossval.train.negative)
      crossval.test = rbind(crossval.test.positive, crossval.test.negative)
      #build the model
      x = model.matrix(active ~ ., data = crossval.train)
      y = data.matrix(crossval.train$active)
      
      x.test = model.matrix(active ~., data=crossval.test)
      y.test = data.matrix(crossval.test$active)
      
      logR.elastic.model = glmnet(x, y, family = "binomial", alpha = alpha.val)

      # predictions, compute auc for the best lambda
      auc.best = 0
      for (lambda.val in logR.elastic.model$lambda) {
        logR.elastic.pred.prob = predict(logR.elastic.model,
                                         type = "response",
                                         newx = x.test,
                                         s = lambda.val)
        logR.elastic.predict = prediction(logR.elastic.pred.prob, crossval.test[,109])
        #compute auc
        logR.elastic.auc.perf = performance(logR.elastic.predict, measure="auc", fpr.stop=0.1)
        logR.elastic.auc = round(logR.elastic.auc.perf@y.values[[1]], 3)
        if (logR.elastic.auc > auc.best) {
          auc.best = logR.elastic.auc
        }
      }
      auc.results = c(auc.results, auc.best)
    }
    test = t.test(auc.results, conf.level=0.95)
    alpha.results = data.frame(alpha.val, mean(auc.results), sd(auc.results), test$conf.int[1], test$conf.int[2])
    names(alpha.results)=results.names
    logR.elastic.results=rbind(logR.elastic.results, alpha.results)
}

#####################################
# 3) models with regularization trained on D1 and tested on D2
#####################################
logR.elastic.test.results = data.frame(numeric(0),
                                  numeric(0),
                                  numeric(0))

results.names = c("alpha", "AUC0.1", "lambda")
names(logR.elastic.results)=results.names
for (alpha.val in alpha.values) {
  x = model.matrix(active ~ ., data = devel1.filtered)
  y = data.matrix(devel1.filtered$active)
  x.test = model.matrix(active ~., data=devel2.filtered)
  y.test = data.matrix(devel2.filtered$active)
  
  logR.elastic.model = glmnet(x, y, family = "binomial", alpha = alpha.val)
  
  # predictions
  auc.best = 0
  lambda.best = 0
  for (lambda.val in logR.elastic.model$lambda) {
    logR.elastic.pred.prob = predict(logR.elastic.model,
                                     type = "response",
                                     newx = x.test,
                                     s = lambda.val)
    logR.elastic.predict = prediction(logR.elastic.pred.prob, y.test)
    #compute auc
    logR.elastic.auc.perf = performance(logR.elastic.predict, measure="auc", fpr.stop=0.1)
    logR.elastic.auc = round(logR.elastic.auc.perf@y.values[[1]], 3)
    if (logR.elastic.auc > auc.best) {
      auc.best = logR.elastic.auc
      lambda.best = lambda.val
    }
  }
  
  alpha.results.test = data.frame(alpha.val, auc.best, lambda.best)
  names(alpha.results.test)=results.names
  logR.elastic.test.results=rbind(logR.elastic.test.results, alpha.results.test)
}

# plot the results
train.results = logR.elastic.results[, c(1,2)]
test.results = logR.elastic.test.results[, c(1,2)]
plot(train.results, ylim = c(0.04, 0.11), xlab = "alpha", ylab = "Mean of AUC0.1", pch=19)
points(test.results, col="red", pch=19)

####################################
# 4) using different data sets - 5-fold cross-validation
####################################

####################################
# 4a) training - 4/5 D1, test - 1/5 D1
####################################

data = devel1.filtered
# divide to positive and negative examples
positives = data[data$active == 1,]
negatives = data[data$active == 0,]

sample.positive = sample(1:nrow(positives))
zeros = 5 - nrow(positives) %% 5
if (zeros < 5) sample.positive = c(sample.positive, rep(0, zeros))
index.positive = matrix(data=sample.positive, nrow=5)
sample.negative = sample(1:nrow(negatives))
zeros = 5 - nrow(negatives) %% 5
if (zeros < 5) sample.negative = c(sample.negative, rep(0, zeros))
index.negative = matrix(data=sample.negative, nrow=5)

auc.results = numeric(0)
for(i in 1:5) {
    cat(".") #TODO smazat
    crossval.train.positive = positives[ - index.positive[i,][index.positive[i,] > 0], ]
    crossval.test.positive  = positives[ index.positive[i,][index.positive[i,] > 0], ]
    crossval.train.negative = negatives[ - index.negative[i,][index.negative[i,] > 0], ]
    crossval.test.negative  = negatives[ index.negative[i,][index.negative[i,] > 0], ]
    crossval.train = rbind(crossval.train.positive, crossval.train.negative)
    crossval.test = rbind(crossval.test.positive, crossval.test.negative)
    #build the model
    x = model.matrix(active ~ ., data = crossval.train)
    y = data.matrix(crossval.train$active)
    
    x.test = model.matrix(active ~., data=crossval.test)
    y.test = data.matrix(crossval.test$active)
    
    logR.elastic.model = glmnet(x, y, family = "binomial", alpha = best.alpha)
    
    # predictions
    auc.best = 0
    for (lambda.val in logR.elastic.model$lambda) {
      logR.elastic.pred.prob = predict(logR.elastic.model,
                                       type = "response",
                                       newx = x.test,
                                       s = lambda.val)
      logR.elastic.predict = prediction(logR.elastic.pred.prob, crossval.test[,109])
      #compute auc
      logR.elastic.auc.perf = performance(logR.elastic.predict, measure="auc", fpr.stop=0.1)
      logR.elastic.auc = round(logR.elastic.auc.perf@y.values[[1]], 3)
      if (logR.elastic.auc > auc.best) {
        auc.best = logR.elastic.auc
        
      }
    }
    auc.results = c(auc.results, auc.best)
}
print("Results of 4a) \n")
mean(auc.results)
sd(auc.results)
test = t.test(auc.results, conf.level=0.95)
test$conf.int

####################################
# 4a) test on D2

x = model.matrix(active ~ ., data = devel1.filtered)
y = data.matrix(devel1.filtered$active)

x.test = model.matrix(active ~., data=devel2.filtered)
y.test = data.matrix(devel2.filtered$active)

logR.elastic.model = glmnet(x, y, family = "binomial", alpha = best.alpha)

# predictions
auc.best = 0
for (lambda.val in logR.elastic.model$lambda) {
  logR.elastic.pred.prob = predict(logR.elastic.model,
                                   type = "response",
                                   newx = x.test,
                                   s = lambda.val)
  logR.elastic.predict = prediction(logR.elastic.pred.prob, y.test)
  #compute auc
  logR.elastic.auc.perf = performance(logR.elastic.predict, measure="auc", fpr.stop=0.1)
  logR.elastic.auc = round(logR.elastic.auc.perf@y.values[[1]], 3)
  if (logR.elastic.auc > auc.best) {
    auc.best = logR.elastic.auc
  }
}
print("4a) tested on D2 \n")
auc.best

####################################
# 4b) training - D1 + 1/5 D2, test - 4/5 D2
####################################

data = devel2.filtered
# divide to positive and negative examples
positives = data[data$active == 1,]
negatives = data[data$active == 0,]

sample.positive = sample(1:nrow(positives))
zeros = 5 - nrow(positives) %% 5
if (zeros < 5) sample.positive = c(sample.positive, rep(0, zeros))
index.positive = matrix(data=sample.positive, nrow=5)
sample.negative = sample(1:nrow(negatives))
zeros = 5 - nrow(negatives) %% 5
if (zeros < 5) sample.negative = c(sample.negative, rep(0, zeros))
index.negative = matrix(data=sample.negative, nrow=5)

auc.results = numeric(0)
for(i in 1:5) {
  cat(".") #TODO smazat
  crossval.test.positive = positives[ - index.positive[i,][index.positive[i,] > 0], ]
  crossval.train.positive  = positives[ index.positive[i,][index.positive[i,] > 0], ]
  crossval.test.negative = negatives[ - index.negative[i,][index.negative[i,] > 0], ]
  crossval.train.negative  = negatives[ index.negative[i,][index.negative[i,] > 0], ]
  crossval.train = rbind(crossval.train.positive, crossval.train.negative, devel1.filtered)
  crossval.test = rbind(crossval.test.positive, crossval.test.negative)
  #build the model
  x = model.matrix(active ~ ., data = crossval.train)
  y = data.matrix(crossval.train$active)
  
  x.test = model.matrix(active ~., data=crossval.test)
  y.test = data.matrix(crossval.test$active)
  
  logR.elastic.model = glmnet(x, y, family = "binomial", alpha = best.alpha)
  
  # predictions
  auc.best = 0
  for (lambda.val in logR.elastic.model$lambda) {
    logR.elastic.pred.prob = predict(logR.elastic.model,
                                     type = "response",
                                     newx = x.test,
                                     s = lambda.val)
    logR.elastic.predict = prediction(logR.elastic.pred.prob, crossval.test[,109])
    #compute auc
    logR.elastic.auc.perf = performance(logR.elastic.predict, measure="auc", fpr.stop=0.1)
    logR.elastic.auc = round(logR.elastic.auc.perf@y.values[[1]], 3)
    if (logR.elastic.auc > auc.best) {
      auc.best = logR.elastic.auc
    }
  }
  auc.results = c(auc.results, auc.best)
}
print("Results of 4b) \n")
mean(auc.results)
sd(auc.results)
test = t.test(auc.results, conf.level=0.95)
test$conf.int
  
  
####################################
# 4c) training - D1 + 4/5 D2, test - 1/5 D2
####################################

data = devel2.filtered
# divide to positive and negative examples
positives = data[data$active == 1,]
negatives = data[data$active == 0,]

sample.positive = sample(1:nrow(positives))
zeros = 5 - nrow(positives) %% 5
if (zeros < 5) sample.positive = c(sample.positive, rep(0, zeros))
index.positive = matrix(data=sample.positive, nrow=5)
sample.negative = sample(1:nrow(negatives))
zeros = 5 - nrow(negatives) %% 5
if (zeros < 5) sample.negative = c(sample.negative, rep(0, zeros))
index.negative = matrix(data=sample.negative, nrow=5)

auc.results = numeric(0)
for(i in 1:5) {
  cat(".") #TODO smazat
  crossval.train.positive = positives[ - index.positive[i,][index.positive[i,] > 0], ]
  crossval.test.positive  = positives[ index.positive[i,][index.positive[i,] > 0], ]
  crossval.train.negative = negatives[ - index.negative[i,][index.negative[i,] > 0], ]
  crossval.test.negative  = negatives[ index.negative[i,][index.negative[i,] > 0], ]
  crossval.train = rbind(crossval.train.positive, crossval.train.negative, devel1.filtered)
  crossval.test = rbind(crossval.test.positive, crossval.test.negative)
  #build the model
  x = model.matrix(active ~ ., data = crossval.train)
  y = data.matrix(crossval.train$active)
  
  x.test = model.matrix(active ~., data=crossval.test)
  y.test = data.matrix(crossval.test$active)
  
  logR.elastic.model = glmnet(x, y, family = "binomial", alpha = best.alpha)
  
  # predictions
  auc.best = 0
  for (lambda.val in logR.elastic.model$lambda) {
    logR.elastic.pred.prob = predict(logR.elastic.model,
                                     type = "response",
                                     newx = x.test,
                                     s = lambda.val)
    logR.elastic.predict = prediction(logR.elastic.pred.prob, crossval.test[,109])
    #compute auc
    logR.elastic.auc.perf = performance(logR.elastic.predict, measure="auc", fpr.stop=0.1)
    logR.elastic.auc = round(logR.elastic.auc.perf@y.values[[1]], 3)
    if (logR.elastic.auc > auc.best) {
      auc.best = logR.elastic.auc
    }
  }
  auc.results = c(auc.results, auc.best)
}
print("Results of 4c) \n")
mean(auc.results)
sd(auc.results)
test = t.test(auc.results, conf.level=0.95)
test$conf.int


####################################
# 4d) training - 4/5 D2, test - 1/5 D2
####################################

data = devel2.filtered
# divide to positive and negative examples
positives = data[data$active == 1,]
negatives = data[data$active == 0,]

sample.positive = sample(1:nrow(positives))
zeros = 5 - nrow(positives) %% 5
if (zeros < 5) sample.positive = c(sample.positive, rep(0, zeros))
index.positive = matrix(data=sample.positive, nrow=5)
sample.negative = sample(1:nrow(negatives))
zeros = 5 - nrow(negatives) %% 5
if (zeros < 5) sample.negative = c(sample.negative, rep(0, zeros))
index.negative = matrix(data=sample.negative, nrow=5)

auc.results = numeric(0)
for(i in 1:5) {
  cat(".") #TODO smazat
  crossval.train.positive = positives[ - index.positive[i,][index.positive[i,] > 0], ]
  crossval.test.positive  = positives[ index.positive[i,][index.positive[i,] > 0], ]
  crossval.train.negative = negatives[ - index.negative[i,][index.negative[i,] > 0], ]
  crossval.test.negative  = negatives[ index.negative[i,][index.negative[i,] > 0], ]
  crossval.train = rbind(crossval.train.positive, crossval.train.negative)
  crossval.test = rbind(crossval.test.positive, crossval.test.negative)
  #build the model
  x = model.matrix(active ~ ., data = crossval.train)
  y = data.matrix(crossval.train$active)
  
  x.test = model.matrix(active ~., data=crossval.test)
  y.test = data.matrix(crossval.test$active)
  
  logR.elastic.model = glmnet(x, y, family = "binomial", alpha = best.alpha)
  
  # predictions
  auc.best = 0
  for (lambda.val in logR.elastic.model$lambda) {
    logR.elastic.pred.prob = predict(logR.elastic.model,
                                     type = "response",
                                     newx = x.test,
                                     s = lambda.val)
    logR.elastic.predict = prediction(logR.elastic.pred.prob, crossval.test[,109])
    #compute auc
    logR.elastic.auc.perf = performance(logR.elastic.predict, measure="auc", fpr.stop=0.1)
    logR.elastic.auc = round(logR.elastic.auc.perf@y.values[[1]], 3)
    if (logR.elastic.auc > auc.best) {
      auc.best = logR.elastic.auc
    }
  }
  auc.results = c(auc.results, auc.best)
}
print("Results of 4d) \n")
mean(auc.results)
sd(auc.results)
test = t.test(auc.results, conf.level=0.95)
test$conf.int

############################################
# 5) final model evaluation
############################################
# divide D2 to positive and negative examples
positives = devel2.filtered[devel2.filtered$active == 1,]
negatives = devel2.filtered[devel2.filtered$active == 0,]

#make samples for 5-fold cross-validation
sample.positive = sample(1:nrow(positives))
zeros = 5 - nrow(positives) %% 5
if (zeros < 5) sample.positive = c(sample.positive, rep(0, zeros))
index.positive = matrix(data=sample.positive, nrow=5)
sample.negative = sample(1:nrow(negatives))
zeros = 5 - nrow(negatives) %% 5
if (zeros < 5) sample.negative = c(sample.negative, rep(0, zeros))
index.negative = matrix(data=sample.negative, nrow=5)


###############################################
# logistic regresion with elastic regularization

alpha.values = seq(0, 1, 0.2)
logR.elastic.results = data.frame(numeric(0),
                                  numeric(0),
                                  numeric(0),
                                  numeric(0),
                                  numeric(0))

results.names = c("alpha", "mean", "standard.deviation", "conf.interval.left", "conf.interval.right")
names(logR.elastic.results)=results.names
for (alpha.val in alpha.values) {
  auc.results = numeric(0)
  for(i in 1:5) {
    cat(".") #TODO smazat
    crossval.train.positive = positives[ - index.positive[i,][index.positive[i,] > 0], ]
    crossval.test.positive  = positives[ index.positive[i,][index.positive[i,] > 0], ]
    crossval.train.negative = negatives[ - index.negative[i,][index.negative[i,] > 0], ]
    crossval.test.negative  = negatives[ index.negative[i,][index.negative[i,] > 0], ]
    crossval.train = rbind(crossval.train.positive, crossval.train.negative)
    crossval.test = rbind(crossval.test.positive, crossval.test.negative)

    #build the model
    x = model.matrix(active ~ ., data = crossval.train)
    y = data.matrix(crossval.train$active)
    
    x.test = model.matrix(active ~., data=crossval.test)
    y.test = data.matrix(crossval.test$active)
    
    logR.elastic.model = glmnet(x, y, family = "binomial", alpha = alpha.val)
    
    # predictions, compute auc for the best lambda
    auc.best = 0
    for (lambda.val in logR.elastic.model$lambda) {
      logR.elastic.pred.prob = predict(logR.elastic.model,
                                       type = "response",
                                       newx = x.test,
                                       s = lambda.val)
      logR.elastic.predict = prediction(logR.elastic.pred.prob, crossval.test[,109])
      #compute auc
      logR.elastic.auc.perf = performance(logR.elastic.predict, measure="auc", fpr.stop=0.1)
      logR.elastic.auc = round(logR.elastic.auc.perf@y.values[[1]], 3)
      if (logR.elastic.auc > auc.best) {
        auc.best = logR.elastic.auc
      }
    }
    auc.results = c(auc.results, auc.best)
  }
  test = t.test(auc.results, conf.level=0.95)
  alpha.results = data.frame(alpha.val, mean(auc.results), sd(auc.results), test$conf.int[1], test$conf.int[2])
  names(alpha.results)=results.names
  logR.elastic.results=rbind(logR.elastic.results, alpha.results)
}

#####################################
# models with regularization trained on D2 and tested on D1
logR.elastic.test.results = data.frame(numeric(0),
                                       numeric(0),
                                       numeric(0))

results.names = c("alpha", "AUC0.1", "lambda")
names(logR.elastic.results)=results.names
for (alpha.val in alpha.values) {
  x = model.matrix(active ~ ., data = devel2.filtered)
  y = data.matrix(devel2.filtered$active)
  x.test = model.matrix(active ~., data=devel1.filtered)
  y.test = data.matrix(devel1.filtered$active)
  
  logR.elastic.model = glmnet(x, y, family = "binomial", alpha = alpha.val)
  
  # predictions
  auc.best = 0
  lambda.best = 0
  for (lambda.val in logR.elastic.model$lambda) {
    logR.elastic.pred.prob = predict(logR.elastic.model,
                                     type = "response",
                                     newx = x.test,
                                     s = lambda.val)
    logR.elastic.predict = prediction(logR.elastic.pred.prob, y.test)
    #compute auc
    logR.elastic.auc.perf = performance(logR.elastic.predict, measure="auc", fpr.stop=0.1)
    logR.elastic.auc = round(logR.elastic.auc.perf@y.values[[1]], 3)
    if (logR.elastic.auc > auc.best) {
      auc.best = logR.elastic.auc
      lambda.best = lambda.val
    }
  }
  
  alpha.results.test = data.frame(alpha.val, auc.best, lambda.best)
  names(alpha.results.test)=results.names
  logR.elastic.test.results=rbind(logR.elastic.test.results, alpha.results.test)
}

# plot the results
train.results = logR.elastic.results[, c(1,2)]
test.results = logR.elastic.test.results[, c(1,2)]
plot(train.results, ylim = c(0.08, 0.1), xlab = "alpha", ylab = "Mean of AUC0.1", pch=19)
points(test.results, col="red", pch=19)

###################################################
# 5) Build the final model - LogR with alpha = 1
###################################################

final.alpha = 1

x = model.matrix(active ~ ., data = devel2.filtered)
y = data.matrix(devel2.filtered$active)
x.test = model.matrix(active ~., data=devel1.filtered)
y.test = data.matrix(devel1.filtered$active)

final.model = glmnet(x, y, family = "binomial", alpha = final.alpha)

auc.best = 0
lambda.best = 0
for (lambda.val in final.model$lambda) {
  final.pred.prob = predict(final.model,
                                   type = "response",
                                   newx = x.test,
                                   s = lambda.val)
  final.predict = prediction(final.pred.prob, y.test)
  #compute auc
  final.auc.perf = performance(final.predict, measure="auc", fpr.stop=0.1)
  final.auc = round(final.auc.perf@y.values[[1]], 3)
  if (final.auc > auc.best) {
    auc.best = final.auc
    lambda.best = lambda.val
  }
}

final.lambda = lambda.best


###################################################
# 5c) Estimate of the precision
###################################################

set.seed(123)
positives = devel1.filtered[devel1.filtered$active == 1,]
negatives = devel1.filtered[devel1.filtered$active == 0,]

proportion = nrow(positives) / nrow(negatives)

subset.positives.length = round(nrow(test.blind.filtered) * proportion)
subset.negatives.length = nrow(test.blind.filtered) - subset.positives.length

precision.50 = numeric(0)
precision.150 = numeric(0)
precision.250 = numeric(0)

for (i in c(1:5000)) {
    #get random sample with right proportion
    s = sample(nrow(positives))
    subset.positives = positives[s[1:subset.positives.length], ]
    s = sample(nrow(negatives))
    subset.negatives = negatives[s[1:subset.negatives.length], ]
    subset = rbind(subset.positives, subset.negatives)
    
    data.blind = subset[, -109]
    data.labels = subset[, 109]
    final.test = model.matrix(~., data=data.blind)
    
    final.pred.prob = predict(final.model,
                              type = "response",
                              newx = final.test,
                              s = final.lambda)
    prob.sorted = sort(final.pred.prob, decreasing = TRUE)
    positives.count = table(final.pred.prob)[2]
    test.prediction.50 = ifelse(final.pred.prob >= prob.sorted[50], 1, 0)
    test.prediction.150 = ifelse(final.pred.prob >= prob.sorted[150], 1, 0)
    test.prediction.250 = ifelse(final.pred.prob >= prob.sorted[250], 1, 0)
    
    cm = confusionMatrix(data.labels, test.prediction.50, negative = 0)
    precision.50 = c(precision.50, (cm[2] / (cm[1] + cm[2])))
  
    cm = confusionMatrix(data.labels, test.prediction.150, negative = 0)
    precision.150 = c(precision.150, (cm[2] / (cm[1] + cm[2])))
    
    cm = confusionMatrix(data.labels, test.prediction.250, negative = 0)
    precision.250 = c(precision.250, (cm[2] / (cm[1] + cm[2])))
    
}

print("precision 50: \n")
mean(precision.50)
sd(precision.50)
print("precision 150: \n")
mean(precision.150)
sd(precision.150)
print("precision 250: \n")
mean(precision.250)
sd(precision.250)

##########################################
## FINALLY - MAKE PREDICTIONS
##########################################

final.test = model.matrix(~., data=test.blind.filtered)

# get predictions with the best lambda
final.pred.prob = predict(final.model,
                          type = "response",
                          newx = final.test,
                          s = final.lambda)
prob.sorted = sort(final.pred.prob, decreasing = TRUE)
positives.count = table(final.pred.prob)[2]
prediction.50 = ifelse(final.pred.prob >= prob.sorted[50], 1, 0)
prediction.150 = ifelse(final.pred.prob >= prob.sorted[150], 1, 0)
prediction.250 = ifelse(final.pred.prob >= prob.sorted[250], 1, 0)

#write to files
write(prediction.50, file = "T.prediction.50.txt",
                   ncolumns = 1,
                   append = FALSE, sep = "\n")
write(prediction.150, file = "T.prediction.150.txt",
      ncolumns = 1,
      append = FALSE, sep = "\n")
write(prediction.250, file = "T.prediction.250.txt",
      ncolumns = 1,
      append = FALSE, sep = "\n")
