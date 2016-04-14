train.data<-read.csv('/Users/raghav/Downloads/train.csv',stringsAsFactors = F)
test.data<-read.csv('/Users/raghav/Downloads/test_dat.csv',stringsAsFactors = F)
library('caret')
library('xgboost')
library('ggplot2')

##REmove id
train.data$ID <- NULL
test.data.id <- test.data$ID
test.data$ID <- NULL


##unique_train_dat<-unique(train.data)


##### Extracting TARGET
train.data.y <- train.data$TARGET
train.data$TARGET <- NULL


##### 0 count per line
count0 <- function(x) {
  return( sum(x == 0) )
}

train.data$n0 <- apply(train.data, 1, FUN=count0)
test.data$n0 <- apply(test.data, 1, FUN=count0)
## Remove near zero variance
nzv <- nearZeroVar(train.data,names=TRUE, freqCut = 95/5,uniqueCut = 10,saveMetrics = TRUE)
summary(nzv)
train.data.nzv <- train.data[, -which(nzv$zeroVar)]

test.data.nzv <- test.data[, -which(nzv$zeroVar)]

precentage.train<-(colSums(train.data.nzv==0))/nrow(train.data.nzv)
remove_data.train<-which(precentage.train>.98)
length(remove_data.train)
train.data.zero<-subset(train.data.nzv, select = -c(remove_data.train) )

test.data.zero<-subset(test.data.nzv, select = -c(remove_data.train) )


### Run correlation
correlationMatrix <- cor(train.data.zero)
highlyCorDescr <- findCorrelation(correlationMatrix, name=TRUE,cutoff = .995)
cols.dont.want<-c(highlyCorDescr)
train.data.cor <- train.data.zero[, ! names(train.data.zero) %in% cols.dont.want, drop = F]

test.data.cor <- test.data.zero[, ! names(test.data.zero) %in% cols.dont.want, drop = F]



##ATTACH BACK THE DATA
train.data.cor$TARGET <- train.data.y

### 1
if("var3" %in% names(train.data.cor)){
##  train.data.out.1<-train.data.cor[train.data.cor$var3>0,]
  train.data.cor$var3[train.data.cor$var3<0]<-2
  test.data.cor$var3[test.data.cor$var3<0]<-2
  
    }
#### 2
# if("imp_op_var39_comer_ult1" %in% names(train.data.out.1)){
#   train.data.out.1<-train.data.out.1[train.data.out.1$imp_op_var39_comer_ult1<7000,]
# summary(train.data.out.1)
# }
# ## Check output with barplot
# # barplot(table(train.data.out.1$TARGET))
# # 
# # df.target<-data.frame(table(train.data.out.1$TARGET))
# # colnames(df.target)<-c('Target','Freq')
# # df.target$Perc<-df.target$Freq/sum(df.target$Freq)
# # ##barplot(df.target$Perc,ylim = 1)
# 
# #hist(train.data.out.1$imp_op_var39_comer_ult1)
# 
# summary(train.data.out.1)
# ### 3
# ##pos.data<-subset(train.data.out.1,TARGET==1)
# ##neg.data<-subset(train.data.out.1,TARGET==0)
# ##plot(train.data.out.1$imp_op_var39_comer_ult3)
# if("imp_op_var39_comer_ult3" %in% names(train.data.out.1)){
# train.data.out.1<-train.data.out.1[train.data.out.1$imp_op_var39_comer_ult3<6000,]
# }
# 
# ## 4
# ##hist(train.data.out.1$imp_op_var41_comer_ult3)
# if("imp_op_var41_comer_ult3" %in% names(train.data.out.1)){
#   train.data.out.1<-train.data.out.1[train.data.out.1$imp_op_var41_comer_ult3<2000,]
# }
# 
# 
# ## 5
# ##plot(train.data.out.1$imp_op_var41_efect_ult1)
# if("imp_op_var41_efect_ult1" %in% names(train.data.out.1)){
# train.data.out.1<-train.data.out.1[train.data.out.1$imp_op_var41_efect_ult1<2000,]
# }
# 
# ## 6
# a<-hist(train.data.out.1$imp_op_var41_efect_ult3)
# if("imp_op_var41_efect_ult3" %in% names(train.data.out.1)){
#   train.data.out.1<-train.data.out.1[train.data.out.1$imp_op_var41_efect_ult3<2000,]
# 
# }
# ## 7
# ##plot(train.data.out.1$saldo_var5)
# if("saldo_var5" %in% names(train.data.out.1)){
# length(train.data.out.1[train.data.out.1$saldo_var5<0,])
# train.data.out.1<-train.data.out.1[train.data.out.1$saldo_var5>0,]
# train.data.out.1<-train.data.out.1[train.data.out.1$saldo_var5<30000,]
# }
# 
# ##8 
# ##plot(train.data.out.1$saldo_var37)
# if("saldo_var30" %in% names(train.data.out.1)){
#   train.data.out.1<-train.data.out.1[train.data.out.1$saldo_var30<20000,]
# }
# 
# ##9
# if("saldo_var37" %in% names(train.data.out.1)){
# train.data.out.1<-train.data.out.1[train.data.out.1$saldo_var37<1800,]
# 
# }
#### Intermediate Model

# train.data.y <- train.data.out.1$TARGET
# train.data.out.1$TARGET<-NULL

### Run correlation
##train.data.out.1$TARGET <- train.data.y
library('Matrix')
train <- sparse.model.matrix(TARGET ~ ., data = train.data.out.1)

dtrain <- xgb.DMatrix(data=train, label=train.data.out.1$TARGET)
watchlist <- list(train=dtrain)

param <- list(  objective           = "reg:logistic", 
                booster             = "gbtree",
                eval_metric         = "auc",
                eta                 = 0.017,
                max_depth           = 5,
                subsample           = 0.7,
                colsample_bytree    = 0.8
)
## reduce subsam
clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 500, 
                    verbose             = 2,
                    watchlist           = watchlist,
                    maximize            = FALSE
)


test.data.cor$TARGET <- -1
test <- sparse.model.matrix(TARGET ~ ., data = test.data.cor)

preds <- predict(clf, test)
submission <- data.frame(ID=test.data.id, TARGET=preds)
cat("saving the submission file\n")
write.csv(submission, "submission.csv", row.names = F)

summary(test.data.cor)

##best 36
# param <- list(  objective           = "reg:logistic", 
#                 booster             = "gbtree",
#                 eval_metric         = "auc",
#                 eta                 = 0.017,
#                 max_depth           = 5,
#                 subsample           = 0.8,
#                 colsample_bytree    = 0.8
# )
# 
# clf <- xgb.train(   params              = param, 
#                     data                = dtrain, 
#                     nrounds             = 500, 
#                     verbose             = 2,
#                     watchlist           = watchlist,
#                     maximize            = FALSE
# )

