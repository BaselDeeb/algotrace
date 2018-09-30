one.var.model.class.cv <- function(i, myform,loop.vars, Data, output.var,Criterion,char,method) {
  k<-5
  tryCatch({
    set.seed(1)
    folds <- createFolds(Data[[output.var]], k = k, list = TRUE, returnTrain = TRUE)
    if(method=="Classification"){
      testxpropHits<-list()
      testxpropHits0<-list()
      testxpropHits1<-list()
      xtable<-list()
      testxauc<-list()
      testxF<-list()
      target_vec<-NULL
      prob_vec<-NULL
      prob.list<-list()
      for(ind in 1:k){
        train<-Data[folds[[ind]],]
        test<-Data[-folds[[ind]],]
        
      if(char=="Logistic" || char=="Naive Logistic"){
        model<-tryCatch({
        speedglm(myform,family=binomial("logit"),data=train) 
        }, error=function(err){
        glm(myform,family=binomial("logit"),data=train) 
        })
        prob<-predict(model,newdata=test,type="response")
      } 
      if(char=="Weighted Logistic" || char=="Naive Weighted Logistic"){
        model<-tryCatch({
          speedglm(myform,family=binomial("logit"),data=train,weights=log.weights.spy[folds[[ind]]]) 
        }, error=function(err){
          glm(myform,family=binomial("logit"),data=train,weights=log.weights.spy[folds[[ind]]]) 
        })
        prob<-predict(model,newdata=test,type="response")
      }   
      if(char=="Xgboost" || char=="Naive Xgboost"){
        Train<- sparse.model.matrix(myform, train) 
        set.seed(xgb.seed);model <- xgboost(data = Train, label = train[[output.var]], nround=xg.nround, params=xg.params,verbose=0)
        Test<- sparse.model.matrix(myform, test) 
        prob<-predict(model,Test)
      }
      if(char=="Recursive Partitioning Tree" || char=="Naive Recursive Partitioning Tree"){
        model<-rpart(myform,data=train,method="class",control=rpart.control(minsplit=rpart.params$minsplit,cp=rpart.params$cp,maxdepth=rpart.params$maxdepth)) 
        prob<-predict(model,newdata=test)[,'1']
      }
      if(char=="Rforest" || char=="Naive Rforest"){
        temp.train<-train   
        temp.train[[output.var]]<-as.factor(temp.train[[output.var]])
        set.seed(12);model<-ranger(myform, data = temp.train, num.trees = 5, write.forest = TRUE,classification=TRUE,probability =TRUE) 
        prob<-predict(model,test)$predictions[,'1']
      }  
      if(char=="Neural Network" || char=="Naive Neural Network"){
        Train<- data.matrix(sparse.model.matrix(myform, train)) 
        mx.set.seed(0)
        model <- mx.mlp(Train, train[[output.var]], hidden_node=5,activation='relu', out_node=2, out_activation="softmax",
                        num.round=20, array.batch.size=15, learning.rate=0.07, momentum=0.9, 
                        eval.metric=mx.metric.accuracy)
        Test<- sparse.model.matrix(myform, test)
        prob<- tryCatch({
          t(predict(model, data.matrix(Test)))[,2]  
        }, error=function(err){
          return(t(predict(model, Test))[,2])
        })
      }  

      prob.list[[ind]]<-prob
      predict_response_test <- ifelse(prob>classification.threshold,1,0) 
      
      ##For AUC and Gini##
      target_vec<-c(target_vec,test[[output.var]])
      prob_vec<-c(prob_vec,prob)
      #######
      testx<-CV.Table(test[[output.var]],predict_response_test)
      xtable[[ind]]<-testx
      ######
      if(Criterion=="Accuracy")
      testxpropHits[[ind]]<-(testx[1,1]+testx[2,2])/sum(testx)
      if(Criterion=="Accuracy 0")
      testxpropHits0[[ind]]<-testx[1,1]/(testx[1,1]+testx[2,1])
      if(Criterion=="Precision")
      testxpropHits1[[ind]]<-testx[2,2]/(testx[2,2]+testx[1,2])
      if(Criterion=="F measure"){
        Precision<-testx[2,2]/(testx[2,2]+testx[1,2])
        Recall<-testx[2,2]/(testx[2,1]+testx[2,2])
        testxF[[ind]]<-2*Recall*Precision/(Recall+Precision)
      }
      if(Criterion=="AUC")
      testxauc[[ind]]<-auc_roc(prob,test[[output.var]])
      }
        
      if(Criterion=="Accuracy")
        stability<-group.distance(testxpropHits)
      if(Criterion=="Accuracy 0")
        stability<-group.distance(testxpropHits0)
      if(Criterion=="Precision")
        stability<-group.distance(testxpropHits1)
      if(Criterion=="F measure")
        stability<-group.distance(testxF)
      if(Criterion=="AUC")
        stability<-group.distance(testxauc)
      
      Table<-Reduce('+',xtable)#test1x+test2x+test3x
      
      Accuracy<-(Table[1,1]+Table[2,2])/sum(Table)
      Accuracy_0 <- Table[1,1]/(Table[1,1]+Table[2,1])
      Precision <- Table[2,2]/(Table[2,2]+Table[1,2])
      False_Positive <- Table[1,2]/(Table[1,1]+Table[1,2])
      False_Negative <- Table[2,1]/(Table[2,1]+Table[2,2])
      #Diff##
      Diff0<-Table[1,1]-Table[2,1]
      Diff1<-Table[2,2]-Table[1,2]
      #Lift
      if(sum(Table[2,])==0){
        Lift_1<-NaN
      } else{
        temp.mean<-sum(Table[2,])/sum(Table)
        Lift_1<-100*(Precision/temp.mean) 
      }

      #F measure#
      Recall<-Table[2,2]/(Table[2,1]+Table[2,2])
      F_measure<-2*Recall*Precision/(Recall+Precision)
      ##auc##
      AUC<-auc_roc(prob_vec,target_vec)
      Gini<-2*AUC-1
      ##
      tmp.df <- data.frame(Accuracy, stability,Accuracy_0,Precision,Recall,False_Positive,False_Negative,
                           Lift_1,Diff0,Diff1,F_measure,AUC,Gini,check.names=F)
      names(tmp.df) <- c('Accuracy', 'Stability','Accuracy 0','Precision','Recall','False Positive','False Negative',
                         'Lift 1(%)','Diff 0','Diff 1','F measure','AUC','Gini')
      rownames(tmp.df) <- i
      return(tmp.df)
    }
    if(method=="Estimation"){
      diff<-list()
      ALL_predict_response_test<-NULL
      ALL_test<-NULL
      pred.list<-list()
      pred.list.train<-list()
      for(ind in 1:k){
        train<-Data[folds[[ind]],]
        test<-Data[-folds[[ind]],]
      if(char=="Linear" || char=="Naive Linear"){
        model<-tryCatch({
          speedlm(myform,data=train) 
        }, error=function(err){
          lm(myform,data=train)
        })
        predict_response_test <- predict(model,newdata=test,type='response')
        predict_response_train<-predict(model,newdata=train,type='response')
      }
        if(char=="Weighted Linear" || char=="Naive Weighted Linear"){
          model<-tryCatch({
            speedlm(myform,data=train,weights=log.weights.spy[folds[[ind]]]) 
          }, error=function(err){
            lm(myform,data=train,weights=log.weights.spy[folds[[ind]]])
          })
          predict_response_test <- predict(model,newdata=test,type='response')
          predict_response_train<-predict(model,newdata=train,type='response')
        }  
      if(char=="Negative Binomial" || char=="Naive Negative Binomial"){
        model<-glm.nb(myform,data=train)
        predict_response_test <- predict(model,newdata=test,type='response')
        predict_response_train<-predict(model,newdata=train,type='response')
      }
      if(char=="Quantile" || char=="Naive Quantile"){
        model<-rq(myform,tau = .5, method = "pfn",data=train)
        predict_response_test <- predict(model,newdata=test,type='response')
        predict_response_train<-predict(model,newdata=train,type='response')
      }
      if(char=="Xgboost" || char=="Naive Xgboost"){
        Train<- sparse.model.matrix(myform, train) 
        set.seed(xgb.seed);model <- xgboost(data = Train, label = train[[output.var]], nround=xg.nround, params=xg.params,verbose=0)
        Test<- sparse.model.matrix(myform, test) 
        predict_response_test <- predict(model,Test)
        Train<- sparse.model.matrix(myform, train) 
        predict_response_train<-predict(model,Train)
      }
      if(char=="Recursive Partitioning Tree" || char=="Naive Recursive Partitioning Tree"){
        model<-rpart(myform,data=train,control=rpart.control(minsplit=rpart.params$minsplit,cp=rpart.params$cp,maxdepth=rpart.params$maxdepth)) 
        predict_response_test <- predict(model,newdata=test)
        predict_response_train<-predict(model,newdata=train)
      }
      if(char=="Rforest" || char=="Naive Rforest"){
        set.seed(12);model<-ranger(myform, data = train, num.trees = 5, write.forest = TRUE,classification=FALSE)
        predict_response_test <- predict(model,test)$predictions
        predict_response_train<-predict(model,train)$predictions
      }
      if(char=="Neural Network" || char=="Naive Neural Network"){
        data <- mx.symbol.Variable("data")
        fc1 <- mx.symbol.FullyConnected(data, num_hidden=1)
        lro <- mx.symbol.LinearRegressionOutput(fc1)
        Train<- data.matrix(sparse.model.matrix(myform, train)) 
        mx.set.seed(0)
        model <- mx.model.FeedForward.create(lro, X=Train, y=train[[output.var]],
                                             ctx=mx.cpu(), num.round=50, array.batch.size=20,
                                             learning.rate=2e-6, momentum=0.9, eval.metric=mx.metric.rmse)
        Test<- sparse.model.matrix(myform, test) 
        predict_response_test<- tryCatch({
          as.numeric(predict(model, data.matrix(Test))) 
        }, error=function(err){
          return(as.numeric(predict(model, Test)))
        })
        predict_response_train<-tryCatch({
          as.numeric(predict(model, data.matrix(Train))) 
        }, error=function(err){
          return(as.numeric(predict(model, Train)))
        })
      }   

      pred.list[[ind]]<-predict_response_test
      pred.list.train[[ind]]<-predict_response_train
        
      ALL_predict_response_test<-c(ALL_predict_response_test,predict_response_test)
      ALL_test<-c(ALL_test,test[[output.var]])
        
      diff[[ind]]<- mean(abs(test[[output.var]]-predict_response_test),na.rm =TRUE)/mean(abs(test[[output.var]]),na.rm =TRUE)

      }
      
      R2<-R2.calulation(act=ALL_test,pred=ALL_predict_response_test)
      R2.Adj<-1-((1-R2)*(length(ALL_test)-1))/(length(ALL_test)-length(loop.vars)-1)
      Cor <- cor(ALL_test,ALL_predict_response_test,use ="complete.obs",method="spearman")
      RMSE<- sqrt(mean((ALL_test-ALL_predict_response_test)^2,na.rm =TRUE))
      MAE<- mean(abs(ALL_test-ALL_predict_response_test),na.rm =TRUE)
      Norm_abs_Difference<- mean(abs(ALL_test-ALL_predict_response_test),na.rm =TRUE)/mean(abs(ALL_test),na.rm =TRUE)
      Difference<- mean(ALL_test-ALL_predict_response_test,na.rm =TRUE)
      Norm_Difference<- mean(ALL_test-ALL_predict_response_test,na.rm =TRUE)/mean(ALL_test,na.rm =TRUE)
      
      Stability<- group.distance(diff)
      Sum_output<-sum(ALL_test,na.rm =TRUE)
      Sum_pred<-sum(ALL_predict_response_test,na.rm =TRUE)
      Abs_Sum_Difference<-abs(Sum_output-Sum_pred)
      Avg_output<-mean(ALL_test,na.rm =TRUE)
      Avg_pred<-mean(ALL_predict_response_test,na.rm =TRUE)
      tmp.df <- data.frame(MAE,Norm_abs_Difference,Difference,Norm_Difference, Stability,Sum_output,Sum_pred,Abs_Sum_Difference,Sum_output/Sum_pred,Avg_output,Avg_pred,RMSE,Cor,R2,R2.Adj,check.names=F)
      names(tmp.df) <- c('MAE','Norm abs Difference','Difference','Norm Difference','Stability','Sum output','Sum pred','Abs Sum Difference','Act div Pred','Avg output','Avg pred','RMSE','Cor(y,yhat)','R2','R2 Adj')
      rownames(tmp.df) <- i
      return(tmp.df)
    }
  },
  error=function(err) {return(NULL)})
}


