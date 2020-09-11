library(e1071)
library(gmodels)
library(ROCR)
library(ggplot2)


Flower<-read.csv("data/flowering_time_Bayes.csv", stringsAsFactors = TRUE)
Genotypes<-read.csv("data/genotype.csv", stringsAsFactors = TRUE)

dim(Flower)
str(Flower)
names(Flower)<-c("Days")
head(Flower)
Flower$Flowering<-NA
for (i in 1:length(Flower$Days)){
  if (Flower$Days[i]<=40){
    Flower$Flowering[i]<-"Fast"
  } else{
    Flower$Flowering[i]<-"Slow"
  }
}
Flower$Flowering<-factor(Flower$Flowering)
dim(Flower)
str(Flower)
head(Flower)
table(Flower$Flowering)
prop.table(table(Flower$Flowering))

dim(Genotypes)
str(Genotypes)
for (i in 1:ncol(Genotypes)){
  Genotypes[,i]<-factor(Genotypes[,i])
}
str(Genotypes)

# Se establece una semilla de pseudoaleatorización. 
set.seed(12345)

# Se crean los marcos de entrenamiento y prueba para el marco de floración.  
Flower.Train<-Flower[1:465, ]
Flower.Test<-Flower[465:696, ]

dim(Flower.Train)
dim(Flower.Test)

table(Flower.Train$Flowering)
table(Flower.Test$Flowering)

prop.table(table(Flower.Train$Flowering))
prop.table(table(Flower.Test$Flowering))

# Se crean los marcos de entrenamiento y prueba para el marco de genotipos. 
Genotypes.Train<-Genotypes[1:465, ]
Genotypes.Test<-Genotypes[465:696, ]



Flower.Bayes <- naiveBayes(Genotypes.Train, Flower.Train$Flowering)
Bayes.Pred <- predict(Flower.Bayes, Genotypes.Test)




Kross1<-CrossTable(Bayes.Pred, Flower.Test$Flowering, prop.chisq = FALSE, prop.t = FALSE, dnn = c('Predicted', 'Actual'))
Flower.Bayes2 <- naiveBayes(Genotypes.Train, Flower.Train$Flowering, laplace = 1)
Bayes2.Pred <- predict(Flower.Bayes2, Genotypes.Test)
Kross2<-CrossTable(Bayes2.Pred, Flower.Test$Flowering, prop.chisq = FALSE, prop.t = FALSE, dnn = c('Predicted', 'Actual'))

Flower.Bayes3 <- naiveBayes(Genotypes.Train, Flower.Train$Flowering, laplace = 2)
Bayes3.Pred <- predict(Flower.Bayes3, Genotypes.Test)
Kross3<-CrossTable(Bayes3.Pred, Flower.Test$Flowering, prop.chisq = FALSE, prop.t = FALSE, dnn = c('Predicted', 'Actual'))



EvalNaiveBayes<-function(lista){
  cat("En el modelo de Naive Bayes utilizado se ha encontrado que:\n")
  cat("    > Los verdaderos positivos son: ", lista$t[4], ".\n", sep = "")
  cat("    > Los falsos negativos encontrados son: ", lista$t[3], ".\n", sep = "")
  cat("    > Los verdaderos negativos son: ", lista$t[1], ".\n", sep = "")
  cat("    > Los falsos positivos hallados son: ", lista$t[2], ".\n\n", sep = "")
  cat("Evaluación del modelo de Naive Bayes:\n")
  Prec<-(lista$t[4]+lista$t[1])/(lista$t[4]+lista$t[3]+lista$t[2]+lista$t[1])
  cat("    > La precisión del modelo es ", Prec, ".\n", sep = "")
  ErrorR<-(lista$t[2]+lista$t[3])/(lista$t[1]+lista$t[2]+lista$t[3]+lista$t[4])
  cat("    > La tasa de error del modelo es ", ErrorR, ".\n", sep = "")
  Pra<-Prec
  Pre<-((lista$t[1]+lista$t[2])/(lista$t[1]+lista$t[2]+lista$t[3]+lista$t[4]))*((lista$t[1]+lista$t[3])/(lista$t[1]+lista$t[2]+lista$t[3]+lista$t[4]))+
    ((lista$t[3]+lista$t[4])/(lista$t[1]+lista$t[2]+lista$t[3]+lista$t[4]))*((lista$t[2]+lista$t[4])/(lista$t[1]+lista$t[2]+lista$t[3]+lista$t[4]))
  kappaR<-(Pra-Pre)/(1-Pre)
  cat("    > El valor de Kappa del modelo es ", kappaR, ".\n", sep = "")
  Sensibilidad<-lista$t[4]/(lista$t[4]+lista$t[3])
  cat("    > La sensibilidad del modelo es ", Sensibilidad, ".\n", sep = "")
  Especificidad<-lista$t[1]/(lista$t[2]+lista$t[1])
  cat("    > La especificidad del modelo es ", Especificidad, ".\n", sep = "")
  Recall<-Sensibilidad
  cat("    > El recall del modelo es ", Recall, ".\n", sep = "")
  Precision<-lista$t[4]/(lista$t[4]+lista$t[2])
  cat("    > El valor predictivo positivo es ", Precision, ".\n", sep = "")
  F.Measure<-(2*lista$t[4])/(2*lista$t[4]+lista$t[2]+lista$t[3])
  cat("    > La medida F del modelo es ", F.Measure, ".\n", sep = "")
  FN<-lista$t[3]
  FP<-lista$t[2]
  Marco<-data.frame(matrix(c(Prec, ErrorR, FP, FN, kappaR, Sensibilidad, Especificidad, Recall, Precision, F.Measure), ncol=10))
}

KrossA<-EvalNaiveBayes(Kross1)
KrossB<-EvalNaiveBayes(Kross2)
KrossC<-EvalNaiveBayes(Kross3)

BayesDF<-rbind(KrossA, KrossB, KrossC)
colnames(BayesDF)<-c("Accuracy","Error", "Falsos Positivos", "Falsos Negativos", "Valor Kappa","Sensibilidad", "Especificidad", "Recall", "Precisión", "Medida-F")
rownames(BayesDF)<-c("Sin Laplace", "Laplace 1", "Laplace 2")
BayesDF

# ROCR

png(filename="results/ROCBayesLaplace0.png")

par(mfrow=c(1,2))
Pred.Prob<-predict(Flower.Bayes, Genotypes.Test, type = "raw")
Pred.Prob<-as.data.frame(Pred.Prob)
pred <- prediction(predictions= Pred.Prob$Slow, labels= Flower.Test$Flowering)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
perf.auc <- performance(pred, measure="auc")
perf.auc <- unlist(perf.auc@y.values)
plot(perf, colorize=TRUE, lwd=2, main=paste("ROC.No Laplace. AUC=", round(perf.auc,3)))
abline(a = 0, b = 1, lwd = 1, lty = 2)
plot(perf, avg="threshold", colorize=TRUE, lwd=2, main=paste("ROC. No Laplace. AUC=", round(perf.auc,3)))
abline(a = 0, b = 1, lwd = 1, lty = 2)

dev.off()

png(filename="results/ROCBayesLaplace1.png")

par(mfrow=c(1,2))
Pred.Prob<-predict(Flower.Bayes2, Genotypes.Test, type = "raw")
Pred.Prob<-as.data.frame(Pred.Prob)
pred <- prediction(predictions= Pred.Prob$Slow, labels= Flower.Test$Flowering)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
perf.auc <- performance(pred, measure="auc")
perf.auc <- unlist(perf.auc@y.values)
plot(perf, colorize=TRUE, lwd=2, main=paste("ROC. Laplace 1. AUC=", round(perf.auc,3)))
abline(a = 0, b = 1, lwd = 1, lty = 2)
plot(perf, avg="threshold", colorize=TRUE, lwd=2, main=paste("ROC. Laplace 1. AUC=", round(perf.auc,3)))
abline(a = 0, b = 1, lwd = 1, lty = 2)

dev.off()


png(filename="results/ROCBayesLaplace2.png")

par(mfrow=c(1,2))
Pred.Prob<-predict(Flower.Bayes3, Genotypes.Test, type = "raw")
Pred.Prob<-as.data.frame(Pred.Prob)
pred <- prediction(predictions= Pred.Prob$Slow, labels= Flower.Test$Flowering)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
perf.auc <- performance(pred, measure="auc")
perf.auc <- unlist(perf.auc@y.values)
plot(perf, colorize=TRUE, lwd=2, main=paste("ROC. Laplace 2. AUC=", round(perf.auc,3)))
abline(a = 0, b = 1, lwd = 1, lty = 2)
plot(perf, avg="threshold", colorize=TRUE, lwd=2, main=paste("ROC. Laplace 2. AUC=", round(perf.auc,3)))
abline(a = 0, b = 1, lwd = 1, lty = 2)

dev.off()



