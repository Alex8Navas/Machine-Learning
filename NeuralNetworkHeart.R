# My First Neural Network from Scratch

library(tidyverse)
library(ggplot2)
library(caret)

# emilla sde pseudoaleatorización: 
set.seed(1234)

# Una capa oculta con cinco neuronas. 
# Funciones de activación a utilizar: tanh y la sigmoide. 

# Marco de datos a cargar: 
# https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5/tables/1
# https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5#Sec2
# https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records
# https://www.kaggle.com/andrewmvd/heart-failure-clinical-data/metadata


heart <- read_csv("data/heart_failure_clinical_records_dataset.csv")
head(heart)
# Reordenar aleatoriamente el marco de datos: 
heart2 <- heart[sample(nrow(heart)), ]
head(heart2)
glimpse(heart2)
heart2$death <- factor(heart2$DEATH_EVENT, labels = c("Superviviente", "Muerto"))

# Divisón del marco de datos en entrenamiento (80%) y prueba (20%): 
index <- round(0.8 * nrow(heart2))
train <- heart2[1:index, ]
head(train)
test <- heart2[(index+1):nrow(heart2), ]
head(test)

# Vemos si se asemejan los casos de riesgo de fallo cardíaco: 
prop.table(table(train$DEATH_EVENT))
prop.table(table(test$DEATH_EVENT))


ggplot(train, aes(x = death, fill = factor(DEATH_EVENT))) + geom_bar(width = 0.5) +
  scale_y_continuous("Número de Sujetos", breaks = c(0,25,50,75,100,125,150,175,200), expand=c(0,0)) +
  scale_fill_manual("Fallo Cardíaco", labels=c("Superviviente","Muerto"), values = c("#B14545", "#4D45B1")) +
  labs(title = "Supervivencia tras Fallo Cardíaco (Marco de Entrenamiento)",
       subtitle = "Conjunto de datos que contiene los registros médicos de 239 pacientes que tuvieron insuficiencia cardíaca,
recogidos durante su período de seguimiento, donde cada perfil de paciente tiene 13 características clínicas.",
       x = "Estado") + 
  theme_light() + theme(legend.position = "bottom",
                        axis.text.x = element_text(angle=55, vjust = 0.5, size = 11))

ggplot(test, aes(x = death, fill = factor(DEATH_EVENT))) + geom_bar(width = 0.5) +
  scale_y_continuous("Número de Sujetos", breaks = c(0,10,20,30,40), expand=c(0,0)) +
  scale_fill_manual("Fallo Cardíaco", labels=c("Superviviente","Muerto"), values = c("#B14545", "#4D45B1")) +
  labs(title = "Supervivencia tras Fallo Cardíaco (Marco de Prueba)",
       subtitle = "Conjunto de datos que contiene los registros médicos de 60 pacientes que tuvieron insuficiencia cardíaca,
recogidos durante su período de seguimiento, donde cada perfil de paciente tiene 13 características clínicas.",
       x = "Estado") + 
  theme_light() + theme(legend.position = "bottom",
                        axis.text.x = element_text(angle=55, vjust = 0.5, size = 11))


# Estandarización de las variables numéricas de los marcos de datos:
# Son age, creatinine_phosphokinase, ejection_fraction, platelets, serum_creatinine, serum_sodium & time.

strain <- data.frame(scale(train[, c("age", "creatinine_phosphokinase", "ejection_fraction", "platelets", "serum_creatinine", "serum_sodium", "time")]))

# Se añaden las variables categóricas: 
# Son anaemia, diabetes, high_blood_pressure, sex & smoking.
strain$anaemia <- train$anaemia
strain$diabetes <- train$diabetes
strain$high_blood_pressure <- train$high_blood_pressure
strain$sex <- train$sex
strain$smoking <- train$smoking

stest <- data.frame(scale(test[, c("age", "creatinine_phosphokinase", "ejection_fraction", "platelets", "serum_creatinine", "serum_sodium", "time")]))

stest$anaemia <- test$anaemia
stest$diabetes <- test$diabetes
stest$high_blood_pressure <- test$high_blood_pressure
stest$sex <- test$sex
stest$smoking <- test$smoking

# Se guardan las variables DEATH_EVENT de los marcos de entrenamiento y prueba. 
strainDeath <- train$DEATH_EVENT # Vector
dim(strainDeath) <- c(length(strainDeath), 1) # Redimensionar el vector, añadir una dimensión más. 
stestDeath <- test$DEATH_EVENT
dim(stestDeath) <- c(length(stestDeath), 1)

# Pasar a matriz y trabajar con la traspuesta (las variables como filas):
strainmatrix <-  t(as.matrix(strain, byrow = TRUE))
stestmatrix <- t(as.matrix(stest, byrow = TRUE))
stestDeathmatrix <- t(as.matrix(stestDeath, byrow = TRUE))
strainDeathmatrix <- t(as.matrix(strainDeath, byrow = TRUE))



# Construcción de la Red Neuronal 
# Función que define el tamaño de las capas de la red. 
getLayerSize <- function(X, y, hidden_neurons, train=TRUE) {
  input <- dim(X)[1]
  hidden <- hidden_neurons
  output <- dim(y)[1]   
  
  size <- list("input" = input, # Número de neuronas de la capa de entrada. 
               "hidden" = hidden, # Número de neuronas de la capa oculta.
               "output" = output) # Número de neuronas de la capa de salida. 
  
  return(size)
}

# Red de cinco neuronas. 
layerSize <- getLayerSize(strainmatrix, strainDeathmatrix, hidden_neurons = 5)
layerSize # 12 de entrada, 5 en la oculta y 1 en la salida. 

# Función para inicializar parámetros basándose en una distribución normal. 
initializeparameters <- function(X, listLayerSize){
  m <- dim(data.matrix(X))[2]
  
  input <- listLayerSize$input
  hidden <- listLayerSize$hidden
  output <- listLayerSize$output
  
  W1 <- matrix(rnorm(hidden * input),
               nrow = hidden, 
               ncol = input,
               byrow = TRUE) * 0.01
  b1 <- matrix(rep(0, hidden), nrow = hidden)
  W2 <- matrix(rnorm(output * hidden), 
               nrow = output,
               ncol = hidden, 
               byrow = TRUE) * 0.01
  b2 <- matrix(rep(0, output), nrow = output)
  
  params <- list("W1" = W1,
                 "b1" = b1, 
                 "W2" = W2,
                 "b2" = b2)
  
  return (params)
}


# Se inicializan los parámetros con la matriz del marco de entrenamiento y el tamaño de la capa. 
parameters <- initializeparameters(strainmatrix, layerSize)
# Observar las dimensiones de los parámetros creados.
lapply(parameters, function(x) dim(x))

# Definición de las Funciones de Activación.
# Función sigmoide.
sigmoide <- function(x){
  return(1 / (1 + exp(-x)))
}
# Función tangente hiperbólica (está en la base de R como tanh(), pero la defino por claridad).
tangentehiperbolica <- function(x){
  return((exp(x) - exp(-x)) / (exp(x) + exp(-x)))
}

# Se grafican las funciones para ver su forma:
x <- seq(from = -10, to = 10, by =0.5)
yth <- tangentehiperbolica(x)
ysig <- sigmoide(x)


plot(x, yth, type = "l", lwd = 3, col = "steelblue",
     ylab = "Tangente Hiperbólica",
     main = "Función Tangente Hiperbólica") 
plot(x, ysig, type = "l", lwd = 3, col = "red",
     ylab = "Sigmoide",
     main = "Función Sigmoide")

# Propagación hacia delante.
forwardpropagation <- function(X, params, listLayeSize){
  
  m <- dim(X)[2]
  hidden <- listLayeSize$hidden
  output <- listLayeSize$output
  
  # Parámetros definidos. 
  W1 <- params$W1
  b1 <- params$b1
  W2 <- params$W2
  b2 <- params$b2
  
  # Redimensionar los interceptos para poder hacer los productos de matrices. 
  # El intercepto es el mismo. Se trata de repetir el valor tantas veces como
  # se lleve a cabo la operación (tantas veces como ecuaciones lineales haya). 
  b1Redim <- matrix(rep(b1, m), nrow = hidden)
  b2Redim <- matrix(rep(b2, m), nrow = output)
  
  Z1 <- W1 %*% X + b1Redim
  A1 <- sigmoide(Z1)
  Z2 <- W2 %*% A1 + b2Redim
  A2 <- sigmoide(Z2)
  
  cache <- list("Z1" = Z1,
                "A1" = A1, 
                "Z2" = Z2,
                "A2" = A2)
  return (cache)
}
# A2 es el valor que se necesita para la propagación hacia adelante. 
# El resto de valores que se devuelven serán utilizados para la retropropagación. 
forwardprop <- forwardpropagation(strainmatrix, parameters, layerSize)
lapply(forwardprop, function(x) dim(x))

# Definición de la función de Coste: 
# Concepto entropía cruzada: https://es.wikipedia.org/wiki/Entrop%C3%ADa_cruzada
costfunction <- function(X, y, cache) {
  m <- dim(X)[2]
  A2 <- cache$A2
  logprobs <- (log(A2) * y) + (log(1-A2) * (1-y)) 
  cost <- -sum(logprobs/m) # Binary Cross-Entropy Loss Function (función binaria de pérdida de la entropía cruzada)
  return (cost)
}


# --- Antiguo Script con factor -----
# Hay que volver a obtener la matriz de la variable categórica como número: 
# J <- as.numeric(strainDeathmatrix)
# J <- if_else(J == 2, 1, 0)

coste <- costfunction(strainmatrix, strainDeathmatrix, forwardprop)
coste

# Se define la función para llevar a cabo la retropropagación: 

backwardpropagation <- function(X, y, cache, params, listLayerSize){
  
  m <- dim(X)[2]
  
  input <- listLayerSize$input
  hidden <- listLayerSize$hidden
  output <- listLayerSize$output
  A2 <- cache$A2
  A1 <- cache$A1
  W2 <- params$W2
  
  
  dZ2 <- A2 - y
  dW2 <- 1/m * (dZ2 %*% t(A1)) 
  db2 <- matrix(1/m * sum(dZ2), nrow = output)
  db2_new <- matrix(rep(db2, m), nrow = output)
  
  dZ1 <- (t(W2) %*% dZ2) * (1 - A1^2)
  dW1 <- 1/m * (dZ1 %*% t(X)) # Derivada de la función de pérdida respecto del peso (W). 
  db1 <- matrix(1/m * sum(dZ1), nrow = hidden) # Derivada de la función de pérdida respecto del sesgo o bias (b).  
  db1_new <- matrix(rep(db1, m), nrow = hidden)
  
  grads <- list("dW1" = dW1, 
                "db1" = db1,
                "dW2" = dW2,
                "db2" = db2)
  
  return(grads)
}

backprop <- backwardpropagation(strainmatrix, strainDeathmatrix,
                                forwardprop, parameters, layerSize)
lapply(backprop, function(x) dim(x))
backprop


# Se define una función para la actualización de los pesos. 
# Parámetros: el gradiente, los parámetros de la red y la ratio de aprendizaje (alfa). 

updateparameters <- function(grads, params, alpha){
  
  # Se almacenan los parámetros en variables.
  W1 <- params$W1
  b1 <- params$b1
  W2 <- params$W2
  b2 <- params$b2
  
  # Se almacenan los gradientes en variables. 
  dW1 <- grads$dW1
  db1 <- grads$db1
  dW2 <- grads$dW2
  db2 <- grads$db2
  
  # Actualización de los pesos. 
  W1 <- W1 - alpha * dW1
  b1 <- b1 - alpha * db1
  W2 <- W2 - alpha * dW2
  b2 <- b2 - alpha * db2
  
  updatedparams <- list("W1" = W1,
                        "b1" = b1,
                        "W2" = W2,
                        "b2" = b2)
  
  return (updatedparams)
}

updated <- updateparameters(backprop, parameters, alpha = 0.01)
lapply(updated, function(x) dim(x))
updated


# Se define la función de entrenamiento del modelo. 

trainModel <- function(X, y, iterations, hiddenNeurons, alpha){
  
  layerSize <- getLayerSize(X, y, hiddenNeurons)
  parameters <- initializeparameters(X, layerSize)
  costHistory <- c() # Historial de los costes por iteración. 
  for (i in 1:iterations) {
    fwdprop <- forwardpropagation(X, parameters, layerSize)
    cost <- costfunction(X, y, fwdprop)
    backprop <- backwardpropagation(X, y, fwdprop, parameters, layerSize)
    updateparameters <- updateParameters(backprop, parameters, alpha)
    parameters <- updateparameters
    costHistory <- c(costHistory, cost)
    
    if (i %% 10000 == 0) cat("Iteration", i, " | Cost: ", cost, "\n")
  }
  
  modelOutput<- list("updatedparameters" = updateparameters,
                    "costHistory" = costHistory)
  return (modelOutput)
}


# Se definen las épocas, el número de neuronas de la capa oculta y la ratio de aprendizaje. 
epochs = 80000
hidden = 12
alfa = 1.1

RNA <- trainModel(strainmatrix, strainDeathmatrix, 
                          hiddenNeurons = hidden,
                          iterations = epochs,
                          alpha = alfa)

# Se construye un modelo de regresión logística con glm() para comparar rendimientos.
linealModel <- glm(strainDeath ~., data = strain)
linealModel
summary(linealModel)
# El modelo de regresión lineal se puede mejorar bastante, pero el propósito es compararlo con la RNA, 
# así que no se estudiará cómo mejorarlo en este script. 
predictione <- round(as.vector(predict(linealModel, stest)))
predictione

# Se evalúa le rendimiento de la RNA. 
performanceRNA <- function(X, y, hiddenNeurons){
  layerSize <- getLayerSize(X, y, hiddenNeurons)
  parameters <- RNA$updatedparameters
  fwdprop <- forwardpropagation(X, parameters, layerSize)
  predictioneRNA <- fwdprop$A2
  
  return (predictioneRNA)
}
predictionsRNA <- performanceRNA(stestmatrix, stestDeathmatrix, hidden)
predictionsRNA <- round(predictionsRNA) # Se pasa a unos y ceros.
predictionsRNA <- as.vector(predictionsRNA)
predictionsRNA

# Se comparan las predicciones con el valor real del evento muerte en el marco de prueba: 
allespred <- as.data.frame(cbind("GLM" = predictione, "RNA" = predictionsRNA, "Real" = as.vector(stestDeath)))
allespred # Marco de datos para observar el acierto y error caso a caso. 
allespred$GLM <- factor(allespred$GLM, labels = c("Superviviente", "Muerto"))
allespred$RNA <- factor(allespred$RNA, labels = c("Superviviente", "Muerto"))
allespred$Real <- factor(allespred$Real, labels = c("Superviviente", "Muerto"))
allespred

# Tablas de doble entrada:
tablaRNA <- table(as.vector(stestDeath), predictionsRNA)
tablaRNA
tablaGLM <- table(as.vector(stestDeath), predictione)
tablaGLM

# Métricas de Evluación de los modelos:
# Precisión = TP/(TP+FP)
# Recall = = TP/(TP+FN)
# F1Score = 2*(Precisión*Recall)/(Precisión+Recall)
# Accuracy = (TP+TF)/n

# La tabla obtenida está con las predicciones en la parte superior,
# luego tb[3] son los falsos positivos (7 para RNA) y tb[2] son los falsos negativos (5 para RNA)
calculateStats <- function(tb, model_name) {
  precision <- tb[4]/(tb[4] + tb[3])
  accuracy <- (tb[1] + tb[4])/(tb[1] + tb[2] + tb[3] + tb[4])
  recall <- tb[4]/(tb[4] + tb[2])
  f1 <- 2 * ((precision * recall) / (precision + recall))
  
  cat(model_name, ": \n")
  cat("\tAccuracy = ", accuracy*100, "%.")
  cat("\n\tPrecision = ", precision*100, "%.")
  cat("\n\tRecall = ", recall*100, "%.")
  cat("\n\tF1 Score = ", f1*100, "%.\n\n")
}
calculateStats(tablaRNA, "Neural Network")
calculateStats(tablaGLM, "GLM")
# Forma rápida de hacerlo con el paquete Caret: 
confusionMatrix(allespred$RNA, allespred$Real, positive = "Muerto")
confusionMatrix(allespred$GLM, allespred$Real, positive = "Muerto")
