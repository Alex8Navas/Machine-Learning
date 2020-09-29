library(class)
library(gmodels)


Wisc<-read.csv("data/wisc_bc_data.csv", stringsAsFactors = TRUE)

head(Wisc)
str(Wisc)

# ID está en la primera columna. Se procede a su eliminación.
rownames(Wisc)<-Wisc[,1]
Wisc<-Wisc[, -1]

# Se comprueba que se ha eliminado la variable ID. 
head(Wisc)
str(Wisc)

# Frecuencias absolutas. 
table(Wisc$diagnosis)
# Frecuencias relativas. 
prop.table(table(Wisc$diagnosis))
# Recodificación. 
Wisc$diagnosis<-factor(Wisc$diagnosis, levels = c("B", "M"), 
                       labels = c("Benigno", "Maligno"))

# Comprobación de la recodificación.
round(prop.table(table(Wisc$diagnosis)) * 100, digits = 1)

summary(Wisc[c("radius_mean", "area_mean", "smoothness_mean")])

# Se crea la función para normalizar. 
Normalizar<-function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
# Se aplica a las 30 variables cuantitativas. 
Wisc.Normalizado<-as.data.frame(lapply(Wisc[2:31], Normalizar))

# Se confirma que la normalización se ha efectuado correctamente. 
summary(Wisc.Normalizado$area_mean)

Wisc_train <- Wisc.Normalizado[1:469, ]
Wisc_test <- Wisc.Normalizado[470:569, ]

Wisc_train_labels<-Wisc[1:469, 1]
Wisc_test_labels<-Wisc[470:569, 1]


Wisc_test_pred<-knn(train = Wisc_train, test = Wisc_test, cl = Wisc_train_labels, k=21)
CrossTable(x=Wisc_test_labels, y=Wisc_test_pred, prop.chisq = FALSE)

# Se estandafrizan todas las columnas menos la de diagnóstico. 
Wisc.Z<-as.data.frame(scale(Wisc[, -c(1)]))

# Se confirma la transformación. 
summary(Wisc.Z$area_mean)


Wisc_train.Z<-Wisc.Z[1:469, ]
Wisc_test.Z<-Wisc.Z[470:569, ]

# El paso de crear los vectores de etiquetas no es necesario, pues son los mismos. 
Wisc_test_pred.Z <- knn(train = Wisc_train.Z, test = Wisc_test.Z,
                        cl = Wisc_train_labels, k=21)
CrossTable(x = Wisc_test_labels, y = Wisc_test_pred.Z, prop.chisq=FALSE)

Wisc_test_pred.1<-knn(train = Wisc_train, test = Wisc_test, cl = Wisc_train_labels, k=1)
CrossTable(x=Wisc_test_labels, y=Wisc_test_pred.1, prop.chisq = FALSE)

Wisc_test_pred.5<-knn(train = Wisc_train, test = Wisc_test, cl = Wisc_train_labels, k=5)
CrossTable(x=Wisc_test_labels, y=Wisc_test_pred.5, prop.chisq = FALSE)

Wisc_test_pred.11<-knn(train = Wisc_train, test = Wisc_test, cl = Wisc_train_labels, k=11)
CrossTable(x=Wisc_test_labels, y=Wisc_test_pred.11, prop.chisq = FALSE)

Wisc_test_pred.15<-knn(train = Wisc_train, test = Wisc_test, cl = Wisc_train_labels, k=15)
CrossTable(x=Wisc_test_labels, y=Wisc_test_pred.15, prop.chisq = FALSE)

Wisc_test_pred.27<-knn(train = Wisc_train, test = Wisc_test, cl = Wisc_train_labels, k=27)
CrossTable(x=Wisc_test_labels, y=Wisc_test_pred.27, prop.chisq = FALSE)

