import numpy as np
import pandas as pd
from sklearn import svm
from scipy.stats import kurtosis
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# inicializacion de las variables
presion = np.zeros(0)
altitud = np.zeros(0)
humedad = np.zeros(0)
temperatura = np.zeros(0)

# For loop para recorrer el csv con intervalos de 10 renglones y generar los calculos
for data in pd.read_csv("Sesion23sep2019.csv", usecols=[
        "Presion", "Altitud", "Humedad", "Temperatura"], chunksize=10):
    # Calculos de la presion
    presion = data["Presion"].values
    presion = presion[np.logical_not(np.isnan(presion))]
    presionMean = presion.mean()
    presionStd = np.std(presion)
    presionKrt = kurtosis(presion)

    # Calculos de la altitud
    altitud = data["Altitud"].values
    altitud = altitud[np.logical_not(np.isnan(altitud))]
    altitudMean = altitud.mean()
    altitudStd = np.std(altitud)
    altitudKrt = kurtosis(altitud)

    # Calculos de la humedad
    humedad = data["Humedad"].values
    humedad = humedad[np.logical_not(np.isnan(humedad))]
    humedadMean = humedad.mean()
    humedadStd = np.std(humedad)
    humedadKrt = kurtosis(humedad)

    # Calculos de la temperatura
    temperatura = data["Temperatura"].values
    temperatura = temperatura[np.logical_not(np.isnan(temperatura))]
    temperaturaMean = temperatura.mean()
    temperaturaStd = np.std(temperatura)
    temperaturaKrt = kurtosis(temperatura)


# ---------------------------------- Clasificacion de instancias usando una SVM-------------------------------#


df = pd.read_csv("Sesion23sep2019.csv", usecols=[
                 "Presion", "Altitud", "Humedad", "Temperatura", "Ocupacion"])
df = df.dropna()
training_set, test_set = train_test_split(df, test_size=0.2, random_state=1)

X_train = training_set.iloc[:, 0:2].values
Y_train = training_set.iloc[:, 4].values
X_test = test_set.iloc[:, 0:2].values
Y_test = test_set.iloc[:, 4].values


classifier = SVC(kernel='rbf', random_state=1)
classifier.fit(X_train, Y_train)


Y_pred = classifier.predict(X_test)

test_set["Predictions"] = Y_pred

cm = confusion_matrix(Y_test, Y_pred)
accuracy = float(cm.diagonal().sum())/len(Y_test)

print("\nAccuracy Of SVM: ", accuracy, "\n")

# Estos prints los habia hecho para checar que el train set y el test set si fueran diferentes (poque da 1.0 el accuracy)
# print(training_set)
# print("-----------------------")
# print(test_set)

# ---------------------------------- Clasificacion de instancias usando una Weighted KNN-------------------------------#
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

test_set["Predictions"] = Y_pred

cm = confusion_matrix(Y_test, Y_pred)
accuracy = float(cm.diagonal().sum())/len(Y_test)

print("\nAccuracy Of WKNN: ", accuracy, "\n")

# ---------------------------------- Clasificacion de instancias usando una Random Forest-------------------------------#
classifier = RandomForestClassifier(
    n_estimators=100, max_depth=2, random_state=0)

classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)


print(classifier.feature_importances_)


# Referencias:
# https://scikit-learn.org/stable/modules/svm.html
