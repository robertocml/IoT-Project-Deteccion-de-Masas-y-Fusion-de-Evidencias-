import numpy as np
import pandas as pd
# from sklearn import svm
from scipy.stats import kurtosis
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
import math

# inicializacion de las variables
presion = np.zeros(0)
altitud = np.zeros(0)
humedad = np.zeros(0)
temperatura = np.zeros(0)


def buildDataF(dataF):
    dfTemp = pd.DataFrame()

    # Divide el dataframe en n grupos, n viene siendo una division del tamño de renglones entre 10
    for data in np.array_split(dataF, math.floor(np.shape(dataF)[0]/10)):
        # Calculos de la presion
        presion = data["Presion"].values
        presion = presion[np.logical_not(np.isnan(presion))]
        if presion.any():
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
            # temperaturaNan = data["Temperatura"].values
            temperatura = temperatura[np.logical_not(np.isnan(temperatura))]
            temperaturaMean = temperatura.mean()
            temperaturaStd = np.std(temperatura)
            temperaturaKrt = kurtosis(temperatura)
        else:
            print("something is null")

        # Junta los datos que se meteran en un renglon del dataframe
        data = [[presionMean, presionStd, presionKrt,
                altitudMean, altitudStd, altitudKrt,
                humedadMean, humedadStd, humedadKrt,
                temperaturaMean, temperaturaStd, temperaturaKrt, dataF.Ocupacion.iloc[0]]]

        names = ["presionMean", "presionStd", "presionKrt",
                "altitudMean", "altitudStd", "altitudKrt",
                "humedadMean", "humedadStd", "humedadKrt",
                "temperaturaMean", "temperaturaStd", "temperaturaKrt", "Ocupacion"]

        # Une este renglon al dataframe que se enviara al final
        dfTemp = dfTemp.append(pd.DataFrame(data, columns=names))

    return dfTemp


df = pd.read_csv("Sensado_GYM_Completo.csv", usecols=[
                 "Fecha", "Presion", "Altitud", "Humedad", "Temperatura", "Ocupacion"])
df = df.dropna()

# Filtra el dataframe para solo contenga el día
df['Fecha'] = pd.to_datetime(df['Fecha']).dt.strftime(
    '%d')  # dt.strftime('%d/%m/%Y %H:%M')

dfLow = df[df["Ocupacion"] == 'L']
dfMed = df[df["Ocupacion"] == 'M']
dfHigh = df[df["Ocupacion"] == 'H']

dfFinal = pd.DataFrame()


while not dfLow.empty:  # Mientras no esten vacias
    # obtiene solo los valores con la primera fecha y lo envía a la funcion para
    dfFinal = dfFinal.append(buildDataF(
        dfLow[dfLow["Fecha"] == dfLow.iloc[0, 0]]))
                                                                    # que construya el dataframe y lo una al dataframe que tendra todos los datos
    # Remueve todos los datos que contengan la primera fecha
    dfLow = dfLow[dfLow.Fecha != dfLow.iloc[0, 0]]

# print(dfFinal)

while not dfMed.empty:  # Mientras no esten vacias
    # obtiene solo los valores con la primera fecha y lo envía a la funcion para
    dfFinal = dfFinal.append(buildDataF(
        dfMed[dfMed["Fecha"] == dfMed.iloc[0, 0]]))
                                                    # que construya el dataframe
    # Remueve todos los datos que contengan la primera fecha
    dfMed = dfMed[dfMed.Fecha != dfMed.iloc[0, 0]]

while not dfHigh.empty:  # Mientras no esten vacias
    # obtiene solo los valores con la primera fecha y lo envía a la funcion para
    dfFinal = dfFinal.append(buildDataF(
        dfHigh[dfHigh["Fecha"] == dfHigh.iloc[0, 0]]))
                                                    # que construya el dataframe
    # Remueve todos los datos que contengan la primera fecha
    dfHigh = dfHigh[dfHigh.Fecha != dfHigh.iloc[0, 0]]


# ---------------------------------- Clasificacion de instancias usando una SVM-------------------------------#
df = dfFinal

training_set, test_set=train_test_split(df, test_size=0.2, random_state=1)

X_train=training_set.iloc[:, 0:12].values
Y_train=training_set.iloc[:, 12].values
X_test=test_set.iloc[:, 0:12].values
Y_test=test_set.iloc[:, 12].values

#----------------- PCA --------------------------#
pca = PCA(n_components=3) # Aca podemos se puede varian el num de comp para buscar el mejor balance (accuracy - n_of_components)

print("xtrain.SHAPE antes del PCA: ", X_train.shape)
X_train = pca.fit_transform(X_train) 
X_test = pca.transform(X_test) 
print("xtrain.SHAPE despues del PCA: ", X_train.shape)

# ------------------ Oversampling ---------------------------------#
# print("Group By: \n",df.groupby('Ocupacion').count())
print("Antes del resampling")
print("Xtrain: ", np.size(X_train))
print("Ytrain: ", np.size(Y_train))

#------------------Tecnicas de Resampling---------------------------#
# sm = RandomOverSampler(random_state = 0)
# sm = SMOTE(random_state = 0)
sm=ADASYN(random_state=0)
X_train, Y_train=sm.fit_sample(X_train, Y_train)

print("Despues del resampling")
print("Xtrain: ", np.size(X_train))
print("Ytrain: ", np.size(Y_train))

classifier=SVC(kernel='rbf', random_state=1)
classifier.fit(X_train, Y_train)
Y_pred=classifier.predict(X_test)

test_set["Predictions"]=Y_pred

cm=confusion_matrix(Y_test, Y_pred)
print("CM SVM: \n", cm)
accuracy=float(cm.diagonal().sum())/len(Y_test)

print("\nAccuracy Of SVM: ", accuracy, "\n")


# ---------------------------------- Clasificacion de instancias usando una Weighted KNN-------------------------------#
classifier=KNeighborsClassifier(n_neighbors=15)
classifier.fit(X_train, Y_train)

Y_pred=classifier.predict(X_test)

test_set["Predictions"]=Y_pred

cm=confusion_matrix(Y_test, Y_pred)
print("CM KNN: \n", cm)
accuracy=float(cm.diagonal().sum())/len(Y_test)

print("\nAccuracy Of WKNN: ", accuracy, "\n")

# ---------------------------------- Clasificacion de instancias usando una Random Forest-------------------------------#
classifier=RandomForestClassifier(
    n_estimators=100, max_depth=2, random_state=0)

classifier.fit(X_train, Y_train)

Y_pred=classifier.predict(X_test)


# print(classifier.feature_importances_)
cm=confusion_matrix(Y_test, Y_pred)
print("CM RF: \n", cm)
accuracy=float(cm.diagonal().sum())/len(Y_test)

print("\nAccuracy Of RF: ", accuracy, "\n")



# -------------------------------- Referencias:

# https://scikit-learn.org/stable/modules/svm.html
