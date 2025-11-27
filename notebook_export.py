!wget --no-cache -O init.py -q https://raw.githubusercontent.com/rramosp/ai4eng.v1/main/content/init.py
import init; init.init(force_download=False); init.get_weblink()

import os
os.environ['KAGGLE_CONFIG_DIR'] = '.'
!chmod 600 ./kaggle.json
!kaggle competitions download -c udea-ai-4-eng-20252-pruebas-saber-pro-colombia

!unzip udea*.zip > /dev/null

import pandas as pd
import numpy as np

z = pd.read_csv("train.csv")
print ("shape of loaded dataframe", z.shape)


z.head()

z = z[['F_EDUCACIONMADRE', 'E_VALORMATRICULAUNIVERSIDAD', 'RENDIMIENTO_GLOBAL']]
z.head()

z.F_EDUCACIONMADRE.value_counts().plot(kind='bar')

z.E_VALORMATRICULAUNIVERSIDAD.value_counts().plot(kind='bar')

from rlxutils import subplots
import matplotlib.pyplot as plt

c = sorted(z.E_VALORMATRICULAUNIVERSIDAD.value_counts().index)
for ax,ci in subplots(c, n_cols=4, usizex=4):
    zc = z[z.E_VALORMATRICULAUNIVERSIDAD==ci]
    zc.RENDIMIENTO_GLOBAL.value_counts()[['bajo', 'medio-bajo', 'medio-alto', 'alto']].plot(kind='bar')
    plt.title(ci)

c = sorted(z.F_EDUCACIONMADRE.value_counts().index)
for ax,ci in subplots(c, n_cols=4, usizex=4):
    zc = z[z.F_EDUCACIONMADRE==ci]
    zc.RENDIMIENTO_GLOBAL.value_counts()[['bajo', 'medio-bajo', 'medio-alto', 'alto']].plot(kind='bar')
    plt.title(ci)

sum(z.F_EDUCACIONMADRE.isna()), sum(z.E_VALORMATRICULAUNIVERSIDAD.isna())

# Reemplazo de valores faltantes por 'no info'
z.F_EDUCACIONMADRE.values[z.F_EDUCACIONMADRE.isna()] = 'no info'
z.E_VALORMATRICULAUNIVERSIDAD.values[z.E_VALORMATRICULAUNIVERSIDAD.isna()] = 'no info'

# Verificación de valores faltantes
sum(z.F_EDUCACIONMADRE.isna()), sum(z.E_VALORMATRICULAUNIVERSIDAD.isna())


# Mapa de conversión de categorías a valores numéricos
cmap = {
    'Entre 1 millón y menos de 2.5 millones': 1.75,
    'Entre 2.5 millones y menos de 4 millones': 3.25,
    'Menos de 500 mil': 0.25,
    'Entre 500 mil y menos de 1 millón': 0.75,
    'Entre 4 millones y menos de 5.5 millones': 4.75,
    'Más de 7 millones': 7.75,
    'Entre 5.5 millones y menos de 7 millones': 6.25,
    'No pagó matrícula': 0,
    'no info': -1
}

# Aplicación del mapa para convertir la columna a valores numéricos
z.E_VALORMATRICULAUNIVERSIDAD = np.array([cmap[i] for i in z.E_VALORMATRICULAUNIVERSIDAD])

# Verificación de la distribución de los valores convertidos
z.E_VALORMATRICULAUNIVERSIDAD.value_counts()

# Creación de una copia del dataset para no afectar el original
z = z.copy()

# Reemplazo de categorías ambiguas por 'no info'
z.F_EDUCACIONMADRE = ['no info' if i in ['No sabe', 'No Aplica'] else i for i in z.F_EDUCACIONMADRE.values]

# Verificación de la distribución de categorías
z.F_EDUCACIONMADRE.value_counts()

# Obtención de los valores únicos
x = z.F_EDUCACIONMADRE.values
F_EDUCACIONMADRE_vals = sorted(np.unique(x))

# Creación de codificación one-hot
F_EDUCACIONMADRE_onehot_vals = {val: np.eye(len(F_EDUCACIONMADRE_vals))[i] for i, val in enumerate(F_EDUCACIONMADRE_vals)}

# Visualización del diccionario one-hot
F_EDUCACIONMADRE_onehot_vals


# Aplicación de la codificación one-hot al dataset
F_EDUCACIONMADRE_onehot_enc = np.array([F_EDUCACIONMADRE_onehot_vals[i] for i in z.F_EDUCACIONMADRE])

# Visualización del array resultante
F_EDUCACIONMADRE_onehot_enc


# Conversión del array one-hot a DataFrame
F_EDUCACIONMADRE_df = pd.DataFrame(
    F_EDUCACIONMADRE_onehot_enc,
    columns=[f"F_EDUCACIONMADRE__{v}" for v in F_EDUCACIONMADRE_onehot_vals]
)

# Visualización del DataFrame resultante
F_EDUCACIONMADRE_df


# Integración del DataFrame one-hot al dataset original y eliminación de la columna original
z = pd.concat([F_EDUCACIONMADRE_df, z], axis=1).drop('F_EDUCACIONMADRE', axis=1)

# Verificación de la nueva forma del dataset
z.shape

z.head()


# Definición del nombre de la columna objetivo
y_col = 'RENDIMIENTO_GLOBAL'

# Mapa de conversión de categorías a valores numéricos
rmap = {'alto': 3, 'bajo': 0, 'medio-bajo': 1, 'medio-alto': 2}

# Aplicación del mapa a la columna objetivo
z[y_col] = [rmap[i] for i in z[y_col]]

# Vista previa del dataset
z.head()


# Ordenamiento de columnas
z = z[sorted(z.columns)]

# Separación de features y variable objetivo
X = z[[c for c in z.columns if c != y_col]].values
y = z[y_col].values

# Verificación de la forma de los arrays
X.shape, y.shape


from sklearn.model_selection import train_test_split

# División del dataset en entrenamiento y prueba (80% - 20%)
Xtr, Xts, ytr, yts = train_test_split(X, y, train_size=0.8)

# Verificación de la forma de los conjuntos resultantes
Xtr.shape, Xts.shape, ytr.shape, yts.shape


from sklearn.linear_model import LogisticRegression

# Creación del modelo de regresión logística
lr = LogisticRegression()

# Entrenamiento del modelo con los datos de entrenamiento
lr.fit(Xtr, ytr)


# Generación de predicciones con el modelo entrenado
preds_tr = lr.predict(Xtr)
preds_ts = lr.predict(Xts)

# Visualización de las primeras 10 predicciones de cada conjunto
print(preds_tr[:10])
print(preds_ts[:10])


import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from rlxutils import subplots

# Cálculo de la precisión en entrenamiento y prueba
np.mean(preds_tr == ytr), np.mean(preds_ts == yts)

# Generación de matrices de confusión
cm_tr = confusion_matrix(ytr, preds_tr)
cm_ts = confusion_matrix(yts, preds_ts)

# Normalización por clase
cm_tr = cm_tr / cm_tr.sum(axis=1).reshape(-1, 1)
cm_ts = cm_ts / cm_ts.sum(axis=1).reshape(-1, 1)

# Visualización de las matrices de confusión
for ax, i in subplots(2, usizex=4):
    if i == 0:
        sns.heatmap(cm_tr, annot=True)
        plt.title("Confusion matrix train")
    if i == 1:
        sns.heatmap(cm_ts, annot=True)
        plt.title("Confusion matrix test")
    plt.ylabel("true")
    plt.xlabel("predicted")


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from rlxutils import subplots

# Selección de la columna de interés y la variable objetivo
zh = z[['E_VALORMATRICULAUNIVERSIDAD', y_col]]

# Separación de features y variable objetivo
X = zh[[c for c in zh.columns if c != y_col]].values
y = zh[y_col].values
X.shape, y.shape

# División del dataset y entrenamiento del modelo pequeño
Xtr, Xts, ytr, yts = train_test_split(X, y, train_size=0.8)
lr_small = LogisticRegression()
lr_small.fit(Xtr, ytr)

# Predicciones
preds_tr = lr_small.predict(Xtr)
preds_ts = lr_small.predict(Xts)

# Precisión
np.mean(preds_tr == ytr), np.mean(preds_ts == yts)

# Matrices de confusión
cm_tr = confusion_matrix(ytr, preds_tr)
cm_ts = confusion_matrix(yts, preds_ts)

# Normalización por clase
cm_tr = cm_tr / cm_tr.sum(axis=1).reshape(-1, 1)
cm_ts = cm_ts / cm_ts.sum(axis=1).reshape(-1, 1)

# Visualización de las matrices de confusión
for ax, i in subplots(2, usizex=4):
    if i == 0:
        sns.heatmap(cm_tr, annot=True)
        plt.title("Confusion matrix train")
    if i == 1:
        sns.heatmap(cm_ts, annot=True)
        plt.title("Confusion matrix test")
    plt.ylabel("true")
    plt.xlabel("predicted")


# Carga del dataset de prueba
zt = pd.read_csv("test.csv")
zt_ids = zt['ID'].values
zt = zt[['F_EDUCACIONMADRE', 'E_VALORMATRICULAUNIVERSIDAD']]
print("Shape of loaded dataframe:", zt.shape)
zt.head()

# Reemplazo de valores faltantes por 'no info'
zt.F_EDUCACIONMADRE.values[zt.F_EDUCACIONMADRE.isna()] = 'no info'
zt.E_VALORMATRICULAUNIVERSIDAD.values[zt.E_VALORMATRICULAUNIVERSIDAD.isna()] = 'no info'

# Conversión de matrícula a valores numéricos
zt = zt.copy()
zt.E_VALORMATRICULAUNIVERSIDAD = np.array([cmap[i] for i in zt.E_VALORMATRICULAUNIVERSIDAD])

# Limpieza de categorías ambiguas en educación de la madre
zt.F_EDUCACIONMADRE = ['no info' if i in ['No sabe', 'No Aplica'] else i for i in zt.F_EDUCACIONMADRE.values]

# Codificación one-hot de educación de la madre
F_EDUCACIONMADRE_onehot_enc = np.array([F_EDUCACIONMADRE_onehot_vals[i] for i in zt.F_EDUCACIONMADRE])
F_EDUCACIONMADRE_df = pd.DataFrame(F_EDUCACIONMADRE_onehot_enc, columns=[f"F_EDUCACIONMADRE__{v}" for v in F_EDUCACIONMADRE_onehot_vals])

# Integración del one-hot al dataset de prueba y eliminación de la columna original
zt = pd.concat([F_EDUCACIONMADRE_df, zt], axis=1).drop('F_EDUCACIONMADRE', axis=1)
zt.shape

# Vista final del dataset de prueba
zt

# Preparación del array de features de prueba
X_test_data = zt[sorted(zt.columns)].values
X_test_data.shape

# Generación de predicciones con el modelo entrenado
preds_test_data = lr.predict(X_test_data)


import pandas as pd

# Mapeo inverso de predicciones a etiquetas textuales
rmapi = {v: k for k, v in rmap.items()}
text_preds_test_data = [rmapi[i] for i in preds_test_data]

# Creación del DataFrame de submission
submission = pd.DataFrame([zt_ids, text_preds_test_data], index=['ID', 'RENDIMIENTO_GLOBAL']).T

# Guardado a CSV
submission.to_csv("my_submission.csv", index=False)

# Vista previa
submission.head()
submission.shape


# Comando para enviar el archivo a la competencia Kaggle
!kaggle competitions submit -c udea-ai-4-eng-20252-pruebas-saber-pro-colombia \
-f my_submission.csv -m "raul ramos submission with linear model"

