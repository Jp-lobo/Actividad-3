from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd 

# Cargar el dataset 
df = pd.read_csv("dataset_viajes.csv")

# Codificar las ubicaciones con valores numéricos
label_encoder = LabelEncoder()
df["Inicio"] = label_encoder.fit_transform(df["Inicio"])
df["Destino"] = label_encoder.fit_transform(df["Destino"])

# Seleccionar características (Inicio, Destino) y la variable objetivo (Tiempo)
X = df[["Inicio", "Destino"]]
y = df["Tiempo"]

# Dividir los datos en entrenamiento y prueba (80% de entrenamiento, 20% de prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar un modelo de regresión (Random Forest)
modelo = RandomForestRegressor(random_state=42)
modelo.fit(X_train, y_train)

# Evaluar el modelo
y_pred = modelo.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)#promedio de errores entre predicciones y valores reales
r2 = r2_score(y_test, y_pred)

# print(f"MAE: {mae}")
# print(f"R²: {r2}")

# Predicción para un nuevo par de nodos
nodo_inicio = label_encoder.transform(["Tocancipa"])[0]
nodo_destino = label_encoder.transform(["Zipaquira"])[0]
tiempo_predicho = modelo.predict([[nodo_inicio, nodo_destino]])

print(f"Tiempo estimado para ir de Tocancipá a Zipaquira: {tiempo_predicho[0]:.2f} minutos")
