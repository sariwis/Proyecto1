import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

datos = 'Datos_Iniciales.xlsx'

#Dataframe
df = pd.read_excel(datos)

#Cambiamos la última columna 
df['Target'] = df['Target'].apply(lambda x: 0 if x == "Dropout" else 1)

#Seleccionamos solo los que son numéricos
df_numeric = df.select_dtypes(include=['number'])

# Calculamos la matriz de correlación
correlation_matrix = df_numeric.corr()

# Configuramos los tamaños
plt.figure(figsize=(12, 10))

# Creamos el mapa de calor
sns.heatmap(correlation_matrix, annot=True, cmap="Spectral", linewidths=.5)

# Título del diagrama
plt.title("Mapa de calor de la matriz de correlación")

# Muestra el mapa de calor
plt.show()


datos2 = 'Factores_Finales.xlsx'

#Dataframe
df = pd.read_excel(datos2)

#Cambiamos la última columna 
df['Target'] = df['Target'].apply(lambda x: 0 if x == "Dropout" else 1)

#Seleccionamos solo los que son numéricos
df_numeric = df.select_dtypes(include=['number'])

# Calculamos la matriz de correlación
correlation_matrix = df_numeric.corr()

# Configuramos los tamaños
plt.figure(figsize=(12, 10))

# Creamos el mapa de calor
sns.heatmap(correlation_matrix, annot=True, cmap="Spectral", linewidths=.5)

# Título del diagrama
plt.title("Mapa de calor de la matriz de correlación de los factores que escogimos")

# Muestra el mapa de calor
plt.show()


import networkx as nx
import matplotlib.pyplot as plt

# Crear un grafo dirigido
G = nx.DiGraph()

# Definir las variables y sus correlaciones
correlations = {
    'T': ['D', 'G', 'Age'],
    'D': ['Scol', 'Prev'],
    'G': ['AO', 'C', 'Scol'],
    'Age': ['AO', 'Prev', 'Scol'],
    'Scol': ['MS'],
    'Prev': ['C', 'AG', 'MQ', 'FQ'],
    'AO': ['MS', 'AG'],
    'C': ['AG'],
    'MS': ['Scol'],
    'MQ': ['FO', 'MO'],
    'FQ': ['FO', 'MO'],
}

# Agregar nodos y bordes al grafo según las correlaciones
for variable, correlaciones in correlations.items():
    G.add_node(variable)
    for correlacion in correlaciones:
        G.add_edge(correlacion, variable)

# Dibujar el grafo
pos = nx.spring_layout(G, seed=42)  # Puedes ajustar el algoritmo de diseño según tus preferencias
nx.draw(G, pos, with_labels=True, node_size=1000, node_color='lightblue', font_size=10, font_color='black', font_weight='bold', arrowsize=20)
plt.title("Red Bayesiana basada en correlaciones")
plt.show()





