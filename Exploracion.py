import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns

datos = 'Datos_Final.xlsx'

df = pd.read_excel(datos)

# Nombre de la columna a graficar
columna_nombre = 'Admission grade'

# Crear el histograma
plt.hist(df[columna_nombre], bins=10)  # Puedes ajustar el número de bins según tus preferencias
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Histograma de Nota ' + columna_nombre)
plt.show()


