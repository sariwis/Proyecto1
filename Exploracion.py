import pandas as pd
import matplotlib.pyplot as plt

datos = 'data_original.xlsx'
df = pd.read_excel(datos)

# Nombre de la columna a graficar
N= 'Nacionality'

# Crear el histograma
plt.hist(df[N], bins=10) 
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Histogram of ' + N)
plt.show()


# Nombre de la columna a graficar
Ed= 'Educational special needs'

# Crear el histograma
plt.hist(df[Ed], bins=10) 
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Histogram of ' + Ed)
plt.show()

# Nombre de la columna a graficar
Sc= 'Scholarship holder'

# Crear el histograma
plt.hist(df[Sc], bins=10) 
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Histogram of ' + Sc)
plt.show()

# Nombre de la columna a graficar
Sc= 'Scholarship holder'

# Crear el histograma
plt.hist(df[Sc], bins=10) 
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Histogram of ' + Sc)
plt.show()

# Nombre de la columna a graficar
I= 'International'

# Crear el histograma
plt.hist(df[I], bins=10) 
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Histogram of ' + I)
plt.show()

# Nombre de la columna a graficar
T= 'Target'

# Crear el histograma
plt.hist(df[T], bins=10) 
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Histogram of ' + T)
plt.show()

# Crear Piechart para edades
bins = [0, 18.9, 20.9, 25.9, 30.9, 39.9, df["Age at enrollment"].max()]
labels = ['17-18 años', '19-20 años', '21-25 años', '26-30 años', '31-39 años', '>40 años']

df['Age Group'] = pd.cut(df['Age at enrollment'], bins=bins, labels=labels, include_lowest=True)

age_group_counts = df['Age Group'].value_counts()

ordered_age_group_counts = {
    '17-18 años': age_group_counts['17-18 años'],
    '19-20 años': age_group_counts['19-20 años'],
    '21-25 años': age_group_counts['21-25 años'],
    '26-30 años': age_group_counts['26-30 años'],
    '31-39 años': age_group_counts['31-39 años'],
    '>40 años': age_group_counts['>40 años']
}

fig, ax = plt.subplots(figsize=(8, 6))
wedges, texts, autotexts = ax.pie(ordered_age_group_counts.values(), labels=None, autopct='', startangle=90, wedgeprops={'linewidth': 1, 'edgecolor': 'gray'})
ax.axis('equal') 
plt.title('Distribución de edades')
legend_labels = [f'{label}: {count}' for label, count in ordered_age_group_counts.items()]
legend = plt.legend(legend_labels, loc='center left', bbox_to_anchor=(-0.2, 0.5))

for text in legend.get_texts():
    text.set_fontsize(8)

plt.show()



