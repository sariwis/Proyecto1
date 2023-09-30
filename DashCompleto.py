import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import re
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import BayesianEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import plotly.express as px
from pgmpy.inference import VariableElimination
from sklearn.metrics import accuracy_score, confusion_matrix

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server

# Define los estilos CSS personalizados para mejorar la apariencia del dashboard
app.layout = html.Div([
    html.Div([
        html.H1('Herramienta de Predicción de Graduación Universitaria', style={'textAlign': 'center', 'color': 'white', 'backgroundColor': '#D4AF37', 'padding': '20px'}),
        html.P('¡Felicitaciones por completar la educación secundaria! Esta herramienta utiliza tus datos personales y académicos para predecir tu probabilidad de graduación o retiro universitario. Responde las siguientes preguntas:', style={'textAlign': 'center', 'fontSize': '20px', 'fontWeight': 'bold', 'color': 'black'}),
    ], style={'backgroundColor': '#f2f2f2'}),
    
    html.Br(),

    html.Div([
        html.Div([
            html.Label('Estado Civil'),
            dcc.Dropdown(id='MS', options=[{'label': 'Soltero/a', 'value': 1}, {'label': 'Casado/a', 'value': 2}, {'label': 'Divorciado/a', 'value': 4}, {'label': 'Unión Libre', 'value': 5}], placeholder='Selecciona estado civil'),
        ], className='four columns', style={'marginTop': '10px'}),
        
        html.Div([
            html.Label('Género'),
            dcc.Dropdown(id='Gen', options=[{'label': 'Femenino', 'value': 0}, {'label': 'Masculino', 'value': 1}], placeholder='Selecciona género'),
        ], className='four columns', style={'marginTop': '10px'}),

        html.Div([
            html.Label('Edad (17-62 años)'),
            dcc.Input(id='Edad', type='number', placeholder='Ingresa tu edad', min=17, max=62),
        ], className='four columns', style={'marginTop': '10px'}),
    ], className='row'),

    html.Br(),

    html.Div([
        html.Div([
            html.Label('¿Cuál fue su nota previa a la universidad? (50-200)'),
            dcc.Input(id='Not_prev', type='number', placeholder='Ingresa tu nota', min=50, max=200),
        ], className='four columns', style={'marginTop': '10px'}),

        html.Div([
            html.Label('¿Cuál fue su calificación de admisión? (50-200)'),
            dcc.Input(id='Not_adm', type='number', placeholder='Ingresa tu nota', min=50, max=200),
        ], className='four columns', style={'marginTop': '10px'}),

        html.Div([
                html.Label('¿Usted o su familia asumirá una dedua para costear la universidad?'),
                dcc.Dropdown(id='Deuda', options=[{'label':'Sí', 'value':1},{'label':'No', 'value':0}], placeholder='Deudor'),
            ], className='four columns', style={'marginTop': '10px'}),

    ], className='row'),

    html.Br(),


    html.Div([
        html.Div([
            html.Label('Curso'),
            dcc.Dropdown(id='Curso', options=[{'label': 'Tecnologías de Producción de Biocombustibles', 'value': 1}, {'label': 'Diseño de Animación y Multimedia', 'value': 2 },{'label': 'Servicio Social (turno vespertino)', 'value': 3},{'label': 'Agronomía', 'value': 4},{'label': 'Diseño de Comunicación', 'value': 5},{'label': 'Enfermería Veterinaria', 'value': 6},{'label': 'Ingeniería Informática', 'value': 7},{'label': 'Equinocultura', 'value': 8},{'label': 'Gestión', 'value': 9},{'label': 'Servicio Social', 'value':10},{'label': 'Turismo', 'value':11},{'label': 'Enfermería', 'value':12},{'label': 'Higiene Oral', 'value':13},{'label': 'Dirección de Publicidad y Marketing', 'value':14},{'label': 'Periodismo y Comunicación', 'value':15},{'label': 'Educación Básica', 'value':16},{'label': 'Gestión (turno en la tarde)', 'value':17}], placeholder='Seleccione su curso'),
        ], className='six columns', style={'marginTop': '10px'}),
        
        html.Div([
            html.Label('¿El curso seleccionado fue su primera opción? (0-6)'),
            dcc.Input(id='Apl_order', type='number', placeholder='Ingresa la orden de aplicación', min=0, max=6),
            html.P('Nota: 0 significa que quedó en su opción preferida y 6 significa la menos preferida.', style={'fontSize': '14px', 'color': 'gray'}),
        ], className='six columns', style={'marginTop': '10px'})
    ], className='row'),

    html.Br(),

    ##

    html.Br(),

    html.Button('CALCULA TU RESULTADO', id='submit', n_clicks=0, style={'backgroundColor': '#D4AF37', 'color': 'white', 'fontSize': '18px'}),
    html.Br(),

    html.Div(id='output'),

    html.Br(),

    dcc.Graph(id='probability-plot'),  # Visualización de probabilidad de graduación o retiro
], className='container')

def by_pred(MS, AO, C, PQ, AG, D, G, AE):
    datos = 'Datos_FINAL.xlsx'
    df = pd.read_excel(datos)

    # Categorizar la variable target
    df['target'] = df['target'].apply(lambda x: 0 if x == "Dropout" else 1)
    df['AG'] = round(df['AG'])
    # Categorizar las calificaciones previas y de ingreso
    lim_Notas = [-1, 49, 99, 149, 201]
    labels = ['Malo', 'Regular', 'Buena', 'Excelente']
    df['PQ'] = pd.cut(df['PQ'], bins=lim_Notas, labels=labels)
    df['PQ'] = df['PQ'].replace({'Malo':1,'Regular':2,'Buena':3, 'Excelente':4})

    df['AG'] = pd.cut(df['AG'], bins=lim_Notas, labels=labels)
    df['AG'] = df['AG'].replace({'Malo':1,'Regular':2,'Buena':3, 'Excelente':4})

    # Categorizar edad
    lim_Edad = [16, 30, 45, 63]
    labels_Edad = ['Joven', 'Adulto', 'Mayor']
    df['AE'] = pd.cut(df['AE'], bins=lim_Edad, labels=labels_Edad)
    df['AE'] = df['AE'].replace({'Joven':1,'Adulto':2,'Mayor':3})

    # Categorizar los cursos
    df['C'] = df['C'].replace({"Biofuel Production Technologies":1,"Animation and Multimedia Design":2,"Social Service (evening attendance)":3,"Agronomy":4,"Communication Design":5,"Veterinary Nursing":6,"Informatics Engineering":7,"Equinculture":8,"Management":9,"Social Service":10,"Tourism":11,"Nursing":12,"Oral Hygiene":13,"Advertising and Marketing Management":14,"Journalism and Communication":15,"Basic Education":16, "Management (evening attendance)":17})

    modelo = BayesianNetwork([('AO', 'AE'), ('AG', 'AO'), ('MS', 'AO'), ('AG', 'C'), ('AG', 'PQ'), ('C', 'G'), ('C', 'PQ'), ('PQ', 'D'), ('PQ', 'AE'), ('AE', 'target'), ('D', 'target')])

    sample_train, sample_test = train_test_split(df, test_size=0.2, random_state=777)
    emv = MaximumLikelihoodEstimator(model = modelo, data = sample_train)

    modelo.fit(data=sample_train, estimator=MaximumLikelihoodEstimator)

    infer = VariableElimination(modelo)

    # Se extraen los valores reales
    y_real = sample_test["target"].values

    df2 = sample_test.drop(columns=['target'])
    y_p = modelo.predict(df2)

    accuracy = accuracy_score(y_real, y_p)
    print(accuracy)

    
    conf = confusion_matrix(y_real, y_p)

    # Extraer verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos
    VerdPos =conf[1, 1]
    FalsoPos = conf[0, 1]
    VerdNeg = conf[0, 0]
    FalsoNeg = conf[1, 0]

    print(VerdPos)
    print(FalsoPos)
    print(VerdNeg)
    print(FalsoNeg)

    print(conf)

    resp = infer.query(['target'], evidence={'MS': MS, 'AO': AO, 'C': C, 'PQ': PQ, 'AG': AG, 'D': D, 'G': G, 'AE': AE})

    return resp


@app.callback(
    Output('output', 'children'),
    Output('probability-plot', 'figure'),
    [Input('submit', 'n_clicks')],
    [State('MS', 'value'),
    State('Gen', 'value'),
    State('Edad', 'value'),
    State('Not_prev', 'value'),
    State('Not_adm', 'value'),
    State('Deuda', 'value'),
    State('Curso', 'value'),
    State('Apl_order', 'value')]
)
def update_output(n_clicks, MS, G, AE, PQ, AG, D, C, AO):
    try:
        if n_clicks > 0:
        
            lim_Edad = [16, 30, 45, 63]
            labels_Edad = [1, 2, 3]
            AE = pd.cut([AE], bins=lim_Edad, labels=labels_Edad)[0]

            lim_Notas = [-1, 49, 99, 149, 201]
            labels_Notas = [1, 2, 3, 4]
            PQ = pd.cut([PQ], bins=lim_Notas, labels=labels_Notas)[0]
            AG = pd.cut([AG], bins=lim_Notas, labels=labels_Notas)[0]

            probabilidad = by_pred(MS, AO, C, PQ, AG, D, G, AE)
            
            # Crea un gráfico de barras para visualizar la probabilidad
            df = pd.DataFrame({'Probabilidad': [probabilidad], 'Resultado': ['Probabilidad de Graduación']})

            prob_df = pd.DataFrame({'Categoría': ['Graduación', 'Retiro'], 'Probabilidad': [probabilidad.values[1], probabilidad.values[0]]})

            fig = px.bar(prob_df, x='Categoría', y='Probabilidad', text='Probabilidad', height=400,
                     labels={'Categoría': 'Resultado', 'Probabilidad': 'Probabilidad'},
                     color='Categoría', title='Probabilidad de Graduación vs. Probabilidad de Retiro')

            # Personaliza el diseño del gráfico
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(legend_title_text='Resultado')
            #fig = {'data': [{'x': ['Probabilidad de Graduación'], 'y': [probabilidad.values[1]], 'type': 'bar'}],'layout': {'xaxis': {'title': 'Resultado'}, 'yaxis': {'title': 'Probabilidad'}}}
            #fig = px.bar(df, x='Resultado', y='Probabilidad', text='Probabilidad', height=400)
            resp = probabilidad.values[1]*100
            return f'Probabilidad de Graduación: {round(resp, 2)}%', fig
    except Exception as e:
        return f'Error: {str(e)}', {}

#Se ejecuta la aplicación
if __name__ == '__main__':
    app.run_server(debug=True,port =8070)
        
