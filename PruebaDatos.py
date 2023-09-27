import dash

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

def by_pred(MS, AO, C, PQG, MQ, FQ, MO, FO, AG, D, G, SH, AE):
    datos = 'DF.csv'

    df = pd.read_csv(datos, names = ['MS', 'AO', 'C', 'PQG', 'MQ', 'FQ', 'MO', 'FO', 'AG', 'D', 'G', 'SH', 'AE', "target"])

    df['target'] = df['target'].apply(lambda x: 0 if x == "Dropout" else 1)

    modelo = BayesianNetwork([("MS", "AO"), ("AO", "C"), ("C", "target"), ("PQG", "AG"), ("MQ", "SH"), ("FQ", "SH"), ("AG", "C"), ("D", "target"), ("G", "AO"), ("SH", "target"), ("AE", "target"), ("MO", "AO"), ("FO", "AO")])

    modelo.fit(df, estimator=MaximumLikelihoodEstimator)

    infer = VariableElimination(modelo)

    resp = infer.query(['target'], evidence={'MS': MS, 'AO': AO, 'C': C, 'PQG': PQG, 'MQ': MQ, 'FQ': FQ, 'MO': MO, 'FO': FO, 'AG': AG, 'D': D, 'G': G, 'SH': SH, 'AE': AE})

    return resp

#RECORDAR PONER VALORES DENTRO DE LOS RANGOS


#DASH

#app.layout = html.Div([html.H1('Predicción de éxito o fracaso de estudiantes recien graduados del colegio', style={'text-align':'center', 'color':'Black', 'font-weight': 'bold', 'background':'#D4AF37', 'z-index':'100'})])
#Se crear la aplicación en dash
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div([html.H1('¡Bienvenidos a la herramienta de Predicción de Graduación Universitaria!', style={'text-align':'center', 'color':'White', 'font-weight': 'bold', 'background':'#D4AF37', 'z-index':'100'}),
                       
        html.Div([html.H6('¡Felicitaciones por haber completado tu educación secundaria y dar el paso hacia la siguiente etapa de tu vida! Sabemos que tomar decisiones sobre tu futuro educativo es emocionante, pero también puede ser un desafío. Esta herramienta está diseñada para utilizar tus datos personales y académicos para predecir si es más probable que te gradues de la universidad o te retires. Queremos brindarte una visión clara de tus probabilidades y ayudarte a tomar las mejores decisiones de forma informada.')], style={'color': 'black', 'font-weight': 'bold'}),

        html.Br(),

        html.Div([html.P('Por favor responde las siguientes preguntas:')], style={'font-size':'20px','color':'black', 'font-weight':'bold'}),

        html.Br(),

        dcc.Store(id='n-clicks', data=0), 
        html.Div([

            html.Div([
                html.Label('¿Cuál es su estado civil?'),
                dcc.Dropdown(id='ME', options=[{'label': 'Soltero/a', 'value': 1}, {'label': 'Casado/a', 'value': 2},{'label': 'Viudo/a', 'value': 3},{'label': 'Divorciado/a', 'value': 4},{'label': 'Unión Libre', 'value': 5},{'label': 'Legalmente Separado/a', 'value': 6}], placeholder='Estado Civil'),
            ], className='four columns', style={'margin-top': '10px'}),
            

            html.Div([
                html.Label('¿Cuál es el nivel de educación de su madre?'),
                dcc.Dropdown(id='Edu_ma', options=[{'label': 'Educación Secundaria - 12º Año de Estudios o Equivalente', 'value': 1}, {'label': 'Educación Superior - Licenciatura', 'value': 2}, {'label': 'Educación Superior - Grado', 'value': 3}, {'label': 'Educación Superior - Máster', 'value': 4}, {'label': 'Educación Superior - Doctorado', 'value': 5}, {'label': 'Frecuencia de Educación Superior', 'value': 6}, {'label': '12º Año de Estudios - No Completado', 'value': 9}, {'label': '11º Año de Estudios - No Completado', 'value': 10}, {'label': '7º Año de Estudios (Antiguo)', 'value': 11}, {'label': 'Otro - 11º Año de Estudios', 'value': 12}, {'label': '10º Año de Estudios', 'value': 14}, {'label': 'Curso de Comercio General', 'value': 18}, {'label': 'Educación Básica - 3er Ciclo (9º/10º/11º Año) o Equivalente', 'value': 19}, {'label': 'Curso Técnico-Profesional', 'value': 22}, {'label': '7º Año de Estudios', 'value': 26}, {'label': '2º Ciclo del Curso de Educación Secundaria General', 'value': 27}, {'label': '9º Año de Estudios - No Completado', 'value': 29}, {'label': '8º Año de Estudios', 'value': 30}, {'label': 'Desconocido', 'value': 34}, {'label': 'No Puede Leer ni Escribir', 'value': 35}, {'label': 'Puede Leer sin haber completado el 4º Año de Estudios', 'value': 36}, {'label': 'Educación Básica - 1er Ciclo (4º/5º Año) o Equivalente', 'value': 37}, {'label': 'Educación Básica - 2º Ciclo (6º/7º/8º Año) o Equivalente', 'value': 38}, {'label': 'Curso de Especialización Tecnológica', 'value': 39}, {'label': 'Educación Superior - Grado (1er Ciclo)', 'value': 40}, {'label': 'Curso de Estudios Superiores Especializados', 'value': 41}, {'label': 'Curso Técnico Superior Profesional', 'value': 42}, {'label': 'Educación Superior - Máster (2º Ciclo)', 'value': 43}, {'label': 'Educación Superior - Doctorado (3er Ciclo)', 'value': 44}, ], placeholder='Educación Madre'),
            ], className='four columns', style={'margin-top': '10px'}),

            html.Div([
                html.Label('¿Cuál es el nivel de educación de su padre?'),
                dcc.Dropdown(id='Edu_pa', options=[{'label': 'Educación Secundaria - 12º Año de Estudios o Equivalente', 'value': 1}, {'label': 'Educación Superior - Licenciatura', 'value': 2}, {'label': 'Educación Superior - Grado', 'value': 3}, {'label': 'Educación Superior - Máster', 'value': 4}, {'label': 'Educación Superior - Doctorado', 'value': 5}, {'label': 'Frecuencia de Educación Superior', 'value': 6}, {'label': '12º Año de Estudios - No Completado', 'value': 9}, {'label': '11º Año de Estudios - No Completado', 'value': 10}, {'label': '7º Año de Estudios (Antiguo)', 'value': 11}, {'label': 'Otro - 11º Año de Estudios', 'value': 12}, {'label': '10º Año de Estudios', 'value': 14}, {'label': 'Curso de Comercio General', 'value': 18}, {'label': 'Educación Básica - 3er Ciclo (9º/10º/11º Año) o Equivalente', 'value': 19}, {'label': 'Curso Técnico-Profesional', 'value': 22}, {'label': '7º Año de Estudios', 'value': 26}, {'label': '2º Ciclo del Curso de Educación Secundaria General', 'value': 27}, {'label': '9º Año de Estudios - No Completado', 'value': 29}, {'label': '8º Año de Estudios', 'value': 30}, {'label': 'Desconocido', 'value': 34}, {'label': 'No Puede Leer ni Escribir', 'value': 35}, {'label': 'Puede Leer sin haber completado el 4º Año de Estudios', 'value': 36}, {'label': 'Educación Básica - 1er Ciclo (4º/5º Año) o Equivalente', 'value': 37}, {'label': 'Educación Básica - 2º Ciclo (6º/7º/8º Año) o Equivalente', 'value': 38}, {'label': 'Curso de Especialización Tecnológica', 'value': 39}, {'label': 'Educación Superior - Grado (1er Ciclo)', 'value': 40}, {'label': 'Curso de Estudios Superiores Especializados', 'value': 41}, {'label': 'Curso Técnico Superior Profesional', 'value': 42}, {'label': 'Educación Superior - Máster (2º Ciclo)', 'value': 43}, {'label': 'Educación Superior - Doctorado (3er Ciclo)', 'value': 44}, ], placeholder='Educación Padre'),
            ], className='four columns', style={'margin-top': '10px'}),
        ], className='row'),
        
        html.Br(),

        html.Div([

            html.Div([
                html.Label('¿Cuál es la profesión de su madre?'),
                dcc.Dropdown(id='Prof_ma', options=[{'label': 'Estudiante', 'value': 0}, {'label': 'Representantes del Poder Legislativo y Órganos Ejecutivos, Directores, Gerentes y Directivos Ejecutivos', 'value': 1}, {'label': 'Especialistas en Actividades Intelectuales y Científicas', 'value': 2}, {'label': 'Técnicos y Profesionales de Nivel Intermedio', 'value': 3}, {'label': 'Personal Administrativo', 'value': 4}, {'label': 'Trabajadores de Servicios Personales, Seguridad y Ventas', 'value': 5}, {'label': 'Agricultores y Trabajadores Especializados en Agricultura, Pesca y Silvicultura', 'value': 6}, {'label': 'Trabajadores Especializados en Industria, Construcción y Artesanos', 'value': 7}, {'label': 'Operadores de Instalaciones y Máquinas y Trabajadores de Ensamblaje', 'value': 8}, {'label': 'Trabajadores No Cualificados', 'value': 9}, {'label': 'Profesiones de las Fuerzas Armadas', 'value': 10}, {'label': 'Otra Situación', 'value': 90}, {'label': '(en blanco)', 'value': 99}, {'label': 'Profesionales de la Salud', 'value': 122}, {'label': 'Profesores', 'value': 123}, {'label': 'Especialistas en Tecnologías de la Información y Comunicación (TIC)', 'value': 125}, {'label': 'Técnicos de Ciencias e Ingeniería de Nivel Intermedio', 'value': 131}, {'label': 'Técnicos y Profesionales de Nivel Intermedio de la Salud', 'value': 132}, {'label': 'Técnicos de Nivel Intermedio en Servicios Legales, Sociales, Deportivos, Culturales y Similares', 'value': 134}, {'label': 'Trabajadores de Oficina, Secretarios en General y Operadores de Procesamiento de Datos', 'value': 141}, {'label': 'Operadores de Datos, Contabilidad, Estadística, Servicios Financieros y Registros Relacionados', 'value': 143}, {'label': 'Otro Personal de Apoyo Administrativo', 'value': 144}, {'label': 'Trabajadores de Servicios Personales', 'value': 151}, {'label': 'Vendedores', 'value': 152}, {'label': 'Técnicos y Profesionales de Nivel Intermedio de la Salud', 'value': 153}, {'label': 'Trabajadores Especializados en Construcción y Similares, Excepto Electricistas', 'value': 171}, {'label': 'Trabajadores Especializados en Impresión, Fabricación de Instrumentos de Precisión, Joyeros, Artesanos y Similares', 'value': 173}, {'label': 'Trabajadores en Procesamiento de Alimentos, Carpintería, Ropa y Otras Industrias y Oficios', 'value': 175}, {'label': 'Trabajadores de Limpieza', 'value': 191}, {'label': 'Trabajadores No Cualificados en Agricultura, Producción Animal, Pesca y Silvicultura', 'value': 192}, {'label': 'Trabajadores No Cualificados en la Industria Extractiva, Construcción, Manufactura y Transporte', 'value': 193}, {'label': 'Asistentes de Preparación de Comidas', 'value': 194}], placeholder='Ocupación Madre'),
            ], className='four columns', style={'margin-top': '10px'}),

            html.Div([
                html.Label('¿Cuál es la profesión de su padre?'),
                dcc.Dropdown(id='Prof_pa', options=[{'label': 'Estudiante', 'value': 0}, {'label': 'Representantes del Poder Legislativo y Órganos Ejecutivos, Directores, Gerentes y Directivos Ejecutivos', 'value': 1},{'label': 'Especialistas en Actividades Intelectuales y Científicas', 'value': 2},{'label': 'Técnicos y Profesionales de Nivel Intermedio', 'value': 3},{'label': 'Personal Administrativo', 'value': 4},{'label': 'Trabajadores de Servicios Personales, Seguridad y Ventas', 'value': 5},{'label': 'Agricultores y Trabajadores Especializados en Agricultura, Pesca y Silvicultura', 'value': 6},{'label': 'Trabajadores Especializados en Industria, Construcción y Artesanos', 'value': 7},{'label': 'Operadores de Instalaciones y Máquinas y Trabajadores de Ensamblaje', 'value': 8},{'label': 'Trabajadores No Cualificados', 'value': 9},{'label': 'Profesiones de las Fuerzas Armadas', 'value': 10},{'label': 'Otra Situación', 'value': 90},{'label': '(en blanco)', 'value': 99},{'label': 'Oficiales de las Fuerzas Armadas', 'value': 101},{'label': 'Sargentos de las Fuerzas Armadas', 'value': 102},{'label': 'Otro personal de las Fuerzas Armadas', 'value': 103},{'label': 'Directores de Servicios Administrativos y Comerciales', 'value': 112},{'label': 'Directores de Hoteles, Restaurantes, Comercio y Otros Servicios', 'value': 114},{'label': 'Especialistas en Ciencias Físicas, Matemáticas, Ingeniería y Técnicas Afines', 'value': 121},{'label': 'Profesionales de la Salud', 'value': 122},{'label': 'Profesores', 'value': 123},{'label': 'Especialistas en Finanzas, Contabilidad, Organización Administrativa, Relaciones Públicas y Comerciales', 'value': 124},{'label': 'Técnicos de Ciencias e Ingeniería de Nivel Intermedio', 'value': 131},{'label': 'Técnicos y Profesionales de Nivel Intermedio de la Salud', 'value': 132},{'label': 'Técnicos de Nivel Intermedio en Servicios Legales, Sociales, Deportivos, Culturales y Similares', 'value': 134},{'label': 'Técnicos en Tecnologías de la Información y la Comunicación (TIC)', 'value': 135},{'label': 'Trabajadores de Oficina, Secretarios en General y Operadores de Procesamiento de Datos', 'value': 141},{'label': 'Operadores de Datos, Contabilidad, Estadística, Servicios Financieros y Registros Relacionados', 'value': 143},{'label': 'Otro Personal de Apoyo Administrativo', 'value': 144},{'label': 'Trabajadores de Servicios Personales', 'value': 151},{'label': 'Vendedores', 'value': 152},{'label': 'Trabajadores de Cuidado Personal y Similares', 'value': 153},{'label': 'Personal de Servicios de Protección y Seguridad', 'value': 154},{'label': 'Agricultores y Trabajadores Especializados en Agricultura y Producción de Animales Orientados al Mercado', 'value': 161},{'label': 'Agricultores, Ganaderos, Pescadores, Cazadores y Recolectores, Subsistencia', 'value': 163},{'label': 'Trabajadores Especializados en Construcción y Similares, Excepto Electricistas', 'value': 171},{'label': 'Trabajadores Especializados en Metalurgia, Trabajo del Metal y Similares', 'value': 172},{'label': 'Trabajadores Especializados en Electricidad y Electrónica', 'value': 174},{'label': 'Trabajadores en Procesamiento de Alimentos, Carpintería, Ropa y Otras Industrias y Oficios', 'value': 175},{'label': 'Operadores de Plantas Fijas y Máquinas', 'value': 181},{'label': 'Trabajadores de Ensamblaje', 'value': 182},{'label': 'Conductores de Vehículos y Operadores de Equipos Móviles', 'value': 183},{'label': 'Trabajadores No Cualificados en Agricultura, Producción Animal, Pesca y Silvicultura', 'value': 192},{'label': 'Trabajadores No Cualificados en la Industria Extractiva, Construcción, Manufactura y Transporte', 'value': 193 },{'label': 'Asistentes de Preparación de Comidas', 'value': 194},{'label': 'Vendedores Ambulantes (excepto Alimentos) y Proveedores de Servicios Callejeros', 'value': 195}], placeholder='Ocupación Padre'),
            ], className='four columns', style={'margin-top': '10px'}),

            html.Div([
                html.Label('¿Cuál es su género?'),
                dcc.Dropdown(id='Gen', options=[{'label':'Femenino', 'value':0}, {'label':'Masculino','value':1}], placeholder = 'Género'),
            ], className='four columns', style={'margin-top': '10px'}),

        ], className='row'),

        html.Br(),

        html.Div([

            html.Div([
                html.Label('¿Cuántos años tiene? (17-62 años)'),
                dcc.Input(id='Edad', type = 'number', placeholder = 'Ingrese su edad', min = 17, max=62),
            ], className='four columns', style={'margin-top': '10px'}),

            html.Div([
                html.Label('¿Cuál fue su nota previa a la universidad? (0-200)'),
                dcc.Input(id='Not_prev', type = 'number', placeholder = 'Ingrese su nota', min = 0, max=200),
            ], className='four columns', style={'margin-top': '10px'}),

            html.Div([
                html.Label('¿Cuál fue su calificación de admisión? (0-200)'),
                dcc.Input(id='Not_adm', type = 'number', placeholder = 'Ingrese su nota', min = 0, max=200),
            ], className='four columns', style={'margin-top': '10px'}),

        ], className='row'),
        
        html.Br(),

        html.Div([

            html.Div([
            html.Label('¿En qué curso quedó inscrito?'),
            dcc.Dropdown(id='C', options=[{'label': 'Tecnologías de Producción de Biocombustibles', 'value': 33}, {'label': 'Diseño de Animación y Multimedia', 'value': 171 },{'label': 'Servicio Social (turno vespertino)', 'value': 8014},{'label': 'Agronomía', 'value': 9003},{'label': 'Diseño de Comunicación', 'value': 9070},{'label': 'Enfermería Veterinaria', 'value': 9085},{'label': 'Ingeniería Informática', 'value': 9119},{'label': 'Equinocultura', 'value': 9130},{'label': 'Gestión', 'value': 9147},{'label': 'Servicio Social', 'value':9238},{'label': 'Turismo', 'value':9254},{'label': 'Enfermería', 'value':9500},{'label': 'Higiene Oral', 'value':9556},{'label': 'Dirección de Publicidad y Marketing', 'value':9670},{'label': 'Periodismo y Comunicación', 'value':9773},{'label': 'Educación Básica', 'value':9853},{'label': 'Gestión (turno en la tarde)', 'value':9991}], placeholder='Curso'),
            ], className='six columns', style={'margin-top': '10px'}),



            html.Div([
                html.Label('¿El curso seleccionado fue su primera opción? (0-9)'),
                html.P('Aquí se debe indicar que 0 es que quedaron en su opción preferida y 9 su menos preferida',
                   style={'border': '1px solid white', 'padding': '10px', 'border-radius': '10px', 'margin-top': '10px', 'margin-bottom': '10px'}),
                dcc.Input(id='Apl_order', type='number', placeholder='Orden de aplicación',min='0', max='9'),
            ], className='six columns', style={'margin-top': '10px'}),
        ], className='row'),
        
        
        html.Br(),

        html.Div([

            html.Div([
                html.Label('¿Usted será deudor?'),
                dcc.Dropdown(id='Deuda', options=[{'label':'Sí', 'value':1},{'label':'No', 'value':0}], placeholder='Deudor'),
            ], className='six columns', style={'margin-top': '10px'}),
            

            html.Div([
                html.Label('¿Estará Becado?'),
                dcc.Dropdown(id='Beca', options=[{'label':'Sí', 'value':1},{'label':'No', 'value':0}], placeholder='Becado'),
            ], className='six columns', style={'margin-top': '10px'}),
        ], className='row'),

        html.Br(),
        html.Br(),
        html.Br(),
        html.Div( id='output'),
        html.Button('¡CALCULA TU RESULTADO!', id='submit', n_clicks=0),
        html.Br(),
        html.Br(),  
        
        
        ], className='container', style={'font-family': 'system-ui', 'background-color': '#f2f2f2'})




#Se ejecuta la aplicación
if __name__ == '__main__':
    app.run_server(debug=True,port =8070)


