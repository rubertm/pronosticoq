import dash
import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Input,Output, State #a
from dash import no_update #a

import xarray as xr
import numpy as np
import pandas as pd
from pandas import Series
#from numpy.random import randn

#import matplotlib.pyplot as plt
#import mpl_toolkits.basemap as Basemap
import cftime
#from matplotlib.colors import BoundaryNorm
#from matplotlib.cm import ScalarMappable
#from mpl_toolkits.basemap import Basemap

import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from PIL import Image
#import json
import base64

from sklearn.model_selection import train_test_split
import sklearn
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, RidgeCV
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.feature_selection import chi2
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.datasets import make_classification

#------------------------------------------------------------------

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#------------------------------------------------------------------

app = dash.Dash(__name__)
server = app.server

#------------------------------------------------------------------

app.layout = html.Div([

    html.H1("Proyecto Final Pronóstico de Caudal (Predicción Ciega)", style={'text-align': 'center'}),
    # html.Br(),
    html.H5("Por: Carlos Saldarraiga, Rubert Montes, Profesor:Julián Rojo", style={'text-align': 'center'}),
    # html.Br(),
    html.H6("Curso de Actualización en Hidroclimatología", style={'text-align': 'center'}),
    html.Br(),
    html.H3("1. Correlación de SST y Caudal del Afluente", style={'text-align': 'left'}),
    dcc.Markdown('''
#### Kaplan Extended SST: Anomalías de SST (Sea Surface Temperature) globales en cuadrícula desde 1856-al presente derivadas de datos de SST de la Oficina Meteorológica del Reino Unido
'''),

    html.Br(),

    # Grafico de macrovariables hidroclimaticas
    # html.Div([
    # dcc.Dropdown(id="slct_mes",
    # options=[
    # {"label": "enero", "value": "0001-01-01"},
    # {"label": "febrero", "value": "0001-02-01"},
    # {"label": "marzo", "value": "0001-03-01"},
    # {"label": "abril", "value": "0001-04-01"},
    # {"label": "mayo", "value": "0001-05-01"},
    # {"label": "junio", "value": "0001-06-01"},
    # {"label": "julio", "value": "0001-07-01"},
    # {"label": "agosto", "value": "0001-08-01"},
    # {"label": "septiembre", "value": "0001-09-01"},
    # {"label": "octubre", "value": "0001-10-01"},
    # {"label": "noviembre", "value": "0001-11-01"},
    # {"label": "diciembre", "value": "0001-12-01"}],
    # multi=False,
    # value="0001-01-01")],
    # style={'width': "40%"}
    # ),

    # html.Div(id='output_container', children=[]),
    # html.Br(),

    # html.Div([
    # dcc.Graph(id='my_bee_map')
    #        ]),
    # html.Br(),
    # html.Br(),
    # html.Br(),
    # Grafico de correlaciones de OLR

    html.Div([
        html.Div([

            dcc.Dropdown(id="slct_mes3",
                         options=[
                             {"label": "enero", "value": 0},
                             {"label": "febrero", "value": 1},
                             {"label": "marzo", "value": 2},
                             {"label": "abril", "value": 3},
                             {"label": "mayo", "value": 4},
                             {"label": "junio", "value": 5},
                             {"label": "julio", "value": 6},
                             {"label": "agosto", "value": 7},
                             {"label": "septiembre", "value": 8},
                             {"label": "octubre", "value": 9},
                             {"label": "noviembre", "value": 10},
                             {"label": "diciembre", "value": 11}],
                         multi=False,
                         value=0),

            # html.Div(id='container3', children=[]),
            # html.Br(),
        ],
            style={'width': '40%', 'float': 'right', 'display': 'inline-block'}  # 'width': "40%"
        ),
        html.Div([
            dcc.Dropdown(id="slct_mes2",
                         options=[
                             {"label": "enero", "value": 0},
                             {"label": "febrero", "value": 1},
                             {"label": "marzo", "value": 2},
                             {"label": "abril", "value": 3},
                             {"label": "mayo", "value": 4},
                             {"label": "junio", "value": 5},
                             {"label": "julio", "value": 6},
                             {"label": "agosto", "value": 7},
                             {"label": "septiembre", "value": 8},
                             {"label": "octubre", "value": 9},
                             {"label": "noviembre", "value": 10},
                             {"label": "diciembre", "value": 11}],
                         multi=False,
                         value=0),
            # html.Div(id='container2', children=[]),
        ],
            style={'width': '40%', 'display': 'inline-block'}
        ),
    ], style={
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '10px 5px'
    }),

    html.Div(id='container2', children=[]),

    html.Br(),

    html.Div([
        dcc.Graph(id='map2')
    ]),

    html.Br(),

    html.H3("2. Dataset con información de los años (1975 - 2019), SST mínima, SST máxima y QRGII",
            style={'text-align': 'left'}),

    # html.Br(),

    html.Div(id='container3', children=[]),
    html.Div(id='container4', children=[]),
    html.Div(id='container5', children=[]),
    html.Div(id='container6', children=[]),
    html.Div(id='container7', children=[]),
    html.Div(id='container8', children=[]),
    html.Div(id='container9', children=[]),
    html.Div(id='container10', children=[]),
    html.Br(),

    html.Div([
        dcc.Graph(id='map23'),
    ], style={'width': '90%', 'padding': '0px 20px 0px 50px'}),

    html.Br(),

    html.H3("3. Modelo de Regresión Líneal Multiple", style={'text-align': 'left'}),
    dcc.Markdown('''
#### La regresión lineal múltiple esta configurada de la siguiente manera:  variable dependiente o respuesta (q - caudal afluente en lps), y variables independientes o predictores (Año, SST mínima y SST máxima).
'''),

    # html.Br(),

    html.Div([
        dcc.Graph(id='map3')
    ]),

    html.Div(id='container11', children=[]),
    html.Div(id='container12', children=[]),
    html.Div(id='container13', children=[]),
    html.Div(id='container14', children=[]),

    html.Br(),

    html.H4("4. Curva ROC con AUC score", style={'text-align': 'left'}),
    dcc.Markdown('''
#### Curva característica de operación del receptor, o curva ROC, ilustra la capacidad de diagnóstico de un sistema clasificador de caudal en tres rangos: bajo lo normal (0 - percentil 33), normal (percentil 33 al 66), y encima de lo normal (percentil 66 al 100). La curva ROC ilustra la tasa de verdaderos positivos (TPR - probabilidad de detección) contra la tasa de falsos positivos (FPR) en el rango de datos. El área bajo la curva (AUC) es igual a la probabilidad de que un clasificador clasifique una instancia positiva elegida al azar (asumiendo que 'positivo' se clasifica más alto que ' negativo').
'''),

    # html.Br(),

    html.Div([
        dcc.Graph(id='map4')
    ]),

    html.H4("5. Estimación del caudal mensual en periodo de análisis", style={'text-align': 'left'}),

    dcc.Markdown('''
#### En el siguiente recuadro introduzca el valor del año asignado a la variable independiente.
'''),

    html.Div([
        dcc.Input(id='ano', type='number', value=0)
    ], style={'text-align': 'left', 'padding': '0px 50px 0px 50px'}),

    html.Div(id='container15', children=[], style={'text-align': 'left', 'padding': '0px 20px 0px 50px'}),  # a

    html.Br(),

    dcc.Markdown('''
#### En el siguiente recuadro introduzca el valor del SST mínima del mes asignado a la variable independiente.
'''),

    html.Div([
        dcc.Input(id='sst_min', type='number', value=0)
    ], style={'text-align': 'left', 'padding': '0px 50px 0px 50px'}),

    html.Br(),

    dcc.Markdown('''
#### En el siguiente recuadro introduzca el valor del SST máximo del mes asignado a la variable independiente.
'''),

    html.Div([
        dcc.Input(id='sst_max', type='number', value=0)
    ], style={'text-align': 'left', 'padding': '0px 50px 0px 50px'}),

    html.Div(id='container16', children=[], style={'text-align': 'left', 'padding': '0px 20px 0px 50px'}),  # a
    html.Div(id='container17', children=[], style={'text-align': 'left', 'padding': '0px 20px 0px 50px'}),  # a
    html.Br(),

    dcc.Markdown('''

fuente:  [https://psl.noaa.gov/cgi-bin/db_search/DBSearch.pl?Dataset=Kaplan+Extended+SST+V2&Variable=Sea+Surface+Temperature](/)

'''),

    html.Br(),
])  # Azul


# Connect the Plotly graphs with Dash Components
# @app.callback(
# [Output(component_id='output_container', component_property='children'),
# Output(component_id='my_bee_map', component_property='figure')],
# [Input(component_id='slct_mes', component_property='value')]
# )


# def update_graph(option_slctd):
# print(option_slctd)
# print(type(option_slctd))

# container = "Seleccione el mes para el análisis de las variables hidroclimáticas: {}".format(option_slctd)

# figure = go.Figure()

# Create quiver - vientos
# u_mes = xr.open_dataset(r"uwnd.mon.ltm.nc")
# v_mes = xr.open_dataset(r"vwnd.mon.ltm.nc")
# u_mes_1 = u_mes.uwnd.sel(time=option_slctd, level=925, lat=slice(35, -35), lon=slice(170, 360))  #lat=slice(90, -90), lon=slice(180, 340)
# v_mes_1 = v_mes.vwnd.sel(time=option_slctd, level=925, lat=slice(35, -35), lon=slice(170, 360))  #lat=slice(90, -90), lon=slice(180, 340)
# m = Basemap()
# y = u_mes_1.lat.values
# x = v_mes_1.lon.values
# xx, yy = np.meshgrid(x,y)
# px, py = m(xx,yy)
# u = u_mes_1.values[0]
# v = v_mes_1.values[0]
# figure = ff.create_quiver(px, py, u, v, scale=.2, arrow_scale=.3,  name='Wind Velocity', line=dict(width=1), scaleratio=0.8) # template="plotly_white"


# Create contornos de OLR
# olr = xr.open_dataset(r'olr.mon.ltm.nc')
# olr_mes_1 = olr.olr.sel(time=option_slctd, lat=slice(35, -35), lon=slice(170, 360)) #lat=slice(40, -40), lon=slice(0, 360)
# figure.add_trace(go.Contour(z=olr_mes_1.values[0], x=olr_mes_1.lon.values, y=olr_mes_1.lat.values,  line_width=0.05, colorscale="RdBu", opacity=0.7, colorbar=dict(
# title='OLR W/m^2', titleside='right', titlefont=dict(size=14, family='Arial, sans-serif'),  thickness=25, thicknessmode='pixels', len=0.8, lenmode='fraction', outlinewidth=0) )) # contours_coloring='lines', line_width=2,

# Create contornos de Omega
# om = xr.open_dataset(r'omega.mon.ltm.nc')
# om_mes_1 = om.omega.sel(time=option_slctd, level=500, lat=slice(35, -35), lon=slice(170, 360), ) #lat=slice(40, -40), lon=slice(0, 360)
# figure.add_trace(go.Contour(z=om_mes_1.values[0], x=om_mes_1.lon.values, y=om_mes_1.lat.values, colorscale='Inferno', name='omega-Pa/s', contours_coloring='lines', showscale=False,  line_width=0.8)) #colorscale=False, autocolorscale=True, showlegend=False,

# Add images
# import base64
# set a local image as a background
# image_filename = 'Colombia_v0.png'
# plotly_logo = base64.b64encode(open(image_filename, 'rb').read())

# figure.update_layout(images= [dict(source='data:image/png;base64,{}'.format(plotly_logo.decode()), xref="x", yref="y",
# x=170, y=35, sizex=187.5, sizey=70, sizing="stretch", opacity= 0.99, visible = True, layer="below")]) # "above" #"stretch", "fill" opacity= 0.99,

# Add title to layout
# figure.update_layout(title='Variables hidroclimáticas, velocidad (u,v), OLR y líneas de contorno - Omega') #coloraxis_showscale=False
# figure.update_layout( template="plotly_white", height=500, width=1300, margin={"r":30,"t":30,"l":30,"b":30}) # height=500, width=1400,

# figure.update_xaxes(title_text='longitud', range=[170, 360],  showgrid=True, gridwidth=1, gridcolor='gray', ticks="outside", tickwidth=2, tickcolor='gray', nticks=9,
# linewidth=1, linecolor='gray', mirror=True) # nticks=9, ticklen=20,
# figure.update_yaxes(title_text='latitud', range=[-35, 35],  showgrid=True, gridwidth=1, gridcolor='gray', ticks="outside", tickwidth=2, tickcolor='gray',
# linewidth=1, linecolor='gray', mirror=True)
# return container, figure

@app.callback(
    [Output(component_id='container2', component_property='children'),
     Output(component_id='map2', component_property='figure'),
     Output(component_id='container3', component_property='children'),
     Output(component_id='container4', component_property='children'),
     Output(component_id='container5', component_property='children'),
     Output(component_id='container6', component_property='children'),
     Output(component_id='container7', component_property='children'),
     Output(component_id='container8', component_property='children'),
     Output(component_id='container9', component_property='children'),
     Output(component_id='container10', component_property='children'),
     Output(component_id='map23', component_property='figure'),
     Output(component_id='map3', component_property='figure'),
     Output(component_id='container11', component_property='children'),
     Output(component_id='container12', component_property='children'),
     Output(component_id='container13', component_property='children'),
     Output(component_id='container14', component_property='children'),
     Output(component_id='map4', component_property='figure'),
     ],
    [Input(component_id='slct_mes3', component_property='value'),
     Input(component_id='slct_mes2', component_property='value')],
)
def update_graph3(option_slctd3, option_slctd2):
    # print(option_slctd)
    # print(type(option_slctd))

    container2 = "En el recuadro izquierdo seleccione el mes para sst / en el derecho seleccione el mes para la variable Caudal Riogrande: {}".format(
        option_slctd3)

    m = option_slctd3
    k = str(m)
    df = pd.read_csv('datos_yearmes.csv', sep=";")
    dff = df[k]
    dff = np.array(dff)

    a_sst_monthly = xr.open_dataset(r"sst.mon.anom.nc")

    n = option_slctd2

    a_sst_monthly2 = a_sst_monthly.sst.sel(time=slice('1975-01-01', '2019-12-01')).fillna(
        a_sst_monthly.sst.mean())  # a_sst_monthly.sst.mean()
    a_sst_monthly3 = a_sst_monthly2[7:52]

    for l in range(n, 540, 12):
        m = int(l / 12)
        a_sst_monthly3[m] = a_sst_monthly2[l]

    correl_sst = a_sst_monthly2.values[0]  # correl_sst es el molde donde se almacenan los calculos de correlación

    import numpy
    for i in range(0, 36):
        for j in range(0, 72):
            sst_ij = a_sst_monthly3.values[:, i, j]
            correl_sst[i][j] = np.corrcoef(sst_ij, dff)[1, 0]

    fig = go.Figure()

    fig.add_trace(go.Contour(z=correl_sst, x=a_sst_monthly.lon.values, y=a_sst_monthly.lat.values, line_width=0.05,
                             name='corr.SST', colorscale="RdBu", opacity=0.8, colorbar=dict(
            title='Correlación SST-q mes', titleside='right', titlefont=dict(size=14, family='Arial, sans-serif'),
            thickness=25, thicknessmode='pixels', len=0.8, lenmode='fraction',
            outlinewidth=0)))  # contours_coloring='lines', line_width=2,

    list_x0 = a_sst_monthly.lon.values.tolist()
    list_x1 = [str(x) for x in list_x0]
    list_y = a_sst_monthly.lat.values.tolist()
    numpy_data = np.array(correl_sst[:])
    df3 = pd.DataFrame({'lat': list_y})
    df4 = pd.DataFrame(data=numpy_data, columns=[list_x1])  # index=[list_y],
    df5 = pd.concat([df3, df4], axis=1)
    df6 = df5.melt(id_vars=['lat'], var_name='lon', value_name="confirmed")
    df7 = df6['lon']
    for i in range(0, len(df6['lon'])):
        tup = (df6['lon'][i])
        df7[i] = float(''.join(tup))
    df7 = pd.DataFrame({'lon': df7})
    df8 = pd.concat([df6['lat'], df7, df6['confirmed']], axis=1)

    dfmin = df8[df8['confirmed'] == df8['confirmed'].min()]
    dfmax = df8[df8['confirmed'] == df8['confirmed'].max()]

    correl_sst_min = dfmin['confirmed'].values[0]
    correl_sst_max = dfmax['confirmed'].values[0]

    correl_min_max = round(correl_sst_min, 2), round(correl_sst_max, 2)
    # Add images
    import base64
    # set a local image as a background
    image_filename2 = 'world.png'
    plotly_logo = base64.b64encode(open(image_filename2, 'rb').read())

    fig.update_layout(images=[dict(source='data:image/png;base64,{}'.format(plotly_logo.decode()), xref="x", yref="y",
                                   x=0, y=90, sizex=357.5, sizey=180, sizing="stretch", opacity=0.99, visible=True,
                                   layer="below")])  # "above" #"stretch", "fill" opacity= 0.99,
    # Add title to layout
    fig.update_layout(
        title='Contornos de correlación entre datos mensuales de SST y Caudal desde 1975 hasta 2019')  # coloraxis_showscale=False
    fig.update_layout(template="plotly_white", height=500, width=1300, margin={"r": 30, "t": 30, "l": 20, "b": 30},
                      annotations=[dict(x=0.90, y=-0.05, xref='paper', yref='paper',
                                        text='correlación min, max: {}'.format(correl_min_max))], )

    # fig.update_layout(annotations = [dict(x=0.25, y=-0.05, xref='paper', yref='paper', text='correlación max: {}'.format(round(correl_sst.max(),2)))])

    fig.update_xaxes(title_text='longitud', range=[0, 360], showgrid=True, gridwidth=1, gridcolor='gray',
                     ticks="outside", tickwidth=2, tickcolor='gray', nticks=9,
                     linewidth=1, linecolor='gray', mirror=True)  # nticks=9, ticklen=20,
    fig.update_yaxes(title_text='latitud', range=[-90, 90], showgrid=True, gridwidth=1, gridcolor='gray',
                     ticks="outside", tickwidth=2, tickcolor='gray',
                     linewidth=1, linecolor='gray', mirror=True)

    lat_min = dfmin['lat'].values[0]
    lat_max = dfmax['lat'].values[0]
    lon_min = dfmin['lon'].values[0]
    lon_max = dfmax['lon'].values[0]

    sst_corr_min = a_sst_monthly3.sel(lat=lat_min, lon=lon_min)
    sst_corr_max = a_sst_monthly3.sel(lat=lat_max, lon=lon_max)
    qmin = round(df[k].min(), 2)
    p33 = round(np.percentile(df[k], 33), 2)
    p66 = round(np.percentile(df[k], 66), 2)
    qmax = round(df[k].max(), 2)

    df9 = pd.DataFrame({'q': df[k]})
    conditions = [
        (df9['q'] < p33),
        (df9['q'] > p66)]
    choices = ['bajo lo normal', 'encima de lo normal']
    df9['estado'] = np.select(conditions, choices, default='normal')
    datos = pd.DataFrame(
        {'year': df['year'].values, 'sst_min': sst_corr_min.values, 'sst_max': sst_corr_max.values, 'q': df[k],
         'estado': df9['estado']})

    container3 = " Longitud ° (con datos de sst con correlación mínima) : {}".format(lon_min)
    container4 = " Latitud ° (con datos de sst con correlación mínima)  : {}".format(lat_min)
    container5 = " Longitud ° (con datos de sst con correlación máxima) : {}".format(lon_max)
    container6 = " Latitud ° (con datos de sst con correlación máxima) : {}".format(lat_max)
    container7 = " Caudal mínimo l/s : {}".format(qmin)
    container8 = " Caudal percentil 33 - l/s : {}".format(p33)
    container9 = " Caudal percentil 66 - l/s : {}".format(p66)
    container10 = " Caudal máximo l/s : {}".format(qmax)

    fig1 = go.Figure(data=[go.Table(
        header=dict(values=list(datos.columns), align='center', fill_color='paleturquoise', line_color='darkslategray',
                    font=dict(color='blue', size=11), height=30),
        cells=dict(values=[datos.year, datos.sst_min, datos.sst_max, datos.q, datos.estado],
                   fill_color='white', align=['left', 'center'], height=30, font=dict(size=11),
                   line_color='darkslategray'))])

    fig1.update_layout(title_text="Dataset con Información de los años (1975 - 2019), SST mínima, SST máxima y QRGII",
                       titlefont=dict(size=14, family='Arial, sans-serif'),
                       margin={"r": 50, "t": 30, "l": 50, "b": 10}, width=1400, height=350)

    train_idx, test_idx = train_test_split(datos.index, test_size=.25, shuffle=False)
    datos['split'] = 'train'
    datos.loc[test_idx, 'split'] = 'test'

    X = datos[['year', 'sst_min', 'sst_max']]
    y = datos['q']
    X_train = datos.loc[train_idx, ['year', 'sst_min', 'sst_max']]
    y_train = datos.loc[train_idx, 'q']

    model = LinearRegression()
    model.fit(X_train, y_train)
    datos['prediction'] = model.predict(X)

    fig2 = px.scatter(
        datos, x='q', y='prediction',
        marginal_x='histogram', marginal_y='histogram',
        color='split', trendline='ols'
    )
    fig2.update_traces(histnorm='probability', selector={'type': 'histogram'})
    fig2.add_shape(
        type="line", line=dict(dash='dash'),
        x0=y.min(), y0=y.min(),
        x1=y.max(), y1=y.max())
    fig2.update_layout(title='Resultados de Entrenamiento y Predicción de Caudal (Método Regresión Líneal Multiple)',
                       height=600, width=1300, )

    coef0 = model.coef_
    coef = [model.intercept_, coef0[0], coef0[1], coef0[2]]

    container11 = " Coeficientes Regresión a0 + a1(Año) + a2(SST mín.) + a3(SST máx.)  : {}".format(coef)
    container12 = " Error Medio Obsoluto (q - Predicción) : {}".format(
        round(metrics.mean_absolute_error(datos['q'], datos['prediction']), 2))
    container13 = " Error Medio Cuadratico (q - Predicción) : {}".format(
        round(metrics.mean_squared_error(datos['q'], datos['prediction']), 2))
    container14 = " Raiz Error Medio Cuadratico (q - Predicción) : {}".format(
        round(np.sqrt(metrics.mean_squared_error(datos['q'], datos['prediction'])), 2))

    in_test = datos['split'] == 'test'
    datos0 = datos[in_test]
    X = datos0.drop(columns=['q', 'estado', 'year', 'split', 'sst_min', 'sst_max'])  # 'split',
    y = datos0['estado']

    # Fit the model
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    y_scores = model.predict_proba(X)

    # One hot encode the labels in order to plot them
    y_onehot = pd.get_dummies(y, columns=model.classes_)

    fig3 = go.Figure()
    fig3.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    for i in range(y_scores.shape[1]):
        y_true = y_onehot.iloc[:, i]
        y_score = y_scores[:, i]

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_score = roc_auc_score(y_true, y_score)

        name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
        fig3.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))
    fig3.update_layout(title='Proporción de Verdaderos Positivos Frente a la Proporción de Falsos Positivos')
    fig3.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=1300, height=600
    )

    return container2, fig, container3, container4, container5, container6, container7, container8, container9, container10, fig1, fig2, container11, container12, container13, container14, fig3  # container2,


@app.callback(
    [Output(component_id='container15', component_property='children'),
     Output(component_id='container16', component_property='children'),
     Output(component_id='container17', component_property='children')],
    [Input('ano', 'value'),
     Input('sst_min', 'value'),
     Input('sst_max', 'value')],
    [State(component_id='slct_mes3', component_property='value'),
     State(component_id='slct_mes2', component_property='value')],
)
def callback_a(a1, a2, a3, option_slctd3, option_slctd2):  # actualizar

    if a1 == 0 or a2 == 0 or a3 == 0:  # actualizar
        return no_update  # actualizar
    elif a1 == None or a2 == None or a3 == None:  # actualizar
        return no_update  # actualizar
    else:  # actualizar
        container15 = "Año de análisis {}".format(a1)  # actualizar

        container16 = "Datos de SST para el mes y año de la variable independiente"

        m = option_slctd3
        k = str(m)
        df = pd.read_csv('datos_yearmes.csv', sep=";")
        dff = df[k]
        dff = np.array(dff)

        a_sst_monthly = xr.open_dataset(r"sst.mon.anom.nc")

        n = option_slctd2

        a_sst_monthly2 = a_sst_monthly.sst.sel(time=slice('1975-01-01', '2019-12-01')).fillna(
            a_sst_monthly.sst.mean())  # a_sst_monthly.sst.mean()
        a_sst_monthly3 = a_sst_monthly2[7:52]

        for l in range(n, 540, 12):
            m = int(l / 12)
            a_sst_monthly3[m] = a_sst_monthly2[l]

        correl_sst = a_sst_monthly2.values[0]  # correl_sst es el molde donde se almacenan los calculos de correlación

        import numpy
        for i in range(0, 36):
            for j in range(0, 72):
                sst_ij = a_sst_monthly3.values[:, i, j]
                correl_sst[i][j] = np.corrcoef(sst_ij, dff)[1, 0]

        list_x0 = a_sst_monthly.lon.values.tolist()
        list_x1 = [str(x) for x in list_x0]
        list_y = a_sst_monthly.lat.values.tolist()
        numpy_data = np.array(correl_sst[:])
        df3 = pd.DataFrame({'lat': list_y})
        df4 = pd.DataFrame(data=numpy_data, columns=[list_x1])  # index=[list_y],
        df5 = pd.concat([df3, df4], axis=1)
        df6 = df5.melt(id_vars=['lat'], var_name='lon', value_name="confirmed")
        df7 = df6['lon']
        for i in range(0, len(df6['lon'])):
            tup = (df6['lon'][i])
            df7[i] = float(''.join(tup))
        df7 = pd.DataFrame({'lon': df7})
        df8 = pd.concat([df6['lat'], df7, df6['confirmed']], axis=1)

        dfmin = df8[df8['confirmed'] == df8['confirmed'].min()]
        dfmax = df8[df8['confirmed'] == df8['confirmed'].max()]

        correl_sst_min = dfmin['confirmed'].values[0]
        correl_sst_max = dfmax['confirmed'].values[0]

        correl_min_max = round(correl_sst_min, 2), round(correl_sst_max, 2)

        lat_min = dfmin['lat'].values[0]
        lat_max = dfmax['lat'].values[0]
        lon_min = dfmin['lon'].values[0]
        lon_max = dfmax['lon'].values[0]

        sst_corr_min = a_sst_monthly3.sel(lat=lat_min, lon=lon_min)
        sst_corr_max = a_sst_monthly3.sel(lat=lat_max, lon=lon_max)
        qmin = round(df[k].min(), 2)
        p33 = round(np.percentile(df[k], 33), 2)
        p66 = round(np.percentile(df[k], 66), 2)
        qmax = round(df[k].max(), 2)

        df9 = pd.DataFrame({'q': df[k]})
        conditions = [
            (df9['q'] < p33),
            (df9['q'] > p66)]
        choices = ['bajo lo normal', 'encima de lo normal']
        df9['estado'] = np.select(conditions, choices, default='normal')
        datos = pd.DataFrame(
            {'year': df['year'].values, 'sst_min': sst_corr_min.values, 'sst_max': sst_corr_max.values, 'q': df[k],
             'estado': df9['estado']})

        train_idx, test_idx = train_test_split(datos.index, test_size=.25, shuffle=False)
        datos['split'] = 'train'
        datos.loc[test_idx, 'split'] = 'test'

        X = datos[['year', 'sst_min', 'sst_max']]
        y = datos['q']
        X_train = datos.loc[train_idx, ['year', 'sst_min', 'sst_max']]
        y_train = datos.loc[train_idx, 'q']

        model = LinearRegression()
        model.fit(X_train, y_train)
        datos['prediction'] = model.predict(X)

        coef0 = model.coef_
        coef = [model.intercept_, coef0[0], coef0[1], coef0[2]]

        q_estimado = model.intercept_ + coef0[0] * a1 + coef0[1] * a2 + coef0[2] * a3

        container17 = "Caudal estimado para el periodo de análisis en lps {}".format(q_estimado)

        return container15, container16, container17

#------------------------------------------------------------------

if __name__ == '__main__':
    app.run_server(debug=True)
#------------------------------------------------------------------


#------------------------------------------------------------------

#------------------------------------------------------------------