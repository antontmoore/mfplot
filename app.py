from dash import Dash, html, dcc
from figure_creation.argoverse import FigureCreator
import dash_bootstrap_components as dbc

fc = FigureCreator()
fig = fc.generate_figure(scene_id='0a0ef009-9d44-4399-99e6-50004d345f34')

dbc_css = dbc.themes.DARKLY
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

app.layout = dbc.Container(children=[
    html.H1(children='Structure'),
    html.Div(children='''
        Web application plotting.
    '''),

    dcc.Graph(
        id='example-graph',
        style={'width': '90vh', 'height': '90vh'},
        figure=fig,
    ),
], class_name='dbc')

if __name__ == '__main__':
    app.run_server(debug=True)