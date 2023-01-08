from dash import Dash, html, dcc
from figure_creation.argoverse import FigureCreator
import dash_bootstrap_components as dbc
import os
from PIL import Image

fc = FigureCreator()
fig = fc.generate_figure(scene_id='0a0ef009-9d44-4399-99e6-50004d345f34')

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
app = Dash(__name__, external_stylesheets=[dbc_css, dbc.themes.DARKLY])

header = html.H4(
    "header text is here", className="bg-primary text-white p-2 mb-2 text-center"
)

graph = dcc.Graph(
        id='example-graph',
        style={'width': '90vh', 'height': '90vh'},
        figure=fig,
    )

waymo_logo = Image.open(os.getcwd() + "/images/waymo logo.png")
argo_logo = Image.open(os.getcwd() + "/images/argo logo.png")
yandex_logo = Image.open(os.getcwd() + "/images/yandex logo.png")
lyft_logo = Image.open(os.getcwd() + "/images/lyft logo.png")
dropdown = dcc.Dropdown(
    [
        {
            "label": html.Span(
                [
                    html.Img(src=argo_logo, height=20),
                    html.Span("Argoverse", style={'font-size': 15, 'padding-left': 10}),
                ], style={'align-items': 'center', 'justify-content': 'center'},
            ),
            "value": "Argoverse",
        },
        {
            "label": html.Span(
                [
                    html.Img(src=waymo_logo, height=20),
                    html.Span("Waymo Open Dataset", style={'font-size': 15, 'padding-left': 10}),
                ], style={'align-items': 'center', 'justify-content': 'center'}
            ),
            "value": "Waymo",
        },
        {
            "label": html.Span(
                [
                    html.Img(src=yandex_logo, height=20),
                    html.Span("Yandex Shifts", style={'font-size': 15, 'padding-left': 10}),
                ], style={'align-items': 'center', 'justify-content': 'center'}
            ),
            "value": "Yandex",
        },
        {
            "label": html.Span(
                [
                    html.Img(src=lyft_logo, height=20),
                    html.Span("Lyft Level 5", style={'font-size': 15, 'padding-left': 10}),
                ], style={'align-items': 'center', 'justify-content': 'center'}
            ),
            "value": "Lyft",
        },
    ],
    value="Argoverse",
)

app.layout = dbc.Container(
    [
        header,
        dbc.Row(
            [
                dbc.Col(
                    [dropdown],
                    width=2
                ),
                dbc.Col([graph], width=8),
            ]
        ),
    ],
    fluid=True,
    className="dbc",
)

if __name__ == '__main__':
    app.run_server(debug=True)
