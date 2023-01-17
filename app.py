from dash import Dash, html, dcc, Input, Output, ctx, State
from figure_creation.argoverse import ArgoverseFigureCreator
import dash_bootstrap_components as dbc
import os
from PIL import Image

fc = ArgoverseFigureCreator()
# fig = fc.generate_figure(scene_id='0a0ef009-9d44-4399-99e6-50004d345f34')
fig = fc.get_current_scene()

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
app = Dash(__name__, external_stylesheets=[dbc_css, dbc.themes.DARKLY])

header = html.H4(
    "header text is here", className="bg-primary text-white p-2 mb-2 text-center"
)

graph = dcc.Graph(
        id='scene_graph',
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
            "disabled": True,
        },
        {
            "label": html.Span(
                [
                    html.Img(src=yandex_logo, height=20),
                    html.Span("Yandex Shifts", style={'font-size': 15, 'padding-left': 10}),
                ], style={'align-items': 'center', 'justify-content': 'center'}
            ),
            "value": "Yandex",
            "disabled": True,
        },
        {
            "label": html.Span(
                [
                    html.Img(src=lyft_logo, height=20),
                    html.Span("Lyft Level 5", style={'font-size': 15, 'padding-left': 10}),
                ], style={'align-items': 'center', 'justify-content': 'center'}
            ),
            "value": "Lyft",
            "disabled": True,
        },
    ],
    value="Argoverse",
    clearable=False
)

button_group = dbc.ButtonGroup(
    [
        dbc.Button("Previous", color="primary", outline=True, id="previous_scene_button"),
        dbc.Input(value="1", type="text", id="scene_id", style={'textAlign': 'center'}),
        dbc.Button("Next", color="primary", id="next_scene_button"),
    ]
)

advanced_settings = html.Div(
    [
        dbc.Button(
            "Advanced settings",
            id="settings-collapse-button",
            className="mb-3",
            color="secondary",
            outline=True,
            n_clicks=0,
        ),
        dbc.Collapse(
            dbc.Card(dbc.CardBody(
                html.Span([
                    dbc.Switch(
                        id="show-trajectory-switch",
                        label="show trajectory",
                        value=False,
                    ),
                    dbc.Label("Scale:"),
                    dbc.RadioItems(
                        options=[
                            {"label": "as is", "value": 1},
                            {"label": "only significant", "value": 2},
                        ],
                        value=2,
                        id="scale-radioitem",
                    )
                ])

            )),
            id="settings-panel",
            is_open=False,
        ),
    ]
)


app.layout = dbc.Container(
    [
        header,
        dbc.Row(
            [
                dbc.Col(
                    [dropdown, html.Br(), button_group, html.Br(), html.Br(), advanced_settings],
                    width=2
                ),
                dbc.Col([graph], width=8),
            ]
        ),
    ],
    fluid=True,
    className="dbc",
)


@app.callback(
    Output("settings-panel", "is_open"),
    [Input("settings-collapse-button", "n_clicks")],
    [State("settings-panel", "is_open")],
)
def toggle_settings_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(
    Output(component_id='scene_graph', component_property='figure'),
    Output(component_id='scene_id', component_property='value'),
    Input(component_id='scene_id', component_property='value'),
    Input(component_id='next_scene_button', component_property='n_clicks'),
    Input(component_id='previous_scene_button', component_property='n_clicks'),
    Input(component_id='show-trajectory-switch', component_property='value'),
    Input(component_id='scale-radioitem', component_property='value')
)
def change_scene_id_by_clicking_buttons(scene_id_from_input, next_clicked, prev_clicked, switch_value, scale_variant):
    trigger_id = ctx.triggered_id

    print(trigger_id)
    new_scene, new_scene_id = fc.current_scene, fc.current_scene_id
    if trigger_id == 'next_scene_button':
        new_scene, new_scene_id = fc.get_next_scene()
    elif trigger_id == 'previous_scene_button':
        new_scene, new_scene_id = fc.get_previous_scene()
    elif trigger_id == 'scene_id':
        new_scene, new_scene_id = fc.get_scene_by_id(int(scene_id_from_input))
    elif trigger_id == 'show-trajectory-switch':
        new_scene, new_scene_id = fc.change_visibility_of_trajectories(switch_value)
    elif trigger_id == 'scale-radioitem':
        new_scene, new_scene_id = fc.change_scene_scale(scale_variant)
    return new_scene, str(new_scene_id)


if __name__ == '__main__':
    app.run_server(debug=True)
