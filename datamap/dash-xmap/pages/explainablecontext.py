import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import dash_table
from utils import Header, make_dash_table
import pandas as pd
import pathlib
import dash_daq as daq
import matplotlib

def create_layout(app, params):
    labels = params["data"][params["target_name"]].values
    colors = ['#2148bf', '#97151c', '#ffd6d6', '#faebeb', '#ffffff']

    return html.Div(
        [
            html.Div([Header(app)]),
            # page 1
            html.Div(
                [
                    # Row 1
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H5("Explainable Contexts"),
                                    html.Br([]),
                                    html.P(
                                        'This step discovers the contexts in which data from the "{}" data set is generated.'
                                        ''
                                        .format(
                                            params["datasetdict"][params["dataname"]]),
                                        style={"color": "#ffffff"},
                                        className="row",
                                    ),
                                ],
                                className="productnew",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.H6(
                                                "Mapping mode",
                                                className="subtitle padded"),
                                            dcc.RadioItems(
                                                id='mapping-mode-dropdown',
                                                options=[
                                                    {'label': i, 'value': i}
                                                    for i in ["Unsupervised", "Supervised"]
                                                ],
                                                value="Unsupervised",
                                                labelStyle={'display': 'inline-block'}
                                            ),
                                        ],
                                        style={'width': '30%', 'display': 'inline-block'},
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [html.H6(
                                                    'Representations',
                                                    className="subtitle padded")],
                                            ),
                                            # html.Br([]),
                                            dcc.RadioItems(
                                                id='representation-dropdown',
                                                options=[
                                                    {'label': i, 'value': i}
                                                    for i in ["Context", "Clustered Map", "Labelled Map", "Network"]
                                                ],
                                                value="Context",
                                                labelStyle={'display': 'inline-block'}
                                            ),
                                        ],
                                        style={'width': '50%', 'display': 'inline-block'},
                                    ),
                                    html.Div(
                                        [
                                            dcc.Dropdown(
                                                id='context-dropdown',
                                                options=[
                                                    {'label': "No Context", 'value': 0}
                                                ],
                                                value=0,
                                                # labelStyle={'display': 'inline-block'}
                                            ),
                                        ],
                                        id="context-select",
                                        style={'width': '20%', 'display': 'inline-block'},
                                    ),
                                ]
                            ),
                            html.Br([]),
                            html.Div(
                                [
                                    dcc.Graph(id="explain-map", selectedData={'points': [], 'range': None},
                                              style={'width': '60%', 'display': 'inline-block'}, className="seven columns",),
                                    html.Div(id="context-notes",
                                         style={'width': '35%', 'display': 'inline-block'}, className="six columns",
                                         )
                                ]
                                , className="row"
                            )
                        ],
                    ),
                    # Row 5
                    html.Br(),
                    html.Div(
                        [
                            html.Div(
                                [
                                    daq.Slider(
                                        id="size-slider-network",
                                        min=1,
                                        max=20,
                                        value=10,
                                        # handleLabel={"showCurrentValue": True,"label": "Size"},
                                        handleLabel='Size',
                                        step=1,
                                        color='#97151c',
                                        marks={'1': 'Min', '20': 'Max'},
                                        size=320
                                    )
                                ],
                                style={'width': '50%', 'display': 'inline-block'},
                            ),
                            html.Div(
                                [
                                    daq.Slider(
                                        id="opacity-slider-network",
                                        min=0,
                                        max=1,
                                        value=0.5,
                                        # handleLabel={"showCurrentValue": True,"label": "Size"},
                                        handleLabel='Opacity',
                                        step=0.1,
                                        color='#97151c',
                                        marks={'0': 'Min', '1': 'Max'},
                                        size=320
                                    )
                                ],
                                style={'width': '50%', 'float': 'right', 'display': 'inline-block'},
                            )
                        ],
                    ),
                    html.Br(), html.Br(),
                    html.Div(
                        id="selected-explain-notes",
                        className="subtitle padded"
                    ),
                    html.Br(),
                    dash_table.DataTable(id="data-explain-table", style_table={'maxHeight': '300px', 'overflowX': 'scroll'},
                                         page_current=0,
                                         page_size=5,
                                         page_action='custom'),
                ],
                className="sub_page",
            ),
        ],
        className="page",
    )
