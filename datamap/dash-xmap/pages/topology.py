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
                                    html.H5("Topology Learning"),
                                    html.Br([]),
                                    html.P(
                                        'The map obtained from "{}" data set processed to identify the topological '
                                        'relations of data inputs. Topological learning algorithm will automatically '
                                        'generates a representation that captures the similarity and dissimilarity '
                                        '(or topology) of the input data.'
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
                                        style={'width': '45%', 'display': 'inline-block'},
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
                                                    for i in ["Network", "Clustered Map", "Labelled Map"]
                                                ],
                                                value="Network",
                                                labelStyle={'display': 'inline-block'}
                                            ),
                                        ],
                                        style={'width': '45%', 'display': 'inline-block'},
                                    ),
                                ]
                            ),
                            html.Br([]),
                            dcc.Graph(id="data-network", selectedData={'points': [], 'range': None})
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
                        id="selected-node-notes",
                        className="subtitle padded"
                    ),
                    html.Br(),
                    dash_table.DataTable(id="node-table", style_table={'maxHeight': '300px', 'overflowX': 'scroll'},
                                         page_current=0,
                                         page_size=5,
                                         page_action='custom'),
                ],
                className="sub_page",
            ),
        ],
        className="page",
    )
