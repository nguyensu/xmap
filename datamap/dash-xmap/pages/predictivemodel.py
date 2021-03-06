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
    # labels = params["data"][params["target_name"]].values
    # colors = ['#2148bf', '#97151c', '#ffd6d6', '#faebeb', '#ffffff']
    params["context_data"] = pd.DataFrame(params["context_vector"],
                                                  columns=["Context #" + str(c + 1) for c in
                                                           range(params["network_ncluster"])])
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
                                    html.H5("Explainable Predictive Modelling"),
                                    html.Br([]),
                                    html.P(
                                        'This step combines the context information from the previous mapping steps '
                                        'to built the predictive models for the data set "{}".'
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
                                                    for i in ["Context", "Clustered Map", "Labelled Map"]
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
                            dash_table.DataTable(id="raw-data-table",
                                                 style_table={'maxHeight': '300px', 'overflowX': 'scroll'},),
                            dash_table.DataTable(id="context-data-table",
                                                 style_table={'maxHeight': '300px', 'overflowX': 'scroll'},
                                                 page_current=0,
                                                 page_size=5,
                                                 page_action='custom'),
                            html.Br([]),
                            html.Div(
                                [
                                    html.H6(
                                        "Select ML Algorithms",
                                        className="subtitle padded", style={'width': '80%', 'display': 'inline-block'}),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            dcc.Dropdown(
                                                id='ml-dropdown',
                                                options=[
                                                    {'label': i, 'value': i}
                                                    for i in ["Logistic Regression", "XMAP Logistic Regression", "XMAP Context-aware Logistic Regression",
                                                              "Decision Tree", "XMAP Decision Tree",
                                                              "ANN", "XMAP ANN",
                                                              "XGBoost", "XMAP XGBoost",
                                                              "Skope Rules", "XMAP Skope Rules"]
                                                ],
                                                value="Logistic Regression",
                                                # labelStyle={'display': 'inline-block'}
                                                style={'width': '90%'},
                                            ),
                                        ],
                                        style={'width': '40%', 'display': 'inline-block'},
                                    ),
                                    # html.Div(),
                                    # html.Div(" ", style={'width': '3%', 'display': 'inline-block'},),
                                    html.Div(
                                        [
                                            daq.Slider(
                                                id="param1",
                                                min=0,
                                                max=1,
                                                value=0.01,
                                                handleLabel={"showCurrentValue": True, "label": "C"},
                                                # handleLabel='Size',
                                                step=0.001,
                                                color='#97151c',
                                                size=150
                                            ),
                                        ],
                                        style={'width': '30%', 'display': 'inline-block'},
                                    ),
                                    html.Div(
                                        [
                                            daq.Slider(
                                                id="param2",
                                                min=0,
                                                max=1,
                                                value=0.01,
                                                handleLabel={"showCurrentValue": True, "label": "C"},
                                                # handleLabel='Size',
                                                step=0.001,
                                                color='#97151c',
                                                size=150
                                            ),
                                        ],
                                        style={'width': '30%', 'display': 'inline-block'},
                                    ),
                                    # html.Div(id="param3",
                                    #          style={'width': '22%', 'display': 'inline-block'},
                                    #          ),
                                ]
                            ),
                            html.Br(),
                            html.Div(
                                [
                                    html.Button("Train/Refresh Model", id="runml-button",
                                                style={'width': '30%', 'display': 'inline-block'}),
                                    dcc.RadioItems(
                                        id='data_select',
                                        options=[
                                            {'label': i, 'value': i}
                                            for i in ["Selected", "All"]
                                        ],
                                        value="Selected",
                                        labelStyle={'display': 'inline-block'}
                                    ),
                                ]
                            ),
                            dcc.Markdown(id='ml-run-log',
                                     children='"Hit the above button to train the model ...'),
                            html.Div(
                                    dcc.Graph(id="feature-contribution"),
                                # style={'width': 1000}
                            ),
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
                            ),
                        ],
                    ),
                ],
                className="sub_page",
            ),
        ],
        className="page",
    )
