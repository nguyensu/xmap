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
                                    html.H5("Data Mapping"),
                                    html.Br([]),
                                    html.P(
                                        'The inputs from "{}" data set is mapped into two dimensional space. '
                                        'Each data instance is represented as a point and the proximity of points on the map represents '
                                        'the similarity of the original data instances.'.format(params["datasetdict"][params["dataname"]]),
                                        style={"color": "#ffffff"},
                                        className="row",
                                    ),
                                ],
                                className="productnew",
                            ),
                            html.Div(
                                [
                                    html.H6(
                                        "Mapping mode",
                                        className="subtitle padded"),
                                    dcc.RadioItems(
                                        id='map-mode-dropdown',
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
                            # html.Table(make_dash_table(df_fund_facts)),
                            dcc.Graph(id="data-map", selectedData={'points': [], 'range': None})
                        ],
                    ),
                    # Row 5
                    html.Br(),
                    html.Div(
                        [
                            html.Div(
                                [
                                    daq.Slider(
                                        id="size-slider",
                                        min=1,
                                        max=20,
                                        value=5,
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
                                        id="opacity-slider",
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
                        id="selected-notes",
                        className="subtitle padded"
                    ),
                    html.Br(),
                    dash_table.DataTable(id="point-table", style_table={'maxHeight': '300px', 'overflowX': 'scroll'},
                                         page_current=0,
                                         page_size=5,
                                         page_action='custom'),
                ],
                className="sub_page",
            ),
        ],
        className="page",
    )

# Load profiling is conducted using Distributed Growing Self-Organizing Map (DGSOM), a CDAC proprietary algorithm. Given the customer profiles (500,000 weekly profiles), DGSOM automatically generates a representation that captures the similarity and dissimilarity (or topology) of the input data. The DGSOM representation is a simple two-dimensional map with a set of nodes, each of which represents a consumption pattern shared by a number of input weekly profiles (e.g. high energy consumption on the weekdays and low consumption on the weekends). Below are some simple guidelines to analyse DGSOM outputs:
#
# · Each node represent a pattern that is mapped to a x-y coordinates; the number next to each node is the number of weekly profiles that are clustered within that node.
#
# · If 2 weekly profiles are similar, they will cluster into the same cluster node or adjacent cluster nodes that are in close proximity
#
# · If 2 weekly profiles are dissimilar, they will cluster into distant cluster nodes
#
# · If there are weekly profiles that are out of ordinary (anomalies), the GSOM map will generate cluster nodes far from the other cluster nodes