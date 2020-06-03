import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import dash_table

from utils import Header, make_dash_table

import pandas as pd
import pathlib

def create_layout(app, params):
    # Page layouts
    return html.Div(
        [
            html.Div([Header(app)]),
            # page 1
            html.Div(
                [
                    # Row 3
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H5("XMAP Summary"),
                                    html.Br([]),
                                    dcc.Markdown(
                                        "\
                                        eXplainable Mapping Analytics Platform (XMAP) is a people-centric analytics solution "
                                        "aiming at providing an effortless and systematic methodology to handle complex data"
                                        " analytics tasks. By taking advantages of advanced artificial intelligence (AI) algorithms,"
                                        " XMAP can cope with big and high-dimensional data without compromising the interpretability"
                                        "of the obtained models.\
                                        ",
                                        style={"color": "#ffffff"},
                                        className="row",
                                    ),
                                    html.P(
                                        "XMAP steps include:",
                                    ),
                                    html.P(
                                        "Data Cleaning",
                                        className="subtitle1 padded",
                                    ),
                                    html.P(
                                        "Data Mapping",
                                        className="subtitle1 padded",
                                    ),
                                    html.P(
                                        "Topology Learning",
                                        className="subtitle1 padded",
                                    ),
                                    html.P(
                                        "Context Learning",
                                        className="subtitle1 padded",
                                    ),
                                ],
                                className="product",
                            )
                        ],
                        className="row",
                    ),
                    # Row 4
                    html.Div(
                        [
                        html.H6(
                            ["Select Data Set"], className="subtitle padded"
                        ),
                        html.P(
                            ["Choose the data set from the below list for further analyses"],
                            id="dataset-label"
                        ),
                        dcc.Dropdown(
                            id='dataset-dropdown',
                            options=[
                                {'label': params["datasetnames"][i], 'value': params["datasets"][i]}
                                for i in range(len(params["datasets"]))
                            ],
                            value=params["datasets"][0] if "data" not in params else params["dataname"]
                        ),
                        html.Br([]),
                        dcc.Markdown(
                            ["Choose the data set from the below list for further analyses"],
                            id="dataset-notes"
                        ),
                        html.Br([]),
                        # html.Table(make_dash_table(df_fund_facts)),
                        dash_table.DataTable(id="data-table", style_table={'maxHeight': '300px', 'overflowX': 'scroll'}, page_current=0,
                                            page_size=5,
                                            page_action='custom')
                        ],
                        className="data-table-sum",
                    ),
                    # Row 5
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        "Class distribution",
                                        className="subtitle padded",
                                    ),
                                    dcc.Graph(
                                        id="graph-2",
                                        # figure={
                                        #     "data": [go.Pie(labels=["A", "B"], values=[10,20],
                                        #                     marker={'colors': ['#97151c', '#dddddd', '#ffd6d6', '#faebeb', '#ffffff']}, textinfo='label')],
                                        #     "layout": go.Layout(title=f"Cases Reported Monthly", margin={"l": 10, "r": 10, },
                                        #     legend={"x": 1, "y": 0.7})},
                                        # config={"displayModeBar": False},
                                    ),
                                ],
                                className="six columns",
                            ),
                            html.Div(
                                [
                                    html.H6(
                                        "Feature correlation",
                                        className="subtitle padded",
                                    ),
                                    html.Div(id="feature-space"),
                                    dcc.Graph(
                                        id="graph-feature_corr",
                                    ),
                                ],
                                className="six columns",
                            )
                        ],
                        className="row ",
                    ),
                ],
                className="sub_page",
            ),
        ],
        className="page",
    )
