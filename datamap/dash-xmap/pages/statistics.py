import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from utils import Header, make_dash_table
import pandas as pd
import pathlib

def create_layout(app, params):
    return html.Div(
        [
            Header(app),
            # page 4
            html.Div(
                [
                    # Row 1
                    html.Div(
                        [
                            html.Div(
                                [html.H6('Descriptive statistics of "{}" data set'.format(params["datasetdict"][params["dataname"]]), className="subtitle padded")],
                            ),
                            # html.Br([]),
                            dcc.RadioItems(
                                id='class-dropdown',
                                options=[
                                    {'label': i, 'value': i}
                                    for i in ["All"] + ["Class " + str(c) for c in list(params["target_classes"])]
                                ],
                                value="All",
                                labelStyle={'display': 'inline-block'}
                            ),
                            # html.Br([]),
                            # html.Table(make_dash_table(df_fund_facts)),
                            html.Div(id="datasum-table"),
                        ],
                    ),
                ],
                className="sub_page",
            ),
        ],
        className="page",
    )
