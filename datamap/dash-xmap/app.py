# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
from datamap.xmap import run_xmap, STEP, LEARN
import numpy as np
import plotly.graph_objs as go
import dash_core_components as dcc
import dash_table
import dash_daq as daq
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

datasets = ["new_german_data", "new_heart_data", "new_churn_data", "new_icu_data", "new_hribm_data", "bank_data", "new_spambase_data", "mushroom_data", "new_breastcancer_data", "adult_data",
                "australian_data", "mammo_data"]
datasets_labels = ["German Credit Risk", "Heart Disease", "Customer Churn", "ICU", "HR-IBM", "Bank", "Spambase", "Mushroom", "Breast Cancer", "Adult",
                "Australian Credit Risk", "Mammo"]

dataset_dict = {datasets[i]: datasets_labels[i] for i in range(len(datasets))}

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)

app.title = "XMAP Prototype"

from pages import (
    overview,
    statistics,
    map,
    topology,
    explainablecontext,
    predictivemodel,
    warning,
)

parameter_dict = {
    "datasets": datasets,
    "datasetnames": datasets_labels,
    "datasetdict": dataset_dict
}

server = app.server

# Describe the layout/ UI of the app
app.layout = html.Div(
    [dcc.Location(id="url", refresh=False), html.Div(id="page-content")]
)
app.config['suppress_callback_exceptions'] = True

# Update page
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def display_page(pathname):
    if "data" not in parameter_dict:
        return overview.create_layout(app, parameter_dict)
    elif pathname == "/dash-xmap/statistics":
        return statistics.create_layout(app, parameter_dict)
    elif pathname == "/dash-xmap/map":
        parameter_dict["map"] = run_xmap(dataset=parameter_dict["dataname"], n_neighbors=15, negative_sample_rate=5,
                                         seed=2,
                                         return_step=STEP.UMAP_TRAINED)
        return map.create_layout(app, parameter_dict)
    elif pathname == "/dash-xmap/topology":
        init_topo()
        return topology.create_layout(app, parameter_dict)
    elif pathname == "/dash-xmap/explainablecontext":
        assign_parameters()
        return explainablecontext.create_layout(app, parameter_dict)
    elif pathname == "/dash-xmap/predictivemodel":
        assign_parameters()
        return predictivemodel.create_layout(app, parameter_dict)
    elif pathname == "/dash-xmap/full-view":
        return (
            overview.create_layout(app, parameter_dict),
            statistics.create_layout(app, parameter_dict),
            map.create_layout(app, parameter_dict),
        )
    else:
        return overview.create_layout(app, parameter_dict)


def init_topo():
    embeddings, nodes, connection, classes, nclusters, node_indices, indices = run_xmap(
        dataset=parameter_dict["dataname"], n_neighbors=15, negative_sample_rate=5, seed=2,
        return_step=STEP.SOINN_TRAINED)
    parameter_dict["map"] = embeddings
    parameter_dict["network_nodes"] = nodes
    parameter_dict["network_connection"] = connection
    parameter_dict["network_cluster_names"] = classes
    parameter_dict["network_ncluster"] = nclusters
    parameter_dict["network_node_indices"] = node_indices
    parameter_dict["network_indices_to_clusters"] = indices


def assign_parameters():
    embeddings, nodes, connection, classes, nclusters, node_indices, indices, cluster_explainer_dict, xcluster_id_details, xfeaturenames \
        = run_xmap(dataset=parameter_dict["dataname"], n_neighbors=15, negative_sample_rate=5, seed=2,
                   return_step=STEP.CONTEXT_EXPLAINED)
    parameter_dict["map"] = embeddings
    parameter_dict["network_nodes"] = nodes
    parameter_dict["network_connection"] = connection
    parameter_dict["network_cluster_names"] = classes
    parameter_dict["network_ncluster"] = nclusters
    parameter_dict["network_node_indices"] = node_indices
    parameter_dict["network_indices_to_clusters"] = indices
    parameter_dict["explain_dict"] = cluster_explainer_dict
    parameter_dict["context_vector"] = xcluster_id_details
    parameter_dict["explain_feature_names"] = xfeaturenames


@app.callback(
    [
        Output('dataset-notes', 'children'),
        Output('data-table', 'columns'),
        Output('data-table', 'style_data_conditional')
    ],
    [
        Input('dataset-dropdown', 'value')
    ])
# @app.callback([], [Input('dataset-dropdown', 'value')])
def update_output(value):
    if "dataname" not in parameter_dict or parameter_dict["dataname"] != value:
        X_norm, Y, scaler, nfeatures, feature_names, target_name = run_xmap(dataset=value, n_neighbors=15, negative_sample_rate=5, seed=2, return_step=STEP.DATA_CLEANED)
        feature_names_display = [ff.replace("ubar", "_").replace("dot", ".") for ff in feature_names]
        data = pd.DataFrame(np.concatenate((Y, X_norm), axis=1), columns=[target_name] + feature_names_display)
        parameter_dict["data"] = data
        parameter_dict["dataname"] = value
        parameter_dict["nfeatures"] = nfeatures
        parameter_dict["feature_names"] = feature_names
        parameter_dict["feature_names_display"] = feature_names_display
        parameter_dict["feature_names_dict"] = {feature_names[i]: feature_names_display[i] for i in range(nfeatures)}
        parameter_dict["target_name"] = target_name
        parameter_dict["target_classes"] = np.unique(Y)
        conditions = [
            {'if': {'column_id': parameter_dict["target_name"]},
             'backgroundColor': '#97151c',
             'color': 'white', }
        ]
        for c in parameter_dict["feature_names_display"]:
            conditions.append(
                {
                    'if': {'column_id': c, 'filter_query': '{{{0}}} != 0'.format(c)},
                    'backgroundColor': '#dddddd',
                    'color': 'black',
                }
            )
        parameter_dict["table_format"] = conditions
    style_data_conditional = parameter_dict["table_format"]
    return 'You have selected "{}" dataset with `{} instances` and `{} features`. Below is a subset of the data set.'.format(dataset_dict[value], len(parameter_dict["data"]), parameter_dict["nfeatures"]), \
           [{"name": i, "id": i} for i in parameter_dict["data"].columns], style_data_conditional

@app.callback(
    Output('data-table', 'data'),
    [
        Input('data-table', 'columns'),
        Input('data-table', "page_current"),
        Input('data-table', "page_size")
    ]
)
def update_data_table(cols, page_current,page_size):
    if "data" not in parameter_dict:
        return pd.DataFrame([]).to_dict('records')
    return parameter_dict["data"].round(3).iloc[page_current*page_size:(page_current+ 1)*page_size].to_dict('records')

@app.callback(
    Output('graph-2', 'figure'),
    [Input('data-table', 'columns')])
def update_distribution_target_chart(cols):
    class_names, occurCount = np.unique(parameter_dict["data"][parameter_dict["target_name"]], return_counts=True)
    # print()
    return {
            "data": [go.Pie(labels=["Class #" + str(int(cn)) for cn in class_names], values=[ct for ct in occurCount],
                            marker={'colors': ['#97151c', '#dddddd', '#ffd6d6', '#faebeb', '#ffffff']}, textinfo='label')],
            "layout": go.Layout(
                # title=f"Cases Reported Monthly",
                margin={"l": 50, "r": 50, },
                showlegend=False)}


@app.callback(
    Output('feature-space', 'children'),
    [Input('data-table', 'columns')])
def update_feature_dropdown(cols):
    if "feature_names_display" not in parameter_dict:
        return dcc.Dropdown(
                id='feature-dropdown',
                options=[
                    {'label': i, 'value': i}
                    for i in ["NA"]
                ],
                value="NA"
                )
    return dcc.Dropdown(
                id='feature-dropdown',
                options=[
                    {'label': parameter_dict["feature_names_display"][i], 'value': parameter_dict["feature_names"][i]}
                    for i in range(len(parameter_dict["feature_names_display"]))
                ],
                value=parameter_dict["feature_names"][0]
                                    )

@app.callback(
    Output('graph-feature_corr', 'figure'),
    [Input('feature-dropdown', 'value')])
def update_figure_feature_corr(value):
    class_names = np.unique(parameter_dict["data"][parameter_dict["target_name"]])
    data, feature = parameter_dict["data"], parameter_dict["feature_names_dict"][value]
    return {
            "data": [go.Bar(
                                x=class_names,
                                y=[data[data[parameter_dict["target_name"]]==c][feature].mean() for c in class_names],
                                marker={"color": "#97151c"}
                            )],
            "layout": go.Layout(title=feature + " vs " + parameter_dict["target_name"],
                                margin={"l": 30, "r": 30, })
                                # legend={"x": 1, "y": 0.7})
    }

@app.callback(
    Output('data-map', 'figure'),
    [
        Input('size-slider', 'value'),
        Input('opacity-slider', "value"),
        Input('map-mode-dropdown', 'value')
    ]
)
def update_map(size, opacity, mapmode):
    if mapmode == "Unsupervised":
        parameter_dict["map"] = run_xmap(dataset=parameter_dict["dataname"], n_neighbors=15, negative_sample_rate=5, seed=2,
                                         return_step=STEP.UMAP_TRAINED)
    else:
        parameter_dict["map"] = run_xmap(dataset=parameter_dict["dataname"], n_neighbors=15, negative_sample_rate=5,
                                         seed=2, return_step=STEP.UMAP_TRAINED, learn_mode=LEARN.SUPERVISED)

    labels = parameter_dict["data"][parameter_dict["target_name"]].values
    colors = ['#2148bf', '#97151c', '#ffd6d6', '#faebeb', '#ffffff']
    return {
            'data': [
                go.Scattergl(
                    x=parameter_dict["map"][labels == i, 0],
                    y=parameter_dict["map"][labels == i, 1],
                    text=np.where(labels == i)[0],
                    mode='markers',
                    marker={
                        'size': size,
                        # 'line': {'width': 0.5, 'color': 'white'},
                        'color': colors[i],
                        'opacity': opacity
                    },
                    name="Class #" + str(i)
                ) for i in [int(c) for c in list(parameter_dict["target_classes"])]
            ],
            'layout': go.Layout(
                xaxis={'title': 'X-axis', 'zeroline': False},
                yaxis={'title': 'Y-axis', 'zeroline': False},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest',
                dragmode='lasso',
            )
        }

@app.callback(
    [
        Output('point-table', 'columns'),
        Output('point-table', 'data'),
        Output('point-table', 'style_data_conditional'),
        Output('selected-notes', 'children')
    ],
    [
        Input('data-map', 'selectedData'),
        Input('point-table', "page_current"),
        Input('point-table', "page_size")
    ]
)
def update_map_points(selected_points, page_current,page_size):
    style_data_conditional = parameter_dict["table_format"]
    # print(selected_points)
    if len(selected_points["points"]) != 0:
        selected_indices = [point["text"] for point in selected_points["points"]]
    else:
        selected_indices = []
    return [{"name": i, "id": i} for i in parameter_dict["data"].columns], \
           parameter_dict["data"].iloc[selected_indices[page_current*page_size:(page_current+ 1)*page_size]].to_dict('records'), \
           style_data_conditional, "{} data points are selected. See the raw data in the table below (from {} to {}):".format(len(selected_indices), page_current*page_size, (page_current+ 1)*page_size)

@app.callback(
    Output('datasum-table', 'children'),
    [Input('class-dropdown', 'value')])
def update_summary(value):
    if "data" not in parameter_dict:
        return pd.DataFrame([]).to_dict('records')
    if value == "All":
        dsum = parameter_dict["data"].describe(percentiles=[.1, .25, .5, .75, .9], include='all')
    else:
        val = value.split(" ")[1]
        dsum = parameter_dict["data"][parameter_dict["data"][parameter_dict["target_name"]]==int(val)].describe(percentiles=[.1, .25, .5, .75, .9], include='all')
    dsum = dsum.transpose()
    cname = dsum.columns
    dsum['stats'] = dsum.index
    dsum = dsum[["stats"] + [c for c in cname]]
    dsum = dsum.round(2)
    conditions = [
        {'if': {'column_id': "stats"},
         'backgroundColor': '#97151c',
         'color': 'white', }
    ]
    for c in dsum.columns[1:]:
        conditions.append(
            {
                'if': {'column_id': c, 'filter_query': '{{{0}}} != 0'.format(c)},
                'backgroundColor': '#dddddd',
                'color': 'black',
            }
        )
    return dash_table.DataTable(
                                 data=dsum.to_dict("records"),
                                 columns=[{"name": i, "id": i} for i in dsum.columns],
                                 fixed_rows={'headers': True, 'data': 0},
                                 # style_cell={'width': '150px'},
                                 style_table={'overflowY': 'scroll'},
                                 style_data_conditional=conditions
                                 )

@app.callback(
    [Output('data-network', 'figure'), Output('data-network', 'selectedData')],
    [
        Input('representation-dropdown', 'value'),
        Input('size-slider-network', 'value'),
        Input('opacity-slider-network', "value"),
        Input('mapping-mode-dropdown', 'value'),
    ]
)
def update_map_topo(plottype, size, opacity, mapmode):
    reload_data_objects(mapmode, plottype)
    return update_map_multi(plottype, size, opacity, mapmode, -1)

@app.callback(
    [
        Output('explain-map', 'figure'),
        Output('explain-map', 'selectedData'),
        Output('context-notes', 'children'),
    ],
    [
        Input('representation-dropdown', 'value'),
        Input('size-slider-network', 'value'),
        Input('opacity-slider-network', "value"),
        Input('mapping-mode-dropdown', 'value'),
        Input('context-dropdown', 'value'),
    ]
)
def update_map_explain(plottype, size, opacity, mapmode, context):
    map, spoints = update_map_multi(plottype, size, opacity, mapmode, context)
    if plottype != "Context":
        notes = html.H6(
                    "Insights",
                    className="subtitle padded"), \
                html.P(
                    'The data set "{}" has {} instances with {} features and can be divided into {} clusters.'
                        .format(dataset_dict[parameter_dict["dataname"]], len(parameter_dict["data"]), parameter_dict["nfeatures"],
                                parameter_dict["network_ncluster"])
                )
    else:
        if context==-1:
            context = 0
        rules = parameter_dict["explain_dict"][context]
        fnames = []
        vals = []
        for truef in rules[1]:
            fnames.append(truef)
            vals.append('True')
        for falsef in rules[2]:
            fnames.append(falsef)
            vals.append('False')
        fnames = [parameter_dict["explain_feature_names"][fn] for fn in fnames]
        fdat = pd.DataFrame.from_dict({'Feature': fnames, 'Value': vals})
        parameter_dict["context_table"] = fdat
        notes = html.H6(
                "Description of Context #{}".format(1+context),
                className="subtitle padded"),\
                html.P(
                id="rule-notes"), \
                dash_table.DataTable(
                    id='context-table',
                    # columns=[{"name": i, "id": i} for i in fdat.columns],
                    columns=[{"name": "Feature", "id": "Feature"}],
                    # data=fdat.to_dict('records'),
                    page_current=0,
                    page_size=7,
                    page_action='custom',
                    style_table={'overflowX': 'scroll'},
                    style_data_conditional=[
                        {
                            'if': {
                                'column_id': 'Feature',
                                'filter_query': '{Value} eq "True"'
                            },
                            'backgroundColor': '#3D9970',
                            'color': 'white',
                        },
                        {
                            'if': {
                                'column_id': 'Feature',
                                'filter_query': '{Value} eq "False"'
                            },
                            'backgroundColor': '#97151c',
                            'color': 'white',
                        },
                    ]
                )

    return map, spoints, notes

@app.callback(
    [
        Output('context-table', 'data'),
        Output('rule-notes', 'children')
    ],
    [
     Input('context-table', "page_current"),
     Input('context-table', "page_size")])
def update_context_table(page_current,page_size):
    if "context_table" not in parameter_dict: # or page_current*page_size > len(parameter_dict["context_table"]):
        return page_current*page_size > len(parameter_dict["context_table"]), "End of Table."
    return parameter_dict["context_table"].iloc[page_current*page_size:(page_current+ 1)*page_size].to_dict('records'), \
           "{} to {} from {} Active Features (Green is for True and Red is for False)".format(page_current*page_size, min((page_current+ 1)*page_size, len(parameter_dict["context_table"])), len(parameter_dict["context_table"]))

def update_map_multi(plottype, size, opacity, mapmode, context):
    colors = 10 * ['#2148bf', '#17b35a', '#b3b315', '#0f0f02', '#3a0fba',
                   '#ba0fac', '#0eb3a5', '#b3660e', '#8db89a',
                   '#00fcb9', '#00a8fc', '#fceb00', '#fc0000']
    if plottype == "Network":
        edge_x = []
        edge_y = []
        for i in range(0, parameter_dict["network_nodes"].shape[0]):
            for j in range(0, parameter_dict["network_nodes"].shape[0]):
                if parameter_dict["network_connection"][i, j] != 0:
                    x0, y0 = parameter_dict["network_nodes"][i, 0], parameter_dict["network_nodes"][i, 1]
                    x1, y1 = parameter_dict["network_nodes"][j, 0], parameter_dict["network_nodes"][j, 1]
                    edge_x.append(x0)
                    edge_x.append(x1)
                    edge_x.append(None)
                    edge_y.append(y0)
                    edge_y.append(y1)
                    edge_y.append(None)
        traces = [
            go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none',
                    mode='lines',
                    name= "Connection"
                )
        ]

        node_ids = parameter_dict["network_cluster_names"]
        for k in np.unique(parameter_dict["network_cluster_names"]):
            traces.append(
                go.Scatter(
                x=parameter_dict["network_nodes"][node_ids == k,0], y=parameter_dict["network_nodes"][node_ids==k,1],
                mode='markers',
                text=np.where(node_ids == k)[0],
                marker={
                    'size': size,
                    'color': colors[k],
                    'opacity': opacity
                },
                name="Cluster #" + str(k)
                )
            )

        return {
                'data': traces,
                'layout': go.Layout(
                    xaxis={'title': ' ', 'zeroline': False},
                    yaxis={'title': ' ', 'zeroline': False},
                    margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                    legend={'orientation':"h"},
                    hovermode='closest',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    dragmode='lasso',
                )
            }, {'points': [], 'range': None}
    else:
        if plottype == "Labelled Map":
            labels = parameter_dict["data"][parameter_dict["target_name"]].values
        elif plottype == "Context":
            labels = parameter_dict["context_vector"][::, context]
        else:
            labels = parameter_dict["network_indices_to_clusters"]
        return {
            'data': [
                go.Scattergl(
                    x=parameter_dict["map"][labels == int(i), 0],
                    y=parameter_dict["map"][labels == int(i), 1],
                    text=np.where(labels == int(i))[0],
                    mode='markers',
                    marker={
                        'size': int(size/3),
                        # 'line': {'width': 0.5, 'color': 'white'},
                        'color': colors[int(i)] if plottype == "Clustered Map" else ['#9c9c9c' , '#000000'][int(i)] if plottype == "Context" else  ['#2148bf', '#97151c'][int(i)],
                        'opacity': opacity
                    },
                    name= "Cluster #" + str(int(i)) if plottype == "Clustered Map" else "Class #" + str(int(i)) if plottype == "Labelled Map" else "Matching=" + str(bool(i))
                ) for i in np.unique(labels) #range(parameter_dict["network_ncluster"])
            ],
            'layout': go.Layout(
                xaxis={'title': '', 'zeroline': False},
                yaxis={'title': '', 'zeroline': False},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'orientation':"h"},
                hovermode='closest',
                dragmode='lasso',
            )
        }, {'points': [], 'range': None}

@app.callback(
    [
        Output('node-table', 'columns'),
        Output('node-table', 'data'),
        Output('node-table', 'style_data_conditional'),
        Output('selected-node-notes', 'children')
    ],
    [
        Input('representation-dropdown', 'value'),
        Input('data-network', 'selectedData'),
        Input('node-table', "page_current"),
        Input('node-table', "page_size"),
    ]
)
def update_network_points(plottype, selected_points, page_current,page_size):
    return update_multi_points(plottype, selected_points, page_current, page_size)

@app.callback(
    [
        Output('data-explain-table', 'columns'),
        Output('data-explain-table', 'data'),
        Output('data-explain-table', 'style_data_conditional'),
        Output('selected-explain-notes', 'children')],
    [
        Input('representation-dropdown', 'value'),
        Input('explain-map', 'selectedData'),
        Input('data-explain-table', "page_current"),
        Input('data-explain-table', "page_size"),
    ]
)
def update_explain_points(plottype, selected_points, page_current,page_size):
    return update_multi_points(plottype, selected_points, page_current,page_size)

def update_multi_points(plottype, selected_points, page_current,page_size):
    if plottype == "Network":
        labels = parameter_dict["network_cluster_names"]
        buffer = 0
    elif plottype == "Labelled Map":
        labels = parameter_dict["data"][parameter_dict["target_name"]].values
        buffer = 0
    else:
        labels = parameter_dict["network_indices_to_clusters"]
        buffer = 1
    indexpoints = {}
    lastpoint = 0
    for i in [int(c) for c in np.unique(labels)]:
        indexpoints[i] = lastpoint
        if plottype == "Network":
            lastpoint = indexpoints[i] + len(parameter_dict["network_nodes"][labels == i, 0])
        else:
            lastpoint = indexpoints[i] + len(parameter_dict["map"][labels == i, 0])
    style_data_conditional = parameter_dict["table_format"]
    if plottype == "Network":
        nnodes = len(selected_points["points"])

    update = not (len(selected_points["points"]) == 0)
    for point in selected_points["points"]:
        if point["curveNumber"]+buffer not in indexpoints:
            update = False
            break

    if update == 0:
        selected_indices = []
    else:
        selected_indices = [point["text"] for point in selected_points["points"]] #indexpoints[point["curveNumber"]+buffer] +
        if plottype == "Network":
            point_node_id = np.array(parameter_dict["network_node_indices"])
            select_list = []
            for node in selected_indices:
                select_list += np.where(point_node_id == node)[0].tolist()
            selected_indices = select_list
    # print(selected_indices)
    if page_current*page_size > len(selected_indices):
        return [{"name": i, "id": i} for i in parameter_dict["data"].columns], \
               pd.DataFrame([]).to_dict('records'), \
               style_data_conditional, "End of Table."
    if plottype == "Network":
        return [{"name": i, "id": i} for i in parameter_dict["data"].columns], \
               parameter_dict["data"].round(3).iloc[selected_indices[page_current*page_size:(page_current+ 1)*page_size]].to_dict('records'), \
               style_data_conditional, "{} data points are extracted from {} selected nodes. See the raw data in the table below (from {} to {}):".format(len(selected_indices), nnodes, page_current*page_size, (page_current+ 1)*page_size)
    else:
        return [{"name": i, "id": i} for i in parameter_dict["data"].columns], \
               parameter_dict["data"].round(3).iloc[
                   selected_indices[page_current * page_size:(page_current + 1) * page_size]].to_dict('records'), \
               style_data_conditional, "{} data points are selected. See the raw data in the table below (from {} to {}):".format(
            len(selected_indices), page_current * page_size, (page_current + 1) * page_size)

@app.callback(
    Output('context-select', 'children'),
    [
        Input('representation-dropdown', 'value'),
        Input('mapping-mode-dropdown', 'value')
    ]
)
def update_context_dropdown(plottype, mapmode):
    reload_data_objects(mapmode, plottype)
    if plottype=="Context":
        return [
            dcc.Dropdown(
                                id='context-dropdown',
                                options=[
                                    {'label': "Context #" + str(i+1), 'value': i}
                                        for i in parameter_dict["explain_dict"]
                                ],
                                value=0,
                                # labelStyle={'display': 'inline-block'}
                            ),
        ]
    else:
        return [
            dcc.Dropdown(
                                id='context-dropdown',
                                options=[
                                    {'label': "No Context", 'value': 0}
                                ],
                                value=0,
                                # labelStyle={'display': 'inline-block'}
                            ),
        ]


def reload_data_objects(mapmode, plottype):
    if plottype == "Context":
        if mapmode == "Unsupervised":
            embeddings, nodes, connection, classes, nclusters, node_indices, indices, cluster_explainer_dict, xcluster_id_details, xfeaturenames \
                = run_xmap(dataset=parameter_dict["dataname"], n_neighbors=15, negative_sample_rate=5, seed=2,
                           return_step=STEP.CONTEXT_EXPLAINED)
        else:
            embeddings, nodes, connection, classes, nclusters, node_indices, indices, cluster_explainer_dict, xcluster_id_details, xfeaturenames \
                = run_xmap(dataset=parameter_dict["dataname"], n_neighbors=15, negative_sample_rate=5,
                           seed=2, return_step=STEP.CONTEXT_EXPLAINED, learn_mode=LEARN.SUPERVISED)
    else:
        if mapmode == "Unsupervised":
            embeddings, nodes, connection, classes, nclusters, node_indices, indices = \
                run_xmap(dataset=parameter_dict["dataname"], n_neighbors=15, negative_sample_rate=5, seed=2,
                         return_step=STEP.SOINN_TRAINED)
        else:
            embeddings, nodes, connection, classes, nclusters, node_indices, indices = \
                run_xmap(dataset=parameter_dict["dataname"], n_neighbors=15, negative_sample_rate=5,
                         seed=2, return_step=STEP.SOINN_TRAINED, learn_mode=LEARN.SUPERVISED)
    parameter_dict["map"] = embeddings
    parameter_dict["network_nodes"] = nodes
    parameter_dict["network_connection"] = connection
    parameter_dict["network_cluster_names"] = classes
    parameter_dict["network_ncluster"] = nclusters
    parameter_dict["network_node_indices"] = node_indices
    parameter_dict["network_indices_to_clusters"] = indices
    if plottype == "Context":
        parameter_dict["explain_dict"] = cluster_explainer_dict
        parameter_dict["context_vector"] = xcluster_id_details
        parameter_dict["explain_feature_names"] = xfeaturenames
        parameter_dict["context_data"] = pd.DataFrame(parameter_dict["context_vector"],
                                                      columns=["Context #" + str(c + 1) for c in
                                                               range(parameter_dict["network_ncluster"])])


@app.callback(
    [
        Output('raw-data-table', 'columns'),
        Output('context-data-table', 'columns'),
        Output('raw-data-table', 'style_data_conditional'),
        Output('context-data-table', 'style_data_conditional')
    ],
    [
        Input('mapping-mode-dropdown', 'value'),
        Input('explain-map', 'figure'),
    ]
)
def update_output(mapmode, figure): #
    allcons = []
    coldata = ['#3D9970', '#f5d442', ]
    count = 0
    for data in [parameter_dict["data"], parameter_dict["context_data"]]:
        conditions = []
        for c in data.columns:
            conditions.append(
                {
                    'if': {'column_id': c, 'filter_query': '{{{0}}} != 0'.format(c)},
                    'backgroundColor': coldata[count],
                    'color': 'black',
                }
            )
        allcons.append(conditions)
        count += 1
    return [{"name": i, "id": i} for i in parameter_dict["data"].columns], \
           [{"name": i, "id": i} for i in parameter_dict["context_data"].columns],\
           allcons[0], allcons[1]

@app.callback(
    [
        Output('raw-data-table', 'data'),
        Output('context-data-table', 'data')
     ],
    [
        Input('context-data-table', "page_current"),
        Input('context-data-table', "page_size"),
        Input('explain-map', 'figure'),
        Input('context-dropdown', 'value'),
        Input('raw-data-table', 'columns'),
    ]
)
def update_table(page_current,page_size, figure, context, dat):
    if "context_data" not in parameter_dict:
        return pd.DataFrame([]).to_dict('records'), pd.DataFrame([]).to_dict('records')
    else:
        filter = parameter_dict["context_data"]["Context #" + str(context+1)] == 1
        return parameter_dict["data"][filter].round(3).iloc[
               page_current * page_size:(page_current + 1) * page_size].to_dict('records'), \
               parameter_dict["context_data"][filter].round(3).iloc[page_current * page_size:(page_current + 1) * page_size].to_dict('records')
    # return parameter_dict["data"].round(3).iloc[page_current*page_size:(page_current+ 1)*page_size].to_dict('records'), parameter_dict["context_data"].round(3).iloc[page_current*page_size:(page_current+ 1)*page_size].to_dict('records')

@app.callback(
    [
        Output('param1', 'min'),
        Output('param1', 'max'),
        Output('param1', 'step'),
        Output('param1', 'value'),
        Output('param1', 'handleLabel'),
        Output('param2', 'min'),
        Output('param2', 'max'),
        Output('param2', 'step'),
        Output('param2', 'value'),
        Output('param2', 'handleLabel'),
    ],
    [
        Input('ml-dropdown', 'value'),
        Input('explain-map', 'figure'),
    ]
)
def update_output(mlalg, fig):
    if "Logistic Regression" in mlalg:
        return 0.01, 2, 0.01, 0.1, {"showCurrentValue": True, "label": "C"}, \
               0, 10, 1e-4, 1e-4, {"showCurrentValue": True, "label": "Tolerance"}
    elif "Decision Tree" in mlalg:
        return 1, 10, 1, 5.0, {"showCurrentValue": True, "label": "Depth"}, \
               2, 20, 1, 2, {"showCurrentValue": True, "label": "MinSplit"}
    elif "ANN" in mlalg:
        return 10, 200, 1, 100, {"showCurrentValue": True, "label": "MaxIter"}, \
               5, 200, 1, 15, {"showCurrentValue": True, "label": "HLSize"}
    elif "XGBoost" in mlalg:
        return 2, 10, 1, 3, {"showCurrentValue": True, "label": "MaxDepth"}, \
               5, 200, 1, 100, {"showCurrentValue": True, "label": "Ensembles"}
    elif "Skope Rules" in mlalg:
        return 2, 10, 1, 2, {"showCurrentValue": True, "label": "MaxDepth"}, \
               5, 200, 1, 10, {"showCurrentValue": True, "label": "Ensembles"}
    return 111, 222, 5, 33

@app.callback(
    [
        Output('ml-run-log', 'children'),
        Output('feature-contribution', 'figure')
    ],
    [
        Input('runml-button', 'n_clicks'),
    ],
    [
        State('context-dropdown', 'value'),
        State('mapping-mode-dropdown', 'value'),
        State('ml-dropdown', 'value'),
        State('param1', 'value'),
        State('param2', 'value'),
        State('data_select', 'value'),
    ]
)
def runml(click, context, learnmode, mlalg, p1, p2, datmode):
    print("Learning ...")
    nfold = 10
    if datmode == "All":
        return runall(nfold)
    if "Logistic Regression" in mlalg:
        if "XMAP" in mlalg:
            if "Context" in mlalg:
                nameclf = "{}_xmapcontextlogreg_{}_{}_{}_fold_{}".format(parameter_dict["dataname"], learnmode, p1, p2, nfold)
            else:
                nameclf = "{}_xmaplogreg_{}_{}_{}_fold_{}".format(parameter_dict["dataname"], learnmode, p1, p2, nfold)
        else:
            nameclf = "{}_logreg_{}_{}_fold_{}".format(parameter_dict["dataname"], p1, p2, nfold)
        # display_clf = "Logistic Regression with C={} and Tolerance={}".format(p1,p2)
        if nameclf not in parameter_dict:
            from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
            clf = LogisticRegression(random_state=0, penalty='l1', tol=p2, solver='liblinear', C=p1, max_iter=1000)
            X, Y, fnames = setup_training_set(mlalg, nameclf)
            # cv
            acc, auc, _, _ = traing_and_get_measures(X, Y, clf, nfold)
            clf.fit(X,Y)
            parameter_dict[nameclf] = (clf, auc, acc)
            # print(clf.coef_)
            sorti = np.argsort(-clf.coef_[0])
            fnames = fnames[sorti]
            coefs = clf.coef_[0][sorti]
            if "Context" not in mlalg:
                pos = np.where(coefs > 0)
                neg = np.where(coefs < 0)
            else:
                coefs = np.array([coefs[i] if ("Original_" in fnames[i] or "Context"+str(context+1)+"_" in fnames[i]) else 0 for i in range(fnames.shape[0])])
                fnames = [f.replace("Original_", "") if "Original_" in f else f for f in fnames]
                # fnames = [f.replace("Context"+str(context+1)+"_", "") if "Context"+str(context+1)+"_" in f else f for f in fnames]
                fnames = np.array(fnames)
                pos = np.where(coefs > 0)
                neg = np.where(coefs < 0)
            return "Complete training `id:{}` -- Metric: `AUC={:.3f}`, `Accuracy={:.3f}`.".format(nameclf, auc, acc), \
                   build_effect_chart(coefs, fnames, neg, pos)
        else:
            # print() #'#3D9970',
            sorti = np.argsort(-parameter_dict[nameclf][0].coef_[0])
            fnames = parameter_dict[nameclf+"final_feature_names"][sorti]
            coefs = parameter_dict[nameclf][0].coef_[0][sorti]
            if "Context" not in mlalg:
                pos = np.where(coefs > 0)
                neg = np.where(coefs < 0)
            else:
                coefs = np.array([coefs[i] if ("Original_" in fnames[i] or "Context"+str(context+1)+"_" in fnames[i]) else 0 for i in range(fnames.shape[0])])
                fnames = [f.replace("Original_", "") if "Original_" in f else f for f in fnames]
                # fnames = [f.replace("Context"+str(context+1)+"_", "") if "Context"+str(context+1)+"_" in f else f for f in fnames]
                fnames = np.array(fnames)
                pos = np.where(coefs > 0)
                neg = np.where(coefs < 0)
            # print(coefs)
            return "Pretrained Model loaded for with `id:{}` -- Metric: `AUC={:.3f}`, `Accuracy={:.3f}`.".format(nameclf,
                                parameter_dict[nameclf][1], parameter_dict[nameclf][2]), \
                   build_effect_chart(coefs, fnames, neg, pos)
    if "Decision Tree" in mlalg:
        if "XMAP" in mlalg:
            nameclf = "{}_xmapDTree_{}_{}_{}_fold_{}".format(parameter_dict["dataname"], learnmode, p1, p2, nfold)
        else:
            nameclf = "{}_DTree_{}_{}_fold_{}".format(parameter_dict["dataname"], p1, p2, nfold)
        # display_clf = "Logistic Regression with C={} and Tolerance={}".format(p1,p2)
        if nameclf not in parameter_dict:
            from sklearn.tree import DecisionTreeClassifier
            clf = DecisionTreeClassifier(random_state=0, max_depth=p1, min_samples_split=p2)
            from sklearn.neighbors import KNeighborsClassifier
            clf = KNeighborsClassifier(n_neighbors=3)
            X, Y, fnames = setup_training_set(mlalg, nameclf)
            # cv
            acc, auc, _, _ = traing_and_get_measures(X, Y, clf, nfold)
            clf.fit(X,Y)
            parameter_dict[nameclf] = (clf, auc, acc)
            return "Complete training `id:{}` -- Metric: `AUC={:.3f}`, `Accuracy={:.3f}`.".format(nameclf, auc, acc), \
                   build_dummy_chart([parameter_dict[nameclf][1], parameter_dict[nameclf][2]])
        else:
            return "Pretrained Model loaded for with `id:{}` -- Metric: `AUC={:.3f}`, `Accuracy={:.3f}`.".format(nameclf,
                                parameter_dict[nameclf][1], parameter_dict[nameclf][2]), \
                   build_dummy_chart([parameter_dict[nameclf][1], parameter_dict[nameclf][2]])
    if "ANN" in mlalg:
        if "XMAP" in mlalg:
            nameclf = "{}_xmapANN_{}_{}_{}_fold_{}".format(parameter_dict["dataname"], learnmode, p1, p2, nfold)
        else:
            nameclf = "{}_ANN_{}_{}_fold_{}".format(parameter_dict["dataname"], p1, p2, nfold)
        # display_clf = "Logistic Regression with C={} and Tolerance={}".format(p1,p2)
        if nameclf not in parameter_dict:
            from sklearn.neural_network import MLPClassifier
            clf = MLPClassifier(hidden_layer_sizes=(p2,), random_state=1, max_iter=p1, warm_start=True)
            X, Y, fnames = setup_training_set(mlalg, nameclf)
            # cv
            acc, auc, _, _ = traing_and_get_measures(X, Y, clf, nfold)
            clf.fit(X,Y)
            parameter_dict[nameclf] = (clf, auc, acc)
            return "Complete training `id:{}` -- Metric: `AUC={:.3f}`, `Accuracy={:.3f}`.".format(nameclf, auc, acc), \
                   build_dummy_chart([parameter_dict[nameclf][1], parameter_dict[nameclf][2]])
        else:
            return "Pretrained Model loaded for with `id:{}` -- Metric: `AUC={:.3f}`, `Accuracy={:.3f}`.".format(nameclf,
                                parameter_dict[nameclf][1], parameter_dict[nameclf][2]), \
                   build_dummy_chart([parameter_dict[nameclf][1], parameter_dict[nameclf][2]])
    if "XGBoost" in mlalg:
        if "XMAP" in mlalg:
            nameclf = "{}_xmapXGBoost_{}_{}_{}_fold_{}".format(parameter_dict["dataname"], learnmode, p1, p2, nfold)
        else:
            nameclf = "{}_XGBoost_{}_{}_fold_{}".format(parameter_dict["dataname"], p1, p2, nfold)
        # display_clf = "Logistic Regression with C={} and Tolerance={}".format(p1,p2)
        if nameclf not in parameter_dict:
            from xgboost import XGBClassifier
            clf = XGBClassifier(max_depth=p1, n_estimators=p2)
            X, Y, fnames = setup_training_set(mlalg, nameclf)
            # cv
            acc, auc, _, _ = traing_and_get_measures(X, Y, clf, nfold)
            clf.fit(X,Y)
            parameter_dict[nameclf] = (clf, auc, acc)
            return "Complete training `id:{}` -- Metric: `AUC={:.3f}`, `Accuracy={:.3f}`.".format(nameclf, auc, acc), \
                   build_dummy_chart([parameter_dict[nameclf][1], parameter_dict[nameclf][2]])
        else:
            return "Pretrained Model loaded for with `id:{}` -- Metric: `AUC={:.3f}`, `Accuracy={:.3f}`.".format(nameclf,
                                parameter_dict[nameclf][1], parameter_dict[nameclf][2]), \
                   build_dummy_chart([parameter_dict[nameclf][1], parameter_dict[nameclf][2]])
    if "Skope Rules" in mlalg:
        if "XMAP" in mlalg:
            nameclf = "{}_xmapSKrules_{}_{}_{}_fold_{}".format(parameter_dict["dataname"], learnmode, p1, p2, nfold)
        else:
            nameclf = "{}_SKrules_{}_{}_fold_{}".format(parameter_dict["dataname"], p1, p2, nfold)
        # display_clf = "Logistic Regression with C={} and Tolerance={}".format(p1,p2)
        if nameclf not in parameter_dict:
            from skrules import SkopeRules
            X, Y, fnames = setup_training_set(mlalg, nameclf)
            clf = SkopeRules(n_estimators=p2, max_depth=p1, #precision_min=0.95,
                             feature_names=fnames, n_jobs=4,
                             random_state=1)
            # cv
            acc, auc, _, _ = traing_and_get_measures(X, Y, clf, nfold)
            clf.fit(X,Y)
            parameter_dict[nameclf] = (clf, auc, acc)
            return "Complete training `id:{}` -- Metric: `AUC={:.3f}`, `Accuracy={:.3f}`.".format(nameclf, auc, acc), \
                   build_dummy_chart([parameter_dict[nameclf][1], parameter_dict[nameclf][2]])
        else:
            return "Pretrained Model loaded for with `id:{}` -- Metric: `AUC={:.3f}`, `Accuracy={:.3f}`.".format(nameclf,
                                parameter_dict[nameclf][1], parameter_dict[nameclf][2]), \
                   build_dummy_chart([parameter_dict[nameclf][1], parameter_dict[nameclf][2]])
    return "Running ..."


def traing_and_get_measures(X, Y, clf, nfold):
    kf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=100)
    scoring = ['roc_auc', 'accuracy']
    scores = cross_validate(clf, X, Y, scoring=scoring, cv=kf, return_train_score=False)
    # print(scores)
    auc = np.mean(scores['test_roc_auc'])
    acc = np.mean(scores['test_accuracy'])
    aucstd = np.std(scores['test_roc_auc'])
    accstd = np.std(scores['test_accuracy'])
    return acc, auc, accstd, aucstd


def setup_training_set(mlalg, nameclf):
    if "XMAP" in mlalg:
        if "Context" in mlalg:
            fnames = list(parameter_dict["data"].columns[1::])
            fnames = ["Original_" + f for f in fnames]
            X = parameter_dict["data"].values[::, 1::]
            for i in range(parameter_dict["context_vector"].shape[1]):
                X_c = ((parameter_dict["data"].values[::, 1::].T*parameter_dict["context_vector"][::, i])).T
                X = np.hstack((X, X_c))
                fnames = fnames + list("Context" + str(i+1) + "_" + c for c in parameter_dict["data"].columns[1::])
            fnames = np.array(fnames)
        else:
            X = np.hstack((parameter_dict["data"].values[::, 1::], parameter_dict["context_vector"]))
            fnames = np.array(list(parameter_dict["data"].columns[1::]) + list(parameter_dict["context_data"]))
    else:
        X = parameter_dict["data"].values[::, 1::]
        fnames = np.array(parameter_dict["data"].columns[1::])
    # X = parameter_dict["context_vector"]
    # fnames = np.array(list(parameter_dict["context_data"]))
    Y = parameter_dict["data"].values[::, 0]
    parameter_dict[nameclf + "final_feature_names"] = fnames
    return X, Y, fnames


def build_effect_chart(coefs, fnames, neg, pos):
    print("Feature impact")
    # print("Positive weights")
    for i in pos[0]:
        print(fnames[i], ",", coefs[i])
    # print("Negative weights")
    for i in neg[0]:
        print(fnames[i], ",",  coefs[i])
    return {
        'data': [
            go.Bar(x=fnames[pos], y=coefs[pos],
                   marker_color='#97151c',
                   text=fnames[pos],
                   name='Increasing Risk', orientation='v'),
            go.Bar(x=fnames[neg], y=coefs[neg],
                   marker_color='#3D9970',
                   text=fnames[neg],
                   name='Decreasing Risk', orientation='v')
        ],
        'layout': go.Layout(
            xaxis=dict(
                title="",
                showgrid=False,
                showline=False,
                showticklabels=False,
                zeroline=False,
                # domain=[0.15, 1]
            ),
            title="Feature Contributions",
            legend={'orientation': "h"},
        )
    }

def build_dummy_chart(measures):
    return {
        'data': [
            go.Bar(y=["AUC", "Accuracy"], x=measures,
                   marker_color='#97151c',
                   name='expenses', orientation='h')
        ],
    }

def runall(nfold):
    dats = datasets
    mlalg = "XMAP"
    X, Y, fnames = setup_training_set(mlalg, "")
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neural_network import MLPClassifier
    from xgboost import XGBClassifier
    from skrules import SkopeRules

    outdata = pd.DataFrame(columns=['Algorithm', 'Dataset', "Parameter", 'AUC', "AUC_STD", "ACC", "ACC_STD", 'XMAP_AUC', 'XMAP_AUC_STD', "XMAP_ACC", "XMAP_ACC_STD", 'LAYER_AUC', 'LAYER_AUC_STD', "LAYER_ACC", "LAYER_ACC_STD"])
    prs = [1.0, 0.5, 0.1]
    count = 0
    for parameterml in prs:
        algs = {
            "Logistic Regression": LogisticRegression(random_state=0, penalty='l1', tol=0.0001, solver='liblinear', C=parameterml, max_iter=5000),
            "Decision Tree": DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_split=2),
            "ANN": MLPClassifier(hidden_layer_sizes=(15,), random_state=1, max_iter=100, warm_start=True),
            "XGBoost": XGBClassifier(max_depth=3, n_estimators=100),
            "ScopeRules": SkopeRules(n_estimators=10, max_depth=2,  # precision_min=0.95,
                         feature_names=fnames, n_jobs=4,
                         random_state=1)
        }
        for data in dats:
            parameter_dict["dataname"] = data
            X_norm, Y, scaler, nfeatures, feature_names, target_name = run_xmap(dataset=data, n_neighbors=15,
                                                                                negative_sample_rate=5, seed=2,
                                                                                return_step=STEP.DATA_CLEANED)
            feature_names_display = [ff.replace("ubar", "_").replace("dot", ".") for ff in feature_names]
            dd = pd.DataFrame(np.concatenate((Y, X_norm), axis=1), columns=[target_name] + feature_names_display)
            parameter_dict["data"] = dd
            parameter_dict["dataname"] = data
            parameter_dict["nfeatures"] = nfeatures
            parameter_dict["feature_names"] = feature_names
            parameter_dict["feature_names_display"] = feature_names_display
            parameter_dict["feature_names_dict"] = {feature_names[i]: feature_names_display[i] for i in range(nfeatures)}
            parameter_dict["target_name"] = target_name
            parameter_dict["target_classes"] = np.unique(Y)
            assign_parameters()
            for alg in algs:
                if alg != "Logistic Regression" and count >=1:
                    break
                print(alg, data, parameterml)
                X, Y, fnames = setup_training_set("", "")
                acc, auc, accstd, aucstd = traing_and_get_measures(X, Y, algs[alg], nfold)
                X, Y, fnames = setup_training_set("XMAP", "")
                xmap_acc, xmap_auc, xmap_accstd, xmap_aucstd = traing_and_get_measures(X, Y, algs[alg], nfold)
                X, Y, fnames = setup_training_set("XMAP Context", "")
                layer_acc, layer_auc, layer_accstd, layer_aucstd = traing_and_get_measures(X, Y, algs[alg], nfold)
                outdata.loc[len(outdata)] = [alg, data, parameterml if alg == "Logistic Regression" else "NA"
                    , auc, aucstd, acc, accstd,
                                             xmap_auc, xmap_aucstd, xmap_acc, xmap_accstd,
                                             layer_auc, layer_aucstd, layer_acc, layer_accstd]
        count += 1

    print(outdata)
    outdata.to_csv("performance_xmap.csv")
    return 1

if __name__ == "__main__":
    app.run_server(debug=True)
