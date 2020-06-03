import dash_html_components as html
import dash_core_components as dcc


def Header(app):
    return html.Div([get_header(app), html.Br([]), get_menu()])


def get_header(app):
    header = html.Div(
        [
            html.Div(
                [
                    html.Img(
                        src=app.get_asset_url("new-xmap-logo3.png"),
                        className="logo"
                    ),
                    html.A(
                        html.Button("Contact", id="contact-button"),
                        href="https://scholars.latrobe.edu.au/display/snguyen",
                    ),
                ],
                className="row",
            ),
            html.Div(
                [
                    html.Div(
                        [html.H5("XMAP: eXplainable Mapping Analytical Process")],
                        className="seven columns main-title",
                    ),
                    html.Div(
                        [
                            dcc.Link(
                                "About",
                                href="/dash-financial-report/full-view",
                                className="full-view-link",
                            )
                        ],
                        className="five columns",
                    ),
                ],
                className="twelve columns",
                style={"padding-left": "0"},
            ),
        ],
        className="row",
    )
    return header


def get_menu():
    menu = html.Div(
        [
            dcc.Link(
                "Overview",
                href="/dash-xmap/overview",
                className="tab first",
            ),
            dcc.Link(
                "Basic Statistics", href="/dash-xmap/statistics", className="tab"
            ),
            dcc.Link(
                "Data Map", href="/dash-xmap/map", className="tab"
            ),
            dcc.Link(
                "Topology", href="/dash-xmap/topology", className="tab"
            ),
            dcc.Link(
                "Explainable Contexts", href="/dash-xmap/explainablecontext", className="tab"
            ),
            dcc.Link(
                "Predictive Modeling", href="/dash-xmap/predictivemodel", className="tab"
            ),
        ],
        className="row all-tabs",
    )
    return menu


def make_dash_table(df):
    """ Return a dash definition of an HTML table for a Pandas dataframe """
    table = []
    for index, row in df.iterrows():
        html_row = []
        for i in range(len(row)):
            html_row.append(html.Td([row[i]]))
        table.append(html.Tr(html_row))
    return table
