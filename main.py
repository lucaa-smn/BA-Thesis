# main.py
import dash
from dash import html, dcc, Input, Output
import pandas as pd
import plotly.express as px
from data_gen import DataGen
from modelle.knn import TimeSeriesNN
from modelle.svm import SVM
from modelle.decision_tree import DecisionTree
from modelle.random_forest import RandomForest
from modelle import section


def create_plot(df, sim):
    df2 = df.copy()
    df2["trajectory_id"] = df2.groupby(["theta", "v0"]).ngroup()

    erste_drei_ids = df2["trajectory_id"].unique()[:3]
    df_drei = df2[df2["trajectory_id"].isin(erste_drei_ids)]

    fig = px.line(
        df_drei,
        x="x",
        y="y",
        color="trajectory_id",
        markers=True,
        title="Erste drei Wurfverläufe (verbundene Zeitreihen)",
        labels={"x": "x [m]", "y": "y [m]", "trajectory_id": "Verlauf-ID"},
    )
    return fig


def main():
    # Daten erzeugen
    sim = DataGen(hs=2.13)
    df = sim.generate_dataset(n=500)
    df_noisy = sim.generate_dataset(n=500, with_noise=True)

    # Dash-App erstellen
    app = dash.Dash(__name__, suppress_callback_exceptions=True)

    # Modelle vorbereiten
    models_normal = {
        "Neural Network": TimeSeriesNN(app=app, data=df),
        "SVM": SVM(app=app, data=df),
        "Decision Tree": DecisionTree(app=app, data=df),
        "Random Forest (Classif.)": RandomForest(
            app=app, data=df, target_column="label", task="classification"
        ),
        "Random Forest (Regr.)": RandomForest(
            app=app, data=df, target_column="label", task="regression"
        ),
    }

    models_noisy = {
        "Neural Network (noisy)": TimeSeriesNN(app=app, data=df_noisy),
        "SVM (noisy)": SVM(app=app, data=df_noisy),
        "Decision Tree (noisy)": DecisionTree(app=app, data=df_noisy),
    }

    # Navbar-Links generieren
    def navbar():
        return html.Div(
            [
                html.H2(
                    "Modelle",
                    className="text-center",
                    style={"margin-bottom": "15px", "color": "#343a40"},
                ),
                html.Hr(),
                html.Div(
                    [
                        dcc.Link(
                            name,
                            href=f"/{name.replace(' ', '_')}",
                            style={
                                "display": "block",
                                "padding": "10px 15px",
                                "margin": "5px 0",
                                "border-radius": "8px",
                                "text-decoration": "none",
                                "color": "#212529",
                                "font-weight": "500",
                            },
                            className="nav-link",
                        )
                        for name in models_normal.keys()
                    ],
                    style={"margin-bottom": "30px"},
                ),
                html.Hr(),
                html.H4(
                    "Noisy Models",
                    className="text-center",
                    style={"margin-bottom": "15px", "color": "#495057"},
                ),
                html.Div(
                    [
                        dcc.Link(
                            name,
                            href=f"/{name.replace(' ', '_')}",
                            style={
                                "display": "block",
                                "padding": "10px 15px",
                                "margin": "5px 0",
                                "border-radius": "8px",
                                "text-decoration": "none",
                                "color": "#495057",
                                "font-weight": "500",
                            },
                            className="nav-link",
                        )
                        for name in models_noisy.keys()
                    ],
                ),
            ],
            style={
                "padding": "20px",
                "background-color": "#ffffff",
                "width": "260px",
                "position": "fixed",
                "height": "100%",
                "overflow-y": "auto",
                "border-right": "1px solid #dee2e6",
                "box-shadow": "2px 0 8px rgba(0,0,0,0.05)",
            },
            className="shadow-sm",
        )

    # Layout mit Navbar + Content
    app.layout = html.Div(
        [
            dcc.Location(id="url"),
            navbar(),
            html.Div(
                id="page-content", style={"margin-left": "22%", "padding": "20px"}
            ),
        ]
    )

    # Callback zum Seitenwechsel
    @app.callback(Output("page-content", "children"), Input("url", "pathname"))
    def display_page(pathname):
        if pathname is None or pathname == "/":
            return html.Div(
                [
                    html.H1("Übersicht"),
                    dcc.Graph(figure=create_plot(df, sim)),
                    html.P("Wähle ein Modell links aus der Navigation."),
                ]
            )
        # Name aus URL extrahieren
        model_name = pathname.strip("/").replace("_", " ")
        if model_name in models_normal:
            return models_normal[model_name].get_html()
        if model_name in models_noisy:
            return models_noisy[model_name].get_html()
        return html.H1("Seite nicht gefunden")

    app.run_server(debug=True)


if __name__ == "__main__":
    main()
