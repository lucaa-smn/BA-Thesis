# main.py
import dash
from dash import html, dcc
import pandas as pd
import plotly.express as px
from data_gen import DataGen
from modelle.knn import TimeSeriesNN
from modelle.svm import SVM
from modelle.decision_tree import DecisionTree
from modelle import section


def main():
    # Simuliere Daten
    sim = DataGen(hs=2.13)
    df = sim.generate_dataset(n=500)
    df_noisy = sim.generate_dataset(n=500, with_noise=True)
    df.to_csv("wurfbahnen_labeled.csv", index=False)
    df_noisy.to_csv("wurfbahnen_labeled_noisy.csv", index=False)

    # Erstelle Dash-App
    app = dash.Dash(__name__)

    df2 = df.copy()

    df2["trajectory_id"] = df2.groupby(["theta", "v0"]).ngroup()

    # Wähle die ersten 3 Verläufe
    erste_drei_ids = df2["trajectory_id"].unique()[:3]
    df_drei = df2[df2["trajectory_id"].isin(erste_drei_ids)]

    # Linienplot
    fig = px.line(
        df_drei,
        x="x",
        y="y",
        color="trajectory_id",
        markers=True,
        title="Erste drei Wurfverläufe (verbundene Zeitreihen)",
        labels={"x": "x [m]", "y": "y [m]", "trajectory_id": "Verlauf-ID"},
    )
    # # Statistik
    # winkel_df = df.copy()
    # winkel_df["theta_deg"] = df["theta"] * 180 / 3.14159

    # winkel_bereiche = (
    #     winkel_df.groupby("label")["theta_deg"].agg(["min", "max"]).round(2)
    # )
    # geschw_bereiche = df.groupby("label")["v0"].agg(["min", "max"]).round(2)

    # bereich_text = "".join(
    #     f"{label}: Winkel {winkel_bereiche.loc[label, 'min']}°–{winkel_bereiche.loc[label, 'max']}°, "
    #     f"v0 {geschw_bereiche.loc[label, 'min']}-{geschw_bereiche.loc[label, 'max']} m/s<br>"
    #     for label in df["label"].unique()
    # )

    # # Plot erstellen
    # df["line_group"] = df.groupby(["theta", "v0"]).ngroup()
    # fig = px.line(
    #     df,
    #     x="x",
    #     y="y",
    #     color="label",
    #     line_group="line_group",
    #     title="Simulierte Wurfparabeln nach Kategorie",
    #     labels={"x": "x [m]", "y": "y [m]"},
    # )

    # # Korb hinzufügen
    # korb_x0 = sim.params["l"] - sim.params["Rr"]
    # korb_x1 = sim.params["l"] + sim.params["Rr"]
    # korb_y = sim.params["hKorb"]
    # fig.add_shape(
    #     type="line",
    #     x0=korb_x0,
    #     x1=korb_x1,
    #     y0=korb_y,
    #     y1=korb_y,
    #     line=dict(color="black", width=6),
    #     name="Korb",
    # )

    # # Annotation
    # fig.add_annotation(
    #     x=0.01,
    #     y=0.99,
    #     xref="paper",
    #     yref="paper",
    #     text=f"<b>Winkel- und Geschwindigkeitsbereiche:</b><br>{bereich_text}",
    #     showarrow=False,
    #     align="left",
    #     bordercolor="black",
    #     borderwidth=1,
    #     bgcolor="white",
    #     opacity=0.8,
    # )

    # fig.update_layout(legend_title_text="Kategorie")

    # Modell-Sektionen
    sections: list[section.Section] = [
        TimeSeriesNN(app=app, data=df),
        SVM(app=app, data=df),
        DecisionTree(app=app, data=df),
        TimeSeriesNN(app=app, data=df_noisy),
        SVM(app=app, data=df_noisy),
        DecisionTree(app=app, data=df_noisy),
    ]

    # Layout der App
    app.layout = html.Div(
        [
            # html.H1("Wurf-Simulation & Klassifikation"),
            # dcc.Graph(figure=fig),
            dcc.Graph(figure=fig),
            *[s.get_html() for s in sections],
        ]
    )

    # Server starten
    app.run_server(debug=True)


if __name__ == "__main__":
    main()
