# modelle/svm.py

import dash
import pandas as pd
import dash.html as html
import dash.dcc as dcc
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.express as px
import matplotlib.pyplot as plt
import base64
import io
from modelle.section import Section


class SVM(Section):
    def __init__(self, app: dash.Dash, data: pd.DataFrame) -> None:
        self.app = app
        self.data = data
        self.sequence_length = 50
        self.results = []

        self.train_and_evaluate()

        self.div = html.Div(
            [
                html.H2("Klassifikation mit SVM"),
                html.H4("Parametervergleich (Accuracy)"),
                dcc.Graph(figure=self.plot_accuracies()),
                html.H4("Beste Konfusionsmatrix"),
                html.Img(src=self.get_confusion_matrix_image()),
            ]
        )

    def get_html(self) -> html.Div:
        return self.div

    def prepare_sequences(self):
        if "wurf_id" not in self.data.columns:
            raise ValueError(
                "Data must contain 'wurf_id' column for sequence grouping."
            )

        grouped = self.data.groupby("wurf_id")
        sequences, labels = [], []

        for _, group in grouped:
            group_sorted = group.sort_values("x")
            sequence = group_sorted[["x", "y"]].values
            if sequence.shape[0] == self.sequence_length:
                sequences.append(sequence.flatten())  # 2D -> 1D für SVM
                labels.append(group_sorted["label"].iloc[0])

        return np.array(sequences), np.array(labels)

    def train_and_evaluate(self):
        X, y = self.prepare_sequences()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        param_grid = [
            {"C": 1, "gamma": 0.01, "kernel": "rbf"},
            {"C": 10, "gamma": 0.01, "kernel": "rbf"},
            {"C": 1, "gamma": 0.001, "kernel": "rbf"},
            {"C": 10, "gamma": 0.001, "kernel": "rbf"},
            {"C": 1, "gamma": "scale", "kernel": "linear"},
        ]

        best_acc = 0
        self.best_cm = None

        for params in param_grid:
            model = svm.SVC(
                C=params["C"], gamma=params["gamma"], kernel=params["kernel"]
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            self.results.append(
                {
                    "C": params["C"],
                    "gamma": params["gamma"],
                    "kernel": params["kernel"],
                    "accuracy": acc,
                }
            )

            if acc > best_acc:
                best_acc = acc
                self.best_cm = confusion_matrix(y_test, y_pred)

    def plot_accuracies(self):
        df = pd.DataFrame(self.results)
        df["gamma"] = df["gamma"].astype(str)  # string für Legende
        df["label"] = df.apply(
            lambda row: f"C={row['C']}, γ={row['gamma']}, kernel={row['kernel']}",
            axis=1,
        )

        fig = px.bar(
            df,
            x="label",
            y="accuracy",
            color="kernel",
            title="SVM-Genauigkeit für verschiedene Parameter",
            labels={"accuracy": "Accuracy", "label": "Parameterkombination"},
        )
        fig.update_layout(xaxis_tickangle=-30)
        return fig

    def get_confusion_matrix_image(self):
        if self.best_cm is None:
            return None

        fig, ax = plt.subplots()
        im = ax.imshow(self.best_cm, cmap="Blues")
        ax.set_title("Beste Konfusionsmatrix")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(np.arange(3))
        ax.set_yticks(np.arange(3))
        ax.set_xticklabels(["Kurz", "Treffer", "Lang"])
        ax.set_yticklabels(["Kurz", "Treffer", "Lang"])
        plt.xlabel("Vorhergesagt")
        plt.ylabel("Tatsächlich")

        for i in range(3):
            for j in range(3):
                ax.text(
                    j, i, self.best_cm[i, j], ha="center", va="center", color="black"
                )

        bio = io.BytesIO()
        plt.savefig(bio, format="png", bbox_inches="tight")
        plt.close(fig)
        src = base64.b64encode(bio.getvalue()).decode()
        return f"data:image/png;base64,{src}"

    def register_callbacks(self):
        pass
