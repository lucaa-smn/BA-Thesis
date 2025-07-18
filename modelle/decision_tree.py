# modelle/decision_tree.py

import dash
import dash.html as html
import dash.dcc as dcc
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import io
import base64
from modelle.section import Section


class DecisionTree(Section):
    def __init__(self, app: dash.Dash, data: pd.DataFrame) -> None:
        self.app = app
        self.data = data
        self.sequence_length = 50

        self.model = DecisionTreeClassifier(random_state=42, max_depth=5)
        self.train_model()

        self.div = html.Div(
            [
                html.H2("Klassifikation mit Decision Tree"),
                html.H4(f"Genauigkeit: {self.accuracy:.2f}"),
                html.H4("Konfusionsmatrix"),
                html.Img(src=self.get_confusion_matrix()),
                html.H4("Baumstruktur"),
                html.Img(src=self.get_tree_plot()),
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
                sequences.append(sequence.flatten())  # ⬅️ Wichtig für Decision Tree!
                labels.append(group_sorted["label"].iloc[0])
        return np.array(sequences), np.array(labels)

    def train_model(self):
        X, y = self.prepare_sequences()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)
        self.y_pred = self.model.predict(X_test)
        self.y_test = y_test
        self.accuracy = accuracy_score(y_test, self.y_pred)

    def get_confusion_matrix(self):
        cm = confusion_matrix(self.y_test, self.y_pred)
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title("Confusion Matrix")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(np.arange(3))
        ax.set_yticks(np.arange(3))
        ax.set_xticklabels(["Kurz", "Treffer", "Lang"])
        ax.set_yticklabels(["Kurz", "Treffer", "Lang"])
        plt.xlabel("Predicted")
        plt.ylabel("True")

        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

        bio = io.BytesIO()
        plt.savefig(bio, format="png")
        plt.close(fig)
        return f"data:image/png;base64,{base64.b64encode(bio.getvalue()).decode()}"

    def get_tree_plot(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_tree(self.model, filled=True, max_depth=2, fontsize=8)
        bio = io.BytesIO()
        plt.savefig(bio, format="png")
        plt.close(fig)
        return f"data:image/png;base64,{base64.b64encode(bio.getvalue()).decode()}"

    def register_callbacks(self):
        pass
