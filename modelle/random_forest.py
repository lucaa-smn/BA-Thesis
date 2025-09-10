# modelle/random_forest.py

import dash
import dash.html as html
import dash.dcc as dcc
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import io
import base64
from modelle.section import Section


class RandomForest(Section):
    def __init__(
        self,
        app: dash.Dash,
        data: pd.DataFrame,
        target_column: str,
        task: str = "regression",
    ) -> None:
        """
        Parameters
        ----------
        app : dash.Dash
            Dash app instance.
        data : pd.DataFrame
            Input dataset with 'wurf_id' and target column.
        target_column : str
            Name of the column to predict.
        task : str
            Either 'regression' or 'classification'.
        """
        if task not in ["regression", "classification"]:
            raise ValueError("task must be either 'regression' or 'classification'")

        self.app = app
        self.data = data
        self.sequence_length = 50
        self.target_column = target_column
        self.task = task

        if self.task == "regression":
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)

        self.train_model()

        if self.task == "regression":
            self.div = html.Div(
                [
                    html.H2("Random Forest Regression"),
                    html.H4(f"MSE: {self.mse:.2f}"),
                    html.H4("True vs. Predicted Plot"),
                    html.Img(src=self.get_true_vs_predicted_plot()),
                ]
            )
        else:
            self.div = html.Div(
                [
                    html.H2("Random Forest Classification"),
                    html.H4(f"Accuracy: {self.accuracy:.2f}"),
                    html.H4("Confusion Matrix"),
                    html.Img(src=self.get_confusion_matrix()),
                ]
            )

    def get_html(self) -> html.Div:
        return self.div

    def prepare_sequences(self):
        if "wurf_id" not in self.data.columns:
            raise ValueError(
                "Data must contain 'wurf_id' column for sequence grouping."
            )
        if self.target_column not in self.data.columns:
            raise ValueError(f"Data must contain target column '{self.target_column}'.")

        grouped = self.data.groupby("wurf_id")
        sequences, targets = [], []
        for _, group in grouped:
            group_sorted = group.sort_values("x")
            sequence = group_sorted[["x", "y"]].values
            if sequence.shape[0] == self.sequence_length:
                sequences.append(sequence.flatten())  # Flatten for RF
                targets.append(group_sorted[self.target_column].iloc[0])
        return np.array(sequences), np.array(targets)

    def train_model(self):
        X, y = self.prepare_sequences()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)
        self.y_pred = self.model.predict(X_test)
        self.y_test = y_test

        if self.task == "regression":
            self.mse = mean_squared_error(y_test, self.y_pred)
        else:
            self.accuracy = accuracy_score(y_test, self.y_pred)

    def get_true_vs_predicted_plot(self):
        fig, ax = plt.subplots()
        ax.scatter(self.y_test, self.y_pred, alpha=0.6)
        ax.plot(
            [self.y_test.min(), self.y_test.max()],
            [self.y_test.min(), self.y_test.max()],
            "r--",
        )
        ax.set_xlabel("True Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("True vs. Predicted")

        bio = io.BytesIO()
        plt.savefig(bio, format="png")
        plt.close(fig)
        return f"data:image/png;base64,{base64.b64encode(bio.getvalue()).decode()}"

    def get_confusion_matrix(self):
        cm = confusion_matrix(self.y_test, self.y_pred)
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title("Confusion Matrix")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

        bio = io.BytesIO()
        plt.savefig(bio, format="png")
        plt.close(fig)
        return f"data:image/png;base64,{base64.b64encode(bio.getvalue()).decode()}"

    def register_callbacks(self):
        pass
