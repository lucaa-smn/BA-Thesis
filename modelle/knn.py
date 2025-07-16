# modelle/time_series_nn.py

import dash
import pandas as pd
import dash.html as html
import dash.dcc as dcc
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import io
import base64
import uuid
from modelle.section import Section


class WurfDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class LSTMClassifier(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=1, num_classes=3):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


class TimeSeriesNN(Section):
    def __init__(self, app: dash.Dash, data: pd.DataFrame) -> None:
        self.app = app
        self.data = data
        self.sequence_length = 50

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMClassifier().to(self.device)

        self.train_model()

        self.div = html.Div(
            [
                html.H2("Zeitreihenanalyse mit LSTM"),
                html.H4("Trainingsmetriken"),
                dcc.Graph(figure=self.plot_metrics()),
                html.H4("Konfusionsmatrix"),
                html.Img(src=self.get_confusion_matrix()),
            ]
        )

    def get_html(self) -> html.Div:
        return self.div

    def train_model(self):
        sequences, labels = self.prepare_sequences()
        X_train, X_test, y_train, y_test = train_test_split(
            sequences, labels, test_size=0.2, random_state=42
        )

        train_loader = DataLoader(
            WurfDataset(X_train, y_train), batch_size=32, shuffle=True
        )
        test_loader = DataLoader(
            WurfDataset(X_test, y_test), batch_size=32, shuffle=False
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.train_loss = []
        self.train_acc = []

        for epoch in range(10):
            self.model.train()
            epoch_loss, correct, total = 0, 0, 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

            self.train_loss.append(epoch_loss / len(train_loader))
            self.train_acc.append(correct / total)

        self.X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        self.y_test = y_test
        self.y_pred = self.predict(self.X_test)

    def prepare_sequences(self):
        grouped = self.data.groupby(["theta", "v0", "label"])
        sequences, labels = [], []
        for _, group in grouped:
            group_sorted = group.sort_values("x")
            sequence = group_sorted[["x", "y"]].values
            if sequence.shape[0] == self.sequence_length:
                sequences.append(sequence)
                labels.append(group_sorted["label"].iloc[0])
        return np.array(sequences), np.array(labels)

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            _, preds = torch.max(outputs, 1)
        return preds.cpu().numpy()

    def plot_metrics(self):
        df = pd.DataFrame(
            {
                "Epoch": list(range(1, len(self.train_loss) + 1)),
                "Loss": self.train_loss,
                "Accuracy": self.train_acc,
            }
        )
        fig = px.line(df, x="Epoch", y=["Loss", "Accuracy"], markers=True)
        fig.update_layout(title="Training Metrics", yaxis_title="Value")
        return fig

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
        src = base64.b64encode(bio.getvalue()).decode()
        return f"data:image/png;base64,{src}"

    def register_callbacks(self):
        pass
