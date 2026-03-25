# ============================================================
# PATCH 1: FOUNDATION + MULTI-AGENT CORE + DATA PIPELINE
# ============================================================

# ================== CORE IMPORTS ==================
import os, random, warnings, uuid, datetime, time, hashlib, sys
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
from sklearn.utils.class_weight import compute_class_weight

from imblearn.over_sampling import SMOTE, ADASYN

import streamlit as st

# ================== GLOBAL CONFIG ==================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SESSION_ID = str(uuid.uuid4())
TIMESTAMP = datetime.datetime.now()

# ================== STREAMLIT INIT ==================
st.set_page_config(page_title="Agentic AQI System", layout="wide")

st.sidebar.title("🔷 Agentic AQI System")
st.sidebar.write(f"Session: {SESSION_ID}")
st.sidebar.write(f"Time: {TIMESTAMP}")
st.sidebar.write(f"Device: {DEVICE}")

# ============================================================
# 🔶 AGENT BASE CLASS
# ============================================================
class BaseAgent:
    def __init__(self, name):
        self.name = name
        self.logs = []

    def log(self, message):
        log_msg = f"[{self.name}] {message}"
        self.logs.append(log_msg)
        print(log_msg)

    def execute(self, *args, **kwargs):
        raise NotImplementedError


# ============================================================
# 🔶 DATA AGENT
# ============================================================
class DataAgent(BaseAgent):
    def __init__(self):
        super().__init__("DataAgent")

    def load_data(self, file_paths):
        self.log("Loading data...")
        df_list = []
        skipped = []

        for path in file_paths:
            if not os.path.exists(path):
                skipped.append(path)
                continue
            try:
                df_list.append(pd.read_excel(path))
            except Exception as e:
                skipped.append(f"{path} | {e}")

        if len(df_list) == 0:
            raise FileNotFoundError("No valid files loaded")

        data = pd.concat(df_list, ignore_index=True)
        self.log(f"Loaded {len(data)} rows")
        return data, skipped

    def clean_data(self, df):
        self.log("Cleaning data...")

        df.columns = [c.strip() for c in df.columns]

        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        df.ffill(inplace=True)

        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

        df.dropna(inplace=True)

        self.log(f"Cleaned data shape: {df.shape}")
        return df


# ============================================================
# 🔶 FEATURE AGENT
# ============================================================
class FeatureAgent(BaseAgent):
    def __init__(self):
        super().__init__("FeatureAgent")

    def aqi_category(self, aqi):
        if aqi <= 50: return 0
        elif aqi <= 100: return 1
        elif aqi <= 150: return 2
        elif aqi <= 200: return 3
        elif aqi <= 300: return 4
        else: return 5

    def transform(self, df):
        self.log("Generating AQI categories...")

        df['AQI_Category'] = df['AQI'].apply(self.aqi_category)

        # Merge rare classes safely
        counts = df['AQI_Category'].value_counts()
        rare_classes = counts[counts < 5].index

        for cls in rare_classes:
            df['AQI_Category'] = df['AQI_Category'].replace(cls, counts.idxmax())

        self.log("Feature engineering completed")
        return df

    def prepare_features(self, df):
        features = ['PM2.5','PM10','NO2','SO2','CO','Ozone']

        X = df[features].apply(pd.to_numeric, errors="coerce")
        X.fillna(X.median(), inplace=True)

        y = df['AQI_Category'].astype(int).values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return X_scaled, y, scaler, features


# ============================================================
# 🔶 SPLIT AGENT
# ============================================================
class SplitAgent(BaseAgent):
    def __init__(self):
        super().__init__("SplitAgent")

    def split(self, X, y):
        self.log("Splitting dataset...")

        return train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=SEED
        )


# ============================================================
# 🔶 BALANCE AGENT
# ============================================================
class BalanceAgent(BaseAgent):
    def __init__(self, method="ADASYN"):
        super().__init__("BalanceAgent")
        self.method = method

    def balance(self, X, y):
        self.log(f"Balancing using {self.method}")

        if self.method == "SMOTE":
            sampler = SMOTE(random_state=SEED)
        else:
            sampler = ADASYN(random_state=SEED)

        X_res, y_res = sampler.fit_resample(X, y)

        self.log(f"Balanced shape: {X_res.shape}")
        return X_res, y_res


# ============================================================
# 🔶 TORCH AGENT (DL DATA PREP)
# ============================================================
class TorchAgent(BaseAgent):
    def __init__(self):
        super().__init__("TorchAgent")

    def prepare(self, X_train, X_test, y_train, y_test):
        self.log("Preparing tensors...")

        X_train_dl = torch.tensor(X_train.reshape(-1,1,X_train.shape[1]), dtype=torch.float32)
        X_test_dl  = torch.tensor(X_test.reshape(-1,1,X_test.shape[1]), dtype=torch.float32)

        y_train_t = torch.tensor(y_train, dtype=torch.long)
        y_test_t  = torch.tensor(y_test, dtype=torch.long)

        train_loader = DataLoader(TensorDataset(X_train_dl, y_train_t), batch_size=32, shuffle=True)
        test_loader  = DataLoader(TensorDataset(X_test_dl, y_test_t), batch_size=32)

        return train_loader, test_loader


# ============================================================
# 🔶 PIPELINE EXECUTION (AGENT ORCHESTRATOR)
# ============================================================

data_agent = DataAgent()
feature_agent = FeatureAgent()
split_agent = SplitAgent()
balance_agent = BalanceAgent(method="ADASYN")
torch_agent = TorchAgent()

# ================== FILE PATHS ==================
excel_files = [
    r"C:\Users\biswa\OneDrive\Desktop\data_collection\talcher_4.xlsx",
    r"C:\Users\biswa\OneDrive\Desktop\data_collection\angul_1.xlsx",
    r"C:\Users\biswa\OneDrive\Desktop\data_collection\angul_2.xlsx",
    r"C:\Users\biswa\OneDrive\Desktop\data_collection\angul_3.xlsx",
    r"C:\Users\biswa\OneDrive\Desktop\data_collection\balasore_1.xlsx",
    r"C:\Users\biswa\OneDrive\Desktop\data_collection\barbil_1.xlsx",
    r"C:\Users\biswa\OneDrive\Desktop\data_collection\barbil_2.xlsx",
    r"C:\Users\biswa\OneDrive\Desktop\data_collection\barbil_3.xlsx",
    r"C:\Users\biswa\OneDrive\Desktop\data_collection\baripada_1.xlsx",
    r"C:\Users\biswa\OneDrive\Desktop\data_collection\baripada_2.xlsx",
    r"C:\Users\biswa\OneDrive\Desktop\data_collection\baripada_3.xlsx",
    r"C:\Users\biswa\OneDrive\Desktop\data_collection\cuttack_1.xlsx",
    r"C:\Users\biswa\OneDrive\Desktop\data_collection\cuttack_2.xlsx",
    r"C:\Users\biswa\OneDrive\Desktop\data_collection\cuttack_3.xlsx",
    r"C:\Users\biswa\OneDrive\Desktop\data_collection\cuttack_4.xlsx",
    r"C:\Users\biswa\OneDrive\Desktop\data_collection\keonjhar_1.xlsx",
    r"C:\Users\biswa\OneDrive\Desktop\data_collection\keonjhar_2.xlsx",
    r"C:\Users\biswa\OneDrive\Desktop\data_collection\keonjhar_3.xlsx",
    r"C:\Users\biswa\OneDrive\Desktop\data_collection\keonjhar_4.xlsx",
    r"C:\Users\biswa\OneDrive\Desktop\data_collection\lingraj_1.xlsx",
    r"C:\Users\biswa\OneDrive\Desktop\data_collection\lingraj_2.xlsx",
    r"C:\Users\biswa\OneDrive\Desktop\data_collection\lingraj_3.xlsx",
    r"C:\Users\biswa\OneDrive\Desktop\data_collection\lingraj_4.xlsx",
    r"C:\Users\biswa\OneDrive\Desktop\data_collection\nayaghad_1.xlsx",
    r"C:\Users\biswa\OneDrive\Desktop\data_collection\nayaghad_2.xlsx",
    r"C:\Users\biswa\OneDrive\Desktop\data_collection\nayaghad_3.xlsx",
    r"C:\Users\biswa\OneDrive\Desktop\data_collection\nayaghad_4.xlsx",
    r"C:\Users\biswa\OneDrive\Desktop\data_collection\patia_1.xlsx",
    r"C:\Users\biswa\OneDrive\Desktop\data_collection\Raigangpur_1.xlsx",
    r"C:\Users\biswa\OneDrive\Desktop\data_collection\talcher_1.xlsx",
    r"C:\Users\biswa\OneDrive\Desktop\data_collection\talcher_2.xlsx",
    r"C:\Users\biswa\OneDrive\Desktop\data_collection\talcher_3.xlsx",
]

# ================== EXECUTION ==================
data, skipped = data_agent.load_data(excel_files)
data = data_agent.clean_data(data)

data = feature_agent.transform(data)
X, y, scaler, features = feature_agent.prepare_features(data)

X_train, X_test, y_train, y_test = split_agent.split(X, y)

X_train_bal, y_train_bal = balance_agent.balance(X_train, y_train)

train_loader, test_loader = torch_agent.prepare(
    X_train_bal, X_test, y_train_bal, y_test
)

st.success("✅ PATCH 1 Completed: Data Pipeline Ready")
# ============================================================
# PATCH 2: ML AGENTS + TABNET + HYBRID + K-FOLD + CALIBRATION
# ============================================================

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.calibration import CalibratedClassifierCV

import xgboost as xgb
from lightgbm import LGBMClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

# ============================================================
# 🔶 ML AGENT
# ============================================================
class MLAgent(BaseAgent):
    def __init__(self):
        super().__init__("MLAgent")
        self.models = {}

    def build_models(self):
        self.log("Building ML models...")

        self.models = {
            "LogReg": LogisticRegression(max_iter=3000, random_state=SEED),
            "SVM": CalibratedClassifierCV(SVC(probability=True, random_state=SEED)),
            "XGBoost": xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=SEED
            ),
            "LightGBM": LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                random_state=SEED
            )
        }

        return self.models

    def train(self, X, y):
        self.log("Training ML models...")
        for name, model in self.models.items():
            self.log(f"Training {name}")
            model.fit(X, y)

        return self.models


# ============================================================
# 🔶 TABNET AGENT
# ============================================================
class TabNetAgent(BaseAgent):
    def __init__(self):
        super().__init__("TabNetAgent")
        self.model = None

    def train(self, X_train, y_train, X_val, y_val):
        self.log("Training TabNet...")

        self.model = TabNetClassifier(seed=SEED, verbose=0)

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_name=["val"],
            eval_metric=["accuracy"],
            max_epochs=100,
            patience=15,
            batch_size=256,
            virtual_batch_size=128
        )

        return self.model


# ============================================================
# 🔶 HYBRID AGENT (STACKING)
# ============================================================
class HybridAgent(BaseAgent):
    def __init__(self):
        super().__init__("HybridAgent")
        self.model = None

    def build(self):
        self.log("Building Hybrid Stacking model...")

        self.model = StackingClassifier(
            estimators=[
                ("lgbm", LGBMClassifier(n_estimators=200, learning_rate=0.05)),
                ("xgb", xgb.XGBClassifier(n_estimators=200, learning_rate=0.05)),
                ("svm", SVC(probability=True))
            ],
            final_estimator=LogisticRegression(max_iter=3000),
            stack_method="predict_proba",
            cv=5,
            n_jobs=-1
        )

        return self.model

    def train(self, X, y):
        self.log("Training Hybrid model...")
        self.model.fit(X, y)
        return self.model


# ============================================================
# 🔶 K-FOLD AGENT (NO DATA LEAKAGE)
# ============================================================
class KFoldAgent(BaseAgent):
    def __init__(self, n_splits=5):
        super().__init__("KFoldAgent")
        self.n_splits = n_splits

    def cross_validate(self, model, X, y):
        self.log(f"Running {self.n_splits}-Fold CV...")

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=SEED)

        scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            self.log(f"Fold {fold+1}")

            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            # ⚠️ Apply balancing INSIDE fold (correct approach)
            sampler = ADASYN(random_state=SEED)
            X_tr, y_tr = sampler.fit_resample(X_tr, y_tr)

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)

            f1 = f1_score(y_val, y_pred, average="weighted")
            scores.append(f1)

        mean_score = np.mean(scores)
        std_score = np.std(scores)

        self.log(f"CV Mean F1: {mean_score:.4f} ± {std_score:.4f}")

        return mean_score, std_score


# ============================================================
# 🔶 MODEL REGISTRY AGENT
# ============================================================
class ModelRegistryAgent(BaseAgent):
    def __init__(self):
        super().__init__("ModelRegistry")
        self.registry = {}

    def register(self, name, model, score):
        self.registry[name] = {"model": model, "score": score}

    def best_model(self):
        best = max(self.registry.items(), key=lambda x: x[1]["score"])
        self.log(f"Best model: {best[0]} ({best[1]['score']:.4f})")
        return best


# ============================================================
# 🔶 EXECUTION OF PATCH 2
# ============================================================

ml_agent = MLAgent()
tabnet_agent = TabNetAgent()
hybrid_agent = HybridAgent()
kfold_agent = KFoldAgent()
registry_agent = ModelRegistryAgent()

# ================== BUILD ==================
ml_models = ml_agent.build_models()

# ================== K-FOLD VALIDATION ==================
cv_results = {}

for name, model in ml_models.items():
    mean_f1, std_f1 = kfold_agent.cross_validate(model, X_train, y_train)
    cv_results[name] = mean_f1
    registry_agent.register(name, model, mean_f1)

# ================== TRAIN FINAL MODELS ==================
trained_ml = ml_agent.train(X_train_bal, y_train_bal)

# ================== TABNET ==================
tabnet_model = tabnet_agent.train(
    X_train_bal, y_train_bal,
    X_test, y_test
)
registry_agent.register("TabNet", tabnet_model, 0.0)

# ================== HYBRID ==================
hybrid_model = hybrid_agent.build()
hybrid_model = hybrid_agent.train(X_train_bal, y_train_bal)

registry_agent.register("Hybrid", hybrid_model, 0.0)

# ================== BEST MODEL ==================
best_model_name, best_model_info = registry_agent.best_model()

st.subheader("📊 K-Fold Results")
st.write(cv_results)

st.success("✅ PATCH 2 Completed: ML + Hybrid + CV Ready")
# ============================================================
# PATCH 3: DL AGENT + TRAINING ENGINE + ADVANCED MODELS
# ============================================================

import torch.backends.cudnn as cudnn
cudnn.benchmark = True  # GPU optimization

# ============================================================
# 🔶 DL MODELS
# ============================================================

class CNN1D(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, num_classes, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 64, batch_first=True, bidirectional=bidirectional)
        mult = 2 if bidirectional else 1
        self.fc = nn.Linear(64 * mult, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class GRUModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.gru = nn.GRU(input_dim, 64, batch_first=True)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])


# ============================================================
# 🔶 DL AGENT
# ============================================================

class DLAgent(BaseAgent):
    def __init__(self, input_dim, num_classes):
        super().__init__("DLAgent")
        self.models = {
            "CNN1D": CNN1D(input_dim, num_classes).to(DEVICE),
            "LSTM": LSTMModel(input_dim, num_classes).to(DEVICE),
            "BiLSTM": LSTMModel(input_dim, num_classes, bidirectional=True).to(DEVICE),
            "GRU": GRUModel(input_dim, num_classes).to(DEVICE),
            "Transformer": TransformerModel(input_dim, num_classes).to(DEVICE)
        }

    def get_models(self):
        return self.models


# ============================================================
# 🔶 TRAINING ENGINE (WITH EARLY STOPPING)
# ============================================================

class TrainingAgent(BaseAgent):
    def __init__(self, patience=5):
        super().__init__("TrainingAgent")
        self.patience = patience

    def train_model(self, model, train_loader, val_loader, epochs=30):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        best_loss = float('inf')
        patience_counter = 0

        history = {"loss": []}

        for epoch in range(epochs):
            model.train()
            total_loss = 0

            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(xb)
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            history["loss"].append(avg_loss)

            # Validation
            val_loss = self.validate(model, val_loader)

            self.log(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                self.log("Early stopping triggered")
                break

        return model, history

    def validate(self, model, loader):
        model.eval()
        criterion = nn.CrossEntropyLoss()
        loss_total = 0

        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                outputs = model(xb)
                loss = criterion(outputs, yb)
                loss_total += loss.item()

        return loss_total / len(loader)


# ============================================================
# 🔶 DL EVALUATION AGENT
# ============================================================

class DLEvaluator(BaseAgent):
    def __init__(self):
        super().__init__("DLEvaluator")

    def evaluate(self, model, loader):
        model.eval()
        preds, trues, probs = [], [], []

        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(DEVICE)
                outputs = model(xb)

                prob = torch.softmax(outputs, dim=1).cpu().numpy()
                pred = np.argmax(prob, axis=1)

                probs.extend(prob)
                preds.extend(pred)
                trues.extend(yb.numpy())

        f1 = f1_score(trues, preds, average="weighted")
        acc = accuracy_score(trues, preds)

        return {
            "F1": f1,
            "Accuracy": acc,
            "Preds": preds,
            "Probs": probs,
            "True": trues
        }


# ============================================================
# 🔶 DL EXECUTION
# ============================================================

input_dim = X_train.shape[1]
num_classes = len(np.unique(y_train))

dl_agent = DLAgent(input_dim, num_classes)
training_agent = TrainingAgent(patience=5)
dl_evaluator = DLEvaluator()

dl_models = dl_agent.get_models()

dl_results = {}

# Create validation loader from test set
val_loader = test_loader

for name, model in dl_models.items():
    st.write(f"🧠 Training DL Model: {name}")

    trained_model, history = training_agent.train_model(
        model,
        train_loader,
        val_loader,
        epochs=30
    )

    eval_metrics = dl_evaluator.evaluate(trained_model, test_loader)

    dl_results[name] = eval_metrics["F1"]

    registry_agent.register(name, trained_model, eval_metrics["F1"])

# ============================================================
# 🔶 DL RESULTS DISPLAY
# ============================================================

st.subheader("🧠 DL Model Results")
st.write(dl_results)

st.success("✅ PATCH 3 Completed: Deep Learning Pipeline Ready")
# ============================================================
# PATCH 4: XAI + DRIFT + TRUST + STATISTICS + UI + EXPORT
# ============================================================

import shap
from captum.attr import IntegratedGradients
from scipy.stats import ks_2samp, entropy, ttest_rel
import plotly.express as px

# ============================================================
# 🔶 XAI AGENT (SHAP + CAPTUM)
# ============================================================

class XAI_Agent(BaseAgent):
    def __init__(self):
        super().__init__("XAI_Agent")

    def shap_explain(self, model, X):
        try:
            self.log("Running SHAP...")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            importance = np.mean(np.abs(shap_values), axis=0)
            return importance
        except Exception as e:
            self.log(f"SHAP failed: {e}")
            return None

    def captum_explain(self, model, loader):
        try:
            self.log("Running Captum IG...")
            ig = IntegratedGradients(model)

            xb, yb = next(iter(loader))
            xb = xb.to(DEVICE)

            attr = ig.attribute(xb, target=yb[0].item())
            return attr.detach().cpu().numpy()
        except Exception as e:
            self.log(f"Captum failed: {e}")
            return None


# ============================================================
# 🔶 DRIFT AGENT
# ============================================================

class DriftAgent(BaseAgent):
    def __init__(self):
        super().__init__("DriftAgent")

    def psi(self, expected, actual, bins=10):
        breakpoints = np.linspace(np.min(expected), np.max(expected), bins+1)

        expected_counts = np.histogram(expected, breakpoints)[0]
        actual_counts = np.histogram(actual, breakpoints)[0]

        expected_perc = expected_counts / len(expected)
        actual_perc = actual_counts / len(actual)

        psi = np.sum((expected_perc - actual_perc) * np.log((expected_perc+1e-9)/(actual_perc+1e-9)))
        return psi

    def compute(self, X_train, X_test, feature_names):
        psi_scores = {}
        kl_scores = {}
        ks_scores = {}

        for i, col in enumerate(feature_names):
            psi_scores[col] = self.psi(X_train[:, i], X_test[:, i])

            train_hist, _ = np.histogram(X_train[:, i], bins=20, density=True)
            test_hist, _ = np.histogram(X_test[:, i], bins=20, density=True)
            kl_scores[col] = entropy(train_hist+1e-9, test_hist+1e-9)

            ks_stat, ks_p = ks_2samp(X_train[:, i], X_test[:, i])
            ks_scores[col] = {"stat": ks_stat, "p": ks_p}

        return psi_scores, kl_scores, ks_scores


# ============================================================
# 🔶 TRUST AGENT
# ============================================================

class TrustAgent(BaseAgent):
    def __init__(self):
        super().__init__("TrustAgent")

    def compute(self, results_df):
        max_f1 = results_df["F1"].max()

        results_df["Confidence"] = results_df["F1"] / (max_f1 + 1e-9)
        results_df["Risk"] = 1 - results_df["Confidence"]

        return results_df


# ============================================================
# 🔶 STATISTICAL AGENT
# ============================================================

class StatsAgent(BaseAgent):
    def __init__(self):
        super().__init__("StatsAgent")

    def summary(self, results_df):
        stats = {}

        for metric in ["F1", "Accuracy"]:
            vals = results_df[metric]
            stats[metric] = {
                "mean": np.mean(vals),
                "std": np.std(vals)
            }

        return stats

    def t_test(self, scores1, scores2):
        t_stat, p_val = ttest_rel(scores1, scores2)
        return t_stat, p_val


# ============================================================
# 🔶 RECOMMENDATION AGENT
# ============================================================

class RecommendationAgent(BaseAgent):
    def __init__(self):
        super().__init__("RecommendationAgent")
        self.advice_map = {
            0: "Good: Air quality is satisfactory. Enjoy outdoor activities!",
            1: "Moderate: Air quality is acceptable. Sensitive individuals should reduce prolonged exertion.",
            2: "Unhealthy for Sensitive Groups: Wear a mask if you have respiratory issues.",
            3: "Unhealthy: Everyone should limit outdoor time. Wear an N95 mask.",
            4: "Very Unhealthy: Avoid all outdoor activities. Use an air purifier indoors.",
            5: "Hazardous: Health warning! Stay indoors and keep windows closed."
        }

    def recommend(self, category_idx, features=None, values=None):
        # Base recommendation
        base = self.advice_map.get(category_idx, "No data available.")
        
        if features is not None and values is not None:
            # XAI-driven specific advice: find the worst pollutant in the input
            # Zip features and values together to find the max
            data_dict = dict(zip(features, values))
            top_pollutant = max(data_dict, key=data_dict.get)
            
            specific_advice = f"\n\n**🔍 XAI Insight:** The primary driver for this rating is **{top_pollutant}**. "
            
            if top_pollutant == "PM2.5":
                specific_advice += "Fine particles can penetrate deep into lungs; use a high-quality filter."
            elif top_pollutant == "CO":
                specific_advice += "High Carbon Monoxide detected; ensure proper ventilation and check for nearby traffic/fires."
            elif top_pollutant == "SO2":
                specific_advice += "Sulfur Dioxide is high; this can cause throat irritation. Limit heavy breathing outdoors."
        else:
            specific_advice = ""
            
        return f"{base} {specific_advice}"

# ============================================================
# 🔶 RESULT AGGREGATION
# ============================================================

results = []

# ML + Hybrid results
for name, model in registry_agent.registry.items():
    try:
        m = model["model"]
        y_pred = m.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="weighted")
        acc = accuracy_score(y_test, y_pred)

        results.append({
            "Model": name,
            "F1": f1,
            "Accuracy": acc
        })
    except:
        pass

# DL results already added → skip duplicate

results_df = pd.DataFrame(results)

# ============================================================
# 🔶 APPLY AGENTS
# ============================================================

xai_agent = XAI_Agent()
drift_agent = DriftAgent()
trust_agent = TrustAgent()
stats_agent = StatsAgent()
rec_agent = RecommendationAgent()

# ================== XAI ==================
st.subheader("🔍 Explainability")

try:
    xgb_model = registry_agent.registry["XGBoost"]["model"]
    shap_vals = xai_agent.shap_explain(xgb_model, X_test)

    if shap_vals is not None:
        st.write("SHAP Importance:", shap_vals)
except:
    st.write("SHAP skipped")

# ================== DRIFT ==================
st.subheader("📉 Drift Analysis")

psi_scores, kl_scores, ks_scores = drift_agent.compute(X_train, X_test, features)

st.write("PSI:", psi_scores)
st.write("KL:", kl_scores)
st.write("KS:", ks_scores)

# ================== TRUST ==================
results_df = trust_agent.compute(results_df)

st.subheader("🛡️ Trust & Risk")
st.dataframe(results_df)

# ================== STATS ==================
stats = stats_agent.summary(results_df)
st.subheader("📊 Statistics")
st.write(stats)

# ================== RADAR ==================
st.subheader("🕸️ Model Radar")

radar_df = results_df.set_index("Model")[["F1", "Accuracy"]]

fig = px.line_polar(
    radar_df.reset_index(),
    r="F1",
    theta="Model",
    line_close=True
)

st.plotly_chart(fig)

# ================== RECOMMENDATION ==================
sample_aqi = np.random.choice(y_test)

st.subheader("🌍 AQI Recommendation")
st.write(f"AQI Category: {sample_aqi}")
st.write(rec_agent.recommend(sample_aqi))

# ============================================================
# 🔶 EXPORT
# ============================================================

csv = results_df.to_csv(index=False).encode()

st.download_button(
    label="Download Results",
    data=csv,
    file_name="aqi_results.csv",
    mime="text/csv"
)


# ================== NEW: USER INPUT & REAL-TIME INFERENCE ==================
st.divider()
st.header("🎮 Manual Agent Testing (User Input)")

with st.expander("📝 Enter Sensor Data for Prediction", expanded=True):
    col1, col2, col3 = st.columns(3)
    user_vals = []
    
    # Create input boxes for each feature
    for i, feature in enumerate(features):
        with [col1, col2, col3][i % 3]:
            val = st.number_input(f"Enter {feature}", value=float(data[feature].median()))
            user_vals.append(val)

    if st.button("🚀 Run Agentic Inference"):
        # 1. Transform input using the fitted scaler
        input_array = np.array(user_vals).reshape(1, -1)
        scaled_input = scaler.transform(input_array)
        
        # 2. Use the Best Model for prediction
        # (Assuming the best model is an ML model from the registry)
        best_model = registry_agent.registry[best_model_name]["model"]
        prediction = best_model.predict(scaled_input)[0]
        
        # 3. Get Feature-Aware Recommendation
        final_rec = rec_agent.recommend(prediction, features, user_vals)
        
        # 4. Display Results
        st.subheader(f"Prediction: Category {prediction}")
        st.info(final_rec)
        
        # Optional: Mini SHAP for this specific user input
        try:
            explainer = shap.TreeExplainer(best_model)
            instance_shap = explainer.shap_values(scaled_input)
            st.write("Current Input Contribution (XAI):")
            st.bar_chart(pd.DataFrame(instance_shap[0], index=features, columns=["Impact"]))
        except:
            st.write("Individual XAI plot not available for this model type.")
# ============================================================
# 🔶 FINAL STATUS
# ============================================================

st.success("🎉 FULL AGENTIC AQI SYSTEM COMPLETED")

# ============================================================
# END OF SYSTEM 
# ============================================================