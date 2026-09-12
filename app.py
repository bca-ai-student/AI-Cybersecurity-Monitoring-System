import os
import re
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# =========================================================
# FLASK APPLICATION
# =========================================================

app = Flask(__name__)
app.secret_key = "cyber-ai-secret-key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATABASE = os.path.join(
    BASE_DIR,
    "cybersecurity.db"
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Maximum CSV size = 50 MB
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


# =========================================================
# FEATURE ALIASES
# =========================================================

ALIASES = {

    "packet_count": [
        "packet_count",
        "total_fwd_packets",
        "tot_fwd_pkts",
        "total_backward_packets",
        "tot_bwd_pkts",
        "packets",
        "total_packets",
        "packetcount",
        "flow_packets"
    ],

    "packet_size": [
        "packet_size",
        "average_packet_size",
        "avg_packet_size",
        "avg_fwd_segment_size",
        "fwd_packet_length_mean",
        "bwd_packet_length_mean",
        "packet_length_mean",
        "flow_bytes_s",
        "flow_bytes_sec",
        "total_length_of_fwd_packets",
        "total_length_of_bwd_packets"
    ],

    "connection_duration": [
        "connection_duration",
        "duration",
        "flow_duration",
        "connection_time",
        "flow_duration_us",
        "flow_duration_ms"
    ],

    "failed_connections": [
        "failed_connections",
        "failed_login",
        "failed_logins",
        "login_attempts",
        "failed_attempts",
        "num_failed_logins",
        "fwd_psh_flags",
        "rst_flag_count",
        "syn_flag_count"
    ]
}


# =========================================================
# LABEL ALIASES
# =========================================================

LABEL_ALIASES = [
    "label",
    "attack",
    "attack_type",
    "attack_category",
    "class",
    "target",
    "category",
    "attack_cat"
]


# =========================================================
# CURRENT AI MODELS
# =========================================================

current_models = {
    "anomaly": None,
    "attack": None,
    "risk": None,
    "attack_accuracy": None,
    "risk_accuracy": None,
    "model_source": "Demo training model"
}


# =========================================================
# CLEAN COLUMN NAME
# =========================================================

def clean_column_name(name):

    name = str(name).strip().lower()

    name = name.replace("/", "_")
    name = name.replace("-", "_")
    name = name.replace(" ", "_")

    return re.sub(
        r"[^a-z0-9_]",
        "",
        name
    )


# =========================================================
# FIND COLUMN
# =========================================================

def find_column(columns, aliases):

    cleaned = {
        clean_column_name(c): c
        for c in columns
    }

    # Exact match
    for alias in aliases:

        if alias in cleaned:
            return cleaned[alias]

    # Partial match
    for c_clean, original in cleaned.items():

        for alias in aliases:

            if alias in c_clean or c_clean in alias:
                return original

    return None


# =========================================================
# NUMERIC COLUMNS
# =========================================================

def numeric_columns(df):

    result = []

    for col in df.columns:

        converted = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        if converted.notna().mean() >= 0.70:
            result.append(col)

    return result


# =========================================================
# BUILD COMMON NETWORK FEATURES
# =========================================================

def build_common_features(raw_df):

    df = raw_df.copy()

    df.columns = [
        clean_column_name(c)
        for c in df.columns
    ]

    nums = numeric_columns(df)

    if len(nums) < 2:

        raise ValueError(
            "The file does not contain enough numeric network features."
        )

    result = pd.DataFrame(index=df.index)

    used = set()

    # Known network columns
    for feature, aliases in ALIASES.items():

        col = find_column(
            df.columns,
            aliases
        )

        if col is not None:

            result[feature] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

            used.add(col)

    # Remaining numeric columns
    available = [
        c for c in nums
        if c not in used
    ]

    fallback_order = [
        "packet_count",
        "packet_size",
        "connection_duration",
        "failed_connections"
    ]

    # Automatic mapping
    for feature in fallback_order:

        if feature not in result.columns and available:

            result[feature] = pd.to_numeric(
                df[available.pop(0)],
                errors="coerce"
            )

    # Check required features
    for feature in fallback_order:

        if feature not in result.columns:

            raise ValueError(
                "Could not map enough network-flow features. "
                "Please upload a network-security CSV containing "
                "numeric traffic features."
            )

    # Numeric conversion
    for column in fallback_order:

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce"
        )

    # Remove infinity
    result = result.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Remove invalid rows
    result = result.dropna(
        subset=fallback_order
    ).copy()

    if len(result) < 5:

        raise ValueError(
            "The uploaded file has fewer than 5 usable network records."
        )

    # Negative values are not useful
    for column in fallback_order:

        result[column] = result[column].clip(
            lower=0
        )

    # Store original indexes
    original_index = result.index.copy()

    # Reset ML index
    result = result.reset_index(drop=True)

    result.attrs["original_index"] = original_index

    return result


# =========================================================
# DATASET DETECTION
# =========================================================

def detect_dataset(raw_df):

    cols = [
        clean_column_name(c)
        for c in raw_df.columns
    ]

    text = " ".join(cols)

    # UNSW-NB15
    if (
        "attack_cat" in cols
        or "sttl" in cols
        or "ct_state_ttl" in cols
    ):

        return "UNSW-NB15 / UNSW-style network dataset"

    # CIC datasets
    if (
        "destination_port" in cols
        or "dst_port" in cols
        or "flow_bytes_s" in cols
        or "total_fwd_packets" in cols
    ):

        return "CIC-IDS / CIC-style network-flow dataset"

    # Generic network dataset
    network_words = [

        "packet",
        "flow",
        "bytes",
        "src_ip",
        "dst_ip",
        "source_ip",
        "destination_ip",
        "protocol",
        "port",
        "duration",
        "attack",
        "label",
        "connection"

    ]

    score = sum(
        word in text
        for word in network_words
    )

    if score >= 2:

        return "Generic compatible network-security dataset"

    raise ValueError(
        "This does not look like a compatible "
        "network-security dataset."
    )


# =========================================================
# MODEL 2 - ATTACK LABELS
# =========================================================

def make_attack_labels(feature_df, raw_df):

    label_col = find_column(
        raw_df.columns,
        LABEL_ALIASES
    )

    # Dataset already contains label
    if label_col is not None:

        labels = (
            raw_df[label_col]
            .astype(str)
            .str.strip()
        )

        labels = labels.replace(
            "",
            "Unknown"
        ).fillna("Unknown")

        original_index = feature_df.attrs.get(
            "original_index"
        )

        if original_index is not None:

            labels = labels.iloc[
                list(original_index)
            ]

        else:

            labels = labels.iloc[
                :len(feature_df)
            ]

        labels = labels.reset_index(drop=True)

        counts = labels.value_counts()

        labels = labels.where(
            labels.map(counts) >= 2,
            "Other"
        )

        if labels.nunique() >= 2:

            return (
                labels,
                "Dataset ground-truth label"
            )

    # =====================================================
    # FALLBACK
    # =====================================================

    f = feature_df

    q_packet = f[
        "packet_count"
    ].quantile(0.85)

    q_failed = f[
        "failed_connections"
    ].quantile(0.85)

    q_small = f[
        "packet_size"
    ].quantile(0.15)

    labels = []

    for _, row in f.iterrows():

        if (
            row["failed_connections"] >= q_failed
            and row["failed_connections"] > 0
        ):

            labels.append(
                "Brute Force / Failed Attempts"
            )

        elif row["packet_count"] >= q_packet:

            labels.append(
                "High Traffic / DoS-like"
            )

        elif row["packet_size"] <= q_small:

            labels.append(
                "Scan-like Traffic"
            )

        else:

            labels.append(
                "Normal"
            )

    return (
        pd.Series(labels),
        "AI traffic-pattern grouping"
    )


# =========================================================
# MODEL 3 - RISK LABELS
# =========================================================

def make_risk_labels(
    feature_df,
    attack_labels
):

    f = feature_df.copy()

    risk = []

    for i, (_, row) in enumerate(
        f.iterrows()
    ):

        packet_score = (
            f["packet_count"]
            <= row["packet_count"]
        ).mean()

        failed_score = (
            f["failed_connections"]
            <= row["failed_connections"]
        ).mean()

        duration_score = (
            f["connection_duration"]
            <= row["connection_duration"]
        ).mean()

        score = (
            0.40 * packet_score
            + 0.40 * failed_score
            + 0.20 * duration_score
        )

        attack_text = str(
            attack_labels.iloc[i]
        ).lower()

        if "normal" not in attack_text:

            score += 0.20

        if score >= 0.80:

            risk.append("Critical")

        elif score >= 0.60:

            risk.append("High")

        elif score >= 0.35:

            risk.append("Medium")

        else:

            risk.append("Low")

    return pd.Series(risk)


# =========================================================
# RANDOM FOREST TRAINING
# =========================================================

def train_rf(
    X,
    y,
    estimators=150
):

    y = pd.Series(y).astype(str)

    counts = y.value_counts()

    if y.nunique() < 2:
        return None, None

    valid_classes = counts[
        counts >= 2
    ].index

    mask = y.isin(valid_classes)

    X2 = X.loc[
        mask
    ].reset_index(drop=True)

    y2 = y.loc[
        mask
    ].reset_index(drop=True)

    if (
        y2.nunique() < 2
        or len(y2) < 6
    ):

        return None, None

    try:

        X_train, X_test, y_train, y_test = train_test_split(

            X2,
            y2,

            test_size=0.20,

            random_state=42,

            stratify=y2
        )

        model = RandomForestClassifier(

            n_estimators=estimators,

            random_state=42,

            class_weight="balanced"
        )

        model.fit(
            X_train,
            y_train
        )

        pred = model.predict(
            X_test
        )

        accuracy = round(
            accuracy_score(
                y_test,
                pred
            ) * 100,
            2
        )

        return (
            model,
            accuracy
        )

    except ValueError:

        model = RandomForestClassifier(

            n_estimators=estimators,

            random_state=42,

            class_weight="balanced"
        )

        model.fit(
            X2,
            y2
        )

        return (
            model,
            None
        )


# =========================================================
# ANALYZE DATASET
# =========================================================

def analyze_dataset(raw_df):

    dataset_name = detect_dataset(
        raw_df
    )

    common = build_common_features(
        raw_df
    )

    feature_cols = [
        "packet_count",
        "packet_size",
        "connection_duration",
        "failed_connections"
    ]

    # =====================================================
    # MODEL 1 - ANOMALY DETECTION
    # =====================================================

    anomaly_model = IsolationForest(

        n_estimators=150,

        contamination="auto",

        random_state=42
    )

    anomaly_model.fit(
        common[feature_cols]
    )

    anomaly_prediction = anomaly_model.predict(
        common[feature_cols]
    )

    common["status"] = pd.Series(
        anomaly_prediction,
        index=common.index
    ).map({

        1: "Normal",

        -1: "Suspicious"

    })

    # =====================================================
    # MODEL 2 - ATTACK CLASSIFICATION
    # =====================================================

    attack_labels, label_source = make_attack_labels(

        common,
        raw_df
    )

    attack_model, attack_accuracy = train_rf(

        common[feature_cols],

        attack_labels,

        150
    )

    if attack_model is not None:

        common["attack_type"] = attack_model.predict(
            common[feature_cols]
        )

    else:

        common["attack_type"] = attack_labels.values

    # =====================================================
    # MODEL 3 - RISK PREDICTION
    # =====================================================

    risk_labels = make_risk_labels(

        common,

        pd.Series(
            common["attack_type"]
        )
    )

    risk_model, risk_accuracy = train_rf(

        common[feature_cols],

        risk_labels,

        150
    )

    if risk_model is not None:

        common["risk_level"] = risk_model.predict(
            common[feature_cols]
        )

    else:

        common["risk_level"] = risk_labels.values

    return (

        common,

        dataset_name,

        label_source,

        attack_accuracy,

        risk_accuracy,

        anomaly_model,

        attack_model,

        risk_model
    )


# =========================================================
# DEFAULT MODELS
# =========================================================

def create_default_models():

    np.random.seed(42)

    n = 500

    demo = pd.DataFrame({

        "packet_count":
            np.random.randint(
                20,
                800,
                n
            ),

        "packet_size":
            np.random.randint(
                40,
                1500,
                n
            ),

        "connection_duration":
            np.random.randint(
                1,
                600,
                n
            ),

        "failed_connections":
            np.random.randint(
                0,
                15,
                n
            )
    })

    # ---------------------------------------------
    # Attack labels for initial manual prediction
    # ---------------------------------------------

    labels = []

    for _, row in demo.iterrows():

        if row["failed_connections"] >= 10:

            labels.append("Brute Force")

        elif row["packet_count"] >= 600:

            labels.append("DDoS / High Traffic")

        elif row["packet_size"] <= 150:

            labels.append("Port Scan")

        else:

            labels.append("Normal")

    labels = pd.Series(labels)

    # ---------------------------------------------
    # Risk labels
    # ---------------------------------------------

    risks = make_risk_labels(
        demo,
        labels
    )

    features = [
        "packet_count",
        "packet_size",
        "connection_duration",
        "failed_connections"
    ]

    # Anomaly
    anomaly_model = IsolationForest(
        n_estimators=150,
        contamination="auto",
        random_state=42
    )

    anomaly_model.fit(
        demo[features]
    )

    # Attack
    attack_model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced"
    )

    attack_model.fit(
        demo[features],
        labels
    )

    # Risk
    risk_model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced"
    )

    risk_model.fit(
        demo[features],
        risks
    )

    current_models["anomaly"] = anomaly_model
    current_models["attack"] = attack_model
    current_models["risk"] = risk_model

    current_models["attack_accuracy"] = None
    current_models["risk_accuracy"] = None

    current_models["model_source"] = (
        "Initial demo training model"
    )


# =========================================================
# DATABASE
# =========================================================

def create_database():

    conn = sqlite3.connect(
        DATABASE
    )

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS security_records (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            packet_count REAL,

            packet_size REAL,

            connection_duration REAL,

            failed_connections REAL,

            status TEXT,

            attack_type TEXT,

            risk_level TEXT,

            source TEXT,

            filename TEXT,

            dataset_type TEXT,

            created_at TEXT

        )
    """)

    conn.commit()

    conn.close()


# =========================================================
# SAVE ANALYSIS
# =========================================================

def save_analysis(
    result_df,
    filename,
    dataset_type
):

    conn = sqlite3.connect(
        DATABASE
    )

    cursor = conn.cursor()

    now = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    for _, row in result_df.iterrows():

        cursor.execute("""
            INSERT INTO security_records (

                packet_count,
                packet_size,
                connection_duration,
                failed_connections,
                status,
                attack_type,
                risk_level,
                source,
                filename,
                dataset_type,
                created_at

            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

        """, (

            float(row["packet_count"]),

            float(row["packet_size"]),

            float(row["connection_duration"]),

            float(row["failed_connections"]),

            str(row["status"]),

            str(row["attack_type"]),

            str(row["risk_level"]),

            "CSV Upload",

            filename,

            dataset_type,

            now

        ))

    conn.commit()

    conn.close()


# =========================================================
# DASHBOARD DATA
# =========================================================

def get_dashboard_data():

    conn = sqlite3.connect(
        DATABASE
    )

    total = pd.read_sql_query(

        """
        SELECT COUNT(*) AS n
        FROM security_records
        """,

        conn

    ).iloc[0]["n"]

    normal = pd.read_sql_query(

        """
        SELECT COUNT(*) AS n
        FROM security_records
        WHERE status='Normal'
        """,

        conn

    ).iloc[0]["n"]

    suspicious = pd.read_sql_query(

        """
        SELECT COUNT(*) AS n
        FROM security_records
        WHERE status='Suspicious'
        """,

        conn

    ).iloc[0]["n"]

    high_risk = pd.read_sql_query(

        """
        SELECT COUNT(*) AS n
        FROM security_records
        WHERE risk_level IN ('High','Critical')
        """,

        conn

    ).iloc[0]["n"]

    attacks = pd.read_sql_query(

        """
        SELECT
            attack_type,
            COUNT(*) AS count
        FROM security_records
        GROUP BY attack_type
        ORDER BY count DESC
        """,

        conn

    )

    risks = pd.read_sql_query(

        """
        SELECT
            risk_level,
            COUNT(*) AS count
        FROM security_records
        GROUP BY risk_level
        """,

        conn

    )

    history = pd.read_sql_query(

        """
        SELECT
            filename,
            dataset_type,
            created_at,
            COUNT(*) AS records,

            SUM(
                CASE
                    WHEN status='Suspicious'
                    THEN 1
                    ELSE 0
                END
            ) AS suspicious,

            SUM(
                CASE
                    WHEN risk_level IN ('High','Critical')
                    THEN 1
                    ELSE 0
                END
            ) AS high_risk

        FROM security_records

        GROUP BY
            filename,
            dataset_type,
            created_at

        ORDER BY created_at DESC

        LIMIT 20
        """,

        conn

    )

    conn.close()

    return {

        "total_records":
            int(total),

        "normal_records":
            int(normal),

        "suspicious_records":
            int(suspicious),

        "high_risk":
            int(high_risk),

        "attack_data":
            attacks.to_dict(
                orient="records"
            ),

        "risk_data":
            risks.to_dict(
                orient="records"
            ),

        "history":
            history.to_dict(
                orient="records"
            )
    }


# =========================================================
# MANUAL PREDICTION
# =========================================================

@app.route(
    "/manual_predict",
    methods=["POST"]
)
def manual_predict():

    try:

        packet_count = float(
            request.form.get(
                "packet_count"
            )
        )

        packet_size = float(
            request.form.get(
                "packet_size"
            )
        )

        connection_duration = float(
            request.form.get(
                "connection_duration"
            )
        )

        failed_connections = float(
            request.form.get(
                "failed_connections"
            )
        )

        if min(
            packet_count,
            packet_size,
            connection_duration,
            failed_connections
        ) < 0:

            raise ValueError(
                "Values cannot be negative."
            )

        features = pd.DataFrame([{

            "packet_count":
                packet_count,

            "packet_size":
                packet_size,

            "connection_duration":
                connection_duration,

            "failed_connections":
                failed_connections

        }])

        anomaly_model = current_models["anomaly"]
        attack_model = current_models["attack"]
        risk_model = current_models["risk"]

        # -------------------------------
        # MODEL 1
        # -------------------------------

        anomaly_prediction = anomaly_model.predict(
            features
        )[0]

        status = (
            "Normal"
            if anomaly_prediction == 1
            else "Suspicious"
        )

        # -------------------------------
        # MODEL 2
        # -------------------------------

        attack_type = attack_model.predict(
            features
        )[0]

        attack_confidence = None

        if hasattr(
            attack_model,
            "predict_proba"
        ):

            probabilities = attack_model.predict_proba(
                features
            )[0]

            attack_confidence = round(
                float(max(probabilities)) * 100,
                2
            )

        # -------------------------------
        # MODEL 3
        # -------------------------------

        risk_level = risk_model.predict(
            features
        )[0]

        risk_confidence = None

        if hasattr(
            risk_model,
            "predict_proba"
        ):

            probabilities = risk_model.predict_proba(
                features
            )[0]

            risk_confidence = round(
                float(max(probabilities)) * 100,
                2
            )

        manual_result = {

            "packet_count":
                packet_count,

            "packet_size":
                packet_size,

            "connection_duration":
                connection_duration,

            "failed_connections":
                failed_connections,

            "status":
                status,

            "attack_type":
                str(attack_type),

            "risk_level":
                str(risk_level),

            "attack_confidence":
                attack_confidence,

            "risk_confidence":
                risk_confidence,

            "model_source":
                current_models["model_source"]
        }

        data = get_dashboard_data()

        return render_template(

            "dashboard.html",

            **data,

            upload_result=None,

            manual_result=manual_result,

            attack_accuracy=
                current_models["attack_accuracy"],

            risk_accuracy=
                current_models["risk_accuracy"],

            label_source=None,

            dataset_type=None
        )

    except Exception as e:

        flash(
            "❌ Manual prediction error: " + str(e),
            "error"
        )

        return redirect(
            url_for("dashboard")
        )


# =========================================================
# DASHBOARD
# =========================================================

@app.route(
    "/",
    methods=["GET"]
)
def dashboard():

    data = get_dashboard_data()

    return render_template(

        "dashboard.html",

        **data,

        upload_result=None,

        manual_result=None,

        attack_accuracy=
            current_models["attack_accuracy"],

        risk_accuracy=
            current_models["risk_accuracy"],

        label_source=None,

        dataset_type=None
    )


# =========================================================
# UPLOAD CSV
# =========================================================

@app.route(
    "/upload_csv",
    methods=["POST"]
)
def upload_csv():

    file = request.files.get(
        "csv_file"
    )

    if (
        file is None
        or file.filename == ""
    ):

        flash(
            "⚠️ Please select a CSV network-security dataset.",
            "error"
        )

        return redirect(
            url_for("dashboard")
        )

    if not file.filename.lower().endswith(".csv"):

        flash(
            "❌ Invalid file! Only CSV files are supported.",
            "error"
        )

        return redirect(
            url_for("dashboard")
        )

    original_filename = secure_filename(
        file.filename
    )

    if not original_filename:

        flash(
            "❌ Invalid filename.",
            "error"
        )

        return redirect(
            url_for("dashboard")
        )

    # =====================================================
    # UNIQUE FILE NAME
    # So multiple uploads are preserved
    # =====================================================

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    filename = (
        timestamp
        + "_"
        + original_filename
    )

    filepath = os.path.join(

        app.config["UPLOAD_FOLDER"],

        filename
    )

    try:

        # Save original uploaded CSV
        file.save(filepath)

        # Read CSV
        raw_df = pd.read_csv(
            filepath,
            low_memory=False
        )

        if raw_df.empty:

            raise ValueError(
                "The uploaded CSV is empty."
            )

        # =================================================
        # ANALYZE
        # =================================================

        (

            result_df,

            dataset_type,

            label_source,

            attack_accuracy,

            risk_accuracy,

            anomaly_model,

            attack_model,

            risk_model

        ) = analyze_dataset(
            raw_df
        )

        # =================================================
        # UPDATE CURRENT MODELS
        # =================================================

        current_models["anomaly"] = anomaly_model
        current_models["attack"] = attack_model
        current_models["risk"] = risk_model

        current_models["attack_accuracy"] = (
            attack_accuracy
        )

        current_models["risk_accuracy"] = (
            risk_accuracy
        )

        current_models["model_source"] = (
            "Models trained using uploaded dataset: "
            + original_filename
        )

        # =================================================
        # SAVE DATABASE HISTORY
        # =================================================

        save_analysis(

            result_df,

            original_filename,

            dataset_type
        )

        # =================================================
        # LATEST RESULT
        # =================================================

        upload_result = {

            "filename":
                original_filename,

            "dataset_type":
                dataset_type,

            "label_source":
                label_source,

            "total":
                len(result_df),

            "normal":
                int(
                    (
                        result_df["status"]
                        == "Normal"
                    ).sum()
                ),

            "suspicious":
                int(
                    (
                        result_df["status"]
                        == "Suspicious"
                    ).sum()
                ),

            "high_risk":
                int(
                    result_df["risk_level"].isin(
                        [
                            "High",
                            "Critical"
                        ]
                    ).sum()
                ),

            "rows":
                result_df[
                    [
                        "packet_count",
                        "packet_size",
                        "connection_duration",
                        "failed_connections",
                        "status",
                        "attack_type",
                        "risk_level"
                    ]
                ].head(25).to_dict(
                    orient="records"
                )
        }

        data = get_dashboard_data()

        return render_template(

            "dashboard.html",

            **data,

            upload_result=upload_result,

            manual_result=None,

            attack_accuracy=
                attack_accuracy,

            risk_accuracy=
                risk_accuracy,

            label_source=
                label_source,

            dataset_type=
                dataset_type
        )

    except Exception as e:

        # Remove failed upload file
        try:

            if os.path.exists(filepath):
                os.remove(filepath)

        except Exception:
            pass

        flash(
            "❌ Dataset processing error: "
            + str(e),
            "error"
        )

        return redirect(
            url_for("dashboard")
        )


# =========================================================
# DELETE ONE DATASET HISTORY
# =========================================================

@app.route(
    "/delete_dataset",
    methods=["POST"]
)
def delete_dataset():

    filename = request.form.get(
        "filename"
    )

    created_at = request.form.get(
        "created_at"
    )

    if (
        not filename
        or not created_at
    ):

        flash(
            "❌ Invalid dataset selected.",
            "error"
        )

        return redirect(
            url_for("dashboard")
        )

    conn = sqlite3.connect(
        DATABASE
    )

    cursor = conn.cursor()

    cursor.execute(

        """
        DELETE FROM security_records
        WHERE filename = ?
        AND created_at = ?
        """,

        (
            filename,
            created_at
        )
    )

    deleted_rows = cursor.rowcount

    conn.commit()

    conn.close()

    if deleted_rows > 0:

        flash(

            "🗑️ Dataset '"
            + filename
            + "' history deleted successfully.",

            "success"
        )

    else:

        flash(
            "⚠️ Dataset history was not found.",
            "error"
        )

    return redirect(
        url_for("dashboard")
    )


# =========================================================
# DELETE ALL HISTORY
# =========================================================

@app.route(
    "/delete_all_history",
    methods=["POST"]
)
def delete_all_history():

    conn = sqlite3.connect(
        DATABASE
    )

    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM security_records"
    )

    conn.commit()

    conn.close()

    flash(
        "🗑️ All uploaded dataset history has been deleted.",
        "success"
    )

    return redirect(
        url_for("dashboard")
    )


# =========================================================
# DATABASE INITIALIZATION
# =========================================================

create_database()

# Initial AI models for manual prediction
create_default_models()


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    print(
        "==================================="
    )

    print(
        " AI CYBERSECURITY MONITORING SYSTEM"
    )

    print(
        "==================================="
    )

    print(
        "ONE DASHBOARD MODE"
    )

    print(
        "CSV Upload + Manual Prediction + 3 AI Models"
    )

    print(
        "SQLite History + Delete Options"
    )

    print(
        "Server: http://127.0.0.1:5000"
    )

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
