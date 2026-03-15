"""Data loading, preprocessing, and feature engineering for NSL-KDD dataset."""

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler

# NSL-KDD column names (41 features + outcome + score)
COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "outcome", "score",
]

# Attack-type to category mapping
# 0=Normal, 1=Probe, 2=R2L, 3=DoS, 4=U2R
ATTACK_CATEGORY_MAP = {
    "normal": 0,
    # DoS attacks
    "neptune": 3, "back": 3, "land": 3, "pod": 3, "smurf": 3,
    "teardrop": 3, "mailbomb": 3, "apache2": 3, "processtable": 3,
    "udpstorm": 3, "worm": 3,
    # Probe attacks
    "ipsweep": 1, "nmap": 1, "portsweep": 1, "satan": 1,
    "mscan": 1, "saint": 1,
    # R2L attacks
    "ftp_write": 2, "guess_passwd": 2, "imap": 2, "multihop": 2,
    "phf": 2, "spy": 2, "warezclient": 2, "warezmaster": 2,
    "sendmail": 2, "named": 2, "snmpgetattack": 2, "snmpguess": 2,
    "xlock": 2, "xsnoop": 2, "httptunnel": 2,
    # U2R attacks
    "buffer_overflow": 4, "loadmodule": 4, "perl": 4, "rootkit": 4,
    "ps": 4, "sqlattack": 4, "xterm": 4,
}

CATEGORY_NAMES = {0: "Normal", 1: "Probe", 2: "R2L", 3: "DoS", 4: "U2R"}


def load_nsl_kdd(train_path, test_path):
    """Load and combine NSL-KDD train and test CSV files.

    Parameters
    ----------
    train_path : str
        Path to KDDTrain+.csv.
    test_path : str
        Path to KDDtest.csv.

    Returns
    -------
    tuple of (pd.DataFrame, int)
        Combined dataframe and the number of training rows.
    """
    df_train = pd.read_csv(train_path, header=None)
    df_test = pd.read_csv(test_path, header=None)
    full = pd.concat([df_train, df_test], ignore_index=True)
    full.columns = COLUMN_NAMES
    return full, df_train.shape[0]


def encode_attack_categories(df):
    """Replace attack names with numeric category labels.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with an ``outcome`` column containing attack names.

    Returns
    -------
    pd.Series
        Integer-encoded category labels.
    """
    return df["outcome"].replace(ATTACK_CATEGORY_MAP).astype(int)


def preprocess_features(df, outcome_series):
    """One-hot encode categorical features and drop metadata columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw combined DataFrame.
    outcome_series : pd.Series
        Encoded outcome labels.

    Returns
    -------
    pd.DataFrame
        DataFrame with dummy-encoded features and outcome column.
    """
    df = df.copy()
    df["outcome"] = outcome_series
    df.drop(columns=["score"], inplace=True, errors="ignore")
    return pd.get_dummies(df, drop_first=True)


def select_features(X_train, y_train, *datasets):
    """Select important features using ExtraTreesClassifier.

    Parameters
    ----------
    X_train : numpy.ndarray
        Training feature matrix.
    y_train : numpy.ndarray
        Training labels.
    *datasets : numpy.ndarray
        Additional datasets to transform with the same selector.

    Returns
    -------
    tuple
        (selector, X_train_selected, *transformed_datasets)
    """
    clf = ExtraTreesClassifier(random_state=42)
    clf.fit(X_train, y_train)
    selector = SelectFromModel(clf, prefit=True)

    results = [selector.transform(X_train)]
    for ds in datasets:
        results.append(selector.transform(ds))

    return (selector, *results)


def scale_features(*datasets):
    """Apply MinMax scaling (0-1) to multiple datasets.

    Parameters
    ----------
    *datasets : numpy.ndarray
        Arrays to scale.

    Returns
    -------
    tuple
        (scaler, *scaled_datasets)
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    results = []
    for i, ds in enumerate(datasets):
        if i == 0:
            results.append(scaler.fit_transform(ds))
        else:
            results.append(scaler.transform(ds))
    return (scaler, *results)
