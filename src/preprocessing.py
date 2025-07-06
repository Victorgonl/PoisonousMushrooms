import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import zscore

CONTINUOUS_FEATURES = ["cap-diameter", "stem-height", "stem-width"]
SURFACE_FEATURES = ["cap-surface", "stem-surface"]
COLORS_FEATURES = [
    "cap-color",
    "stem-color",
    "veil-color",
    "gill-color",
    "spore-print-color",
]
BINARY_FEATURES = ["veil-universal", "has-ring", "does-bruise-or-bleed"]
SURFACES = ["i", "g", "y", "s", "h", "l", "k", "t", "w", "e", "o", "f"]
COLORS = ["n", "b", "g", "r", "p", "u", "e", "w", "y", "l", "o", "k", "f"]


class MushroomPreprocessor:
    def __init__(self, secondary_mushroom: pd.DataFrame):
        self.secondary_mushroom = secondary_mushroom
        self.standard_scalers = {}
        self.surface_encoder = LabelEncoder()
        self.color_encoder = LabelEncoder()
        self.encoders_map = {}

    def fill_nan(self, data, feats, value):
        data[feats] = data[feats].fillna(value)
        return data

    def fill_missing_by_group_mean(self, data, target, references):
        data = data.copy()
        ref_combinations = [references[:i] for i in range(len(references), 0, -1)]
        group_means_dict = {tuple(refs): data.groupby(refs)[target].mean() for refs in ref_combinations}
        global_mean = data[target].mean()

        def fill_aux(row):
            if pd.isna(row[target]):
                for refs in ref_combinations:
                    key = tuple(row[ref] for ref in refs)
                    val = group_means_dict[tuple(refs)].get(key, np.nan)
                    if not pd.isna(val):
                        return val
                return global_mean
            return row[target]

        data[target] = data.apply(fill_aux, axis=1)
        return data

    def detect_outliers_zscore(self, series, threshold=3):
        z_scores = zscore(series.dropna())
        outlier_mask = pd.Series(False, index=series.index)
        outlier_mask[series.dropna().index] = np.abs(z_scores) > threshold
        return outlier_mask

    def _common_cleaning_steps(self, data):
        for feat in self.secondary_mushroom.select_dtypes(include="object").columns:
            if feat in data.columns:
                valid_values = self.secondary_mushroom[feat].dropna().unique()
                data[feat] = data[feat].where(data[feat].isin(valid_values))
        data["cap-surface"] = data["cap-surface"].replace({"d": np.nan})

        columns_to_drop = ["id", "has-ring", "veil-color", "spore-print-color", "stem-surface", "stem-root"]
        if "class" in data:
            columns_to_drop.append("class")
        data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

        data = self.fill_nan(
            data, ["stem-color", "gill-attachment", "gill-spacing", "gill-color", "ring-type", "cap-color"], "f"
        )
        data = self.fill_nan(data, ["cap-shape"], "o")
        data = self.fill_nan(data, ["veil-type"], "p")
        h = data["habitat"].value_counts().idxmax()
        b = data["does-bruise-or-bleed"].value_counts().idxmax()
        data = self.fill_nan(data, ["habitat"], h)
        data = self.fill_nan(data, ["does-bruise-or-bleed"], b)
        data = self.fill_nan(data, ["cap-surface"], "o")
        data = self.fill_missing_by_group_mean(data, "cap-diameter", ["cap-surface", "cap-shape", "cap-color"])

        data["ring-type"] = data["ring-type"].replace({"z": "o"})
        data["habitat"] = data["habitat"].replace({"u": "o", "w": "o"})

        return data

    def _encode_nominal_features(self, data):
        data["veil-universal"] = data["veil-type"].map({"p": 0, "u": 1}).astype("object")
        data = data.drop(columns=["veil-type"])
        data["does-bruise-or-bleed"] = data["does-bruise-or-bleed"].map({"f": 0, "t": 1}).astype("object")

        for feat in ["cap-color", "stem-color", "gill-color"]:
            data[feat] = self.color_encoder.transform(data[feat]).astype("object")

        data["cap-surface"] = self.surface_encoder.transform(data["cap-surface"]).astype("object")

        data["ring-type"] = data["ring-type"].replace({"m": "o"}).astype("object")
        data["habitat"] = data["habitat"].replace({"p": "o"}).astype("object")

        for feat in ["cap-shape", "gill-attachment", "gill-spacing", "ring-type", "habitat", "season"]:
            encoder = self.encoders_map[feat]
            data[feat] = encoder.transform(data[feat]).astype("object")

        return data

    def fit(self, data: pd.DataFrame):
        data = data.copy()
        tqdm.write("Starting fit preprocessing...")
        data = self._common_cleaning_steps(data)

        for feat in tqdm(CONTINUOUS_FEATURES, desc="Fitting scalers"):
            scaler = StandardScaler()
            self.standard_scalers[feat] = scaler
            scaler.fit(data[[feat]])

        self.surface_encoder.fit(SURFACES)
        self.color_encoder.fit(COLORS)

        other_nominal_features = [
            feat for feat in data if feat not in BINARY_FEATURES + COLORS_FEATURES + SURFACE_FEATURES + CONTINUOUS_FEATURES
        ]
        for feat in tqdm(other_nominal_features, desc="Fitting label encoders"):
            encoder = LabelEncoder()
            encoder.fit(data[feat])
            self.encoders_map[feat] = encoder

    def transform(self, data: pd.DataFrame, return_ids=False):
        data = data.copy()
        target = None
        ids = data["id"].copy() if "id" in data.columns else None

        tqdm.write("Starting transform preprocessing...")
        if "class" in data:
            target = (
                pd.DataFrame(data["class"]).replace({"e": 0, "p": 1}).rename(columns={"class": "poisonous"}).astype("object")
            )

        data = self._common_cleaning_steps(data)

        for feat in CONTINUOUS_FEATURES:
            scaler = self.standard_scalers[feat]
            data[feat] = scaler.transform(data[[feat]])

        data = self._encode_nominal_features(data)

        for feat in tqdm(CONTINUOUS_FEATURES, desc="Fixing outliers"):
            outliers = self.detect_outliers_zscore(data[feat])
            references = [t for t in data.columns if feat.split("-")[0] in t and t != feat]
            data.loc[outliers, feat] = np.nan
            data = self.fill_missing_by_group_mean(data, feat, references)

        if ids is not None and return_ids:
            return (
                data.to_numpy(dtype=np.float32),
                target.to_numpy(dtype=np.uint16).ravel() if target is not None else None,
                ids.to_numpy() if ids is not None and return_ids else None,
            )
        return data.to_numpy(dtype=np.float32), target.to_numpy(dtype=np.uint16).ravel() if target is not None else None

    def fit_transform(self, data: pd.DataFrame):
        self.fit(data.copy())
        return self.transform(data.copy())
