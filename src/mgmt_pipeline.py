import glob
import pandas as pd
from PIL import Image

from src.utils import get_clean_smiles, get_converted_unit, detect_csv_delimiter

import logging
from src.predictor.predictor_base import PredictorBase
from src.data.featurizer import FeaturizerBase
from src.data.reducer import ReducerBase
from src.data.split import DataSplitterBase
import gin
import numpy as np
from pathlib import Path


@gin.configurable()
class ManagementPipeline:
    """'meta-Pipeline' (sounds cool, eh?) for handling temporary/one-off work"""

    possible_smiles_cols = ["SMILES", "Smiles", "smiles", "molecule"]
    root_categories = None

    def __init__(
        self,
        raw_input_dir: Path | str,
        normalized_input_dir: Path | str,
        output_dir: Path | str,
        mode: str,
        force_normalize_all: bool = False,
        root_categories: list[str] | None = None,
        classify_datasets_list: list[str] | None = None,
        classify_thresholds: list[float] | None = None,
        classify_target_unit: str | None = "uM",
        explore_datasets_list: list[str] | None = None,
        explore_datasets_categories: list[str] | None = None,
        plot_title: str | None = None,
        reducer: ReducerBase = None,
        splitter: DataSplitterBase = None,
        predictor: PredictorBase = None,
        featurizer: FeaturizerBase = None,
    ):
        self._raw_input_dir: Path = Path(raw_input_dir)
        self.normalized_input_dir: Path = Path(normalized_input_dir)
        self.output_dir: Path = Path(output_dir)

        self.mode = mode
        self.force_normalize_all = force_normalize_all
        self.classify_datasets_list = classify_datasets_list
        self.classify_thresholds = classify_thresholds
        self.classify_target_unit = classify_target_unit
        self.explore_datasets_list = explore_datasets_list
        self.explore_datasets_categories = explore_datasets_categories

        ManagementPipeline.root_categories = root_categories

        self.splitter = splitter
        self.predictor = predictor
        self.featurizer = featurizer
        self.reducer = reducer
        if self.reducer:
            self.visualizer = reducer.get_associated_visualizer()
            if plot_title:
                self.visualizer.set_plot_title(plot_title)
            else:
                self.visualizer.set_plot_title(
                    f"{self.reducer.name} projection of {self.featurizer.feature_name} features"
                )

    def run(self):
        if self.mode == "classify":
            assert self.classify_datasets_list, (
                "Ran ManagementPipeline in 'classify' mode without "
                "specifying which datasets to convert to classification"
            )
            assert self.classify_thresholds, (
                "Ran ManagementPipeline in 'classify' mode without "
                "specifying thresholds with which to convert to classification"
            )
            assert self.classify_target_unit, (
                "Ran ManagementPipeline in 'classify' mode without "
                "specifying which unit to convert all of the Standard Values to"
            )
            self.make_datasets_into_classification()
        elif self.mode == "normalize":
            assert self.force_normalize_all is not None, (
                "Ran ManagementPipeline in 'normalize' mode without "
                "specifying .force_normalize_all attribute"
            )
            self.normalize_datasets()
        elif self.mode == "visualize":
            assert self.reducer is not None, (
                "Ran ManagementPipeline in 'visualize' mode without "
                "specifying .reducer: ReducerBase attribute"
            )
            assert (
                self.explore_datasets_list and not self.explore_datasets_categories
            ) or (
                not self.explore_datasets_list and self.explore_datasets_categories
            ), "Either dataset categories or an explicit list must be specified, not both."

            self.dump_exploratory_visualization()

    @staticmethod
    def get_clean_smiles_df(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
        """Consolidate pandas, our-internal NaN-dropping into one function"""
        pre_dropna_len = len(df)
        df = df.dropna(subset=smiles_col)

        pre_cleaning_len = len(df)
        df[smiles_col].apply(get_clean_smiles)

        df = df.dropna(subset=[smiles_col]).reset_index(drop=True)

        if pre_dropna_len != pre_cleaning_len:
            logging.info(
                f"Dropped {pre_dropna_len - pre_cleaning_len} 'nan' SMILES after pd.read_csv"
            )
        if pre_cleaning_len != len(df):
            logging.info(f"Dropped {pre_cleaning_len - len(df)} invalid SMILES")
        logging.info(f"Dataset size: {len(df)}")

        return df

    @staticmethod
    def get_canon_smiles_df(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
        pre_canonicalization_len = len(df)
        df[smiles_col].apply(get_clean_smiles)

        df = df.dropna(subset=[smiles_col]).reset_index(drop=True)

        if pre_canonicalization_len != len(df):
            logging.info(
                f"Canonicalization resulted in {pre_canonicalization_len - len(df)} "
                "'None' SMILES, all were dropped."
            )

        return df

    @classmethod
    def get_normalized_filename(cls, ds_globbed_path: Path) -> str:
        dataset_dir_parts = ds_globbed_path.parent.parts

        begin_index = None
        for root_domain in cls.root_categories:
            if root_domain in dataset_dir_parts:
                begin_index = dataset_dir_parts.index(root_domain)
                break

        if begin_index is None:
            raise ValueError(
                f"Unable to match any part of Path: {str(ds_globbed_path)} "
                f"to one of root domains (of interest): {str(cls.root_categories)}"
            )

        basename_parts = dataset_dir_parts[begin_index:]
        normalized_basename = "_".join(basename_parts).replace("-", "").lower()

        return normalized_basename

    @classmethod
    def get_smiles_col_in_raw(cls, raw_df) -> str:
        for smiles_col_variant in cls.possible_smiles_cols:
            if smiles_col_variant in raw_df.columns:
                return smiles_col_variant

        # Failed
        raise ValueError(
            "Failed to find one of SMILES column name variants:",
            str(cls.possible_smiles_cols),
            "in dataframe:",
            str(raw_df.head()),
        )

    def make_datasets_into_classification(self):
        for dataset in self.classify_datasets_list:
            dataset_path = self.normalized_input_dir / dataset

            dataset_df = pd.read_csv(
                dataset_path, delimiter=detect_csv_delimiter(dataset_path)
            )
            if "Standard Units" not in dataset_df.columns:
                logging.info(
                    f"Dataset '{dataset}' has no 'Standard Units' column, skipping."
                )
                continue

            classification_df = self.get_dataset_as_classification(dataset_df)

            (self.output_dir / "classification").mkdir(parents=True, exist_ok=True)

            classification_df.to_csv(
                self.output_dir / "classification" / dataset, index=False
            )

    def get_dataset_as_classification(self, dataset_df: pd.DataFrame) -> pd.DataFrame:
        normalized_value_colname = f"normalized_value_{self.classify_target_unit}"
        classification_colname = "class"

        def normalize_row_value(pd_row):
            val = float(pd_row["Standard Value"])
            from_unit = pd_row["Standard Units"]
            mol_weight = pd_row["Molecular Weight"]

            normalized_value: float = get_converted_unit(
                val, from_unit, self.classify_target_unit, mol_weight=mol_weight
            )

            return normalized_value

        def assign_class(pd_row):
            relation = pd_row["Standard Relation"]
            normalized_val = pd_row[normalized_value_colname]

            if "'" in relation:
                relation = relation.split("'")[1]

            min_class_val = self.classify_thresholds[0]
            max_class_val = self.classify_thresholds[-1]

            if relation == "<" and normalized_val < min_class_val:
                return 0
            elif relation == "<=" and normalized_val <= min_class_val:
                return 0
            # recall that 1 threshold => 2 classes, hence not "len - 1"
            elif relation == ">" and normalized_val > max_class_val:
                return len(self.classify_thresholds)
            elif relation == ">=" and normalized_val >= min_class_val:
                return len(self.classify_thresholds)

            return np.nan

        leave_cols = {
            "smiles",
            "Standard Value",
            "Standard Units",
            "Standard Relation",
            "Molecular Weight",
            normalized_value_colname,
            classification_colname,
        }

        pre_normalization_len = len(dataset_df)

        dataset_df[normalized_value_colname] = dataset_df.apply(
            normalize_row_value, axis=1
        )

        dataset_df.dropna(subset=[normalized_value_colname], inplace=True)
        pre_classification_len = len(dataset_df)

        if pre_normalization_len != pre_classification_len:
            logging.info(
                f"Dropped {pre_normalization_len - pre_classification_len} 'nan' "
                f"SMILES after value unit normalization"
            )

        if pre_classification_len > 0:
            dataset_df[classification_colname] = dataset_df.apply(assign_class, axis=1)
            dataset_df.dropna(subset=[classification_colname], inplace=True)
            dataset_df[classification_colname] = dataset_df[
                classification_colname
            ].astype("int32")

        if pre_classification_len != len(dataset_df):
            logging.info(
                f"Dropped {pre_classification_len - len(dataset_df)} 'nan' SMILES after "
                f"'real-valued => class' conversion"
            )

        cols_to_drop = [col for col in dataset_df.columns if col not in leave_cols]
        dataset_df.drop(cols_to_drop, axis=1, inplace=True)

        return dataset_df

    def normalize_datasets(self, force_normalize_all: bool = False):
        datasets = glob.glob(f"{str(self._raw_input_dir)}/**/*.csv", recursive=True)

        # pairs of "before, after" paths
        datasets_paths: list[tuple[Path, Path]] = [
            (
                Path(ds_glob),
                self.get_df_output_path(self.get_normalized_filename(Path(ds_glob))),
            )
            for ds_glob in datasets
        ]

        # filter already normalized
        if not force_normalize_all:
            datasets_paths = [
                (ds_glob, ds_path)
                for ds_glob, ds_path in datasets_paths
                if not ds_path.exists()
            ]

        for ds_glob, ds_path in datasets_paths:
            normalized_dataset_df = self.get_normalized_df(ds_glob)

            normalized_dataset_df.to_csv(ds_path, index=False)

    def get_normalized_df(self, ds_globbed_path: Path) -> pd.DataFrame:
        """Get ready-to-save df without NaNs and with canonical SMILES"""

        logging.info(f"Reading dataset: {str(ds_globbed_path)}")
        df_to_normalize = pd.read_csv(
            ds_globbed_path, delimiter=detect_csv_delimiter(ds_globbed_path)
        )
        logging.debug(f"Raw dataset size: {len(df_to_normalize)}")
        logging.debug(f"Raw dataset columns: {df_to_normalize.columns.tolist()}")

        df_to_normalize.rename(
            columns={self.get_smiles_col_in_raw(df_to_normalize): "smiles"},
            inplace=True,
        )

        df_to_normalize = self.get_clean_smiles_df(df_to_normalize, smiles_col="smiles")

        df_to_normalize = self.get_canon_smiles_df(df_to_normalize, smiles_col="smiles")

        return df_to_normalize

    def get_df_output_path(
        self,
        normalized_basename: str,
        prefix: str = "",
        suffix: str = "",
        extension: str = "csv",
    ) -> Path:
        """Get path to save normalized df at"""
        if prefix:
            normalized_basename = f"{prefix}_{normalized_basename}"
        if suffix:
            normalized_basename = f"{normalized_basename}_{suffix}"

        output_path_str = (
            f"{str(self.normalized_input_dir)}/" f"{normalized_basename}.{extension}"
        )
        return Path(output_path_str)

    def get_featurized_dataset_df(self, dataset_path: Path) -> pd.DataFrame:
        """
        Featurizes the entire training dataset.
        Returns pd.DataFrame with a featurized input column.

        dataset_path: Path to dataset in self.normalized_input_dir
        """

        df_to_featurize = pd.read_csv(
            dataset_path, delimiter=detect_csv_delimiter(dataset_path)
        )

        len_before_feat = len(df_to_featurize)

        feature_col_name = self.featurizer.feature_name
        df_to_featurize[feature_col_name] = df_to_featurize["smiles"].apply(
            lambda smiles: self.featurizer.feature_to_str(
                self.featurizer.featurize([smiles])
            )
        )

        len_after_feat = len(df_to_featurize)
        assert len_before_feat == len_after_feat, (
            f"{len_before_feat - len_after_feat} SMILES failed to featurize with"
            f"{self.featurizer.name}. 'get_featurized_dataset_df' expects featurizable SMILES."
        )

        return df_to_featurize

    def save_featurized_dataset(self, dataset_path: Path, df_featurized: pd.DataFrame):
        """dataset_path: Path to dataset in self.output_dir (normalized data & path)."""

        feature_col_name = self.featurizer.feature_name
        df_featurized[feature_col_name] = df_featurized[feature_col_name].apply(
            lambda feature: self.featurizer.feature_to_str(feature)
        )

        basename = dataset_path.name
        df_featurized.to_csv(self.output_dir / self.featurizer.name / basename)

    def load_featurized_dataset(self, dataset_path: Path):
        """dataset_path: Path to dataset in self.output_dir (normalized data & path)."""
        basename = dataset_path.name
        df_featurized = pd.read_csv(self.output_dir / self.featurizer.name / basename)

        feature_col_name = self.featurizer.feature_name
        df_featurized[feature_col_name] = df_featurized[feature_col_name].apply(
            lambda str_rep: self.featurizer.str_to_feature(str_rep)
        )

        return df_featurized

    def get_pca_input_form(self, featurized_dataset_df: pd.DataFrame) -> pd.DataFrame:
        """Get the form that PCA dim reduction anticipates from a featurized dataset (curr.: ECFP)"""
        featurized_dataset_df.drop(columns="smiles", inplace=True)
        pca_ready_df = pd.concat(
            [
                featurized_dataset_df[self.featurizer.feature_name]
                .apply(lambda s: pd.Series(list(map(int, s))))
                .add_prefix("bit_")
            ],
            axis=1,
        )

        return pca_ready_df

    def get_visualization(
        self, featurized_df_dict: dict[str, pd.DataFrame]
    ) -> Image.Image:
        """Take in the featurized versions of dataframes, dim-reduce them and return a vis. image"""
        reduced_df_dict = {
            k: self.reducer.get_reduced_df(self.get_pca_input_form(v))
            for k, v in featurized_df_dict.items()
        }

        visualization = self.visualizer.get_visualization(reduced_df_dict)

        return visualization

    def save_visualization(self, visualization: Image.Image):
        """Take a visualization image and save it to visualization output dir."""
        vis_output_dir = Path(self.output_dir) / "visualizations"

        vis_output_dir.mkdir(parents=True, exist_ok=True)

        if self.explore_datasets_categories:
            categories_prefix = "_".join(
                [cat[:2] for cat in self.explore_datasets_categories]
            )
            output_path = (
                vis_output_dir
                / f"{categories_prefix}_{self.featurizer.name}_visualization.png"
            )
        else:
            output_path = vis_output_dir / f"{self.featurizer.name}_visualization.png"

        visualization.save(output_path)

    def dump_exploratory_visualization(self):
        """Go over what is to be visualized from .cfg and save it to disk."""
        dataset_paths = None
        if self.explore_datasets_list:
            dataset_paths: list[Path] = [
                self.normalized_input_dir / Path(dataset_path)
                for dataset_path in self.explore_datasets_list
            ]
        elif self.explore_datasets_categories:
            candidates = glob.glob(str(self.normalized_input_dir / "*.csv"))
            dataset_paths = [
                Path(candidate)
                for candidate in candidates
                for category in self.explore_datasets_categories
                if category in candidate.lower()
            ]

        assert dataset_paths

        featurized_df_dict: dict[str, pd.DataFrame] = {
            str(ds_path): self.get_featurized_dataset_df(ds_path)
            for ds_path in dataset_paths
        }

        visualization = self.get_visualization(featurized_df_dict)

        self.save_visualization(visualization)
