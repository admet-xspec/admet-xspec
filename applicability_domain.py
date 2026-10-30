"""Fit applicability domains for trained models in the cache, and calibrate them.

Needs only the cache: each model dir `data/cache/models/<model_key>/<split_key>/`
names its split, whose domain is fitted once and saved under
`data/cache/splits/<split_key>/applicability_domain/`. Models with a `model.pkl`
also get a calibration report (test metrics inside vs outside the domain).

Example:
    python applicability_domain.py --models 'AChE_human_pIC50_*' --splits 'scaffold_*'
"""

import argparse
import fnmatch
import logging
import pathlib
import time
from collections import defaultdict

import gin

from src.data.applicability_domain import (
    ApplicabilityDomainBase,
    fit_domain_for_model,
    get_or_fit_domain,
)
from src.data.data_interface import DataInterface

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


@gin.configurable
def _configured(
    data_interface: DataInterface, applicability_domain: ApplicabilityDomainBase
):
    """Return the gin-configured data interface and domain."""
    return data_interface, applicability_domain


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--cfg",
        "-c",
        type=pathlib.Path,
        default=pathlib.Path("configs/applicability_domains/tanimoto_knn.gin"),
        help="Gin config of the applicability domain",
    )
    parser.add_argument(
        "--data-interface-cfg",
        type=pathlib.Path,
        default=pathlib.Path("configs/data_interface/data_interface.gin"),
        help="Gin config of the data interface (cache location)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="*",
        help="Glob pattern of model cache keys, e.g. 'AChE_human_pIC50_*'",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="*",
        help="Glob pattern of split (data) cache keys, e.g. 'scaffold_*'",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Only fit the domains; skip calibrating them against saved models",
    )
    parser.add_argument(
        "--override-cache",
        action="store_true",
        help="Refit domains and recalibrate even if they are already saved",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Can be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )
    return parser.parse_args()


def find_model_dirs(
    data_interface: DataInterface, models_pattern: str, splits_pattern: str
) -> dict[str, list[str]]:
    """Return split key -> model keys of the cached model dirs matching the patterns."""
    models_by_split = defaultdict(list)
    for model_dir in sorted(data_interface.models_dir.glob(f"{models_pattern}/*/")):
        split_key = model_dir.name
        if fnmatch.fnmatch(split_key, splits_pattern):
            models_by_split[split_key].append(model_dir.parent.name)
    return dict(models_by_split)


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(level=args.log_level, format="%(message)s")

    for cfg in (args.data_interface_cfg, args.cfg):
        if not cfg.is_file():
            raise FileNotFoundError(f"Config file {cfg} not found.")
    gin.parse_config_files_and_bindings(
        [str(args.data_interface_cfg), str(args.cfg)],
        [
            "_configured.data_interface = %data_interface",
            "_configured.applicability_domain = %applicability_domain",
        ],
    )
    data_interface, domain = _configured()
    if domain is None:
        raise ValueError(f"{args.cfg} configures no applicability domain.")
    data_interface.set_override_cache(args.override_cache)

    time_start = time.time()
    models_by_split = find_model_dirs(data_interface, args.models, args.splits)
    n_models = sum(len(keys) for keys in models_by_split.values())
    logging.info("Applicability domain configuration summary:")
    logging.info(f"* Domain: {domain.get_cache_key()}")
    logging.info(f"* Model dirs: {n_models} over {len(models_by_split)} splits")
    logging.info(f"* Calibrate: {not args.no_calibrate}")

    n_fitted, n_calibrated, failed = 0, 0, []
    for split_key, model_keys in models_by_split.items():
        if not data_interface.check_train_test_split_exists(split_key):
            logging.warning(f"Split `{split_key}` is not cached; skipping its models.")
            failed.append(split_key)
            continue
        try:
            train_df, test_df = data_interface.load_train_test_split(split_key)
            for refit in (False, True):
                get_or_fit_domain(
                    domain, data_interface, split_key, refit, train_df, test_df
                )
            n_fitted += 1
        except Exception as exc:
            logging.error(f"Failed to fit the domain of split `{split_key}`: {exc}")
            failed.append(split_key)
            continue

        if args.no_calibrate:
            continue
        for model_key in model_keys:
            if not data_interface.check_model_exists(model_key, split_key):
                continue
            exists = data_interface.load_domain_calibration(
                model_key, split_key, domain.get_cache_key()
            )
            if exists and not args.override_cache:
                continue
            logging.info(f"Calibrating `{model_key}/{split_key}`")
            try:
                fit_domain_for_model(
                    domain, data_interface, model_key, split_key, train_df, test_df
                )
                n_calibrated += 1
            except Exception as exc:
                logging.error(f"Failed to calibrate `{model_key}/{split_key}`: {exc}")
                failed.append(f"{model_key}/{split_key}")

    time_elapsed = time.time() - time_start
    logging.info(
        f"Fitted domains of {n_fitted} splits and calibrated {n_calibrated} models "
        f"in {round(time_elapsed / 60, 2)} minutes."
    )
    if failed:
        logging.warning(f"{len(failed)} failures: {failed[:20]}")
