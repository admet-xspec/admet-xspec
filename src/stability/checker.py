"""Chemical stability triage for small molecules, from RDKit's BRENK catalog.

Caveat: BRENK is a lead-unsuitability filter, not a stability filter. Common
liabilities (alkyl esters, beta-lactams, sulfonyl fluorides, vinyl sulfones) pass
it untouched; see the README limitations before trusting a clean result.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import logging
import pickle
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from ._clean import CleanFn, resolve_cleaner
from .catalog import CatalogSchema, build_catalog, build_schema
from .config import AXES, load_config
from .scoring import Scorer

logger = logging.getLogger(__name__)

IDENTITY_COLUMNS = ("input_smiles", "clean_smiles")
STATUS_COLUMNS = ("valid",)
SUMMARY_COLUMNS = (
    "n_alerts",
    "score_electrophile",
    "score_hydrolytic",
    "score_redox_thermal",
    "score_total",
    "tier",
    "alerts",
)

_ALERT_DTYPE = "Int8"
_SCORE_DTYPE = "Float32"
_COUNT_DTYPE = "Int16"

# Worker-process globals, set by `_init_worker`.
_WORKER_CATALOG = None
_WORKER_CLEANER = None
_WORKER_REMOVE_SALT = True


@contextlib.contextmanager
def _rdkit_quiet(enabled: bool):
    """Silence RDKit C++ parse logging if `enabled`; `valid` still records failures."""
    if not enabled:
        yield
        return
    RDLogger.DisableLog("rdApp.*")
    try:
        yield
    finally:
        RDLogger.EnableLog("rdApp.*")


class BrenkChecker:
    """Check chemical stability of SMILES against RDKit's BRENK alert catalog.

    SMILES are cleaned with `src.utils.get_clean_smiles`. Rows it rejects get
    `valid=False` and `pd.NA` in every alert, score and summary column, so a 0 in
    an alert column always means "examined, not matched".

    Args:
        smiles_col: Input column holding SMILES.
        id_col: Optional identifier column, copied to the output.
        remove_salt: Passed to `get_clean_smiles`.
        config_path: Optional YAML/JSON overriding severities, aggregation or tiers.
        prefix: Prefix of the binary alert columns.
        n_jobs: Worker processes; used only above `parallel_threshold` unique
            SMILES, since pool overhead dominates on small inputs.
        parallel_threshold: Minimum number of unique SMILES to use a pool.
        clean_fn: Cleaner override, mainly for testing.
        strict_config: Raise, rather than warn, on config keys absent from BRENK.
        quiet_rdkit: Suppress RDKit parse logging while assessing.

    Example:
        >>> checker = BrenkChecker(smiles_col="smiles", id_col="id")
        >>> result = checker.assess(df)
        >>> combined = df.join(result.drop(columns=["input_smiles"]))
    """

    def __init__(
        self,
        smiles_col: str = "smiles",
        id_col: Optional[str] = None,
        remove_salt: bool = True,
        config_path: Optional[str | Path] = None,
        prefix: str = "brenk_",
        n_jobs: int = 1,
        parallel_threshold: int = 20_000,
        clean_fn: Optional[CleanFn] = None,
        strict_config: bool = True,
        quiet_rdkit: bool = False,
    ):
        self.smiles_col = smiles_col
        self.id_col = id_col
        self.remove_salt = remove_salt
        self.prefix = prefix
        self.n_jobs = max(1, int(n_jobs))
        self.parallel_threshold = int(parallel_threshold)
        self.quiet_rdkit = bool(quiet_rdkit)

        self._clean_fn = resolve_cleaner(clean_fn)
        self.schema: CatalogSchema = build_schema(prefix)
        self._catalog = build_catalog()
        self.config = load_config(config_path)
        self.scorer = Scorer(self.config)

        unknown = self.scorer.validate_against(self.schema.descriptions)
        if unknown:
            message = (
                f"Config references {len(unknown)} entries absent from the BRENK "
                f"catalog, which will have no effect: {unknown}"
            )
            if strict_config:
                raise ValueError(message)
            logger.warning(message)

        missing = self.scorer.unweighted(self.schema.descriptions)
        if missing:
            logger.info(
                "%d BRENK entries have no severity configured and score 0: %s",
                len(missing),
                missing,
            )

    @property
    def alert_columns(self) -> List[str]:
        """Return all binary alert columns, known before any data is read."""
        return list(self.schema.columns)

    @property
    def output_columns(self) -> List[str]:
        """Return every column `assess` produces, in output order."""
        columns = []
        if self.id_col is not None:
            columns.append(self.id_col)
        columns.extend(IDENTITY_COLUMNS)
        columns.extend(STATUS_COLUMNS)
        columns.extend(SUMMARY_COLUMNS)
        columns.extend(self.schema.columns)
        return columns

    @property
    def dtypes(self) -> dict:
        """Return the output dtype map, stored in the sidecar for CSV round-trips."""
        mapping = {
            "input_smiles": "object",
            "clean_smiles": "object",
            "valid": "boolean",
            "n_alerts": _COUNT_DTYPE,
            "tier": "object",
            "alerts": "object",
        }
        for axis in AXES:
            mapping[f"score_{axis}"] = _SCORE_DTYPE
        mapping["score_total"] = _SCORE_DTYPE
        for column in self.schema.columns:
            mapping[column] = _ALERT_DTYPE
        return mapping

    def assess(
        self,
        df: pd.DataFrame,
        include_input: bool = False,
    ) -> pd.DataFrame:
        """Assess every row of `df`, returning a frame on the same index.

        The result holds generated columns plus `id_col`, or with `include_input`,
        the input columns followed by generated ones. `df` is never mutated.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"expected a DataFrame, got {type(df).__name__}")
        if self.smiles_col not in df.columns:
            raise KeyError(
                f"SMILES column {self.smiles_col!r} not in frame. "
                f"Available: {list(df.columns)}"
            )
        if self.id_col is not None and self.id_col not in df.columns:
            raise KeyError(f"id column {self.id_col!r} not in frame")

        generated = [c for c in self.output_columns if c != self.id_col]
        collisions = sorted(set(generated) & set(df.columns))
        if collisions:
            raise ValueError(
                "Input frame already contains columns this checker would "
                f"generate: {collisions}. Rename them or change `prefix`."
            )

        result = self._build_frame(df)
        if include_input:
            id_cols = [self.id_col] if self.id_col else []
            return pd.concat([df, result.drop(columns=id_cols)], axis=1)
        return result

    def assess_smiles(self, smiles: Iterable[str]) -> pd.DataFrame:
        """Assess a bare sequence of SMILES."""
        frame = pd.DataFrame({self.smiles_col: list(smiles)})
        checker = self._without_id() if self.id_col is not None else self
        return checker.assess(frame)

    def assess_to_csv(
        self,
        df: pd.DataFrame,
        path: str | Path,
        write_metadata: bool = True,
        **to_csv_kwargs,
    ) -> Path:
        """Assess `df` and write a CSV, plus a `.meta.json` sidecar."""
        from .io import write_metadata as _write_metadata

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        result = self.assess(df)
        to_csv_kwargs.setdefault("index", False)
        result.to_csv(path, **to_csv_kwargs)
        if write_metadata:
            _write_metadata(self, result, path)
        return path

    def _without_id(self) -> "BrenkChecker":
        """Return a shallow copy with `id_col` unset."""
        clone = object.__new__(BrenkChecker)
        clone.__dict__.update(self.__dict__)
        clone.id_col = None
        return clone

    def _build_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assess unique SMILES once, then expand results back over all rows."""
        index = df.index
        raw = df[self.smiles_col]
        n_rows = len(df)
        n_alerts_total = len(self.schema)

        # Non-strings (NaN, None, numbers) are invalid by contract.
        is_str = raw.map(lambda v: isinstance(v, str)).to_numpy(dtype=bool)

        if n_rows == 0:
            unique_smiles: np.ndarray = np.array([], dtype=object)
            codes = np.array([], dtype=np.int64)
        else:
            # Screening files often carry 10-30% duplicates.
            categorical = pd.Categorical(raw.where(pd.Series(is_str, index=index)))
            unique_smiles = np.asarray(categorical.categories, dtype=object)
            codes = np.asarray(categorical.codes, dtype=np.int64)

        clean, matrix = self._assess_unique(unique_smiles)
        valid_unique = np.array([c is not None for c in clean], dtype=bool)

        # code == -1 marks a non-string.
        row_valid = np.zeros(n_rows, dtype=bool)
        clean_out = np.full(n_rows, None, dtype=object)
        alert_matrix = np.zeros((n_rows, n_alerts_total), dtype=np.int8)

        has_code = codes >= 0
        if has_code.any():
            picked = codes[has_code]
            row_valid[has_code] = valid_unique[picked]
            clean_out[has_code] = np.asarray(clean, dtype=object)[picked]
            alert_matrix[has_code] = matrix[picked]

        frame = pd.DataFrame(index=index)
        if self.id_col is not None:
            frame[self.id_col] = df[self.id_col].to_numpy()
        frame["input_smiles"] = raw.to_numpy()
        frame["clean_smiles"] = clean_out
        frame["valid"] = pd.array(row_valid, dtype="boolean")

        alerts = pd.DataFrame(alert_matrix, columns=self.schema.columns, index=index)
        descriptions = np.asarray(self.schema.descriptions, dtype=object)

        n_hits = alert_matrix.sum(axis=1)
        row_hits = [
            list(descriptions[alert_matrix[i].astype(bool)]) if row_valid[i] else None
            for i in range(n_rows)
        ]
        hit_names = [";".join(h) if h is not None else None for h in row_hits]
        score_rows = [self.scorer.score(h) if h is not None else None for h in row_hits]

        frame["n_alerts"] = pd.array(
            [int(n) if v else None for n, v in zip(n_hits, row_valid)],
            dtype=_COUNT_DTYPE,
        )
        for axis in AXES:
            frame[f"score_{axis}"] = pd.array(
                [s[axis] if s is not None else None for s in score_rows],
                dtype=_SCORE_DTYPE,
            )
        frame["score_total"] = pd.array(
            [s["total"] if s is not None else None for s in score_rows],
            dtype=_SCORE_DTYPE,
        )
        frame["tier"] = [
            self.scorer.tier(s["total"]) if s is not None else None for s in score_rows
        ]
        frame["alerts"] = hit_names

        alerts = alerts.astype(_ALERT_DTYPE)
        if (~row_valid).any():
            # Unexamined molecules get NA, not 0.
            alerts.loc[~row_valid, :] = pd.NA

        return pd.concat([frame, alerts], axis=1)[self.output_columns]

    def _assess_unique(self, unique_smiles: np.ndarray):
        """Clean and match each unique SMILES; return cleaned SMILES and hit matrix."""
        n_unique = len(unique_smiles)
        matrix = np.zeros((n_unique, len(self.schema)), dtype=np.int8)
        if n_unique == 0:
            return [], matrix

        use_pool = self.n_jobs > 1 and n_unique >= self.parallel_threshold
        if use_pool and not _is_picklable(self._clean_fn):
            logger.warning("clean_fn is not picklable; assessing serially")
            use_pool = False
        with _rdkit_quiet(self.quiet_rdkit):
            if use_pool:
                results = self._assess_parallel(unique_smiles)
            else:
                results = [
                    _assess_one(s, self._clean_fn, self._catalog, self.remove_salt)
                    for s in unique_smiles
                ]

        column_of = self.schema.description_to_column
        position_of = {c: i for i, c in enumerate(self.schema.columns)}
        clean = []
        for row, (clean_smi, hits) in enumerate(results):
            clean.append(clean_smi)
            for description in hits:
                column = column_of.get(description)
                if column is None:  # pragma: no cover - schema guarantees presence
                    logger.warning("Unmapped BRENK description %r", description)
                    continue
                matrix[row, position_of[column]] = 1
        return clean, matrix

    def _assess_parallel(self, unique_smiles: np.ndarray):
        """Run `_assess_one` over a spawn-based process pool."""
        import multiprocessing as mp

        chunksize = max(1, len(unique_smiles) // (self.n_jobs * 8))
        with mp.get_context("spawn").Pool(
            processes=self.n_jobs,
            initializer=_init_worker,
            initargs=(self._clean_fn, self.remove_salt, self.quiet_rdkit),
        ) as pool:
            return pool.map(
                _assess_one_worker, list(unique_smiles), chunksize=chunksize
            )


def _assess_one(smiles, clean_fn, catalog, remove_salt):
    """Clean one SMILES and return `(clean_smiles, hits)`; `(None, [])` if rejected."""
    try:
        if _accepts_remove_salt(clean_fn):
            clean = clean_fn(smiles, remove_salt=remove_salt)
        else:
            clean = clean_fn(smiles)
    except Exception:
        logger.warning("cleaner raised for %r", smiles, exc_info=True)
        return None, []

    if clean is None:
        return None, []

    mol = Chem.MolFromSmiles(clean)
    if mol is None or mol.GetNumAtoms() == 0:
        # Unreachable per the cleaner contract; an empty mol must never score clean.
        logger.warning("cleaned SMILES %r did not re-parse to a molecule", clean)
        return None, []

    hits = [entry.GetDescription() for entry in catalog.GetMatches(mol)]
    return clean, hits


@functools.lru_cache(maxsize=None)
def _accepts_remove_salt(clean_fn: CleanFn) -> bool:
    """Return whether `clean_fn` takes a `remove_salt` keyword."""
    try:
        params = inspect.signature(clean_fn).parameters.values()
    except (TypeError, ValueError):  # builtins without a signature
        return False
    return any(
        p.name == "remove_salt" or p.kind is inspect.Parameter.VAR_KEYWORD
        for p in params
    )


def _is_picklable(obj) -> bool:
    """Return whether `obj` can be sent to a spawned worker."""
    try:
        pickle.dumps(obj)
    except Exception:
        return False
    return True


def _init_worker(clean_fn: CleanFn, remove_salt: bool, quiet: bool):  # pragma: no cover
    """Set per-process catalog and cleaner."""
    global _WORKER_CATALOG, _WORKER_CLEANER, _WORKER_REMOVE_SALT
    _WORKER_CATALOG = build_catalog()
    _WORKER_CLEANER = clean_fn
    _WORKER_REMOVE_SALT = remove_salt
    if quiet:
        RDLogger.DisableLog("rdApp.*")


def _assess_one_worker(smiles):  # pragma: no cover - subprocess
    return _assess_one(smiles, _WORKER_CLEANER, _WORKER_CATALOG, _WORKER_REMOVE_SALT)
