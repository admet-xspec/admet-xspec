"""Severity weights for BRENK alerts, tuned for assay and physiological conditions.

Target: aqueous buffer at pH 7.0-7.4 and 37 C over hours to days, with glutathione,
serum hydrolases and DMSO stocks under freeze-thaw. Hence e.g. azides weigh
moderate (inert in buffer) while catechols weigh high (autoxidise in media).

Axes:
    electrophile: covalent reactivity toward thiols, amines and nucleophiles.
    hydrolytic: solvolysis in buffer or cleavage by serum hydrolases.
    redox_thermal: autoxidation, redox cycling, photolability, weak-bond homolysis.

Severity is 1 (minor), 2 (material) or 3 (severe); 0 / empty means reported but
not scored. Override via `BrenkChecker(config_path=...)` with a YAML/JSON file of
the same structure, keyed by BRENK description.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

AXES: Tuple[str, ...] = ("electrophile", "hydrolytic", "redox_thermal")

E, H, R = AXES

# description -> list of (axis, severity)
DEFAULT_SEVERITIES: Dict[str, List[Tuple[str, int]]] = {
    # --- severe electrophiles: acylating or alkylating on contact -------------
    "acid_halide": [(E, 3), (H, 3)],
    "sulfonyl_cyanide": [(E, 3), (H, 3)],
    "acyl_cyanide": [(E, 3), (H, 3)],
    "isocyanate": [(E, 3), (H, 3)],
    "ketene": [(E, 3), (H, 3)],
    "triflate": [(E, 3), (H, 3)],
    "silicon_halogen": [(E, 3), (H, 3)],
    "Carbo_cation/anion": [(E, 3), (H, 3)],
    "beta-keto/anhydride": [(E, 3), (H, 3)],
    "ester_of_HOBT": [(E, 3), (H, 3)],
    "four_member_lactones": [(E, 3), (H, 3)],
    "thioester": [(E, 3), (H, 3)],
    "alkyl_halide": [(E, 3)],
    "N-C-halo": [(E, 3)],
    "N-halo": [(E, 3), (R, 3)],
    "Three-membered_heterocycle": [(E, 3)],
    "cyanate_/aminonitrile_/thiocyanate": [(E, 3), (H, 2)],
    # --- Michael acceptors ----------------------------------------------------
    # Michael_acceptor_1 fires on the whole class (acrylamides, cinnamic acids,
    # enones) and cannot rank within it; 3 since GSH adduction dominates in cells.
    "Michael_acceptor_1": [(E, 3)],
    "Michael_acceptor_2": [(E, 3)],
    "Michael_acceptor_3": [(E, 3)],
    "Michael_acceptor_4": [(E, 3)],
    "Michael_acceptor_5": [(E, 3)],
    # --- moderate electrophiles ----------------------------------------------
    "aldehyde": [(E, 2)],
    "2-halo_pyridine": [(E, 2)],
    "conjugated_nitrile_group": [(E, 2)],
    "cyanamide": [(E, 2)],
    "sulphate": [(E, 2), (H, 2)],
    # Also matches reactive maleimides and NHS esters; a false alarm on inert
    # phthalimides is cheaper than a missed warhead.
    "phthalimide": [(E, 2)],
    "phenol_ester": [(E, 2), (H, 3)],
    "phenyl_carbonate": [(E, 2), (H, 3)],
    "imine_1": [(E, 2), (H, 3)],
    "imine_2": [(E, 2), (H, 3)],
    "cyanohydrins": [(E, 1), (H, 2)],
    "diketo_group": [(E, 1)],
    "halogenated_ring_1": [(E, 1)],
    "halogenated_ring_2": [(E, 1)],
    "Thiocarbonyl_group": [(E, 1), (R, 1)],
    # --- hydrolytic ------------------------------------------------------------
    "enamine": [(H, 3)],
    "acyclic_C=C-O": [(H, 2)],
    "het-C-het_not_in_ring": [(H, 2)],
    ">_2_ester_groups": [(H, 2)],
    "cumarine": [(H, 1)],
    "phosphor": [(H, 1)],
    "triphenyl_methyl-silyl": [(H, 1)],
    "oxime_1": [(H, 1)],
    "oxime_2": [(H, 1)],
    "acyl_hydrazine": [(H, 1), (R, 1)],
    "hydroxamic_acid": [(H, 1), (R, 1)],
    # --- redox, autoxidation, photolability ------------------------------------
    "peroxide": [(R, 3)],
    "catechol": [(R, 3)],
    "hydroquinone": [(R, 3)],
    "chinone_1": [(E, 3), (R, 3)],
    "chinone_2": [(E, 3), (R, 3)],
    "disulphide": [(R, 2)],
    "thiol_1": [(R, 2)],
    "thiol_2": [(R, 2)],
    "hydrazine": [(R, 2)],
    "N-nitroso": [(R, 2)],
    "N-hydroxyl_pyridine": [(R, 2)],
    "Azido_group": [(R, 2)],
    "Azo_group": [(R, 2)],
    "diazo_group": [(R, 2)],
    "diaminobenzene_1": [(R, 2)],
    "diaminobenzene_2": [(R, 2)],
    "diaminobenzene_3": [(R, 2)],
    "benzidine": [(R, 2)],
    "polyene": [(R, 2)],
    "sulfinic_acid": [(R, 2)],
    "sulfur_oxygen_single_bond": [(R, 2)],
    "sulphur_nitrogen_single_bond": [(R, 2)],
    "N-acyl-2-amino-5-mercapto-1,3,4-_thiadiazole": [(R, 2)],
    "heavy_metal": [(R, 2)],
    # Fires on every nitro group and peroxide; kept low to avoid double-penalising
    # nitroaromatics, which are stable at pH 7.4.
    "Oxygen-nitrogen_single_bond": [(R, 1)],
    "nitro_group": [(R, 1)],
    "N_oxide": [(R, 1)],
    "aniline": [(R, 1)],
    "iodine": [(R, 1)],
    "stilbene": [(R, 1)],
    "methylidene-1,3-dithiole": [(R, 1)],
    "Thiobenzothiazole_1": [(R, 1)],
    "thiobenzothiazole_2": [(R, 1)],
    "Polycyclic_aromatic_hydrocarbon_1": [(R, 1)],
    "Polycyclic_aromatic_hydrocarbon_2": [(R, 1)],
    "Polycyclic_aromatic_hydrocarbon_3": [(R, 1)],
    # --- reported but not scored: lead-likeness/toxicophore flags -------------
    "Aliphatic_long_chain": [],
    "Perfluorinated_chain": [],
    "Sulfonic_acid_1": [],
    "Sulfonic_acid_2": [],
    "amidotetrazole": [],
    "azepane": [],
    "azocane": [],
    "biotin_analogue": [],
    "charged_oxygen_or_sulfur_atoms": [],
    "crown_ether": [],
    "cycloheptane_1": [],
    "cycloheptane_2": [],
    "cyclooctane_1": [],
    "cyclooctane_2": [],
    "hydantoin": [],
    "isolated_alkene": [],
    "quaternary_nitrogen_1": [],
    "quaternary_nitrogen_2": [],
    "quaternary_nitrogen_3": [],
    "saponine_derivative": [],
    "triple_bond": [],
}

# Score = worst hit + increment per extra hit, capped. A plain sum would rank four
# mild flags above one acyl chloride.
AGGREGATION = {
    "extra_hit_increment": 0.5,
    "axis_cap": 4.0,
    "cross_axis_increment": 0.5,
    "total_cap": 4.0,
}

# (min score_total, tier label)
TIER_THRESHOLDS = [
    (0.0, "clean"),
    (0.01, "watch"),
    (2.0, "labile"),
    (3.0, "avoid"),
]


def load_config(path=None) -> dict:
    """Return the scoring config, optionally overridden from YAML or JSON."""
    config = {
        "severities": {k: [tuple(p) for p in v] for k, v in DEFAULT_SEVERITIES.items()},
        "aggregation": dict(AGGREGATION),
        "tiers": [tuple(t) for t in TIER_THRESHOLDS],
    }
    if path is None:
        return config

    import json
    from pathlib import Path

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        user = yaml.safe_load(text) or {}
    else:
        user = json.loads(text)

    if "severities" in user:
        config["severities"].update(
            {k: [tuple(p) for p in v] for k, v in user["severities"].items()}
        )
    if "aggregation" in user:
        config["aggregation"].update(user["aggregation"])
    if "tiers" in user:
        config["tiers"] = [tuple(t) for t in user["tiers"]]

    bad_axes = {
        axis
        for pairs in config["severities"].values()
        for axis, _ in pairs
        if axis not in AXES
    }
    if bad_axes:
        raise ValueError(f"Unknown axis in config: {sorted(bad_axes)}. Valid: {AXES}")
    return config
