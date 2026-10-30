# The structural alerts, one by one

Every molecule is matched against the 105 entries of RDKit's BRENK catalog. This document says what each entry matches and why it does or does not matter under assay conditions: aqueous buffer at pH 7.4 and 37 °C, hours to days, glutathione in cells, serum hydrolases in plasma media, DMSO stocks under freeze-thaw.

That context sets the weights. Each alert carries a severity of 1–3 on up to three axes: **E** electrophilic (reacts with thiols and amines), **H** hydrolytic (cleaved by water or esterases), **R** redox (autoxidation, redox cycling, photolability). Alerts marked "unscored" are reported but score zero — lead-likeness or toxicophore flags with no bearing on stability in buffer. Weights live in `config.py`; the behaviour below was verified against this RDKit build, not read off the entry names.

The relevance column always reads in the same order: what water does to it, what it does to protein, which side chains it goes for, and what else in the medium it reacts with.

## Severe electrophiles: acylating and alkylating

| Alert | Score | Matches | Stability relevance |
| --- | --- | --- | --- |
| acid_halide | E3 H3 | Acyl chloride or bromide | Hydrolysed by water on contact; acylates protein; mainly lysine amines; also reacts with glutathione |
| acyl_cyanide | E3 H3 | Acyl group bonded to a nitrile | Hydrolysed by water on contact; acylates protein; mainly lysine amines; also releases cyanide into medium |
| sulfonyl_cyanide | E3 H3 | Sulfonyl group bonded to a nitrile | Hydrolysed by water on contact; sulfonylates protein; mainly lysine amines and serine hydroxyls; also reacts with glutathione |
| isocyanate | E3 H3 | N=C=O | Hydrolysed by water on contact; carbamoylates protein; mainly lysine amines; also reacts with glutathione |
| ketene | E3 H3 | C=C=O | Hydrolysed by water on contact; acylates protein; mainly lysine amines; also reacts with glutathione |
| triflate | E3 H3 | Trifluoromethanesulfonate ester | Hydrolysed by water within hours; alkylates protein; mainly cysteine thiols; also reacts with glutathione |
| silicon_halogen | E3 H3 | Si–Cl or Si–Br | Hydrolysed by water on contact; silylates protein; mainly serine hydroxyls; hydrolysis usually wins in medium |
| ester_of_HOBT | E3 H3 | Benzotriazolyl ester | Hydrolysed by water within hours; acylates protein; mainly lysine amines; also reacts with glutathione |
| four_member_lactones | E3 H3 | β-lactone | Hydrolysed by water within hours; acylates protein; mainly cysteine thiols and serine hydroxyls; also reacts with glutathione |
| thioester | E3 H3 | C(=O)S | Hydrolysed by water slowly; acylates protein; mainly cysteine thiols and lysine amines; also exchanges with glutathione |
| beta-keto/anhydride | E3 H3 | Anhydride or β-keto ester | Hydrolysed by water on contact; acylates protein; mainly lysine amines; also reacts with glutathione |
| Carbo_cation/anion | E3 H3 | A formally charged carbon | Quenched by water on contact; alkylates protein; any nucleophilic side chain; also reacts with glutathione |
| alkyl_halide | E3 | sp³ carbon bearing Cl, Br or I. Aryl halides and CF₃ groups do not match | Stable to water; alkylates protein; mainly cysteine thiols; also reacts with glutathione |
| N-halo | E3 R3 | N–Cl or N–Br | Hydrolysed by water within hours; oxidises and halogenates protein; mainly cysteine thiols and methionine; also consumed by glutathione |
| N-C-halo | E3 | Halogen on a carbon adjacent to nitrogen | Stable to water; alkylates protein; mainly cysteine thiols; also reacts with glutathione |
| Three-membered_heterocycle | E3 | Epoxide, aziridine or thiirane | Stable to water near pH 7.4; alkylates protein; mainly cysteine thiols; also reacts with glutathione |
| cyanate_/aminonitrile_/thiocyanate | E3 H2 | O–C≡N, N–C≡N or S–C≡N | Hydrolysed by water within hours; carbamoylates protein; mainly lysine amines; also releases cyanide into medium |

## Michael acceptors

All five score E3, since glutathione adduction dominates in cells. The catalog cannot rank within this class: intrinsic reactivity across its members spans orders of magnitude.

| Alert | Score | Matches | Stability relevance |
| --- | --- | --- | --- |
| Michael_acceptor_1 | E3 | Enones, acrylates, acrylamides, fumarates, cinnamates | Stable to water; adds covalently to protein; mainly cysteine thiols; consumed by glutathione in cells |
| Michael_acceptor_2 | E3 | Ynones | Stable to water; adds covalently to protein; mainly cysteine thiols; consumed by glutathione in cells |
| Michael_acceptor_3 | E3 | Vinyl-activated heteroatom systems | Stable to water; adds covalently to protein; mainly cysteine thiols; consumed by glutathione in cells |
| Michael_acceptor_4 | E3 | Further conjugated acceptor patterns | Stable to water; adds covalently to protein; mainly cysteine thiols; consumed by glutathione in cells |
| Michael_acceptor_5 | E3 | Propiolates | Stable to water; adds covalently to protein; mainly cysteine thiols; consumed by glutathione in cells |

## Moderate electrophiles

| Alert | Score | Matches | Stability relevance |
| --- | --- | --- | --- |
| aldehyde | E2 | CHO, and formamides too | Hydrated by water reversibly; forms reversible adducts with protein; mainly lysine amines and cysteine thiols; also reacts with glutathione |
| 2-halo_pyridine | E2 | Halogen at pyridine C2 | Stable to water; arylates protein; mainly cysteine thiols; also reacts with glutathione |
| conjugated_nitrile_group | E2 | C=C–C≡N | Stable to water; adds covalently to protein; mainly cysteine thiols; consumed by glutathione in cells |
| cyanamide | E2 | N–C≡N on an amine | Hydrolysed by water slowly; adds covalently to protein; mainly cysteine thiols; also reacts with glutathione |
| sulphate | E2 H2 | Sulfate ester | Hydrolysed by water within hours; alkylates protein; mainly cysteine thiols; also reacts with glutathione |
| phthalimide | E2 | Phthalimides, and also maleimides and NHS esters | Hydrolysed by water slowly; adds covalently to or acylates protein; mainly cysteine thiols and lysine amines; consumed by glutathione in cells |
| phenol_ester | E2 H3 | Aryl ester | Hydrolysed by water and by serum esterases; acylates protein; mainly lysine amines; also reacts with glutathione |
| phenyl_carbonate | E2 H3 | Aryl carbonate | Hydrolysed by water and by serum esterases; acylates protein; mainly lysine amines; also reacts with glutathione |
| imine_1 | E2 H3 | C=N, including amidines, guanidines and biguanides | Hydrolysed by water within hours; forms reversible adducts with protein; mainly lysine amines; conjugated amidines do neither, so metformin is a false positive |
| imine_2 | E2 H3 | Aryl-substituted C=N | Hydrolysed by water within hours; forms reversible adducts with protein; mainly lysine amines; same caveat as imine_1 |
| cyanohydrins | E1 H2 | HO–C–C≡N | Hydrolysed by water within hours; little direct reaction with protein; releases cyanide into medium |
| diketo_group | E1 | 1,2-diketone | Hydrated by water reversibly; forms reversible adducts with protein; mainly lysine amines and cysteine thiols; also reacts with glutathione |
| halogenated_ring_1 | E1 | Polyhalogenated arene | Stable to water; arylates protein slowly; mainly cysteine thiols; also reacts with glutathione |
| halogenated_ring_2 | E1 | An alternative polyhalogenation pattern | Stable to water; arylates protein slowly; mainly cysteine thiols; also reacts with glutathione |
| Thiocarbonyl_group | E1 R1 | C=S, as in thioamides and thioureas | Stable to water; binds protein weakly; mainly cysteine thiols; also consumed by dissolved oxygen |

## Hydrolytically labile

| Alert | Score | Matches | Stability relevance |
| --- | --- | --- | --- |
| enamine | H3 | N–C=C; also fires on cyanamides and ynamines | Hydrolysed by water within hours; no covalent reaction with protein; releases the carbonyl and amine into medium |
| acyclic_C=C-O | H2 | Vinyl ether outside a ring | Hydrolysed by water, faster at acidic pH; no covalent reaction with protein; releases the carbonyl into medium |
| het-C-het_not_in_ring | H2 | One carbon bearing two heteroatoms outside a ring: acetal, ketal, aminal | Hydrolysed by water, faster at acidic pH; no covalent reaction with protein; a lysosomal concern more than a pH 7.4 one |
| >_2_ester_groups | H2 | Three or more esters | Hydrolysed by water and by serum esterases; no covalent reaction with protein; releases the acids and alcohols into medium |
| cumarine | H1 | Coumarin | Hydrolysed by water slowly; no covalent reaction with protein; ring-opens to the hydroxy acid in medium |
| phosphor | H1 | Any phosphorus | Hydrolysed by water slowly and by serum phosphatases; no covalent reaction with protein; releases the alcohol into medium |
| triphenyl_methyl-silyl | H1 | Trityl or bulky silyl groups | Hydrolysed by water, faster at acidic pH; no covalent reaction with protein; releases the free alcohol or thiol into medium |
| oxime_1 | H1 | Aldoxime or ketoxime | Hydrolysed by water slowly; no covalent reaction with protein; releases the carbonyl into medium |
| oxime_2 | H1 | Aryl oxime | Hydrolysed by water slowly; no covalent reaction with protein; releases the carbonyl into medium |
| acyl_hydrazine | H1 R1 | C(=O)N–N | Hydrolysed by water slowly; no covalent reaction with protein; also consumed by dissolved oxygen |
| hydroxamic_acid | H1 R1 | C(=O)N–OH | Hydrolysed by water slowly; no covalent reaction with protein; also chelates metal ions in medium |

## Redox-active and photolabile

| Alert | Score | Matches | Stability relevance |
| --- | --- | --- | --- |
| peroxide | R3 | O–O | Stable to water; oxidises protein; mainly cysteine thiols and methionine; also consumed by glutathione |
| catechol | R3 | 1,2-dihydroxybenzene | Stable to water; oxidises protein indirectly; mainly cysteine thiols; consumed by dissolved oxygen, generating hydrogen peroxide in medium |
| hydroquinone | R3 | 1,4-dihydroxybenzene, and para-aminophenols, so paracetamol flags | Stable to water; oxidises protein indirectly; mainly cysteine thiols; consumed by dissolved oxygen, generating hydrogen peroxide in medium |
| chinone_1 | E3 R3 | para-quinone | Stable to water; adds covalently to and oxidises protein; mainly cysteine thiols; consumed by glutathione in cells |
| chinone_2 | E3 R3 | ortho-quinone | Stable to water; adds covalently to and oxidises protein; mainly cysteine thiols; consumed by glutathione in cells |
| disulphide | R2 | S–S | Stable to water; exchanges with protein disulfides; mainly cysteine thiols; also exchanges with glutathione |
| thiol_1 | R2 | Thiolate | Stable to water; forms mixed disulfides with protein; mainly cysteine thiols; consumed by dissolved oxygen |
| thiol_2 | R2 | Free S–H | Stable to water; forms mixed disulfides with protein; mainly cysteine thiols; consumed by dissolved oxygen |
| hydrazine | R2 | N–N | Stable to water; little direct reaction with protein; consumed by dissolved oxygen |
| N-nitroso | R2 | N–N=O | Stable to water; little direct reaction with protein; degraded by light and reduced in cells |
| N-hydroxyl_pyridine | R2 | N-hydroxy pyridine or pyridone | Stable to water; little direct reaction with protein; consumed by dissolved oxygen and chelates metal ions |
| Azido_group | R2 | N₃ | Stable to water; little direct reaction with protein; reduced in cells and degraded by light |
| Azo_group | R2 | N=N | Stable to water; little direct reaction with protein; reduced by azoreductases and degraded by light |
| diazo_group | R2 | Diazo compounds, and also organic azides and azo compounds | Stable to water; little direct reaction with protein; reduced in cells and degraded by light |
| diaminobenzene_1 | R2 | ortho-diaminobenzene | Stable to water; oxidises to a quinone-imine that binds protein; mainly cysteine thiols; consumed by dissolved oxygen |
| diaminobenzene_2 | R2 | The meta isomer | Stable to water; oxidises to a quinone-imine that binds protein; mainly cysteine thiols; consumed by dissolved oxygen |
| diaminobenzene_3 | R2 | The para isomer | Stable to water; oxidises to a quinone-imine that binds protein; mainly cysteine thiols; consumed by dissolved oxygen |
| benzidine | R2 | 4,4′-diaminobiphenyl | Stable to water; oxidises to a quinone-imine that binds protein; mainly cysteine thiols; consumed by dissolved oxygen |
| polyene | R2 | Two or more conjugated C=C | Stable to water; little direct reaction with protein; consumed by dissolved oxygen and degraded by light |
| sulfinic_acid | R2 | S(=O)OH | Stable to water; little direct reaction with protein; consumed by dissolved oxygen |
| sulfur_oxygen_single_bond | R2 | Sulfenate S–O | Hydrolysed by water slowly; cleaved by protein thiols; mainly cysteine thiols; also exchanges with glutathione |
| sulphur_nitrogen_single_bond | R2 | Sulfenamide S–N | Hydrolysed by water slowly; cleaved by protein thiols; mainly cysteine thiols; also exchanges with glutathione |
| N-acyl-2-amino-5-mercapto-1,3,4-_thiadiazole | R2 | That scaffold | Stable to water; forms mixed disulfides with protein; mainly cysteine thiols; consumed by dissolved oxygen |
| heavy_metal | R2 | Metals, and also boron, silicon, selenium and arsenic | Hydrated by water reversibly; binds protein reversibly; mainly serine hydroxyls and cysteine thiols; also catalyses oxidation in medium |
| nitro_group | R1 | Ar–NO₂ | Stable to water; no covalent reaction with protein; reduced by nitroreductases in cells |
| Oxygen-nitrogen_single_bond | R1 | Every N–O: nitro groups, oximes, hydrazines, N-oxides, peroxides | Stable to water; no covalent reaction with protein; reduced in cells. Kept at 1 so nitroaromatics are not penalised twice |
| N_oxide | R1 | N→O | Stable to water; no covalent reaction with protein; reduced in cells |
| aniline | R1 | Primary aryl amine only; N-methylaniline does not match | Stable to water; oxidises to species that bind protein; mainly cysteine thiols; consumed by dissolved oxygen |
| iodine | R1 | Alkyl or aryl iodide | Stable to water; no covalent reaction with protein; degraded by light |
| stilbene | R1 | 1,2-diarylethene | Stable to water; no covalent reaction with protein; degraded by light |
| methylidene-1,3-dithiole | R1 | That motif | Stable to water; little direct reaction with protein; consumed by dissolved oxygen |
| Thiobenzothiazole_1 | R1 | 2-substituted benzothiazole-thiol | Stable to water; forms mixed disulfides with protein; mainly cysteine thiols; consumed by dissolved oxygen |
| thiobenzothiazole_2 | R1 | A related pattern | Stable to water; forms mixed disulfides with protein; mainly cysteine thiols; consumed by dissolved oxygen |
| Polycyclic_aromatic_hydrocarbon_1 | R1 | Fused aromatic systems | Stable to water; no covalent reaction with protein; degraded by light and bioactivated by cellular oxidases |
| Polycyclic_aromatic_hydrocarbon_2 | R1 | Anthracene-type systems | Stable to water; no covalent reaction with protein; degraded by light and bioactivated by cellular oxidases |
| Polycyclic_aromatic_hydrocarbon_3 | R1 | Phenanthrene-type systems | Stable to water; no covalent reaction with protein; degraded by light and bioactivated by cellular oxidases |

## Reported but not scored

These are lead-likeness, permeability and toxicophore flags. They appear in the output and in the `alerts` column, but contribute nothing to the score, because they say little about stability in buffer.

| Alert | Score | Matches | Stability relevance |
| --- | --- | --- | --- |
| Sulfonic_acid_1 | unscored | Sulfonate ester, e.g. a mesylate | Hydrolysed by water slowly; alkylates protein; mainly cysteine thiols; also reacts with glutathione. Arguably mis-weighted; see the blind spots below |
| Sulfonic_acid_2 | unscored | Free sulfonic acid | Stable to water; no covalent reaction with protein; stays ionised in medium |
| Aliphatic_long_chain | unscored | Six or more contiguous sp³ carbons, so hexane flags | Stable to water; binds protein non-covalently; mainly hydrophobic pockets; also partitions into membranes |
| Perfluorinated_chain | unscored | CF₂ chains | Stable to water; binds protein non-covalently; mainly hydrophobic pockets; also partitions into membranes |
| quaternary_nitrogen_1 | unscored | Charged nitrogen patterns | Stable to water; no covalent reaction with protein; stays charged in medium |
| quaternary_nitrogen_2 | unscored | Tetraalkylammonium and N-oxide | Stable to water; no covalent reaction with protein; stays charged in medium |
| quaternary_nitrogen_3 | unscored | Charged nitrogen in azides and ylides | Stable to water; no covalent reaction with protein; part of why an azide trips three alerts at once |
| charged_oxygen_or_sulfur_atoms | unscored | Charged O or S, such as a sulfonium | Stable to water; a sulfonium alkylates protein; mainly cysteine thiols; otherwise stays charged in medium |
| isolated_alkene | unscored | Unconjugated C=C | Stable to water; no covalent reaction with protein; slowly consumed by dissolved oxygen |
| triple_bond | unscored | C≡C | Stable to water; no covalent reaction with protein; inert in medium unless conjugated to a carbonyl |
| amidotetrazole | unscored | Acylated tetrazole | Stable to water; no covalent reaction with protein; stays ionised in medium |
| hydantoin | unscored | Imidazolidine-2,4-dione | Stable to water; no covalent reaction with protein; inert in medium |
| biotin_analogue | unscored | Biotin-like bicycle | Stable to water; binds protein non-covalently; mainly the biotin site of avidin and streptavidin; inert in medium |
| saponine_derivative | unscored | Steroidal glycoside | Stable to water; no covalent reaction with protein; lyses membranes in medium |
| crown_ether | unscored | Macrocyclic polyether, though not 18-crown-6 or 12-crown-4 | Stable to water; no covalent reaction with protein; sequesters metal ions in medium |
| cycloheptane_1 | unscored | 7-membered carbocycle | Stable to water; no covalent reaction with protein; inert in medium |
| cycloheptane_2 | unscored | A variant pattern | Stable to water; no covalent reaction with protein; inert in medium |
| cyclooctane_1 | unscored | 8-membered carbocycle | Stable to water; no covalent reaction with protein; inert in medium |
| cyclooctane_2 | unscored | A variant pattern | Stable to water; no covalent reaction with protein; inert in medium |
| azepane | unscored | 7-membered nitrogen heterocycle | Stable to water; no covalent reaction with protein; inert in medium |
| azocane | unscored | 8-membered nitrogen heterocycle | Stable to water; no covalent reaction with protein; inert in medium |

## Names that mislead

| Alert | What it really catches |
| --- | --- |
| Michael_acceptor_1 | The entire class, warheads and inert conjugates alike; it cannot rank within it |
| phthalimide | Maleimides and NHS esters |
| heavy_metal | Boronic acids |
| diazo_group | Organic azides |
| imine_1, imine_2 | Guanidines, so metformin scores as hydrolytically labile despite being indefinitely stable |
| hydroquinone | Para-aminophenols |
| aniline | Primary aryl amines only; it misses every N-alkyl aniline |
| enamine | Cyanamides, while missing simple dialkyl enamines |

## What is not covered

Verified silent on this build: **plain alkyl esters** (ethyl benzoate, and therefore enalapril), **β-lactams** (penicillin G — the catalog has a lactone entry but no lactam one), and **sulfonyl fluorides** and **vinyl sulfones**, both established covalent warheads. Nothing covers carbamates, carbonates, lactones above four atoms, boronic esters, silyl ethers, or anhydrides as a distinct class. Sulfonate esters are matched but score zero.

Esters are the commonest hydrolytic liability in medicinal chemistry and the basis of most ester prodrugs. **A zero score is not a stability claim** — read the structure for esters, carbamates and lactams yourself.

## Reading the output

The score within an axis is the worst hit plus 0.5 per additional distinct hit, capped at 4. It is not a sum, which would rank four mild flags above one acid chloride and would triple-count a benzyl azide, since `Azido_group`, `diazo_group` and `quaternary_nitrogen_3` all fire on the same functional group. Tiers follow the total: `clean` at 0, `watch` below 2, `labile` below 3, `avoid` at 3 and above.

Two molecules with the same alert set receive identical scores. There is no electrophilicity index and no "borderline", and substructure matching is blind to pH, concentration and steric context. Covalent inhibitors are meant to flag: ibrutinib and osimertinib both trip `Michael_acceptor_1`, which is confirmation rather than a defect. Use the score to order a list, and the `alerts` column to decide what to do.
