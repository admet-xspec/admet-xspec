from chembl_webresource_client.new_client import new_client
import pandas as pd

df = pd.read_csv("CNS_target_list.csv", sep=";")

for target_id in df["chembl_id"]:
    try:
        activity = new_client.activity
        res = activity.filter(target_chembl_id=target_id).only(
            [
                "canonical_smiles",
                "standard_value",
                "standard_units",
                "standard_relation",
                "assay_type",
            ]
        )
        res_df = (
            pd.DataFrame(res)[
                ["canonical_smiles", "relation", "value", "units", "type"]
            ]
            if res
            else pd.DataFrame()
        )
        res_df.drop_duplicates(subset="canonical_smiles", inplace=True)
        print(f"{target_id}: {len(res_df)} compounds")
        res_df.to_csv(f"datasets/chembl_{target_id}.csv", index=False)
    except Exception as e:
        print(f"Error for {target_id}: {e}")
