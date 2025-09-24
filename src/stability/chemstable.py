import pathlib
import pandas as pd
import tempfile
import subprocess


class DeepChemStableWrapper:
    """Wrapper for DeepChemStable to predict chemical stability from SMILES strings."""

    def __init__(self, dcs_path="external/DeepChemStable", conda_env="stable"):
        self.dcs_path = pathlib.Path(dcs_path)
        self.dcs_figures = self.dcs_path / "figures"
        self.dcs_conda_env = conda_env
        self.dcs_output_file = self.dcs_path / "results.csv"
        self._check_dcs_directory()

    def predict(self, smiles_list):

        # Prepare the input file
        with tempfile.NamedTemporaryFile(
            dir=self.dcs_path, delete=True, suffix=".csv"
        ) as temp_input_file:
            self._prepare_dcs_input(smiles_list, input_file_path=temp_input_file.name)

            # Delete any old output files
            if self.dcs_output_file.exists():
                self.dcs_output_file.unlink()

            # Run DeepChemStable
            command = f"""cd {self.dcs_path} && conda run -n {self.dcs_conda_env} python predict.py {temp_input_file.name} {len(smiles_list)}"""
            with subprocess.Popen(
                command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            ) as process:
                stdout, stderr = process.communicate()
                if process.returncode != 0:
                    raise RuntimeError(
                        f"DeepChemStable prediction failed with error: {stderr.decode('utf-8')}"
                    )

            # Check if the output file exists
            if not self.dcs_output_file.exists():
                raise FileNotFoundError(
                    f"DeepChemStable output file not found at {self.dcs_output_file}. "
                    "Ensure that the DeepChemStable script ran successfully."
                )

            # Parse the output
            output = self._parse_dcs_output(self.dcs_output_file)
            # Add the original SMILES to the output
            output["smiles"] = smiles_list
        return output

    def _prepare_dcs_input(self, smiles_list, input_file_path=None):
        """Prepare input for DeepChemStable and return the path to the temporary input file."""
        # Convert a list of SMILES to the format expected by DeepChemStable
        df = pd.DataFrame({"smiles": smiles_list})
        df["substance_id"] = df.index
        df["label"] = 1
        df.to_csv(input_file_path, index=False, header=True)

    def _check_dcs_directory(self):
        """Check if the DeepChemStable directory exists."""
        if not self.dcs_path.exists():
            raise FileNotFoundError(
                f"""DeepChemStable directory not found at {self.dcs_path}. Please set up 
            DeepChemStable according to the instructions provided in the README."""
            )
        # Ensure the cache directory exists
        self.dcs_figures.mkdir(parents=True, exist_ok=True)

    def _parse_dcs_output(self, output_file_path):
        """Parse the output from DeepChemStable.
        Returns a DataFrame with columns: 'smiles', 'probability', 'label'."""
        df = pd.read_csv(output_file_path)
        df["Probability"] = df["Probability"].apply(lambda x: 1 - x)
        df["Label"] = df["Label"].apply(lambda x: 1 if x == "Stable" else 0)
        df = df.rename(
            columns={"Probability": "class_probability", "Label": "is_stable"}
        )
        return df[["class_probability", "is_stable"]]
