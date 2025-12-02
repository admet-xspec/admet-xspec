# Find every 'prepared_data.csv' file in the 'data/datasets/' directory and delete it.
import pathlib

data_dir = pathlib.Path("data/datasets/")
prepared_files = list(data_dir.rglob("prepared_data.csv"))
for file_path in prepared_files:
    file_path.unlink()
    print(f"Deleted: {file_path}")
print("Cache clearing complete.")