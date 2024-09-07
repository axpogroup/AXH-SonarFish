from pathlib import Path

import pandas as pd


def save_classified_trajectories(
    dump_df_list: list[pd.DataFrame],
    measurement_csv_paths: list[str],
    save_dir: str,
    name_extension: str = "_classified",
    drop_columns: list[str] = ["image_tile", "binary_image"],
) -> None:
    """
    Dump a list of dataframes to CSV files in the specified directory.
    Names of the CSV files are based on the input measurement CSV files.
    """
    
    if len(dump_df_list) != len(measurement_csv_paths):
        raise ValueError("The number of video IDs does not match the number of file paths.")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for df, measurement_csv_path in zip(dump_df_list, measurement_csv_paths):
        base_filename_without_ext = Path(measurement_csv_path).stem

        output_filename = f"{base_filename_without_ext}{name_extension}.csv"
        output_filepath = Path(save_dir) / output_filename

        df.drop(columns=drop_columns, inplace=True, errors="ignore")
        df.to_csv(output_filepath, index=False)
        print(f"Saved {output_filepath}")
