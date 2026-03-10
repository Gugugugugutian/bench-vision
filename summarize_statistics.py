import os
import glob
import pandas as pd

def main():
    csv_files = glob.glob(os.path.join("./statistics/*", "*.csv"))
    
    if not csv_files:
        print("No CSV files found in ./statistics directory.")
        return
    
    dataframes = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # df["source_file"] = os.path.basename(file)
            assert len(df) == 1, f"Expected 1 row in {file}, but found {len(df)}"
            dataframes.append(df)
            # print(f"Read {file} ({len(df)} rows)")
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not dataframes:
        print("No dataframes to combine.")
        return
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    output_path = "./statistics/bench_vision_summary.csv"
    combined_df.to_csv(output_path, index=False)

    print(f"Saved {len(combined_df)} statistics into file: {output_path}")

if __name__ == "__main__":
    main()