import os
import glob
import pandas as pd


def merge_summary_results(folder_path, output_file="final_combined_results.csv"):
    """
    Merge all CSV files in a folder into one dataset.
    Replace shortened model names with full names.
    Remove duplicates if any.
    """

    # Full model name mapping
    model_name_map = {
        "KNN": "K-Nearest Neighbors",
        "SVM": "Support Vector Machine",
        "MLP": "Multi-Layer Perceptron",
        "RF": "Random Forest",
        "LR": "Logistic Regression",
        "DT": "Decision Tree",
        "NB": "Naive Bayes",
        "GB": "Gradient Boosting"
    }

    # Get all CSV files
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    if not csv_files:
        print("No CSV files found.")
        return

    all_dataframes = []

    for file in csv_files:
        try:
            df = pd.read_csv(file)

            # Detect model name from filename
            filename = os.path.basename(file).upper()

            for short, full in model_name_map.items():
                if short in filename:
                    df["Model"] = full
                    break

            all_dataframes.append(df)
            print(f"Loaded: {file}")

        except Exception as e:
            print(f"Failed to load {file}: {e}")

    # Merge everything
    final_df = pd.concat(all_dataframes, ignore_index=True)

    # Remove duplicates
    final_df.drop_duplicates(inplace=True)

    # Save final result
    output_path = os.path.join(folder_path, output_file)
    final_df.to_csv(output_path, index=False)

    print("=" * 60)
    print(f"Final merged dataset saved to: {output_path}")
    print(f"Total rows: {len(final_df)}")
    print("=" * 60)

    return final_df


