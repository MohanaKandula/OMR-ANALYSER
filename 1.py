import pandas as pd

# Load Excel file
file_path = "Key (Set A and B).xlsx"

# Load sheets
set_a_df = pd.read_excel("Key (Set A and B).xlsx", sheet_name="Set - A")
set_b_df = pd.read_excel("Key (Set A and B).xlsx", sheet_name="Set - B")

# Clean column names (remove spaces, lowercase)
set_a_df.columns = set_a_df.columns.str.strip().str.lower()
set_b_df.columns = set_b_df.columns.str.strip().str.lower()

# Standardize "satistics" typo
set_a_df = set_a_df.rename(columns={"satistics": "statistics"})

# Function to extract just the answer letter
def extract_answers(series):
    return series.dropna().astype(str).str.split("-").str[-1].str.strip().str.upper().tolist()

# Build dictionary
set_a_dict = {
    "subject_1": extract_answers(set_a_df["python"]),
    "subject_2": extract_answers(set_a_df["eda"]),
    "subject_3": extract_answers(set_a_df["sql"]),
    "subject_4": extract_answers(set_a_df["power bi"]),
    "subject_5": extract_answers(set_a_df["statistics"]),
}

set_b_dict = {
    "subject_1": extract_answers(set_b_df["python"]),
    "subject_2": extract_answers(set_b_df["eda"]),
    "subject_3": extract_answers(set_b_df["sql"]),
    "subject_4": extract_answers(set_b_df["power bi"]),
    "subject_5": extract_answers(set_b_df["statistics"]),
}

final_dict = {"A": set_a_dict, "B": set_b_dict}

print(final_dict)
