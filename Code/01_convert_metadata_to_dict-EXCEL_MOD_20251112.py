# generate trial dictionary.csv file from the meta_data_summary.xls
# Import necessary libraries
import pandas as pd
import os

# Step 1: Prompt the user to enter the file path and trial name
# enter Trial_v4_SDTM_LibraryInfo.xlsx
file_path = input("Please enter the path to the meta_data_summary.xlsx file: ")
trial_name = input("Please enter the trial name (e.g., FLINT2, CYNCH, PIVENS): ").upper()

# Print current working directory to help with troubleshooting
print(f"Current working directory: {os.getcwd()}")

# Step 2: Check if the file exists
if not os.path.exists(file_path):
    print(f"Error: The file {file_path} does not exist.")
    exit()

# Load the Excel file
excel_data = pd.ExcelFile(file_path)

# Initialize an empty DataFrame for the combined result
combined_df = pd.DataFrame()

# Loop through all sheets except the first one
for sheet in excel_data.sheet_names[0:]:
    # Load the sheet data, specifying that the header is in row 3 (index 2)
    df = pd.read_excel(file_path, sheet_name=sheet, header=2)

    # Normalize column names to uppercase for consistent comparison
    df.columns = df.columns.str.upper()

    # Add the sheet name as a new column for reference
    df['Source_Tab'] = sheet

    # Append to the combined DataFrame
    combined_df = pd.concat([combined_df, df], ignore_index=True)

# Step 3: Create the 'dir' directory if it doesn't exist
output_dir = 'dict'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Directory '{output_dir}' created.")

# Step 4: Save the combined DataFrame to a new CSV file in the 'dir' directory
output_file_path = os.path.join(output_dir, f"{trial_name}_dictionary.csv")
combined_df.to_csv(output_file_path, index=False)
output_file_path_excel = os.path.join(output_dir, f"{trial_name}_dictionary.xlsx")
combined_df.to_excel(output_file_path_excel, index=False)

# Print success message with the file path
print(f"Combined CSV data has been saved to {output_file_path}")
print(f"Combined Excel data has been saved to {output_file_path_excel}")
