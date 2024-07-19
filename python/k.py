import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np
import os # Import the os module

# Load the Excel file
file_path = '/home/dugong/Desktop/desktop/k/pandas/pandas/python/kcostcodes.ods'  # Replace with the actual path to your file

if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
else:
    # Load the Excel file
    data = pd.read_excel(file_path)

# Iterate through all Excel files in the directory
# Initialize an empty list to store dataframes
dataframes = []
directory = '/home/dugong/Desktop/desktop/k/pandas/pandas/python/' # Replace with your actual directory
for filename in os.listdir(directory):
    if filename.endswith('.xlsx'):
        # Read the Excel file into a pandas dataframe
        filepath = os.path.join(directory, filename)
        df = pd.read_excel(filepath)
        # Append the dataframe to the list
        dataframes.append(df)

# Concatenate all dataframes into a single dataframe
concat_df = pd.concat(dataframes)
# List of selected cost codes to filter
selected_cost_codes = ['11-110', '11-123', '11-132', '11-135', '11-140', '11-150', '11-152',
                       '11-153', '11-155', '11-156', '11-176', '11-182', '11-183', '11-184',
                       '11-185', '11-186', '11-205', '11-206', '11-208', '11-210', '11-220',
                       '11-222', '11-225', '11-230', '11-231', '11-232', '11-235', '11-236',
                       '11-240', '11-245', '11-250', '11-300', '11-301', '11-388', '11-398',
                       '11-399', '11-400', '11-401', '11-500', '11-900', '11-910', '11-920',
                       '80-000', '80-001', '80-002', '80-003', '80-004', '80-005', '80-006',
                       '80-007', '80-008', '80-009', '80-010', '80-011', '80-012', '80-013',
                       '80-014', '80-015', '80-016', '80-017', '80-020', '80-021', '80-022',
                       '80-023', '80-024', '80-025', '80-026', '80-027', '80-028', '80-029',
                       '80-030', '80-031', '80-032', '80-035', '80-040', '80-050', '80-060',
                       '80-075', '80-100', '80-101', '80-110', '80-111', '80-113', '80-114',
                       '80-120', '80-140', '80-150', '80-200', '80-220', '80-221', '80-250',
                       '80-300', '81-000', '81-001', '81-002', '81-003', '81-004', '81-005',
                       '81-006', '81-007', '81-008', '82-000', '82-001', '82-100', '82-105',
                       '82-110', '82-116', '82-117', '82-118', '82-119', '82-121', '82-122',
                       '84-001', '84-002', '86-001', '86-004', '86-005', '86-010', '86-011',
                       '86-012', '86-023', '86-034', '86-036', '86-103', '86-104', '86-105',
                       '86-107', '86-109', '86-908', '87-001', '87-002', '87-003', '88-002',
                       '88-003', '88-004', '88-005', '88-006', '88-007', '88-008', '88-009',
                       '88-010', '88-011', '88-021', '88-100', '88-310', '88-500', '88-999',
                       '89-001', '89-002', '89-003', '89-004', '89-005', '89-006', '89-007',
                       '89-008', '89-009', '89-010', '89-011', '89-012', '89-013', '89-020',
                       '89-021', '89-025', '89-030', '89-100', '89-110', '89-111', '89-112',
                       '89-113', '89-114', '89-115', '89-116', '89-117', '89-135', '89-145',
                       '89-150', '89-160', '89-172', '89-200', '89-201', '89-205', '89-206',
                       '89-215', '89-216', '89-220', '89-225', '89-250', '89-255', '89-300',
                       '89-301', '89-303', '89-304', '89-305', '89-307', '89-308', '89-310',
                       '89-311', '89-312', '89-315', '89-350', '89-400', '89-404', '89-405',
                       '89-406', '89-407', '89-408', '89-409', '89-410', '89-411', '89-412',
                       '89-416', '89-418', '89-420', '89-422', '89-450', '89-460', '89-465',
                       '89-500', '89-550', '89-552', '89-556', '89-600', '89-700', '89-701',
                       '89-702', '89-800', '90-001', '90-100', '99-000', '99-001', '99-002',
                       '99-003', '99-004', '99-051', '99-100', '99-110', '99-400', '99-666',
                       '99-700', '99-798', '99-799', '99-99800', '99-800', '99-801', '99-900',
                       '99-990', '99-999']

# Filter the dataframe for the selected cost codes
filtered_data = data[~data['Cost Code'].isin(selected_cost_codes)]

distinct_records = filtered_data[['Cost Code', 'Unit']].drop_duplicates()
distinct_cost_codes_df = pd.DataFrame(distinct_records)

distinct_cost_codes_df = distinct_cost_codes_df[distinct_cost_codes_df['Cost Code'] != 0]
distinct_cost_codes_df = distinct_cost_codes_df[distinct_cost_codes_df['Cost Code'].notna()]

filtered_data = filtered_data[filtered_data['Cost Code'] != 0]
filtered_data = filtered_data[filtered_data['Cost Code'].notna()]


filtered_data = filtered_data[(filtered_data['Actual Quantity'] != 0) &
                              (filtered_data['Actual Quantity'].notna()) &
                              (filtered_data['Actual Quantity'] > 0)]

'''
filtered_data_expected = filtered_data[filtered_data['Expected Labor Hours'] > 0] # Add this line
filtered_data_Actual = filtered_data[filtered_data['Actual Labor Hours'] > 0] # Add this line
'''

filtered_data_expected = filtered_data
filtered_data_Actual = filtered_data

print(filtered_data_expected)
print(filtered_data_Actual)
filtered_data_expected['Expected Productivity'] = filtered_data.apply(
    lambda row: row['Actual Quantity'] / row['Expected Labor Hours'] if row['Expected Labor Hours'] != 0 else 0,
    axis=1, result_type="expand"
)

filtered_data_Actual['Actual Productivity'] = filtered_data.apply(
    lambda row: row['Actual Quantity'] / row['Actual Labor Hours'] if row['Actual Labor Hours'] != 0 else 0,
    axis=1, result_type="expand"
)

filtered_data_expected.head()

std_expected = filtered_data_expected.groupby(['Cost Code', 'Unit'])['Expected Productivity']
std_expected.columns = ['Cost Code', 'Unit', 'std_expected']

std_expected.head()

std_expected = filtered_data_expected.groupby(['Cost Code', 'Unit'])['Expected Productivity'].std().reset_index()
std_expected.columns = ['Cost Code', 'Unit', 'std_expected']
std_expected.head()

import pandas as pd
from sklearn.ensemble import IsolationForest

# Calculate standard deviation for cost_code = "MH" and its corresponding unit for expected productivity
std_expected = filtered_data_expected.groupby(['Cost Code', 'Unit'])['Expected Productivity'].std().reset_index()
std_expected.columns = ['Cost Code', 'Unit', 'std_expected']
#print("std_expected = ",std_expected)

# Initialize variables
results_list = []

random_state = 1
init_contamination = 0.005

while init_contamination <= 0.05:
    print(f'{init_contamination:.4f}')

    # Initialize the Isolation Forest model
    iso_forest_actual = IsolationForest(contamination=init_contamination, random_state=random_state)
    iso_forest_expected = IsolationForest(contamination=init_contamination, random_state=random_state)

    # Fit the model and predict outliers for Actual Productivity
    iso_forest_actual.fit(filtered_data_Actual[filtered_data_Actual['Actual Productivity'] != 0][['Actual Productivity']])
    filtered_data_Actual['is_outlier_actual'] = iso_forest_actual.predict(filtered_data_Actual[['Actual Productivity']])

    # Filter inliers
    inliers_actual = filtered_data_Actual[filtered_data_Actual['is_outlier_actual'] == 1]

    '''
    # Fit the model and predict outliers for Expected Productivity
    iso_forest_expected.fit(filtered_data_expected[['Expected Productivity']])
    filtered_data_expected['is_outlier_expected'] = iso_forest_expected.predict(filtered_data_expected[['Expected Productivity']])
    '''

    # Calculate standard deviation for each cost_code and its corresponding unit for actual productivity disregarding zeroes and outliers
    std_actual = inliers_actual.groupby(['Cost Code', 'Unit'])['Actual Productivity'].std().reset_index()
    std_actual.columns = ['Cost Code', 'Unit', 'std_actual']
    #print(std_actual)
    #std_actual.to_excel(f'std_actual_{random_state:.4f}_{init_contamination}.xlsx')

    # Find difference between the standard deviation of actual with the corresponding standard deviation of expected grouped by cost code and unit
    result_df = pd.merge(std_actual, std_expected, on=['Cost Code', 'Unit'])
    result_df['difference'] = result_df['std_actual'] - result_df['std_expected']
    result_df['random_state'] = random_state
    result_df['init_contamination'] = init_contamination

    # Append to results list
    results_list.append(result_df)

    random_state += 1
    init_contamination += 0.005
    print(result_df)

# Concatenate all results into a single DataFrame
all_results_df = pd.concat(results_list)






############




# Define a function to find the minimum difference and corresponding values
def find_min_difference(filtered_df):
    if filtered_df.empty:  # Check if DataFrame is empty
        return None
    else:
        filtered_df['difference'] = filtered_df['difference'].fillna(filtered_df['difference'].max())
        min_results = filtered_df.loc[filtered_df.groupby('Unit')['difference'].idxmin()]
        return min_results

# Dictionary to hold the results for each cost code
cost_code_results = {}

# Loop through each unique cost code
for cost_code in all_results_df['Cost Code'].unique():
    filtered_results = all_results_df[all_results_df['Cost Code'] == cost_code]
    min_results = find_min_difference(filtered_results)
    cost_code_results[cost_code] = min_results

# Display the results as DataFrames for each cost code
for cost_code, df in cost_code_results.items():
    print(f"Results for Cost Code: {cost_code}")
    display(df[['Unit', 'difference', 'random_state', 'init_contamination']])

# Optional: Convert dictionary of DataFrames to a dictionary of dictionaries for more detailed display
unit_results = {}
for cost_code, df in cost_code_results.items():
    unit_results[cost_code] = {}
    for unit in df['Unit'].unique():
        min_difference_row = df[df['Unit'] == unit].sort_values(by='difference').iloc[0]
        unit_results[cost_code][unit] = {
            'Unit': min_difference_row['Unit'],
            'Minimum Difference': min_difference_row['difference'],
            'Random State': min_difference_row['random_state'],
            'Contamination': min_difference_row['init_contamination'],
        }

# Print the results in a readable format
for cost_code, units in unit_results.items():
    print(f"Results for Cost Code: {cost_code}")
    for unit, result in units.items():
        print(f"  Unit: {result['Unit']}")
        print(f"    Minimum Difference: {result['Minimum Difference']}")
        print(f"    Random State: {result['Random State']}")
        print(f"    Contamination: {result['Contamination']}")
        print()


############