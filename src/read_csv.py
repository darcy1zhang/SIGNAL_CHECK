

import csv

def read_csv_to_nested_dict(csv_file_path, primary_key_field):
    """
    Reads a CSV file and stores it in a two-dimensional dictionary.
    
    Parameters:
    csv_file_path (str): The path to the CSV file.
    primary_key_field (str): The column name to be used as the primary key in the first layer.
    
    Returns:
    dict: A two-dimensional dictionary with the specified primary key.
    """
    data_dict = {}

    try:
        # Open the CSV file
        with open(csv_file_path, mode='r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Check if the file is empty
            if reader.fieldnames is None:
                print("Error: The CSV file is empty.")
                return data_dict

            # Check if the specified primary key exists in the columns
            if primary_key_field not in reader.fieldnames:
                print(f"Error: The specified primary key '{primary_key_field}' is not a valid column name.")
                return data_dict
            
            # Populate the dictionary
            for row in reader:
                primary_key = row[primary_key_field]
                data_dict[primary_key] = {column: value for column, value in row.items()}
        
    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
    except IOError:
        print(f"Error: Could not read the file '{csv_file_path}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    return data_dict

# Example usage
# csv_file_path = 'path/to/your/file.csv'
# primary_key_field = 'id'  # Change this to your desired primary key field
# data = read_csv_to_nested_dict(csv_file_path, primary_key_field)
# print(data)
