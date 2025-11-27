import re
from datetime import datetime
import os
import csv

def check_file(file_path:str):
    """
        Check if path exists
    """

    return True if os.path.exists(file_path) else False

def open_file(file_path:str):
    """
        Open the file with the raw data and read it.
        Fixed the problem of reading the same instance on multiple lines (see the records variable)
    """

    if check_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        content_without_header = content.split('\n', 1)[1]
        records = re.findall(r'.+?"(?:\([^)]*\)|[^"])*"', content_without_header, flags=re.DOTALL)
        return records
    else:
        raise ValueError('Error Path!') 


def process_data(start_ts, end_ts, patient_str):
    """
        Process the data and adapt it with the correct format
    """

    try:
        start_dt = datetime.strptime(start_ts, '%Y-%m-%dT%H:%M:%S')
        end_dt = datetime.strptime(end_ts, '%Y-%m-%dT%H:%M:%S')
        start_date = start_dt.strftime('%d/%m/%Y')
        start_time = start_dt.strftime('%H:%M')
        end_date = end_dt.strftime('%d/%m/%Y')
        end_time = end_dt.strftime('%H:%M')
        patient_match = re.match(r'([^,]+), ([^(]+) \((\d+)\)', patient_str)

        if not patient_match:
            raise Exception
        
        first_name, last_name, patient_id = patient_match.groups()

        """Check if date is WEEKDAY"""
        if start_dt.weekday() >= 5:
            raise Exception
        
        return [start_date, start_time, end_date, end_time,
                first_name.strip(), last_name.strip(), int(patient_id)]

    except Exception as e:
        return None
    
def find_match(lines):
    """
        Find the match from the raw-data given the regular expression.
        Process the data after.
    """

    output_rows = []
    for line in lines[0:]:

        """
            PHASE 2 - Remove newline and replace it with whitespace
        """
        print(line)
        line = re.sub(r'\s+', ' ', line.strip())
        print(line)
        matches = re.match(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}),'
                        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}),'
                        r'"(.+)"', line)
        
        if not matches:
            continue
    
        start_ts, end_ts, patient_str = matches.groups()

        """
            PHASE 4
        """
        row = process_data(start_ts, end_ts, patient_str)
        if row:
            output_rows.append(row)
    
    return output_rows

def print_on_file(rows, filepath_out):
    """
        Print the data formatted into a csv file
    """
    if check_file(filepath_out):
        with open(filepath_out, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['start_date','start_time','end_date','end_time',
                            'patient_first_name','patient_last_name','patient_id'])
            writer.writerows(rows)

    print(f"CSV file successfully created: {FILE_OUT}")        
        


DIR_INPUT = "input_file"
FILE_IN = "data.txt"
FILE_OUT = "dataformatted.csv"
DIR_OUTPUT = "output_file"

if __name__ == '__main__':

    """
        MAIN FUNCTION
    """

    curr_working_directory = os.path.dirname(os.path.realpath(__file__))
    filepath_in = os.path.join(curr_working_directory, DIR_INPUT)
    filepath_in = os.path.join(filepath_in, FILE_IN)
    filepath_out = os.path.join(curr_working_directory, DIR_OUTPUT)
    filepath_out = os.path.join(filepath_out, FILE_OUT)
    
    
    """
        PHASE 1
    """
    records = open_file(file_path=filepath_in)

    """
        PHASE 2-3
    """
    rows = find_match(records)
    
    """
        PHASE 5-6
    """
    print_on_file(rows, filepath_out)