# import csv

# def process_dataset(file_path, output_file):
#     with open(file_path, 'r', encoding='utf-8') as file, open(output_file, 'w', newline='', encoding='utf-8') as output:
#         csv_writer = csv.writer(output)
#         csv_writer.writerow(['message_body'])  # Writing header

#         for line in file:
#             parts = line.strip().split('\t')  # Splitting each line by tab character
#             if len(parts) > 2:
#                 message_body = parts[2]  # Extracting the message body
#                 csv_writer.writerow([message_body])  # Writing the message body to the CSV

# # Replace 'path/to/dataset.txt' with the path to your dataset file
# # Replace 'output.csv' with the desired output CSV file name
# process_dataset('C:/Users/Admin/Desktop/text.txt', 'output.csv')



# import csv

# def remove_quotation_marks(input_file, output_file):
#     modified_data = []
#     # Try different encodings if 'utf-8' doesn't work, like 'ISO-8859-1', 'cp1252'
#     encoding_to_try = ['utf-8', 'ISO-8859-1', 'cp1252']

#     for encoding in encoding_to_try:
#         try:
#             with open(input_file, 'r', encoding=encoding) as file:
#                 csv_reader = csv.reader(file)
#                 modified_data = [ [item.replace('"', '') for item in row] for row in csv_reader]
#             break
#         except UnicodeDecodeError:
#             print(f"Trying with encoding: {encoding}")

#     # If none of the encodings worked, you may use 'utf-8' with 'ignore' or 'replace' error handling
#     if not modified_data:
#         with open(input_file, 'r', encoding='utf-8', errors='ignore') as file:
#             csv_reader = csv.reader(file)
#             modified_data = [ [item.replace('"', '') for item in row] for row in csv_reader]

#     # Writing the modified data back to the CSV file
#     with open(output_file, 'w', newline='', encoding='utf-8') as file:
#         csv_writer = csv.writer(file)
#         csv_writer.writerows(modified_data)

# remove_quotation_marks('C:/Users/Admin/Desktop/505 project/CS505Project/output.csv', 'C:/Users/Admin/Desktop/505 project/CS505Project/output.csv')
