import csv
import random
import re
from transformers import AutoTokenizer

def data_augmentation(file_path):
    model_name = "kykim/bert-kor-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=160)
    counter = 0
    values = [4.8, 4.9, 5.0]

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)

            # Skip the first row
            next(reader)

            # Get the total number of lines in the file
            total_lines = sum(1 for _ in reader)

            # Reset the file pointer to the beginning of the file
            file.seek(0)

            # Skip the first row again
            next(reader)

            # Define a variable to store the new data
            new_data = []

            # Iterate through the rows in the original file (starting from the second row)
            for i, row in enumerate(reader):
                # Process each row as needed to create new data
                # Example: Extract the numeric part from the id column
                match = re.search(r'(\d+)$', row[0])
                id_numeric_part = match.group(1) if match else None

                # Tokenize the sentence in the third column using the transformers tokenizer
                sentence_1_tokens = tokenizer.encode(row[2], max_length=160, truncation=True)
                sentence_2_tokens = tokenizer.encode(row[3], max_length=160, truncation=True)

                # Remove the token from sentence_tokens if the similarity is less than 2
                if len(sentence_1_tokens) > 4 and float(row[4]) < 2:
                    del sentence_1_tokens[-2]

                    # Use regular expression to find the last number and replace it with the custom value
                    modified_id = re.sub(r'\d+$', str(total_lines+counter), row[0])
                    counter += 1

                    # Decode the modified tokens to obtain the word forms
                    modified_sentence_words = tokenizer.decode(sentence_1_tokens, skip_special_tokens=True)

                    selected_value = random.choice(values)

                    # Append the numeric part and tokenized sentence to the new data
                    new_row = [modified_id, row[1], row[2], modified_sentence_words, str(selected_value), '1.0']

                    new_data.append(new_row)

                # Remove the token from sentence_tokens if the similarity is less than 2
                if len(sentence_2_tokens) > 4 and float(row[4]) < 2:
                    del sentence_2_tokens[-2]

                    # Use regular expression to find the last number and replace it with the custom value
                    modified_id = re.sub(r'\d+$', str(total_lines+counter), row[0])
                    counter += 1

                    # Decode the modified tokens to obtain the word forms
                    modified_sentence_words = tokenizer.decode(sentence_2_tokens, skip_special_tokens=True)

                    selected_value = random.choice(values)

                    # Append the numeric part and tokenized sentence to the new data
                    new_row = [modified_id, row[1], modified_sentence_words, row[3], str(selected_value), '1.0']

                    new_data.append(new_row)

                # Reverse sentence order
                elif float(row[4]) >= 2 and float(row[4]) < 4.8:
                    modified_id = re.sub(r'\d+$', str(total_lines+counter), row[0])
                    counter += 1

                    new_row = [modified_id, row[1], row[3], row[2], row[4], row[5]]

                    new_data.append(new_row)

            # You can now use the 'new_data' variable for further processing or save it to a new file
            try:
                with open(file_path, 'a', newline='') as file:  # Open the existing file in append mode
                    writer = csv.writer(file)
                    writer.writerows(new_data)

                print(f"Combined data written to {file_path}")

            except Exception as e:
                print(f"An error occurred while writing the combined data: {e}")

    except FileNotFoundError:
        print(f"The file '{file_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    data_augmentation('../../data/train.csv')
    data_augmentation('../../data/dev.csv')
