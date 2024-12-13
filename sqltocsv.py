import re
import csv
from collections import defaultdict

def extract_values(sql_content, table_name):
    """
    Extract values from INSERT statements for the specified table.
    This handles various edge cases, such as quoted values and escaped characters.
    """
    # Find all INSERT statements for the table
    pattern = rf"INSERT INTO `{table_name}`\s*\(.*?\)\s*VALUES\s*(\(.*?\));"
    matches = re.finditer(pattern, sql_content, re.MULTILINE | re.DOTALL)
    
    all_values = []
    for match in matches:
        values_str = match.group(1)
        # Extract multiple rows of values if present
        rows = re.findall(r"\((.*?)\)", values_str)
        for row in rows:
            values = []
            current_value = ''
            in_quotes = False
            escape_next = False
            
            for char in row:
                if escape_next:  # Handle escaped characters
                    current_value += char
                    escape_next = False
                    continue
                
                if char == '\\':  # Start escape sequence
                    escape_next = True
                    continue
                
                if char in ['"', "'"]:
                    if in_quotes:
                        in_quotes = False
                    else:
                        in_quotes = True
                elif char == ',' and not in_quotes:
                    values.append(current_value.strip().strip("'").strip('"'))
                    current_value = ''
                    continue
                current_value += char
            
            values.append(current_value.strip().strip("'").strip('"'))
            all_values.append(values)
    
    return all_values

def create_csv_from_sql_dumps(questions_sql_path, answers_sql_path, output_csv_path):
    # Read SQL files
    with open(questions_sql_path, 'r', encoding='utf-8') as f:
        questions_content = f.read()
    with open(answers_sql_path, 'r', encoding='utf-8') as f:
        answers_content = f.read()
    
    # Parse questions
    questions_data = []
    for values in extract_values(questions_content, "evaluation_questions"):
        questions_data.append({
            'id': values[0],
            'text': values[1],
            'order': int(values[2])
        })
    
    # Sort questions by order
    questions_data.sort(key=lambda x: x['order'])
    
    # Create question mapping
    questions_map = {q['id']: q['text'] for q in questions_data}
    
    # Parse answers
    grouped_answers = defaultdict(dict)
    for values in extract_values(answers_content, "evaluation_answers"):
        eval_id = values[1]
        question_id = values[2]
        score = values[3]
        grouped_answers[eval_id][question_id] = score
    
    # Prepare CSV headers
    headers = ['evaluation_id'] + [q['text'] for q in questions_data]
    
    # Prepare rows
    rows = []
    for eval_id in sorted(grouped_answers.keys(), key=int):
        row = [eval_id]
        for question in questions_data:
            score = grouped_answers[eval_id].get(question['id'], '')
            row.append(score)
        rows.append(row)
    
    # Write to CSV
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

if __name__ == "__main__":
    questions_sql_path = "evaluation_questions.sql"
    answers_sql_path = "evaluation_answers.sql"
    output_csv_path = "evaluation_results.csv"
    
    create_csv_from_sql_dumps(questions_sql_path, answers_sql_path, output_csv_path)
    print(f"CSV file has been created at: {output_csv_path}")
