import csv

with open('test_questions.csv') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        print(f'{i+1}. [{row["question_type"]}] {row["question"]}')
        print(f'   Expected doc: {row["expected_doc_id"]}')
        print(f'   Keywords: {row["expected_keywords"]}')
        print()