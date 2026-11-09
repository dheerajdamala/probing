import requests
import io
import pandas as pd
import re

sql_regex = re.compile(
    r'(select\s+[\s\S]*?\s+from|insert\s+into|update\s+[\s\S]*?\s+set|delete\s+from|'
    r'cursor\.execute|sqlite3|psycopg2|sqlalchemy|raw\(|execute_sql|\.raw_query|'
    r'db\.session\.execute|cursor\(\)\.execute|\.execute\(["\']\s*(select|insert|update|delete|create|drop))',
    re.IGNORECASE
)

url = "https://huggingface.co/datasets/lemon42-ai/Code_Vulnerability_Labeled_Dataset/resolve/main/code_vul_dataset.csv"
print("Downloading lemon42-ai code_vul_dataset.csv...")
r = requests.get(url, timeout=30)
if r.status_code == 200:
    df = pd.read_csv(io.BytesIO(r.content))
    print("lemon42-ai shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Sample:\n", df.head(2))
    if 'language' in df.columns:
        print("Languages:", df['language'].value_counts())
    # Check SQL
    code_col = 'code' if 'code' in df.columns else df.columns[0]
    has_sql = df[code_col].astype(str).apply(lambda s: bool(sql_regex.search(s)))
    print("Has SQL:", has_sql.sum())
