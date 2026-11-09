def repo_exec(q):
    cursor.execute(f"SELECT * FROM records WHERE {q}")
