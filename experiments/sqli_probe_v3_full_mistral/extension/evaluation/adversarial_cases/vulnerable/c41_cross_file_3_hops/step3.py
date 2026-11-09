def exec_sink(v):
    cursor.execute(f"SELECT * FROM data WHERE v = '{v}'")
