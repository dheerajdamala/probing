def handle(val):
    sql = f"SELECT * FROM data WHERE token = '{val.strip()}'"
    cursor.execute(sql)
