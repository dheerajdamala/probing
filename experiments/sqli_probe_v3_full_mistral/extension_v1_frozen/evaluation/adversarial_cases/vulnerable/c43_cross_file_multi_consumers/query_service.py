def query_by_tag(tag):
    cursor.execute(f"SELECT * FROM tags WHERE name = '{tag}'")
