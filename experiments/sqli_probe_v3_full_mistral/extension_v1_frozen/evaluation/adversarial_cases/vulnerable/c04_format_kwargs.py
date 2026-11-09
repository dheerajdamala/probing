def search(val):
    q = "SELECT * FROM articles WHERE title LIKE '%{k}%'".format(k=val)
    cursor.execute(q)
