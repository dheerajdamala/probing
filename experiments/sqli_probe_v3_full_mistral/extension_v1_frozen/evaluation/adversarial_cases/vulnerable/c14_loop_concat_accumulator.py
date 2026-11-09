def loop_concat(inputs):
    base = 'SELECT * FROM metrics WHERE id = '
    for inp in inputs:
        stmt = base + inp
        cursor.execute(stmt)
