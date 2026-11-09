def run_raw_sql(sql_str):
    cursor.execute(sql_str)
def process():
    cat = request.form['cat']
    sql = "SELECT * FROM catalog WHERE c = '" + cat + "'"
    run_raw_sql(sql)
