def run_select_all(tbl_name):
    cursor.execute(f"SELECT * FROM {tbl_name}")
def handle():
    target = request.args['target_table']
    run_select_all(target)
