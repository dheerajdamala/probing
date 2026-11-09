def loop_reassign(id_list):
    for uid in id_list:
        stmt = f"SELECT * FROM items WHERE id = {uid}"
        cursor.execute(stmt)
def handle(): loop_reassign(request.args.getlist('id'))
