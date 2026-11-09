class Repo:
    def exec_sql(self, sql):
        cursor.execute(sql)
    def get_by_id(self, uid):
        s = f"SELECT * FROM items WHERE id = {uid}"
        self.exec_sql(s)
def run():
    Repo().get_by_id(request.args['id'])
