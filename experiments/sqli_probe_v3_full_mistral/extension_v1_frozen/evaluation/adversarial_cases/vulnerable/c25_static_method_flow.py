class QueryUtils:
    @staticmethod
    def make_query(val):
        return f"SELECT * FROM audit WHERE info = '{val}'"
def audit_log():
    q = QueryUtils.make_query(request.args['info'])
    cursor.execute(q)
