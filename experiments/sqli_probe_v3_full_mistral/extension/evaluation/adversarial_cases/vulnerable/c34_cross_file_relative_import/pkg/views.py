from .sql_utils import render_clause
def run_view():
    v = request.args['code']
    q = render_clause(v)
    cursor.execute(q)
