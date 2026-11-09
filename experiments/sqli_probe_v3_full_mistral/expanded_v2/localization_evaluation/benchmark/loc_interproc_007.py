def build_query_7(filter_val):
    return f"SELECT * FROM records WHERE status = '{filter_val}'"

def run_search_7(conn, user_tag):
    q = build_query_7(user_tag)
    return conn.execute(q).fetchall()
