def build_query_11(filter_val):
    return f"SELECT * FROM records WHERE status = '{filter_val}'"

def run_search_11(conn, user_tag):
    q = build_query_11(user_tag)
    return conn.execute(q).fetchall()
