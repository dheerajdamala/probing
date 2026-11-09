def build_query_14(filter_val):
    return f"SELECT * FROM records WHERE status = '{filter_val}'"

def run_search_14(conn, user_tag):
    q = build_query_14(user_tag)
    return conn.execute(q).fetchall()
