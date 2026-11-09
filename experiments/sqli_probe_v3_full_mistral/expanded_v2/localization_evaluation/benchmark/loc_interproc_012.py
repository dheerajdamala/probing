def build_query_12(filter_val):
    return f"SELECT * FROM records WHERE status = '{filter_val}'"

def run_search_12(conn, user_tag):
    q = build_query_12(user_tag)
    return conn.execute(q).fetchall()
