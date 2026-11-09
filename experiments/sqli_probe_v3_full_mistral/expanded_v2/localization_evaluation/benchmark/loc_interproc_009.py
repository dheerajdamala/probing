def build_query_9(filter_val):
    return f"SELECT * FROM records WHERE status = '{filter_val}'"

def run_search_9(conn, user_tag):
    q = build_query_9(user_tag)
    return conn.execute(q).fetchall()
