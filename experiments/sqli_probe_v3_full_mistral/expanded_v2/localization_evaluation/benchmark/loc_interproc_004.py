def build_query_4(filter_val):
    return f"SELECT * FROM records WHERE status = '{filter_val}'"

def run_search_4(conn, user_tag):
    q = build_query_4(user_tag)
    return conn.execute(q).fetchall()
