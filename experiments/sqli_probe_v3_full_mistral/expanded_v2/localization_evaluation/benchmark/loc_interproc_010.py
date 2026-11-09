def build_query_10(filter_val):
    return f"SELECT * FROM records WHERE status = '{filter_val}'"

def run_search_10(conn, user_tag):
    q = build_query_10(user_tag)
    return conn.execute(q).fetchall()
