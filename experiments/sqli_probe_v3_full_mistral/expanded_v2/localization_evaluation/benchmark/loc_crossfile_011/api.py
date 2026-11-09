from db_ops import execute_custom_11

def handle_request_11(req, payload):
    keyword = payload.get('q')
    return execute_custom_11(keyword)
