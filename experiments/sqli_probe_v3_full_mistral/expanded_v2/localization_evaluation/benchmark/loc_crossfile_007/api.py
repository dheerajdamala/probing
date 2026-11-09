from db_ops import execute_custom_7

def handle_request_7(req, payload):
    keyword = payload.get('q')
    return execute_custom_7(keyword)
