from db_ops import execute_custom_12

def handle_request_12(req, payload):
    keyword = payload.get('q')
    return execute_custom_12(keyword)
