from db_ops import execute_custom_15

def handle_request_15(req, payload):
    keyword = payload.get('q')
    return execute_custom_15(keyword)
