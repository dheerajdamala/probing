from db_ops import execute_custom_13

def handle_request_13(req, payload):
    keyword = payload.get('q')
    return execute_custom_13(keyword)
