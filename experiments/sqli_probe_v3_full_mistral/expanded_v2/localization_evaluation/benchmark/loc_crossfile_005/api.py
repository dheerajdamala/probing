from db_ops import execute_custom_5

def handle_request_5(req, payload):
    keyword = payload.get('q')
    return execute_custom_5(keyword)
