from db_ops import execute_custom_3

def handle_request_3(req, payload):
    keyword = payload.get('q')
    return execute_custom_3(keyword)
