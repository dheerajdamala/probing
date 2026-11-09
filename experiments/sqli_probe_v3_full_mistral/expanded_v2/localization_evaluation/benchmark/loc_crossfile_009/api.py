from db_ops import execute_custom_9

def handle_request_9(req, payload):
    keyword = payload.get('q')
    return execute_custom_9(keyword)
