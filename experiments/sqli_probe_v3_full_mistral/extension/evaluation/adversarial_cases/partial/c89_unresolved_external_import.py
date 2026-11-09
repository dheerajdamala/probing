from third_party_security_lib import mystery_sanitizer
def process():
    val = mystery_sanitizer(request.args['val'])
    cursor.execute(val)
