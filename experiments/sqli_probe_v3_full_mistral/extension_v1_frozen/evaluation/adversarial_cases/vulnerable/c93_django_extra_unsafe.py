def filter_name(val):
    return Entry.objects.extra(where=["title = '" + val + "'"])
def handle(): filter_name(request.args['t'])
