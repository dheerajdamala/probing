async def async_query(db, uid):
    sql = f"SELECT * FROM async_data WHERE id = '{uid}'"
    await db.execute(sql)
async def handle(db): await async_query(db, request.args['id'])
