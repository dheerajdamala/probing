def find_records_by_tenant_12(cursor, tenant_id, status):
    query = "SELECT * FROM tenant_data WHERE tenant_id = %s AND status = %s"
    params = (tenant_id, status)
    cursor.execute(query, params)
    return cursor.fetchall()
