def walk_tree(node):
    if node.has_child:
        walk_tree(node.child)
    cursor.execute(f"SELECT * FROM nodes WHERE id = '{node.id}'")
def handle(root): walk_tree(root)
