import json

nb = json.load(open('notebooks/run_static_steering_experiment.ipynb', encoding='utf-8'))
print('Cells count:', len(nb['cells']))
for i, cell in enumerate(nb['cells'][:10]):
    ctype = cell.get('cell_type')
    src = ''.join(cell.get('source', []))[:100].replace('\n', ' ')
    print(f'Cell {i} ({ctype}): {src}')
