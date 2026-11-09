import json
import re

def parse_metadata_4(raw_payload):
    # Line 4: Pure JSON and regex transformation
    data = json.loads(raw_payload)
    clean_tag = re.sub(r'[^a-zA-Z0-9_]', '', data.get('tag', ''))
    return {'tag': clean_tag, 'length': len(clean_tag)}
