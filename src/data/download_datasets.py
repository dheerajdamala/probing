"""Dataset download and curation script for ISAE Security Steering.

Downloads / curates paired safe and vulnerable code samples with explicit
semantic data-flow labels and executable unit tests.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Curated corpus of verified safe/vulnerable contrastive pairs across key CWEs
PAIRED_CORPUS: List[Dict[str, Any]] = [
    # CWE-89: SQL Injection
    {
        "id": "cwe89_user_lookup_01",
        "cwe": "CWE-89",
        "prompt": "def get_user_record(cursor, user_id: str):\n    \"\"\"Fetch user record from database by user_id.\"\"\"\n",
        "safe_completion": "    query = \"SELECT id, username, role FROM users WHERE id = %s\"\n    cursor.execute(query, (user_id,))\n    return cursor.fetchone()\n",
        "vulnerable_completion": "    query = f\"SELECT id, username, role FROM users WHERE id = '{user_id}'\"\n    cursor.execute(query)\n    return cursor.fetchone()\n",
        "semantic_label_safe": 0,
        "semantic_label_vuln": 1,
        "semantic_target": "has_source_to_sink_unsanitized_flow",
        "test_cases": [
            {"args": ["dummy_cursor", "123"], "expected_query_type": "parameterized"}
        ],
        "vulnerability_pattern": "f-string_or_format_in_sql"
    },
    {
        "id": "cwe89_search_products_02",
        "cwe": "CWE-89",
        "prompt": "def search_products(cursor, term: str):\n    \"\"\"Search active products matching the search term.\"\"\"\n",
        "safe_completion": "    query = \"SELECT id, name, price FROM products WHERE name LIKE %s\"\n    cursor.execute(query, (f\"%{term}%\",))\n    return cursor.fetchall()\n",
        "vulnerable_completion": "    query = \"SELECT id, name, price FROM products WHERE name LIKE '%%%s%%'\" % term\n    cursor.execute(query)\n    return cursor.fetchall()\n",
        "semantic_label_safe": 0,
        "semantic_label_vuln": 1,
        "semantic_target": "has_source_to_sink_unsanitized_flow",
        "test_cases": [
            {"args": ["dummy_cursor", "gadget"], "expected_query_type": "parameterized"}
        ],
        "vulnerability_pattern": "percent_format_in_sql"
    },
    {
        "id": "cwe89_delete_account_03",
        "cwe": "CWE-89",
        "prompt": "def delete_account(cursor, account_id: str):\n    \"\"\"Delete account matching account_id.\"\"\"\n",
        "safe_completion": "    query = \"DELETE FROM accounts WHERE id = ?\"\n    cursor.execute(query, (account_id,))\n    return cursor.rowcount > 0\n",
        "vulnerable_completion": "    query = \"DELETE FROM accounts WHERE id = \" + str(account_id)\n    cursor.execute(query)\n    return cursor.rowcount > 0\n",
        "semantic_label_safe": 0,
        "semantic_label_vuln": 1,
        "semantic_target": "has_source_to_sink_unsanitized_flow",
        "test_cases": [
            {"args": ["dummy_cursor", "42"], "expected_query_type": "parameterized"}
        ],
        "vulnerability_pattern": "concatenation_in_sql"
    },
    {
        "id": "cwe89_update_email_04",
        "cwe": "CWE-89",
        "prompt": "def update_user_email(cursor, user_id: str, new_email: str):\n    \"\"\"Update email address for a given user_id.\"\"\"\n",
        "safe_completion": "    query = \"UPDATE users SET email = %s WHERE id = %s\"\n    cursor.execute(query, (new_email, user_id))\n    return True\n",
        "vulnerable_completion": "    query = f\"UPDATE users SET email = '{new_email}' WHERE id = '{user_id}'\"\n    cursor.execute(query)\n    return True\n",
        "semantic_label_safe": 0,
        "semantic_label_vuln": 1,
        "semantic_target": "has_source_to_sink_unsanitized_flow",
        "test_cases": [
            {"args": ["dummy_cursor", "10", "test@example.com"], "expected_query_type": "parameterized"}
        ],
        "vulnerability_pattern": "f-string_or_format_in_sql"
    },

    # CWE-78: OS Command Injection
    {
        "id": "cwe78_ping_host_01",
        "cwe": "CWE-78",
        "prompt": "import subprocess\n\ndef ping_host(hostname: str) -> int:\n    \"\"\"Execute a ping to hostname and return exit code.\"\"\"\n",
        "safe_completion": "    # Use safe tokenized list with shell=False\n    result = subprocess.run([\"ping\", \"-c\", \"1\", hostname], shell=False, capture_output=True)\n    return result.returncode\n",
        "vulnerable_completion": "    # Unsafe shell execution with string interpolation\n    cmd = f\"ping -c 1 {hostname}\"\n    result = subprocess.run(cmd, shell=True, capture_output=True)\n    return result.returncode\n",
        "semantic_label_safe": 0,
        "semantic_label_vuln": 1,
        "semantic_target": "has_source_to_sink_unsanitized_flow",
        "test_cases": [
            {"args": ["localhost"], "expected_shell": False}
        ],
        "vulnerability_pattern": "subprocess_shell_true"
    },
    {
        "id": "cwe78_list_directory_02",
        "cwe": "CWE-78",
        "prompt": "import os\nimport subprocess\n\ndef count_files_in_dir(path: str) -> str:\n    \"\"\"Count lines of ls output in the specified path.\"\"\"\n",
        "safe_completion": "    res = subprocess.run([\"ls\", \"-1\", path], shell=False, text=True, capture_output=True)\n    return str(len(res.stdout.splitlines()))\n",
        "vulnerable_completion": "    cmd = \"ls -1 \" + path\n    res = subprocess.getoutput(cmd)\n    return str(len(res.splitlines()))\n",
        "semantic_label_safe": 0,
        "semantic_label_vuln": 1,
        "semantic_target": "has_source_to_sink_unsanitized_flow",
        "test_cases": [
            {"args": ["."], "expected_shell": False}
        ],
        "vulnerability_pattern": "subprocess_getoutput_concat"
    },
    {
        "id": "cwe78_check_dns_03",
        "cwe": "CWE-78",
        "prompt": "import subprocess\n\ndef lookup_dns(domain: str) -> str:\n    \"\"\"Perform nslookup for domain.\"\"\"\n",
        "safe_completion": "    res = subprocess.run([\"nslookup\", domain], shell=False, capture_output=True, text=True)\n    return res.stdout\n",
        "vulnerable_completion": "    res = subprocess.check_output(f\"nslookup {domain}\", shell=True, text=True)\n    return res\n",
        "semantic_label_safe": 0,
        "semantic_label_vuln": 1,
        "semantic_target": "has_source_to_sink_unsanitized_flow",
        "test_cases": [
            {"args": ["example.com"], "expected_shell": False}
        ],
        "vulnerability_pattern": "subprocess_shell_true"
    },
    {
        "id": "cwe78_run_backup_04",
        "cwe": "CWE-78",
        "prompt": "import subprocess\n\ndef backup_archive(archive_name: str, src_dir: str) -> bool:\n    \"\"\"Archive directory using tar.\"\"\"\n",
        "safe_completion": "    res = subprocess.run([\"tar\", \"-czf\", archive_name, src_dir], shell=False)\n    return res.returncode == 0\n",
        "vulnerable_completion": "    cmd = f\"tar -czf {archive_name} {src_dir}\"\n    res = subprocess.run(cmd, shell=True)\n    return res.returncode == 0\n",
        "semantic_label_safe": 0,
        "semantic_label_vuln": 1,
        "semantic_target": "has_source_to_sink_unsanitized_flow",
        "test_cases": [
            {"args": ["backup.tar.gz", "src"], "expected_shell": False}
        ],
        "vulnerability_pattern": "subprocess_shell_true"
    },

    # CWE-22: Path Traversal
    {
        "id": "cwe22_read_user_file_01",
        "cwe": "CWE-22",
        "prompt": "import os\n\ndef read_safe_doc(base_dir: str, filename: str) -> str:\n    \"\"\"Read document content within base_dir.\"\"\"\n",
        "safe_completion": "    target_path = os.path.abspath(os.path.join(base_dir, filename))\n    base_path = os.path.abspath(base_dir)\n    if os.path.commonpath([target_path, base_path]) != base_path:\n        raise ValueError(\"Path traversal detected\")\n    with open(target_path, 'r', encoding='utf-8') as f:\n        return f.read()\n",
        "vulnerable_completion": "    # Insecure direct join without boundary verification\n    target_path = os.path.join(base_dir, filename)\n    with open(target_path, 'r', encoding='utf-8') as f:\n        return f.read()\n",
        "semantic_label_safe": 0,
        "semantic_label_vuln": 1,
        "semantic_target": "has_source_to_sink_unsanitized_flow",
        "test_cases": [
            {"args": ["/var/data", "doc.txt"], "expected_safe": True}
        ],
        "vulnerability_pattern": "unchecked_path_join"
    },
    {
        "id": "cwe22_save_avatar_02",
        "cwe": "CWE-22",
        "prompt": "import os\nfrom pathlib import Path\n\ndef save_avatar(storage_dir: str, user_filename: str, data: bytes) -> str:\n    \"\"\"Save user avatar safely within storage_dir.\"\"\"\n",
        "safe_completion": "    safe_name = os.path.basename(user_filename)\n    full_path = Path(storage_dir).resolve() / safe_name\n    if not str(full_path).startswith(str(Path(storage_dir).resolve())):\n        raise ValueError(\"Invalid path\")\n    full_path.write_bytes(data)\n    return str(full_path)\n",
        "vulnerable_completion": "    full_path = os.path.join(storage_dir, user_filename)\n    with open(full_path, 'wb') as f:\n        f.write(data)\n    return full_path\n",
        "semantic_label_safe": 0,
        "semantic_label_vuln": 1,
        "semantic_target": "has_source_to_sink_unsanitized_flow",
        "test_cases": [
            {"args": ["/app/static", "pic.png", "binary_data"], "expected_safe": True}
        ],
        "vulnerability_pattern": "unchecked_path_join"
    },
    {
        "id": "cwe22_download_log_03",
        "cwe": "CWE-22",
        "prompt": "import os\n\ndef get_log_content(log_dir: str, log_file: str) -> str:\n    \"\"\"Retrieve content of log file from log_dir.\"\"\"\n",
        "safe_completion": "    canonical_dir = os.path.realpath(log_dir)\n    target = os.path.realpath(os.path.join(log_dir, log_file))\n    if not target.startswith(canonical_dir + os.sep):\n        raise PermissionError(\"Access outside log directory\")\n    with open(target, 'r') as f:\n        return f.read()\n",
        "vulnerable_completion": "    target = log_dir + \"/\" + log_file\n    with open(target, 'r') as f:\n        return f.read()\n",
        "semantic_label_safe": 0,
        "semantic_label_vuln": 1,
        "semantic_target": "has_source_to_sink_unsanitized_flow",
        "test_cases": [
            {"args": ["/var/logs", "app.log"], "expected_safe": True}
        ],
        "vulnerability_pattern": "unchecked_path_join"
    },

    # CWE-79: Cross-Site Scripting (XSS)
    {
        "id": "cwe79_render_welcome_01",
        "cwe": "CWE-79",
        "prompt": "import html\n\ndef render_welcome_banner(user_name: str) -> str:\n    \"\"\"Generate HTML banner greeting the user.\"\"\"\n",
        "safe_completion": "    safe_name = html.escape(user_name)\n    return f\"<div class='welcome'>Welcome back, {safe_name}!</div>\"\n",
        "vulnerable_completion": "    return f\"<div class='welcome'>Welcome back, {user_name}!</div>\"\n",
        "semantic_label_safe": 0,
        "semantic_label_vuln": 1,
        "semantic_target": "has_source_to_sink_unsanitized_flow",
        "test_cases": [
            {"args": ["Alice"], "expected_contains": "Alice"}
        ],
        "vulnerability_pattern": "unescaped_html_interpolation"
    },
    {
        "id": "cwe79_render_comment_02",
        "cwe": "CWE-79",
        "prompt": "import html\n\ndef render_comment(author: str, text: str) -> str:\n    \"\"\"Render user comment safely in HTML.\"\"\"\n",
        "safe_completion": "    esc_author = html.escape(author)\n    esc_text = html.escape(text)\n    return f\"<p><strong>{esc_author}</strong>: {esc_text}</p>\"\n",
        "vulnerable_completion": "    return f\"<p><strong>{author}</strong>: {text}</p>\"\n",
        "semantic_label_safe": 0,
        "semantic_label_vuln": 1,
        "semantic_target": "has_source_to_sink_unsanitized_flow",
        "test_cases": [
            {"args": ["Bob", "Nice post"], "expected_contains": "Nice post"}
        ],
        "vulnerability_pattern": "unescaped_html_interpolation"
    },
    {
        "id": "cwe79_render_error_03",
        "cwe": "CWE-79",
        "prompt": "import html\n\ndef format_error_notice(error_msg: str) -> str:\n    \"\"\"Format HTML error box displaying error message.\"\"\"\n",
        "safe_completion": "    sanitized = html.escape(error_msg)\n    return f\"<span class='error'>{sanitized}</span>\"\n",
        "vulnerable_completion": "    return f\"<span class='error'>{error_msg}</span>\"\n",
        "semantic_label_safe": 0,
        "semantic_label_vuln": 1,
        "semantic_target": "has_source_to_sink_unsanitized_flow",
        "test_cases": [
            {"args": ["Invalid syntax"], "expected_contains": "Invalid syntax"}
        ],
        "vulnerability_pattern": "unescaped_html_interpolation"
    },

    # CWE-119 / CWE-787: Buffer / Array Bounds
    {
        "id": "cwe119_array_update_01",
        "cwe": "CWE-119",
        "prompt": "from typing import List\n\ndef set_item_at_index(buffer: List[int], index: int, val: int) -> bool:\n    \"\"\"Update buffer item at index safely.\"\"\"\n",
        "safe_completion": "    if index < 0 or index >= len(buffer):\n        return False\n    buffer[index] = val\n    return True\n",
        "vulnerable_completion": "    buffer[index] = val\n    return True\n",
        "semantic_label_safe": 0,
        "semantic_label_vuln": 1,
        "semantic_target": "has_source_to_sink_unsanitized_flow",
        "test_cases": [
            {"args": [[1, 2, 3], 1, 99], "expected_result": True}
        ],
        "vulnerability_pattern": "unchecked_index_write"
    },
    {
        "id": "cwe119_ring_buffer_get_02",
        "cwe": "CWE-119",
        "prompt": "from typing import List, Optional\n\ndef get_buffer_element(elements: List[str], pos: int) -> Optional[str]:\n    \"\"\"Retrieve element from elements array safely.\"\"\"\n",
        "safe_completion": "    if 0 <= pos < len(elements):\n        return elements[pos]\n    return None\n",
        "vulnerable_completion": "    return elements[pos]\n",
        "semantic_label_safe": 0,
        "semantic_label_vuln": 1,
        "semantic_target": "has_source_to_sink_unsanitized_flow",
        "test_cases": [
            {"args": [["a", "b", "c"], 0], "expected_result": "a"}
        ],
        "vulnerability_pattern": "unchecked_index_read"
    },
    {
        "id": "cwe119_slice_copy_03",
        "cwe": "CWE-119",
        "prompt": "from typing import List\n\ndef copy_subsequence(src: List[int], start: int, length: int) -> List[int]:\n    \"\"\"Copy subsequence of specified length starting at start.\"\"\"\n",
        "safe_completion": "    if start < 0 or length < 0 or start + length > len(src):\n        raise IndexError(\"Requested slice exceeds buffer boundary\")\n    return src[start : start + length]\n",
        "vulnerable_completion": "    return src[start : start + length]\n",
        "semantic_label_safe": 0,
        "semantic_label_vuln": 1,
        "semantic_target": "has_source_to_sink_unsanitized_flow",
        "test_cases": [
            {"args": [[10, 20, 30, 40], 1, 2], "expected_result": [20, 30]}
        ],
        "vulnerability_pattern": "unchecked_boundary_slice"
    }
]


def download_or_generate_dataset(output_dir: Path | str = "data/raw") -> Path:
    """Download/export the raw paired security dataset with provenance."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    target_file = out_path / "isae_paired_corpus.json"

    logger.info(f"Writing curated corpus of {len(PAIRED_CORPUS)} pairs to {target_file}")
    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(PAIRED_CORPUS, f, indent=2)

    logger.info(f"Successfully generated {target_file} ({target_file.stat().st_size} bytes)")
    return target_file


if __name__ == "__main__":
    download_or_generate_dataset()
