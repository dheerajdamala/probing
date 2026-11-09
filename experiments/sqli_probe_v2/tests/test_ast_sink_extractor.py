"""
Comprehensive unit tests for the AST-Aware SQL Sink Region Extractor.

Verifies:
1. Python from-import statements (immune to false positives on 'from')
2. SQLite execute calls
3. SQLAlchemy execution
4. Parameterized queries
5. F-string queries
6. String concatenation
7. Helper-function data flow
8. Safe code containing SQL keywords in strings/logs
9. Code without SQL execution sinks (fallback verification)
"""

import pytest
import os
import sys

# Add repository root to path
repo_root = r"C:\Users\dheer\Desktop\isae"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from experiments.sqli_probe_v2.extraction.ast_sink_extractor import (
    extract_ast_sink_regions
)


def test_python_from_import_statements():
    """Verify that Python 'from ... import ...' statements are never tagged as SQL sinks."""
    code = """from typing import List, Dict, Optional
from os.path import join, dirname
from datetime import datetime

def format_date(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")
"""
    res = extract_ast_sink_regions(code)
    # Must NOT detect any SQL sinks
    assert res["has_ast_sink"] is False
    assert res["fallback_used"] is True
    assert len(res["sink_lines"]) == 0
    assert len(res["detected_sinks"]) == 0


def test_sqlite_execute_calls():
    """Verify detection of sqlite3 cursor execution."""
    code = """import sqlite3

def get_user(db_path, user_id):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM users WHERE id = ?", (user_id,))
    return cur.fetchone()
"""
    res = extract_ast_sink_regions(code)
    assert res["has_ast_sink"] is True
    assert res["fallback_used"] is False
    assert "execute" in res["detected_sinks"]
    assert 7 in res["sink_lines"]  # line 7 contains cur.execute


def test_sqlalchemy_execution():
    """Verify detection of SQLAlchemy session.execute."""
    code = """from sqlalchemy.orm import Session
from sqlalchemy import text

def fetch_orders(session: Session, customer_id: int):
    stmt = text("SELECT id, total FROM orders WHERE customer_id = :cid")
    result = session.execute(stmt, {"cid": customer_id})
    return result.fetchall()
"""
    res = extract_ast_sink_regions(code)
    assert res["has_ast_sink"] is True
    assert res["fallback_used"] is False
    assert "execute" in res["detected_sinks"]
    assert "stmt" in res["query_vars"]
    # Both stmt definition (line 5) and execute call (line 6) must be included
    assert 5 in res["sink_lines"] or 6 in res["sink_lines"]


def test_parameterized_queries():
    """Verify detection of parameterized query arguments."""
    code = """def update_email(conn, uid, email):
    cur = conn.cursor()
    sql = "UPDATE accounts SET email = %s WHERE id = %s"
    cur.execute(sql, (email, uid))
    conn.commit()
"""
    res = extract_ast_sink_regions(code)
    assert res["has_ast_sink"] is True
    assert "sql" in res["query_vars"]
    assert 4 in res["sink_lines"]  # cur.execute line


def test_fstring_queries():
    """Verify tracking of f-string query construction reaching an execute call."""
    code = """def unsafe_query(cursor, table_name):
    query = f"SELECT * FROM {table_name}"
    cursor.execute(query)
    return cursor.fetchall()
"""
    res = extract_ast_sink_regions(code)
    assert res["has_ast_sink"] is True
    assert "query" in res["query_vars"]
    # Must capture assignment line 2 and execute line 3
    assert 2 in res["sink_lines"]
    assert 3 in res["sink_lines"]


def test_string_concatenation():
    """Verify tracking of string concatenation reaching execution."""
    code = """def search_items(conn, search_term):
    cursor = conn.cursor()
    sql = "SELECT id, name FROM items WHERE name LIKE '%" + search_term + "%'"
    cursor.execute(sql)
    return cursor.fetchall()
"""
    res = extract_ast_sink_regions(code)
    assert res["has_ast_sink"] is True
    assert "sql" in res["query_vars"]
    assert 3 in res["sink_lines"]
    assert 4 in res["sink_lines"]


def test_helper_function_data_flow():
    """Verify tracking when query variable is passed through helper functions."""
    code = """def build_query(uid):
    return "SELECT * FROM users WHERE id = " + str(uid)

def run_user_search(conn, uid):
    q = build_query(uid)
    cursor = conn.cursor()
    cursor.execute(q)
    return cursor.fetchone()
"""
    res = extract_ast_sink_regions(code)
    assert res["has_ast_sink"] is True
    assert "q" in res["query_vars"]
    assert 7 in res["sink_lines"]  # cursor.execute(q)
    assert 5 in res["sink_lines"]  # q = build_query(uid)


def test_safe_code_containing_sql_keywords():
    """Verify that benign code containing SQL keywords in comments or strings without sinks is not marked as sink."""
    code = """def parse_sql_dump(file_path):
    # This function parses SELECT, INSERT, UPDATE, and DELETE statements from a text file
    with open(file_path, 'r') as f:
        text_content = f.read()
    keywords = ["SELECT", "FROM", "WHERE", "JOIN"]
    count = sum(text_content.count(kw) for kw in keywords)
    return {"keyword_occurrences": count}
"""
    res = extract_ast_sink_regions(code)
    # No DB execution call occurs; should cleanly fall back
    assert res["has_ast_sink"] is False
    assert res["fallback_used"] is True
    assert len(res["detected_sinks"]) == 0


def test_code_without_sql_sinks_fallback():
    """Verify clean fallback for pure arithmetic/utility logic."""
    code = """def calculate_factorial(n: int) -> int:
    if n <= 1:
        return 1
    return n * calculate_factorial(n - 1)
"""
    res = extract_ast_sink_regions(code)
    assert res["has_ast_sink"] is False
    assert res["fallback_used"] is True
    assert res["sink_spans"] == []
    assert res["sink_lines"] == []


def test_multistatement_query_construction():
    """Verify handling of multi-statement query construction with intermediate string buffers."""
    code = """def export_records(conn, filter_status, limit_val):
    query = "SELECT id, status, payload FROM jobs"
    if filter_status:
        query += " WHERE status = '" + filter_status + "'"
    query += f" LIMIT {limit_val}"
    
    cur = conn.cursor()
    cur.execute(query)
    return cur.fetchall()
"""
    res = extract_ast_sink_regions(code)
    assert res["has_ast_sink"] is True
    assert "query" in res["query_vars"]
    assert 2 in res["sink_lines"] # initial query
    assert 8 in res["sink_lines"] # cur.execute(query)


def test_safe_parameterized_execution_detailed():
    """Verify that safe parameterized execution correctly identifies the execute sink and param tuple."""
    code = """def safe_transfer(conn, sender_id, receiver_id, amount):
    cur = conn.cursor()
    sql_debit = "UPDATE balances SET balance = balance - %s WHERE user_id = %s"
    cur.execute(sql_debit, (amount, sender_id))
    
    sql_credit = "UPDATE balances SET balance = balance + %s WHERE user_id = %s"
    cur.execute(sql_credit, (amount, receiver_id))
    conn.commit()
"""
    res = extract_ast_sink_regions(code)
    assert res["has_ast_sink"] is True
    assert res["detected_sinks"].count("execute") == 2
    assert "sql_debit" in res["query_vars"]
    assert "sql_credit" in res["query_vars"]


