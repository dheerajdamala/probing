"""
Adversarial Benchmark Suite Builder for SQL Injection Probe V3 Extension.
Constructs 100 challenging, structurally diverse benchmark cases spanning 10 categories:
1. Direct and indirect SQL injection (15 cases)
2. Cross-function propagation (15 cases)
3. Cross-file propagation (15 cases)
4. Safe parameterized queries (10 cases)
5. Sanitization and validation (10 cases)
6. Dynamic SQL identifiers (10 cases)
7. Control-flow and reassignment (10 cases)
8. Recursion and unresolved calls (5 cases)
9. ORM and database API variations (5 cases)
10. Syntax and unsupported constructs (5 cases)
"""

import os
import json
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASES_DIR = os.path.join(BASE_DIR, "adversarial_cases")
MANIFEST_PATH = os.path.join(BASE_DIR, "adversarial_manifest.json")

def create_all_cases():
    cases_meta = []

    # =========================================================================
    # CATEGORY 1: Direct and indirect SQL injection (15 cases)
    # Target dir: adversarial_cases/vulnerable/
    # =========================================================================

    # 1. Nested call f-string
    c01 = {
        "id": "ADV_001_NESTED_FSTRING",
        "category": "direct_and_indirect_sql_injection",
        "is_project": False,
        "filename": "vulnerable/c01_nested_fstring.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Unsafe f-string directly inside execute call parameter",
        "expected_sink_line": 2,
        "expected_vuln_lines": [2],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "code": (
            "def lookup_user():\n"
            "    cursor.execute(f\"SELECT * FROM users WHERE id = '{request.args.get('id')}'\")\n"
        )
    }

    # 2. Multiline binop concatenation
    c02 = {
        "id": "ADV_002_BINOP_CONCAT_MULTILINE",
        "category": "direct_and_indirect_sql_injection",
        "is_project": False,
        "filename": "vulnerable/c02_binop_concat_multiline.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Multi-line parenthesized string concatenation with intermediate variable",
        "expected_sink_line": 5,
        "expected_vuln_lines": [3],
        "expected_subtype": "DIRECT_STRING_CONCATENATION",
        "code": (
            "def query_products(tag):\n"
            "    query = (\n"
            "        \"SELECT id, name FROM items WHERE tag = '\" + tag + \"'\"\n"
            "    )\n"
            "    db.execute(query)\n"
        )
    }

    # 3. Percent formatting tuple
    c03 = {
        "id": "ADV_003_PERCENT_TUPLE",
        "category": "direct_and_indirect_sql_injection",
        "is_project": False,
        "filename": "vulnerable/c03_percent_tuple.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Percent formatting tuple with constant and untrusted operand",
        "expected_sink_line": 3,
        "expected_vuln_lines": [2],
        "expected_subtype": "PERCENT_FORMATTING",
        "code": (
            "def auth(user, pwd):\n"
            "    sql = \"SELECT id FROM accounts WHERE u='%s' AND p='%s'\" % (user, pwd)\n"
            "    cursor.execute(sql)\n"
        )
    }

    # 4. Format kwargs
    c04 = {
        "id": "ADV_004_FORMAT_KWARGS",
        "category": "direct_and_indirect_sql_injection",
        "is_project": False,
        "filename": "vulnerable/c04_format_kwargs.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Str.format with keyword arguments",
        "expected_sink_line": 3,
        "expected_vuln_lines": [2],
        "expected_subtype": "FORMAT_METHOD",
        "code": (
            "def search(val):\n"
            "    q = \"SELECT * FROM articles WHERE title LIKE '%{k}%'\".format(k=val)\n"
            "    cursor.execute(q)\n"
        )
    }

    # 5. Format map
    c05 = {
        "id": "ADV_005_FORMAT_MAP",
        "category": "direct_and_indirect_sql_injection",
        "is_project": False,
        "filename": "vulnerable/c05_format_map.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Str.format_map with dict input",
        "expected_sink_line": 4,
        "expected_vuln_lines": [3],
        "expected_subtype": "FORMAT_METHOD",
        "code": (
            "def render_sql():\n"
            "    ctx = {'user': request.args['user']}\n"
            "    stmt = \"SELECT * FROM users WHERE username = '{user}'\".format_map(ctx)\n"
            "    conn.execute(stmt)\n"
        )
    }

    # 6. F-string with expression method call
    c06 = {
        "id": "ADV_006_FSTRING_METHOD_CALL",
        "category": "direct_and_indirect_sql_injection",
        "is_project": False,
        "filename": "vulnerable/c06_fstring_method_call.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "F-string calling .strip() on tainted input",
        "expected_sink_line": 3,
        "expected_vuln_lines": [2],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "code": (
            "def handle(val):\n"
            "    sql = f\"SELECT * FROM data WHERE token = '{val.strip()}'\"\n"
            "    cursor.execute(sql)\n"
        )
    }

    # 7. Indirect 4-variable reassignment chain
    c07 = {
        "id": "ADV_007_INDIRECT_VAR_CHAIN",
        "category": "direct_and_indirect_sql_injection",
        "is_project": False,
        "filename": "vulnerable/c07_indirect_var_chain.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Taint passed through a chain of 4 local variables",
        "expected_sink_line": 6,
        "expected_vuln_lines": [5],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "code": (
            "def chain_assign():\n"
            "    v1 = request.args['x']\n"
            "    v2 = v1\n"
            "    v3 = v2\n"
            "    sql = f\"SELECT * FROM logs WHERE entry = '{v3}'\"\n"
            "    cursor.execute(sql)\n"
        )
    }

    # 8. Aliased cursor execute method
    c08 = {
        "id": "ADV_008_ALIASED_CURSOR_CALL",
        "category": "direct_and_indirect_sql_injection",
        "is_project": False,
        "filename": "vulnerable/c08_aliased_cursor_call.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Cursor execute method aliased to local variable",
        "expected_sink_line": 4,
        "expected_vuln_lines": [3],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "code": (
            "def alias_exec(uid):\n"
            "    run = cursor.execute\n"
            "    sql = f\"SELECT * FROM t WHERE id = {uid}\"\n"
            "    run(sql)\n"
        )
    }

    # 9. String join construction
    c09 = {
        "id": "ADV_009_STRING_JOIN_CONSTRUCT",
        "category": "direct_and_indirect_sql_injection",
        "is_project": False,
        "filename": "vulnerable/c09_string_join_construct.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Join used to assemble query fragments with untrusted string",
        "expected_sink_line": 3,
        "expected_vuln_lines": [2],
        "expected_subtype": "DIRECT_STRING_CONCATENATION",
        "code": (
            "def assemble_join(user_input):\n"
            "    query = \"\".join([\"SELECT * FROM users WHERE name = '\", user_input, \"'\"])\n"
            "    cursor.execute(query)\n"
        )
    }

    # 10. Ternary conditional expression
    c10 = {
        "id": "ADV_010_TERNARY_CONDITIONAL",
        "category": "direct_and_indirect_sql_injection",
        "is_project": False,
        "filename": "vulnerable/c10_ternary_conditional.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Ternary expression selecting tainted input",
        "expected_sink_line": 4,
        "expected_vuln_lines": [3],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "code": (
            "def ternary_flow(flag):\n"
            "    val = request.args['u'] if flag else 'guest'\n"
            "    sql = f\"SELECT * FROM sessions WHERE user = '{val}'\"\n"
            "    cursor.execute(sql)\n"
        )
    }

    # 11. List item indexing
    c11 = {
        "id": "ADV_011_LIST_ITEM_INDEXING",
        "category": "direct_and_indirect_sql_injection",
        "is_project": False,
        "filename": "vulnerable/c11_list_item_indexing.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Taint stored in list and retrieved via index",
        "expected_sink_line": 4,
        "expected_vuln_lines": [3],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "code": (
            "def list_flow():\n"
            "    items = [request.args['param']]\n"
            "    sql = f\"SELECT * FROM items WHERE code = '{items[0]}'\"\n"
            "    cursor.execute(sql)\n"
        )
    }

    # 12. Dictionary lookup
    c12 = {
        "id": "ADV_012_DICT_LOOKUP",
        "category": "direct_and_indirect_sql_injection",
        "is_project": False,
        "filename": "vulnerable/c12_dict_lookup.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Taint stored in dict and looked up via key",
        "expected_sink_line": 4,
        "expected_vuln_lines": [3],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "code": (
            "def dict_flow():\n"
            "    params = {'filter': request.args['filter']}\n"
            "    sql = f\"SELECT * FROM records WHERE filter = '{params['filter']}'\"\n"
            "    cursor.execute(sql)\n"
        )
    }

    # 13. Tuple unpacking assignment
    c13 = {
        "id": "ADV_013_UNPACKING_ASSIGNMENT",
        "category": "direct_and_indirect_sql_injection",
        "is_project": False,
        "filename": "vulnerable/c13_unpacking_assignment.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Tuple unpacking assigning taint to variable",
        "expected_sink_line": 4,
        "expected_vuln_lines": [3],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "code": (
            "def unpack_flow():\n"
            "    raw_a, safe_b = request.args['x'], 100\n"
            "    sql = f\"SELECT * FROM inventory WHERE batch = '{raw_a}'\"\n"
            "    cursor.execute(sql)\n"
        )
    }

    # 14. Loop concatenation accumulator
    c14 = {
        "id": "ADV_014_LOOP_CONCAT_ACCUMULATOR",
        "category": "direct_and_indirect_sql_injection",
        "is_project": False,
        "filename": "vulnerable/c14_loop_concat_accumulator.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Untrusted input concatenated into SQL inside a loop",
        "expected_sink_line": 6,
        "expected_vuln_lines": [5],
        "expected_subtype": "DIRECT_STRING_CONCATENATION",
        "code": (
            "def loop_concat(inputs):\n"
            "    base = 'SELECT * FROM metrics WHERE id = '\n"
            "    for inp in inputs:\n"
            "        stmt = base + inp\n"
            "        cursor.execute(stmt)\n"
        )
    }

    # 15. Try-except block taint
    c15 = {
        "id": "ADV_015_TRY_EXCEPT_TAINT",
        "category": "direct_and_indirect_sql_injection",
        "is_project": False,
        "filename": "vulnerable/c15_try_except_taint.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Tainted input read in try block and executed in sink",
        "expected_sink_line": 6,
        "expected_vuln_lines": [5],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "code": (
            "def try_flow():\n"
            "    try:\n"
            "        val = request.args['secret']\n"
            "    except KeyError:\n"
            "        val = 'default'\n"
            "    sql = f\"SELECT * FROM secrets WHERE key = '{val}'\"\n"
            "    cursor.execute(sql)\n"
        )
    }

    cases_meta.extend([c01, c02, c03, c04, c05, c06, c07, c08, c09, c10, c11, c12, c13, c14, c15])

    # =========================================================================
    # CATEGORY 2: Cross-function propagation (15 cases)
    # Target dir: adversarial_cases/vulnerable/
    # =========================================================================

    # 16. 5-function deep chain
    c16 = {
        "id": "ADV_016_FIVE_FUNCTION_CHAIN",
        "category": "cross_function_propagation",
        "is_project": False,
        "filename": "vulnerable/c16_five_function_chain.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Taint passed through 5 sequential helper functions",
        "expected_sink_line": 2,
        "expected_vuln_lines": [2],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "code": (
            "def f5_sink(v):\n"
            "    cursor.execute(f\"SELECT * FROM t WHERE id = '{v}'\")\n"
            "def f4(v): f5_sink(v)\n"
            "def f3(v): f4(v)\n"
            "def f2(v): f3(v)\n"
            "def f1(v): f2(v)\n"
            "def entry(): f1(request.args['id'])\n"
        )
    }

    # 17. Helper returns untrusted input
    c17 = {
        "id": "ADV_017_HELPER_RETURNS_INPUT",
        "category": "cross_function_propagation",
        "is_project": False,
        "filename": "vulnerable/c17_helper_returns_input.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Helper returns raw HTTP request argument",
        "expected_sink_line": 6,
        "expected_vuln_lines": [5],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "code": (
            "def fetch_param():\n"
            "    return request.args.get('target')\n"
            "def handler():\n"
            "    target = fetch_param()\n"
            "    stmt = f\"SELECT * FROM nodes WHERE name = '{target}'\"\n"
            "    cursor.execute(stmt)\n"
        )
    }

    # 18. Helper builds query string
    c18 = {
        "id": "ADV_018_HELPER_BUILDS_QUERY",
        "category": "cross_function_propagation",
        "is_project": False,
        "filename": "vulnerable/c18_helper_builds_query.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Helper builds unescaped query string and caller executes",
        "expected_sink_line": 6,
        "expected_vuln_lines": [2],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "code": (
            "def build_user_query(uid):\n"
            "    return f\"SELECT * FROM users WHERE id = {uid}\"\n"
            "def handle():\n"
            "    u = request.args['id']\n"
            "    q = build_user_query(u)\n"
            "    cursor.execute(q)\n"
        )
    }

    # 19. Executor helper receives built query
    c19 = {
        "id": "ADV_019_EXECUTOR_HELPER",
        "category": "cross_function_propagation",
        "is_project": False,
        "filename": "vulnerable/c19_executor_helper.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Caller builds query and passes to execution helper",
        "expected_sink_line": 2,
        "expected_vuln_lines": [5],
        "expected_subtype": "DIRECT_STRING_CONCATENATION",
        "code": (
            "def run_raw_sql(sql_str):\n"
            "    cursor.execute(sql_str)\n"
            "def process():\n"
            "    cat = request.form['cat']\n"
            "    sql = \"SELECT * FROM catalog WHERE c = '\" + cat + \"'\"\n"
            "    run_raw_sql(sql)\n"
        )
    }

    # 20. Two helpers: builder and runner
    c20 = {
        "id": "ADV_020_TWO_HELPERS_BUILD_RUN",
        "category": "cross_function_propagation",
        "is_project": False,
        "filename": "vulnerable/c20_two_helpers_build_run.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "One helper builds query, second helper executes it",
        "expected_sink_line": 4,
        "expected_vuln_lines": [2],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "code": (
            "def builder(val):\n"
            "    return f\"SELECT * FROM data WHERE key = '{val}'\"\n"
            "def runner(query):\n"
            "    cursor.execute(query)\n"
            "def controller():\n"
            "    runner(builder(request.args['k']))\n"
        )
    }

    # 21. Multiple arguments, one tainted
    c21 = {
        "id": "ADV_021_MULTI_ARGS_ONE_TAINTED",
        "category": "cross_function_propagation",
        "is_project": False,
        "filename": "vulnerable/c21_multi_args_one_tainted.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Function receives constant table and untrusted filter",
        "expected_sink_line": 2,
        "expected_vuln_lines": [2],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "code": (
            "def find_records(table, filter_arg):\n"
            "    cursor.execute(f\"SELECT * FROM {table} WHERE val = '{filter_arg}'\")\n"
            "def route():\n"
            "    find_records('users', request.args['f'])\n"
        )
    }

    # 22. Keyword arguments propagation
    c22 = {
        "id": "ADV_022_KWARGS_PROPAGATION",
        "category": "cross_function_propagation",
        "is_project": False,
        "filename": "vulnerable/c22_kwargs_propagation.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Untrusted parameter passed as named keyword argument",
        "expected_sink_line": 2,
        "expected_vuln_lines": [2],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "code": (
            "def query_user(user_id=None):\n"
            "    cursor.execute(f\"SELECT * FROM u WHERE id = {user_id}\")\n"
            "def api():\n"
            "    query_user(user_id=request.args['id'])\n"
        )
    }

    # 23. Class method to class method
    c23 = {
        "id": "ADV_023_CLASS_METHOD_CHAIN",
        "category": "cross_function_propagation",
        "is_project": False,
        "filename": "vulnerable/c23_class_method_chain.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Self.build() passed to self.execute()",
        "expected_sink_line": 3,
        "expected_vuln_lines": [5],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "code": (
            "class Repo:\n"
            "    def exec_sql(self, sql):\n"
            "        cursor.execute(sql)\n"
            "    def get_by_id(self, uid):\n"
            "        s = f\"SELECT * FROM items WHERE id = {uid}\"\n"
            "        self.exec_sql(s)\n"
            "def run():\n"
            "    Repo().get_by_id(request.args['id'])\n"
        )
    }

    # 24. Subclass inheritance flow
    c24 = {
        "id": "ADV_024_SUBCLASS_INHERITANCE",
        "category": "cross_function_propagation",
        "is_project": False,
        "filename": "vulnerable/c24_subclass_inheritance.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Subclass calls inherited execution method",
        "expected_sink_line": 3,
        "expected_vuln_lines": [6],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "code": (
            "class BaseDB:\n"
            "    def run_query(self, query):\n"
            "        cursor.execute(query)\n"
            "class UserDB(BaseDB):\n"
            "    def find_user(self, name):\n"
            "        self.run_query(f\"SELECT * FROM users WHERE name = '{name}'\")\n"
            "def api():\n"
            "    UserDB().find_user(request.args['name'])\n"
        )
    }

    # 25. Static method propagation
    c25 = {
        "id": "ADV_025_STATIC_METHOD_FLOW",
        "category": "cross_function_propagation",
        "is_project": False,
        "filename": "vulnerable/c25_static_method_flow.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Static method query builder executed in caller",
        "expected_sink_line": 7,
        "expected_vuln_lines": [4],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "code": (
            "class QueryUtils:\n"
            "    @staticmethod\n"
            "    def make_query(val):\n"
            "        return f\"SELECT * FROM audit WHERE info = '{val}'\"\n"
            "def audit_log():\n"
            "    q = QueryUtils.make_query(request.args['info'])\n"
            "    cursor.execute(q)\n"
        )
    }

    # 26. Init stores taint, method executes
    c26 = {
        "id": "ADV_026_INIT_STORES_TAINT",
        "category": "cross_function_propagation",
        "is_project": False,
        "filename": "vulnerable/c26_init_stores_taint.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Object initializer accepts taint, method executes",
        "expected_sink_line": 5,
        "expected_vuln_lines": [5],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "code": (
            "class SearchContext:\n"
            "    def __init__(self, query_term):\n"
            "        self.term = query_term\n"
            "    def execute(self):\n"
            "        cursor.execute(f\"SELECT * FROM docs WHERE doc = '{self.term}'\")\n"
            "def search():\n"
            "    ctx = SearchContext(request.args['q'])\n"
            "    ctx.execute()\n"
        )
    }

    # 27. Generator yields tainted input
    c27 = {
        "id": "ADV_027_GENERATOR_YIELD_TAINT",
        "category": "cross_function_propagation",
        "is_project": False,
        "filename": "vulnerable/c27_generator_yield_taint.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Generator yields untrusted parameter",
        "expected_sink_line": 6,
        "expected_vuln_lines": [5],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "code": (
            "def item_generator():\n"
            "    yield request.args['id']\n"
            "def run():\n"
            "    for item in item_generator():\n"
            "        sql = f\"SELECT * FROM queue WHERE id = {item}\"\n"
            "        cursor.execute(sql)\n"
        )
    }

    # 28. Closure captures outer scope taint
    c28 = {
        "id": "ADV_028_CLOSURE_CAPTURES_TAINT",
        "category": "cross_function_propagation",
        "is_project": False,
        "filename": "vulnerable/c28_closure_captures_taint.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Nested function captures parameter from enclosing function",
        "expected_sink_line": 4,
        "expected_vuln_lines": [3],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "code": (
            "def outer_handler(user_input):\n"
            "    def inner_executor():\n"
            "        query = f\"SELECT * FROM users WHERE name = '{user_input}'\"\n"
            "        cursor.execute(query)\n"
            "    inner_executor()\n"
            "def route():\n"
            "    outer_handler(request.args['name'])\n"
        )
    }

    # 29. Callback invocation receives untrusted
    c29 = {
        "id": "ADV_029_CALLBACK_INVOCATION",
        "category": "cross_function_propagation",
        "is_project": False,
        "filename": "vulnerable/c29_callback_invocation.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Callback receives tainted input and executes query",
        "expected_sink_line": 2,
        "expected_vuln_lines": [2],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "code": (
            "def db_callback(val):\n"
            "    cursor.execute(f\"SELECT * FROM entries WHERE v = '{val}'\")\n"
            "def dispatch(cb, arg):\n"
            "    cb(arg)\n"
            "def run():\n"
            "    dispatch(db_callback, request.args['v'])\n"
        )
    }

    # 30. Varargs wrapper function
    c30 = {
        "id": "ADV_030_VARARGS_WRAPPER",
        "category": "cross_function_propagation",
        "is_project": False,
        "filename": "vulnerable/c30_varargs_wrapper.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Variable args *args passed to executor",
        "expected_sink_line": 2,
        "expected_vuln_lines": [5],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "code": (
            "def raw_exec(*args):\n"
            "    cursor.execute(args[0])\n"
            "def entry():\n"
            "    uid = request.args['id']\n"
            "    sql = f\"SELECT * FROM accounts WHERE id = {uid}\"\n"
            "    raw_exec(sql)\n"
        )
    }

    cases_meta.extend([c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30])

    # =========================================================================
    # CATEGORY 3: Cross-file propagation (15 cases)
    # Target dir: adversarial_cases/vulnerable/ or cross_file subdirs
    # =========================================================================

    # 31. Cross-file builder
    c31 = {
        "id": "ADV_031_CROSS_FILE_BUILDER",
        "category": "cross_file_propagation",
        "is_project": True,
        "filename": "vulnerable/c31_cross_file_builder",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Query builder defined in separate helper module",
        "expected_sink_line": 5,
        "expected_vuln_lines": [2],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "files": {
            "builder.py": (
                "def build_user_sql(uid):\n"
                "    return f\"SELECT * FROM users WHERE id = '{uid}'\"\n"
            ),
            "main.py": (
                "from builder import build_user_sql\n"
                "def run():\n"
                "    u = request.args['id']\n"
                "    sql = build_user_sql(u)\n"
                "    cursor.execute(sql)\n"
            )
        }
    }

    # 32. Cross-file executor
    c32 = {
        "id": "ADV_032_CROSS_FILE_EXECUTOR",
        "category": "cross_file_propagation",
        "is_project": True,
        "filename": "vulnerable/c32_cross_file_executor",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Execution helper defined in db_driver module",
        "expected_sink_line": 2,
        "expected_vuln_lines": [4],
        "expected_subtype": "DIRECT_STRING_CONCATENATION",
        "files": {
            "db_driver.py": (
                "def execute_statement(stmt):\n"
                "    cursor.execute(stmt)\n"
            ),
            "main.py": (
                "from db_driver import execute_statement\n"
                "def handle():\n"
                "    name = request.args['name']\n"
                "    query = \"SELECT * FROM staff WHERE n = '\" + name + \"'\"\n"
                "    execute_statement(query)\n"
            )
        }
    }

    # 33. Cross-file 3 tiers
    c33 = {
        "id": "ADV_033_CROSS_FILE_3_TIERS",
        "category": "cross_file_propagation",
        "is_project": True,
        "filename": "vulnerable/c33_cross_file_3_tiers",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Taint traverses web -> service -> repository modules",
        "expected_sink_line": 2,
        "expected_vuln_lines": [2],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "files": {
            "repo.py": (
                "def repo_exec(q):\n"
                "    cursor.execute(f\"SELECT * FROM records WHERE {q}\")\n"
            ),
            "service.py": (
                "from repo import repo_exec\n"
                "def service_find(criteria):\n"
                "    repo_exec(criteria)\n"
            ),
            "controller.py": (
                "from service import service_find\n"
                "def endpoint():\n"
                "    raw = request.args['filter']\n"
                "    service_find(raw)\n"
            )
        }
    }

    # 34. Cross-file relative import
    c34 = {
        "id": "ADV_034_CROSS_FILE_RELATIVE_IMPORT",
        "category": "cross_file_propagation",
        "is_project": True,
        "filename": "vulnerable/c34_cross_file_relative_import",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Relative import in package directory",
        "expected_sink_line": 5,
        "expected_vuln_lines": [2],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "files": {
            "pkg/__init__.py": "",
            "pkg/sql_utils.py": (
                "def render_clause(val):\n"
                "    return f\"SELECT * FROM items WHERE code = '{val}'\"\n"
            ),
            "pkg/views.py": (
                "from .sql_utils import render_clause\n"
                "def run_view():\n"
                "    v = request.args['code']\n"
                "    q = render_clause(v)\n"
                "    cursor.execute(q)\n"
            )
        }
    }

    # 35. Cross-file relative parent import
    c35 = {
        "id": "ADV_035_CROSS_FILE_PARENT_IMPORT",
        "category": "cross_file_propagation",
        "is_project": True,
        "filename": "vulnerable/c35_cross_file_parent_import",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Relative import from parent package",
        "expected_sink_line": 2,
        "expected_vuln_lines": [4],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "files": {
            "app/__init__.py": "",
            "app/db.py": (
                "def exec_query(sql):\n"
                "    cursor.execute(sql)\n"
            ),
            "app/sub/__init__.py": "",
            "app/sub/action.py": (
                "from ..db import exec_query\n"
                "def do_action():\n"
                "    val = request.args['x']\n"
                "    s = f\"SELECT * FROM t WHERE val = '{val}'\"\n"
                "    exec_query(s)\n"
            )
        }
    }

    # 36. Cross-file import alias
    c36 = {
        "id": "ADV_036_CROSS_FILE_ALIAS",
        "category": "cross_file_propagation",
        "is_project": True,
        "filename": "vulnerable/c36_cross_file_alias",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Imported helper used via alias name",
        "expected_sink_line": 5,
        "expected_vuln_lines": [2],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "files": {
            "queries.py": (
                "def assemble_sql(uid):\n"
                "    return f\"SELECT * FROM users WHERE id = '{uid}'\"\n"
            ),
            "main.py": (
                "from queries import assemble_sql as build_fn\n"
                "def run():\n"
                "    sql = build_fn(request.args['id'])\n"
                "    cursor.execute(sql)\n"
            )
        }
    }

    # 37. Cross-file star import
    c37 = {
        "id": "ADV_037_CROSS_FILE_STAR_IMPORT",
        "category": "cross_file_propagation",
        "is_project": True,
        "filename": "vulnerable/c37_cross_file_star_import",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Import * pulls in query execution helper",
        "expected_sink_line": 2,
        "expected_vuln_lines": [4],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "files": {
            "helpers.py": (
                "def raw_query(sql):\n"
                "    cursor.execute(sql)\n"
            ),
            "main.py": (
                "from helpers import *\n"
                "def entry():\n"
                "    inp = request.args['q']\n"
                "    s = f\"SELECT * FROM docs WHERE q = '{inp}'\"\n"
                "    raw_query(s)\n"
            )
        }
    }

    # 38. Cross-file module attribute call
    c38 = {
        "id": "ADV_038_CROSS_FILE_MOD_ATTR",
        "category": "cross_file_propagation",
        "is_project": True,
        "filename": "vulnerable/c38_cross_file_mod_attr",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Module imported and function invoked as module.func()",
        "expected_sink_line": 2,
        "expected_vuln_lines": [4],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "files": {
            "database_module.py": (
                "def query_database(statement):\n"
                "    cursor.execute(statement)\n"
            ),
            "main.py": (
                "import database_module\n"
                "def handle():\n"
                "    v = request.args['token']\n"
                "    sql = f\"SELECT * FROM auth WHERE tok = '{v}'\"\n"
                "    database_module.query_database(sql)\n"
            )
        }
    }

    # 39. Cross-file class import
    c39 = {
        "id": "ADV_039_CROSS_FILE_CLASS_IMPORT",
        "category": "cross_file_propagation",
        "is_project": True,
        "filename": "vulnerable/c39_cross_file_class_import",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Repository class imported from data layer module",
        "expected_sink_line": 3,
        "expected_vuln_lines": [5],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "files": {
            "models.py": (
                "class AccountRepository:\n"
                "    def execute(self, sql):\n"
                "        cursor.execute(sql)\n"
                "    def find_account(self, uid):\n"
                "        self.execute(f\"SELECT * FROM accounts WHERE id = '{uid}'\")\n"
            ),
            "views.py": (
                "from models import AccountRepository\n"
                "def get_account():\n"
                "    repo = AccountRepository()\n"
                "    repo.find_account(request.args['id'])\n"
            )
        }
    }

    # 40. Cross-file constant safe table
    c40 = {
        "id": "ADV_040_CROSS_FILE_SAFE_TABLE",
        "category": "cross_file_propagation",
        "is_project": True,
        "filename": "vulnerable/c40_cross_file_safe_table",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Constants file supplies table name, main supplies untrusted where",
        "expected_sink_line": 5,
        "expected_vuln_lines": [4],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "files": {
            "constants.py": (
                "USER_TABLE = 'app_users'\n"
            ),
            "main.py": (
                "from constants import USER_TABLE\n"
                "def run():\n"
                "    uid = request.args['id']\n"
                "    sql = f\"SELECT * FROM {USER_TABLE} WHERE id = '{uid}'\"\n"
                "    cursor.execute(sql)\n"
            )
        }
    }

    # 41. Cross-file 3-hop helper chain
    c41 = {
        "id": "ADV_041_CROSS_FILE_3_HOPS",
        "category": "cross_file_propagation",
        "is_project": True,
        "filename": "vulnerable/c41_cross_file_3_hops",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Taint traverses step1.py -> step2.py -> step3.py",
        "expected_sink_line": 2,
        "expected_vuln_lines": [2],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "files": {
            "step3.py": (
                "def exec_sink(v):\n"
                "    cursor.execute(f\"SELECT * FROM data WHERE v = '{v}'\")\n"
            ),
            "step2.py": (
                "from step3 import exec_sink\n"
                "def forward(v): exec_sink(v)\n"
            ),
            "step1.py": (
                "from step2 import forward\n"
                "def start(): forward(request.args['v'])\n"
            )
        }
    }

    # 42. Cross-file package init re-export
    c42 = {
        "id": "ADV_042_CROSS_FILE_INIT_REEXPORT",
        "category": "cross_file_propagation",
        "is_project": True,
        "filename": "vulnerable/c42_cross_file_init_reexport",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Function re-exported through package __init__.py",
        "expected_sink_line": 2,
        "expected_vuln_lines": [4],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "files": {
            "database/__init__.py": "from .core import run_sql",
            "database/core.py": (
                "def run_sql(s):\n"
                "    cursor.execute(s)\n"
            ),
            "main.py": (
                "from database import run_sql\n"
                "def handle():\n"
                "    val = request.args['code']\n"
                "    sql = f\"SELECT * FROM codes WHERE c = '{val}'\"\n"
                "    run_sql(sql)\n"
            )
        }
    }

    # 43. Cross-file multiple consumers
    c43 = {
        "id": "ADV_043_CROSS_FILE_MULTI_CONSUMERS",
        "category": "cross_file_propagation",
        "is_project": True,
        "filename": "vulnerable/c43_cross_file_multi_consumers",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Multiple callers in consumer.py invoke common query function",
        "expected_sink_line": 2,
        "expected_vuln_lines": [2],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "files": {
            "query_service.py": (
                "def query_by_tag(tag):\n"
                "    cursor.execute(f\"SELECT * FROM tags WHERE name = '{tag}'\")\n"
            ),
            "consumer.py": (
                "from query_service import query_by_tag\n"
                "def route1(): query_by_tag(request.args['t1'])\n"
                "def route2(): query_by_tag(request.args['t2'])\n"
            )
        }
    }

    # 44. Cross-file dataclass transfer
    c44 = {
        "id": "ADV_044_CROSS_FILE_DATACLASS",
        "category": "cross_file_propagation",
        "is_project": True,
        "filename": "vulnerable/c44_cross_file_dataclass",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Parameter transferred inside dataclass attribute across files",
        "expected_sink_line": 3,
        "expected_vuln_lines": [3],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "files": {
            "dto.py": (
                "class QueryRequest:\n"
                "    def __init__(self, key):\n"
                "        self.key = key\n"
            ),
            "db.py": (
                "def exec_request(req):\n"
                "    cursor.execute(f\"SELECT * FROM store WHERE k = '{req.key}'\")\n"
            ),
            "main.py": (
                "from dto import QueryRequest\n"
                "from db import exec_request\n"
                "def run():\n"
                "    req = QueryRequest(request.args['key'])\n"
                "    exec_request(req)\n"
            )
        }
    }

    # 45. Cross-file re-export chain
    c45 = {
        "id": "ADV_045_CROSS_FILE_REEXPORT_CHAIN",
        "category": "cross_file_propagation",
        "is_project": True,
        "filename": "vulnerable/c45_cross_file_reexport_chain",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Helper re-exported through middle module",
        "expected_sink_line": 2,
        "expected_vuln_lines": [4],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "files": {
            "raw_db.py": (
                "def execute_direct(sql):\n"
                "    cursor.execute(sql)\n"
            ),
            "facade.py": (
                "from raw_db import execute_direct as exec_fn\n"
            ),
            "client.py": (
                "from facade import exec_fn\n"
                "def action():\n"
                "    v = request.args['val']\n"
                "    stmt = f\"SELECT * FROM t WHERE val = '{v}'\"\n"
                "    exec_fn(stmt)\n"
            )
        }
    }

    cases_meta.extend([c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45])

    # =========================================================================
    # CATEGORY 4: Safe parameterized queries (10 cases)
    # Target dir: adversarial_cases/safe/
    # =========================================================================

    # 46. Safe qmark
    c46 = {
        "id": "ADV_046_SAFE_QMARK",
        "category": "safe_parameterized_queries",
        "is_project": False,
        "filename": "safe/c46_safe_qmark.py",
        "expected_label": "SAFE",
        "expected_status": "COMPLETE",
        "security_mechanism": "DB-API qmark placeholder binds parameter securely",
        "expected_sink_line": 3,
        "expected_vuln_lines": [],
        "expected_subtype": None,
        "code": (
            "def safe_qmark(user_id):\n"
            "    cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))\n"
            "def handle(): safe_qmark(request.args['id'])\n"
        )
    }

    # 47. Safe pyformat dict
    c47 = {
        "id": "ADV_047_SAFE_PYFORMAT_DICT",
        "category": "safe_parameterized_queries",
        "is_project": False,
        "filename": "safe/c47_safe_pyformat_dict.py",
        "expected_label": "SAFE",
        "expected_status": "COMPLETE",
        "security_mechanism": "PyFormat %(key)s placeholder with dict argument",
        "expected_sink_line": 4,
        "expected_vuln_lines": [],
        "expected_subtype": None,
        "code": (
            "def safe_pyformat(user_val):\n"
            "    params = {'u': user_val}\n"
            "    cursor.execute('SELECT * FROM accounts WHERE u = %(u)s', params)\n"
            "def handle(): safe_pyformat(request.args['u'])\n"
        )
    }

    # 48. Safe numeric placeholder
    c48 = {
        "id": "ADV_048_SAFE_NUMERIC",
        "category": "safe_parameterized_queries",
        "is_project": False,
        "filename": "safe/c48_safe_numeric.py",
        "expected_label": "SAFE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Numeric placeholder :1 binds argument safely",
        "expected_sink_line": 3,
        "expected_vuln_lines": [],
        "expected_subtype": None,
        "code": (
            "def safe_numeric(tok):\n"
            "    cursor.execute('SELECT * FROM tokens WHERE t = :1', (tok,))\n"
            "def handle(): safe_numeric(request.args['t'])\n"
        )
    }

    # 49. Safe multiple parameters
    c49 = {
        "id": "ADV_049_SAFE_MULTI_PARAMS",
        "category": "safe_parameterized_queries",
        "is_project": False,
        "filename": "safe/c49_safe_multi_params.py",
        "expected_label": "SAFE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Multiple parameters securely passed via tuple",
        "expected_sink_line": 3,
        "expected_vuln_lines": [],
        "expected_subtype": None,
        "code": (
            "def safe_multi(a, b):\n"
            "    cursor.execute('SELECT * FROM t WHERE a = %s AND b = %s', (a, b))\n"
            "def handle(): safe_multi(request.args['a'], request.args['b'])\n"
        )
    }

    # 50. Safe executemany batch
    c50 = {
        "id": "ADV_050_SAFE_EXECUTEMANY",
        "category": "safe_parameterized_queries",
        "is_project": False,
        "filename": "safe/c50_safe_executemany.py",
        "expected_label": "SAFE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Executemany securely binds batch of tuples",
        "expected_sink_line": 3,
        "expected_vuln_lines": [],
        "expected_subtype": None,
        "code": (
            "def safe_batch(items):\n"
            "    rows = [(i,) for i in items]\n"
            "    cursor.executemany('INSERT INTO queue VALUES (%s)', rows)\n"
            "def handle(): safe_batch(request.args.getlist('item'))\n"
        )
    }

    # 51. Safe param in helper function
    c51 = {
        "id": "ADV_051_SAFE_PARAM_HELPER",
        "category": "safe_parameterized_queries",
        "is_project": False,
        "filename": "safe/c51_safe_param_helper.py",
        "expected_label": "SAFE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Helper function encapsulates parameterized execution",
        "expected_sink_line": 2,
        "expected_vuln_lines": [],
        "expected_subtype": None,
        "code": (
            "def run_safe(uid):\n"
            "    cursor.execute('SELECT * FROM users WHERE id = %s', (uid,))\n"
            "def process():\n"
            "    u = request.args.get('id')\n"
            "    run_safe(u)\n"
        )
    }

    # 52. Safe dynamic IN placeholders
    c52 = {
        "id": "ADV_052_SAFE_DYNAMIC_IN_CLAUSE",
        "category": "safe_parameterized_queries",
        "is_project": False,
        "filename": "safe/c52_safe_dynamic_in_clause.py",
        "expected_label": "SAFE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Placeholders string generated safely, parameters passed separately",
        "expected_sink_line": 5,
        "expected_vuln_lines": [],
        "expected_subtype": None,
        "code": (
            "def query_in_clause(id_list):\n"
            "    placeholders = ', '.join(['%s'] * len(id_list))\n"
            "    stmt = f'SELECT * FROM items WHERE id IN ({placeholders})'\n"
            "    cursor.execute(stmt, id_list)\n"
            "def handle(): query_in_clause(request.args.getlist('ids'))\n"
        )
    }

    # 53. Safe repository class
    c53 = {
        "id": "ADV_053_SAFE_CLASS_REPO",
        "category": "safe_parameterized_queries",
        "is_project": False,
        "filename": "safe/c53_safe_class_repo.py",
        "expected_label": "SAFE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Repository class encapsulates parameterized queries",
        "expected_sink_line": 3,
        "expected_vuln_lines": [],
        "expected_subtype": None,
        "code": (
            "class SafeRepository:\n"
            "    def find(self, key):\n"
            "        cursor.execute('SELECT * FROM data WHERE k = %s', (key,))\n"
            "def run():\n"
            "    SafeRepository().find(request.args['k'])\n"
        )
    }

    # 54. Safe kwargs dict passing
    c54 = {
        "id": "ADV_054_SAFE_KWARGS_DICT",
        "category": "safe_parameterized_queries",
        "is_project": False,
        "filename": "safe/c54_safe_kwargs_dict.py",
        "expected_label": "SAFE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Dictionary parameters passed directly to cursor.execute",
        "expected_sink_line": 4,
        "expected_vuln_lines": [],
        "expected_subtype": None,
        "code": (
            "def execute_with_dict(val):\n"
            "    query = 'SELECT * FROM metrics WHERE metric_name = :m'\n"
            "    cursor.execute(query, {'m': val})\n"
            "def api(): execute_with_dict(request.args['m'])\n"
        )
    }

    # 55. Safe across files
    c55 = {
        "id": "ADV_055_SAFE_ACROSS_FILES",
        "category": "safe_parameterized_queries",
        "is_project": True,
        "filename": "safe/c55_safe_across_files",
        "expected_label": "SAFE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Cross-file helper safely executes parameterized query",
        "expected_sink_line": 2,
        "expected_vuln_lines": [],
        "expected_subtype": None,
        "files": {
            "db.py": (
                "def run_param(stmt, params):\n"
                "    cursor.execute(stmt, params)\n"
            ),
            "main.py": (
                "from db import run_param\n"
                "def handle():\n"
                "    uid = request.args['id']\n"
                "    run_param('SELECT * FROM users WHERE id = %s', (uid,))\n"
            )
        }
    }

    cases_meta.extend([c46, c47, c48, c49, c50, c51, c52, c53, c54, c55])

    # =========================================================================
    # CATEGORY 5: Sanitization and validation (10 cases)
    # Target dir: adversarial_cases/safe/
    # =========================================================================

    # 56. Sanitized int cast
    c56 = {
        "id": "ADV_056_SANITIZED_INT_CAST",
        "category": "sanitization_and_validation",
        "is_project": False,
        "filename": "safe/c56_sanitized_int_cast.py",
        "expected_label": "SAFE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Builtin int() cast guarantees value cannot inject SQL syntax",
        "expected_sink_line": 4,
        "expected_vuln_lines": [],
        "expected_subtype": None,
        "code": (
            "def get_user():\n"
            "    clean_id = int(request.args['id'])\n"
            "    sql = f\"SELECT * FROM users WHERE id = {clean_id}\"\n"
            "    cursor.execute(sql)\n"
        )
    }

    # 57. Sanitized float cast
    c57 = {
        "id": "ADV_057_SANITIZED_FLOAT_CAST",
        "category": "sanitization_and_validation",
        "is_project": False,
        "filename": "safe/c57_sanitized_float_cast.py",
        "expected_label": "SAFE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Builtin float() cast ensures numerical value only",
        "expected_sink_line": 4,
        "expected_vuln_lines": [],
        "expected_subtype": None,
        "code": (
            "def filter_price():\n"
            "    price = float(request.args['max_price'])\n"
            "    sql = f\"SELECT * FROM products WHERE price <= {price}\"\n"
            "    cursor.execute(sql)\n"
        )
    }

    # 58. Sanitized bool cast
    c58 = {
        "id": "ADV_058_SANITIZED_BOOL_CAST",
        "category": "sanitization_and_validation",
        "is_project": False,
        "filename": "safe/c58_sanitized_bool_cast.py",
        "expected_label": "SAFE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Builtin bool() cast produces True or False literal",
        "expected_sink_line": 4,
        "expected_vuln_lines": [],
        "expected_subtype": None,
        "code": (
            "def query_active():\n"
            "    is_active = bool(request.args.get('active'))\n"
            "    sql = f\"SELECT * FROM accounts WHERE active = {is_active}\"\n"
            "    cursor.execute(sql)\n"
        )
    }

    # 59. Sanitized uuid cast
    c59 = {
        "id": "ADV_059_SANITIZED_UUID_CAST",
        "category": "sanitization_and_validation",
        "is_project": False,
        "filename": "safe/c59_sanitized_uuid_cast.py",
        "expected_label": "SAFE",
        "expected_status": "COMPLETE",
        "security_mechanism": "UUID cast validates strict 36-char hexadecimal format",
        "expected_sink_line": 5,
        "expected_vuln_lines": [],
        "expected_subtype": None,
        "code": (
            "import uuid\n"
            "def fetch_session():\n"
            "    clean_uuid = str(uuid.UUID(request.args['token']))\n"
            "    sql = f\"SELECT * FROM sessions WHERE tok = '{clean_uuid}'\"\n"
            "    cursor.execute(sql)\n"
        )
    }

    # 60. Custom sanitizer helper
    c60 = {
        "id": "ADV_060_CUSTOM_INT_HELPER",
        "category": "sanitization_and_validation",
        "is_project": False,
        "filename": "safe/c60_custom_int_helper.py",
        "expected_label": "SAFE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Helper parse_id validates and converts input to integer",
        "expected_sink_line": 6,
        "expected_vuln_lines": [],
        "expected_subtype": None,
        "code": (
            "def clean_id(raw):\n"
            "    return int(raw)\n"
            "def run():\n"
            "    safe_id = clean_id(request.args['id'])\n"
            "    stmt = f\"SELECT * FROM t WHERE id = {safe_id}\"\n"
            "    cursor.execute(stmt)\n"
        )
    }

    # 61. Whitelist membership in list
    c61 = {
        "id": "ADV_061_WHITELIST_IN_LIST",
        "category": "sanitization_and_validation",
        "is_project": False,
        "filename": "safe/c61_whitelist_in_list.py",
        "expected_label": "SAFE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Input must be member of static ALLOWED list before execution",
        "expected_sink_line": 4,
        "expected_vuln_lines": [],
        "expected_subtype": None,
        "code": (
            "ALLOWED = ['title', 'created_at']\n"
            "def sort_posts(col):\n"
            "    if col in ALLOWED:\n"
            "        cursor.execute(f\"SELECT * FROM posts ORDER BY {col}\")\n"
            "def handle(): sort_posts(request.args['sort'])\n"
        )
    }

    # 62. Whitelist equality checks
    c62 = {
        "id": "ADV_062_WHITELIST_EQUALITY",
        "category": "sanitization_and_validation",
        "is_project": False,
        "filename": "safe/c62_whitelist_equality.py",
        "expected_label": "SAFE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Direction must equal 'asc' or 'desc' literal",
        "expected_sink_line": 4,
        "expected_vuln_lines": [],
        "expected_subtype": None,
        "code": (
            "def sort_order(direction):\n"
            "    if direction == 'asc' or direction == 'desc':\n"
            "        cursor.execute(f\"SELECT * FROM items ORDER BY id {direction}\")\n"
            "def handle(): sort_order(request.args['dir'])\n"
        )
    }

    # 63. Whitelist dict lookup mapping
    c63 = {
        "id": "ADV_063_WHITELIST_DICT_MAP",
        "category": "sanitization_and_validation",
        "is_project": False,
        "filename": "safe/c63_whitelist_dict_map.py",
        "expected_label": "SAFE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Dictionary mapping translates user input to hardcoded column",
        "expected_sink_line": 4,
        "expected_vuln_lines": [],
        "expected_subtype": None,
        "code": (
            "MAPPING = {'name': 'username', 'email': 'user_email'}\n"
            "def query_mapped(field):\n"
            "    col = MAPPING.get(field, 'id')\n"
            "    cursor.execute(f\"SELECT {col} FROM accounts\")\n"
            "def handle(): query_mapped(request.args['f'])\n"
        )
    }

    # 64. Regex digits validation
    c64 = {
        "id": "ADV_064_REGEX_DIGITS_CHECK",
        "category": "sanitization_and_validation",
        "is_project": False,
        "filename": "safe/c64_regex_digits_check.py",
        "expected_label": "SAFE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Regex guard ensures parameter contains strictly digits",
        "expected_sink_line": 5,
        "expected_vuln_lines": [],
        "expected_subtype": None,
        "code": (
            "import re\n"
            "def query_numeric(raw):\n"
            "    if re.match(r'^\\d+$', raw):\n"
            "        cursor.execute(f\"SELECT * FROM table WHERE id = {raw}\")\n"
            "def handle(): query_numeric(request.args['id'])\n"
        )
    }

    # 65. Early return guard
    c65 = {
        "id": "ADV_065_EARLY_RETURN_GUARD",
        "category": "sanitization_and_validation",
        "is_project": False,
        "filename": "safe/c65_early_return_guard.py",
        "expected_label": "SAFE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Early return when input is not in whitelist",
        "expected_sink_line": 5,
        "expected_vuln_lines": [],
        "expected_subtype": None,
        "code": (
            "VALID_STATUS = ['OPEN', 'CLOSED']\n"
            "def filter_status(s):\n"
            "    if s not in VALID_STATUS:\n"
            "        return\n"
            "    cursor.execute(f\"SELECT * FROM tickets WHERE status = '{s}'\")\n"
            "def handle(): filter_status(request.args['s'])\n"
        )
    }

    cases_meta.extend([c56, c57, c58, c59, c60, c61, c62, c63, c64, c65])

    # =========================================================================
    # CATEGORY 6: Dynamic SQL identifiers (10 cases)
    # Target dir: vulnerable/ (66-72, 75) and safe/ (73, 74)
    # =========================================================================

    # 66. Unsafe dynamic table f-string
    c66 = {
        "id": "ADV_066_DYNAMIC_TABLE_FSTRING",
        "category": "dynamic_sql_identifiers",
        "is_project": False,
        "filename": "vulnerable/c66_dynamic_table_fstring.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Unescaped table name in FROM clause",
        "expected_sink_line": 3,
        "expected_vuln_lines": [2],
        "expected_subtype": "DYNAMIC_TABLE_NAME",
        "code": (
            "def query_table(tbl):\n"
            "    sql = f\"SELECT * FROM {tbl} WHERE active = 1\"\n"
            "    cursor.execute(sql)\n"
            "def handle(): query_table(request.args['table'])\n"
        )
    }

    # 67. Unsafe dynamic table concatenation
    c67 = {
        "id": "ADV_067_DYNAMIC_TABLE_CONCAT",
        "category": "dynamic_sql_identifiers",
        "is_project": False,
        "filename": "vulnerable/c67_dynamic_table_concat.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Direct concatenation after FROM clause",
        "expected_sink_line": 3,
        "expected_vuln_lines": [2],
        "expected_subtype": "DYNAMIC_TABLE_NAME",
        "code": (
            "def fetch_from_table(tbl):\n"
            "    q = \"SELECT * FROM \" + tbl\n"
            "    cursor.execute(q)\n"
            "def handle(): fetch_from_table(request.args['t'])\n"
        )
    }

    # 68. Unsafe dynamic column selection
    c68 = {
        "id": "ADV_068_DYNAMIC_COLUMN_SELECT",
        "category": "dynamic_sql_identifiers",
        "is_project": False,
        "filename": "vulnerable/c68_dynamic_column_select.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Unescaped column identifier in SELECT clause",
        "expected_sink_line": 3,
        "expected_vuln_lines": [2],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "code": (
            "def select_column(col_name):\n"
            "    sql = f\"SELECT {col_name} FROM users\"\n"
            "    cursor.execute(sql)\n"
            "def handle(): select_column(request.args['col'])\n"
        )
    }

    # 69. Unsafe dynamic order by
    c69 = {
        "id": "ADV_069_DYNAMIC_ORDER_BY",
        "category": "dynamic_sql_identifiers",
        "is_project": False,
        "filename": "vulnerable/c69_dynamic_order_by.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Untrusted column in ORDER BY clause",
        "expected_sink_line": 3,
        "expected_vuln_lines": [2],
        "expected_subtype": "DYNAMIC_ORDER_BY",
        "code": (
            "def order_items(sort_col):\n"
            "    stmt = f\"SELECT * FROM items ORDER BY {sort_col}\"\n"
            "    cursor.execute(stmt)\n"
            "def handle(): order_items(request.args['sort'])\n"
        )
    }

    # 70. Unsafe dynamic order direction
    c70 = {
        "id": "ADV_070_DYNAMIC_ORDER_DIR",
        "category": "dynamic_sql_identifiers",
        "is_project": False,
        "filename": "vulnerable/c70_dynamic_order_dir.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Untrusted ordering direction (ASC/DESC/injection)",
        "expected_sink_line": 3,
        "expected_vuln_lines": [2],
        "expected_subtype": "DYNAMIC_ORDER_BY",
        "code": (
            "def sort_direction(direction):\n"
            "    sql = f\"SELECT * FROM logs ORDER BY ts {direction}\"\n"
            "    cursor.execute(sql)\n"
            "def handle(): sort_direction(request.args['dir'])\n"
        )
    }

    # 71. Unsafe dynamic limit clause
    c71 = {
        "id": "ADV_071_DYNAMIC_LIMIT_CLAUSE",
        "category": "dynamic_sql_identifiers",
        "is_project": False,
        "filename": "vulnerable/c71_dynamic_limit_clause.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Untrusted limit parameter interpolated without cast",
        "expected_sink_line": 3,
        "expected_vuln_lines": [2],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "code": (
            "def query_limit(lim):\n"
            "    query = f\"SELECT * FROM items LIMIT {lim}\"\n"
            "    cursor.execute(query)\n"
            "def handle(): query_limit(request.args['limit'])\n"
        )
    }

    # 72. Unsafe dynamic join table
    c72 = {
        "id": "ADV_072_DYNAMIC_JOIN_TABLE",
        "category": "dynamic_sql_identifiers",
        "is_project": False,
        "filename": "vulnerable/c72_dynamic_join_table.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Untrusted joined table name in JOIN clause",
        "expected_sink_line": 3,
        "expected_vuln_lines": [2],
        "expected_subtype": "DYNAMIC_TABLE_NAME",
        "code": (
            "def join_table(sec_tbl):\n"
            "    sql = f\"SELECT * FROM orders JOIN {sec_tbl} ON orders.id = {sec_tbl}.oid\"\n"
            "    cursor.execute(sql)\n"
            "def handle(): join_table(request.args['sec'])\n"
        )
    }

    # 73. Safe dynamic table whitelisted
    c73 = {
        "id": "ADV_073_SAFE_TABLE_WHITELISTED",
        "category": "dynamic_sql_identifiers",
        "is_project": False,
        "filename": "safe/c73_safe_table_whitelisted.py",
        "expected_label": "SAFE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Table name verified against ALLOWED_TABLES before FROM clause",
        "expected_sink_line": 4,
        "expected_vuln_lines": [],
        "expected_subtype": None,
        "code": (
            "ALLOWED_TABLES = ['users', 'admins']\n"
            "def fetch_data(t):\n"
            "    if t in ALLOWED_TABLES:\n"
            "        cursor.execute(f\"SELECT * FROM {t}\")\n"
            "def handle(): fetch_data(request.args['t'])\n"
        )
    }

    # 74. Safe dynamic order whitelisted
    c74 = {
        "id": "ADV_074_SAFE_ORDER_WHITELISTED",
        "category": "dynamic_sql_identifiers",
        "is_project": False,
        "filename": "safe/c74_safe_order_whitelisted.py",
        "expected_label": "SAFE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Sort column verified against ALLOWED_COLUMNS before ORDER BY",
        "expected_sink_line": 4,
        "expected_vuln_lines": [],
        "expected_subtype": None,
        "code": (
            "ALLOWED_COLUMNS = ['date', 'amount']\n"
            "def sort_finance(col):\n"
            "    if col in ALLOWED_COLUMNS:\n"
            "        cursor.execute(f\"SELECT * FROM txs ORDER BY {col}\")\n"
            "def handle(): sort_finance(request.args['c'])\n"
        )
    }

    # 75. Dynamic table in helper unsafe
    c75 = {
        "id": "ADV_075_DYNAMIC_TABLE_HELPER_UNSAFE",
        "category": "dynamic_sql_identifiers",
        "is_project": False,
        "filename": "vulnerable/c75_dynamic_table_helper_unsafe.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Helper accepts table name argument without validation",
        "expected_sink_line": 2,
        "expected_vuln_lines": [2],
        "expected_subtype": "DYNAMIC_TABLE_NAME",
        "code": (
            "def run_select_all(tbl_name):\n"
            "    cursor.execute(f\"SELECT * FROM {tbl_name}\")\n"
            "def handle():\n"
            "    target = request.args['target_table']\n"
            "    run_select_all(target)\n"
        )
    }

    cases_meta.extend([c66, c67, c68, c69, c70, c71, c72, c73, c74, c75])

    # =========================================================================
    # CATEGORY 7: Control-flow and reassignment (10 cases)
    # Target dir: safe/ (76, 83, 84) and vulnerable/ (77, 78, 79, 80, 81, 82, 85)
    # =========================================================================

    # 76. Overwrite taint with safe constant
    c76 = {
        "id": "ADV_076_OVERWRITE_WITH_CONSTANT",
        "category": "control_flow_and_reassignment",
        "is_project": False,
        "filename": "safe/c76_overwrite_with_constant.py",
        "expected_label": "SAFE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Tainted variable is completely reassigned to safe constant literal",
        "expected_sink_line": 5,
        "expected_vuln_lines": [],
        "expected_subtype": None,
        "code": (
            "def overwrite_flow():\n"
            "    q = f\"SELECT * FROM users WHERE id = {request.args['id']}\"\n"
            "    q = \"SELECT * FROM users WHERE id = 1\"\n"
            "    cursor.execute(q)\n"
        )
    }

    # 77. Overwrite safe with taint
    c77 = {
        "id": "ADV_077_OVERWRITE_SAFE_WITH_TAINT",
        "category": "control_flow_and_reassignment",
        "is_project": False,
        "filename": "vulnerable/c77_overwrite_safe_with_taint.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Initial constant query overwritten with tainted f-string",
        "expected_sink_line": 5,
        "expected_vuln_lines": [4],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "code": (
            "def overwrite_safe():\n"
            "    q = 'SELECT 1'\n"
            "    if True:\n"
            "        q = f\"SELECT * FROM users WHERE id = {request.args['id']}\"\n"
            "    cursor.execute(q)\n"
        )
    }

    # 78. Conditional both branches tainted
    c78 = {
        "id": "ADV_078_BOTH_BRANCHES_TAINTED",
        "category": "control_flow_and_reassignment",
        "is_project": False,
        "filename": "vulnerable/c78_both_branches_tainted.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Both conditional branches construct unsafe queries",
        "expected_sink_line": 6,
        "expected_vuln_lines": [3, 5],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "code": (
            "def both_tainted(cond):\n"
            "    if cond:\n"
            "        q = f\"SELECT * FROM a WHERE x = '{request.args['x']}'\"\n"
            "    else:\n"
            "        q = f\"SELECT * FROM b WHERE y = '{request.args['y']}'\"\n"
            "    cursor.execute(q)\n"
        )
    }

    # 79. Conditional one branch safe, one tainted
    c79 = {
        "id": "ADV_079_ONE_BRANCH_TAINTED",
        "category": "control_flow_and_reassignment",
        "is_project": False,
        "filename": "vulnerable/c79_one_branch_tainted.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "One execution branch executes tainted query string",
        "expected_sink_line": 6,
        "expected_vuln_lines": [3],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "code": (
            "def branch_tainted(cond):\n"
            "    if cond:\n"
            "        sql = f\"SELECT * FROM u WHERE id = {request.args['id']}\"\n"
            "    else:\n"
            "        sql = 'SELECT * FROM u WHERE id = 1'\n"
            "    cursor.execute(sql)\n"
        )
    }

    # 80. Taint reassigned in for loop
    c80 = {
        "id": "ADV_080_LOOP_REASSIGNMENT",
        "category": "control_flow_and_reassignment",
        "is_project": False,
        "filename": "vulnerable/c80_loop_reassignment.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Loop iteration variable holds tainted input for query construction",
        "expected_sink_line": 4,
        "expected_vuln_lines": [3],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "code": (
            "def loop_reassign(id_list):\n"
            "    for uid in id_list:\n"
            "        stmt = f\"SELECT * FROM items WHERE id = {uid}\"\n"
            "        cursor.execute(stmt)\n"
            "def handle(): loop_reassign(request.args.getlist('id'))\n"
        )
    }

    # 81. Taint in try block, executed outside
    c81 = {
        "id": "ADV_081_TRY_BLOCK_REASSIGN",
        "category": "control_flow_and_reassignment",
        "is_project": False,
        "filename": "vulnerable/c81_try_block_reassign.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Query string constructed inside try block and executed in main block",
        "expected_sink_line": 6,
        "expected_vuln_lines": [3],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "code": (
            "def try_reassign():\n"
            "    try:\n"
            "        q = f\"SELECT * FROM logs WHERE l = '{request.args['l']}'\"\n"
            "    except Exception:\n"
            "        q = 'SELECT 1'\n"
            "    cursor.execute(q)\n"
        )
    }

    # 82. Sanitization bypassed in alternate branch
    c82 = {
        "id": "ADV_082_SANITIZATION_BYPASSED",
        "category": "control_flow_and_reassignment",
        "is_project": False,
        "filename": "vulnerable/c82_sanitization_bypassed.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Sanitizing int() branch is bypassed when condition is false",
        "expected_sink_line": 6,
        "expected_vuln_lines": [5],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "code": (
            "def bypass_flow(cond, raw_val):\n"
            "    if cond:\n"
            "        val = int(raw_val)\n"
            "    else:\n"
            "        val = raw_val\n"
            "    cursor.execute(f\"SELECT * FROM data WHERE v = {val}\")\n"
            "def api(): bypass_flow(False, request.args['v'])\n"
        )
    }

    # 83. Hardcoded boolean constant
    c83 = {
        "id": "ADV_083_BOOLEAN_CONSTANT_SAFE",
        "category": "control_flow_and_reassignment",
        "is_project": False,
        "filename": "safe/c83_boolean_constant_safe.py",
        "expected_label": "SAFE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Variable is assigned boolean True constant",
        "expected_sink_line": 4,
        "expected_vuln_lines": [],
        "expected_subtype": None,
        "code": (
            "def bool_constant_flow():\n"
            "    is_admin = True\n"
            "    query = f\"SELECT * FROM users WHERE admin = {is_admin}\"\n"
            "    cursor.execute(query)\n"
        )
    }

    # 84. Reassigned to integer literal
    c84 = {
        "id": "ADV_084_REASSIGNED_TO_INT_SAFE",
        "category": "control_flow_and_reassignment",
        "is_project": False,
        "filename": "safe/c84_reassigned_to_int_safe.py",
        "expected_label": "SAFE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Tainted variable overwritten with constant int 42",
        "expected_sink_line": 5,
        "expected_vuln_lines": [],
        "expected_subtype": None,
        "code": (
            "def reassign_safe():\n"
            "    var = request.args['x']\n"
            "    var = 42\n"
            "    sql = f\"SELECT * FROM items WHERE id = {var}\"\n"
            "    cursor.execute(sql)\n"
        )
    }

    # 85. Default fallback expression
    c85 = {
        "id": "ADV_085_FALLBACK_EXPRESSION",
        "category": "control_flow_and_reassignment",
        "is_project": False,
        "filename": "vulnerable/c85_fallback_expression.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Logical OR fallback retains taint when request param is provided",
        "expected_sink_line": 4,
        "expected_vuln_lines": [3],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "code": (
            "def fallback_flow():\n"
            "    param = request.args.get('term') or 'default'\n"
            "    sql = f\"SELECT * FROM search WHERE term = '{param}'\"\n"
            "    cursor.execute(sql)\n"
        )
    }

    cases_meta.extend([c76, c77, c78, c79, c80, c81, c82, c83, c84, c85])

    # =========================================================================
    # CATEGORY 8: Recursion and unresolved calls (5 cases)
    # Target dir: vulnerable/ (86, 87) and partial/ (88, 89, 90)
    # =========================================================================

    # 86. Direct recursion tree walk
    c86 = {
        "id": "ADV_086_DIRECT_RECURSION",
        "category": "recursion_and_unresolved",
        "is_project": False,
        "filename": "vulnerable/c86_direct_recursion.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Recursive function terminates traversal and reaches unsafe sink",
        "expected_sink_line": 4,
        "expected_vuln_lines": [4],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "code": (
            "def walk_tree(node):\n"
            "    if node.has_child:\n"
            "        walk_tree(node.child)\n"
            "    cursor.execute(f\"SELECT * FROM nodes WHERE id = '{node.id}'\")\n"
            "def handle(root): walk_tree(root)\n"
        )
    }

    # 87. Mutual recursion call graph
    c87 = {
        "id": "ADV_087_MUTUAL_RECURSION",
        "category": "recursion_and_unresolved",
        "is_project": False,
        "filename": "vulnerable/c87_mutual_recursion.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Ping-pong mutual recursion cycle detected without crash",
        "expected_sink_line": 5,
        "expected_vuln_lines": [5],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "code": (
            "def ping(x): pong(x)\n"
            "def pong(y):\n"
            "    if False: ping(y)\n"
            "def entry():\n"
            "    cursor.execute(f\"SELECT * FROM t WHERE val = '{request.args['p']}'\")\n"
        )
    }

    # 88. Nested call depth exceeds max
    c88 = {
        "id": "ADV_088_DEPTH_EXCEEDS_MAX",
        "category": "recursion_and_unresolved",
        "is_project": False,
        "filename": "partial/c88_depth_exceeds_max.py",
        "expected_label": "UNCERTAIN",
        "expected_status": "PARTIAL",
        "security_mechanism": "Call chain 12 functions deep triggers depth cutoff",
        "expected_sink_line": 13,
        "expected_vuln_lines": [],
        "expected_subtype": None,
        "code": (
            "def f1(x): return f2(x)\n"
            "def f2(x): return f3(x)\n"
            "def f3(x): return f4(x)\n"
            "def f4(x): return f5(x)\n"
            "def f5(x): return f6(x)\n"
            "def f6(x): return f7(x)\n"
            "def f7(x): return f8(x)\n"
            "def f8(x): return f9(x)\n"
            "def f9(x): return f10(x)\n"
            "def f10(x): return f11(x)\n"
            "def f11(x): return f12(x)\n"
            "def f12(x): return f\"SELECT * FROM t WHERE id = '{x}'\"\n"
            "def run(): cursor.execute(f1(request.args['id']))\n"
        )
    }

    # 89. Unresolved external package import
    c89 = {
        "id": "ADV_089_UNRESOLVED_EXTERNAL_IMPORT",
        "category": "recursion_and_unresolved",
        "is_project": False,
        "filename": "partial/c89_unresolved_external_import.py",
        "expected_label": "UNCERTAIN",
        "expected_status": "PARTIAL",
        "security_mechanism": "External library helper cannot be inspected via local AST",
        "expected_sink_line": 4,
        "expected_vuln_lines": [],
        "expected_subtype": None,
        "code": (
            "from third_party_security_lib import mystery_sanitizer\n"
            "def process():\n"
            "    val = mystery_sanitizer(request.args['val'])\n"
            "    cursor.execute(val)\n"
        )
    }

    # 90. Missing imported module file
    c90 = {
        "id": "ADV_090_MISSING_IMPORTED_FILE",
        "category": "recursion_and_unresolved",
        "is_project": True,
        "filename": "partial/c90_missing_imported_file",
        "expected_label": "SAFE",
        "expected_status": "PARTIAL",
        "security_mechanism": "Missing file recorded as unresolved import in partial report",
        "expected_sink_line": 3,
        "expected_vuln_lines": [],
        "expected_subtype": None,
        "files": {
            "main.py": (
                "from non_existent_database_module import missing_func\n"
                "def run():\n"
                "    cursor.execute('SELECT 1')\n"
            )
        }
    }

    cases_meta.extend([c86, c87, c88, c89, c90])

    # =========================================================================
    # CATEGORY 9: ORM and database API variations (5 cases)
    # Target dir: vulnerable/ (91, 92, 93, 94) and safe/ (95)
    # =========================================================================

    # 91. SQLAlchemy session.execute(text(...))
    c91 = {
        "id": "ADV_091_SQLALCHEMY_TEXT_UNSAFE",
        "category": "orm_variations",
        "is_project": False,
        "filename": "vulnerable/c91_sqlalchemy_text_unsafe.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "SQLAlchemy session.execute with unescaped text f-string",
        "expected_sink_line": 3,
        "expected_vuln_lines": [2],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "code": (
            "def query_user(session, uid):\n"
            "    stmt = text(f\"SELECT * FROM users WHERE id = '{uid}'\")\n"
            "    session.execute(stmt)\n"
            "def handle(s): query_user(s, request.args['id'])\n"
        )
    }

    # 92. Django raw SQL query
    c92 = {
        "id": "ADV_092_DJANGO_RAW_UNSAFE",
        "category": "orm_variations",
        "is_project": False,
        "filename": "vulnerable/c92_django_raw_unsafe.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Django Model.objects.raw with interpolated f-string",
        "expected_sink_line": 2,
        "expected_vuln_lines": [2],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "code": (
            "def fetch_members(group_id):\n"
            "    return User.objects.raw(f\"SELECT * FROM auth_user WHERE gid = {group_id}\")\n"
            "def view(): fetch_members(request.args['gid'])\n"
        )
    }

    # 93. Django extra where clause
    c93 = {
        "id": "ADV_093_DJANGO_EXTRA_UNSAFE",
        "category": "orm_variations",
        "is_project": False,
        "filename": "vulnerable/c93_django_extra_unsafe.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Django QuerySet.extra where clause concatenation",
        "expected_sink_line": 2,
        "expected_vuln_lines": [2],
        "expected_subtype": "DIRECT_STRING_CONCATENATION",
        "code": (
            "def filter_name(val):\n"
            "    return Entry.objects.extra(where=[\"title = '\" + val + \"'\"])\n"
            "def handle(): filter_name(request.args['t'])\n"
        )
    }

    # 94. Peewee execute_sql
    c94 = {
        "id": "ADV_094_PEEWEE_RAW_UNSAFE",
        "category": "orm_variations",
        "is_project": False,
        "filename": "vulnerable/c94_peewee_raw_unsafe.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Peewee database.execute_sql with f-string interpolation",
        "expected_sink_line": 2,
        "expected_vuln_lines": [2],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "code": (
            "def raw_peewee(code):\n"
            "    db.execute_sql(f\"SELECT * FROM inventory WHERE code = '{code}'\")\n"
            "def api(): raw_peewee(request.args['c'])\n"
        )
    }

    # 95. Safe SQLAlchemy parameterized text
    c95 = {
        "id": "ADV_095_SAFE_SQLALCHEMY_PARAMS",
        "category": "orm_variations",
        "is_project": False,
        "filename": "safe/c95_safe_sqlalchemy_params.py",
        "expected_label": "SAFE",
        "expected_status": "COMPLETE",
        "security_mechanism": "SQLAlchemy text binds :id parameter securely via dict",
        "expected_sink_line": 3,
        "expected_vuln_lines": [],
        "expected_subtype": None,
        "code": (
            "def safe_sqlalchemy(session, uid):\n"
            "    stmt = text(\"SELECT * FROM users WHERE id = :id\")\n"
            "    session.execute(stmt, {'id': uid})\n"
            "def handle(s): safe_sqlalchemy(s, request.args['id'])\n"
        )
    }

    cases_meta.extend([c91, c92, c93, c94, c95])

    # =========================================================================
    # CATEGORY 10: Syntax and unsupported constructs (5 cases)
    # Target dir: edge_cases/
    # =========================================================================

    # 96. Syntax error in one multi-file project file
    c96 = {
        "id": "ADV_096_SYNTAX_ERROR_IN_PROJECT",
        "category": "syntax_and_unsupported",
        "is_project": True,
        "filename": "edge_cases/c96_syntax_error_in_project",
        "expected_label": "VULNERABLE",
        "expected_status": "PARTIAL",
        "security_mechanism": "Broken file gracefully skipped while valid file is analyzed",
        "expected_sink_line": 3,
        "expected_vuln_lines": [2],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "files": {
            "broken.py": "def broken(x: invalid syntax here!!!",
            "valid.py": (
                "def handler():\n"
                "    sql = f\"SELECT * FROM users WHERE id = '{request.args['id']}'\"\n"
                "    cursor.execute(sql)\n"
            )
        }
    }

    # 97. Empty file
    c97 = {
        "id": "ADV_097_EMPTY_FILE",
        "category": "syntax_and_unsupported",
        "is_project": False,
        "filename": "edge_cases/c97_empty_file.py",
        "expected_label": "SAFE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Empty file contains zero database sinks",
        "expected_sink_line": None,
        "expected_vuln_lines": [],
        "expected_subtype": None,
        "code": ""
    }

    # 98. File with only comments and docstrings
    c98 = {
        "id": "ADV_098_COMMENTS_ONLY",
        "category": "syntax_and_unsupported",
        "is_project": False,
        "filename": "edge_cases/c98_comments_only.py",
        "expected_label": "SAFE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Docstring-only file contains no executable statements",
        "expected_sink_line": None,
        "expected_vuln_lines": [],
        "expected_subtype": None,
        "code": (
            "\"\"\"\n"
            "Module documentation string describing the architecture.\n"
            "# Database queries will be added here in future release.\n"
            "\"\"\"\n"
        )
    }

    # 99. Exec meta call
    c99 = {
        "id": "ADV_099_EXEC_META_CALL",
        "category": "syntax_and_unsupported",
        "is_project": False,
        "filename": "edge_cases/c99_exec_meta_call.py",
        "expected_label": "SAFE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Dynamic exec() call without direct recognized database sink",
        "expected_sink_line": None,
        "expected_vuln_lines": [],
        "expected_subtype": None,
        "code": (
            "def meta_exec(cmd):\n"
            "    exec(cmd)\n"
            "def handle(): meta_exec(request.args['code'])\n"
        )
    }

    # 100. Async await database call
    c100 = {
        "id": "ADV_100_ASYNC_AWAIT_CALL",
        "category": "syntax_and_unsupported",
        "is_project": False,
        "filename": "edge_cases/c100_async_await_call.py",
        "expected_label": "VULNERABLE",
        "expected_status": "COMPLETE",
        "security_mechanism": "Asynchronous await db.execute sink with f-string interpolation",
        "expected_sink_line": 3,
        "expected_vuln_lines": [2],
        "expected_subtype": "F_STRING_INTERPOLATION",
        "code": (
            "async def async_query(db, uid):\n"
            "    sql = f\"SELECT * FROM async_data WHERE id = '{uid}'\"\n"
            "    await db.execute(sql)\n"
            "async def handle(db): await async_query(db, request.args['id'])\n"
        )
    }

    cases_meta.extend([c96, c97, c98, c99, c100])

    return cases_meta

def write_suite_to_disk():
    meta_list = create_all_cases()
    print(f"Total benchmark cases defined: {len(meta_list)}")

    written_count = 0
    manifest = []

    for item in meta_list:
        cid = item["id"]
        rel_fn = item["filename"]
        target_path = os.path.join(CASES_DIR, rel_fn)

        if item.get("is_project"):
            if os.path.exists(target_path):
                shutil.rmtree(target_path)
            os.makedirs(target_path, exist_ok=True)
            for sub_file, content in item["files"].items():
                full_sub_path = os.path.join(target_path, sub_file)
                os.makedirs(os.path.dirname(full_sub_path), exist_ok=True)
                with open(full_sub_path, "w", encoding="utf-8") as f:
                    f.write(content)
        else:
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(item.get("code", ""))

        written_count += 1
        # Store metadata in manifest without internal code payload
        m_entry = {k: v for k, v in item.items() if k not in ("code", "files")}
        manifest.append(m_entry)

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Successfully wrote {written_count} cases to {CASES_DIR}")
    print(f"Successfully generated manifest at {MANIFEST_PATH}")

if __name__ == "__main__":
    write_suite_to_disk()


if __name__ == "__main__":
    meta = create_all_cases()
    print(f"Generated {len(meta)} case metadata entries so far.")
