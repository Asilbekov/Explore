"""Local unit tests for the v3 harness pure functions.

Extracts the actual FunctionDef/Assign/Import nodes from
kaggle/kernel_harness_v3.py via AST and execs only those, so the tests
exercise the exact code that ships to Kaggle (no model, no GPU).
"""

import ast
import json
import sys

HARNESS = "/home/z/my-project/wet_ink/kaggle/kernel_harness_v3.py"
src = open(HARNESS).read()
tree = ast.parse(src)

WANTED_FUNCS = {
    "mcnemar_exact", "wilson", "_last_number", "_boxed_answer", "_numbers_equal",
    "score_gsm8k", "he_extract", "_run_check_program", "load_humaneval",
}
WANTED_ASSIGNS = {"HUMANEVAL_GZ_B64", "INSTRUCTION", "_NUMBER_RE", "_BOXED_RE"}

body = []
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in WANTED_FUNCS:
        body.append(node)
    elif isinstance(node, ast.Assign):
        tgt = node.targets[0]
        if isinstance(tgt, ast.Name) and tgt.id in WANTED_ASSIGNS:
            body.append(node)
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        # keep stdlib imports only (json, math, re, os, sys, tempfile, base64,
        # gzip, subprocess, time, concurrent.futures); drop heavy ML ones
        names = [a.name for a in getattr(node, "names", [])]
        mods = [getattr(n, "module", "") or "" for n in [node]] if isinstance(node, ast.ImportFrom) else names
        heavy = any(m.split(".")[0] in ("torch", "transformers", "datasets",
                                        "huggingface_hub") for m in mods)
        if not heavy:
            body.append(node)

mod = ast.Module(body=body, type_ignores=[])
ns = {}
exec(compile(mod, HARNESS, "exec"), ns)

fails = []


def check(name, cond, info=""):
    status = "PASS" if cond else "FAIL"
    if not cond:
        fails.append(name)
    print(f"[{status}] {name} {info}")


# ---------------------------------------------------------------- mcnemar_exact
m = ns["mcnemar_exact"]
check("mcnemar no discordant", m(0, 0) == 1.0)
check("mcnemar 5/0", abs(m(5, 0) - 0.0625) < 1e-12, f"got {m(5,0)}")
check("mcnemar symmetric", m(0, 5) == m(5, 0))
# exact: 2*P(Bin(12,.5)<=2) = 2*(1+12+66)/4096
check("mcnemar 10/2", abs(m(10, 2) - 2 * 79 / 4096) < 1e-12, f"got {m(10,2)}")
check("mcnemar 12/12", m(12, 12) == 1.0)

# ---------------------------------------------------------------- wilson
w = ns["wilson"]
lo, hi = w(86, 100)
check("wilson 86/100", 0.77 < lo < 0.80 and 0.90 < hi < 0.93, f"({lo:.3f},{hi:.3f})")
lo, hi = w(0, 0)
check("wilson n=0", (lo, hi) == (0.0, 1.0))

# ---------------------------------------------------------------- gsm8k scoring
s = ns["score_gsm8k"]
check("gsm8k boxed", s(r"... $$\boxed{72}$$", "72") is True)
check("gsm8k boxed last wins", s(r"\boxed{3} then \boxed{72}", "72") is True)
check("gsm8k boxed wrong", s(r"\boxed{7}", "72") is False)
check("gsm8k last number fallback", s("The answer is 72.", "72") is True)
check("gsm8k commas stripped", s("So 1,200 total.", "1200") is True)
check("gsm8k no numbers", s("I cannot solve this.", "72") is False)
check("gsm8k decimal", s("= 3.5 units", "3.5") is True)
check("gsm8k negative", s("result: -42", "-42") is True)

# ---------------------------------------------------------------- humaneval load
docs = ns["load_humaneval"]()
check("humaneval 164 docs", len(docs) == 164, f"got {len(docs)}")
check("humaneval fields", all({"task_id", "prompt", "test", "entry_point"} <= set(d) for d in docs))

# ---------------------------------------------------------------- execution runner
rc = ns["_run_check_program"]

# ---------------------------------------------------------------- he_extract (Saber verbatim)
# NOTE (verified against the published handler): the regex group starts right
# after the def signature, so the extracted completion INCLUDES the prompt's
# docstring; in the merged program the docstring appears twice, which is valid
# Python (adjacent string literals). We assert the real protocol semantics.
ex = ns["he_extract"]
doc0 = docs[0]  # has_close_elements
gen_ok = ('    for idx, elem in enumerate(numbers):\n'
          '        return True\n```\nHere is the explanation.')
c = ex(gen_ok, doc0)
check("extract includes docstring (Saber semantics)",
      c.startswith('    """ Check if in given list'), repr(c[:40]))
check("extract contains generated body", "return True" in c and "```" not in c)
gen_nofence = "    x = 1\n    return x\n"
c = ex(gen_nofence, doc0)
check("extract fallback no fence", c == gen_nofence.rstrip(), repr(c[:40]))
gen_prose_first = "Sure! Here is the solution:\n```python\n    return True\n```\n"
c = ex(gen_prose_first, doc0)
# Protocol behaviour (verified): the non-greedy regex stops at the FIRST
# "\n```" — with prose before the code fence the completion is docstring+prose
# and the code is dropped. The trailing-prompt protocol makes prose-first
# generations rare (the model continues inside the open code block).
check("extract prose-first drops code (Saber quirk)",
      "return True" not in c and "Here is the solution" in c, repr(c[:60]))

# end-to-end Saber pipeline: prompt + completion (+ duplicated docstring) must
# be executable and pass the canonical test when the generated body is correct
body_ok = ("    for idx, elem in enumerate(numbers):\n"
           "        for idx2, elem2 in enumerate(numbers):\n"
           "            if idx != idx2:\n"
           "                distance = abs(elem - elem2)\n"
           "                if distance < threshold:\n"
           "                    return True\n"
           "    return False\n```\nExplanation follows.")
completion = ex(body_ok, doc0)
program = doc0["prompt"] + completion + "\n" + doc0["test"] + "\n" + f"check({doc0['entry_point']})"
idx, ok, info = rc((7, program, 10))
check("saber pipeline end-to-end passes", ok is True, info)
check("saber pipeline docstring duplicated",
      program.count('"""') >= 4, "two docstrings present (harmless)")

# ---------------------------------------------------------------- execution semantics
prog_pass = "def f():\n    return 1\nassert f() == 1\n"
prog_fail = "def f():\n    return 1\nassert f() == 2\n"
prog_syntax = "def f(:\n"
prog_hang = "while True:\n    pass\n"
idx, ok, info = rc((1, prog_pass, 5))
check("exec pass", ok is True, info)
idx, ok, info = rc((2, prog_fail, 5))
check("exec assert-fail", ok is False, info)
idx, ok, info = rc((3, prog_syntax, 5))
check("exec syntax-error", ok is False, info)
t0 = __import__("time").time()
idx, ok, info = rc((4, prog_hang, 2))
dt = __import__("time").time() - t0
check("exec timeout killed", ok is False and info == "timeout", f"{info} dt={dt:.1f}s")
check("exec timeout prompt", 1.5 < dt < 6, f"dt={dt:.1f}s")

# real HumanEval-style check program (canonical solution must pass)
d = docs[0]
canon = d["canonical_solution"]
prog = d["prompt"] + canon + "\n" + d["test"] + "\n" + f"check({d['entry_point']})"
idx, ok, info = rc((5, prog, 10))
check("exec canonical solution passes", ok is True, info)
# broken solution must fail
prog_bad = d["prompt"] + "    return False\n" + "\n" + d["test"] + "\n" + f"check({d['entry_point']})"
idx, ok, info = rc((6, prog_bad, 10))
check("exec broken solution fails", ok is False, info)

print()
if fails:
    print("FAILED:", fails)
    sys.exit(1)
print("ALL HARNESS UNIT TESTS PASSED")
