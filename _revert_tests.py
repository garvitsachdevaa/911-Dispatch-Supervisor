import re

with open('tests/test_inference.py', 'r') as f:
    code = f.read()

# block 1
code = re.sub(
    r'for line in stdout.split\(\"\\n\"\):\n\s+if \'\"type\": \"START\"\' in line:\n\s+try:\n\s+import json\n\s+d = json\.loads\(line\)\n\s+tasks_run\.append\(d\.get\(\"task\"\)\)\n\s+except:\n\s+pass',
    r'for line in stdout.split("\n"):\n            if line.startswith("[START]"):\n                match = re.match(r"\[START\] task=(\S+) env=(\S+) model=(\S+)", line)\n                assert match\n                tasks_run.append(match.group(1))',
    code
)

# block 2
code = re.sub(
    r'for line in stdout\.split\(\"\\n\"\):\n\s+if \'\"type\": \"START\"\' in line:\n\s+import json\n\s+d = json\.loads\(line\)\n\s+assert d\.get\(\"task\"\) in self\.TASK_IDS\n\s+assert d\.get\(\"env\"\) == \"citywide-dispatch-supervisor\"\n\s+assert d\.get\(\"model\"\) == \"test-model\"',
    r'pattern = r"\[START\] task=\S+ env=citywide-dispatch-supervisor model=\S+"\n        for line in stdout.split("\n"):\n            if line.startswith("[START]"):\n                assert re.match(pattern, line)',
    code
)

# block 3
code = re.sub(
    r'valid_errors = \{None, \"max_steps_exceeded\", \"illegal_transition\", \"step_error\"\}\n\s+for line in stdout\.split\(\"\\n\"\):\n\s+if \'\"type\": \"STEP\"\' in line:\n\s+import json\n\s+d = json\.loads\(line\)\n\s+assert d\.get\(\"error\"\) in valid_errors or isinstance\(d\.get\(\"error\"\), str\)',
    r'valid_errors = {"null", "max_steps_exceeded", "illegal_transition", "step_error"}\n        for line in stdout.split("\n"):\n            if not line.startswith("[STEP]"):\n                continue\n            match = re.match(r"\[STEP\].+ error=(.+)", line)\n            assert match\n            assert match.group(1) in valid_errors',
    code
)

with open('tests/test_inference.py', 'w') as f:
    f.write(code)

print('tests done reverting')
