"""Print the real collected test count (used by README and the docs check)."""
import re
import subprocess
import sys

out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], check=False, capture_output=True, text=True).stdout
m = re.search(r"(\d+) tests? collected", out)
if not m:
    sys.exit("could not determine test count:\n" + out)
print(m.group(1))
