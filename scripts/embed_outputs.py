"""Embed docs/output/*.txt into markdown between <!-- OUTPUT:name --> ... <!-- /OUTPUT --> markers.
--check exits 1 if any embedded block differs from its file."""
import re
import sys
from pathlib import Path

pat = re.compile(r"(<!-- OUTPUT:(\S+) -->\n)(.*?)(<!-- /OUTPUT -->)", re.DOTALL)
stale = 0
for md in [Path("README.md"), *Path("docs").glob("*.md")]:
    text = md.read_text()

    def sub(m):
        body = "```text\n" + Path("docs/output", m.group(2)).read_text().rstrip() + "\n```\n"
        return m.group(1) + body + m.group(4)

    new = pat.sub(sub, text)
    if new != text:
        if "--check" in sys.argv:
            print(f"stale embedded output in {md}")
            stale = 1
        else:
            md.write_text(new)
sys.exit(stale)
