"""Fail if a routine in theblas.h has no theblas_-prefixed counterpart in theblas_c.h."""

import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
cpp = (root / "include/theblas/theblas.h").read_text()
c = (root / "bindings/c/include/theblas/theblas_c.h").read_text()

cpp = re.sub(r"/\*.*?\*/|//[^\n]*", "", cpp, flags=re.S)
cpp_names = set(re.findall(r"\b(\w+)\s*\(", cpp)) - {"void"}
c_names = {name.removeprefix("theblas_") for name in re.findall(r"\b(theblas_\w+)\s*\(", c)}

missing = sorted(cpp_names - c_names)
if missing:
    sys.exit(f"theblas_c.h is missing wrappers for: {', '.join(missing)}")
print(f"theblas_c.h covers all {len(cpp_names)} routines of theblas.h")
