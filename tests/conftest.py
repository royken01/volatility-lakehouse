"""The tests never connect anywhere: dummy catalog settings, and the framework settings come from the
built-in defaults (no user framework.toml, no LAKEHOUSE_* overrides)."""
import os
import tempfile

for name in ("R2_CATALOG_URI", "R2_CATALOG_WAREHOUSE", "R2_CATALOG_TOKEN", "R2_ENDPOINT", "R2_ACCESS_KEY_ID",
             "R2_SECRET_ACCESS_KEY"):
    os.environ.setdefault(name, "unused-in-tests")
for name in list(os.environ):
    if name.startswith("LAKEHOUSE_"):
        del os.environ[name]
os.environ["LAKEHOUSE_HOME"] = tempfile.mkdtemp(prefix="lakehouse-tests-")
