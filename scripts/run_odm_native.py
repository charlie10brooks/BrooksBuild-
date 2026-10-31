"""Run the ODM installed in this image the same way BrooksBuild does (native_runner + native_env). Usage: run_odm_native.py <project-parent> <name> [odm options]"""
import subprocess
import sys
from types import SimpleNamespace

sys.path.insert(0, "/app/backend")
from app.config import Settings
from app.services import odm

app = SimpleNamespace(settings=Settings())
run = odm.native_runner(app)
if not run:
    sys.exit("native ODM not found (TERRASCOPE_ODM_HOME=%s)" % app.settings.odm_home)
parent, name, *opts = sys.argv[1:]
cmd = [*run, "--project-path", parent, *opts, name]
print("$", " ".join(cmd), flush=True)
sys.exit(subprocess.call(cmd, env=odm.native_env(app)))
