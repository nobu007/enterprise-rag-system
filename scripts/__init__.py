# Explicit package marker.
#
# Without this file `scripts` is only a namespace package, so a foreign
# `scripts` package earlier on sys.path (e.g. a harness exporting
# PYTHONPATH=<some other repo>) shadows it and `import scripts.ingest`
# resolves to the wrong tree. A regular package here wins because pytest
# prepends the repository root to sys.path.
