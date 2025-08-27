import os, sys
# Ensure repository root on sys.path for 'src' package resolution
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
