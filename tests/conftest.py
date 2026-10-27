import sys
from pathlib import Path

# Ensure project root is in sys.path during test execution
ROOT_DIR = Path(__file__).parent.parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
