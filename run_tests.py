# run_tests.py
import sys
import pytest

if __name__ == "__main__":
    # -q: quiet, --disable-warnings, --maxfail=1 stops on first failure
    sys.exit(pytest.main(["-q", "--disable-warnings", "--maxfail=1"]))