import time
import pytest
import sys

def run_tests():
    print("Running pytest with testpaths=tests...")
    t0 = time.time()
    code = pytest.main(["-v", "tests"])
    print(f"Pytest exited with code {code} in {time.time() - t0:.2f}s")
    sys.exit(code)

if __name__ == "__main__":
    run_tests()
