"""Compatibility entry point; use report.py safety for new workflows."""
from reporting.examples import safe_window
from reporting.safety import report

if __name__ == "__main__":
    from report import main
    main(["safety", *__import__("sys").argv[1:]])
