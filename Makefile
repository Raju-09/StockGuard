# StockGuard — common commands
# Usage:  make <target>
# Windows users: run these from Git Bash or WSL if `make` isn't on PATH.
# PowerShell alternative for each target is shown in a comment.

PYTHON  = .\venv\Scripts\python.exe
PIP     = .\venv\Scripts\pip.exe
RUFF    = .\venv\Scripts\ruff.exe
PYTEST  = .\venv\Scripts\pytest.exe

.PHONY: install simulate lint format test clean help

## install   Install all dependencies (including dev tools) into the venv.
install:
	$(PIP) install -e ".[dev]"

## simulate  Regenerate the simulated dataset CSVs.
simulate:
	$(PYTHON) src/simulate.py

## lint      Run ruff linter. Fails on any error.
lint:
	$(RUFF) check src/ tests/

## format    Auto-fix safe lint issues + format code.
format:
	$(RUFF) check --fix src/ tests/
	$(RUFF) format src/ tests/

## test      Run all tests with coverage summary.
test:
	$(PYTEST) tests/ -v --tb=short

## clean     Remove Python caches and test artifacts.
clean:
	if exist __pycache__ rmdir /s /q __pycache__
	if exist .ruff_cache rmdir /s /q .ruff_cache
	if exist .pytest_cache rmdir /s /q .pytest_cache
	if exist htmlcov rmdir /s /q htmlcov
	for /r %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"

## help      Show this help message.
help:
	@findstr /r "^## " Makefile
