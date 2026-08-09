BLUE := \033[1;34m
GREEN := \033[1;32m
RESET := \033[0m
PYTHON_VERSIONS := 3.11.13 3.12.10 3.13.9

.PHONY: format check-formatted check check-matrix tests check-coverage docs all prod pre-commit

format:
	@printf "$(BLUE)==>$(RESET) Formatting code with ruff...\n"
	@uv run --extra dev ruff format
	@printf "$(GREEN)✔ Formatting complete.$(RESET)\n"

check-formatted:
	@printf "$(BLUE)==>$(RESET) Checking formatting with ruff format --check...\n"
	@uv run --extra dev ruff format --check
	@printf "$(GREEN)✔ Formatting check complete.$(RESET)\n"

check:
	@printf "$(BLUE)==>$(RESET) Running ruff checks...\n"
	@uv run --extra dev ruff check
	@printf "$(BLUE)==>$(RESET) Type checking with ty...\n"
	@uv run --extra dev ty check
	@printf "$(BLUE)==>$(RESET) Type checking with basedpyright...\n"
	@uv run --extra dev basedpyright
	@printf "$(GREEN)✔ Checking complete.$(RESET)\n"

check-matrix:
	@for version in $(PYTHON_VERSIONS); do \
		short_version=$${version%.*}; \
		printf "$(BLUE)==>$(RESET) Running validation matrix for Python $$version...\n"; \
		uv run --extra dev --python $$version ruff check src tests || exit $$?; \
		uv run --extra dev --python $$version ty check --python-version $$short_version || exit $$?; \
		uv run --extra dev --python $$version basedpyright --pythonversion $$short_version src tests || exit $$?; \
	done
	@printf "$(GREEN)✔ Matrix checking complete.$(RESET)\n"

tests:
	@printf "$(BLUE)==>$(RESET) Running tests with pytest...\n"
	@uv run --extra dev pytest
	@printf "$(GREEN)✔ Tests complete.$(RESET)\n"

check-coverage:
	@printf "$(BLUE)==>$(RESET) Checking 100%% line and branch coverage...\n"
	@uv run --extra dev pytest --cov=. -q
	@printf "$(GREEN)✔ Coverage thresholds satisfied.$(RESET)\n"

docs-llms:
	@printf "$(BLUE)==>$(RESET) Generating LLM documentation bundle...\n"
	@uv run --extra docs python scripts/llms.py --write
	@printf "$(GREEN)✔ LLM documentation bundle generated.$(RESET)\n"

docs:
	@uv run --extra docs python scripts/llms.py --check
	@uv run --extra docs python scripts/llms.py --stage
	@printf "$(BLUE)==>$(RESET) Building Zensical site in strict mode...\n"
	@uv run --extra docs zensical build --clean --strict
	@printf "$(GREEN)✔ Documentation build complete.$(RESET)\n"

docs-serve: docs
	@printf "$(BLUE)==>$(RESET) Serving Zensical documentation...\n"
	@uv run --extra docs zensical serve --open

all: format check

prod: tests check-coverage format check-matrix docs

pre-commit:
	@printf "$(BLUE)==>$(RESET) Running pre-commit checks...\n"
	@uv run --extra dev pre-commit run --all-files
	@printf "$(GREEN)✔ Pre-commit checks complete.$(RESET)\n"
