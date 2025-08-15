# ====== settings ======
ENV_NAME ?= DA4CHE
DOCS     ?= .
HTMLDIR  := $(DOCS)/_build/html
CACHE    := $(DOCS)/_exec_cache
PORT     ?= 8000

TOC      := _toc.yml
TOCGEN   := generate_toc.py

# Constrain native threading + headless plotting to prevent kernel crashes
ENVVARS := MPLBACKEND=Agg \
           OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# Use conda (or mamba if available) without shell activation
CONDA       := $(shell command -v conda 2>/dev/null)
MAMBA       := $(shell command -v mamba 2>/dev/null)
PKG_MGR     := $(if $(MAMBA),mamba,conda)
CONDA_RUN   := conda run -n $(ENV_NAME)

.DEFAULT_GOAL := help

.PHONY: help env build fast serve watch open clean retoc pdf conda-info

## help: Show common commands
help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' Makefile | \
		sort | awk 'BEGIN {FS=":.*?## "}; {printf "\033[36m%-14s\033[0m %s\n", $$1, $$2}'

conda-info: ## Print conda/mamba info (debug)
	@echo "conda: $(CONDA)"
	@echo "mamba: $(MAMBA)"
	@echo "env:   $(ENV_NAME)"

# ---------- Environment management ----------
env: ## Create or update the conda environment if needed
	@if ! conda env list | awk '{print $$1}' | grep -qx $(ENV_NAME); then \
	  echo ">> Creating conda env $(ENV_NAME)"; \
	  if [ -f environment.yml ]; then \
	    $(PKG_MGR) env create -n $(ENV_NAME) -f environment.yml; \
	  elif [ -f requirements.txt ]; then \
	    $(PKG_MGR) create -y -n $(ENV_NAME) python=3.10; \
	    conda run -n $(ENV_NAME) pip install -r requirements.txt; \
	  else \
	    $(PKG_MGR) create -y -n $(ENV_NAME) python=3.10 jupyter-book myst-nb jupyter-cache sphinx-autobuild ipykernel numpy pandas scipy scikit-learn matplotlib statsmodels requests openpyxl; \
	  fi; \
	else \
	  echo ">> Found conda env $(ENV_NAME)", proceeding without updating; \
	fi
	@echo ">> Ensuring kernel registration as 'Python 3'"
	@$(CONDA_RUN) python -m ipykernel install --user --name=python3 --display-name="Python 3" >/dev/null 2>&1 || true

# ---------- ToC autogeneration ----------
$(TOC): $(TOCGEN)
	@echo ">> Generating $(TOC) ..."
	@python $(TOCGEN)

retoc: ## Force-regenerate _toc.yml without a full clean
	rm -f $(TOC)
	$(MAKE) $(TOC)

# ---------- Build/serve ----------
build: env $(TOC) ## Build the book inside conda env (with execution)
	@echo ">> Building book with $(ENV_NAME) (thread-limited, headless MPL)"
	@env $(ENVVARS) $(CONDA_RUN) jupyter-book build $(DOCS) --all

fast: env $(TOC) ## Fast build (no execution) if you have _config-fast.yml
	@env $(ENVVARS) $(CONDA_RUN) jupyter-book build $(DOCS) --config _config-fast.yml

serve: build ## Build then serve & open browser
	@echo ">> Serving $(HTMLDIR) on http://localhost:$(PORT)/"
	@if command -v xdg-open >/dev/null; then xdg-open "http://localhost:$(PORT)/index.html"; \
	elif command -v open >/dev/null; then open "http://localhost:$(PORT)/index.html"; \
	else start "http://localhost:$(PORT)/index.html"; fi
	cd $(HTMLDIR) && python -m http.server $(PORT)

watch: env ## Auto-rebuild & serve (no execution; good for editing prose)
	@env $(ENVVARS) $(CONDA_RUN) sphinx-autobuild -b html $(DOCS) $(HTMLDIR) \
	  --port $(PORT) -a -E -j auto

open: ## Open the built site in your default browser
	@if command -v xdg-open >/dev/null; then xdg-open "$(HTMLDIR)/index.html"; \
	elif command -v open >/dev/null; then open "$(HTMLDIR)/index.html"; \
	else start "$(HTMLDIR)/index.html"; fi

clean: ## Remove build artifacts, caches, and _toc.yml (forces regen next build)
	rm -rf $(DOCS)/_build $(CACHE) .jupyter_cache
	rm -f $(TOC)

pdf: build ## Build a single-file PDF from HTML
	@env $(ENVVARS) $(CONDA_RUN) jupyter-book build $(DOCS) --builder pdfhtml

# ---- nbmake testing helpers (mirror full source tree into build/nbtest) ----
# Mirrors your book into $(NBTEST_WORKDIR), converts target .md files to .ipynb
# *inside* the mirror, and runs pytest+nbmake there. Avoids arrays to maximize
# portability on older /bin/bash versions.
#
# Usage:
#   make test FILE='2-regression/Topic2.2-Model_Validation.md'
#   make test FILE='2-regression/*.md'
#   make test FILE='2-regression/Topic2.{2,3}-*.md'

SHELL := /bin/bash
PYTEST ?= pytest
JUPYTEXT ?= jupytext
RSYNC ?= rsync
NBMAKE_FLAGS ?= --nbmake --nbmake-timeout=180
NBTEST_OUTDIR ?= build/nbtest
NBTEST_WORKDIR ?= $(NBTEST_OUTDIR)/work
NBTEST_SRCDIR ?= .

# Exclusions to avoid copying bulky/irrelevant dirs
RSYNC_EXCLUDES := \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '.mypy_cache' \
  --exclude '.pytest_cache' \
  --exclude '__pycache__' \
  --exclude '.ipynb_checkpoints' \
  --exclude 'node_modules' \
  --exclude '.DS_Store' \
  --exclude 'build' \
  --exclude '_build'

.PHONY: test nbtest check-tools prepare-nbtest clean-tests

check-tools:
	@command -v $(JUPYTEXT) >/dev/null || { echo "Missing 'jupytext' in this env"; exit 1; }
	@command -v $(PYTEST)   >/dev/null || { echo "Missing 'pytest' in this env";   exit 1; }
	@command -v $(RSYNC)    >/dev/null || { echo "Missing 'rsync' in this env";    exit 1; }

prepare-nbtest: check-tools
	@echo ">> Preparing nbtest workdir: $(NBTEST_WORKDIR)"
	@rm -rf "$(NBTEST_WORKDIR)"
	@mkdir -p "$(NBTEST_WORKDIR)"
	@echo ">> Mirroring source tree from $(NBTEST_SRCDIR) -> $(NBTEST_WORKDIR)"
	@$(RSYNC) -a $(RSYNC_EXCLUDES) "$(NBTEST_SRCDIR)/" "$(NBTEST_WORKDIR)/"

# User-facing alias
test: nbtest

# Core logic: operate entirely inside the mirrored tree
nbtest: prepare-nbtest
	@if [ -z "$(FILE)" ]; then \
	  echo "ERROR: provide FILE='<one or more .md/.ipynb paths or globs>'"; \
	  exit 2; \
	fi; \
	status=0; \
	cd "$(NBTEST_WORKDIR)"; \
	shopt -s nullglob; \
	for pattern in $(FILE); do \
	  for g in $$pattern; do \
	    case "$$g" in \
	      *.md) \
	        out_ipynb="$${g%.md}.ipynb"; \
	        echo ">> Converting $$g -> $$out_ipynb"; \
	        d="$${out_ipynb%/*}"; [ "$$d" = "$$out_ipynb" ] || mkdir -p "$$d"; \
	        $(JUPYTEXT) --to ipynb "$$g" -o "$$out_ipynb" ;; \
	      *.ipynb) \
	        out_ipynb="$$g"; \
	        echo ">> Using notebook $$out_ipynb" ;; \
	      *) \
	        echo ">> Skipping $$g (not .md or .ipynb)"; \
	        continue ;; \
	    esac; \
	    echo ">> Running nbmake on $$out_ipynb"; \
	    $(PYTEST) $(NBMAKE_FLAGS) "$$out_ipynb" || status=$$?; \
	  done; \
	done; \
	exit $$status

# Remove only nbmake artifacts
clean-tests:
	@echo "Removing $(NBTEST_OUTDIR) …"; \
	rm -rf "$(NBTEST_OUTDIR)"
