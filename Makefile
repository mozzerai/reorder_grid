.PHONY: help setup doctor update outdated clean fmt fmt-check analyze check test test-arch test-file test-name coverage gate validate pre-commit imports imports-package doc publish-check

# Default target
help:
	@echo "reorder_grid (Flutter package) Makefile"
	@echo "---------------------------------------"
	@echo "Setup Commands:"
	@echo "  setup           - Install dependencies"
	@echo "  doctor          - Run Flutter doctor to check environment"
	@echo "  update          - Upgrade dependencies and raise pubspec lower bounds"
	@echo "  outdated        - Check for outdated packages"
	@echo "  clean           - Clean build artifacts and caches"
	@echo ""
	@echo "Gate Commands (what a release has to pass):"
	@echo "  gate            - Run the full gate: fmt-check + analyze + test"
	@echo "  validate        - Alias for gate"
	@echo "  fmt-check       - Fail if lib/, test/ or tool/ is unformatted"
	@echo "  analyze         - flutter analyze --fatal-infos --fatal-warnings"
	@echo "  test            - Run all tests"
	@echo "  pre-commit      - Format, organize imports, then run the gate"
	@echo ""
	@echo "Development Commands:"
	@echo "  fmt             - Format lib/, test/ and tool/"
	@echo "  check           - Alias for analyze"
	@echo "  test-arch       - Run only the architecture tests"
	@echo "  test-file       - Run one test file (FILE=test/reorder_grid_test.dart)"
	@echo "  test-name       - Run tests by name (NAME='lands the dropped tile in its slot')"
	@echo "  coverage        - Run the tests and write coverage/lcov.info"
	@echo "  imports         - Fix directive order/grouping in lib/"
	@echo "  imports-package - Rewrite relative imports to package:reorder_grid/..."
	@echo ""
	@echo "Release Commands:"
	@echo "  doc             - Generate the API docs into doc/api/"
	@echo "  publish-check   - Dry-run the pub.dev publish"

# Install dependencies
setup:
	@echo "Installing dependencies..."
	@flutter pub get
	@echo "Dependencies installed"

# Check Flutter environment
doctor:
	@flutter doctor -v

# Upgrade dependencies, in two passes, because the flags do not compose:
# `--major-versions` rewrites a constraint only when the major itself moves, and
# it suppresses `--tighten` when both are passed. Without the second pass a
# minor bump resolves and is tested but never reaches pubspec.yaml, leaving the
# published lower bound pointing at a version this package never ran against.
#
# The trailing "N packages have newer versions incompatible with dependency
# constraints" is usually about transitive packages the Flutter SDK pins
# (material_color_utilities, test_api). Nothing here can move those — run
# `make outdated` to see whether any DIRECT dependency is actually behind.
update:
	@echo "Upgrading dependencies..."
	@flutter pub upgrade --major-versions
	@flutter pub upgrade --tighten

# Check for outdated packages
outdated:
	@flutter pub outdated

# Clean build artifacts and caches.
# pubspec.lock is gitignored here: libraries do not commit a lock file, so a
# clean is always followed by a fresh resolve.
clean:
	@echo "Cleaning build artifacts..."
	@flutter clean
	@rm -rf build/
	@rm -rf .dart_tool/
	@rm -rf coverage/
	@rm -rf doc/api/
	@echo "Clean complete"

# Format code
fmt:
	@dart format lib test tool

# Formatting check, exactly as the gate runs it
fmt-check:
	@dart format --output=none --set-exit-if-changed lib test tool

# Analyze. `--fatal-infos` because the curated lint set in analysis_options.yaml
# reports at info level and the package is expected to stay at zero.
analyze:
	@flutter analyze --fatal-infos --fatal-warnings

# Alias kept for muscle memory with the consumer apps
check: analyze

# Run all tests (includes test/architecture/)
test:
	@flutter test

# Run only the architecture tests: the import order/style rules and the
# setState ban.
test-arch:
	@echo "🔍 Validando regras de arquitetura..."
	@flutter test test/architecture/
	@echo "✅ Testes arquiteturais passaram!"

# Run a single test file: make test-file FILE=test/reorder_grid_test.dart
test-file:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make test-file FILE=test/reorder_grid_test.dart"; \
		exit 1; \
	fi
	@flutter test $(FILE)

# Run tests by name: make test-name NAME='lands the dropped tile in its slot'
test-name:
	@if [ -z "$(NAME)" ]; then \
		echo "Usage: make test-name NAME='lands the dropped tile in its slot'"; \
		exit 1; \
	fi
	@flutter test --plain-name "$(NAME)"

# Line coverage into coverage/lcov.info. `genhtml coverage/lcov.info -o
# coverage/html` turns it into a browsable report.
coverage:
	@flutter test --coverage
	@echo "Coverage written to coverage/lcov.info"

# The gate every change has to pass. Plain `flutter analyze` is looser than
# this — run it before pushing or publishing.
gate: fmt-check analyze test
	@echo "✅ Core validation complete!"

validate: gate

# Fix what is fixable, then prove the gate passes
pre-commit: fmt imports
	@$(MAKE) --no-print-directory gate
	@echo "✅ Pronto para commit!"

# Fix directive order/grouping in lib/ (enforced by import_order_test.dart)
imports:
	@dart run tool/organize_imports.dart

# Rewrite relative imports to package:reorder_grid/...
# (enforced by import_style_test.dart)
imports-package:
	@dart run tool/rewrite_imports.dart

# Generate the API docs pub.dev will build on publish
doc:
	@dart doc
	@echo "Docs written to doc/api/"

# Everything pub.dev checks, without uploading. Run after `gate`.
publish-check:
	@flutter pub publish --dry-run
