# Top-level Makefile for common development workflows.

BUILD_DIR ?= build
CONFIG ?= Release
PREFIX ?= $(CURDIR)/install

.PHONY: help configure build test install docs clean rebuild

help:
	@echo "Available targets:"
	@echo "  configure  - Configure CMake in $(BUILD_DIR)"
	@echo "  build      - Build project"
	@echo "  test       - Run tests with CTest"
	@echo "  install    - Install to $(PREFIX)"
	@echo "  docs       - Generate Doxygen docs (requires Doxyfile)"
	@echo "  clean      - Remove $(BUILD_DIR)"
	@echo "  rebuild    - Clean, configure, and build"

configure:
	@if [ -f "$(BUILD_DIR)/CMakeCache.txt" ] && ! grep -q "CMAKE_HOME_DIRECTORY:INTERNAL=$(CURDIR)" "$(BUILD_DIR)/CMakeCache.txt"; then \
		echo "Detected stale CMake cache in $(BUILD_DIR); removing it."; \
		rm -rf "$(BUILD_DIR)"; \
	fi
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CONFIG)

build: configure
	cmake --build $(BUILD_DIR) --config $(CONFIG) --parallel

test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure -C $(CONFIG)

install: build
	cmake --install $(BUILD_DIR) --prefix $(PREFIX) --config $(CONFIG)

docs:
	doxygen Doxyfile

clean:
	rm -rf $(BUILD_DIR)

rebuild: clean build
