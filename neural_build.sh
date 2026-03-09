#!/bin/bash

# neural_build.sh - C++20 project build helper
# Usage: source neural_build.sh || . neural_build.sh

# Configure alias
alias configure_project='cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Debug -S . -B build_debug && cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release -S . -B build_release'

# Build aliases (no need to cd)
alias build_debug='ninja -C build_debug'
alias build_release='ninja -C build_release'
alias build='build_debug'

# Run aliases (no need to cd)
alias run_debug='./build_debug/neural_implementation'
alias run_release='./build_release/neural_implementation'

# Clean alias (no need to cd)
alias clean='rm -rf build_debug/build build_release/build'

# Help alias
alias help_build='echo "Available commands:" && echo "  configure_project - Configure Debug/Release builds" && echo "  build/debug - Build and run Debug version" && echo "  build/release - Build and run Release version" && echo "  run/debug - Run Debug version" && echo "  run/release - Run Release version" && echo "  clean - Clean build directories"'

# Note: Run configure_project first to set up builds
