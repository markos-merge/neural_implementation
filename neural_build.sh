#!/bin/bash

# neural_build.sh - C++20 project build helper
# Usage: source neural_build.sh || . neural_build.sh

# Configure alias (or: cmake --preset <debug|release|relwithdebinfo>)
alias configure_project='cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Debug -S . -B build_debug && cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release -S . -B build_release && cmake -G "Ninja" -DCMAKE_BUILD_TYPE=RelWithDebInfo -S . -B build_relwithdebinfo'

# Build aliases (no need to cd)
alias build_debug='ninja -C build_debug'
alias build_release='ninja -C build_release'
alias build_relwithdebinfo='ninja -C build_relwithdebinfo'
alias build='build_debug'

# Run aliases (no need to cd)
alias run_debug='./build_debug/neural_implementation'
alias run_release='./build_release/neural_implementation'
alias run_relwithdebinfo='./build_relwithdebinfo/neural_implementation'

# Clean alias (no need to cd)
alias clean='rm -rf build_debug/build build_release/build'

# Help alias
alias help_build='echo "Available commands:" && echo "  configure_project - Configure Debug/Release/RelWithDebInfo builds" && echo "  build_debug / build_release / build_relwithdebinfo" && echo "  run_debug / run_release / run_relwithdebinfo" && echo "  cmake --preset relwithdebinfo (VS Code / CLI)" && echo "  clean - Clean build directories"'

# Note: Run configure_project first to set up builds
