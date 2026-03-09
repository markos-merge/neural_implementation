# neural_implementation

A **neural network from scratch** in C++20: tensors (Eigen backend), layers, loss, backprop, and optimizers. Learning-oriented implementation with theory docs in `docs/`.

## 📁 Project Structure
```
neural_implementation/
├── CMakeLists.txt          # Project configuration
├── include/                # Header files
│   └── neural_network.hpp  # NeuralNetwork class declaration
├── src/                    # Source files
│   └── main.cpp            # Main implementation
├── build_debug/            # Debug build directory
├── build_release/          # Release build directory
└── neural_build.sh         # Build script with aliases
```

## 📌 Commands
```bash
# Source the build script
source neural_build.sh

# Configure Debug/Release builds
configure_project

# Build and run Debug version
build_debug

# Build and run Release version
build_release

# Run existing Debug build
run_debug

# Run existing Release build
run_release

# Clean build directories
clean

# Show help
help_build
```

## 📝 Usage Instructions
1. **First-time setup**:
   ```bash
   source neural_build.sh
   configure_project
   ```

2. **Build and run**:
   ```bash
   build_debug    # Build and run Debug version
   run_release    # Run Release version (no rebuild needed)
   ```

3. **Clean up**:
   ```bash
   clean
   ```

## 🧠 Notes
- `configure_project` sets up both Debug and Release builds
- `build_debug`/`build_release` handle full build + run
- `run_debug`/`run_release` run pre-built binaries
- Clean command removes build artifacts safely

## 📚 File Contents
- `CMakeLists.txt`: C++20 project configuration
- `neural_build.sh`: Build script with aliases for Debug/Release
- `neural_network.hpp`: NeuralNetwork class declaration
- `main.cpp`: Main implementation with print statement
