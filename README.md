# TensorLib: Tensor and Operations Team

`TensorLib` is a C++ library for tensor manipulations with an emphasis on performance, featuring both CPU and CUDA GPU backends.

This explains how to build `TensorLib` and link it against your own C++ projects.

## Requirements
CUDA Toolkit version used: 13.0
CUDA Toolkit Required version (minimum): >12.5

g++ Version: > 11



## Part 1: Building the Library

The first step is to compile `TensorLib` into shared (`.so`) and static (`.a`) library files.

### Prerequisites

Ensure the following are installed on your system:
*   A C++ compiler supporting C++20 (e.g., `g++` 10+)
*   The NVIDIA CUDA Toolkit (for `nvcc`)
*   GNU Make
*   Intel TBB (`libtbb-dev`)

### Compilation Steps

1.  **Clone the Repository**
    Open your terminal and clone this repository to your local machine.
    ```bash
    git clone https://github.com/kathir-23s/Tensor-Implementations.git
    ```

2.  **Navigate to the Directory**
    ```bash
    cd TensorLib/
    ```

3.  **Build the Library**
    Run the `make rebuild` command. This performs a clean, full compilation of the library.
    ```bash
    make rebuild
    ```

4.  **Verify the Output**
    After the build completes, the compiled libraries will be located in the `lib/` directory. You should see:
    *   `lib/libtensor.so`: The shared library.
    *   `lib/libtensor.a`: The static library.

The library is now ready to be used in other projects.

---

## Part 2: Using the Library in Your Project

Once `TensorLib` is built, you can link it against your own C++ applications.

### Directory Structure (Example)

Let's imagine you have a new project with the following structure. You've placed the compiled `TensorLib` folder inside a `vendor/` directory.

```
my_awesome_project/
├── my_app.cpp
└── vendor/
    └── TensorLib/
        ├── include/
        │   └── TensorLib.h
        └── lib/
            ├── libtensor.so
            └── libtensor.a
```

### Compilation Command

To compile `my_app.cpp`, you need to tell your compiler where to find the `TensorLib` header files (`-I`) and library files (`-L`), and which library to link (`-ltensor`).

Here is a complete example command. **Remember to replace `/path/to/` with the actual path to your project.**

```bash
g++ -std=c++20 \
    -I/path/to/my_awesome_project/vendor/TensorLib/include \
    -L/path/to/my_awesome_project/vendor/TensorLib/lib \
    my_app.cpp \
    -o my_app \
    -ltensor \
    -Wl,-rpath,'$ORIGIN/vendor/TensorLib/lib'
```

#### Breakdown of the Command:
*   `-I.../TensorLib/include`: Tells the compiler where to find `#include "TensorLib.h"`.
*   `-L.../TensorLib/lib`: Tells the linker where to find the library files.
*   `my_app.cpp`: Your source code.
*   `-o my_app`: The name of your final executable program.
*   `-ltensor`: Links your program with `libtensor.so`.
*   `-Wl,-rpath,'$ORIGIN/vendor/TensorLib/lib'`: **(Important)** This embeds the relative path to `libtensor.so` inside your executable. This allows you to run `./my_app` without setting `LD_LIBRARY_PATH`.

#### Compile using make
* You can also compile files using the make command
``` 
make run-snippet FILE=<filename>
```
* This will create a temporary object file called snippet_runner which will run and show you output and delete by itself

### Verifying if the library is compiled properly and available to use

* Once imported and run ```make rebuild``` command on terminal, to verify if its working properly.
* Run this command ```make run-snippet FILE=starter_examples/test_library_setup.cpp ```
* Verify if the terminal showing the following output:

```
--- Compiling snippet: Example/test_library_setup.cpp ---
g++ -Iinclude -I/usr/local/cuda/include -DWITH_CUDA -std=c++20 -fPIC -Wall -Wextra -g -fopenmp -o snippet_runner Example/test_library_setup.cpp -L/usr/local/cuda/lib64 -Llib -Xlinker -rpath -Xlinker '$ORIGIN/lib' -ltensor -lcudart -ltbb -lcurand

--- Running snippet_runner ---
./snippet_runner
--- Code is Running ---
Testing TensorLib Integration
Successfully created and displayed a Tensor:
Tensor(shape=(2, 2), dtype=float32, device='cpu')
[[3.1400, 3.1400],
 [3.1400, 3.1400]]

---  TensorLib Integration Test Successful ---



--- Cleaning up snippet ---
rm -f snippet_runner
```
* If so, then the library has been compiled properly and available to use
