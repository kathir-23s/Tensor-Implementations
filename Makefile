# =============================================================================
# Configuration
# =============================================================================
CXX = g++
NVCC = nvcc
SRCDIR := src
OBJDIR := lib/objects
LIBDIR := lib
TARGET_A := $(LIBDIR)/libtensor.a
TARGET_SO := $(LIBDIR)/libtensor.so

# =============================================================================
# SLEEF Configuration (FIXED)
# =============================================================================

# Check if user wants SLEEF (default: enabled)
# Use ?= so it can be overridden by environment
USE_SLEEF ?= 1

# Export to sub-makes
export USE_SLEEF

# Silent SLEEF detection and installation
ifeq ($(USE_SLEEF),1)
    # Check if SLEEF is available via pkg-config
    SLEEF_AVAILABLE := $(shell pkg-config --exists sleef 2>/dev/null && echo 1 || echo 0)
    
    # If not available, try to install it silently
    ifeq ($(SLEEF_AVAILABLE),0)
        # Check if we can install via package manager
        ifeq ($(shell command -v apt-get 2>/dev/null),/usr/bin/apt-get)
            $(shell sudo apt-get update > /dev/null 2>&1 && sudo apt-get install -y libsleef-dev > /dev/null 2>&1 && echo 1 || echo 0)
            SLEEF_AVAILABLE := $(shell pkg-config --exists sleef 2>/dev/null && echo 1 || echo 0)
        else ifeq ($(shell command -v brew 2>/dev/null),/usr/local/bin/brew)
            $(shell brew install sleef > /dev/null 2>&1 && echo 1 || echo 0)
            SLEEF_AVAILABLE := $(shell pkg-config --exists sleef 2>/dev/null && echo 1 || echo 0)
        else ifeq ($(shell command -v dnf 2>/dev/null),/usr/bin/dnf)
            $(shell sudo dnf install -y sleef-devel > /dev/null 2>&1 && echo 1 || echo 0)
            SLEEF_AVAILABLE := $(shell pkg-config --exists sleef 2>/dev/null && echo 1 || echo 0)
        endif
    endif
    
    # Set flags based on availability
    ifeq ($(SLEEF_AVAILABLE),1)
        SLEEF_DEFINE = -DWITH_SLEEF
        SLEEF_CFLAGS = $(shell pkg-config --cflags sleef 2>/dev/null)
        SLEEF_LIBS = $(shell pkg-config --libs sleef 2>/dev/null || echo "-lsleef")
    else
        SLEEF_DEFINE =
        SLEEF_CFLAGS =
        SLEEF_LIBS =
    endif
else
    # SLEEF explicitly disabled
    SLEEF_DEFINE =
    SLEEF_CFLAGS =
    SLEEF_LIBS =
endif

CPPFLAGS = -Iinclude -I/usr/local/cuda/include -DWITH_CUDA $(SLEEF_DEFINE) $(SLEEF_CFLAGS)
CXXFLAGS = -std=c++20 -fPIC -Wall -Wextra -g -fopenmp
NVCCFLAGS = -std=c++20 -Xcompiler="-fPIC" -arch=sm_86 -g

RPATH = -Xlinker -rpath -Xlinker '$$ORIGIN/lib'
LDFLAGS = -L/usr/local/cuda/lib64 -L$(LIBDIR) $(RPATH)
LDLIBS = -lcudart -ltbb -lcurand $(SLEEF_LIBS)

# =============================================================================
# File Discovery (Automatic)
# =============================================================================
CPP_SOURCES := $(shell find $(SRCDIR) -name '*.cpp')
CU_SOURCES := $(shell find $(SRCDIR) -name '*.cu')

CU_SOURCES_FOR_DLINK := \
	src/Views/ContiguousKernel.cu \
	src/UnaryOps/cuda/ReductionKernels.cu \
	src/UnaryOps/cuda/ReductionImplGPU.cu

CU_SOURCES_REGULAR := $(filter-out $(CU_SOURCES_FOR_DLINK), $(CU_SOURCES))
OBJECTS_FROM_CPP := $(patsubst %.cpp,$(OBJDIR)/%.o,$(CPP_SOURCES))
OBJECTS_FROM_CU_REGULAR := $(patsubst %.cu,$(OBJDIR)/%.o,$(CU_SOURCES_REGULAR))
OBJECTS_FOR_DLINK := $(patsubst %.cu,$(OBJDIR)/%.o,$(CU_SOURCES_FOR_DLINK))
DEVICE_LINK_OBJ := $(OBJDIR)/device_link.o
ALL_OBJECTS := $(OBJECTS_FROM_CPP) $(OBJECTS_FROM_CU_REGULAR) $(OBJECTS_FOR_DLINK)

# =============================================================================
# Main Build Rules
# =============================================================================
.PHONY: all
all: $(TARGET_SO) $(TARGET_A)
	@echo "\n✅ Library build complete."

$(TARGET_SO): $(ALL_OBJECTS) $(DEVICE_LINK_OBJ)
	@echo "--- Creating shared library: $@"
	$(NVCC) -shared $(NVCCFLAGS) $(ALL_OBJECTS) $(DEVICE_LINK_OBJ) $(LDFLAGS) $(LDLIBS) -o $@

$(TARGET_A): $(ALL_OBJECTS)
	@echo "--- Creating static library: $@"
	ar rcs $@ $(ALL_OBJECTS)

# --- Compilation Pattern Rules ---
$(DEVICE_LINK_OBJ): $(OBJECTS_FOR_DLINK)
	@echo "--- Linking CUDA device code..."
	$(NVCC) $(NVCCFLAGS) -dlink $^ -o $@

$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(@D)
	@echo "Compiling [CXX]: $<"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(OBJECTS_FOR_DLINK): NVCC_DEVICE_FLAGS = -dc
$(OBJECTS_FOR_DLINK): $(OBJDIR)/%.o: %.cu
	@mkdir -p $(@D)
	@echo "Compiling [CUDA -dc]: $<"
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(NVCC_DEVICE_FLAGS) -c $< -o $@

$(OBJECTS_FROM_CU_REGULAR): $(OBJDIR)/%.o: %.cu
	@mkdir -p $(@D)
	@echo "Compiling [CUDA]: $<"
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c $< -o $@

# =============================================================================
# Custom Action Rules
# =============================================================================
.PHONY: run-snippet
run-snippet: $(TARGET_SO)
	@if [ -z "$(FILE)" ]; then \
		echo "ERROR: Please specify a file to run."; \
		echo "Usage: make run-snippet FILE=path/to/your/file.cpp"; \
		exit 1; \
	fi
	@echo "--- Compiling snippet: $(FILE)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o snippet_runner $(FILE) $(LDFLAGS) -ltensor $(LDLIBS)
	@echo "--- Running snippet_runner"
	./snippet_runner
	@rm -f snippet_runner

.PHONY: rebuild
rebuild:
	@$(MAKE) clean
	@$(MAKE) all

.PHONY: clean
clean:
	@echo "--- Cleaning build files"
	rm -rf $(OBJDIR) $(TARGET_A) $(TARGET_SO)

# =============================================================================
# Build Variants (FIXED)
# =============================================================================
.PHONY: with-sleef
with-sleef:
	@echo "Building WITH SLEEF vectorization..."
	@$(MAKE) clean
	@$(MAKE) USE_SLEEF=1

.PHONY: without-sleef
without-sleef:
	@echo "Building WITHOUT SLEEF (standard library)..."
	@$(MAKE) clean
	@$(MAKE) USE_SLEEF=0

.PHONY: check-sleef
check-sleef:
	@echo "SLEEF Status:"
	@if pkg-config --exists sleef 2>/dev/null; then \
		echo "  ✅ Available"; \
		echo "  CFLAGS: $(shell pkg-config --cflags sleef 2>/dev/null)"; \
		echo "  LIBS: $(shell pkg-config --libs sleef 2>/dev/null)"; \
	else \
		echo "  ❌ Not available"; \
	fi
	@echo "Current build configuration:"
	@echo "  USE_SLEEF: $(USE_SLEEF)"
	@echo "  SLEEF_DEFINE: $(SLEEF_DEFINE)"
	@echo "  SLEEF_LIBS: $(SLEEF_LIBS)"

# =============================================================================
# Help
# =============================================================================
.PHONY: help
help:
	@echo "Build targets:"
	@echo "  all              - Build libraries (SLEEF auto-detected)"
	@echo "  with-sleef       - Build with SLEEF vectorization"
	@echo "  without-sleef    - Build without SLEEF"
	@echo "  rebuild          - Clean and rebuild"
	@echo "  clean            - Remove build artifacts"
	@echo ""
	@echo "Development:"
	@echo "  run-snippet FILE=path - Compile and run test file"
	@echo "  check-sleef      - Show SLEEF status"
	@echo ""
	@echo "Features:"
	@echo "  • SLEEF auto-detection and installation"
	@echo "  • Vectorized math when available"
	@echo "  • Silent operation by default"