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

CPPFLAGS = -Iinclude -I/usr/local/cuda/include -DWITH_CUDA
CXXFLAGS = -std=c++20 -fPIC -Wall -Wextra -g -fopenmp
NVCCFLAGS = -std=c++20 -Xcompiler="-fPIC" -arch=sm_86 -g

RPATH = -Xlinker -rpath -Xlinker '$$ORIGIN/lib'
LDFLAGS = -L/usr/local/cuda/lib64 -L$(LIBDIR) $(RPATH)
LDLIBS = -lcudart -ltbb -lcurand

# =============================================================================
# File Discovery (Automatic)
# =============================================================================
CPP_SOURCES := $(shell find $(SRCDIR) -name '*.cpp')
CU_SOURCES := $(shell find $(SRCDIR) -name '*.cu')

CU_SOURCES_FOR_DLINK := \
	src/Views/ContiguousKernel.cu \
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
	@echo "\n\n✅ Library build is up-to-date."

$(TARGET_SO): $(ALL_OBJECTS) $(DEVICE_LINK_OBJ)
	@echo "\n\n--- Creating shared library: $@"
	$(NVCC) -shared $(NVCCFLAGS) $(ALL_OBJECTS) $(DEVICE_LINK_OBJ) $(LDFLAGS) $(LDLIBS) -o $@

$(TARGET_A): $(ALL_OBJECTS)
	@echo "\n\n--- Creating static library: $@"
	ar rcs $@ $(ALL_OBJECTS)

# --- Compilation Pattern Rules ---
$(DEVICE_LINK_OBJ): $(OBJECTS_FOR_DLINK)
	@echo "\n\n--- Linking CUDA device code..."
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
	@echo "--- Compiling snippet: $(FILE) ---"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o snippet_runner $(FILE) $(LDFLAGS) -ltensor $(LDLIBS)
	@echo "\n--- Running snippet_runner ---"
	./snippet_runner
	@echo "\n"
	@echo "\n--- Cleaning up snippet ---"
	rm -f snippet_runner

.PHONY: rebuild
rebuild:
	@$(MAKE) clean && $(MAKE) all

.PHONY: clean
clean:
	@echo "--- Cleaning up build files ---"
	rm -rf $(OBJDIR) $(TARGET_A) $(TARGET_SO) 