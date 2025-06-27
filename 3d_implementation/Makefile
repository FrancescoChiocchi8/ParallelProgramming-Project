# Makefile for testing N-Body algorithms
CC = gcc
MPICC = mpicc
CFLAGS = -O3 -Wall
PYTHON = python3

# Executable files
SEQ_EXEC = nbody_sequential
PAR_EXEC = nbody_parallel

# Dataset files
DATASET_DIR = dataset
DATASET_FILES = bodies_10.csv bodies_100.csv bodies_500.csv bodies_1000.csv bodies_2000.csv bodies_3000.csv bodies_5000.csv bodies_10000.csv
DATASETS = $(addprefix $(DATASET_DIR)/,$(DATASET_FILES))

# Number of processors to be tested
PROCESSORS = 2 3 4 5 6 7 8

# Directory for results
RESULTS_DIR = results
PLOTS_DIR = plots

.PHONY: all compile test clean plots help

all: compile test plots

# Compilation of executables
compile: $(SEQ_EXEC) $(PAR_EXEC)

$(SEQ_EXEC): nbody_sequential.c
	$(CC) $(CFLAGS) -o $@ $< -lm

$(PAR_EXEC): nbody_parallel.c
	$(MPICC) $(CFLAGS) -o $@ $< -lm

# Directory creation
$(RESULTS_DIR):
	mkdir -p $(RESULTS_DIR)

$(PLOTS_DIR):
	mkdir -p $(PLOTS_DIR)

# Comprehensive test
test: compile $(RESULTS_DIR) test-sequential test-parallel

# Sequential test
test-sequential: $(SEQ_EXEC) $(RESULTS_DIR)
	@echo "=== TESTING SEQUENTIAL VERSION ==="
	@for dataset in $(DATASETS); do \
		if [ -f $$dataset ]; then \
			echo "Testing $$dataset..."; \
			./$(SEQ_EXEC) $$dataset > $(RESULTS_DIR)/seq_$$dataset.log 2>&1; \
		else \
			echo "Warning: $$dataset not found, skipping..."; \
		fi; \
	done

# Parallel testing
test-parallel: $(PAR_EXEC) $(RESULTS_DIR)
	@echo "=== TESTING PARALLEL VERSION ==="
	@for dataset in $(DATASETS); do \
		if [ -f $$dataset ]; then \
			echo "Testing $$dataset with different processor counts:"; \
			for proc in $(PROCESSORS); do \
				echo "  - $$proc processors..."; \
				mpirun -np $$proc ./$(PAR_EXEC) $$dataset > $(RESULTS_DIR)/par_$${dataset}_p$$proc.log 2>&1; \
			done; \
		else \
			echo "Warning: $$dataset not found, skipping..."; \
		fi; \
	done

# Time extraction and plot generation
plots: $(PLOTS_DIR)
	@echo "=== EXTRACTING PERFORMANCE DATA AND GENERATING PLOTS ==="
	$(PYTHON) extract_performance.py

# Test dataset generation
generate-datasets:
	@echo "=== GENERATING TEST DATASETS ==="
	$(PYTHON) data_acquisition.py

# Cleaning
clean:
	rm -f $(SEQ_EXEC) $(PAR_EXEC)
	rm -rf $(RESULTS_DIR)
	rm -rf $(PLOTS_DIR)
	rm -f output_*.csv

# Rapid test with small dataset
quick-test: compile $(RESULTS_DIR)
	@echo "=== QUICK TEST ==="
	@if [ -f bodies_10.csv ]; then \
		echo "Sequential test..."; \
		./$(SEQ_EXEC) bodies_10.csv > $(RESULTS_DIR)/quick_seq.log 2>&1; \
		echo "Parallel test (2 processors)..."; \
		mpirun -np 2 ./$(PAR_EXEC) bodies_10.csv > $(RESULTS_DIR)/quick_par.log 2>&1; \
		echo "Quick test completed. Check $(RESULTS_DIR)/ for logs."; \
	else \
		echo "bodies_10.csv not found. Please ensure the dataset exists."; \
	fi

# Help
help:
	@echo "N-Body Performance Testing Makefile"
	@echo "===================================="
	@echo "Available targets:"
	@echo "  all              - Compile, test, and generate plots"
	@echo "  compile          - Compile both sequential and parallel versions"
	@echo "  test             - Run all performance tests"
	@echo "  test-sequential  - Run only sequential tests"
	@echo "  test-parallel    - Run only parallel tests"
	@echo "  plots            - Extract data and generate performance plots"
	@echo "  generate-datasets- Generate test datasets"
	@echo "  quick-test       - Quick test with small dataset"
	@echo "  clean            - Clean all generated files"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Required files:"
	@echo "  - nbody_sequential.c"
	@echo "  - nbody_parallel.c"
	@echo "  - extract_performance.py"
	@echo "  - Dataset files: $(DATASETS)"
	@echo ""
	@echo "Requirements:"
	@echo "  - GCC compiler"
	@echo "  - MPI implementation"
	@echo "  - Python 3 with matplotlib and pandas"