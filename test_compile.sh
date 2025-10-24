#!/bin/bash
# Test compilation and execution script for tensor library

# Create test output directory if it doesn't exist
mkdir -p local_test

# Initialize test report
REPORT_FILE="local_test/test_report.md"
echo "# Test Execution Report" > "$REPORT_FILE"
echo "**Generated on:** $(date)" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "## Summary" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# Counters for summary
TOTAL_TESTS=0
SUCCESSFUL_COMPILES=0
SUCCESSFUL_RUNS=0
FAILED_COMPILES=0
FAILED_RUNS=0

# Function to compile and optionally run a test
compile_and_run() {
    local test_type=$1
    local test_num=$2
    local run_tests=$3
    
    local test_file="Tests/Tensor Tests/${test_type}_${test_num}.cpp"
    local output_file="local_test/${test_type}_${test_num}"
    
    ((TOTAL_TESTS++))
    
    echo "## ${test_type}_${test_num}" >> "$REPORT_FILE"
    echo "**File:** \`$test_file\`" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    if [ -f "$test_file" ]; then
        echo "Compiling $test_file..."
        echo "**Compilation:** " >> "$REPORT_FILE"
        
        # Capture compilation output
        COMPILE_OUTPUT=$(g++ -std=c++20 -Iinclude -DWITH_CUDA  "$test_file" \
            -o "$output_file" \
            lib/libtensor.so \
            -lcudart -lcurand 2>&1)
        
        if [ $? -eq 0 ]; then
            echo "✓ Success: $output_file"
            echo "✅ **SUCCESS**" >> "$REPORT_FILE"
            ((SUCCESSFUL_COMPILES++))
            
            # Run the test if requested
            if [ "$run_tests" = "run" ]; then
                echo "   Running test..."
                echo "**Execution:** " >> "$REPORT_FILE"
                
                # Capture test execution output with timeout
                TEST_OUTPUT=$("$output_file" 2>&1)
                TEST_EXIT_CODE=$?
                
                if [ $TEST_EXIT_CODE -eq 0 ]; then
                    echo "   ✅ Test passed"
                    echo "✅ **TEST PASSED**" >> "$REPORT_FILE"
                    ((SUCCESSFUL_RUNS++))
                elif [ $TEST_EXIT_CODE -eq 124 ]; then
                    echo "   ⚠ Test timed out"
                    echo "⚠ **TEST TIMED OUT**" >> "$REPORT_FILE"
                    ((FAILED_RUNS++))
                else
                    echo "   ❌ Test failed with exit code: $TEST_EXIT_CODE"
                    echo "❌ **TEST FAILED** (Exit code: $TEST_EXIT_CODE)" >> "$REPORT_FILE"
                    ((FAILED_RUNS++))
                fi
                
                # Add test output to report
                if [ -n "$TEST_OUTPUT" ]; then
                    echo "" >> "$REPORT_FILE"
                    echo "### Test Output:" >> "$REPORT_FILE"
                    echo '```' >> "$REPORT_FILE"
                    echo "$TEST_OUTPUT" >> "$REPORT_FILE"
                    echo '```' >> "$REPORT_FILE"
                fi
            else
                echo "**Execution:** Not run" >> "$REPORT_FILE"
            fi
        else
            echo "✗ Failed: $test_file"
            echo "❌ **COMPILATION FAILED**" >> "$REPORT_FILE"
            ((FAILED_COMPILES++))
            
            # Add compilation errors to report
            echo "" >> "$REPORT_FILE"
            echo "### Compilation Errors:" >> "$REPORT_FILE"
            echo '```' >> "$REPORT_FILE"
            echo "$COMPILE_OUTPUT" >> "$REPORT_FILE"
            echo '```' >> "$REPORT_FILE"
        fi
    else
        echo "⚠ Warning: $test_file not found"
        echo "❌ **FILE NOT FOUND**" >> "$REPORT_FILE"
        ((FAILED_COMPILES++))
    fi
    
    echo "" >> "$REPORT_FILE"
    echo "---" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
}

# Check if user wants to run tests immediately
RUN_TESTS=""
if [ "$1" = "--run" ] || [ "$1" = "-r" ]; then
    RUN_TESTS="run"
    echo "Compiling and running all tests..."
else
    echo "Compiling all tests..."
fi

# Clear terminal and show header
clear
echo "========================================="
echo "    Tensor Library Test Runner"
echo "========================================="
echo ""

# Compile Unit Tests (1-11)
echo "=== Unit Tests ==="
for i in {1..11}; do
    compile_and_run "UnitTest" "$i" "$RUN_TESTS"
done

# Compile CUDA Tests (1-7)
echo "=== CUDA Tests ==="
for i in {1..7}; do
    compile_and_run "CudaTest" "$i" "$RUN_TESTS"
done

# Update summary in report
{
    echo "### Statistics:"
    echo "- **Total Tests:** $TOTAL_TESTS"
    echo "- **Successful Compilations:** $SUCCESSFUL_COMPILES"
    echo "- **Failed Compilations:** $FAILED_COMPILES"
    if [ "$RUN_TESTS" = "run" ]; then
        echo "- **Successful Runs:** $SUCCESSFUL_RUNS"
        echo "- **Failed Runs:** $FAILED_RUNS"
        echo "- **Success Rate:** $(( (SUCCESSFUL_RUNS * 100) / TOTAL_TESTS ))%"
    fi
    echo ""
    echo "### Generated Files:"
    echo "All compiled test binaries are available in \`./local_test/\`"
} >> "$REPORT_FILE"

# Print final summary to console
echo ""
echo "========================================="
echo "            TEST SUMMARY"
echo "========================================="
echo "Total Tests:        $TOTAL_TESTS"
echo "Successful Compiles: $SUCCESSFUL_COMPILES"
echo "Failed Compiles:     $FAILED_COMPILES"
if [ "$RUN_TESTS" = "run" ]; then
    echo "Successful Runs:     $SUCCESSFUL_RUNS"
    echo "Failed Runs:         $FAILED_RUNS"
    SUCCESS_RATE=$(( (SUCCESSFUL_RUNS * 100) / TOTAL_TESTS ))
    echo "Overall Success Rate: ${SUCCESS_RATE}%"
fi
echo ""
echo "Detailed report: $REPORT_FILE"
echo "Test binaries:   ./local_test/"
echo "========================================="

if [ "$RUN_TESTS" != "run" ]; then
    echo ""
    echo "You can run tests with: ./local_test/<test_name>"
    echo "Or use: $0 --run to compile and run all tests automatically"
fi