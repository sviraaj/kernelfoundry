cmake_minimum_required(VERSION 3.20)

# ---------- ncu / Nsight Compute integration ----------
# Targets:
#   ncu_full          — full section report, saved to build/reports/
#   ncu_stalls        — occupancy + stall breakdown (fast, no replay)
#   ncu_bandwidth     — HBM / L2 / smem byte counts
#   ncu_tensorcore    — tensor core utilization
#   ncu_roofline      — roofline model (SpeedOfLight_HierarchicalTensorRooflineChart)
#
# Usage (from build/):
#   cmake --build . --target ncu_full
#   ncu-ui reports/ncu_full.ncu-rep
#
# Requires: ncu on PATH or NCU_PATH set at configure time.
#   cmake -DNCU_PATH=/usr/local/cuda-12.9/bin/ncu ..

find_program(NCU_EXECUTABLE ncu
    HINTS "${NCU_PATH}" /usr/local/cuda-12.9/bin /usr/local/cuda/bin
    DOC "Nsight Compute CLI"
)

set(NCU_REPORTS_DIR "${CMAKE_BINARY_DIR}/reports")
set(NCU_KERNEL        flash_attn_kernel)
set(NCU_LAUNCH_COUNT  1)
set(NCU_LAUNCH_SKIP   5)   # skip warmup iterations inside bench

set(NCU_COMMON_FLAGS
    --kernel-name     ${NCU_KERNEL}
    --launch-count    ${NCU_LAUNCH_COUNT}
    --launch-skip     ${NCU_LAUNCH_SKIP}
    --target-processes all
)

set(NCU_STALL_METRICS
    # SM-level absolute warp count (= kFlashWarps when 1 block/SM due to smem)
    sm__warps_active.avg.per_cycle_active
    # SM occupancy as % of peak (peak = 48–64 warps/SM on Blackwell)
    sm__warps_active.avg.pct_of_peak_sustained_active
    # Per-SMSP eligible warps — an SM has 4 SMSPs, so this = sm_count/4
    # Shows "1" when kFlashWarps=4 and smem forces 1 block/SM (expected, not a bug)
    smsp__warps_eligible.avg.per_cycle_active
    smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
    smsp__warp_issue_stalled_wait_per_warp_active.pct
    smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct
)

set(NCU_BW_METRICS
    dram__bytes_read.sum
    dram__bytes_write.sum
    lts__t_bytes.sum
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum
    l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum
    l1tex__t_bytes_pipe_lsu_mem_shared_op_ld.sum
    l1tex__t_bytes_pipe_lsu_mem_shared_op_st.sum
)

set(NCU_TC_METRICS
    sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active
    sm__inst_executed_pipe_tensor_op_hmma.sum
    smsp__sass_thread_inst_executed_op_hmma_pred_on.sum
)

# Helper: join a list into a comma-separated string
function(join_metrics OUT_VAR)
    set(_list ${ARGN})
    string(REPLACE ";" "," _joined "${_list}")
    set(${OUT_VAR} "${_joined}" PARENT_SCOPE)
endfunction()

join_metrics(NCU_STALL_METRICS_STR  ${NCU_STALL_METRICS})
join_metrics(NCU_BW_METRICS_STR     ${NCU_BW_METRICS})
join_metrics(NCU_TC_METRICS_STR     ${NCU_TC_METRICS})

if(NCU_EXECUTABLE)
    add_custom_target(ncu_full
        COMMAND ${CMAKE_COMMAND} -E make_directory ${NCU_REPORTS_DIR}
        COMMAND ${NCU_EXECUTABLE}
            ${NCU_COMMON_FLAGS}
            --set full
            --import-source yes
            --source-folders ${PROJECT_SOURCE_DIR}
            -o ${NCU_REPORTS_DIR}/ncu_full
            $<TARGET_FILE:bench_flash_attn>
        DEPENDS bench_flash_attn
        COMMENT "ncu full report → build/reports/ncu_full.ncu-rep"
        VERBATIM
    )

    add_custom_target(ncu_stalls
        COMMAND ${CMAKE_COMMAND} -E make_directory ${NCU_REPORTS_DIR}
        COMMAND ${NCU_EXECUTABLE}
            ${NCU_COMMON_FLAGS}
            --metrics ${NCU_STALL_METRICS_STR}
            $<TARGET_FILE:bench_flash_attn>
        DEPENDS bench_flash_attn
        COMMENT "ncu occupancy + stall breakdown"
        VERBATIM
    )

    add_custom_target(ncu_bandwidth
        COMMAND ${CMAKE_COMMAND} -E make_directory ${NCU_REPORTS_DIR}
        COMMAND ${NCU_EXECUTABLE}
            ${NCU_COMMON_FLAGS}
            --metrics ${NCU_BW_METRICS_STR}
            $<TARGET_FILE:bench_flash_attn>
        DEPENDS bench_flash_attn
        COMMENT "ncu HBM + L2 + smem bandwidth"
        VERBATIM
    )

    add_custom_target(ncu_tensorcore
        COMMAND ${CMAKE_COMMAND} -E make_directory ${NCU_REPORTS_DIR}
        COMMAND ${NCU_EXECUTABLE}
            ${NCU_COMMON_FLAGS}
            --metrics ${NCU_TC_METRICS_STR}
            $<TARGET_FILE:bench_flash_attn>
        DEPENDS bench_flash_attn
        COMMENT "ncu tensor core utilization"
        VERBATIM
    )

    add_custom_target(ncu_roofline
        COMMAND ${CMAKE_COMMAND} -E make_directory ${NCU_REPORTS_DIR}
        COMMAND ${NCU_EXECUTABLE}
            ${NCU_COMMON_FLAGS}
            --section SpeedOfLight_HierarchicalTensorRooflineChart
            --import-source yes
            --source-folders ${PROJECT_SOURCE_DIR}
            -o ${NCU_REPORTS_DIR}/ncu_roofline
            $<TARGET_FILE:bench_flash_attn>
        DEPENDS bench_flash_attn
        COMMENT "ncu roofline → build/reports/ncu_roofline.ncu-rep"
        VERBATIM
    )
    # PyTorch FA2 (SDPA) profiling — same metrics as ncu_stalls for direct comparison.
    # Requires Python + PyTorch in PYTHON_EXECUTABLE (default: searches PATH).
    find_program(PYTHON_EXECUTABLE python3
        HINTS "${PYTHON_PATH}" /home/sviraaj/projects/AI/.venv/bin
        DOC "Python interpreter for bench_flash_attn2.py"
    )
    set(NCU_SDPA_SCRIPT "${CMAKE_SOURCE_DIR}/tests/bench_flash_attn2.py")

    if(PYTHON_EXECUTABLE)
        add_custom_target(ncu_sdpa_discover
            COMMAND ${NCU_EXECUTABLE}
                --target-processes all
                --metrics smsp__cycles_elapsed.avg
                ${PYTHON_EXECUTABLE} ${NCU_SDPA_SCRIPT}
            COMMENT "ncu kernel name discovery for PyTorch SDPA (grep output for flash/fmha)"
            VERBATIM
        )

        add_custom_target(ncu_sdpa_stalls
            COMMAND ${NCU_EXECUTABLE}
                --target-processes all
                --kernel-name "regex:.*flash.*"
                --launch-count    ${NCU_LAUNCH_COUNT}
                --launch-skip     ${NCU_LAUNCH_SKIP}
                --metrics ${NCU_STALL_METRICS_STR},dram__bytes_read.sum,dram__bytes_write.sum
                ${PYTHON_EXECUTABLE} ${NCU_SDPA_SCRIPT}
            COMMENT "ncu stall + bandwidth breakdown for PyTorch FA2 (compare with ncu_stalls)"
            VERBATIM
        )

        add_custom_target(ncu_sdpa_full
            COMMAND ${CMAKE_COMMAND} -E make_directory ${NCU_REPORTS_DIR}
            COMMAND ${NCU_EXECUTABLE}
                --target-processes all
                --kernel-name "regex:.*flash.*"
                --launch-count    ${NCU_LAUNCH_COUNT}
                --launch-skip     ${NCU_LAUNCH_SKIP}
                --set full
                -o ${NCU_REPORTS_DIR}/ncu_sdpa_full
                ${PYTHON_EXECUTABLE} ${NCU_SDPA_SCRIPT}
            COMMENT "ncu full report for PyTorch FA2 → build/reports/ncu_sdpa_full.ncu-rep"
            VERBATIM
        )
    else()
        message(STATUS "Python not found — ncu_sdpa_* targets disabled. Set -DPYTHON_PATH=<dir> to enable.")
    endif()

else()
    message(STATUS "ncu not found — profiling targets disabled. Set -DNCU_PATH=<dir> to enable.")
endif()

# ---------- Flash Attention end-to-end tests ----------
add_executable(test_flash_attn
    test_flash_attn.cu
)
target_include_directories(test_flash_attn PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(test_flash_attn PRIVATE CUDA::cudart)
set_target_properties(test_flash_attn PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
)
add_test(NAME flash_attn COMMAND test_flash_attn)

# ---------- Flash Attention benchmark ----------
add_executable(bench_flash_attn
    bench_flash_attn.cu
)
target_include_directories(bench_flash_attn PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(bench_flash_attn PRIVATE CUDA::cudart)
target_compile_options(bench_flash_attn PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
set_target_properties(bench_flash_attn PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
)
