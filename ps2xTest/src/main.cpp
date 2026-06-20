#include "MiniTest.h"
#include <cstdlib>
#include <iostream>

void register_code_generator_tests();
void register_r5900_decoder_tests();
void register_elf_analyzer_tests();
void register_pad_input_tests();
void register_ps2_runtime_io_tests();
void register_ps2_runtime_kernel_tests();
void register_ps2_runtime_interrupt_tests();
void register_ps2_memory_tests();
void register_ps2_gs_tests();
void register_ps2_sif_rpc_tests();
void register_ps2_sif_dma_tests();
void register_ps2_recompiler_tests();
void register_ps2_runtime_expansion_tests();
void register_scheduler_tests();
void register_scheduler_protocol_tests();
void register_scheduler_race_tests();
void register_scheduler_stress_tests();
void register_scheduler_vsync_priority_tests();
void register_scheduler_lifecycle_tests();
void register_scheduler_borrowed_worker_tests();
void register_scheduler_window_tests();
void register_scheduler_sleep_resume_tests();
void register_scheduler_shutdown_clean_tests();
void register_scheduler_borrowed_guard_tests();
void register_scheduler_sema_delete_tests();
void register_scheduler_tid_reuse_tests();

int main()
{
    register_code_generator_tests();
    register_r5900_decoder_tests();
    register_elf_analyzer_tests();
    register_pad_input_tests();
    register_ps2_runtime_io_tests();
    register_ps2_runtime_kernel_tests();
    register_ps2_runtime_interrupt_tests();
    register_ps2_memory_tests();
    register_ps2_gs_tests();
    register_ps2_sif_rpc_tests();
    register_ps2_sif_dma_tests();
    register_ps2_recompiler_tests();
    register_ps2_runtime_expansion_tests();
    register_scheduler_tests();
    register_scheduler_protocol_tests();
    register_scheduler_race_tests();
    register_scheduler_stress_tests();
    register_scheduler_vsync_priority_tests();
    register_scheduler_lifecycle_tests();
    register_scheduler_borrowed_worker_tests();
    register_scheduler_window_tests();
    register_scheduler_sleep_resume_tests();
    register_scheduler_shutdown_clean_tests();
    register_scheduler_borrowed_guard_tests();
    register_scheduler_sema_delete_tests();
    register_scheduler_tid_reuse_tests();
    int res = MiniTest::Run();
    std::cout.flush();
    std::cerr.flush();
    std::_Exit(res);
}
