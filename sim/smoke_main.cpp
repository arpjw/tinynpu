// Verify the C++ -> Verilator -> RTL execution path, not accelerator behavior.
#include "Vtoolchain_smoke.h"
#include "verilated.h"
#include <iostream>

int main(int argc, char** argv) {
    VerilatedContext context;
    context.commandArgs(argc, argv);
    Vtoolchain_smoke dut{&context};
    auto tick = [&]() {
        dut.clk = 0;
        dut.eval();
        dut.clk = 1;
        dut.eval();
    };
    dut.rst = 1;
    dut.enable = 1;
    dut.data_in = 255;
    tick();
    if (dut.data_out != 0) return 1;
    dut.rst = 0;
    dut.data_in = 165;
    tick();
    if (dut.data_out != 165) return 2;
    dut.enable = 0;
    dut.data_in = 90;
    tick();
    if (dut.data_out != 165) return 3;
    dut.rst = 1;
    tick();
    if (dut.data_out != 0) return 4;
    dut.final();
    std::cout << "PASS: Verilator toolchain smoke (not NPU verification)\n";
    return 0;
}
