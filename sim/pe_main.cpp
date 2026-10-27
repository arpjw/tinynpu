// Deterministic direct PE verification independent of the array scheduler.
#include "Vpe.h"
#include "verilated.h"
#include <cstdint>
#include <iostream>
#include <random>

static int signed8(unsigned value) { return value < 128 ? int(value) : int(value) - 256; }
static int64_t signed32(uint32_t value) {
    return value <= 0x7fffffffU ? int64_t(value) : int64_t(value) - 0x100000000LL;
}

int main(int argc, char** argv) {
    VerilatedContext context;
    context.commandArgs(argc, argv);
    Vpe dut{&context};
    int expected_a = 0, expected_b = 0;
    bool expected_av = false, expected_bv = false;
    int64_t expected_acc = 0;
    int checks = 0;
    auto edge = [&](int a, int b, bool av, bool bv, bool enabled, bool clear, bool reset) {
        dut.a_in = uint8_t(a); dut.b_in = uint8_t(b);
        dut.a_valid_in = av; dut.b_valid_in = bv;
        dut.enable = enabled; dut.clear = clear; dut.rst = reset;
        dut.clk = 0; dut.eval();
        dut.clk = 1; dut.eval();
        if (reset || clear) {
            expected_a = expected_b = 0;
            expected_av = expected_bv = false;
            expected_acc = 0;
        } else if (enabled) {
            expected_a = a; expected_b = b;
            expected_av = av; expected_bv = bv;
            if (av && bv) expected_acc += int64_t(a) * int64_t(b);
        }
        ++checks;
        if (signed8(dut.a_out) != expected_a || signed8(dut.b_out) != expected_b ||
            bool(dut.a_valid_out) != expected_av || bool(dut.b_valid_out) != expected_bv ||
            signed32(dut.accumulator) != expected_acc) {
            std::cerr << "PE mismatch at edge " << checks << '\n';
            return false;
        }
        return true;
    };
    // All clear/reset/enable combinations, including reset and clear when disabled.
    for (int reset = 0; reset < 2; ++reset)
        for (int clear = 0; clear < 2; ++clear)
            for (int enable = 0; enable < 2; ++enable) {
                if (!edge(0, 0, false, false, false, false, true)) return 1;
                if (!edge(-128, 127, true, true, true, false, false)) return 1;
                if (!edge(127, -128, true, true, enable, clear, reset)) return 1;
            }
    for (int a : {-128, -1, 0, 1, 127}) for (int b : {-128, -1, 0, 1, 127})
        for (int av = 0; av < 2; ++av) for (int bv = 0; bv < 2; ++bv)
            if (!edge(a, b, av, bv, true, false, false)) return 1;
    std::mt19937 rng(20261027);
    for (int step = 0; step < 10000; ++step) {
        int a = int(rng() % 256) - 128, b = int(rng() % 256) - 128;
        bool av = rng() % 2, bv = rng() % 2, enabled = rng() % 2;
        bool clear = rng() % 31 == 0, reset = rng() % 67 == 0;
        if (!edge(a, b, av, bv, enabled, clear, reset)) return 1;
    }
    dut.final();
    std::cout << "PASS: " << checks << " direct PE state checks; seed=20261027\n";
}
