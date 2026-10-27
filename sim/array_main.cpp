// Persistent text request / JSON response driver. Matrix arithmetic is RTL only.
#include "Varray.h"
#include "verilated.h"
#include <array>
#include <cstdint>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct OperandState {
    int a, b;
    bool av, bv, active, useful;
};

static int signed8(unsigned value) { return value < 128 ? int(value) : int(value) - 256; }
static int64_t signed32(uint32_t value) {
    return value <= 0x7fffffffU ? int64_t(value) : int64_t(value) - 0x100000000LL;
}

int main(int argc, char** argv) {
    VerilatedContext context;
    context.commandArgs(argc, argv);
    Varray dut{&context};
    auto tick = [&]() {
        dut.clk = 0; dut.eval();
        dut.clk = 1; dut.eval();
    };
    dut.rst = 1; dut.clear = 0; dut.enable = 0;
    for (int i = 0; i < 4; ++i) {
        dut.a_in[i] = dut.b_in[i] = 0;
        dut.a_valid_in[i] = dut.b_valid_in[i] = 0;
    }
    tick();
    dut.rst = 0;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "QUIT") break;
        try {
            std::istringstream request(line);
            int m, n, k, tracing, job, tile, pause_count;
            if (!(request >> m >> n >> k >> tracing >> job >> tile >> pause_count) ||
                m < 1 || m > 4 || n < 1 || n > 4 || k < 1 || k > 64 ||
                pause_count < 0 || pause_count > k + 6)
                throw std::runtime_error("invalid request header");
            std::set<int> pauses;
            int value;
            for (int p = 0; p < pause_count; ++p) {
                if (!(request >> value) || value < 0 || value >= k + 6)
                    throw std::runtime_error("invalid pause step");
                pauses.insert(value);
            }
            int a[4][64] = {}, b[64][4] = {};
            for (int i = 0; i < m; ++i) for (int t = 0; t < k; ++t) {
                if (!(request >> value) || value < -128 || value > 127)
                    throw std::runtime_error("invalid A operand");
                a[i][t] = value;
            }
            for (int t = 0; t < k; ++t) for (int j = 0; j < n; ++j) {
                if (!(request >> value) || value < -128 || value > 127)
                    throw std::runtime_error("invalid B operand");
                b[t][j] = value;
            }
            std::string extra;
            if (request >> extra) throw std::runtime_error("unexpected trailing input");
            int cycle = 0, valid_macs = 0, useful_macs = 0;
            int compute_cycles = 0, pause_cycles = 0, clear_cycles = 0;
            std::ostringstream trace;
            bool first_cycle = true;
            auto edge = [&](int step, const char* phase, bool enabled, bool clearing) {
                dut.enable = enabled; dut.clear = clearing;
                dut.clk = 0; dut.eval();
                OperandState before[4][4];
                for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) {
                    auto& pe = before[i][j];
                    pe = {signed8(dut.debug_a[i][j]), signed8(dut.debug_b[i][j]),
                          bool(dut.debug_a_valid[i][j]), bool(dut.debug_b_valid[i][j]),
                          false, false};
                    pe.active = enabled && !clearing && pe.av && pe.bv;
                    pe.useful = pe.active && i < m && j < n;
                    valid_macs += pe.active;
                    useful_macs += pe.useful;
                }
                dut.clk = 1; dut.eval();
                // Count executed rising edges from their actual control state.
                if (dut.clear) ++clear_cycles;
                else if (dut.enable) ++compute_cycles;
                else ++pause_cycles;
                if (tracing) {
                    if (!first_cycle) trace << ',';
                    first_cycle = false;
                    trace << "{\"schema_version\":1,\"job_id\":" << job
                          << ",\"tile_id\":" << tile << ",\"cycle\":" << cycle
                          << ",\"logical_step\":" << step << ",\"phase\":\"" << phase
                          << "\",\"enable\":" << (enabled ? "true" : "false") << ",\"pes\":[";
                    for (int i = 0; i < 4; ++i) {
                        if (i) trace << ',';
                        trace << '[';
                        for (int j = 0; j < 4; ++j) {
                            if (j) trace << ',';
                            const auto& pe = before[i][j];
                            trace << "{\"a\":" << pe.a << ",\"b\":" << pe.b
                                  << ",\"a_valid\":" << (pe.av ? "true" : "false")
                                  << ",\"b_valid\":" << (pe.bv ? "true" : "false")
                                  << ",\"accumulator\":" << signed32(dut.accumulators[i][j])
                                  << ",\"active\":" << (pe.active ? "true" : "false")
                                  << ",\"useful\":" << (pe.useful ? "true" : "false")
                                  << ",\"padded\":" << (pe.active && !pe.useful ? "true" : "false")
                                  << '}';
                        }
                        trace << ']';
                    }
                    trace << "]}";
                }
                ++cycle;
            };
            for (int i = 0; i < 4; ++i) {
                dut.a_in[i] = dut.b_in[i] = 0;
                dut.a_valid_in[i] = dut.b_valid_in[i] = 0;
            }
            edge(-1, "clear", false, true);
            for (int step = 0; step < k + 6; ++step) {
                for (int lane = 0; lane < 4; ++lane) {
                    int t = step - lane;
                    bool valid = t >= 0 && t < k;
                    dut.a_in[lane] = valid ? uint8_t(a[lane][t]) : 0;
                    dut.b_in[lane] = valid ? uint8_t(b[t][lane]) : 0;
                    dut.a_valid_in[lane] = dut.b_valid_in[lane] = valid;
                }
                if (pauses.count(step)) edge(step, "pause", false, false);
                edge(step, "compute", true, false);
            }
            std::cout << "{\"output\":[";
            for (int i = 0; i < m; ++i) {
                if (i) std::cout << ',';
                std::cout << '[';
                for (int j = 0; j < n; ++j) {
                    if (j) std::cout << ',';
                    std::cout << signed32(dut.accumulators[i][j]);
                }
                std::cout << ']';
            }
            std::cout << "],\"compute_cycles\":" << compute_cycles
                      << ",\"pause_cycles\":" << pause_cycles
                      << ",\"clear_cycles\":" << clear_cycles
                      << ",\"valid_macs\":" << valid_macs << ",\"useful_macs\":" << useful_macs
                      << ",\"padded_macs\":" << valid_macs - useful_macs
                      << ",\"trace\":[" << trace.str() << "]}" << std::endl;
        } catch (const std::exception& error) {
            std::cout << "{\"error\":\"" << error.what() << "\"}" << std::endl;
        }
    }
    dut.final();
}
