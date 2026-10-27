// Fixed 4x4 array. Debug outputs expose actual wires consumed by every PE.
module array (
    input logic clk,
    input logic rst,
    input logic clear,
    input logic enable,
    input logic signed [7:0] a_in [4],
    input logic signed [7:0] b_in [4],
    input logic a_valid_in [4],
    input logic b_valid_in [4],
    output logic signed [31:0] accumulators [4][4],
    output logic signed [7:0] debug_a [4][4],
    output logic signed [7:0] debug_b [4][4],
    output logic debug_a_valid [4][4],
    output logic debug_b_valid [4][4]
);
    logic signed [7:0] a_forward [4][4];
    logic signed [7:0] b_forward [4][4];
    logic a_forward_valid [4][4];
    logic b_forward_valid [4][4];

    for (genvar i = 0; i < 4; i++) begin : rows
        for (genvar j = 0; j < 4; j++) begin : columns
            if (j == 0) begin : left_boundary
                assign debug_a[i][j] = a_in[i];
                assign debug_a_valid[i][j] = a_valid_in[i];
            end else begin : left_neighbor
                assign debug_a[i][j] = a_forward[i][j-1];
                assign debug_a_valid[i][j] = a_forward_valid[i][j-1];
            end
            if (i == 0) begin : top_boundary
                assign debug_b[i][j] = b_in[j];
                assign debug_b_valid[i][j] = b_valid_in[j];
            end else begin : top_neighbor
                assign debug_b[i][j] = b_forward[i-1][j];
                assign debug_b_valid[i][j] = b_forward_valid[i-1][j];
            end
            pe element (
                .clk(clk), .rst(rst), .clear(clear), .enable(enable),
                .a_in(debug_a[i][j]), .b_in(debug_b[i][j]),
                .a_valid_in(debug_a_valid[i][j]), .b_valid_in(debug_b_valid[i][j]),
                .a_out(a_forward[i][j]), .b_out(b_forward[i][j]),
                .a_valid_out(a_forward_valid[i][j]),
                .b_valid_out(b_forward_valid[i][j]),
                .accumulator(accumulators[i][j])
            );
        end
    end
endmodule
