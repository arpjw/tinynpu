// Output-stationary processing element. All state obeys rst > clear > enable.
module pe (
    input  logic clk,
    input  logic rst,
    input  logic clear,
    input  logic enable,
    input  logic signed [7:0] a_in,
    input  logic signed [7:0] b_in,
    input  logic a_valid_in,
    input  logic b_valid_in,
    output logic signed [7:0] a_out,
    output logic signed [7:0] b_out,
    output logic a_valid_out,
    output logic b_valid_out,
    output logic signed [31:0] accumulator
);
    logic signed [15:0] product;
    logic signed [31:0] extended_product;
    assign product = a_in * b_in;
    assign extended_product = {{16{product[15]}}, product};

    always_ff @(posedge clk) begin
        if (rst || clear) begin
            a_out <= '0;
            b_out <= '0;
            a_valid_out <= 1'b0;
            b_valid_out <= 1'b0;
            accumulator <= '0;
        end else if (enable) begin
            a_out <= a_in;
            b_out <= b_in;
            a_valid_out <= a_valid_in;
            b_valid_out <= b_valid_in;
            if (a_valid_in && b_valid_in)
                accumulator <= accumulator + extended_product;
        end
    end
endmodule
