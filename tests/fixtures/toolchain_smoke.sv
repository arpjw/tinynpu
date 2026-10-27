// Toolchain fixture only. This register is not a TinyNPU processing element.
module toolchain_smoke (
    input  logic       clk,
    input  logic       rst,
    input  logic       enable,
    input  logic [7:0]  data_in,
    output logic [7:0]  data_out
);
    always_ff @(posedge clk) begin
        if (rst)
            data_out <= '0;
        else if (enable)
            data_out <= data_in;
    end
endmodule
