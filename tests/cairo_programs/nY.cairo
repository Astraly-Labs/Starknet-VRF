%builtins output range_check bitwise
from lib.curve import P_low, P_high
from lib.ed25519 import ExtendedPoint, TwistedArithmetics
from lib.u255 import u255
from starkware.cairo.common.cairo_builtins import BitwiseBuiltin
from starkware.cairo.common.uint256 import Uint256

func main{output_ptr: felt*, range_check_ptr, bitwise_ptr: BitwiseBuiltin*}() {
    // __setup__();
    %{
        def bin_c(u):
            b=bin(u)
            f = b[0:10] + ' ' + b[10:19] + '...' + b[-16:-8] + ' ' + b[-8:]
            return f
        def bin_64(u):
            b=bin(u)
            little = '0b'+b[2:][::-1]
            f='0b'+' '.join([b[2:][i:i+64] for i in range(0, len(b[2:]), 64)])
            return f
        def bin_8(u):
            b=bin(u)
            little = '0b'+b[2:][::-1]
            f="0b"+' '.join([little[2:][i:i+8] for i in range(0, len(little[2:]), 8)])
            return f
        def _inverse(x):
            x=x.low + (x.high<<128)
            print("to inverse:", x)
            PRIME = 2**255-19
            return pow(x, PRIME - 2, PRIME)
        def print_from_extended(p):
            x = p.x.low + (p.x.high<<128)
            y = p.y.low + (p.y.high<<128)
            z = p.z.low + (p.z.high<<128)
            PRIME = 2**255-19

            invZ = _inverse(p.z)
            print("invz", invZ)
            assert invZ * z % PRIME == 1

            print(f"x={x*invZ%PRIME}")
            print(f"y={y*invZ%PRIME}")

        def print_u_256_info(u, un):
            u = u.low + (u.high << 128) 
            print(f" {un}_{u.bit_length()}bits = {bin_c(u)}")
            print(f" {un} = {u}")
        def print_felt_info(u, un):
            print(f" {un}_{u.bit_length()}bits = {bin_8(u)}")
            print(f" {un} = {u}")

        def print_u_512_info(u, un):
            u = u.d0 + (u.d1 << 128) + (u.d2<<256) + (u.d3<<384) 
            print(f" {un}_{u.bit_length()}bits = {bin_64(u)}")
            print(f" {un} = {u}")
        def print_u_512_info_u(l, h, un):
            u = l.low + (l.high << 128) + (h.low<<256) + (h.high<<384) 
            print(f" {un}_{u.bit_length()}bits = {bin_64(u)}")
            print(f" {un} = {u}")

        def print_u_256_neg(u, un):
            u = 2**256 - (u.low + (u.high << 128))
            print(f"-{un}_{u.bit_length()}bits = {bin_c(u)}")
            print(f"-{un} = {u}")

        def print_sub(a, an, b, bn, res, resn):
            print (f"----------------Subbing {resn} = {an} - {bn}------------------")
            print_u_256_info(a, an)
            print('\n')

            print_u_256_info(b, bn)
            print_u_256_neg(b, bn)
            print('\n')

            print_u_256_info(res, resn)
            print ('---------------------------------------------------------')
    %}
    let ZERO = ExtendedPoint(Uint256(0, 0), Uint256(1, 0), Uint256(1, 0), Uint256(0, 0));
    // s= 1657008116471393480753518857542198008763130933602946594165815725401803042495
    // let sG: ExtendedPoint = _scalar_multiply_G_by_s_loop(index=0, res=ZERO, s=Uint256(1000, 0));
    let cY: ExtendedPoint = TwistedArithmetics._scalar_multiply_Y_by_c_loop(
        index=0, res=ZERO, c=67939000374577434858470669064940821215
    );

    %{ print_u_256_info(ids.cY.x, "x") %}
    %{ print_u_256_info(ids.cY.y, "y") %}
    %{ print_u_256_info(ids.cY.z, "z") %}
    %{ print_u_256_info(ids.cY.t, "t") %}
    %{ print_from_extended(ids.cY) %}
    // assert cY.x.low = 42114794351893256184233311756735798781;
    // assert cY.x.high = 105724198134614256290713683444580514556;
    // assert cY.y.low = 239217761667596995751113313307787875712;
    // assert cY.y.high = 69762747036280158809166715932357629792;
    return ();
}
