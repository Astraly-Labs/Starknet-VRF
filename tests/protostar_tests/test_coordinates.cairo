%lang starknet
from starkware.cairo.common.cairo_builtins import HashBuiltin
from starkware.cairo.common.cairo_builtins import BitwiseBuiltin
from starkware.cairo.common.uint256 import Uint256, uint256_mul
from lib.curve import (
    P_low,
    P_high,
    P_min_1_div_2_low,
    P_min_1_div_2_high,
    ExtendedPoint,
    Gx_low,
    Gx_high,
    Gy_low,
    Gy_high,
)
from lib.utils import clear_high_order_bit_128, split_64, split_128
from lib.u255 import u512
from lib.ed25519 import bijections, WeierstrassArithmetics, TwistedArithmetics
from lib.curve import AffinePoint, EcPoint

from starkware.cairo.common.cairo_secp.bigint import BigInt3, uint256_to_bigint, bigint_to_uint256

@external
func __setup__() {
    %{
        PRIME = 2**255-19

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
    assert 1 = 1;
    return ();
}

@external
func test_convert_round_trip{syscall_ptr: felt*, pedersen_ptr: HashBuiltin*, range_check_ptr}() {
    __setup__();
    let G = AffinePoint(Uint256(Gx_low, Gx_high), Uint256(Gy_low, Gy_high));
    let G_Mt = bijections.to_montgomery(G);
    let G_We = bijections.to_weierstrass(G_Mt);
    %{ print_u_256_info(ids.G_We.x, "G_We.x") %}
    %{ print_u_256_info(ids.G_We.y, "G_We.Y") %}
    let res_Mt = bijections.from_weierstrass(G_We);
    %{ print_u_256_info(ids.res_Mt.x, "res_Mt.x") %}
    %{ print_u_256_info(ids.res_Mt.y, "res_Mt.Y") %}

    let res_Ed = bijections.from_montgomery(res_Mt);

    let res_x = res_Ed.x;
    let res_y = res_Ed.y;
    assert Gx_low = res_x.low;
    assert Gx_high = res_x.high;
    assert Gy_low = res_y.low;
    assert Gy_high = res_y.high;
    return ();
}

@external
func test_convert_round_mt{syscall_ptr: felt*, pedersen_ptr: HashBuiltin*, range_check_ptr}() {
    alloc_locals;

    __setup__();
    tempvar G = AffinePoint(Uint256(Gx_low, Gx_high), Uint256(Gy_low, Gy_high));
    %{ print_u_256_info(ids.G.x, "G.x") %}
    %{ print_u_256_info(ids.G.y, "G.Y") %}
    let G_Mt = bijections.to_montgomery(G);
    %{ print_u_256_info(ids.G_Mt.x, "G_Mt.x") %}
    %{ print_u_256_info(ids.G_Mt.y, "G_Mt.Y") %}
    let res_Ed = bijections.from_montgomery(G_Mt);
    %{ print_u_256_info(ids.res_Ed.x, "Res_Ed.x") %}
    %{ print_u_256_info(ids.res_Ed.y, "Res_ed.Y") %}
    let res_x = res_Ed.x;
    let res_y = res_Ed.y;
    assert Gx_low = res_x.low;
    assert Gx_high = res_x.high;
    assert Gy_low = res_y.low;
    assert Gy_high = res_y.high;

    return ();
}

@external
func test_add_weierstrass2{range_check_ptr, bitwise_ptr: BitwiseBuiltin*}() {
    alloc_locals;
    __setup__();
    let ZERO = ExtendedPoint(Uint256(0, 0), Uint256(1, 0), Uint256(1, 0), Uint256(0, 0));
    let G = ExtendedPoint(
        Uint256(Gx_low, Gx_high),
        Uint256(Gy_low, Gy_high),
        Uint256(1, 0),
        Uint256(43784682192424479926751423844859764131, 137613371725791198171896662509163897725),
    );
    // s= 1657008116471393480753518857542198008763130933602946594165815725401803042495
    // let sG: ExtendedPoint = TwistedArithmetics._scalar_multiply_G_by_s_loop(index=0, res=ZERO, s=Uint256(111, 0));
    // %{ print_u_256_info(ids.sG.x, "x") %}
    // %{ print_u_256_info(ids.sG.y, "y") %}
    // %{ print_u_256_info(ids.sG.z, "z") %}
    // %{ print_u_256_info(ids.sG.t, "t") %}

    // %{ print_from_extended(ids.sG) %}

    let alpha = Uint256(
        143186476941636880901214103594843510573, 124026708105846590725274683684370988502
    );
    let (alpha_big) = uint256_to_bigint(alpha);
    %{ print_felt_info(ids.alpha_big.d0, "alpha_big.d0") %}
    %{ print_felt_info(ids.alpha_big.d1, "alpha_big.d1") %}
    %{ print_felt_info(ids.alpha_big.d2, "alpha_big.d2") %}

    let G_affine = AffinePoint(Uint256(Gx_low, Gx_high), Uint256(Gy_low, Gy_high));
    let G_Mt = bijections.to_montgomery(G_affine);
    let G_We = bijections.to_weierstrass(G_Mt);
    %{ print_u_256_info(ids.G_We.x, "G_We.x") %}
    %{ print_u_256_info(ids.G_We.y, "G_We.Y") %}

    let (G_we_x_Bigint: BigInt3) = uint256_to_bigint(G_We.x);
    let (G_we_y_Bigint: BigInt3) = uint256_to_bigint(G_We.y);
    let G_we_Bigint = EcPoint(G_we_x_Bigint, G_we_y_Bigint);
    let (res_We_ecPoint: EcPoint) = WeierstrassArithmetics.double(G_we_Bigint);
    let (res_we_x: Uint256) = bigint_to_uint256(res_We_ecPoint.x);
    let (res_we_y: Uint256) = bigint_to_uint256(res_We_ecPoint.y);

    tempvar res_We = AffinePoint(res_we_x, res_we_y);
    %{ print_u_256_info(ids.res_We.x, "res_We.x") %}
    %{ print_u_256_info(ids.res_We.y, "res_We.y") %}
    // let G_We = AffinePoint(
    //     Uint256(226854911280625642308916404954512303194, 56713727820156410577229101238628035242),
    //     Uint256(194385589255140534133473644113281602521, 43439275391615930100543143142450843980),
    // );
    // let res_We: AffinePoint = WeierstrassArithmetics.double(G_We);  // , Uint256(111, 0));

    // let res_Mt = bijections.from_weierstrass(res_We);
    // let res_Ed = bijections.from_montgomery(res_Mt);
    // %{ print_u_256_info(ids.res_Ed.x, "res_ed.x") %}
    // %{ print_u_256_info(ids.res_Ed.y, "res_ed.y") %}

    return ();
}

@external
func test_mod_double{range_check_ptr, bitwise_ptr: BitwiseBuiltin*}() {
    alloc_locals;
    __setup__();
    let ZERO = ExtendedPoint(Uint256(0, 0), Uint256(1, 0), Uint256(1, 0), Uint256(0, 0));
    let G = ExtendedPoint(
        Uint256(Gx_low, Gx_high),
        Uint256(Gy_low, Gy_high),
        Uint256(1, 0),
        Uint256(43784682192424479926751423844859764131, 137613371725791198171896662509163897725),
    );

    // let G_We = AffinePoint(
    //     Uint256(
    //     225642940045174434350236119063462604597,
    //     56455135111361580725276067162451215896),
    //     Uint256(275730766544855198133974066020848770324, 55527814943294651515557878935034888995),
    // );

    // %{ print_u_256_info(ids.G_We.x, "G_We.x") %}
    // %{ print_u_256_info(ids.G_We.y, "G_We.Y") %}

    let (G_we_x_Bigint: BigInt3) = uint256_to_bigint(
        Uint256(
        225642940045174434350236119063462604597,
        56455135111361580725276067162451215896),
    );
    let (G_we_y_Bigint: BigInt3) = uint256_to_bigint(
        Uint256(275730766544855198133974066020848770324, 55527814943294651515557878935034888995)
    );
    let G_we_Bigint = EcPoint(G_we_x_Bigint, G_we_y_Bigint);
    let (res_We_ecPoint: EcPoint) = WeierstrassArithmetics.double(G_we_Bigint);
    // let (res_we_x: Uint256) = bigint_to_uint256(res_We_ecPoint.x);
    // let (res_we_y: Uint256) = bigint_to_uint256(res_We_ecPoint.y);

    // tempvar res_We = AffinePoint(res_we_x, res_we_y);
    // %{ print_u_256_info(ids.res_We.x, "res_We.x") %}
    // %{ print_u_256_info(ids.res_We.y, "res_We.y") %}

    return ();
}

@external
func test_mod_mul{range_check_ptr, bitwise_ptr: BitwiseBuiltin*}() {
    alloc_locals;
    __setup__();
    let ZERO = ExtendedPoint(Uint256(0, 0), Uint256(1, 0), Uint256(1, 0), Uint256(0, 0));
    let G = ExtendedPoint(
        Uint256(Gx_low, Gx_high),
        Uint256(Gy_low, Gy_high),
        Uint256(1, 0),
        Uint256(43784682192424479926751423844859764131, 137613371725791198171896662509163897725),
    );

    // let G_We = AffinePoint(
    //     Uint256(
    //     225642940045174434350236119063462604597,
    //     56455135111361580725276067162451215896),
    //     Uint256(275730766544855198133974066020848770324, 55527814943294651515557878935034888995),
    // );

    // %{ print_u_256_info(ids.G_We.x, "G_We.x") %}
    // %{ print_u_256_info(ids.G_We.y, "G_We.Y") %}

    let (G_we_x_Bigint: BigInt3) = uint256_to_bigint(
        Uint256(
        225642940045174434350236119063462604597,
        56455135111361580725276067162451215896),
    );
    let (G_we_y_Bigint: BigInt3) = uint256_to_bigint(
        Uint256(275730766544855198133974066020848770324, 55527814943294651515557878935034888995)
    );
    let s = BigInt3(
        35150989507602953651859135, 33076049803196078643974141, 276799551127567447115016
    );

    let G_we_Bigint = EcPoint(G_we_x_Bigint, G_we_y_Bigint);
    let res_We_ecPoint: EcPoint = WeierstrassArithmetics.scalar_mul(G_we_Bigint, s);
    let (res_we_x: Uint256) = bigint_to_uint256(res_We_ecPoint.x);
    let (res_we_y: Uint256) = bigint_to_uint256(res_We_ecPoint.y);

    tempvar res_We = AffinePoint(res_we_x, res_we_y);
    %{ print_u_256_info(ids.res_We.x, "res_We.x") %}
    %{ print_u_256_info(ids.res_We.y, "res_We.y") %}

    return ();
}

// @external
// func test_mod_mul{range_check_ptr, bitwise_ptr: BitwiseBuiltin*}() {
//     alloc_locals;
//     __setup__();
//     let ZERO = ExtendedPoint(Uint256(0, 0), Uint256(1, 0), Uint256(1, 0), Uint256(0, 0));
//     let G = ExtendedPoint(
//         Uint256(Gx_low, Gx_high),
//         Uint256(Gy_low, Gy_high),
//         Uint256(1, 0),
//         Uint256(43784682192424479926751423844859764131, 137613371725791198171896662509163897725),
//     );

// // %{ print_u_256_info(ids.G_We.x, "G_We.x") %}
//     // %{ print_u_256_info(ids.G_We.y, "G_We.Y") %}

// let (G_we_x_Bigint: BigInt3) = uint256_to_bigint(
//         Uint256(
//         225642940045174434350236119063462604597,
//         56455135111361580725276067162451215896),
//     );
//     let (G_we_y_Bigint: BigInt3) = uint256_to_bigint(
//         Uint256(275730766544855198133974066020848770324, 55527814943294651515557878935034888995)
//     );
//     // let s = BigInt3(
//     //     35150989507602953651859135, 33076049803196078643974141, 276799551127567447115016
//     // );
//     let s = BigInt3(2, 0, 0);

// let G_we_Bigint = EcPoint(G_we_x_Bigint, G_we_y_Bigint);
//     let (res_We_ecPoint: EcPoint) = WeierstrassArithmetics.scalar_mul(G_we_Bigint, s);
//     let (res_we_x: Uint256) = bigint_to_uint256(res_We_ecPoint.x);
//     let (res_we_y: Uint256) = bigint_to_uint256(res_We_ecPoint.y);

// tempvar res_We = AffinePoint(res_we_x, res_we_y);
//     %{ print_u_256_info(ids.res_We.x, "res_We.x") %}
//     %{ print_u_256_info(ids.res_We.y, "res_We.y") %}

// return ();
// }
