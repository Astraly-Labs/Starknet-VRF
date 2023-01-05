from starkware.cairo.common.alloc import alloc
from starkware.cairo.common.cairo_builtins import BitwiseBuiltin
from starkware.cairo.common.math import unsigned_div_rem as felt_divmod
from starkware.cairo.common.cairo_keccak.keccak import keccak, finalize_keccak, keccak_add_uint256s
from starkware.cairo.common.uint256 import Uint256, uint256_reverse_endian, uint256_unsigned_div_rem
from starkware.cairo.common.cairo_secp.bigint import BigInt3, uint256_to_bigint, bigint_to_uint256
from lib.curve import EcPoint
from lib.utils import encode_packed
from lib.u255 import u255, Uint384, Uint768
from lib.f25519 import f25519
from lib.h2c import _ecvrf_hash_to_curve_elligator2_25519
from lib.ed25519 import WeierstrassArithmetics, bijections

from lib.curve import (
    P_low,
    P_high,
    P_minus_A_low,
    P_minus_A_high,
    A,
    P_min_1_div_2_low,
    P_min_1_div_2_high,
    minus_A_low,
    minus_A_high,
    II_low,
    II_high,
    D_low,
    D_high,
    Gx_low,
    Gx_high,
    Gy_low,
    Gy_high,
    AffinePoint,
)

const Y_x_low = 216777731952371746420703551196102372892;
const Y_x_high = 93091407441439483731424260048654748620;
const Y_y_low = 279514667457029326575282730897876635171;
const Y_y_high = 169682413972660886576215326048749138346;

// Returns 0 if x is even <=> least significant bit = 0
func LSB{range_check_ptr}(x: felt) -> Uint256 {
    let (q, r) = felt_divmod(x, 2);
    if (r == 0) {
        let res = Uint256(0, 0);
        return res;
    } else {
        let res = Uint256(0, 170141183460469231731687303715884105728);
        return res;
    }
}
func encode_point{range_check_ptr, bitwise_ptr: BitwiseBuiltin*}(p: AffinePoint) -> Uint256 {
    alloc_locals;
    let lsb_x: Uint256 = LSB(p.x.low);
    let res = u255.add(lsb_x, p.y);
    // let (res) = uint256_reverse_endian(res);
    return res;
}
func ecvrf_verify{range_check_ptr, bitwise_ptr: BitwiseBuiltin*}(
    alpha_string: Uint256,
    gamma_string: AffinePoint,
    c_string_little: felt,
    s_string_little: Uint256,
) -> () {
    alloc_locals;
    tempvar encoded_public_key_point = Uint256(
        279514667457029326575282730897876635171, 169682413972660886576215326048749138346
        );
    %{ print_u_256_info(ids.encoded_public_key_point, 'encoded Y') %}
    let (keccak_ptr: felt*) = alloc();
    local keccak_ptr_start: felt* = keccak_ptr;
    let (s) = uint256_to_bigint(s_string_little);
    let (c) = uint256_to_bigint(Uint256(c_string_little, 0));
    let (H: EcPoint, H_Te: AffinePoint) = _ecvrf_hash_to_curve_elligator2_25519(
        encoded_public_key_point, alpha_string
    );
    let gamma_Mt = bijections.to_montgomery(gamma_string);
    let gamma_We = bijections.to_weierstrass(gamma_Mt);
    let gamma = bijections.affine_to_ec_point(gamma_We);

    // 5. u =  s*B - c*Y
    let G = AffinePoint(Uint256(Gx_low, Gx_high), Uint256(Gy_low, Gy_high));
    let G_Mt: AffinePoint = bijections.to_montgomery(G);
    let G_We: AffinePoint = bijections.to_weierstrass(G_Mt);
    let G_We_Bigint: EcPoint = bijections.affine_to_ec_point(G_We);

    %{ print_u_256_info(ids.G_We.x, "G_We.x") %}
    %{ print_u_256_info(ids.G_We.y, "G_We.Y") %}
    let Y = AffinePoint(Uint256(Y_x_low, Y_x_high), Uint256(Y_y_low, Y_y_high));
    let Y_Mt: AffinePoint = bijections.to_montgomery(Y);
    let Y_We: AffinePoint = bijections.to_weierstrass(Y_Mt);
    let Y_We_Bigint: EcPoint = bijections.affine_to_ec_point(Y_We);

    let s_b = WeierstrassArithmetics.scalar_mul_s_by_G(G_We_Bigint, s);
    let c_y = WeierstrassArithmetics.scalar_mul_Y_by_c(Y_We_Bigint, c);
    let (nc_y) = WeierstrassArithmetics.ec_negate(c_y);
    let (u_Wei) = WeierstrassArithmetics.ec_add(s_b, nc_y);

    // 6. v = s*H - c*Gamma
    // %{ print_u_256_info(ids.res_Mt.x, "res_Mt.x") %}
    // %{ print_u_256_info(ids.res_Mt.y, "res_Mt.Y") %}

    let s_h = WeierstrassArithmetics.scalar_mul(H, s);
    let c_g = WeierstrassArithmetics.scalar_mul_by_c(gamma, c);
    let (nc_g) = WeierstrassArithmetics.ec_negate(c_g);
    let (v_Wei) = WeierstrassArithmetics.ec_add(s_h, nc_g);

    // Convert back to Twisted Edwards
    let u_wei_affine = bijections.ec_point_to_affine(u_Wei);
    let v_wei_affine = bijections.ec_point_to_affine(v_Wei);
    let u_Mt = bijections.from_weierstrass(u_wei_affine);
    let v_Mt = bijections.from_weierstrass(v_wei_affine);

    let u: AffinePoint = bijections.from_montgomery(u_Mt);
    let v: AffinePoint = bijections.from_montgomery(v_Mt);

    // 7. Hash Points
    let h_compact: Uint256 = encode_point(H_Te);
    let gamma_compact: Uint256 = encode_point(gamma_string);
    let u_compact: Uint256 = encode_point(u);
    let v_compact: Uint256 = encode_point(v);
    %{ print_affine_info(ids.H_Te, 'p1') %}
    %{ print_affine_info(ids.gamma_string, 'p2') %}
    %{ print_affine_info(ids.u, 'p3') %}
    %{ print_affine_info(ids.v, 'p4') %}

    %{ print_u_256_info(ids.h_compact, 'p1') %}
    %{ print_u_256_info(ids.gamma_compact, 'p2') %}
    %{ print_u_256_info(ids.u_compact, 'p3') %}
    %{ print_u_256_info(ids.v_compact, 'p4') %}
    let packed_points_input: Uint256* = alloc();
    assert packed_points_input[0] = h_compact;
    assert packed_points_input[1] = gamma_compact;
    assert packed_points_input[2] = u_compact;
    assert packed_points_input[3] = v_compact;
    let packed_input: Uint256* = encode_packed.pack_u256_little_auto(packed_points_input, 4);
    // %{ print_u_256_info(ids.packed_input, 'packed_input') %}
    %{ print_u256_array_little(ids.packed_input.address_, 4) %}
    let (inputs) = alloc();
    let inputs_start = inputs;

    keccak_add_uint256s{inputs=inputs}(n_elements=4, elements=packed_input, bigend=0);
    assert inputs_start[16] = 516;  // int.from_bytes(b'\x04\x02', 'little'), suite string = \x04
    with keccak_ptr {
        let (message_hash) = keccak(inputs=inputs_start, n_bytes=130);
    }
    %{ print_u_256_info(ids.message_hash, 'keccak(to_hash)') %}
    let (_, cp) = uint256_unsigned_div_rem(message_hash, Uint256(0, 1));
    // let res_Mt = bijections.from_weierstrass(res);
    // let res_Ed = bijections.from_montgomery(res_Mt);

    %{
        print_felt_info_little(ids.cp.low + (ids.cp.high <<128), 'cp')
        print_felt_info_little(ids.c_string_little,'c')
    %}

    with_attr error_message("Invalid") {
        assert cp.low = c_string_little;
    }
    // output : hash(SUITE_STRING + three_string + _encode_point(cofactor_gamma))
    return ();
}
