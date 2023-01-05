from starkware.cairo.common.cairo_builtins import BitwiseBuiltin
from starkware.cairo.common.uint256 import Uint256, uint256_reverse_endian, uint256_eq
from lib.f25519 import f25519
from lib.ed25519 import WeierstrassArithmetics, bijections
from lib.u255 import u255, u512, Uint768
from lib.utils import clear_high_order_bit_128
from lib.curve import AffinePoint, EcPoint
from starkware.cairo.common.cairo_secp.bigint import BigInt3
from starkware.cairo.common.alloc import alloc
from starkware.cairo.common.math import unsigned_div_rem as felt_divmod

from starkware.cairo.common.registers import get_fp_and_pc
from starkware.cairo.common.cairo_keccak.keccak import keccak, finalize_keccak, keccak_add_uint256s
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
    minus_D_low,
    minus_D_high,
    II_low,
    II_high,
    D_low,
    D_high,
)

func _ecvrf_hash_to_curve_elligator2_25519{range_check_ptr, bitwise_ptr: BitwiseBuiltin*}(
    public_key: Uint256, alpha_string: Uint256
) -> (h_We: EcPoint, h_Te: AffinePoint) {
    alloc_locals;
    let (__fp__, _) = get_fp_and_pc();
    // keccak_ptr needs to be initialized to use the keccak functions
    let (keccak_ptr: felt*) = alloc();
    local keccak_ptr_start: felt* = keccak_ptr;
    // //////////////////////
    let (alpha_string_split) = alloc();
    let alpha_string_split_start = alpha_string_split;
    keccak_add_uint256s{inputs=alpha_string_split}(
        n_elements=1, elements=cast(&alpha_string, Uint256*), bigend=0
    );
    %{ print_u_256_info(ids.alpha_string, 'alpha_string') %}

    with keccak_ptr {
        let (k_alpha_string: Uint256) = keccak(inputs=alpha_string_split_start, n_bytes=32);
    }
    %{ print_u_256_info(ids.k_alpha_string, 'keccak(alpha_string)') %}
    // let (k_alpha_string) = uint256_reverse_endian(k_alpha_string);
    let (local input_256: Uint256*) = alloc();
    assert input_256[0] = k_alpha_string;  // This is 32 bytes length
    assert input_256[1] = public_key;  // This is 32 bytes length

    let (inputs) = alloc();
    let inputs_start = inputs;

    // We use keccak_add_uint256s to properly format the the two 32 bytes (Uint256) input into an array of 8-bytes felts.

    keccak_add_uint256s{inputs=inputs}(n_elements=2, elements=input_256, bigend=0);
    assert inputs_start[8] = 1025;  // int.from_bytes(b'\x01\x04', 'little'), suite string = \x04
    %{ print_felt_info(memory[ids.inputs_start+0],'0') %}
    %{ print_felt_info(memory[ids.inputs_start+1],'1') %}
    %{ print_felt_info(memory[ids.inputs_start+2],'2') %}
    %{ print_felt_info(memory[ids.inputs_start+3],'3') %}
    %{ print_felt_info(memory[ids.inputs_start+4],'4') %}
    %{ print_felt_info(memory[ids.inputs_start+5],'5') %}
    %{ print_felt_info(memory[ids.inputs_start+6],'6') %}
    %{ print_felt_info(memory[ids.inputs_start+7],'7') %}
    %{ print_felt_info(memory[ids.inputs_start+8],'8') %}

    with keccak_ptr {
        let (message_hash) = keccak(inputs=inputs_start, n_bytes=66);
    }
    %{ print_u_256_info(ids.message_hash, 'keccak(to_hash)') %}

    %{
        def from_uint(uint):
            """Takes in uint256-ish tuple, returns value."""
            return uint.low + (uint.high << 128)
        print("yo")
    %}
    %{ print(f"message_hash : {int.to_bytes(ids.message_hash.low+(ids.message_hash.high<<128), 32, 'little')}") %}
    %{ print(f"message_hash : {ids.message_hash.low+(ids.message_hash.high<<128)}") %}

    let high_order_bit_cleared = clear_high_order_bit_128(message_hash.high);
    tempvar message_hash2 = Uint256(low=message_hash.low, high=high_order_bit_cleared);
    %{ print(f"message_hash2 : {int.to_bytes(ids.message_hash2.low+(ids.message_hash2.high<<128), 32, 'little')}") %}
    %{ print(f"message_hash2 : {ids.message_hash2.low+(ids.message_hash2.high<<128)}") %}

    // 8. u = - A / (1 + 2*(r^2) ) mod p (note: the inverse of (1+2*(r^2)) modulo p is guaranteed to exist)
    // let (two_r, carry) = u255.add_carry(message_hash2, message_hash2);
    // %{ print(f"carry1 : {ids.carry}") %}
    let two_r_squared: u512 = u255.mul2ab(message_hash2, message_hash2);
    %{ print_u_512_info(ids.two_r_squared, "2rr") %}

    // let (two_r_squared_low: Uint256, carry: felt) = u255.add_carry(
    //     Uint256(1, 0), Uint256(two_r_squared.d0, two_r_squared.d1)
    // );
    // %{ print(f"carry2 : {ids.carry}, 1+2rr={pack_512(ids.two_r_squared_low, ids.two_r_squared_high, 128)}") %}
    // assert carry = 0;
    let one_plus_2r_squared: u512 = u255.add_u512_and_u256(two_r_squared, Uint256(1, 0));
    // tempvar one_plus_2r_squared = u512(
    //     two_r_squared_low.low, two_r_squared_low.low, two_r_squared.d2, two_r_squared.d3
    //     );
    %{ print_u_512_info(ids.one_plus_2r_squared, "1+2rr") %}
    let inv: Uint256 = f25519.inv_mod_p_uint512(one_plus_2r_squared);
    %{ print_u_256_info(ids.inv, "inv") %}

    let u_512: u512 = u255.mul(Uint256(P_minus_A_low, P_minus_A_high), inv);
    let u: Uint256 = f25519.u512_modulo_p_25519(u_512);
    %{ print(f"u={ids.u.low + (ids.u.high<< 128)}") %}

    // 9. w = u * (u ** 2 + A * u + 1) % PRIME. A is 19 bits, fixed.

    let u_sq: Uint256 = f25519.square(u);
    let Au: u512 = u255.mul(Uint256(A, 0), u);
    let (Au_one_low: Uint256, carry) = u255.add_carry(Uint256(Au.d0, Au.d1), Uint256(1, 0));
    assert carry = 0;

    let Au_one: Uint256 = f25519.u512_modulo_p_25519(
        u512(Au_one_low.low, Au_one_low.high, Au.d2, Au.d3)
    );
    let u_sq_au_one: Uint256 = f25519.add(Au_one, u_sq);
    // let u_sq_au_one_reduced: Uint256 = f25519.u512_modulo_p_25519(u_sq_au_one);

    let w: Uint256 = f25519.mul(u, u_sq_au_one);
    %{ print(f"w_255={ids.w.low + (ids.w.high<<128)}") %}

    // 10-11
    let (is_sq, _) = f25519.get_square_root(w);
    local final_u: Uint256;
    if (is_sq == 0) {
        // tempvar e = Uint256(1, 0);
        let sss = u255.super_sub(Uint256(minus_A_low, minus_A_high), u);
        assert final_u.low = sss.low;
        assert final_u.high = sss.high;
        // tempvar final_u = final_u;
        tempvar range_check_ptr = range_check_ptr;
        // tempvar bitwise_ptr = bitwise_ptr;
    } else {
        assert final_u.low = u.low;
        assert final_u.high = u.high;
        // tempvar final_u = final_u;

        tempvar range_check_ptr = range_check_ptr;
        // tempvar bitwise_ptr = bitwise_ptr;
    }

    // 12-13.
    let final_u_minus_one = f25519.sub(final_u, Uint256(1, 0));
    let final_u_plus_one = u255.add(final_u, Uint256(1, 0));

    let y_coordinate = f25519.div(final_u_minus_one, final_u_plus_one);
    let y_coordinate_high = clear_high_order_bit_128(y_coordinate.high);
    tempvar yy = Uint256(y_coordinate.low, y_coordinate_high);
    %{ print(f"final_u={ids.final_u.low + (ids.final_u.low<< 128)}") %}
    %{ print_u_256_info(ids.yy, 'yy') %}
    // 14
    // no need to remove the first bit as we operate within p ?

    let x = _x_recover_hint(yy);
    %{ print(f"xx={ids.x.low + (ids.x.high<< 128)}") %}

    // After we are done with using keccak functions, we call finalize_keccak to prevent malicious prover actions.
    finalize_keccak(keccak_ptr_start=keccak_ptr_start, keccak_ptr_end=keccak_ptr);

    let h_prelim = AffinePoint(x, yy);
    let h_prelim = bijections.to_montgomery(h_prelim);
    let h_prelim = bijections.to_weierstrass(h_prelim);
    let h: EcPoint = bijections.affine_to_ec_point(h_prelim);
    let h = WeierstrassArithmetics.scalar_mul_by_8(h, BigInt3(8, 0, 0));
    let h_We: AffinePoint = bijections.ec_point_to_affine(h);
    let h_Mt = bijections.from_weierstrass(h_We);
    let h_Te = bijections.from_montgomery(h_Mt);
    return (h, h_Te);
}

func _x_recover_hint{range_check_ptr}(y: Uint256) -> Uint256 {
    alloc_locals;
    let y_sq = f25519.square(y);
    let u = f25519.sub(y_sq, Uint256(1, 0));

    let D_times_y_sq: Uint256 = f25519.mul(Uint256(D_low, D_high), y_sq);

    let v: Uint256 = f25519.add(D_times_y_sq, Uint256(1, 0));

    let xx = f25519.div(u, v);
    local x: Uint256;
    // To whitelist
    %{
        PRIME = 2**255 - 19
        II = pow(2, (PRIME - 1) // 4, PRIME)

        xx = ids.xx.low + (ids.xx.high<<128)
        x = pow(xx, (PRIME + 3) // 8, PRIME)
        if (x * x - xx) % PRIME != 0:
            x = (x * II) % PRIME
        if x % 2 != 0:
            x = PRIME - x
        ids.x.low = x & ((1<<128)-1)
        ids.x.high = x >> 128
    %}
    let x_sq = f25519.square(x);
    let x_sq_y_sq = f25519.mul(x_sq, y_sq);
    let D_x_sq_y_sq = f25519.mul(Uint256(minus_D_low, minus_D_high), x_sq_y_sq);
    let right_side = u255.super_sub(Uint256(1, 0), D_x_sq_y_sq);
    let left_side = u255.super_sub(y_sq, x_sq);
    // Cofactors ?
    assert left_side.high = right_side.high;
    assert left_side.low = right_side.low;
    %{ print_u_256_info(ids.x,'H2CX') %}
    %{ print_u_256_info(ids.y,'H2CY') %}

    return x;
}
// TODO : optimize
func _x_recover{range_check_ptr}(y: Uint256) -> Uint256 {
    alloc_locals;
    let y_sq = f25519.square(y);
    let y_sq_min_one = f25519.sub(y_sq, Uint256(1, 0));

    %{ print(f"y_sq_min_one={ids.y_sq_min_one.low + (ids.y_sq_min_one.high<< 128)}") %}

    let D_times_y_sq: Uint256 = f25519.mul(Uint256(D_low, D_high), y_sq);

    %{ print(f"d_ysq={ids.D_times_y_sq.low + (ids.D_times_y_sq.high<< 128)}") %}
    let u = y_sq_min_one;
    let v: Uint256 = f25519.add(D_times_y_sq, Uint256(1, 0));

    %{ print(f"v={ids.v.low + (ids.v.high<< 128)}") %}

    let xx = f25519.div(y_sq_min_one, v);
    %{ print_u_256_info(ids.xx, 'xx xrecover') %}
    let x = f25519.pow_prime_3_div_8(xx);
    %{ print_u_256_info(ids.x, 'x xrecover') %}

    let x_sq = f25519.square(x);
    let xx_min_x_sq = f25519.sub(x, x_sq);
    let (is_zero) = uint256_eq(xx_min_x_sq, Uint256(0, 0));
    tempvar x_low = x.low;
    local new_x: Uint256;
    if (is_zero == 0) {
        let z = f25519.mul(x, Uint256(II_low, II_high));
        assert new_x.low = z.low;
        assert new_x.high = z.high;
        tempvar range_check_ptr = range_check_ptr;
    } else {
        tempvar range_check_ptr = range_check_ptr;
        assert new_x.low = x.low;
        assert new_x.high = x.high;
    }

    let x_is_odd = is_odd(x_low);
    local new_new_x: Uint256;
    if (x_is_odd == 1) {
        let zz = u255.super_sub(Uint256(P_low, P_high), x);
        assert new_new_x.low = zz.low;
        assert new_new_x.high = zz.high;
        tempvar range_check_ptr = range_check_ptr;
    } else {
        tempvar range_check_ptr = range_check_ptr;
        assert new_new_x.low = new_x.low;
        assert new_new_x.high = new_x.high;
    }

    return new_new_x;
}

// Returns 0 if x is even <=> least significant bit = 0
func is_odd{range_check_ptr}(x: felt) -> felt {
    alloc_locals;
    let (q, r) = felt_divmod(x, 2);
    if (r == 0) {
        return 0;
    }
    return 1;
}
