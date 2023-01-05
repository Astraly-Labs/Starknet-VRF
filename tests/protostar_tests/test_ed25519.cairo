%lang starknet
from starkware.cairo.common.cairo_builtins import HashBuiltin
from starkware.cairo.common.cairo_builtins import BitwiseBuiltin

from starkware.cairo.common.uint256 import Uint256, uint256_mul
from lib.h2c import _ecvrf_hash_to_curve_elligator2_25519
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
from lib.ed25519 import TwistedArithmetics
from lib.f25519 import f25519
from lib.utils import clear_high_order_bit_128, split_64, split_128
from lib.u255 import u255, u512, Uint768, Uint384, HALF_SHIFT

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
func square_e{range_check_ptr}(a: Uint384) -> (low: Uint384, high: Uint384) {
    alloc_locals;
    let (a0, a1) = split_64(a.d0);
    let (a2, a3) = split_64(a.d1);
    let (a4, a5) = split_64(a.d2);

    const HALF_SHIFT2 = 2 * HALF_SHIFT;
    local a0_2 = a0 * 2;
    local a34 = a3 + a4 * HALF_SHIFT2;

    let (res0, carry) = split_128(a0 * (a0 + a1 * HALF_SHIFT2));
    let (res2, carry) = split_128(a.d1 * a0_2 + a1 * (a1 + a2 * HALF_SHIFT2) + carry);
    let (res4, carry) = split_128(
        a.d2 * a0_2 + (a3 + a34) * a1 + a2 * (a2 + a3 * HALF_SHIFT2) + carry
    );
    let (res6, carry) = split_128((a5 * a1 + a.d2 * a2) * 2 + a3 * a34 + carry);
    let (res8, carry) = split_128(a5 * (a3 + a34) + a4 * a4 + carry);
    // let (res10, carry) = split_64(a5*a5 + carry)

    return (
        low=Uint384(d0=res0, d1=res2, d2=res4), high=Uint384(d0=res6, d1=res8, d2=a5 * a5 + carry),
    );
}
@external
func test_clear_high_order_bit{
    syscall_ptr: felt*, range_check_ptr, pedersen_ptr: HashBuiltin*, bitwise_ptr: BitwiseBuiltin*
}() {
    alloc_locals;
    let x = 2 ** 128 - 1;
    let res = clear_high_order_bit_128(x);
    %{ print(f"res {ids.res}") %}
    assert res = 2 ** 127 - 1;
    return ();
}
// @external
// func test_clear_first_64{
//     syscall_ptr: felt*, range_check_ptr, pedersen_ptr: HashBuiltin*, bitwise_ptr: BitwiseBuiltin*
// }() {
//     alloc_locals;
//     let x = 2 ** 128 - 1 - 2 ** 20;
//     let res = clear_first_64_bits(x);
//     %{ print(f"res {ids.res}") %}
//     assert res = 18446744073708503039;
//     return ();
// }
func split_64_scam{range_check_ptr}(a: felt) -> (low: felt, high: felt) {
    alloc_locals;
    local low: felt;
    local high: felt;
    low = a;
    high = 0;
    assert a = low + high * HALF_SHIFT;

    return (low, high);
}
func uint256_mul_scammed{range_check_ptr}(a: Uint256, b: Uint256) -> (low: Uint256, high: Uint256) {
    alloc_locals;
    let (a0, a1) = split_64_scam(a.low);
    let (a2, a3) = split_64_scam(a.high);
    let (b0, b1) = split_64_scam(b.low);
    let (b2, b3) = split_64_scam(b.high);
    let (res0, carry) = split_64_scam(a0 * b0);
    let (res1, carry) = split_64_scam(a1 * b0 + a0 * b1 + carry);
    let (res2, carry) = split_64_scam(a2 * b0 + a1 * b1 + a0 * b2 + carry);
    let (res3, carry) = split_64_scam(a3 * b0 + a2 * b1 + a1 * b2 + a0 * b3 + carry);
    let (res4, carry) = split_64_scam(a3 * b1 + a2 * b2 + a1 * b3 + carry);
    let (res5, carry) = split_64_scam(a3 * b2 + a2 * b3 + carry);
    let (res6, carry) = split_64_scam(a3 * b3 + carry);

    return (
        low=Uint256(low=res0 + HALF_SHIFT * res1, high=res2 + HALF_SHIFT * res3),
        high=Uint256(low=res4 + HALF_SHIFT * res5, high=res6 + HALF_SHIFT * carry),
    );
}

@external
func test_h2c{
    syscall_ptr: felt*, range_check_ptr, pedersen_ptr: HashBuiltin*, bitwise_ptr: BitwiseBuiltin*
}() {
    alloc_locals;
    __setup__();
    let pbk = Uint256(
        49797504309839299353147461194798889772, 302399301418007558314037202906863157199
    );  // to_uint(int.from_bytes(_encode_point(_scalar_multiply(p=BASE,e=10)), 'little'))
    let alpha = Uint256(
        low=163705051694923355836323137323695835676, high=267062820370413217226717485711181863254
    );  // to_uint(int.from_bytes(keccak(b"hello"), 'little'))
    _ecvrf_hash_to_curve_elligator2_25519(public_key=pbk, alpha_string=alpha);
    // let pbk = Uint256(
    //     24384833419280938441248282869368043283, 230706774633386159594410180559446675406
    // );  // to_uint(int.from_bytes(_encode_point(_scalar_multiply(p=BASE,e=11)), 'little'))
    // _ecvrf_hash_to_curve_elligator2_25519(public_key=pbk, alpha_string=alpha);

    return ();
}
@external
func test_inv_mod{
    syscall_ptr: felt*, range_check_ptr, pedersen_ptr: HashBuiltin*, bitwise_ptr: BitwiseBuiltin*
}() {
    alloc_locals;
    let x = u512(
        199922541599641497118593248398018987107,
        96593935205676288982313329380101447963,
        133077342791584132607185689248168144743,
        106672199179311949703724999866412007623,
    );

    let x_inv = f25519.inv_mod_p_uint512(x);
    assert x_inv.low = 284946103120669617102927834399154717678;
    assert x_inv.high = 54996237036786620694603418430173126452;
    return ();
}

@external
func test_pow{
    syscall_ptr: felt*, range_check_ptr, pedersen_ptr: HashBuiltin*, bitwise_ptr: BitwiseBuiltin*
}() {
    alloc_locals;
    let x = Uint256(
        268691350397907185738371061122832407442, 144555504316151822805459321401694188150
    );  // xx%PRIME
    let res = f25519.pow(
        x, Uint256(340282366920938463463374607431768211454, 21267647932558653966460912964485513215)
    );
    // %{ print(f"res {ids.res}") %}
    assert res.low = 39100679060079264080458647432603284997;
    assert res.high = 33128569487467647680503087226593960032;
    // 11273068037900272381704093689909174923119171119597651152594999593181591811589
    return ();
}

@external
func test_pow2{
    syscall_ptr: felt*, range_check_ptr, pedersen_ptr: HashBuiltin*, bitwise_ptr: BitwiseBuiltin*
}() {
    alloc_locals;
    __setup__();
    let x = Uint256(
        268691350397907185738371061122832407442, 144555504316151822805459321401694188150
    );
    let pow_p_5_8 = f25519.pow_prime_3_div_8(x);

    assert pow_p_5_8.low = 39100679060079264080458647432603284997;
    assert pow_p_5_8.high = 33128569487467647680503087226593960032;
    return ();
}
// Basic test to copy and paste
//
// @external
// func test{
//     syscall_ptr : felt*, range_check_ptr, pedersen_ptr : HashBuiltin*, bitwise_ptr : BitwiseBuiltin*
// }(){
//     alloc_locals;
//     tempvar contract_address;
//     %{ ids.contract_address = context.contract_a_address %}
// return ();
// }

@external
func test_mul2{
    syscall_ptr: felt*, range_check_ptr, pedersen_ptr: HashBuiltin*, bitwise_ptr: BitwiseBuiltin*
}() {
    alloc_locals;
    let x = Uint256(
        268691350397907185738371061122832407442, 144555504316151822805459321401694188150
    );  // xx%PRIME
    let (res_low, res_high) = uint256_mul(x, x);
    %{
        def pack_512(u256_low, u256_high, num_bits_shift: int) -> int:
            limbs = (u256_low.low, u256_low.high, u256_high.low, u256_high.high)
            return sum(limb << (num_bits_shift * i) for i, limb in enumerate(limbs))
        print(pack_512(ids.res_low, ids.res_high, 128))
        print(ids.res_low.low, ids.res_low.high, ids.res_high.low, ids.res_high.high)
    %}
    assert res_low.low = 47351901840359301914610819813999648580;
    assert res_low.high = 4604374406688834925008046075820590703;
    assert res_high.low = 199045124928047743750422239792234787156;
    assert res_high.high = 61408688370126605859411795069843935425;
    let res2 = Uint768(res_low.low, res_low.high, res_high.low, res_high.high, 0, 0);
    return ();
}

@external
func test_mul3{
    syscall_ptr: felt*, range_check_ptr, pedersen_ptr: HashBuiltin*, bitwise_ptr: BitwiseBuiltin*
}() {
    alloc_locals;
    let x = Uint256(
        268691350397907185738371061122832407442, 144555504316151822805459321401694188150
    );  // xx%PRIME
    let res: u512 = u255.mul(x, x);
    assert res.d0 = 47351901840359301914610819813999648580;
    assert res.d1 = 4604374406688834925008046075820590703;
    assert res.d2 = 199045124928047743750422239792234787156;
    assert res.d3 = 61408688370126605859411795069843935425;
    return ();
}
@external
func test_add{
    syscall_ptr: felt*, range_check_ptr, pedersen_ptr: HashBuiltin*, bitwise_ptr: BitwiseBuiltin*
}() {
    alloc_locals;
    __setup__();
    let x = ExtendedPoint(
        Uint256(43192411618168306500369449636968528207, 127836812826537975251982703603520145015),
        Uint256(49797504309839299353147461194798889772, 132258117957538326582349899190979051471),
        Uint256(1, 0),
        Uint256(155466007355907588088263444829604398654, 72943317090724618555666500857162389375),
    );  // [to_uint(x) for x in to_extended([43500613248243327786121022071801015118933854441360174117148262713429272820047,45005105423099817237495816771148012388779685712352441364231470781391834741548 ])]
    let y = ExtendedPoint(
        Uint256(140259942142931294965626752625963497903, 137224085978424599938040703774585469598),
        Uint256(253144262263300720435102317091852409400, 25820880030508234660306446697074341360),
        Uint256(1, 0),
        Uint256(253879765200236411362134985095524777690, 26353387897677070768219998042781955100),
    );  // [to_uint(x) for x in to_extended((46694936775300686710656303283485882876784402425210400817529601134760286812591,8786390172762935853260670851718824721296437982862763585171334833968259029560))]

    let res: ExtendedPoint = TwistedArithmetics.add(x, y);

    %{ print(ids.res.x.low, ids.res.x.high, ids.res.y.low, ids.res.y.high) %}
    // assert res.x.d0 = 91297808838743694984660001710711986787;
    // assert res.x.d1 = 29711444290802994633705050679624375622;
    // assert res.y.d0 = 104021219304868236826575167050365978179;
    // assert res.y.d0 = 120910031807270091425397192357442325518;
    return ();
}

// @external
// func test_multiply_G{
//     syscall_ptr: felt*, pedersen_ptr: HashBuiltin*, range_check_ptr, bitwise_ptr: BitwiseBuiltin*
// }() {
//     __setup__();
//     let ZERO = ExtendedPoint(Uint256(0, 0), Uint256(1, 0), Uint256(1, 0), Uint256(0, 0));
//     let G = ExtendedPoint(
//         Uint256(Gx_low, Gx_high),
//         Uint256(Gy_low, Gy_high),
//         Uint256(1, 0),
//         Uint256(43784682192424479926751423844859764131, 137613371725791198171896662509163897725),
//     );
//     // s= 1657008116471393480753518857542198008763130933602946594165815725401803042495
//     let sG: ExtendedPoint = TwistedArithmetics._scalar_multiply_G_by_s_loop(
//         index=0, res=ZERO, s=Uint256(111, 0)
//     );
//     %{ print_u_256_info(ids.sG.x, "x") %}
//     %{ print_u_256_info(ids.sG.y, "y") %}
//     %{ print_u_256_info(ids.sG.z, "z") %}
//     %{ print_u_256_info(ids.sG.t, "t") %}

// %{ print_from_extended(ids.sG) %}

// return ();
// }
