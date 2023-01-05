%lang starknet
from starkware.cairo.common.cairo_builtins import HashBuiltin
from starkware.cairo.common.cairo_builtins import BitwiseBuiltin

from lib.verifier import ecvrf_verify
from starkware.cairo.common.uint256 import Uint256
from lib.u255 import u255
from lib.f25519 import f25519
from lib.ed25519 import bijections, EcPoint
from lib.curve import ExtendedPoint, AffinePoint
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
        def print_affine_info(p, pn):
            print(f"Affine Point {pn}")
            print_u_256_info(p.x, 'X')
            print_u_256_info(p.y, 'Y')

        def print_felt_info(u, un):
            print(f" {un}_{u.bit_length()}bits = {bin_8(u)}")
            print(f" {un} = {u}")
            print(f" {un} = {int.to_bytes(u, 8, 'little')}")

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
func test_verify{
    syscall_ptr: felt*, pedersen_ptr: HashBuiltin*, range_check_ptr, bitwise_ptr: BitwiseBuiltin*
}() {
    __setup__();
    let ZERO = ExtendedPoint(Uint256(0, 0), Uint256(1, 0), Uint256(1, 0), Uint256(0, 0));
    let gamma_Ted = AffinePoint(
        Uint256(192390340321515207807992917489602166392, 58758014839227359774768445342975695172),
        Uint256(18239844443173624386396059217848374739, 84035145635319752902361312099286819162),
    );

    let c = 7182018137573024061866498914007889479;
    let s = Uint256(204173955805175125808790801027651232919, 956074987569246721549546242559416368);
    let alpha_string = Uint256('ponmlkjihgfedcba', 'fedcbazyxwvutsrq');
    ecvrf_verify(alpha_string, gamma_Ted, c, s);

    return ();
}

@external
func test_encode_point{
    syscall_ptr: felt*, pedersen_ptr: HashBuiltin*, range_check_ptr, bitwise_ptr: BitwiseBuiltin*
}() {
    __setup__();
    let ZERO = ExtendedPoint(Uint256(0, 0), Uint256(1, 0), Uint256(1, 0), Uint256(0, 0));
    let gamma_Ted = AffinePoint(
        Uint256(186618695346065647475975190267294414760, 118040052493330149454319627980738656502),
        Uint256(199083777198409727313884107955243172217, 56789223528657733838044095376412440596),
    );

    return ();
}
