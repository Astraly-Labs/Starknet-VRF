from tools.minimal import SUITE_STRING, pow2, _encode_point, A, ORDER, _hash, PRIME, TWO_INV, _inverse, _decode_point, \
    _scalar_multiply, BASE, BASEx, D, _edwards_add, _ecvrf_hash_points, _ecvrf_nonce_generation_rfc8032, _ecvrf_decode_proof
from starkware.python.math_utils import is_quad_residue
import random
from starkware.python.math_utils import sqrt, isqrt, ec_double, ec_double_slope


def test_generator():
    count = 0
    total = 10000
    for i in range(1, PRIME):
        r = random.randint(1, PRIME)
        if not is_quad_residue(r, PRIME):
            count += 1
            print(r)
            assert is_quad_residue(r * ((PRIME - 1) // 2) % PRIME, PRIME)
    print(count, total, count / total, count / PRIME)


def check_is_on_curve_TEd(p):
    """Check to confirm point is on TwEd25519 ; return boolean"""
    x = p[0]
    y = p[1]
    result = (-x * x + y * y - 1 - D * x * x * y * y) % PRIME
    return result == 0


def check_is_on_curve_W(a, b, p,  prime):
    x = p[0]
    y = p[1]
    res = (x**3 + a*x + b) % prime
    assert res == y**2 % prime


def check_is_on_curve_Mt(A, B, p,  prime):
    x = p[0]
    y = p[1]
    res = (x**3 + A*x**2 + x) % prime
    assert res == B*y**2 % prime


# check_is_on_curve_W(0, 7, GG[0], GG[1], 0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f)
a = (3-A*A)*_inverse(3) % PRIME
b = (2*A*A*A % PRIME - 9*A % PRIME) % PRIME*_inverse(27) % PRIME

p = ()
p = (19298681539552699237261830834781317975544997444273427339909597334652188435546,
     14781619447589544791020593568409986887264606134616475288964881837755586237401)

check_is_on_curve_W(a, b, p, 2**255-19)


def to_montgomery(p_ed):
    A = 486662
    B = 4*_inverse(-1-D) % PRIME
    u_mt = (1+p_ed[1])*_inverse(1-p_ed[1]) % PRIME
    v_mt = (1+p_ed[1]) * _inverse((1-p_ed[1])*p_ed[0]) % PRIME
    check_is_on_curve_Mt(A=A, B=B, p=(u_mt, v_mt), prime=PRIME)
    return (u_mt, v_mt)


def to_weierstrass(p_mt):
    A = 486662
    B = 4*_inverse(-1-D) % PRIME
    t_We = (p_mt[0]*_inverse(B) % PRIME + A*_inverse(3*B) % PRIME) % PRIME
    v_We = p_mt[1]*_inverse(B) % PRIME

    a_raw = (3-A*A)*_inverse(3*B*B) % PRIME
    b_raw = (2*A*A*A % PRIME - 9*A % PRIME) % PRIME*_inverse(27*B*B*B) % PRIME

    check_is_on_curve_W(a_raw, b_raw, (t_We, v_We), 2**255-19)
    return (t_We, v_We)


# Convert to Mt and check on Mt (scaled, unscaled)
A = 486662
B = 4*_inverse(-1-D) % PRIME
u_mt = (1+BASE[1])*_inverse(1-BASE[1]) % PRIME
v_mt = (1+BASE[1]) * _inverse((1-BASE[1])*BASE[0]) % PRIME
scaling_factor = _inverse(sqrt(_inverse(B), PRIME))
v_mt_scaled = v_mt * scaling_factor % PRIME
check_is_on_curve_Mt(A=A, B=1, p=(u_mt, v_mt_scaled), prime=PRIME)
check_is_on_curve_Mt(A=A, B=B, p=(u_mt, v_mt), prime=PRIME)

# Convert Mt to We
t_We = (u_mt*_inverse(B) % PRIME + A*_inverse(3*B) % PRIME) % PRIME
v_We = v_mt*_inverse(B) % PRIME

a_raw = (3-A*A)*_inverse(3*B*B) % PRIME
b_raw = (2*A*A*A % PRIME - 9*A % PRIME) % PRIME*_inverse(27*B*B*B) % PRIME

check_is_on_curve_W(a_raw, b_raw, (t_We, v_We), 2**255-19)
check_is_on_curve_W(a, b, p, 2**255-19)
poo = ec_double((t_We, v_We), a_raw, 2**255-19)
check_is_on_curve_W(a_raw, b_raw, poo, PRIME)

# play

slope = ec_double_slope((t_We, v_We), a_raw, PRIME) % PRIME
x_sqr = t_We**2
slope_y = slope*v_We
to_assert = 3*x_sqr - 2 * slope_y + a_raw


def split_64(a):
    return (a & ((1 << 64) - 1), a >> 64)


def to_bigint(a):

    RC_BOUND = 2 ** 128
    BASE = 2**86
    low, high = split_128(a)
    D1_HIGH_BOUND = BASE ** 2 // RC_BOUND
    D1_LOW_BOUND = RC_BOUND // BASE
    d1_low, d0 = divmod(low, BASE)
    d2, d1_high = divmod(high, D1_HIGH_BOUND)
    d1 = d1_high * D1_LOW_BOUND + d1_low

    return (d0, d1, d2)


def bin_c(u):
    b = bin(u)
    f = b[0:10] + ' ' + b[10:19] + '...' + b[-16:-8] + ' ' + b[-8:]
    return f


def bin_64(u):
    b = bin(u)
    little = '0b' + b[2:][::-1]
    f = '0b' + ' '.join([b[2:][i:i + 64] for i in range(0, len(b[2:]), 64)])
    return f


def bin_8(u):
    b = bin(u)
    little = '0b' + b[2:][::-1]
    f = "0b" + ' '.join([little[2:][i:i + 8]
                         for i in range(0, len(little[2:]), 8)])
    return f


def from_uint(a):
    return a[0] + (a[1] << 128)


def split_128(a):
    return (a & ((1 << 128) - 1), a >> 128)


def square_e(a):
    (a0a1, a2a3) = split_128(a)
    (a0, a1) = split_64(a0a1)
    (a2, a3) = split_64(a2a3)
    half_shift = 2 ** 128

    (res0, carry) = split_128(a0 * (a0 + a1 * 2 ** 128))
    (res2, carry) = split_128(2 * a0 * a2a3 + a1 * (a1 + a2 * 2 ** 128) + carry)
    (res4, carry) = split_128(2 * a3 + 0 + a2 * (a2 + a3 * 2 ** 128) + carry)

    print(res0, res2, res4, a3 * a3 + carry)
    return res0 + (res2 << 128) + (res4 << 256) + ((a3 * a3 + carry) << 384)


def to_extended(p):
    # Takes (x,y), gives (x,y,z,t)
    return (p[0], p[1], 1, p[0] * p[1] % PRIME)


def eadd(a, b):
    x1 = a[0]
    x2 = b[0]
    y1 = a[1]
    y2 = b[1]
    z1 = a[2]
    z2 = b[2]
    t1 = a[3]
    t2 = b[3]
    A = (y1 - x1) * (y2 + x2) % PRIME
    print(f"ya_min_xa={y1 - x1}")
    print(f"yb_xb={y2 + x2}")
    print(f"A={A}")
    B = (y1 + x1) * (y2 - x2) % PRIME
    print(f"ya_xa={y1 + x1}")
    print(f"yb_min_xb={y2 - x2}")
    print(f"B={B}")
    F = (B - A) % PRIME
    print(f"F={F}")
    C = z1 * 2 * t2 % PRIME
    print(f"C_ur={z1 * 2 * t2}")
    print(f"C={C}")
    D = t1 * 2 * z2 % PRIME
    print(f"D_ur={t1 * 2 * z2}")
    print(f"D={D}")
    E = D + C
    print(f"E={E}")
    G = B + A
    print(f"G={G}")
    H = D - C
    print(f"H={H}")
    X3 = E * F % PRIME
    print(f"X3={X3}")
    Y3 = G * H % PRIME
    print(f"Y3={Y3}")
    T3 = E * H % PRIME
    Z3 = F * G % PRIME
    print(f"Z3={Z3}")
    print(f"T3={T3}")
    return ((X3, Y3, Z3, T3))


#######################################################
# p = _scalar_multiply(p=BASE, e=10)
# pp = _edwards_add(p, p)
# eadd(to_extended(p), to_extended(pp))

#######################################################

def ecvrf_prove(sk, alpha_string):
    """
    Input:
        sk - VRF private key
        alpha_string - input alpha, an octet string
    Output:
        pi_string - VRF proof, octet string of length ptLen+n+qLen
    """
    # 1. Use sk to derive the VRF secret scalar x and the VRF public key y = x*B
    #    (this derivation depends on the ciphersuite, as per Section 5.5; these values can
    #    be cached, for example, after key generation, and need not be re-derived each time)
    """Calculate and return the secret_scalar and the corresponding public_key
       secret_scalar is an integer; public_key is an encoded point string
    """
    h = bytearray(_hash(sk)[0:32])
    h[31] = int((h[31] & 0x7f) | 0x40)
    h[0] = int(h[0] & 0xf8)
    secret_scalar_x = int.from_bytes(h, 'little')
    public_point: List[int, int] = _scalar_multiply(p=BASE, e=secret_scalar_x)
    public_key_y: bytes = _encode_point(public_point)
    print(f"Public key point for {sk}: \n", public_point)
    # 2. H = ECVRF_hash_to_curve(suite_string, y, alpha_string)
    h = _ecvrf_hash_to_curve_elligator2_25519(
        SUITE_STRING, public_key_y, alpha_string)

    # 3. h_string = point_to_string(H)
    h_string = _decode_point(h)

    # 4. Gamma = x*H
    gamma = _scalar_multiply(p=h_string, e=secret_scalar_x)

    # 5. k = ECVRF_nonce_generation(sk, h_string)
    k = _ecvrf_nonce_generation_rfc8032(sk, h)

    # 6. c = ECVRF_hash_points(H, Gamma, k*B, k*H)
    k_b = _scalar_multiply(p=BASE, e=k)
    k_h = _scalar_multiply(p=h_string, e=k)
    c = _ecvrf_hash_points(h_string, gamma, k_b, k_h)

    # 7. s = (k + c*x) mod q
    s = (k + c * secret_scalar_x) % ORDER

    # 8. pi_string = point_to_string(Gamma) || int_to_string(c, n) || int_to_string(s, qLen)
    pi_string = _encode_point(gamma) + int.to_bytes(c,
                                                    16, 'little') + int.to_bytes(s, 32, 'little')
    # 9. Output pi_string
    return pi_string


def ecvrf_verify(y, pi_string, alpha_string):
    """
    Input:
        y - public key, an EC point
        pi_string - VRF proof, octet string of length ptLen+n+qLen
        alpha_string - VRF input, octet string
    Output:
        ("VALID", beta_string), where beta_string is the VRF hash output, octet string
        of length hLen; or "INVALID"
    """
    # 1. D = ECVRF_decode_proof(pi_string)
    d = _ecvrf_decode_proof(pi_string)

    # 2. If D is "INVALID", output "INVALID" and stop
    if d == "INVALID":
        return "INVALID"

    # 3. (Gamma, c, s) = D
    gamma, c, s = d
    print("c", c)
    print("s", s)
    # gamma : ec point
    # c : 128 bits, little endian
    # s : 253 bits, little endian
    # 4. H = ECVRF_hash_to_curve(suite_string, y, alpha_string)
    h = _ecvrf_hash_to_curve_elligator2_25519(SUITE_STRING, y, alpha_string)
    print('H DECODE POINT', _decode_point(h))
    # 5. U = s*B - c*y
    s_b = _scalar_multiply(p=BASE, e=s)
    y_point = _decode_point(y)
    c_y = _scalar_multiply(p=y_point, e=c)
    nc_y = [PRIME - c_y[0], c_y[1]]

    u = _edwards_add(s_b, nc_y)

    # 6. V = s*H - c*Gamma
    s_h = _scalar_multiply(p=_decode_point(h), e=s)
    c_g = _scalar_multiply(p=gamma, e=c)
    nc_g = [PRIME - c_g[0], c_g[1]]
    v = _edwards_add(nc_g, s_h)

    # 7. c’ = ECVRF_hash_points(H, Gamma, U, V)
    cp = _ecvrf_hash_points(_decode_point(h), gamma, u, v)

    # 8. If c and c’ are equal, output ("VALID", ECVRF_proof_to_hash(pi_string)); else output "INVALID"
    if c != cp:
        return "INVALID"

    else:
        return "VALID"  # , ecvrf_proof_to_hash(pi_string)


def prepare_input_for_ecvrf_hash_to_curve_elligator2_25519(suite_string: bytes, y: bytes, alpha_string: bytes) -> bytes:
    # 1 byte, 1 byte, 32 bytes, 32 bytes
    res = suite_string + b'\x01' + y + alpha_string
    print("YYYY", y)
    res = _hash(alpha_string) + y + b'\x01' + suite_string
    return res


def _ecvrf_hash_to_curve_elligator2_25519(suite_string, y, alpha_string):
    """
    Input:
        suite_string - a single octet specifying ECVRF ciphersuite.
        alpha_string - value to be hashed, an octet string
        y - public key, an EC point
    Output:
        H - hashed value, a finite EC point in G
    Fixed options:
        p = 2^255-19, the size of the finite field F, a prime, for edwards25519 and curve25519 curves
        A = 486662, Montgomery curve constant for curve25519
        cofactor = 8, the cofactor for edwards25519 and curve25519 curves
    """
    assert suite_string == SUITE_STRING

    to_hash = prepare_input_for_ecvrf_hash_to_curve_elligator2_25519(
        suite_string, y, alpha_string)
    print(f"To hash : {to_hash}")
    # 3. hash_string = Hash(suite_string || one_string || PK_string || alpha_string )
    hash_string = _hash(to_hash)
    print(f"hashed : {hash_string}")
    print(f"hashed : {int.from_bytes(hash_string, 'little')}")
    # 4. truncated_h_string = hash_string[0]...hash_string[31]
    truncated_h_string = bytearray(hash_string[0:32])
    print(f"bytearray : {truncated_h_string}")
    # 5. oneTwentySeven_string = 0x7F = int_to_string(127, 1) (a single octet with value 127)
    one_twenty_seven_string = 0x7f
    print(f" hash[31] : {truncated_h_string[31]}")
    print(
        f"hash[31] & 0b1111111 {truncated_h_string[31] & one_twenty_seven_string}")
    # 6. truncated_h_string[31] = truncated_h_string[31] & oneTwentySeven_string (this step clears the high-order bit of octet 31)
    truncated_h_string[31] = int(
        truncated_h_string[31] & one_twenty_seven_string)
    print(f"truncated hstring={truncated_h_string} \n ")
    # 7. r = string_to_int(truncated_h_string)
    r = int.from_bytes(truncated_h_string, 'little')

    print(f"r={r}")

    # 8. u = - A / (1 + 2*(r^2) ) mod p (note: the inverse of (1+2*(r^2)) modulo p is guaranteed to exist)
    u = (PRIME - A) * _inverse(1 + 2 * (r ** 2)) % PRIME
    print(f"u={u}")
    # 9. w = u * (u^2 + A*u + 1) mod p (this step evaluates the Montgomery equation for Curve25519)
    w = u * (u ** 2 + A * u + 1) % PRIME
    print(f"w={w}")
    # 10. Let e equal the Legendre symbol of w and p (see note below on how to compute e)
    e = pow(w, (PRIME - 1) // 2, PRIME)
    print(f"e={e}")

    # 11. If e is equal to 1 then final_u = u; else final_u = (-A - u) mod p
    #     (note: final_u is the Montgomery u-coordinate of the output; see  note below on how to compute it)
    final_u = (e * u + (e - 1) * A * TWO_INV) % PRIME
    print(f"final_u={final_u}")
    # 12. y_coordinate = (final_u - 1) / (final_u + 1) mod p
    #     (note 1: y_coordinate is the Edwards coordinate corresponding to final_u)
    #     (note 2: the inverse of (final_u + 1) modulo p is guaranteed to exist)
    y_coordinate = (final_u - 1) * _inverse(final_u + 1) % PRIME
    print(f"y_coordinate={y_coordinate}")
    # 13. h_string = int_to_string (y_coordinate, 32)
    h_string = int.to_bytes(y_coordinate, 32, 'little')
    print(f"h_string={h_string}")
    # 14. H_prelim = string_to_point(h_string) (note: string_to_point will not return INVALID by correctness of Elligator2)
    h_prelim = _decode_point(h_string)
    print(f"h_prelim={h_prelim}")

    # 15. Set H = cofactor * H_prelim
    h = _scalar_multiply(p=h_prelim, e=8)
    print(f"h={h}")

    # 16. Output H
    h_point = _encode_point(h)
    print(f"h_point={h_point}")

    return h_point


# test_generator()

def print_ng(p=BASE, name='nG', n=253):
    file = open('./client/dw/precomputed_'+name+'.txt', 'w')

    ng = p
    pp = to_extended(p)
    x = split_128(pp[0])
    y = split_128(pp[1])
    z = split_128(pp[2])
    t = split_128(pp[3])
    file.write('    ' + 'dw '+str(x[0])+';\n')
    file.write('    ' + 'dw '+str(x[1])+';\n')
    file.write('    ' + 'dw '+str(y[0])+';\n')
    file.write('    ' + 'dw '+str(y[1])+';\n')
    file.write('    ' + 'dw '+str(z[0])+';\n')
    file.write('    ' + 'dw '+str(z[1])+';\n')
    file.write('    ' + 'dw '+str(t[0])+';\n')
    file.write('    ' + 'dw '+str(t[1])+';\n')
    file.write('\n')
    for _ in range(0, n):
        ng = _edwards_add(ng, ng)
        pp = to_extended(ng)
        x = split_128(pp[0])
        y = split_128(pp[1])
        z = split_128(pp[2])
        t = split_128(pp[3])
        file.write('    ' + 'dw '+str(x[0])+';\n')
        file.write('    ' + 'dw '+str(x[1])+';\n')
        file.write('    ' + 'dw '+str(y[0])+';\n')
        file.write('    ' + 'dw '+str(y[1])+';\n')
        # file.write('    ' + 'dw '+str(z[0])+';\n')
        # file.write('    ' + 'dw '+str(z[1])+';\n')
        file.write('    ' + 'dw '+str(t[0])+';\n')
        file.write('    ' + 'dw '+str(t[1])+';\n')
        file.write('\n')


def print_ng_wei(p=BASE, name='nG', n=253):
    file = open('./client/dw/precomputed_'+name+'_wei.txt', 'w')

    ng = p
    pp = [to_bigint(x) for x in to_weierstrass(to_montgomery(p))]

    file.write('    ' + 'dw '+str(pp[0][0])+';\n')
    file.write('    ' + 'dw '+str(pp[0][1])+';\n')
    file.write('    ' + 'dw '+str(pp[0][2])+';\n')
    file.write('    ' + 'dw '+str(pp[1][0])+';\n')
    file.write('    ' + 'dw '+str(pp[1][1])+';\n')
    file.write('    ' + 'dw '+str(pp[1][2])+';\n')
    file.write('\n')
    for _ in range(0, n):
        ng = _edwards_add(ng, ng)
        pp = [to_bigint(x) for x in to_weierstrass(to_montgomery(ng))]

        file.write('    ' + 'dw '+str(pp[0][0])+';\n')
        file.write('    ' + 'dw '+str(pp[0][1])+';\n')
        file.write('    ' + 'dw '+str(pp[0][2])+';\n')
        file.write('    ' + 'dw '+str(pp[1][0])+';\n')
        file.write('    ' + 'dw '+str(pp[1][1])+';\n')
        file.write('    ' + 'dw '+str(pp[1][2])+';\n')
        file.write('\n')


# print_ng(BASE, 'nG', 252)
# print_ng(Y, 'nY', 128)
# print_ng_wei(BASE, 'nG', 253)
# print_ng_wei(Y, 'nY', 129)


# p = _scalar_multiply(p=BASE, e=10)

# _ecvrf_hash_to_curve_elligator2_25519(
#     SUITE_STRING, _encode_point(Y), alpha_string)


#######################################################
alpha_string = b'abcdefghijklmnopqrstuvwxyzabcdef'
pi_string = ecvrf_prove(
    0x76ac232256b548d60d1273bac6a984f1d9da7fecf428092e711808980b118650.to_bytes(32, "big"), alpha_string)


Y = (31677364464174491700709790792734682609129727062794799651632892241694786563612,
     57739933451475567400550753339703382323534342777832397104845965682571402726947)
ecvrf_verify(y=_encode_point(Y), pi_string=pi_string,
             alpha_string=alpha_string)

#######################################################


def pi_string_to_input_call(pi_string, alpha_string: bytes):
    if type(alpha_string) == bytes:
        (a0, a1) = split_128(int.from_bytes(alpha_string, 'little'))
    if type(alpha_string) == int:
        (a0, a1) = split_128(alpha_string)

    gamma, c, s = _ecvrf_decode_proof(pi_string)
    (gx0, gx1) = split_128(gamma[0])
    (gy0, gy1) = split_128(gamma[1])
    (s0, s1) = split_128(s)
    res = {'alpha_string_low': a0, 'alpha_string_high': a1,
           'gamma_string_x_low': gx0, 'gamma_string_x_high': gx1,
           'gamma_string_y_low': gy0, 'gamma_string_y_high': gy1,
           'c_string_little': c, 's_string_little_low': s0, 's_string_little_high': s1}
    return res


pi_string_to_input_call(pi_string, alpha_string)
