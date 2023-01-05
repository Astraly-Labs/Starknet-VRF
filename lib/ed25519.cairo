from lib.u255 import u255, u512, Uint768
from lib.f25519 import f25519
from starkware.cairo.common.uint256 import Uint256
from starkware.cairo.common.cairo_builtins import BitwiseBuiltin
from lib.nGnY_Weierstrass import get_2_pow_n_times_G_We, get_2_pow_n_times_Y_We
from starkware.cairo.common.alloc import alloc
from starkware.cairo.common.math import unsigned_div_rem as felt_divmod
from starkware.cairo.common.cairo_keccak.keccak import keccak, finalize_keccak, keccak_add_uint256s
from starkware.cairo.common.cairo_secp.bigint import (
    BigInt3,
    UnreducedBigInt3,
    nondet_bigint3,
    bigint_to_uint256,
    uint256_to_bigint,
)
from lib.utils import clear_high_order_bit_128
from lib.curve import ExtendedPoint, AffinePoint, EcPoint
from lib.nGnY import get_2_pow_n_times_G, get_2_pow_n_times_Y

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
    B_low,
    B_high,
)

const BASE = 2 ** 86;
const SECP_REM = 19;

namespace bijections {
    func to_montgomery{range_check_ptr}(p: AffinePoint) -> AffinePoint {
        alloc_locals;
        let x = p.x;
        let y = p.y;

        let one_plus_y = u255.add(Uint256(1, 0), y);
        let one_min_y = u255.super_sub(Uint256(1, 0), y);
        let u = f25519.div(one_plus_y, one_min_y);

        let one_min_y_times_x = f25519.mul(one_min_y, x);
        let v = f25519.div(one_plus_y, one_min_y_times_x);
        // let scaled_v = f25519.mul(v, Uint256(scaling_factor_low, scaling_factor_high));

        let res = AffinePoint(u, v);
        return res;
    }

    // Weierstrass : v² = t³ + a t + b
    // convert x,y from Montgomery to (t,v) in Weierstrass form
    func to_weierstrass{range_check_ptr}(p: AffinePoint) -> AffinePoint {
        alloc_locals;

        let x = p.x;
        let y = p.y;
        let t = f25519.mul(
            x,
            Uint256(340136930372684318508332973124842267076, 142887562981738775072382971032201430489),
        );  // split_128(_inverse(B))
        let t = f25519.add(
            t,
            Uint256(226951868979461738945610827825796103865, 131596535959466458927327757599711187310),
        );  // >>> split_128(A*_inverse(3)%PRIME) split_128(A*_inverse(3*B)%PRIME)
        let v = f25519.mul(
            y,
            Uint256(340136930372684318508332973124842267076, 142887562981738775072382971032201430489),
        );  // split_128(_inverse(B))
        let res = AffinePoint(t, v);
        return res;
    }

    const alpha_low = 113330497941476724517763779605972107572;
    const alpha_high = 38544647501002772804359546116172918417;
    // alpha = sqrt((_inverse(B)**2%PRIME-a_raw)%PRIME*_inverse(3), PRIME)
    const m_alpha_low = 226951868979461738945610827825796103865;
    const m_alpha_high = 131596535959466458927327757599711187310;

    func from_weierstrass{range_check_ptr}(p: AffinePoint) -> AffinePoint {
        alloc_locals;

        let t = p.x;
        let v = p.y;
        let v = f25519.mul(Uint256(B_low, B_high), v);
        let x = u255.super_sub(t, Uint256(m_alpha_low, m_alpha_high));
        let x = f25519.mul(Uint256(B_low, B_high), x);
        let res = AffinePoint(x, v);
        return res;
    }

    func from_montgomery{range_check_ptr}(p: AffinePoint) -> AffinePoint {
        alloc_locals;

        let u = p.x;
        let v = p.y;
        let x = f25519.div(u, v);
        let u_minus_one = u255.super_sub(u, Uint256(1, 0));
        let u_plus_one = u255.add(u, Uint256(1, 0));
        let y = f25519.div(u_minus_one, u_plus_one);
        let res = AffinePoint(x, y);
        return res;
    }

    func affine_to_ec_point{range_check_ptr}(p: AffinePoint) -> EcPoint {
        alloc_locals;
        let (x_Bigint) = uint256_to_bigint(p.x);
        let (y_Bigint) = uint256_to_bigint(p.y);
        let res = EcPoint(x_Bigint, y_Bigint);
        return res;
    }
    func ec_point_to_affine{range_check_ptr}(p: EcPoint) -> AffinePoint {
        alloc_locals;
        let (x_256) = bigint_to_uint256(p.x);
        let (y_256) = bigint_to_uint256(p.y);
        let res = AffinePoint(x_256, y_256);
        return res;
    }
}

namespace TwistedArithmetics {
    // func scalar_multiply{range_check_ptr, bitwise_ptr: BitwiseBuiltin*}(
    //     n: felt, P: ExtendedPoint
    // ) -> ExtendedPoint {
    //     if (n == 0) {
    //         tempvar res = ExtendedPoint(Uint256(0, 0), Uint256(1, 0), Uint256(0, 0), Uint256(0, 0));
    //         return res;
    //     }
    //     if (n == 1) {
    //         return P;
    //     }
    //     let (quotient: Uint256, n_mod_2: felt) = u255.modulo_2(n);

    // if (n_mod_2 == 1) {
    //         let to_add = scalar_multiply(n - 1, P);
    //         return add(P, to_add);
    //     } else {
    //         let pdouble = scalar_multiply(quotient, P);
    //         return pdouble;
    //     }

    // let q = scalar_multiply(n_div_by_2, P);
    //     let q = double(q, q);

    // let G_pow_n = get_2_pow_n_times_G(n);
    //     return q;
    // }

    func _scalar_multiply_G_by_s_loop{range_check_ptr, bitwise_ptr: BitwiseBuiltin*}(
        index: felt, res: ExtendedPoint, s: Uint256
    ) -> ExtendedPoint {
        alloc_locals;
        %{ print("index:",ids.index) %}
        if (index == 253) {
            return res;
        }
        let eq = u255.eq(s, Uint256(0, 0));
        if (eq == 1) {
            %{ print('quotient is zero, returning res') %}
            return res;
        }
        let (quotient: Uint256, s_mod_2: felt) = u255.modulo_2(s);

        // s impair, dernier bit 1.
        if (s_mod_2 == 1) {
            %{ print(f"get 2^{ids.index} * G") %}
            let _2_pow_n_times_G = get_2_pow_n_times_G(index);
            %{ print_u_256_info(ids._2_pow_n_times_G.x, "x") %}
            %{ print_u_256_info(ids._2_pow_n_times_G.y, "y") %}
            %{ print_u_256_info(ids._2_pow_n_times_G.z, "z") %}
            %{ print_u_256_info(ids._2_pow_n_times_G.t, "t") %}

            %{ print_from_extended(ids._2_pow_n_times_G) %}
            %{ print_from_extended(ids.res) %}

            let res = add_b_z_1(res, _2_pow_n_times_G);
            %{ print_u_256_info(ids.res.x, "resx") %}
            %{ print_u_256_info(ids.res.y, "resy") %}
            %{ print_u_256_info(ids.res.z, "resz") %}
            %{ print_u_256_info(ids.res.t, "rest") %}
            %{ print_from_extended(ids.res) %}
            return _scalar_multiply_G_by_s_loop(index + 1, res, quotient);
        } else {
            return _scalar_multiply_G_by_s_loop(index + 1, res, quotient);
        }
    }

    func _scalar_multiply_Y_by_c_loop{range_check_ptr, bitwise_ptr: BitwiseBuiltin*}(
        index: felt, res: ExtendedPoint, c: felt
    ) -> ExtendedPoint {
        alloc_locals;
        %{ print("index:",ids.index) %}
        if (index == 128) {
            return res;
        }
        // let eq = u255.eq(c, Uint256(0, 0));
        if (c == 0) {
            %{ print('quotient is zero, returning res') %}
            return res;
        }
        let (quotient: felt, c_mod_2: felt) = felt_divmod(c, 2);

        // c impair, dernier bit 1.
        if (c_mod_2 == 1) {
            %{ print(f"get 2^{ids.index} * Y") %}
            let _2_pow_n_times_G = get_2_pow_n_times_Y(index);
            // %{ print_u_256_info(ids._2_pow_n_times_G.x, "x") %}
            // %{ print_u_256_info(ids._2_pow_n_times_G.y, "y") %}
            // %{ print_u_256_info(ids._2_pow_n_times_G.z, "z") %}
            // %{ print_u_256_info(ids._2_pow_n_times_G.t, "t") %}

            // %{ print_from_extended(ids._2_pow_n_times_G) %}
            // %{ print_from_extended(ids.res) %}

            let res = add(res, _2_pow_n_times_G);
            // %{ print_u_256_info(ids.res.x, "resx") %}
            // %{ print_u_256_info(ids.res.y, "resy") %}
            // %{ print_u_256_info(ids.res.z, "resz") %}
            // %{ print_u_256_info(ids.res.t, "rest") %}
            // %{ print_from_extended(ids.res) %}
            return _scalar_multiply_Y_by_c_loop(index + 1, res, quotient);
        } else {
            return _scalar_multiply_Y_by_c_loop(index + 1, res, quotient);
        }
    }

    func double{range_check_ptr, bitwise_ptr: BitwiseBuiltin*}(a: ExtendedPoint) -> ExtendedPoint {
        alloc_locals;
        let X1 = a.x;
        let Y1 = a.y;
        let Z1 = a.z;

        let A: Uint256 = f25519.square(X1);
        let B: Uint256 = f25519.square(Y1);
        let C: Uint256 = f25519.square2(Z1);
        let mA = u255.neg(A);
        let D: Uint256 = u255.a_modulo_2_255_19(mA);

        let x1y1 = u255.add(X1, Y1);
        let x1y1_sq = f25519.square(x1y1);
        let E = f25519.sub(x1y1_sq, A);

        let E = f25519.sub(E, B);
        let G = u255.add(D, B);
        let F = f25519.sub(G, C);
        let H = f25519.sub(D, B);

        let X3 = f25519.mul(E, F);
        let Y3 = f25519.mul(G, H);
        let T3 = f25519.mul(E, H);
        let Z3 = f25519.mul(F, G);
        let res = ExtendedPoint(X3, Y3, Z3, T3);
        return res;
    }
    func add{range_check_ptr, bitwise_ptr: BitwiseBuiltin*}(
        a: ExtendedPoint, b: ExtendedPoint
    ) -> ExtendedPoint {
        alloc_locals;
        // let (le) = uint384_lib.lt(a.x, a.y);
        // assert le = 1;
        let Ya_min_Xa = u255.super_sub(a.y, a.x);
        let Yb_min_Xb = u255.super_sub(b.y, b.x);

        let Yb_plus_Xb = u255.add(b.y, b.x);
        let Ya_plus_Xa = u255.add(a.y, a.x);
        // %{ print_u_256_info(ids.Ya_min_Xa, 'Ya_min_xa') %}
        // %{ print_u_256_info(ids.Yb_plus_Xb, "yb+xb") %}
        // %{ print(f"yb-xb={ids.Yb_min_Xb.low + (ids.Yb_min_Xb.high<< 128)}") %}

        let A = f25519.mul(Ya_min_Xa, Yb_plus_Xb);
        let B = f25519.mul(Ya_plus_Xa, Yb_min_Xb);
        let fzero = u255.eq(A, B);
        if (fzero == 1) {
            %{ print('retourning double(a)') %}
            return double(a);
        }
        let F = u255.super_sub(B, A);
        let C = f25519.mul2ab(a.z, b.t);

        %{ print(f"A={ids.A.low + (ids.A.high<< 128)}") %}
        %{ print(f"B={ids.B.low + (ids.B.high<< 128)}") %}
        %{ print(f"F={ids.F.low + (ids.F.high<< 128)}") %}
        // %{ print(f"C={ids.C.low + (ids.C.high<< 128)}") %}

        let D = f25519.mul2ab(a.t, b.z);
        // %{ print(f"D={ids.D.low + (ids.D.high<< 128)}") %}
        let E = u255.add(D, C);
        let G = u255.add(B, A);
        let H = u255.super_sub(D, C);

        // %{ print(f"E={ids.E.low + (ids.E.high<< 128)}") %}
        // %{ print(f"G={ids.G.low + (ids.G.high<< 128)}") %}
        // %{ print(f"H={ids.H.low + (ids.H.high<< 128)}") %}
        // %{ print(f"H={ids.H.low + (ids.H.high<< 128)}") %}

        let X3 = f25519.mul(E, F);
        let Y3 = f25519.mul(G, H);
        let Z3 = f25519.mul(F, G);
        let T3 = f25519.mul(E, H);

        // %{ print(f"X3={ids.X3.low + (ids.X3.high<< 128)}") %}
        // %{ print(f"Y3={ids.Y3.low + (ids.Y3.high<< 128)}") %}
        // %{ print(f"Z3={ids.Z3.low + (ids.Z3.high<< 128)}") %}
        // %{ print(f"T3={ids.T3.low + (ids.T3.high<< 128)}") %}
        let res = ExtendedPoint(X3, Y3, Z3, T3);
        return res;
    }
    func add_b_z_1{range_check_ptr, bitwise_ptr: BitwiseBuiltin*}(
        a: ExtendedPoint, b: ExtendedPoint
    ) -> ExtendedPoint {
        alloc_locals;
        // let (le) = uint384_lib.lt(a.x, a.y);
        // assert le = 1;
        let Ya_min_Xa = u255.super_sub(a.y, a.x);
        let Yb_min_Xb = u255.super_sub(b.y, b.x);

        let Yb_plus_Xb = u255.add(b.y, b.x);
        let Ya_plus_Xa = u255.add(a.y, a.x);
        // %{ print_u_256_info(ids.Ya_min_Xa, 'Ya_min_xa') %}
        // %{ print_u_256_info(ids.Yb_plus_Xb, "yb+xb") %}
        // %{ print(f"yb-xb={ids.Yb_min_Xb.low + (ids.Yb_min_Xb.high<< 128)}") %}

        let A = f25519.mul(Ya_min_Xa, Yb_plus_Xb);
        let B = f25519.mul(Ya_plus_Xa, Yb_min_Xb);
        let fzero = u255.eq(A, B);
        // if (fzero == 1) {
        //     %{ print('retourning double(a)') %}
        //     return double(a);
        // }
        let F = u255.super_sub(B, A);
        let C = f25519.mul2ab(a.z, b.t);

        // let D = f25519.mul2ab(a.t, b.z);
        let D = u255.double_u255(a.t);

        // let D = u255.a_modulo_p_255_19(D); uncomment in case of trouble
        let E = u255.add(D, C);
        let G = u255.add(B, A);
        let H = u255.super_sub(D, C);

        let X3 = f25519.mul(E, F);
        let Y3 = f25519.mul(G, H);
        let Z3 = f25519.mul(F, G);
        let T3 = f25519.mul(E, H);
        let res = ExtendedPoint(X3, Y3, Z3, T3);
        return res;
    }
}

namespace WeierstrassArithmetics {
    // Computes the negation of a point on the elliptic curve, which is a point with the same x value
    // and the negation of the y value. If the point is the zero point, returns the zero point.
    //
    // Arguments:
    //   point - The point to operate on.
    //
    // Returns:
    //   point - The negation of the given point.
    func ec_negate{range_check_ptr}(point: EcPoint) -> (point: EcPoint) {
        alloc_locals;
        %{
            from starkware.cairo.common.cairo_secp.secp_utils import pack
            SECP_P = 2**255-19

            y = pack(ids.point.y, PRIME) % SECP_P
            # The modulo operation in python always returns a nonnegative number.
            value = (-y) % SECP_P
        %}
        let (minus_y) = nondet_bigint3();
        verify_zero(
            UnreducedBigInt3(
            d0=minus_y.d0 + point.y.d0,
            d1=minus_y.d1 + point.y.d1,
            d2=minus_y.d2 + point.y.d2),
        );

        return (point=EcPoint(x=point.x, y=minus_y));
    }

    // Computes the slope of the elliptic curve at a given point.
    // The slope is used to compute point + point.
    //
    // Arguments:
    //   point - the point to operate on.
    //
    // Returns:
    //   slope - the slope of the curve at point, in BigInt3 representation.
    //
    // Assumption: point != 0.
    func compute_doubling_slope{range_check_ptr}(point: EcPoint) -> (slope: BigInt3) {
        alloc_locals;
        // Note that y cannot be zero: assume that it is, then point = -point, so 2 * point = 0, which
        // contradicts the fact that the size of the curve is odd.
        %{
            from starkware.python.math_utils import ec_double_slope
            from starkware.cairo.common.cairo_secp.secp_utils import pack
            SECP_P = 2**255-19

            # Compute the slope.
            x = pack(ids.point.x, PRIME)
            y = pack(ids.point.y, PRIME)
            value = slope = ec_double_slope(point=(x, y), alpha=42204101795669822316448953119945047945709099015225996174933988943478124189485, p=SECP_P)
        %}
        let (slope: BigInt3) = nondet_bigint3();
        // let alpha = Uint256(
        //     143186476941636880901214103594843510573, 124026708105846590725274683684370988502
        // );
        let (x_sqr: UnreducedBigInt3) = unreduced_sqr(point.x);
        let (slope_y: UnreducedBigInt3) = unreduced_mul(slope, point.y);
        let to_assert = UnreducedBigInt3(
            d0=3 * x_sqr.d0 - 2 * slope_y.d0 + 44933163489768861888943917,
            d1=3 * x_sqr.d1 - 2 * slope_y.d1 + 5088459194227531129123890,
            d2=3 * x_sqr.d2 - 2 * slope_y.d2 + 7050102118787810395887998,
        );
        // let to_assert256 = bigint_to_uint256(to_assert);
        // %{ print_u_256_info(ids.to_assert256, 'to_assert') %}

        verify_zero(to_assert);

        return (slope=slope);
    }

    // Computes the slope of the line connecting the two given points.
    // The slope is used to compute point0 + point1.
    //
    // Arguments:
    //   point0, point1 - the points to operate on.
    //
    // Returns:
    //   slope - the slope of the line connecting point0 and point1, in BigInt3 representation.
    //
    // Assumptions:
    // * point0.x != point1.x (mod 25519).
    // * point0, point1 != 0.
    func compute_slope{range_check_ptr}(point0: EcPoint, point1: EcPoint) -> (slope: BigInt3) {
        alloc_locals;
        %{
            from starkware.python.math_utils import line_slope
            from starkware.cairo.common.cairo_secp.secp_utils import pack
            SECP_P = 2**255-19
            # Compute the slope.
            x0 = pack(ids.point0.x, PRIME)
            y0 = pack(ids.point0.y, PRIME)
            x1 = pack(ids.point1.x, PRIME)
            y1 = pack(ids.point1.y, PRIME)
            value = slope = line_slope(point1=(x0, y0), point2=(x1, y1), p=SECP_P)
        %}
        let (slope) = nondet_bigint3();

        let x_diff = BigInt3(
            d0=point0.x.d0 - point1.x.d0, d1=point0.x.d1 - point1.x.d1, d2=point0.x.d2 - point1.x.d2
        );
        let (x_diff_slope: UnreducedBigInt3) = unreduced_mul(x_diff, slope);

        verify_zero(
            UnreducedBigInt3(
            d0=x_diff_slope.d0 - point0.y.d0 + point1.y.d0,
            d1=x_diff_slope.d1 - point0.y.d1 + point1.y.d1,
            d2=x_diff_slope.d2 - point0.y.d2 + point1.y.d2),
        );

        return (slope=slope);
    }

    // Computes the addition of a given point to itself.
    //
    // Arguments:
    //   point - the point to operate on.
    //
    // Returns:
    //   res - a point representing point + point.
    func double{range_check_ptr}(point: EcPoint) -> (res: EcPoint) {
        alloc_locals;
        // The zero point.
        if (point.x.d0 == 0) {
            if (point.x.d1 == 0) {
                if (point.x.d2 == 0) {
                    return (res=point);
                }
            }
        }

        let (slope: BigInt3) = compute_doubling_slope(point);
        let (slope_sqr: UnreducedBigInt3) = unreduced_sqr(slope);

        %{
            from starkware.cairo.common.cairo_secp.secp_utils import pack
            SECP_P = 2**255-19

            slope = pack(ids.slope, PRIME)
            x = pack(ids.point.x, PRIME)
            y = pack(ids.point.y, PRIME)

            value = new_x = (pow(slope, 2, SECP_P) - 2 * x) % SECP_P
        %}
        let (new_x: BigInt3) = nondet_bigint3();

        %{ value = new_y = (slope * (x - new_x) - y) % SECP_P %}
        let (new_y: BigInt3) = nondet_bigint3();

        verify_zero(
            UnreducedBigInt3(
            d0=slope_sqr.d0 - new_x.d0 - 2 * point.x.d0,
            d1=slope_sqr.d1 - new_x.d1 - 2 * point.x.d1,
            d2=slope_sqr.d2 - new_x.d2 - 2 * point.x.d2),
        );

        let (x_diff_slope: UnreducedBigInt3) = unreduced_mul(
            BigInt3(d0=point.x.d0 - new_x.d0, d1=point.x.d1 - new_x.d1, d2=point.x.d2 - new_x.d2),
            slope,
        );

        verify_zero(
            UnreducedBigInt3(
            d0=x_diff_slope.d0 - point.y.d0 - new_y.d0,
            d1=x_diff_slope.d1 - point.y.d1 - new_y.d1,
            d2=x_diff_slope.d2 - point.y.d2 - new_y.d2),
        );

        return (res=EcPoint(new_x, new_y));
    }

    // Computes the addition of two given points.
    //
    // Arguments:
    //   point0, point1 - the points to operate on.
    //
    // Returns:
    //   res - the sum of the two points (point0 + point1).
    //
    // Assumption: point0.x != point1.x (however, point0 = point1 = 0 is allowed).
    // Note that this means that the function cannot be used if point0 = point1 != 0
    // (use ec_double() in this case) or point0 = -point1 != 0 (the result is 0 in this case).
    func fast_ec_add{range_check_ptr}(point0: EcPoint, point1: EcPoint) -> (res: EcPoint) {
        // Check whether point0 is the zero point.
        alloc_locals;
        if (point0.x.d0 == 0) {
            if (point0.x.d1 == 0) {
                if (point0.x.d2 == 0) {
                    return (res=point1);
                }
            }
        }

        // Check whether point1 is the zero point.
        if (point1.x.d0 == 0) {
            if (point1.x.d1 == 0) {
                if (point1.x.d2 == 0) {
                    return (res=point0);
                }
            }
        }

        let (slope: BigInt3) = compute_slope(point0, point1);
        let (slope_sqr: UnreducedBigInt3) = unreduced_sqr(slope);

        %{
            from starkware.cairo.common.cairo_secp.secp_utils import pack
            SECP_P = 2**255-19

            slope = pack(ids.slope, PRIME)
            x0 = pack(ids.point0.x, PRIME)
            x1 = pack(ids.point1.x, PRIME)
            y0 = pack(ids.point0.y, PRIME)

            value = new_x = (pow(slope, 2, SECP_P) - x0 - x1) % SECP_P
        %}
        let (new_x: BigInt3) = nondet_bigint3();

        %{ value = new_y = (slope * (x0 - new_x) - y0) % SECP_P %}
        let (new_y: BigInt3) = nondet_bigint3();

        verify_zero(
            UnreducedBigInt3(
            d0=slope_sqr.d0 - new_x.d0 - point0.x.d0 - point1.x.d0,
            d1=slope_sqr.d1 - new_x.d1 - point0.x.d1 - point1.x.d1,
            d2=slope_sqr.d2 - new_x.d2 - point0.x.d2 - point1.x.d2),
        );

        let (x_diff_slope: UnreducedBigInt3) = unreduced_mul(
            BigInt3(d0=point0.x.d0 - new_x.d0, d1=point0.x.d1 - new_x.d1, d2=point0.x.d2 - new_x.d2),
            slope,
        );

        verify_zero(
            UnreducedBigInt3(
            d0=x_diff_slope.d0 - point0.y.d0 - new_y.d0,
            d1=x_diff_slope.d1 - point0.y.d1 - new_y.d1,
            d2=x_diff_slope.d2 - point0.y.d2 - new_y.d2),
        );

        return (res=EcPoint(new_x, new_y));
    }

    // Same as fast_ec_add, except that the cases point0 = +/-point1 are supported.
    func ec_add{range_check_ptr}(point0: EcPoint, point1: EcPoint) -> (res: EcPoint) {
        let x_diff = BigInt3(
            d0=point0.x.d0 - point1.x.d0, d1=point0.x.d1 - point1.x.d1, d2=point0.x.d2 - point1.x.d2
        );
        let (same_x: felt) = is_zero(x_diff);
        if (same_x == 0) {
            // point0.x != point1.x so we can use fast_ec_add.
            return fast_ec_add(point0, point1);
        }

        // We have point0.x = point1.x. This implies point0.y = +/-point1.y.
        // Check whether point0.y = -point1.y.
        let y_sum = BigInt3(
            d0=point0.y.d0 + point1.y.d0, d1=point0.y.d1 + point1.y.d1, d2=point0.y.d2 + point1.y.d2
        );
        let (opposite_y: felt) = is_zero(y_sum);
        if (opposite_y != 0) {
            // point0.y = -point1.y.
            // Note that the case point0 = point1 = 0 falls into this branch as well.
            let ZERO_POINT = EcPoint(BigInt3(0, 0, 0), BigInt3(0, 0, 0));
            return (res=ZERO_POINT);
        } else {
            // point0.y = point1.y.
            return double(point0);
        }
    }

    // Given a scalar, an integer m in the range [0, 250), and a point on the elliptic curve, point,
    // verifies that 0 <= scalar < 2**m and returns (2**m * point, scalar * point).
    func ec_mul_inner{range_check_ptr}(point: EcPoint, scalar: felt, m: felt) -> (
        pow2: EcPoint, res: EcPoint
    ) {
        if (m == 0) {
            with_attr error_message("Too large scalar") {
                scalar = 0;
            }
            let ZERO_POINT = EcPoint(BigInt3(0, 0, 0), BigInt3(0, 0, 0));
            return (pow2=point, res=ZERO_POINT);
        }

        alloc_locals;
        let (double_point: EcPoint) = double(point);
        %{ memory[ap] = (ids.scalar % PRIME) % 2 %}
        jmp odd if [ap] != 0, ap++;
        return ec_mul_inner(point=double_point, scalar=scalar / 2, m=m - 1);

        odd:
        let (local inner_pow2: EcPoint, inner_res: EcPoint) = ec_mul_inner(
            point=double_point, scalar=(scalar - 1) / 2, m=m - 1
        );
        // Here inner_res = (scalar - 1) / 2 * double_point = (scalar - 1) * point.
        // Assume point != 0 and that inner_res = +/-point. We obtain (scalar - 1) * point = +/-point =>
        // scalar - 1 = +/-1 (mod N) => scalar = 0 or 2 (mod N).
        // By induction, we know that (scalar - 1) / 2 must be in the range [0, 2**(m-1)),
        // so scalar is an odd number in the range [0, 2**m), and we get a contradiction.
        let (res: EcPoint) = fast_ec_add(point0=point, point1=inner_res);
        return (pow2=inner_pow2, res=res);
    }

    func ec_mul_inner_by_G{range_check_ptr}(point: EcPoint, scalar: felt, m: felt, index: felt) -> (
        pow2: EcPoint, res: EcPoint
    ) {
        %{ print('index, m : ', ids.index, ids.m) %}

        if (m == 0) {
            with_attr error_message("Too large scalar") {
                scalar = 0;
            }
            let ZERO_POINT = EcPoint(BigInt3(0, 0, 0), BigInt3(0, 0, 0));
            return (pow2=point, res=ZERO_POINT);
        }

        alloc_locals;
        let double_point: EcPoint = get_2_pow_n_times_G_We(index);
        %{ memory[ap] = (ids.scalar % PRIME) % 2 %}
        jmp odd if [ap] != 0, ap++;
        return ec_mul_inner_by_G(point=double_point, scalar=scalar / 2, m=m - 1, index=index + 1);

        odd:
        %{ print('odd: ') %}

        let (local inner_pow2: EcPoint, inner_res: EcPoint) = ec_mul_inner_by_G(
            point=double_point, scalar=(scalar - 1) / 2, m=m - 1, index=index + 1
        );

        let (res: EcPoint) = fast_ec_add(point0=point, point1=inner_res);
        return (pow2=inner_pow2, res=res);
    }

    func ec_mul_inner_by_Y{range_check_ptr}(point: EcPoint, scalar: felt, m: felt, index: felt) -> (
        pow2: EcPoint, res: EcPoint
    ) {
        %{ print('index, m : ', ids.index, ids.m) %}

        if (m == 0) {
            with_attr error_message("Too large scalar") {
                scalar = 0;
            }
            let ZERO_POINT = EcPoint(BigInt3(0, 0, 0), BigInt3(0, 0, 0));
            return (pow2=point, res=ZERO_POINT);
        }

        alloc_locals;
        let double_point: EcPoint = get_2_pow_n_times_Y_We(index);
        %{ memory[ap] = (ids.scalar % PRIME) % 2 %}
        jmp odd if [ap] != 0, ap++;
        return ec_mul_inner_by_Y(point=double_point, scalar=scalar / 2, m=m - 1, index=index + 1);

        odd:
        %{ print('odd: ') %}

        let (local inner_pow2: EcPoint, inner_res: EcPoint) = ec_mul_inner_by_Y(
            point=double_point, scalar=(scalar - 1) / 2, m=m - 1, index=index + 1
        );

        let (res: EcPoint) = fast_ec_add(point0=point, point1=inner_res);
        return (pow2=inner_pow2, res=res);
    }
    func _ec_mul_inner_by_G{range_check_ptr}(res: EcPoint, scalar: felt, index: felt) -> EcPoint {
        alloc_locals;
        %{ print("index:",ids.index) %}

        if (scalar == 0) {
            %{ print('quotient is zero, returning res') %}
            return res;
        }
        let (quotient: felt, s_mod_2: felt) = felt_divmod(scalar, 2);

        // s impair, dernier bit 1.
        if (s_mod_2 == 1) {
            %{ print(f"get 2^{ids.index} * G") %}
            let _2_pow_n_times_G = get_2_pow_n_times_G_We(index);
            let (res) = ec_add(res, _2_pow_n_times_G);  // fast ec add fails at one

            return _ec_mul_inner_by_G(res, quotient, index + 1);
        } else {
            return _ec_mul_inner_by_G(res, quotient, index + 1);
        }
    }

    // Given a point and a 253-bit scalar, returns scalar * point.
    func scalar_mul{range_check_ptr}(point: EcPoint, scalar: BigInt3) -> EcPoint {
        alloc_locals;
        let (pow2_0: EcPoint, local res0: EcPoint) = ec_mul_inner(point, scalar.d0, 86);
        let (pow2_1: EcPoint, local res1: EcPoint) = ec_mul_inner(pow2_0, scalar.d1, 86);
        let (_, local res2: EcPoint) = ec_mul_inner(pow2_1, scalar.d2, 81);
        let (res: EcPoint) = ec_add(res0, res1);
        let (res: EcPoint) = ec_add(res, res2);
        return res;
    }
    // Given a point and a 128-bit scalar, returns scalar * point.
    func scalar_mul_by_c{range_check_ptr}(point: EcPoint, scalar: BigInt3) -> EcPoint {
        alloc_locals;
        let (pow2_0: EcPoint, local res0: EcPoint) = ec_mul_inner(point, scalar.d0, 86);
        let (pow2_1: EcPoint, local res1: EcPoint) = ec_mul_inner(pow2_0, scalar.d1, 42);
        let (res: EcPoint) = ec_add(res0, res1);
        return res;
    }
    func scalar_mul_Y_by_c{range_check_ptr}(point: EcPoint, scalar: BigInt3) -> EcPoint {
        alloc_locals;
        let (pow2_0: EcPoint, local res0: EcPoint) = ec_mul_inner_by_Y(point, scalar.d0, 86, 1);
        let (pow2_1: EcPoint, local res1: EcPoint) = ec_mul_inner_by_Y(pow2_0, scalar.d1, 42, 87);
        let (res: EcPoint) = ec_add(res0, res1);
        return res;
    }
    func scalar_mul_by_8{range_check_ptr}(point: EcPoint, scalar: BigInt3) -> EcPoint {
        alloc_locals;
        let (pow2_0: EcPoint, local res0: EcPoint) = ec_mul_inner(point, scalar.d0, 4);
        return res0;
    }
    func scalar_mul_s_by_G{range_check_ptr}(point: EcPoint, scalar: BigInt3) -> EcPoint {
        alloc_locals;
        let (pow2_0: EcPoint, local res0: EcPoint) = ec_mul_inner_by_G(point, scalar.d0, 86, 1);
        %{ print('First Done: ') %}

        let (pow2_1: EcPoint, local res1: EcPoint) = ec_mul_inner_by_G(pow2_0, scalar.d1, 86, 87);
        let (_, local res2: EcPoint) = ec_mul_inner_by_G(pow2_1, scalar.d2, 81, 173);
        let (res: EcPoint) = ec_add(res0, res1);
        let (res: EcPoint) = ec_add(res, res2);
        return res;
    }

    func _scalar_mul_s_by_G{range_check_ptr}(point: EcPoint, scalar: BigInt3) -> EcPoint {
        alloc_locals;
        let zero = EcPoint(BigInt3(0, 0, 0), BigInt3(0, 0, 0));
        let res0: EcPoint = _ec_mul_inner_by_G(zero, scalar.d0, 1);
        %{ print('First Done: ') %}

        let res1: EcPoint = _ec_mul_inner_by_G(res0, scalar.d1, 87);
        let res2: EcPoint = _ec_mul_inner_by_G(res1, scalar.d2, 173);
        let (res: EcPoint) = ec_add(res0, res1);
        let (res: EcPoint) = ec_add(res, res2);
        return res;
    }
    func unreduced_mul(a: BigInt3, b: BigInt3) -> (res_low: UnreducedBigInt3) {
        // The result of the product is:
        //   sum_{i, j} a.d_i * b.d_j * BASE**(i + j)
        // Since we are computing it mod secp256k1_prime, we replace the term
        //   a.d_i * b.d_j * BASE**(i + j)
        // where i + j >= 3 with
        //   a.d_i * b.d_j * BASE**(i + j - 3) * 4 * SECP_REM
        // since BASE ** 3 = 4 * SECP_REM (mod secp256k1_prime).
        return (
            UnreducedBigInt3(
            d0=a.d0 * b.d0 + (a.d1 * b.d2 + a.d2 * b.d1) * (8 * SECP_REM),
            d1=a.d0 * b.d1 + a.d1 * b.d0 + (a.d2 * b.d2) * (8 * SECP_REM),
            d2=a.d0 * b.d2 + a.d1 * b.d1 + a.d2 * b.d0),
        );
    }

    // Computes the square of a big integer, given in BigInt3 representation, modulo the
    // secp256k1 prime.
    //
    // Has the same guarantees as in unreduced_mul(a, a).
    func unreduced_sqr(a: BigInt3) -> (res_low: UnreducedBigInt3) {
        tempvar twice_d0 = a.d0 * 2;
        return (
            UnreducedBigInt3(
            d0=a.d0 * a.d0 + (a.d1 * a.d2) * (2 * 8 * SECP_REM),
            d1=twice_d0 * a.d1 + (a.d2 * a.d2) * (8 * SECP_REM),
            d2=twice_d0 * a.d2 + a.d1 * a.d1),
        );
    }

    func verify_zero{range_check_ptr}(val: UnreducedBigInt3) {
        let q = [ap];
        %{
            from starkware.cairo.common.cairo_secp.secp_utils import pack
            SECP_P = 2**255-19
            to_assert = pack(ids.val, PRIME)
            q, r = divmod(pack(ids.val, PRIME), SECP_P)
            assert r == 0, f"verify_zero: Invalid input {ids.val.d0, ids.val.d1, ids.val.d2}."
            ids.q = q % PRIME
        %}
        let q_biased = [ap + 1];
        q_biased = q + 2 ** 127, ap++;
        [range_check_ptr] = q_biased, ap++;
        // This implies that q is in the range [-2**127, 2**127).

        tempvar r1 = (val.d0 + q * SECP_REM) / BASE;
        assert [range_check_ptr + 1] = r1 + 2 ** 127;
        // This implies that r1 is in the range [-2**127, 2**127).
        // Therefore, r1 * BASE is in the range [-2**213, 2**213).
        // By the soundness assumption, val.d0 is in the range (-2**250, 2**250).
        // This implies that r1 * BASE = val.d0 + q * SECP_REM (as integers).

        tempvar r2 = (val.d1 + r1) / BASE;
        assert [range_check_ptr + 2] = r2 + 2 ** 127;
        // Similarly, this implies that r2 * BASE = val.d1 + r1 (as integers).
        // Therefore, r2 * BASE**2 = val.d1 * BASE + r1 * BASE.

        assert val.d2 = q * (BASE / 8) - r2;
        // Similarly, this implies that q * BASE / 4 = val.d2 + r2 (as integers).
        // Therefore,
        //   q * BASE**3 / 4 = val.d2 * BASE**2 + r2 * BASE ** 2 =
        //   val.d2 * BASE**2 + val.d1 * BASE + r1 * BASE =
        //   val.d2 * BASE**2 + val.d1 * BASE + val.d0 + q * SECP_REM =
        //   val + q * SECP_REM.
        // Hence, val = q * (BASE**3 / 4 - SECP_REM) = q * (2**256 - SECP_REM) = q * secp256k1_prime.

        let range_check_ptr = range_check_ptr + 3;
        return ();
    }
    // Returns 1 if x == 0 (mod secp256k1_prime), and 0 otherwise.
    //
    // Completeness assumption: x's limbs are in the range (-BASE, 2*BASE).
    // Soundness assumption: x's limbs are in the range (-2**107.49, 2**107.49).
    func is_zero{range_check_ptr}(x: BigInt3) -> (res: felt) {
        %{
            from starkware.cairo.common.cairo_secp.secp_utils import pack
            SECP_P=2**255-19

            x = pack(ids.x, PRIME) % SECP_P
        %}
        if (nondet %{ x == 0 %} != 0) {
            verify_zero(UnreducedBigInt3(d0=x.d0, d1=x.d1, d2=x.d2));
            return (res=1);
        }

        %{
            SECP_P=2**255-19
            from starkware.python.math_utils import div_mod

            value = x_inv = div_mod(1, x, SECP_P)
        %}
        let (x_inv) = nondet_bigint3();
        let (x_x_inv) = unreduced_mul(x, x_inv);

        // Check that x * x_inv = 1 to verify that x != 0.
        verify_zero(UnreducedBigInt3(
            d0=x_x_inv.d0 - 1,
            d1=x_x_inv.d1,
            d2=x_x_inv.d2));
        return (res=0);
    }
}
