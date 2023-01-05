from starkware.cairo.common.uint256 import Uint256
from starkware.cairo.common.cairo_secp.bigint import BigInt3

struct ExtendedPoint {
    x: Uint256,
    y: Uint256,
    z: Uint256,
    t: Uint256,
}

struct AffinePoint {
    x: Uint256,
    y: Uint256,
}

// Represents a point on a Weierstrass elliptic curve.
// The zero point is represented as a point with x = 0 (there is no point on the curve with a zero
// x value).
struct EcPoint {
    x: BigInt3,
    y: BigInt3,
}
const P = 2 ** 255 - 19;
const P_low = 340282366920938463463374607431768211437;
const P_high = 170141183460469231731687303715884105727;

const A = 486662;
const TWO_INV_low = 340282366920938463463374607431768211447;
const TWO_INV_high = 85070591730234615865843651857942052863;
const P_minus_A_low = 340282366920938463463374607431767724775;
const P_minus_A_high = P_high;

const P_min_1_div_2_low = 340282366920938463463374607431768211446;
const P_min_1_div_2_high = 85070591730234615865843651857942052863;

const minus_A_low = 340282366920938463463374607431767724775;
const minus_A_high = 170141183460469231731687303715884105727;

const D_low = 581746193016579820166537227703777443;
const D_high = 109014481914921826637217330734730700952;
const minus_D_low = 339700620727921883643208070204064433994;
const minus_D_high = 61126701545547405094469972981153404775;
const II_low = 62822086998211147343238952298832896176;
const II_high = 57837735039853669993003572062525839271;

// q = 2 ** 252 + 27742317777372353535851937790883648493;
// q = ORDER
const q_low = 27742317777372353535851937790883648493;
const q_high = 21267647932558653966460912964485513216;

const Gx_low = 139801444470765215774282931664016758042;
const Gx_high = 44410827061887073916105350349541792860;
const Gy_low = 136112946768375385385349842972707284568;
const Gy_high = 136112946768375385385349842972707284582;

const Y = 216777731952371746420703551196102372892;
const Yy = 57739933451475567400550753339703382323534342777832397104845965682571402726947;

// ///////////////////////////////////////////////////////////////////////////

// >>> A = split_128((-2 + 2 * D) * _inverse(-1 - D) % PRIME) = 486662
const A_low = 486662;
const A_high = 0;

// >>> B = split_128(4 * _inverse(-1 - D) % PRIME)
const B_low = 340282366920938463463374607431767724773;
const B_high = 170141183460469231731687303715884105727;
// Sc

// twisted : a x² + y² = 1 + D x² y²
// twisted ed25519,  a = -1

// Montgomery : B v² = u³ + A u² + u
// convert x,y from Twisted Edwards curve to u,v Montgomery coordinates
// B = 4 / (-1-D)
