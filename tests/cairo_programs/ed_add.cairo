%builtins output range_check bitwise
from lib.curve import P_low, P_high
from lib.ed25519 import ExtendedPoint, TwistedArithmetics
from lib.u255 import u255
from starkware.cairo.common.cairo_builtins import BitwiseBuiltin
from starkware.cairo.common.uint256 import Uint256

func main{output_ptr: felt*, range_check_ptr, bitwise_ptr: BitwiseBuiltin*}() {
    alloc_locals;
    let P = Uint256(P_low, P_high);
    let a = ExtendedPoint(
        x=Uint256(43192411618168306500369449636968528207, 127836812826537975251982703603520145015),
        y=Uint256(49797504309839299353147461194798889772, 132258117957538326582349899190979051471),
        z=Uint256(1, 0),
        t=Uint256(155466007355907588088263444829604398654, 72943317090724618555666500857162389375),
    );  // [to_uint(x) for x in to_extended([43500613248243327786121022071801015118933854441360174117148262713429272820047,45005105423099817237495816771148012388779685712352441364231470781391834741548 ])]
    let b = ExtendedPoint(
        x=Uint256(140259942142931294965626752625963497903, 137224085978424599938040703774585469598),
        y=Uint256(253144262263300720435102317091852409400, 25820880030508234660306446697074341360),
        z=Uint256(1, 0),
        t=Uint256(253879765200236411362134985095524777690, 26353387897677070768219998042781955100),
    );  // [to_uint(x) for x in to_extended((46694936775300686710656303283485882876784402425210400817529601134760286812591,8786390172762935853260670851718824721296437982862763585171334833968259029560))]
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
    let res: ExtendedPoint = TwistedArithmetics.add(a, b);
    // let (l, h) = u256.mul2b(P, P);
    // %{ print_u_512_info_u(ids.l, ids.h, '2PP') %}

    assert res.x.low = 91297808838743694984660001710711986787;
    assert res.x.high = 29711444290802994633705050679624375622;
    assert res.y.low = 104021219304868236826575167050365978179;
    assert res.y.high = 120910031807270091425397192357442325518;

    %{ print(ids.res.x.low, ids.res.x.high, ids.res.y.low, ids.res.y.high) %}
    return ();
}
