import nose.tools as nt
import vi


def test_vi():
    N=4
    B=(0,1,1,1)
    expectedValue = 0.25
    answer = vi.get_value(N, B)
    nt.eq_(answer, expectedValue)

    N=8
    B=(1,0,1,0,1,1,1,0)
    expectedValue = 1.8125
    answer = vi.get_value(N, B)
    nt.eq_(answer, expectedValue)

    N=6
    B=(1,0,1,1,0,1)
    expectedValue = 1.1666
    answer = vi.get_value(N, B)
    nt.eq_(answer, expectedValue)
