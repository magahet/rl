import nose.tools as nt
import vi


def test_vi():
    
    def trunc(v):
        s = str(v)
        return s[:min(6, len(s))]
        
    N=4
    B=(0,1,1,1)
    expectedValue = '0.25'
    answer = vi.get_value(N, B)
    nt.eq_(trunc(answer), expectedValue)
    print

    N=8
    B=(1,0,1,0,1,1,1,0)
    expectedValue = '1.8125'
    answer = vi.get_value(N, B)
    nt.eq_(trunc(answer), expectedValue)
    print

    N=6
    B=(1,0,1,1,0,1)
    expectedValue = '1.1666'
    answer = vi.get_value(N, B)
    nt.eq_(trunc(answer), expectedValue)
    print
    
    N=10
    B=(1,0,1,1,1,1,1,0,1,0)
    expectedValue='2.06'
    answer = vi.get_value(N, B)
    nt.eq_(trunc(answer), expectedValue)
    print
     
    N=18
    B=(1,0,0,1,0,0,0,1,0,1,0,1,1,1,1,1,0,0)
    expectedValue='4.9308'
    answer = vi.get_value(N, B)
    nt.eq_(trunc(answer), expectedValue)
    print
    
    N=26
    B=(1,0,0,0,0,1,1,0,1,1,0,1,1,0,0,0,1,1,1,1,1,1,0,0,0,1)
    expectedValue='6.4733'
    answer = vi.get_value(N, B)
    nt.eq_(trunc(answer), expectedValue)
