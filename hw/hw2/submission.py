#!/usr/bin/env python

import solve


cases = (
    dict(probToState=0.29,valueEstimates=(0.0,0.3,21.6,0.0,6.1,16.1,11.8),rewards=(-2.1,6.4,2.5,5.9,9.3,4.3,-2.3)),
    dict(probToState=0.0,valueEstimates=(0.0,1.5,0.0,23.7,19.8,21.2,0.7),rewards=(8.6,0.0,4.1,-3.0,1.5,-2.0,4.1)),
    dict(probToState=0.0,valueEstimates=(0.0,1.3,0.0,14.3,-2.3,10.7,4.0),rewards=(8.0,8.8,0.7,0.0,-3.9,4.8,2.4)),
    dict(probToState=0.79,valueEstimates=(0.0,0.0,3.8,25.0,0.0,20.5,16.9),rewards=(6.5,3.1,-0.6,1.6,0.0,9.3,-1.0)),
    dict(probToState=0.2,valueEstimates=(0.0,14.7,0.0,3.4,16.7,-4.7,0.0),rewards=(9.2,9.0,4.2,4.9,-3.9,6.0,0.0))
)


for case in cases:
    print case
    print solve.solve(case['probToState'], case['valueEstimates'], case['rewards'], True)
