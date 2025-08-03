

import numpy as np

def sqm2rai_fmt( sqm, NDIGIT=1 ):
    def truncate_decimals(x, decimals=1):
        factor = 10 ** decimals
        return np.floor(x * factor) / factor
    ''' NDIGTI after comma for "wa" !!! '''
    assert NDIGIT>=0
    up = 4*5/np.power(10,NDIGIT+1)    # 
    _sqm = sqm + 5/np.power(10,NDIGIT+1)    # 0 : 0.5 , 1 : 0.05 ,   2 : 0:005 

    rai, _rem1 = divmod( _sqm, 1600 )
    ngan, _rem2 =  divmod( _rem1, 400 )
    wah = truncate_decimals( _rem2/4, NDIGIT  )
    fmt = f'{{:.0f}}-{{:.0f}}-{{:.{NDIGIT}f}}'
    area_tri = (rai,ngan,wah)
    area_fmt = fmt.format( rai, ngan, wah )
    area_rai = sqm/1600. 
    #import pdb ; pdb.set_trace()
    return area_tri, area_fmt, area_rai


#####################################################################
for sqm_at in [100, 200, 400, 1600, 2000 ]:
    print( f'================== {sqm_at} ================')
    for sqm in np.arange( sqm_at-0.2 ,sqm_at+0.2, 0.01):    # sqm
            hms , fmt,rai = sqm2rai_fmt( sqm ,1)
            print( f'{sqm:.7f} sqm =====> {fmt}  , {rai:.8f} {hms} ')
