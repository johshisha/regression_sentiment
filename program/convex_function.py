#coding: utf-8

import math

"""
*****memo*****
----t: 0.500000,a: 3, b: 9-----
y1: 0.3 diff: 0.2 loss: 2.47334956165
y2: 0.8 diff: -0.3 loss: 1.06876475393 l_diff: 1.40458480772
y3: 0.7 diff: -0.2 loss: 0.744958572005 l_diff: 0.323806181925
y4: 0.4 diff: 0.1 loss: 0.501214547363 l_diff: 0.243744024642
y5: 0.6 diff: -0.1 loss: 0.275072375772 l_diff: 0.226142171591

----t: 0.300000,a: 3, b: 9-----
y1: 0.1 diff: 0.2 loss: 3.1442435287
y2: 0.6 diff: -0.3 loss: 2.47565138195 l_diff: 0.668592146746
y3: 0.5 diff: -0.2 loss: 1.53046763088 l_diff: 0.945183751071
y4: 0.2 diff: 0.1 loss: 0.718405553741 l_diff: 0.81206207714
y5: 0.4 diff: -0.1 loss: 0.501214547363 l_diff: 0.217191006378






"""


def convex(x, a=3, b=2):
    return b*math.exp(-a*x**2 / 2)
    

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import sys
    
    
    t = float(sys.argv[1])
    print 't:', t
    for b in range(3,10):
        for a in range(1,6):
            y1 = t - 0.2
            n1 = convex(y1, a=a, b=b)
            l1 = ((t-y1) * n1)**2
            
            y2 = t + 0.3
            n2 = convex(y2, a=a, b=b)
            l2 = ((t-y2) * n2)**2
            
            y3 = t + 0.2
            n3 = convex(y3, a=a, b=b)
            l3 = ((t-y3) * n3)**2
            
            y4 = t - 0.1
            n4 = convex(y4, a=a, b=b)
            l4 = ((t-y4) * n4)**2
            
            y5 = t + 0.1
            n5 = convex(y5, a=a, b=b)
            l5 = ((t-y5) * n5)**2
            
            
            if l1 > l2 and l2 > l3 and l3 > l4 and l4 > l5:
                print '\n----a: %d, b: %d-----'%(a,b)
                print 'y1:', y1, 'diff:', (t-y1), 'loss:', l1
                print 'y2:', y2, 'diff:', (t-y2), 'loss:', l2, 'l_diff:', l1-l2
                print 'y3:', y3, 'diff:', (t-y3), 'loss:', l3, 'l_diff:', l2-l3
                print 'y4:', y4, 'diff:', (t-y4), 'loss:', l4, 'l_diff:', l3-l4
                print 'y5:', y5, 'diff:', (t-y5), 'loss:', l5, 'l_diff:', l4-l5
                    
    
    while(True):     
        print '\nplease input parms'
        print 't a b'
        raw = raw_input()
        if raw == 'exit':
            sys.exit(0)
            
        t, a, b = raw.split(' ')
        t = float(t)
        a = float(a)
        b = float(b)   
        
        x = map(lambda x: x/10.0, range(-10,11))
        y = map(lambda y: convex(y, a=a, b=b), x)
        plt.plot(x, y, label='%d'%a)
        
        y1 = t - 0.2
        n1 = convex(y1, a=a, b=b)
        l1 = ((t-y1) * n1)**2
        
        y2 = t + 0.3
        n2 = convex(y2, a=a, b=b)
        l2 = ((t-y2) * n2)**2
        
        y3 = t + 0.2
        n3 = convex(y3, a=a, b=b)
        l3 = ((t-y3) * n3)**2
        
        y4 = t - 0.1
        n4 = convex(y4, a=a, b=b)
        l4 = ((t-y4) * n4)**2
        
        y5 = t + 0.1
        n5 = convex(y5, a=a, b=b)
        l5 = ((t-y5) * n5)**2
        
        print '\n----t: %f,a: %d, b: %d-----'%(t,a,b)
        print 'y1:', y1, 'diff:', (t-y1), 'loss:', l1
        print 'y2:', y2, 'diff:', (t-y2), 'loss:', l2, 'l_diff:', l1-l2
        print 'y3:', y3, 'diff:', (t-y3), 'loss:', l3, 'l_diff:', l2-l3
        print 'y4:', y4, 'diff:', (t-y4), 'loss:', l4, 'l_diff:', l3-l4
        print 'y5:', y5, 'diff:', (t-y5), 'loss:', l5, 'l_diff:', l4-l5
            
        plt.legend()
        plt.show()


    
    
