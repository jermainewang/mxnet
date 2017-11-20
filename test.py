import mxnet as mx
import mxnet.ndarray as nd
import mxnet.autograd as ag

ctx = mx.cpu(0)

def PP(arr):
    if arr is None:
        print('None')
    else:
        print(arr.asnumpy()[0, 0])

def test():
    a = nd.ones((2, 3), ctx=ctx)  # a is 1
    a_grad = nd.zeros((2, 3), ctx=ctx)
    ag.mark_variables(a, a_grad, 'write')
    d = nd.zeros((2, 3), ctx=ctx) + 100  # d is 100
    d_grad = nd.zeros((2, 3), ctx=ctx)
    ag.mark_variables(d, d_grad, 'write')
    for i in range(10):
        print('>>>>>>iter', i)
        with ag.record():
            b = a * (i + 1)
            if i % 2 == 0:
                b_grad = nd.zeros((2, 3), ctx=ctx)
                ag.mark_variables(b, b_grad)
            t = b * (i + 2)
            if i % 3 == 0:
                t = t * d
            t.backward(retain_graph=True)
            PP(a.grad)
            PP(b.grad)
            PP(d.grad)

def test1():
    a = nd.ones((2, 3), ctx=ctx)  # a is 1
    a_grad = nd.zeros((2, 3), ctx=ctx)
    ag.mark_variables(a, a_grad, 'write')
    with ag.record():
        b = a * 2
        b.attach_grad()
        t = b * 3
        t.backward()
        PP(a.grad)
        PP(b.grad)

def test2():
    a = nd.ones((2, 3), ctx=ctx)  # a is 1
    a.attach_grad()
    d = nd.zeros((2, 3), ctx=ctx) + 100  # d is 100
    d.attach_grad()
    with ag.record():
        b = a * 2
        t = b * d
        t.backward()
        PP(d.grad)
    with ag.record():
        b = a * 2
        t = b * 3
        t.backward()
        PP(d.grad)

#test()
#test1()
test2()
