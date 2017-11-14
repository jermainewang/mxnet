import mxnet as mx
import mxnet.ndarray as nd
from mxnet import gluon
from mxnet.gluon import nn, Block, HybridBlock
from mxnet import autograd

import time

ctx = mx.cpu(0)

def sigmoid(F, x):
    return 1 / (1 + -F.exp(x))

class LSTMCell(HybridBlock):
    def __init__(self, hidden_size, input_size, prefix=None, params=None):
        super(LSTMCell, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self._hidden_size = hidden_size
            self._input_size = input_size
            self.Wxi = self.params.get('Wxi', shape=(input_size, hidden_size), init='zeros')
            self.Wxf = self.params.get('Wxf', shape=(input_size, hidden_size), init='zeros')
            self.Wxo = self.params.get('Wxo', shape=(input_size, hidden_size), init='zeros')
            self.Wxg = self.params.get('Wxg', shape=(input_size, hidden_size), init='zeros')
            self.Whi = self.params.get('Whi', shape=(hidden_size, hidden_size), init='zeros')
            self.Whf = self.params.get('Whf', shape=(hidden_size, hidden_size), init='zeros')
            self.Who = self.params.get('Who', shape=(hidden_size, hidden_size), init='zeros')
            self.Whg = self.params.get('Whg', shape=(hidden_size, hidden_size), init='zeros')
            #self.bi = self.params.get('bi', shape=(hidden_size,), init='zeros')
            #self.bf = self.params.get('bf', shape=(hidden_size,), init='zeros')
            #self.bo = self.params.get('bo', shape=(hidden_size,), init='zeros')
            #self.bg = self.params.get('bg', shape=(hidden_size,), init='zeros')

    def hybrid_forward(self, F, X, states,
                       Wxi, Wxf, Wxo, Wxg,
                       Whi, Whf, Who, Whg):
                       #bi, bf, bo, bg):
        h, c = states
        i = F.sigmoid(F.dot(X, Wxi) + F.dot(h, Whi))
        f = F.sigmoid(F.dot(X, Wxf) + F.dot(h, Whf))
        o = F.sigmoid(F.dot(X, Wxo) + F.dot(h, Who))
        g = F.tanh(F.dot(X, Wxg) + F.dot(h, Whg))
        c = f * c + i * g
        h = o * F.tanh(c)
        return h, [h, c]

class LSTM(Block):
    def __init__(self, hidden_size, input_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        with self.name_scope():
            self.cell = LSTMCell(hidden_size, input_size)

    def forward(self, data):
        N, _ = data[0].shape
        h = nd.zeros((N, self.hidden_size), ctx=ctx)
        c = nd.zeros((N, self.hidden_size), ctx=ctx)
        w = nd.zeros((self.input_size, self.hidden_size), ctx=ctx)
        states = [h, c]
        outputs = []
        for t in range(len(data)):
            out, states = self.cell(data[t], states)
            outputs.append(out)
        return outputs, states

class CrossEntropy(HybridBlock):
    def __init__(self):
        super(CrossEntropy, self).__init__()
    def hybrid_forward(self, F, yhat, y):
        return -F.mean(F.sum(y * F.log(yhat), axis=0, exclude=True))


class TrashLossLSTM(Block):
    def __init__(self, hidden_size, input_size):
        super(TrashLossLSTM, self).__init__()
        self.lstm = LSTM(hidden_size, input_size)
        self.cross_entropy = CrossEntropy()

    def forward(self, data):
        outputs, _ = self.lstm(data)
        total_loss = 0.
        label = nd.zeros(outputs[0].shape, ctx=ctx)
        for output in outputs:
            total_loss = total_loss + self.cross_entropy(output, label)
        return total_loss / len(outputs)

def test_lstm():
    N = 1
    hidden_size = 128
    input_size = 8
    length = 100
    data = [nd.zeros((N, input_size), ctx=ctx) for i in range(length)]
    lstm = TrashLossLSTM(hidden_size, input_size)
    lstm.hybridize()
    lstm.collect_params().initialize(ctx=ctx)
    for i in range(20):
        t0 = time.time()
        #loss = lstm(data)
        with autograd.record():
            loss = lstm(data)
            loss.backward()
        loss.wait_to_read()
        print('Iter #%d, takes %fs' % (i, (time.time() - t0)))

if __name__ == '__main__':
    test_lstm()
