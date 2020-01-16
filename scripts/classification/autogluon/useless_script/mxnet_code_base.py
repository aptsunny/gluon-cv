from mxnet import nd
from mxnet.gluon import nn
from mxnet import autograd
from mxnet.gluon.data.vision import datasets, transforms

from mxnet import nd, gpu, gluon, autograd
from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms
import time

y = nd.random.uniform(-1,1,(2,3))
y = nd.full((2,3), 2.0)
x = nd.random.uniform(shape=(4,1,28,28))
print(y,x)

layer = nn.Dense(3)
print(layer)
layer.initialize()
print(x, layer(x))
# print(layer)


#autograd
x = nd.array([[1, 2], [3, 4]])
x.attach_grad()
print(x)
with autograd.record():
    y = 2 * x * x
y.backward()

print(x.grad)


#
# X = nd.random.uniform(shape=(4,3,28,28))
# transformer = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(0.13, 0.31)])
#
# net = nn.Dense(5)
#
# preds = []
# for x in X:
#     # x = transformer(x).expand_dims(axis=0)
#     pred = net(x).argmax(axis=1)
#     preds.append(pred.astype('int32').asscalar())
#
# print(preds,preds.dtype,preds.shape)

x = nd.ones((3,4), ctx=gpu())
x.copyto(gpu(1))
y = nd.random.uniform(shape=(3,4), ctx=gpu())
print(x + y)

