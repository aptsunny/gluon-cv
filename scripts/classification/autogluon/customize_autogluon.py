import autogluon as ag
import numpy as np
@ag.args(
    lr=ag.space.Real(1e-3, 1e-2, log=True),
    wd=ag.space.Real(1e-3, 1e-2))
def ag_train_fn(args, reporter):
    # epoch -> reporter

    for e in range(10):
        dummy_accuracy = 1 - np.power(1.8, -np.random.uniform(e, 2*e))

        reporter(epoch=e, accuracy=dummy_accuracy, lr=args.lr, wd=args.wd)

# x, y = np.linspace(0, 99, 100), np.linspace(0, 99, 100)
# X, Y = np.meshgrid(x, y)
# Z = np.zeros(X.shape)
#
# @ag.args(
#     x=ag.space.Categorical(*list(range(100))),
#     y=ag.space.Categorical(*list(range(100))),
# )
# def rl_simulation(args, reporter):
#     x, y = args.x, args.y
#     reporter(accuracy=Z[y][x])

myscheduler = ag.scheduler.FIFOScheduler(ag_train_fn,
                                        num_trials=3,
                                        reward_attr='accuracy',
                                        #time_attr='epoch',
                                        #grace_period=1
                                        )

myscheduler.run()
myscheduler.join_tasks()
myscheduler.get_training_curves(plot=True,use_legend=False)
print('The Best Configuration and Accuracy are: {}, {}'.format(myscheduler.get_best_config(),
                                                               myscheduler.get_best_reward()))

# myscheduler = ag.scheduler.RLScheduler(rl_simulation,
#                                         resource={'num_cpus': 4, 'num_gpus': 1},
#                                         num_trials=15,
#                                         reward_attr="accuracy",
#                                         controller_batch_size=4,
#                                         controller_lr=5e-3)

# myscheduler = ag.scheduler.HyperbandScheduler(ag_train,# ag_train_fn
#                                                 resource={'num_cpus': 4, 'num_gpus': 1},
#                                                 num_trials=3,
#                                                 reward_attr='accuracy',
#                                                 time_attr='epoch',
#                                                 grace_period=1)


# detection quick start
# import autogluon as ag
# from autogluon import ObjectDetection as task
# root = './'
# filename_zip = ag.download('https://autogluon.s3.amazonaws.com/datasets/tiny_motorbike.zip',
#                         path=root)
# filename = ag.unzip(filename_zip, root=root)
#
# import os
# data_root = os.path.join(root, filename)
# dataset_train = task.Dataset(data_root, classes=('motorbike',))
#
# time_limits = 5*60*60  # 5 hours
# epochs = 1
# detector = task.fit(dataset_train,
#                     num_trials=2,
#                     epochs=epochs,
#                     lr=ag.Categorical(5e-4, 1e-4),
#                     ngpus_per_trial=1,
#                     time_limits=time_limits)
#
# dataset_test = task.Dataset(data_root, index_file_name='test', classes=('motorbike',))
#
# test_map = detector.evaluate(dataset_test)
# print("mAP on test dataset: {}".format(test_map[1][1]))
#
# image = '000467.jpg'
# image_path = os.path.join(data_root, 'JPEGImages', image)
# ind, prob, loc = detector.predict(image_path)

# customize training script
# import os
# import numpy as np
#
# import mxnet as mx
# from mxnet import gluon, init
# from autogluon.task.image_classification.nets import get_built_in_network
#
# def get_dataset_meta(dataset, basedir='./datasets'):
#     if dataset.lower() == 'apparel':
#         num_classes = 18
#         rec_train = os.path.join(basedir, 'Apparel_train.rec')
#         rec_train_idx = os.path.join(basedir, 'Apparel_train.idx')
#         rec_val = os.path.join(basedir, 'Apparel_test.rec')
#         rec_val_idx = os.path.join(basedir, 'Apparel_test.idx')
#     else:
#         raise NotImplemented
#     return num_classes, rec_train, rec_train_idx, rec_val, rec_val_idx
#
# def test(net, val_data, ctx, batch_fn):
#     metric = mx.metric.Accuracy()
#     val_data.reset()
#     for i, batch in enumerate(val_data):
#         data, label = batch_fn(batch, ctx)
#         outputs = [net(X) for X in data]
#         metric.update(label, outputs)
#
#     return metric.get()
#
#
# def train_loop(args, reporter):
#     lr_steps = [int(args.epochs*0.75), np.inf]
#     ctx = [mx.gpu(i) for i in range(args.num_gpus)] if args.num_gpus > 0 else [mx.cpu()]
#
#     num_classes, rec_train, rec_train_idx, rec_val, rec_val_idx = get_dataset_meta(args.dataset)
#     net = get_built_in_network(args.net, num_classes, ctx)
#
#     train_data, val_data, batch_fn = get_data_rec(
#             args.input_size, args.crop_ratio, rec_train, rec_train_idx,
#             rec_val, rec_val_idx, args.batch_size, args.num_workers,
#             args.jitter_param, args.max_rotate_angle)
#
#     trainer = gluon.Trainer(net.collect_params(), 'sgd', {
#                             'learning_rate': args.lr, 'momentum': args.momentum, 'wd': args.wd})
#     metric = mx.metric.Accuracy()
#     L = gluon.loss.SoftmaxCrossEntropyLoss()
#
#     lr_counter = 0
#     for epoch in range(args.epochs):
#         if epoch == lr_steps[lr_counter]:
#             trainer.set_learning_rate(trainer.learning_rate*args.lr_factor)
#             lr_counter += 1
#
#         train_data.reset()
#         metric.reset()
#         for i, batch in enumerate(train_data):
#             data, label = batch_fn(batch, ctx)
#             with mx.autograd.record():
#                 outputs = [net(X) for X in data]
#                 loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
#             for l in loss:
#                 l.backward()
#
#             trainer.step(args.batch_size)
#             metric.update(label, outputs)
#
#         _, train_acc = metric.get()
#         _, val_acc = test(net, val_data, ctx, batch_fn)
#
#         if reporter is not None:
#             # reporter enables communications with autogluon
#             reporter(epoch=epoch, accuracy=val_acc)
#         else:
#             print('[Epoch %d] Train-acc: %.3f | Val-acc: %.3f' %
#                   (epoch, train_acc, val_acc))
#
# import autogluon as ag
# from autogluon.utils.mxutils import get_data_rec
#
# @ag.args(
#     dataset='apparel',
#     net='resnet18_v1b',
#     epochs=ag.Choice(40, 80),
#     lr=ag.Real(1e-4, 1e-2, log=True),
#     lr_factor=ag.Real(0.1, 1, log=True),
#     batch_size=256,
#     momentum=0.9,
#     wd=ag.Real(1e-5, 1e-3, log=True),
#     num_gpus=1,
#     num_workers=30,
#     input_size=ag.Choice(224, 256),
#     crop_ratio=0.875,
#     jitter_param=ag.Real(0.1, 0.4),
#     max_rotate_angle=ag.space.Int(0, 10),
# )
# def train_finetune(args, reporter):
#     return train_loop(args, reporter)
#
# myscheduler = ag.scheduler.FIFOScheduler(train_finetune,
#                                          resource={'num_cpus': 8, 'num_gpus': 1},
#                                          num_trials=5,
#                                          time_attr='epoch',
#                                          reward_attr="accuracy")
# print(myscheduler)
#
#
# myscheduler.run()
# myscheduler.join_tasks()
#
# myscheduler.get_training_curves(plot=True,use_legend=False)
# print('The Best Configuration and Accuracy are: {}, {}'.format(myscheduler.get_best_config(),
#                                                                myscheduler.get_best_reward()))