from sklearn.datasets import make_classification
import numpy as np
import random
import time, copy

def all_np(arr):
    arr = np.array(arr)
    key = np.unique(arr)
    result = {}
    for k in key:
        mask = (arr == k)
        arr_new = arr[mask]
        v = arr_new.size
        result[k] = v
    return result

def all_list(arr):  # list
    result = {}
    for i in set(arr):
        result[i] = arr.count(i)
    return result

def balance(train_data):
    label_group = []
    for i, image_label in enumerate(train_data._data.items):
        label_group.append(image_label[1])

    result = all_list(label_group)

    # batch = 45
    # s = [train_data[idx] for idx in batch]
    print("class_balance")
    return train_data

def balance_train(list_label_group):
    origin_label_group = []
    origin_images_group = []
    for i, image_label in enumerate(list_label_group):
        origin_label_group.append(image_label[1])
        origin_images_group.append(image_label[0])

    # list(tuple) -> dict
    dict_train = {}
    for name, label in zip(origin_images_group, origin_label_group):
        # label exist
        if not label in dict_train.keys():
            dict_train[label] = [name]
        else:
            dict_train[label].append(name)

    # frequency function
    # start_time = time.perf_counter()
    # frequency = all_np(origin_label_group)
    # end_time = time.perf_counter()
    # time1 = end_time - start_time
    # print('frequency in {} seconds'.format(end_time - start_time))
    # start_time = time.perf_counter()
    frequency = all_list(origin_label_group)
    # end_time = time.perf_counter()
    # time2 = end_time - start_time
    # print('frequency_2 in {} seconds'.format(end_time - start_time))
    # print('1/2 in {} '.format(time1/time2))

    # label balance

    # start_time = time.perf_counter()
    # for i in dict_train.keys():
    #     start = 0
    #     end = start + frequency[i]
    #     list_value = origin_images_group[start:end]
    #     start = end
    #     while len(dict_train[i]) < max(frequency.values()):
    #         dict_train[i].append(random.choice(list_value))
    # end_time = time.perf_counter()
    # time0 = end_time - start_time
    # print('0 in {} seconds'.format(end_time - start_time))


    start_time = time.perf_counter()
    for i in dict_train.keys():
        list_value = dict_train[i]
        while len(dict_train[i]) < max(frequency.values()):
            dict_train[i].append(random.choice(list_value))
    end_time = time.perf_counter()
    time1 = end_time - start_time
    print('1 in {} seconds'.format(time1))

    # start_time = time.perf_counter()
    # for i in dict_train.keys():
    #     # list_value = dict_train[i]
    #     list_value_2  = copy.deepcopy(dict_train[i])
    #     while len(dict_train[i]) < max(frequency.values()):
    #         dict_train[i].append(random.choice(list_value_2))
    #
    # end_time = time.perf_counter()
    # time2 = end_time - start_time
    # print('2 in {} seconds'.format(end_time - start_time))
    #
    # print('1/0 in {} '.format(time1 / time0))
    # print('1/2 in {} '.format(time1 / time2))
    # print('0/2 in {} '.format(time0 / time2))


    # dict -> list(tuple)
    balance_list_label_group = []
    for i, j in dict_train.items():
        for jj in j:
            balance_list_label_group.append(tuple((jj, i)))

    return balance_list_label_group

# FORMAT
# input
list_label_group = [('a',1),('r',1),
                    ('b',2),('f',2),
                    ('c',3),('1a',3), ('1b',3), ('1c',3),('f8',3),
                    ('1d',4), ('d',4), ('2a',4), ('2b',4),('2c',4),('2d',4),('3d',4),('4d',4),('5d',4),('8f',4),('9d',4)
                    ]

# unbance data into balance
dict_train = balance_train(list_label_group)

print(dict_train)


# min_num_classes =0
# labels_0 = list(dict_train.keys())
# labels = [k for k in dict_train.keys() if len(dict_train[k]) >= min_num_classes]






