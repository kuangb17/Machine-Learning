import numpy as np
import math
import statistics

import matplotlib.pyplot as plot

with open('train.csv', mode='r') as f:
    train_data = []
    for line in f:
        train_matrix = line.strip().split(',')
        train_data.append(train_matrix)

num_ind = [0, 5, 9, 11, 12, 13, 14]

for i in range(len(train_data)):
    for j in num_ind:
        train_data[i][j] = float(train_data[i][j])  # change datatype

# obj_dic = {'age': 0, 'balance': 0, 'day': 0, 'duration': 0, 'campaign': 0, 'pdays': 0, 'previous': 0}
obj_dic = {0: 0, 5: 0, 9: 0, 11: 0, 12: 0, 13: 0, 14: 0}
for i in obj_dic:
    obj_dic[i] = statistics.median([row[i] for row in train_data])
#    print(i)

for row in train_data:
    for i in obj_dic:
        if row[i] >= obj_dic[i]:
            row[i] = 'yes'
        else:
            row[i] = 'no'

# print(obj_dic)

with open('test.csv', mode='r') as f:
    test_data = []
    for line in f:
        test_matrix = line.strip().split(',')
        test_data.append(test_matrix)

for i in range(len(test_data)):
    for j in num_ind:
        test_data[i][j] = float(test_data[i][j])  # change datatype

for i in obj_dic:
    obj_dic[i] = statistics.median([row[i] for row in test_data])

for row in test_data:
    for i in obj_dic:
        if row[i] >= obj_dic[i]:
            row[i] = 'yes'
        else:
            row[i] = 'no'
        # ------------------------------------------------------------------------------
Attr_dict = {'age': ['yes', 'no'],
             'job': ['admin.', 'unknown', 'unemployed', 'management', 'housemaid', 'entrepreneur', 'student',
                     'blue-collar', 'self-employed', 'retired', 'technician', 'services'],
             'martial': ['married', 'divorced', 'single'],
             'education': ['unknown', 'secondary', 'primary', 'tertiary'],
             'default': ['yes', 'no'],
             'balance': ['yes', 'no'],
             'housing': ['yes', 'no'],
             'loan': ['yes', 'no'],
             'contact': ['unknown', 'telephone', 'cellular'],
             'day': ['yes', 'no'],
             'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
             'duration': ['yes', 'no'],
             'campaign': ['yes', 'no'],
             'pdays': ['yes', 'no'],
             'previous': ['yes', 'no'],
             'poutcome': ['unknown', 'other', 'failure', 'success']}

keys = []
for i in Attr_dict:
    keys.append(i)
# print(keys)
Attr_set = set(keys)


# print(Attr_set)


# attr_index= []
# for i in keys
#   attr_index.append = keys.index(i) #get index of attr


# ----------------------------------------------------------------------------
def create_list(attr):
    obj_dic = {}
    for attr_val in Attr_dict[attr]:
        obj_dic[attr_val] = []
    return obj_dic  # dict type with list value type


def create_list_0(attr):
    obj_dic = {}
    for attr_val in attr:
        obj_dic[attr_val] = 0
    return obj_dic  # dict type with float value type   dict=(key,value)


def exp_entropy(groups, classes):
    Q = 0.0
    tp = 0.0
    for attr_val in groups:
        tp = sum([row[-1] for row in groups[attr_val]])
        Q = Q + tp

    exp_ent = 0.0
    for attr_val in groups:
        size = float(len(groups[attr_val]))
        if size == 0:
            continue
        score = 0
        q = sum([row[-1] for row in groups[attr_val]])
        for class_val in classes:
            p = sum([row[-1] for row in groups[attr_val] if row[-2] == class_val]) / q
            if p == 0:
                temp = 0
            else:
                temp = p * math.log2(1 / p)
            score += temp
        exp_ent += score * sum([row[-1] for row in groups[attr_val]]) / Q
    return exp_ent


def data_split(attr, dataset):
    branch_obj_dic = create_list(attr)  # this may result in empty dict elements
    for row in dataset:
        for attr_val in Attr_dict[attr]:
            if row[keys.index(attr)] == attr_val:
                branch_obj_dic[attr_val].append(row)
    return branch_obj_dic


def find_best_split(dataset):
    if dataset == []:
        return
    label_values = list(set(row[-2] for row in dataset))
    metric_obj_dic = create_list_0(Attr_dict)
    for attr in Attr_dict:
        groups = data_split(attr, dataset)
        metric_obj_dic[attr] = exp_entropy(groups, label_values)  # change metric here
    best_attr = min(metric_obj_dic, key=metric_obj_dic.get)
    best_groups = data_split(best_attr, dataset)
    return {'best_attr': best_attr, 'best_groups': best_groups}


def leaf_node_label(group):
    majority_labels = [row[-2] for row in group]
    return max(set(majority_labels), key=majority_labels.count)


def if_node_divisible(branch_obj_dic):
    non_empty_indices = [key for key in branch_obj_dic if not (not branch_obj_dic[key])]
    if len(non_empty_indices) == 1:
        return False
    else:
        return True


def child_node(node, max_depth, curr_depth):
    if curr_depth >= max_depth:
        for key in node['best_groups']:
            if node['best_groups'][key] != []:  # and ( not isinstance(node['best_groups'][key],str)):
                # extract nonempty branches
                node[key] = leaf_node_label(node['best_groups'][key])
            else:
                node[key] = leaf_node_label(sum(node['best_groups'].values(), []))
        return
    for key in node['best_groups']:
        if node['best_groups'][key] != []:  # and ( not isinstance(node['best_groups'][key],str)):
            node[key] = find_best_split(node['best_groups'][key])
            child_node(node[key], max_depth, curr_depth + 1)
        else:
            node[key] = leaf_node_label(sum(node['best_groups'].values(), []))


def tree_build(train_data, max_depth):
    root = find_best_split(train_data)
    child_node(root, max_depth, 1)
    return root


def label_predict(node, inst):
    if isinstance(node[inst[keys.index(node['best_attr'])]], dict):
        return label_predict(node[inst[pos(node['best_attr'])]], inst)
    else:
        return node[inst[keys.index(node['best_attr'])]]  # leaf node


def sign_func(val):
    if val > 0:
        return 1.0
    else:
        return -1.0


def label_return(dataset, tree):
    true_label = []
    pred_seq = []  # predicted sequence
    for row in dataset:
        true_label.append(row[-2])
        pre = label_predict(tree, row)
        pred_seq.append(pre)
    return [true_label, pred_seq]


def list_obj_dic(n):
    obj_dic = {}
    for i in range(n):
        obj_dic[i] = []
    return obj_dic


def bin_quan(llist):
    bin_list = []
    for i in range(len(llist)):
        if llist[i] == 'yes':
            bin_list.append(1.0)
        else:
            bin_list.append(-1.0)
    return bin_list


def wt_update(curr_wt, vote, bin_true, bin_pred):  # updating weights
    next_wt = []  # updated wieght
    for i in range(len(bin_true)):
        next_wt.append(curr_wt[i] * math.e ** (- vote * bin_true[i] * bin_pred[i]))
    next_weight = [x / sum(next_wt) for x in next_wt]
    return next_weight


def wt_append(mylist, weights):
    for i in range(len(mylist)):
        mylist[i].append(weights[i])
    return mylist


def wt_update_2_data(data, weight):
    for i in range(len(data)):
        data[i][-1] = weight[i]
    return data


def fin_dec(indiv_pred, vote, data_len, _T):
    fin_pred = []
    for j in range(data_len):
        score = sum([indiv_pred[i][0][j] * vote[i] for i in range(_T)])
        fin_pred.append(sign_func(score))
    return fin_pred


def wt_error(true_label, predicted, weights):
    count = 0  # correct predication count
    for i in range(len(true_label)):
        if true_label[i] != predicted[i]:
            count += weights[i]
    return count


def _error(_true_lb, _pred_lb):
    count = 0
    size = len(_true_lb)
    for i in range(size):
        if _true_lb[i] != _pred_lb[i]:
            count += 1
    return count / size


delta = 1e-8
T = 50


def ada_boost(_T, _delta, train_data):
    pred_result = list_obj_dic(_T)  # +1,-1 dict ele
    vote_say = []
    weights = [row[-1] for row in train_data]
    for i in range(_T):
        tree = tree_build(train_data, 1)  # train stumps
        print(tree['best_attr'])
        [pp_true, qq_pred] = label_return(train_data, tree)  # prediction result 'yes or no'
        pred_result[i].append(bin_quan(qq_pred))
        err = wt_error(pp_true, qq_pred, weights)  # + _delta
        print(err)  # from the 2nd stump err is always clsoe to 0.5
        print(weights[0])
        vote_say.append(0.5 * math.log((1 - err) / err))  # final vote of each stump
        weights = wt_update(weights, 0.5 * math.log((1 - err) / err), bin_quan(pp_true), bin_quan(qq_pred))
        train_data = wt_update_2_data(train_data, weights)
    return [pred_result, vote_say, weights]


W_1 = np.ones(len(train_data)) / len(train_data)  # wt initialization
train_data = wt_append(train_data, W_1)
true_label_bin = bin_quan([row[-2] for row in train_data])


def iteration_error(T_max):
    ERR = []
    for t in range(1, T_max):
        [aa_pred, bb_vote, weights] = ada_boost(t, .001, train_data)
        fin_pred = fin_dec(aa_pred, bb_vote, len(train_data), t)
        ERR.append(_error(true_label_bin, fin_pred))
    return ERR


Err = iteration_error(10)

plot.plot(Err)
plot.ylabel('loss function value')
plot.xlabel('No. of iterations')
plot.title('tolerance= 0.000001, # passings =20000 ')
plot.show()
W_1 = np.ones(len(train_data)) / len(train_data)  # wt initialization
train_data = wt_append(train_data, W_1)
[aa_pred, bb_vote, weights] = ada_boost(T, delta, train_data)
train_data = wt_update_2_data(train_data, weights)
tree_1 = tree_build(train_data, 1)
[pp, qq] = label_return(train_data, tree_1)


def compare(x, y):
    count = 0
    for i in range(len(x)):
        if x[i] != y[i]:
            count += 1
    return count


print(compare(pp, qq))

print(wt_error(bin_quan(pp), bin_quan(qq), weights))

fin_pred = fin_dec(aa_pred, bb_vote, len(train_data), T)
true_label = bin_quan([row[-2] for row in train_data])
print(_error(true_label, fin_pred))

W_1 = np.random.random(len(train_data))
MM = len(train_data)
for i in range(MM):
    if i <= 1000:
        W_1[i] = 100
    else:
        W_1[i] = 80
WW = [x / sum(W_1) for x in W_1]

train_data = wt_append(train_data, WW)
tree = tree_build(train_data, 1)
print(tree['best_attr'])
