import math

with open('train.csv', mode='r') as f:
    myList_train = [];
    for line in f:
        terms = line.strip().split(',')  # 7*N matrix
        myList_train.append(terms)

Attr_dict = {'buying': ['vhigh', 'high', 'med', 'low'],
             'maint': ['vhigh', 'high', 'med', 'low'],
             'doors': ['2', '3', '4', '5more'],
             'persons': ['2', '4', 'more'],
             'lug_boot': ['small', 'med', 'big'],
             'safety': ['low', 'med', 'high']}


# myList_train


# function of entropy: ent=-sum([p_i*log(p_i)
# attribute: different attributes values
# lable: possible lables
def entropy(attribute, lable):
    ent = 0.0
    N_ex = float(sum([len(attribute[attr_val]) for attr_val in lable])) # number of the example

    for attr_val in attribute:
        size = float(len(attribute[attr_val]))

        if size == 0: # avoid devide by 0
            continue
        score = 0.0
        for class_val in lable:
            p = [row[-1] for row in attribute[attr_val]].count(class_val) / size
            if p == 0:
                temp = 0
            else:
                temp = p * math.log2(1 / p)
            score += temp

        ent += (size / N_ex)*score
    return ent

# function of  gini_index: 1-sum(p_k*p_k)
# attribute: different attributes values
# lable: possible lables
def gini_index(attribute, lable):
    gini = 0.0
    N_ex = float(sum([len(attribute[attr_val]) for attr_val in lable]))

    for attr_val in attribute:
        size = float(len(lable[attr_val]))

        if size == 0:
            continue
        score = 0.0
        for class_val in lable:  # label values
            p = [row[-1] for row in attribute[attr_val]].count(class_val) / size
            score += p * p
        gini += (size / N_ex)*(1.0 - score)

    return gini


# function of  MajorityError: 1-max(p_1,p_2,...,p_v)
# attribute: different attributes values
# lable: possible lables
def m_error(attribute, lable):
    m_err = 0.0
    N_ex = float(sum([len(attribute[attr_val]) for attr_val in lable]))

    for attr_val in lable:
        size = float(len(attribute[attr_val]))
        if size == 0:
            continue
        score = 0.0
        temp = 0
        for class_val in lable:
            p = [row[-1] for row in lable[attr_val]].count(class_val) / size
            temp = max(temp, p)
            score = 1 - temp
        m_err +=(size / N_ex) *score

    return m_err
