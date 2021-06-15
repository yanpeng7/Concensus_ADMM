import os


def check_and_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def dict2pformat(x: dict):
    ret = ''
    for k in x:
        ret += ' %s: [%s]' % (k, str(x[k]))
    return ret


def dict2md_table(ipt: dict):
    ret = str()
    for section in ipt.keys():

        ret += '## ' + section + '\n'
        ret += '|  Key  |  Value |\n|:----:|:---:|\n'

        for i in ipt[section].keys():
            ret += '|' + i + '|' + str(ipt[section][i]) + '|\n'

        ret += '\n\n'

    return ret
