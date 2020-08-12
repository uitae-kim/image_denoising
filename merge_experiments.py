import os
import numpy as np
import itertools


def merge(files, sigma_list):
    result = ''
    for i, file in enumerate(files):
        if i == 0:
            result += ','.join(file[0]) +',avg,best' + '\n'

        result += 'noise level,' + sigma_list[i] + '\n'

        for j in range(1, len(file)):
            val = list(map(float, file[j][1:]))
            avg = np.mean(val)
            max_val = max(val)
            max_idx = val.index(max_val)
            max_tech_name = file[0][max_idx + 1]

            result += file[j][0]

            for v in val:
                result += f',{v:.3f}'

            result += f',{avg},{max_tech_name}' + '\n'

    return result


if __name__ == '__main__':
    sizes = ['064', '128', '256', '512']
    sigma = ['15', '25', '50']
    fdict = {
        sizes[0]: [],
        sizes[1]: [],
        sizes[2]: [],
        sizes[3]: [],
    }

    chain = itertools.chain(itertools.product(sizes, sigma))
    for c in chain:
        fname = f'./experiments/{c[0]}_{c[1]}.csv'
        with open(fname) as f:
            item = []
            while True:
                line = f.readline()
                if line == '':
                    break
                item.append(line.rstrip().split(','))
            fdict[c[0]].append(item)

    for size in sizes:
        result = merge(fdict[size], sigma)
        with open(f'./experiments/{size}.csv', 'w') as f:
            f.write(result)
