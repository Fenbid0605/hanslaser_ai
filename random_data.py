import random

import numpy as np
import pandas as pd

from dataset import DataSet

if __name__ == '__main__':
    dataset = DataSet()
    rows = dataset.universal.rows
    # L 排序
    rows.sort(key=lambda x: float(x[4].value))

    random.seed(1003)

    choices = []
    for row in rows:
        random_number = random.randint(0, 100)
        if random_number < 9:
            choices.append([float(c.value) for c in row])

    frame = pd.DataFrame(np.array(choices),
                         columns=['速度', '电流', 'Q频', 'Q释放',
                                  'L', 'A', 'B'])
    print(frame)
    frame.to_excel('random_data.xlsx')
