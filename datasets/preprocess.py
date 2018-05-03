import sys, os
import numpy as np
import pandas as pd
from tqdm import tqdm
sys.path.append('./')


def drop_singleton(data_dir, target_dir, buy=False):
    buffer_size = 1000
    id_cache = None
    session = []
    saved = []
    drop_count = 0

    with open(target_dir, 'wt') as t:
        # write header
        t.writelines("sess_id, item_id, category\n")

    with open(data_dir, 'rt') as f:
        for i, line in enumerate(f):
            sess_id, time, item_id, category = line.split(sep=',')
            # drop time, regularize "S" character in category
            category = category.replace("S", "13")
            new_line = f"{sess_id},{str(item_id)},{category}"
            
            #detecting session change
            if id_cache != sess_id: 
                id_cache = sess_id
                
                # save non-singletons only
                if len(session) < 2 and i is not 0:
                    drop_count += 1
                    continue
                else:
                    saved += session
                
                session = [new_line]
            else:
                session.append(new_line)

            if len(saved) > buffer_size:
                with open(target_dir, 'at') as t:
                    t.writelines(saved)
                    saved = []
        # write remaining lines at the end of loop
        with open(target_dir, 'at') as t:
            t.writelines(saved)
            saved = []
    print(f"file {data_dir} converted to {target_dir}.\n {drop_count} sessions dropped.")
    
        
def is_singleton(parameter_list):
    pass


if __name__ == '__main__':
    data_click_dir = 'datasets/data/yoochoose-clicks.dat'
    data_buys_dir = 'datasets/data/yoochoose-buys.dat'
    data_test_dir = 'datasets/data/yoochoose-test.dat'

    target_train = 'datasets/data/train.csv'
    target_buys = 'datasets/data/buys.csv'
    target_test = 'datasets/data/test.csv'
    

    # drop_singleton(data_buys_dir, target_buys)
    drop_singleton(data_test_dir, target_test)
    drop_singleton(data_click_dir, target_train)
