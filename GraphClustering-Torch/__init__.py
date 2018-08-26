#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

File Name :
File Description :
Author : Liangwei Li

"""

if __name__ == '__main__':
    from tqdm import tqdm
    import time

    with tqdm(total=100) as pbar:
        for i in range(10):
            time.sleep(1)
            pbar.update(10)