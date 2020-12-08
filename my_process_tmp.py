import os
from tqdm import trange  # 显示进度条
from multiprocessing import cpu_count  # 查看cpu核心数
from multiprocessing import Pool  # 并行处理必备，进程池
import cv2
import numpy as np
matches = [100, 200, 300, 400, 500, 600, 700, 800]

# def setcallback(x):
#     with open('result.txt', 'a+') as f:
#         line = str(x) + "\n"
#         f.write(line)

def single_worker(List_imgs, src_path):
    print('begin')
    for i in trange(len(List_imgs)):
        if not List_imgs[i].endswith('.png'):
            continue
        file = List_imgs[i]
        filepath = os.path.join(src_path, file)
        seg = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        w, h = seg.shape
        count = np.zeros(8)
        for m in matches:
            seg[seg == m] = matches.index(m)
        for i in range(w):
            for j in range(h):
                count[seg[i, j]] += 1
        if max(count) > w * h / 2:
            with open('result.txt', 'a+') as f:
                line = str(filepath) + "\n"
                f.write(line)

def single_worker2(List_imgs, src_path, count):
    print('begin')
    for i in trange(len(List_imgs)):
        if not List_imgs[i].endswith('.png'):
            continue
        file = List_imgs[i]
        filepath = os.path.join(src_path, file)
        seg = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        w, h = seg.shape
        for m in matches:
            seg[seg == m] = matches.index(m)
        for i in range(w):
            for j in range(h):
                count[seg[i, j]] += 1
    print(count)
    return count

def single_worker3(List_imgs, src_path, num):
    print('begin')
    count = np.zeros(8)
    for i in trange(len(List_imgs)):
        if not List_imgs[i].endswith('.png'):
            continue
        file = List_imgs[i]
        filepath = os.path.join(src_path, file)
        seg = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        w, h = seg.shape
        for m in matches:
            seg[seg == m] = matches.index(m)
        for i in range(w):
            for j in range(h):
                count[seg[i, j]] += 1
        num[np.argmax(count)] += 1
    return num

if __name__=='__main__':

    if os.path.exists('./result.txt'):
        os.remove('./result.txt')

    results=[]
    SourceImgs = '../badone/label/'
    # 以下重点
    # 将数据集下的图片名加载到List_imgs中
    List_imgs = os.listdir(SourceImgs)
    Len_imgs = len(List_imgs)  # 数据集长度
    num_cores = cpu_count()  # cpu核心数

    if num_cores == 2:  # 双核，将所有数据集分成两个子数据集
        subset1 = List_imgs[:Len_imgs // 2]
        subset2 = List_imgs[Len_imgs // 2:]

        List_subsets = [subset1, subset2]
    elif num_cores == 4:  # 四核，将所有数据集分成四个子数据集
        subset1 = List_imgs[:Len_imgs // 4]
        subset2 = List_imgs[Len_imgs // 4: Len_imgs // 2]
        subset3 = List_imgs[Len_imgs // 2: (Len_imgs * 3) // 4]
        subset4 = List_imgs[(Len_imgs * 3) // 4:]

        List_subsets = [subset1, subset2, subset3, subset4]
    elif num_cores >= 8:  # 八核以上，将所有数据集分成八个子数据集
        num_cores = 8
        subset1 = List_imgs[:Len_imgs // 8]
        subset2 = List_imgs[Len_imgs // 8: Len_imgs // 4]
        subset3 = List_imgs[Len_imgs // 4: (Len_imgs * 3) // 8]
        subset4 = List_imgs[(Len_imgs * 3) // 8: Len_imgs // 2]
        subset5 = List_imgs[Len_imgs // 2: (Len_imgs * 5) // 8]
        subset6 = List_imgs[(Len_imgs * 5) // 8: (Len_imgs * 6) // 8]
        subset7 = List_imgs[(Len_imgs * 6) // 8: (Len_imgs * 7) // 8]
        subset8 = List_imgs[(Len_imgs * 7) // 8:]

        List_subsets = [subset1, subset2, subset3, subset4,
                        subset5, subset6, subset7, subset8]

    # 开辟进程池，不需要改动
    # num_cores为cpu核心数，也就是开启的进程数
    p = Pool(num_cores)
    #
    # 对每个进程分配工作
    for i in range(num_cores):
        #
        # 格式：p.apply_async(task, args=(...))
        # task：当前进程需要进行的任务/函数，只需要填写函数名
        # args：task函数中所需要传入的参数
        # 注意看 List_subsets[i] 就是传入不同的数据子集
        # p.apply_async(func=single_worker, args=(List_subsets[i], SourceImgs,))
        # results.append(p.apply_async(func=single_worker2, args=(List_subsets[i], SourceImgs, np.zeros(8),)))
        results.append(p.apply_async(func=single_worker3, args=(List_subsets[i], SourceImgs, np.zeros(8),)))

    # 当进程完成时，关闭进程池
    # 以下两行代码不需要改动
    p.close()
    p.join()

    count = np.zeros(8)
    for i in results:
        print(i.get())
        count += i.get()
    print('end with')
    print(count)