import os  
import shutil  
from tqdm import tqdm  
if __name__=='__main__':  
    from imagededup.methods import PHash  
    phasher = PHash()  
    imgs_path = r"E:\Projects\CG\WeatherSimulation\Raining\Experiment\RainDataset\MSPFN\mixtrain\label"
    merge = r"E:\Projects\CG\WeatherSimulation\Raining\Experiment\RainDataset\MSPFN\MSPFN_train"
    new_dataset_duplicates = phasher.find_duplicates(image_dir=imgs_path)  
    print(new_dataset_duplicates)  
    new_k = []  
    new_v = []  
    for k, v in tqdm(new_dataset_duplicates.items(), desc="筛选新数据集"):  
        '''其实没必要搞这么麻烦，主要是防止出现类似 1：[], 2：[3, 1]或者 
        2：[3, 1], 1：[]的情况，但会大大增加计算量，要求不大的话一层for循环就能搞定''' 
        if len(v) == 0 and (k not in new_v):  
            new_k.append(k)  
            new_v.extend(v)  
            continue  
        if k not in new_v:  
            if len(new_k) == 0:  
                new_k.append(k)  
                new_v.extend(v)  
                continue  
            for jug in new_k:  
                if jug in v:  
                    break  
                else:  
                    new_k.append(k)  
                    new_v.extend(v)  
                    break  
    # 将新数据集中不重复的数据迁移到另一个文件夹中  
    for q in tqdm(new_k, desc='迁移不重复新数据集'):  
        local = os.path.join(imgs_path, q)  
        shutil.copy(local, merge)
