# multiprocessing render
import json
import multiprocessing
import subprocess
from dataclasses import dataclass
from typing import Optional
import os
import numpy as np




import argparse

parser = argparse.ArgumentParser(description='distributed rendering')

parser.add_argument('--path_aug', type=str,
                    help='path_aug')
parser.add_argument('--path_orig', type=str,
                    help='path_orig')
args = parser.parse_args()


dirs_aug=os.listdir(args.path_aug)
dirs_orig=os.listdir(args.path_orig)
data=[]

for x in dirs_aug:
  y=x.split("_")[0]

  sub_data={
      
            "path_aug":args.path_aug+"/"+x,
            "path_orig":args.path_orig+"/"+y+"_lower.obj",
            "path_target_aug": x.split(".")[0],

  }
  data.append(sub_data)

VIEWS = ["_front", "_back", "_right", "_left", "_front_right", "_front_left", "_back_right", "_back_left", "_top"]

def check_task_finish(render_dir, view_index):
    files_type = ['rgb', 'normals']
    flag = True
    view_index = "%03d" % view_index
    if os.path.exists(render_dir):
        for t in files_type:
            if t=="rgb":
                folder="image"
            else:
                folder="normal"
            for face in VIEWS:
                fpath = os.path.join(render_dir, f'{folder}/{t}_{view_index}{face}.png')
                # print(fpath)
                if not os.path.exists(fpath):
                    flag = False
    else:
        flag = False

    return flag

def worker(
    queue: multiprocessing.JoinableQueue,
    count: multiprocessing.Value,
    gpu: int,
) -> None:
    while True:
        item = queue.get()
        if item is None:
            break

        view_path_gt= os.path.join("/content/data/gt", item["path_target_aug"]   )
        view_path_input = os.path.join("/content/data/input", item["path_target_aug"]  )

        if check_task_finish(view_path_gt, 0) and check_task_finish(view_path_input, 0) :
            queue.task_done()
            print('========', item, 'rendered', '========')
            continue


        delta_z = str(np.random.uniform(-60, 60, 1)[0]) # left right rotate
        delta_x = str(np.random.uniform(-15, 30, 1)[0])  # up and down rotate
        delta_y = str(0)
        # Perform some operation on the item
        print(item["path_aug"], gpu)
        path_aug = item["path_aug"]
        path_orig = item["path_orig"]
        path_target_aug = item["path_target_aug"]
        command1 = (
            f" CUDA_VISIBLE_DEVICES={gpu} "
            f" blenderproc run --blender-install-path ./ /content/instant-nsr-pl/render/render_blenderproc_persp.py"
            f" --object_path {path_aug} --view 0"
            f" --output_folder /content/data/input"
            f" --object_uid {path_target_aug}"
            f" --resolution 256 "
            f" --radius 1.35 "
            f" --delta_z {delta_z}"
            f" --delta_x {delta_x}"
            f" --delta_y {delta_y} "
        )
        command2 = (
            f" CUDA_VISIBLE_DEVICES={gpu} "
            f" blenderproc run --blender-install-path ./ /content/instant-nsr-pl/render/render_blenderproc_persp.py"
            f" --object_path {path_orig} --view 0"
            f" --output_folder /content/data/gt"
            f" --object_uid {path_target_aug}"
            f" --resolution 256 "
            f" --radius 1.35 "
            f" --delta_z {delta_z}"
            f" --delta_x {delta_x}"
            f" --delta_y {delta_y} "
        )
    
        print(command1)
        print(command2)

        subprocess.run(command1, shell=True)
        subprocess.run(command2, shell=True)

        with count.get_lock():
            count.value += 1

        queue.task_done()

        os.system('cls')


queue = multiprocessing.JoinableQueue()
count = multiprocessing.Value("i", 0)

gpu_list=[0]
num_gpus = 1
workers_per_gpu = 32
# Start worker processes on each of the GPUs
for gpu_i in range(num_gpus):
    for worker_i in range(workers_per_gpu):
        worker_i = gpu_i * workers_per_gpu + worker_i
        process = multiprocessing.Process(
            target=worker, args=(queue, count, gpu_list[gpu_i])
        )
        process.daemon = True
        process.start()
    


for item in data:

    queue.put(item)

# Wait for all tasks to be completed
queue.join()

# Add sentinels to the queue to stop the worker processes
for i in range(num_gpus * workers_per_gpu):
    queue.put(None)