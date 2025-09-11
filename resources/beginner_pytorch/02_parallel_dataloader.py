from multiprocessing import freeze_support
import time

import torch
from torch.utils.data import Dataset, DataLoader
from timed_decorator.simple_timed import timed

class SleepyDataset(Dataset):
    def __init__(self):
        self.data = [1, 2, 3, 4]
        self.labels = ["odd", "even", "odd", "even"]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            print(f"Loading item {i} in main process")
        else:
            print(f"Worker {worker_info.id}/{worker_info.num_workers} is loading item {i}")

        time.sleep(1)  # Simulate a slow loading process
        return self.data[i], self.labels[i]
    

@timed(use_seconds=True, show_args=True, return_time=True)
def load_data(num_workers: int):
    dataset = SleepyDataset()
    dataloader = DataLoader(dataset, batch_size=1, num_workers=num_workers)
    for _ in dataloader:
        time.sleep(1)  # Simulate training


if __name__ == "__main__":
    freeze_support()
    _, t0 = load_data(0)
    _, t1 = load_data(2)
    _, t2 = load_data(4)
    _, t3 = load_data(8)
    print()
    print(f"num_workers: {0}, time: {t0} seconds")
    print(f"num_workers: {2}, time: {t1} seconds")
    print(f"num_workers: {4}, time: {t2} seconds")
    print(f"num_workers: {8}, time: {t3} seconds")
    print(f"Speedup: {t0/t1}, {t0/t2}, {t0/t3}")
