from tqdm import tqdm
import time

for _ in tqdm(range(100), desc="Progress", leave=False):
    time.sleep(0.1)

print('Done')
