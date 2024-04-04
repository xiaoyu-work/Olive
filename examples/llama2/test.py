import time

from olive.workflows import run as olive_run

start_time = time.time()
olive_run("llama2_qlora.json")
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
