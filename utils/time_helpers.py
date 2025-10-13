import time

def print_local_time():
    local_time = time.localtime()
    formatted_local_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    current_time = time.time()

    print(f"Time: {formatted_local_time}\n")
