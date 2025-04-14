import time

TIME_PRINT_HISTORY = {}

def print_local_time():
    local_time = time.localtime()
    formatted_local_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    current_time = time.time()

    if 'time_printed' not in TIME_PRINT_HISTORY:
        TIME_PRINT_HISTORY['time_printed'] = current_time
        print("Time:", formatted_local_time)
    else:
        elapsed_time = current_time - TIME_PRINT_HISTORY['time_printed']
        if elapsed_time >= 15 * 60:  # 15 minutes in seconds
            TIME_PRINT_HISTORY['time_printed'] = current_time
            print("Time:", formatted_local_time)
