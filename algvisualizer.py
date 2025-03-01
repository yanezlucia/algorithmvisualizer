import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random
from tkinter import *
from tkinter import messagebox
import time

# Global variables
array = []
algo_vars = {}
size_entry = None
custom_entry = None
array_display = None
target_entry = None  # For Linear Search

PAUSE_TIME = 0.05  # Consistent animation speed for all algorithms

# At the top of the file, add these color constants
BUBBLE_COLOR = '#FF9999'    # Light red
MERGE_COLOR = '#66B2FF'     # Light blue
QUICK_COLOR = '#99FF99'     # Light green
LINEAR_COLOR = '#FFCC99'    # Light orange
MSD_COLOR = '#FF99CC'       # Light pink
LSD_COLOR = '#99FFCC'       # Light cyan

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
            plt.clf()
            plt.bar(range(len(arr)), arr, color=BUBBLE_COLOR)
            plt.title('Bubble Sort')
            plt.pause(PAUSE_TIME)

def merge_sort(arr):
    def merge_sort_recursive(arr):
        if len(arr) > 1:
            mid = len(arr) // 2
            left_half = arr[:mid]
            right_half = arr[mid:]
            yield from merge_sort_recursive(left_half)
            yield from merge_sort_recursive(right_half)
            i = j = k = 0
            while i < len(left_half) and j < len(right_half):
                if left_half[i] <= right_half[j]:
                    arr[k] = left_half[i]
                    i += 1
                else:
                    arr[k] = right_half[j]
                    j += 1
                k += 1
                yield arr.copy()
            while i < len(left_half):
                arr[k] = left_half[i]
                i += 1
                k += 1
                yield arr.copy()
            while j < len(right_half):
                arr[k] = right_half[j]
                j += 1
                k += 1
                yield arr.copy()
    yield from merge_sort_recursive(arr)

def linear_search(arr, target):
    if not arr:  # Handle empty array
        return -1
    
    try:
        for i in range(len(arr)):
            plt.clf()
            colors = ['red' if j == i else 'blue' for j in range(len(arr))]
            plt.bar(range(len(arr)), arr, color=LINEAR_COLOR)
            plt.title('Linear Search')
            plt.pause(PAUSE_TIME)
            
            if arr[i] == target:
                plt.clf()
                colors = ['green' if j == i else 'blue' for j in range(len(arr))]
                plt.bar(range(len(arr)), arr, color=LINEAR_COLOR)
                plt.title('Linear Search - Found!')
                plt.pause(PAUSE_TIME)
                plt.close()
                return i
        
        plt.close()
        return -1
    except Exception as e:
        plt.close()
        raise e

def quick_sort(arr):
    if not arr:
        return arr
    if len(arr) <= 1:
        return arr

    def quick_sort_recursive(arr, start, end):
        if start >= end:
            return arr

        pivot_idx = (start + end) // 2
        pivot = arr[pivot_idx]
        
        # Move pivot to end
        arr[pivot_idx], arr[end] = arr[end], arr[pivot_idx]
        plt.clf()
        plt.bar(range(len(arr)), arr, color='purple')  # Add color
        plt.title('Quick Sort')
        plt.pause(PAUSE_TIME)

        store_idx = start
        for i in range(start, end):
            if arr[i] < pivot:
                arr[store_idx], arr[i] = arr[i], arr[store_idx]
                store_idx += 1
                plt.clf()
                plt.bar(range(len(arr)), arr, color='purple')  # Add color
                plt.title('Quick Sort')
                plt.pause(PAUSE_TIME)

        arr[store_idx], arr[end] = arr[end], arr[store_idx]
        plt.clf()
        plt.bar(range(len(arr)), arr, color='purple')  # Add color
        plt.title('Quick Sort')
        plt.pause(PAUSE_TIME)

        quick_sort_recursive(arr, start, store_idx - 1)
        quick_sort_recursive(arr, store_idx + 1, end)
        return arr  # Add return statement

    try:
        arr = quick_sort_recursive(arr, 0, len(arr) - 1)  # Capture return value
        plt.show()
        plt.close()
        return arr
    except Exception as e:
        plt.close()
        raise e

def lsd_radix_sort(arr):
    if not arr:
        return arr
    if len(arr) <= 1:
        return arr

    try:
        max_num = max(arr)
        exp = 1
        
        while max_num // exp > 0:
            counting = [0] * 10
            output = [0] * len(arr)

            for i in range(len(arr)):
                index = arr[i] // exp % 10
                counting[index] += 1
                plt.clf()
                plt.bar(range(len(arr)), arr, color=LSD_COLOR)
                plt.title(f'LSD Radix Sort (Digit: {exp})')
                plt.pause(PAUSE_TIME)

            for i in range(1, 10):
                counting[i] += counting[i - 1]

            i = len(arr) - 1
            while i >= 0:
                index = arr[i] // exp % 10
                output[counting[index] - 1] = arr[i]
                counting[index] -= 1
                i -= 1
                plt.clf()
                plt.bar(range(len(arr)), output, color=LSD_COLOR)
                plt.title(f'LSD Radix Sort (Building output)')
                plt.pause(PAUSE_TIME)

            for i in range(len(arr)):
                arr[i] = output[i]
                plt.clf()
                plt.bar(range(len(arr)), arr, color=LSD_COLOR)
                plt.title('LSD Radix Sort')
                plt.pause(PAUSE_TIME)

            exp *= 10

        plt.show()
        plt.close()
        return arr
    except Exception as e:
        plt.close()
        raise e

def msd_radix_sort(arr):
    if not arr:
        return arr
    if len(arr) <= 1:
        return arr

    def get_digit(num, digit_place):
        return (num // (10 ** digit_place)) % 10

    def msd_radix_sort_recursive(arr, start, end, digit_place):
        if start >= end or digit_place < 0:
            return

        buckets = [[] for _ in range(10)]

        for i in range(start, end + 1):
            digit = get_digit(arr[i], digit_place)
            buckets[digit].append(arr[i])
            plt.clf()
            plt.bar(range(len(arr)), arr, color=MSD_COLOR)
            plt.title(f'MSD Radix Sort (Digit: {digit_place})')
            plt.pause(PAUSE_TIME)

        pos = start
        for bucket in buckets:
            for num in bucket:
                arr[pos] = num
                pos += 1
                plt.clf()
                plt.bar(range(len(arr)), arr, color=MSD_COLOR)
                plt.title('MSD Radix Sort')
                plt.pause(PAUSE_TIME)

            bucket_start = pos - len(bucket)
            if len(bucket) > 1 and digit_place > 0:
                msd_radix_sort_recursive(arr, bucket_start, pos - 1, digit_place - 1)

    try:
        max_num = max(arr)
        max_digits = len(str(max_num))
        msd_radix_sort_recursive(arr, 0, len(arr) - 1, max_digits - 1)
        plt.show()
        plt.close()
        return arr
    except Exception as e:
        plt.close()
        raise e

def start_sorting():
    if not array:
        messagebox.showerror("Error", "Please generate an array first")
        return

    selected = [algo for algo, var in algo_vars.items() if var.get()]
    if not selected:
        messagebox.showerror("Error", "Please select at least one algorithm")
        return

    # Get target value for linear search
    if "Linear Search" in selected:
        try:
            target = int(target_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid target value for linear search")
            return
    # Create figure with subplots
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(15, 6))

    # Create main sorting subplot and timer subplot
    ax = plt.subplot2grid((1, 5), (0, 0), colspan=4)
    timer_ax = plt.subplot2grid((1, 5), (0, 4))
    ax.set_title('Algorithm Visualizer')
    ax.set_xlabel('Array Index')
    ax.set_ylabel('Value')

    # Hide timer axis but keep the subplot
    timer_ax.axis('off')
    timer_text = {}
    timer_position = 0.9

    # Create timer display for each algorithm
    for algo in selected:
        color = f'C{selected.index(algo)}'
        timer_text[algo] = timer_ax.text(0.1, timer_position, f'{algo}: 0.000s',
                                        fontsize=11, color=color)
        timer_position -= 0.2

    # Make a copy of the array for each algorithm.
    arrays = {algo: array.copy() for algo in selected}
    bars = {algo: ax.bar(range(len(array)), arrays[algo], alpha=0.3, label=algo)
            for algo in selected}
    ax.legend()
    plt.tight_layout()

    def update_plot(times):
        for algo in selected:
            for idx, (rect, val) in enumerate(zip(bars[algo], arrays[algo])):
                rect.set_height(val)
                # Special highlighting for Linear Search.
                if algo == "Linear Search":
                    curr_idx = states[algo]['current_index']
                    if idx < curr_idx:
                        if idx in states[algo]['found']:
                            rect.set_color("green")
                        else:
                            rect.set_color("blue")
                    elif idx == curr_idx:
                        rect.set_color("red")
                    else:
                        rect.set_color("gray")
                # For LSD Radix Sort, highlight the current element.
                elif algo == "LSD Radix Sort":
                    phase = states[algo]["phase"]
                    curr_i = states[algo]["i"]
                    if phase in ["count", "build_output", "copy_back"]:
                        if idx == curr_i:
                            rect.set_color("red")
                        else:
                            rect.set_color("gray")
                    else:
                        rect.set_color("gray")
                # For MSD Radix Sort, highlight the element currently being processed.
                elif algo == "MSD Radix Sort":
                    if states[algo]["stack"]:
                        current_task = states[algo]["stack"][-1]
                        if current_task["phase"] == "bucket":
                            target_index = current_task["start"] + current_task["i"]
                            if idx == target_index:
                                rect.set_color("red")
                            else:
                                rect.set_color("gray")
                        elif current_task["phase"] == "copy_back":
                            target_index = current_task["start"] + current_task["i"]
                            if idx == target_index:
                                rect.set_color("red")
                            else:
                                rect.set_color("gray")
                        else:
                            rect.set_color("gray")
                    else:
                        rect.set_color("gray")
                else:
                    rect.set_color("gray")
            if sorting_complete[algo]:
                timer_text[algo].set_text(f'{algo}: {times[algo]:.3f}s ✓')
            else:
                current_time = time.time() - start_times[algo]
                timer_text[algo].set_text(f'{algo}: {current_time:.3f}s')
        fig.canvas.draw_idle()
        plt.pause(0.01)

    # Initialize per-algorithm state.
    states = {}
    for algo in selected:
        if algo == 'Bubble Sort':
            states[algo] = {'i': 0, 'j': 0, 'n': len(arrays[algo])}
        elif algo == 'Merge Sort':
            states[algo] = {'sorted_size': 1}
        elif algo == 'Quick Sort':
            states[algo] = {'stack': [(0, len(arrays[algo]) - 1)]}
        elif algo == 'Linear Search':
            states[algo] = {'current_index': 0, 'found': []}
        elif algo == 'LSD Radix Sort':
            states[algo] = {
                "exp": 1,
                "phase": "count",  # phases: "count", "cumulative", "build_output", "copy_back"
                "i": 0,
                "n": len(arrays[algo]),
                "max_num": max(arrays[algo]) if arrays[algo] else 0,
                "count": [0] * 10,
                "output": [0] * len(arrays[algo])
            }
        elif algo == 'MSD Radix Sort':
            # Initialize a stack for recursive tasks.
            max_num = max(arrays[algo]) if arrays[algo] else 0
            max_digits = len(str(max_num)) if max_num > 0 else 1
            states[algo] = {
                "stack": [
                    {
                        "start": 0,
                        "end": len(arrays[algo]) - 1,
                        "digit_place": max_digits - 1,
                        "phase": "bucket",  # phases: "bucket", then "copy_back"
                        "i": 0,
                        "buckets": [ [] for _ in range(10) ]
                    }
                ]
            }

    sorting_complete = {algo: False for algo in selected}
    start_times = {algo: time.time() for algo in selected}
    completion_times = {algo: 0 for algo in selected}

    # Main simulation loop.
    while not all(sorting_complete.values()):
        for algo in selected:
            if sorting_complete[algo]:
                continue

            arr = arrays[algo]
            state = states[algo]

            if algo == 'Bubble Sort':
                if state['j'] < state['n'] - state['i'] - 1:
                    if arr[state['j']] > arr[state['j'] + 1]:
                        arr[state['j']], arr[state['j'] + 1] = arr[state['j'] + 1], arr[state['j']]
                    state['j'] += 1
                else:
                    state['i'] += 1
                    state['j'] = 0
                    if state['i'] >= state['n']:
                        sorting_complete[algo] = True
                        completion_times[algo] = time.time() - start_times[algo]

            elif algo == 'Merge Sort':
                if state['sorted_size'] < len(arr):
                    for start in range(0, len(arr), 2 * state['sorted_size']):
                        mid = min(start + state['sorted_size'], len(arr))
                        end = min(start + 2 * state['sorted_size'], len(arr))
                        left = arr[start:mid]
                        right = arr[mid:end]
                        i = j = 0
                        k = start
                        while i < len(left) and j < len(right):
                            if left[i] <= right[j]:
                                arr[k] = left[i]
                                i += 1
                            else:
                                arr[k] = right[j]
                                j += 1
                            k += 1
                        while i < len(left):
                            arr[k] = left[i]
                            i += 1
                            k += 1
                        while j < len(right):
                            arr[k] = right[j]
                            j += 1
                            k += 1
                    state['sorted_size'] *= 2
                else:
                    sorting_complete[algo] = True
                    completion_times[algo] = time.time() - start_times[algo]

            elif algo == 'Quick Sort':
                if state['stack']:
                    low, high = state['stack'].pop()
                    if low < high:
                        pivot = arr[high]
                        i = low - 1
                        for j in range(low, high):
                            if arr[j] <= pivot:
                                i += 1
                                arr[i], arr[j] = arr[j], arr[i]
                        arr[i + 1], arr[high] = arr[high], arr[i + 1]
                        pi = i + 1
                        state['stack'].append((pi + 1, high))
                        state['stack'].append((low, pi - 1))
                else:
                    sorting_complete[algo] = True
                    completion_times[algo] = time.time() - start_times[algo]

            elif algo == 'Linear Search':
                if state['current_index'] < len(arr):
                    if arr[state['current_index']] == target:
                        state['found'].append(state['current_index'])
                    state['current_index'] += 1
                else:
                    sorting_complete[algo] = True
                    completion_times[algo] = time.time() - start_times[algo]

            elif algo == 'LSD Radix Sort':
                # Check if more passes are required.
                if state["max_num"] // state["exp"] == 0:
                    sorting_complete[algo] = True
                    completion_times[algo] = time.time() - start_times[algo]
                else:
                    if state["phase"] == "count":
                        if state["i"] < state["n"]:
                            idx = (arr[state["i"]] // state["exp"]) % 10
                            state["count"][idx] += 1
                            state["i"] += 1
                        else:
                            state["phase"] = "cumulative"
                            state["i"] = 1
                    elif state["phase"] == "cumulative":
                        if state["i"] < 10:
                            state["count"][state["i"]] += state["count"][state["i"] - 1]
                            state["i"] += 1
                        else:
                            state["phase"] = "build_output"
                            state["i"] = state["n"] - 1
                    elif state["phase"] == "build_output":
                        if state["i"] >= 0:
                            idx = (arr[state["i"]] // state["exp"]) % 10
                            state["output"][state["count"][idx] - 1] = arr[state["i"]]
                            state["count"][idx] -= 1
                            state["i"] -= 1
                        else:
                            state["phase"] = "copy_back"
                            state["i"] = 0
                    elif state["phase"] == "copy_back":
                        if state["i"] < state["n"]:
                            arr[state["i"]] = state["output"][state["i"]]
                            state["i"] += 1
                        else:
                            state["exp"] *= 10
                            if state["max_num"] // state["exp"] == 0:
                                sorting_complete[algo] = True
                                completion_times[algo] = time.time() - start_times[algo]
                            else:
                                state["phase"] = "count"
                                state["i"] = 0
                                state["count"] = [0] * 10
                                state["output"] = [0] * state["n"]

            elif algo == 'MSD Radix Sort':
                # Use a stack of tasks to simulate recursive MSD.
                if state["stack"]:
                    current_task = state["stack"][-1]
                    if current_task["phase"] == "bucket":
                        length = current_task["end"] - current_task["start"] + 1
                        if current_task["i"] < length:
                            idx = current_task["start"] + current_task["i"]
                            elem = arr[idx]
                            bucket_index = (elem // (10 ** current_task["digit_place"])) % 10
                            current_task["buckets"][bucket_index].append(elem)
                            current_task["i"] += 1
                        else:
                            current_task["phase"] = "copy_back"
                            # Build flattened list from buckets.
                            current_task["flattened"] = []
                            for b in range(10):
                                current_task["flattened"].extend(current_task["buckets"][b])
                            current_task["i"] = 0
                    elif current_task["phase"] == "copy_back":
                        if current_task["i"] < len(current_task["flattened"]):
                            arr[current_task["start"] + current_task["i"]] = current_task["flattened"][current_task["i"]]
                            current_task["i"] += 1
                        else:
                            # Determine bucket boundaries.
                            boundaries = []
                            start_index = current_task["start"]
                            for b in range(10):
                                bucket_len = len(current_task["buckets"][b])
                                boundaries.append((start_index, start_index + bucket_len - 1))
                                start_index += bucket_len
                            state["stack"].pop()  # Finished with current task.
                            # Push new tasks for buckets that need further sorting.
                            for b in range(10):
                                b_start, b_end = boundaries[b]
                                if b_start <= b_end and (b_end - b_start + 1) > 1 and current_task["digit_place"] > 0:
                                    new_task = {
                                        "start": b_start,
                                        "end": b_end,
                                        "digit_place": current_task["digit_place"] - 1,
                                        "phase": "bucket",
                                        "i": 0,
                                        "buckets": [ [] for _ in range(10) ]
                                    }
                                    state["stack"].append(new_task)
                            if not state["stack"]:
                                sorting_complete[algo] = True
                                completion_times[algo] = time.time() - start_times[algo]
                else:
                    sorting_complete[algo] = True
                    completion_times[algo] = time.time() - start_times[algo]

        update_plot(completion_times)

    # Show final times in sorted order
    final_text = "Final Times:\n"
    sorted_times = sorted(completion_times.items(), key=lambda x: x[1])
    for algo, time_taken in sorted_times:
        final_text += f"{algo}: {time_taken:.3f}s\n"
        if algo == "Linear Search":
            if states[algo]['found']:
                final_text += f"Found target at indices: {states[algo]['found']}\n"
            else:
                final_text += "Target not found\n"
    messagebox.showinfo("Sorting/Search Complete", final_text)
    plt.show()

# Array Helpers #
def generate_array():
    try:
        size = int(size_entry.get())
        global array
        array = [random.randint(1, 100) for _ in range(size)]
        update_array_display()
    except ValueError:
        messagebox.showerror("Error", "Please enter a valid number")

def use_custom_array():
    try:
        input_str = custom_entry.get()
        global array
        array = [int(x.strip()) for x in input_str.split(',')]
        update_array_display()
    except ValueError:
        messagebox.showerror("Error", "Please enter valid comma-separated numbers")

def update_array_display():
    array_display.delete(1.0, END)
    array_display.insert(END, str(array))

def select_all():
    for var in algo_vars.values():
        var.set(True)

def deselect_all():
    for var in algo_vars.values():
        var.set(False)

# Create main window
root = Tk()
root.title("Sorting/Search Algorithm Visualizer")

# Create and pack widgets
frame = Frame(root)
frame.pack(padx=10, pady=10)

# Array size input
Label(frame, text="Array Size:").grid(row=0, column=0, padx=5)
size_entry = Entry(frame, width=10)
size_entry.grid(row=0, column=1, padx=5)
Button(frame, text="Generate Random Array", command=generate_array).grid(row=0, column=2, padx=5)

# Custom array input
Label(frame, text="Custom Array (comma-separated):").grid(row=1, column=0, padx=5)
custom_entry = Entry(frame, width=30)
custom_entry.grid(row=1, column=1, columnspan=2, padx=5)
Button(frame, text="Use Custom Array", command=use_custom_array).grid(row=1, column=3, padx=5)

Label(frame, text="Target Value (for Linear Search):").grid(row=2, column=0, padx=5)
target_entry = Entry(frame, width=10)
target_entry.grid(row=2, column=1, padx=5)

# Array display
array_display = Text(root, height=2, width=50)
array_display.pack(pady=5)

# Algorithm selection
algo_frame = Frame(root)
algo_frame.pack(pady=5)
Label(algo_frame, text="Select Algorithms:").pack()

# Include all algorithms in the selection list.
for algo in ['Bubble Sort', 'Merge Sort', 'Quick Sort', 'Linear Search', 'LSD Radix Sort', 'MSD Radix Sort']:
    var = BooleanVar()
    algo_vars[algo] = var
    Checkbutton(algo_frame, text=algo, variable=var).pack()

# Select and Deselect button
button_frame = Frame(root)
button_frame.pack(pady=5)
Button(button_frame, text="Select All", command=select_all).pack(side=LEFT, padx=5)
Button(button_frame, text="Deselect All", command=deselect_all).pack(side=LEFT, padx=5)

# Start button
Button(root, text="Start", command=start_sorting, width=20, height=2).pack(pady=10)

if __name__ == "__main__":
    root.mainloop()


