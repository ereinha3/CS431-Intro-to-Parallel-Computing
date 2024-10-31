import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Set the base directory containing thread subdirectories
base_dir = "."
output_dir = "./graphs/"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Regular expressions to match data in the output files
element_pattern = re.compile(r'using (\d+) elements')
timing_patterns = {
    'O(N-1)': re.compile(r'Time to do O\(N-1\) prefix sum on a \d+ elements: ([\d.]+) \(s\)'),
    'O(NlogN)': re.compile(r'Time to do O\(NlogN\) //prefix sum on a \d+ elements: ([\d.]+) \(s\)'),
    '2(N-1)': re.compile(r'Time to do 2\(N-1\) //prefix sum on a \d+ elements: ([\d.]+) \(s\)')
}

# List to store all parsed data
data = {}

# Traverse thread subdirectories
for thread_dir in os.listdir(base_dir):
    if thread_dir.startswith("thread"):
        thread_path = os.path.join(base_dir, thread_dir)
        thread_num = int(thread_dir.replace("thread", ""))

        data[thread_num] = {}

        tempObj = {}

        # Traverse files within each thread directory
        for filename in os.listdir(thread_path):
            if filename.endswith(".out"):
                file_path = os.path.join(thread_path, filename)
                index = int(filename.split('_')[1].replace(".out", ""))

                # Calculate elements processed based on index
                elements = 2 ** (18 + 2 * index)

                # Variables to store extracted data for each file
                times = {"thread": thread_num, "index": index, "elements": elements}
                
                triple = []
                # Parse each file
                with open(file_path, 'r') as file:
                    for line in file:
                        for key, pattern in timing_patterns.items():
                            if time_match := pattern.search(line):
                                triple.append(float(time_match.group(1)))

                if index in tempObj.keys():
                    tempObj[index][0].append(triple)
                    tempObj[index][1] += 1
                else:
                    tempObj[index] = [[triple], 1]

        for key in tempObj.keys():
            sumList = list(sum(values) for values in zip(*tempObj[key][0]))
            sumList = [x / tempObj[key][1] for x in sumList]
            data[thread_num][key] = sumList  # Store average times


# Graph 1: thread count vs completion time, holding elements at index 1
threads_index1 = list(data.keys())
threads_index1.sort()
times_O_N1_index1 = [data[thread][1][0] for thread in threads_index1]  # O(N-1)
times_O_NlogN_index1 = [data[thread][1][1] for thread in threads_index1]  # O(NlogN)
times_2_N1_index1 = [data[thread][1][2] for thread in threads_index1]  # 2(N-1)

plt.figure()
plt.plot(threads_index1, times_O_N1_index1, label='O(N-1)', marker='o')
plt.plot(threads_index1, times_O_NlogN_index1, label='O(NlogN)', marker='o')
plt.plot(threads_index1, times_2_N1_index1, label='2(N-1)', marker='o')
plt.xlabel("Thread Count")
plt.ylabel("Completion Time (s)")
plt.title("Thread Count vs Completion Time (Elements at Index 1)")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_dir, "thread_count_vs_completion_time_index1.png"))
plt.close()  # Close the plot to free memory

# Graph 2: thread count vs completion time, holding elements at index 4
times_O_N1_index4 = [data[thread][4][0] for thread in threads_index1]  # O(N-1)
times_O_NlogN_index4 = [data[thread][4][1] for thread in threads_index1]  # O(NlogN)
times_2_N1_index4 = [data[thread][4][2] for thread in threads_index1]  # 2(N-1)

plt.figure()
plt.plot(threads_index1, times_O_N1_index4, label='O(N-1)', marker='o')
plt.plot(threads_index1, times_O_NlogN_index4, label='O(NlogN)', marker='o')
plt.plot(threads_index1, times_2_N1_index4, label='2(N-1)', marker='o')
plt.xlabel("Thread Count")
plt.ylabel("Completion Time (s)")
plt.title("Thread Count vs Completion Time (Elements at Index 4)")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_dir, "thread_count_vs_completion_time_index4.png"))
plt.close()

# Graph 3: #items processed vs processing time, holding thread count at 21
thread21_data = data[21]
elements_thread21 = [key for key in thread21_data.keys()]
elements_thread21.sort()
times_O_N1_thread21 = [thread21_data[key][0] for key in elements_thread21]  # O(N-1)
times_O_NlogN_thread21 = [thread21_data[key][1] for key in elements_thread21]  # O(NlogN)
times_2_N1_thread21 = [thread21_data[key][2] for key in elements_thread21]  # 2(N-1)

items21 = [2**(16+key*2) for key in elements_thread21]
items21.sort()

plt.figure()
plt.plot(items21, times_O_N1_thread21, label='O(N-1)', marker='o')
plt.plot(items21, times_O_NlogN_thread21, label='O(NlogN)', marker='o')
plt.plot(items21, times_2_N1_thread21, label='2(N-1)', marker='o')
plt.xlabel("# Items Processed")
plt.ylabel("Completion Time (s)")
plt.xscale('log')  # Log scale for clarity
plt.title("Items Processed vs Completion Time (Thread Count 21)")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_dir, "items_processed_vs_completion_time_thread21.png"))
plt.close()

# Graph 4: #items processed vs processing time, holding thread count at 14
thread14_data = data[14]
elements_thread14 = [key for key in thread14_data.keys()]
elements_thread14.sort()
times_O_N1_thread14 = [thread14_data[key][0] for key in elements_thread14]  # O(N-1)
times_O_NlogN_thread14 = [thread14_data[key][1] for key in elements_thread14]  # O(NlogN)
times_2_N1_thread14 = [thread14_data[key][2] for key in elements_thread14]  # 2(N-1)

items14 = [2**(16+key*2) for key in elements_thread14]
items14.sort()

plt.figure()
plt.plot(items14, times_O_N1_thread14, label='O(N-1)', marker='o')
plt.plot(items14, times_O_NlogN_thread14, label='O(NlogN)', marker='o')
plt.plot(items14, times_2_N1_thread14, label='2(N-1)', marker='o')
plt.xlabel("# Items Processed")
plt.ylabel("Completion Time (s)")
plt.xscale('log')  # Log scale for clarity
plt.title("Items Processed vs Completion Time (Thread Count 14)")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_dir, "items_processed_vs_completion_time_thread14.png"))
plt.close()