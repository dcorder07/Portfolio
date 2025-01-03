import csv

Ages = []

file_path = 'C:/Users/dcord/Downloads/train.csv'

with open(file_path, 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    # get the index of the column you want to extract
    column_index = 5
    # loop through the rows and extract the column value
    for row in reader:
        Age = row[column_index]
        if Age != '' and Age != 'Age':
            #print(column_value)
            Ages.append(Age)

print("Total No. of valid Age values = ", len(Ages))

#---------------Merge------------------
def merge_sort(arr):
    if len(arr) > 1:
        middle = len(arr) // 2
        lhalf = arr[:middle]
        rhalf = arr[middle:]
        merge_sort(lhalf)
        merge_sort(rhalf)
        i = 0
        j = 0
        k = 0
        while i < len(lhalf) and j < len(rhalf):
            if lhalf[i] < rhalf[j]:
                arr[k] = lhalf[i]
                i = i + 1
            else:
                arr[k] = rhalf[j]
                j = j + 1
            k = k + 1

        while i < len(lhalf):
            arr[k] = lhalf[i]
            i = i + 1
            k = k + 1

        while j < len(rhalf):
            arr[k] = rhalf[j]
            j = j + 1
            k = k + 1
    return arr

#---------------Bubble------------------
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr[:20]

#---------------Insertio------------------
def insertion_sort(arr):
    n = len(arr)
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j+1] = arr[j]
            j -= 1
        arr[j+1] = key
    return arr[:20]

#---------------Selection------------------
def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr[:20]

#---------------Quick------------------
def quick_sort(arr):
    n = len(arr)
    if n <= 1:
        return arr
    pivot = arr[0]
    left = []
    right = []
    for i in range(1, n):
        if arr[i] < pivot:
            left.append(arr[i])
        else:
            right.append(arr[i])
    left = quick_sort(left)
    right = quick_sort(right)
    return left[:20] + [pivot] + right[:20]

#---------------Printing----------------

print("Merge Sort: ", merge_sort(Ages[:20]))
print("Bubble Sort: ", bubble_sort(Ages))
print("Insertion Sort: ", insertion_sort(Ages))
print("Selection Sort: ", selection_sort(Ages))
print("Quick Sort: ", quick_sort(Ages))
