def quick_sort(lst):
    pivot = lst.pop(len(lst)//2)
    if len(lst) <= 1:
        return lst
    else:
        left, right = [], []
        for i in lst:
            if i > pivot:
                right.append(i)
            else:
                left.append(i)
        return quick_sort(left) + [pivot] + quick_sort(right)


def merge_sort(left, right):
    new_list = []
    if len(left) == 1 and len(right) == 1:
        if left[0] < right[0]:
            new_list.append(left[0])
            new_list.append(right[0])
        else:
            new_list.append(right[0])
            new_list.append(left[0])
    else:
        while len(left) >= 1 and len(right) >= 1:
            if left[0] <= right[0]:
                new_list.append(left.pop(0))
            else:
                new_list.append(right.pop(0))

        new_list = new_list + left
        new_list = new_list + right
    return new_list


def merge(lst):
    if len(lst) == 1:
        return lst

    left = lst[:len(lst)//2]
    right = lst[len(lst)//2:]
    left_sorted = merge(left)
    right_sorted = merge(right)

    return merge_sort(left_sorted, right_sorted)
