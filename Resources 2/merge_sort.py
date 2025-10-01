def max_area(height):
    max_area = 0
    i = 0
    j = 0
    for i in range(len(height)-1):
        for j in range(len(height)-1):
            lowest_height = 0
            if height[i] > height[j]:
                lowest_height = height[j]
            else:
                lowest_height = height[i]
            lenght = j - i
            area = lowest_height*lenght
            if max_area < area:
                max_area = area
    return area


print(max_area([1, 8, 6, 2, 5, 4, 8, 3, 7]))