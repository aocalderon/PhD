import math

def calculateDisks(p1_x,p1_y,p2_x,p2_y):

    r2 = math.pow(epsilon/2,2)
    disks = []
    
    X = p1_x - p2_x
    Y = p1_y - p2_y
    D2 = math.pow(X, 2) + math.pow(Y, 2)
    
    if (D2 == 0):
        return []

    expression = abs(4 * (r2 / D2) - 1)
    root = math.pow(expression, 0.5)
    h_1 = ((X + Y * root) / 2) + p2_x
    h_2 = ((X - Y * root) / 2) + p2_x
    k_1 = ((Y - X * root) / 2) + p2_y
    k_2 = ((Y + X * root) / 2) + p2_y
    
    return [h_1, k_1, h_2, k_2]

epsilon = 2
print calculateDisks(1,2,3,4)
