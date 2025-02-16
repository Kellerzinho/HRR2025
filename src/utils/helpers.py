def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))

def distance(a, b):
    # Distância euclidiana 2D
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5
