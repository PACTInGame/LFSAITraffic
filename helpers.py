import math

import psutil


def is_lfs_running():
    for proc in psutil.process_iter():
        try:
            if proc.name() == "LFS.exe":
                print("LFS.exe seems to be running. Starting!\n\n")

                return True

        except psutil.AccessDenied:
            print(
                "It seems like you do not have sufficient permissions to check for System Apps. Cannot automatically detect if LFS is running!")
            return True

    return False


def is_spotify_running():
    for proc in psutil.process_iter():
        try:
            if proc.name() == "Spotify.exe":
                return True

        except psutil.AccessDenied:
            print(
                "It seems like you do not have sufficient permissions to check for System Apps. Cannot automatically detect if LFS is running!")
            return True

    return False


def calc_polygon_points(own_x, own_y, length, angle):
    # Calculate the coordinates of a point at a certain distance and angle from a given point.
    return own_x + length * math.cos(math.radians(angle)), own_y + length * math.sin(math.radians(angle))


def point_in_rectangle(point_x, point_y, rect_corners):
    """
    Check if a point is inside a rectangle using the cross product method.
    This works for any rectangle orientation (rotated rectangles).

    Args:
        point_x, point_y: Coordinates of the point to check
        rect_corners: List of 4 tuples [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                     representing rectangle corners in order

    Returns:
        bool: True if point is inside rectangle, False otherwise
    """

    def cross_product(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def point_in_triangle(p, a, b, c):
        # Check if point p is inside triangle abc using cross products
        cp1 = cross_product(a, b, p)
        cp2 = cross_product(b, c, p)
        cp3 = cross_product(c, a, p)

        return (cp1 >= 0 and cp2 >= 0 and cp3 >= 0) or (cp1 <= 0 and cp2 <= 0 and cp3 <= 0)

    # Split rectangle into two triangles and check if point is in either
    p = (point_x, point_y)
    x1, y1 = rect_corners[0]
    x2, y2 = rect_corners[1]
    x3, y3 = rect_corners[2]
    x4, y4 = rect_corners[3]

    # Triangle 1: corners 0, 1, 2
    # Triangle 2: corners 0, 2, 3
    return (point_in_triangle(p, rect_corners[0], rect_corners[1], rect_corners[2]) or
            point_in_triangle(p, rect_corners[0], rect_corners[2], rect_corners[3]))
