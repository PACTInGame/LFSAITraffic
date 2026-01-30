import json
import numpy as np
import matplotlib

# Set backend to TkAgg as requested for PyCharm compatibility
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.spatial import KDTree, distance_matrix


class MapGenerator:
    def __init__(self, raw_objects):
        """
        raw_objects: List of tuples (index, x, y, z)
        """
        self.raw_objects = raw_objects
        self.roads = {}  # {road_index: [ordered (x,y,z) points]}
        self.closed_roads = set()  # Stores indices of roads that are loops
        self.junctions = []
        self.junction_radius = 3.0  # meters
        self.loop_threshold = 50.0  # meters - Distance to auto-close loop

    def process(self):
        print("1. Grouping and ordering road points...")
        self._build_ordered_roads()

        print("2. Checking for closed loops...")
        self._handle_closed_loops()

        print("3. Detecting junctions in 3D space...")
        self._find_junctions()

        return self.roads, self.junctions

    def _build_ordered_roads(self):
        """
        Groups points by index and sorts them using a Nearest Neighbor approach,
        then corrects for 'rotated' loops by cutting at the largest physical gap.
        """
        # 1. Group by Index
        grouped = {}
        for idx, x, y, z in self.raw_objects:
            if idx not in grouped:
                grouped[idx] = []
            grouped[idx].append([x, y, z])

        # 2. Sort points for each road
        for idx, points in grouped.items():
            points_np = np.array(points)

            # If only 1 or 2 points, order is trivial
            if len(points_np) <= 2:
                self.roads[idx] = points_np.tolist()
                continue

            # --- STEP A: Initial Greedy Sort (Existing Logic) ---
            d_mat = distance_matrix(points_np, points_np)
            # Start from one of the furthest points
            i, _ = np.unravel_index(d_mat.argmax(), d_mat.shape)

            start_idx = i
            ordered_indices = [start_idx]
            remaining_indices = set(range(len(points_np)))
            remaining_indices.remove(start_idx)

            curr_idx = start_idx

            while remaining_indices:
                dists = d_mat[curr_idx]
                min_dist = float('inf')
                next_idx = None

                for candidate in remaining_indices:
                    if dists[candidate] < min_dist:
                        min_dist = dists[candidate]
                        next_idx = candidate

                if next_idx is not None:
                    ordered_indices.append(next_idx)
                    remaining_indices.remove(next_idx)
                    curr_idx = next_idx
                else:
                    break

            sorted_points = points_np[ordered_indices]

            # --- STEP B: The Fix (Gap Correction) ---
            # Calculate distances between adjacent points in the sorted list
            # shape: (N-1,)
            internal_dists = np.linalg.norm(sorted_points[:-1] - sorted_points[1:], axis=1)

            # Calculate the "wrap-around" distance (Last point -> First point)
            wrap_dist = np.linalg.norm(sorted_points[-1] - sorted_points[0])

            # Find the largest gap inside the current path
            max_internal_gap = internal_dists.max()
            max_internal_idx = internal_dists.argmax()

            # Logic: If the largest physical gap is inside the list (and not the wrap-around),
            # then the Greedy algorithm mistakenly bridged the Start-End gap.
            # We must cut the list at that internal gap.
            if max_internal_gap > wrap_dist:
                # The gap is between index `max_internal_idx` and `max_internal_idx + 1`
                # We roll the array so `max_internal_idx + 1` becomes the new index 0
                shift_amount = -(max_internal_idx + 1)
                sorted_points = np.roll(sorted_points, shift_amount, axis=0)
                print(f"Fixed rotated road {idx}: cut at gap {max_internal_gap:.2f}m")

            self.roads[idx] = sorted_points.tolist()

    def _handle_closed_loops(self):
        """
        Checks if start and end points are within threshold.
        If so, physically closes the loop and marks it.
        """
        for r_idx, points in self.roads.items():
            # Need at least 3 points to form a meaningful loop
            if len(points) < 3:
                continue

            start_pt = np.array(points[0])
            end_pt = np.array(points[-1])

            # Calculate Euclidean distance between start and end
            dist = np.linalg.norm(start_pt - end_pt)
            # check that the longest gap in the road is not larger than the loop threshold (in case start and end are detected wrong)
            for i in range(len(points)-1):
                seg_dist = np.linalg.norm(np.array(points[i]) - np.array(points[i+1]))
                if seg_dist > dist:
                    dist = seg_dist

            if dist < self.loop_threshold:
                print(f"-> Road {r_idx} detected as Loop (Gap: {dist:.2f}m). Closing loop.")

                # 1. Mark as closed
                self.closed_roads.add(r_idx)

                # 2. Add start point to the end to close geometry
                # We append the list version of the point, not the numpy array
                self.roads[r_idx].append(points[0])

    def _find_junctions(self):
        """
        Uses a KDTree to find points from DIFFERENT road indices
        that are within 3 meters of each other.
        """
        # Flatten data for KDTree: [x, y, z, road_index]
        all_points = []
        for r_idx, pts in self.roads.items():
            for p in pts:
                all_points.append(p + [r_idx])

        data = np.array(all_points)
        if len(data) == 0:
            return

        coords = data[:, :3]  # x, y, z
        road_ids = data[:, 3]  # indices

        tree = KDTree(coords)
        pairs = tree.query_pairs(r=self.junction_radius)

        found_junctions = {}

        for i, j in pairs:
            id_a = road_ids[i]
            id_b = road_ids[j]

            if id_a != id_b:
                loc = (coords[i] + coords[j]) / 2.0
                conn_key = frozenset([int(id_a), int(id_b)])

                if conn_key not in found_junctions:
                    found_junctions[conn_key] = {
                        "location": loc.tolist(),
                        "connected_roads": list(conn_key)
                    }

        self.junctions = list(found_junctions.values())

    def save_to_json(self, filename="track_data.json"):
        output = {
            "metadata": {
                "unit": "meters",
                "coordinate_system": "x, y, z"
            },
            "roads": [],
            "junctions": self.junctions
        }

        for r_idx, points in self.roads.items():
            # Check if this index is in our set of closed loops
            is_closed = r_idx in self.closed_roads

            output["roads"].append({
                "road_id": int(r_idx),
                "point_count": len(points),
                "closed_loop": is_closed,  # <--- New Flag
                "path": points
            })

        with open(filename, 'w') as f:
            json.dump(output, f, indent=4)
        print(f"Map saved to {filename}")

    def debug_plot(self):
        print("Opening interactive plot...")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot Roads
        colors = matplotlib.cm.tab20(np.linspace(0, 1, len(self.roads)))
        for (r_idx, points), color in zip(self.roads.items(), colors):
            pts = np.array(points)

            # Label style changes if it's a loop
            label = f"Road {r_idx}"
            if r_idx in self.closed_roads:
                label += " (Loop)"

            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], c=color, label=label)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color, s=5)

            # Label the start
            ax.text(pts[0, 0], pts[0, 1], pts[0, 2], f"Start {r_idx}", size=8)

        # Plot Junctions
        for j in self.junctions:
            loc = j['location']
            ax.scatter(loc[0], loc[1], loc[2], c='red', s=100, marker='X')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')
        plt.title("Track Debug Map")
        plt.tight_layout()
        plt.show()


# --- Example Usage ---
if __name__ == "__main__":
    # 1. Standard Line
    road_1 = [(20, x, 0, 0) for x in [0, 10, 2, 8, 4, 6]]

    # 2. A Loop (Circle-ish)
    # Points roughly in a circle, start (0,0) and end (1,1) are close
    theta = np.linspace(0, 2 * np.pi - 0.2, 10)  # Stop short of full circle
    r = 100
    road_loop_pts = []
    for i, t in enumerate(theta):
        x = r * np.cos(t)
        y = r * np.sin(t)
        road_loop_pts.append((99, x, y, 0))

    # 3. Mock Data assembly
    input_data = road_1 + road_loop_pts

    # Run Generator
    gen = MapGenerator(input_data)
    gen.process()
    gen.save_to_json()
    gen.debug_plot()