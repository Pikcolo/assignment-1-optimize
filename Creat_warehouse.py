"""
=============================================================================
 Assignment 1 : หาเส้นทางหยิบสินค้าที่สั้นที่สุดในโกดัง
 (Warehouse Order-Picking Shortest Route)
=============================================================================
 โจทย์
   - โกดังเป็นตาราง 2 มิติ  0 = ทางเดิน (เดินได้) , 1 = ชั้นวาง (เดินไม่ได้)
   - เริ่มที่ entrance -> หยิบสินค้าให้ครบทุกจุด -> จบที่ exit_point
   - เดินได้ 4 ทิศทาง (บน/ล่าง/ซ้าย/ขวา) เท่านั้น
   - จุดหยิบสินค้าอยู่บน "ชั้นวาง" พนักงานเพียงเดินไปยืนบน "ช่องทางเดินที่ติดกับ
     ชั้นวางนั้น" ก็ถือว่าหยิบของได้
   - ไม่ต้องหยิบเรียงลำดับ ขอแค่ครบทุกจุด

 โครงสร้างไฟล์
   ส่วนที่ 1-6   : แผนที่โกดัง / สุ่มจุดหยิบ / วาดแผนที่  (ของเดิมจากโจทย์)
   ส่วนที่ 7-9   : สร้างแบบจำลองกราฟ (BFS ระยะทางจริงในเขาวงกต)
   ส่วนที่ 10    : ตัวถอดรหัส "ลำดับการหยิบ -> เส้นทางจริง" ด้วย DP
   ส่วนที่ 11-15 : วิธี optimize 5 แบบ + ตัวตรวจคำตอบที่เหมาะที่สุดแบบ exact
   ส่วนที่ 16-18 : รันเปรียบเทียบ / แสดงผล / วาดเส้นทาง / ทดลองขยาย n
=============================================================================
"""

import os
import time
import random
import heapq
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# ตั้งเป็น 0 เพื่อไม่ให้หน้าต่างกราฟเด้งขึ้นมา (ยังเซฟไฟล์รูปให้เหมือนเดิม)
SHOW_PLOTS = os.environ.get("WH_SHOW_PLOTS", "1") == "1"
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

INF = float("inf")

# =========================================================
# 1) กำหนดแผนที่โกดัง
#    0 = ทางเดิน
#    1 = ชั้นวางสินค้า / สิ่งกีดขวาง
# =========================================================
warehouse = np.array([
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]])

# =========================================================
# 2) การเคลื่อนที่แบบ Manhattan: บน ล่าง ซ้าย ขวา
# =========================================================
movements = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1)
]


# =========================================================
# 3) ตรวจสอบว่าพิกัดอยู่ภายในแผนที่หรือไม่
# =========================================================
def is_valid_position(warehouse, position):
    row, col = position
    rows, cols = warehouse.shape
    return 0 <= row < rows and 0 <= col < cols


def is_walkable(warehouse, position):
    """เป็นช่องที่เดินได้จริงหรือไม่ (อยู่ในแผนที่ และเป็นทางเดิน 0)"""
    return is_valid_position(warehouse, position) and warehouse[position] == 0


# =========================================================
# 4) ตรวจสอบประเภทของจุด
# =========================================================
def validate_points(warehouse, entrance, exit_point, pickup_points):
    # ทางเข้าและทางออกต้องอยู่บนทางเดิน
    if warehouse[entrance] != 0:
        raise ValueError("Entrance ต้องอยู่บนทางเดินที่มีค่า 0")

    if warehouse[exit_point] != 0:
        raise ValueError("Exit ต้องอยู่บนทางเดินที่มีค่า 0")

    # จุดหยิบสินค้าทุกจุดต้องอยู่บนชั้นวาง
    for point in pickup_points:
        if warehouse[point] != 1:
            raise ValueError(
                f"Pickup point {point} ต้องอยู่บนชั้นวางที่มีค่า 1"
            )


# =========================================================
# 5) สุ่มจุดหยิบสินค้าจากตำแหน่งชั้นวาง
# =========================================================
def generate_pickup_points(warehouse, n, seed=42):
    # หาพิกัดทุกจุดที่เป็นชั้นวาง (ค่า 1)
    shelf_positions = list(zip(*np.where(warehouse == 1)))
    if n > len(shelf_positions):
        raise ValueError(
            f"ไม่สามารถสร้างจุดหยิบสินค้า {n} จุดได้ "
            f"เพราะมีชั้นวางทั้งหมดเพียง {len(shelf_positions)} จุด"
        )
    # ตั้งค่า seed เพื่อให้ผลการสุ่มเหมือนเดิมทุกครั้ง
    rng = np.random.default_rng(seed)

    # เลือกตำแหน่งชั้นวางแบบไม่ซ้ำกัน
    selected_indices = rng.choice(
        len(shelf_positions),
        size=n,
        replace=False
    )

    pickup_points = [
        tuple(int(v) for v in shelf_positions[index])
        for index in selected_indices
    ]
    return pickup_points


def generate_pickup_points2(warehouse, n):
    # หาพิกัดทุกจุดที่เป็นชั้นวาง (ค่า 1)
    shelf_positions = list(zip(*np.where(warehouse == 1)))

    if n > len(shelf_positions):
        raise ValueError(
            f"ไม่สามารถสร้างจุดหยิบสินค้า {n} จุดได้ "
            f"เพราะมีชั้นวางทั้งหมดเพียง {len(shelf_positions)} จุด"
        )

    # ใช้เวลาปัจจุบันเป็น seed
    seed = int(time.time())

    # สร้าง random generator
    rng = np.random.default_rng(seed)

    # เลือกตำแหน่งชั้นวางแบบไม่ซ้ำกัน
    selected_indices = rng.choice(
        len(shelf_positions),
        size=n,
        replace=False
    )

    pickup_points = [
        tuple(int(v) for v in shelf_positions[index])
        for index in selected_indices
    ]

    return pickup_points


# =========================================================
# 6) แสดงแผนที่ พร้อมทางเข้า ทางออก และจุดหยิบสินค้า
# =========================================================
def _draw_grid(warehouse, title):
    """วาดพื้นหลังโกดัง (ใช้ร่วมกันระหว่างรูปแผนที่กับรูปเส้นทาง)"""
    rows, cols = warehouse.shape
    colors = [
        "#EAF6FF",  # ฟ้าอ่อน = ทางเดิน
        "#4A4A4A"   # เทาเข้ม = ชั้นวาง
    ]
    warehouse_cmap = ListedColormap(colors)

    plt.figure(figsize=(15, 10))
    plt.imshow(
        warehouse,
        cmap=warehouse_cmap,
        origin="upper",
        interpolation="nearest"
    )

    plt.xticks(np.arange(cols))
    plt.yticks(np.arange(rows))
    plt.xticks(np.arange(-0.5, cols, 1), minor=True)
    plt.yticks(np.arange(-0.5, rows, 1), minor=True)
    plt.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    plt.tick_params(which="minor", bottom=False, left=False)

    plt.title(title, fontsize=15, fontweight="bold")
    plt.xlabel("Column")
    plt.ylabel("Row")


def _draw_terminals(entrance, exit_point):
    plt.scatter(entrance[1], entrance[0], color="#27AE60", s=220, marker="o",
                edgecolors="black", linewidths=1.2, zorder=5, label="Entrance")
    plt.text(entrance[1], entrance[0], "IN", ha="center", va="center",
             fontsize=8, fontweight="bold", color="white", zorder=6)

    plt.scatter(exit_point[1], exit_point[0], color="#E74C3C", s=220, marker="o",
                edgecolors="black", linewidths=1.2, zorder=5, label="Exit")
    plt.text(exit_point[1], exit_point[0], "OUT", ha="center", va="center",
             fontsize=7, fontweight="bold", color="white", zorder=6)


def _legend(extra=None):
    legend_elements = [
        Patch(facecolor="#EAF6FF", edgecolor="black", label="Walkway (0)"),
        Patch(facecolor="#4A4A4A", edgecolor="black", label="Shelf (1)"),
        plt.Line2D([0], [0], marker="o", color="w", label="Entrance",
                   markerfacecolor="#27AE60", markeredgecolor="black", markersize=10),
        plt.Line2D([0], [0], marker="o", color="w", label="Exit",
                   markerfacecolor="#E74C3C", markeredgecolor="black", markersize=10),
        plt.Line2D([0], [0], marker="s", color="w", label="Pickup Point",
                   markerfacecolor="#F1C40F", markeredgecolor="black", markersize=10),
    ]
    if extra:
        legend_elements += extra
    plt.legend(handles=legend_elements, loc="upper center",
               bbox_to_anchor=(0.5, -0.08), ncol=len(legend_elements))


def _finish(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, filename), dpi=120, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def plot_warehouse(warehouse, entrance, exit_point, pickup_points):
    _draw_grid(warehouse, "Warehouse Map with Entrance, Exit and Pickup Points")
    _draw_terminals(entrance, exit_point)

    # แสดงจุดหยิบสินค้า
    for index, point in enumerate(pickup_points, start=1):
        plt.scatter(point[1], point[0], color="#F1C40F", s=180, marker="s",
                    edgecolors="black", linewidths=1.0, zorder=3)
        plt.text(point[1], point[0], str(index), ha="center", va="center",
                 fontsize=9, fontweight="bold", color="black", zorder=4)

    _legend()
    _finish("warehouse_map.png")


def plot_solution(warehouse, entrance, exit_point, pickup_points,
                  path, order, stops, title, filename):
    """
    วาดเส้นทางคำตอบ
      path  : list ของช่องทางเดินทั้งหมดตั้งแต่ IN ถึง OUT
      order : ลำดับ index ของ pickup ที่ "ตั้งใจ" เดินไปหยิบ
      stops : ช่องทางเดินที่ใช้ยืนหยิบของแต่ละจุดใน order
    """
    _draw_grid(warehouse, title)

    ys = [c[0] for c in path]
    xs = [c[1] for c in path]
    plt.plot(xs, ys, color="#2980B9", linewidth=2.6, alpha=0.9, zorder=2)
    plt.scatter(xs, ys, color="#2980B9", s=14, zorder=2)

    # จุดหยิบสินค้าทั้งหมด (สีเหลือง = เลขจุดเดิมจากโจทย์)
    for index, point in enumerate(pickup_points, start=1):
        plt.scatter(point[1], point[0], color="#F1C40F", s=180, marker="s",
                    edgecolors="black", linewidths=1.0, zorder=3)
        plt.text(point[1], point[0], str(index), ha="center", va="center",
                 fontsize=9, fontweight="bold", color="black", zorder=4)

    # ช่องที่ยืนหยิบของ พร้อมลำดับการหยิบ 1,2,3,...
    for step, (p_idx, cell) in enumerate(zip(order, stops), start=1):
        plt.scatter(cell[1], cell[0], color="#8E44AD", s=150, marker="D",
                    edgecolors="white", linewidths=1.0, zorder=4)
        plt.text(cell[1], cell[0], str(step), ha="center", va="center",
                 fontsize=8, fontweight="bold", color="white", zorder=5)

    _draw_terminals(entrance, exit_point)
    _legend(extra=[
        plt.Line2D([0], [0], color="#2980B9", lw=3, label="Route"),
        plt.Line2D([0], [0], marker="D", color="w", label="Pick position (order)",
                   markerfacecolor="#8E44AD", markeredgecolor="black", markersize=9),
    ])
    _finish(filename)


# =========================================================
# 7) แบบจำลองปัญหาเป็นกราฟ  (Topic 08 : Graph Optimization)
# ---------------------------------------------------------
#  แนวคิด
#   - พนักงานเดินบน "ช่องทางเดิน" เท่านั้น -> มองแผนที่เป็นกราฟที่แต่ละช่อง 0
#     เป็น node และเชื่อมกับเพื่อนบ้าน 4 ทิศด้วย edge น้ำหนัก 1
#   - จุดหยิบสินค้า i ไม่ใช่ node เดียว แต่เป็น "กลุ่มของช่องทางเดินที่ติดชั้นวาง"
#     เรียกว่า access cells ของจุดนั้น  (ยืนช่องไหนก็ได้ในกลุ่มนี้ = หยิบได้)
#   - ปัญหาจึงกลายเป็น Generalized TSP-Path :
#       เริ่ม entrance -> แตะกลุ่ม access ของทุกจุดหยิบอย่างน้อยกลุ่มละ 1 ช่อง
#       -> จบที่ exit  โดยระยะทางระหว่างช่องคือ shortest path จริงในเขาวงกต
#   - น้ำหนัก edge เท่ากันหมด (=1) จึงใช้ BFS แทน Dijkstra ได้ และได้คำตอบ
#     เท่ากันแต่เร็วกว่า (Dijkstra ที่ n้ำหนักเท่ากัน degenerate เป็น BFS)
# =========================================================
def get_access_cells(warehouse, shelf_point):
    """ช่องทางเดินที่ติดกับชั้นวาง shelf_point ในทิศ บน/ล่าง/ซ้าย/ขวา"""
    cells = []
    for dr, dc in movements:
        neighbour = (shelf_point[0] + dr, shelf_point[1] + dc)
        if is_walkable(warehouse, neighbour):
            cells.append(neighbour)
    return cells


def bfs_shortest_paths(warehouse, source):
    """
    BFS จาก source ไปทุกช่องทางเดิน
    คืน (dist, parent) : dist[cell] = จำนวนก้าวน้อยที่สุด, parent ใช้ย้อนรอยเส้นทาง
    """
    dist = {source: 0}
    parent = {source: None}
    queue = deque([source])
    while queue:
        cell = queue.popleft()
        for dr, dc in movements:
            nxt = (cell[0] + dr, cell[1] + dc)
            if nxt not in dist and is_walkable(warehouse, nxt):
                dist[nxt] = dist[cell] + 1
                parent[nxt] = cell
                queue.append(nxt)
    return dist, parent


def astar_shortest_path(warehouse, source, target):
    """
    A* (Topic 08.2) — ใช้ Manhattan distance เป็น heuristic ซึ่ง admissible
    เพราะเดินได้แค่ 4 ทิศและทุกก้าวมีค่า 1
    ในโปรแกรมนี้ใช้เพื่อ "ยืนยัน" ว่า BFS ให้ระยะทางเท่ากัน (ดูส่วนที่ 16)
    """
    def h(c):
        return abs(c[0] - target[0]) + abs(c[1] - target[1])

    open_heap = [(h(source), 0, source)]
    g = {source: 0}
    parent = {source: None}
    while open_heap:
        _, gc, cell = heapq.heappop(open_heap)
        if cell == target:
            break
        if gc > g.get(cell, INF):
            continue
        for dr, dc in movements:
            nxt = (cell[0] + dr, cell[1] + dc)
            if not is_walkable(warehouse, nxt):
                continue
            ng = gc + 1
            if ng < g.get(nxt, INF):
                g[nxt] = ng
                parent[nxt] = cell
                heapq.heappush(open_heap, (ng + h(nxt), ng, nxt))
    return g.get(target, INF)


# =========================================================
# 8) คลาสเก็บแบบจำลองปัญหา + ตารางระยะทางระหว่าง node สำคัญ
# =========================================================
class WarehouseProblem:
    """
    node id ที่ใช้ภายใน (เป็น int เพื่อให้คำนวณเร็ว)
        0            = entrance
        1            = exit_point
        2, 3, 4, ... = access cell ต่าง ๆ (ไม่ซ้ำกัน)
    groups[i] = list ของ node id ที่เป็น access cell ของจุดหยิบที่ i
    """

    START = 0
    END = 1

    def __init__(self, warehouse, entrance, exit_point, pickup_points):
        self.warehouse = warehouse
        self.entrance = entrance
        self.exit_point = exit_point
        self.pickups = list(pickup_points)
        self.n = len(self.pickups)

        # ---- 8.1 หา access cells ของทุกจุดหยิบ ----
        access = [get_access_cells(warehouse, p) for p in self.pickups]
        for i, cells in enumerate(access):
            if not cells:
                raise ValueError(
                    f"จุดหยิบสินค้า {self.pickups[i]} ไม่มีช่องทางเดินติดอยู่เลย "
                    f"จึงหยิบไม่ได้"
                )
        self.access_cells = access

        # ---- 8.2 ตั้งรหัส node ----
        self.cells = [entrance, exit_point]
        self.node_of = {entrance: self.START, exit_point: self.END}
        for cells in access:
            for c in cells:
                if c not in self.node_of:
                    self.node_of[c] = len(self.cells)
                    self.cells.append(c)
        self.groups = [[self.node_of[c] for c in cells] for cells in access]

        # ---- 8.3 ช่องไหนหยิบของชิ้นไหนได้บ้าง (bitmask) ----
        # ใช้ตอนตัดจุดที่ "หยิบผ่านทางได้ฟรี" ในส่วนที่ 15
        self.cell_mask = {}
        for i, cells in enumerate(access):
            for c in cells:
                self.cell_mask[c] = self.cell_mask.get(c, 0) | (1 << i)
        self.FULL_MASK = (1 << self.n) - 1

        # ---- 8.4 BFS จากทุก node สำคัญ -> ตารางระยะทาง + ตารางย้อนรอย ----
        m = len(self.cells)
        self.dist = [[INF] * m for _ in range(m)]
        self.parents = []
        self.bfs_count = 0
        for i, src in enumerate(self.cells):
            d, par = bfs_shortest_paths(warehouse, src)
            self.bfs_count += 1
            self.parents.append(par)
            row = self.dist[i]
            for j, dst in enumerate(self.cells):
                if dst in d:
                    row[j] = d[dst]

        if any(self.dist[self.START][g] == INF for grp in self.groups for g in grp):
            raise ValueError("มีจุดหยิบสินค้าที่เดินไปไม่ถึงจากทางเข้า")

        # ---- 8.5 ตารางระยะทางระดับ "กลุ่ม" ใช้กับ NN / ACO ----
        # group_dist[a][b] = ระยะทางสั้นสุดระหว่างกลุ่ม a กับกลุ่ม b
        # index: 0 = entrance, 1..n = จุดหยิบ, n+1 = exit
        k = self.n + 2
        self.group_dist = [[0] * k for _ in range(k)]
        for a in range(k):
            for b in range(k):
                if a == b:
                    continue
                self.group_dist[a][b] = min(
                    self.dist[u][v]
                    for u in self._nodes_of(a)
                    for v in self._nodes_of(b)
                )

    def _nodes_of(self, idx):
        if idx == 0:
            return [self.START]
        if idx == self.n + 1:
            return [self.END]
        return self.groups[idx - 1]

    # ----- ย้อนรอยเส้นทางจริงระหว่าง node สองตัว -----
    def cell_path(self, node_a, node_b):
        par = self.parents[node_a]
        cell_a, cell_b = self.cells[node_a], self.cells[node_b]
        out = [cell_b]
        while out[-1] != cell_a:
            out.append(par[out[-1]])
        out.reverse()
        return out


# =========================================================
# 9) ตัวถอดรหัส : ลำดับการหยิบ  ->  ระยะทางจริง  (DP บน access cells)
# ---------------------------------------------------------
#  ปัญหาย่อย: ถ้า "ลำดับ" การหยิบถูกกำหนดมาแล้ว เรายังต้องเลือกว่าแต่ละจุด
#  จะยืนหยิบที่ช่องไหน (มีได้ถึง 4 ช่อง) ซึ่งเลือกไม่ดีก็เสียก้าวฟรี ๆ
#  แก้ด้วย Dynamic Programming แบบ layered graph (shortest path บน DAG):
#      f(ขั้นที่ t, ช่อง c) = min over ช่อง c' ของขั้น t-1 [ f(t-1,c') + d(c',c) ]
#  ทำงาน O(n * A^2) เมื่อ A <= 4 จึงเร็วมาก และให้คำตอบ "ดีที่สุดของลำดับนั้น"
# =========================================================
def evaluate_order(problem, order):
    """คืน (ระยะทางรวม, ข้อมูลย้อนรอย) ของลำดับการหยิบที่กำหนด"""
    dist = problem.dist
    prev_nodes = [problem.START]
    prev_cost = [0]
    back = []

    for p in order:
        cur_nodes = problem.groups[p]
        cur_cost = []
        cur_back = []
        for v in cur_nodes:
            best, arg = INF, 0
            for k, u in enumerate(prev_nodes):
                val = prev_cost[k] + dist[u][v]
                if val < best:
                    best, arg = val, k
            cur_cost.append(best)
            cur_back.append(arg)
        prev_nodes, prev_cost = cur_nodes, cur_cost
        back.append(cur_back)

    best, arg = INF, 0
    for k, u in enumerate(prev_nodes):
        val = prev_cost[k] + dist[u][problem.END]
        if val < best:
            best, arg = val, k
    return best, (back, arg)


def order_cost(problem, order):
    return evaluate_order(problem, order)[0]


def build_route(problem, order):
    """
    คืน (ระยะทาง, เส้นทางเป็น list ของช่อง, ช่องที่ยืนหยิบของแต่ละจุดใน order)
    """
    total, (back, arg) = evaluate_order(problem, order)
    if total == INF:
        return INF, [], []

    idx = arg
    stop_nodes = []
    for t in range(len(order) - 1, -1, -1):
        stop_nodes.append(problem.groups[order[t]][idx])
        idx = back[t][idx]
    stop_nodes.reverse()

    node_seq = [problem.START] + stop_nodes + [problem.END]
    path = [problem.cells[problem.START]]
    for a, b in zip(node_seq, node_seq[1:]):
        path += problem.cell_path(a, b)[1:]

    stops = [problem.cells[v] for v in stop_nodes]
    return total, path, stops


def collected_mask(problem, path):
    """สินค้าที่หยิบได้จริงจากการเดินตามเส้นทางนี้ (รวมที่หยิบผ่านทางได้ฟรี)"""
    mask = 0
    for cell in path:
        mask |= problem.cell_mask.get(cell, 0)
    return mask


# =========================================================
# 10) วิธีที่ 1 : Nearest Neighbour (Greedy Constructive Heuristic)
# ---------------------------------------------------------
#  หลักการ: อยู่ที่ไหนก็เดินไปหยิบจุดที่ "ใกล้ที่สุด" ที่ยังไม่ได้หยิบ
#  ปรับใช้กับโจทย์: ระยะทางที่ใช้ตัดสินใจไม่ใช่ระยะเส้นตรง แต่เป็นระยะ BFS จริง
#  ระหว่าง "กลุ่ม access cell" จึงไม่โดนกำแพงชั้นวางหลอก
#  ข้อดี เร็วมาก O(n^2) / ข้อเสีย มองสั้น มักทิ้งจุดโดดไว้ท้ายสุด
# =========================================================
def solve_nearest_neighbour(problem, first=None):
    """first = บังคับให้จุดแรกเป็นจุดนี้ (ใช้สร้างประชากรเริ่มต้นแบบหลากหลาย)"""
    k = problem.n + 2
    unvisited = set(range(1, k - 1))
    current = 0
    order = []
    if first is not None:
        current = first + 1
        order.append(first)
        unvisited.discard(current)
    while unvisited:
        nxt = min(unvisited, key=lambda j: problem.group_dist[current][j])
        order.append(nxt - 1)
        unvisited.discard(nxt)
        current = nxt
    return order


# =========================================================
# 11) วิธีที่ 2 : Local Search (2-opt + Or-opt)
# ---------------------------------------------------------
#  2-opt  : กลับด้านลำดับช่วง [i..j]  -> แก้เส้นทางที่ "ไขว้กัน"
#  Or-opt : ย้ายชิ้นส่วนยาว 1-3 จุด ไปแทรกตำแหน่งอื่น (กลับด้านได้)
#  ปรับใช้กับโจทย์: ปกติ 2-opt ของ TSP คำนวณ delta จากระยะ 4 เส้น แต่ที่นี่
#  การกลับด้านทำให้ "ช่องที่ยืนหยิบ" เปลี่ยนไปด้วย จึงประเมินใหม่ด้วย DP
#  ในส่วนที่ 9 ทุกครั้ง (ยังเร็วเพราะ DP เป็น O(n*16))
#  ใช้กลยุทธ์ best-improvement วนจนไม่มีอะไรดีขึ้น (local optimum)
# =========================================================
def two_opt(problem, order, best=None):
    if best is None:
        best = order_cost(problem, order)
    improved = True
    while improved:
        improved = False
        for i in range(len(order) - 1):
            for j in range(i + 1, len(order)):
                cand = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                cost = order_cost(problem, cand)
                if cost < best:
                    order, best, improved = cand, cost, True
    return order, best


def or_opt(problem, order, best=None, seg_lengths=(1, 2, 3)):
    if best is None:
        best = order_cost(problem, order)
    improved = True
    while improved:
        improved = False
        for L in seg_lengths:
            for i in range(len(order) - L + 1):
                seg = order[i:i + L]
                rest = order[:i] + order[i + L:]
                for j in range(len(rest) + 1):
                    for piece in (seg, seg[::-1]):
                        cand = rest[:j] + piece + rest[j:]
                        if cand == order:
                            continue
                        cost = order_cost(problem, cand)
                        if cost < best:
                            order, best, improved = cand, cost, True
    return order, best


def local_search(problem, order):
    """สลับ 2-opt กับ Or-opt จนกว่าจะไม่มีอะไรดีขึ้นทั้งคู่"""
    best = order_cost(problem, order)
    while True:
        order, c1 = two_opt(problem, order, best)
        order, c2 = or_opt(problem, order, c1)
        if c2 >= best:
            return order, c2
        best = c2


def quick_or_opt(problem, order, window=6, max_passes=8):
    """
    Or-opt แบบจำกัดระยะ (ย้ายทีละ 1 จุด ไปไม่เกิน window ตำแหน่ง)
    ใช้เป็น memetic step ภายใน GA เพราะเร็วกว่า or_opt เต็มรูปแบบมาก
    (or_opt เต็มรูปแบบเป็น O(n^2) ต่อ 1 pass ส่วนอันนี้เป็น O(n*window))
    """
    best = order_cost(problem, order)
    n = len(order)
    for _ in range(max_passes):
        improved = False
        for i in range(n):
            rest = order[:i] + order[i + 1:]
            lo, hi = max(0, i - window), min(len(rest), i + window)
            for j in range(lo, hi + 1):
                cand = rest[:j] + [order[i]] + rest[j:]
                if cand == order:
                    continue
                cost = order_cost(problem, cand)
                if cost < best:
                    order, best, improved = cand, cost, True
                    break
        if not improved:
            break
    return order, best


# =========================================================
# 12) วิธีที่ 3 : Genetic Algorithm  (Topic 10.2)
# ---------------------------------------------------------
#  การเข้ารหัส (encoding): โครโมโซม = permutation ของหมายเลขจุดหยิบ 0..n-1
#      ไม่ต้องใส่ entrance/exit ในยีน เพราะถูกตรึงเป็นหัว/ท้ายเสมอ
#  ฟิตเนส: ระยะทางจาก DP ส่วนที่ 9 (ยิ่งน้อยยิ่งดี)
#  Selection : tournament ขนาด 3
#  Crossover : OX (Order Crossover) — รักษาความเป็น permutation
#  Mutation  : สุ่มเลือกระหว่าง inversion กับ swap
#  Elitism   : เก็บตัวที่ดีที่สุด elite ตัวไปรุ่นถัดไปตรง ๆ
#  ปรับใช้กับโจทย์ (ส่วนที่ปรับปรุงจาก GA มาตรฐาน)
#    1) ใส่คำตอบจาก Nearest Neighbour เป็น seed ตัวแรกของประชากร
#       ทำให้ไม่ต้องเริ่มจากศูนย์ ลดจำนวนรุ่นที่ต้องใช้
#    2) เป็น Memetic GA — ทุก ls_every รุ่น เอาตัวที่ดีที่สุดไปทำ quick Or-opt
#       (GA หา "ย่าน" ที่ดี, local search ขัดให้เนียน)
#    3) ถ้าไม่ดีขึ้นติดกันนาน (stagnation) จะ restart ประชากรส่วนล่างแบบสุ่ม
# =========================================================
def solve_genetic(problem, population_size=None, generations=None,
                  mutation_rate=0.5, elite=2, tournament=3,
                  ls_every=50, seed=1, seed_order=None, refine=True):
    n = problem.n
    if n <= 2:
        return list(range(n)), order_cost(problem, list(range(n)))

    population_size = population_size or min(200, max(60, 4 * n))
    generations = generations or min(600, max(200, 12 * n))

    rnd = random.Random(seed)
    base = list(range(n))

    # ประชากรเริ่มต้น = คำตอบ NN 1 ตัว + NN ที่บังคับจุดเริ่มต่างกันครึ่งหนึ่ง
    # + สุ่มล้วนอีกครึ่งหนึ่ง (คุมสมดุลระหว่างคุณภาพกับความหลากหลาย)
    population = []
    if seed_order is not None:
        population.append(list(seed_order))
    half = population_size // 2
    starts = rnd.sample(base, min(n, half))
    for s in starts:
        if len(population) >= half:
            break
        population.append(solve_nearest_neighbour(problem, first=s))
    while len(population) < population_size:
        population.append(rnd.sample(base, n))
    fitness = [order_cost(problem, ind) for ind in population]

    best_idx = min(range(population_size), key=lambda i: fitness[i])
    best_order, best_cost = population[best_idx][:], fitness[best_idx]
    stagnation = 0

    for gen in range(generations):
        ranked = sorted(range(population_size), key=lambda i: fitness[i])
        new_pop = [population[i][:] for i in ranked[:elite]]
        new_fit = [fitness[i] for i in ranked[:elite]]

        while len(new_pop) < population_size:
            # --- tournament selection ---
            def pick():
                cand = rnd.sample(range(population_size), tournament)
                return population[min(cand, key=lambda i: fitness[i])]

            parent_a, parent_b = pick(), pick()

            # --- order crossover (OX) ---
            i, j = sorted(rnd.sample(range(n), 2))
            child = [None] * n
            child[i:j + 1] = parent_a[i:j + 1]
            taken = set(parent_a[i:j + 1])
            fill = [g for g in parent_b if g not in taken]
            k = 0
            for t in range(n):
                if child[t] is None:
                    child[t] = fill[k]
                    k += 1

            # --- mutation ---
            if rnd.random() < mutation_rate:
                i, j = sorted(rnd.sample(range(n), 2))
                if rnd.random() < 0.5:
                    child[i:j + 1] = child[i:j + 1][::-1]   # inversion
                else:
                    child[i], child[j] = child[j], child[i]  # swap

            new_pop.append(child)
            new_fit.append(order_cost(problem, child))

        population, fitness = new_pop, new_fit

        # --- memetic step: ขัดตัวที่ดีที่สุด 3 ตัวด้วย local search เบา ๆ ---
        if ls_every and (gen + 1) % ls_every == 0:
            ranked = sorted(range(population_size), key=lambda i: fitness[i])
            for b in ranked[:3]:
                improved, cost = quick_or_opt(problem, population[b])
                population[b], fitness[b] = improved, cost

        b = min(range(population_size), key=lambda i: fitness[i])
        if fitness[b] < best_cost:
            best_cost, best_order = fitness[b], population[b][:]
            stagnation = 0
        else:
            stagnation += 1

        # --- restart ครึ่งล่างของประชากรเมื่อหยุดนิ่งนานเกินไป ---
        if stagnation >= 60:
            ranked = sorted(range(population_size), key=lambda i: fitness[i])
            for i in ranked[population_size // 2:]:
                population[i] = rnd.sample(base, n)
                fitness[i] = order_cost(problem, population[i])
            stagnation = 0

    # --- ขัดคำตอบสุดท้ายด้วย Local Search เต็มรูปแบบ (memetic algorithm) ---
    if refine:
        best_order, best_cost = local_search(problem, best_order)
    return best_order, best_cost


# =========================================================
# 13) วิธีที่ 4 : Ant Colony Optimization  (Topic 10.4)
# ---------------------------------------------------------
#  มดแต่ละตัวสร้างลำดับการหยิบทีละจุด โดยเลือกจุดถัดไปด้วยความน่าจะเป็น
#        P(i->j) = [tau_ij ^ alpha] * [eta_ij ^ beta] / sum(...)
#  โดย eta_ij = 1 / group_dist[i][j] (ยิ่งใกล้ยิ่งน่าเลือก)
#  จบรอบ -> ระเหยฟีโรโมน tau *= (1-rho) -> โรยฟีโรโมน Q/ระยะทาง บนเส้นที่ใช้
#  ปรับใช้กับโจทย์ (ส่วนที่ปรับปรุงจาก ACO มาตรฐาน)
#    1) ระยะทางฮิวริสติก eta ใช้ระยะ BFS จริงระหว่างกลุ่ม access cell
#       ไม่ใช่ระยะยุคลิด/Manhattan ตรง ๆ (สำคัญมาก เพราะมีชั้นวางขวาง)
#    2) ใช้ elitist ant — โรยฟีโรโมนเพิ่มให้เส้นทางที่ดีที่สุดที่เคยเจอทุกรอบ
#       เพื่อเร่งการลู่เข้า
#    3) จำกัดฟีโรโมนไว้ในช่วง [tau_min, tau_max] แบบ Max-Min Ant System
#       กันไม่ให้ลู่เข้าเร็วเกินจนติด local optimum
#    4) เมื่อจบทุกรอบ นำลำดับที่ดีที่สุดไปขัดด้วย Local Search ส่วนที่ 11
# =========================================================
def solve_ant_colony(problem, n_ants=None, iterations=None,
                     alpha=1.0, beta=3.0, rho=0.1, Q=100.0,
                     elitist=2.0, seed=1, refine=True):
    n = problem.n
    if n <= 2:
        return list(range(n)), order_cost(problem, list(range(n)))

    n_ants = n_ants or min(40, max(20, n))
    iterations = iterations or min(150, max(80, 3 * n))

    rnd = random.Random(seed)
    k = n + 2
    START, END = 0, k - 1

    tau_max, tau_min = 5.0, 0.05
    tau = [[1.0] * k for _ in range(k)]
    eta = [[0.0] * k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            d = problem.group_dist[i][j]
            eta[i][j] = 0.0 if (i == j or d == 0 or d == INF) else 1.0 / d

    best_order, best_cost = None, INF

    for _ in range(iterations):
        solutions = []
        for _ in range(n_ants):
            unvisited = set(range(1, END))
            current = START
            order = []
            while unvisited:
                cands = list(unvisited)
                weights = [(tau[current][j] ** alpha) * (eta[current][j] ** beta)
                           for j in cands]
                total = sum(weights)
                nxt = rnd.choices(cands, weights=weights)[0] if total > 0 \
                    else rnd.choice(cands)
                order.append(nxt - 1)
                unvisited.discard(nxt)
                current = nxt
            cost = order_cost(problem, order)
            solutions.append((order, cost))
            if cost < best_cost:
                best_order, best_cost = order[:], cost

        # --- ระเหย ---
        for i in range(k):
            row = tau[i]
            for j in range(k):
                row[j] *= (1.0 - rho)

        # --- โรยฟีโรโมนจากมดทุกตัว ---
        for order, cost in solutions:
            nodes = [START] + [p + 1 for p in order] + [END]
            add = Q / cost
            for a, b in zip(nodes, nodes[1:]):
                tau[a][b] += add
                tau[b][a] += add

        # --- elitist ant: เน้นเส้นทางที่ดีที่สุดที่เคยเจอ ---
        nodes = [START] + [p + 1 for p in best_order] + [END]
        add = elitist * Q / best_cost
        for a, b in zip(nodes, nodes[1:]):
            tau[a][b] += add
            tau[b][a] += add

        # --- Max-Min: จำกัดช่วงฟีโรโมน ---
        for i in range(k):
            row = tau[i]
            for j in range(k):
                if row[j] > tau_max:
                    row[j] = tau_max
                elif row[j] < tau_min:
                    row[j] = tau_min

    if refine:
        best_order, best_cost = local_search(problem, best_order)
    return best_order, best_cost


# =========================================================
# 14) วิธีที่ 5 : Held-Karp Dynamic Programming  (exact, n เล็ก)
# ---------------------------------------------------------
#  DP บน bitmask แบบ TSP มาตรฐาน แต่ขยาย state ให้รวม "ช่องที่ยืนหยิบ" ด้วย
#      f(mask, i, c) = ระยะทางน้อยสุดที่เริ่มจาก entrance หยิบของครบตาม mask
#                      แล้วมาจบที่จุด i โดยยืนอยู่ช่อง c
#  คำตอบ = min over (i,c) ของ f(FULL, i, c) + d(c, exit)
#  ให้คำตอบที่ดีที่สุดของ "แบบจำลองแวะทุกจุดตามลำดับ" แบบการันตี
#  ความซับซ้อน O(2^n * n^2 * A^2) จึงใช้ได้เฉพาะ n <= ~12
# =========================================================
EXACT_MAX_N = 12


def solve_held_karp(problem):
    n = problem.n
    if n > EXACT_MAX_N:
        return None, INF
    dist = problem.dist
    groups = problem.groups

    # dp[mask][i] = list ของต้นทุน ต่อ access cell แต่ละช่องของจุด i
    dp = [[None] * n for _ in range(1 << n)]
    par = [[None] * n for _ in range(1 << n)]
    for i in range(n):
        dp[1 << i][i] = [dist[problem.START][v] for v in groups[i]]
        par[1 << i][i] = [None] * len(groups[i])

    for mask in range(1, 1 << n):
        for i in range(n):
            cur = dp[mask][i]
            if cur is None:
                continue
            gi = groups[i]
            for j in range(n):
                if mask >> j & 1:
                    continue
                nmask = mask | (1 << j)
                gj = groups[j]
                target = dp[nmask][j]
                if target is None:
                    target = [INF] * len(gj)
                    dp[nmask][j] = target
                    par[nmask][j] = [None] * len(gj)
                tpar = par[nmask][j]
                for bj, v in enumerate(gj):
                    best, arg = target[bj], tpar[bj]
                    for bi, u in enumerate(gi):
                        val = cur[bi] + dist[u][v]
                        if val < best:
                            best, arg = val, (i, bi)
                    target[bj], tpar[bj] = best, arg

    full = (1 << n) - 1
    best, state = INF, None
    for i in range(n):
        cur = dp[full][i]
        if cur is None:
            continue
        for bi, u in enumerate(groups[i]):
            val = cur[bi] + dist[u][problem.END]
            if val < best:
                best, state = val, (full, i, bi)

    # ย้อนรอยเพื่อเอา "ลำดับการหยิบ"
    order = []
    mask, i, bi = state
    while True:
        order.append(i)
        prev = par[mask][i][bi]
        if prev is None:
            break
        mask ^= (1 << i)
        i, bi = prev
    order.reverse()
    return order, best


# =========================================================
# 15) ตัวช่วยสำคัญ : "หยิบผ่านทางได้ฟรี" (Free-pickup reduction)
# ---------------------------------------------------------
#  ข้อสังเกตจากโจทย์: พนักงานหยิบของได้ทันทีที่ "ยืนบนช่องติดชั้นวาง"
#  ดังนั้นระหว่างเดินจาก A ไป B ถ้าเผอิญเดินผ่านช่องที่ติดชั้นวางของจุด k
#  ก็ถือว่าหยิบจุด k ได้ฟรี ไม่ต้องแวะเป็นจุดแยกอีก
#  แบบจำลอง TSP ปกติมองไม่เห็นเรื่องนี้ ทำให้ได้คำตอบยาวเกินจริง
#  ฟังก์ชันนี้จึงไล่ลบจุดที่ "หยิบฟรีได้อยู่แล้ว" ออกจากลำดับ แล้วคำนวณซ้ำ
#  ตราบใดที่เส้นทางใหม่ยังหยิบครบทุกจุดและสั้นลง -> เป็นการ optimize เพิ่ม
#  ที่ใช้ได้กับ "ทุกวิธี" ข้างบน
# =========================================================
def free_pickup_reduction(problem, order):
    cost, path, stops = build_route(problem, order)
    while True:
        changed = False
        for i in range(len(order)):
            cand = order[:i] + order[i + 1:]
            c2, p2, s2 = build_route(problem, cand)
            if c2 < cost and collected_mask(problem, p2) == problem.FULL_MASK:
                order, cost, path, stops = cand, c2, p2, s2
                changed = True
                break
        if not changed:
            return order, cost, path, stops


# =========================================================
# 16) ตัวตรวจคำตอบ : BFS บน state space (cell, ของที่หยิบแล้ว)
# ---------------------------------------------------------
#  พิสูจน์ค่า optimum จริง ๆ ของโจทย์ (รวมกรณีหยิบผ่านทางได้ฟรี)
#  state = (ช่องที่ยืน, bitmask ของที่หยิบแล้ว) , edge น้ำหนัก 1 -> BFS ได้เลย
#  จำนวน state = (#ช่องทางเดิน) x 2^n  จึงใช้ได้เฉพาะ n เล็ก
# =========================================================
def solve_exact_state_bfs(problem, max_n=EXACT_MAX_N):
    n = problem.n
    if n > max_n:
        return INF, [], 0

    wh = problem.warehouse
    full = problem.FULL_MASK
    start = (problem.entrance, problem.cell_mask.get(problem.entrance, 0))

    dist = {start: 0}
    parent = {start: None}
    queue = deque([start])
    goal = None
    while queue:
        state = queue.popleft()
        cell, mask = state
        if cell == problem.exit_point and mask == full:
            goal = state
            break
        d = dist[state]
        for dr, dc in movements:
            nxt = (cell[0] + dr, cell[1] + dc)
            if not is_walkable(wh, nxt):
                continue
            nstate = (nxt, mask | problem.cell_mask.get(nxt, 0))
            if nstate not in dist:
                dist[nstate] = d + 1
                parent[nstate] = state
                queue.append(nstate)

    if goal is None:
        return INF, [], len(dist)

    path = []
    s = goal
    while s is not None:
        path.append(s[0])
        s = parent[s]
    path.reverse()
    return dist[goal], path, len(dist)


# =========================================================
# 17) ตัวช่วยแสดงผล
# =========================================================
def format_path(path, per_line=8):
    lines = []
    for i in range(0, len(path), per_line):
        chunk = path[i:i + per_line]
        lines.append("    " + " -> ".join(f"({r:2d},{c:2d})" for r, c in chunk))
    return "\n".join(lines)


def run_method(name, fn, problem, apply_reduction=True):
    """รันวิธีหนึ่ง ๆ แล้วคืนผลลัพธ์พร้อมเวลาที่ใช้"""
    t0 = time.perf_counter()
    order, raw_cost = fn(problem)
    if order is None:
        return None
    if apply_reduction:
        order, cost, path, stops = free_pickup_reduction(problem, order)
    else:
        cost, path, stops = build_route(problem, order)
    elapsed = time.perf_counter() - t0

    ok = collected_mask(problem, path) == problem.FULL_MASK
    return {
        "name": name,
        "order": order,
        "raw_cost": raw_cost,
        "cost": cost,
        "steps": len(path) - 1,
        "path": path,
        "stops": stops,
        "time": elapsed,
        "valid": ok and cost == len(path) - 1,
    }


def print_result_detail(problem, res):
    print()
    print("=" * 78)
    print(f" ผลลัพธ์ละเอียดของวิธีที่ดีที่สุด : {res['name']}")
    print("=" * 78)

    # ----- 5.1 ลำดับการหยิบสินค้า -----
    print("\n[5.1] ลำดับการหยิบสินค้า")
    print(f"  จำนวนจุดที่ต้องแวะจริง {len(res['order'])} จุด "
          f"(จากทั้งหมด {problem.n} จุด)")
    print(f"  {'ลำดับ':<6}{'จุดที่':<8}{'ชั้นวาง (r,c)':<18}{'ยืนหยิบที่ (r,c)':<20}")
    print("  " + "-" * 56)
    for step, (p, cell) in enumerate(zip(res["order"], res["stops"]), start=1):
        print(f"  {step:<8}{p + 1:<8}{str(problem.pickups[p]):<18}{str(cell):<20}")

    free = [i for i in range(problem.n) if i not in res["order"]]
    if free:
        names = ", ".join(str(i + 1) for i in free)
        print(f"  * จุดที่ {names} หยิบได้ 'ฟรี' ระหว่างเดินผ่าน "
              f"ไม่ต้องแวะเป็นจุดแยก")

    # ----- 5.2 เส้นทางทั้งหมด -----
    print(f"\n[5.2] เส้นทางทั้งหมด ({len(res['path'])} ช่อง)")
    print(format_path(res["path"]))

    # ----- 5.3 จำนวนก้าวรวม -----
    print(f"\n[5.3] จำนวนก้าวรวมที่น้อยที่สุด = {res['steps']} ก้าว")
    print(f"      ตรวจสอบความถูกต้องของเส้นทาง : "
          f"{'ผ่าน' if res['valid'] else 'ไม่ผ่าน'}")


# =========================================================
# 18) การทดลองขยายจำนวนจุด n = 10, 20, 30, 50  (ข้อ 7.2.4)
# =========================================================
def scaling_experiment(warehouse, entrance, exit_point,
                       sizes=(10, 20, 30, 50), seed=20):
    print()
    print("=" * 78)
    print(" [7.2.4] ผลของการเพิ่มจำนวนจุดหยิบสินค้า n = 10, 20, 30, 50")
    print("=" * 78)
    header = (f"{'n':>4} | {'NN':^17} | {'NN+LocalSearch':^17} | "
              f"{'GA':^17} | {'ACO':^17} | {'Held-Karp':^17}")
    print(header)
    print("-" * len(header))

    for n in sizes:
        pts = generate_pickup_points(warehouse, n, seed=seed)
        problem = WarehouseProblem(warehouse, entrance, exit_point, pts)

        nn_order = solve_nearest_neighbour(problem)
        results = [
            run_method("NN", lambda p: (solve_nearest_neighbour(p), 0), problem),
            run_method("NN+LS",
                       lambda p: local_search(p, solve_nearest_neighbour(p)),
                       problem),
            run_method("GA",
                       lambda p: solve_genetic(p, seed=7, seed_order=nn_order),
                       problem),
            run_method("ACO", lambda p: solve_ant_colony(p, seed=7), problem),
            run_method("Held-Karp", solve_held_karp, problem),
        ]

        cells = []
        for r in results:
            if r is None or r["cost"] == INF:
                cells.append(f"{'n/a':^17}")
            else:
                cells.append(f"{r['steps']:>7} ({r['time']:6.2f}s)")
        print(f"{n:>4} | " + " | ".join(cells))

    print()
    print(" หมายเหตุ  n/a = วิธี exact ใช้ไม่ได้เพราะ state space โตแบบ 2^n")
    print(" ค่าในวงเล็บคือเวลาประมวลผล (รวมขั้นตอน free-pickup reduction)")


# =========================================================
# 19) กำหนดทางเข้า ทางออก และจำนวนจุดหยิบสินค้า
# =========================================================
if __name__ == "__main__":
    # พิกัดอยู่ในรูปแบบ (row, column)
    entrance = (0, 0)      # ต้องเป็นทางเดิน: ค่า 0
    exit_point = (17, 19)  # ต้องเป็นทางเดิน: ค่า 0

    # จำนวนจุดที่ต้องหยิบสินค้า
    n = 10

    # สุ่มจุดหยิบสินค้าบนชั้นวาง: ทุกจุดจะมีค่า 1 แน่นอน
    pickup_points = generate_pickup_points(warehouse=warehouse, n=n, seed=20)
    # pickup_points = generate_pickup_points2(warehouse=warehouse, n=n)

    # ตรวจสอบความถูกต้องของจุดทั้งหมด
    validate_points(
        warehouse=warehouse,
        entrance=entrance,
        exit_point=exit_point,
        pickup_points=pickup_points
    )

    # -----------------------------------------------------
    # 19.1 แสดงข้อมูลพิกัด
    # -----------------------------------------------------
    print("=" * 78)
    print(" โจทย์ : หาเส้นทางหยิบสินค้าที่สั้นที่สุดในโกดัง")
    print("=" * 78)
    print(f"ขนาดโกดัง : {warehouse.shape[0]} x {warehouse.shape[1]} "
          f"(ทางเดิน {int((warehouse == 0).sum())} ช่อง, "
          f"ชั้นวาง {int((warehouse == 1).sum())} ช่อง)")
    print("Entrance:", entrance)
    print("Exit:", exit_point)
    print(f"Pickup points ({n} points):")
    for index, point in enumerate(pickup_points, start=1):
        print(f"  Pickup {index}: {point}")

    # -----------------------------------------------------
    # 19.2 สร้างแบบจำลองกราฟ
    # -----------------------------------------------------
    t0 = time.perf_counter()
    problem = WarehouseProblem(warehouse, entrance, exit_point, pickup_points)
    build_time = time.perf_counter() - t0

    print(f"\n[Graph model] node สำคัญ {len(problem.cells)} ช่อง "
          f"(entrance + exit + access cells), "
          f"ทำ BFS {problem.bfs_count} ครั้ง ใช้เวลา {build_time:.3f}s")
    print("  access cells ของแต่ละจุดหยิบ (ช่องที่ยืนแล้วหยิบของได้):")
    for i, cells in enumerate(problem.access_cells, start=1):
        print(f"    จุด {i:>2} {str(problem.pickups[i - 1]):<10} -> {cells}")

    # ตรวจว่า BFS กับ A* ให้ระยะทางตรงกัน (ยืนยันความถูกต้องของตารางระยะทาง)
    check_a = problem.groups[0][0]
    d_bfs = problem.dist[problem.START][check_a]
    d_astar = astar_shortest_path(warehouse, entrance, problem.cells[check_a])
    print(f"  ตรวจสอบ BFS vs A* : entrance -> {problem.cells[check_a]} "
          f"= {d_bfs} / {d_astar}  "
          f"({'ตรงกัน' if d_bfs == d_astar else 'ไม่ตรงกัน'})")

    # -----------------------------------------------------
    # 19.3 รันทุกวิธีแล้วเปรียบเทียบ (ข้อ 6, 7.2.2, 7.2.3)
    # -----------------------------------------------------
    nn_seed_order = solve_nearest_neighbour(problem)

    methods = [
        ("1. Nearest Neighbour (greedy)",
         lambda p: (solve_nearest_neighbour(p), 0)),
        ("2. NN + Local Search (2-opt/Or-opt)",
         lambda p: local_search(p, solve_nearest_neighbour(p))),
        ("3. Genetic Algorithm (memetic)",
         lambda p: solve_genetic(p, seed=7, seed_order=nn_seed_order)),
        ("4. Ant Colony Optimization",
         lambda p: solve_ant_colony(p, seed=7)),
        ("5. Held-Karp DP (exact ordering)",
         solve_held_karp),
    ]

    results = []
    for name, fn in methods:
        res = run_method(name, fn, problem)
        if res is not None and res["cost"] != INF:
            results.append(res)

    # ค่า optimum จริงจาก BFS บน state space
    t0 = time.perf_counter()
    opt_steps, opt_path, n_states = solve_exact_state_bfs(problem)
    opt_time = time.perf_counter() - t0

    print()
    print("=" * 78)
    print(" [7.2.2 / 7.2.3] เปรียบเทียบผลลัพธ์ของแต่ละวิธี")
    print("=" * 78)
    header = (f"{'วิธี':<38}{'ก่อน reduce':>12}{'ก้าวรวม':>10}"
              f"{'เวลา (s)':>11}{'gap':>8}")
    print(header)
    print("-" * 80)
    best_steps = min(r["steps"] for r in results)
    ref = opt_steps if opt_steps != INF else best_steps
    for r in results:
        raw = f"{int(r['raw_cost'])}" if r["raw_cost"] else "-"
        gap = (r["steps"] - ref) / ref * 100
        print(f"{r['name']:<38}{raw:>12}{r['steps']:>10}"
              f"{r['time']:>11.3f}{gap:>7.1f}%")
    print("-" * 80)
    if opt_steps != INF:
        print(f"{'** Optimum จริง (state-space BFS)':<38}{'-':>12}"
              f"{opt_steps:>10}{opt_time:>11.3f}{0.0:>7.1f}%")
        print(f"   (ค้นหา {n_states:,} state = ช่องทางเดิน x 2^{n})")
    print()
    print(" 'ก่อน reduce' = ระยะทางที่วิธีนั้นหาได้ก่อนตัดจุดที่หยิบผ่านทางได้ฟรี")
    print(" 'gap' = ห่างจากคำตอบที่ดีที่สุดกี่เปอร์เซ็นต์")

    # -----------------------------------------------------
    # 19.4 แสดงรายละเอียดคำตอบที่ดีที่สุด (ข้อ 5.1 - 5.3)
    # -----------------------------------------------------
    best = min(results, key=lambda r: (r["steps"], r["time"]))
    print_result_detail(problem, best)

    if opt_steps != INF:
        print(f"\n  เทียบกับค่าที่พิสูจน์ได้ว่าดีที่สุด = {opt_steps} ก้าว -> "
              f"{'ได้คำตอบที่ดีที่สุด (optimal)' if best['steps'] == opt_steps else 'ยังไม่ optimal'}")

    # -----------------------------------------------------
    # 19.5 วาดแผนที่และเส้นทาง
    # -----------------------------------------------------
    plot_warehouse(
        warehouse=warehouse,
        entrance=entrance,
        exit_point=exit_point,
        pickup_points=pickup_points
    )

    plot_solution(
        warehouse=warehouse,
        entrance=entrance,
        exit_point=exit_point,
        pickup_points=pickup_points,
        path=best["path"],
        order=best["order"],
        stops=best["stops"],
        title=f"Optimal Picking Route - {best['steps']} steps "
              f"({best['name']})",
        filename="warehouse_route.png"
    )

    # -----------------------------------------------------
    # 19.6 การทดลองขยายจำนวนจุด (ข้อ 7.2.4)
    # -----------------------------------------------------
    scaling_experiment(warehouse, entrance, exit_point,
                       sizes=(10, 20, 30, 50), seed=20)
