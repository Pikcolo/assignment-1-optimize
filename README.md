# Assignment 1 — หาเส้นทางหยิบสินค้าที่สั้นที่สุดในโกดัง

241-405 Optimization · งานกลุ่ม 3 คน

## วิธีที่กลุ่มเลือกใช้

| สมาชิก | วิธี | ประเภท |
|---|---|---|
| คนที่ 1 | Held-Karp Dynamic Programming | Exact |
| คนที่ 2 | Genetic Algorithm | Metaheuristic (ประชากร) |
| คนที่ 3 | Ant Colony Optimization | Metaheuristic (ฝูง) |

## รันโปรแกรม

```bash
python Creat_warehouse.py                  # รันทั้งหมด (~1.5-2 นาที)
WH_SHOW_PLOTS=0 python Creat_warehouse.py  # ไม่ให้หน้าต่างกราฟเด้ง (ยังเซฟรูปให้)
```

## ไฟล์

| ไฟล์ | คืออะไร |
|---|---|
| `Creat_warehouse.py` | โปรแกรมทั้งหมด |
| `REPORT.md` | รายงานหลัก |
| `ACO_presentation.md` | เอกสารเจาะลึก ACO สำหรับการนำเสนอ |
| `warehouse_map.png` | แผนที่โกดัง |
| `warehouse_route.png` | เส้นทางคำตอบ 72 ก้าว |
| `aco_convergence.png` | กราฟการลู่เข้าของ ACO |
