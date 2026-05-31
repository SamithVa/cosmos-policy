import json, csv, os

base="/home/data/wanshan/workspace/vla/cosmos-policy/LIBERO-plus/libero/libero"
cls=json.load(open(base+"/benchmark/task_classification.json"))
categories = {
    0: "Camera Viewpoints",
    1: "Robot Initial States",
    2: "Language Instructions",
    3: "Objects Layout",
}
SUITE="libero_10"
CATEGORY=0

rows=[(x["id"]-1, 0, x["name"], x.get("difficulty_level")) for x in cls[SUITE] if x["category"]==categories[CATEGORY]]
os.makedirs("exp", exist_ok=True)
out=f"exp/{SUITE}_{categories[CATEGORY].replace(' ', '_').lower()}.csv"
with open(out,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["task_id","episode_idx","name","difficulty_level"])
    for r in rows: w.writerow(r)
print(f"Wrote {len(rows)} tasks to {out}")
print("first 3:", rows[:3])
