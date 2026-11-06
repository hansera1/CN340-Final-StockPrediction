"""รอบ 7 ข้อ 4: ช่องโมเดลของวันจริง (rules.yaml → live.models / direction_target / shadow_slots)

ทิศทางและราคาเลือกแยกรอบได้ · spec = ทางใต้ outputs/models/ · model_dir (เช่น R4ATC) ต่อหน้าให้เอง
เป้าหมาย up_notup ใช้โฟลเดอร์ <model_dir>_UN (04_lock_model.py --target up_notup) ไม่ปนกับโมเดลขึ้น/ลง
"""
TARGETS = ("up_down", "up_notup")
UN_SUFFIX = "_UN"


def direction_target(rules, session):
    t = rules["live"]["direction_target"][session]
    if t not in TARGETS:
        raise ValueError(f"live.direction_target.{session} ไม่รู้จัก: {t}")
    return t


def _prefix(rules, un=False):
    base = rules["live"].get("model_dir") or ""
    base = f"{base}{UN_SUFFIX}" if un else base
    return f"{base}/" if base else ""


def direction_spec(rules, session):
    un = direction_target(rules, session) == "up_notup"
    return _prefix(rules, un) + rules["live"]["models"][f"{session}_dir"]


def price_specs(rules, session):
    """→ (spec ของ P(ราคาปลาย = P(c)), spec ของ P(ราคาปลาย > P(c) | ไม่เท่า))"""
    m = rules["live"]["models"][f"{session}_price"]
    return _prefix(rules) + m["eq"], _prefix(rules) + m["up"]


def support_spec(rules, key):
    """โมเดลประกอบที่ไม่แยกช่อง: shadow (ช่วงราคา + รันเงา E0-ud), flat_dir"""
    return _prefix(rules) + rules["live"][key]


def live_specs(rules):
    """{ชื่อที่แสดงใน check_ready: spec} ทุกโมเดลที่วันจริงต้องโหลดได้ (รวมชุดรันเงา)"""
    out = {}
    for s in rules["sessions"]:
        out[f"{s}_dir"] = direction_spec(rules, s)
        out[f"{s}_price_eq"], out[f"{s}_price_up"] = price_specs(rules, s)
    out["shadow"], out["flat_dir"] = support_spec(rules, "shadow"), support_spec(rules, "flat_dir")
    for i, x in enumerate(rules["live"].get("shadow_slots") or []):
        out[f"เงา {x['slot']} {i + 1}"] = x["spec"]
    return out


def lock_specs(rules):
    """spec (ไม่มี model_dir) ที่ 04_lock_model.py ต้องฝึก เรียงไม่ซ้ำ"""
    live, out = rules["live"], []
    for s in rules["sessions"]:
        out.append(live["models"][f"{s}_dir"])
    for s in rules["sessions"]:
        out += [live["models"][f"{s}_price"]["eq"], live["models"][f"{s}_price"]["up"]]
    return list(dict.fromkeys(out + [live["shadow"], live["flat_dir"]]))


def decide(p_up, target):
    """up_down: P ≥ 0.5 (กติกาที่ล็อกเดิม) · up_notup: P > 0.5 · ไฟล์ส่งยังเป็น ขึ้น / ลง"""
    return "ขึ้น" if (p_up > 0.5 if target == "up_notup" else p_up >= 0.5) else "ลง"
