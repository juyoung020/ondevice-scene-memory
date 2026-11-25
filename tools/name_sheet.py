# 물체 대표 사진 + 이름 한 장 모음(눈 판정용). 사용: name_sheet.py <out> <png>
import json, sys
from PIL import Image, ImageDraw
out, dst = sys.argv[1], sys.argv[2]
d = json.load(open(f"{out}/view.json"))
objs = [o for o in d["objects"] if o.get("rgbd") and not o.get("structural")]
objs.sort(key=lambda o: o["id"])
S, C = 200, 8
rows = (len(objs) + C - 1) // C
sheet = Image.new("RGB", (C * S, rows * (S + 30)), "white")
dr = ImageDraw.Draw(sheet)
for i, o in enumerate(objs):
    im = Image.open(f"{out}/{o['rgbd']['rgb']}").convert("RGB")
    try:
        mk = Image.open(f"{out}/{o['rgbd']['mask']}").convert("L").resize(im.size)
        im = Image.composite(Image.blend(im, Image.new("RGB", im.size, (255, 0, 0)), 0.35), im, mk)
    except Exception:
        pass
    im.thumbnail((S, S))
    x, y = (i % C) * S, (i // C) * (S + 30)
    sheet.paste(im, (x, y))
    dr.text((x + 2, y + S + 2), f"O{o['id']} {o['name']} {o.get('name_p',-1):.2f}", fill="black")
    dr.text((x + 2, y + S + 15), f"n{o['n_obs']} {o['state']}", fill="gray")
sheet.save(dst)
print(len(objs))
