"""씬그래프 층: 방(점유 지도의 빈 공간을 구역으로) + 방 이름(SigLIP 2 제로샷) + 엣지(물체–물체, 방–물체).

object-memory-robot 의 scenemap rooms(빈 공간 분할 → 이름) 를 파이썬으로 줄인 판.
- 방 나누기: 빈칸 거리변환 → 문턱(통로보다 넓은 곳) 이상인 덩어리를 씨앗으로 → 빈칸을 가까운 씨앗으로 키워 나눔.
- 방 이름: 키프레임 사진의 SigLIP 2 영상 벡터를 카메라가 있던 방에 모아 평균 → 방 종류 문장 벡터와 코사인.
- 엣지: near(물체 중심 0.8 m 안, 같은 높이대), on(위 물체 바닥 ≈ 아래 물체 윗면, 평면 겹침), contains(방 → 물체).
"""
import cv2
import numpy as np

ROOM_TYPES = {  # 표시 이름: 문장
    'office': 'a photo of an office with desks and computers',
    'meeting room': 'a photo of a meeting room',
    'kitchen': 'a photo of a kitchen',
    'living room': 'a photo of a living room with a sofa',
    'bedroom': 'a photo of a bedroom',
    'bathroom': 'a photo of a bathroom',
    'corridor': 'a photo of a corridor or hallway',
    'lobby': 'a photo of a lobby or lounge with tables and chairs',
    'cafe': 'a photo of a cafe',
    'classroom': 'a photo of a classroom',
    'storage': 'a photo of a storage room',
}
OBJECT_NAMES = [  # COCO 실내 + 사무실·집 물건
    'chair', 'office chair', 'sofa', 'table', 'desk', 'bed', 'monitor', 'computer screen', 'laptop', 'keyboard',
    'computer mouse', 'cell phone', 'book', 'bottle', 'cup', 'mug', 'bowl', 'plate', 'vase', 'potted plant', 'teddy bear',
    'globe', 'ball', 'backpack', 'handbag', 'bag', 'box', 'lamp', 'clock', 'telephone', 'printer', 'speaker', 'trash can',
    'cabinet', 'shelf', 'drawer', 'refrigerator', 'microwave', 'sink', 'toy', 'robot arm', 'scissors', 'paper', 'folder',
    'tape', 'pen', 'headphones', 'cable', 'pillow', 'tv', 'whiteboard', 'picture frame', 'fan', 'umbrella', 'bucket']
STRUCTURAL_NAMES = ['wall', 'floor', 'ceiling', 'window', 'door', 'office partition', 'pillar', 'carpet', 'curtain', 'glass wall']
PALETTE = [(66, 133, 244), (52, 168, 83), (251, 188, 5), (234, 67, 53), (171, 71, 188), (0, 172, 193), (255, 112, 67)]


class Embedder:
    """SigLIP 2 (models/siglip2-base-patch32-256). 영상·글 벡터, 정규화."""

    def __init__(self, path='models/siglip2-base-patch32-256'):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).cuda().eval()
        names = list(ROOM_TYPES)
        self.room_names = names
        self.room_txt = self.text([ROOM_TYPES[n] for n in names])

    def set_labels(self, names, structural):
        """ObjectSAM(종류 없는 분할)용 이름표: 물체 이름 + 구조물(벽·바닥…) 이름. 구조물로 판정되면 물체에서 뺀다."""
        self.lab_names = list(names) + list(structural)
        self.lab_struct = np.array([False] * len(names) + [True] * len(structural))
        self.lab_txt = self.text([f'a photo of a {n}' for n in self.lab_names])

    def labels(self, crops):
        """[(잘라낸 BGR, 마스크)] → [(이름, 확률, 구조물인가)]. 마스크 밖은 회색. 한 장면의 물체를 한 번에(배치)."""
        if not crops:
            return []
        ims = []
        for bgr, mask in crops:
            im = cv2.resize(bgr, (256, 256), interpolation=cv2.INTER_AREA)
            mk = cv2.resize(mask.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
            im[mk == 0] = 127
            ims.append(im[:, :, ::-1])
        with self.torch.no_grad():
            x = self.torch.from_numpy(np.stack(ims)).cuda().permute(0, 3, 1, 2).float() / 255.0
            x = (x - 0.5) / 0.5  # SigLIP 정규화(평균 0.5, 표준편차 0.5)
            v = self.model.get_image_features(pixel_values=x)
            v = getattr(v, 'pooler_output', v).float().cpu().numpy()
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        s = v @ self.lab_txt.T
        out = []
        for row in s:
            p = np.exp((row - row.max()) * 50)
            p /= p.sum()
            k = int(np.argmax(row))
            out.append((self.lab_names[k], float(p[k]), bool(self.lab_struct[k])))
        return out

    def image(self, bgr):
        with self.torch.no_grad():
            x = self.proc(images=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), return_tensors='pt').to('cuda')
            v = self.model.get_image_features(**x)
            v = getattr(v, 'pooler_output', v)
            v = v[0].float().cpu().numpy()
        return v / np.linalg.norm(v)

    def text(self, texts):
        with self.torch.no_grad():
            x = self.proc(text=texts, padding='max_length', max_length=64, return_tensors='pt').to('cuda')
            v = self.model.get_text_features(**x)
            v = getattr(v, 'pooler_output', v)
            v = v.float().cpu().numpy()
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def segment_rooms(img, meta, min_width=1.2, min_area=3.0):
    """img: map.pgm 배열(254 빈칸, 0 막힘, 205 모름), 위가 +y. 반환: 라벨 배열(-1 = 방 아님), 방 수."""
    res = meta['resolution']
    free = (img == 254).astype(np.uint8)
    if free.sum() == 0:
        return np.full(img.shape, -1, np.int32), 0
    free = cv2.morphologyEx(free, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    dist = cv2.distanceTransform(free, cv2.DIST_L2, 5) * res
    seeds = (dist >= min_width / 2).astype(np.uint8)
    n, lab = cv2.connectedComponents(seeds)
    if n <= 1:  # 넓은 곳이 없으면 빈칸 전체가 한 방
        n, lab = cv2.connectedComponents(free)
    # 빈칸을 가까운 씨앗으로 키움(경계 넘지 않게 빈칸 안에서만 한 칸씩)
    lab = np.where(lab > 0, lab, 0).astype(np.int32)
    k = np.ones((3, 3), np.uint8)
    for _ in range(400):
        grow = np.zeros_like(lab)
        for r in range(1, n):
            d = cv2.dilate((lab == r).astype(np.uint8), k)
            grow[(d > 0) & (lab == 0) & (free > 0) & (grow == 0)] = r
        if not grow.any():
            break
        lab = np.where(grow > 0, grow, lab)
    out = np.full(img.shape, -1, np.int32)
    rid = 0
    for r in range(1, n):
        m = lab == r
        if m.sum() * res * res >= min_area:
            out[m] = rid
            rid += 1
    return out, rid


def cell_of(meta, h, xy):
    res = meta['resolution']
    x0, y0 = meta['origin']
    cx = int((xy[0] - x0) / res)
    cy = h - 1 - int((xy[1] - y0) / res)
    return cy, cx


def build(img, meta, objs, kf_embs, emb):
    """objs: view 물체 목록(pos, extent, id, state). kf_embs: [(cam_xy, 영상 벡터)].
    반환: rooms(뷰어 형식), edges([a, b, rel])."""
    lab, nroom = segment_rooms(img, meta)
    h, w = lab.shape

    def room_at(xy):
        cy, cx = cell_of(meta, h, xy)
        if 0 <= cy < h and 0 <= cx < w and lab[cy, cx] >= 0:
            return int(lab[cy, cx])
        # 막힘 칸 위 물체(책상 등): 0.5 m 안의 가장 가까운 방 칸
        r = int(0.5 / meta['resolution'])
        y0, y1, x0, x1 = max(cy - r, 0), min(cy + r + 1, h), max(cx - r, 0), min(cx + r + 1, w)
        sub = lab[y0:y1, x0:x1]
        ys, xs = np.nonzero(sub >= 0)
        if len(ys) == 0:
            return None
        k = np.argmin((ys + y0 - cy) ** 2 + (xs + x0 - cx) ** 2)
        return int(sub[ys[k], xs[k]])

    res = meta['resolution']
    rooms = []
    for r in range(nroom):
        ys, xs = np.nonzero(lab == r)
        cx = meta['origin'][0] + (xs.mean() + 0.5) * res
        cy = meta['origin'][1] + (h - 1 - ys.mean() + 0.5) * res
        vecs = [v for xy, v in kf_embs if room_at(xy) == r]
        name, conf = f'room {r}', 0.0
        if vecs and emb is not None:
            m = np.mean(vecs, 0)
            m /= np.linalg.norm(m)
            s = emb.room_txt @ m
            p = np.exp((s - s.max()) * 50)
            p /= p.sum()
            k = int(np.argmax(s))
            name, conf = emb.room_names[k], float(p[k])
        rooms.append({'id': r, 'name': name, 'conf': conf, 'centroid': [float(cx), float(cy)],
                      'area_m2': float(len(ys) * res * res), 'color': list(PALETTE[r % len(PALETTE)]), 'objects': []})
    edges = []
    live = [o for o in objs if o['state'] != 'gone']
    for o in live:
        r = room_at(o['pos'][:2])
        if r is not None:
            rooms[r]['objects'].append(o['id'])
            edges.append([f'R{r}', f'O{o["id"]}', 'contains'])
    for i, a in enumerate(live):
        pa, ea = np.array(a['pos']), np.array(a['extent'])
        for b in live[i + 1:]:
            pb, eb = np.array(b['pos']), np.array(b['extent'])
            dxy = np.linalg.norm(pa[:2] - pb[:2])
            over = np.all(np.abs(pa[:2] - pb[:2]) < (ea[:2] + eb[:2]) / 2 * 0.8)
            if over and abs((pa[2] - ea[2] / 2) - (pb[2] + eb[2] / 2)) < 0.12:
                edges.append([f'O{a["id"]}', f'O{b["id"]}', 'on'])
            elif over and abs((pb[2] - eb[2] / 2) - (pa[2] + ea[2] / 2)) < 0.12:
                edges.append([f'O{b["id"]}', f'O{a["id"]}', 'on'])
            elif dxy < 0.8 and abs(pa[2] - pb[2]) < 0.5:
                edges.append([f'O{a["id"]}', f'O{b["id"]}', 'near', float(dxy)])
    # near 는 물체마다 가장 가까운 2 개까지만(모든 쌍을 이으면 그물이 됨)
    deg, kept = {}, []
    for e in sorted(edges, key=lambda e: e[3] if len(e) > 3 else -1):
        if e[2] == 'near':
            if deg.get(e[0], 0) >= 2 or deg.get(e[1], 0) >= 2:
                continue
            deg[e[0]] = deg.get(e[0], 0) + 1
            deg[e[1]] = deg.get(e[1], 0) + 1
        kept.append(e[:3])
    return rooms, kept
