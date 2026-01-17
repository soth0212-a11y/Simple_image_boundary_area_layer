use crate::config::AppConfig;
use crate::rle_ccl_gpu::OutBox;

pub const FLAG_MERGED: u32 = 1 << 0;
pub const FLAG_CONTAIN_USED: u32 = 1 << 1;
pub const FLAG_GAP_BRIDGE: u32 = 1 << 3;

#[derive(Clone, Debug)]
pub struct L3Config {
    pub min_w: u32,
    pub min_h: u32,
    pub min_area: u64,
    pub color_tol: u32,
    pub iou_th: f32,
    pub gap_x_th: u32,
    pub gap_y_th: u32,
    pub overlap_x_th: u32,
    pub overlap_y_th: u32,
    pub contain_ratio_th: f32,
    pub nms_iou_th: f32,
}

impl Default for L3Config {
    fn default() -> Self {
        Self {
            min_w: 2,
            min_h: 2,
            min_area: 16,
            color_tol: 1,
            iou_th: 0.5,
            gap_x_th: 2,
            gap_y_th: 2,
            overlap_x_th: 1,
            overlap_y_th: 1,
            contain_ratio_th: 0.90,
            nms_iou_th: 0.85,
        }
    }
}

impl L3Config {
    pub fn from_app_config(cfg: &AppConfig) -> Self {
        Self {
            min_w: cfg.l3_min_w,
            min_h: cfg.l3_min_h,
            min_area: cfg.l3_min_area,
            color_tol: cfg.l3_color_tol,
            iou_th: cfg.l3_iou_th,
            gap_x_th: cfg.l3_gap_x_th,
            gap_y_th: cfg.l3_gap_y_th,
            overlap_x_th: cfg.l3_overlap_x_th,
            overlap_y_th: cfg.l3_overlap_y_th,
            contain_ratio_th: cfg.l3_contain_ratio_th,
            nms_iou_th: cfg.l3_nms_iou_th,
        }
    }
}

pub fn l3_merge_and_refine_cpu(l2_boxes: &[OutBox], cfg: &L3Config) -> Vec<OutBox> {
    let mut boxes = Vec::with_capacity(l2_boxes.len());
    boxes.extend_from_slice(l2_boxes);
    let n = boxes.len();
    if n == 0 {
        return Vec::new();
    }

    let mut uf = UnionFind::new(n);
    for i in 0..n {
        for j in (i + 1)..n {
            let a = &boxes[i];
            let b = &boxes[j];
            if !color_close_565(a.color565, b.color565, cfg.color_tol) {
                continue;
            }
            let (merge, flags) = spatial_merge(a, b, cfg);
            if merge {
                uf.union(i, j, flags);
            }
        }
    }

    let mut root_to_comp = vec![usize::MAX; n];
    let mut comps: Vec<Component> = Vec::new();
    for (idx, b) in boxes.iter().enumerate() {
        let root = uf.find(idx);
        let comp_idx = if root_to_comp[root] == usize::MAX {
            let edge_flags = uf.edge_flags(root);
            let next = comps.len();
            comps.push(Component::new(edge_flags));
            root_to_comp[root] = next;
            next
        } else {
            root_to_comp[root]
        };
        comps[comp_idx].update(b);
    }

    let mut merged = Vec::with_capacity(comps.len());
    for comp in comps {
        let mut flags = comp.edge_flags;
        if comp.size > 1 {
            flags |= FLAG_MERGED;
        }
        merged.push(OutBox::from_xy(
            comp.minx,
            comp.miny,
            comp.maxx,
            comp.maxy,
            comp.best_color,
            flags,
        ));
    }

    post_refine(merged, cfg)
}

fn post_refine(boxes: Vec<OutBox>, cfg: &L3Config) -> Vec<OutBox> {
    let n = boxes.len();
    if n <= 1 {
        return boxes;
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        boxes[b]
            .area()
            .cmp(&boxes[a].area())
            .then_with(|| a.cmp(&b))
    });
    let mut keep = vec![true; n];
    for (oi, &i) in order.iter().enumerate() {
        if !keep[i] {
            continue;
        }
        for &j in order.iter().skip(oi + 1) {
            if !keep[j] {
                continue;
            }
            if boxes[i].color565 != boxes[j].color565 {
                continue;
            }
            if contains(&boxes[i], &boxes[j]) {
                keep[j] = false;
                continue;
            }
            let iou = iou(&boxes[i], &boxes[j]);
            if iou > cfg.nms_iou_th {
                keep[j] = false;
            }
        }
    }
    let mut out = Vec::new();
    for (idx, b) in boxes.into_iter().enumerate() {
        if keep[idx] {
            out.push(b);
        }
    }
    out
}

fn color_close_565(a: u32, b: u32, tol: u32) -> bool {
    let a = a & 0xFFFFu32;
    let b = b & 0xFFFFu32;
    if tol == 0 {
        return a == b;
    }
    let ar = (a >> 11) & 31;
    let ag = (a >> 5) & 63;
    let ab = a & 31;
    let br = (b >> 11) & 31;
    let bg = (b >> 5) & 63;
    let bb = b & 31;
    let dr = abs_diff(ar, br);
    let dg = abs_diff(ag, bg);
    let db = abs_diff(ab, bb);
    dr <= tol && dg <= tol.saturating_mul(2) && db <= tol
}

fn spatial_merge(a: &OutBox, b: &OutBox, cfg: &L3Config) -> (bool, u32) {
    let overlap_x = overlap_x(a, b);
    let overlap_y = overlap_y(a, b);
    let gap_x = gap_x(a, b);
    let gap_y = gap_y(a, b);
    let iou_val = iou(a, b);
    let iou_ok = iou_val >= cfg.iou_th;

    let mut flags = 0u32;
    let contain_ok = contain_merge(a, b, cfg.contain_ratio_th);
    if contain_ok {
        flags |= FLAG_CONTAIN_USED;
    }

    let gap_ok = (gap_x <= cfg.gap_x_th && overlap_y >= cfg.overlap_y_th)
        || (gap_y <= cfg.gap_y_th && overlap_x >= cfg.overlap_x_th);
    if gap_ok {
        flags |= FLAG_GAP_BRIDGE;
    }

    (iou_ok || contain_ok || gap_ok, flags)
}

fn contain_merge(a: &OutBox, b: &OutBox, ratio_th: f32) -> bool {
    let a_contains_b = contains(a, b);
    let b_contains_a = contains(b, a);
    if !a_contains_b && !b_contains_a {
        return false;
    }
    let area_a = a.area();
    let area_b = b.area();
    if area_a == 0 || area_b == 0 {
        return false;
    }
    let (small, large) = if area_a <= area_b { (area_a, area_b) } else { (area_b, area_a) };
    (small as f32) / (large as f32) >= ratio_th
}

fn contains(a: &OutBox, b: &OutBox) -> bool {
    a.minx() <= b.minx()
        && a.miny() <= b.miny()
        && a.maxx() >= b.maxx()
        && a.maxy() >= b.maxy()
}

fn iou(a: &OutBox, b: &OutBox) -> f32 {
    let ox = overlap_x(a, b) as u64;
    let oy = overlap_y(a, b) as u64;
    let inter = ox * oy;
    let area_a = a.area();
    let area_b = b.area();
    let union = area_a + area_b - inter;
    if union == 0 {
        return 0.0;
    }
    (inter as f32) / (union as f32)
}

fn overlap_x(a: &OutBox, b: &OutBox) -> u32 {
    let max_min = a.minx().max(b.minx());
    let min_max = a.maxx().min(b.maxx());
    min_max.saturating_sub(max_min)
}

fn overlap_y(a: &OutBox, b: &OutBox) -> u32 {
    let max_min = a.miny().max(b.miny());
    let min_max = a.maxy().min(b.maxy());
    min_max.saturating_sub(max_min)
}

fn gap_x(a: &OutBox, b: &OutBox) -> u32 {
    if a.minx() > b.maxx() {
        a.minx() - b.maxx()
    } else if b.minx() > a.maxx() {
        b.minx() - a.maxx()
    } else {
        0
    }
}

fn gap_y(a: &OutBox, b: &OutBox) -> u32 {
    if a.miny() > b.maxy() {
        a.miny() - b.maxy()
    } else if b.miny() > a.maxy() {
        b.miny() - a.maxy()
    } else {
        0
    }
}

fn abs_diff(a: u32, b: u32) -> u32 {
    if a >= b { a - b } else { b - a }
}

struct Component {
    minx: u32,
    miny: u32,
    maxx: u32,
    maxy: u32,
    size: u32,
    edge_flags: u32,
    color_counts: Vec<(u32, u32)>,
    best_color: u32,
    best_count: u32,
}

impl Component {
    fn new(edge_flags: u32) -> Self {
        Self {
            minx: u32::MAX,
            miny: u32::MAX,
            maxx: 0,
            maxy: 0,
            size: 0,
            edge_flags,
            color_counts: Vec::new(),
            best_color: 0,
            best_count: 0,
        }
    }

    fn update(&mut self, b: &OutBox) {
        let minx = b.minx();
        let miny = b.miny();
        let maxx = b.maxx();
        let maxy = b.maxy();
        self.minx = self.minx.min(minx);
        self.miny = self.miny.min(miny);
        self.maxx = self.maxx.max(maxx);
        self.maxy = self.maxy.max(maxy);
        self.size += 1;

        let color = b.color565 & 0xFFFFu32;
        let mut found = false;
        for entry in self.color_counts.iter_mut() {
            if entry.0 == color {
                entry.1 += 1;
                if entry.1 > self.best_count || (entry.1 == self.best_count && color < self.best_color) {
                    self.best_color = color;
                    self.best_count = entry.1;
                }
                found = true;
                break;
            }
        }
        if !found {
            self.color_counts.push((color, 1));
            if self.best_count == 0 || color < self.best_color {
                self.best_color = color;
                self.best_count = 1;
            }
        }
    }
}

struct UnionFind {
    parent: Vec<usize>,
    edge_flags: Vec<u32>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        let mut parent = Vec::with_capacity(n);
        for i in 0..n {
            parent.push(i);
        }
        Self { parent, edge_flags: vec![0u32; n] }
    }

    fn find(&mut self, x: usize) -> usize {
        let p = self.parent[x];
        if p == x {
            return x;
        }
        let root = self.find(p);
        self.parent[x] = root;
        root
    }

    fn union(&mut self, a: usize, b: usize, flags: u32) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            self.edge_flags[ra] |= flags;
            return;
        }
        let (hi, lo) = if ra > rb { (ra, rb) } else { (rb, ra) };
        self.parent[hi] = lo;
        let merged_flags = self.edge_flags[lo] | self.edge_flags[hi] | flags;
        self.edge_flags[lo] = merged_flags;
    }

    fn edge_flags(&mut self, idx: usize) -> u32 {
        let root = self.find(idx);
        self.edge_flags[root]
    }
}
