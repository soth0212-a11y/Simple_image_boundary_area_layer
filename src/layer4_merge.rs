use crate::rle_ccl_gpu::OutBox;
use std::collections::BTreeMap;
use std::ops::RangeInclusive;

#[derive(Clone, Copy, Debug)]
pub struct L4MergeCfg {
    pub bin_size: u32,
    pub neighbor_radius: i32,
    pub gap_thr: i32,
    pub color_tol_r: i32,
    pub color_tol_g: i32,
    pub color_tol_b: i32,
    pub enable_area_ratio: bool,
    pub area_ratio_thr: u32, // max(big/small) allowed
    pub enable_iou: bool,
    pub iou_num: u32, // e.g. 3
    pub iou_den: u32, // e.g. 10 for 0.3
}

impl Default for L4MergeCfg {
    fn default() -> Self {
        Self {
            bin_size: 64,
            neighbor_radius: 1,
            gap_thr: 4,
            color_tol_r: 4,
            color_tol_g: 4,
            color_tol_b: 4,
            enable_area_ratio: true,
            area_ratio_thr: 30,
            enable_iou: true,
            iou_num: 3,
            iou_den: 10,
        }
    }
}

pub fn l4_merge_final(boxes: &[OutBox], cfg: &L4MergeCfg) -> Vec<OutBox> {
    if boxes.is_empty() {
        return Vec::new();
    }
    let n = boxes.len();
    if n == 1 {
        return vec![boxes[0]];
    }

    let mut uf = UnionFind::new(n);
    let bin_map = build_bin_map(boxes, cfg.bin_size.max(1));
    let neighbor_radius = cfg.neighbor_radius.max(0);

    for i in 0..n {
        let (bx, by) = box_bin_center(boxes[i], cfg.bin_size.max(1));
        for ny in neighbors(by, neighbor_radius) {
            for nx in neighbors(bx, neighbor_radius) {
                if let Some(cands) = bin_map.get(&(nx, ny)) {
                    for &j in cands {
                        if j <= i {
                            continue;
                        }
                        if !color_close_565(
                            boxes[i].color565 as u16,
                            boxes[j].color565 as u16,
                            cfg.color_tol_r,
                            cfg.color_tol_g,
                            cfg.color_tol_b,
                        ) {
                            continue;
                        }
                        if !near_aabb(boxes[i], boxes[j], cfg.gap_thr) {
                            continue;
                        }
                        if cfg.enable_area_ratio && !area_ratio_ok(boxes[i], boxes[j], cfg.area_ratio_thr) {
                            continue;
                        }
                        if cfg.enable_iou && !iou_ok(boxes[i], boxes[j], cfg.iou_num, cfg.iou_den) {
                            continue;
                        }
                        uf.union(i, j);
                    }
                }
            }
        }
    }

    let mut comps: Vec<Option<CompAcc>> = (0..n).map(|_| None).collect();
    for (i, b) in boxes.iter().enumerate() {
        let r = uf.find(i);
        let acc = comps[r].get_or_insert_with(CompAcc::default);
        acc.minx = acc.minx.min(b.minx());
        acc.miny = acc.miny.min(b.miny());
        acc.maxx = acc.maxx.max(b.maxx());
        acc.maxy = acc.maxy.max(b.maxy());
        if i < acc.rep_i {
            acc.rep_i = i;
            acc.rep_color = b.color565 & 0xFFFF;
            acc.rep_flags = b.flags;
        }
    }

    let mut out: Vec<OutBox> = Vec::new();
    for acc in comps.into_iter().flatten() {
        if acc.maxx <= acc.minx || acc.maxy <= acc.miny {
            continue;
        }
        out.push(OutBox::from_xy(
            acc.minx,
            acc.miny,
            acc.maxx,
            acc.maxy,
            acc.rep_color,
            acc.rep_flags,
        ));
    }
    out
}

fn neighbors(center: i32, radius: i32) -> RangeInclusive<i32> {
    (center - radius)..=(center + radius)
}

fn build_bin_map(boxes: &[OutBox], bin_size: u32) -> BTreeMap<(i32, i32), Vec<usize>> {
    let mut map: BTreeMap<(i32, i32), Vec<usize>> = BTreeMap::new();
    for (i, &b) in boxes.iter().enumerate() {
        let key = box_bin_center(b, bin_size);
        map.entry(key).or_default().push(i);
    }
    map
}

fn box_bin_center(b: OutBox, bin_size: u32) -> (i32, i32) {
    let bs = bin_size.max(1) as i64;
    let cx = ((b.minx() as i64) + (b.maxx() as i64)) / 2;
    let cy = ((b.miny() as i64) + (b.maxy() as i64)) / 2;
    ((cx / bs) as i32, (cy / bs) as i32)
}

fn rgb565(c: u16) -> (i32, i32, i32) {
    let r5 = ((c >> 11) & 0x1F) as i32;
    let g6 = ((c >> 5) & 0x3F) as i32;
    let b5 = (c & 0x1F) as i32;
    (r5, g6, b5)
}

fn color_close_565(a: u16, b: u16, tol_r: i32, tol_g: i32, tol_b: i32) -> bool {
    let (ar, ag, ab) = rgb565(a);
    let (br, bg, bb) = rgb565(b);
    (ar - br).abs() <= tol_r && (ag - bg).abs() <= tol_g && (ab - bb).abs() <= tol_b
}

fn near_aabb(a: OutBox, b: OutBox, gap_thr: i32) -> bool {
    let ax0 = a.minx() as i32;
    let ay0 = a.miny() as i32;
    let ax1 = a.maxx() as i32;
    let ay1 = a.maxy() as i32;
    let bx0 = b.minx() as i32;
    let by0 = b.miny() as i32;
    let bx1 = b.maxx() as i32;
    let by1 = b.maxy() as i32;

    let overlap_x = (ax1.min(bx1) - ax0.max(bx0)).max(0);
    let overlap_y = (ay1.min(by1) - ay0.max(by0)).max(0);
    let gap_x = (ax0 - bx1).max(bx0 - ax1).max(0);
    let gap_y = (ay0 - by1).max(by0 - ay1).max(0);

    (overlap_x > 0 && overlap_y > 0)
        || (gap_x <= gap_thr && overlap_y > 0)
        || (gap_y <= gap_thr && overlap_x > 0)
}

fn area_u64(b: OutBox) -> u64 {
    let w = b.w() as u64;
    let h = b.h() as u64;
    w * h
}

fn area_ratio_ok(a: OutBox, b: OutBox, ratio_thr: u32) -> bool {
    if ratio_thr == 0 {
        return true;
    }
    let aa = area_u64(a);
    let ab = area_u64(b);
    if aa == 0 || ab == 0 {
        return false;
    }
    let big = aa.max(ab);
    let small = aa.min(ab).max(1);
    big / small <= ratio_thr as u64
}

fn iou_ok(a: OutBox, b: OutBox, num: u32, den: u32) -> bool {
    if den == 0 {
        return true;
    }
    let ax0 = a.minx();
    let ay0 = a.miny();
    let ax1 = a.maxx();
    let ay1 = a.maxy();
    let bx0 = b.minx();
    let by0 = b.miny();
    let bx1 = b.maxx();
    let by1 = b.maxy();
    let ox = ax1.min(bx1).saturating_sub(ax0.max(bx0)) as u64;
    let oy = ay1.min(by1).saturating_sub(ay0.max(by0)) as u64;
    let inter = ox * oy;
    if inter == 0 {
        return false;
    }
    let aa = area_u64(a);
    let ab = area_u64(b);
    let union = aa + ab - inter;
    if union == 0 {
        return false;
    }
    (inter as u128) * (den as u128) >= (union as u128) * (num as u128)
}

#[derive(Clone, Copy, Debug)]
struct CompAcc {
    minx: u32,
    miny: u32,
    maxx: u32,
    maxy: u32,
    rep_i: usize,
    rep_color: u32,
    rep_flags: u32,
}

impl Default for CompAcc {
    fn default() -> Self {
        Self {
            minx: u32::MAX,
            miny: u32::MAX,
            maxx: 0,
            maxy: 0,
            rep_i: usize::MAX,
            rep_color: 0,
            rep_flags: 0,
        }
    }
}

struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self { parent: (0..n).collect() }
    }

    fn find(&mut self, x: usize) -> usize {
        let mut root = x;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        let mut cur = x;
        while self.parent[cur] != cur {
            let p = self.parent[cur];
            self.parent[cur] = root;
            cur = p;
        }
        root
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        // Deterministic: smallest root wins.
        if ra < rb {
            self.parent[rb] = ra;
        } else {
            self.parent[ra] = rb;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(x0: u32, y0: u32, x1: u32, y1: u32, c: u32) -> OutBox {
        OutBox::from_xy(x0, y0, x1, y1, c, 0)
    }

    #[test]
    fn merges_same_color_near() {
        let cfg = L4MergeCfg { enable_iou: false, enable_area_ratio: false, ..Default::default() };
        let boxes = vec![b(0, 0, 10, 10, 0x07E0), b(12, 0, 20, 10, 0x07E0)];
        let out = l4_merge_final(&boxes, &cfg);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].minx(), 0);
        assert_eq!(out[0].maxx(), 20);
    }

    #[test]
    fn does_not_merge_different_color() {
        let cfg = L4MergeCfg { enable_iou: false, enable_area_ratio: false, ..Default::default() };
        let boxes = vec![b(0, 0, 10, 10, 0x07E0), b(12, 0, 20, 10, 0xF800)];
        let out = l4_merge_final(&boxes, &cfg);
        assert_eq!(out.len(), 2);
    }
}

