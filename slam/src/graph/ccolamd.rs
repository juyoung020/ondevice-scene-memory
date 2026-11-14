//! CCOLAMD 2.x (GTSAM 4.2 gtsam/3rdparty/CCOLAMD/Source/ccolamd.c, BSD-3) 를 GTSAM 이 부르는 그대로 옮김:
//! gtsam::Ordering::Colamd(VariableIndex) → ColamdConstrained(cmember 모두 0) → ccolamd(knobs: dense 행·열 = -1, aggressive = 1, LU 아님).
//! 정수만 쓰는 알고리즘이라 원본과 같은 순서가 나온다(배열 길이 Alen 도 원본과 같게 잡아 쓰레기 수집 시점까지 같음).
//! 원본에서 A 안에 겹쳐 두던 Row·Col·Front 배열은 따로 둔다(서로 겹쳐 쓰이는 일이 없음을 확인).

const EMPTY: i32 = -1;

#[derive(Clone, Copy, Default)]
struct Col {
    start: i32,
    length: i32,
    /// thickness | parent
    s1: i32,
    /// score | order
    s2: i32,
    /// headhash | hash | prev
    s3: i32,
    /// degree_next | hash_next
    s4: i32,
    nextcol: i32,
    lastcol: i32,
}

#[derive(Clone, Copy, Default)]
struct Row {
    start: i32,
    length: i32,
    /// degree | p
    s1: i32,
    /// mark | first_column
    s2: i32,
    thickness: i32,
    front: i32,
}

fn clear_mark(tag_mark: i32, max_mark: i32, row: &mut [Row]) -> i32 {
    if tag_mark <= 0 || tag_mark >= max_mark {
        for r in row.iter_mut() {
            if r.s2 >= 0 {
                r.s2 = 0;
            }
        }
        return 1;
    }
    tag_mark
}

fn garbage_collection(row: &mut [Row], col: &mut [Col], a: &mut [i32], pfree: usize) -> usize {
    let mut pdest = 0usize;
    for c in 0..col.len() {
        if col[c].start >= 0 {
            let mut psrc = col[c].start as usize;
            col[c].start = pdest as i32;
            let length = col[c].length;
            for _ in 0..length {
                let r = a[psrc];
                psrc += 1;
                if row[r as usize].s2 >= 0 {
                    a[pdest] = r;
                    pdest += 1;
                }
            }
            col[c].length = pdest as i32 - col[c].start;
        }
    }
    for r in 0..row.len() {
        if row[r].s2 < 0 || row[r].length == 0 {
            row[r].s2 = -1;
        } else {
            let psrc = row[r].start as usize;
            row[r].s2 = a[psrc];
            a[psrc] = -(r as i32) - 1;
        }
    }
    let mut psrc = pdest;
    while psrc < pfree {
        let v = a[psrc];
        psrc += 1;
        if v < 0 {
            psrc -= 1;
            let r = (-v - 1) as usize;
            a[psrc] = row[r].s2;
            row[r].start = pdest as i32;
            let length = row[r].length;
            for _ in 0..length {
                let c = a[psrc];
                psrc += 1;
                if col[c as usize].start >= 0 {
                    a[pdest] = c;
                    pdest += 1;
                }
            }
            row[r].length = pdest as i32 - row[r].start;
        }
    }
    pdest
}

fn detect_super_cols(col: &mut [Col], a: &[i32], head: &mut [i32], row_start: usize, row_length: usize) {
    for rp in row_start..row_start + row_length {
        let c0 = a[rp] as usize;
        if col[c0].start < 0 {
            continue;
        }
        let hash = col[c0].s3 as usize;
        let head_column = head[hash];
        let first_col = if head_column > EMPTY { col[head_column as usize].s3 } else { -(head_column + 2) };
        let mut super_c = first_col;
        while super_c != EMPTY {
            let sc = super_c as usize;
            let length = col[sc].length;
            let mut prev_c = super_c;
            let mut c = col[sc].s4;
            while c != EMPTY {
                let cu = c as usize;
                if col[cu].length != length || col[cu].s2 != col[sc].s2 {
                    prev_c = c;
                    c = col[cu].s4;
                    continue;
                }
                let (p1, p2) = (col[sc].start as usize, col[cu].start as usize);
                let mut i = 0;
                while i < length as usize {
                    if a[p1 + i] != a[p2 + i] {
                        break;
                    }
                    i += 1;
                }
                if i != length as usize {
                    prev_c = c;
                    c = col[cu].s4;
                    continue;
                }
                col[sc].s1 += col[cu].s1;
                col[cu].s1 = super_c;
                col[cu].start = -2;
                col[cu].s2 = EMPTY;
                col[prev_c as usize].s4 = col[cu].s4;
                let lc = col[sc].lastcol as usize;
                col[lc].nextcol = c;
                col[sc].lastcol = col[cu].lastcol;
                c = col[cu].s4;
            }
            super_c = col[sc].s4;
        }
        if head_column > EMPTY {
            col[head_column as usize].s3 = EMPTY;
        } else {
            head[hash] = EMPTY;
        }
    }
}

fn post_tree(root: i32, mut k: i32, child: &mut [i32], sibling: &[i32], order: &mut [i32], stack: &mut [i32]) -> i32 {
    let mut head: i32 = 0;
    stack[0] = root;
    while head >= 0 {
        let i = stack[head as usize] as usize;
        if child[i] != EMPTY {
            let mut f = child[i];
            while f != EMPTY {
                head += 1;
                f = sibling[f as usize];
            }
            let mut h = head;
            let mut f = child[i];
            while f != EMPTY {
                stack[h as usize] = f;
                h -= 1;
                f = sibling[f as usize];
            }
            child[i] = EMPTY;
        } else {
            head -= 1;
            order[i] = k;
            k += 1;
        }
    }
    k
}

/// 열(변수) n_col 개, 행(인자) n_row 개. cols[j] = 변수 j 가 들어간 인자 번호들(오름차순, VariableIndex 순서).
/// 돌려주는 perm[k] = k 번째로 소거할 변수 번호
pub fn colamd(n_row: usize, cols: &[Vec<i32>]) -> Vec<usize> {
    let n_col = cols.len();
    if n_col == 0 {
        return vec![];
    }
    if n_col == 1 {
        return vec![0];
    }
    let nnz: usize = cols.iter().map(|c| c.len()).sum();
    // Alen = ccolamd_recommended(nnz, n_row, n_col) 에서 ccolamd2 가 Row·Col·Front 몫을 뺀 길이
    let alen = (2 * nnz).max(4 * n_col) + n_col + nnz / 5;
    let mut a = vec![0i32; alen];
    let mut p = vec![0i32; n_col + 1];
    let mut cnt = 0;
    for (j, c) in cols.iter().enumerate() {
        for &f in c {
            a[cnt] = f;
            cnt += 1;
        }
        p[j + 1] = cnt as i32;
    }
    let (nr, nc) = (n_row as i32, n_col as i32);
    let mut col = vec![Col::default(); n_col + 1];
    let mut row = vec![Row::default(); n_row + 1];
    // cset: 집합 하나(cmember 모두 0) → cset = n_col-1 .. 0
    let cset: Vec<i32> = (0..nc).rev().collect();
    let cset_start = [0, nc];
    let mut dead_cols = [0i32; 1];

    // ---- init_rows_cols ----
    for c in 0..n_col {
        col[c] = Col { start: p[c], length: p[c + 1] - p[c], s1: 1, s2: 0, s3: EMPTY, s4: EMPTY, nextcol: EMPTY, lastcol: c as i32 };
    }
    let mut jumbled = false;
    for r in 0..n_row {
        row[r].length = 0;
        row[r].s2 = -1;
        row[r].thickness = 1;
        row[r].front = EMPTY;
    }
    for c in 0..n_col {
        let mut last_row = -1;
        for cp in p[c] as usize..p[c + 1] as usize {
            let r = a[cp] as usize;
            if a[cp] <= last_row || row[r].s2 == c as i32 {
                jumbled = true;
            }
            if row[r].s2 != c as i32 {
                row[r].length += 1;
            } else {
                col[c].length -= 1;
            }
            row[r].s2 = c as i32;
            last_row = a[cp];
        }
    }
    row[0].start = p[n_col];
    row[0].s1 = row[0].start;
    row[0].s2 = -1;
    for r in 1..n_row {
        row[r].start = row[r - 1].start + row[r - 1].length;
        row[r].s1 = row[r].start;
        row[r].s2 = -1;
    }
    for c in 0..n_col {
        for cp in p[c] as usize..p[c + 1] as usize {
            let r = a[cp] as usize;
            if jumbled {
                if row[r].s2 != c as i32 {
                    a[row[r].s1 as usize] = c as i32;
                    row[r].s1 += 1;
                    row[r].s2 = c as i32;
                }
            } else {
                a[row[r].s1 as usize] = c as i32;
                row[r].s1 += 1;
            }
        }
    }
    for r in 0..n_row {
        row[r].s2 = 0;
        row[r].s1 = row[r].length;
    }
    if jumbled {
        col[0].start = 0;
        p[0] = 0;
        for c in 1..n_col {
            col[c].start = col[c - 1].start + col[c - 1].length;
            p[c] = col[c].start;
        }
        for r in 0..n_row {
            for rp in row[r].start as usize..(row[r].start + row[r].length) as usize {
                let c = a[rp] as usize;
                a[p[c] as usize] = r as i32;
                p[c] += 1;
            }
        }
    }

    // ---- init_scoring (knobs dense = -1) ----
    let head = &mut p; // head = p (길이 n_col+1)
    let dense_row_count = nc - 1;
    let dense_col_count = nr - 1;
    let mut max_deg = 0;
    let mut n_col2 = nc;
    head[0] = cset_start[1];
    for c in (0..n_col).rev() {
        if col[c].length == 0 {
            head[0] -= 1;
            col[c].s2 = head[0];
            n_col2 -= 1;
            dead_cols[0] += 1;
            col[c].start = -1;
        }
    }
    for c in (0..n_col).rev() {
        if col[c].start < 0 {
            continue;
        }
        if col[c].length > dense_col_count {
            head[0] -= 1;
            col[c].s2 = head[0];
            n_col2 -= 1;
            dead_cols[0] += 1;
            for cp in col[c].start as usize..(col[c].start + col[c].length) as usize {
                row[a[cp] as usize].s1 -= 1;
            }
            col[c].start = -1;
        }
    }
    for r in 0..n_row {
        let deg = row[r].s1;
        if deg > dense_row_count || deg == 0 {
            row[r].s2 = -1;
            row[r].thickness = 0;
        } else {
            max_deg = max_deg.max(deg);
        }
    }
    for c in (0..n_col).rev() {
        if col[c].start < 0 {
            continue;
        }
        let mut score = 0;
        let st = col[c].start as usize;
        let mut new_cp = st;
        for cp in st..st + col[c].length as usize {
            let r = a[cp];
            if row[r as usize].s2 < 0 {
                continue;
            }
            a[new_cp] = r;
            new_cp += 1;
            score += row[r as usize].s1 - 1;
            score = score.min(nc);
        }
        let col_length = (new_cp - st) as i32;
        if col_length == 0 {
            head[0] -= 1;
            col[c].s2 = head[0];
            n_col2 -= 1;
            dead_cols[0] += 1;
            col[c].start = -1;
        } else {
            col[c].length = col_length;
            col[c].s2 = score;
        }
    }
    for c in 0..=n_col {
        head[c] = EMPTY;
    }

    // ---- find_ordering ----
    let mut front_npivcol = vec![0i32; n_col + 1];
    let mut front_nrows = vec![0i32; n_col + 1];
    let mut front_ncols = vec![0i32; n_col + 1];
    let mut front_parent = vec![EMPTY; n_col + 1];
    let mut front_cols = vec![EMPTY; n_col + 1];
    let mut in_front = vec![EMPTY; n_row + 1];
    let alen_i = alen as i32;
    let mut pfree = 2 * nnz as i32;
    let max_mark = i32::MAX - nc;
    let mut tag_mark = clear_mark(0, max_mark, &mut row[..n_row]);
    let mut min_score = 0;
    let mut current_set: i32 = -1;
    let mut deadcol = 0;
    let mut nfr: i32 = 0;
    let mut k: i32 = 0;
    'outer: while k < nc {
        while k + deadcol == cset_start[(current_set + 1) as usize] {
            current_set += 1;
            k += deadcol;
            deadcol = 0;
            if k == nc {
                break 'outer;
            }
            min_score = nc;
            let mut colstart = cset_start[current_set as usize];
            let colend = cset_start[(current_set + 1) as usize];
            while colstart < colend {
                let c = cset[colstart as usize] as usize;
                colstart += 1;
                if col[c].start < 0 {
                    if col[c].s2 != EMPTY {
                        deadcol += 1;
                    }
                    continue;
                }
                let score = col[c].s2;
                let next_col = head[score as usize];
                col[c].s3 = EMPTY;
                col[c].s4 = next_col;
                if next_col != EMPTY {
                    col[next_col as usize].s3 = c as i32;
                }
                head[score as usize] = c as i32;
                min_score = min_score.min(score);
            }
        }
        while head[min_score as usize] == EMPTY && min_score < nc {
            min_score += 1;
        }
        let pivot_col = head[min_score as usize] as usize;
        let next_col = col[pivot_col].s4;
        head[min_score as usize] = next_col;
        if next_col != EMPTY {
            col[next_col as usize].s3 = EMPTY;
        }
        let pivot_col_score = col[pivot_col].s2;
        col[pivot_col].s2 = k;
        let mut pivot_col_thickness = col[pivot_col].s1;
        k += pivot_col_thickness;
        let needed_memory = pivot_col_score.min(nc - k);
        if pfree + needed_memory >= alen_i {
            pfree = garbage_collection(&mut row[..n_row], &mut col[..n_col], &mut a, pfree as usize) as i32;
            tag_mark = clear_mark(0, max_mark, &mut row[..n_row]);
        }
        let pivot_row_start = pfree;
        let mut pivot_row_degree = 0;
        let mut pivot_row_thickness = 0;
        col[pivot_col].s1 = -pivot_col_thickness;
        let pcs = col[pivot_col].start as usize;
        for cp in pcs..pcs + col[pivot_col].length as usize {
            let r = a[cp] as usize;
            if row[r].s2 >= 0 {
                pivot_row_thickness += row[r].thickness;
                let rs = row[r].start as usize;
                for rp in rs..rs + row[r].length as usize {
                    let c = a[rp] as usize;
                    let ct = col[c].s1;
                    if ct > 0 && col[c].start >= 0 {
                        col[c].s1 = -ct;
                        a[pfree as usize] = c as i32;
                        pfree += 1;
                        pivot_row_degree += ct;
                    }
                }
            }
        }
        col[pivot_col].s1 = pivot_col_thickness;
        max_deg = max_deg.max(pivot_row_degree);
        for cp in pcs..pcs + col[pivot_col].length as usize {
            let r = a[cp] as usize;
            if row[r].s2 >= 0 {
                if row[r].front != EMPTY {
                    front_parent[row[r].front as usize] = nfr;
                } else {
                    in_front[r] = nfr;
                }
            }
            row[r].s2 = -1;
            row[r].thickness = 0;
        }
        let pivot_row_length = pfree - pivot_row_start;
        let pivot_row = if pivot_row_length > 0 { a[col[pivot_col].start as usize] } else { EMPTY };
        let (prs, prl) = (pivot_row_start as usize, pivot_row_length as usize);
        for rp in prs..prs + prl {
            let c = a[rp] as usize;
            let col_thickness = -col[c].s1;
            col[c].s1 = col_thickness;
            // cmember 모두 0 == current_set(0)
            {
                let cur_score = col[c].s2;
                let prev_col = col[c].s3;
                let next_col = col[c].s4;
                if prev_col == EMPTY {
                    head[cur_score as usize] = next_col;
                } else {
                    col[prev_col as usize].s4 = next_col;
                }
                if next_col != EMPTY {
                    col[next_col as usize].s3 = prev_col;
                }
            }
            let cs = col[c].start as usize;
            for cp in cs..cs + col[c].length as usize {
                let r = a[cp] as usize;
                let row_mark = row[r].s2;
                if row_mark < 0 {
                    continue;
                }
                let mut set_difference = row_mark - tag_mark;
                if set_difference < 0 {
                    set_difference = row[r].s1;
                }
                set_difference -= col_thickness;
                if set_difference == 0 {
                    // aggressive
                    if row[r].front != EMPTY {
                        front_parent[row[r].front as usize] = nfr;
                    } else {
                        in_front[r] = nfr;
                    }
                    row[r].s2 = -1;
                    pivot_row_thickness += row[r].thickness;
                    row[r].thickness = 0;
                } else {
                    row[r].s2 = set_difference + tag_mark;
                }
            }
        }
        for rp in prs..prs + prl {
            let c = a[rp] as usize;
            let mut hash: u32 = 0;
            let mut cur_score = 0;
            let cs = col[c].start as usize;
            let mut new_cp = cs;
            for cp in cs..cs + col[c].length as usize {
                let r = a[cp];
                let row_mark = row[r as usize].s2;
                if row_mark < 0 {
                    continue;
                }
                a[new_cp] = r;
                new_cp += 1;
                hash = hash.wrapping_add(r as u32);
                cur_score += row_mark - tag_mark;
                cur_score = cur_score.min(nc);
            }
            col[c].length = (new_cp - cs) as i32;
            if col[c].length == 0 {
                col[c].start = -1;
                pivot_row_degree -= col[c].s1;
                col[c].s2 = k;
                k += col[c].s1;
                pivot_col_thickness += col[c].s1;
                let lc = col[c].lastcol as usize;
                col[lc].nextcol = front_cols[nfr as usize];
                front_cols[nfr as usize] = c as i32;
            } else {
                col[c].s2 = cur_score;
                hash %= (n_col + 1) as u32;
                let head_column = head[hash as usize];
                let first_col;
                if head_column > EMPTY {
                    first_col = col[head_column as usize].s3;
                    col[head_column as usize].s3 = c as i32;
                } else {
                    first_col = -(head_column + 2);
                    head[hash as usize] = -(c as i32 + 2);
                }
                col[c].s4 = first_col;
                col[c].s3 = hash as i32;
            }
        }
        detect_super_cols(&mut col, &a, head, prs, prl);
        col[pivot_col].start = -1;
        let lc = col[pivot_col].lastcol as usize;
        col[lc].nextcol = front_cols[nfr as usize];
        front_cols[nfr as usize] = pivot_col as i32;
        tag_mark = clear_mark(tag_mark + max_deg + 1, max_mark, &mut row[..n_row]);
        let mut new_rp = prs;
        for rp in prs..prs + prl {
            let c = a[rp];
            let cu = c as usize;
            if col[cu].start < 0 {
                continue;
            }
            a[new_rp] = c;
            new_rp += 1;
            let at = (col[cu].start + col[cu].length) as usize;
            a[at] = pivot_row;
            col[cu].length += 1;
            let mut cur_score = col[cu].s2 + pivot_row_degree;
            let max_score = nc - k - col[cu].s1;
            cur_score -= col[cu].s1;
            cur_score = cur_score.min(max_score);
            col[cu].s2 = cur_score;
            let next_col = head[cur_score as usize];
            col[cu].s4 = next_col;
            col[cu].s3 = EMPTY;
            if next_col != EMPTY {
                col[next_col as usize].s3 = c;
            }
            head[cur_score as usize] = c;
            min_score = min_score.min(cur_score);
        }
        let nf = nfr as usize;
        front_npivcol[nf] = pivot_col_thickness;
        front_nrows[nf] = pivot_row_thickness;
        front_ncols[nf] = pivot_col_thickness + pivot_row_degree;
        front_parent[nf] = EMPTY;
        pivot_row_thickness -= pivot_col_thickness;
        pivot_row_thickness = pivot_row_thickness.max(0);
        if pivot_row_degree > 0 {
            let pr = pivot_row as usize;
            row[pr].start = pivot_row_start;
            row[pr].length = (new_rp - prs) as i32;
            row[pr].s1 = pivot_row_degree;
            row[pr].s2 = 0;
            row[pr].thickness = pivot_row_thickness;
            row[pr].front = nfr;
        }
        nfr += 1;
    }
    let _ = (n_col2, &in_front, &front_nrows);

    // ---- fsize · postorder · apply_order ----
    let nn = nfr as usize;
    let mut fsize = vec![EMPTY; nn];
    for j in 0..nn {
        if front_npivcol[j] > 0 {
            let parent = front_parent[j];
            let (r, c) = (front_nrows[j], front_ncols[j]);
            let (dr, dc) = (r as f64, c as f64);
            let frsize = if !((dr * dc) * (1.0 + 1e-8) <= i32::MAX as f64) || (dr * dc).is_nan() { i32::MAX } else { r.wrapping_mul(c) };
            fsize[j] = fsize[j].max(frsize);
            if parent != EMPTY {
                fsize[parent as usize] = fsize[parent as usize].max(fsize[j]);
            }
        }
    }
    let mut child = vec![EMPTY; nn];
    let mut sibling = vec![EMPTY; nn];
    for j in (0..nn).rev() {
        if front_npivcol[j] > 0 {
            let parent = front_parent[j];
            if parent != EMPTY {
                sibling[j] = child[parent as usize];
                child[parent as usize] = j as i32;
            }
        }
    }
    for i in 0..nn {
        if front_npivcol[i] > 0 && child[i] != EMPTY {
            let (mut fprev, mut maxfrsize, mut bigfprev, mut bigf) = (EMPTY, EMPTY, EMPTY, EMPTY);
            let mut f = child[i];
            while f != EMPTY {
                let frsize = fsize[f as usize];
                if frsize >= maxfrsize {
                    maxfrsize = frsize;
                    bigfprev = fprev;
                    bigf = f;
                }
                fprev = f;
                f = sibling[f as usize];
            }
            let fnext = sibling[bigf as usize];
            if fnext != EMPTY {
                if bigfprev == EMPTY {
                    child[i] = fnext;
                } else {
                    sibling[bigfprev as usize] = fnext;
                }
                sibling[bigf as usize] = EMPTY;
                sibling[fprev as usize] = bigf;
            }
        }
    }
    let mut order = vec![EMPTY; nn];
    let mut stack = vec![0i32; nn];
    let mut kk = 0;
    for i in 0..nn {
        if front_parent[i] == EMPTY && front_npivcol[i] > 0 {
            kk = post_tree(i as i32, kk, &mut child, &sibling, &mut order, &mut stack);
        }
    }
    // Front_cols 만 순서대로(나머지 Front 배열은 결과에 안 쓰임)
    let mut temp = vec![0i32; nn];
    for i in 0..nn {
        if order[i] != EMPTY {
            temp[order[i] as usize] = front_cols[i];
        }
    }
    let front_cols_o = temp;
    // 순서 매기기
    let mut amap = vec![EMPTY; n_col];
    let mut perm = vec![0i32; n_col];
    let mut kq: i32 = 0;
    for i in 0..nn {
        // set1/set2 는 모두 0: dead_cols 를 더하지 않음
        let mut c = front_cols_o[i];
        while c != EMPTY {
            perm[kq as usize] = c;
            amap[c as usize] = kq;
            kq += 1;
            c = col[c as usize].nextcol;
        }
    }
    if n_col2 < nc {
        for c in 0..n_col {
            if amap[c] == EMPTY {
                let kd = col[c].s2;
                perm[kd as usize] = c as i32;
                amap[c] = kd;
            }
        }
    }
    perm.into_iter().map(|v| v as usize).collect()
}
