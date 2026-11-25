//! libstdc++ std::sort(내부 정렬 · 힙 정렬 · 삽입 정렬)를 그대로 옮긴 것. std::sort 는 안정 정렬이 아니라
//! 비교자가 같다고 보는 원소의 순서가 이 알고리즘에 달려 있다 → 원본과 비트 동일하려면 같은 알고리즘이 필요.
//! less(a, b) = 원본 비교자 comp(a, b).

const THRESHOLD: usize = 16;

pub fn sort_by<T: Clone>(v: &mut [T], less: &mut impl FnMut(&T, &T) -> bool) {
    let n = v.len();
    if n == 0 {
        return;
    }
    let depth = 2 * (usize::BITS - 1 - n.leading_zeros()) as usize;
    introsort_loop(v, 0, n, depth, less);
    final_insertion_sort(v, 0, n, less);
}

fn introsort_loop<T: Clone>(v: &mut [T], first: usize, mut last: usize, mut depth: usize, less: &mut impl FnMut(&T, &T) -> bool) {
    while last - first > THRESHOLD {
        if depth == 0 {
            heap_sort(&mut v[first..last], less);
            return;
        }
        depth -= 1;
        let cut = partition_pivot(v, first, last, less);
        introsort_loop(v, cut, last, depth, less);
        last = cut;
    }
}

fn partition_pivot<T: Clone>(v: &mut [T], first: usize, last: usize, less: &mut impl FnMut(&T, &T) -> bool) -> usize {
    let mid = first + (last - first) / 2;
    move_median_to_first(v, first, first + 1, mid, last - 1, less);
    unguarded_partition(v, first + 1, last, first, less)
}

fn move_median_to_first<T>(v: &mut [T], r: usize, a: usize, b: usize, c: usize, less: &mut impl FnMut(&T, &T) -> bool) {
    if less(&v[a], &v[b]) {
        if less(&v[b], &v[c]) {
            v.swap(r, b)
        } else if less(&v[a], &v[c]) {
            v.swap(r, c)
        } else {
            v.swap(r, a)
        }
    } else if less(&v[a], &v[c]) {
        v.swap(r, a)
    } else if less(&v[b], &v[c]) {
        v.swap(r, c)
    } else {
        v.swap(r, b)
    }
}

fn unguarded_partition<T>(v: &mut [T], mut first: usize, mut last: usize, pivot: usize, less: &mut impl FnMut(&T, &T) -> bool) -> usize {
    loop {
        while less(&v[first], &v[pivot]) {
            first += 1;
        }
        last -= 1;
        while less(&v[pivot], &v[last]) {
            last -= 1;
        }
        if first >= last {
            return first;
        }
        v.swap(first, last);
        first += 1;
    }
}

fn final_insertion_sort<T: Clone>(v: &mut [T], first: usize, last: usize, less: &mut impl FnMut(&T, &T) -> bool) {
    if last - first > THRESHOLD {
        insertion_sort(v, first, first + THRESHOLD, less);
        for i in first + THRESHOLD..last {
            unguarded_linear_insert(v, i, less);
        }
    } else {
        insertion_sort(v, first, last, less);
    }
}

fn insertion_sort<T: Clone>(v: &mut [T], first: usize, last: usize, less: &mut impl FnMut(&T, &T) -> bool) {
    if first == last {
        return;
    }
    for i in first + 1..last {
        if less(&v[i], &v[first]) {
            v[first..=i].rotate_right(1);
        } else {
            unguarded_linear_insert(v, i, less);
        }
    }
}

fn unguarded_linear_insert<T: Clone>(v: &mut [T], mut last: usize, less: &mut impl FnMut(&T, &T) -> bool) {
    let val = v[last].clone();
    let mut next = last - 1;
    while less(&val, &v[next]) {
        v[last] = v[next].clone();
        last = next;
        next = next.wrapping_sub(1);
    }
    v[last] = val;
}

fn heap_sort<T: Clone>(v: &mut [T], less: &mut impl FnMut(&T, &T) -> bool) {
    let len = v.len();
    if len >= 2 {
        let mut parent = (len - 2) / 2;
        loop {
            let val = v[parent].clone();
            adjust_heap(v, parent, len, val, less);
            if parent == 0 {
                break;
            }
            parent -= 1;
        }
    }
    let mut last = len;
    while last > 1 {
        last -= 1;
        let val = v[last].clone();
        v[last] = v[0].clone();
        adjust_heap(v, 0, last, val, less);
    }
}

fn adjust_heap<T: Clone>(v: &mut [T], mut hole: usize, len: usize, val: T, less: &mut impl FnMut(&T, &T) -> bool) {
    let top = hole;
    let mut second = hole;
    while len >= 1 && second < (len - 1) / 2 {
        second = 2 * (second + 1);
        if less(&v[second], &v[second - 1]) {
            second -= 1;
        }
        v[hole] = v[second].clone();
        hole = second;
    }
    if len & 1 == 0 && len >= 2 && second == (len - 2) / 2 {
        second = 2 * (second + 1);
        v[hole] = v[second - 1].clone();
        hole = second - 1;
    }
    // push_heap
    let mut parent = hole.wrapping_sub(1) / 2;
    while hole > top && less(&v[parent], &val) {
        v[hole] = v[parent].clone();
        hole = parent;
        parent = (hole.wrapping_sub(1)) / 2;
    }
    v[hole] = val;
}
