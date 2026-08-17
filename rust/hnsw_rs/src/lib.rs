// Rust port of submission/src/hnsw.cpp — same algorithm, same locking
// discipline, same SIMD width — for a language-level performance comparison.
// Modes ported: float, sq8, sq8pd (the shipping surface). sq4pd is parked in
// the C++ and intentionally not ported.

use std::cell::RefCell;
use std::cmp::Ordering as CmpOrdering;
use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};
use std::sync::Mutex;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// SIMD kernels — compile-time dispatch like simd.h (-C target-cpu=native).
// Early-exit cadence mirrors the C++ (partial reduce every 96 lanes).
// ---------------------------------------------------------------------------
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
mod kernels {
    use std::arch::x86_64::*;

    #[inline]
    unsafe fn reduce(v: __m512) -> f32 {
        _mm512_reduce_add_ps(v)
    }

    #[inline]
    pub unsafe fn l2(a: *const f32, b: *const f32, dim: usize, threshold: f32) -> f32 {
        let mut acc = _mm512_setzero_ps();
        let mut i = 0usize;
        while i + 16 <= dim {
            let d = _mm512_sub_ps(_mm512_loadu_ps(a.add(i)), _mm512_loadu_ps(b.add(i)));
            acc = _mm512_fmadd_ps(d, d, acc);
            if i % 96 == 80 {
                let s = reduce(acc);
                if s >= threshold {
                    return s;
                }
            }
            i += 16;
        }
        let mut s = reduce(acc);
        while i < dim {
            let d = *a.add(i) - *b.add(i);
            s += d * d;
            i += 1;
        }
        s
    }

    #[inline]
    pub unsafe fn l2_sq8(a: *const u8, q: *const f32, dim: usize, scale: f32, offset: f32, threshold: f32) -> f32 {
        let vs = _mm512_set1_ps(scale);
        let vo = _mm512_set1_ps(offset);
        let mut acc = _mm512_setzero_ps();
        let mut i = 0usize;
        while i + 16 <= dim {
            let bytes = _mm_loadu_si128(a.add(i) as *const __m128i);
            let f = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(bytes));
            let dq = _mm512_fmadd_ps(f, vs, vo);
            let d = _mm512_sub_ps(dq, _mm512_loadu_ps(q.add(i)));
            acc = _mm512_fmadd_ps(d, d, acc);
            if i % 96 == 80 {
                let s = reduce(acc);
                if s >= threshold {
                    return s;
                }
            }
            i += 16;
        }
        let mut s = reduce(acc);
        while i < dim {
            let d = *a.add(i) as f32 * scale + offset - *q.add(i);
            s += d * d;
            i += 1;
        }
        s
    }

    #[inline]
    pub unsafe fn l2_sq8pd(a: *const u8, q_shifted: *const f32, dim: usize, scale: *const f32, threshold: f32) -> f32 {
        let mut acc = _mm512_setzero_ps();
        let mut i = 0usize;
        while i + 16 <= dim {
            let bytes = _mm_loadu_si128(a.add(i) as *const __m128i);
            let f = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(bytes));
            let d = _mm512_fmsub_ps(f, _mm512_loadu_ps(scale.add(i)), _mm512_loadu_ps(q_shifted.add(i)));
            acc = _mm512_fmadd_ps(d, d, acc);
            if i % 96 == 80 {
                let s = reduce(acc);
                if s >= threshold {
                    return s;
                }
            }
            i += 16;
        }
        let mut s = reduce(acc);
        while i < dim {
            let d = *a.add(i) as f32 * *scale.add(i) - *q_shifted.add(i);
            s += d * d;
            i += 1;
        }
        s
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
mod kernels {
    #[inline]
    pub unsafe fn l2(a: *const f32, b: *const f32, dim: usize, threshold: f32) -> f32 {
        let mut s = 0.0f32;
        for i in 0..dim {
            let d = *a.add(i) - *b.add(i);
            s += d * d;
            if i % 16 == 15 && s >= threshold {
                return s;
            }
        }
        s
    }
    #[inline]
    pub unsafe fn l2_sq8(a: *const u8, q: *const f32, dim: usize, scale: f32, offset: f32, threshold: f32) -> f32 {
        let mut s = 0.0f32;
        for i in 0..dim {
            let d = *a.add(i) as f32 * scale + offset - *q.add(i);
            s += d * d;
            if i % 16 == 15 && s >= threshold {
                return s;
            }
        }
        s
    }
    #[inline]
    pub unsafe fn l2_sq8pd(a: *const u8, q_shifted: *const f32, dim: usize, scale: *const f32, threshold: f32) -> f32 {
        let mut s = 0.0f32;
        for i in 0..dim {
            let d = *a.add(i) as f32 * *scale.add(i) - *q_shifted.add(i);
            s += d * d;
            if i % 16 == 15 && s >= threshold {
                return s;
            }
        }
        s
    }
}

// ---------------------------------------------------------------------------
#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Float,
    Sq8,
    Sq8Pd,
}

// (distance, node) with total order — Rust's BinaryHeap needs Ord.
#[derive(Clone, Copy, PartialEq)]
struct P(f32, i32);
impl Eq for P {}
impl PartialOrd for P {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}
impl Ord for P {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        self.0.total_cmp(&other.0).then(self.1.cmp(&other.1))
    }
}

struct SearchTracker {
    visited_gen: Vec<u32>,
    current_gen: u32,
    rng: u64, // xorshift64* state
}
impl SearchTracker {
    fn new(n: usize, seed: u64) -> Self {
        Self { visited_gen: vec![0; n], current_gen: 1, rng: seed | 1 }
    }
    #[inline]
    fn next_f64(&mut self) -> f64 {
        // xorshift64*
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64
    }
}

const LOCK_POOL_SIZE: usize = 4096;

struct Inner {
    npts: usize,
    dim: usize,
    m: usize,
    mmax0: usize,
    efc: usize,
    ml: f64,
    max_level: i32,
    entry_point: i32,
    mode: Mode,
    heuristic: bool,
    is_building: bool,
    data: Vec<f32>,
    graph: Vec<Vec<Vec<i32>>>,
    codes: Vec<u8>,
    scale: f32,
    offset: f32,
    pd_scale: Vec<f32>,
    pd_offset: Vec<f32>,
    new_to_orig: Vec<i32>,
}

// Shared across build threads exactly like the C++ `this`: graph mutations
// are guarded by the striped lock pool, entry point by the global lock.
struct Shared {
    inner: *mut Inner,
    node_locks: Vec<Mutex<()>>,
    global: Mutex<()>,
    n_distances: AtomicI64,
}
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

thread_local! {
    static Q_SHIFT: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

impl Shared {
    #[inline]
    unsafe fn i(&self) -> &Inner {
        &*self.inner
    }
    #[inline]
    #[allow(clippy::mut_from_ref)]
    unsafe fn i_mut(&self) -> &mut Inner {
        &mut *self.inner
    }

    #[inline]
    unsafe fn dist_to(&self, node: i32, q: *const f32, threshold: f32) -> f32 {
        let s = self.i();
        kernels::l2(s.data.as_ptr().add(node as usize * s.dim), q, s.dim, threshold)
    }

    // Beam search at one layer; returns the UNORDERED candidate window.
    unsafe fn search_layer(&self, q: *const f32, mut _ep: i32, ef: usize, layer: usize, count: bool, tracker: &mut SearchTracker) -> Vec<P> {
        let s = self.i();
        let ep = _ep;
        tracker.current_gen = tracker.current_gen.wrapping_add(1);
        if tracker.current_gen == 0 {
            tracker.visited_gen.fill(0);
            tracker.current_gen = 1;
        }
        let gen = tracker.current_gen;
        tracker.visited_gen[ep as usize] = gen;

        let inf = f32::MAX;
        let q_shifted: *const f32 = if s.mode == Mode::Sq8Pd {
            Q_SHIFT.with(|b| {
                let mut b = b.borrow_mut();
                b.resize(s.dim, 0.0);
                for d in 0..s.dim {
                    b[d] = *q.add(d) - s.pd_offset[d];
                }
                b.as_ptr()
            })
        } else {
            std::ptr::null()
        };

        let d_ep = match s.mode {
            Mode::Sq8 => kernels::l2_sq8(s.codes.as_ptr().add(ep as usize * s.dim), q, s.dim, s.scale, s.offset, inf),
            Mode::Sq8Pd => kernels::l2_sq8pd(s.codes.as_ptr().add(ep as usize * s.dim), q_shifted, s.dim, s.pd_scale.as_ptr(), inf),
            Mode::Float => self.dist_to(ep, q, inf),
        };
        if count {
            self.n_distances.fetch_add(1, Ordering::Relaxed);
        }

        let mut c_heap = std::collections::BinaryHeap::new(); // max-heap of Reverse => min
        let mut w_heap = std::collections::BinaryHeap::new(); // max-heap: furthest on top
        c_heap.push(std::cmp::Reverse(P(d_ep, ep)));
        w_heap.push(P(d_ep, ep));

        let mut neighbors_copy: Vec<i32> = Vec::new();
        while let Some(&std::cmp::Reverse(P(d_c, c))) = c_heap.peek() {
            c_heap.pop();
            if d_c > w_heap.peek().unwrap().0 {
                break;
            }
            // During build: copy under the striped lock (concurrent inserts
            // mutate adjacency). At query time the graph is immutable.
            let neighbors: &[i32] = if s.is_building {
                let _g = self.node_locks[c as usize % LOCK_POOL_SIZE].lock().unwrap();
                neighbors_copy.clear();
                neighbors_copy.extend_from_slice(&s.graph[c as usize][layer]);
                &neighbors_copy
            } else {
                &s.graph[c as usize][layer]
            };

            let n_n = neighbors.len();
            for idx in 0..n_n {
                let e = neighbors[idx];
                if tracker.visited_gen[e as usize] == gen {
                    continue;
                }
                tracker.visited_gen[e as usize] = gen;

                if idx + 2 < n_n {
                    let pf = neighbors[idx + 2] as usize;
                    if s.mode != Mode::Float {
                        std::arch::x86_64::_mm_prefetch(s.codes.as_ptr().add(pf * s.dim) as *const i8, std::arch::x86_64::_MM_HINT_T0);
                    } else {
                        std::arch::x86_64::_mm_prefetch(s.data.as_ptr().add(pf * s.dim) as *const i8, std::arch::x86_64::_MM_HINT_T0);
                    }
                }

                let threshold = if w_heap.len() >= ef { w_heap.peek().unwrap().0 } else { inf };
                let d_e = match s.mode {
                    Mode::Sq8 => kernels::l2_sq8(s.codes.as_ptr().add(e as usize * s.dim), q, s.dim, s.scale, s.offset, threshold),
                    Mode::Sq8Pd => kernels::l2_sq8pd(s.codes.as_ptr().add(e as usize * s.dim), q_shifted, s.dim, s.pd_scale.as_ptr(), threshold),
                    Mode::Float => self.dist_to(e, q, threshold),
                };
                if count {
                    self.n_distances.fetch_add(1, Ordering::Relaxed);
                }
                if w_heap.len() < ef || d_e < w_heap.peek().unwrap().0 {
                    c_heap.push(std::cmp::Reverse(P(d_e, e)));
                    w_heap.push(P(d_e, e));
                    if w_heap.len() > ef {
                        w_heap.pop();
                    }
                }
            }
        }
        w_heap.into_vec()
    }

    unsafe fn select_from_sorted(&self, cand: &[P], m_max: usize) -> Vec<i32> {
        let s = self.i();
        let mut selected: Vec<i32> = Vec::with_capacity(m_max);
        if !s.heuristic || cand.len() <= m_max {
            for p in cand.iter().take(m_max) {
                selected.push(p.1);
            }
            return selected;
        }
        let mut discarded: Vec<i32> = Vec::new();
        for &P(d_base, e) in cand {
            if selected.len() >= m_max {
                break;
            }
            let pe = s.data.as_ptr().add(e as usize * s.dim);
            let mut diverse = true;
            for &sel in &selected {
                let d_es = kernels::l2(s.data.as_ptr().add(sel as usize * s.dim), pe, s.dim, d_base);
                if d_es < d_base {
                    diverse = false;
                    break;
                }
            }
            if diverse {
                selected.push(e);
            } else {
                discarded.push(e);
            }
        }
        let mut i = 0;
        while selected.len() < m_max && i < discarded.len() {
            selected.push(discarded[i]);
            i += 1;
        }
        selected
    }

    unsafe fn select_neighbors(&self, mut cand: Vec<P>, m_max: usize) -> Vec<i32> {
        cand.sort_unstable();
        self.select_from_sorted(&cand, m_max)
    }

    unsafe fn prune(&self, node: i32, layer: usize, m_max: usize) {
        let s = self.i_mut();
        let nbrs_len = s.graph[node as usize][layer].len();
        if nbrs_len <= m_max {
            return;
        }
        let q = s.data.as_ptr().add(node as usize * s.dim);
        let mut cand: Vec<P> = s.graph[node as usize][layer]
            .iter()
            .map(|&e| P(kernels::l2(s.data.as_ptr().add(e as usize * s.dim), q, s.dim, f32::MAX), e))
            .collect();
        cand.sort_unstable();
        s.graph[node as usize][layer] = self.select_from_sorted(&cand, m_max);
    }

    unsafe fn insert(&self, idx: i32, tracker: &mut SearchTracker) {
        let s = self.i_mut();
        let u: f64 = tracker.next_f64();
        let l = (((-(u + 1e-10).ln()) * s.ml) as i32).min(32);
        s.graph[idx as usize] = vec![Vec::new(); (l + 1) as usize];

        let (mut ep, max_l) = {
            let _g = self.global.lock().unwrap();
            (s.entry_point, s.max_level)
        };
        let q = s.data.as_ptr().add(idx as usize * s.dim);

        let mut lc = max_l;
        while lc > l {
            let w = self.search_layer(q, ep, 1, lc as usize, false, tracker);
            ep = w[0].1;
            lc -= 1;
        }
        let mut lc = l.min(max_l);
        while lc >= 0 {
            let m_max = if lc == 0 { s.mmax0 } else { s.m };
            let w = self.search_layer(q, ep, s.efc, lc as usize, false, tracker);
            let neighbors = self.select_neighbors(w, m_max);
            ep = neighbors[0];
            {
                let _g = self.node_locks[idx as usize % LOCK_POOL_SIZE].lock().unwrap();
                s.graph[idx as usize][lc as usize] = neighbors.clone();
            }
            for &nb in &neighbors {
                let _g = self.node_locks[nb as usize % LOCK_POOL_SIZE].lock().unwrap();
                s.graph[nb as usize][lc as usize].push(idx);
                if s.graph[nb as usize][lc as usize].len() > m_max {
                    self.prune(nb, lc as usize, m_max);
                }
            }
            lc -= 1;
        }

        if l > max_l {
            let _g = self.global.lock().unwrap();
            if l > s.max_level {
                s.entry_point = idx;
                s.max_level = l;
            }
        }
    }

    unsafe fn reorder_for_locality(&self) {
        let s = self.i_mut();
        let n = s.npts;
        let mut order: Vec<i32> = Vec::with_capacity(n);
        let mut seen = vec![0u8; n];
        order.push(s.entry_point);
        seen[s.entry_point as usize] = 1;
        let mut head = 0usize;
        while head < order.len() {
            let c = order[head];
            head += 1;
            for &nb in &s.graph[c as usize][0] {
                if seen[nb as usize] == 0 {
                    seen[nb as usize] = 1;
                    order.push(nb);
                }
            }
        }
        for i in 0..n {
            if seen[i] == 0 {
                order.push(i as i32);
            }
        }
        let mut new_id = vec![0i32; n];
        for (i, &o) in order.iter().enumerate() {
            new_id[o as usize] = i as i32;
        }
        {
            let mut tmp = vec![0.0f32; n * s.dim];
            for (i, &o) in order.iter().enumerate() {
                tmp[i * s.dim..(i + 1) * s.dim].copy_from_slice(&s.data[o as usize * s.dim..(o as usize + 1) * s.dim]);
            }
            s.data = tmp;
        }
        if !s.codes.is_empty() {
            let mut tmp = vec![0u8; n * s.dim];
            for (i, &o) in order.iter().enumerate() {
                tmp[i * s.dim..(i + 1) * s.dim].copy_from_slice(&s.codes[o as usize * s.dim..(o as usize + 1) * s.dim]);
            }
            s.codes = tmp;
        }
        {
            let mut tmp: Vec<Vec<Vec<i32>>> = vec![Vec::new(); n];
            for (i, &o) in order.iter().enumerate() {
                tmp[i] = std::mem::take(&mut s.graph[o as usize]);
                for layer in &mut tmp[i] {
                    for e in layer.iter_mut() {
                        *e = new_id[*e as usize];
                    }
                }
            }
            s.graph = tmp;
        }
        s.entry_point = new_id[s.entry_point as usize];
        s.new_to_orig = order;
    }
}

#[pyclass]
struct HNSWIndex {
    inner: Box<Inner>,
    node_locks_seed: u64,
    n_distances: AtomicI64,
}

#[pymethods]
impl HNSWIndex {
    #[new]
    fn new() -> Self {
        Self {
            inner: Box::new(Inner {
                npts: 0,
                dim: 0,
                m: 16,
                mmax0: 32,
                efc: 100,
                ml: 0.0,
                max_level: -1,
                entry_point: -1,
                mode: Mode::Float,
                heuristic: false,
                is_building: false,
                data: Vec::new(),
                graph: Vec::new(),
                codes: Vec::new(),
                scale: 1.0,
                offset: 0.0,
                pd_scale: Vec::new(),
                pd_offset: Vec::new(),
                new_to_orig: Vec::new(),
            }),
            node_locks_seed: 0x9E3779B97F4A7C15,
            n_distances: AtomicI64::new(0),
        }
    }

    #[pyo3(signature = (data, m=16, ef_construction=100, mode="float", heuristic=false, reorder=false))]
    fn fit(&mut self, py: Python<'_>, data: PyReadonlyArray2<f32>, m: usize, ef_construction: usize, mode: &str, heuristic: bool, reorder: bool) -> PyResult<()> {
        let arr = data.as_array();
        let (npts, dim) = (arr.shape()[0], arr.shape()[1]);
        if npts == 0 {
            return Err(PyRuntimeError::new_err("Input must be a non-empty 2D array"));
        }
        let s = &mut *self.inner;
        s.npts = npts;
        s.dim = dim;
        s.m = m;
        s.mmax0 = 2 * m;
        s.efc = ef_construction;
        s.ml = 1.0 / (m as f64).ln();
        s.heuristic = heuristic;
        s.mode = match mode {
            "sq8" => Mode::Sq8,
            "sq8pd" => Mode::Sq8Pd,
            "float" => Mode::Float,
            other => return Err(PyRuntimeError::new_err(format!("unknown mode '{other}' (rust port: float, sq8 or sq8pd)"))),
        };
        s.data = arr.as_slice().map(|sl| sl.to_vec()).unwrap_or_else(|| arr.iter().copied().collect());

        match s.mode {
            Mode::Sq8 => {
                let (mut mn, mut mx) = (f32::MAX, f32::MIN);
                for &v in &s.data {
                    mn = mn.min(v);
                    mx = mx.max(v);
                }
                if mx - mn > 1e-8 {
                    s.scale = (mx - mn) / 255.0;
                    s.offset = mn;
                } else {
                    s.scale = 1.0;
                    s.offset = 0.0;
                }
                s.codes = s.data.iter().map(|&v| (((v - s.offset) / s.scale).round() as i32).clamp(0, 255) as u8).collect();
            }
            Mode::Sq8Pd => {
                s.pd_scale = vec![0.0; dim];
                s.pd_offset = vec![0.0; dim];
                for d in 0..dim {
                    let (mut mn, mut mx) = (f32::MAX, f32::MIN);
                    for i in 0..npts {
                        let v = s.data[i * dim + d];
                        mn = mn.min(v);
                        mx = mx.max(v);
                    }
                    if mx - mn > 1e-8 {
                        s.pd_scale[d] = (mx - mn) / 255.0;
                        s.pd_offset[d] = mn;
                    } else {
                        s.pd_scale[d] = 1.0;
                        s.pd_offset[d] = mn;
                    }
                }
                s.codes = (0..npts * dim)
                    .map(|j| {
                        let d = j % dim;
                        (((s.data[j] - s.pd_offset[d]) / s.pd_scale[d]).round() as i32).clamp(0, 255) as u8
                    })
                    .collect();
            }
            Mode::Float => {}
        }

        s.graph = vec![Vec::new(); npts];
        s.max_level = -1;
        s.entry_point = -1;
        self.n_distances.store(0, Ordering::Relaxed);
        s.is_building = true;

        // First node sequentially — every worker starts with a valid entry point.
        {
            let mut t0 = SearchTracker::new(0, 42);
            let u = t0.next_f64();
            let l0 = (((-(u + 1e-10).ln()) * s.ml) as i32).min(32);
            s.graph[0] = vec![Vec::new(); (l0 + 1) as usize];
            s.entry_point = 0;
            s.max_level = l0;
        }

        let shared = Shared {
            inner: &mut *self.inner as *mut Inner,
            node_locks: (0..LOCK_POOL_SIZE).map(|_| Mutex::new(())).collect(),
            global: Mutex::new(()),
            n_distances: AtomicI64::new(0),
        };
        let seed_base = self.node_locks_seed;

        py.allow_threads(|| {
            let next_idx = AtomicUsize::new(1);
            let nthreads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
            let npts_local = npts;
            std::thread::scope(|scope| {
                for t in 0..nthreads {
                    let shared_ref = &shared;
                    let next = &next_idx;
                    scope.spawn(move || {
                        let mut tracker = SearchTracker::new(npts_local, seed_base ^ (t as u64 + 1).wrapping_mul(0xA24BAED4963EE407));
                        loop {
                            let idx = next.fetch_add(1, Ordering::Relaxed);
                            if idx >= npts_local {
                                break;
                            }
                            unsafe { shared_ref.insert(idx as i32, &mut tracker) };
                        }
                    });
                }
            });
        });

        self.inner.is_building = false;
        if reorder {
            let shared = Shared {
                inner: &mut *self.inner as *mut Inner,
                node_locks: Vec::new(),
                global: Mutex::new(()),
                n_distances: AtomicI64::new(0),
            };
            unsafe { shared.reorder_for_locality() };
        }
        Ok(())
    }

    #[pyo3(signature = (query, k, ef=100))]
    fn query<'py>(&self, py: Python<'py>, query: PyReadonlyArray1<f32>, k: usize, ef: usize) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let qv = query.as_array();
        let s = &*self.inner;
        if qv.len() != s.dim {
            return Err(PyRuntimeError::new_err("Query dimension mismatch"));
        }
        let qcopy: Vec<f32> = qv.iter().copied().collect();
        let ef = ef.max(k);

        let shared = Shared {
            inner: &*self.inner as *const Inner as *mut Inner,
            node_locks: Vec::new(),
            global: Mutex::new(()),
            n_distances: AtomicI64::new(0),
        };

        let results: Vec<P> = py.allow_threads(|| {
            let mut tracker = SearchTracker::new(s.npts, 7);
            let q = qcopy.as_ptr();
            unsafe {
                let mut ep = s.entry_point;
                let mut l = s.max_level;
                while l > 0 {
                    let w = shared.search_layer(q, ep, 1, l as usize, true, &mut tracker);
                    ep = w[0].1;
                    l -= 1;
                }
                let w = shared.search_layer(q, ep, ef, 0, true, &mut tracker);
                let mut results: Vec<P> = if s.mode == Mode::Float {
                    w
                } else {
                    w.iter().map(|p| P(shared.dist_to(p.1, q, f32::MAX), p.1)).collect()
                };
                results.sort_unstable();
                results
            }
        });
        self.n_distances.fetch_add(shared.n_distances.load(Ordering::Relaxed), Ordering::Relaxed);

        let out_k = k.min(results.len());
        let out: Vec<i64> = results[..out_k]
            .iter()
            .map(|p| {
                if s.new_to_orig.is_empty() {
                    p.1 as i64
                } else {
                    s.new_to_orig[p.1 as usize] as i64
                }
            })
            .collect();
        Ok(out.into_pyarray(py))
    }

    fn total_distances_count(&self) -> i64 {
        self.n_distances.load(Ordering::Relaxed)
    }
}

#[pymodule]
fn hnsw_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<HNSWIndex>()?;
    Ok(())
}
