//! Symmetric Jacobi and Shifted Lanczos Fiedler eigensolvers for the 32x32 assignment Laplacian.

use super::*;

/// Complete fixed-array symmetric Jacobi eigensolver for the active 16x16 Laplacian subgraph.
/// Used by standalone `chronos_tick` so pre-assignment Fiedler costs are always certified
/// without seed symmetry blindness.
pub(crate) fn solve_fiedler_jacobi_16x16(a: &mut Arena) {
    let mut act = [0usize; GRAPH_NODES];
    let mut n = 0usize;
    for i in 0..GRAPH_NODES {
        if a.laplacian[i * GRAPH_NODES + i] > 1e-6 {
            act[n] = i;
            n += 1;
        }
    }

    for i in 0..GRAPH_NODES {
        a.fiedler_vec[i] = 0.0;
    }
    if n < 2 {
        a.fiedler_lambda2 = 0.0;
        a.fiedler_gap = 0.0;
        return;
    }

    for v in &mut a.jacobi_mat[..n * n] {
        *v = 0.0;
    }
    for v in &mut a.jacobi_vecs[..n * n] {
        *v = 0.0;
    }
    for r in 0..n {
        let ir = act[r];
        a.jacobi_vecs[r * n + r] = 1.0;
        for c in 0..n {
            a.jacobi_mat[r * n + c] = a.laplacian[ir * GRAPH_NODES + act[c]];
        }
    }
    let mat = &mut a.jacobi_mat;
    let vecs = &mut a.jacobi_vecs;

    for _sweep in 0..12usize {
        let mut max_off = 0.0f32;
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = mat[p * n + q];
                let abs_apq = apq.abs();
                if abs_apq > max_off {
                    max_off = abs_apq;
                }
                if abs_apq < 1e-7 {
                    continue;
                }
                let app = mat[p * n + p];
                let aqq = mat[q * n + q];
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + wasm_sqrtf(1.0 + tau * tau))
                } else {
                    -1.0 / (-tau + wasm_sqrtf(1.0 + tau * tau))
                };
                let c = 1.0 / wasm_sqrtf(1.0 + t * t);
                let s = t * c;

                mat[p * n + p] = app - t * apq;
                mat[q * n + q] = aqq + t * apq;
                mat[p * n + q] = 0.0;
                mat[q * n + p] = 0.0;

                for k in 0..n {
                    if k != p && k != q {
                        let akp = mat[k * n + p];
                        let akq = mat[k * n + q];
                        let nkp = c * akp - s * akq;
                        let nkq = s * akp + c * akq;
                        mat[k * n + p] = nkp;
                        mat[p * n + k] = nkp;
                        mat[k * n + q] = nkq;
                        mat[q * n + k] = nkq;
                    }
                    let vkp = vecs[k * n + p];
                    let vkq = vecs[k * n + q];
                    vecs[k * n + p] = c * vkp - s * vkq;
                    vecs[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
        if max_off < 1e-6 {
            break;
        }
    }

    // Sort eigenvalues ascending: evals[0] ~ 0 (1_n), evals[1] = lambda_2, evals[2] = lambda_3
    let mut order = [0usize; GRAPH_NODES];
    for i in 0..n {
        order[i] = i;
    }
    for i in 0..n {
        for j in (i + 1)..n {
            if mat[order[j] * n + order[j]] < mat[order[i] * n + order[i]] {
                let tmp = order[i];
                order[i] = order[j];
                order[j] = tmp;
            }
        }
    }

    let c1 = order[1];
    let lambda2 = mat[c1 * n + c1].max(0.0);
    let lambda3 = if n >= 3 {
        mat[order[2] * n + order[2]].max(lambda2)
    } else {
        lambda2
    };

    // Orthogonalize against 1_n and normalize
    let mut mean = 0.0f32;
    for r in 0..n {
        mean += vecs[r * n + c1];
    }
    mean /= n as f32;

    let mut norm2 = 0.0f32;
    let mut fvec = [0.0f32; GRAPH_NODES];
    for r in 0..n {
        let v = vecs[r * n + c1] - mean;
        fvec[r] = v;
        norm2 += v * v;
    }
    let inv_norm = 1.0 / wasm_sqrtf(norm2.max(1e-12));
    for r in 0..n {
        fvec[r] *= inv_norm;
    }

    // Temporal sign continuity against prior track Fiedler components
    let mut dot_prev = 0.0f32;
    for r in 0..n {
        let idx = act[r];
        if idx < MAX_TARGETS {
            dot_prev += fvec[r] * a.prev_fiedler_track[idx];
        }
    }
    let sign = if dot_prev < -1e-4 || (dot_prev.abs() <= 1e-4 && fvec[0] < 0.0) {
        -1.0
    } else {
        1.0
    };

    for r in 0..n {
        let idx = act[r];
        let val = fvec[r] * sign;
        a.fiedler_vec[idx] = val;
        if idx < MAX_TARGETS {
            a.prev_fiedler_track[idx] = val;
        }
    }
    a.fiedler_lambda2 = lambda2;
    a.fiedler_gap = (lambda3 - lambda2).max(0.0);
}


#[inline(always)]
unsafe fn dot_f64_padded4(a_ptr: *const f64, b_ptr: *const f64, n4: usize) -> f64 {
    let mut s0 = f64x2_splat(0.0);
    let mut s1 = f64x2_splat(0.0);
    let mut i = 0usize;
    while i < n4 {
        let a0 = v128_load(a_ptr.add(i) as *const v128);
        let b0 = v128_load(b_ptr.add(i) as *const v128);
        s0 = f64x2_add(s0, f64x2_mul(a0, b0));
        let a1 = v128_load(a_ptr.add(i + 2) as *const v128);
        let b1 = v128_load(b_ptr.add(i + 2) as *const v128);
        s1 = f64x2_add(s1, f64x2_mul(a1, b1));
        i += 4;
    }
    let s = f64x2_add(s0, s1);
    f64x2_extract_lane::<0>(s) + f64x2_extract_lane::<1>(s)
}

#[inline(always)]
unsafe fn sub_scaled_f64_padded4(w_ptr: *mut f64, q_ptr: *const f64, scale: f64, n4: usize) {
    let vs = f64x2_splat(scale);
    let mut i = 0usize;
    while i < n4 {
        let w0 = v128_load(w_ptr.add(i) as *const v128);
        let q0 = v128_load(q_ptr.add(i) as *const v128);
        v128_store(w_ptr.add(i) as *mut v128, f64x2_sub(w0, f64x2_mul(vs, q0)));
        let w1 = v128_load(w_ptr.add(i + 2) as *const v128);
        let q1 = v128_load(q_ptr.add(i + 2) as *const v128);
        v128_store(w_ptr.add(i + 2) as *mut v128, f64x2_sub(w1, f64x2_mul(vs, q1)));
        i += 4;
    }
}

unsafe fn find_orthogonal_basis_seed_f64(
    ls: &mut LanczosScratch,
    n: usize,
    n4: usize,
    num_existing: usize,
    out_offset: usize,
) -> bool {
    let inv_n = 1.0f64 / (n as f64);
    for cand in 0..n {
        for idx in 0..n {
            ls.w[idx] = if idx == cand { 1.0 } else { 0.0 } - inv_n;
        }
        for idx in n..n4 {
            ls.w[idx] = 0.0;
        }
        for _pass in 0..2usize {
            for k in 0..num_existing {
                let k_off = k * GRAPH_NODES;
                let dot = dot_f64_padded4(ls.q.as_ptr().add(k_off), ls.w.as_ptr(), n4);
                sub_scaled_f64_padded4(ls.w.as_mut_ptr(), ls.q.as_ptr().add(k_off), dot, n4);
            }
            let mut mean = 0.0f64;
            for idx in 0..n {
                mean += ls.w[idx];
            }
            mean *= inv_n;
            for idx in 0..n {
                ls.w[idx] -= mean;
            }
            for idx in n..n4 {
                ls.w[idx] = 0.0;
            }
        }
        let norm2 = dot_f64_padded4(ls.w.as_ptr(), ls.w.as_ptr(), n4);
        if norm2 > 1e-12 {
            let inv_norm = 1.0 / wasm_sqrt_f64(norm2);
            for idx in 0..n {
                ls.q[out_offset + idx] = ls.w[idx] * inv_norm;
            }
            for idx in n..n4 {
                ls.q[out_offset + idx] = 0.0;
            }
            return true;
        }
    }
    false
}

unsafe fn solve_tridiag_tql2_f64(ls: &mut LanczosScratch, m_dim: usize) -> (usize, i32, f64, f64) {
    let m4 = (m_dim + 3) & !3;
    for i in 0..m_dim {
        ls.d_diag[i] = ls.alpha[i];
        ls.e_sub[i] = if i + 1 < m_dim { ls.beta[i] } else { 0.0 };
        let r_off = i * GRAPH_NODES;
        ls.z[r_off..(r_off + m4)].fill(0.0);
        ls.z[r_off + i] = 1.0;
    }

    for l in 0..m_dim {
        let mut iter = 0usize;
        while iter < 40 {
            iter += 1;
            let mut m = l;
            while m < m_dim - 1 {
                if abs_f64(ls.e_sub[m])
                    <= 1e-14 * (abs_f64(ls.d_diag[m]) + abs_f64(ls.d_diag[m + 1])) + 1e-30
                {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            let mut g = (ls.d_diag[l + 1] - ls.d_diag[l]) / (2.0 * ls.e_sub[l]);
            let ag = abs_f64(g);
            let mut r = if ag < 1e100 {
                wasm_sqrt_f64(g * g + 1.0)
            } else {
                ag
            };
            g = ls.d_diag[m] - ls.d_diag[l] + ls.e_sub[l] / (g + if g >= 0.0 { r } else { -r });
            let mut s = 1.0f64;
            let mut c = 1.0f64;
            let mut p = 0.0f64;
            for i in (l..m).rev() {
                let f = s * ls.e_sub[i];
                let b = c * ls.e_sub[i];
                let r2 = f * f + g * g;
                r = if r2 > 1e-200 && r2 < 1e200 {
                    wasm_sqrt_f64(r2)
                } else {
                    hypot_f64(f, g)
                };
                ls.e_sub[i + 1] = r;
                if r < 1e-30 {
                    ls.d_diag[i + 1] -= p;
                    ls.e_sub[m] = 0.0;
                    break;
                }
                s = f / r;
                c = g / r;
                g = ls.d_diag[i + 1] - p;
                r = (ls.d_diag[i] - g) * s + 2.0 * c * b;
                p = s * r;
                ls.d_diag[i + 1] = g + p;
                g = c * r - b;
                let vs = f64x2_splat(s);
                let vc = f64x2_splat(c);
                let z_ptr = ls.z.as_mut_ptr();
                let row_i = i * GRAPH_NODES;
                let row_i1 = row_i + GRAPH_NODES;
                let mut k = 0usize;
                while k < m4 {
                    let fz0 = v128_load(z_ptr.add(row_i1 + k) as *const v128);
                    let zk0 = v128_load(z_ptr.add(row_i + k) as *const v128);
                    v128_store(
                        z_ptr.add(row_i1 + k) as *mut v128,
                        f64x2_add(f64x2_mul(vs, zk0), f64x2_mul(vc, fz0)),
                    );
                    v128_store(
                        z_ptr.add(row_i + k) as *mut v128,
                        f64x2_sub(f64x2_mul(vc, zk0), f64x2_mul(vs, fz0)),
                    );
                    let fz1 = v128_load(z_ptr.add(row_i1 + k + 2) as *const v128);
                    let zk1 = v128_load(z_ptr.add(row_i + k + 2) as *const v128);
                    v128_store(
                        z_ptr.add(row_i1 + k + 2) as *mut v128,
                        f64x2_add(f64x2_mul(vs, zk1), f64x2_mul(vc, fz1)),
                    );
                    v128_store(
                        z_ptr.add(row_i + k + 2) as *mut v128,
                        f64x2_sub(f64x2_mul(vc, zk1), f64x2_mul(vs, fz1)),
                    );
                    k += 4;
                }
            }
            if r < 1e-30 && ls.e_sub[m] == 0.0 {
                continue;
            }
            ls.d_diag[l] -= p;
            ls.e_sub[l] = g;
            ls.e_sub[m] = 0.0;
        }
    }

    let mut best_col = 0usize;
    let mut sec_col = -1i32;
    let mut theta_max = -1e30f64;
    let mut theta_second = -1e30f64;
    for c in 0..m_dim {
        let val = ls.d_diag[c];
        if val > theta_max {
            theta_second = theta_max;
            sec_col = best_col as i32;
            theta_max = val;
            best_col = c;
        } else if val > theta_second {
            theta_second = val;
            sec_col = c as i32;
        }
    }
    (best_col, sec_col, theta_max, theta_second)
}

pub(crate) unsafe fn solve_fiedler_lanczos(a: &mut Arena) -> f32 {
    let ls = &mut a.lanczos_scratch;
    let mut n = 0usize;
    for i in 0..GRAPH_NODES {
        if a.laplacian[i * GRAPH_NODES + i] > 1e-6 {
            ls.act_idx[n] = i;
            n += 1;
        }
    }
    a.fiedler_vec.fill(0.0);
    if n < 2 {
        ls.last_n = 0;
        ls.last_lambda2 = 0.0;
        a.fiedler_lambda2 = 0.0;
        a.fiedler_gap = 0.0;
        return 0.0;
    }

    let n4 = (n + 3) & !3;
    let mut mu = 0.0f64;
    for idx in 0..n {
        let ia = ls.act_idx[idx] * GRAPH_NODES;
        let r_off = idx * GRAPH_NODES;
        let diag = a.laplacian[ia + ls.act_idx[idx]] as f64;
        ls.sub_l[r_off + idx] = diag;
        for b in 0..idx {
            let v = a.laplacian[ia + ls.act_idx[b]] as f64;
            ls.sub_l[r_off + b] = v;
            ls.sub_l[b * GRAPH_NODES + idx] = v;
        }
        for b in n..n4 {
            ls.sub_l[r_off + b] = 0.0;
        }
    }
    for idx in 0..n {
        let r_off = idx * GRAPH_NODES;
        let mut row_sum = ls.sub_l[r_off + idx];
        for b in 0..n {
            if idx != b {
                row_sum += abs_f64(ls.sub_l[r_off + b]);
            }
        }
        if row_sum > mu {
            mu = row_sum;
        }
    }
    mu += 0.1;

    let m_steps = (n - 1).min(MAX_KRYLOV);
    let inv_n = 1.0f64 / (n as f64);
    let mut mean0 = 0.0f64;
    for idx in 0..n {
        let val = ((idx + 1) as f64) - 0.5 * ((n + 1) as f64);
        ls.q[idx] = val;
        mean0 += val;
    }
    mean0 *= inv_n;
    let mut norm0 = 0.0f64;
    for idx in 0..n {
        ls.q[idx] -= mean0;
        norm0 += ls.q[idx] * ls.q[idx];
    }
    let inv_norm0 = 1.0f64 / wasm_sqrt_f64(norm0.max(1e-18));
    for idx in 0..n {
        ls.q[idx] *= inv_norm0;
    }
    for idx in n..n4 {
        ls.q[idx] = 0.0;
        ls.w[idx] = 0.0;
    }

    let mut actual_m = 0usize;
    for j in 0..m_steps {
        let q_off = j * GRAPH_NODES;
        let q_ptr = ls.q.as_ptr().add(q_off);
        for idx in 0..n {
            let sum_l = dot_f64_padded4(ls.sub_l.as_ptr().add(idx * GRAPH_NODES), q_ptr, n4);
            ls.w[idx] = mu * ls.q[q_off + idx] - sum_l;
        }
        let mut w_mean = 0.0f64;
        for idx in 0..n {
            w_mean += ls.w[idx];
        }
        w_mean *= inv_n;
        if j > 0 && ls.beta[j - 1] > 0.0 {
            let prev_off = (j - 1) * GRAPH_NODES;
            let beta_prev = ls.beta[j - 1];
            for idx in 0..n {
                ls.w[idx] = (ls.w[idx] - w_mean) - beta_prev * ls.q[prev_off + idx];
            }
        } else {
            for idx in 0..n {
                ls.w[idx] -= w_mean;
            }
        }
        for idx in n..n4 {
            ls.w[idx] = 0.0;
        }

        let alpha = dot_f64_padded4(q_ptr, ls.w.as_ptr(), n4);
        ls.alpha[j] = alpha;
        sub_scaled_f64_padded4(ls.w.as_mut_ptr(), q_ptr, alpha, n4);
        let pre_norm2 = dot_f64_padded4(ls.w.as_ptr(), ls.w.as_ptr(), n4);

        for pass in 0..2usize {
            for k in 0..=j {
                let k_ptr = ls.q.as_ptr().add(k * GRAPH_NODES);
                let dot = dot_f64_padded4(k_ptr, ls.w.as_ptr(), n4);
                sub_scaled_f64_padded4(ls.w.as_mut_ptr(), k_ptr, dot, n4);
            }
            let mut wm = 0.0f64;
            for idx in 0..n {
                wm += ls.w[idx];
            }
            wm *= inv_n;
            for idx in 0..n {
                ls.w[idx] -= wm;
            }
            for idx in n..n4 {
                ls.w[idx] = 0.0;
            }
            let post2 = dot_f64_padded4(ls.w.as_ptr(), ls.w.as_ptr(), n4);
            if pass == 0 && post2 > 0.5 * pre_norm2 {
                break;
            }
        }

        let beta2 = dot_f64_padded4(ls.w.as_ptr(), ls.w.as_ptr(), n4);
        let beta = wasm_sqrt_f64(beta2.max(0.0));
        actual_m = j + 1;

        if j + 1 >= m_steps {
            ls.beta[j] = 0.0;
            break;
        }
        let next_off = (j + 1) * GRAPH_NODES;
        if beta < 1e-8 {
            ls.beta[j] = 0.0;
            if !find_orthogonal_basis_seed_f64(ls, n, n4, j + 1, next_off) {
                break;
            }
        } else {
            ls.beta[j] = beta;
            let inv_beta = 1.0 / beta;
            for idx in 0..n {
                ls.q[next_off + idx] = ls.w[idx] * inv_beta;
            }
            for idx in n..n4 {
                ls.q[next_off + idx] = 0.0;
            }
        }
    }

    let (best_col, _, theta_max, theta_second) = solve_tridiag_tql2_f64(ls, actual_m);
    let lambda2 = (mu - theta_max).max(0.0);
    let lambda3 = if actual_m >= 2 {
        (mu - theta_second).max(lambda2)
    } else {
        lambda2
    };
    let spectral_gap = (lambda3 - lambda2).max(0.0);

    let mut eig_acc0 = [f64x2_splat(0.0); GRAPH_NODES / 2];
    let best_z_off = best_col * GRAPH_NODES;
    let n2 = (n + 1) >> 1;
    for j in 0..actual_m {
        let vz = f64x2_splat(ls.z[best_z_off + j]);
        let q_ptr = ls.q.as_ptr().add(j * GRAPH_NODES);
        for u in 0..n2 {
            let qv = v128_load(q_ptr.add(u * 2) as *const v128);
            eig_acc0[u] = f64x2_add(eig_acc0[u], f64x2_mul(qv, vz));
        }
    }
    for u in 0..n2 {
        v128_store(ls.eig_vec.as_mut_ptr().add(u * 2) as *mut v128, eig_acc0[u]);
    }
    for idx in n..n4 {
        ls.eig_vec[idx] = 0.0;
    }

    let mut dot_prev = 0.0f64;
    for idx in 0..n {
        let node_idx = ls.act_idx[idx];
        if node_idx < MAX_TARGETS {
            dot_prev += ls.eig_vec[idx] * ls.prev_track[node_idx];
        }
    }

    let sign_fix = if dot_prev < -1e-4 || (abs_f64(dot_prev) <= 1e-4 && ls.eig_vec[0] < 0.0) {
        -1.0f64
    } else {
        1.0f64
    };
    for idx in 0..n {
        let node_idx = ls.act_idx[idx];
        let val = ls.eig_vec[idx] * sign_fix;
        a.fiedler_vec[node_idx] = val as f32;
        if node_idx < MAX_TARGETS {
            ls.prev_track[node_idx] = val;
        }
    }

    ls.last_n = n;
    ls.last_lambda2 = lambda2;
    let l2_f = lambda2 as f32;
    let gap_f = spectral_gap as f32;
    a.fiedler_lambda2 = if l2_f.is_finite() { l2_f.max(0.0) } else { 0.0 };
    a.fiedler_gap = if gap_f.is_finite() { gap_f.max(0.0) } else { 0.0 };
    a.fiedler_lambda2
}

pub(crate) unsafe fn get_fiedler_res_norm(a: &mut Arena) -> f32 {
    let ls = &a.lanczos_scratch;
    let n = ls.last_n;
    if n < 2 {
        return 0.0;
    }
    let n4 = (n + 3) & !3;
    let lambda2 = ls.last_lambda2;
    let mut res2 = 0.0f64;
    for idx in 0..n {
        let lv = dot_f64_padded4(
            ls.sub_l.as_ptr().add(idx * GRAPH_NODES),
            ls.eig_vec.as_ptr(),
            n4,
        );
        let diff = lv - lambda2 * ls.eig_vec[idx];
        res2 += diff * diff;
    }
    wasm_sqrt_f64(res2) as f32
}

