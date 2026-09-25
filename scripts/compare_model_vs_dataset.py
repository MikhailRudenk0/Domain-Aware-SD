"""Compare model-mode vs dataset-mode eval for base drafter on all 66 clusters."""
import json, glob, numpy as np

model_dir = 'eval_results/model_mode_v2'
dataset_dir = 'eval_results'

model_files = sorted(glob.glob(f'{model_dir}/eval__Lite-Mistral-150M-v2-Instruct__*.json'))
dataset_files = sorted(glob.glob(f'{dataset_dir}/eval__Lite-Mistral-150M-v2-Instruct__*.json'))

print(f'Model-mode files: {len(model_files)}')
print(f'Dataset-mode files: {len(dataset_files)}')
print()

# Verify modes
with open(dataset_files[0]) as f:
    d = json.load(f)
    print(f'Dataset file target_source: {d["metadata"]["target_source"]}')
with open(model_files[0]) as f:
    d = json.load(f)
    print(f'Model file target_source: {d["metadata"]["target_source"]}')
print()

rows = []
for mf in model_files:
    cluster = mf.split('__')[-1].replace('.json','')
    df_path = f'{dataset_dir}/eval__Lite-Mistral-150M-v2-Instruct__{cluster}.json'

    with open(mf) as f:
        m_data = json.load(f)
    with open(df_path) as f:
        d_data = json.load(f)

    mr = m_data['results']
    dr = d_data['results']

    m_oa = np.mean(mr.get('single__overlap_area__full_target', []))
    m_t1 = np.mean(mr.get('single__top1_match__full_target', []))
    m_tk = np.mean(mr.get('single__topk_overlap__full_target', []))
    m_kl = np.mean(mr.get('single__kl__full_target', []))

    d_keys = list(dr.keys())
    d_oa_key = [k for k in d_keys if 'overlap_area' in k][0]
    d_t1_key = [k for k in d_keys if 'top1_match' in k][0]
    d_tk_key = [k for k in d_keys if 'topk_overlap' in k][0]
    d_kl_key = [k for k in d_keys if '__kl__' in k][0]

    d_oa = np.mean(dr[d_oa_key])
    d_t1 = np.mean(dr[d_t1_key])
    d_tk = np.mean(dr[d_tk_key])
    d_kl = np.mean(dr[d_kl_key])

    name = cluster.replace('_10templates','')
    rows.append((name, m_oa, d_oa, m_t1, d_t1, m_tk, d_tk, m_kl, d_kl))

rows.sort(key=lambda x: x[0])

# Header
hdr = f"{'Cluster':<40} | {'OA_mod':>7} {'OA_ds':>7} {'d':>6} | {'T1_mod':>7} {'T1_ds':>7} {'d':>6} | {'TK_mod':>7} {'TK_ds':>7} {'d':>6} | {'KL_mod':>7} {'KL_ds':>7} {'d':>6}"
print(hdr)
print('-' * len(hdr))

m_oa_all, d_oa_all = [], []
m_t1_all, d_t1_all = [], []
m_tk_all, d_tk_all = [], []
m_kl_all, d_kl_all = [], []

for r in rows:
    name, m_oa, d_oa, m_t1, d_t1, m_tk, d_tk, m_kl, d_kl = r
    m_oa_all.append(m_oa); d_oa_all.append(d_oa)
    m_t1_all.append(m_t1); d_t1_all.append(d_t1)
    m_tk_all.append(m_tk); d_tk_all.append(d_tk)
    m_kl_all.append(m_kl); d_kl_all.append(d_kl)

    print(f"{name:<40} | {m_oa:7.4f} {d_oa:7.4f} {d_oa-m_oa:+6.3f} | {m_t1:7.4f} {d_t1:7.4f} {d_t1-m_t1:+6.3f} | {m_tk:7.4f} {d_tk:7.4f} {d_tk-m_tk:+6.3f} | {m_kl:7.4f} {d_kl:7.4f} {d_kl-m_kl:+6.3f}")

print('-' * len(hdr))
print(f"{'MEAN':<40} | {np.mean(m_oa_all):7.4f} {np.mean(d_oa_all):7.4f} {np.mean(d_oa_all)-np.mean(m_oa_all):+6.3f} | {np.mean(m_t1_all):7.4f} {np.mean(d_t1_all):7.4f} {np.mean(d_t1_all)-np.mean(m_t1_all):+6.3f} | {np.mean(m_tk_all):7.4f} {np.mean(d_tk_all):7.4f} {np.mean(d_tk_all)-np.mean(m_tk_all):+6.3f} | {np.mean(m_kl_all):7.4f} {np.mean(d_kl_all):7.4f} {np.mean(d_kl_all)-np.mean(m_kl_all):+6.3f}")
print(f"{'MIN':<40} | {np.min(m_oa_all):7.4f} {np.min(d_oa_all):7.4f}        | {np.min(m_t1_all):7.4f} {np.min(d_t1_all):7.4f}        | {np.min(m_tk_all):7.4f} {np.min(d_tk_all):7.4f}        | {np.min(m_kl_all):7.4f} {np.min(d_kl_all):7.4f}")
print(f"{'MAX':<40} | {np.max(m_oa_all):7.4f} {np.max(d_oa_all):7.4f}        | {np.max(m_t1_all):7.4f} {np.max(d_t1_all):7.4f}        | {np.max(m_tk_all):7.4f} {np.max(d_tk_all):7.4f}        | {np.max(m_kl_all):7.4f} {np.max(d_kl_all):7.4f}")

from scipy import stats
corr_oa, _ = stats.pearsonr(m_oa_all, d_oa_all)
corr_t1, _ = stats.pearsonr(m_t1_all, d_t1_all)
corr_tk, _ = stats.pearsonr(m_tk_all, d_tk_all)
corr_kl, _ = stats.pearsonr(m_kl_all, d_kl_all)
print(f'\nPearson correlation (model vs dataset):')
print(f'  overlap_area: r = {corr_oa:.4f}')
print(f'  top1_match:   r = {corr_t1:.4f}')
print(f'  topk_overlap: r = {corr_tk:.4f}')
print(f'  kl:           r = {corr_kl:.4f}')
