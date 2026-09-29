"""Local-only, exploratory short-video / Doppler association analysis.

Run from any directory with explicit local input paths; see docs/PAPER2_DOPPLER_ASSOCIATIONS.md.
Inputs are read only. No patient-level joined table is exported.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

STAT = Path(__file__).resolve().with_name('06_analyze_clinician_activity_association.py')
DOPPLER = {'SD': 'flow_sd', 'PI': 'flow_pi', 'RI': 'flow_ri',
           'VI': 'flow_vi', 'FI': 'flow_fi', 'VFI': 'flow_vfi'}
QUALITY = ['global_speed_median_original_px_s', 'outer_all_fb_p95_original_px',
           'fps', 'duration_s', 'outer_all_pcc_median', 'formal_pair_valid_ratio']
MODELS = {
    'age_thickness': ['female_age', 'endometrial_thickness_mm'],
    'age_thickness_BMI': ['female_age', 'endometrial_thickness_mm', 'female_bmi'],
    'acquisition': ['female_age', 'endometrial_thickness_mm', 'fps', 'duration_s'],
    'global_motion': ['female_age', 'endometrial_thickness_mm', QUALITY[0]],
    'tracking_error': ['female_age', 'endometrial_thickness_mm', QUALITY[1]],
    'motion_and_error': ['female_age', 'endometrial_thickness_mm', QUALITY[0], QUALITY[1]],
}

REFERENCE_RHO = {
    ('F09', 'VI'): -0.2380,
    ('F09', 'VFI'): -0.2288,
    ('F15', 'VI'): -0.2143,
    ('F15', 'VFI'): -0.2121,
}
SENSITIVITY_PAIRS = tuple(REFERENCE_RHO)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(series):
    return series.astype('string').str.strip().str.replace(r'\.0$', '', regex=True)


def unique(table, key):
    table[key] = normalize(table[key])
    if table[key].isna().any() or table[key].eq('').any() or table[key].duplicated().any():
        raise ValueError(f'Invalid join key: {key}')


def verified_inputs(master_path, manifest_path, audit_path, stats):
    """Load the exact script-05 master and clinical workbook from one frozen run."""
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if sha(master_path) != manifest.get('output_sha256', {}).get('master_csv'):
        raise ValueError('Master CSV SHA256 differs from script 05 manifest')
    if sha(audit_path) != manifest.get('input_sha256', {}).get('patient_audit'):
        raise ValueError('Patient audit SHA256 differs from script 05 manifest')
    gate_path = Path(manifest['input_paths']['frozen_gate'])
    if sha(gate_path) != manifest.get('frozen_gate_sha256'):
        raise ValueError('Frozen gate SHA256 differs from script 05 manifest')
    master = pd.read_csv(master_path, dtype={'case_id': 'string'}, low_memory=False)
    stats.validate_master(master, manifest)
    unique(master, 'case_id')
    audit = pd.read_excel(audit_path, sheet_name='02_患者级状态', dtype={'case_id': 'string'})
    unique(audit, 'case_id')
    if set(master.case_id) != set(audit.case_id):
        raise ValueError('Patient audit case IDs differ from frozen master')
    for column in ('matched_exam_date8', 'video_real_exam_date8', 'final_audit_status'):
        if column not in master or column not in audit:
            raise ValueError(f'Missing examination audit column: {column}')
    aligned = audit.set_index('case_id').loc[master.case_id]
    for column in ('matched_exam_date8', 'video_real_exam_date8'):
        frozen = normalize(master[column]).reset_index(drop=True)
        source = normalize(aligned[column]).reset_index(drop=True)
        if (not frozen.str.fullmatch(r'20\d{6}').fillna(False).all()
                or not frozen.eq(source).fillna(False).all()):
            raise ValueError(f'Patient audit {column} differs from frozen master')
    if (not master.final_audit_status.eq('PASS_THREE_WAY').all()
            or not aligned.final_audit_status.eq('PASS_THREE_WAY').all()
            or not master.date_audit_pass.astype(str).str.lower().eq('true').all()):
        raise ValueError('Doppler cohort requires the verified three-way date audit')
    return master, audit, gate_path, manifest


def verified_repair_ids(repair_path, master):
    """Check the original 14-case repair inventory before exclusion analyses."""
    repaired = pd.read_csv(repair_path, dtype={'case_id': 'string'})
    if 'case_id' not in repaired:
        raise ValueError('Topology repair inventory lacks case_id')
    parsed = repaired.case_id.str.extract(r'^.+_(\d+)_(20\d{6})$')
    if len(parsed) != 14 or parsed.isna().any().any() or parsed[0].duplicated().any():
        raise ValueError('Topology repair inventory must have 14 distinct valid case IDs')
    if not set(parsed[0]).issubset(set(master.case_id)):
        raise ValueError('Topology repair case ID absent from frozen master')
    dates = normalize(master.set_index('case_id').loc[parsed[0], 'paper1_filename_date8'])
    if not dates.reset_index(drop=True).eq(parsed[1]).fillna(False).all():
        raise ValueError('Topology repair source date differs from frozen Paper 1 source')
    return set(parsed[0])


def partial(a):
    """Rank all continuous columns, regress x and y on controls plus intercept."""
    if len(a) <= a.shape[1] + 2:
        return np.nan
    r = rankdata(a, axis=0)
    c = np.column_stack([np.ones(len(a)), r[:, 2:]])
    if np.linalg.matrix_rank(c) < c.shape[1]:
        return np.nan
    residual = r[:, :2] - c @ np.linalg.lstsq(c, r[:, :2], rcond=None)[0]
    if np.any(np.std(residual, axis=0) < 1e-10):
        return np.nan
    return float(np.corrcoef(residual.T)[0, 1])


def boot_partial(a, seed, b=1000):
    if len(a) <= a.shape[1] + 2:
        return np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    vals = np.array([partial(a[rng.integers(0, len(a), len(a))]) for _ in range(b)])
    good = vals[np.isfinite(vals)]
    ci = np.percentile(good, [2.5, 97.5]) if len(good) else [np.nan, np.nan]
    return *ci, len(good)


def markdown(table):
    def value(x):
        if isinstance(x, (float, np.floating)):
            return f'{x:.4g}' if np.isfinite(x) else 'NA'
        return str(x).replace('|', '/')
    rows = ['|' + '|'.join(map(str, table.columns)) + '|',
            '|' + '|'.join(['---'] * len(table.columns)) + '|']
    rows += ['|' + '|'.join(value(x) for x in row) + '|' for row in table.itertuples(index=False, name=None)]
    return '\n'.join(rows)


def association_table(table, stats, features):
    rows = []
    for index, (f, d) in enumerate((f, d) for f in features for d in DOPPLER):
        v = table[[features[f], d]].dropna().to_numpy(float)
        result = stats.association(v[:, 0], v[:, 1], index)
        if result['status'] == 'EVALUABLE':
            if not np.isclose(result['rho'], spearmanr(v[:, 0], v[:, 1]).statistic, atol=1e-12):
                raise ValueError(f'Spearman cross-check failed for {f}-{d}')
        elif not np.isnan(result['rho']):
            raise ValueError(f'Non-evaluable comparison has a rho for {f}-{d}')
        rows.append({'feature': f, 'doppler': d, 'n': len(v), **result})
        print(f'Completed {f}-{d}: n={len(v)}, rho={result["rho"]:.4f}', flush=True)
    results = pd.DataFrame(rows)
    results['holm_p_24'] = stats.holm_with_planned_family(results.raw_p)
    planned_p = results.raw_p.fillna(1).to_numpy(float)
    order = np.argsort(planned_p, kind='stable')
    check = np.maximum.accumulate((len(results)-np.arange(len(results)))*planned_p[order]).clip(0, 1)
    observed = results.holm_p_24.fillna(1).to_numpy(float)[order]
    if not np.allclose(observed, check):
        raise ValueError('Independent Holm cross-check failed')
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--master', type=Path, required=True,
                        help='Frozen Paper 2 patient-level master CSV')
    parser.add_argument('--master-manifest', type=Path, required=True,
                        help='Manifest written alongside the frozen master by script 05')
    parser.add_argument('--audit', type=Path, required=True,
                        help='Patient-level clinical audit workbook')
    parser.add_argument('--qc', type=Path, required=True,
                        help='Directory with patient_quality.csv and PRIVATE_source_inventory.csv')
    parser.add_argument('--repair', type=Path, required=True,
                        help='14-case topology repair CSV')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument(
        '--verify-reference-rho',
        action='store_true',
        help='Legacy reproduction gate: require the four archived original-cohort rho values.',
    )
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output must be a new directory')
    master_path, manifest_path, audit_path = args.master, args.master_manifest, args.audit
    qc_dir, repair_path = args.qc, args.repair
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('existing_statistics', STAT)
    stats = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stats)
    features = stats.FEATURES
    master, audit, gate_path, master_manifest = verified_inputs(
        master_path, manifest_path, audit_path, stats)
    sources = [master_path, manifest_path, audit_path, gate_path,
               qc_dir/'patient_quality.csv', qc_dir/'PRIVATE_source_inventory.csv',
               repair_path, STAT, Path(__file__)]
    hashes = {str(p): sha(p) for p in sources}
    mapping = {}
    for key, prefix in DOPPLER.items():
        found = [c for c in audit if str(c).split('\n')[0] == prefix]
        if len(found) != 1:
            raise ValueError(f'Ambiguous column: {prefix}')
        mapping[found[0]] = key
    original_columns = dict(mapping)
    clinical = audit[['case_id', *mapping]].rename(columns=mapping)
    table = master.merge(clinical, on='case_id', how='left', validate='one_to_one', indicator=True)
    if not table['_merge'].eq('both').all():
        raise ValueError('Unmatched clinical rows')
    table = table.drop(columns='_merge')
    inventory = pd.read_csv(qc_dir/'PRIVATE_source_inventory.csv', dtype={'case_id': 'string'})
    parsed = inventory.case_id.str.extract(r'_(\d+)_(\d{8})$')
    if parsed.isna().any().any():
        raise ValueError('Unexpected QC source identifier format')
    inventory['case_id'] = parsed[0]
    inventory['qc_filename_date'] = pd.to_numeric(parsed[1])
    unique(inventory, 'case_id')
    unique(inventory, 'analysis_id')
    quality = pd.read_csv(qc_dir/'patient_quality.csv')
    unique(quality, 'analysis_id')
    quality = inventory[['case_id', 'analysis_id', 'qc_filename_date']].merge(quality, on='analysis_id', validate='one_to_one')
    quality = quality.rename(columns={'F01':'qc_F01'})
    table = table.merge(quality[['case_id', 'status', 'qc_filename_date', 'qc_F01', *QUALITY]], on='case_id', how='left', validate='one_to_one')
    available = pd.to_numeric(table[features['F09']],errors='coerce').notna()
    if not table.loc[available,'status'].eq('OK').all():
        raise ValueError('Feature-available cases lack valid QC mapping')
    if not (pd.to_numeric(table.loc[available,'paper1_filename_date8']) == table.loc[available,'qc_filename_date']).all():
        raise ValueError('QC source filename date does not match feature source')
    repaired_ids = verified_repair_ids(repair_path, master)
    table['topology_repaired'] = table.case_id.isin(repaired_ids)
    unaffected = available & ~table.topology_repaired
    if not np.allclose(table.loc[unaffected,features['F01']],table.loc[unaffected,'qc_F01'],equal_nan=False):
        raise ValueError('Unexpected QC/feature difference outside documented 14 repaired cases')
    fields = [*features.values(), *DOPPLER, 'female_age', 'female_bmi', 'endometrial_thickness_mm', *QUALITY]
    availability = []
    for column in fields:
        raw = table[column]
        numeric = pd.to_numeric(raw, errors='coerce').replace([np.inf, -np.inf], np.nan)
        availability.append({'field': column, 'total': len(table), 'finite': numeric.notna().sum(),
                             'missing': numeric.isna().sum(), 'zero': numeric.eq(0).sum(),
                             'negative': numeric.lt(0).sum(), 'unique': numeric.nunique(),
                             'min': numeric.min(), 'median': numeric.median(), 'max': numeric.max(),
                             'nonempty_unparsed': (raw.notna() & numeric.isna()).sum()})
        table[column] = numeric
    # Clinician activity availability does not govern Doppler inclusion.
    results = association_table(table, stats, features)
    # Optional legacy reproduction gate. It is not a validity condition for a
    # legitimately corrected or future cohort.
    if args.verify_reference_rho:
        indexed = results.set_index(['feature', 'doppler'])
        if not all(abs(indexed.loc[key, 'rho'] - value) < .000051
                   for key, value in REFERENCE_RHO.items()):
            raise ValueError('Reference cohort point estimates differ from archived values')
    out = args.output
    out.mkdir(parents=True)
    pd.DataFrame(availability).to_csv(out/'availability.csv', index=False)
    results.to_csv(out/'associations_24.csv', index=False)
    sensitivity = []
    strata = []
    robust = []
    for f, d in SENSITIVITY_PAIRS:
        v = table[[features[f],d]].dropna().to_numpy(float)
        x,y = v.T
        keep = ((x>=np.quantile(x,.01)) & (x<=np.quantile(x,.99)) &
                (y>=np.quantile(y,.01)) & (y<=np.quantile(y,.99)))
        loo = [stats.spearman(np.delete(x,i),np.delete(y,i)) for i in range(len(x))]
        robust.append({'feature':f,'doppler':d,'n':len(v),'rho':stats.spearman(x,y),
                       'nonzero_n':int((y>0).sum()),'nonzero_rho':stats.spearman(x[y>0],y[y>0]),
                       'trim_n':int(keep.sum()),'trim_rho':stats.spearman(x[keep],y[keep]),
                       'leave_one_out_min':min(loo),'leave_one_out_max':max(loo)})
        for name, controls in MODELS.items():
            a = table[[features[f], d, *controls]].dropna().to_numpy(float)
            lower, upper, valid = boot_partial(a, 202609280 + len(sensitivity))
            sensitivity.append({'feature': f, 'doppler': d, 'model': name, 'n': len(a),
                                'unadjusted_same_sample': stats.spearman(a[:,0],a[:,1]),
                                'adjusted_rank_rho': partial(a), 'ci_lower': lower,
                                'ci_upper': upper, 'bootstrap_valid': valid})
        controls = MODELS['motion_and_error']
        a = table.loc[~table.topology_repaired, [features[f],d,*controls]].dropna().to_numpy(float)
        lower, upper, valid = boot_partial(a,202609280+len(sensitivity))
        sensitivity.append({'feature':f,'doppler':d,'model':'motion_error_exclude_14_repaired',
                            'n':len(a),'unadjusted_same_sample':stats.spearman(a[:,0],a[:,1]),
                            'adjusted_rank_rho':partial(a),'ci_lower':lower,'ci_upper':upper,'bootstrap_valid':valid})
        for q in QUALITY[:2]:
            a = table[[features[f], d, q]].dropna()
            cut = a[q].median()
            for label, mask in [('lower_half', a[q].le(cut)), ('upper_half', a[q].gt(cut))]:
                v = a.loc[mask].to_numpy(float)
                strata.append({'feature': f, 'doppler': d, 'stratifier': q, 'group': label,
                               'cutoff': cut, 'n': len(v), 'rho': stats.spearman(v[:,0],v[:,1]) if len(v)>2 else np.nan})
        print(f'Completed quality/clinical sensitivities {f}-{d}', flush=True)
    sensitivity = pd.DataFrame(sensitivity)
    sensitivity.to_csv(out/'adjusted_sensitivities.csv', index=False)
    pd.DataFrame(robust).to_csv(out/'outlier_sensitivities.csv',index=False)
    pd.DataFrame(strata).to_csv(out/'quality_strata_descriptive.csv', index=False)
    diagnostics = []
    for f in ['F09', 'F15', 'VI', 'VFI']:
        c = features.get(f, f)
        for q in QUALITY:
            a = table[[c, q]].dropna().to_numpy(float)
            diagnostics.append({'variable': f, 'quality_variable': q, 'n': len(a),
                                'rho': stats.spearman(a[:,0],a[:,1]) if np.unique(a[:,1]).size>1 else np.nan})
    pd.DataFrame(diagnostics).to_csv(out/'quality_correlations_descriptive.csv', index=False)
    # Show distributions, original values, and monotonic relations without
    # patient labels; no clinically meaningful threshold is inferred.
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, key in zip(axes.flat, ['F09','F15','VI','VFI']):
        ax.hist(table[features.get(key,key)].dropna(), bins=35, color='#397da8', edgecolor='white')
        ax.set(xlabel=key, ylabel='Patients', title=f'{key}: observed distribution')
    fig.savefig(out/'01_distributions.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, (f,d) in zip(axes.flat, expected):
        a = table[[features[f],d]].dropna().to_numpy(float)
        ax.scatter(a[:,0],a[:,1],s=15,alpha=.45,color='#287da4',edgecolors='none')
        row = results.set_index(['feature','doppler']).loc[(f,d)]
        ax.set(xlabel=f, ylabel=d, title=f'n={len(a)}; Spearman rho={row.rho:.3f}')
    fig.savefig(out/'02_raw_scatter.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, (f,d) in zip(axes.flat, expected):
        a = table[[features[f],d]].dropna().to_numpy(float)
        r = rankdata(a,axis=0)/len(a)
        ax.scatter(r[:,0],r[:,1],s=14,alpha=.35,color='#287da4',edgecolors='none')
        bins = pd.qcut(r[:,0],5,duplicates='drop')
        grouped = pd.DataFrame({'x':r[:,0],'y':r[:,1],'bin':bins}).groupby('bin',observed=True).median()
        ax.plot(grouped.x,grouped.y,'o-',color='#b34932',label='5-bin median (descriptive)')
        ax.set(xlabel=f'{f} rank percentile',ylabel=f'{d} rank percentile',xlim=(0,1),ylim=(0,1))
        ax.legend(fontsize=8)
    fig.savefig(out/'03_rank_scatter.png', dpi=180); plt.close(fig)
    # Clinical metadata inventory: aggregate categories, no names or identifiers.
    metadata = {c: {str(k): int(v) for k,v in master[c].value_counts(dropna=False).items()}
                for c in ['cycle_type','endometrial_type']}
    metadata['treatment_protocol_counts'] = {str(k):int(v) for k,v in audit.treatment_protocol.value_counts(dropna=False).items()}
    (out/'clinical_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    if not all(sha(Path(p)) == h for p,h in hashes.items()):
        raise ValueError('Inputs changed during analysis')
    manifest = {'status':'complete', 'exploratory':True, 'base_n':len(master),
                'qc_matched_n':int(table.status.notna().sum()),'input_sha256':hashes,
                'script05_manifest_sha256': hashes[str(manifest_path)],
                'script05_cohort_counts': master_manifest['cohort_counts'],
                'doppler_source_columns':original_columns, 'permutations':9999,'bootstraps':5000,
                'permutation_seed_base':20260922,'bootstrap_seed_base':20270922,
                'seed_note':'Reuses project statistical implementation; original Doppler run seed was not archived. Exact Monte Carlo P/CI equality to old text is not required.',
                'reference_rho_check_requested': bool(args.verify_reference_rho),
                'sensitivity_bootstraps':1000,'sensitivity_seed_base':202609280,
                'controls':MODELS,'python':platform.python_version(),
                'numpy':np.__version__,'pandas':pd.__version__,
                'verification':['Script-05 master, patient audit and frozen gate hashes verified; case IDs and confirmed exam dates agree',
                                'Repair inventory has 14 distinct master case IDs with matching Paper 1 source dates',
                                'QC IDs parsed by filename suffix; feature source dates verified; F01 agrees outside documented 14 topology-repaired cases',
                                '24 rho values cross-checked with scipy.stats.spearmanr',
                                'Holm verified with independent formula',
                                'legacy reference rho gate passed' if args.verify_reference_rho else
                                'legacy reference rho gate not requested',
                                'unique one-to-one joins enforced', 'all input hashes unchanged']}
    selected = results.loc[results.holm_p_24.lt(.05),['feature','doppler','n','rho','ci_lower','ci_upper','raw_p','holm_p_24']]
    report = '# 短视频动态特征与内膜多普勒：现有队列完整复算\n\n'
    report += '日期：2026-09-28。全部分析仍为探索性。测量部位和同次检查对应沿用研究者已确认的信息。\n\n'
    report += ('旧队列四个rho复现门槛：已启用。\\n\\n' if args.verify_reference_rho
               else '旧队列四个rho复现门槛：未启用；这不是新/修正队列的有效性条件。\\n\\n')

    report += '## 24项分析\n\n四项冻结特征与六项多普勒指标逐对使用有效病例；缺失不填零。双侧置换9,999次，配对病例bootstrap 5,000次，Holm覆盖全部24项。完整结果见 associations_24.csv。\n\n'
    report += '通过Holm校正的比较：\n\n'+markdown(selected)+'\n\n'
    report += '## 临床、采集及质量敏感性\n\n所有连续变量转秩后，分别将特征和多普勒对协变量回归，计算残差相关；95%区间用1,000次患者重抽样，每次重新转秩和拟合。同一完整病例集的未调整值同时列出，避免把缺失导致的样本变化误当作调整效果。它们是事后敏感性检查，不增加独立确认性结论。\n\n'
    report += markdown(sensitivity)+'\n\n'
    report += '零值、两端1%截除与逐例剔除检查见 outlier_sensitivities.csv。\n\n'
    report += '质量数据是全视频层面汇总，冻结特征中14例已屏蔽拓扑风险帧。两者F01差异已由原始修复清单解释；增加排除这14例的质量调整敏感性分析，不把全视频质量值当作屏蔽后的精确误差。BMI原表存在101.9等极值，BMI仅作秩调整补充，不自动改值。\n\n'
    report += '## 图与分层\n\n01为原始分布，02为不截除极值的原始散点，03为秩散点及五组中位数，分组仅帮助阅读。quality_strata_descriptive.csv按整体位移和追踪往返误差各自中位数分层，该切点不是合格标准。quality_correlations_descriptive.csv仅报告描述性相关，不按显著性挑选协变量。\n\n'
    report += '## 边界与后续\n\n- 临床周期类型和治疗方案不等同于视频采集时的月经周期阶段；当前字段无法可靠完成该阶段调整。\n- 现有质量指标源于同一视频与追踪过程，可能同时反映真实运动和测量困难；调整后减弱不能单独证明伪影。\n- 帧率、时长可检查；多普勒增益、PRF、设备/操作者与同步激素资料未进入当前数据表，无法据此排除其影响。\n- F09/F15及VI/VFI高度相关，四项结果不是四次独立重复。\n- 本次保留既定四项特征和24项比较；F09–VI作为已观察线索的主要展示，不改写成预先指定的检验。\n- 当前分析单位沿用患者主表病例记录；自然人是否跨编号重复未另外核实。\n- 新病例到来前应锁定其分析规则，单独报告新病例结果，再提供合并结果。\n- 旧Doppler分析未保存随机种子，本次可复现运行以manifest为准；随机置换P值和bootstrap区间可能与旧表略有不同，点估计必须一致。\n\n## 复现\n\n运行 scripts/11_analyze_doppler_associations.py，指定 --master、--master-manifest、--audit、--qc、--repair 和 --output（尚不存在的目录）。原始输入未修改，未导出患者级合并表。manifest记录输入哈希、统计脚本哈希、种子和核验结果。\n'
    (out/'REPORT.md').write_text(report,encoding='utf-8')
    manifest['output_sha256'] = {p.name:sha(p) for p in out.iterdir() if p.is_file()}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DONE', out, flush=True)


if __name__ == '__main__':
    main()
