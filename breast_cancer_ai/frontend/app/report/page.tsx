'use client';
import React, { useEffect, useState } from 'react';
import DashboardLayout from '../components/DashboardLayout';
import { api } from '../lib/api';
import { FileText, DownloadSimple } from '@phosphor-icons/react';

export default function ReportPage() {
  const [overview, setOverview] = useState<any>(null);
  const [evaluation, setEvaluation] = useState<any>(null);
  const [biomarkers, setBiomarkers] = useState<any[]>([]);
  const [validation, setValidation] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      api.getOverview().catch(() => null),
      api.getEvaluation().catch(() => null),
      api.getBiomarkers('all', 'all', 15).catch(() => null),
      api.getAllValidation().catch(() => null),
    ]).then(([ov, ev, bm, val]) => {
      setOverview(ov);
      setEvaluation(ev);
      setBiomarkers(bm?.biomarkers || []);
      setValidation(val);
      setLoading(false);
    });
  }, []);

  const now = new Date().toISOString().slice(0, 19).replace('T', ' ');
  const pipelineRan = overview?.pipeline_ran;

  const Section = ({ title, children }: { title: string; children: React.ReactNode }) => (
    <div className="bg-slate-800/40 border border-slate-700/60 rounded-sm p-6 mb-6">
      <h2 className="font-chivo font-bold text-xs uppercase tracking-widest text-slate-500 mb-4 pb-2 border-b border-slate-700/50">{title}</h2>
      {children}
    </div>
  );

  const Row = ({ label, value }: { label: string; value: any }) => (
    <div className="flex items-center justify-between py-1.5 border-b border-slate-800/40 last:border-0">
      <span className="text-xs font-mono text-slate-500">{label}</span>
      <span className="text-xs font-mono text-slate-200 font-bold">{value ?? '—'}</span>
    </div>
  );

  const bestModel = evaluation?.model_results?.sort((a: any, b: any) => (b.roc_auc || 0) - (a.roc_auc || 0))[0];

  return (
    <DashboardLayout pageTitle="Research Report" pageSubtitle="Full pipeline summary and findings">
      <div className="flex items-center justify-between mb-6">
        <p className="text-xs font-mono text-slate-600">Generated: {now}</p>
        <button
          onClick={() => window.print()}
          className="btn-secondary flex items-center gap-2 text-xs"
        >
          <DownloadSimple size={14} /> Export / Print
        </button>
      </div>

      {!pipelineRan && (
        <div className="p-4 mb-6 bg-amber-950/30 border border-amber-800/40 rounded-sm">
          <p className="text-xs font-mono text-amber-400">Pipeline has not been run yet. Run the pipeline to populate this report with real results.</p>
        </div>
      )}

      {/* Title */}
      <div className="text-center py-8 mb-6 border border-slate-700/40 rounded-sm bg-slate-900/30">
        <div className="flex justify-center mb-4"><FileText size={40} weight="duotone" className="text-blue-400" /></div>
        <h1 className="font-chivo font-bold text-2xl uppercase tracking-widest text-slate-100">Adaptive Multi-Omics Graph Learning</h1>
        <h2 className="font-chivo font-bold text-lg uppercase tracking-wider text-slate-400 mt-1">for Breast Cancer pCR Prediction</h2>
        <p className="text-xs font-mono text-slate-600 mt-4">ISPY2 · GSE122630 · GSE240671</p>
      </div>

      <Section title="1. Dataset Summary">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[
            { name: 'ISPY2', role: 'Training / Test', patients: overview?.total_patients ?? 736, features: (overview?.total_rna_features ?? 19134) + (overview?.total_protein_features ?? 139), pcr_rate: overview?.pcr_rate ?? 0.348 },
            { name: 'GSE122630', role: 'Temporal Validation', patients: 34, features: 19502, pcr_rate: null },
            { name: 'GSE240671', role: 'External Validation', patients: 61, features: 58243, pcr_rate: null },
          ].map(ds => (
            <div key={ds.name} className="p-4 bg-slate-900/50 rounded-sm border border-slate-700/40">
              <p className="font-mono font-bold text-sm text-slate-200 mb-2">{ds.name}</p>
              <Row label="Role" value={ds.role} />
              <Row label="Patients" value={ds.patients} />
              <Row label="Features" value={ds.features?.toLocaleString()} />
              {ds.pcr_rate !== null && <Row label="pCR Rate" value={`${(ds.pcr_rate * 100).toFixed(1)}%`} />}
            </div>
          ))}
        </div>
      </Section>

      <Section title="2. Feature Selection">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center p-3 bg-slate-900/50 rounded-sm border border-slate-800">
            <p className="text-2xl font-bold font-mono text-slate-100">{overview?.total_rna_features?.toLocaleString() ?? '19,134'}</p>
            <p className="text-[10px] font-mono text-slate-500 uppercase mt-1">Initial RNA</p>
          </div>
          <div className="text-center p-3 bg-slate-900/50 rounded-sm border border-slate-800">
            <p className="text-2xl font-bold font-mono text-blue-400">LASSO</p>
            <p className="text-[10px] font-mono text-slate-500 uppercase mt-1">→ ~500 genes</p>
          </div>
          <div className="text-center p-3 bg-slate-900/50 rounded-sm border border-slate-800">
            <p className="text-2xl font-bold font-mono text-teal-400">Boruta</p>
            <p className="text-[10px] font-mono text-slate-500 uppercase mt-1">→ All-relevant</p>
          </div>
          <div className="text-center p-3 bg-slate-900/50 rounded-sm border border-slate-800">
            <p className="text-2xl font-bold font-mono text-slate-100">{overview?.total_protein_features ?? 139}</p>
            <p className="text-[10px] font-mono text-slate-500 uppercase mt-1">RPPA Proteins</p>
          </div>
        </div>
      </Section>

      <Section title="3. Model Performance">
        {bestModel ? (
          <div className="space-y-2">
            {evaluation?.model_results?.map((m: any) => (
              <div key={m.model_name} className="flex items-center justify-between p-3 bg-slate-900/50 rounded-sm border border-slate-800">
                <span className="font-mono text-sm text-slate-200">{m.model_name}</span>
                <div className="flex items-center gap-6 text-xs font-mono">
                  <span className="text-slate-500">AUC: <strong className="text-slate-100">{m.roc_auc?.toFixed(4)}</strong></span>
                  <span className="text-slate-500">F1: <strong className="text-slate-100">{m.f1?.toFixed(4)}</strong></span>
                  <span className="text-slate-500">Acc: <strong className="text-slate-100">{m.accuracy?.toFixed(4)}</strong></span>
                </div>
              </div>
            ))}
          </div>
        ) : <p className="text-slate-600 font-mono text-sm">Run the pipeline to see model results.</p>}
      </Section>

      <Section title="4. Top Biomarkers (SHAP)">
        {biomarkers.length > 0 ? (
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-xs font-mono text-blue-400 uppercase tracking-wider mb-2">pCR Promoters</p>
              {biomarkers.filter(b => b.direction === 'positive').slice(0, 8).map((b, i) => (
                <div key={b.gene} className="flex items-center justify-between py-1 border-b border-slate-800/40 last:border-0">
                  <span className="font-mono text-xs text-slate-300">#{i+1} {b.gene}</span>
                  <span className="font-mono text-xs text-blue-400">+{b.mean_abs_shap?.toFixed(4)}</span>
                </div>
              ))}
            </div>
            <div>
              <p className="text-xs font-mono text-red-400 uppercase tracking-wider mb-2">pCR Inhibitors</p>
              {biomarkers.filter(b => b.direction === 'negative').slice(0, 8).map((b, i) => (
                <div key={b.gene} className="flex items-center justify-between py-1 border-b border-slate-800/40 last:border-0">
                  <span className="font-mono text-xs text-slate-300">#{i+1} {b.gene}</span>
                  <span className="font-mono text-xs text-red-400">{b.mean_shap?.toFixed(4)}</span>
                </div>
              ))}
            </div>
          </div>
        ) : <p className="text-slate-600 font-mono text-sm">Run pipeline steps 1-8 to discover biomarkers.</p>}
      </Section>

      <Section title="5. External Validation (GSE240671)">
        {validation?.external?.roc_auc ? (
          <div className="grid grid-cols-4 gap-4">
            {[['AUC-ROC', validation.external.roc_auc], ['F1', validation.external.f1], ['Accuracy', validation.external.accuracy], ['Samples', validation.external.n_samples]].map(([l, v]: any) => (
              <div key={l} className="text-center p-3 bg-slate-900/50 rounded-sm border border-slate-800">
                <p className="text-xl font-bold font-mono text-slate-100">{typeof v === 'number' && v < 2 ? v.toFixed(3) : v}</p>
                <p className="text-[10px] font-mono text-slate-500 uppercase mt-1">{l}</p>
              </div>
            ))}
          </div>
        ) : <p className="text-slate-600 font-mono text-sm">External validation not yet computed.</p>}
      </Section>

      <Section title="6. Temporal Analysis (GSE122630)">
        {validation?.temporal?.mapped_count > 0 ? (
          <p className="text-xs font-mono text-slate-300">
            {validation.temporal.mapped_count} biomarker genes tracked across T1→T4 timepoints.<br />
            {validation.temporal.unmapped_count} genes unmapped (ENSG→symbol mapping required for remaining genes).
          </p>
        ) : <p className="text-slate-600 font-mono text-sm">Temporal analysis not yet computed.</p>}
      </Section>
    </DashboardLayout>
  );
}
