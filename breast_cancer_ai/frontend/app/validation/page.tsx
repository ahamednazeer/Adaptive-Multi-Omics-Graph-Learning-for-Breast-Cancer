'use client';
import React, { useEffect, useState } from 'react';
import DashboardLayout from '../components/DashboardLayout';
import EmptyState from '../components/EmptyState';
import ROCChart from '../components/ROCChart';
import { api } from '../lib/api';
import { CheckCircle } from '@phosphor-icons/react';

type Tab = 'external' | 'temporal' | 'resistance';

export default function ValidationPage() {
  const [tab, setTab] = useState<Tab>('external');
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getAllValidation().then(setData).catch(() => null).finally(() => setLoading(false));
  }, []);

  const TABS: { key: Tab; label: string }[] = [
    { key: 'external', label: 'External Validation (GSE240671)' },
    { key: 'temporal', label: 'Temporal Analysis (GSE122630)' },
    { key: 'resistance', label: 'Resistance Biomarkers' },
  ];

  if (loading) return (
    <DashboardLayout pageTitle="Validation" pageSubtitle="Temporal and external validation results">
      <div className="text-center py-16 text-slate-600 font-mono animate-pulse">Loading...</div>
    </DashboardLayout>
  );

  const notRan = !data?.pipeline_ran;

  return (
    <DashboardLayout pageTitle="Validation" pageSubtitle="Temporal and external dataset validation">
      {/* Tabs */}
      <div className="flex gap-1 mb-6 bg-slate-900/50 border border-slate-800 rounded-sm p-1 w-fit">
        {TABS.map(t => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`text-xs font-mono px-3 py-1.5 rounded-sm transition-colors uppercase tracking-wider
              ${tab === t.key ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-slate-200'}`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {notRan ? (
        <EmptyState title="No Validation Results" message="Run the full pipeline through step 9 to see validation." icon={<CheckCircle size={48} weight="duotone" />} />
      ) : tab === 'external' ? (
        <ExternalTab data={data?.external} />
      ) : tab === 'temporal' ? (
        <TemporalTab data={data?.temporal} />
      ) : (
        <ResistanceTab data={data?.resistance} />
      )}
    </DashboardLayout>
  );
}

function ExternalTab({ data }: { data: any }) {
  if (!data || !data.roc_auc) return <div className="text-slate-600 font-mono text-sm">No external validation data available.</div>;
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div className="bg-slate-800/40 border border-slate-700/60 rounded-sm p-6">
        <p className="font-chivo font-bold text-xs uppercase tracking-widest text-slate-500 mb-4">GSE240671 — Independent Validation</p>
        <div className="grid grid-cols-2 gap-3">
          {[['AUC-ROC', data.roc_auc], ['F1', data.f1], ['Accuracy', data.accuracy], ['Precision', data.precision]].map(([l, v]: any) => (
            <div key={l} className="p-3 bg-slate-900/50 rounded-sm border border-slate-800 text-center">
              <p className="text-[10px] font-mono text-slate-500 uppercase">{l}</p>
              <p className="text-xl font-bold font-mono text-slate-100 mt-1">{v?.toFixed(3) ?? '—'}</p>
            </div>
          ))}
        </div>
        <div className="mt-4 space-y-1.5">
          <div className="flex justify-between text-xs font-mono"><span className="text-slate-500">Samples</span><span className="text-slate-300">{data.n_samples}</span></div>
          <div className="flex justify-between text-xs font-mono"><span className="text-slate-500">Features used</span><span className="text-slate-300">{data.n_features_used}</span></div>
          <div className="flex justify-between text-xs font-mono"><span className="text-slate-500">Missing features</span><span className="text-amber-400">{data.n_features_missing}</span></div>
        </div>
      </div>
      {data.roc_curve?.length > 0 && (
        <div className="bg-slate-800/40 border border-slate-700/60 rounded-sm p-6">
          <p className="font-chivo font-bold text-xs uppercase tracking-widest text-slate-500 mb-4">ROC Curve</p>
          <ROCChart data={data.roc_curve} auc={data.roc_auc} color="#3b82f6" label="External" />
        </div>
      )}
    </div>
  );
}

function TemporalTab({ data }: { data: any }) {
  if (!data?.genes?.length) return <div className="text-slate-600 font-mono text-sm">No temporal data. GSE122630 uses ENSG IDs — gene mapping required.</div>;
  const genes = (data.genes || []).slice(0, 15);
  return (
    <div className="bg-slate-800/40 border border-slate-700/60 rounded-sm overflow-hidden">
      <div className="px-6 py-4 border-b border-slate-700/50">
        <p className="font-chivo font-bold text-xs uppercase tracking-widest text-slate-500">
          Gene Expression T1→T4 ({data.mapped_count} genes mapped, {data.unmapped_count} unmapped)
        </p>
      </div>
      <table className="w-full">
        <thead className="bg-slate-900/50">
          <tr>
            {['Gene','T1 (Pre)','T2 (Early)','T3 (Mid)','T4 (Surgery)','Direction','Change'].map(h => (
              <th key={h} className="px-4 py-3 text-left text-xs font-mono text-slate-500 uppercase tracking-wider">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800/50">
          {genes.map((g: any) => (
            <tr key={g.gene} className="hover:bg-slate-800/30">
              <td className="px-4 py-3 font-mono text-sm font-bold text-slate-200">{g.gene}</td>
              {['T1','T2','T3','T4'].map(tp => (
                <td key={tp} className="px-4 py-3 font-mono text-xs text-slate-300">
                  {g.timepoints?.[tp]?.mean?.toFixed(3) ?? '—'}
                </td>
              ))}
              <td className="px-4 py-3">
                <span className={`text-xs font-mono ${g.direction === 'up' ? 'text-blue-400' : 'text-red-400'}`}>
                  {g.direction === 'up' ? '↑ Up' : '↓ Down'}
                </span>
              </td>
              <td className="px-4 py-3 font-mono text-xs text-slate-400">{g.fold_change?.toFixed(3) ?? '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ResistanceTab({ data }: { data: any }) {
  if (!data?.genes?.length) return <div className="text-slate-600 font-mono text-sm">No resistance data available.</div>;
  return (
    <div className="space-y-4">
      {data.resistance_genes?.length > 0 && (
        <div className="p-4 bg-red-950/20 border border-red-800/40 rounded-sm">
          <p className="text-xs font-mono text-red-400 uppercase tracking-wider mb-2">Resistance-Associated Genes (upregulated in No-pCR)</p>
          <p className="font-mono text-sm text-slate-300">{data.resistance_genes.map((g: any) => g.gene).join(', ')}</p>
        </div>
      )}
      <div className="bg-slate-800/40 border border-slate-700/60 rounded-sm overflow-hidden">
        <table className="w-full">
          <thead className="bg-slate-900/50">
            <tr>
              {['Gene','Pre Mean','Post Mean','Fold Change','Direction','Resistance?'].map(h => (
                <th key={h} className="px-4 py-3 text-left text-xs font-mono text-slate-500 uppercase tracking-wider">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/50">
            {data.genes.slice(0, 20).map((g: any) => (
              <tr key={g.gene} className="hover:bg-slate-800/30">
                <td className="px-4 py-3 font-mono text-sm font-bold text-slate-200">{g.gene}</td>
                <td className="px-4 py-3 font-mono text-xs text-slate-300">{g.pre_mean?.toFixed(3)}</td>
                <td className="px-4 py-3 font-mono text-xs text-slate-300">{g.post_mean?.toFixed(3)}</td>
                <td className="px-4 py-3 font-mono text-xs text-slate-300">{g.fold_change?.toFixed(3)}</td>
                <td className="px-4 py-3"><span className={`font-mono text-xs ${g.direction === 'up' ? 'text-blue-400' : 'text-red-400'}`}>{g.direction === 'up' ? '↑' : '↓'}</span></td>
                <td className="px-4 py-3">{g.resistance_associated ? <span className="text-[10px] font-mono text-red-400 bg-red-950/40 border border-red-800/40 px-2 py-0.5 rounded-sm">YES</span> : <span className="text-slate-600">—</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
