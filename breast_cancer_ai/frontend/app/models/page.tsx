'use client';
import React, { useEffect, useState } from 'react';
import DashboardLayout from '../components/DashboardLayout';
import EmptyState from '../components/EmptyState';
import ROCChart from '../components/ROCChart';
import { api } from '../lib/api';
import { Robot } from '@phosphor-icons/react';

const COLORS: Record<string, string> = {
  'Logistic Regression': '#94a3b8',
  'XGBoost Baseline': '#f59e0b',
  'Fused Model': '#3b82f6',
};

export default function ModelsPage() {
  const [data, setData] = useState<any>(null);
  const [selected, setSelected] = useState('Fused Model');
  const [detail, setDetail] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getModelComparison().then(setData).catch(() => null).finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!selected) return;
    api.getModelDetail(selected).then(setDetail).catch(() => setDetail(null));
  }, [selected]);

  const models = data?.models || [];

  return (
    <DashboardLayout pageTitle="Models" pageSubtitle="Baseline vs proposed model comparison">
      {loading ? (
        <div className="text-center py-16 text-slate-600 font-mono animate-pulse">Loading...</div>
      ) : !data?.pipeline_ran ? (
        <EmptyState title="No Model Results" message="Run the pipeline to train and evaluate models." icon={<Robot size={48} weight="duotone" />} />
      ) : (
        <div className="space-y-6">
          {/* Model cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {models.map((m: any) => (
              <button
                key={m.model_name}
                onClick={() => setSelected(m.model_name)}
                className={`text-left p-5 rounded-sm border transition-all
                  ${selected === m.model_name
                    ? 'bg-blue-950/30 border-blue-600/50'
                    : 'bg-slate-800/40 border-slate-700/60 hover:border-slate-500'}`}
              >
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-2 h-2 rounded-full" style={{ backgroundColor: COLORS[m.model_name] || '#64748b' }} />
                  <p className="font-chivo font-bold text-xs uppercase tracking-wider text-slate-300">{m.model_name}</p>
                </div>
                <p className="text-3xl font-bold font-mono text-slate-100">{m.roc_auc?.toFixed(3) ?? '—'}</p>
                <p className="text-[10px] font-mono text-slate-500 uppercase tracking-wider mt-1">AUC-ROC</p>
                <div className="grid grid-cols-2 gap-2 mt-3 text-xs font-mono">
                  <div><span className="text-slate-500">F1: </span><span className="text-slate-300">{m.f1?.toFixed(3)}</span></div>
                  <div><span className="text-slate-500">Acc: </span><span className="text-slate-300">{m.accuracy?.toFixed(3)}</span></div>
                </div>
              </button>
            ))}
          </div>

          {/* Detail for selected model */}
          {detail && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-slate-800/40 border border-slate-700/60 rounded-sm p-6">
                <p className="font-chivo font-bold text-xs uppercase tracking-widest text-slate-500 mb-4">{selected} — Full Metrics</p>
                <div className="space-y-3">
                  {[['AUC-ROC', 'roc_auc'], ['F1 Score', 'f1'], ['Accuracy', 'accuracy'], ['Precision', 'precision'], ['Recall', 'recall']].map(([label, key]) => (
                    <div key={key} className="flex items-center justify-between">
                      <span className="text-xs font-mono text-slate-500 uppercase tracking-wider">{label}</span>
                      <div className="flex items-center gap-3">
                        <div className="w-32 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full"
                            style={{ width: `${((detail[key] || 0) * 100)}%`, backgroundColor: COLORS[selected] || '#3b82f6' }}
                          />
                        </div>
                        <span className="text-sm font-bold font-mono text-slate-100 w-14 text-right">
                          {detail[key]?.toFixed(4) ?? '—'}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>

                {detail.confusion_matrix && (
                  <div className="mt-6 pt-4 border-t border-slate-700/50">
                    <p className="text-xs font-mono text-slate-500 uppercase tracking-wider mb-3">Confusion Matrix</p>
                    <div className="grid grid-cols-2 gap-2 text-center text-xs font-mono">
                      {[
                        { label: 'TP', val: detail.confusion_matrix.tp, color: 'text-green-400 bg-green-950/30 border-green-800/40' },
                        { label: 'FP', val: detail.confusion_matrix.fp, color: 'text-red-400 bg-red-950/30 border-red-800/40' },
                        { label: 'FN', val: detail.confusion_matrix.fn, color: 'text-amber-400 bg-amber-950/30 border-amber-800/40' },
                        { label: 'TN', val: detail.confusion_matrix.tn, color: 'text-green-400 bg-green-950/30 border-green-800/40' },
                      ].map(c => (
                        <div key={c.label} className={`p-3 border rounded-sm ${c.color}`}>
                          <div className="text-xl font-bold">{c.val}</div>
                          <div className="text-[10px] opacity-70 uppercase tracking-wider">{c.label}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {detail.roc_curve?.length > 0 && (
                <div className="bg-slate-800/40 border border-slate-700/60 rounded-sm p-6">
                  <p className="font-chivo font-bold text-xs uppercase tracking-widest text-slate-500 mb-4">ROC Curve</p>
                  <ROCChart data={detail.roc_curve} auc={detail.roc_auc} color={COLORS[selected]} label={selected} />
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </DashboardLayout>
  );
}
