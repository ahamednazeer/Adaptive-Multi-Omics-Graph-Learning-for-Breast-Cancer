'use client';
import React, { useEffect, useState } from 'react';
import DashboardLayout from '../components/DashboardLayout';
import EmptyState from '../components/EmptyState';
import ConfusionMatrix from '../components/ConfusionMatrix';
import ROCChart from '../components/ROCChart';
import { api } from '../lib/api';
import { ChartLineUp } from '@phosphor-icons/react';

const MODEL_COLORS: Record<string, string> = {
  'Logistic Regression': '#94a3b8',
  'XGBoost Baseline': '#f59e0b',
  'Fused Model': '#3b82f6',
};

export default function EvaluationPage() {
  const [data, setData] = useState<any>(null);
  const [selected, setSelected] = useState('Fused Model');
  const [detail, setDetail] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getEvaluation().then(setData).catch(() => null).finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!selected) return;
    api.getModelDetail(selected).then(setDetail).catch(() => setDetail(null));
  }, [selected]);

  const models = data?.model_results || [];

  const metricBar = (label: string, value: number | null | undefined, color: string) => (
    <div>
      <div className="flex items-center justify-between mb-1">
        <p className="text-[10px] font-mono text-slate-500 uppercase tracking-wider">{label}</p>
        <p className="text-xs font-mono font-bold text-slate-200">{value !== null && value !== undefined ? (value * 100).toFixed(1) + '%' : '—'}</p>
      </div>
      <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all" style={{ width: `${(value || 0) * 100}%`, backgroundColor: color }} />
      </div>
    </div>
  );

  return (
    <DashboardLayout pageTitle="Evaluation" pageSubtitle="Model performance and comparison">
      {loading ? (
        <div className="text-center py-16 text-slate-600 font-mono animate-pulse">Loading...</div>
      ) : !data?.pipeline_ran ? (
        <EmptyState title="No Evaluation Results" message="Run the full pipeline to see model evaluation." icon={<ChartLineUp size={48} weight="duotone" />} />
      ) : (
        <div className="space-y-6">
          {/* Comparison table */}
          <div className="bg-slate-800/40 border border-slate-700/60 rounded-sm overflow-hidden">
            <div className="px-6 py-4 border-b border-slate-700/50">
              <p className="font-chivo font-bold text-xs uppercase tracking-widest text-slate-500">Model Comparison</p>
            </div>
            <table className="w-full">
              <thead className="bg-slate-900/50">
                <tr>
                  {['Model','AUC','F1','Accuracy','Precision','Recall'].map(h => (
                    <th key={h} className="px-4 py-3 text-left text-xs font-mono text-slate-500 uppercase tracking-wider">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/50">
                {models.map((m: any) => (
                  <tr
                    key={m.model_name}
                    onClick={() => setSelected(m.model_name)}
                    className={`cursor-pointer transition-colors ${selected === m.model_name ? 'bg-blue-950/20' : 'hover:bg-slate-800/40'}`}
                  >
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full" style={{ backgroundColor: MODEL_COLORS[m.model_name] || '#64748b' }} />
                        <span className="font-mono text-sm text-slate-200">{m.model_name}</span>
                      </div>
                    </td>
                    {['roc_auc','f1','accuracy','precision','recall'].map(k => (
                      <td key={k} className="px-4 py-3 font-mono text-sm text-slate-200">
                        {m[k] !== null ? m[k]?.toFixed(4) : '—'}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Model detail */}
          {detail && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-slate-800/40 border border-slate-700/60 rounded-sm p-6">
                <p className="font-chivo font-bold text-xs uppercase tracking-widest text-slate-500 mb-4">{selected} — Metrics</p>
                <div className="space-y-3">
                  {metricBar('AUC-ROC', detail.roc_auc, MODEL_COLORS[selected] || '#3b82f6')}
                  {metricBar('F1 Score', detail.f1, '#10b981')}
                  {metricBar('Accuracy', detail.accuracy, '#8b5cf6')}
                  {metricBar('Precision', detail.precision, '#f59e0b')}
                  {metricBar('Recall', detail.recall, '#06b6d4')}
                </div>
              </div>

              <div className="bg-slate-800/40 border border-slate-700/60 rounded-sm p-6">
                <p className="font-chivo font-bold text-xs uppercase tracking-widest text-slate-500 mb-4">Confusion Matrix</p>
                {detail.confusion_matrix ? (
                  <ConfusionMatrix
                    tp={detail.confusion_matrix.tp} tn={detail.confusion_matrix.tn}
                    fp={detail.confusion_matrix.fp} fn={detail.confusion_matrix.fn}
                  />
                ) : <p className="text-slate-600 font-mono text-sm">No confusion matrix data.</p>}
              </div>

              {detail.roc_curve?.length > 0 && (
                <div className="lg:col-span-2 bg-slate-800/40 border border-slate-700/60 rounded-sm p-6">
                  <p className="font-chivo font-bold text-xs uppercase tracking-widest text-slate-500 mb-4">ROC Curve</p>
                  <ROCChart data={detail.roc_curve} auc={detail.roc_auc} color={MODEL_COLORS[selected]} label={selected} />
                </div>
              )}
            </div>
          )}

          {/* Cross-validation */}
          {data?.cross_validation && (
            <div className="bg-slate-800/40 border border-slate-700/60 rounded-sm p-6">
              <p className="font-chivo font-bold text-xs uppercase tracking-widest text-slate-500 mb-4">Cross-Validation Results</p>
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-4">
                {['accuracy','f1','roc_auc','precision','recall'].map(k => (
                  <div key={k} className="text-center p-3 bg-slate-900/50 rounded-sm border border-slate-800">
                    <p className="text-[10px] font-mono text-slate-500 uppercase tracking-wider">{k.replace('roc_','')}</p>
                    <p className="text-lg font-bold font-mono text-slate-100 mt-1">
                      {data.cross_validation[`${k}_mean`]?.toFixed(3) ?? '—'}
                    </p>
                    <p className="text-[10px] font-mono text-slate-600">
                      ±{data.cross_validation[`${k}_std`]?.toFixed(3) ?? '—'}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </DashboardLayout>
  );
}
