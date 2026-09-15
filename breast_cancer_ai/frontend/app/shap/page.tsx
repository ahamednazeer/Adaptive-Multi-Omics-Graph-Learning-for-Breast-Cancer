'use client';
import React, { useEffect, useState } from 'react';
import DashboardLayout from '../components/DashboardLayout';
import EmptyState from '../components/EmptyState';
import SHAPChart from '../components/SHAPChart';
import { api } from '../lib/api';
import { ChartBar } from '@phosphor-icons/react';

export default function SHAPPage() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getGlobalSHAP(40).then(setData).catch(() => setData(null)).finally(() => setLoading(false));
  }, []);

  const features = data?.features || [];
  const positive = features.filter((f: any) => f.value >= 0);
  const negative = features.filter((f: any) => f.value < 0);

  return (
    <DashboardLayout pageTitle="SHAP Explorer" pageSubtitle="Global feature importance from explainable AI">
      {loading ? (
        <div className="text-center py-16 text-slate-600 font-mono animate-pulse">Loading...</div>
      ) : !data?.pipeline_ran ? (
        <EmptyState title="No SHAP Data" message="Run the pipeline through step 8 (Explainability) to see SHAP values." icon={<ChartBar size={48} weight="duotone" />} />
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* All features */}
          <div className="lg:col-span-2 bg-slate-800/40 border border-slate-700/60 rounded-sm p-6">
            <p className="font-chivo font-bold text-xs uppercase tracking-widest text-slate-500 mb-6">
              Global SHAP — Top {features.length} Features (by |SHAP|)
            </p>
            <SHAPChart features={features.slice(0, 30)} title="" maxItems={30} />
          </div>

          {/* Positive drivers */}
          <div className="bg-slate-800/40 border border-slate-700/60 rounded-sm p-6">
            <p className="font-chivo font-bold text-xs uppercase tracking-widest text-blue-400 mb-4">
              pCR Promoters — Top Positive Features
            </p>
            <div className="space-y-2">
              {positive.slice(0, 10).map((f: any, i: number) => (
                <div key={f.name} className="flex items-center justify-between gap-3 py-1.5 border-b border-slate-800/40 last:border-0">
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] font-mono text-slate-600 w-4">#{i+1}</span>
                    <span className="font-mono text-xs text-slate-200">{f.name}</span>
                  </div>
                  <span className="font-mono text-xs text-blue-400">+{f.value.toFixed(4)}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Negative drivers */}
          <div className="bg-slate-800/40 border border-slate-700/60 rounded-sm p-6">
            <p className="font-chivo font-bold text-xs uppercase tracking-widest text-red-400 mb-4">
              pCR Inhibitors — Top Negative Features
            </p>
            <div className="space-y-2">
              {negative.slice(0, 10).map((f: any, i: number) => (
                <div key={f.name} className="flex items-center justify-between gap-3 py-1.5 border-b border-slate-800/40 last:border-0">
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] font-mono text-slate-600 w-4">#{i+1}</span>
                    <span className="font-mono text-xs text-slate-200">{f.name}</span>
                  </div>
                  <span className="font-mono text-xs text-red-400">{f.value.toFixed(4)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </DashboardLayout>
  );
}
