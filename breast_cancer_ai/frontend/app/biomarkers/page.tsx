'use client';
import React, { useEffect, useState } from 'react';
import DashboardLayout from '../components/DashboardLayout';
import EmptyState from '../components/EmptyState';
import { api } from '../lib/api';
import { Dna } from '@phosphor-icons/react';

export default function BiomarkersPage() {
  const [data, setData] = useState<any>(null);
  const [filter, setFilter] = useState({ direction: 'all', type: 'all' });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getBiomarkers(filter.direction, filter.type, 50)
      .then(setData).catch(() => setData(null)).finally(() => setLoading(false));
  }, [filter]);

  const biomarkers = data?.biomarkers || [];

  return (
    <DashboardLayout pageTitle="Biomarkers" pageSubtitle="SHAP-ranked gene and protein importance">
      {/* Filter bar */}
      <div className="flex items-center gap-3 mb-6">
        {[['Direction', 'direction', ['all','positive','negative']],
          ['Type', 'type', ['all','rna','protein']]].map(([label, key, opts]: any) => (
          <div key={key}>
            <label className="text-[10px] font-mono text-slate-600 uppercase tracking-wider mr-2">{label}</label>
            <select
              className="input-modern text-xs"
              value={(filter as any)[key]}
              onChange={e => setFilter(f => ({ ...f, [key]: e.target.value }))}
            >
              {opts.map((o: string) => <option key={o} value={o}>{o}</option>)}
            </select>
          </div>
        ))}
        {data?.total && <p className="text-xs font-mono text-slate-500 ml-auto">{data.total} biomarkers</p>}
      </div>

      {loading ? (
        <div className="text-center py-16 text-slate-600 font-mono animate-pulse">Loading...</div>
      ) : !data?.pipeline_ran ? (
        <EmptyState title="No Biomarkers Yet" message="Run the pipeline (steps 1-8) to discover biomarkers." icon={<Dna size={48} weight="duotone" />} />
      ) : (
        <div className="bg-slate-800/40 border border-slate-700/60 rounded-sm overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-slate-900/50">
                <tr>
                  {['Rank','Gene / Protein','Type','SHAP Score','Direction','Pathway'].map(h => (
                    <th key={h} className="px-4 py-3 text-left text-xs font-mono text-slate-500 uppercase tracking-wider">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/50">
                {biomarkers.map((b: any) => (
                  <tr key={b.gene} className="hover:bg-slate-800/30 transition-colors">
                    <td className="px-4 py-3 font-mono text-xs text-slate-600">#{b.rank}</td>
                    <td className="px-4 py-3 font-mono text-sm font-bold text-slate-200">{b.gene}</td>
                    <td className="px-4 py-3">
                      <span className={`text-[10px] font-mono px-2 py-0.5 rounded-sm uppercase tracking-wider
                        ${b.feature_type === 'rna' ? 'bg-blue-950/50 text-blue-400 border border-blue-800' : 'bg-purple-950/50 text-purple-400 border border-purple-800'}`}>
                        {b.feature_type}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <div className="w-24 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full ${b.direction === 'positive' ? 'bg-blue-500' : 'bg-red-500'}`}
                            style={{ width: `${Math.min(100, b.mean_abs_shap * 500)}%` }}
                          />
                        </div>
                        <span className="font-mono text-xs text-slate-300">{b.mean_abs_shap?.toFixed(4)}</span>
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <span className={`text-xs font-mono ${b.direction === 'positive' ? 'text-blue-400' : 'text-red-400'}`}>
                        {b.direction === 'positive' ? '↑ pCR' : '↓ pCR'}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-xs font-mono text-slate-500">{b.pathway || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </DashboardLayout>
  );
}
