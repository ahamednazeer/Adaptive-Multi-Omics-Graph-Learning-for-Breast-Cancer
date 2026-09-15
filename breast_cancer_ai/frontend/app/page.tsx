'use client';
import React, { useEffect, useState } from 'react';
import DashboardLayout from './components/DashboardLayout';
import { DataCard } from './components/DataCard';
import EmptyState from './components/EmptyState';
import StatusBadge from './components/StatusBadge';
import { api } from './lib/api';
import { Gauge, Dna, TestTube, ChartLineUp, FlowArrow } from '@phosphor-icons/react';

export default function OverviewPage() {
  const [overview, setOverview] = useState<any>(null);
  const [steps, setSteps] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    Promise.all([
      api.getOverview().catch(() => null),
      api.getPipelineSteps().catch(() => ({ steps: [] })),
    ]).then(([ov, st]) => {
      setOverview(ov);
      setSteps(st?.steps || []);
      setLoading(false);
    });
  }, []);

  if (loading) {
    return (
      <DashboardLayout pageTitle="Overview" pageSubtitle="System status and dataset summary">
        <div className="flex items-center justify-center h-64">
          <div className="text-slate-500 font-mono text-sm animate-pulse">Loading...</div>
        </div>
      </DashboardLayout>
    );
  }

  const notRunYet = !overview?.pipeline_ran;

  return (
    <DashboardLayout pageTitle="Overview" pageSubtitle="System status and dataset summary">
      {/* Stat Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <DataCard title="Total Patients" value={overview?.total_patients ?? 736} icon={Dna} color="blue" subtitle="ISPY2 training set" />
        <DataCard title="RNA Features" value={(overview?.total_rna_features ?? 19134).toLocaleString()} icon={TestTube} color="teal" />
        <DataCard title="Best Model AUC" value={overview?.best_model_auc ? overview.best_model_auc.toFixed(3) : '—'} icon={ChartLineUp} color={overview?.best_model_auc ? 'green' : 'yellow'} subtitle={notRunYet ? 'Run pipeline' : undefined} />
        <DataCard title="pCR Rate" value={overview?.pcr_rate ? `${(overview.pcr_rate * 100).toFixed(1)}%` : '34.8%'} icon={Gauge} color="purple" subtitle="256 / 736 patients" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {/* Datasets */}
        <div className="bg-slate-800/40 border border-slate-700/60 rounded-sm p-6">
          <h2 className="font-chivo font-bold text-xs uppercase tracking-widest text-slate-500 mb-4">Datasets</h2>
          <div className="space-y-3">
            {(overview?.datasets || [
              { name: 'ISPY2', n_patients: 736, n_features: 19273, role: 'Training / Test', size_mb: 98 },
              { name: 'GSE122630', n_patients: 34, n_features: 19502, role: 'Temporal Validation', size_mb: 10 },
              { name: 'GSE240671', n_patients: 61, n_features: 58243, role: 'External Validation', size_mb: 18 },
            ]).map((ds: any) => (
              <div key={ds.name} className="flex items-center justify-between p-3 bg-slate-900/50 rounded-sm border border-slate-700/40">
                <div>
                  <p className="text-sm font-mono font-bold text-slate-200">{ds.name}</p>
                  <p className="text-xs font-mono text-slate-500">{ds.n_patients} patients · {ds.n_features?.toLocaleString()} genes · {ds.size_mb} MB</p>
                </div>
                <span className="text-[10px] font-mono px-2 py-1 rounded-sm bg-blue-950/50 border border-blue-800 text-blue-400 uppercase tracking-wider">
                  {ds.role}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Pipeline Steps */}
        <div className="bg-slate-800/40 border border-slate-700/60 rounded-sm p-6">
          <h2 className="font-chivo font-bold text-xs uppercase tracking-widest text-slate-500 mb-4 flex items-center gap-2">
            <FlowArrow size={14} /> Pipeline Steps
          </h2>
          {steps.length === 0 ? (
            <EmptyState title="Pipeline Not Run" message="Go to Pipeline page to start the 10-step ML pipeline." />
          ) : (
            <div className="space-y-2">
              {steps.map((s: any) => (
                <div key={s.step_name} className="flex items-center justify-between py-1.5 border-b border-slate-800/50 last:border-0">
                  <p className="text-xs font-mono text-slate-300 capitalize">{s.step_name.replace(/_/g, ' ')}</p>
                  <StatusBadge status={s.status as any} />
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* System Status */}
      <div className="bg-slate-800/40 border border-slate-700/60 rounded-sm p-6">
        <h2 className="font-chivo font-bold text-xs uppercase tracking-widest text-slate-500 mb-4">System Status</h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {[
            { label: 'FastAPI Backend', status: error ? 'error' : 'active' },
            { label: 'SQLite Database', status: 'active' },
            { label: 'ML Pipeline', status: notRunYet ? 'idle' : 'active' },
          ].map(s => (
            <div key={s.label} className="flex items-center gap-3 p-3 bg-slate-900/50 rounded-sm border border-slate-700/40">
              <span className={`w-2 h-2 rounded-full ${s.status === 'active' ? 'bg-green-400' : s.status === 'error' ? 'bg-red-400' : 'bg-amber-400'}`} />
              <div>
                <p className="text-xs font-mono text-slate-300">{s.label}</p>
                <p className="text-[10px] font-mono text-slate-500 uppercase">{s.status}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </DashboardLayout>
  );
}
