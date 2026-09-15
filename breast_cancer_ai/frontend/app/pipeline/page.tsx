'use client';
import React, { useEffect, useState } from 'react';
import DashboardLayout from '../components/DashboardLayout';
import StatusBadge from '../components/StatusBadge';
import { api } from '../lib/api';
import { Play, ArrowClockwise } from '@phosphor-icons/react';
import { Square } from 'lucide-react';

const STEP_DESCRIPTIONS: Record<string, string> = {
  data_inspection: 'Load ISPY2, GSE122630, GSE240671 — validate shapes, missing values',
  preprocessing: 'Impute protein (~20% missing), normalize RNA/protein, one-hot encode treatments',
  splitting_baseline: 'Patient-level stratified split 70/10/20 + LR and XGBoost baseline',
  feature_selection: 'LASSO on 19K RNA → BORUTA refinement + protein quality filter',
  graph_construction: 'Map selected genes to STRING DB → build biological interaction graph',
  gcn_gat_training: 'GCN and GAT molecular representation on patient graphs',
  adaptive_fusion: 'Combine molecular embedding + clinical + treatment → pCR prediction',
  explainability: 'SHAP values → global biomarker ranking + patient cards',
  temporal_external_validation: 'GSE122630 T1-T4 trends + GSE240671 independent validation',
  evaluation: 'Cross-validation, calibration, ablation, model comparison',
};

export default function PipelinePage() {
  const [steps, setSteps] = useState<any[]>([]);
  const [status, setStatus] = useState<any>(null);
  const [running, setRunning] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [polling, setPolling] = useState(false);

  const fetchStatus = async () => {
    const [st, s] = await Promise.all([
      api.getPipelineSteps().catch(() => ({ steps: [] })),
      api.getPipelineStatus().catch(() => null),
    ]);
    setSteps(st?.steps || []);
    setStatus(s);
    if (s?.status === 'running') setPolling(true);
    else setPolling(false);
  };

  useEffect(() => { fetchStatus(); }, []);

  // Poll while running
  useEffect(() => {
    if (!polling) return;
    const id = setInterval(fetchStatus, 2000);
    return () => clearInterval(id);
  }, [polling]);

  const handleRun = async () => {
    setRunning(true);
    try {
      await api.runPipeline();
      setPolling(true);
      await fetchStatus();
    } finally {
      setRunning(false);
    }
  };

  const handleStop = async () => {
    setStopping(true);
    try {
      await api.stopPipeline();
      setPolling(false);
      setRunning(false);
      await fetchStatus();
    } catch (err) {
      console.error('Failed to stop pipeline:', err);
    } finally {
      setStopping(false);
    }
  };

  const completedCount = steps.filter(s => s.status === 'done').length;
  const totalSteps = steps.length || 10;
  const pct = Math.round((completedCount / totalSteps) * 100);

  const isRunning = status?.status === 'running' || running;

  return (
    <DashboardLayout pageTitle="Pipeline" pageSubtitle="10-step ML pipeline execution">
      {/* Header controls */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <p className="text-xs font-mono text-slate-500 uppercase tracking-wider">
            {status?.status === 'running' ? `Step: ${status.current_step?.replace(/_/g, ' ') || '...'}` :
             status?.status === 'done' ? `Completed — ${completedCount}/${totalSteps} steps` :
             status?.status === 'stopped' ? 'Pipeline stopped' :
             'Ready to run'}
          </p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={fetchStatus}
            className="btn-secondary flex items-center gap-2 text-xs"
          >
            <ArrowClockwise size={14} /> Refresh
          </button>
          {isRunning ? (
            <button
              onClick={handleStop}
              disabled={stopping}
              className="px-4 py-2 bg-red-600/90 hover:bg-red-500 active:bg-red-700 text-white font-mono text-xs font-semibold rounded-sm flex items-center gap-2 transition-all border border-red-500 shadow-lg shadow-red-900/40 cursor-pointer disabled:opacity-50"
            >
              <Square size={12} className="fill-current" />
              {stopping ? 'Stopping...' : 'Stop Pipeline'}
            </button>
          ) : (
            <button
              onClick={handleRun}
              disabled={running}
              className="btn-primary flex items-center gap-2 text-xs disabled:opacity-50"
            >
              <Play size={14} weight="fill" />
              Run Pipeline
            </button>
          )}
        </div>
      </div>

      {/* Progress bar */}
      {steps.length > 0 && (
        <div className="mb-6">
          <div className="flex items-center justify-between mb-1.5">
            <p className="text-xs font-mono text-slate-500 uppercase tracking-wider">Progress</p>
            <p className="text-xs font-mono text-blue-400">{pct}%</p>
          </div>
          <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-500 transition-all duration-500 rounded-full"
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>
      )}

      {/* Steps list */}
      <div className="space-y-3">
        {(steps.length > 0 ? steps : Object.keys(STEP_DESCRIPTIONS).map(k => ({ step_name: k, status: 'idle', message: '' }))).map((step: any, i: number) => (
          <div key={step.step_name} className={`bg-slate-800/40 border rounded-sm p-4 transition-all
            ${step.status === 'running' ? 'border-blue-500/60 bg-blue-950/10' :
              step.status === 'done' ? 'border-green-800/40' :
              step.status === 'error' ? 'border-red-800/50' :
              'border-slate-700/60'}`}
          >
            <div className="flex items-start justify-between gap-4">
              <div className="flex items-start gap-3 flex-1">
                <span className="text-xs font-mono text-slate-600 mt-0.5 w-4 shrink-0">{String(i + 1).padStart(2, '0')}</span>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-mono text-slate-200 capitalize">{step.step_name.replace(/_/g, ' ')}</p>
                  <p className="text-xs font-mono text-slate-500 mt-0.5">{STEP_DESCRIPTIONS[step.step_name] || ''}</p>
                  {step.message && step.status === 'running' && (
                    <p className="text-xs font-mono text-blue-400 mt-1 animate-pulse">{step.message}</p>
                  )}
                  {step.message && step.status === 'error' && (
                    <p className="text-xs font-mono text-red-400 mt-1">{step.message}</p>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-3 shrink-0">
                {step.duration_seconds && (
                  <span className="text-[10px] font-mono text-slate-600">{step.duration_seconds}s</span>
                )}
                <StatusBadge status={step.status === 'idle' ? 'idle' : step.status as any} />
              </div>
            </div>
          </div>
        ))}
      </div>

      {status?.error_message && (
        <div className="mt-4 p-4 bg-red-950/30 border border-red-800/50 rounded-sm">
          <p className="text-xs font-mono text-red-400">{status.error_message}</p>
        </div>
      )}
    </DashboardLayout>
  );
}
