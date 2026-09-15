'use client';
import React from 'react';

interface Props {
  tp: number; tn: number; fp: number; fn: number;
}

export default function ConfusionMatrix({ tp, tn, fp, fn }: Props) {
  const total = tp + tn + fp + fn;
  const accuracy = total > 0 ? ((tp + tn) / total) : 0;
  const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
  const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0;
  const f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0;

  const Cell = ({ count, label, color }: { count: number; label: string; color: string }) => (
    <div className={`flex flex-col items-center justify-center p-6 border ${color} rounded-sm`}>
      <span className="text-3xl font-bold font-mono">{count}</span>
      <span className="text-xs font-mono uppercase tracking-widest mt-2 opacity-70">{label}</span>
    </div>
  );

  const Metric = ({ label, value }: { label: string; value: number }) => (
    <div className="text-center">
      <p className="text-xs font-mono text-slate-500 uppercase tracking-wider">{label}</p>
      <p className="text-xl font-bold font-mono text-slate-100">{(value * 100).toFixed(1)}%</p>
    </div>
  );

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-2">
        <Cell count={tp} label="True Positive" color="text-green-400 bg-green-950/30 border-green-800/50" />
        <Cell count={fp} label="False Positive" color="text-red-400 bg-red-950/30 border-red-800/50" />
        <Cell count={fn} label="False Negative" color="text-amber-400 bg-amber-950/30 border-amber-800/50" />
        <Cell count={tn} label="True Negative" color="text-green-400 bg-green-950/30 border-green-800/50" />
      </div>
      <div className="grid grid-cols-4 gap-4 pt-2 border-t border-slate-700/50">
        <Metric label="Accuracy" value={accuracy} />
        <Metric label="Precision" value={precision} />
        <Metric label="Recall" value={recall} />
        <Metric label="F1 Score" value={f1} />
      </div>
    </div>
  );
}
