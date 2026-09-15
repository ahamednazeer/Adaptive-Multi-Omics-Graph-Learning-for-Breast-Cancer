'use client';
import React from 'react';

type Status = 'done' | 'running' | 'error' | 'pending' | 'idle' | 'active' | 'pcr' | 'no-pcr' | string;

interface Props {
  status: Status;
  className?: string;
}

const CONFIG: Record<string, { dot: string; text: string; label: string }> = {
  done:    { dot: 'bg-green-400',  text: 'text-green-400',  label: 'Done' },
  active:  { dot: 'bg-green-400',  text: 'text-green-400',  label: 'Active' },
  running: { dot: 'bg-blue-400 animate-pulse', text: 'text-blue-400', label: 'Running' },
  pending: { dot: 'bg-amber-400',  text: 'text-amber-400',  label: 'Pending' },
  idle:    { dot: 'bg-slate-500',  text: 'text-slate-500',  label: 'Idle' },
  error:   { dot: 'bg-red-400',    text: 'text-red-400',    label: 'Error' },
  pcr:     { dot: 'bg-green-400',  text: 'text-green-400',  label: 'pCR' },
  'no-pcr':{ dot: 'bg-red-400',    text: 'text-red-400',    label: 'No pCR' },
};

export default function StatusBadge({ status, className = '' }: Props) {
  const cfg = CONFIG[status] || { dot: 'bg-slate-600', text: 'text-slate-400', label: status };
  return (
    <span className={`inline-flex items-center gap-1.5 ${className}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${cfg.dot}`} />
      <span className={`text-[10px] font-mono uppercase tracking-wider ${cfg.text}`}>{cfg.label}</span>
    </span>
  );
}
