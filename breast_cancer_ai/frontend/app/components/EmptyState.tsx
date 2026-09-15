'use client';
import React, { ReactNode } from 'react';
import { CircleDashed } from '@phosphor-icons/react';

interface Props {
  title?: string;
  message?: string;
  action?: ReactNode;
  icon?: ReactNode;
}

export default function EmptyState({
  title = 'No Data Yet',
  message = 'Run the pipeline first to see results here.',
  action,
  icon,
}: Props) {
  return (
    <div className="flex flex-col items-center justify-center py-24 text-center">
      <div className="text-slate-600 mb-4">
        {icon || <CircleDashed size={48} weight="duotone" />}
      </div>
      <p className="font-chivo font-bold text-lg uppercase tracking-widest text-slate-400 mb-2">{title}</p>
      <p className="text-sm font-mono text-slate-600 max-w-sm mb-6">{message}</p>
      {action}
    </div>
  );
}
