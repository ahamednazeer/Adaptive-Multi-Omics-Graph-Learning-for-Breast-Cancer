'use client';
import React from 'react';

interface DataCardProps {
  title: string;
  value: string | number;
  icon?: React.ElementType;
  className?: string;
  subtitle?: string;
  trend?: { value: number; label: string };
  color?: 'blue' | 'green' | 'red' | 'yellow' | 'purple' | 'teal';
}

const COLOR_MAP: Record<string, string> = {
  blue:   'text-blue-400',
  green:  'text-green-400',
  red:    'text-red-400',
  yellow: 'text-amber-400',
  purple: 'text-purple-400',
  teal:   'text-teal-400',
};

export function DataCard({ title, value, icon: Icon, className = '', subtitle, trend, color = 'blue' }: DataCardProps) {
  const colorClass = COLOR_MAP[color] || COLOR_MAP.blue;

  return (
    <div className={`bg-slate-800/40 border border-slate-700/60 rounded-sm p-5 ${className}`}>
      <div className="flex items-start justify-between mb-3">
        <div className="min-w-0 flex-1">
          <p className="text-[10px] font-mono text-slate-500 uppercase tracking-wider truncate">{title}</p>
        </div>
        {Icon && <Icon size={22} weight="duotone" className={`${colorClass} ml-3 shrink-0`} />}
      </div>
      <p className={`text-2xl font-bold font-mono text-slate-100 truncate`}>{value}</p>
      {subtitle && <p className="text-[10px] font-mono text-slate-600 mt-1 uppercase tracking-wider">{subtitle}</p>}
      {trend && (
        <div className="mt-2 flex items-center gap-1.5">
          <span className={`text-[10px] font-mono ${trend.value >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {trend.value >= 0 ? '+' : ''}{trend.value}%
          </span>
          <span className="text-[10px] font-mono text-slate-600">{trend.label}</span>
        </div>
      )}
    </div>
  );
}

export default DataCard;
