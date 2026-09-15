'use client';
import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, Cell, ResponsiveContainer, ReferenceLine } from 'recharts';

interface SHAPFeature {
  name: string;
  value: number;
  type?: 'positive' | 'negative';
}

interface Props {
  features: SHAPFeature[];
  title?: string;
  maxItems?: number;
}

export default function SHAPChart({ features, title = 'Feature Importance', maxItems = 15 }: Props) {
  const data = features
    .slice(0, maxItems)
    .sort((a, b) => b.value - a.value)
    .map(f => ({
      name: f.name.length > 22 ? f.name.slice(0, 22) + '…' : f.name,
      value: f.value,
      type: f.type ?? (f.value >= 0 ? 'positive' : 'negative'),
    }));

  const CustomTooltip = ({ active, payload }: any) => {
    if (!active || !payload?.length) return null;
    const d = payload[0].payload;
    return (
      <div className="bg-slate-900 border border-slate-700 rounded-sm px-3 py-2 text-xs font-mono">
        <p className="text-slate-300">{d.name}</p>
        <p className={d.value >= 0 ? 'text-blue-400' : 'text-red-400'}>
          SHAP: {d.value.toFixed(4)}
        </p>
      </div>
    );
  };

  return (
    <div className="w-full">
      {title && (
        <p className="text-xs font-mono text-slate-500 uppercase tracking-widest mb-4">{title}</p>
      )}
      <ResponsiveContainer width="100%" height={Math.max(200, data.length * 28)}>
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24, top: 4, bottom: 4 }}>
          <XAxis
            type="number"
            tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'JetBrains Mono' }}
            axisLine={{ stroke: '#334155' }}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="name"
            width={150}
            tick={{ fill: '#94a3b8', fontSize: 10, fontFamily: 'JetBrains Mono' }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
          <ReferenceLine x={0} stroke="#475569" strokeDasharray="3 3" />
          <Bar dataKey="value" radius={[0, 2, 2, 0]}>
            {data.map((entry, index) => (
              <Cell key={index} fill={entry.value >= 0 ? '#3b82f6' : '#ef4444'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
