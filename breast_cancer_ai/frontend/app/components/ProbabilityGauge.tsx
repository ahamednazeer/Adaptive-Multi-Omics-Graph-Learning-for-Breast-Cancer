'use client';
import React from 'react';

interface Props {
  probability: number; // 0 to 1
  size?: 'sm' | 'md' | 'lg';
}

export default function ProbabilityGauge({ probability, size = 'md' }: Props) {
  const pct = Math.max(0, Math.min(1, probability));
  const isPCR = pct >= 0.5;

  const dims = { sm: 120, md: 180, lg: 240 };
  const dim = dims[size];
  const cx = dim / 2;
  const cy = dim / 2;
  const r = dim * 0.38;
  const strokeW = dim * 0.065;

  // Arc from 210° to 330° (spans 300° total)
  const startAngle = 210;
  const endAngle = 330; // going clockwise past 0°
  const totalDeg = 300;
  const filledDeg = pct * totalDeg;

  function polarToXY(angle: number, radius: number) {
    const rad = (angle * Math.PI) / 180;
    return { x: cx + radius * Math.cos(rad), y: cy + radius * Math.sin(rad) };
  }

  function arc(startDeg: number, sweepDeg: number, radius: number) {
    const s = polarToXY(startDeg, radius);
    const e = polarToXY(startDeg + sweepDeg, radius);
    const largeArc = sweepDeg > 180 ? 1 : 0;
    return `M ${s.x} ${s.y} A ${radius} ${radius} 0 ${largeArc} 1 ${e.x} ${e.y}`;
  }

  const trackColor = '#1e293b';
  const fillColor = isPCR ? '#22c55e' : '#ef4444';
  const textColor = isPCR ? '#22c55e' : '#ef4444';
  const glowColor = isPCR ? 'rgba(34,197,94,0.25)' : 'rgba(239,68,68,0.25)';

  const fontSize = { sm: 18, md: 28, lg: 36 }[size];
  const labelSize = { sm: 8, md: 10, lg: 12 }[size];

  return (
    <div className="flex flex-col items-center gap-2">
      <div style={{ filter: `drop-shadow(0 0 12px ${glowColor})` }}>
        <svg width={dim} height={dim * 0.75} viewBox={`0 0 ${dim} ${dim * 0.75}`}>
          {/* Track */}
          <path
            d={arc(startAngle, totalDeg, r)}
            fill="none"
            stroke={trackColor}
            strokeWidth={strokeW}
            strokeLinecap="round"
          />
          {/* Fill */}
          {pct > 0 && (
            <path
              d={arc(startAngle, filledDeg, r)}
              fill="none"
              stroke={fillColor}
              strokeWidth={strokeW}
              strokeLinecap="round"
            />
          )}
          {/* Percentage text */}
          <text
            x={cx}
            y={cy * 0.95}
            textAnchor="middle"
            dominantBaseline="middle"
            fill={textColor}
            fontSize={fontSize}
            fontFamily="JetBrains Mono, monospace"
            fontWeight="bold"
          >
            {Math.round(pct * 100)}%
          </text>
          {/* Label */}
          <text
            x={cx}
            y={cy * 0.95 + fontSize * 0.9}
            textAnchor="middle"
            dominantBaseline="middle"
            fill="#64748b"
            fontSize={labelSize}
            fontFamily="JetBrains Mono, monospace"
            letterSpacing="2"
          >
            PROBABILITY
          </text>
        </svg>
      </div>
      <div className={`text-sm font-bold font-mono uppercase tracking-widest ${isPCR ? 'text-green-400' : 'text-red-400'}`}>
        {isPCR ? 'pCR PREDICTED' : 'NO pCR PREDICTED'}
      </div>
    </div>
  );
}
