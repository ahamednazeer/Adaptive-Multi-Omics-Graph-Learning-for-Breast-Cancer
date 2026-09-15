'use client';
import React, { useState } from 'react';
import DashboardLayout from '../components/DashboardLayout';
import ProbabilityGauge from '../components/ProbabilityGauge';
import SHAPChart from '../components/SHAPChart';
import { api } from '../lib/api';
import { Info, Sparkles, Dna, Activity, AlertCircle, CheckCircle2, FlaskConical } from 'lucide-react';

interface Preset {
  name: string;
  badge: string;
  badgeColor: string;
  her2: string;
  hr: string;
  treatment_arm: string;
  description: string;
}

const PRESETS: Preset[] = [
  {
    name: 'Triple Negative (TNBC)',
    badge: 'High Chemo Sensitivity',
    badgeColor: 'bg-purple-950/60 text-purple-300 border-purple-800',
    her2: '0',
    hr: '0',
    treatment_arm: 'Paclitaxel + Pembrolizumab',
    description: 'Aggressive tumor lacking hormone & HER2 receptors. Highest response to chemo + immunotherapy combinations.',
  },
  {
    name: 'HER2-Enriched',
    badge: 'Targeted Candidate',
    badgeColor: 'bg-blue-950/60 text-blue-300 border-blue-800',
    her2: '1',
    hr: '0',
    treatment_arm: 'Paclitaxel + Pertuzumab',
    description: 'Driven by HER2 protein overexpression. Exceptional response when combined with targeted HER2 inhibitors.',
  },
  {
    name: 'Triple-Positive (Luminal B)',
    badge: 'Dual Receptor',
    badgeColor: 'bg-teal-950/60 text-teal-300 border-teal-800',
    her2: '1',
    hr: '1',
    treatment_arm: 'Paclitaxel + Neratinib',
    description: 'Expresses both hormone receptors and HER2. Requires multi-agent dual blockade to prevent resistance.',
  },
  {
    name: 'Hormone-Driven (Luminal A)',
    badge: 'Endocrine Driven',
    badgeColor: 'bg-amber-950/60 text-amber-300 border-amber-800',
    her2: '0',
    hr: '1',
    treatment_arm: 'Paclitaxel (Control)',
    description: 'Slow-growing, hormone-dependent. Often exhibits lower response to chemo alone; primarily relies on endocrine pills.',
  },
];

const TREATMENT_DETAILS: Record<string, { category: string; description: string }> = {
  'Paclitaxel + Pembrolizumab': {
    category: 'Chemo + Immunotherapy',
    description: 'Trains the immune system to recognize and attack tumor cells alongside chemotherapy.',
  },
  'Paclitaxel + Pertuzumab': {
    category: 'Chemo + Targeted HER2 Antibody',
    description: 'Dual HER2 receptor blockade preventing cancer growth signals and triggering antibody cytotoxicity.',
  },
  'Paclitaxel + Neratinib': {
    category: 'Chemo + Pan-HER Tyrosine Kinase Inhibitor',
    description: 'Oral small-molecule inhibitor blocking HER1, HER2, and HER4 intracellular signaling.',
  },
  'Paclitaxel + MK-2206': {
    category: 'Chemo + AKT Survival Inhibitor',
    description: 'Blocks the PI3K/AKT cell survival pathway, preventing cancer cells from surviving chemo.',
  },
  'Paclitaxel + ABT 888 + Carboplatin': {
    category: 'Chemo + PARP DNA-Repair Inhibitor',
    description: 'Induces synthetic lethality in tumors with BRCA or DNA damage repair defects.',
  },
  'Paclitaxel + AMG-386': {
    category: 'Chemo + Angiogenesis Inhibitor',
    description: 'Inhibits Angiopoietin-1 and 2 to starve the tumor of new blood vessel supply.',
  },
  'Paclitaxel + T-DM1/Pertuzumab': {
    category: 'Chemo + Antibody-Drug Conjugate',
    description: 'Delivers toxic payload directly inside HER2-positive cells with minimal off-target toxicity.',
  },
  'Paclitaxel (Control)': {
    category: 'Standard Chemotherapy Benchmark',
    description: 'Standard neoadjuvant taxane monotherapy without additional targeted agents.',
  },
};

export default function PredictionPage() {
  const [form, setForm] = useState({
    her2: '0',
    hr: '1',
    treatment_arm: 'Paclitaxel + Pembrolizumab',
  });
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const applyPreset = (preset: Preset) => {
    setForm({
      her2: preset.her2,
      hr: preset.hr,
      treatment_arm: preset.treatment_arm,
    });
    setResult(null);
    setError('');
  };

  const getSubtypeInfo = () => {
    const isHer2Pos = form.her2 === '1';
    const isHrPos = form.hr === '1';

    if (!isHer2Pos && !isHrPos) {
      return {
        name: 'Triple-Negative (TNBC)',
        summary: 'Aggressive phenotype with high proliferation. Chemosensitive and benefits strongly from immunotherapy.',
        badgeColor: 'text-purple-400 bg-purple-950/50 border-purple-800/60',
      };
    }
    if (isHer2Pos && !isHrPos) {
      return {
        name: 'HER2-Enriched',
        summary: 'High HER2 oncogene expression. Excellent candidates for targeted dual antibody therapies (Pertuzumab).',
        badgeColor: 'text-blue-400 bg-blue-950/50 border-blue-800/60',
      };
    }
    if (isHer2Pos && isHrPos) {
      return {
        name: 'Triple-Positive / Luminal HER2',
        summary: 'Driven by both hormones and HER2. Best treated with targeted HER2 inhibitors followed by endocrine maintenance.',
        badgeColor: 'text-teal-400 bg-teal-950/50 border-teal-800/60',
      };
    }
    return {
      name: 'Luminal / Hormone-Receptor Positive',
      summary: 'Estrogen/progesterone driven. Usually slower growing; responds best to endocrine therapy with variable chemo response.',
      badgeColor: 'text-amber-400 bg-amber-950/50 border-amber-800/60',
    };
  };

  const subtype = getSubtypeInfo();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const res = await api.predictPatient({
        her2: Number(form.her2),
        hr: Number(form.hr),
        treatment_arm: form.treatment_arm,
      });
      setResult(res);
    } catch (err: any) {
      setError(err.message || 'Prediction failed. Please ensure the pipeline has completed training.');
    } finally {
      setLoading(false);
    }
  };

  const probability = result?.probability ?? 0;
  const isHighResponse = probability >= 0.50;
  const isModerateResponse = probability >= 0.35 && probability < 0.50;

  return (
    <DashboardLayout pageTitle="New Patient Prediction" pageSubtitle="Simulate and predict pathological complete response (pCR)">
      <div className="space-y-8 max-w-5xl">
        {/* Presets Header */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-sm p-5 backdrop-blur-sm">
          <div className="flex items-center gap-2 mb-2 text-slate-300">
            <Sparkles size={16} className="text-blue-400" />
            <h2 className="text-xs font-mono uppercase tracking-wider font-semibold">1-Click Clinical Presets (Quick Profiles)</h2>
          </div>
          <p className="text-xs text-slate-500 mb-4">
            Click any clinical profile below to automatically populate the biopsy receptors and optimal treatment arm:
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {PRESETS.map((p) => {
              const isSelected = form.her2 === p.her2 && form.hr === p.hr && form.treatment_arm === p.treatment_arm;
              return (
                <button
                  key={p.name}
                  type="button"
                  onClick={() => applyPreset(p)}
                  className={`text-left p-3 rounded-sm border transition-all cursor-pointer ${
                    isSelected
                      ? 'bg-blue-950/40 border-blue-500 shadow-md shadow-blue-950/50'
                      : 'bg-slate-800/40 border-slate-700/60 hover:bg-slate-800/80 hover:border-slate-600'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-xs font-bold text-slate-200">{p.name}</span>
                  </div>
                  <span className={`inline-block text-[10px] font-mono px-1.5 py-0.5 rounded-xs border mb-2 ${p.badgeColor}`}>
                    {p.badge}
                  </span>
                  <p className="text-[11px] text-slate-400 line-clamp-2 leading-relaxed">{p.description}</p>
                </button>
              );
            })}
          </div>
        </div>

        {/* Main Grid: Form & Result */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Left Form (7 cols) */}
          <div className="lg:col-span-7 bg-slate-900/60 border border-slate-800 rounded-sm p-6 backdrop-blur-sm space-y-6">
            <div className="flex items-center justify-between border-b border-slate-800 pb-4">
              <div className="flex items-center gap-2">
                <Dna size={16} className="text-blue-400" />
                <h3 className="font-chivo font-bold text-xs uppercase tracking-widest text-slate-300">Biopsy & Prescription Parameters</h3>
              </div>
              <span className={`text-xs font-mono px-2 py-0.5 rounded-sm border ${subtype.badgeColor}`}>
                {subtype.name}
              </span>
            </div>

            <form onSubmit={handleSubmit} className="space-y-6">
              {/* HER2 Status Field */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-mono text-slate-300 uppercase tracking-wider font-semibold">
                    1. HER2 Receptor Status
                  </label>
                  <span className="text-[11px] font-mono text-slate-500">IHC / FISH Score</span>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <button
                    type="button"
                    onClick={() => setForm((f) => ({ ...f, her2: '0' }))}
                    className={`p-3 text-left rounded-sm border transition-all cursor-pointer ${
                      form.her2 === '0'
                        ? 'bg-blue-950/40 border-blue-400 text-slate-100'
                        : 'bg-slate-800/40 border-slate-700/60 text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    <div className="font-mono text-xs font-bold mb-1">Negative (0)</div>
                    <div className="text-[11px] text-slate-400">Normal HER2 gene copies. No excess growth antenna on tumor cells.</div>
                  </button>

                  <button
                    type="button"
                    onClick={() => setForm((f) => ({ ...f, her2: '1' }))}
                    className={`p-3 text-left rounded-sm border transition-all cursor-pointer ${
                      form.her2 === '1'
                        ? 'bg-blue-950/40 border-blue-400 text-slate-100'
                        : 'bg-slate-800/40 border-slate-700/60 text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    <div className="font-mono text-xs font-bold mb-1">Positive (1)</div>
                    <div className="text-[11px] text-slate-400">High HER2 overexpression. Sensitive to anti-HER2 targeted drugs.</div>
                  </button>
                </div>
              </div>

              {/* Hormone Receptor Status Field */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-mono text-slate-300 uppercase tracking-wider font-semibold">
                    2. Hormone Receptor (HR) Status
                  </label>
                  <span className="text-[11px] font-mono text-slate-500">Estrogen / Progesterone</span>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <button
                    type="button"
                    onClick={() => setForm((f) => ({ ...f, hr: '1' }))}
                    className={`p-3 text-left rounded-sm border transition-all cursor-pointer ${
                      form.hr === '1'
                        ? 'bg-blue-950/40 border-blue-400 text-slate-100'
                        : 'bg-slate-800/40 border-slate-700/60 text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    <div className="font-mono text-xs font-bold mb-1">Positive (1)</div>
                    <div className="text-[11px] text-slate-400">Tumor feeds on estrogen/progesterone. Responsive to hormone blockers.</div>
                  </button>

                  <button
                    type="button"
                    onClick={() => setForm((f) => ({ ...f, hr: '0' }))}
                    className={`p-3 text-left rounded-sm border transition-all cursor-pointer ${
                      form.hr === '0'
                        ? 'bg-blue-950/40 border-blue-400 text-slate-100'
                        : 'bg-slate-800/40 border-slate-700/60 text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    <div className="font-mono text-xs font-bold mb-1">Negative (0)</div>
                    <div className="text-[11px] text-slate-400">Hormone independent. Cells do not require estrogen to proliferate.</div>
                  </button>
                </div>
              </div>

              {/* Treatment Regimen Field */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-mono text-slate-300 uppercase tracking-wider font-semibold">
                    3. Prescribed Neoadjuvant Chemotherapy Regimen
                  </label>
                  <span className="text-[11px] font-mono text-slate-500">I-SPY2 Trial Arms</span>
                </div>
                <select
                  className="w-full bg-slate-900 border border-slate-700 rounded-sm p-3 text-sm text-slate-200 focus:outline-none focus:border-blue-500 font-mono"
                  value={form.treatment_arm}
                  onChange={(e) => setForm((f) => ({ ...f, treatment_arm: e.target.value }))}
                >
                  {Object.keys(TREATMENT_DETAILS).map((arm) => (
                    <option key={arm} value={arm}>
                      {arm} — [{TREATMENT_DETAILS[arm].category}]
                    </option>
                  ))}
                </select>
                {TREATMENT_DETAILS[form.treatment_arm] && (
                  <div className="p-3 bg-slate-800/40 border border-slate-700/50 rounded-sm flex items-start gap-2.5 mt-2">
                    <FlaskConical size={16} className="text-blue-400 shrink-0 mt-0.5" />
                    <div>
                      <p className="text-xs font-semibold text-blue-300 font-mono">
                        {TREATMENT_DETAILS[form.treatment_arm].category}
                      </p>
                      <p className="text-[11px] text-slate-400 mt-0.5 leading-relaxed">
                        {TREATMENT_DETAILS[form.treatment_arm].description}
                      </p>
                    </div>
                  </div>
                )}
              </div>

              {/* Informative Note */}
              <div className="p-3 bg-slate-800/30 border border-slate-800 rounded-sm flex items-start gap-2 text-slate-400">
                <Info size={15} className="text-slate-500 shrink-0 mt-0.5" />
                <p className="text-[11px] leading-relaxed">
                  The model aligns patient clinical receptors with population median multi-omics gene expression from 736 patients to calculate personal response probability.
                </p>
              </div>

              {error && (
                <div className="p-3 bg-red-950/40 border border-red-800/60 rounded-sm text-xs text-red-300 flex items-center gap-2 font-mono">
                  <AlertCircle size={16} className="text-red-400 shrink-0" />
                  {error}
                </div>
              )}

              <button
                type="submit"
                disabled={loading}
                className="w-full py-3 bg-blue-600 hover:bg-blue-500 active:bg-blue-700 text-white font-mono text-xs font-semibold uppercase tracking-wider rounded-sm transition-all shadow-lg shadow-blue-950/50 flex items-center justify-center gap-2 cursor-pointer disabled:opacity-50"
              >
                {loading ? (
                  <>
                    <Activity size={14} className="animate-spin" /> Calculating Response Probability...
                  </>
                ) : (
                  <>
                    <Sparkles size={14} /> Calculate pCR Response Probability
                  </>
                )}
              </button>
            </form>
          </div>

          {/* Right Result (5 cols) */}
          <div className="lg:col-span-5 space-y-6">
            {result ? (
              <div className="space-y-6 animate-fade-in">
                {/* Result Card */}
                <div className="bg-slate-900/60 border border-slate-800 rounded-sm p-6 flex flex-col items-center backdrop-blur-sm">
                  <p className="font-chivo font-bold text-xs uppercase tracking-widest text-slate-400 mb-4">
                    Predicted Response Outcome
                  </p>
                  <ProbabilityGauge probability={result.probability} size="lg" />

                  {/* Clinical Response Summary Badge */}
                  <div className="mt-5 w-full">
                    {isHighResponse ? (
                      <div className="p-3 bg-emerald-950/40 border border-emerald-800/60 rounded-sm text-center">
                        <div className="flex items-center justify-center gap-1.5 text-emerald-300 font-mono text-xs font-bold mb-1">
                          <CheckCircle2 size={15} /> FAVORABLE RESPONSE EXPECTED
                        </div>
                        <p className="text-[11px] text-emerald-400/90 leading-relaxed">
                          Tumor has high likelihood of Pathological Complete Response (cancer completely eradicated before surgery).
                        </p>
                      </div>
                    ) : isModerateResponse ? (
                      <div className="p-3 bg-amber-950/40 border border-amber-800/60 rounded-sm text-center">
                        <div className="flex items-center justify-center gap-1.5 text-amber-300 font-mono text-xs font-bold mb-1">
                          <AlertCircle size={15} /> INTERMEDIATE RESPONSE LIKELIHOOD
                        </div>
                        <p className="text-[11px] text-amber-400/90 leading-relaxed">
                          Partial tumor shrinkage anticipated. Close longitudinal ultrasound monitoring during early cycles is recommended.
                        </p>
                      </div>
                    ) : (
                      <div className="p-3 bg-rose-950/40 border border-rose-800/60 rounded-sm text-center">
                        <div className="flex items-center justify-center gap-1.5 text-rose-300 font-mono text-xs font-bold mb-1">
                          <AlertCircle size={15} /> LOW RESPONSE / RESISTANT PROFILE
                        </div>
                        <p className="text-[11px] text-rose-400/90 leading-relaxed">
                          Chemo monotherapy unlikely to eradicate tumor. Consider multi-agent targeted clinical trial or alternative regimen.
                        </p>
                      </div>
                    )}
                  </div>

                  {/* Metadata Stats */}
                  <div className="mt-4 grid grid-cols-2 gap-3 w-full">
                    <div className="text-center p-3 bg-slate-900/80 rounded-sm border border-slate-800">
                      <p className="text-[10px] font-mono text-slate-500 uppercase">Model Architecture</p>
                      <p className="text-xs font-mono text-slate-200 mt-1 font-semibold">{result.model_type}</p>
                    </div>
                    <div className="text-center p-3 bg-slate-900/80 rounded-sm border border-slate-800">
                      <p className="text-[10px] font-mono text-slate-500 uppercase">pCR Probability</p>
                      <p className="text-base font-bold font-mono text-blue-400 mt-1">{(result.probability * 100).toFixed(1)}%</p>
                    </div>
                  </div>
                </div>

                {/* SHAP Drivers if available */}
                {result.top_features?.length > 0 && (
                  <div className="bg-slate-900/60 border border-slate-800 rounded-sm p-6 backdrop-blur-sm">
                    <p className="font-chivo font-bold text-xs uppercase tracking-widest text-slate-400 mb-2">
                      Key Biological Drivers
                    </p>
                    <p className="text-[11px] text-slate-500 mb-4 font-mono">
                      Features pushing probability higher (Blue) or lower (Red)
                    </p>
                    <SHAPChart
                      features={result.top_features.slice(0, 8).map((f: any) => ({
                        name: f.name,
                        value: f.shap_value,
                        type: f.type,
                      }))}
                    />
                  </div>
                )}
              </div>
            ) : (
              <div className="bg-slate-900/40 border border-slate-800/80 rounded-sm p-8 flex flex-col items-center justify-center text-center h-full min-h-[340px]">
                <div className="w-12 h-12 rounded-full bg-slate-800/50 flex items-center justify-center mb-3 border border-slate-700/50">
                  <Activity size={24} className="text-slate-500" />
                </div>
                <h4 className="text-sm font-semibold text-slate-300 font-mono mb-1">Awaiting Patient Parameters</h4>
                <p className="text-xs text-slate-500 max-w-xs leading-relaxed">
                  Select an example preset or choose custom biopsy receptors and click &quot;Calculate pCR Response Probability&quot;.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}
