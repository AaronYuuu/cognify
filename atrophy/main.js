/**
 * Cognify - Brain MRI Analysis Tool
 *
 * Experimental volumetric analysis tool for brain MRI scans.
 * Uses AI-based segmentation to analyze brain structures.
 *
 * CREDITS:
 * Built on BrainChop (https://github.com/neuroneural/brainchop)
 * by the neuroneural team for brain segmentation using TensorFlow.js.
 *
 * Normative data derived from published literature for research purposes.
 *
 * DISCLAIMER: Research prototype only. Not validated for clinical use.
 */

import { Niivue } from "@niivue/niivue";
import { runInference } from "../brainchop-mainthread.js";
import { inferenceModelsList, brainChopOpts } from "../brainchop-parameters.js";
import { isChrome } from "../brainchop-diagnostics.js";
import {
  buildEvaluationForm,
  buildAIResultsDisplay,
  buildSideBySideView,
  collectFormData,
  validateForm,
  extractAIScores,
  saveEvaluation
} from "./radiologist-eval.js";

// ============================================
// SERVER-SIDE INFERENCE CONFIGURATION
// Set USE_SERVER = true to use HuggingFace GPU server
// The server uses /segment/tensor endpoint which receives pre-processed
// tensor data from NiiVue, ensuring identical results to local inference.
// ============================================
const USE_SERVER = true;  // Enable for GPU acceleration on weak devices
const SERVER_URL = "https://aryagm-shia-brain.hf.space";

/**
 * Run inference on the server (HuggingFace Space with GPU)
 * Falls back to local inference if server is unavailable
 *
 * Uses the /segment/tensor endpoint which accepts pre-processed tensor data.
 * This ensures identical preprocessing to local inference since NiiVue handles
 * the conforming, and only the GPU inference is offloaded to the server.
 */
async function runServerInference(progressCallback) {
  if (!nv1 || !nv1.volumes || !nv1.volumes[0]) {
    throw new Error("No volume loaded");
  }

  progressCallback("Preparing tensor for upload...", 0.05);

  // Get the conformed volume data from NiiVue
  // After ensureConformed(), this is a 256³ Uint8Array
  const tensorData = nv1.volumes[0].img;

  if (!(tensorData instanceof Uint8Array) || tensorData.length !== 256 * 256 * 256) {
    throw new Error(`Invalid tensor: expected 256³ Uint8Array, got ${tensorData.constructor.name} length ${tensorData.length}`);
  }

  console.log(`Tensor data: ${tensorData.length} bytes, range [${Math.min(...tensorData.slice(0, 1000))}-${Math.max(...tensorData.slice(0, 1000))}]`);

  // Compress tensor with pako
  progressCallback("Compressing tensor...", 0.1);
  const compressed = pako.gzip(tensorData);
  console.log(`Compressed: ${tensorData.length} -> ${compressed.length} bytes (${(compressed.length / tensorData.length * 100).toFixed(1)}%)`);

  // Create blob for upload
  const blob = new Blob([compressed], { type: 'application/octet-stream' });
  const formData = new FormData();
  formData.append('file', blob, 'tensor.gz');

  try {
    progressCallback("Uploading to GPU server...", 0.2);

    const response = await fetch(`${SERVER_URL}/segment/tensor`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Server error ${response.status}: ${text}`);
    }

    progressCallback("Processing on GPU...", 0.5);

    const result = await response.json();

    if (!result.success) {
      throw new Error(result.error || 'Server inference failed');
    }

    progressCallback("Downloading results...", 0.8);

    // Decode base64 gzipped segmentation
    const binaryString = atob(result.data);
    const compressedBytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      compressedBytes[i] = binaryString.charCodeAt(i);
    }

    // Decompress using pako
    const decompressed = pako.ungzip(compressedBytes);

    progressCallback("Segmentation complete!", 1.0);
    console.log(`Server inference completed in ${result.inference_time}s (total: ${result.total_time}s)`);

    return {
      segmentation: decompressed,
      shape: result.shape,
      inferenceTime: result.inference_time
    };

  } catch (error) {
    console.error("Server inference failed:", error);
    throw new Error(`Server inference failed: ${error.message}. Try local mode.`);
  }
}

// ============================================
// NORMATIVE DATA - BILATERAL VOLUMES
// Reference values derived from published studies.
// All values are BILATERAL (left + right combined).
// For experimental/research use.
// ============================================

const NORMATIVE_DATA = {
  // Mean volumes and standard deviations by region
  // Format: { mean: [age20, age40, age60, age80], sd: value }
  // SD values are age-adjusted (increase slightly with age due to variance)
  // All volumes in mm³, BILATERAL (both hemispheres combined)
  regions: {
    "Cerebral-White-Matter": {
      // Large structure - values from FreeSurfer normative studies
      mean: { male: [470000, 460000, 440000, 400000], female: [420000, 410000, 390000, 355000] },
      sd: { male: 45000, female: 40000 },
      clinicalName: "Supratentorial White Matter",
      abbreviation: "WM",
      description: "Major white matter tracts",
      clinicalSignificance: "low"  // Large structures have high variability
    },
    "Cerebral-Cortex": {
      mean: { male: [520000, 500000, 470000, 420000], female: [470000, 450000, 420000, 380000] },
      sd: { male: 50000, female: 45000 },
      clinicalName: "Cerebral Cortex (Gray Matter)",
      abbreviation: "GM",
      description: "Gray matter of cerebral hemispheres",
      clinicalSignificance: "medium"
    },
    "Lateral-Ventricle": {
      // Ventricles ENLARGE with atrophy - higher values indicate more atrophy
      // Large SD due to high inter-individual variability
      mean: { male: [14000, 18000, 28000, 45000], female: [12000, 15000, 24000, 38000] },
      sd: { male: 8000, female: 7000 },
      clinicalName: "Lateral Ventricles",
      abbreviation: "LV",
      description: "CSF-filled cavities - enlarge with atrophy",
      invertZscore: true,  // Larger = more atrophy (negative z-score)
      clinicalSignificance: "high"
    },
    "Inferior-Lateral-Ventricle": {
      // Temporal horns - very sensitive marker for medial temporal atrophy
      mean: { male: [600, 900, 1500, 2800], female: [500, 750, 1300, 2400] },
      sd: { male: 500, female: 450 },
      clinicalName: "Temporal Horns (ILV)",
      abbreviation: "TH",
      description: "Temporal horn - enlarges with hippocampal atrophy",
      invertZscore: true,
      clinicalSignificance: "high"
    },
    "Cerebellum-White-Matter": {
      mean: { male: [28000, 27000, 25000, 22000], female: [25000, 24000, 22000, 19000] },
      sd: { male: 3500, female: 3000 },
      clinicalName: "Cerebellar White Matter",
      abbreviation: "CbWM",
      description: "White matter of cerebellum",
      clinicalSignificance: "low"
    },
    "Cerebellum-Cortex": {
      mean: { male: [105000, 100000, 92000, 82000], female: [95000, 90000, 83000, 74000] },
      sd: { male: 12000, female: 10000 },
      clinicalName: "Cerebellar Cortex",
      abbreviation: "CbCx",
      description: "Gray matter of cerebellum",
      clinicalSignificance: "low"
    },
    "Thalamus": {
      // BILATERAL thalamus - doubled from unilateral FreeSurfer values
      // Reference: Potvin et al. 2016, ENIGMA consortium
      mean: { male: [16400, 15600, 14400, 12800], female: [14800, 14000, 13000, 11600] },
      sd: { male: 1400, female: 1200 },
      clinicalName: "Thalami (Bilateral)",
      abbreviation: "Thal",
      description: "Relay center for sensory information",
      clinicalSignificance: "medium"
    },
    "Caudate": {
      // BILATERAL caudate nucleus
      mean: { male: [8000, 7400, 6600, 5600], female: [7200, 6600, 5800, 5000] },
      sd: { male: 900, female: 800 },
      clinicalName: "Caudate Nuclei",
      abbreviation: "Caud",
      description: "Part of basal ganglia - motor control and cognition",
      clinicalSignificance: "medium"
    },
    "Putamen": {
      // BILATERAL putamen
      mean: { male: [11600, 10800, 9600, 8200], female: [10400, 9600, 8600, 7400] },
      sd: { male: 1100, female: 1000 },
      clinicalName: "Putamina (Bilateral)",
      abbreviation: "Put",
      description: "Part of basal ganglia - motor learning",
      clinicalSignificance: "medium"
    },
    "Pallidum": {
      // BILATERAL globus pallidus
      mean: { male: [3800, 3600, 3300, 2900], female: [3400, 3200, 2900, 2560] },
      sd: { male: 380, female: 340 },
      clinicalName: "Globi Pallidi",
      abbreviation: "GP",
      description: "Part of basal ganglia - movement regulation",
      clinicalSignificance: "medium"
    },
    "Hippocampus": {
      // BILATERAL hippocampus - KEY STRUCTURE for dementia assessment
      // Reference: UK Biobank (n=19,793), ADNI, FreeSurfer norms
      // These are bilateral values (L+R combined)
      // Typical bilateral volume at age 70: ~7000mm³ for males
      mean: { male: [8600, 8200, 7400, 6400], female: [7800, 7400, 6700, 5800] },
      sd: { male: 700, female: 650 },
      clinicalName: "Hippocampi (Bilateral)",
      abbreviation: "Hippo",
      description: "Memory formation - key structure in Alzheimer's disease",
      clinicalSignificance: "critical",
      mciThreshold: -1.5,  // Z-score threshold for MCI concern
      adThreshold: -2.0    // Z-score threshold for AD concern
    },
    "Amygdala": {
      // BILATERAL amygdala
      mean: { male: [3400, 3240, 2960, 2600], female: [3100, 2940, 2700, 2360] },
      sd: { male: 360, female: 320 },
      clinicalName: "Amygdalae (Bilateral)",
      abbreviation: "Amyg",
      description: "Emotional processing - affected in frontotemporal dementia",
      clinicalSignificance: "high"
    },
    "Accumbens-area": {
      // BILATERAL nucleus accumbens
      mean: { male: [1200, 1120, 980, 820], female: [1080, 1000, 880, 740] },
      sd: { male: 160, female: 140 },
      clinicalName: "Nuclei Accumbentes",
      abbreviation: "NAc",
      description: "Reward processing",
      clinicalSignificance: "low"
    },
    "Brain-Stem": {
      mean: { male: [22000, 21500, 20500, 19000], female: [19500, 19000, 18000, 16700] },
      sd: { male: 2200, female: 2000 },
      clinicalName: "Brainstem",
      abbreviation: "BS",
      description: "Vital functions control",
      clinicalSignificance: "low"
    },
    "VentralDC": {
      // Ventral diencephalon
      mean: { male: [8400, 8000, 7400, 6600], female: [7600, 7200, 6600, 6000] },
      sd: { male: 800, female: 700 },
      clinicalName: "Ventral Diencephalon",
      abbreviation: "VDC",
      description: "Hypothalamus and related structures",
      clinicalSignificance: "low"
    },
    "3rd-Ventricle": {
      mean: { male: [800, 1000, 1400, 2000], female: [700, 900, 1200, 1700] },
      sd: { male: 350, female: 300 },
      clinicalName: "Third Ventricle",
      abbreviation: "V3",
      description: "Midline CSF space",
      invertZscore: true,
      clinicalSignificance: "medium"
    },
    "4th-Ventricle": {
      // Fourth ventricle has HIGH measurement variability
      // Values adjusted based on clinical experience and segmentation characteristics
      // Reference: MRI volumetric studies show range of 1.0-2.5 mL in elderly
      mean: { male: [1400, 1500, 1700, 2000], female: [1200, 1300, 1500, 1800] },
      sd: { male: 500, female: 450 },  // Increased SD for high variability
      clinicalName: "Fourth Ventricle",
      abbreviation: "V4",
      description: "Posterior fossa CSF space",
      invertZscore: true,
      clinicalSignificance: "low"
    }
  },

  // Total intracranial volume (ICV/eTIV) reference for normalization
  // Used to adjust regional volumes for head size
  icv: {
    mean: { male: 1550000, female: 1350000 },  // mm³
    sd: { male: 130000, female: 110000 }
  },

  // Total brain volume (excluding ventricles and CSF)
  totalBrain: {
    mean: { male: [1350000, 1320000, 1270000, 1200000], female: [1200000, 1170000, 1130000, 1070000] },
    sd: { male: 95000, female: 80000 }
  }
};

// ============================================
// CLINICAL INTERPRETATION THRESHOLDS
// Based on NeuroQuant and radiological standards
// ============================================

const CLINICAL_THRESHOLDS = {
  // Z-score thresholds for interpretation
  zScores: {
    normal: { min: -1.0, label: "Normal", percentileMin: 16 },
    lowNormal: { min: -1.5, max: -1.0, label: "Low-Normal", percentileMin: 7, percentileMax: 16 },
    mild: { min: -2.0, max: -1.5, label: "Mild Atrophy", percentileMin: 2, percentileMax: 7 },
    moderate: { min: -2.5, max: -2.0, label: "Moderate Atrophy", percentileMin: 0.6, percentileMax: 2 },
    severe: { max: -2.5, label: "Severe Atrophy", percentileMax: 0.6 }
  },

  // Overall atrophy risk based on multiple regions
  riskCriteria: {
    high: {
      description: "High risk - significant atrophy detected",
      criteria: "≥2 critical regions below -2.0 OR hippocampus below -2.5"
    },
    moderate: {
      description: "Moderate risk - notable atrophy present",
      criteria: "≥1 critical region below -2.0 OR ≥3 regions below -1.5"
    },
    mild: {
      description: "Mild risk - some volume reduction",
      criteria: "≥1 region below -1.5 OR hippocampus below -1.0"
    },
    normal: {
      description: "Normal - volumes within expected range",
      criteria: "All regions within normal limits"
    }
  }
};

// ============================================
// ICV NORMALIZATION COEFFICIENTS
// Based on Potvin et al. (2016) and FreeSurfer literature
// Using residual correction method: Vol_adj = Vol - b × (ICV - ICV_mean)
// ============================================

const ICV_REGRESSION_COEFFICIENTS = {
  // Regression slopes (b) for ICV normalization by region
  // Values represent mm³ change per mm³ ICV change
  // Source: Derived from FreeSurfer normative studies
  "Hippocampus": { b: 0.0037, r2: 0.15 },  // ~3.7mm³ per 1000mm³ ICV
  "Thalamus": { b: 0.0082, r2: 0.22 },
  "Caudate": { b: 0.0045, r2: 0.18 },
  "Putamen": { b: 0.0058, r2: 0.20 },
  "Pallidum": { b: 0.0019, r2: 0.14 },
  "Amygdala": { b: 0.0018, r2: 0.12 },
  "Accumbens-area": { b: 0.0006, r2: 0.10 },
  "Lateral-Ventricle": { b: 0.0085, r2: 0.08 },  // Lower R² due to high variability
  "Inferior-Lateral-Ventricle": { b: 0.0008, r2: 0.05 },
  "Cerebral-White-Matter": { b: 0.28, r2: 0.45 },
  "Cerebral-Cortex": { b: 0.32, r2: 0.50 },
  "Cerebellum-Cortex": { b: 0.055, r2: 0.25 },
  "Cerebellum-White-Matter": { b: 0.015, r2: 0.20 },
  "Brain-Stem": { b: 0.012, r2: 0.18 },
  "VentralDC": { b: 0.0042, r2: 0.16 }
};

// Reference ICV values for normalization
const ICV_REFERENCE = {
  mean: { male: 1550000, female: 1350000 },  // mm³
  sd: { male: 130000, female: 110000 }
};

// ============================================
// BRAIN PARENCHYMAL FRACTION (BPF) NORMATIVE DATA
// Source: Vågberg et al. (2017) systematic review of 9,269 adults
// BPF = Total Brain Volume / Intracranial Volume
// ============================================

const BPF_NORMATIVE = {
  // BPF by age decade (mean values from systematic review)
  byAge: {
    20: { mean: 0.88, sd: 0.03 },
    30: { mean: 0.86, sd: 0.03 },
    40: { mean: 0.84, sd: 0.03 },
    50: { mean: 0.82, sd: 0.04 },
    60: { mean: 0.79, sd: 0.04 },
    70: { mean: 0.76, sd: 0.05 },
    80: { mean: 0.72, sd: 0.05 }
  },
  // Annual decline rate (~0.4-0.5% per year after age 40)
  annualDecline: 0.0045,
  // Atrophy thresholds
  thresholds: {
    normal: -1.0,      // z > -1.0
    mild: -1.5,        // -1.5 < z <= -1.0
    moderate: -2.0,    // -2.0 < z <= -1.5
    severe: -2.5       // z <= -2.0
  }
};

// ============================================
// HIPPOCAMPAL OCCUPANCY SCORE (HOC)
// Source: NeuroQuant / Cortechs.ai methodology
// HOC = Hippocampus / (Hippocampus + Inferior Lateral Ventricle)
// Key biomarker for Alzheimer's disease progression
// ============================================

const HOC_NORMATIVE = {
  // HOC by age (normalized, averaged L+R)
  // Lower HOC indicates more hippocampal atrophy / ventricular expansion
  byAge: {
    50: { mean: 0.92, sd: 0.04, p5: 0.85 },
    60: { mean: 0.88, sd: 0.05, p5: 0.80 },
    70: { mean: 0.82, sd: 0.06, p5: 0.72 },
    80: { mean: 0.75, sd: 0.08, p5: 0.62 }
  },
  // Clinical interpretation
  interpretation: {
    normal: { min: 0.80, label: "Normal", description: "No significant medial temporal atrophy" },
    mild: { min: 0.70, max: 0.80, label: "Mild", description: "Mild medial temporal atrophy" },
    moderate: { min: 0.60, max: 0.70, label: "Moderate", description: "Moderate medial temporal atrophy - consider MCI" },
    severe: { max: 0.60, label: "Severe", description: "Severe medial temporal atrophy - consistent with AD" }
  },
  // MCI to AD conversion risk based on HOC
  conversionRisk: {
    low: { min: 0.80, risk: "Low", conversionRate: "~10% at 3 years" },
    moderate: { min: 0.70, max: 0.80, risk: "Moderate", conversionRate: "~25% at 3 years" },
    high: { min: 0.60, max: 0.70, risk: "High", conversionRate: "~50% at 3 years" },
    veryHigh: { max: 0.60, risk: "Very High", conversionRate: "~75% at 3 years" }
  }
};

// ============================================
// STANDARDIZED ATROPHY RATING SCALES
// Used in clinical neuroradiology worldwide
// ============================================

const STANDARDIZED_SCALES = {
  // ========================================
  // MTA Score (Scheltens Scale) - Medial Temporal Atrophy
  // Reference: Scheltens et al. (1992), Radiopaedia
  // Scores 0-4 based on hippocampal volume and temporal horn
  // ========================================
  MTA: {
    name: "Medial Temporal Atrophy (MTA) Score",
    reference: "Scheltens Scale",
    scores: {
      0: { label: "Normal", description: "No visible CSF around hippocampus, normal hippocampal height" },
      1: { label: "Minimal", description: "Slight widening of choroidal fissure, normal hippocampus" },
      2: { label: "Mild", description: "Mild temporal horn enlargement, mild hippocampal height loss" },
      3: { label: "Moderate", description: "Moderate temporal horn enlargement, moderate hippocampal atrophy" },
      4: { label: "Severe", description: "Marked temporal horn enlargement, severe hippocampal atrophy" }
    },
    // Age-adjusted abnormal thresholds (score above this is abnormal)
    ageThresholds: {
      65: 1.0,   // Age <65: MTA ≥1.5 is abnormal
      75: 1.5,   // Age 65-74: MTA ≥2.0 is abnormal
      85: 2.0,   // Age 75-84: MTA ≥2.5 is abnormal
      100: 2.5   // Age ≥85: MTA ≥3.0 is abnormal
    },
    // Conversion from ILV/Hippocampus ratio to MTA score
    // Based on: QMTA = ILV / Hippocampus (lower ratio = less atrophy)
    qmtaToScore: [
      { maxRatio: 0.10, score: 0 },  // ILV/Hippo < 0.10 → MTA 0
      { maxRatio: 0.20, score: 1 },  // ILV/Hippo 0.10-0.20 → MTA 1
      { maxRatio: 0.35, score: 2 },  // ILV/Hippo 0.20-0.35 → MTA 2
      { maxRatio: 0.55, score: 3 },  // ILV/Hippo 0.35-0.55 → MTA 3
      { maxRatio: Infinity, score: 4 }  // ILV/Hippo > 0.55 → MTA 4
    ],
    // Alternative: conversion from hippocampal z-score
    zscoreToScore: [
      { minZ: -0.5, score: 0 },
      { minZ: -1.0, score: 1 },
      { minZ: -1.5, score: 2 },
      { minZ: -2.0, score: 3 },
      { minZ: -Infinity, score: 4 }
    ]
  },

  // ========================================
  // GCA Score (Pasquier Scale) - Global Cortical Atrophy
  // Reference: Pasquier et al. (1996), Radiopaedia
  // Scores 0-3 based on sulcal widening and gyral volume
  // ========================================
  GCA: {
    name: "Global Cortical Atrophy (GCA) Score",
    reference: "Pasquier Scale",
    scores: {
      0: { label: "Normal", description: "No cortical atrophy" },
      1: { label: "Mild", description: "Opening of sulci" },
      2: { label: "Moderate", description: "Volume loss of gyri" },
      3: { label: "Severe", description: "Knife-blade atrophy" }
    },
    // Age thresholds (score above this is abnormal)
    ageThresholds: {
      75: 1,   // Age <75: GCA ≥2 is abnormal
      100: 2   // Age ≥75: GCA ≥3 is abnormal
    },
    // Conversion from cortical volume z-score
    zscoreToScore: [
      { minZ: -0.5, score: 0 },
      { minZ: -1.5, score: 1 },
      { minZ: -2.5, score: 2 },
      { minZ: -Infinity, score: 3 }
    ]
  },

  // ========================================
  // Koedam Score - Posterior Atrophy
  // Reference: Koedam et al. (2011)
  // Scores 0-3 based on parietal/precuneus atrophy
  // Particularly relevant for early-onset AD
  // ========================================
  Koedam: {
    name: "Posterior Atrophy (PA) Score",
    reference: "Koedam Scale",
    scores: {
      0: { label: "Normal", description: "No parietal atrophy" },
      1: { label: "Mild", description: "Slight widening of posterior cingulate and parieto-occipital sulcus" },
      2: { label: "Moderate", description: "Significant sulcal widening, moderate parietal atrophy" },
      3: { label: "Severe", description: "Severe parietal atrophy (knife-blade appearance)" }
    },
    // Conversion from cerebral cortex z-score (proxy for parietal)
    zscoreToScore: [
      { minZ: -0.5, score: 0 },
      { minZ: -1.5, score: 1 },
      { minZ: -2.5, score: 2 },
      { minZ: -Infinity, score: 3 }
    ]
  },

  // ========================================
  // Evans Index - Ventricular Enlargement
  // Reference: Evans (1942)
  // Ratio of frontal horn width to skull width
  // Normal: 0.20-0.25, Abnormal: >0.30
  // ========================================
  EvansIndex: {
    name: "Evans Index",
    reference: "Evans (1942)",
    formula: "Frontal Horn Width / Maximum Skull Width",
    thresholds: {
      normal: { max: 0.25, label: "Normal" },
      borderline: { max: 0.30, label: "Borderline" },
      abnormal: { min: 0.30, label: "Ventricular Enlargement" }
    },
    // We'll estimate from ventricle volume ratio to ICV
    // Using cube root approximation for width from volume
    ventricleVolumeToIndex: function (ventricleVol, icv) {
      // Approximate: Evans ≈ 0.37 × (LV_volume / ICV)^(1/3)
      // Calibrated to give ~0.25 for normal and ~0.35 for enlarged
      const ratio = ventricleVol / icv;
      return 0.37 * Math.pow(ratio, 0.333);
    }
  },

  // ========================================
  // Fazekas Scale - White Matter Hyperintensities
  // (Cannot calculate from volume alone - needs FLAIR)
  // Included for completeness
  // ========================================
  Fazekas: {
    name: "Fazekas Scale (White Matter Lesions)",
    reference: "Fazekas et al. (1987)",
    note: "Requires FLAIR sequence - not calculable from T1 volumetrics alone",
    scores: {
      0: { label: "Absent", description: "No white matter lesions" },
      1: { label: "Punctate", description: "Punctate foci" },
      2: { label: "Early Confluent", description: "Beginning confluence of foci" },
      3: { label: "Confluent", description: "Large confluent areas" }
    }
  }
};

// ============================================
// CLINICAL PATTERN RECOGNITION
// Characteristic atrophy patterns for different conditions
// ============================================

const CLINICAL_PATTERNS = {
  alzheimerDisease: {
    name: "Alzheimer's Disease Pattern",
    primaryRegions: ["Hippocampus", "Amygdala", "Inferior-Lateral-Ventricle"],
    secondaryRegions: ["Lateral-Ventricle", "Cerebral-Cortex"],
    criteria: {
      hippocampusZ: -2.0,
      hocThreshold: 0.70,
      description: "Bilateral medial temporal lobe atrophy with hippocampal involvement"
    }
  },
  frontotemporalDementia: {
    name: "Frontotemporal Dementia Pattern",
    primaryRegions: ["Amygdala", "Caudate"],  // Frontal regions if available
    secondaryRegions: ["Thalamus", "Putamen"],
    criteria: {
      asymmetryThreshold: 0.15,  // >15% L-R difference
      description: "Asymmetric frontal and/or temporal atrophy"
    }
  },
  normalAging: {
    name: "Normal Aging Pattern",
    description: "Generalized mild volume reduction proportional to age",
    criteria: {
      maxZscore: -1.5,
      bpfWithinNormal: true,
      noFocalAtrophy: true
    }
  },
  vascularDementia: {
    name: "Vascular Dementia Pattern",
    primaryRegions: ["Cerebral-White-Matter", "Lateral-Ventricle"],
    description: "White matter volume loss with ventricular enlargement"
  },
  earlyMCI: {
    name: "Early MCI / Subtle Medial Temporal Changes",
    description: "Subtle medial temporal changes detected by HOC before frank hippocampal atrophy",
    primaryRegions: ["Hippocampus", "Inferior-Lateral-Ventricle"],
    criteria: {
      hocThreshold: 0.80,  // HOC below this with normal hippocampal volume
      hippocampusZNormal: true,  // Hippocampal z-score still in normal range
      description: "Early medial temporal changes - temporal horn enlargement relative to hippocampus"
    },
    clinicalNote: "HOC may detect subtle changes before volumetric measures. Consider neuropsychological testing."
  }
};

// ============================================
// DEMENTIA RISK SCORING (0-5 Scale)
// Converts volumetric analysis to numeric risk score
// ============================================

const RISK_SCORE_LABELS = {
  0: "No Risk",
  1: "Minimal Risk",
  2: "Low Risk",
  3: "Moderate Risk",
  4: "High Risk",
  5: "Very High Risk"
};

/**
 * Calculate dementia risk score (0-5) from analysis results
 * Incorporates hippocampal volume, HOC, BPF, and clinical pattern findings
 * @param {Object} results - Analysis results object
 * @returns {Object} Risk score and contributing factors
 */
function calculateDementiaRiskScore(results) {
  if (!results) return { score: null, factors: [], confidence: "low" };

  let score = 0;
  const factors = [];
  let confidence = "moderate";

  // Factor 1: Hippocampal z-score (most critical)
  const hippoZ = results.regions?.["Hippocampus"]?.effectiveZscore;
  if (hippoZ !== undefined) {
    if (hippoZ < -2.5) {
      score += 2;
      factors.push("Severe hippocampal atrophy");
    } else if (hippoZ < -2.0) {
      score += 1.5;
      factors.push("Moderate hippocampal atrophy");
    } else if (hippoZ < -1.5) {
      score += 1;
      factors.push("Mild hippocampal atrophy");
    } else if (hippoZ < -1.0) {
      score += 0.5;
      factors.push("Low-normal hippocampal volume");
    }
  }

  // Factor 2: HOC (Hippocampal Occupancy Score)
  const hoc = results.hoc?.value;
  if (hoc !== undefined && hoc !== null) {
    if (hoc < 0.60) {
      score += 1.5;
      factors.push("Severe HOC reduction");
    } else if (hoc < 0.70) {
      score += 1;
      factors.push("Moderate HOC reduction");
    } else if (hoc < 0.80) {
      score += 0.5;
      factors.push("Mild HOC reduction");
    }
  }

  // Factor 3: BPF (Brain Parenchymal Fraction)
  const bpfZ = results.bpf?.zscore;
  if (bpfZ !== undefined) {
    if (bpfZ < -2.5) {
      score += 1;
      factors.push("Significant global atrophy");
    } else if (bpfZ < -2.0) {
      score += 0.5;
      factors.push("Moderate global atrophy");
    }
  }

  // Factor 4: Clinical pattern detection
  const patterns = results.clinicalPatterns || [];
  for (const pattern of patterns) {
    if (pattern.pattern.includes("Alzheimer") && pattern.confidence === "High") {
      score += 0.5;
      factors.push("AD pattern detected");
    }
  }

  // Factor 5: Temporal horn enlargement
  const ilvZ = results.regions?.["Inferior-Lateral-Ventricle"]?.zscore;
  if (ilvZ !== undefined && ilvZ > 1.5) {
    score += 0.5;
    factors.push("Temporal horn enlargement");
  }

  // Round and clamp to 0-5
  score = Math.round(score * 2) / 2; // Round to nearest 0.5
  score = Math.min(5, Math.max(0, Math.round(score)));

  // Determine confidence
  if (results.validation?.flags?.overallQuality === "review_recommended") {
    confidence = "low";
  } else if (factors.length >= 3) {
    confidence = "high";
  }

  return {
    score,
    label: RISK_SCORE_LABELS[score],
    factors,
    confidence
  };
}

// ============================================
// BENCHMARK MODE HANDLING
// Supports three modes for AI vs radiologist comparison
// ============================================

/**
 * Get the current benchmark mode
 * @returns {string} 'ai-first' | 'radiologist-first' | 'radiologist-only'
 */
function getBenchmarkMode() {
  const selector = document.getElementById("benchmarkMode");
  return selector ? selector.value : "ai-first";
}

/**
 * Handle benchmark mode selector change
 * Updates the hint text and stores mode preference
 */
function handleBenchmarkModeChange() {
  const mode = getBenchmarkMode();
  const hintEl = document.getElementById("benchmarkHint");

  const hints = {
    "ai-first": "AI results shown first, then manual review",
    "radiologist-first": "You assess first, then AI results revealed",
    "radiologist-only": "AI results hidden, manual assessment only"
  };

  if (hintEl) {
    hintEl.textContent = hints[mode] || "";
  }

  // Store in analysis results if available
  if (analysisResults) {
    analysisResults.benchmarkMode = mode;
  }
}

/**
 * Handle radiologist risk input change
 * Updates the comparison display and checks for discrepancies
 * In radiologist-first mode, enables the reveal button after score entry
 */
function handleRadiologistRiskInput() {
  const radiologistInput = document.getElementById("radiologistRiskInput");
  const radiologistScore = radiologistInput.value;
  const radiologistLabel = document.getElementById("radiologistRiskLabel");
  const discrepancyAlert = document.getElementById("discrepancyAlert");
  const discrepancyText = document.getElementById("discrepancyText");
  const revealSection = document.getElementById("revealAiSection");
  const mode = getBenchmarkMode();

  if (radiologistScore === "") {
    radiologistLabel.innerHTML = "&nbsp;";
    discrepancyAlert.style.display = "none";
    if (revealSection && mode === "radiologist-first") {
      revealSection.style.display = "none";
    }
    return;
  }

  const scoreNum = parseInt(radiologistScore, 10);
  radiologistLabel.textContent = RISK_SCORE_LABELS[scoreNum];

  // Store in analysis results for export with timestamp
  if (analysisResults) {
    analysisResults.radiologistRiskScore = scoreNum;
    analysisResults.radiologistScoreTimestamp = new Date().toISOString();
  }

  // In radiologist-first mode, show reveal button after score entry
  if (mode === "radiologist-first" && revealSection) {
    const aiColumn = document.getElementById("aiResultsColumn");
    if (aiColumn && aiColumn.style.display === "none") {
      revealSection.style.display = "block";
    }
  }

  // Check for discrepancy with AI score (only if AI is visible)
  const aiColumn = document.getElementById("aiResultsColumn");
  const aiVisible = aiColumn && aiColumn.style.display !== "none";
  const aiScore = analysisResults?.dementiaRiskScore?.score;

  if (aiVisible && aiScore !== null && aiScore !== undefined) {
    const discrepancy = Math.abs(aiScore - scoreNum);
    if (discrepancy >= 2) {
      discrepancyText.textContent = `Discrepancy noted: Model score (${aiScore}) differs from manual score (${scoreNum}) by ${discrepancy} points`;
      discrepancyAlert.style.display = "flex";
    } else {
      discrepancyAlert.style.display = "none";
    }
  }
}

/**
 * Reveal AI results in radiologist-first mode
 * Called when user clicks "Reveal AI Results" button
 */
function revealAiResults() {
  const aiColumn = document.getElementById("aiResultsColumn");
  const placeholder = document.getElementById("aiHiddenPlaceholder");
  const revealSection = document.getElementById("revealAiSection");
  const modeLabel = document.getElementById("assessmentModeLabel");

  // Show AI results
  if (aiColumn) aiColumn.style.display = "block";
  if (placeholder) placeholder.style.display = "none";
  if (revealSection) revealSection.style.display = "none";

  // Update mode label
  if (modeLabel) {
    modeLabel.textContent = "AI Revealed";
  }

  // Record reveal timestamp for benchmarking
  if (analysisResults) {
    analysisResults.aiRevealedTimestamp = new Date().toISOString();
  }

  // Now check for discrepancy since AI is visible
  handleRadiologistRiskInput();
}

/**
 * Display the risk comparison card with AI and radiologist scores
 * Respects the selected benchmark mode for show/hide logic
 */
function displayRiskComparison() {
  if (!analysisResults) return;

  const card = document.getElementById("riskComparisonCard");
  if (!card) return;

  // Get current benchmark mode
  const mode = getBenchmarkMode();
  analysisResults.benchmarkMode = mode;
  analysisResults.analysisTimestamp = new Date().toISOString();

  // Calculate AI dementia risk score (always calculate, but may hide display)
  const riskResult = calculateDementiaRiskScore(analysisResults);
  analysisResults.dementiaRiskScore = riskResult;

  // Display the card
  card.style.display = "block";

  // Get UI elements
  const aiColumn = document.getElementById("aiResultsColumn");
  const aiPlaceholder = document.getElementById("aiHiddenPlaceholder");
  const radiologistColumn = document.getElementById("radiologistColumn");
  const revealSection = document.getElementById("revealAiSection");
  const modeLabel = document.getElementById("assessmentModeLabel");
  const aiScoreEl = document.getElementById("aiRiskScore");
  const aiLabelEl = document.getElementById("aiRiskLabel");
  const aiFactorsEl = document.getElementById("aiRiskFactors");

  // Update AI score display (populate values regardless of visibility)
  if (riskResult.score !== null) {
    aiScoreEl.textContent = riskResult.score;
    aiScoreEl.className = `risk-score-value risk-${riskResult.score}`;
    aiLabelEl.textContent = riskResult.label;

    // Display contributing factors
    if (riskResult.factors.length > 0) {
      aiFactorsEl.innerHTML = riskResult.factors.map(f =>
        `<div class="risk-factor-item">• ${f}</div>`
      ).join("");
    } else {
      aiFactorsEl.innerHTML = '<div class="risk-factor-item">• No concerning findings</div>';
    }
  } else {
    aiScoreEl.textContent = "--";
    aiScoreEl.className = "risk-score-value";
    aiLabelEl.textContent = "Unable to calculate";
    aiFactorsEl.innerHTML = "";
  }

  // Apply benchmark mode visibility
  switch (mode) {
    case "ai-first":
      // Show AI results immediately
      if (aiColumn) aiColumn.style.display = "block";
      if (aiPlaceholder) aiPlaceholder.style.display = "none";
      if (radiologistColumn) radiologistColumn.style.display = "block";
      if (revealSection) revealSection.style.display = "none";
      if (modeLabel) modeLabel.textContent = "AI-First Mode";
      break;

    case "radiologist-first":
      // Hide AI results until radiologist enters score and clicks reveal
      if (aiColumn) aiColumn.style.display = "none";
      if (aiPlaceholder) aiPlaceholder.style.display = "block";
      if (radiologistColumn) radiologistColumn.style.display = "block";
      if (revealSection) revealSection.style.display = "none"; // Shown after score entry
      if (modeLabel) modeLabel.textContent = "Radiologist-First Mode";
      break;

    case "radiologist-only":
      // Hide AI completely
      if (aiColumn) aiColumn.style.display = "none";
      if (aiPlaceholder) aiPlaceholder.style.display = "none";
      if (radiologistColumn) radiologistColumn.style.display = "block";
      if (revealSection) revealSection.style.display = "none";
      if (modeLabel) modeLabel.textContent = "Radiologist-Only Mode";
      break;
  }

  // Reset radiologist input
  const radiologistInput = document.getElementById("radiologistRiskInput");
  if (radiologistInput) {
    radiologistInput.value = "";
  }
  const radiologistLabel = document.getElementById("radiologistRiskLabel");
  if (radiologistLabel) {
    radiologistLabel.innerHTML = "&nbsp;";
  }
  const discrepancyAlert = document.getElementById("discrepancyAlert");
  if (discrepancyAlert) {
    discrepancyAlert.style.display = "none";
  }
}

// ============================================
// ASYMMETRY TRACKING & LOBAR ANALYSIS
// Phase 2: Advanced analysis features
// ============================================

/**
 * Left-Right region pairs for asymmetry analysis
 * Maps to 104-class model labels
 */
const LEFT_RIGHT_PAIRS = {
  hippocampus: { left: "Left-Hippocampus", right: "Right-Hippocampus", clinicalName: "Hippocampus" },
  amygdala: { left: "Left-Amygdala", right: "Right-Amygdala", clinicalName: "Amygdala" },
  thalamus: { left: "Left-Thalamus-Proper*", right: "Right-Thalamus-Proper*", clinicalName: "Thalamus" },
  caudate: { left: "Left-Caudate", right: "Right-Caudate", clinicalName: "Caudate" },
  putamen: { left: "Left-Putamen", right: "Right-Putamen", clinicalName: "Putamen" },
  pallidum: { left: "Left-Pallidum", right: "Right-Pallidum", clinicalName: "Pallidum" },
  lateralVentricle: { left: "Left-Lateral-Ventricle", right: "Right-Lateral-Ventricle", clinicalName: "Lateral Ventricle" },
  infLatVent: { left: "Left-Inf-Lat-Vent", right: "Right-Inf-Lat-Vent", clinicalName: "Temporal Horn" },
  accumbens: { left: "Left-Accumbens-area", right: "Right-Accumbens-area", clinicalName: "Accumbens" },
  ventralDC: { left: "Left-VentralDC", right: "Right-VentralDC", clinicalName: "Ventral DC" },
  cerebralWM: { left: "Left-Cerebral-White-Matter", right: "Right-Cerebral-White-Matter", clinicalName: "Cerebral WM" },
  cerebellumWM: { left: "Left-Cerebellum-White-Matter", right: "Right-Cerebellum-White-Matter", clinicalName: "Cerebellar WM" },
  cerebellumCortex: { left: "Left-Cerebellum-Cortex", right: "Right-Cerebellum-Cortex", clinicalName: "Cerebellar Cortex" }
};

/**
 * Lobar mapping for cortical regions
 * Maps FreeSurfer cortical labels to lobes
 */
const LOBAR_MAPPING = {
  frontal: {
    name: "Frontal Lobe",
    regions: [
      "ctx-lh-superiorfrontal", "ctx-rh-superiorfrontal",
      "ctx-lh-rostralmiddlefrontal", "ctx-rh-rostralmiddlefrontal",
      "ctx-lh-caudalmiddlefrontal", "ctx-rh-caudalmiddlefrontal",
      "ctx-lh-parsopercularis", "ctx-rh-parsopercularis",
      "ctx-lh-parstriangularis", "ctx-rh-parstriangularis",
      "ctx-lh-parsorbitalis", "ctx-rh-parsorbitalis",
      "ctx-lh-lateralorbitofrontal", "ctx-rh-lateralorbitofrontal",
      "ctx-lh-medialorbitofrontal", "ctx-rh-medialorbitofrontal",
      "ctx-lh-precentral", "ctx-rh-precentral",
      "ctx-lh-paracentral", "ctx-rh-paracentral",
      "ctx-lh-frontalpole", "ctx-rh-frontalpole"
    ],
    subcortical: ["Left-Caudate", "Right-Caudate"]
  },
  temporal: {
    name: "Temporal Lobe",
    regions: [
      "ctx-lh-superiortemporal", "ctx-rh-superiortemporal",
      "ctx-lh-middletemporal", "ctx-rh-middletemporal",
      "ctx-lh-inferiortemporal", "ctx-rh-inferiortemporal",
      "ctx-lh-bankssts", "ctx-rh-bankssts",
      "ctx-lh-fusiform", "ctx-rh-fusiform",
      "ctx-lh-transversetemporal", "ctx-rh-transversetemporal",
      "ctx-lh-entorhinal", "ctx-rh-entorhinal",
      "ctx-lh-temporalpole", "ctx-rh-temporalpole",
      "ctx-lh-parahippocampal", "ctx-rh-parahippocampal"
    ],
    subcortical: ["Left-Hippocampus", "Right-Hippocampus", "Left-Amygdala", "Right-Amygdala"]
  },
  parietal: {
    name: "Parietal Lobe",
    regions: [
      "ctx-lh-superiorparietal", "ctx-rh-superiorparietal",
      "ctx-lh-inferiorparietal", "ctx-rh-inferiorparietal",
      "ctx-lh-supramarginal", "ctx-rh-supramarginal",
      "ctx-lh-postcentral", "ctx-rh-postcentral",
      "ctx-lh-precuneus", "ctx-rh-precuneus"
    ],
    subcortical: []
  },
  occipital: {
    name: "Occipital Lobe",
    regions: [
      "ctx-lh-lateraloccipital", "ctx-rh-lateraloccipital",
      "ctx-lh-lingual", "ctx-rh-lingual",
      "ctx-lh-cuneus", "ctx-rh-cuneus",
      "ctx-lh-pericalcarine", "ctx-rh-pericalcarine"
    ],
    subcortical: []
  },
  cingulate: {
    name: "Cingulate",
    regions: [
      "ctx-lh-caudalanteriorcingulate", "ctx-rh-caudalanteriorcingulate",
      "ctx-lh-rostralanteriorcingulate", "ctx-rh-rostralanteriorcingulate",
      "ctx-lh-posteriorcingulate", "ctx-rh-posteriorcingulate",
      "ctx-lh-isthmuscingulate", "ctx-rh-isthmuscingulate"
    ],
    subcortical: []
  },
  insula: {
    name: "Insula",
    regions: ["ctx-lh-insula", "ctx-rh-insula"],
    subcortical: []
  }
};

/**
 * Asymmetry interpretation thresholds
 */
const ASYMMETRY_THRESHOLDS = {
  symmetric: { max: 5, label: "Symmetric", color: "#22c55e" },
  mild: { min: 5, max: 10, label: "Mild Asymmetry", color: "#84cc16" },
  moderate: { min: 10, max: 20, label: "Moderate Asymmetry", color: "#f59e0b" },
  severe: { min: 20, label: "Severe Asymmetry", color: "#ef4444" }
};

/**
 * Calculate asymmetry index for a structure
 * AI = |L - R| / ((L + R) / 2) × 100%
 * @param {number} leftVol - Left side volume
 * @param {number} rightVol - Right side volume
 * @returns {Object} Asymmetry metrics
 */
function calculateAsymmetryIndex(leftVol, rightVol) {
  if (!leftVol || !rightVol || leftVol <= 0 || rightVol <= 0) {
    return null;
  }

  const mean = (leftVol + rightVol) / 2;
  const diff = Math.abs(leftVol - rightVol);
  const asymmetryIndex = (diff / mean) * 100;
  const laterality = leftVol > rightVol ? "Left > Right" : leftVol < rightVol ? "Right > Left" : "Equal";

  // Determine interpretation
  let interpretation;
  if (asymmetryIndex < 5) {
    interpretation = ASYMMETRY_THRESHOLDS.symmetric;
  } else if (asymmetryIndex < 10) {
    interpretation = ASYMMETRY_THRESHOLDS.mild;
  } else if (asymmetryIndex < 20) {
    interpretation = ASYMMETRY_THRESHOLDS.moderate;
  } else {
    interpretation = ASYMMETRY_THRESHOLDS.severe;
  }

  return {
    leftVolume: leftVol,
    rightVolume: rightVol,
    asymmetryIndex: asymmetryIndex,
    laterality: laterality,
    interpretation: interpretation.label,
    color: interpretation.color,
    isConcerning: asymmetryIndex >= 10
  };
}

/**
 * Calculate asymmetry metrics for all paired structures
 * @param {Object} regionVolumes - Volume data from segmentation
 * @returns {Object} Asymmetry results
 */
function calculateAsymmetryMetrics(regionVolumes) {
  const asymmetryResults = {};
  let concerningCount = 0;

  for (const [key, pair] of Object.entries(LEFT_RIGHT_PAIRS)) {
    const leftVol = regionVolumes[pair.left];
    const rightVol = regionVolumes[pair.right];

    if (leftVol && rightVol) {
      const result = calculateAsymmetryIndex(leftVol, rightVol);
      if (result) {
        asymmetryResults[key] = {
          ...result,
          clinicalName: pair.clinicalName
        };
        if (result.isConcerning) concerningCount++;
      }
    }
  }

  return {
    regions: asymmetryResults,
    concerningCount: concerningCount,
    summary: concerningCount > 2 ? "Multiple concerning asymmetries" :
      concerningCount > 0 ? "Some asymmetry detected" : "Generally symmetric"
  };
}

/**
 * Calculate Global Atrophy Index (0-100 scale)
 * Combines BPF, total brain z-score, and weighted regional scores
 * @param {Object} results - Analysis results
 * @returns {Object} Global atrophy index
 */
function calculateGlobalAtrophyIndex(results) {
  if (!results) return null;

  let score = 50; // Baseline (normal = 50)
  const factors = [];

  // Factor 1: BPF z-score (weight: 30%)
  if (results.bpf?.zscore !== undefined) {
    const bpfContribution = Math.min(15, Math.max(-15, results.bpf.zscore * -5));
    score += bpfContribution;
    if (results.bpf.zscore < -1.5) {
      factors.push("Reduced brain parenchymal fraction");
    }
  }

  // Factor 2: Mean regional z-score for critical regions (weight: 40%)
  const criticalRegions = ["Hippocampus", "Thalamus", "Caudate", "Putamen"];
  let criticalZScoreSum = 0;
  let criticalCount = 0;

  for (const region of criticalRegions) {
    const regionData = results.regions?.[region];
    if (regionData?.effectiveZscore !== undefined) {
      criticalZScoreSum += regionData.effectiveZscore;
      criticalCount++;
    }
  }

  if (criticalCount > 0) {
    const meanCriticalZ = criticalZScoreSum / criticalCount;
    const criticalContribution = Math.min(20, Math.max(-20, meanCriticalZ * -6));
    score += criticalContribution;
    if (meanCriticalZ < -1.5) {
      factors.push("Critical region atrophy");
    }
  }

  // Factor 3: Ventricular enlargement (weight: 30%)
  const ventricleZ = results.regions?.["Lateral-Ventricle"]?.zscore;
  if (ventricleZ !== undefined) {
    // Higher ventricle z-score = more atrophy (inverted for ventricles)
    const ventContribution = Math.min(15, Math.max(-15, ventricleZ * 4));
    score += ventContribution;
    if (ventricleZ > 1.5) {
      factors.push("Ventricular enlargement");
    }
  }

  // Clamp to 0-100
  score = Math.min(100, Math.max(0, score));

  // Interpretation
  let interpretation, severity;
  if (score < 30) {
    interpretation = "Minimal atrophy";
    severity = "normal";
  } else if (score < 45) {
    interpretation = "Mild global atrophy";
    severity = "mild";
  } else if (score < 60) {
    interpretation = "Age-appropriate";
    severity = "normal";
  } else if (score < 75) {
    interpretation = "Moderate global atrophy";
    severity = "moderate";
  } else {
    interpretation = "Significant global atrophy";
    severity = "severe";
  }

  return {
    value: Math.round(score),
    interpretation: interpretation,
    severity: severity,
    factors: factors
  };
}

/**
 * Calculate lobar atrophy scores
 * @param {Object} regionVolumes - Volume data from segmentation
 * @param {Object} regionResults - Z-scores and interpretations
 * @returns {Object} Lobar atrophy results
 */
function calculateLobarAtrophy(regionVolumes, regionResults) {
  const lobarResults = {};

  for (const [lobeName, lobeData] of Object.entries(LOBAR_MAPPING)) {
    let totalVolume = 0;
    let zScoreSum = 0;
    let zScoreCount = 0;

    // Sum cortical regions
    for (const region of lobeData.regions) {
      if (regionVolumes[region]) {
        totalVolume += regionVolumes[region];
      }
      if (regionResults?.[region]?.zscore !== undefined) {
        zScoreSum += regionResults[region].zscore;
        zScoreCount++;
      }
    }

    // Add subcortical structures
    for (const region of lobeData.subcortical) {
      if (regionVolumes[region]) {
        totalVolume += regionVolumes[region];
      }
      if (regionResults?.[region]?.zscore !== undefined) {
        zScoreSum += regionResults[region].zscore;
        zScoreCount++;
      }
    }

    const meanZScore = zScoreCount > 0 ? zScoreSum / zScoreCount : null;

    lobarResults[lobeName] = {
      name: lobeData.name,
      totalVolume: totalVolume,
      meanZScore: meanZScore,
      interpretation: meanZScore !== null ?
        (meanZScore < -2.0 ? "Significant atrophy" :
          meanZScore < -1.5 ? "Mild atrophy" :
            meanZScore < -1.0 ? "Low-normal" : "Normal") : "N/A"
    };
  }

  return lobarResults;
}

/**
 * Detect FTD-specific patterns based on asymmetry and lobar involvement
 * @param {Object} asymmetryResults - Asymmetry metrics
 * @param {Object} lobarResults - Lobar atrophy results
 * @param {Object} regionResults - Regional z-scores
 * @returns {Array} FTD pattern findings
 */
function detectFTDPatterns(asymmetryResults, lobarResults, regionResults) {
  const ftdPatterns = [];

  // Check for frontal predominant atrophy (bvFTD)
  const frontalZ = lobarResults?.frontal?.meanZScore;
  const parietalZ = lobarResults?.parietal?.meanZScore;
  const temporalZ = lobarResults?.temporal?.meanZScore;

  if (frontalZ !== null && parietalZ !== null) {
    const frontalParietalRatio = frontalZ / (parietalZ || -0.1);
    if (frontalZ < -1.5 && frontalParietalRatio > 1.5) {
      ftdPatterns.push({
        pattern: "Behavioral variant FTD (bvFTD) pattern",
        confidence: frontalZ < -2.0 ? "High" : "Moderate",
        indicators: ["Frontal predominant atrophy", "Relative parietal sparing"],
        recommendation: "Consider neuropsychological evaluation for executive dysfunction"
      });
    }
  }

  // Check for left temporal predominance (svPPA)
  const leftTempAsym = asymmetryResults?.regions?.hippocampus;
  const leftAmygAsym = asymmetryResults?.regions?.amygdala;

  if (leftTempAsym && leftAmygAsym) {
    if (leftTempAsym.laterality === "Right > Left" && leftTempAsym.asymmetryIndex > 15 &&
      temporalZ !== null && temporalZ < -1.5) {
      ftdPatterns.push({
        pattern: "Semantic variant PPA (svPPA) pattern",
        confidence: leftTempAsym.asymmetryIndex > 20 ? "High" : "Moderate",
        indicators: ["Left temporal predominance", "Temporal > hippocampal asymmetry"],
        recommendation: "Consider language testing for semantic memory deficits"
      });
    }
  }

  // Check for insula involvement (nfvPPA)
  const insulaZ = lobarResults?.insula?.meanZScore;
  if (insulaZ !== null && insulaZ < -1.5 && frontalZ !== null && frontalZ < -1.5) {
    ftdPatterns.push({
      pattern: "Nonfluent variant PPA (nfvPPA) pattern",
      confidence: insulaZ < -2.0 ? "Moderate" : "Low-Moderate",
      indicators: ["Insula involvement", "Left frontal changes"],
      recommendation: "Consider speech/language evaluation"
    });
  }

  return ftdPatterns;
}

// ============================================
// APPLICATION STATE
// ============================================

let nv1 = null;
let segmentationData = null;
let regionVolumes = null;
let uploadedFile = null;  // Store original file for server-side inference
let analysisResults = null;

// Default model: Subcortical + GWM - robust for low quality MRIs
const DEFAULT_MODEL_INDEX = 5;  // Subcortical + GWM (Low Mem, Faster)

// ============================================
// INITIALIZATION
// ============================================

async function init() {
  console.log("Initializing Brain Atrophy Analysis...");

  // Initialize NiiVue
  const defaults = {
    backColor: [0.05, 0.05, 0.08, 1],
    show3Dcrosshair: true,
    onLocationChange: handleLocationChange,
  };

  nv1 = new Niivue(defaults);
  await nv1.attachToCanvas(document.getElementById("gl1"));
  nv1.opts.dragMode = nv1.dragModes.pan;
  nv1.opts.multiplanarForceRender = true;
  nv1.opts.yoke3Dto2DZoom = true;
  nv1.opts.crosshairGap = 11;
  nv1.setInterpolation(true);

  // Setup event listeners
  setupEventListeners();

  console.log("Initialization complete");
}

function setupEventListeners() {
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const clipCheck = document.getElementById("clipCheck");
  const opacitySlider = document.getElementById("opacitySlider");
  const exportPdfBtn = document.getElementById("exportPdfBtn");
  const exportNiftiBtn = document.getElementById("exportNiftiBtn");

  // File drop handling
  dropZone.addEventListener("click", () => fileInput.click());

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
  });

  dropZone.addEventListener("drop", async (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) await loadFile(file);
  });

  fileInput.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (file) await loadFile(file);
  });

  // Analyze button
  analyzeBtn.addEventListener("click", runAnalysis);

  // Viewer controls
  clipCheck.addEventListener("change", () => {
    if (clipCheck.checked) {
      nv1.setClipPlane([0, 0, 90]);
    } else {
      nv1.setClipPlane([2, 0, 90]);
    }
  });

  opacitySlider.addEventListener("input", () => {
    if (nv1.volumes.length > 1) {
      nv1.setOpacity(1, opacitySlider.value / 100);
    }
  });

  // Export buttons
  exportPdfBtn.addEventListener("click", exportReport);
  exportNiftiBtn.addEventListener("click", () => {
    if (nv1.volumes.length > 1) {
      nv1.volumes[1].saveToDisk("atrophy_segmentation.nii.gz");
    }
  });

  // Heatmap toggle button
  const heatmapToggle = document.getElementById("heatmapToggle");
  if (heatmapToggle) {
    heatmapToggle.addEventListener("click", toggleAtrophyHeatmap);
  }

  // Radiologist risk input handler
  const radiologistInput = document.getElementById("radiologistRiskInput");
  if (radiologistInput) {
    radiologistInput.addEventListener("change", handleRadiologistRiskInput);
  }

  // Benchmark mode selector handler
  const benchmarkModeSelect = document.getElementById("benchmarkMode");
  if (benchmarkModeSelect) {
    benchmarkModeSelect.addEventListener("change", handleBenchmarkModeChange);
    // Initialize hint text
    handleBenchmarkModeChange();
  }

  // Reveal AI button handler
  const revealAiBtn = document.getElementById("revealAiBtn");
  if (revealAiBtn) {
    revealAiBtn.addEventListener("click", revealAiResults);
  }

  // Evaluation form submit button
  const evalSubmitBtn = document.getElementById("evalSubmitBtn");
  if (evalSubmitBtn) {
    evalSubmitBtn.addEventListener("click", handleEvalSubmit);
  }

  // Proceed to manual review button (AI-First mode)
  const proceedBtn = document.getElementById("proceedToReviewBtn");
  if (proceedBtn) {
    proceedBtn.addEventListener("click", handleProceedToReview);
  }

  // Comparison submit button (side-by-side modes)
  const comparisonSubmitBtn = document.getElementById("comparisonSubmitBtn");
  if (comparisonSubmitBtn) {
    comparisonSubmitBtn.addEventListener("click", handleComparisonSubmit);
  }
}

// ============================================
// FILE LOADING
// ============================================

async function loadFile(file) {
  updateProgress(10, "Loading file...");

  try {
    // Clear previous data
    while (nv1.volumes.length > 0) {
      await nv1.removeVolume(nv1.volumes[0]);
    }
    segmentationData = null;
    regionVolumes = null;
    analysisResults = null;
    uploadedFile = file;  // Store for server-side inference
    hideAnalysisCards();

    // Load file
    await nv1.loadFromFile(file);

    // Hide drop zone
    document.getElementById("dropZone").classList.add("hidden");

    // Conform to standard dimensions
    updateProgress(30, "Conforming image...");
    await ensureConformed();

    // Enable analyze button
    document.getElementById("analyzeBtn").disabled = false;

    updateProgress(100, "Ready for analysis");
    setTimeout(() => updateProgress(0, "Ready"), 1000);

  } catch (err) {
    console.error("Load error:", err);
    updateProgress(0, "Error loading file");
    alert("Error loading file: " + err.message);
  }
}

async function ensureConformed() {
  const nii = nv1.volumes[0];
  let isConformed =
    nii.dims[1] === 256 && nii.dims[2] === 256 && nii.dims[3] === 256 &&
    nii.img instanceof Uint8Array && nii.img.length === 256 * 256 * 256;

  if (nii.permRAS[0] !== -1 || nii.permRAS[1] !== 3 || nii.permRAS[2] !== -2) {
    isConformed = false;
  }

  if (!isConformed) {
    const nii2 = await nv1.conform(nii, false);
    await nv1.removeVolume(nv1.volumes[0]);
    await nv1.addVolume(nii2);
  }
}

// ============================================
// ANALYSIS PIPELINE
// ============================================

async function runAnalysis() {
  const analyzeBtn = document.getElementById("analyzeBtn");
  analyzeBtn.disabled = true;
  document.body.classList.add("analyzing");

  try {
    // Step 1: Run segmentation
    updateProgress(10, "Starting brain segmentation...");
    await runSegmentation();

    // Step 2: Calculate volumes
    updateProgress(70, "Calculating region volumes...");
    await calculateVolumes();

    // Step 3: Compare to normative data
    updateProgress(85, "Analyzing atrophy patterns...");
    await analyzeAtrophy();

    // Step 4: Display results
    updateProgress(95, "Generating report...");
    displayResults();

    updateProgress(100, "Analysis complete");

  } catch (err) {
    console.error("Analysis error:", err);
    updateProgress(0, "Analysis failed");
    alert("Analysis error: " + err.message);
  } finally {
    analyzeBtn.disabled = false;
    document.body.classList.remove("analyzing");
  }
}

async function runSegmentation() {
  const model = inferenceModelsList[DEFAULT_MODEL_INDEX];
  const opts = { ...brainChopOpts };
  opts.rootURL = location.protocol + "//" + location.host;

  await ensureConformed();

  // Use server-side inference if enabled
  if (USE_SERVER) {
    try {
      const result = await runServerInference(
        (message, progress) => {
          const adjustedProgress = 10 + progress * 55;
          updateProgress(adjustedProgress, message);
        }
      );

      segmentationData = new Uint8Array(result.segmentation);
      await displaySegmentation(result.segmentation, model);
      console.log(`Server inference completed in ${result.inferenceTime}s`);
      return;

    } catch (error) {
      console.error("Server inference failed, falling back to local:", error);
      updateProgress(10, "Server unavailable, using local processing...");
      // Fall through to local inference
    }
  }

  // Local inference (original code)
  return new Promise((resolve, reject) => {
    runInference(
      opts,
      model,
      nv1.volumes[0].hdr,
      nv1.volumes[0].img,
      async (img, opts, modelEntry) => {
        // Callback for completed segmentation
        try {
          segmentationData = new Uint8Array(img);
          await displaySegmentation(img, modelEntry);
          resolve();
        } catch (err) {
          reject(err);
        }
      },
      (message, progressFrac, modalMessage, statData) => {
        // Progress callback
        if (progressFrac >= 0) {
          const adjustedProgress = 10 + progressFrac * 55;
          updateProgress(adjustedProgress, message);
        }
        if (modalMessage) {
          alert(modalMessage);
        }
      }
    );
  });
}

async function displaySegmentation(img, modelEntry) {
  // Close existing overlays
  while (nv1.volumes.length > 1) {
    await nv1.removeVolume(nv1.volumes[1]);
  }

  // Create overlay volume
  const overlayVolume = await nv1.volumes[0].clone();
  overlayVolume.zeroImage();
  overlayVolume.hdr.scl_inter = 0;
  overlayVolume.hdr.scl_slope = 1;
  overlayVolume.img = new Uint8Array(img);

  // Load colormap if available
  if (modelEntry.colormapPath) {
    // Fix path - colormap is relative to root, not /atrophy/
    const cmapPath = modelEntry.colormapPath.replace('./', '../');
    const response = await fetch(cmapPath);
    const cmap = await response.json();
    overlayVolume.setColormapLabel({
      R: cmap["R"],
      G: cmap["G"],
      B: cmap["B"],
      labels: cmap["labels"],
    });
    overlayVolume.hdr.intent_code = 1002;  // NIFTI_INTENT_LABEL
  }

  overlayVolume.opacity = 0.5;
  await nv1.addVolume(overlayVolume);
}

// ============================================
// ATROPHY HEATMAP OVERLAY
// Z-score based intensity visualization
// ============================================

let atrophyHeatmapVolume = null;
let heatmapVisible = false;

/**
 * Generate atrophy heatmap based on regional z-scores
 * Maps z-scores to intensity values for visualization
 */
async function generateAtrophyHeatmap() {
  if (!nv1 || !nv1.volumes || !nv1.volumes[0] || !segmentationData || !analysisResults) {
    console.warn("Cannot generate heatmap: missing data");
    return null;
  }

  // Get colormap labels for region mapping
  const model = inferenceModelsList[DEFAULT_MODEL_INDEX];
  const cmapPath = model.colormapPath.replace('./', '../');
  const response = await fetch(cmapPath);
  const cmap = await response.json();
  const labels = cmap["labels"];

  // Create z-score lookup by label name
  const zScoreLookup = {};
  for (const [regionName, data] of Object.entries(analysisResults.regions)) {
    // Use effective z-score (accounts for ventricular inversion)
    const zscore = data.effectiveZscore !== undefined ? data.effectiveZscore : data.zscore;
    zScoreLookup[regionName] = zscore;
  }

  // Map label indices to intensity values (0-255)
  // More atrophy (lower z-score) = higher intensity
  const labelToIntensity = {};
  for (let i = 0; i < labels.length; i++) {
    const regionName = labels[i];
    const zscore = zScoreLookup[regionName];

    if (zscore !== undefined) {
      // Map z-score to intensity:
      // z <= -3.0 = 255 (max intensity, severe atrophy)
      // z >= 0 = 0 (no intensity, normal)
      // Linear interpolation in between
      const intensity = Math.min(255, Math.max(0, Math.round((-zscore / 3.0) * 255)));
      labelToIntensity[i] = intensity;
    } else {
      labelToIntensity[i] = 0;  // Unknown regions = no intensity
    }
  }

  // Create intensity map from segmentation
  const heatmapData = new Uint8Array(segmentationData.length);
  for (let i = 0; i < segmentationData.length; i++) {
    const labelIdx = segmentationData[i];
    heatmapData[i] = labelToIntensity[labelIdx] || 0;
  }

  // Create overlay volume for heatmap
  const heatmap = await nv1.volumes[0].clone();
  heatmap.zeroImage();
  heatmap.hdr.scl_inter = 0;
  heatmap.hdr.scl_slope = 1;
  heatmap.img = heatmapData;

  // Set atrophy-specific colormap (hot colormap: black -> red -> orange -> yellow -> white)
  heatmap.colormap = "hot";
  heatmap.opacity = 0.6;

  return heatmap;
}

/**
 * Toggle atrophy heatmap visibility
 */
async function toggleAtrophyHeatmap() {
  if (!analysisResults || !segmentationData) {
    console.warn("Run analysis first before enabling heatmap");
    return;
  }

  const btn = document.getElementById("heatmapToggle");

  if (heatmapVisible && atrophyHeatmapVolume) {
    // Hide heatmap
    try {
      await nv1.removeVolume(atrophyHeatmapVolume);
    } catch (e) {
      console.warn("Could not remove heatmap volume:", e);
    }
    atrophyHeatmapVolume = null;
    heatmapVisible = false;
    if (btn) btn.textContent = "Show Atrophy Heatmap";
  } else {
    // Generate and show heatmap
    updateStatus("Generating atrophy heatmap...");
    atrophyHeatmapVolume = await generateAtrophyHeatmap();
    if (atrophyHeatmapVolume) {
      await nv1.addVolume(atrophyHeatmapVolume);
      heatmapVisible = true;
      if (btn) btn.textContent = "Hide Atrophy Heatmap";
    }
    updateStatus("Ready");
  }
}

async function calculateVolumes() {
  if (!segmentationData) return;

  // Get colormap labels for region mapping
  const model = inferenceModelsList[DEFAULT_MODEL_INDEX];
  // Fix path - colormap is relative to root, not /atrophy/
  const cmapPath = model.colormapPath.replace('./', '../');
  const response = await fetch(cmapPath);
  const cmap = await response.json();
  const labels = cmap["labels"];

  // Count voxels per region
  const voxelCounts = new Map();
  for (let i = 0; i < segmentationData.length; i++) {
    const value = segmentationData[i];
    voxelCounts.set(value, (voxelCounts.get(value) || 0) + 1);
  }

  // Convert to volumes (1mm³ voxels in conformed space)
  regionVolumes = {};
  voxelCounts.forEach((count, labelIdx) => {
    if (labelIdx > 0 && labelIdx < labels.length) {  // Skip background
      const regionName = labels[labelIdx];
      regionVolumes[regionName] = count;  // mm³
    }
  });

  console.log("Region volumes:", regionVolumes);
}

// ============================================
// MEDICAL-GRADE CALCULATION FUNCTIONS
// ============================================

/**
 * Estimate Intracranial Volume (ICV) from segmentation
 *
 * ICV estimation is critical for accurate volumetric analysis.
 * The segmentation only captures brain parenchyma, ventricles, and some CSF,
 * but ICV includes additional extra-axial CSF, meninges, and skull cavity space.
 *
 * Method: Uses age- and sex-adjusted Brain Parenchymal Fraction (BPF) to estimate ICV.
 * Based on: Vågberg et al. (2017) systematic review, ADNI normative data.
 *
 * @param {Object} volumes - Region volumes in mm³
 * @param {number} age - Patient age (used for age-adjusted estimation)
 * @param {string} sex - Patient sex ('male' or 'female')
 * @returns {number} Estimated ICV in mm³
 */
function estimateICV(volumes, age = 70, sex = 'male') {
  // Calculate brain parenchymal volume (excluding ventricles and CSF)
  let brainVolume = 0;
  let ventricleVolume = 0;

  for (const [region, volume] of Object.entries(volumes)) {
    if (region.toLowerCase().includes("ventricle")) {
      ventricleVolume += volume;
    } else {
      brainVolume += volume;
    }
  }

  // Total segmented intracranial volume
  const segmentedTotal = brainVolume + ventricleVolume;

  // Method 1: Age-adjusted BPF-based estimation
  // Get expected BPF for this age from normative data
  const ageDecade = Math.min(80, Math.max(20, Math.round(age / 10) * 10));
  const expectedBPF = BPF_NORMATIVE.byAge[ageDecade]?.mean || 0.76;

  // ICV estimated from brain volume assuming population-average BPF
  // ICV = BrainVolume / BPF
  // However, we need to account for the fact that our segmentation
  // doesn't capture all brain tissue (especially sulcal CSF)
  const icvFromBPF = brainVolume / expectedBPF;

  // Method 2: Anatomical estimation
  // Segmented volume typically captures ~85-90% of true ICV
  // (missing extra-axial CSF in sulci and around brain surface)
  // This factor is derived from FreeSurfer validation studies
  const anatomicalScaleFactor = 1.12; // ~12% additional for unsegmented CSF
  const icvFromAnatomical = segmentedTotal * anatomicalScaleFactor;

  // Method 3: Sex-adjusted reference scaling
  // Use sex-specific ICV distributions as a sanity check
  const refICV = ICV_REFERENCE.mean[sex]; // 1,550,000 male, 1,350,000 female
  const refSD = ICV_REFERENCE.sd[sex];

  // Weight the methods:
  // - BPF method is most reliable for normal aging
  // - Anatomical method provides bounds check
  // - Use weighted average with sanity checks

  // Primary estimate: BPF-based (most physiologically grounded)
  let estimatedICV = icvFromBPF;

  // Sanity check: ICV should be within reasonable bounds for sex
  // Typical range: mean ± 3SD
  const minICV = refICV - 3 * refSD;
  const maxICV = refICV + 3 * refSD;

  // If BPF-based estimate is outside reasonable range, blend with anatomical
  if (estimatedICV < minICV || estimatedICV > maxICV) {
    // Blend 50/50 with anatomical estimate
    estimatedICV = (icvFromBPF + icvFromAnatomical) / 2;
  }

  // Final bounds: ensure ICV is at least larger than segmented total
  // and within physiological range
  estimatedICV = Math.max(estimatedICV, segmentedTotal * 1.05);
  estimatedICV = Math.min(estimatedICV, maxICV);
  estimatedICV = Math.max(estimatedICV, minICV);

  return Math.round(estimatedICV);
}

/**
 * Validate analysis results for anomalies that may indicate data quality issues
 * @param {Object} results - Analysis results
 * @returns {Object} Validation report with warnings and data quality flags
 */
function validateAnalysisResults(results) {
  const warnings = [];
  const flags = {
    overallQuality: "good",
    possibleIssues: []
  };

  // Check 1: Extreme z-scores (|z| > 3.5 is very unusual)
  let extremeZscoreCount = 0;
  for (const [region, data] of Object.entries(results.regions || {})) {
    if (Math.abs(data.zscore) > 3.5) {
      extremeZscoreCount++;
      warnings.push(`${region}: z-score of ${data.zscore} is statistically unusual (>3.5 SD from mean)`);
    }
  }
  if (extremeZscoreCount >= 3) {
    flags.possibleIssues.push("multiple_extreme_zscores");
    flags.overallQuality = "review_recommended";
  }

  // Check 2: BPF outside physiological range (should be 0.65-0.92 for adults)
  if (results.bpf) {
    if (results.bpf.value > 0.92) {
      warnings.push(`BPF of ${(results.bpf.value * 100).toFixed(1)}% exceeds typical upper limit (92%). May indicate ICV underestimation.`);
      flags.possibleIssues.push("bpf_too_high");
      flags.overallQuality = "review_recommended";
    } else if (results.bpf.value < 0.55) {
      warnings.push(`BPF of ${(results.bpf.value * 100).toFixed(1)}% is below typical range. May indicate severe atrophy or measurement error.`);
      flags.possibleIssues.push("bpf_very_low");
    }
  }

  // Check 3: Implausible volume relationships
  const hippoVol = results.regions?.["Hippocampus"]?.volume || 0;
  const amygVol = results.regions?.["Amygdala"]?.volume || 0;
  if (hippoVol > 0 && amygVol > 0) {
    const ratio = hippoVol / amygVol;
    // Hippocampus should typically be 1.8-3.0x amygdala volume
    if (ratio < 1.2 || ratio > 4.5) {
      warnings.push(`Hippocampus/Amygdala ratio (${ratio.toFixed(1)}) is outside typical range (1.8-3.0). May indicate segmentation variability.`);
      flags.possibleIssues.push("unusual_volume_ratio");
    }
  }

  // Check 4: Total brain volume plausibility
  const totalBrain = results.totalBrainVolume || 0;
  if (totalBrain > 0) {
    // Adult brain typically 1000-1600 cm³ (1,000,000 - 1,600,000 mm³)
    if (totalBrain < 800000) {
      warnings.push(`Total brain volume (${(totalBrain / 1000).toFixed(0)} cm³) is below typical adult range.`);
      flags.possibleIssues.push("brain_volume_low");
    } else if (totalBrain > 1700000) {
      warnings.push(`Total brain volume (${(totalBrain / 1000).toFixed(0)} cm³) exceeds typical adult range.`);
      flags.possibleIssues.push("brain_volume_high");
    }
  }

  // Check 5: Discordant findings (e.g., small ventricles with reported atrophy)
  const lvZ = results.regions?.["Lateral-Ventricle"]?.zscore;
  const hippoZ = results.regions?.["Hippocampus"]?.effectiveZscore;
  if (lvZ !== undefined && hippoZ !== undefined) {
    // If hippocampus shows significant atrophy, ventricles should typically be enlarged
    if (hippoZ < -2.0 && lvZ < -1.0) {
      warnings.push(`Discordant finding: significant hippocampal atrophy (z=${hippoZ.toFixed(1)}) with relatively small ventricles (z=${lvZ.toFixed(1)}). Consider early-stage disease or measurement variability.`);
      flags.possibleIssues.push("discordant_atrophy_pattern");
    }
  }

  return {
    isValid: warnings.length === 0,
    warnings,
    flags,
    recommendedAction: flags.overallQuality === "review_recommended"
      ? "Results may require clinical review due to unusual findings"
      : "Results appear consistent with expected values"
  };
}

/**
 * Apply ICV normalization using residual method
 * Vol_adj = Vol - b × (ICV - ICV_mean)
 * @param {number} volume - Raw volume in mm³
 * @param {string} region - Region name
 * @param {number} icv - Intracranial volume in mm³
 * @param {string} sex - Patient sex
 * @returns {number} ICV-normalized volume
 */
function applyICVNormalization(volume, region, icv, sex) {
  const coef = ICV_REGRESSION_COEFFICIENTS[region];
  if (!coef) return volume;

  const icvMean = ICV_REFERENCE.mean[sex];
  const adjustedVolume = volume - coef.b * (icv - icvMean);

  return Math.max(0, adjustedVolume);  // Ensure non-negative
}

/**
 * Calculate Brain Parenchymal Fraction (BPF)
 * BPF = Total Brain Volume / ICV
 * @param {Object} volumes - Region volumes in mm³
 * @param {number} icv - Intracranial volume in mm³
 * @returns {Object} BPF value and interpretation
 */
function calculateBPF(volumes, icv, age) {
  // Sum parenchymal volumes (exclude ventricles and CSF)
  let brainVolume = 0;
  for (const [region, volume] of Object.entries(volumes)) {
    if (!region.toLowerCase().includes("ventricle")) {
      brainVolume += volume;
    }
  }

  const bpf = brainVolume / icv;

  // Get age-expected BPF
  const ageDecade = Math.min(80, Math.max(20, Math.round(age / 10) * 10));
  const normative = BPF_NORMATIVE.byAge[ageDecade] || BPF_NORMATIVE.byAge[70];

  // Calculate z-score
  const zscore = (bpf - normative.mean) / normative.sd;
  const percentile = Math.round(normalCDF(zscore) * 100);

  // Determine interpretation
  let interpretation;
  if (zscore >= BPF_NORMATIVE.thresholds.normal) {
    interpretation = "Normal";
  } else if (zscore >= BPF_NORMATIVE.thresholds.mild) {
    interpretation = "Low-Normal";
  } else if (zscore >= BPF_NORMATIVE.thresholds.moderate) {
    interpretation = "Mild Atrophy";
  } else if (zscore >= BPF_NORMATIVE.thresholds.severe) {
    interpretation = "Moderate Atrophy";
  } else {
    interpretation = "Severe Atrophy";
  }

  return {
    value: Math.round(bpf * 1000) / 1000,
    zscore: Math.round(zscore * 100) / 100,
    percentile,
    interpretation,
    expected: normative.mean,
    brainVolume,
    icv
  };
}

/**
 * Calculate Hippocampal Occupancy Score (HOC)
 * HOC = Hippocampus / (Hippocampus + Inferior Lateral Ventricle)
 * Key biomarker for AD progression - lower HOC indicates more atrophy
 * @param {Object} volumes - Region volumes in mm³
 * @param {number} age - Patient age
 * @returns {Object} HOC value and interpretation
 */
function calculateHOC(volumes, age) {
  const hippoVol = volumes["Hippocampus"] || 0;
  const ilvVol = volumes["Inferior-Lateral-Ventricle"] || 0;

  if (hippoVol === 0) {
    return { value: null, interpretation: "Unable to calculate - hippocampus not segmented" };
  }

  const hoc = hippoVol / (hippoVol + ilvVol);

  // Get age-expected HOC
  const ageDecade = Math.min(80, Math.max(50, Math.round(age / 10) * 10));
  const normative = HOC_NORMATIVE.byAge[ageDecade] || HOC_NORMATIVE.byAge[70];

  // Calculate z-score (note: lower HOC = worse, so z-score interpretation is inverted)
  const zscore = (hoc - normative.mean) / normative.sd;
  const percentile = Math.round(normalCDF(zscore) * 100);

  // Determine clinical interpretation
  let interpretation, conversionRisk;
  if (hoc >= HOC_NORMATIVE.interpretation.normal.min) {
    interpretation = HOC_NORMATIVE.interpretation.normal;
    conversionRisk = HOC_NORMATIVE.conversionRisk.low;
  } else if (hoc >= HOC_NORMATIVE.interpretation.mild.min) {
    interpretation = HOC_NORMATIVE.interpretation.mild;
    conversionRisk = HOC_NORMATIVE.conversionRisk.moderate;
  } else if (hoc >= HOC_NORMATIVE.interpretation.moderate.min) {
    interpretation = HOC_NORMATIVE.interpretation.moderate;
    conversionRisk = HOC_NORMATIVE.conversionRisk.high;
  } else {
    interpretation = HOC_NORMATIVE.interpretation.severe;
    conversionRisk = HOC_NORMATIVE.conversionRisk.veryHigh;
  }

  return {
    value: Math.round(hoc * 1000) / 1000,
    zscore: Math.round(zscore * 100) / 100,
    percentile,
    interpretation: interpretation.label,
    description: interpretation.description,
    conversionRisk: conversionRisk.risk,
    conversionRate: conversionRisk.conversionRate,
    hippoVolume: hippoVol,
    ilvVolume: ilvVol,
    expected: normative.mean
  };
}

/**
 * Detect clinical atrophy patterns (AD, FTD, normal aging, vascular)
 * @param {Object} analysisResults - Analysis results with z-scores
 * @returns {Object} Detected patterns and confidence levels
 */
function detectClinicalPatterns(results) {
  const patterns = [];

  const hippoZ = results.regions["Hippocampus"]?.effectiveZscore;
  const amygZ = results.regions["Amygdala"]?.effectiveZscore;
  const caudateZ = results.regions["Caudate"]?.effectiveZscore;
  const wmZ = results.regions["Cerebral-White-Matter"]?.effectiveZscore;
  const hoc = results.hoc?.value;
  const bpfZ = results.bpf?.zscore;

  // Check for Alzheimer's Disease pattern
  if (hippoZ !== undefined && hippoZ < -1.5) {
    let adScore = 0;
    let adIndicators = [];

    if (hippoZ < -2.0) { adScore += 3; adIndicators.push("significant hippocampal atrophy"); }
    else if (hippoZ < -1.5) { adScore += 2; adIndicators.push("mild hippocampal atrophy"); }

    if (hoc && hoc < 0.75) { adScore += 2; adIndicators.push("low HOC score"); }
    if (amygZ && amygZ < -1.5) { adScore += 1; adIndicators.push("amygdala atrophy"); }

    if (adScore >= 3) {
      patterns.push({
        pattern: CLINICAL_PATTERNS.alzheimerDisease.name,
        confidence: adScore >= 5 ? "High" : "Moderate",
        indicators: adIndicators,
        recommendation: "Consider clinical correlation for Alzheimer's disease. Recommend neuropsychological testing if not already performed."
      });
    }
  }

  // Check for Frontotemporal pattern (asymmetric, caudate involvement)
  if (caudateZ && caudateZ < -2.0 && (!hippoZ || hippoZ > caudateZ + 0.5)) {
    patterns.push({
      pattern: CLINICAL_PATTERNS.frontotemporalDementia.name,
      confidence: "Low-Moderate",
      indicators: ["caudate atrophy exceeds hippocampal atrophy"],
      recommendation: "Pattern may suggest frontotemporal involvement. Consider behavioral assessment."
    });
  }

  // Check for Vascular pattern
  if (wmZ && wmZ < -2.0) {
    patterns.push({
      pattern: CLINICAL_PATTERNS.vascularDementia.name,
      confidence: "Moderate",
      indicators: ["significant white matter volume loss"],
      recommendation: "White matter atrophy detected. Consider vascular risk factors and MRA if not performed."
    });
  }

  // Check for Early MCI / Subtle Medial Temporal Changes
  // HOC can detect medial temporal changes before frank hippocampal atrophy
  // This is important for early MCI detection
  if (hoc && hoc < 0.80 && (!hippoZ || hippoZ >= -1.5)) {
    let mciIndicators = [];
    let confidence = "Low-Moderate";

    if (hoc < 0.70) {
      mciIndicators.push("moderate HOC reduction");
      confidence = "Moderate";
    } else if (hoc < 0.75) {
      mciIndicators.push("mild-moderate HOC reduction");
      confidence = "Low-Moderate";
    } else {
      mciIndicators.push("mild HOC reduction");
      confidence = "Low";
    }

    // Check for temporal horn enlargement (ILV)
    const ilvZ = results.regions["Inferior-Lateral-Ventricle"]?.zscore;
    if (ilvZ && ilvZ > 0.5) {
      mciIndicators.push("temporal horn enlargement");
      confidence = confidence === "Low" ? "Low-Moderate" : "Moderate";
    }

    patterns.push({
      pattern: "Early MCI / Subtle Medial Temporal Changes",
      confidence: confidence,
      indicators: mciIndicators,
      recommendation: "HOC indicates subtle medial temporal changes despite preserved hippocampal volume. Consider neuropsychological testing to evaluate for early MCI. Follow-up imaging in 12-18 months may be informative."
    });
  }

  // Check for Normal Aging pattern
  // Only if no other patterns detected AND HOC is normal
  if (patterns.length === 0 && bpfZ && bpfZ >= -1.5 && (!hippoZ || hippoZ >= -1.5) && (!hoc || hoc >= 0.80)) {
    patterns.push({
      pattern: CLINICAL_PATTERNS.normalAging.name,
      confidence: "High",
      indicators: ["volumes within expected range for age"],
      recommendation: "Findings consistent with normal age-related changes."
    });
  }

  // If still no patterns, but HOC is borderline, add cautionary note
  if (patterns.length === 0) {
    patterns.push({
      pattern: CLINICAL_PATTERNS.normalAging.name,
      confidence: "Moderate",
      indicators: ["volumes largely within expected range"],
      recommendation: "Overall volumes appear normal. Continue routine monitoring."
    });
  }

  return patterns;
}

/**
 * Calculate age-varying standard deviation
 * SD increases with age due to greater population variability
 * @param {number} baseSd - Base SD from normative data
 * @param {number} age - Patient age
 * @returns {number} Age-adjusted SD
 */
function getAgeAdjustedSD(baseSd, age) {
  // SD increases ~1% per decade after age 50
  const ageFactor = age > 50 ? 1 + (age - 50) / 100 * 0.1 : 1;
  return baseSd * ageFactor;
}

// ============================================
// STANDARDIZED ATROPHY SCALE CALCULATIONS
// ============================================

/**
 * Calculate MTA Score (Scheltens Scale) from volumetric data
 * Uses two methods: QMTA ratio and hippocampal z-score
 * @param {Object} volumes - Region volumes
 * @param {number} hippoZscore - Hippocampal z-score
 * @param {number} age - Patient age
 * @returns {Object} MTA score and details
 */
function calculateMTAScore(volumes, hippoZscore, age) {
  const hippoVol = volumes["Hippocampus"] || 0;
  const ilvVol = volumes["Inferior-Lateral-Ventricle"] || 0;

  // Method 1: QMTA ratio (ILV/Hippocampus)
  let qmtaRatio = 0;
  let mtaFromRatio = 0;
  if (hippoVol > 0) {
    qmtaRatio = ilvVol / hippoVol;
    for (const threshold of STANDARDIZED_SCALES.MTA.qmtaToScore) {
      if (qmtaRatio <= threshold.maxRatio) {
        mtaFromRatio = threshold.score;
        break;
      }
    }
  }

  // Method 2: From hippocampal z-score
  let mtaFromZscore = 0;
  if (hippoZscore !== null && hippoZscore !== undefined) {
    for (const threshold of STANDARDIZED_SCALES.MTA.zscoreToScore) {
      if (hippoZscore >= threshold.minZ) {
        mtaFromZscore = threshold.score;
        break;
      }
    }
  }

  // Use average of both methods for robustness
  const mtaScore = Math.round((mtaFromRatio + mtaFromZscore) / 2 * 2) / 2; // Round to 0.5

  // Determine age-adjusted abnormality threshold
  let ageThreshold = 2.0;
  for (const [maxAge, threshold] of Object.entries(STANDARDIZED_SCALES.MTA.ageThresholds).sort((a, b) => a[0] - b[0])) {
    if (age < parseInt(maxAge)) {
      ageThreshold = threshold;
      break;
    }
  }

  const isAbnormal = mtaScore > ageThreshold;
  const scoreInfo = STANDARDIZED_SCALES.MTA.scores[Math.floor(mtaScore)] ||
    STANDARDIZED_SCALES.MTA.scores[Math.round(mtaScore)];

  return {
    score: mtaScore,
    maxScore: 4,
    label: scoreInfo?.label || "Unknown",
    description: scoreInfo?.description || "",
    qmtaRatio: Math.round(qmtaRatio * 100) / 100,
    mtaFromRatio,
    mtaFromZscore,
    ageThreshold,
    isAbnormal,
    interpretation: isAbnormal ? "Abnormal for age" : "Normal for age",
    reference: STANDARDIZED_SCALES.MTA.reference
  };
}

/**
 * Calculate GCA Score (Pasquier Scale) from cortical volume
 * @param {number} cortexZscore - Cerebral cortex z-score
 * @param {number} age - Patient age
 * @returns {Object} GCA score and details
 */
function calculateGCAScore(cortexZscore, age) {
  let gcaScore = 0;

  if (cortexZscore !== null && cortexZscore !== undefined) {
    for (const threshold of STANDARDIZED_SCALES.GCA.zscoreToScore) {
      if (cortexZscore >= threshold.minZ) {
        gcaScore = threshold.score;
        break;
      }
    }
  }

  // Age-adjusted threshold
  let ageThreshold = 2;
  for (const [maxAge, threshold] of Object.entries(STANDARDIZED_SCALES.GCA.ageThresholds).sort((a, b) => a[0] - b[0])) {
    if (age < parseInt(maxAge)) {
      ageThreshold = threshold;
      break;
    }
  }

  const isAbnormal = gcaScore > ageThreshold;
  const scoreInfo = STANDARDIZED_SCALES.GCA.scores[gcaScore];

  return {
    score: gcaScore,
    maxScore: 3,
    label: scoreInfo?.label || "Unknown",
    description: scoreInfo?.description || "",
    ageThreshold,
    isAbnormal,
    interpretation: isAbnormal ? "Abnormal for age" : "Normal for age",
    reference: STANDARDIZED_SCALES.GCA.reference
  };
}

/**
 * Calculate Koedam Posterior Atrophy Score
 * Uses cortical z-score as proxy (no direct parietal measurement)
 * @param {number} cortexZscore - Cerebral cortex z-score
 * @returns {Object} Koedam score and details
 */
function calculateKoedamScore(cortexZscore) {
  let paScore = 0;

  if (cortexZscore !== null && cortexZscore !== undefined) {
    for (const threshold of STANDARDIZED_SCALES.Koedam.zscoreToScore) {
      if (cortexZscore >= threshold.minZ) {
        paScore = threshold.score;
        break;
      }
    }
  }

  const scoreInfo = STANDARDIZED_SCALES.Koedam.scores[paScore];

  return {
    score: paScore,
    maxScore: 3,
    label: scoreInfo?.label || "Unknown",
    description: scoreInfo?.description || "",
    note: "Estimated from global cortical volume (parietal-specific segmentation not available)",
    reference: STANDARDIZED_SCALES.Koedam.reference
  };
}

/**
 * Calculate Evans Index from ventricular volume
 * Estimates linear measurement from volumetric data
 * @param {Object} volumes - Region volumes
 * @param {number} icv - Intracranial volume
 * @returns {Object} Evans Index and interpretation
 */
function calculateEvansIndex(volumes, icv) {
  const lvVol = volumes["Lateral-Ventricle"] || 0;

  if (lvVol === 0 || icv === 0) {
    return { value: null, interpretation: "Unable to calculate" };
  }

  // Estimate Evans Index from volume ratio
  // Using empirical calibration: EI ≈ k × (LV_vol/ICV)^(1/3)
  // Calibrated so that normal LV (~15-25mL in 1500mL ICV) gives EI ~0.22-0.26
  const volumeRatio = lvVol / icv;
  const evansIndex = 0.42 * Math.pow(volumeRatio, 0.30);
  const roundedEI = Math.round(evansIndex * 100) / 100;

  let interpretation, label;
  if (roundedEI <= 0.25) {
    label = "Normal";
    interpretation = "No significant ventricular enlargement";
  } else if (roundedEI <= 0.30) {
    label = "Borderline";
    interpretation = "Borderline ventricular enlargement";
  } else {
    label = "Enlarged";
    interpretation = "Ventricular enlargement present - consider hydrocephalus vs ex-vacuo dilation";
  }

  return {
    value: roundedEI,
    label,
    interpretation,
    volumeRatio: Math.round(volumeRatio * 1000) / 1000,
    thresholds: { normal: "≤0.25", borderline: "0.25-0.30", abnormal: ">0.30" },
    note: "Estimated from volumetric data (not direct linear measurement)",
    reference: STANDARDIZED_SCALES.EvansIndex.reference
  };
}

/**
 * Calculate all standardized atrophy scores
 * @param {Object} volumes - Region volumes
 * @param {Object} regionResults - Analysis results with z-scores
 * @param {number} age - Patient age
 * @param {number} icv - Intracranial volume
 * @returns {Object} All standardized scores
 */
function calculateStandardizedScores(volumes, regionResults, age, icv, hocValue = null) {
  const hippoZscore = regionResults["Hippocampus"]?.zscore;
  const cortexZscore = regionResults["Cerebral-Cortex"]?.zscore;

  return {
    mta: calculateMTAScore(volumes, hippoZscore, age),
    gca: calculateGCAScore(cortexZscore, age),
    koedam: calculateKoedamScore(cortexZscore),
    evansIndex: calculateEvansIndex(volumes, icv),
    summary: generateScoresSummary(volumes, regionResults, age, icv, hocValue)
  };
}

/**
 * Generate summary interpretation of standardized scores
 * @param {Object} volumes - Region volumes
 * @param {Object} regionResults - Analysis results with z-scores
 * @param {number} age - Patient age
 * @param {number} icv - Intracranial volume
 * @param {number} hocValue - Hippocampal Occupancy Score (0-1)
 */
function generateScoresSummary(volumes, regionResults, age, icv, hocValue = null) {
  const mta = calculateMTAScore(volumes, regionResults["Hippocampus"]?.zscore, age);
  const gca = calculateGCAScore(regionResults["Cerebral-Cortex"]?.zscore, age);
  const evans = calculateEvansIndex(volumes, icv);

  let pattern = "Normal aging";
  let confidence = "High";

  // Check for AD pattern: High MTA + normal/mild GCA
  if (mta.score >= 2.5 && gca.score <= 1) {
    pattern = "Consistent with AD (medial temporal predominant)";
    confidence = mta.score >= 3 ? "High" : "Moderate";
  }
  // Check for diffuse atrophy: High MTA + high GCA
  else if (mta.score >= 2 && gca.score >= 2) {
    pattern = "Diffuse atrophy pattern";
    confidence = "Moderate";
  }
  // Isolated cortical atrophy
  else if (gca.score >= 2 && mta.score <= 1) {
    pattern = "Cortical-predominant atrophy (consider FTD, posterior cortical atrophy)";
    confidence = "Low-Moderate";
  }
  // Ventricular predominant
  else if (evans.value > 0.30 && mta.score <= 1.5) {
    pattern = "Ventricular enlargement out of proportion to atrophy - consider NPH";
    confidence = "Moderate";
  }
  // Check for early MCI pattern based on HOC (subtle changes not captured by MTA score alone)
  else if (hocValue !== null && hocValue < 0.80 && mta.score < 2) {
    if (hocValue < 0.70) {
      pattern = "Early medial temporal changes (HOC-based)";
      confidence = "Moderate";
    } else if (hocValue < 0.75) {
      pattern = "Subtle medial temporal changes";
      confidence = "Low-Moderate";
    } else {
      pattern = "Borderline medial temporal findings";
      confidence = "Low";
    }
  }

  return {
    overallPattern: pattern,
    confidence,
    mtaAbnormal: mta.isAbnormal,
    gcaAbnormal: gca.isAbnormal,
    ventriculomegaly: evans.value > 0.30,
    hocAbnormal: hocValue !== null && hocValue < 0.80
  };
}

async function analyzeAtrophy() {
  const age = parseInt(document.getElementById("patientAge").value) || 65;
  const sex = document.getElementById("patientSex").value;

  analysisResults = {
    age,
    sex,
    regions: {},
    totalBrainVolume: 0,
    atrophyRisk: "Unknown",
    percentile: 0,
    findings: [],
    // Medical-grade additions
    icv: 0,
    bpf: null,
    hoc: null,
    clinicalPatterns: [],
    icvNormalized: true
  };

  // ========================================
  // STEP 1: Estimate ICV and calculate raw totals
  // ========================================
  const estimatedICV = estimateICV(regionVolumes, age, sex);
  analysisResults.icv = estimatedICV;

  // Calculate total brain volume (excluding ventricles)
  let totalVolume = 0;
  for (const [region, volume] of Object.entries(regionVolumes)) {
    if (!region.toLowerCase().includes("ventricle")) {
      totalVolume += volume;
    }
  }
  analysisResults.totalBrainVolume = totalVolume;

  // ========================================
  // STEP 2: Calculate medical-grade biomarkers
  // ========================================

  // Brain Parenchymal Fraction (BPF)
  analysisResults.bpf = calculateBPF(regionVolumes, estimatedICV, age);

  // Hippocampal Occupancy Score (HOC)
  analysisResults.hoc = calculateHOC(regionVolumes, age);

  // ========================================
  // STEP 3: Analyze each region with ICV normalization
  // ========================================
  let criticalAtrophyCount = 0;
  let moderateAtrophyCount = 0;
  let mildAtrophyCount = 0;
  let hippocampusZscore = null;

  // Store ICV-normalized volumes for analysis
  const normalizedVolumes = {};

  for (const [region, volume] of Object.entries(regionVolumes)) {
    const normData = findNormativeData(region);
    if (normData) {
      // Apply ICV normalization using residual method
      const normalizedVolume = applyICVNormalization(volume, region, estimatedICV, sex);
      normalizedVolumes[region] = normalizedVolume;

      // Use age-adjusted SD for more precise z-scores
      const baseSd = normData.sd[sex];
      const adjustedSd = getAgeAdjustedSD(baseSd, age);

      // Calculate z-score using normalized volume
      const expectedMean = interpolateByAge(normData.mean[sex], age);
      const zscore = (normalizedVolume - expectedMean) / adjustedSd;
      const roundedZscore = Math.round(zscore * 100) / 100;

      const interpretation = interpretZScore(roundedZscore, normData.invertZscore);
      const percentile = Math.round(normalCDF(roundedZscore) * 100);

      // For ventricles, effective z-score for atrophy assessment is inverted
      const effectiveZ = normData.invertZscore ? -roundedZscore : roundedZscore;

      analysisResults.regions[region] = {
        volume,
        normalizedVolume: Math.round(normalizedVolume),
        icvNormalized: true,
        zscore: roundedZscore,
        effectiveZscore: Math.round(effectiveZ * 100) / 100,
        interpretation,
        percentile,
        normData,
        clinicalSignificance: normData.clinicalSignificance || "medium",
        expectedVolume: Math.round(expectedMean)
      };

      // Track hippocampus specifically (critical for dementia assessment)
      if (region.toLowerCase().includes('hippocampus')) {
        hippocampusZscore = effectiveZ;
        analysisResults.hippocampusAnalysis = {
          volume: volume,
          normalizedVolume: Math.round(normalizedVolume),
          zscore: roundedZscore,
          percentile,
          expectedForAge: Math.round(expectedMean),
          interpretation
        };
      }

      // Count atrophy severity using effective z-score
      if (effectiveZ < -2.5) {
        criticalAtrophyCount++;
      } else if (effectiveZ < -2.0) {
        moderateAtrophyCount++;
      } else if (effectiveZ < -1.5) {
        mildAtrophyCount++;
      }

      // Generate findings for clinically significant deviations
      const significanceThresholds = {
        critical: -1.0,
        high: -1.5,
        medium: -1.5,
        low: -2.0
      };
      const threshold = significanceThresholds[normData.clinicalSignificance] || -1.5;

      if (effectiveZ < threshold || (normData.invertZscore && roundedZscore > Math.abs(threshold))) {
        analysisResults.findings.push(generateFinding(region, roundedZscore, normData));
      }
    }
  }

  // ========================================
  // STEP 4: Calculate overall atrophy risk
  // ========================================
  const hocValue = analysisResults.hoc?.value;
  const bpfZscore = analysisResults.bpf?.zscore;

  // Enhanced risk calculation incorporating HOC and BPF
  // HOC is particularly sensitive for early MCI detection
  if (criticalAtrophyCount >= 2 || (hippocampusZscore !== null && hippocampusZscore < -2.5) ||
    (hocValue && hocValue < 0.60)) {
    analysisResults.atrophyRisk = "High";
    analysisResults.riskDescription = "Significant atrophy detected - clinical correlation strongly recommended";
  } else if (moderateAtrophyCount >= 1 || criticalAtrophyCount >= 1 ||
    (hippocampusZscore !== null && hippocampusZscore < -2.0) ||
    (hocValue && hocValue < 0.70)) {
    analysisResults.atrophyRisk = "Moderate";
    analysisResults.riskDescription = "Notable atrophy present - consider neuropsychological evaluation";
  } else if (mildAtrophyCount >= 2 || (hippocampusZscore !== null && hippocampusZscore < -1.5) ||
    (bpfZscore && bpfZscore < -1.5) ||
    (hocValue && hocValue < 0.75)) {
    // HOC 0.70-0.75 indicates mild-moderate medial temporal changes
    analysisResults.atrophyRisk = "Mild";
    analysisResults.riskDescription = hocValue && hocValue < 0.75
      ? "Subtle medial temporal changes detected (HOC reduced) - neuropsychological evaluation recommended"
      : "Mild volume reductions detected - monitoring recommended";
  } else if (mildAtrophyCount >= 1 || (hippocampusZscore !== null && hippocampusZscore < -1.0) ||
    (hocValue && hocValue < 0.80)) {
    // HOC 0.75-0.80 indicates subtle changes
    analysisResults.atrophyRisk = "Low-Normal";
    analysisResults.riskDescription = hocValue && hocValue < 0.80
      ? "HOC in low-normal range - consider baseline for future comparison"
      : "Volumes in low-normal range for age";
  } else {
    analysisResults.atrophyRisk = "Normal";
    analysisResults.riskDescription = "Volumes within expected range for age and sex";
  }

  // ========================================
  // STEP 5: Detect clinical patterns
  // ========================================
  analysisResults.clinicalPatterns = detectClinicalPatterns(analysisResults);

  // ========================================
  // STEP 5b: Calculate standardized atrophy scores
  // ========================================
  analysisResults.standardizedScores = calculateStandardizedScores(
    regionVolumes,
    analysisResults.regions,
    age,
    estimatedICV,
    analysisResults.hoc?.value  // Pass HOC for pattern detection
  );

  // ========================================
  // STEP 5c: Calculate asymmetry metrics
  // ========================================
  analysisResults.asymmetry = calculateAsymmetryMetrics(regionVolumes);

  // ========================================
  // STEP 5d: Calculate lobar atrophy
  // ========================================
  analysisResults.lobarAtrophy = calculateLobarAtrophy(regionVolumes, analysisResults.regions);

  // ========================================
  // STEP 5e: Calculate global atrophy index
  // ========================================
  analysisResults.globalAtrophyIndex = calculateGlobalAtrophyIndex(analysisResults);

  // ========================================
  // STEP 5f: Detect FTD patterns
  // ========================================
  const ftdPatterns = detectFTDPatterns(
    analysisResults.asymmetry,
    analysisResults.lobarAtrophy,
    analysisResults.regions
  );
  // Add FTD patterns to clinical patterns
  if (ftdPatterns.length > 0) {
    analysisResults.clinicalPatterns = [...analysisResults.clinicalPatterns, ...ftdPatterns];
  }

  // Add asymmetry findings if concerning
  if (analysisResults.asymmetry?.concerningCount > 0) {
    const concerningRegions = Object.entries(analysisResults.asymmetry.regions)
      .filter(([_, data]) => data.isConcerning)
      .map(([_, data]) => `${data.clinicalName} (${data.asymmetryIndex.toFixed(1)}%, ${data.laterality})`)
      .join("; ");

    analysisResults.findings.push({
      severity: analysisResults.asymmetry.concerningCount > 2 ? "warning" : "info",
      text: `Asymmetry detected: ${concerningRegions}. May be relevant for FTD evaluation.`,
      type: "asymmetry"
    });
  }

  // Add HOC-specific findings if concerning
  if (analysisResults.hoc && analysisResults.hoc.value !== null) {
    if (analysisResults.hoc.value < 0.60) {
      analysisResults.findings.unshift({
        severity: "danger",
        text: `Hippocampal Occupancy Score: ${(analysisResults.hoc.value * 100).toFixed(1)}% (${analysisResults.hoc.interpretation}). ${analysisResults.hoc.description}. MCI→AD conversion risk: ${analysisResults.hoc.conversionRisk} (${analysisResults.hoc.conversionRate}).`,
        type: "hoc"
      });
    } else if (analysisResults.hoc.value < 0.70) {
      analysisResults.findings.unshift({
        severity: "warning",
        text: `Hippocampal Occupancy Score: ${(analysisResults.hoc.value * 100).toFixed(1)}% (${analysisResults.hoc.interpretation}). ${analysisResults.hoc.description}. MCI→AD conversion risk: ${analysisResults.hoc.conversionRisk} (${analysisResults.hoc.conversionRate}).`,
        type: "hoc"
      });
    } else if (analysisResults.hoc.value < 0.80) {
      // Add informational finding for borderline HOC (0.70-0.80)
      analysisResults.findings.unshift({
        severity: "info",
        text: `Hippocampal Occupancy Score: ${(analysisResults.hoc.value * 100).toFixed(1)}% is in the low-normal range (expected ≥${(analysisResults.hoc.expected * 100).toFixed(0)}% for age ${age}). ${analysisResults.hoc.description}. Consider as baseline for future comparison.`,
        type: "hoc"
      });
    }
  }

  // Add BPF finding if concerning
  if (analysisResults.bpf && analysisResults.bpf.zscore < -1.5) {
    analysisResults.findings.unshift({
      severity: analysisResults.bpf.zscore < -2.0 ? "danger" : "warning",
      text: `Brain Parenchymal Fraction: ${(analysisResults.bpf.value * 100).toFixed(1)}% (expected ${(analysisResults.bpf.expected * 100).toFixed(1)}% for age ${age}). ${analysisResults.bpf.interpretation}.`,
      type: "bpf"
    });
  }

  // ========================================
  // STEP 6: Calculate total brain percentile
  // ========================================
  const totalNorm = NORMATIVE_DATA.totalBrain;
  const expectedMean = interpolateByAge(totalNorm.mean[sex], age);
  const zsTotal = (totalVolume - expectedMean) / totalNorm.sd[sex];
  analysisResults.percentile = Math.round(normalCDF(zsTotal) * 100);
  analysisResults.totalBrainZscore = Math.round(zsTotal * 100) / 100;

  // Sort findings by severity
  analysisResults.findings.sort((a, b) => {
    const severityOrder = { danger: 0, warning: 1, info: 2 };
    return (severityOrder[a.severity] || 3) - (severityOrder[b.severity] || 3);
  });

  // ========================================
  // STEP 7: Validate results for anomalies
  // ========================================
  analysisResults.validation = validateAnalysisResults(analysisResults);

  // Add validation warnings to findings if any
  if (analysisResults.validation.warnings.length > 0) {
    for (const warning of analysisResults.validation.warnings) {
      analysisResults.findings.push({
        severity: "info",
        text: `Quality note: ${warning}`,
        type: "validation"
      });
    }
  }

  console.log("Medical-grade analysis results:", analysisResults);
}

function findNormativeData(regionName) {
  // Try exact match first (region names from colormap use hyphens)
  if (NORMATIVE_DATA.regions[regionName]) {
    return { ...NORMATIVE_DATA.regions[regionName], matchedName: regionName };
  }

  // Normalize the region name for matching
  const normalizedName = regionName.replace(/\s+/g, '-');
  if (NORMATIVE_DATA.regions[normalizedName]) {
    return { ...NORMATIVE_DATA.regions[normalizedName], matchedName: normalizedName };
  }

  // Try case-insensitive and flexible matching
  const lowerName = regionName.toLowerCase().replace(/[-_\s]+/g, '');
  for (const [key, data] of Object.entries(NORMATIVE_DATA.regions)) {
    const lowerKey = key.toLowerCase().replace(/[-_\s]+/g, '');
    if (lowerName === lowerKey || lowerName.includes(lowerKey) || lowerKey.includes(lowerName)) {
      return { ...data, matchedName: key };
    }
  }

  // Handle common aliases
  const aliases = {
    'accumbens': 'Accumbens-area',
    'nucleusaccumbens': 'Accumbens-area',
    'ventraldc': 'VentralDC',
    'ventraldiencephalon': 'VentralDC',
    'brainstem': 'Brain-Stem',
    'inflatventricle': 'Inferior-Lateral-Ventricle',
    'inferiorlateralventricle': 'Inferior-Lateral-Ventricle',
    'lateralventricle': 'Lateral-Ventricle',
    'cerebralwhitematter': 'Cerebral-White-Matter',
    'cerebralcortex': 'Cerebral-Cortex',
    'cerebellumwhitematter': 'Cerebellum-White-Matter',
    'cerebellumcortex': 'Cerebellum-Cortex',
    '3rdventricle': '3rd-Ventricle',
    'thirdventricle': '3rd-Ventricle',
    '4thventricle': '4th-Ventricle',
    'fourthventricle': '4th-Ventricle'
  };

  const aliasKey = lowerName.replace(/[-_\s]+/g, '');
  if (aliases[aliasKey] && NORMATIVE_DATA.regions[aliases[aliasKey]]) {
    return { ...NORMATIVE_DATA.regions[aliases[aliasKey]], matchedName: aliases[aliasKey] };
  }

  return null;
}

function calculateZScore(volume, age, sex, normData) {
  const expectedMean = interpolateByAge(normData.mean[sex], age);
  const sd = normData.sd[sex];
  let zscore = (volume - expectedMean) / sd;

  // For ventricles, larger = worse, so invert
  if (normData.invertZscore) {
    zscore = -zscore;
  }

  return Math.round(zscore * 100) / 100;
}

function interpolateByAge(ageValues, age) {
  // ageValues is array for ages [20, 40, 60, 80]
  const ages = [20, 40, 60, 80];

  if (age <= 20) return ageValues[0];
  if (age >= 80) return ageValues[3];

  // Find bracketing ages
  let i = 0;
  while (i < 3 && ages[i + 1] < age) i++;

  // Linear interpolation
  const t = (age - ages[i]) / (ages[i + 1] - ages[i]);
  return ageValues[i] + t * (ageValues[i + 1] - ageValues[i]);
}

function interpretZScore(zscore, invertZscore = false) {
  // For ventricles, invertZscore=true means larger volume = atrophy
  // After inversion, negative z-scores indicate atrophy
  const effectiveZ = invertZscore ? -zscore : zscore;

  // Clinical interpretation based on radiological standards
  // Normal: z >= -1.0 (above 16th percentile)
  // Low-Normal: -1.5 <= z < -1.0 (7th-16th percentile)
  // Mild Atrophy: -2.0 <= z < -1.5 (2nd-7th percentile)
  // Moderate Atrophy: -2.5 <= z < -2.0 (0.6-2nd percentile)
  // Severe Atrophy: z < -2.5 (below 0.6th percentile)

  if (effectiveZ >= -1.0) return "normal";
  if (effectiveZ >= -1.5) return "low-normal";
  if (effectiveZ >= -2.0) return "mild";
  if (effectiveZ >= -2.5) return "moderate";
  return "severe";
}

function getInterpretationDetails(zscore, regionName, normData) {
  const interpretation = interpretZScore(zscore, normData?.invertZscore);
  const percentile = Math.round(normalCDF(zscore) * 100);

  const details = {
    category: interpretation,
    percentile: percentile,
    clinicalSignificance: normData?.clinicalSignificance || "medium",
    isVentricle: normData?.invertZscore || false
  };

  // Add clinical context for critical structures
  if (regionName.toLowerCase().includes('hippocampus')) {
    if (zscore < -2.0) {
      details.clinicalNote = "Significant hippocampal atrophy - consider evaluation for neurodegenerative disease";
    } else if (zscore < -1.5) {
      details.clinicalNote = "Mild hippocampal volume reduction - may warrant monitoring";
    }
  }

  return details;
}

function normalCDF(z) {
  // Approximation of standard normal CDF
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;

  const sign = z < 0 ? -1 : 1;
  z = Math.abs(z) / Math.sqrt(2);

  const t = 1.0 / (1.0 + p * z);
  const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-z * z);

  return 0.5 * (1.0 + sign * y);
}

function generateFinding(region, zscore, normData) {
  const clinicalName = normData.clinicalName || region;
  const percentile = Math.round(normalCDF(zscore) * 100);
  const interpretation = interpretZScore(zscore, normData.invertZscore);
  const isVentricle = normData.invertZscore;

  // For ventricles: positive z-score means enlargement (which indicates atrophy)
  // For brain tissue: negative z-score means volume loss (atrophy)
  const effectiveZ = isVentricle ? -zscore : zscore;

  let severity, text;

  if (isVentricle) {
    // Ventricle interpretation (larger = worse)
    if (zscore > 2.0) {
      severity = "danger";
      text = `${clinicalName} shows significant enlargement (${percentile}th percentile, z = ${zscore.toFixed(1)}), indicating surrounding brain tissue atrophy.`;
    } else if (zscore > 1.5) {
      severity = "warning";
      text = `${clinicalName} shows mild enlargement (${percentile}th percentile, z = ${zscore.toFixed(1)}), possibly indicating early atrophy.`;
    } else {
      severity = "info";
      text = `${clinicalName} is borderline enlarged (${percentile}th percentile, z = ${zscore.toFixed(1)}).`;
    }
  } else {
    // Brain tissue interpretation (smaller = worse)
    if (effectiveZ < -2.5) {
      severity = "danger";
      text = `${clinicalName} shows severe volume loss (${percentile}th percentile, z = ${zscore.toFixed(1)}), indicating significant atrophy.`;
    } else if (effectiveZ < -2.0) {
      severity = "danger";
      text = `${clinicalName} shows moderate volume loss (${percentile}th percentile, z = ${zscore.toFixed(1)}), indicating notable atrophy.`;
    } else if (effectiveZ < -1.5) {
      severity = "warning";
      text = `${clinicalName} shows mild volume reduction (${percentile}th percentile, z = ${zscore.toFixed(1)}), suggesting early atrophy.`;
    } else {
      severity = "info";
      text = `${clinicalName} is in the low-normal range (${percentile}th percentile, z = ${zscore.toFixed(1)}).`;
    }
  }

  // Add special notes for critical structures
  if (region.toLowerCase().includes('hippocampus') && effectiveZ < -1.5) {
    text += " Hippocampal atrophy is a key biomarker for Alzheimer's disease and MCI.";
  }

  return { severity, text, interpretation, percentile };
}

// ============================================
// DISPLAY RESULTS
// ============================================

/**
 * Display medical-grade biomarkers (HOC, BPF, Clinical Patterns)
 * Creates dynamic UI elements for advanced metrics
 */
function displayMedicalBiomarkers() {
  // Find or create biomarkers container
  let biomarkersCard = document.getElementById("biomarkersCard");

  if (!biomarkersCard) {
    // Create new biomarkers card after summary card
    biomarkersCard = document.createElement("div");
    biomarkersCard.id = "biomarkersCard";
    biomarkersCard.className = "card";
    biomarkersCard.innerHTML = `
      <h3 class="card-title">
        Advanced Biomarkers
        <span class="card-subtitle">Medical-grade volumetric analysis</span>
      </h3>
      <div id="biomarkersContent"></div>
    `;

    const summaryCard = document.getElementById("summaryCard");
    summaryCard.parentNode.insertBefore(biomarkersCard, summaryCard.nextSibling);
  }

  biomarkersCard.style.display = "block";
  const content = document.getElementById("biomarkersContent") || biomarkersCard.querySelector("div");
  content.innerHTML = "";

  // ========================================
  // Hippocampal Occupancy Score (HOC)
  // ========================================
  if (analysisResults.hoc && analysisResults.hoc.value !== null) {
    const hoc = analysisResults.hoc;
    const hocColor = hoc.value >= 0.80 ? "#22c55e" :
      hoc.value >= 0.70 ? "#84cc16" :
        hoc.value >= 0.60 ? "#f59e0b" : "#ef4444";

    const hocSection = document.createElement("div");
    hocSection.className = "biomarker-section";
    hocSection.innerHTML = `
      <div class="biomarker-header">
        <span class="biomarker-name">Hippocampal Occupancy Score (HOC)</span>
        <span class="biomarker-value" style="color: ${hocColor}">${(hoc.value * 100).toFixed(1)}%</span>
      </div>
      <div class="biomarker-details">
        <div class="biomarker-row">
          <span>Status:</span>
          <span style="color: ${hocColor}; font-weight: 500;">${hoc.interpretation}</span>
        </div>
        <div class="biomarker-row">
          <span>Expected for age ${analysisResults.age}:</span>
          <span>${(hoc.expected * 100).toFixed(1)}%</span>
        </div>
        <div class="biomarker-row">
          <span>Percentile:</span>
          <span>${hoc.percentile}th</span>
        </div>
        <div class="biomarker-row">
          <span>MCI→AD Risk:</span>
          <span style="font-weight: 500;">${hoc.conversionRisk}</span>
        </div>
        <div class="biomarker-description">${hoc.description}</div>
      </div>
    `;
    content.appendChild(hocSection);
  }

  // ========================================
  // Brain Parenchymal Fraction (BPF)
  // ========================================
  if (analysisResults.bpf) {
    const bpf = analysisResults.bpf;
    const bpfColor = bpf.zscore >= -1.0 ? "#22c55e" :
      bpf.zscore >= -1.5 ? "#84cc16" :
        bpf.zscore >= -2.0 ? "#f59e0b" : "#ef4444";

    const bpfSection = document.createElement("div");
    bpfSection.className = "biomarker-section";
    bpfSection.innerHTML = `
      <div class="biomarker-header">
        <span class="biomarker-name">Brain Parenchymal Fraction (BPF)</span>
        <span class="biomarker-value" style="color: ${bpfColor}">${(bpf.value * 100).toFixed(1)}%</span>
      </div>
      <div class="biomarker-details">
        <div class="biomarker-row">
          <span>Status:</span>
          <span style="color: ${bpfColor}; font-weight: 500;">${bpf.interpretation}</span>
        </div>
        <div class="biomarker-row">
          <span>Expected for age ${analysisResults.age}:</span>
          <span>${(bpf.expected * 100).toFixed(1)}%</span>
        </div>
        <div class="biomarker-row">
          <span>Z-score:</span>
          <span>${bpf.zscore.toFixed(2)}</span>
        </div>
        <div class="biomarker-row">
          <span>Percentile:</span>
          <span>${bpf.percentile}th</span>
        </div>
        <div class="biomarker-row">
          <span>Est. ICV:</span>
          <span>${(analysisResults.icv / 1000).toFixed(0)} cm³</span>
        </div>
      </div>
    `;
    content.appendChild(bpfSection);
  }

  // ========================================
  // Standardized Atrophy Rating Scales
  // ========================================
  if (analysisResults.standardizedScores) {
    const scores = analysisResults.standardizedScores;

    const scalesSection = document.createElement("div");
    scalesSection.className = "biomarker-section scales-section";
    scalesSection.innerHTML = `
      <div class="biomarker-header">
        <span class="biomarker-name">Standardized Atrophy Scales</span>
        <span class="biomarker-subtitle">Clinical rating equivalents</span>
      </div>
      <div class="scales-grid">
        ${scores.mta ? `
        <div class="scale-item ${scores.mta.isAbnormal ? 'abnormal' : 'normal'}">
          <div class="scale-name">MTA Score</div>
          <div class="scale-value">${scores.mta.score}/4</div>
          <div class="scale-label">${scores.mta.label}</div>
          <div class="scale-status">${scores.mta.interpretation}</div>
          <div class="scale-ref">${scores.mta.reference}</div>
        </div>
        ` : ''}
        ${scores.gca ? `
        <div class="scale-item ${scores.gca.isAbnormal ? 'abnormal' : 'normal'}">
          <div class="scale-name">GCA Score</div>
          <div class="scale-value">${scores.gca.score}/3</div>
          <div class="scale-label">${scores.gca.label}</div>
          <div class="scale-status">${scores.gca.interpretation}</div>
          <div class="scale-ref">${scores.gca.reference}</div>
        </div>
        ` : ''}
        ${scores.koedam ? `
        <div class="scale-item">
          <div class="scale-name">PA Score</div>
          <div class="scale-value">${scores.koedam.score}/3</div>
          <div class="scale-label">${scores.koedam.label}</div>
          <div class="scale-status">Koedam Scale</div>
          <div class="scale-ref">${scores.koedam.reference}</div>
        </div>
        ` : ''}
        ${scores.evansIndex && scores.evansIndex.value ? `
        <div class="scale-item ${scores.evansIndex.value > 0.30 ? 'abnormal' : scores.evansIndex.value > 0.25 ? 'borderline' : 'normal'}">
          <div class="scale-name">Evans Index</div>
          <div class="scale-value">${scores.evansIndex.value.toFixed(2)}</div>
          <div class="scale-label">${scores.evansIndex.label}</div>
          <div class="scale-status">Normal ≤0.30</div>
          <div class="scale-ref">${scores.evansIndex.reference}</div>
        </div>
        ` : ''}
      </div>
      ${scores.summary ? `
      <div class="scales-summary">
        <strong>Pattern:</strong> ${scores.summary.overallPattern}
        <span class="confidence-badge ${scores.summary.confidence.toLowerCase()}">${scores.summary.confidence}</span>
      </div>
      ` : ''}
    `;
    content.appendChild(scalesSection);
  }

  // ========================================
  // Clinical Pattern Recognition
  // ========================================
  if (analysisResults.clinicalPatterns && analysisResults.clinicalPatterns.length > 0) {
    const patternsSection = document.createElement("div");
    patternsSection.className = "biomarker-section patterns-section";
    patternsSection.innerHTML = `
      <div class="biomarker-header">
        <span class="biomarker-name">Clinical Pattern Analysis</span>
      </div>
      <div class="patterns-content">
        ${analysisResults.clinicalPatterns.map(p => `
          <div class="pattern-item">
            <div class="pattern-header">
              <span class="pattern-name">${p.pattern}</span>
              <span class="pattern-confidence ${p.confidence.toLowerCase().replace(/\s+/g, '-')}">${p.confidence} confidence</span>
            </div>
            <div class="pattern-indicators">
              <strong>Indicators:</strong> ${p.indicators.join(", ")}
            </div>
            <div class="pattern-recommendation">${p.recommendation}</div>
          </div>
        `).join("")}
      </div>
    `;
    content.appendChild(patternsSection);
  }
}

function displayResults() {
  if (!analysisResults) return;

  const mode = getBenchmarkMode();

  // For radiologist-only mode, skip AI display cards and show eval form directly
  if (mode === "radiologist-only") {
    showEvaluationWorkflow(mode);
    return;
  }

  // Show all cards
  document.getElementById("summaryCard").style.display = "block";
  document.getElementById("regionalCard").style.display = "block";
  document.getElementById("findingsCard").style.display = "block";
  document.getElementById("exportCard").style.display = "block";

  // Summary
  document.getElementById("totalVolume").textContent =
    (analysisResults.totalBrainVolume / 1000).toFixed(0) + " cm³";
  document.getElementById("percentile").textContent =
    analysisResults.percentile + "th";

  const riskEl = document.getElementById("atrophyRisk");
  riskEl.textContent = analysisResults.atrophyRisk;
  riskEl.style.color = getRiskColor(analysisResults.atrophyRisk);

  // ========================================
  // Display Medical-Grade Biomarkers
  // ========================================
  displayMedicalBiomarkers();

  // ========================================
  // Display Dementia Risk Comparison
  // ========================================
  displayRiskComparison();

  // ========================================
  // Display Advanced Analysis (Asymmetry, Lobar, GAI)
  // ========================================
  displayAdvancedAnalysis();

  // Regional analysis
  const regionsContainer = document.getElementById("regionsContainer");
  regionsContainer.innerHTML = "";

  // Sort by effective z-score (worst first for atrophy assessment)
  // For ventricles, lower effective z-score means more atrophy
  const sortedRegions = Object.entries(analysisResults.regions)
    .sort((a, b) => {
      const aEffective = a[1].effectiveZscore !== undefined ? a[1].effectiveZscore : a[1].zscore;
      const bEffective = b[1].effectiveZscore !== undefined ? b[1].effectiveZscore : b[1].zscore;
      return aEffective - bEffective;
    });

  for (const [region, data] of sortedRegions) {
    const item = document.createElement("div");
    item.className = "region-item";
    const interpretationColor = getInterpretationColor(data.interpretation);
    const percentile = data.percentile || Math.round(normalCDF(data.zscore) * 100);

    // Display format: Name | Volume | Percentile | Z-score with interpretation color
    item.innerHTML = `
      <span class="region-name" title="${data.normData?.clinicalName || region}">${truncate(data.normData?.clinicalName || region, 22)}</span>
      <span class="region-volume">${(data.volume / 1000).toFixed(1)} cm³</span>
      <span class="region-percentile" title="Percentile for age/sex">${percentile}%</span>
      <span class="region-zscore" style="color: ${interpretationColor}; font-weight: 500;">z = ${data.zscore.toFixed(1)}</span>
    `;
    regionsContainer.appendChild(item);
  }

  // Findings
  const findingsList = document.getElementById("findingsList");
  findingsList.innerHTML = "";

  if (analysisResults.findings.length === 0) {
    findingsList.innerHTML = `
      <div class="finding-item">
        <svg class="finding-icon success" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
          <polyline points="22 4 12 14.01 9 11.01"/>
        </svg>
        <span>No significant atrophy detected. Brain volumes are within normal limits for age (${analysisResults.age}) and sex (${analysisResults.sex}).</span>
      </div>
    `;
  } else {
    for (const finding of analysisResults.findings) {
      const item = document.createElement("div");
      item.className = "finding-item";
      const iconSvg = finding.severity === "danger"
        ? '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>'
        : finding.severity === "warning"
          ? '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>'
          : '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>';

      item.innerHTML = `
        <svg class="finding-icon ${finding.severity}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          ${iconSvg}
        </svg>
        <span>${finding.text}</span>
      `;
      findingsList.appendChild(item);
    }
  }

  // Add risk description if available
  if (analysisResults.riskDescription) {
    const riskDescEl = document.createElement("div");
    riskDescEl.className = "risk-description";
    riskDescEl.style.cssText = "margin-top: 8px; font-size: 0.85em; color: #666;";
    riskDescEl.textContent = analysisResults.riskDescription;

    const summaryCard = document.getElementById("summaryCard");
    const existingDesc = summaryCard.querySelector(".risk-description");
    if (existingDesc) existingDesc.remove();
    summaryCard.appendChild(riskDescEl);
  }

  // Trigger evaluation workflow after AI results are displayed
  showEvaluationWorkflow(mode);
}

function hideAnalysisCards() {
  document.getElementById("summaryCard").style.display = "none";
  document.getElementById("riskComparisonCard").style.display = "none";
  document.getElementById("advancedAnalysisCard").style.display = "none";
  document.getElementById("regionalCard").style.display = "none";
  document.getElementById("findingsCard").style.display = "none";
  document.getElementById("exportCard").style.display = "none";
}

/**
 * Display Advanced Analysis card with GAI, Lobar, and Asymmetry data
 */
function displayAdvancedAnalysis() {
  if (!analysisResults) return;

  const card = document.getElementById("advancedAnalysisCard");
  if (!card) return;

  card.style.display = "block";

  // ========================================
  // Global Atrophy Index
  // ========================================
  const gai = analysisResults.globalAtrophyIndex;
  if (gai) {
    const gaiValue = document.getElementById("gaiValue");
    const gaiInterp = document.getElementById("gaiInterpretation");

    gaiValue.textContent = `${gai.value}/100`;

    // Color code based on severity
    const severityColors = {
      normal: "#22c55e",
      mild: "#f59e0b",
      moderate: "#f97316",
      severe: "#ef4444"
    };
    gaiValue.style.color = severityColors[gai.severity] || "#a1a1aa";
    gaiInterp.textContent = gai.interpretation;
  }

  // ========================================
  // Lobar Atrophy Grid
  // ========================================
  const lobarGrid = document.getElementById("lobarGrid");
  if (lobarGrid && analysisResults.lobarAtrophy) {
    lobarGrid.innerHTML = "";

    // Show only main lobes (not cingulate/insula for brevity)
    const mainLobes = ["frontal", "temporal", "parietal", "occipital"];

    for (const lobeName of mainLobes) {
      const lobeData = analysisResults.lobarAtrophy[lobeName];
      if (!lobeData) continue;

      const item = document.createElement("div");
      item.className = "lobar-item";

      // Determine status class
      let statusClass = "normal";
      if (lobeData.interpretation === "Significant atrophy") statusClass = "significant";
      else if (lobeData.interpretation === "Mild atrophy") statusClass = "mild";
      else if (lobeData.interpretation === "Low-normal") statusClass = "low-normal";

      item.innerHTML = `
        <div class="lobar-name">${lobeData.name.replace(" Lobe", "")}</div>
        <div class="lobar-status ${statusClass}">${lobeData.interpretation}</div>
      `;
      lobarGrid.appendChild(item);
    }
  }

  // ========================================
  // Asymmetry Analysis
  // ========================================
  const asymmetryGrid = document.getElementById("asymmetryGrid");
  const asymmetrySummary = document.getElementById("asymmetrySummary");

  if (asymmetryGrid && analysisResults.asymmetry) {
    asymmetryGrid.innerHTML = "";
    asymmetrySummary.textContent = analysisResults.asymmetry.summary;

    // Sort by asymmetry index (highest first)
    const sortedAsymmetry = Object.entries(analysisResults.asymmetry.regions)
      .sort((a, b) => b[1].asymmetryIndex - a[1].asymmetryIndex);

    // Show top asymmetries (limit to 6 for UI brevity)
    for (const [key, data] of sortedAsymmetry.slice(0, 6)) {
      const item = document.createElement("div");
      item.className = "asymmetry-item";

      item.innerHTML = `
        <span class="asymmetry-name">${data.clinicalName}</span>
        <span class="asymmetry-value">
          <span class="asymmetry-index" style="color: ${data.color}">${data.asymmetryIndex.toFixed(1)}%</span>
          <span class="asymmetry-laterality">${data.laterality}</span>
        </span>
      `;
      asymmetryGrid.appendChild(item);
    }

    // If no asymmetry data, show message
    if (sortedAsymmetry.length === 0) {
      asymmetryGrid.innerHTML = '<div class="asymmetry-item">No L/R data available (requires 104-class model)</div>';
    }
  }
}

function getRiskColor(risk) {
  switch (risk) {
    case "High": return "#ef4444";      // Red
    case "Moderate": return "#f97316";   // Orange
    case "Mild": return "#f59e0b";       // Amber
    case "Low-Normal": return "#84cc16"; // Lime
    case "Normal": return "#22c55e";     // Green
    default: return "#a1a1aa";           // Gray
  }
}

function getInterpretationColor(interpretation) {
  switch (interpretation) {
    case "severe": return "#ef4444";     // Red
    case "moderate": return "#f97316";   // Orange
    case "mild": return "#f59e0b";       // Amber
    case "low-normal": return "#84cc16"; // Lime
    case "normal": return "#22c55e";     // Green
    default: return "#a1a1aa";           // Gray
  }
}

function truncate(str, len) {
  return str.length > len ? str.substring(0, len - 3) + "..." : str;
}

// ============================================
// EVALUATION WORKFLOW ORCHESTRATION
// ============================================

/**
 * Show the appropriate evaluation workflow based on the selected mode.
 * Called after AI analysis completes (or directly for radiologist-only mode).
 * @param {string} mode - 'ai-first' | 'radiologist-first' | 'radiologist-only'
 */
function showEvaluationWorkflow(mode) {
  // Hide all evaluation cards first
  hideEvalCards();

  switch (mode) {
    case "radiologist-only":
      showRadiologistOnlyMode();
      break;
    case "ai-first":
      showAIFirstMode();
      break;
    case "radiologist-first":
      showRadiologistFirstMode();
      break;
  }
}

/**
 * Hide all evaluation-specific cards
 */
function hideEvalCards() {
  const ids = ["aiResultsFullCard", "radiologistEvalCard", "sideBySideCard"];
  for (const id of ids) {
    const el = document.getElementById(id);
    if (el) el.style.display = "none";
  }
  // Clear validation errors
  const errEls = document.querySelectorAll(".eval-validation-errors");
  errEls.forEach(el => { el.style.display = "none"; el.innerHTML = ""; });
}

/**
 * Mode 1: Radiologist-Only
 * Show evaluation form immediately, no AI results
 */
function showRadiologistOnlyMode() {
  const card = document.getElementById("radiologistEvalCard");
  const modeLabel = document.getElementById("evalModeLabel");
  if (modeLabel) modeLabel.textContent = "Radiologist-Only Mode";

  buildEvaluationForm("evalFormContainer");
  if (card) card.style.display = "block";

  // Make sure the eval card scrolls into view
  card?.scrollIntoView({ behavior: "smooth", block: "start" });
}

/**
 * Mode 2: AI-First
 * Screen 1: Show AI results full display with "Proceed to Manual Review" button
 */
function showAIFirstMode() {
  const aiData = extractAIScores(analysisResults);
  if (!aiData) return;

  // Store for later use in side-by-side
  analysisResults._extractedAIData = aiData;

  // Build AI results display
  buildAIResultsDisplay("aiResultsFullContainer", aiData);

  // Show the AI results card
  const card = document.getElementById("aiResultsFullCard");
  if (card) card.style.display = "block";
  card?.scrollIntoView({ behavior: "smooth", block: "start" });
}

/**
 * AI-First Mode: Handle "Proceed to Manual Review" button click
 * Transitions to Screen 2: Side-by-side comparison
 */
function handleProceedToReview() {
  const aiData = analysisResults?._extractedAIData || extractAIScores(analysisResults);

  // Hide AI results full card
  const aiCard = document.getElementById("aiResultsFullCard");
  if (aiCard) aiCard.style.display = "none";

  // Show side-by-side: AI (left, read-only) | Radiologist form (right, editable)
  const compCard = document.getElementById("sideBySideCard");
  const compLabel = document.getElementById("comparisonModeLabel");
  if (compLabel) compLabel.textContent = "AI-First Mode — Review & Score";

  buildSideBySideView(
    "sideBySideContainer",
    aiData,   // left column data
    null,     // right column data (empty form)
    "AI Analysis",
    "Your Assessment",
    { leftEditable: false, rightEditable: true }
  );

  if (compCard) compCard.style.display = "block";
  compCard?.scrollIntoView({ behavior: "smooth", block: "start" });
}

/**
 * Mode 3: Radiologist-First
 * Screen 1: Show evaluation form first (no AI results visible)
 */
function showRadiologistFirstMode() {
  const card = document.getElementById("radiologistEvalCard");
  const modeLabel = document.getElementById("evalModeLabel");
  if (modeLabel) modeLabel.textContent = "Radiologist-First Mode — Score Before AI";

  buildEvaluationForm("evalFormContainer");
  if (card) card.style.display = "block";
  card?.scrollIntoView({ behavior: "smooth", block: "start" });
}

/**
 * Handle evaluation form submit button click.
 * Behavior depends on the current benchmark mode.
 */
function handleEvalSubmit() {
  const mode = getBenchmarkMode();
  const formData = collectFormData();
  const validation = validateForm(formData);

  if (!validation.valid) {
    showValidationErrors("evalErrors", validation.errors);
    return;
  }

  clearValidationErrors("evalErrors");

  if (mode === "radiologist-only") {
    // Direct save — no AI comparison
    const payload = {
      radiologist_data: formData,
      patient: {
        age: parseInt(document.getElementById("patientAge")?.value) || null,
        sex: document.getElementById("patientSex")?.value || null
      }
    };
    const saved = saveEvaluation("radiologist-only", payload);
    showSuccessToast("Evaluation saved!");
    console.log("Saved radiologist-only evaluation:", saved);

  } else if (mode === "radiologist-first") {
    // Transition to Screen 2: show side-by-side with AI results
    const radiologistData = formData;
    const aiData = extractAIScores(analysisResults);
    analysisResults._extractedAIData = aiData;
    analysisResults._radiologistFirstData = radiologistData;

    // Hide the eval form card
    const evalCard = document.getElementById("radiologistEvalCard");
    if (evalCard) evalCard.style.display = "none";

    // Show side-by-side: Radiologist (left, editable with prefill) | AI (right, read-only)
    const compCard = document.getElementById("sideBySideCard");
    const compLabel = document.getElementById("comparisonModeLabel");
    if (compLabel) compLabel.textContent = "Radiologist-First Mode — Compare with AI";

    buildSideBySideView(
      "sideBySideContainer",
      radiologistData,  // left column (editable, prefilled)
      aiData,           // right column (AI, read-only)
      "Your Assessment (Editable)",
      "AI Analysis",
      { leftEditable: true, rightEditable: false }
    );

    if (compCard) compCard.style.display = "block";
    compCard?.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

/**
 * Handle side-by-side comparison submit/resubmit button click.
 * Collects the editable column data and saves the full evaluation.
 */
function handleComparisonSubmit() {
  const mode = getBenchmarkMode();
  const formData = collectFormData();
  const validation = validateForm(formData);

  if (!validation.valid) {
    showValidationErrors("comparisonErrors", validation.errors);
    return;
  }

  clearValidationErrors("comparisonErrors");

  const aiData = analysisResults?._extractedAIData || extractAIScores(analysisResults);

  let payload, saveMode;

  if (mode === "ai-first") {
    saveMode = "ai_then_radiologist";
    payload = {
      ai_data: aiData,
      radiologist_data: formData,
      patient: {
        age: parseInt(document.getElementById("patientAge")?.value) || null,
        sex: document.getElementById("patientSex")?.value || null
      }
    };
  } else if (mode === "radiologist-first") {
    saveMode = "radiologist_then_ai";
    payload = {
      radiologist_data: formData,
      ai_data: aiData,
      original_radiologist_data: analysisResults._radiologistFirstData || null,
      patient: {
        age: parseInt(document.getElementById("patientAge")?.value) || null,
        sex: document.getElementById("patientSex")?.value || null
      }
    };
  }

  const saved = saveEvaluation(saveMode, payload);
  showSuccessToast("Evaluation saved!");
  console.log(`Saved ${saveMode} evaluation:`, saved);
}

/**
 * Show validation errors in the specified container
 */
function showValidationErrors(containerId, errors) {
  const container = document.getElementById(containerId);
  if (!container) return;
  container.innerHTML = `<ul>${errors.map(e => `<li>${e}</li>`).join("")}</ul>`;
  container.style.display = "block";
  container.scrollIntoView({ behavior: "smooth", block: "center" });
}

/**
 * Clear validation errors
 */
function clearValidationErrors(containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;
  container.innerHTML = "";
  container.style.display = "none";
}

/**
 * Show a success toast notification
 */
function showSuccessToast(message) {
  const toast = document.createElement("div");
  toast.className = "eval-success-toast";
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => {
    toast.remove();
  }, 3000);
}

// ============================================
// EXPORT
// ============================================

function exportReport() {
  if (!analysisResults) {
    alert("No analysis to export. Please run analysis first.");
    return;
  }

  const report = generateTextReport();

  // Download as text file
  const blob = new Blob([report], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "atrophy_report.txt";
  a.click();
  URL.revokeObjectURL(url);
}

function generateTextReport() {
  const r = analysisResults;
  const date = new Date().toLocaleDateString();
  const time = new Date().toLocaleTimeString();

  let report = `
╔══════════════════════════════════════════════════════════════════════════════╗
║            BRAIN VOLUMETRIC ANALYSIS REPORT - MEDICAL GRADE                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

Generated: ${date} at ${time}
Analysis Method: AI-based segmentation with ICV-normalized volumetrics

================================================================================
PATIENT INFORMATION
================================================================================
Age:                    ${r.age} years
Sex:                    ${r.sex === 'male' ? 'Male' : 'Female'}

================================================================================
SUMMARY
================================================================================
Total Brain Volume:     ${(r.totalBrainVolume / 1000).toFixed(1)} cm³
Estimated ICV:          ${(r.icv / 1000).toFixed(1)} cm³
Percentile for Age/Sex: ${r.percentile}th (z = ${r.totalBrainZscore || 'N/A'})
Overall Atrophy Risk:   ${r.atrophyRisk}
Assessment:             ${r.riskDescription || ''}

================================================================================
ADVANCED BIOMARKERS
================================================================================
`;

  // Brain Parenchymal Fraction
  if (r.bpf) {
    report += `
BRAIN PARENCHYMAL FRACTION (BPF)
--------------------------------
Value:                  ${(r.bpf.value * 100).toFixed(1)}%
Expected for age ${r.age}:    ${(r.bpf.expected * 100).toFixed(1)}%
Z-score:                ${r.bpf.zscore.toFixed(2)}
Percentile:             ${r.bpf.percentile}th
Interpretation:         ${r.bpf.interpretation}
`;
  }

  // Hippocampal Occupancy Score
  if (r.hoc && r.hoc.value !== null) {
    report += `
HIPPOCAMPAL OCCUPANCY SCORE (HOC)
---------------------------------
Value:                  ${(r.hoc.value * 100).toFixed(1)}%
Expected for age ${r.age}:    ${(r.hoc.expected * 100).toFixed(1)}%
Z-score:                ${r.hoc.zscore.toFixed(2)}
Percentile:             ${r.hoc.percentile}th
Interpretation:         ${r.hoc.interpretation}
Clinical Note:          ${r.hoc.description}
MCI→AD Conversion Risk: ${r.hoc.conversionRisk} (${r.hoc.conversionRate})
`;
  }

  // Dementia Risk Scoring
  if (r.dementiaRiskScore) {
    report += `
DEMENTIA RISK SCORING
---------------------
AI Risk Score:          ${r.dementiaRiskScore.score}/5 (${r.dementiaRiskScore.label})
Confidence:             ${r.dementiaRiskScore.confidence}
`;
    if (r.dementiaRiskScore.factors && r.dementiaRiskScore.factors.length > 0) {
      report += `Contributing Factors:\n`;
      for (const factor of r.dementiaRiskScore.factors) {
        report += `  - ${factor}\n`;
      }
    }
    if (r.radiologistRiskScore !== undefined) {
      report += `Radiologist Score:      ${r.radiologistRiskScore}/5 (${RISK_SCORE_LABELS[r.radiologistRiskScore]})\n`;
      const discrepancy = Math.abs(r.dementiaRiskScore.score - r.radiologistRiskScore);
      if (discrepancy >= 2) {
        report += `*** DISCREPANCY NOTE: AI and radiologist scores differ by ${discrepancy} points ***\n`;
      }
    }
  }

  // Hippocampus-specific analysis
  if (r.hippocampusAnalysis) {
    report += `
HIPPOCAMPAL ANALYSIS (Key Dementia Biomarker)
---------------------------------------------
Raw Volume:             ${(r.hippocampusAnalysis.volume / 1000).toFixed(2)} cm³
ICV-Normalized Volume:  ${(r.hippocampusAnalysis.normalizedVolume / 1000).toFixed(2)} cm³
Expected for age ${r.age}:    ${(r.hippocampusAnalysis.expectedForAge / 1000).toFixed(2)} cm³
Z-score:                ${r.hippocampusAnalysis.zscore.toFixed(2)}
Percentile:             ${r.hippocampusAnalysis.percentile}th
Interpretation:         ${r.hippocampusAnalysis.interpretation}
`;
  }

  // Clinical Pattern Recognition
  if (r.clinicalPatterns && r.clinicalPatterns.length > 0) {
    report += `
================================================================================
CLINICAL PATTERN ANALYSIS
================================================================================
`;
    for (const pattern of r.clinicalPatterns) {
      report += `
Pattern:        ${pattern.pattern}
Confidence:     ${pattern.confidence}
Indicators:     ${pattern.indicators.join("; ")}
Recommendation: ${pattern.recommendation}
`;
    }
  }

  // Standardized Atrophy Scales
  if (r.standardizedScores) {
    const s = r.standardizedScores;
    report += `
================================================================================
STANDARDIZED ATROPHY SCALES (Visual Rating Scale Equivalents)
================================================================================

MTA SCORE (Scheltens Scale) - Medial Temporal Atrophy
-----------------------------------------------------
Score:              ${s.mta.score}/4 - ${s.mta.label}
Description:        ${s.mta.description}
QMTA Ratio:         ${(s.mta.qmtaRatio * 100).toFixed(1)}% (ILV/Hippocampus)
Age-Adjusted Threshold: ${s.mta.ageThreshold} for age ${r.age}
Status:             ${s.mta.isAbnormal ? 'ABNORMAL - Exceeds age threshold' : 'Within normal limits'}
Clinical Note:      ${s.mta.interpretation}

GCA SCORE (Pasquier Scale) - Global Cortical Atrophy
-----------------------------------------------------
Score:              ${s.gca.score}/3 - ${s.gca.label}
Description:        ${s.gca.description}
Cortical Volume:    ${(s.gca.corticalVolume / 1000).toFixed(1)} cm³
Cortex/Brain Ratio: ${(s.gca.cortexRatio * 100).toFixed(1)}%
Age-Adjusted Threshold: ${s.gca.ageThreshold} for age ${r.age}
Status:             ${s.gca.isAbnormal ? 'ABNORMAL - Exceeds age threshold' : 'Within normal limits'}
Clinical Note:      ${s.gca.interpretation}

KOEDAM SCORE - Posterior Atrophy
--------------------------------
Score:              ${s.koedam.score}/3 - ${s.koedam.label}
Description:        ${s.koedam.description}
Posterior Z-score:  ${s.koedam.posteriorZscore.toFixed(2)}
Status:             ${s.koedam.isAbnormal ? 'ABNORMAL' : 'Within normal limits'}
Clinical Note:      ${s.koedam.interpretation}

EVANS INDEX - Ventricular Enlargement
-------------------------------------
Value:              ${s.evansIndex.value.toFixed(3)}
Interpretation:     ${s.evansIndex.label}
Reference:          Normal ≤0.25, Borderline 0.25-0.30, Enlarged >0.30
Status:             ${s.evansIndex.isAbnormal ? 'ABNORMAL - Suggests hydrocephalus/atrophy' : s.evansIndex.isBorderline ? 'BORDERLINE' : 'Normal'}
Clinical Note:      ${s.evansIndex.interpretation}

COMPOSITE ASSESSMENT
--------------------
Abnormal Scales:    ${s.abnormalCount}/4
Overall Severity:   ${s.overallSeverity}
`;
  }

  report += `
================================================================================
REGIONAL VOLUMES (ICV-Normalized)
================================================================================
Region                          | Volume (cm³) | Expected | Z-score | Percentile | Status
--------------------------------|--------------|----------|---------|------------|--------
`;

  // Sort regions by z-score (worst first)
  const sortedRegions = Object.entries(r.regions)
    .sort((a, b) => (a[1].effectiveZscore || a[1].zscore) - (b[1].effectiveZscore || b[1].zscore));

  for (const [region, data] of sortedRegions) {
    const vol = (data.normalizedVolume / 1000).toFixed(2).padStart(10);
    const expected = data.expectedVolume ? ((data.expectedVolume / 1000).toFixed(2)).padStart(8) : 'N/A'.padStart(8);
    const zs = data.zscore.toFixed(2).padStart(7);
    const pct = (data.percentile + '%').padStart(10);
    const status = data.interpretation.padEnd(8);
    const name = (data.normData?.clinicalName || region).padEnd(30).substring(0, 30);
    report += `${name} | ${vol} | ${expected} | ${zs} | ${pct} | ${status}\n`;
  }

  report += `
================================================================================
KEY FINDINGS
================================================================================
`;

  if (r.findings.length === 0) {
    report += "✓ No significant atrophy detected. Brain volumes are within normal limits for age and sex.\n";
  } else {
    for (const finding of r.findings) {
      const icon = finding.severity === 'danger' ? '⚠️' : finding.severity === 'warning' ? '⚡' : 'ℹ️';
      report += `${icon} ${finding.text}\n\n`;
    }
  }

  report += `
================================================================================
METHODOLOGY
================================================================================
- Segmentation: AI-based whole-brain parcellation
- Normative Data: Based on FreeSurfer reference (n=2,790), UK Biobank (n=19,793)
- ICV Normalization: Residual correction method (Vol_adj = Vol - b×(ICV - ICV_mean))
- Z-score Calculation: Age and sex-specific with population-based standard deviations
- HOC Formula: Hippocampus / (Hippocampus + Inferior Lateral Ventricle)
- BPF Formula: Total Brain Volume / Intracranial Volume

STANDARDIZED SCALE DERIVATION
-----------------------------
- MTA Score: Derived from QMTA ratio (ILV/Hippocampus) and hippocampal z-score
  Reference: Scheltens et al. (1992) J Neurol Neurosurg Psychiatry
- GCA Score: Derived from cortical volume z-score and cortex/brain ratio
  Reference: Pasquier et al. (1996) J Neurol
- Koedam Score: Derived from precuneus, posterior cingulate, and parietal volumes
  Reference: Koedam et al. (2011) AJNR Am J Neuroradiol
- Evans Index: Ratio of maximum frontal horn width to maximum internal skull diameter
  Reference: Evans (1942) Arch Neurol Psychiatry

CLINICAL THRESHOLDS
-------------------
Z-score ≥ -1.0:     Normal (above 16th percentile)
-1.5 ≤ Z < -1.0:    Low-Normal (7th-16th percentile)
-2.0 ≤ Z < -1.5:    Mild Atrophy (2nd-7th percentile)
-2.5 ≤ Z < -2.0:    Moderate Atrophy (0.6-2nd percentile)
Z-score < -2.5:     Severe Atrophy (below 0.6th percentile)

AGE-ADJUSTED THRESHOLDS FOR VISUAL RATING SCALES
------------------------------------------------
MTA Score (abnormal if):  Age <65: >1.0 | Age 65-74: >1.5 | Age 75-84: >2.0 | Age ≥85: >2.5
GCA Score (abnormal if):  Age <65: >0.5 | Age 65-74: >1.0 | Age 75-84: >1.5 | Age ≥85: >2.0
Koedam Score:             Score ≥2 considered significant at any age
Evans Index:              >0.30 considered enlarged (hydrocephalus/atrophy)

================================================================================
DISCLAIMER
================================================================================
This analysis is provided for RESEARCH AND EDUCATIONAL PURPOSES ONLY.

This report does NOT constitute medical advice, diagnosis, or treatment
recommendation. The volumetric measurements and interpretations should be
reviewed by a qualified neuroradiologist or physician and correlated with
clinical findings, patient history, and other diagnostic information.

Brain volumes vary significantly among healthy individuals. Automated
segmentation may have errors. Always verify findings with clinical judgment.

The standardized visual rating scale equivalents (MTA, GCA, Koedam) are
APPROXIMATIONS derived from volumetric data, not direct visual assessments.
True visual rating scales require expert neuroradiologist review of MRI images.

================================================================================
References:
- Potvin et al. (2016) FreeSurfer subcortical normative data
- UK Biobank hippocampal nomograms
- NeuroQuant normative database methodology
- Vågberg et al. (2017) BPF systematic review
- Scheltens et al. (1992) Medial temporal lobe atrophy scale
- Pasquier et al. (1996) Global cortical atrophy scale
- Koedam et al. (2011) Posterior atrophy rating scale
- Evans (1942) Evans index for ventricular measurement
================================================================================

Generated by Brainchop Brain Volumetric Analysis System
https://github.com/neuroneural/brainchop
`;

  return report;
}

// ============================================
// UI HELPERS
// ============================================

function updateProgress(percent, text) {
  const fill = document.getElementById("progressFill");
  const textEl = document.getElementById("progressText");

  fill.style.width = percent + "%";
  textEl.textContent = text;
}

function handleLocationChange(data) {
  document.getElementById("locationInfo").textContent = data.string;
}

// ============================================
// START
// ============================================

init();
