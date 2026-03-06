/**
 * Radiologist Evaluation Module
 * 
 * Handles the radiologist evaluation form for three workflow modes:
 * - Mode 1: Radiologist-Only (No AI)
 * - Mode 2: AI-First (AI → Review)
 * - Mode 3: Radiologist-First (Manual → AI)
 */

// ============================================
// DATA CONSTANTS
// ============================================

export const ATROPHY_LEVELS = [
    { value: "", label: "Select..." },
    { value: "none", label: "None" },
    { value: "mild", label: "Mild" },
    { value: "moderate", label: "Moderate" },
    { value: "high", label: "High" }
];

export const BRAIN_REGIONS = [
    // Ventricular
    { id: "lateral-ventricles", name: "Lateral Ventricles", group: "Ventricular", key: "Lateral-Ventricle" },
    { id: "third-ventricle", name: "Third Ventricle", group: "Ventricular", key: "3rd-Ventricle" },
    { id: "temporal-horns", name: "Temporal Horns of the Lateral Ventricles", group: "Ventricular", key: "Inferior-Lateral-Ventricle" },
    { id: "fourth-ventricle", name: "Fourth Ventricle", group: "Ventricular", key: "4th-Ventricle" },

    // Subcortical
    { id: "thalami", name: "Thalami (Bilateral)", group: "Subcortical", key: "Thalamus" },
    { id: "caudate-nuclei", name: "Caudate Nuclei", group: "Subcortical", key: "Caudate" },
    { id: "putamina", name: "Putamina (Bilateral)", group: "Subcortical", key: "Putamen" },
    { id: "globi-pallidi", name: "Globi Pallidi", group: "Subcortical", key: "Pallidum" },
    { id: "hippocampi", name: "Hippocampi (Bilateral)", group: "Subcortical", key: "Hippocampus" },
    { id: "amygdalae", name: "Amygdalae (Bilateral)", group: "Subcortical", key: "Amygdala" },
    { id: "nuclei-accumbentes", name: "Nuclei Accumbentes", group: "Subcortical", key: "Accumbens-area" },
    { id: "ventral-diencephalon", name: "Ventral Diencephalon", group: "Subcortical", key: "VentralDC" },

    // Cortical & White Matter
    { id: "cerebral-cortex", name: "Cerebral Cortex (Gray Matter)", group: "Cortical", key: "Cerebral-Cortex" },
    { id: "supratentorial-wm", name: "Supratentorial White Matter", group: "Cortical", key: "Cerebral-White-Matter" },
    { id: "brainstem", name: "Brainstem", group: "Cortical", key: "Brain-Stem" },

    // Cerebellar
    { id: "cerebellar-cortex", name: "Cerebellar Cortex", group: "Cerebellar", key: "Cerebellum-Cortex" },
    { id: "cerebellar-wm", name: "Cerebellar White Matter", group: "Cerebellar", key: "Cerebellum-White-Matter" }
];

// Note: 17 regions listed. The spec says 18 but Brainstem is counted separately from cortical.
// We have exactly the 18 listed in the spec minus one that maps to the same structure.
// The 18th region was listed twice (Brainstem appears in the spec list).

export const LOBES = [
    { id: "frontal", name: "Frontal" },
    { id: "temporal", name: "Temporal" },
    { id: "parietal", name: "Parietal" },
    { id: "occipital", name: "Occipital" }
];

/**
 * Map normative/bilateral region keys to 104-class (Aparc+Aseg) label names.
 * Used so extractAIScores can resolve regional atrophy from left/right segments.
 */
const REGION_KEY_TO_104_LABELS = {
    "Lateral-Ventricle": ["Left-Lateral-Ventricle", "Right-Lateral-Ventricle"],
    "3rd-Ventricle": ["3rd-Ventricle"],
    "Inferior-Lateral-Ventricle": ["Left-Inf-Lat-Vent", "Right-Inf-Lat-Vent"],
    "4th-Ventricle": ["4th-Ventricle"],
    "Thalamus": ["Left-Thalamus-Proper*", "Right-Thalamus-Proper*"],
    "Caudate": ["Left-Caudate", "Right-Caudate"],
    "Putamen": ["Left-Putamen", "Right-Putamen"],
    "Pallidum": ["Left-Pallidum", "Right-Pallidum"],
    "Hippocampus": ["Left-Hippocampus", "Right-Hippocampus"],
    "Amygdala": ["Left-Amygdala", "Right-Amygdala"],
    "Accumbens-area": ["Left-Accumbens-area", "Right-Accumbens-area"],
    "VentralDC": ["Left-VentralDC", "Right-VentralDC"],
    "Cerebral-Cortex": [], // 104 has many ctx-*; no single key
    "Cerebral-White-Matter": ["Left-Cerebral-White-Matter", "Right-Cerebral-White-Matter"],
    "Brain-Stem": ["Brain-Stem"],
    "Cerebellum-Cortex": ["Left-Cerebellum-Cortex", "Right-Cerebellum-Cortex"],
    "Cerebellum-White-Matter": ["Left-Cerebellum-White-Matter", "Right-Cerebellum-White-Matter"]
};

export const GLOBAL_SCORES = [
    { id: "mta", name: "MTA", fullName: "Medial Temporal Atrophy", min: 0, max: 4, step: 1, type: "integer" },
    { id: "pa", name: "PA", fullName: "Parietal Atrophy (Koedam)", min: 0, max: 3, step: 1, type: "integer" },
    { id: "gca", name: "GCA", fullName: "Global Cortical Atrophy", min: 0, max: 3, step: 1, type: "integer" },
    { id: "evans-index", name: "Evans Index", fullName: "Evans Index", min: 0.10, max: 0.50, step: 0.01, type: "float" }
];

// ============================================
// FORM BUILDING
// ============================================

/**
 * Build the full radiologist evaluation form
 * @param {string} containerId - DOM id to render the form into
 * @param {Object} options - { editable: true, prefill: null }
 */
export function buildEvaluationForm(containerId, options = {}) {
    const { editable = true, prefill = null } = options;
    const container = document.getElementById(containerId);
    if (!container) return;

    const disabled = editable ? "" : "disabled";

    let html = `<div class="eval-form" id="evalFormInner">`;

    // Global Scores section
    html += `
    <div class="eval-section">
      <h4 class="eval-section-title">Global Clinical Scores</h4>
      <div class="eval-grid">`;

    for (const score of GLOBAL_SCORES) {
        const val = prefill?.globalScores?.[score.id] ?? "";
        html += `
        <div class="eval-field">
          <label class="eval-label" for="eval-${score.id}">
            <span class="eval-label-abbr">${score.name}</span>
            <span class="eval-label-full">${score.fullName}</span>
          </label>
          <input type="number" id="eval-${score.id}" class="eval-input"
            min="${score.min}" max="${score.max}" step="${score.step}"
            placeholder="${score.min}–${score.max}" value="${val}" ${disabled}>
        </div>`;
    }

    html += `
      </div>
    </div>`;

    // Lobar Atrophy section
    html += `
    <div class="eval-section">
      <h4 class="eval-section-title">Lobar Atrophy</h4>
      <div class="eval-grid">`;

    for (const lobe of LOBES) {
        const val = prefill?.lobarAtrophy?.[lobe.id] ?? "";
        html += `
        <div class="eval-field">
          <label class="eval-label" for="eval-lobe-${lobe.id}">${lobe.name}</label>
          <select id="eval-lobe-${lobe.id}" class="eval-dropdown" ${disabled}>
            ${ATROPHY_LEVELS.map(l => `<option value="${l.value}" ${l.value === val ? "selected" : ""}>${l.label}</option>`).join("")}
          </select>
        </div>`;
    }

    html += `
      </div>
    </div>`;

    // Regional Atrophy section - grouped
    const groups = {};
    for (const region of BRAIN_REGIONS) {
        if (!groups[region.group]) groups[region.group] = [];
        groups[region.group].push(region);
    }

    html += `
    <div class="eval-section">
      <h4 class="eval-section-title">Regional Atrophy (18 Brain Structures)</h4>`;

    for (const [groupName, regions] of Object.entries(groups)) {
        html += `
      <div class="eval-region-group">
        <span class="eval-group-label">${groupName}</span>
        <div class="eval-grid">`;

        for (const region of regions) {
            const val = prefill?.regionalAtrophy?.[region.id] ?? "";
            html += `
          <div class="eval-field">
            <label class="eval-label" for="eval-region-${region.id}" title="${region.name}">${region.name}</label>
            <select id="eval-region-${region.id}" class="eval-dropdown" ${disabled}>
              ${ATROPHY_LEVELS.map(l => `<option value="${l.value}" ${l.value === val ? "selected" : ""}>${l.label}</option>`).join("")}
            </select>
          </div>`;
        }

        html += `
        </div>
      </div>`;
    }

    html += `
    </div>`;

    // Scan Number
    const scanVal = prefill?.scanNumber ?? "";
    html += `
    <div class="eval-section">
      <h4 class="eval-section-title">Metadata</h4>
      <div class="eval-grid eval-grid-single">
        <div class="eval-field">
          <label class="eval-label" for="eval-scan-number">Scan Number</label>
          <input type="number" id="eval-scan-number" class="eval-input"
            min="1" step="1" placeholder="1, 2, 3..." value="${scanVal}" ${disabled}>
        </div>
      </div>
    </div>`;

    html += `</div>`;
    container.innerHTML = html;
}

// ============================================
// AI RESULTS DISPLAY
// ============================================

/**
 * Build a read-only display of AI results in the same format as the evaluation form
 * @param {string} containerId - DOM element id
 * @param {Object} aiData - Extracted AI scores
 */
export function buildAIResultsDisplay(containerId, aiData) {
    const container = document.getElementById(containerId);
    if (!container || !aiData) return;

    let html = `<div class="eval-results-display">`;

    // Global Scores
    html += `
    <div class="eval-section">
      <h4 class="eval-section-title">Global Clinical Scores <span class="eval-ai-badge">AI</span></h4>
      <div class="eval-results-grid">`;

    for (const score of GLOBAL_SCORES) {
        const val = aiData.globalScores?.[score.id];
        const displayVal = val !== null && val !== undefined ? (score.type === "float" ? val.toFixed(2) : val) : "—";
        html += `
        <div class="eval-result-item">
          <span class="eval-result-label">${score.name}</span>
          <span class="eval-result-value">${displayVal}</span>
        </div>`;
    }

    html += `
      </div>
    </div>`;

    // Lobar Atrophy
    html += `
    <div class="eval-section">
      <h4 class="eval-section-title">Lobar Atrophy <span class="eval-ai-badge">AI</span></h4>
      <div class="eval-results-grid">`;

    for (const lobe of LOBES) {
        const val = aiData.lobarAtrophy?.[lobe.id] ?? "—";
        const label = ATROPHY_LEVELS.find(l => l.value === val)?.label ?? capitalize(val);
        html += `
        <div class="eval-result-item">
          <span class="eval-result-label">${lobe.name}</span>
          <span class="eval-result-value eval-atrophy-${val}">${label}</span>
        </div>`;
    }

    html += `
      </div>
    </div>`;

    // Regional Atrophy - grouped
    const groups = {};
    for (const region of BRAIN_REGIONS) {
        if (!groups[region.group]) groups[region.group] = [];
        groups[region.group].push(region);
    }

    html += `
    <div class="eval-section">
      <h4 class="eval-section-title">Regional Atrophy <span class="eval-ai-badge">AI</span></h4>`;

    for (const [groupName, regions] of Object.entries(groups)) {
        html += `
      <div class="eval-region-group">
        <span class="eval-group-label">${groupName}</span>
        <div class="eval-results-grid">`;

        for (const region of regions) {
            const val = aiData.regionalAtrophy?.[region.id] ?? "—";
            const label = ATROPHY_LEVELS.find(l => l.value === val)?.label ?? capitalize(val);
            html += `
          <div class="eval-result-item">
            <span class="eval-result-label" title="${region.name}">${region.name}</span>
            <span class="eval-result-value eval-atrophy-${val}">${label}</span>
          </div>`;
        }

        html += `
        </div>
      </div>`;
    }

    html += `
    </div>`;

    html += `</div>`;
    container.innerHTML = html;
}

// ============================================
// SIDE-BY-SIDE COMPARISON
// ============================================

/**
 * Build a side-by-side comparison view
 * @param {string} containerId - DOM element id
 * @param {Object} leftData - Data for left column
 * @param {Object} rightData - Data for right column
 * @param {string} leftLabel - e.g. "AI Analysis" or "Your Assessment"
 * @param {string} rightLabel - e.g. "Your Assessment" or "AI Analysis"
 * @param {Object} options - { leftEditable: false, rightEditable: false, rightSubtitle?: string }
 */
export function buildSideBySideView(containerId, leftData, rightData, leftLabel, rightLabel, options = {}) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const rightSubtitle = options.rightSubtitle ? `<span class="side-by-side-subtitle">${options.rightSubtitle}</span>` : "";
    const html = `
    <div class="side-by-side-container">
      <div class="side-by-side-column" id="sideBySideLeft">
        <div class="side-by-side-header">
          <h4>${leftLabel}</h4>
        </div>
        <div class="side-by-side-body" id="sideBySideLeftBody"></div>
      </div>
      <div class="side-by-side-column side-by-side-editable" id="sideBySideRight">
        <div class="side-by-side-header">
          <h4>${rightLabel}</h4>
          ${rightSubtitle}
        </div>
        <div class="side-by-side-body" id="sideBySideRightBody"></div>
      </div>
    </div>`;

    container.innerHTML = html;

    // Render each column
    if (options.leftEditable) {
        buildEvaluationForm("sideBySideLeftBody", { editable: true, prefill: leftData });
    } else {
        buildAIResultsDisplay("sideBySideLeftBody", leftData);
    }

    if (options.rightEditable) {
        buildEvaluationForm("sideBySideRightBody", { editable: true, prefill: rightData });
    } else {
        buildAIResultsDisplay("sideBySideRightBody", rightData);
    }
}

// ============================================
// FORM DATA COLLECTION
// ============================================

/**
 * Collect all form field values into a structured object
 * @returns {Object} Radiologist evaluation data
 */
export function collectFormData() {
    const data = {
        globalScores: {},
        lobarAtrophy: {},
        regionalAtrophy: {},
        scanNumber: null
    };

    // Global scores
    for (const score of GLOBAL_SCORES) {
        const el = document.getElementById(`eval-${score.id}`);
        if (el && el.value !== "") {
            data.globalScores[score.id] = score.type === "float" ? parseFloat(el.value) : parseInt(el.value, 10);
        } else {
            data.globalScores[score.id] = null;
        }
    }

    // Lobar atrophy
    for (const lobe of LOBES) {
        const el = document.getElementById(`eval-lobe-${lobe.id}`);
        data.lobarAtrophy[lobe.id] = el?.value || null;
    }

    // Regional atrophy
    for (const region of BRAIN_REGIONS) {
        const el = document.getElementById(`eval-region-${region.id}`);
        data.regionalAtrophy[region.id] = el?.value || null;
    }

    // Scan number
    const scanEl = document.getElementById("eval-scan-number");
    if (scanEl && scanEl.value !== "") {
        data.scanNumber = parseInt(scanEl.value, 10);
    }

    return data;
}

/**
 * Collect form data from the side-by-side editable column
 * Looks for fields inside #sideBySideLeftBody or #sideBySideRightBody
 */
export function collectSideBySideFormData(columnId = "sideBySideLeftBody") {
    // The form fields are rendered inside the specified column
    // Since IDs are global, collectFormData() will find them regardless of container
    return collectFormData();
}

// ============================================
// VALIDATION
// ============================================

/**
 * Validate the evaluation form data
 * @param {Object} data - From collectFormData()
 * @returns {Object} { valid: boolean, errors: string[] }
 */
export function validateForm(data) {
    const errors = [];

    // Global scores validation
    for (const score of GLOBAL_SCORES) {
        const val = data.globalScores[score.id];
        if (val === null || val === undefined) {
            errors.push(`${score.fullName} is required`);
        } else if (val < score.min || val > score.max) {
            errors.push(`${score.fullName} must be between ${score.min} and ${score.max}`);
        }
    }

    // Lobar atrophy - all required
    for (const lobe of LOBES) {
        if (!data.lobarAtrophy[lobe.id]) {
            errors.push(`${lobe.name} lobe atrophy rating is required`);
        }
    }

    // Regional atrophy - all required
    for (const region of BRAIN_REGIONS) {
        if (!data.regionalAtrophy[region.id]) {
            errors.push(`${region.name} atrophy rating is required`);
        }
    }

    // Scan number
    if (!data.scanNumber || data.scanNumber < 1) {
        errors.push("Scan number must be 1 or greater");
    }

    return { valid: errors.length === 0, errors };
}

// ============================================
// AI DATA EXTRACTION
// ============================================

/**
 * Extract AI scores from analysisResults into the same format as radiologist data
 * Maps z-scores to None/Mild/Moderate/High
 * @param {Object} analysisResults - The global analysis results object
 * @returns {Object} AI scores in the same shape as radiologist data
 */
export function extractAIScores(analysisResults) {
    if (!analysisResults) return null;

    const aiData = {
        globalScores: {},
        lobarAtrophy: {},
        regionalAtrophy: {}
    };

    // Global scores from standardized scales
    const scales = analysisResults.standardizedScores;
    if (scales) {
        aiData.globalScores.mta = scales.mta?.score ?? null;
        aiData.globalScores.pa = scales.koedam?.score ?? null;
        aiData.globalScores.gca = scales.gca?.score ?? null;
        aiData.globalScores["evans-index"] = scales.evansIndex?.value
            ? parseFloat(scales.evansIndex.value.toFixed(2))
            : null;
    }

    // Regional atrophy - map z-scores to severity (support 104-class left/right labels)
    for (const region of BRAIN_REGIONS) {
        const effectiveZ = getEffectiveZForRegion(analysisResults.regions, region.key);
        aiData.regionalAtrophy[region.id] = effectiveZ !== null ? zscoreToAtrophyLevel(effectiveZ) : null;
    }

    // Lobar atrophy from lobar analysis
    const lobar = analysisResults.lobarAtrophy;
    for (const lobe of LOBES) {
        if (lobar?.[lobe.id]) {
            const meanZ = lobar[lobe.id].meanZScore;
            aiData.lobarAtrophy[lobe.id] = zscoreToAtrophyLevel(meanZ);
        } else {
            aiData.lobarAtrophy[lobe.id] = null;
        }
    }

    return aiData;
}

/**
 * Get effective z-score for a region from analysisResults.regions.
 * Supports bilateral key (e.g. Hippocampus) and 104-class left/right labels.
 * @param {Object} regions - analysisResults.regions
 * @param {string} regionKey - BRAIN_REGIONS[].key
 * @returns {number|null} effective z (lower = more atrophy), or null
 */
function getEffectiveZForRegion(regions, regionKey) {
    if (!regions) return null;
    const direct = regions[regionKey];
    if (direct?.effectiveZscore != null) return direct.effectiveZscore;
    if (direct?.zscore != null) return direct.zscore;
    const labels = REGION_KEY_TO_104_LABELS[regionKey];
    if (!labels || labels.length === 0) return null;
    const zs = [];
    for (const name of labels) {
        const r = regions[name];
        const z = r?.effectiveZscore ?? r?.zscore;
        if (z != null) zs.push(z);
    }
    if (zs.length === 0) return null;
    // Use worst (most negative) z so atrophy in either side is reflected
    return Math.min(...zs);
}

/**
 * Convert a z-score to an atrophy level
 * @param {number} zscore - Effective z-score (lower = more atrophy)
 * @returns {string} none | mild | moderate | high
 */
function zscoreToAtrophyLevel(zscore) {
    if (zscore === null || zscore === undefined) return null;
    if (zscore >= -1.0) return "none";
    if (zscore >= -1.5) return "mild";
    if (zscore >= -2.0) return "moderate";
    return "high";
}

// ============================================
// SAVE / EXPORT
// ============================================

/**
 * Save the evaluation as a JSON download
 * @param {string} mode - 'radiologist-only' | 'ai_then_radiologist' | 'radiologist_then_ai'
 * @param {Object} payload - The evaluation data to save
 */
export function saveEvaluation(mode, payload) {
    const submission = {
        timestamp: new Date().toISOString(),
        mode: mode,
        ...payload
    };

    const blob = new Blob([JSON.stringify(submission, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `evaluation_${mode}_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    return submission;
}

// ============================================
// UTILITY
// ============================================

function capitalize(str) {
    if (!str) return str;
    return str.charAt(0).toUpperCase() + str.slice(1);
}
