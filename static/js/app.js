/* ═══════════════════════════════════════════════════════════════════════
   MediPredict AI — Frontend Application Logic
   ═══════════════════════════════════════════════════════════════════════ */

'use strict';

// ─── State ──────────────────────────────────────────────────────────────────
let allSymptoms    = [];   // [{id, label}, ...]
let selectedSet    = new Set();
let currentMode    = 'checklist';
let nlpDebounceId  = null;
let probabilityChart = null;
let currentTopDisease = null;  // top predicted disease for hospital search

// ─── Initialise ──────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadSymptoms();
  
  // Set language toggle button states based on current language
  const currentLang = document.cookie.match(/googtrans=\/en\/([a-z]{2})/);
  const activeLang = currentLang ? currentLang[1] : 'en';
  
  const btnHi = document.getElementById('btnTextHi');
  const btnMr = document.getElementById('btnTextMr');
  
  if (btnHi) btnHi.textContent = activeLang === 'hi' ? 'English' : 'हिंदी';
  if (btnMr) btnMr.textContent = activeLang === 'mr' ? 'English' : 'मराठी';
});

async function loadSymptoms() {
  try {
    const res  = await fetch('/symptoms');
    const data = await res.json();
    allSymptoms = data.symptoms || [];
    renderSymptomGrid(allSymptoms);
  } catch (err) {
    document.getElementById('symptomGrid').innerHTML =
      '<p style="color:#f87171;padding:16px;grid-column:1/-1">Failed to load symptoms. Is Flask running?</p>';
  }
}

// ─── Mode Switching ───────────────────────────────────────────────────────────
function switchMode(mode) {
  currentMode = mode;

  document.getElementById('checklistModeBtn').classList.toggle('active', mode === 'checklist');
  document.getElementById('textModeBtn').classList.toggle('active', mode === 'text');

  document.getElementById('checklistPanel').classList.toggle('hidden', mode !== 'checklist');
  document.getElementById('textPanel').classList.toggle('hidden', mode !== 'text');

  updatePredictButton();
}

// ─── Symptom Grid ─────────────────────────────────────────────────────────────
function renderSymptomGrid(symptoms) {
  const grid = document.getElementById('symptomGrid');
  if (!symptoms.length) {
    grid.innerHTML = '<p style="color:#94a3b8;padding:16px;grid-column:1/-1;text-align:center">No symptoms match your search.</p>';
    return;
  }

  grid.innerHTML = symptoms.map(s => `
    <button
      class="symptom-chip${selectedSet.has(s.id) ? ' selected' : ''}"
      id="chip-${s.id}"
      onclick="toggleSymptom('${s.id}', '${escapeAttr(s.label)}')"
      aria-pressed="${selectedSet.has(s.id)}"
      title="${escapeAttr(s.label)}"
    >
      <span class="chip-check" aria-hidden="true"></span>
      ${escapeHtml(s.label)}
    </button>
  `).join('');
}

function filterSymptoms(query) {
  const q = query.toLowerCase().trim();
  const filtered = q
    ? allSymptoms.filter(s => s.label.toLowerCase().includes(q) || s.id.includes(q.replace(/ /g, '_')))
    : allSymptoms;
  renderSymptomGrid(filtered);
}

function toggleSymptom(id, label) {
  if (selectedSet.has(id)) {
    selectedSet.delete(id);
  } else {
    selectedSet.add(id);
  }
  renderSelectedTags();
  updatePredictButton();

  // Update chip state
  const chip = document.getElementById(`chip-${id}`);
  if (chip) {
    chip.classList.toggle('selected', selectedSet.has(id));
    chip.setAttribute('aria-pressed', selectedSet.has(id));
  }

  // Update count
  const count = selectedSet.size;
  document.getElementById('selectedCount').textContent = count;
  document.getElementById('minSymptomHint').style.display = count >= 3 ? 'none' : '';
}

function renderSelectedTags() {
  const container = document.getElementById('selectedTags');
  if (selectedSet.size === 0) {
    container.innerHTML = '';
    return;
  }

  container.innerHTML = [...selectedSet].map(id => {
    const sym = allSymptoms.find(s => s.id === id);
    const label = sym ? sym.label : id.replace(/_/g, ' ');
    return `
      <span class="tag" id="tag-${id}">
        ${escapeHtml(label)}
        <button class="tag-remove" onclick="removeSymptom('${id}')" aria-label="Remove ${escapeAttr(label)}">×</button>
      </span>
    `;
  }).join('');
}

function removeSymptom(id) {
  selectedSet.delete(id);
  renderSelectedTags();
  updatePredictButton();
  document.getElementById('selectedCount').textContent = selectedSet.size;
  document.getElementById('minSymptomHint').style.display = selectedSet.size >= 3 ? 'none' : '';

  const chip = document.getElementById(`chip-${id}`);
  if (chip) {
    chip.classList.remove('selected');
    chip.setAttribute('aria-pressed', 'false');
  }
}

// ─── Free Text NLP ────────────────────────────────────────────────────────────
function onTextInput(text) {
  clearTimeout(nlpDebounceId);
  updatePredictButton();

  if (text.trim().length < 5) {
    document.getElementById('nlpPreview').classList.add('hidden');
    return;
  }

  nlpDebounceId = setTimeout(() => previewNlpMatch(text), 600);
}

async function previewNlpMatch(text) {
  if (text.trim().length < 5) return;
  try {
    const res  = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();

    const preview = document.getElementById('nlpPreview');
    const tagsEl  = document.getElementById('nlpMatchedTags');

    const matched = data.free_text_matched || [];
    if (matched.length > 0) {
      preview.classList.remove('hidden');
      tagsEl.innerHTML = matched.map(s => `<span class="nlp-tag">${escapeHtml(s)}</span>`).join('');
    } else {
      preview.classList.add('hidden');
    }
  } catch (_) { /* silent preview failure */ }
}

// ─── Predict Button State ─────────────────────────────────────────────────────
function updatePredictButton() {
  const btn = document.getElementById('predictBtn');
  let enabled = false;

  if (currentMode === 'checklist') {
    enabled = selectedSet.size >= 3;
  } else {
    const text = (document.getElementById('freeTextInput')?.value || '').trim();
    enabled = text.length >= 10;
  }

  btn.disabled = !enabled;
  btn.setAttribute('aria-disabled', !enabled);
}

// ─── Run Prediction ───────────────────────────────────────────────────────────
async function runPrediction() {
  const btn = document.getElementById('predictBtn');
  const content = btn.querySelector('.predict-btn-content');
  const spinner = btn.querySelector('.btn-spinner');

  // Loading state
  btn.disabled = true;
  content.classList.add('hidden');
  spinner.classList.remove('hidden');

  try {
    let body;
    if (currentMode === 'checklist') {
      body = { symptoms: [...selectedSet] };
    } else {
      const text = document.getElementById('freeTextInput').value.trim();
      body = { text };
    }

    const res  = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    const data = await res.json();

    if (!res.ok) {
      showError(data.error || 'Prediction failed. Please try again.');
      return;
    }

    renderResults(data);

  } catch (err) {
    showError('Network error. Please check Flask is running and try again.');
  } finally {
    btn.disabled = false;
    content.classList.remove('hidden');
    spinner.classList.add('hidden');
    updatePredictButton();
  }
}

function showError(msg) {
  // Simple inline error
  const existing = document.getElementById('errorMsg');
  if (existing) existing.remove();

  const el = document.createElement('div');
  el.id = 'errorMsg';
  el.style.cssText = `
    background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);
    border-radius:12px;padding:14px 18px;color:#f87171;font-size:0.9rem;
    margin-top:12px;display:flex;align-items:center;gap:10px;
  `;
  el.innerHTML = `<span>⚠️</span><span>${escapeHtml(msg)}</span>`;
  document.getElementById('predictBtn').insertAdjacentElement('afterend', el);
  setTimeout(() => el.remove(), 6000);
}

// ─── Render Results ───────────────────────────────────────────────────────────
function renderResults(data) {
  const section = document.getElementById('resultsSection');
  section.classList.remove('hidden');

  // Store top disease for hospital finder
  if (data.predictions && data.predictions.length > 0) {
    currentTopDisease = data.predictions[0].disease;
  }

  // Symptoms used
  const symptomsEl = document.getElementById('symptomsUsedList');
  const used = data.symptoms_used || [];
  symptomsEl.textContent = used.length
    ? used.map(s => s.toLowerCase()).join(', ')
    : '—';

  // Emergency banner
  const emergencyBanner = document.getElementById('emergencyBanner');
  const emergencyText   = document.getElementById('emergencyText');
  if (data.is_emergency && data.red_flags && data.red_flags.length > 0) {
    emergencyBanner.classList.remove('hidden');
    emergencyText.textContent = `Red-flag symptoms detected: ${data.red_flags.join(', ')}. Please seek immediate medical attention.`;
  } else {
    emergencyBanner.classList.add('hidden');
  }

  // Prediction cards
  renderPredictionCards(data.predictions || []);

  // Chart
  renderChart(data.predictions || []);

  // Reset hospital finder for new search
  document.getElementById('pincodeInput').value = '';
  document.getElementById('hospitalResults').classList.add('hidden');
  document.getElementById('hospitalResults').innerHTML = '';

  // Scroll into view
  section.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ─── Hospital Finder ──────────────────────────────────────────────────────────
async function findHospitals() {
  const pincode = document.getElementById('pincodeInput').value.trim();
  if (!/^\d{6}$/.test(pincode)) {
    showHospitalError('Please enter a valid 6-digit PIN code.');
    return;
  }

  const btn      = document.getElementById('findHospitalsBtn');
  const btnText  = btn.querySelector('.pincode-btn-text');
  const spinner  = btn.querySelector('.pincode-spinner');
  const resultsEl = document.getElementById('hospitalResults');

  // Loading state
  btn.disabled = true;
  btnText.classList.add('hidden');
  spinner.classList.remove('hidden');
  resultsEl.classList.remove('hidden');
  resultsEl.innerHTML = `
    <div style="display:flex;align-items:center;gap:12px;padding:20px;color:var(--text-muted);font-size:0.9rem;">
      <div class="spinner-sm"></div>
      Searching for hospitals near PIN ${pincode}…
    </div>`;

  try {
    const res  = await fetch('/hospitals', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pincode, disease: currentTopDisease || '' }),
    });
    const data = await res.json();

    if (!res.ok) {
      showHospitalError(data.error || 'Hospital search failed.');
      return;
    }
    renderHospitals(data);

  } catch (err) {
    showHospitalError('Network error. Please check your internet connection.');
  } finally {
    btn.disabled = false;
    btnText.classList.remove('hidden');
    spinner.classList.add('hidden');
  }
}

function showHospitalError(msg) {
  const el = document.getElementById('hospitalResults');
  el.classList.remove('hidden');
  el.innerHTML = `
    <div style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.25);border-radius:12px;padding:16px 20px;color:#f87171;font-size:0.9rem;display:flex;align-items:center;gap:10px;">
      <span>⚠️</span><span>${escapeHtml(msg)}</span>
    </div>`;
}

function renderHospitals(data) {
  const el = document.getElementById('hospitalResults');
  const hospitals = data.hospitals || [];

  if (hospitals.length === 0) {
    el.innerHTML = `
      <div class="hosp-none">
        <div class="hosp-none-icon">🏥</div>
        <p>No hospitals found within ${data.radius_km} km of PIN code <strong>${escapeHtml(data.pincode)}</strong>.</p>
        <p style="margin-top:6px;font-size:0.8rem;">Try a different PIN code or search in a nearby area.</p>
      </div>`;
    return;
  }

  const specialtyLabel = (data.specialties_searched || [])
    .filter(s => s !== 'hospital' && s !== 'clinic')
    .map(s => s.charAt(0).toUpperCase() + s.slice(1))
    .join(', ') || 'General';

  let html = `
    <div class="hosp-area-banner">
      📍 ${escapeHtml(data.area)} (PIN: ${escapeHtml(data.pincode)}) &nbsp;·&nbsp;
      ${hospitals.length} of ${data.total_found} hospitals shown &nbsp;·&nbsp;
      Recommended specialty: <strong>${escapeHtml(specialtyLabel)}</strong>
    </div>`;

  hospitals.forEach((h, i) => {
    const distText = h.distance_km < 1
      ? `${h.distance_m} m away`
      : `${h.distance_km} km away`;

    const typeLabel = h.amenity === 'clinic' ? 'Clinic' : 'Hospital';
    const animDelay = `animation-delay:${i * 0.06}s`;

    html += `
      <div class="hosp-card ${h.emergency ? 'emergency-hosp' : ''}" style="${animDelay}">
        <div class="hosp-rank-badge">${i + 1}</div>
        <div class="hosp-body">
          <div class="hosp-name" title="${escapeAttr(h.name)}">${escapeHtml(h.name)}</div>
          <div class="hosp-address">${escapeHtml(h.address)}</div>
          <div class="hosp-tags">
            <span class="hosp-tag hosp-tag-type">${typeLabel}</span>
            ${h.emergency ? '<span class="hosp-tag hosp-tag-emergency">🚨 Emergency</span>' : ''}
            ${h.speciality ? `<span class="hosp-tag hosp-tag-specialty">${escapeHtml(h.speciality)}</span>` : ''}
            ${h.phone ? `<span class="hosp-tag" style="background:rgba(74,222,128,0.1);border:1px solid rgba(74,222,128,0.25);color:var(--green-400);">📞 ${escapeHtml(h.phone)}</span>` : ''}
          </div>
        </div>
        <div class="hosp-right">
          <span class="hosp-distance">📍 ${distText}</span>
          <a class="hosp-maps-btn" href="${escapeAttr(h.maps_url)}" target="_blank" rel="noopener noreferrer">
            <svg viewBox="0 0 24 24" fill="none"><path d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
            View on Maps
          </a>
        </div>
      </div>`;
  });

  el.innerHTML = html;
}


function renderPredictionCards(predictions) {
  const container = document.getElementById('predictionCards');
  const ranks = ['🥇', '🥈', '🥉'];

  container.innerHTML = predictions.map((pred, i) => {
    const confPct = Math.round(pred.confidence * 100);
    const confClass = confPct >= 60 ? 'conf-high' : confPct >= 35 ? 'conf-med' : 'conf-low';

    const sevClass = {
      severe: 'badge-severe',
      moderate: 'badge-moderate',
      mild: 'badge-mild',
    }[pred.severity] || 'badge-mild';

    const sevIcon = {
      severe: '🔴',
      moderate: '🟡',
      mild: '🟢',
    }[pred.severity] || '⚪';

    const precautionsHtml = (pred.precautions || [])
      .map(p => `<li>${escapeHtml(p)}</li>`)
      .join('');

    return `
      <article class="pred-card" aria-label="Prediction ${i+1}: ${escapeAttr(pred.disease)}">
        <div class="pred-rank" aria-hidden="true">${String(i+1).padStart(2,'0')}</div>

        <div class="pred-header">
          <div>
            <div style="font-size:0.82rem;color:var(--text-muted);font-weight:600;margin-bottom:4px;letter-spacing:0.05em;text-transform:uppercase">
              ${ranks[i] || `#${i+1}`} Possible Condition
            </div>
            <h3 class="pred-disease">${escapeHtml(pred.disease)}</h3>
          </div>
          <div class="pred-confidence">
            <span class="confidence-label">Confidence</span>
            <span class="confidence-value ${confClass}">${confPct}%</span>
          </div>
        </div>

        <div class="confidence-bar-wrap">
          <div class="confidence-bar-track" role="progressbar" aria-valuenow="${confPct}" aria-valuemin="0" aria-valuemax="100" aria-label="Confidence ${confPct}%">
            <div class="confidence-bar-fill" style="width:${confPct}%"></div>
          </div>
        </div>

        <p class="pred-description">${escapeHtml(pred.description)}</p>

        <div class="pred-precautions">
          <div class="precautions-title">Home Care & Precautions</div>
          <ul class="precautions-list">${precautionsHtml}</ul>
        </div>

        <div class="pred-badges">
          ${pred.see_doctor
            ? '<span class="badge badge-see-doctor">🏥 See a Doctor</span>'
            : '<span class="badge badge-home-care">🏠 Home Care May Suffice</span>'
          }
          <span class="badge ${sevClass}">${sevIcon} ${capitalize(pred.severity || 'unknown')} Severity</span>
        </div>
      </article>
    `;
  }).join('');
}

// ─── Chart ────────────────────────────────────────────────────────────────────
function renderChart(predictions) {
  const ctx = document.getElementById('probabilityChart').getContext('2d');

  if (probabilityChart) {
    probabilityChart.destroy();
    probabilityChart = null;
  }

  const labels = predictions.map(p => p.disease);
  const values = predictions.map(p => Math.round(p.confidence * 100));
  const colors = ['rgba(168,85,247,0.8)', 'rgba(6,182,212,0.7)', 'rgba(236,72,153,0.6)'];
  const borderColors = ['#a855f7', '#06b6d4', '#ec4899'];

  probabilityChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Confidence (%)',
        data: values,
        backgroundColor: colors.slice(0, predictions.length),
        borderColor: borderColors.slice(0, predictions.length),
        borderWidth: 1.5,
        borderRadius: 8,
        borderSkipped: false,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: 'y',
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.parsed.x}% confidence`,
          },
          backgroundColor: 'rgba(13,13,26,0.95)',
          titleColor: '#f8fafc',
          bodyColor: '#94a3b8',
          borderColor: 'rgba(255,255,255,0.08)',
          borderWidth: 1,
          padding: 12,
          cornerRadius: 8,
        },
      },
      scales: {
        x: {
          min: 0,
          max: 100,
          ticks: {
            color: '#475569',
            callback: v => `${v}%`,
            font: { family: "'Inter', sans-serif", size: 11 },
          },
          grid: {
            color: 'rgba(255,255,255,0.05)',
            drawBorder: false,
          },
          border: { display: false },
        },
        y: {
          ticks: {
            color: '#94a3b8',
            font: { family: "'Inter', sans-serif", size: 12, weight: '600' },
          },
          grid: { display: false },
          border: { display: false },
        },
      },
      animation: {
        duration: 900,
        easing: 'easeOutQuart',
      },
    },
  });
}

// ─── Reset ────────────────────────────────────────────────────────────────────
function resetApp() {
  // Clear selections
  selectedSet.clear();
  renderSelectedTags();
  document.getElementById('selectedCount').textContent = '0';
  document.getElementById('minSymptomHint').style.display = '';

  // Clear search
  const search = document.getElementById('symptomSearch');
  if (search) { search.value = ''; filterSymptoms(''); }

  // Clear free text
  const textarea = document.getElementById('freeTextInput');
  if (textarea) textarea.value = '';
  document.getElementById('nlpPreview').classList.add('hidden');

  // Hide results
  document.getElementById('resultsSection').classList.add('hidden');

  // Reset hospital finder
  document.getElementById('pincodeInput').value = '';
  document.getElementById('hospitalResults').classList.add('hidden');
  document.getElementById('hospitalResults').innerHTML = '';
  currentTopDisease = null;

  // Destroy chart
  if (probabilityChart) { probabilityChart.destroy(); probabilityChart = null; }

  // Reset all chips
  document.querySelectorAll('.symptom-chip').forEach(chip => {
    chip.classList.remove('selected');
    chip.setAttribute('aria-pressed', 'false');
  });

  updatePredictButton();

  // Scroll to top
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
function escapeHtml(str) {
  const d = document.createElement('div');
  d.textContent = str || '';
  return d.innerHTML;
}

function escapeAttr(str) {
  return (str || '').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function capitalize(str) {
  if (!str) return '';
  return str.charAt(0).toUpperCase() + str.slice(1);
}

// ─── Language Toggle ────────────────────────────────────────────────────────
function setLanguage(langCode) {
  const currentLang = document.cookie.match(/googtrans=\/en\/([a-z]{2})/);
  const activeLang = currentLang ? currentLang[1] : 'en';
  
  // If clicking the currently active language, toggle back to English
  const newLang = activeLang === langCode ? 'en' : langCode;
  
  document.cookie = `googtrans=/en/${newLang}; path=/; domain=${window.location.hostname}`;
  document.cookie = `googtrans=/en/${newLang}; path=/`;
  
  // Update button texts immediately
  const btnHi = document.getElementById('btnTextHi');
  const btnMr = document.getElementById('btnTextMr');
  if (btnHi) btnHi.textContent = newLang === 'hi' ? 'English' : 'हिंदी';
  if (btnMr) btnMr.textContent = newLang === 'mr' ? 'English' : 'मराठी';
  
  // Trigger Google Translate dynamically without reloading
  const selectElement = document.querySelector('.goog-te-combo');
  if (selectElement) {
    selectElement.value = newLang;
    selectElement.dispatchEvent(new Event('change'));
  } else {
    // Fallback if widget hasn't loaded properly
    window.location.reload();
  }
}
