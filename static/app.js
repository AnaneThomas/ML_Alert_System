async function postJSON(url, data) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  const payload = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = payload && payload.error ? payload.error : `Request failed (${res.status})`;
    throw new Error(msg);
  }
  return payload;
}

function byId(id) {
  return document.getElementById(id);
}

function renderResult(payload) {
  const alert = payload.alert === 1;
  const risk = typeof payload.risk_score === 'number' ? payload.risk_score : null;
  const threshold = typeof payload.threshold === 'number' ? payload.threshold : 0.5;
  const color = alert ? '#dc2626' : '#059669';
  const bgLight = alert ? '#fef2f2' : '#f0fdf4';
  const border = alert ? '#fecaca' : '#a7f3d0';
  const meterColor = risk !== null && risk >= threshold ? '#ef4444' : '#10b981';
  const meterBg = risk !== null && risk >= threshold ? '#fee2e2' : '#d1fae5';
  const status = alert ? 'ALERT' : 'SAFE';
  const reason = payload.reason || 'No reason returned.';

  const riskMeter = risk === null ? '' : `
    <div style="display:flex;flex-direction:column;gap:8px;">
      <div style="display:flex;justify-content:space-between;">
        <span style="font-size:12px;font-weight:500;color:#64748b;">ML Risk Score</span>
        <span style="font-size:14px;font-weight:700;color:${meterColor};">${risk.toFixed(4)}</span>
      </div>
      <div style="height:8px;border-radius:4px;overflow:hidden;background:${meterBg};">
        <div style="height:100%;border-radius:4px;width:${Math.min(risk, 1) * 100}%;background:${meterColor};transition:width .7s ease;"></div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:12px;color:#94a3b8;">
        <span>0</span><span>Threshold ${threshold}</span><span>1</span>
      </div>
    </div>`;

  return `
    <div style="border-radius:12px;border:1px solid ${border};background:${bgLight};padding:16px;display:flex;flex-direction:column;gap:12px;">
      <div style="display:flex;align-items:center;gap:8px;font-weight:700;font-size:14px;color:${color};">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="${color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          ${alert ? '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>' : '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>'}
        </svg>
        ${status}
      </div>
      <p style="font-size:12px;line-height:1.6;color:${color};margin:0;">${reason}</p>
      ${riskMeter}
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
        <div style="background:rgba(255,255,255,.7);border-radius:8px;padding:8px;">
          <div style="font-size:11px;color:#64748b;">Rule Alert</div>
          <div style="font-size:16px;font-weight:700;color:${payload.rule_alert ? '#dc2626' : '#059669'};margin-top:2px;">${payload.rule_alert ?? 0}</div>
        </div>
        <div style="background:rgba(255,255,255,.7);border-radius:8px;padding:8px;">
          <div style="font-size:11px;color:#64748b;">ML Alert</div>
          <div style="font-size:16px;font-weight:700;color:${payload.ml_alert ? '#dc2626' : '#059669'};margin-top:2px;">${payload.ml_alert ?? 0}</div>
        </div>
      </div>
    </div>`;
}

function debounce(fn, waitMs) {
  let t = null;
  return (...args) => {
    if (t) clearTimeout(t);
    t = setTimeout(() => fn(...args), waitMs);
  };
}

async function getJSON(url) {
  const res = await fetch(url);
  const payload = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = payload && payload.error ? payload.error : `Request failed (${res.status})`;
    throw new Error(msg);
  }
  return payload;
}

let selectedPatientId = null;

function setSelectedPatientCard(patient) {
  const card = byId('patient-selected');
  if (!card) return;
  card.hidden = false;
  const name = patient.name || 'Unknown';
  const age = typeof patient.age === 'number' ? Math.round(patient.age) : '';
  const gender = patient.gender || '';
  card.innerHTML = `Selected: <span class="value">${name}</span> <span class="muted mono">(${patient.id})</span> <span class="muted">${gender}${age !== '' ? ' • ' + age + 'y' : ''}</span>`;
}

function hideResults() {
  const box = byId('patient-results');
  if (box) {
    box.hidden = true;
    box.innerHTML = '';
  }
}

function renderResults(results) {
  const box = byId('patient-results');
  if (!box) return;
  if (!results || results.length === 0) {
    box.hidden = true;
    box.innerHTML = '';
    return;
  }
  box.hidden = false;
  box.innerHTML = results
    .map((r) => {
      const name = r.name || 'Unknown';
      const age = typeof r.age === 'number' ? Math.round(r.age) : '';
      const gender = r.gender || '';
      return `<button type="button" class="result-item" data-id="${r.id}"><span class="value">${name}</span><span class="muted mono">${r.id}</span><span class="muted">${gender}${age !== '' ? ' • ' + age + 'y' : ''}</span></button>`;
    })
    .join('');
}

async function selectPatient(patientId) {
  selectedPatientId = patientId;
  hideResults();
  const patient = await getJSON(`/api/patients/${encodeURIComponent(patientId)}`);
  setSelectedPatientCard(patient);
  byId('age').value = typeof patient.age === 'number' ? patient.age.toFixed(1) : '';
  byId('gender').value = patient.gender || '';
  byId('conditions').value = patient.conditions || '';
  byId('allergies').value = patient.allergies || '';
}

function setupPatientSearch() {
  const input = byId('patient-search');
  const box = byId('patient-results');
  if (!input || !box) return;

  const run = debounce(async () => {
    const q = input.value.trim();
    if (!q) {
      hideResults();
      return;
    }
    const payload = await getJSON(`/api/patients/search?q=${encodeURIComponent(q)}`);
    renderResults(payload.results || []);
  }, 250);

  input.addEventListener('input', () => {
    run();
  });

  box.addEventListener('click', async (e) => {
    const btn = e.target.closest('.result-item');
    if (!btn) return;
    const pid = btn.getAttribute('data-id');
    if (!pid) return;
    try {
      await selectPatient(pid);
      input.value = '';
    } catch (err) {
      const resultCard = byId('result-card');
      const result = byId('result');
      resultCard.hidden = false;
      result.innerHTML = `<div class="error">${err.message}</div>`;
    }
  });

  document.addEventListener('click', (e) => {
    if (e.target === input || box.contains(e.target)) return;
    hideResults();
  });
}

setupPatientSearch();

const predictForm = byId('predict-form');
if (predictForm) predictForm.addEventListener('submit', async (e) => {
  e.preventDefault();

  const data = {
    age: byId('age').value,
    gender: byId('gender').value,
    conditions: byId('conditions').value,
    allergies: byId('allergies').value,
    medication: byId('medication').value,
  };

  const resultCard = byId('result-card');
  const result = byId('result');
  resultCard.hidden = false;
  result.innerHTML = '<div class="muted">Checking...</div>';

  try {
    if (selectedPatientId) {
      // Refresh patient context to ensure latest CSV-derived values are used
      await selectPatient(selectedPatientId);
    }
    const payload = await postJSON('/predict', data);
    result.innerHTML = renderResult(payload);
  } catch (err) {
    result.innerHTML = `<div class="error">${err.message}</div>`;
  }
});

const exampleBtn = byId('example-btn');
if (exampleBtn) exampleBtn.addEventListener('click', () => {
  const age = byId('age');
  const gender = byId('gender');
  if (age && !age.readOnly) age.value = '32';
  if (gender && !gender.disabled && !gender.readOnly) gender.value = 'F';
  byId('conditions').value = 'pregnancy';
  byId('allergies').value = 'latex allergy';
  byId('medication').value = 'warfarin';
});

const clearBtn = byId('clear-btn');
if (clearBtn) clearBtn.addEventListener('click', () => {
  selectedPatientId = null;
  const selected = byId('patient-selected');
  if (selected) {
    selected.hidden = true;
    selected.innerHTML = '';
  }
  const search = byId('patient-search');
  if (search) search.value = '';
  hideResults();

  const age = byId('age');
  const gender = byId('gender');
  const conditions = byId('conditions');
  const allergies = byId('allergies');
  const medication = byId('medication');
  if (age) age.value = '';
  if (gender) gender.value = '';
  if (conditions) conditions.value = '';
  if (allergies) allergies.value = '';
  if (medication) medication.value = '';

  const resultCard = byId('result-card');
  const result = byId('result');
  if (resultCard) resultCard.hidden = true;
  if (result) result.innerHTML = '';
});
