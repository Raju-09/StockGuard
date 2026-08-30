/**
 * app/static/app.js
 * StockGuard — Inventory Intelligence Dashboard
 * Complete frontend logic: data fetching, chart rendering, table population.
 */

'use strict';

const BASE_URL = '';

// ─── Chart instance (module-level so we can destroy/recreate) ───────────────
let demandChart = null;

// ─── Toast auto-dismiss timer ─────────────────────────────────────────────────
let toastTimer = null;

// ─── Cached product list for metadata display ─────────────────────────────────
let productCache = [];

// ═══════════════════════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * onLoad — called via body onload.
 * Populates product dropdown and simulation results table.
 */
async function onLoad() {
  await Promise.allSettled([loadProducts(), loadSimulationResults()]);
}

// ═══════════════════════════════════════════════════════════════════════════════
// DATA LOADERS
// ═══════════════════════════════════════════════════════════════════════════════

async function loadProducts() {
  try {
    const res = await fetch(`${BASE_URL}/products`);
    if (!res.ok) throw new Error(`Server error ${res.status}`);
    const data = await res.json();
    const products = data.products || [];
    productCache = products;

    const select = document.getElementById('product');
    // Clear existing options except placeholder
    select.innerHTML = '<option value="" disabled selected>Select a product…</option>';

    products.forEach((p) => {
      const opt = document.createElement('option');
      opt.value = p.name;
      opt.textContent = p.name;
      select.appendChild(opt);
    });

    // Wire up product-change handler for metadata display
    select.addEventListener('change', onProductChange);

    // Auto-select first product and trigger initial forecast on page load
    if (products.length > 0) {
      select.value = products[0].name;
      onProductChange({ target: select });
      const invInput = document.getElementById('inventory');
      const priceInput = document.getElementById('price');
      if (invInput && !invInput.value) invInput.value = '24';
      if (priceInput && !priceInput.value) priceInput.value = '55';
      submit();
    }
  } catch (err) {
    console.error('Failed to load products:', err);
    showToast('Could not load products. Check that the API is running.');
  }
}

async function loadSimulationResults() {
  try {
    const res = await fetch(`${BASE_URL}/simulation-results`);
    if (!res.ok) throw new Error(`Server error ${res.status}`);
    const data = await res.json();
    updateSimTable(data.results || []);
  } catch (err) {
    console.error('Failed to load simulation results:', err);
    // Render empty state — non-fatal
    updateSimTable([]);
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PRODUCT CHANGE HANDLER
// ═══════════════════════════════════════════════════════════════════════════════

function onProductChange(e) {
  const name = e.target.value;
  const product = productCache.find((p) => p.name === name);
  const infoEl = document.getElementById('product-info');
  if (!product || !infoEl) return;

  document.getElementById('product-avg-demand').textContent =
    product.avg_daily_demand != null
      ? `${Number(product.avg_daily_demand).toFixed(1)} units/day`
      : '—';
  document.getElementById('product-stockout-days').textContent =
    product.stockout_days != null ? `${product.stockout_days} days` : '—';

  infoEl.classList.add('visible');
}

// ═══════════════════════════════════════════════════════════════════════════════
// SUBMIT / PREDICT
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * submit — reads form, validates, calls POST /predict, renders results.
 */
async function submit() {
  // ── Gather inputs ──────────────────────────────────────────────────────────
  const product = document.getElementById('product').value;
  const inventoryRaw = document.getElementById('inventory').value;
  const priceRaw = document.getElementById('price').value;
  const promo = document.getElementById('promo').checked;

  const currentInventory = parseFloat(inventoryRaw);
  const price = parseFloat(priceRaw);

  // ── Validate ───────────────────────────────────────────────────────────────
  if (!product) {
    showToast('Please select a product before running the forecast.');
    return;
  }
  if (inventoryRaw === '' || isNaN(currentInventory) || currentInventory < 0) {
    showToast('Current inventory must be a number ≥ 0.');
    return;
  }
  if (priceRaw === '' || isNaN(price) || price <= 0) {
    showToast('Price must be a number greater than 0.');
    return;
  }

  // ── Show spinner, disable button ───────────────────────────────────────────
  showSpinner();
  const btn = document.getElementById('predict-btn');
  btn.disabled = true;

  try {
    const res = await fetch(`${BASE_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        product,
        current_inventory: currentInventory,
        price,
        promo,
      }),
    });

    if (!res.ok) {
      let msg = `Server returned ${res.status}`;
      try {
        const errData = await res.json();
        if (errData.detail) msg = errData.detail;
      } catch (_) { /* ignore parse error */ }
      throw new Error(msg);
    }

    const data = await res.json();

    // ── Update metric cards ────────────────────────────────────────────────
    setMetric('forecast-demand', Math.round(data.forecast_demand), 'units');
    setMetric('stockout-risk', formatPercent(data.stockout_risk), '');
    setMetric('recommended-order', Math.round(data.recommended_order), 'units');
    setMetric('days-until-stockout', data.days_until_stockout, 'days');

    // ── Risk badge ─────────────────────────────────────────────────────────
    setRisk(data.stockout_risk);

    // ── Model version ──────────────────────────────────────────────────────
    const mvEl = document.getElementById('model-version');
    if (mvEl && data.model_version) {
      mvEl.textContent = `v${data.model_version}`;
      mvEl.style.display = 'inline-flex';
    }

    // ── Explanation & PM Insights ─────────────────────────────────────────
    const explCard = document.getElementById('explanation-card');
    const explText = document.getElementById('explanation-text');
    if (explCard && explText && data.explanation) {
      let html = '';
      if (typeof data.explanation === 'object') {
        html += `<div style="margin-bottom:0.5rem; font-family:monospace; font-size:0.95em;"><strong>Replenishment:</strong> ${data.explanation.recommended_order_formula}</div>`;
      } else {
        html += data.explanation;
      }
      if (data.financial_impact) {
        html += `<div style="margin-top:0.5rem; padding-top:0.5rem; border-top:1px solid rgba(255,255,255,0.1); font-size:0.85em; color:#8892a4;">
          <strong>Financial Optimization:</strong> Est. Holding Cost: $${data.financial_impact.daily_holding_cost}/day | Expected Lost Margin: $${data.financial_impact.expected_lost_margin_per_day}/day.<br>
          <span style="color:#ffffff;">${data.financial_impact.trade_off_recommendation}</span>
        </div>`;
      }
      if (data.substitution_alert && data.substitution_alert.is_active) {
        html += `<div style="margin-top:0.5rem; padding:0.5rem 0.75rem; background:rgba(231,76,60,0.15); border-left:3px solid #e74c3c; border-radius:4px; font-size:0.85em; color:#ff8b80;">
          <strong>⚠️ Cross-SKU Substitution Alert:</strong> ${data.substitution_alert.message}
        </div>`;
      }
      explText.innerHTML = html;
      explCard.classList.add('visible');
    }

    // ── Chart ──────────────────────────────────────────────────────────────
    if (data.history) {
      const h = data.history;
      updateChart(h.dates || [], h.true_demand || [], h.sales || [], h.opening_stock || []);
    }
  } catch (err) {
    console.error('Predict error:', err);
    showToast(err.message || 'An unexpected error occurred.');
  } finally {
    hideSpinner();
    btn.disabled = false;
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CHART
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * updateChart — renders or replaces the Chart.js line chart.
 * @param {string[]} dates        x-axis labels
 * @param {number[]} true_demand  dashed line (hidden true demand)
 * @param {number[]} sales        observed sales line
 * @param {number[]} opening_stock opening stock line
 */
function updateChart(dates, true_demand, sales, opening_stock) {
  const placeholder = document.getElementById('chart-placeholder');
  const wrapperEl = document.getElementById('chart-wrapper');
  const canvas = document.getElementById('demand-chart');

  if (placeholder) placeholder.style.display = 'none';
  if (wrapperEl) wrapperEl.style.display = 'block';

  // Destroy previous chart instance to avoid canvas reuse errors
  if (demandChart) {
    demandChart.destroy();
    demandChart = null;
  }

  const ctx = canvas.getContext('2d');

  demandChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: dates,
      datasets: [
        {
          label: 'True Demand (hidden)',
          data: true_demand,
          borderColor: 'rgba(231, 76, 60, 0.85)',
          backgroundColor: 'rgba(231, 76, 60, 0.08)',
          borderWidth: 2,
          borderDash: [6, 4],
          pointRadius: 0,
          pointHoverRadius: 4,
          tension: 0.35,
          fill: false,
          order: 3,
        },
        {
          label: 'Observed Sales',
          data: sales,
          borderColor: 'rgba(79, 142, 247, 0.9)',
          backgroundColor: 'rgba(79, 142, 247, 0.08)',
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 4,
          tension: 0.35,
          fill: true,
          order: 2,
        },
        {
          label: 'Opening Stock',
          data: opening_stock,
          borderColor: 'rgba(39, 174, 96, 0.85)',
          backgroundColor: 'transparent',
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 4,
          tension: 0.35,
          fill: false,
          order: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'index',
        intersect: false,
      },
      plugins: {
        legend: {
          display: false, // We use a custom legend in HTML
        },
        tooltip: {
          backgroundColor: '#1a1d27',
          borderColor: '#2d3142',
          borderWidth: 1,
          titleColor: '#ffffff',
          bodyColor: '#8892a4',
          padding: 10,
          callbacks: {
            label(ctx) {
              return ` ${ctx.dataset.label}: ${ctx.parsed.y}`;
            },
          },
        },
      },
      scales: {
        x: {
          grid: {
            color: 'rgba(45, 49, 66, 0.6)',
            drawBorder: false,
          },
          ticks: {
            color: '#8892a4',
            font: { size: 11 },
            maxTicksLimit: 10,
            maxRotation: 0,
          },
        },
        y: {
          grid: {
            color: 'rgba(45, 49, 66, 0.6)',
            drawBorder: false,
          },
          ticks: {
            color: '#8892a4',
            font: { size: 11 },
          },
          title: {
            display: true,
            text: 'Units',
            color: '#8892a4',
            font: { size: 11 },
          },
        },
      },
    },
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// SIMULATION TABLE
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * updateSimTable — renders the simulation comparison table.
 * @param {Array<{policy, stockout_pct, holding_cost, lost_sales_cost, total_cost}>} results
 */
function updateSimTable(results) {
  const tbody = document.querySelector('#sim-table tbody');
  if (!tbody) return;

  if (!results || results.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" class="table-empty">No simulation data available yet.</td></tr>`;
    return;
  }

  // Find minimum total_cost for highlighting
  const minCost = Math.min(...results.map((r) => r.total_cost));

  tbody.innerHTML = results
    .map((r) => {
      const isBest = r.total_cost === minCost;
      const rowClass = isBest ? 'best-row' : '';
      const bestBadge = isBest
        ? `<span class="best-badge">★ Best</span>`
        : '';
      return `
        <tr class="${rowClass}">
          <td>${escapeHtml(String(r.policy))}${bestBadge}</td>
          <td>${formatPercent(r.stockout_pct)}</td>
          <td>${formatCurrency(r.holding_cost)}</td>
          <td>${formatCurrency(r.lost_sales_cost)}</td>
          <td>${formatCurrency(r.total_cost)}</td>
        </tr>`;
    })
    .join('');
}

// ═══════════════════════════════════════════════════════════════════════════════
// RISK BADGE
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * setRisk — updates the stockout-risk metric card with a coloured badge.
 * Accepts either a numeric probability (0–1) or a string ('HIGH'|'MEDIUM'|'LOW').
 * @param {number|string} riskValue
 */
function setRisk(riskValue) {
  const badgeEl = document.getElementById('risk-badge');
  if (!badgeEl) return;

  let level;

  if (typeof riskValue === 'string') {
    level = riskValue.toUpperCase();
  } else {
    // Numeric probability → classify
    const pct = riskValue > 1 ? riskValue : riskValue * 100;
    if (pct >= 60) level = 'HIGH';
    else if (pct >= 30) level = 'MEDIUM';
    else level = 'LOW';
  }

  badgeEl.className = 'risk-badge';

  switch (level) {
    case 'HIGH':
      badgeEl.classList.add('risk-high');
      badgeEl.textContent = '⚠ HIGH';
      break;
    case 'MEDIUM':
      badgeEl.classList.add('risk-medium');
      badgeEl.textContent = '◉ MEDIUM';
      break;
    default:
      badgeEl.classList.add('risk-low');
      badgeEl.textContent = '✔ LOW';
  }

  badgeEl.style.display = 'inline-flex';
}

// ═══════════════════════════════════════════════════════════════════════════════
// METRIC HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

function setMetric(id, value, unit) {
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.remove('placeholder');
  el.textContent = value != null ? `${value}${unit ? ' ' + unit : ''}` : '—';
}

// ═══════════════════════════════════════════════════════════════════════════════
// SPINNER
// ═══════════════════════════════════════════════════════════════════════════════

function showSpinner() {
  const overlay = document.getElementById('loading-overlay');
  if (overlay) overlay.classList.add('active');
}

function hideSpinner() {
  const overlay = document.getElementById('loading-overlay');
  if (overlay) overlay.classList.remove('active');
}

// ═══════════════════════════════════════════════════════════════════════════════
// ERROR TOAST
// ═══════════════════════════════════════════════════════════════════════════════

function showToast(message) {
  const toast = document.getElementById('error-toast');
  const msgEl = document.getElementById('toast-message');
  if (!toast || !msgEl) return;

  msgEl.textContent = message;
  toast.classList.add('visible');

  clearTimeout(toastTimer);
  toastTimer = setTimeout(hideToast, 5000);
}

function hideToast() {
  const toast = document.getElementById('error-toast');
  if (toast) toast.classList.remove('visible');
}

// ═══════════════════════════════════════════════════════════════════════════════
// FORMATTING UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════

function formatPercent(value) {
  if (value == null) return '—';
  const pct = value > 1 ? value : value * 100;
  return `${pct.toFixed(1)}%`;
}

function formatCurrency(value) {
  if (value == null) return '—';
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
}

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
