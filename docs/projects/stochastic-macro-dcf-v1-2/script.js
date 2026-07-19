// =============================================================
// script.js — Probabilistic DCF Engine (JavaScript v1)
//
// Core simulation + valuation logic is UNCHANGED from v1.
// This file adds:
//   - Contextual help on field focus
//   - Chart.js histogram with hover tooltips
//   - Chart.js line charts for commodity, FCF, WACC paths
//   - Random-by-default seed behavior
//
// STRUCTURE:
//   1.  Random Number Generation
//   2.  State Path Simulation
//   3.  Operating Model
//   4.  Valuation Model
//   5.  Summary Statistics
//   6.  Sample Path Selection
//   7.  Input Help System
//   8.  Chart Rendering (Chart.js)
//   9.  Summary Cards
//   10. Read Inputs
//   11. Main: runModel()
//   12. Utilities
// =============================================================
 
 
// =============================================================
// 1. SEEDED RANDOM NUMBER GENERATOR
//
// Math.random() can't be seeded, so we use mulberry32.
// Leaving the seed blank = fresh random result every run.
// =============================================================
 
function makePRNG(seed) {
  let s = seed >>> 0;
  return function () {
    s += 0x6d2b79f5;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
 
// Box-Muller: turns two uniform [0,1] numbers into one N(0,1) sample
function makeNormalSampler(rand) {
  return function () {
    const u1 = rand();
    const u2 = rand();
    return Math.sqrt(-2 * Math.log(u1 + 1e-15)) * Math.cos(2 * Math.PI * u2);
  };
}
 
 
// =============================================================
// 2. STATE PATH SIMULATION
//
// Commodity price → Ornstein-Uhlenbeck (mean-reverting)
// Interest rate   → Ornstein-Uhlenbeck (mean-reverting)
// Demand index    → Geometric Brownian Motion (growth)
// All three are independent in v1.
// =============================================================
 
function simulateStatePaths(inputs, randn) {
  const n = inputs.num_simulations;
  const T = inputs.forecast_years;
 
  const commodity = simulateOU(
    inputs.commodity_current,
    inputs.commodity_reversion_speed,
    inputs.commodity_long_run_mean,
    inputs.commodity_vol * inputs.commodity_current, // level-scaled absolute vol
    n, T, randn, 1.0
  );
 
  const rate = simulateOU(
    inputs.rate_current,
    inputs.rate_drift,
    inputs.rate_long_run_mean,
    inputs.rate_vol,
    n, T, randn, 0.001
  );
 
  const demand = simulateGBM(
    inputs.demand_current,
    inputs.demand_growth,
    inputs.demand_vol,
    n, T, randn
  );
 
  return { commodity, rate, demand };
}
 
// Discrete Ornstein-Uhlenbeck: X(t+1) = X(t) + κ(θ - X(t)) + σ·ε
function simulateOU(x0, kappa, theta, sigma, n, T, randn, floor) {
  const paths = [];
  for (let i = 0; i < n; i++) {
    const path = new Array(T);
    let x = x0;
    for (let t = 0; t < T; t++) {
      x = x + kappa * (theta - x) + sigma * randn();
      if (floor !== undefined) x = Math.max(x, floor);
      path[t] = x;
    }
    paths.push(path);
  }
  return paths;
}
 
// Geometric Brownian Motion: X(t+1) = X(t) · exp((μ - σ²/2) + σ·ε)
function simulateGBM(x0, mu, sigma, n, T, randn) {
  const drift = mu - 0.5 * sigma * sigma;
  const paths = [];
  for (let i = 0; i < n; i++) {
    const path = new Array(T);
    let x = x0;
    for (let t = 0; t < T; t++) {
      x = x * Math.exp(drift + sigma * randn());
      path[t] = x;
    }
    paths.push(path);
  }
  return paths;
}
 
 
// =============================================================
// 3. OPERATING MODEL
//
// revenue → margin → EBIT → D&A → capex → WC → FCF
// FCF = EBIT*(1-tax) + D&A - Capex - ΔWC
// =============================================================
 
function buildOperatingPaths(statePaths, inputs) {
  const { commodity, demand } = statePaths;
  const n = commodity.length;
  const T = commodity[0].length;
 
  const rev0        = inputs.current_revenue;
  const prod0       = inputs.current_production;
  const comm0       = inputs.commodity_current;
  const demand0     = inputs.demand_current;
  const baseMargin  = inputs.base_operating_margin;
  const baseCapex   = inputs.capex_percent_revenue;
  const daPct       = inputs.da_percent_revenue;
  const wcPct       = inputs.working_capital_percent_revenue;
  const tax         = inputs.tax_rate;
 
  const revSensComm   = inputs.revenue_sensitivity_to_commodity;
  const revSensDem    = inputs.revenue_sensitivity_to_demand;
  const marginSens    = inputs.margin_sensitivity_to_commodity;
  const breakeven     = inputs.cash_breakeven_price;
  const belowMult     = inputs.below_breakeven_multiplier;
  const marginFloor   = inputs.structural_margin_floor;
  const capexSens     = inputs.capex_sensitivity_to_commodity;
  const prodReinvSens = inputs.production_growth_sensitivity_to_reinvestment;
  const breakevenDev  = (breakeven - comm0) / comm0;
 
  const production  = zeros2D(n, T);
  const revenue     = zeros2D(n, T);
  const margin      = zeros2D(n, T);
  const ebit        = zeros2D(n, T);
  const da          = zeros2D(n, T);
  const capexArr    = zeros2D(n, T);
  const wcArr       = zeros2D(n, T);
  const deltaWcArr  = zeros2D(n, T);
  const fcf         = zeros2D(n, T);
 
  for (let i = 0; i < n; i++) {
    let prodPrev = prod0;
    let wcPrev   = wcPct * rev0;
 
    for (let t = 0; t < T; t++) {
      const commT   = commodity[i][t];
      const demandT = demand[i][t];
      const commDev   = (commT - comm0) / comm0;
      const demandDev = (demandT - demand0) / demand0;
 
      const prodT = prodPrev;
      production[i][t] = prodT;
 
      let revT = rev0
        * (1.0 + revSensComm * commDev)
        * (1.0 + revSensDem  * demandDev)
        * (prodT / prod0);
      revT = Math.max(revT, 0);
      revenue[i][t] = revT;
 
      // Kinked margin sensitivity: normal slope above cash breakeven,
      // steeper slope below it (fixed costs don't scale down, so margin
      // compression accelerates once price falls under cash cost).
      let marginT;
      if (commT >= breakeven) {
        marginT = baseMargin + marginSens * commDev;
      } else {
        const normalPortion  = marginSens * breakevenDev;
        const steepPortion   = marginSens * belowMult * (commDev - breakevenDev);
        marginT = baseMargin + normalPortion + steepPortion;
      }
      // Structural floor: the point where a producer would shut in
      // production rather than operate at a larger loss.
      marginT = Math.min(Math.max(marginT, marginFloor), 0.95);
      margin[i][t] = marginT;
 
      const ebitT = revT * marginT;
      ebit[i][t] = ebitT;
 
      const daT = daPct * revT;
      da[i][t] = daT;
 
      let capexPctT = baseCapex + capexSens * commDev;
      capexPctT = Math.min(Math.max(capexPctT, 0), 0.60);
      const capexT = capexPctT * revT;
      capexArr[i][t] = capexT;
 
      const wcT     = wcPct * revT;
      const deltaWcT = wcT - wcPrev;
      wcArr[i][t]    = wcT;
      deltaWcArr[i][t] = deltaWcT;
 
      // FCF = EBIT*(1-T) + D&A - Capex - ΔWC
      fcf[i][t] = ebitT * (1.0 - tax) + daT - capexT - deltaWcT;
 
      const extraCapexPct = Math.max(capexPctT - baseCapex, 0);
      prodPrev = prodT * (1.0 + prodReinvSens * extraCapexPct);
      wcPrev   = wcT;
    }
  }
 
  return { production, revenue, margin, ebit, da,
           capex: capexArr, workingCapital: wcArr, deltaWC: deltaWcArr, fcf };
}
 
 
// =============================================================
// 4. VALUATION MODEL
//
// WACC = equity_weight*Ke + debt_weight*Kd*(1-tax)
// Only rf varies with the simulation; all other WACC inputs fixed.
// EV = sum(PV of FCFs) + PV of terminal value
// Terminal-year FCF is smoothed over the final N forecast years (per path)
// rather than a single raw year, so one noisy draw doesn't dominate the
// ~60-70% of value that terminal value typically represents.
// Equity = EV - net_debt  →  Price = Equity / shares
// =============================================================
 
function buildWaccPaths(statePaths, inputs) {
  const { rate } = statePaths;
  const n = rate.length;
  const T = rate[0].length;
  const { beta, equity_risk_premium: erp, debt_spread,
          debt_weight, equity_weight, tax_rate } = inputs;
 
  const wacc = zeros2D(n, T);
  for (let i = 0; i < n; i++) {
    for (let t = 0; t < T; t++) {
      const rf = rate[i][t];
      const ke = rf + beta * erp;
      const kd = rf + debt_spread;
      wacc[i][t] = Math.max(equity_weight * ke + debt_weight * kd * (1 - tax_rate), 0.01);
    }
  }
  return wacc;
}
 
function discountValuationPaths(operatingPaths, waccPaths, inputs) {
  const { fcf } = operatingPaths;
  const n = fcf.length;
  const T = fcf[0].length;
  const { terminal_growth: g, net_debt, shares_outstanding } = inputs;
  const smoothYears = Math.max(1, Math.min(inputs.terminal_smoothing_years || 1, T));
  const prices = new Array(n);
 
  for (let i = 0; i < n; i++) {
    let running = 1.0;
    const cumDisc = new Array(T);
    for (let t = 0; t < T; t++) {
      running *= (1.0 + waccPaths[i][t]);
      cumDisc[t] = running;
    }
 
    let pvFcf = 0;
    for (let t = 0; t < T; t++) {
      pvFcf += fcf[i][t] / cumDisc[t];
    }
 
    const waccTerminal = waccPaths[i][T - 1];
    // Average the final `smoothYears` years of this path's own FCF
    // instead of using a single raw draw as the terminal-year base.
    let fcfSum = 0;
    for (let t = T - smoothYears; t < T; t++) {
      fcfSum += fcf[i][t];
    }
    const fcfTerminal  = fcfSum / smoothYears;
    const tvDenom      = waccTerminal - g;
    const tv = tvDenom > 0.001 ? fcfTerminal * (1.0 + g) / tvDenom : 0;
 
    const ev    = pvFcf + tv / cumDisc[T - 1];
    prices[i]   = (ev - net_debt) / shares_outstanding;
  }
 
  return prices;
}
 
 
// =============================================================
// 5. SUMMARY STATISTICS
// =============================================================
 
function summarizeResults(prices, currentPrice) {
  const sorted = [...prices].sort((a, b) => a - b);
  return {
    mean:            mean(prices),
    median:          percentile(sorted, 50),
    p5:              percentile(sorted, 5),
    p25:             percentile(sorted, 25),
    p75:             percentile(sorted, 75),
    p95:             percentile(sorted, 95),
    probBelowMarket: prices.filter(p => p < currentPrice).length / prices.length,
  };
}
 
function percentile(sorted, q) {
  const idx = (q / 100) * (sorted.length - 1);
  const lo  = Math.floor(idx);
  const hi  = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (idx - lo) * (sorted[hi] - sorted[lo]);
}
 
function mean(arr) {
  return arr.reduce((s, v) => s + v, 0) / arr.length;
}
 
 
// =============================================================
// 6. SAMPLE PATH SELECTION
//
// Picks nSamples paths evenly spread across the sorted price
// distribution. Marks the one closest to the median as central.
// =============================================================
 
function selectSamplePaths(statePaths, operatingPaths, waccPaths, prices, nSamples) {
  const n = prices.length;
  const sortedIdx = Array.from({ length: n }, (_, i) => i)
    .sort((a, b) => prices[a] - prices[b]);
 
  const selected = [];
  for (let k = 0; k < nSamples; k++) {
    const pos = Math.round((k / (nSamples - 1)) * (n - 1));
    selected.push(sortedIdx[pos]);
  }
 
  const med = percentile([...prices].sort((a, b) => a - b), 50);
  const samplePrices = selected.map(i => prices[i]);
  const centralIdx = samplePrices.reduce((best, p, k) =>
    Math.abs(p - med) < Math.abs(samplePrices[best] - med) ? k : best, 0);
 
  return {
    commodity:  selected.map(i => statePaths.commodity[i]),
    rate:       selected.map(i => statePaths.rate[i]),
    demand:     selected.map(i => statePaths.demand[i]),
    fcf:        selected.map(i => operatingPaths.fcf[i]),
    wacc:       selected.map(i => waccPaths[i]),
    prices:     samplePrices,
    centralIdx,
  };
}
 
 
// =============================================================
// 7. INPUT HELP SYSTEM
//
// Each .field element has data-help-title and data-help-body
// attributes set in the HTML. On focus of any child input,
// we read those attributes and update the shared help box.
// =============================================================
 
function initHelpSystem() {
  // Listen for focus on any input inside a .field wrapper
  document.querySelectorAll('.field').forEach(fieldEl => {
    const input = fieldEl.querySelector('input');
    if (!input) return;
 
    input.addEventListener('focus', () => {
      const title = fieldEl.dataset.helpTitle;
      const body  = fieldEl.dataset.helpBody;
      if (title && body) {
        document.getElementById('help-title').textContent = title;
        document.getElementById('help-body').textContent  = body;
      }
    });
  });
}
 
 
// =============================================================
// 8. CHART RENDERING (Chart.js)
//
// We keep track of existing chart instances so we can destroy
// them before re-drawing on a second run. Otherwise Chart.js
// throws errors about canvas already being in use.
// =============================================================
 
// Store chart instances so we can destroy and rebuild on re-run
const _charts = {};
 
function destroyChart(id) {
  if (_charts[id]) {
    _charts[id].destroy();
    delete _charts[id];
  }
}
 
// Shared Chart.js defaults for all charts
const CHART_DEFAULTS = {
  animation: false,
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { display: false },
  },
  scales: {
    x: {
      grid: { color: '#1a2438' },
      ticks: { color: '#4b6280', font: { family: 'DM Mono', size: 11 } },
    },
    y: {
      grid: { color: '#1a2438' },
      ticks: { color: '#4b6280', font: { family: 'DM Mono', size: 11 } },
    },
  },
};
 
// ---- 8A. Histogram of per-share value distribution ----
// Renders bars + a smooth KDE overlay line derived from the actual data.
// The KDE line does NOT assume a normal distribution — it uses a
// Gaussian kernel evaluated at each bin midpoint.
function renderHistogram(prices, currentPrice) {
  destroyChart('hist');
 
  const sorted   = [...prices].sort((a, b) => a - b);
  const minVal   = sorted[0];
  const maxVal   = sorted[sorted.length - 1];
  const numBins  = 40;
  const binWidth = (maxVal - minVal) / numBins || 1;
  const n        = prices.length;
 
  // Count prices into bins
  const counts = new Array(numBins).fill(0);
  for (const p of prices) {
    const bin = Math.min(Math.floor((p - minVal) / binWidth), numBins - 1);
    counts[bin]++;
  }
 
  // --- Kernel Density Estimate (KDE) ---
  // Uses a Gaussian kernel with bandwidth chosen by Silverman's rule.
  // Evaluated at each bin midpoint so it traces the actual data shape.
  const stdDev = Math.sqrt(prices.reduce((s, p) => s + (p - mean(prices)) ** 2, 0) / n);
  const bandwidth = 1.06 * stdDev * Math.pow(n, -0.2); // Silverman's rule
 
  function gaussianKernel(u) {
    return Math.exp(-0.5 * u * u) / Math.sqrt(2 * Math.PI);
  }
 
  // Evaluate KDE at each bin midpoint, then scale to match histogram counts
  const kdeMidpoints = counts.map((_, b) => {
    const x = minVal + (b + 0.5) * binWidth;
    let density = 0;
    for (const p of prices) {
      density += gaussianKernel((x - p) / bandwidth);
    }
    density /= (n * bandwidth);
    // Scale density to histogram count units: density * n * binWidth = expected count
    return density * n * binWidth;
  });
 
  // Bar labels
  const labels = counts.map((_, b) =>
    `$${(minVal + b * binWidth).toFixed(1)}–$${(minVal + (b + 1) * binWidth).toFixed(1)}`
  );
 
  // Color bars: accent blue, slightly red-tinted near the market price
  const barColors = counts.map((_, b) => {
    const binMid = minVal + (b + 0.5) * binWidth;
    return Math.abs(binMid - currentPrice) < binWidth * 1.5
      ? 'rgba(248,113,113,0.4)'
      : 'rgba(77,151,239,0.35)';
  });
 
  const ctx = document.getElementById('hist-canvas').getContext('2d');
 
  _charts['hist'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        // Dataset 0: histogram bars
        {
          type: 'bar',
          data: counts,
          backgroundColor: barColors,
          borderColor:     barColors.map(c => c.replace(/[\d.]+\)$/, '0.8)')),
          borderWidth: 1,
          borderRadius: 2,
          order: 2,
        },
        // Dataset 1: KDE overlay line
        {
          type: 'line',
          data: kdeMidpoints,
          borderColor: 'rgba(77,151,239,0.85)',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.4,        // smooth the KDE curve visually
          fill: false,
          order: 1,            // draw on top of bars
        },
      ],
    },
    options: {
      ...CHART_DEFAULTS,
      plugins: {
        legend: { display: false },
        tooltip: {
          // Only show tooltip for the bar dataset (index 0)
          filter: item => item.datasetIndex === 0,
          callbacks: {
            title: ctx => ctx[0].label,
            label: ctx => {
              const pct = ((ctx.raw / prices.length) * 100).toFixed(1);
              return `${ctx.raw} paths · ${pct}% of simulations`;
            },
          },
          backgroundColor: '#131923',
          borderColor: '#232f42',
          borderWidth: 1,
          titleColor: '#dce6f0',
          bodyColor: '#6b7e96',
          titleFont: { family: 'DM Mono', size: 11 },
          bodyFont:  { family: 'DM Mono', size: 11 },
        },
      },
      scales: {
        ...CHART_DEFAULTS.scales,
        x: {
          ...CHART_DEFAULTS.scales.x,
          ticks: {
            ...CHART_DEFAULTS.scales.x.ticks,
            maxTicksLimit: 6,
            maxRotation: 0,
          },
        },
        y: {
          ...CHART_DEFAULTS.scales.y,
          title: {
            display: true,
            text: 'Paths',
            color: '#334155',
            font: { family: 'DM Mono', size: 10 },
          },
        },
      },
    },
    plugins: [
      // Custom plugin: vertical red dashed line at the market price bin
      {
        id: 'marketLine',
        afterDraw(chart) {
          const marketBin = Math.min(
            Math.floor((currentPrice - minVal) / binWidth),
            numBins - 1
          );
          if (marketBin < 0 || marketBin >= numBins) return;
 
          const meta = chart.getDatasetMeta(0);
          const bar  = meta.data[marketBin];
          if (!bar) return;
 
          const { ctx: c, chartArea: { top, bottom } } = chart;
          c.save();
          c.beginPath();
          c.moveTo(bar.x, top);
          c.lineTo(bar.x, bottom);
          c.strokeStyle = '#f87171';
          c.lineWidth   = 1.5;
          c.setLineDash([4, 3]);
          c.stroke();
 
          c.fillStyle = '#f87171';
          c.font      = '10px DM Mono, monospace';
          c.textAlign = 'center';
          c.fillText(`$${currentPrice.toFixed(0)}`, bar.x, top - 5);
          c.restore();
        },
      },
    ],
  });
}
 
// ---- 8B. Generic line chart for sample paths ----
// tension: 0 = straight segments (jagged, visibly volatile)
// Central path highlighted; all others very faint.
function renderLineChart(canvasId, sampleData, labels, centralIdx, formatY, yLabel) {
  destroyChart(canvasId);
 
  const datasets = sampleData.map((path, k) => {
    const isCentral = (k === centralIdx);
    return {
      data:        path,
      borderColor: isCentral ? '#4d97ef' : 'rgba(77,151,239,0.14)',
      borderWidth: isCentral ? 2 : 1,
      pointRadius: 0,
      tension:     0,      // straight line segments — no smoothing
      fill:        false,
    };
  });
 
  const ctx = document.getElementById(canvasId).getContext('2d');
 
  _charts[canvasId] = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      ...CHART_DEFAULTS,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          filter: item => item.datasetIndex === centralIdx,
          callbacks: {
            label: ctx => `Central: ${formatY(ctx.raw)}`,
          },
          backgroundColor: '#131923',
          borderColor: '#232f42',
          borderWidth: 1,
          titleColor: '#dce6f0',
          bodyColor:  '#4d97ef',
          titleFont:  { family: 'DM Mono', size: 11 },
          bodyFont:   { family: 'DM Mono', size: 11 },
        },
      },
      scales: {
        ...CHART_DEFAULTS.scales,
        y: {
          ...CHART_DEFAULTS.scales.y,
          title: {
            display: true, text: yLabel,
            color: '#334155', font: { family: 'DM Mono', size: 10 },
          },
        },
      },
    },
  });
}
 
// ---- 8C. Render all three path charts ----
function renderPathCharts(samplePaths, T) {
  const years = Array.from({ length: T }, (_, i) => `Yr ${i + 1}`);
  const { centralIdx } = samplePaths;
 
  renderLineChart(
    'commodity-canvas', samplePaths.commodity, years, centralIdx,
    v => `$${v.toFixed(1)}/bbl`, '$/bbl'
  );
 
  renderLineChart(
    'fcf-canvas', samplePaths.fcf, years, centralIdx,
    v => `$${v.toFixed(0)}M`, '$M'
  );
 
  renderLineChart(
    'wacc-canvas', samplePaths.wacc, years, centralIdx,
    v => `${(v * 100).toFixed(2)}%`, 'WACC'
  );
}
 
 
// =============================================================
// 9. SUMMARY CARDS
// =============================================================
 
function renderSummary(stats, currentPrice) {
  const grid = document.getElementById('summary-grid');
  const probPct = (stats.probBelowMarket * 100).toFixed(1);
  const probClass = stats.probBelowMarket > 0.5 ? 'red' : 'green';
 
  grid.innerHTML = `
    <div class="summary-cell highlight">
      <span class="cell-label">Mean Price</span>
      <span class="cell-value accent">$${stats.mean.toFixed(2)}</span>
      <span class="cell-sub">avg across all paths</span>
    </div>
    <div class="summary-cell">
      <span class="cell-label">Median Price</span>
      <span class="cell-value">$${stats.median.toFixed(2)}</span>
      <span class="cell-sub">50th percentile</span>
    </div>
    <div class="summary-cell">
      <span class="cell-label">P5 — P95</span>
      <span class="cell-value">$${stats.p5.toFixed(2)}</span>
      <span class="cell-sub">to $${stats.p95.toFixed(2)}</span>
    </div>
    <div class="summary-cell">
      <span class="cell-label">P25 — P75</span>
      <span class="cell-value">$${stats.p25.toFixed(2)}</span>
      <span class="cell-sub">to $${stats.p75.toFixed(2)}</span>
    </div>
    <div class="summary-cell">
      <span class="cell-label">Market Price</span>
      <span class="cell-value">$${currentPrice.toFixed(2)}</span>
      <span class="cell-sub">current</span>
    </div>
    <div class="summary-cell ${stats.probBelowMarket > 0.5 ? 'danger' : ''}">
      <span class="cell-label">Prob Below Market</span>
      <span class="cell-value ${probClass}">${probPct}%</span>
      <span class="cell-sub">of paths below current price</span>
    </div>
  `;
}
 
 
// =============================================================
// 10. READ INPUTS
//
// Reads all form fields into a plain object.
// Random seed: blank → fresh random each run.
// =============================================================
 
function readInputs() {
  const get    = id => parseFloat(document.getElementById(id).value);
  const getInt = id => parseInt(document.getElementById(id).value, 10);
  const getSeed = () => {
    const v = document.getElementById('random_seed').value.trim();
    // Blank seed → pick a fresh random integer each run
    return v === '' ? Math.floor(Math.random() * 1e9) : parseInt(v, 10);
  };
 
  return {
    // State variables (simple)
    commodity_current:          get('commodity_current'),
    commodity_vol:              get('commodity_vol'),
    commodity_long_run_mean:    get('commodity_long_run_mean'),
    rate_current:               get('rate_current'),
    demand_growth:              get('demand_growth'),
 
    // State variables (advanced)
    commodity_reversion_speed:  get('commodity_reversion_speed'),
    rate_drift:                 get('rate_drift'),
    rate_vol:                   get('rate_vol'),
    rate_long_run_mean:         get('rate_long_run_mean'),
    demand_current:             get('demand_current'),
    demand_vol:                 get('demand_vol'),
 
    // Company (simple)
    current_revenue:            get('current_revenue'),
    current_production:         get('current_production'),
    base_operating_margin:      get('base_operating_margin'),
    capex_percent_revenue:      get('capex_percent_revenue'),
    da_percent_revenue:         get('da_percent_revenue'),
    working_capital_percent_revenue: get('working_capital_percent_revenue'),
    shares_outstanding:         get('shares_outstanding'),
    net_debt:                   get('net_debt'),
    current_stock_price:        get('current_stock_price'),
 
    // Sensitivities (advanced)
    revenue_sensitivity_to_commodity: get('revenue_sensitivity_to_commodity'),
    revenue_sensitivity_to_demand:    get('revenue_sensitivity_to_demand'),
    margin_sensitivity_to_commodity:  get('margin_sensitivity_to_commodity'),
    cash_breakeven_price:             get('cash_breakeven_price'),
    below_breakeven_multiplier:       get('below_breakeven_multiplier'),
    structural_margin_floor:          get('structural_margin_floor'),
    capex_sensitivity_to_commodity:   get('capex_sensitivity_to_commodity'),
    production_growth_sensitivity_to_reinvestment: get('production_growth_sensitivity_to_reinvestment'),
 
    // Valuation (advanced)
    tax_rate:             get('tax_rate'),
    beta:                 get('beta'),
    equity_risk_premium:  get('equity_risk_premium'),
    debt_spread:          get('debt_spread'),
    debt_weight:          get('debt_weight'),
    equity_weight:        get('equity_weight'),
    terminal_growth:      get('terminal_growth'),
    terminal_smoothing_years: getInt('terminal_smoothing_years'),
 
    // Simulation
    num_simulations: Math.min(getInt('num_simulations'), 5000),
    forecast_years:  getInt('forecast_years'),
    random_seed:     getSeed(),
  };
}
 
 
// =============================================================
// 11. MAIN ENTRY POINT: runModel()
//
// Called by the "Run Simulation" button.
// Runs the full pipeline, then renders all outputs.
// =============================================================
 
function runModel() {
  const btn    = document.getElementById('run-btn');
  const status = document.getElementById('run-status');
 
  btn.disabled = true;
  status.textContent = 'Running…';
 
  // Small delay lets the browser repaint (show "Running…") before heavy work
  setTimeout(() => {
    try {
      const inputs = readInputs();
 
      // Set up seeded PRNG
      const rand  = makePRNG(inputs.random_seed);
      const randn = makeNormalSampler(rand);
 
      // Pipeline
      const statePaths     = simulateStatePaths(inputs, randn);
      const operatingPaths = buildOperatingPaths(statePaths, inputs);
      const waccPaths      = buildWaccPaths(statePaths, inputs);
      const prices         = discountValuationPaths(operatingPaths, waccPaths, inputs);
      const stats          = summarizeResults(prices, inputs.current_stock_price);
      const samplePaths    = selectSamplePaths(statePaths, operatingPaths, waccPaths, prices, 15);
 
      // Render
      renderSummary(stats, inputs.current_stock_price);
      renderHistogram(prices, inputs.current_stock_price);
      renderPathCharts(samplePaths, inputs.forecast_years);
 
      // Update meta line and show results
      document.getElementById('results-meta').textContent =
        `${inputs.num_simulations.toLocaleString()} paths · seed ${inputs.random_seed}`;
 
      const resultsSection = document.getElementById('results-section');
      resultsSection.style.display = 'block';
      resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
 
      status.textContent = 'Done';
 
    } catch (err) {
      status.textContent = 'Error: ' + err.message;
      console.error(err);
    }
 
    btn.disabled = false;
  }, 30);
}
 
 
// =============================================================
// 12. UTILITIES
// =============================================================
 
function zeros2D(rows, cols) {
  return Array.from({ length: rows }, () => new Array(cols).fill(0));
}
 
 
// =============================================================
// INIT — runs once when the page loads
// =============================================================
 
document.addEventListener('DOMContentLoaded', () => {
  initHelpSystem();
});
