/* portfolio.js — Paper Portfolio System with localStorage */

/* ============================================================
   DATA MODEL
   State stored in localStorage as:
   - 'opr_portfolio': { cash, transactions: [], history: [] }
   ============================================================ */

const STORAGE_KEY   = 'opr_portfolio';
const STARTING_CASH = 100000; // Default starting capital

let priceMap = {}; // Loaded from portfolio_prices.json
let portfolio = loadPortfolio();

/* --- Initialize --- */
document.addEventListener('DOMContentLoaded', async () => {
  // Load current prices
  const priceData = await fetchJSON('data/portfolio_prices.json');
  if (priceData?.prices) {
    priceMap = priceData.prices;
  } else {
    showToast('Price data unavailable — using last known prices.', 'error');
  }

  // Seed with sample trades if brand new
  if (!portfolio.bootstrapped) {
    seedSamplePortfolio();
  }

  // Render everything
  renderAll();

  // Wire up form
  document.getElementById('addTradeBtn').addEventListener('click', handleAddTrade);
  document.getElementById('exportBtn').addEventListener('click', handleExport);
  document.getElementById('importBtn').addEventListener('click', handleImport);
  document.getElementById('importFile').addEventListener('change', handleImportFile);
  document.getElementById('resetBtn').addEventListener('click', handleReset);
});

/* ============================================================
   PORTFOLIO STORAGE
   ============================================================ */

function loadPortfolio() {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) return JSON.parse(stored);
  } catch (e) {
    console.warn('Failed to parse portfolio:', e);
  }
  return { cash: STARTING_CASH, transactions: [], history: [], bootstrapped: false };
}

function savePortfolio() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(portfolio));
  } catch (e) {
    showToast('Storage error — portfolio may not be saved.', 'error');
  }
}

/* ============================================================
   SAMPLE SEED DATA
   ============================================================ */

function seedSamplePortfolio() {
  const seedTrades = [
    { date: '2026-01-15', ticker: 'NVDA', side: 'buy',  shares: 10,  price: 780.00, note: 'AI cycle position' },
    { date: '2026-01-20', ticker: 'JPM',  side: 'buy',  shares: 25,  price: 188.50, note: 'Bank earnings play' },
    { date: '2026-02-03', ticker: 'GLD',  side: 'buy',  shares: 40,  price: 216.80, note: 'Stagflation hedge' },
    { date: '2026-02-14', ticker: 'MSFT', side: 'buy',  shares: 15,  price: 395.20, note: 'Azure re-acceleration' },
    { date: '2026-03-01', ticker: 'LLY',  side: 'buy',  shares: 5,   price: 690.00, note: 'GLP-1 secular trend' },
    { date: '2026-03-10', ticker: 'NVDA', side: 'buy',  shares: 5,   price: 825.00, note: 'Add on dip' },
    { date: '2026-03-20', ticker: 'NKE',  side: 'buy',  shares: 30,  price: 89.00,  note: 'Value play — China recovery' },
    { date: '2026-04-01', ticker: 'NKE',  side: 'sell', shares: 15,  price: 82.00,  note: 'Cut on China miss' },
    { date: '2026-04-10', ticker: 'OXY',  side: 'buy',  shares: 50,  price: 63.00,  note: 'Permian E&P exposure' },
    { date: '2026-04-15', ticker: 'JPM',  side: 'buy',  shares: 10,  price: 195.00, note: 'Add before earnings' },
  ];

  seedTrades.forEach(t => applyTransaction(t, false));
  portfolio.bootstrapped = true;

  // Build initial history from seed
  rebuildHistory();
  savePortfolio();
}

/* ============================================================
   PORTFOLIO LOGIC
   ============================================================ */

/**
 * Apply a single transaction to portfolio state.
 * If saveAfter = true, persist to storage.
 */
function applyTransaction(tx, saveAfter = true) {
  const trade = {
    id:     Date.now() + Math.random(),
    date:   tx.date,
    ticker: tx.ticker.toUpperCase().trim(),
    side:   tx.side,
    shares: parseFloat(tx.shares),
    price:  parseFloat(tx.price),
    note:   tx.note || ''
  };

  const cost = trade.shares * trade.price;

  if (trade.side === 'buy') {
    if (portfolio.cash < cost) {
      showToast('Insufficient cash for this trade.', 'error');
      return false;
    }
    portfolio.cash -= cost;
  } else {
    // sell
    portfolio.cash += cost;
  }

  portfolio.transactions.push(trade);

  // Snapshot history
  const totalVal = computeTotalValue();
  portfolio.history.push({
    date:  trade.date,
    value: totalVal + portfolio.cash
  });

  if (saveAfter) savePortfolio();
  return true;
}

/**
 * Compute all current holdings from transaction history.
 * Returns array of { ticker, shares, avgCost, totalCost }
 */
function computeHoldings() {
  const positions = {}; // ticker -> { shares, totalCost }

  portfolio.transactions.forEach(tx => {
    if (!positions[tx.ticker]) {
      positions[tx.ticker] = { shares: 0, totalCost: 0 };
    }
    const pos = positions[tx.ticker];

    if (tx.side === 'buy') {
      pos.totalCost += tx.shares * tx.price;
      pos.shares    += tx.shares;
    } else {
      // Reduce on average cost basis
      const avgCost  = pos.shares > 0 ? pos.totalCost / pos.shares : 0;
      const costReduction = avgCost * tx.shares;
      pos.totalCost  = Math.max(0, pos.totalCost - costReduction);
      pos.shares     = Math.max(0, pos.shares - tx.shares);
    }
  });

  return Object.entries(positions)
    .filter(([, pos]) => pos.shares > 0.0001)
    .map(([ticker, pos]) => ({
      ticker,
      shares:    pos.shares,
      avgCost:   pos.shares > 0 ? pos.totalCost / pos.shares : 0,
      totalCost: pos.totalCost
    }));
}

/**
 * Compute total market value of all holdings (no cash).
 */
function computeTotalValue() {
  const holdings = computeHoldings();
  return holdings.reduce((sum, h) => {
    const price = priceMap[h.ticker] || h.avgCost;
    return sum + h.shares * price;
  }, 0);
}

/**
 * Rebuild the portfolio history from scratch (used after seeding).
 */
function rebuildHistory() {
  portfolio.history = [];
  // Walk transactions in order
  const sorted = [...portfolio.transactions].sort((a, b) => a.date.localeCompare(b.date));
  let runningCash = STARTING_CASH;
  const runningPos = {};

  sorted.forEach(tx => {
    const cost = tx.shares * tx.price;
    if (tx.side === 'buy') {
      runningCash -= cost;
      if (!runningPos[tx.ticker]) runningPos[tx.ticker] = { shares: 0, totalCost: 0 };
      runningPos[tx.ticker].shares    += tx.shares;
      runningPos[tx.ticker].totalCost += cost;
    } else {
      runningCash += cost;
      if (runningPos[tx.ticker]) {
        runningPos[tx.ticker].shares = Math.max(0, runningPos[tx.ticker].shares - tx.shares);
      }
    }

    const equityVal = Object.entries(runningPos).reduce((sum, [ticker, pos]) => {
      const p = priceMap[ticker] || (pos.shares > 0 ? pos.totalCost / pos.shares : 0);
      return sum + pos.shares * p;
    }, 0);

    portfolio.history.push({ date: tx.date, value: runningCash + equityVal });
  });
}

/* ============================================================
   RENDERING
   ============================================================ */

function renderAll() {
  renderMetrics();
  renderChart();
  renderHoldingsTable();
  renderLedger();
}

/* --- Summary Metrics --- */
function renderMetrics() {
  const holdings   = computeHoldings();
  const equityVal  = holdings.reduce((sum, h) => {
    return sum + h.shares * (priceMap[h.ticker] || h.avgCost);
  }, 0);
  const totalValue = equityVal + portfolio.cash;
  const totalCost  = holdings.reduce((sum, h) => sum + h.totalCost, 0);
  const unrealPnL  = equityVal - totalCost;
  const returnPct  = totalCost > 0 ? (unrealPnL / STARTING_CASH) * 100 : 0;

  // Daily P&L: compare last two history entries
  let dailyPnL = 0;
  if (portfolio.history.length >= 2) {
    const last    = portfolio.history[portfolio.history.length - 1].value;
    const prevDay = portfolio.history[portfolio.history.length - 2].value;
    dailyPnL = last - prevDay;
  }

  setMetric('metricTotal',  formatCurrency(totalValue));
  setMetric('metricCash',   formatCurrency(portfolio.cash));
  setMetric('metricDaily',  formatCurrency(dailyPnL), dailyPnL >= 0 ? 'up' : 'down');
  setMetric('metricPnL',    formatCurrency(unrealPnL), unrealPnL >= 0 ? 'up' : 'down');
  setMetric('metricReturn', formatPct(returnPct),       returnPct >= 0 ? 'up' : 'down');
}

function setMetric(id, value, colorClass = '') {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = value;
  el.className = 'metric-value' + (colorClass ? ' ' + colorClass : '');
}

/* --- Portfolio Chart --- */
function renderChart() {
  const canvas = document.getElementById('portfolioChart');
  if (!canvas) return;

  const history = portfolio.history;
  if (history.length < 2) {
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#435570';
    ctx.font = '13px IBM Plex Mono, monospace';
    ctx.textAlign = 'center';
    canvas.width  = canvas.offsetWidth || 800;
    canvas.height = 240;
    ctx.fillText('Add trades to see portfolio performance chart.', canvas.width / 2, 120);
    return;
  }

  // Set canvas size
  canvas.width  = canvas.offsetWidth || canvas.parentElement.offsetWidth || 800;
  canvas.height = 240;

  const ctx    = canvas.getContext('2d');
  const W      = canvas.width;
  const H      = canvas.height;
  const padL   = 72;
  const padR   = 24;
  const padT   = 20;
  const padB   = 36;
  const chartW = W - padL - padR;
  const chartH = H - padT - padB;

  const values = history.map(h => h.value);
  const labels = history.map(h => h.date);
  const minVal = Math.min(...values) * 0.98;
  const maxVal = Math.max(...values) * 1.02;
  const range  = maxVal - minVal || 1;

  const xScale = i => padL + (i / (values.length - 1)) * chartW;
  const yScale = v => padT + chartH - ((v - minVal) / range) * chartH;

  // Clear
  ctx.clearRect(0, 0, W, H);

  // Grid lines (horizontal)
  ctx.strokeStyle = '#1e293b';
  ctx.lineWidth   = 1;
  const gridLines = 4;
  for (let i = 0; i <= gridLines; i++) {
    const y = padT + (i / gridLines) * chartH;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(W - padR, y);
    ctx.stroke();

    // Y axis labels
    const val = maxVal - (i / gridLines) * range;
    ctx.fillStyle = '#435570';
    ctx.font      = '10px IBM Plex Mono, monospace';
    ctx.textAlign = 'right';
    ctx.fillText('$' + Math.round(val).toLocaleString(), padL - 8, y + 4);
  }

  // Determine line color (up or down overall)
  const isUp = values[values.length - 1] >= values[0];
  const lineColor = isUp ? '#22c55e' : '#ef4444';
  const fillColor = isUp ? 'rgba(34,197,94,0.08)' : 'rgba(239,68,68,0.08)';

  // Fill area
  ctx.beginPath();
  ctx.moveTo(xScale(0), yScale(values[0]));
  values.forEach((v, i) => ctx.lineTo(xScale(i), yScale(v)));
  ctx.lineTo(xScale(values.length - 1), padT + chartH);
  ctx.lineTo(padL, padT + chartH);
  ctx.closePath();
  ctx.fillStyle = fillColor;
  ctx.fill();

  // Line
  ctx.beginPath();
  ctx.moveTo(xScale(0), yScale(values[0]));
  values.forEach((v, i) => ctx.lineTo(xScale(i), yScale(v)));
  ctx.strokeStyle  = lineColor;
  ctx.lineWidth    = 2;
  ctx.lineJoin     = 'round';
  ctx.shadowColor  = lineColor;
  ctx.shadowBlur   = 6;
  ctx.stroke();
  ctx.shadowBlur   = 0;

  // X axis labels (show a subset)
  ctx.fillStyle = '#435570';
  ctx.font      = '10px IBM Plex Mono, monospace';
  ctx.textAlign = 'center';
  const step = Math.max(1, Math.floor(labels.length / 5));
  labels.forEach((label, i) => {
    if (i % step === 0 || i === labels.length - 1) {
      ctx.fillText(label, xScale(i), H - 8);
    }
  });

  // Current value label (top right)
  const lastVal = values[values.length - 1];
  ctx.fillStyle  = lineColor;
  ctx.font       = 'bold 11px IBM Plex Mono, monospace';
  ctx.textAlign  = 'right';
  ctx.fillText('$' + Math.round(lastVal).toLocaleString(), W - padR, padT + 14);
}

/* --- Holdings Table --- */
function renderHoldingsTable() {
  const holdings  = computeHoldings();
  const totalVal  = computeTotalValue() + portfolio.cash;
  const tbody     = document.getElementById('holdingsTbody');
  if (!tbody) return;

  if (!holdings.length) {
    tbody.innerHTML = `<tr><td colspan="7" class="empty-state">No open positions.</td></tr>`;
    return;
  }

  tbody.innerHTML = holdings.map(h => {
    const price    = priceMap[h.ticker] || h.avgCost;
    const mktVal   = h.shares * price;
    const unrealPnL = mktVal - h.totalCost;
    const weight   = totalVal > 0 ? (mktVal / totalVal) * 100 : 0;
    const pnlClass = unrealPnL >= 0 ? 'up' : 'down';

    return `
      <tr>
        <td class="mono strong">${h.ticker}</td>
        <td class="mono">${h.shares.toFixed(2)}</td>
        <td class="mono">${formatCurrency(h.avgCost)}</td>
        <td class="mono">${formatCurrency(price)}</td>
        <td class="mono">${formatCurrency(mktVal)}</td>
        <td class="mono ${pnlClass}">${formatCurrency(unrealPnL)} (${formatPct((unrealPnL / h.totalCost) * 100)})</td>
        <td class="mono">${weight.toFixed(1)}%</td>
      </tr>
    `;
  }).join('');
}

/* --- Trade Ledger --- */
function renderLedger() {
  const tbody = document.getElementById('ledgerTbody');
  if (!tbody) return;

  const txns = [...portfolio.transactions].reverse();

  if (!txns.length) {
    tbody.innerHTML = `<tr><td colspan="6" class="empty-state">No trades recorded.</td></tr>`;
    return;
  }

  tbody.innerHTML = txns.map(tx => `
    <tr>
      <td class="mono">${tx.date}</td>
      <td class="mono strong">${tx.ticker}</td>
      <td><span class="side-${tx.side}">${tx.side.toUpperCase()}</span></td>
      <td class="mono">${tx.shares}</td>
      <td class="mono">${formatCurrency(tx.price)}</td>
      <td style="color:var(--text-muted);font-size:0.78rem;">${tx.note || '—'}</td>
    </tr>
  `).join('');
}

/* ============================================================
   FORM HANDLERS
   ============================================================ */

function handleAddTrade() {
  const ticker = document.getElementById('tradeTicker').value.trim().toUpperCase();
  const side   = document.getElementById('tradeSide').value;
  const shares = parseFloat(document.getElementById('tradeShares').value);
  const price  = parseFloat(document.getElementById('tradePrice').value);
  const date   = document.getElementById('tradeDate').value;
  const note   = document.getElementById('tradeNote').value.trim();

  if (!ticker || !side || isNaN(shares) || shares <= 0 || isNaN(price) || price <= 0 || !date) {
    showToast('Please fill in all required trade fields.', 'error');
    return;
  }

  // Validate sell against holdings
  if (side === 'sell') {
    const holdings = computeHoldings();
    const pos = holdings.find(h => h.ticker === ticker);
    if (!pos || pos.shares < shares) {
      showToast(`Cannot sell ${shares} shares of ${ticker} — insufficient holdings.`, 'error');
      return;
    }
  }

  const ok = applyTransaction({ ticker, side, shares, price, date, note }, true);
  if (ok) {
    showToast(`${side.toUpperCase()} ${shares} ${ticker} @ ${formatCurrency(price)} recorded.`, 'success');
    clearForm();
    renderAll();
  }
}

function clearForm() {
  ['tradeTicker','tradeShares','tradePrice','tradeNote'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.value = '';
  });
  // Reset date to today
  const dateEl = document.getElementById('tradeDate');
  if (dateEl) dateEl.value = new Date().toISOString().split('T')[0];
}

function handleExport() {
  const blob = new Blob([JSON.stringify(portfolio, null, 2)], { type: 'application/json' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = `opr_portfolio_${new Date().toISOString().split('T')[0]}.json`;
  a.click();
  URL.revokeObjectURL(url);
  showToast('Portfolio exported.', 'success');
}

function handleImport() {
  document.getElementById('importFile').click();
}

function handleImportFile(e) {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (ev) => {
    try {
      const imported = JSON.parse(ev.target.result);
      if (!imported.transactions || !Array.isArray(imported.transactions)) {
        throw new Error('Invalid portfolio format');
      }
      portfolio = imported;
      savePortfolio();
      renderAll();
      showToast('Portfolio imported successfully.', 'success');
    } catch (err) {
      showToast('Import failed: ' + err.message, 'error');
    }
  };
  reader.readAsText(file);
  e.target.value = ''; // reset input
}

function handleReset() {
  if (!confirm('Reset portfolio? This will clear all trades and cannot be undone.')) return;
  portfolio = { cash: STARTING_CASH, transactions: [], history: [], bootstrapped: true };
  savePortfolio();
  renderAll();
  showToast('Portfolio reset.', 'default');
}
