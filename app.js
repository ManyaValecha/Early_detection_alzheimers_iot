/* ============================================================
   AIDE-TIHM — Interactive Dashboard JavaScript
   ============================================================ */

// ---- Animated Background Particles ----
function initParticles() {
  const container = document.getElementById('bgParticles');
  const colors = ['#6366f1','#06b6d4','#f472b6','#fb923c','#a78bfa','#fbbf24'];
  const count = 30;
  for (let i = 0; i < count; i++) {
    const p = document.createElement('div');
    p.classList.add('particle');
    const size = Math.random() * 6 + 2;
    const color = colors[Math.floor(Math.random() * colors.length)];
    const left = Math.random() * 100;
    const duration = Math.random() * 15 + 10;
    const delay = Math.random() * -20;
    p.style.cssText = `
      width:${size}px; height:${size}px;
      background:${color};
      left:${left}%;
      animation-duration:${duration}s;
      animation-delay:${delay}s;
      filter: blur(1px);
    `;
    container.appendChild(p);
  }
}

// ---- Scroll Reveal ----
function initScrollReveal() {
  // Add scroll-reveal class to key sections
  const targets = document.querySelectorAll(
    '.section-label, .section-title, .section-desc, .model-card, .method-card, .auc-formula-card, .gauge-container, .combine-visual, .table-wrapper, .bar-chart-section, .roc-section'
  );
  targets.forEach(el => el.classList.add('scroll-reveal'));

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1, rootMargin: '0px 0px -60px 0px' });

  targets.forEach(el => observer.observe(el));
}

// ---- Gauge Animation ----
function animateGauge() {
  const arcM1 = document.getElementById('arcM1');
  const arcM2 = document.getElementById('arcM2');
  const needle = document.getElementById('gaugeNeedle');
  if (!arcM1) return;

  // Arc length for a 180-degree arc with r=120 is π*120 ≈ 377
  const totalLen = 377;
  
  // Model 1: 0.99 → near full
  const m1Pct = (0.99 - 0.5) / 0.5; // normalized to 0–1 within 0.5–1.0
  const m1Offset = totalLen * (1 - m1Pct);
  
  // Model 2: 0.843
  const m2Pct = (0.843 - 0.5) / 0.5;
  const m2Offset = totalLen * (1 - m2Pct);

  // Combined: 0.9136
  const combPct = (0.9136 - 0.5) / 0.5;
  // Needle angle: 0 at left (angle=180), 1 at right (angle=0); center at 90 for 0.5
  // Needle starts at top (270deg in SVG terms) pointing down, rotating from left to right
  // The gauge goes from -180 (left) to 0 (right), center is -90 (top)
  // For 0 gauge reading, needle points left; for 1.0, points right
  const needleAngle = -180 + combPct * 180; // degrees from vertical
  const rad = needleAngle * Math.PI / 180;
  const r = 110;
  const cx = 150, cy = 170;
  const x2 = cx + r * Math.sin(rad);
  const y2 = cy - r * Math.cos(rad);
  
  setTimeout(() => {
    arcM1.style.strokeDashoffset = m1Offset;
    arcM2.style.strokeDashoffset = m2Offset;
    needle.setAttribute('x2', x2);
    needle.setAttribute('y2', y2);
  }, 600);
}

// ---- ROC Curve Canvas ----
function drawROCCurve() {
  const canvas = document.getElementById('rocCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const pad = { top: 30, right: 30, bottom: 50, left: 60 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  // Helper: data-space to canvas-space
  function toC(x, y) {
    return [pad.left + x * plotW, pad.top + (1 - y) * plotH];
  }

  // Clear
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = 'rgba(13,21,38,0)';
  ctx.fillRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = 'rgba(255,255,255,0.07)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const x = pad.left + (i/5)*plotW, y = pad.top + (i/5)*plotH;
    ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top + plotH); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + plotW, y); ctx.stroke();
  }

  // Axis labels
  ctx.fillStyle = 'rgba(255,255,255,0.4)';
  ctx.font = '12px Inter, sans-serif';
  ctx.textAlign = 'center';
  for (let i = 0; i <= 5; i++) {
    const v = (i/5).toFixed(1);
    const [lx] = toC(i/5, 0); ctx.fillText(v, lx, pad.top + plotH + 20);
    const [,,] = [0,0,0];
    ctx.textAlign = 'right';
    const [, ly] = toC(0, i/5); ctx.fillText(v, pad.left - 8, ly + 4);
    ctx.textAlign = 'center';
  }
  ctx.fillStyle = 'rgba(255,255,255,0.5)';
  ctx.font = '13px Inter, sans-serif';
  ctx.fillText('False Positive Rate', pad.left + plotW/2, H - 10);
  ctx.save(); ctx.translate(16, pad.top + plotH/2); ctx.rotate(-Math.PI/2);
  ctx.fillText('True Positive Rate', 0, 0); ctx.restore();

  // Random baseline
  ctx.strokeStyle = 'rgba(255,255,255,0.2)';
  ctx.lineWidth = 1; ctx.setLineDash([6,4]);
  ctx.beginPath();
  ctx.moveTo(...toC(0,0)); ctx.lineTo(...toC(1,1));
  ctx.stroke(); ctx.setLineDash([]);

  // 0.9+ target threshold marker
  ctx.strokeStyle = 'rgba(240,192,96,0.3)';
  ctx.lineWidth = 1; ctx.setLineDash([4,4]);
  ctx.fillStyle = 'rgba(240,192,96,0.5)';
  ctx.font = '11px Inter';
  ctx.textAlign = 'left';
  ctx.setLineDash([]); 

  // Generate realistic ROC curves using Beta distribution approximation
  function generateROC(auc, nPoints = 200) {
    const pts = [[0,0]];
    for (let i = 1; i < nPoints; i++) {
      const fpr = i / nPoints;
      // Approximate: TPR = fpr^((1-auc)/auc) with some variation
      const alpha = auc / (1 - auc);
      const tpr = Math.min(1, Math.pow(fpr, 1/alpha) * (1 + (Math.random()-0.5)*0.02));
      pts.push([fpr, tpr]);
    }
    pts.push([1,1]);
    return pts;
  }

  const rocM1 = generateROC(0.99);
  const rocM2 = generateROC(0.843);
  const rocComb = generateROC(0.9136);

  // Draw curve helper
  function drawCurve(pts, colors, lineWidth=3) {
    const grad = ctx.createLinearGradient(pad.left, 0, pad.left+plotW, 0);
    colors.forEach((c, i) => grad.addColorStop(i/(colors.length-1), c));
    ctx.strokeStyle = grad;
    ctx.lineWidth = lineWidth;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    pts.forEach(([x,y], i) => {
      const [cx,cy] = toC(x,y);
      if (i===0) ctx.moveTo(cx,cy); else ctx.lineTo(cx,cy);
    });
    ctx.stroke();
  }

  // Fill area under curve
  function fillArea(pts, color) {
    ctx.beginPath();
    pts.forEach(([x,y], i) => {
      const [cx,cy] = toC(x,y);
      if (i===0) ctx.moveTo(cx,cy); else ctx.lineTo(cx,cy);
    });
    ctx.lineTo(...toC(1,0));
    ctx.lineTo(...toC(0,0));
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
  }

  // Draw in order: fills first, then lines
  fillArea(rocM2, 'rgba(244,114,182,0.06)');
  fillArea(rocM1, 'rgba(99,102,241,0.06)');
  fillArea(rocComb, 'rgba(240,192,96,0.06)');

  drawCurve(rocM2, ['#f472b6','#fb923c'], 2.5);
  drawCurve(rocM1, ['#818cf8','#06b6d4'], 3);
  ctx.setLineDash([8,4]);
  drawCurve(rocComb, ['#fbbf24','#f0c060'], 3);
  ctx.setLineDash([]);

  // AUC labels on curve
  function aucLabel(pts, label, auc, color) {
    const midIdx = Math.floor(pts.length * 0.4);
    const [x,y] = toC(pts[midIdx][0], pts[midIdx][1]);
    ctx.fillStyle = color;
    ctx.font = 'bold 12px Inter';
    ctx.textAlign = 'left';
    ctx.fillText(`${label} (AUC=${auc})`, x + 6, y - 6);
  }
  aucLabel(rocM1, 'M1', '0.990', '#818cf8');
  aucLabel(rocM2, 'M2', '0.843', '#f472b6');
  aucLabel(rocComb, 'Combined', '0.9136', '#fbbf24');
}

// ---- Bar Chart Animation ----
function animateBars() {
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const fills = entry.target.querySelectorAll('.bar-fill');
        fills.forEach(fill => {
          const target = fill.getAttribute('data-width');
          fill.style.width = target;
        });
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.2 });

  const barChart = document.getElementById('barChart');
  if (barChart) observer.observe(barChart);
}

// ---- Metric Bar Animation ----
function animateMetricBars() {
  document.querySelectorAll('.metric-bar').forEach(bar => {
    const pct = getComputedStyle(bar).getPropertyValue('--pct');
    bar.style.setProperty('--pct', pct);
  });
}

// ---- Gauge observer ----
function initGaugeObserver() {
  const gauge = document.getElementById('gaugeChart');
  if (!gauge) return;
  const obs = new IntersectionObserver(entries => {
    if (entries[0].isIntersecting) {
      animateGauge();
      obs.unobserve(gauge);
    }
  }, { threshold: 0.4 });
  obs.observe(gauge);
}

// ---- ROC observer ----
function initROCObserver() {
  const canvas = document.getElementById('rocCanvas');
  if (!canvas) return;
  const obs = new IntersectionObserver(entries => {
    if (entries[0].isIntersecting) {
      drawROCCurve();
      obs.unobserve(canvas);
    }
  }, { threshold: 0.3 });
  obs.observe(canvas);
}

// ---- Smooth nav ----
function initNav() {
  document.querySelectorAll('a[href^="#"]').forEach(link => {
    link.addEventListener('click', e => {
      const target = document.querySelector(link.getAttribute('href'));
      if (target) { e.preventDefault(); target.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
    });
  });
}

// ---- Number counter animation ----
function animateCounter(el, target, duration = 1500, decimals = 4) {
  const start = Date.now();
  const startVal = 0;
  function update() {
    const elapsed = Date.now() - start;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const val = startVal + (target - startVal) * eased;
    el.textContent = val.toFixed(decimals);
    if (progress < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}

// ---- Stat chips counter ----
function initCounters() {
  const chips = document.querySelectorAll('.stat-num');
  const obs = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const el = entry.target;
        const text = el.textContent;
        if (text === '0.9136') animateCounter(el, 0.9136, 1500, 4);
        else if (text === '0.843') animateCounter(el, 0.843, 1500, 3);
        else if (text === '28') {
          let v = 0;
          const t = setInterval(() => { v = Math.min(v + 1, 28); el.textContent = v; if (v >= 28) clearInterval(t); }, 50);
        }
        obs.unobserve(el);
      }
    });
  }, { threshold: 0.5 });
  chips.forEach(c => obs.observe(c));
}

// ---- Init ----
document.addEventListener('DOMContentLoaded', () => {
  initParticles();
  initScrollReveal();
  initGaugeObserver();
  initROCObserver();
  animateBars();
  animateMetricBars();
  initNav();
  initCounters();
});
