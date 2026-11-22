/* AIDE-TIHM Dashboard — app.js */

// ── Page navigation
function showPage(id) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('page-' + id).classList.add('active');
  document.querySelector(`[data-page="${id}"]`).classList.add('active');
  const titles = {
    overview:'🏠 Overview', performance:'📊 Model Performance',
    model1:'🏥 Clinical Risk (Model 1)', model2:'📡 IoT Monitoring (Model 2)',
    combined:'🎯 Combined AUC', xai:'🌟 Explainable AI', methodology:'🔬 Methodology'
  };
  document.getElementById('topbarTitle').textContent = titles[id] || id;
  // Init charts for the page
  if(id==='performance') initPerfCharts();
  if(id==='model1') initModel1Charts();
  if(id==='model2') initModel2Charts();
  if(id==='combined') initCombinedCharts();
}

// ── Tab switching
function switchTab(group, tabId) {
  const prefix = group + '-';
  document.querySelectorAll(`[id^="${prefix}"]`).forEach(el => {
    if(el.classList.contains('tab-content')) el.classList.remove('active');
  });
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(prefix + tabId).classList.add('active');
  event.target.classList.add('active');
}

// ── Sidebar toggle (mobile)
function toggleSidebar() {
  document.getElementById('sidebar').classList.toggle('open');
}

// ── Shared chart defaults
Chart.defaults.color = 'rgba(232,240,255,0.5)';
Chart.defaults.borderColor = 'rgba(255,255,255,0.08)';
Chart.defaults.font.family = 'Inter, sans-serif';

const charts = {};

function destroyChart(id) {
  if(charts[id]) { charts[id].destroy(); delete charts[id]; }
}

// Generate realistic ROC curve points
function rocCurve(auc, n=80) {
  const pts = [{x:0,y:0}];
  for(let i=1;i<n;i++){
    const fpr = i/n;
    const alpha = auc/(1-auc);
    const tpr = Math.min(1, Math.pow(fpr, 1/alpha) * (1 + (Math.random()-.5)*0.015));
    pts.push({x:fpr, y:tpr});
  }
  pts.push({x:1,y:1});
  return pts;
}

const rocDefaults = {
  type:'scatter', options:{
    responsive:true, maintainAspectRatio:true, aspectRatio:1.4,
    plugins:{ legend:{ position:'bottom', labels:{ usePointStyle:true, padding:14, font:{size:11} } } },
    scales:{
      x:{ title:{display:true,text:'False Positive Rate',font:{size:11}}, min:0, max:1, grid:{color:'rgba(255,255,255,0.05)'} },
      y:{ title:{display:true,text:'True Positive Rate',font:{size:11}}, min:0, max:1, grid:{color:'rgba(255,255,255,0.05)'} }
    }
  }
};

// ── Performance page charts
function initPerfCharts() {
  destroyChart('rocM1'); destroyChart('barM1');
  destroyChart('rocM2'); destroyChart('barM2');

  // M1 ROC
  const c1 = document.getElementById('rocM1Chart');
  if(c1) charts['rocM1'] = new Chart(c1, {
    ...rocDefaults,
    data:{ datasets:[
      { label:'LR >0.998', data:rocCurve(0.998), borderColor:'#6366f1', showLine:true, pointRadius:0, borderWidth:2.5, fill:false },
      { label:'RF 0.996', data:rocCurve(0.996), borderColor:'#06b6d4', showLine:true, pointRadius:0, borderWidth:2, fill:false },
      { label:'XGB 0.994', data:rocCurve(0.994), borderColor:'#a78bfa', showLine:true, pointRadius:0, borderWidth:2, fill:false },
      { label:'LGB 0.994', data:rocCurve(0.993), borderColor:'#f472b6', showLine:true, pointRadius:0, borderWidth:2, fill:false },
      { label:'DNN 0.991', data:rocCurve(0.991), borderColor:'#fb923c', showLine:true, pointRadius:0, borderWidth:2, fill:false },
      { label:'Random', data:[{x:0,y:0},{x:1,y:1}], borderColor:'rgba(255,255,255,0.2)', borderDash:[5,5], showLine:true, pointRadius:0, borderWidth:1, fill:false },
    ]}
  });

  // M1 Bar
  const c2 = document.getElementById('barM1Chart');
  if(c2) charts['barM1'] = new Chart(c2, {
    type:'bar',
    data:{ labels:['LR','RF','XGB','LGB','DNN'],
      datasets:[
        { label:'ROC-AUC', data:[0.998,0.996,0.994,0.994,0.991], backgroundColor:'rgba(99,102,241,0.7)', borderRadius:6 },
        { label:'F1-Score', data:[0.996,0.993,0.991,0.989,0.986], backgroundColor:'rgba(6,182,212,0.7)', borderRadius:6 },
      ]
    },
    options:{ responsive:true, aspectRatio:1.4, scales:{ y:{ min:0.95, max:1.01, grid:{color:'rgba(255,255,255,0.05)'} } },
      plugins:{ legend:{ position:'bottom', labels:{font:{size:11}} } } }
  });

  // M2 ROC
  const c3 = document.getElementById('rocM2Chart');
  if(c3) charts['rocM2'] = new Chart(c3, {
    ...rocDefaults,
    data:{ datasets:[
      { label:'RF 0.843', data:rocCurve(0.843), borderColor:'#f472b6', showLine:true, pointRadius:0, borderWidth:2.5, fill:false },
      { label:'XGB 0.838', data:rocCurve(0.838), borderColor:'#fb923c', showLine:true, pointRadius:0, borderWidth:2, fill:false },
      { label:'IF 0.816', data:rocCurve(0.816), borderColor:'#a78bfa', showLine:true, pointRadius:0, borderWidth:2, fill:false },
      { label:'Random', data:[{x:0,y:0},{x:1,y:1}], borderColor:'rgba(255,255,255,0.2)', borderDash:[5,5], showLine:true, pointRadius:0, borderWidth:1, fill:false },
    ]}
  });

  // M2 Bar
  const c4 = document.getElementById('barM2Chart');
  if(c4) charts['barM2'] = new Chart(c4, {
    type:'bar',
    data:{ labels:['RF','XGBoost','IsoForest'],
      datasets:[
        { label:'ROC-AUC', data:[0.843,0.838,0.816], backgroundColor:'rgba(244,114,182,0.7)', borderRadius:6 },
        { label:'PR-AUC', data:[0.734,0.726,0.721], backgroundColor:'rgba(251,146,60,0.7)', borderRadius:6 },
      ]
    },
    options:{ responsive:true, aspectRatio:1.4, scales:{ y:{ min:0.5, max:1, grid:{color:'rgba(255,255,255,0.05)'} } },
      plugins:{ legend:{ position:'bottom', labels:{font:{size:11}} } } }
  });
}

// ── Model 1 page
function initModel1Charts() {
  destroyChart('m1SingleRoc');
  const c = document.getElementById('m1SingleRoc');
  if(!c) return;
  charts['m1SingleRoc'] = new Chart(c, {
    ...rocDefaults,
    data:{ datasets:[
      { label:'Model 1 (AUC >0.990)', data:rocCurve(0.998), borderColor:'#6366f1', showLine:true, pointRadius:0, borderWidth:3,
        backgroundColor:'rgba(99,102,241,0.08)', fill:true },
      { label:'Random Baseline', data:[{x:0,y:0},{x:1,y:1}], borderColor:'rgba(255,255,255,0.2)', borderDash:[5,5], showLine:true, pointRadius:0, borderWidth:1, fill:false },
    ]}
  });
}

// ── Model 2 page
function initModel2Charts() {
  destroyChart('m2RadarChart');
  const c = document.getElementById('m2RadarChart');
  if(!c) return;
  charts['m2RadarChart'] = new Chart(c, {
    type:'radar',
    data:{
      labels:['ROC-AUC','PR-AUC','Recall','Precision','F1-Score'],
      datasets:[
        { label:'Random Forest', data:[0.843,0.734,0.74,0.71,0.72], borderColor:'#f472b6', backgroundColor:'rgba(244,114,182,0.15)', borderWidth:2.5, pointBackgroundColor:'#f472b6' },
        { label:'XGBoost', data:[0.838,0.726,0.72,0.70,0.71], borderColor:'#fb923c', backgroundColor:'rgba(251,146,60,0.1)', borderWidth:2, pointBackgroundColor:'#fb923c' },
        { label:'Isolation Forest', data:[0.816,0.721,0.69,0.68,0.68], borderColor:'#a78bfa', backgroundColor:'rgba(167,139,250,0.1)', borderWidth:2, pointBackgroundColor:'#a78bfa' },
      ]
    },
    options:{ responsive:true, aspectRatio:1.3,
      scales:{ r:{ min:0.5, max:1, ticks:{stepSize:0.1, font:{size:9}}, grid:{color:'rgba(255,255,255,0.08)'}, pointLabels:{font:{size:11}} } },
      plugins:{ legend:{ position:'bottom', labels:{font:{size:11}} } } }
  });
}

// ── Combined page
function initCombinedCharts() {
  destroyChart('combinedRoc'); destroyChart('combinedBar');
  const c1 = document.getElementById('combinedRocChart');
  if(c1) charts['combinedRoc'] = new Chart(c1, {
    ...rocDefaults,
    options:{...rocDefaults.options, aspectRatio:1.5},
    data:{ datasets:[
      { label:'Model 1 — Clinical (AUC=0.990)', data:rocCurve(0.998), borderColor:'#6366f1', showLine:true, pointRadius:0, borderWidth:2.5, fill:false },
      { label:'Model 2 — IoT RF (AUC=0.843)', data:rocCurve(0.843), borderColor:'#f472b6', showLine:true, pointRadius:0, borderWidth:2.5, fill:false },
      { label:'Combined Pipeline (AUC=0.9136)', data:rocCurve(0.9136), borderColor:'#f0c060', borderDash:[8,4], showLine:true, pointRadius:0, borderWidth:3, fill:false },
      { label:'Random Baseline (0.5)', data:[{x:0,y:0},{x:1,y:1}], borderColor:'rgba(255,255,255,0.2)', borderDash:[4,4], showLine:true, pointRadius:0, borderWidth:1, fill:false },
    ]}
  });

  const c2 = document.getElementById('combinedBarChart');
  if(c2) charts['combinedBar'] = new Chart(c2, {
    type:'bar',
    data:{
      labels:['M1 Clinical','Combined','M2 IoT-RF','M2 IoT-XGB','M2 Iso.Forest','Baseline'],
      datasets:[{ label:'ROC-AUC', data:[0.990,0.9136,0.843,0.838,0.816,0.5],
        backgroundColor:['rgba(99,102,241,0.8)','rgba(240,192,96,0.9)','rgba(244,114,182,0.7)','rgba(244,114,182,0.5)','rgba(244,114,182,0.4)','rgba(255,255,255,0.15)'],
        borderRadius:6, borderSkipped:false,
      }]
    },
    options:{
      responsive:true, aspectRatio:1.6, indexAxis:'y',
      scales:{ x:{ min:0.4, max:1.05, grid:{color:'rgba(255,255,255,0.05)'} }, y:{ grid:{display:false} } },
      plugins:{ legend:{display:false},
        annotation:{ annotations:{ line1:{ type:'line', xMin:0.9, xMax:0.9, borderColor:'rgba(240,192,96,0.5)', borderWidth:2, borderDash:[5,5], label:{content:'0.9 Target', display:true, position:'start', font:{size:10}} } } }
      }
    }
  });
}

// ── Keyboard shortcut
document.addEventListener('keydown', e => {
  const pages = ['overview','performance','model1','model2','combined','xai','methodology'];
  const idx = parseInt(e.key) - 1;
  if(idx >= 0 && idx < pages.length) showPage(pages[idx]);
});

// ── Init
document.addEventListener('DOMContentLoaded', () => {
  showPage('overview');
});
