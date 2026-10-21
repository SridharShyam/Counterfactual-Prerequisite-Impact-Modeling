// CFLP Frontend Research Interaction Controller
document.addEventListener('DOMContentLoaded', () => {
  // Theme Management
  const themeToggleBtn = document.getElementById('themeToggleBtn');
  const themeIcon = document.getElementById('themeIcon');
  const themeText = document.getElementById('themeText');

  const currentTheme = localStorage.getItem('cflp-theme') || 'light';
  applyTheme(currentTheme);

  themeToggleBtn.addEventListener('click', () => {
    const active = document.documentElement.getAttribute('data-theme');
    const target = active === 'light' ? 'dark' : 'light';
    applyTheme(target);
  });

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('cflp-theme', theme);
    if (theme === 'dark') {
      themeIcon.textContent = '🌙';
      themeText.textContent = 'Light Theme';
    } else {
      themeIcon.textContent = '☀️';
      themeText.textContent = 'Dark Theme';
    }
  }

  // Modals for Privacy & Terms (Considerations 26 & 27)
  const privacyModal = document.getElementById('privacyModal');
  const termsModal = document.getElementById('termsModal');
  document.getElementById('openPrivacyBtn').addEventListener('click', () => privacyModal.classList.add('is-open'));
  document.getElementById('openTermsBtn').addEventListener('click', () => termsModal.classList.add('is-open'));

  document.querySelectorAll('.modal-close-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const modalId = e.target.getAttribute('data-close');
      const el = document.getElementById(modalId);
      if (el) el.classList.remove('is-open');
    });
  });

  window.addEventListener('click', (e) => {
    if (e.target.classList.contains('modal-overlay')) {
      e.target.classList.remove('is-open');
    }
  });

  // Intervention slider value sync
  const interventionRange = document.getElementById('interventionRange');
  const interventionValDisplay = document.getElementById('interventionValDisplay');
  interventionRange.addEventListener('input', (e) => {
    interventionValDisplay.textContent = parseFloat(e.target.value).toFixed(2);
  });

  // Concept Selector and Data Initialization
  const targetConceptSelect = document.getElementById('targetConceptSelect');
  const targetPrereqCount = document.getElementById('targetPrereqCount');
  const systemStatusBadge = document.getElementById('systemStatusBadge');
  const runAnalysisBtn = document.getElementById('runAnalysisBtn');
  const rankingsTableBody = document.getElementById('rankingsTableBody');
  const learningPathContainer = document.getElementById('learningPathContainer');
  const factualProbabilityDisplay = document.getElementById('factualProbabilityDisplay');
  const stepCountBadge = document.getElementById('stepCountBadge');

  let conceptsMap = new Map();

  // 1. Fetch System Health & Metadata
  fetch('/health')
    .then(res => res.json())
    .then(data => {
      if (data.status === 'healthy' && data.gnn_model_loaded) {
        systemStatusBadge.textContent = 'Online (GNN Loaded)';
        systemStatusBadge.className = 'badge badge-success';
      } else {
        systemStatusBadge.textContent = 'Degraded';
        systemStatusBadge.className = 'badge badge-warning';
      }
    })
    .catch(() => {
      systemStatusBadge.textContent = 'Offline';
      systemStatusBadge.className = 'badge badge-warning';
    });

  // 2. Fetch Concepts with Prerequisites
  fetch('/concepts?limit=100&only_with_prerequisites=true')
    .then(res => res.json())
    .then(concepts => {
      targetConceptSelect.innerHTML = '';
      concepts.forEach(c => {
        conceptsMap.set(c.kc_id, c);
        const opt = document.createElement('option');
        opt.value = c.kc_id;
        opt.textContent = `KC ${c.kc_id}: ${c.name_en}`;
        targetConceptSelect.appendChild(opt);
      });

      // Default to KC 4 (Object quantity combination) or first available
      if (conceptsMap.has(4)) {
        targetConceptSelect.value = "4";
      } else if (concepts.length > 0) {
        targetConceptSelect.value = concepts[0].kc_id.toString();
      }
      updateSelectedConceptMetadata();
    })
    .catch(err => {
      targetConceptSelect.innerHTML = '<option value="">Error loading concepts</option>';
    });

  targetConceptSelect.addEventListener('change', updateSelectedConceptMetadata);

  function updateSelectedConceptMetadata() {
    const selectedId = parseInt(targetConceptSelect.value, 10);
    const concept = conceptsMap.get(selectedId);
    if (concept) {
      targetPrereqCount.textContent = `${concept.in_degree} In-Degree Prerequisites (Direct: [${concept.direct_prerequisites.join(', ')}])`;
    }
  }

  // Student Presets Data Map
  const studentPresets = {
    test_student_8572: {
      concepts: [0, 2, 2, 3],
      responses: [1, 0, 1, 1]
    },
    weak_foundations: {
      concepts: [0, 0, 1, 2, 3],
      responses: [0, 0, 0, 0, 1]
    },
    intermediate: {
      concepts: [0, 1, 2, 3, 4],
      responses: [1, 1, 0, 1, 0]
    }
  };

  // 3. Run Counterfactual Analysis Trigger
  runAnalysisBtn.addEventListener('click', runCounterfactualWorkflow);

  function runCounterfactualWorkflow() {
    const targetId = parseInt(targetConceptSelect.value, 10);
    if (isNaN(targetId)) return;

    const interventionLvl = parseFloat(interventionRange.value);
    const presetKey = document.getElementById('studentPresetSelect').value;
    const studentHistory = studentPresets[presetKey];

    // Show Skeleton Loaders (Consideration 21)
    runAnalysisBtn.disabled = true;
    runAnalysisBtn.textContent = 'Computing Graph Propagation...';
    factualProbabilityDisplay.textContent = '...';

    renderSkeletonTable();
    renderSkeletonPath();

    const requestPayload = {
      target_kc: targetId,
      student_history: studentHistory,
      intervention_level: interventionLvl
    };

    // Parallel execution for prediction, ranking, path generation, and live graph topology
    Promise.all([
      fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target_kc: targetId, student_history: studentHistory })
      }).then(r => r.json()),

      fetch('/counterfactual/rank', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestPayload)
      }).then(r => r.json()),

      fetch('/learning-path/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...requestPayload, max_path_length: 5 })
      }).then(r => r.json()),

      fetch(`/graph/subgraph/${targetId}`).then(r => r.json())
    ])
    .then(([predRes, rankRes, pathRes, graphRes]) => {
      // 1. Update Factual Display
      const factualPct = (predRes.predicted_success_probability * 100).toFixed(1);
      factualProbabilityDisplay.textContent = `${factualPct}%`;

      // 2. Render Live Graph Topology
      renderLiveGraph(graphRes, rankRes.rankings);

      // 3. Render Rankings Table
      renderRankingsTable(rankRes.rankings);

      // 4. Render Learning Path
      renderLearningPath(pathRes);
    })
    .catch(err => {
      rankingsTableBody.innerHTML = `<tr><td colspan="6" style="color:var(--color-error); text-align:center; padding:1.5rem;">Error calculating counterfactual inference: ${err.message}</td></tr>`;
      learningPathContainer.innerHTML = `<div style="color:var(--color-error); padding:1rem;">Failed to generate learning path.</div>`;
    })
    .finally(() => {
      runAnalysisBtn.disabled = false;
      runAnalysisBtn.textContent = 'Run Counterfactual Analysis';
    });
  }

  function renderLiveGraph(graphData, rankings) {
    const svgEdges = document.getElementById('graphEdgesGroup');
    const svgNodes = document.getElementById('graphNodesGroup');
    if (!svgEdges || !svgNodes) return;

    svgEdges.innerHTML = '';
    svgNodes.innerHTML = '';

    if (!graphData || !graphData.nodes || graphData.nodes.length === 0) {
      return;
    }

    const impactMap = new Map();
    if (rankings) {
      rankings.forEach(r => impactMap.set(r.kc_id, r.estimated_impact));
    }

    // Dynamic layer layout
    // Target on right (x=620), direct prerequisites in middle (x=360), indirect prerequisites on left (x=140)
    const targetNode = graphData.nodes.find(n => n.is_target);
    const directNodes = graphData.nodes.filter(n => n.is_direct);
    const indirectNodes = graphData.nodes.filter(n => !n.is_target && !n.is_direct);

    const positions = new Map();

    if (targetNode) {
      positions.set(targetNode.id, { x: 620, y: 130 });
    }

    directNodes.forEach((node, idx) => {
      const step = 260 / (directNodes.length + 1);
      positions.set(node.id, { x: 370, y: step * (idx + 1) });
    });

    indirectNodes.forEach((node, idx) => {
      const step = 260 / (indirectNodes.length + 1);
      positions.set(node.id, { x: 130, y: step * (idx + 1) });
    });

    // Render Edges
    graphData.edges.forEach(edge => {
      const p1 = positions.get(edge.source);
      const p2 = positions.get(edge.target);
      if (p1 && p2) {
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', p1.x);
        line.setAttribute('y1', p1.y);
        line.setAttribute('x2', p2.x);
        line.setAttribute('y2', p2.y);
        line.setAttribute('class', 'graph-edge is-active');
        line.setAttribute('marker-end', 'url(#arrowhead)');
        svgEdges.appendChild(line);
      }
    });

    // Render Nodes
    graphData.nodes.forEach(node => {
      const pos = positions.get(node.id);
      if (!pos) return;

      const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      let nodeClass = 'graph-node';
      if (node.is_target) nodeClass += ' is-target';
      else if (node.is_direct) nodeClass += ' is-direct';

      g.setAttribute('class', nodeClass);
      g.setAttribute('transform', `translate(${pos.x}, ${pos.y})`);

      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('r', node.is_target ? '26' : '22');
      g.appendChild(circle);

      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      text.setAttribute('y', '0');
      text.textContent = `KC ${node.id}`;
      g.appendChild(text);

      const desc = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      desc.setAttribute('y', node.is_target ? '38' : '34');
      desc.setAttribute('class', 'graph-node-desc');
      const shortName = node.name_en.length > 20 ? node.name_en.slice(0, 18) + '...' : node.name_en;
      const impact = impactMap.get(node.id);
      desc.textContent = impact !== undefined ? `${shortName} (+${(impact * 100).toFixed(1)}%)` : shortName;
      g.appendChild(desc);

      svgNodes.appendChild(g);
    });
  }

  function renderSkeletonTable() {
    rankingsTableBody.innerHTML = `
      <tr>
        <td colspan="6" style="padding:1rem;">
          <div class="skeleton skeleton-text" style="width:100%; height:1.5rem;"></div>
          <div class="skeleton skeleton-text" style="width:90%; height:1.5rem; margin-top:0.5rem;"></div>
        </td>
      </tr>
    `;
  }

  function renderSkeletonPath() {
    learningPathContainer.innerHTML = `
      <div class="skeleton skeleton-card" style="margin-bottom:0.75rem;"></div>
      <div class="skeleton skeleton-card"></div>
    `;
  }

  function renderRankingsTable(rankings) {
    if (!rankings || rankings.length === 0) {
      rankingsTableBody.innerHTML = `<tr><td colspan="6" style="text-align:center; color:var(--color-text-muted); padding:2rem;">No connected candidate prerequisites found for this concept.</td></tr>`;
      return;
    }

    rankingsTableBody.innerHTML = rankings.map((item, idx) => {
      const deltaSign = item.estimated_impact >= 0 ? '+' : '';
      const deltaColor = item.estimated_impact >= 0 ? 'var(--color-success)' : 'var(--color-error)';
      return `
        <tr>
          <td><strong>#${idx + 1}</strong></td>
          <td>
            <strong>KC ${item.kc_id}</strong>
            <div style="font-size:0.8125rem; color:var(--color-text-muted);">${item.kc_name_en}</div>
          </td>
          <td>${(item.current_mastery * 100).toFixed(1)}%</td>
          <td>${(item.counterfactual_target_prediction * 100).toFixed(1)}%</td>
          <td style="color:${deltaColor}; font-weight:700;">
            ${deltaSign}${(item.estimated_impact * 100).toFixed(2)}%
          </td>
          <td>
            <span class="badge ${item.relationship.includes('direct') ? 'badge-primary' : 'badge-warning'}">
              ${item.relationship}
            </span>
          </td>
        </tr>
      `;
    }).join('');
  }

  function renderLearningPath(pathData) {
    if (!pathData || !pathData.steps || pathData.steps.length === 0) {
      learningPathContainer.innerHTML = '<div style="color:var(--color-text-muted); padding:1rem;">No valid path discovered.</div>';
      stepCountBadge.textContent = '0 Steps';
      return;
    }

    stepCountBadge.textContent = `${pathData.total_steps} Steps`;

    learningPathContainer.innerHTML = pathData.steps.map(step => {
      const isTarget = step.is_target;
      const stepTypeBadge = isTarget
        ? '<span class="badge badge-primary">Goal Target Concept</span>'
        : `<span class="badge badge-success">Impact: +${(step.estimated_impact * 100).toFixed(2)}%</span>`;

      return `
        <div class="timeline-step ${isTarget ? 'is-target' : ''}">
          <div class="timeline-marker">${step.step_number}</div>
          <div class="step-header">
            <div class="step-title">
              Step ${step.step_number}: KC ${step.kc_id} &mdash; ${step.kc_name_en}
            </div>
            ${stepTypeBadge}
          </div>
          <div class="step-rationale">${step.rationale}</div>
          <div style="font-size:0.75rem; color:var(--color-text-muted); margin-top:0.375rem; font-variant-numeric:tabular-nums;">
            Current Estimated Mastery: ${(step.current_mastery * 100).toFixed(1)}%
          </div>
        </div>
      `;
    }).join('');
  }
});
