/* ══════════════════════════════════════════════
   ExoFinder — 3D Scene & Interactions
   ══════════════════════════════════════════════ */

// ── LOADING SCREEN ──
window.addEventListener('load', () => {
    setTimeout(() => {
        document.querySelector('.loader-screen')?.classList.add('hidden');
    }, 1200);
});

// ── CUSTOM CURSOR ──
(function() {
    const glow = document.querySelector('.cursor-glow');
    const dot = document.querySelector('.cursor-dot');
    if (!glow || !dot || window.matchMedia('(max-width: 768px)').matches) return;
    let cx = 0, cy = 0, dx = 0, dy = 0;
    document.addEventListener('mousemove', e => { cx = e.clientX; cy = e.clientY; });
    document.querySelectorAll('a, button, .info-card, .stat-card, .plot-card, .tech-bubble, .roadmap-card').forEach(el => {
        el.addEventListener('mouseenter', () => glow.classList.add('hover'));
        el.addEventListener('mouseleave', () => glow.classList.remove('hover'));
    });
    (function loop() {
        dx += (cx - dx) * 0.15; dy += (cy - dy) * 0.15;
        glow.style.left = dx + 'px'; glow.style.top = dy + 'px';
        dot.style.left = cx + 'px'; dot.style.top = cy + 'px';
        requestAnimationFrame(loop);
    })();
})();

// ── NAV SCROLL ──
const nav = document.querySelector('.nav');
window.addEventListener('scroll', () => {
    nav?.classList.toggle('scrolled', window.scrollY > 60);
}, { passive: true });

// ── THREE.JS HERO SCENE ──
if (typeof THREE !== 'undefined') { try { (function() {
    const canvas = document.getElementById('heroCanvas');
    if (!canvas) return;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 500);
    camera.position.set(0, 0, 10);
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);

    // ─── LIGHTS ───
    scene.add(new THREE.AmbientLight('#0a0a20', 0.6));
    const warmLight = new THREE.PointLight('#ffeedd', 1.2, 20);
    warmLight.position.set(6, -4, 2);
    scene.add(warmLight);

    // ─── STARFIELD — 7000 particles, deep space ───
    const fieldCount = 7000;
    const fieldPos = new Float32Array(fieldCount * 3);
    for (let i = 0; i < fieldCount; i++) {
        const r = 25 + Math.random() * 200;
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos(2 * Math.random() - 1);
        fieldPos[i*3]   = r * Math.sin(phi) * Math.cos(theta);
        fieldPos[i*3+1] = r * Math.sin(phi) * Math.sin(theta);
        fieldPos[i*3+2] = r * Math.cos(phi);
    }
    const fieldGeo = new THREE.BufferGeometry();
    fieldGeo.setAttribute('position', new THREE.BufferAttribute(fieldPos, 3));
    const fieldMat = new THREE.PointsMaterial({ color: 0xffffff, size: 0.06, sizeAttenuation: true, transparent: true, opacity: 0.8 });
    const starField = new THREE.Points(fieldGeo, fieldMat);
    scene.add(starField);

    // Brighter accent stars
    const accentCount = 150;
    const accentPos = new Float32Array(accentCount * 3);
    for (let i = 0; i < accentCount; i++) {
        const r = 15 + Math.random() * 80;
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos(2 * Math.random() - 1);
        accentPos[i*3]   = r * Math.sin(phi) * Math.cos(theta);
        accentPos[i*3+1] = r * Math.sin(phi) * Math.sin(theta);
        accentPos[i*3+2] = r * Math.cos(phi);
    }
    const accentGeo = new THREE.BufferGeometry();
    accentGeo.setAttribute('position', new THREE.BufferAttribute(accentPos, 3));
    scene.add(new THREE.Points(accentGeo, new THREE.PointsMaterial({ color: 0xbbddff, size: 0.15, sizeAttenuation: true, transparent: true, opacity: 0.6 })));

    // ─── NEBULA GLOWS (background depth) ───
    function makeNebula(x, y, z, size, r, g, b, a) {
        const cv = document.createElement('canvas');
        cv.width = 256; cv.height = 256;
        const ctx = cv.getContext('2d');
        const grad = ctx.createRadialGradient(128,128,0,128,128,128);
        grad.addColorStop(0, `rgba(${r},${g},${b},${a})`);
        grad.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.fillStyle = grad; ctx.fillRect(0,0,256,256);
        const mat = new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(cv), transparent: true, blending: THREE.AdditiveBlending, depthWrite: false });
        const s = new THREE.Sprite(mat);
        s.position.set(x, y, z); s.scale.set(size, size, 1);
        return s;
    }
    scene.add(makeNebula(-8, 3, -30, 30, 15, 25, 70, 0.07));
    scene.add(makeNebula(10, -5, -40, 35, 35, 12, 55, 0.05));
    scene.add(makeNebula(0, -2, -35, 25, 8, 35, 60, 0.04));

    // ─── HOST STAR (bottom-right corner, far from text) ───
    const sX = 6, sY = -4.5, sZ = 2;
    const hostStar = new THREE.Mesh(
        new THREE.SphereGeometry(0.25, 32, 32),
        new THREE.MeshBasicMaterial({ color: 0xfff4dd })
    );
    hostStar.position.set(sX, sY, sZ);
    scene.add(hostStar);

    // Star glow — compact, won't reach center
    function makeStarGlow(size, opacity) {
        const cv = document.createElement('canvas');
        cv.width = 128; cv.height = 128;
        const ctx = cv.getContext('2d');
        const g = ctx.createRadialGradient(64,64,0,64,64,64);
        g.addColorStop(0, `rgba(255,240,210,${opacity})`);
        g.addColorStop(0.4, `rgba(255,200,140,${opacity*0.2})`);
        g.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.fillStyle = g; ctx.fillRect(0,0,128,128);
        const mat = new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(cv), transparent: true, blending: THREE.AdditiveBlending, depthWrite: false });
        const s = new THREE.Sprite(mat);
        s.scale.set(size, size, 1); s.position.set(sX, sY, sZ);
        return s;
    }
    scene.add(makeStarGlow(2.0, 0.6));
    scene.add(makeStarGlow(3.5, 0.12));

    // ─── PLANET (small, orbiting the star in bottom-right) ───
    const orbitR = 1.2;
    const planetMesh = new THREE.Mesh(
        new THREE.SphereGeometry(0.08, 32, 32),
        new THREE.MeshStandardMaterial({ color: 0x2a3a6e, roughness: 0.5, metalness: 0.1, emissive: 0x0a0a30, emissiveIntensity: 0.3 })
    );
    scene.add(planetMesh);
    // Atmosphere rim
    const rim = new THREE.Mesh(
        new THREE.SphereGeometry(0.095, 32, 32),
        new THREE.MeshBasicMaterial({ color: 0x00bbff, transparent: true, opacity: 0.1, side: THREE.BackSide })
    );
    scene.add(rim);

    // ─── LIGHT CURVE (animated transit dip at bottom) ───
    const lcPoints = [];
    for (let i = 0; i <= 200; i++) {
        const x = (i / 200) * 12 - 6;
        let y = 0;
        const d = Math.abs(x);
        if (d < 0.8) y = -0.25 * Math.cos((d / 0.8) * Math.PI * 0.5);
        y += Math.sin(i * 0.3) * 0.015;
        lcPoints.push(new THREE.Vector3(x, y - 4.2, 4));
    }
    const lcGeo = new THREE.BufferGeometry().setFromPoints(lcPoints);
    const lcMat = new THREE.LineBasicMaterial({ color: 0x00d4ff, transparent: true, opacity: 0.25 });
    const lcLine = new THREE.Line(lcGeo, lcMat);
    scene.add(lcLine);

    // ─── INTERACTION ───
    let mouseX = 0, mouseY = 0;
    window.addEventListener('mousemove', e => {
        mouseX = (e.clientX / window.innerWidth - 0.5) * 2;
        mouseY = (e.clientY / window.innerHeight - 0.5) * 2;
    }, { passive: true });

    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });

    let scrollProg = 0;
    window.addEventListener('scroll', () => {
        scrollProg = Math.min(window.scrollY / (window.innerHeight * 0.8), 1);
    }, { passive: true });

    // ─── ANIMATION ───
    const clock = new THREE.Clock();
    function animate() {
        requestAnimationFrame(animate);
        const t = clock.getElapsedTime();

        // Planet orbits star in bottom-right
        const a = t * 0.3;
        planetMesh.position.set(sX + Math.cos(a) * orbitR, sY + Math.sin(a) * orbitR * 0.3, sZ + Math.sin(a) * orbitR * 0.5);
        planetMesh.rotation.y += 0.006;
        rim.position.copy(planetMesh.position);

        // Star subtle pulse
        hostStar.scale.setScalar(1 + Math.sin(t * 1.5) * 0.04);

        // Starfield drift
        starField.rotation.y += 0.00003;
        starField.rotation.x += 0.00001;

        // Animate light curve noise
        const lcArr = lcLine.geometry.attributes.position.array;
        for (let i = 0; i <= 200; i++) {
            const x = (i / 200) * 12 - 6;
            let y = 0;
            const d = Math.abs(x);
            if (d < 0.8) y = -0.25 * Math.cos((d / 0.8) * Math.PI * 0.5);
            y += Math.sin(i * 0.3 + t * 2) * 0.012;
            lcArr[i * 3 + 1] = y - 4.2;
        }
        lcLine.geometry.attributes.position.needsUpdate = true;

        // Camera: straight ahead + parallax (center stays clear)
        camera.position.x += (mouseX * 0.4 - camera.position.x) * 0.02;
        camera.position.y += (mouseY * -0.2 - camera.position.y) * 0.02;
        camera.position.z = 10 - scrollProg * 2;
        camera.lookAt(0, 0, 0);

        renderer.render(scene, camera);
    }
    animate();
})(); } catch(e) { console.warn('Three.js scene error:', e); } }

// ── HERO TEXT ENTRANCE ──
window.addEventListener('load', () => {
    const anim = (el, delay) => setTimeout(() => {
        if (!el) return;
        el.style.transition = 'opacity 0.9s cubic-bezier(0.4,0,0.2,1), transform 0.9s cubic-bezier(0.4,0,0.2,1)';
        el.style.opacity = '1'; el.style.transform = 'translateY(0)';
    }, delay);
    anim(document.getElementById('heroBadge'), 400);
    anim(document.getElementById('heroTitle'), 700);
    anim(document.getElementById('heroSub'), 1000);
    anim(document.getElementById('heroBtns'), 1300);
});

// ── SCROLL REVEAL ──
const revealObs = new IntersectionObserver(entries => {
    entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('visible'); });
}, { threshold: 0.06, rootMargin: '0px 0px -40px 0px' });
document.querySelectorAll('.reveal, .reveal-left, .reveal-right, .reveal-scale').forEach(el => revealObs.observe(el));

// ── ANIMATED COUNTERS ──
const counterObs = new IntersectionObserver(entries => {
    entries.forEach(e => {
        if (!e.isIntersecting) return;
        const target = parseFloat(e.target.dataset.counter);
        const start = performance.now();
        (function tick(now) {
            const p = Math.min((now - start) / 1600, 1);
            const ease = 1 - Math.pow(1 - p, 4);
            e.target.textContent = (target * ease).toFixed(4);
            if (p < 1) requestAnimationFrame(tick);
        })(start);
        counterObs.unobserve(e.target);
    });
}, { threshold: 0.5 });
document.querySelectorAll('[data-counter]').forEach(el => counterObs.observe(el));

// ── PIPELINE ANIMATION ──
const pipeObs = new IntersectionObserver(entries => {
    entries.forEach(e => {
        if (!e.isIntersecting) return;
        const nodes = e.target.querySelectorAll('.pipeline-node');
        const conns = e.target.querySelectorAll('.pipeline-connector');
        let i = 0;
        function step() {
            if (i < nodes.length) nodes[i].classList.add('active');
            if (i > 0 && i - 1 < conns.length) conns[i-1].classList.add('active');
            i++;
            if (i <= nodes.length) setTimeout(step, 350);
        }
        step();
        pipeObs.unobserve(e.target);
    });
}, { threshold: 0.3 });
const pipeEl = document.getElementById('pipelineContainer');
if (pipeEl) pipeObs.observe(pipeEl);

// ── COMPARISON BAR CHART ──
(function() {
    const data = [
        { label: 'Precision', baseline: 0.2000, hybrid: 0.6667 },
        { label: 'Recall', baseline: 0.2000, hybrid: 0.4000 },
        { label: 'F1-Score', baseline: 0.2000, hybrid: 0.5000 },
        { label: 'PR-AUC', baseline: 0.1011, hybrid: 0.5606 },
        { label: 'ROC-AUC', baseline: 0.9437, hybrid: 0.8727 },
        { label: 'MCC', baseline: 0.1929, hybrid: 0.5132 },
        { label: 'Accuracy', baseline: 0.9860, hybrid: 0.9930 },
    ];
    const container = document.getElementById('barChart');
    if (!container) return;
    data.forEach(d => {
        const row = document.createElement('div'); row.className = 'bar-row';
        row.innerHTML = `
            <div class="bar-label">${d.label}</div>
            <div class="bar-pair">
                <div class="bar-tag">Hybrid CNN+RF</div>
                <div class="bar-container"><div class="bar-fill hybrid" data-width="${d.hybrid*100}" style="width:0">${d.hybrid.toFixed(4)}</div></div>
                <div class="bar-tag">Baseline RF</div>
                <div class="bar-container"><div class="bar-fill baseline" data-width="${d.baseline*100}" style="width:0">${d.baseline.toFixed(4)}</div></div>
            </div>`;
        container.appendChild(row);
    });
    const barObs = new IntersectionObserver(entries => {
        entries.forEach(e => {
            if (!e.isIntersecting) return;
            e.target.querySelectorAll('.bar-fill').forEach(b => { b.style.width = b.dataset.width + '%'; });
            barObs.unobserve(e.target);
        });
    }, { threshold: 0.2 });
    barObs.observe(container);
})();

// ── SMOOTH NAV SCROLL ──
document.querySelectorAll('.nav-links a, .btn-primary, .btn-secondary').forEach(a => {
    a.addEventListener('click', function(e) {
        const href = this.getAttribute('href');
        if (href && href.startsWith('#')) {
            e.preventDefault();
            const t = document.querySelector(href);
            if (t) window.scrollTo({ top: t.getBoundingClientRect().top + window.pageYOffset - 70, behavior: 'smooth' });
        }
    });
});

// ── TILT EFFECT ON CARDS ──
document.querySelectorAll('.info-card, .stat-card, .arch-panel').forEach(card => {
    card.addEventListener('mousemove', e => {
        const rect = card.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width - 0.5;
        const y = (e.clientY - rect.top) / rect.height - 0.5;
        card.style.transform = `translateY(-8px) perspective(600px) rotateX(${-y*6}deg) rotateY(${x*6}deg)`;
    });
    card.addEventListener('mouseleave', () => {
        card.style.transform = '';
    });
});

// ── RADIAL GAUGE CHARTS (Phase 4) ──
(function() {
    const gaugeData = [
        { label: 'Precision', value: 0.6667, color: '#30d158' },
        { label: 'Recall', value: 0.4000, color: '#00d4ff' },
        { label: 'F1-Score', value: 0.5000, color: '#bf5af2' },
        { label: 'PR-AUC', value: 0.5606, color: '#ff9f0a' },
        { label: 'ROC-AUC', value: 0.8727, color: '#00d4ff' },
        { label: 'MCC', value: 0.5132, color: '#ff375f' },
    ];
    const grid = document.getElementById('gaugeGrid');
    if (!grid) return;
    const r = 45, circ = 2 * Math.PI * r;
    gaugeData.forEach(d => {
        const card = document.createElement('div');
        card.className = 'gauge-card';
        card.innerHTML = `
            <svg class="gauge-svg" viewBox="0 0 100 100">
                <circle class="gauge-bg" cx="50" cy="50" r="${r}"/>
                <circle class="gauge-fill" cx="50" cy="50" r="${r}"
                    stroke="${d.color}"
                    stroke-dasharray="${circ}"
                    stroke-dashoffset="${circ}"
                    data-target="${circ - circ * d.value}"/>
            </svg>
            <div class="gauge-value" style="color:${d.color}">${d.value.toFixed(4)}</div>
            <div class="gauge-label">${d.label}</div>`;
        grid.appendChild(card);
    });
    const gaugeObs = new IntersectionObserver(entries => {
        entries.forEach(e => {
            if (!e.isIntersecting) return;
            e.target.querySelectorAll('.gauge-fill').forEach(c => {
                c.style.strokeDashoffset = c.dataset.target;
            });
            gaugeObs.unobserve(e.target);
        });
    }, { threshold: 0.3 });
    gaugeObs.observe(grid);
})();

// ── ARCHITECTURE LAYER-BY-LAYER REVEAL (Phase 3) ──
(function() {
    document.querySelectorAll('.arch-panel').forEach(panel => {
        const layers = panel.querySelectorAll('.arch-layer, .arch-arrow');
        const obs = new IntersectionObserver(entries => {
            entries.forEach(e => {
                if (!e.isIntersecting) return;
                layers.forEach((layer, i) => {
                    setTimeout(() => layer.classList.add('visible'), i * 120);
                });
                obs.unobserve(e.target);
            });
        }, { threshold: 0.2 });
        obs.observe(panel);
    });
})();

// ── PIPELINE PARTICLE CANVAS (Phase 2) ──
(function() {
    const section = document.getElementById('pipeline');
    if (!section) return;
    const canvas = document.createElement('canvas');
    canvas.id = 'pipelineCanvas';
    section.style.position = 'relative';
    section.insertBefore(canvas, section.firstChild);
    const ctx = canvas.getContext('2d');
    let w, h, dots = [];

    function resize() {
        w = canvas.width = section.offsetWidth;
        h = canvas.height = section.offsetHeight;
    }
    resize();
    window.addEventListener('resize', resize);

    // Create floating particles
    for (let i = 0; i < 60; i++) {
        dots.push({
            x: Math.random() * w,
            y: Math.random() * h,
            vx: (Math.random() - 0.5) * 0.4,
            vy: (Math.random() - 0.5) * 0.3,
            r: 1 + Math.random() * 2,
            alpha: 0.1 + Math.random() * 0.3
        });
    }

    let animating = false;
    function draw() {
        if (!animating) return;
        ctx.clearRect(0, 0, w, h);
        dots.forEach(d => {
            d.x += d.vx; d.y += d.vy;
            if (d.x < 0 || d.x > w) d.vx *= -1;
            if (d.y < 0 || d.y > h) d.vy *= -1;
            ctx.beginPath();
            ctx.arc(d.x, d.y, d.r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(0,212,255,${d.alpha})`;
            ctx.fill();
        });
        // Draw connections between nearby particles
        for (let i = 0; i < dots.length; i++) {
            for (let j = i + 1; j < dots.length; j++) {
                const dx = dots[i].x - dots[j].x;
                const dy = dots[i].y - dots[j].y;
                const dist = Math.sqrt(dx*dx + dy*dy);
                if (dist < 120) {
                    ctx.beginPath();
                    ctx.moveTo(dots[i].x, dots[i].y);
                    ctx.lineTo(dots[j].x, dots[j].y);
                    ctx.strokeStyle = `rgba(0,212,255,${0.06 * (1 - dist/120)})`;
                    ctx.stroke();
                }
            }
        }
        requestAnimationFrame(draw);
    }

    const canvasObs = new IntersectionObserver(entries => {
        entries.forEach(e => {
            animating = e.isIntersecting;
            if (animating) draw();
        });
    }, { threshold: 0.1 });
    canvasObs.observe(section);
})();
