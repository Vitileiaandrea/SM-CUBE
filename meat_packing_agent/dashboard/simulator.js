const CONFIG = {
    cubeSize: { width: 210, length: 210, height: 250 },
    conveyorWidth: 500, conveyorLength: 2000, conveyorHeight: 400,
    robotArmRadius: 750, robotColumnHeight: 700,
    handSize: 130, scale: 0.001,
    sliceApproachTime: 2000, rotationTime: 800, lowerTime: 600, pickTime: 300, raiseTime: 500
};

let isRunning = false, scene, camera, renderer, controls;
let robot, robotBase, robotArm, robotHand, vacuumCups3D = [];
let conveyor, cubeContainer, placedSlicesInCube = [];
let sliceOnConveyor = null, sliceBeingCarried = null, animationId = null;
let animState = { phase: 'idle', slicesData: [], currentSliceIndex: 0, animationStart: 0, startValue: 0, endValue: 0, finalFillPct: 0 };
let simState = { fillPercentage: 0, slicesPlaced: 0, slicesDiscarded: 0, currentLayer: 1, status: 'Pronto', robotRotation: 0, robotAction: 'Fermo' };
const API_URL = window.location.origin;

// Key positions in world coordinates (calculated after scene creation)
let conveyorSurfaceY = 0;
let cubeBottomY = 0;
let cubeOriginX = 0;
let cubeOriginZ = 0;
let pickupX = 0;
let pickupZ = 0;

function easeInOutQuad(t) { return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2; }

function initScene() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);
    camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.01, 100);
    camera.position.set(1.5, 1.2, 1.5);
    camera.lookAt(0.3, 0.4, 0);
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.shadowMap.enabled = true;
    document.getElementById('canvas-container').appendChild(renderer.domElement);
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0.3, 0.4, 0);
    scene.add(new THREE.AmbientLight(0xffffff, 0.5));
    const mainLight = new THREE.DirectionalLight(0xffffff, 0.8);
    mainLight.position.set(2, 4, 2);
    mainLight.castShadow = true;
    scene.add(mainLight);
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(5, 5), new THREE.MeshStandardMaterial({ color: 0x2d3748 }));
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);
    scene.add(new THREE.GridHelper(3, 30, 0x444444, 0x333333));
    createConveyor();
    createCube();
    createRobot();
    createVacuumGridUI();
    window.addEventListener('resize', onWindowResize);
    animate();
}

function createConveyor() {
    const g = new THREE.Group(), s = CONFIG.scale;
    const h = CONFIG.conveyorHeight * s, w = CONFIG.conveyorWidth * s, l = CONFIG.conveyorLength * s;
    const fm = new THREE.MeshStandardMaterial({ color: 0x4a5568, metalness: 0.6 });
    const lg = new THREE.BoxGeometry(0.03, h, 0.03);
    [[-l/2+0.05, h/2, w/2-0.02], [-l/2+0.05, h/2, -w/2+0.02], [l/2-0.05, h/2, w/2-0.02], [l/2-0.05, h/2, -w/2+0.02]].forEach(p => { const m = new THREE.Mesh(lg, fm); m.position.set(...p); g.add(m); });
    const rg = new THREE.BoxGeometry(l, 0.04, 0.02);
    const lr = new THREE.Mesh(rg, fm); lr.position.set(0, h, w/2); g.add(lr);
    const rr = new THREE.Mesh(rg, fm); rr.position.set(0, h, -w/2); g.add(rr);
    const belt = new THREE.Mesh(new THREE.BoxGeometry(l - 0.02, 0.01, w - 0.04), new THREE.MeshStandardMaterial({ color: 0x1a1a1a }));
    belt.position.set(0, h + 0.005, 0); g.add(belt);
    g.position.set(-0.6, 0, 0);
    scene.add(g);
    conveyor = g;
    
    // Calculate conveyor surface Y in world coordinates
    conveyorSurfaceY = g.position.y + h + 0.01;
    // Pickup position (end of conveyor near robot)
    pickupX = g.position.x + l/2 - 0.2;
    pickupZ = g.position.z;
}

function createCube() {
    const g = new THREE.Group(), s = CONFIG.scale;
    const w = CONFIG.cubeSize.width * s, l = CONFIG.cubeSize.length * s, h = CONFIG.cubeSize.height * s;
    const baseY = CONFIG.conveyorHeight * s;
    const bottom = new THREE.Mesh(new THREE.BoxGeometry(w + 0.01, 0.005, l + 0.01), new THREE.MeshStandardMaterial({ color: 0x6b7280, metalness: 0.7 }));
    bottom.position.y = baseY; g.add(bottom);
    const wm = new THREE.MeshStandardMaterial({ color: 0x9ca3af, transparent: true, opacity: 0.25, side: THREE.DoubleSide });
    const fg = new THREE.BoxGeometry(w, h, 0.003);
    const fw = new THREE.Mesh(fg, wm); fw.position.set(0, baseY + h/2, l/2); g.add(fw);
    const bw = new THREE.Mesh(fg, wm); bw.position.set(0, baseY + h/2, -l/2); g.add(bw);
    const sg = new THREE.BoxGeometry(0.003, h, l);
    const lw = new THREE.Mesh(sg, wm); lw.position.set(-w/2, baseY + h/2, 0); g.add(lw);
    const rw = new THREE.Mesh(sg, wm); rw.position.set(w/2, baseY + h/2, 0); g.add(rw);
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(w, h, l)), new THREE.LineBasicMaterial({ color: 0x60a5fa }));
    edges.position.y = baseY + h/2; g.add(edges);
    g.position.set(0.4, 0, 0);
    scene.add(g);
    cubeContainer = g;
    
    // Calculate cube bottom Y and origin in world coordinates
    cubeBottomY = g.position.y + baseY + 0.005;
    // Cube origin (0,0) is at the corner (-w/2, -l/2) relative to cube center
    cubeOriginX = g.position.x - w/2;
    cubeOriginZ = g.position.z - l/2;
}

function createRobot() {
    const g = new THREE.Group(), s = CONFIG.scale;
    const ym = new THREE.MeshStandardMaterial({ color: 0xf5a623, metalness: 0.3 });
    const bm = new THREE.MeshStandardMaterial({ color: 0x1a1a1a });
    const gm = new THREE.MeshStandardMaterial({ color: 0x6b7280, metalness: 0.7 });
    const bp = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.14, 0.02, 32), gm);
    bp.position.y = 0.01; g.add(bp);
    const bg = new THREE.Group();
    const base = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.12, 0.08, 32), ym);
    base.position.y = 0.06; bg.add(base);
    const ch = CONFIG.robotColumnHeight * s;
    const col = new THREE.Mesh(new THREE.BoxGeometry(0.1, ch, 0.1), ym);
    col.position.y = 0.1 + ch/2; bg.add(col);
    const sh = new THREE.Mesh(new THREE.SphereGeometry(0.06, 16, 16), bm);
    sh.position.y = 0.1 + ch; bg.add(sh);
    const ag = new THREE.Group();
    ag.position.y = 0.1 + ch;
    const al = CONFIG.robotArmRadius * s;
    const arm = new THREE.Mesh(new THREE.BoxGeometry(al, 0.06, 0.06), ym);
    arm.position.x = al / 2; ag.add(arm);
    const wr = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 0.08, 16), bm);
    wr.position.x = al; wr.rotation.z = Math.PI / 2; ag.add(wr);
    const hg = new THREE.Group();
    hg.position.x = al;
    const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 0.15, 16), gm);
    shaft.position.y = -0.075; hg.add(shaft);
    const hs = CONFIG.handSize * s;
    const hp = new THREE.Mesh(new THREE.BoxGeometry(hs, 0.015, hs), gm);
    hp.position.y = -0.16; hg.add(hp);
    const cg = new THREE.CylinderGeometry(0.012, 0.008, 0.02, 12);
    const cm = new THREE.MeshStandardMaterial({ color: 0x374151 });
    const cs = hs / 5;
    for (let r = 0; r < 4; r++) {
        for (let c = 0; c < 4; c++) {
            const cup = new THREE.Mesh(cg, cm.clone());
            cup.position.set((c - 1.5) * cs, -0.18, (r - 1.5) * cs);
            hg.add(cup);
            vacuumCups3D.push(cup);
        }
    }
    ag.add(hg);
    robotHand = hg;
    bg.add(ag);
    robotArm = ag;
    g.add(bg);
    robotBase = bg;
    // Position robot between conveyor and cube
    g.position.set(0, 0, -0.4);
    scene.add(g);
    robot = g;
    robotBase.rotation.y = Math.PI * 0.3;
}

function createVacuumGridUI() {
    const grid = document.getElementById('vacuum-grid');
    grid.innerHTML = '';
    for (let i = 0; i < 16; i++) {
        const cup = document.createElement('div');
        cup.className = 'vacuum-cup';
        cup.id = 'vacuum-cup-' + i;
        grid.appendChild(cup);
    }
}

function updateVacuumCups(pattern) {
    for (let i = 0; i < 16; i++) {
        const ui = document.getElementById('vacuum-cup-' + i);
        const c3d = vacuumCups3D[i];
        const active = pattern.includes(i);
        if (ui) ui.className = active ? 'vacuum-cup active' : 'vacuum-cup';
        if (c3d) c3d.material.color.setHex(active ? 0x22c55e : 0x374151);
    }
}

function createMeatSlice(data) {
    const s = CONFIG.scale;
    const w = data.width * s, l = data.length * s;
    const t = ((data.thickness_min + data.thickness_max) / 2) * s;
    const geo = new THREE.BoxGeometry(w, t, l, 3, 1, 3);
    const pos = geo.attributes.position;
    for (let i = 0; i < pos.count; i++) {
        const x = pos.getX(i), z = pos.getZ(i);
        if (Math.abs(x) > w * 0.35 || Math.abs(z) > l * 0.35) {
            pos.setX(i, x + (Math.random() - 0.5) * w * 0.15);
            pos.setZ(i, z + (Math.random() - 0.5) * l * 0.15);
        }
    }
    geo.computeVertexNormals();
    const hue = 0.98 + Math.random() * 0.04;
    const color = new THREE.Color().setHSL(hue % 1, 0.65, 0.45);
    const mat = new THREE.MeshStandardMaterial({ color: color, roughness: 0.8 });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.castShadow = true;
    mesh.userData = { ...data, thickness: t, widthMm: data.width, lengthMm: data.length };
    return mesh;
}

function animate() {
    animationId = requestAnimationFrame(animate);
    controls.update();
    if (isRunning) updateAnimation();
    renderer.render(scene, camera);
}

function getProgress(duration) {
    return Math.min(1, (Date.now() - animState.animationStart) / duration);
}

// Calculate angle from robot to a world position
function angleToPosition(worldX, worldZ) {
    const dx = worldX - robot.position.x;
    const dz = worldZ - robot.position.z;
    return Math.atan2(dx, -dz);
}

// Calculate hand Y position to reach a target world Y
function handYForWorldY(targetWorldY) {
    // Robot shoulder is at robot.position.y + 0.1 + columnHeight
    const shoulderY = robot.position.y + 0.1 + CONFIG.robotColumnHeight * CONFIG.scale;
    // Hand cups are at robotHand local Y = -0.18
    // When robotHand.position.y = 0, cup world Y = shoulderY - 0.18
    // To reach targetWorldY: shoulderY + robotHand.position.y - 0.18 = targetWorldY
    return targetWorldY - shoulderY + 0.18;
}

function updateAnimation() {
    const s = CONFIG.scale;
    const convAngle = angleToPosition(pickupX, pickupZ);
    const cubeAngle = angleToPosition(cubeContainer.position.x, cubeContainer.position.z);
    
    switch (animState.phase) {
        case 'slice_approaching': {
            const p = getProgress(CONFIG.sliceApproachTime);
            if (sliceOnConveyor) {
                const startX = conveyor.position.x - 0.8;
                sliceOnConveyor.position.x = startX + (pickupX - startX) * easeInOutQuad(p);
            }
            if (p >= 1) {
                animState.phase = 'rotating_to_conveyor';
                animState.animationStart = Date.now();
                animState.startValue = robotBase.rotation.y;
                animState.endValue = convAngle;
                simState.robotAction = 'Ruotando';
                updateStats();
            }
            break;
        }
        case 'rotating_to_conveyor': {
            const p = getProgress(CONFIG.rotationTime);
            robotBase.rotation.y = animState.startValue + (animState.endValue - animState.startValue) * easeInOutQuad(p);
            simState.robotRotation = Math.round(robotBase.rotation.y * 180 / Math.PI);
            document.getElementById('robot-rotation').textContent = simState.robotRotation + String.fromCharCode(176);
            if (p >= 1) {
                animState.phase = 'lowering_to_pick';
                animState.animationStart = Date.now();
                animState.startValue = robotHand.position.y;
                // Lower hand to conveyor surface
                const sliceThickness = sliceOnConveyor ? sliceOnConveyor.userData.thickness : 0.02;
                animState.endValue = handYForWorldY(conveyorSurfaceY + sliceThickness/2);
                simState.robotAction = 'Scendendo';
                updateStats();
            }
            break;
        }
        case 'lowering_to_pick': {
            const p = getProgress(CONFIG.lowerTime);
            robotHand.position.y = animState.startValue + (animState.endValue - animState.startValue) * easeInOutQuad(p);
            if (p >= 1) {
                animState.phase = 'picking';
                animState.animationStart = Date.now();
                updateVacuumCups(calculateVacuumPattern(sliceOnConveyor ? sliceOnConveyor.userData : {}));
                simState.robotAction = 'Afferrando';
                updateStats();
            }
            break;
        }
        case 'picking': {
            const p = getProgress(CONFIG.pickTime);
            if (p >= 1) {
                if (sliceOnConveyor) {
                    sliceBeingCarried = sliceOnConveyor;
                    scene.remove(sliceOnConveyor);
                    sliceOnConveyor = null;
                    sliceBeingCarried.position.set(0, -0.20, 0);
                    robotHand.add(sliceBeingCarried);
                }
                animState.phase = 'raising_after_pick';
                animState.animationStart = Date.now();
                animState.startValue = robotHand.position.y;
                animState.endValue = 0;
                simState.robotAction = 'Sollevando';
                updateStats();
            }
            break;
        }
        case 'raising_after_pick': {
            const p = getProgress(CONFIG.raiseTime);
            robotHand.position.y = animState.startValue + (animState.endValue - animState.startValue) * easeInOutQuad(p);
            if (p >= 1) {
                animState.phase = 'rotating_to_cube';
                animState.animationStart = Date.now();
                animState.startValue = robotBase.rotation.y;
                animState.endValue = cubeAngle;
                simState.robotAction = 'Ruotando';
                updateStats();
            }
            break;
        }
        case 'rotating_to_cube': {
            const p = getProgress(CONFIG.rotationTime);
            robotBase.rotation.y = animState.startValue + (animState.endValue - animState.startValue) * easeInOutQuad(p);
            simState.robotRotation = Math.round(robotBase.rotation.y * 180 / Math.PI);
            document.getElementById('robot-rotation').textContent = simState.robotRotation + String.fromCharCode(176);
            if (p >= 1) {
                const sd = animState.slicesData[animState.currentSliceIndex];
                // Calculate target Y: cube bottom + slice z position + half thickness
                const sliceThickness = sliceBeingCarried ? sliceBeingCarried.userData.thickness : 0.02;
                const targetY = cubeBottomY + (sd ? sd.z * s : 0) + sliceThickness/2;
                animState.phase = 'lowering_to_place';
                animState.animationStart = Date.now();
                animState.startValue = robotHand.position.y;
                animState.endValue = handYForWorldY(targetY);
                simState.robotAction = 'Scendendo';
                updateStats();
            }
            break;
        }
        case 'lowering_to_place': {
            const p = getProgress(CONFIG.lowerTime);
            robotHand.position.y = animState.startValue + (animState.endValue - animState.startValue) * easeInOutQuad(p);
            if (p >= 1) {
                animState.phase = 'placing';
                animState.animationStart = Date.now();
                simState.robotAction = 'Posando';
                updateStats();
            }
            break;
        }
        case 'placing': {
            const p = getProgress(CONFIG.pickTime);
            if (p >= 1) {
                if (sliceBeingCarried) {
                    robotHand.remove(sliceBeingCarried);
                    const sd = animState.slicesData[animState.currentSliceIndex];
                    if (sd) {
                        // API returns x, y as corner position in mm from cube origin (0,0)
                        // Calculate slice center position
                        const sliceWidthMm = sd.width || 120;
                        const sliceLengthMm = sd.length || 100;
                        const centerXmm = sd.x + sliceWidthMm / 2;
                        const centerZmm = sd.y + sliceLengthMm / 2;
                        
                        // Clamp to keep slice inside cube
                        const cubeW = CONFIG.cubeSize.width;
                        const cubeL = CONFIG.cubeSize.length;
                        const clampedCenterXmm = Math.max(sliceWidthMm/2, Math.min(cubeW - sliceWidthMm/2, centerXmm));
                        const clampedCenterZmm = Math.max(sliceLengthMm/2, Math.min(cubeL - sliceLengthMm/2, centerZmm));
                        
                        // Convert to world coordinates
                        const worldX = cubeOriginX + clampedCenterXmm * s;
                        const worldZ = cubeOriginZ + clampedCenterZmm * s;
                        const worldY = cubeBottomY + sd.z * s + sliceBeingCarried.userData.thickness/2;
                        
                        sliceBeingCarried.position.set(worldX, worldY, worldZ);
                        sliceBeingCarried.rotation.y = (sd.rotation || 0) * Math.PI / 180;
                    }
                    scene.add(sliceBeingCarried);
                    placedSlicesInCube.push(sliceBeingCarried);
                    sliceBeingCarried = null;
                    simState.slicesPlaced++;
                    if (animState.slicesData.length > 0) {
                        simState.fillPercentage = (simState.slicesPlaced / animState.slicesData.length) * animState.finalFillPct;
                    }
                }
                updateVacuumCups([]);
                animState.phase = 'raising_after_place';
                animState.animationStart = Date.now();
                animState.startValue = robotHand.position.y;
                animState.endValue = 0;
                simState.robotAction = 'Sollevando';
                updateStats();
            }
            break;
        }
        case 'raising_after_place': {
            const p = getProgress(CONFIG.raiseTime);
            robotHand.position.y = animState.startValue + (animState.endValue - animState.startValue) * easeInOutQuad(p);
            if (p >= 1) {
                animState.currentSliceIndex++;
                if (animState.currentSliceIndex < animState.slicesData.length) {
                    spawnSliceOnConveyor();
                    animState.phase = 'slice_approaching';
                    animState.animationStart = Date.now();
                    simState.robotAction = 'Attendendo';
                } else {
                    animState.phase = 'idle';
                    simState.status = 'Completato!';
                    simState.robotAction = 'Fermo';
                    simState.fillPercentage = animState.finalFillPct;
                    isRunning = false;
                    document.getElementById('btn-start').disabled = false;
                    document.getElementById('btn-stop').disabled = true;
                }
                updateStats();
            }
            break;
        }
    }
}

function spawnSliceOnConveyor() {
    const sd = animState.slicesData[animState.currentSliceIndex];
    if (!sd) return;
    sliceOnConveyor = createMeatSlice({ width: sd.width, length: sd.length, thickness_min: sd.thickness * 0.9, thickness_max: sd.thickness * 1.1 });
    // Place slice on conveyor surface at the start
    sliceOnConveyor.position.set(conveyor.position.x - 0.8, conveyorSurfaceY + sliceOnConveyor.userData.thickness/2, conveyor.position.z);
    scene.add(sliceOnConveyor);
}

function calculateVacuumPattern(data) {
    const pattern = [];
    const sw = data.widthMm || data.width || 150, sl = data.lengthMm || data.length || 150;
    const cx = Math.min(4, Math.ceil(sw / 40)), cz = Math.min(4, Math.ceil(sl / 40));
    const sx = Math.floor((4 - cx) / 2), sz = Math.floor((4 - cz) / 2);
    for (let z = sz; z < sz + cz && z < 4; z++) {
        for (let x = sx; x < sx + cx && x < 4; x++) {
            pattern.push(z * 4 + x);
        }
    }
    return pattern;
}

function updateStats() {
    document.getElementById('fill-pct').textContent = simState.fillPercentage.toFixed(1) + '%';
    document.getElementById('fill-bar').style.width = Math.min(100, simState.fillPercentage) + '%';
    document.getElementById('slices-placed').textContent = simState.slicesPlaced;
    document.getElementById('slices-discarded').textContent = simState.slicesDiscarded;
    document.getElementById('current-layer').textContent = simState.currentLayer;
    document.getElementById('status').textContent = simState.status;
    document.getElementById('robot-action').textContent = simState.robotAction;
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

async function startSimulation() {
    if (isRunning) return;
    isRunning = true;
    simState.status = 'Caricamento...';
    simState.robotAction = 'Inizializzazione';
    updateStats();
    document.getElementById('btn-start').disabled = true;
    document.getElementById('btn-stop').disabled = false;
    try {
        const cubeId = Math.floor(Math.random() * 1000);
        const response = await fetch(API_URL + '/cube/fill_training_algorithm?cube_id=' + cubeId, { method: 'POST' });
        if (!response.ok) throw new Error('API error: ' + response.status);
        const result = await response.json();
        animState.slicesData = result.placed_slices || [];
        animState.currentSliceIndex = 0;
        animState.finalFillPct = result.fill_percentage || 0;
        simState.fillPercentage = 0;
        simState.slicesDiscarded = result.slices_discarded || 0;
        simState.currentLayer = result.layers_completed || 1;
        if (animState.slicesData.length > 0) {
            simState.status = 'In esecuzione';
            simState.robotAction = 'Attendendo';
            spawnSliceOnConveyor();
            animState.phase = 'slice_approaching';
            animState.animationStart = Date.now();
        } else {
            simState.status = 'Nessuna fettina';
            isRunning = false;
            document.getElementById('btn-start').disabled = false;
            document.getElementById('btn-stop').disabled = true;
        }
        updateStats();
    } catch (error) {
        console.error('Simulation error:', error);
        simState.status = 'Errore: ' + error.message;
        simState.robotAction = 'Errore';
        isRunning = false;
        document.getElementById('btn-start').disabled = false;
        document.getElementById('btn-stop').disabled = true;
        updateStats();
    }
}

function stopSimulation() {
    isRunning = false;
    animState.phase = 'idle';
    simState.status = 'Fermato';
    simState.robotAction = 'Fermo';
    updateStats();
    document.getElementById('btn-start').disabled = false;
    document.getElementById('btn-stop').disabled = true;
}

async function resetSimulation() {
    stopSimulation();
    placedSlicesInCube.forEach(slice => scene.remove(slice));
    placedSlicesInCube = [];
    if (sliceOnConveyor) { scene.remove(sliceOnConveyor); sliceOnConveyor = null; }
    if (sliceBeingCarried) { robotHand.remove(sliceBeingCarried); sliceBeingCarried = null; }
    robotBase.rotation.y = Math.PI * 0.3;
    robotHand.position.y = 0;
    animState = { phase: 'idle', slicesData: [], currentSliceIndex: 0, animationStart: 0, startValue: 0, endValue: 0, finalFillPct: 0 };
    simState = { fillPercentage: 0, slicesPlaced: 0, slicesDiscarded: 0, currentLayer: 1, status: 'Pronto', robotRotation: Math.round(Math.PI * 0.3 * 180 / Math.PI), robotAction: 'Fermo' };
    updateStats();
    updateVacuumCups([]);
    try { await fetch(API_URL + '/cube/reset', { method: 'POST' }); } catch (e) {}
}

document.getElementById('btn-start').addEventListener('click', startSimulation);
document.getElementById('btn-stop').addEventListener('click', stopSimulation);
document.getElementById('btn-reset').addEventListener('click', resetSimulation);
window.addEventListener('load', function() { initScene(); updateStats(); });
