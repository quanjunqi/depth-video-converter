/**
 * 深度视频转换器 - 视频转黑白深度视频转换器
 *
 * 使用 Transformers.js + Depth Anything 模型在浏览器中
 * 逐帧进行深度估计，生成黑白深度视频。
 *
 * 兼容 file:// 协议：双击 index.html 即可直接使用，无需搭建服务器。
 */

// Transformers.js 模块引用（动态加载）
let _transformers = null;

// CDN 源（按优先顺序尝试）— 使用 dist/transformers.js（自包含 ESM，无外部 import）
const TRANSFORMERS_CDN = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5/dist/transformers.js';
const TRANSFORMERS_BACKUPS = [
    'https://unpkg.com/@huggingface/transformers@3.7.5/dist/transformers.js',
    'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5/dist/transformers.min.js',
];

// 模型源配置：优先使用 ModelScope 国内镜像，失败时回退到 HuggingFace 官方
const MODEL_SOURCES = [
    {
        name: 'ModelScope 国内镜像',
        host: 'https://www.modelscope.cn/models/',
        pathTemplate: '{model}/resolve/master/',
        supports: ['onnx-community/depth-anything-v2-small-ONNX'],
    },
    {
        name: 'HuggingFace 官方',
        host: 'https://huggingface.co/',
        pathTemplate: '{model}/resolve/{revision}/',
        supports: '*', // 支持任意模型 ID
    },
];

// ============================================
// State
// ============================================
const state = {
    videoFile: null,
    videoElement: null,
    depthEstimator: null,
    modelLoaded: false,
    loadedModelId: null,
    isProcessing: false,
    resultBlob: null,
    resultUrl: null,
    settings: {
        modelId: 'da3:da3-small',
        invert: false,
        contrast: 0,
        brightness: 0,
        fps: 0,
        resolution: 'min640',
        keepAudio: false,
        poseEnabled: false,
    },
    processing: {
        startTime: 0,
        totalFrames: 0,
        processedFrames: 0,
        lastUpdate: 0,
    },
    taskManage: false,
};

// ============================================
// DOM Helper
// ============================================
const $ = (id) => document.getElementById(id);
const $$ = (sel) => document.querySelectorAll(sel);

// ============================================
// file:// Protocol Detection
// ============================================
const IS_FILE_PROTOCOL = window.location.protocol === 'file:';

/**
 * 从 CDN 加载 ES 模块，兼容 file:// 协议。
 *
 * 策略（按顺序尝试）：
 * 1. 直接 import(url) — 在 http:// 或部分 file:// 环境下可用
 * 2. fetch(url) → Blob URL → import(blobUrl) — file:// 下 import() 被阻止时的回退方案
 *    原理：fetch 对 https CDN 的跨域请求不受 file:// 同源限制影响（CDN 返回 CORS *），
 *    Blob URL 属于同源，import(blobUrl) 不受 file:// ES Module 限制。
 *
 * @param {string[]} urls — CDN URL 列表，按优先级排列
 * @returns {Promise<Object>} — 模块的命名空间对象
 */
async function loadESModule(urls) {
    let lastError = null;

    // 策略 1：直接 import()
    for (const url of urls) {
        try {
            const mod = await import(url);
            console.log(`[模块加载] import() 成功: ${url}`);
            return mod;
        } catch (err) {
            console.warn(`[模块加载] import() 失败: ${url}`, err.message);
            lastError = err;
        }
    }

    // 策略 2：fetch → Blob URL → import()
    for (const url of urls) {
        try {
            console.log(`[模块加载] 尝试 fetch+blob 回退: ${url}`);
            const response = await fetch(url);
            if (!response.ok) {
                console.warn(`[模块加载] fetch 返回 ${response.status}: ${url}`);
                continue;
            }
            const text = await response.text();
            const blob = new Blob([text], { type: 'text/javascript' });
            const blobUrl = URL.createObjectURL(blob);
            const mod = await import(blobUrl);
            // 注意：不立即 revoke，模块可能内部引用 blobUrl
            console.log(`[模块加载] fetch+blob 成功: ${url}`);
            return mod;
        } catch (err) {
            console.warn(`[模块加载] fetch+blob 失败: ${url}`, err.message);
            lastError = err;
        }
    }

    // 所有策略均失败
    const protocolHint = IS_FILE_PROTOCOL
        ? '\n\n当前正在使用 file:// 协议打开。如果浏览器阻止了模块加载，请尝试以下方案：\n'
          + '1. 使用最新版 Chrome 或 Edge 浏览器\n'
          + '2. 或运行 start-server.bat 启动本地静态服务器\n'
          + '3. 或通过命令行执行 python -m http.server 8000 后访问 http://localhost:8000'
        : '';
    throw new Error(`无法加载 ES 模块，所有 CDN 源和回退策略均失败。\n原始错误: ${lastError?.message || 'Unknown'}${protocolHint}`);
}

// ============================================
// UI State Management
// ============================================
function showSection(id) {
    $$('.section').forEach(s => s.classList.remove('active'));
    $(id).classList.add('active');
    try { localStorage.setItem('dvc_section', id); } catch (e) { /* ignore */ }
}

// 刷新后恢复上次所在页面（无其他恢复逻辑覆盖时使用）
function restoreLastSection() {
    const saved = (() => { try { return localStorage.getItem('dvc_section'); } catch (e) { return null; } })();
    if (saved && $(saved)) { showSection(saved); return; }
    showSection('upload-section');
}

function showToast(message, type = 'info', duration = 4000) {
    const container = $('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => {
        toast.classList.add('removing');
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

function log(message, type = 'info') {
    const logEl = $('processing-log');
    const entry = document.createElement('div');
    entry.className = `log-entry log-${type}`;
    const time = new Date().toLocaleTimeString('zh-CN', { hour12: false });
    entry.innerHTML = `<span class="log-time">[${time}]</span> ${message}`;
    logEl.appendChild(entry);
    logEl.scrollTop = logEl.scrollHeight;
}

function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
}

function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

// ============================================
// WebGPU Detection
// ============================================
async function checkWebGPU() {
    if (!navigator.gpu) return false;
    try {
        const adapter = await navigator.gpu.requestAdapter();
        return !!adapter;
    } catch {
        return false;
    }
}

// Check WebCodecs support
function checkWebCodecs() {
    return typeof VideoEncoder !== 'undefined' && typeof VideoFrame !== 'undefined';
}

// ============================================
// Model Download UI
// ============================================

/** Show the model download panel and reset its state */
function showModelDownloadPanel(sourceName) {
    const panel = $('model-download-panel');
    const progressSection = $('progress-section');
    if (panel) {
        panel.style.display = 'block';
        // Reset to downloading state
        const icon = $('model-dl-icon');
        icon.classList.remove('done');
        $('model-dl-title').textContent = '正在下载 AI 深度模型';
        $('model-dl-subtitle').innerHTML = `首次使用需要从 <strong>${sourceName}</strong> 下载约 <strong>240MB</strong> 模型文件。下载完成后会<strong>自动永久缓存</strong>到浏览器，之后任何时候打开网页都无需重复下载`;
        $('model-dl-fill').classList.remove('done');
        $('model-dl-fill').style.width = '0%';
        $('model-dl-fill').style.background = '';
        $('model-dl-percent').classList.remove('done');
        $('model-dl-percent').textContent = '0%';
        $('model-dl-speed').textContent = '';
        $('model-dl-files').innerHTML = '';
    }
    // Hide frame processing progress during model download
    if (progressSection) {
        progressSection.style.display = 'none';
    }
    $('processing-status').textContent = '正在下载 AI 模型，请耐心等待...';
}

/** Update the model download panel with per-file progress */
function updateModelDownloadUI(fileProgress, startTime, sourceName) {
    const files = Object.entries(fileProgress);
    if (files.length === 0) return;

    // Calculate overall progress (weighted by file count)
    const totalProgress = files.reduce((sum, [_, f]) => sum + (f.progress || 0), 0);
    const overallPct = Math.round(totalProgress / files.length);

    // Update overall bar
    $('model-dl-fill').style.width = `${overallPct}%`;
    $('model-dl-percent').textContent = `${overallPct}%`;

    // Calculate download speed
    if (startTime) {
        const elapsed = (Date.now() - startTime) / 1000;
        const totalLoaded = files.reduce((sum, [_, f]) => sum + (f.loaded || 0), 0);
        if (elapsed > 0.5 && totalLoaded > 0) {
            const speed = totalLoaded / elapsed;
            $('model-dl-speed').textContent = `已下载 ${formatBytes(totalLoaded)} · 速度 ${formatBytes(speed)}/s`;
        }
    }

    // Update subtitle with live progress hint
    const allDone = files.every(([_, f]) => f.done);
    if (!allDone && overallPct > 0) {
        $('model-dl-title').textContent = `正在下载 AI 深度模型 · ${overallPct}%`;
    }

    // Render file list
    const filesContainer = $('model-dl-files');
    filesContainer.innerHTML = files.map(([name, f]) => {
        const shortName = name.split('/').pop();
        const pct = Math.round(f.progress || 0);
        const fileSize = f.total ? formatBytes(f.total) : '';
        return `
            <div class="model-dl-file">
                <div class="model-dl-file-name" title="${name}">${shortName}</div>
                <div class="model-dl-file-bar">
                    <div class="model-dl-file-fill ${f.done ? 'done' : ''}" style="width: ${pct}%"></div>
                </div>
                <span class="model-dl-file-status ${f.done ? 'done' : ''}">
                    ${f.done ? '✓ 完成' : (pct > 0 ? pct + '%' : '等待中')}
                </span>
            </div>
        `;
    }).join('');
}

/** Show completion state for model download */
function completeModelDownloadPanel() {
    const icon = $('model-dl-icon');
    icon.classList.add('done');
    // Change icon to checkmark
    icon.innerHTML = `<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>`;
    $('model-dl-title').textContent = 'AI 模型下载完成！';
    $('model-dl-subtitle').textContent = '模型已自动缓存到浏览器，下次打开网页无需重复下载，即将开始处理视频...';
    $('model-dl-fill').classList.add('done');
    $('model-dl-fill').style.width = '100%';
    $('model-dl-percent').classList.add('done');
    $('model-dl-percent').textContent = '100%';
    $('model-dl-speed').textContent = '下载完成，正在初始化模型...';

    // Mark all files as done
    const fileItems = $('model-dl-files').querySelectorAll('.model-dl-file');
    fileItems.forEach(item => {
        const fill = item.querySelector('.model-dl-file-fill');
        const status = item.querySelector('.model-dl-file-status');
        fill.classList.add('done');
        fill.style.width = '100%';
        status.classList.add('done');
        status.textContent = '✓ 完成';
    });

    // After a short delay, hide the panel and show frame processing progress
    setTimeout(() => {
        $('model-download-panel').style.display = 'none';
        $('progress-section').style.display = 'block';
        $('processing-status').textContent = '正在逐帧处理深度估计...';
    }, 1500);
}

/** Hide the model download panel (used on error) */
function hideModelDownloadPanel() {
    const panel = $('model-download-panel');
    if (panel) panel.style.display = 'none';
    const progressSection = $('progress-section');
    if (progressSection) progressSection.style.display = 'block';
}

// ============================================
// Persistent Model Cache
// ============================================
// 原实现依赖 Transformers.js 的 env.useBrowserCache（浏览器 Cache Storage API），
// 但该 API 在 file://（双击打开）等场景下不可用或不持久，导致每次打开都重新下载模型。
// 这里改为通过 Transformers.js 官方扩展点 env.customCache，
// 实现「IndexedDB 持久缓存 + 可选本地文件夹落盘」的双层缓存：
//   1. 本地文件夹（用户通过"选择模型保存文件夹"授权，模型文件直接保存到磁盘）
//   2. IndexedDB（file:// 协议下可持久化，关闭网页后仍然有效）
//   3. 网络下载（兜底，下载完成后自动写入以上两层）
// 一次下载，任何时间打开网页都直接使用，无需重复下载。

const IDB_CACHE_NAME = 'depth-video-converter-model-cache';
const IDB_CACHE_STORE = 'files';   // key: 完整模型文件 URL, value: { blob, savedAt }
const IDB_META_STORE = 'meta';     // key: 元数据 key, value: 任意值
const IDB_DIR_KEY = 'models-dir-handle'; // 保存 FileSystemDirectoryHandle

let _cacheDb = null;

function openCacheDb() {
    return new Promise((resolve, reject) => {
        if (_cacheDb) return resolve(_cacheDb);
        if (!window.indexedDB) return reject(new Error('IndexedDB 不可用'));
        const req = indexedDB.open(IDB_CACHE_NAME, 1);
        req.onupgradeneeded = (e) => {
            const db = e.target.result;
            if (!db.objectStoreNames.contains(IDB_CACHE_STORE)) {
                db.createObjectStore(IDB_CACHE_STORE);
            }
            if (!db.objectStoreNames.contains(IDB_META_STORE)) {
                db.createObjectStore(IDB_META_STORE);
            }
        };
        req.onsuccess = (e) => { _cacheDb = e.target.result; resolve(_cacheDb); };
        req.onerror = () => reject(req.error || new Error('打开 IndexedDB 失败'));
    });
}

function idbOp(db, store, method, key, value) {
    return new Promise((resolve, reject) => {
        try {
            const tx = db.transaction(store, method === 'put' || method === 'delete' ? 'readwrite' : 'readonly');
            const os = tx.objectStore(store);
            let req;
            if (method === 'get') req = os.get(key);
            else if (method === 'put') req = os.put(value, key);
            else if (method === 'delete') req = os.delete(key);
            else if (method === 'getAll') req = os.getAll();
            else return reject(new Error('未知操作: ' + method));
            req.onsuccess = () => resolve(req.result);
            req.onerror = () => reject(req.error);
        } catch (err) {
            reject(err);
        }
    });
}

/** 从 IndexedDB 读取缓存项 */
async function cacheGet(url) {
    try {
        const db = await openCacheDb();
        const item = await idbOp(db, IDB_CACHE_STORE, 'get', url);
        return (item && item.blob && item.blob.size > 0) ? item : null;
    } catch (err) {
        console.warn('[模型缓存] IndexedDB 读取失败:', err);
        return null;
    }
}

/** 写入 IndexedDB 缓存 */
async function cacheSet(url, blob) {
    try {
        const db = await openCacheDb();
        await idbOp(db, IDB_CACHE_STORE, 'put', url, { blob, savedAt: Date.now() });
        return true;
    } catch (err) {
        console.warn('[模型缓存] IndexedDB 写入失败:', err);
        return false;
    }
}

/** 读取全部缓存项（用于迁移到本地文件夹） */
async function cacheGetAll() {
    try {
        const db = await openCacheDb();
        const entries = await idbOp(db, IDB_CACHE_STORE, 'getAll');
        const keys = await new Promise((resolve, reject) => {
            const tx = db.transaction(IDB_CACHE_STORE, 'readonly');
            const req = tx.objectStore(IDB_CACHE_STORE).getAllKeys();
            req.onsuccess = () => resolve(req.result);
            req.onerror = () => reject(req.error);
        });
        return entries.map((item, i) => ({ url: keys[i], blob: item.blob })).filter(x => x.url && x.blob && x.blob.size > 0);
    } catch (err) {
        console.warn('[模型缓存] 读取全部缓存失败:', err);
        return [];
    }
}

async function metaGet(key) {
    try {
        const db = await openCacheDb();
        return await idbOp(db, IDB_META_STORE, 'get', key);
    } catch (err) {
        return undefined;
    }
}

async function metaSet(key, value) {
    try {
        const db = await openCacheDb();
        await idbOp(db, IDB_META_STORE, 'put', key, value);
        return true;
    } catch (err) {
        return false;
    }
}

/**
 * 从模型文件 URL 解析出「模型 ID + 文件相对路径」，
 * 用于在本地文件夹中以 {模型ID}/{文件路径} 的结构保存，多个模型互不冲突。
 */
function parseModelUrl(url) {
    try {
        // ModelScope: https://www.modelscope.cn/models/{modelId}/resolve/{revision}/{path}
        let m = url.match(/\/models\/(.+?)\/resolve\/(?:master|main|v?[\w.-]+)\/(.+)$/i);
        if (m) return { modelId: m[1].replace(/\/+$/, ''), relPath: m[2].split('?')[0] };
        // HuggingFace / hf-mirror: https://huggingface.co/{modelId}/resolve/{revision}/{path}
        m = url.match(/^https?:\/\/(?:huggingface\.co|hf-mirror\.com)\/(.+?)\/resolve\/(?:main|v?[\w.-]+)\/(.+)$/i);
        if (m) return { modelId: m[1].replace(/\/+$/, ''), relPath: m[2].split('?')[0] };
        // 其他（WASM 运行时等）：统一放入 _runtime 目录，按文件名区分
        const filename = decodeURIComponent(url.split('/').pop() || 'file.bin').split('?')[0];
        return { modelId: '_runtime', relPath: filename };
    } catch (err) {
        return { modelId: '_runtime', relPath: 'file.bin' };
    }
}

/** 读取本地文件夹中的模型文件（返回 Blob/File 或 null） */
async function readFileFromDir(url) {
    const dirHandle = await metaGet(IDB_DIR_KEY);
    if (!dirHandle) return null;
    try {
        const perm = await dirHandle.queryPermission({ mode: 'read' });
        if (perm !== 'granted') return null;
        const { modelId, relPath } = parseModelUrl(url);
        let current = dirHandle;
        const segments = [...modelId.split('/').filter(Boolean), ...relPath.split('/').filter(Boolean)];
        for (let i = 0; i < segments.length - 1; i++) {
            current = await current.getDirectoryHandle(segments[i]);
        }
        const fileHandle = await current.getFileHandle(segments[segments.length - 1]);
        const file = await fileHandle.getFile();
        return (file && file.size > 0) ? file : null;
    } catch (err) {
        return null; // 文件不存在 / 权限不足 / 目录结构不符
    }
}

/** 将模型文件写入本地文件夹 */
async function writeFileToDir(url, blob) {
    const dirHandle = await metaGet(IDB_DIR_KEY);
    if (!dirHandle) return false;
    try {
        const perm = await dirHandle.queryPermission({ mode: 'readwrite' });
        if (perm !== 'granted') return false;
        const { modelId, relPath } = parseModelUrl(url);
        let current = dirHandle;
        const segments = [...modelId.split('/').filter(Boolean), ...relPath.split('/').filter(Boolean)];
        for (let i = 0; i < segments.length - 1; i++) {
            current = await current.getDirectoryHandle(segments[i], { create: true });
        }
        const fileHandle = await current.getFileHandle(segments[segments.length - 1], { create: true });
        const writable = await fileHandle.createWritable();
        await writable.write(blob);
        await writable.close();
        return true;
    } catch (err) {
        console.warn('[模型缓存] 写入本地文件夹失败:', err);
        return false;
    }
}

/** 加载模型前确保本地文件夹的读取权限（用户未设置过文件夹则直接跳过，不打扰） */
let _dirPermissionChecked = false; // 同一会话只尝试授权一次，避免重复弹窗
async function ensureDirReadPermission() {
    if (_dirPermissionChecked) return false;
    _dirPermissionChecked = true;

    const dirHandle = await metaGet(IDB_DIR_KEY);
    if (!dirHandle) return false;
    try {
        let perm = await dirHandle.queryPermission({ mode: 'read' });
        if (perm !== 'granted') {
            perm = await dirHandle.requestPermission({ mode: 'read' });
        }
        return perm === 'granted';
    } catch (err) {
        return false;
    }
}

/** 在加载模型前快速检测网络是否可达 */
async function checkNetworkConnection() {
    try {
        // 尝试 fetch CDN 上的一个小文件（package.json 只有几 KB）来检测网络是否可用
        await fetch(TRANSFORMERS_CDN.replace('/dist/transformers.js', '/package.json'), {
            method: 'HEAD',
            mode: 'no-cors',
            cache: 'no-store',
        });
        return true;
    } catch {
        return false;
    }
}

/** 将 Blob/File 构造成带 content-length 的 Response，供 Transformers.js 消费 */
function cachedResponse(blob) {
    return new Response(blob, {
        status: 200,
        headers: {
            'Content-Length': String(blob.size),
            'Content-Type': blob.type || 'application/octet-stream',
        },
    });
}

/** Transformers.js 官方扩展点：env.customCache 需要实现 match / put（Web Cache API 接口） */
const modelCache = {
    async match(request) {
        const url = typeof request === 'string' ? request : (request && request.url);
        if (!url) return undefined;

        // 1) 优先本地文件夹（真正保存到磁盘的模型）
        try {
            const localFile = await readFileFromDir(url);
            if (localFile) {
                console.log(`[模型缓存] 从本地文件夹命中: ${url.split('/').pop()}`);
                return cachedResponse(localFile);
            }
        } catch (e) { /* 忽略 */ }

        // 2) IndexedDB 持久缓存
        try {
            const item = await cacheGet(url);
            if (item) {
                console.log(`[模型缓存] 从 IndexedDB 命中: ${url.split('/').pop()}`);
                return cachedResponse(item.blob);
            }
        } catch (e) { /* 忽略 */ }

        return undefined; // 未命中，走网络下载
    },

    async put(request, response) {
        const url = typeof request === 'string' ? request : (request && request.url);
        if (!url || !response || !response.ok) return;

        try {
            const blob = await response.clone().blob();
            if (!blob || blob.size === 0) return;

            // 写入 IndexedDB（必须成功，这是核心持久层）
            await cacheSet(url, blob);

            // 若用户已授权本地文件夹，同时落盘到磁盘（不阻塞下载流程）
            writeFileToDir(url, blob).then((written) => {
                if (written) {
                    console.log(`[模型缓存] 已保存到本地文件夹: ${url.split('/').pop()}`);
                }
            }).catch(() => { /* 忽略 */ });
        } catch (err) {
            console.warn('[模型缓存] 缓存写入失败:', err);
        }
    },
};

/** 将 IndexedDB 中已有缓存迁移到本地文件夹 */
async function migrateCacheToDir() {
    const items = await cacheGetAll();
    if (items.length === 0) return;
    let ok = 0;
    for (const item of items) {
        if (await writeFileToDir(item.url, item.blob)) ok++;
    }
    showToast(
        ok === items.length
            ? `已将 ${ok} 个模型缓存文件保存到本地文件夹，之后可完全离线使用`
            : `已保存 ${ok}/${items.length} 个缓存文件到本地文件夹`,
        ok === items.length ? 'success' : 'warning',
        6000
    );
}

/** 刷新"选择模型保存文件夹"区域的 UI 状态 */
async function updateModelDirUI() {
    const btn = $('choose-model-dir-btn');
    const statusEl = $('model-dl-local-status');
    if (!btn || !statusEl) return;

    const dirHandle = await metaGet(IDB_DIR_KEY);
    if (dirHandle) {
        btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/><path d="M12 11v6"/><path d="M9 14l3 3 3-3"/></svg> 更改保存文件夹`;
        statusEl.textContent = `模型将保存到文件夹：「${dirHandle.name}」，以后打开网页将直接从磁盘加载，无需网络`;
        statusEl.className = 'model-dl-local-status ok';
    } else {
        btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/><path d="M12 11v6"/><path d="M9 14l3 3 3-3"/></svg> 选择模型保存文件夹（推荐）`;
        statusEl.textContent = '模型会自动缓存到浏览器（IndexedDB），下次打开无需重复下载；选择文件夹后模型文件将直接保存到磁盘，永久离线可用';
        statusEl.className = 'model-dl-local-status';
    }
}

/** 让用户选择一个本地文件夹用于保存模型文件（可选功能，不强制） */
async function chooseModelDir() {
    if (typeof window.showDirectoryPicker !== 'function') {
        showToast('当前浏览器不支持选择文件夹，模型已自动缓存到浏览器，无需操作', 'info', 4000);
        return;
    }
    try {
        const handle = await window.showDirectoryPicker({
            id: 'depth-video-converter-models',
            mode: 'readwrite',
        });
        await metaSet(IDB_DIR_KEY, handle);
        await updateModelDirUI();
        showToast(`模型保存文件夹已设置为：「${handle.name}」`, 'success', 4000);
        // 把已缓存好的模型文件同步到新文件夹（不阻塞后续流程）
        migrateCacheToDir().catch(() => { /* 忽略迁移错误，不影响主流程 */ });
    } catch (err) {
        if (err.name === 'AbortError') return; // 用户取消
        showToast(`设置保存文件夹失败: ${err.message}`, 'error', 5000);
    }
}


// ============================================
// Model Loading
// ============================================
async function loadModel(requestedModelId) {
    // 根据模型 ID 选择合适的源（优先匹配支持该模型的源）
    const sourcesToTry = MODEL_SOURCES.filter(s => s.supports === '*' || s.supports.includes(requestedModelId));
    if (sourcesToTry.length === 0) {
        throw new Error(`没有可用的模型源支持 ${requestedModelId}`);
    }

    if (state.modelLoaded && state.loadedModelId === requestedModelId) {
        return state.depthEstimator;
    }

    log(`准备加载模型: ${requestedModelId}`, 'info');

    // 动态加载 Transformers.js，依次尝试多个 CDN 源
    if (!_transformers) {
        log('正在加载 Transformers.js 运行时...', 'info');
        const sources = [TRANSFORMERS_CDN, ...TRANSFORMERS_BACKUPS];

        try {
            _transformers = await loadESModule(sources);
            log('Transformers.js 运行时加载成功', 'success');
        } catch (err) {
            console.error('All Transformers.js CDN sources failed:', err);
            throw new Error(
                '无法加载 AI 运行时。可能原因：\n' +
                '1. 网络连接问题或 CDN 被拦截\n' +
                '2. 浏览器扩展（广告拦截、隐私保护）阻止了脚本加载\n' +
                '3. 公司网络/防火墙限制\n' +
                (IS_FILE_PROTOCOL ? '4. file:// 协议下浏览器阻止了模块加载，请尝试运行 start-server.bat\n' : '') +
                '\n建议：检查网络、关闭广告拦截扩展、或使用本地静态服务器。'
            );
        }
    }

    const { pipeline, env } = _transformers;
    env.allowLocalModels = false;

    // 持久化缓存：使用自定义缓存（IndexedDB + 可选本地文件夹），
    // 不再依赖在 file:// 下不可靠的浏览器 Cache Storage API（useBrowserCache）。
    env.useBrowserCache = false;
    env.useCustomCache = true;
    env.customCache = modelCache;

    // 若用户之前设置过本地模型文件夹，先请求读取权限（未设置过则直接跳过）
    await ensureDirReadPermission();

    const hasWebGPU = await checkWebGPU();
    const device = hasWebGPU ? 'webgpu' : 'wasm';
    log(`使用设备: ${device.toUpperCase()}`, 'info');

    if (hasWebGPU) {
        showToast('检测到 WebGPU 支持，将使用 GPU 加速推理', 'success');
    } else {
        showToast('未检测到 WebGPU，使用 WASM 模式（较慢）', 'warning');
    }

    // 提前检查网络连接，避免 Transformers.js 内部产生未捕获的 fetch 失败
    const networkOk = await checkNetworkConnection();
    if (!networkOk) {
        log('网络连接检测失败，CDN 可能不可访问', 'warning');
        showToast('网络连接不可用，模型下载可能失败。请检查网络或关闭广告拦截扩展', 'warning', 6000);
    } else {
        log('网络连接检测通过，CDN 可访问', 'success');
    }

    // 依次尝试各个模型源
    let lastError = null;
    for (const source of sourcesToTry) {
        try {
            log(`尝试从 ${source.name} 加载模型...`, 'info');

            // 设置模型下载源
            env.remoteHost = source.host;
            env.remotePathTemplate = source.pathTemplate;
            log(`模型基础 URL: ${source.host}${source.pathTemplate.replace('{model}', requestedModelId).replace('{revision}', 'main').slice(0, -1)}`, 'info');

            // 显示模型下载面板
            showModelDownloadPanel(source.name);

            // 用于追踪多文件下载进度
            const fileProgress = {}; // { filename: { progress, loaded, total, done } }
            let downloadStartTime = null;

            state.depthEstimator = await pipeline('depth-estimation', requestedModelId, {
                device: device,
                dtype: hasWebGPU ? 'fp32' : 'q8',
                progress_callback: (progress) => {
                    try {
                        const file = progress.file || 'unknown';

                        if (progress.status === 'initiate') {
                            // 文件开始下载
                            if (!downloadStartTime) downloadStartTime = Date.now();
                            fileProgress[file] = { progress: 0, loaded: 0, total: 0, done: false };
                            updateModelDownloadUI(fileProgress, downloadStartTime, source.name);
                            log(`开始下载: ${file}`, 'info');

                        } else if (progress.status === 'progress') {
                            // 文件下载进度更新
                            if (!downloadStartTime) downloadStartTime = Date.now();
                            if (!fileProgress[file]) {
                                fileProgress[file] = { progress: 0, loaded: 0, total: 0, done: false };
                            }
                            fileProgress[file].progress = progress.progress || 0;
                            fileProgress[file].loaded = progress.loaded || 0;
                            fileProgress[file].total = progress.total || 0;
                            fileProgress[file].done = false;
                            updateModelDownloadUI(fileProgress, downloadStartTime, source.name);

                        } else if (progress.status === 'done') {
                            // 单个文件下载完成
                            if (fileProgress[file]) {
                                fileProgress[file].progress = 100;
                                fileProgress[file].done = true;
                            } else {
                                fileProgress[file] = { progress: 100, loaded: 0, total: 0, done: true };
                            }
                            updateModelDownloadUI(fileProgress, downloadStartTime, source.name);
                            log(`下载完成: ${file}`, 'info');
                        }
                    } catch (callbackErr) {
                        // 确保 progress_callback 不会抛出错误导致未捕获的 Promise rejection
                        console.warn('[模型下载进度回调] 错误:', callbackErr);
                    }
                },
            });

            // 模型下载完成
            state.modelLoaded = true;
            state.loadedModelId = requestedModelId;
            log(`模型加载完成（来自 ${source.name}）`, 'success');
            completeModelDownloadPanel();
            showToast('AI 模型下载完成，开始处理视频！', 'success', 4000);
            return state.depthEstimator;

        } catch (err) {
            lastError = err;
            console.warn(`Failed to load model from ${source.name}:`, err);
            log(`从 ${source.name} 加载失败: ${err.message}`, 'warning');
            hideModelDownloadPanel();
        }
    }

    // 所有源都失败
    console.error('All model sources failed:', lastError);
    throw new Error(
        '模型加载失败，所有可用源均无法下载。\n\n' +
        '可能原因及解决方案：\n' +
        '1. 当前网络无法访问 ModelScope 或 HuggingFace\n' +
        '2. 浏览器扩展拦截了跨域请求\n' +
        '3. 模型文件较大（约 240MB），下载超时\n' +
        (IS_FILE_PROTOCOL ? '4. file:// 协议下部分浏览器会阻止跨域请求，请尝试运行 start-server.bat\n' : '') +
        '\n建议：\n' +
        '• 刷新页面后重试\n' +
        '• 关闭广告拦截/隐私保护扩展\n' +
        '• 切换网络环境（手机热点 / 公司网络）\n' +
        '• 开启可访问 HuggingFace 的 VPN/代理\n' +
        (IS_FILE_PROTOCOL ? '• 或运行 start-server.bat 启动本地服务器后访问 http://localhost:8000\n' : '') +
        `\n原始错误: ${lastError?.message || 'Unknown error'}`
    );
}

// ============================================
// Session Store（刷新恢复：参数 / 视频 / 任务）
// ============================================
const SESSION_KEYS = { settings: 'dvc_settings', job: 'dvc_active_job' };
const IDB_NAME = 'dvc-session';
const IDB_STORE = 'files';
const IDB_VIDEO_KEY = 'current-video';

function idbOpen() {
    return new Promise((resolve, reject) => {
        const req = indexedDB.open(IDB_NAME, 1);
        req.onupgradeneeded = () => req.result.createObjectStore(IDB_STORE);
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
    });
}
async function idbSet(key, value) {
    try {
        const db = await idbOpen();
        return new Promise((resolve, reject) => {
            const tx = db.transaction(IDB_STORE, 'readwrite');
            tx.objectStore(IDB_STORE).put(value, key);
            tx.oncomplete = () => resolve();
            tx.onerror = () => reject(tx.error);
        });
    } catch (e) { console.warn('IDB set failed:', e); }
}
async function idbGet(key) {
    try {
        const db = await idbOpen();
        return new Promise((resolve, reject) => {
            const req = db.transaction(IDB_STORE, 'readonly').objectStore(IDB_STORE).get(key);
            req.onsuccess = () => resolve(req.result);
            req.onerror = () => reject(req.error);
        });
    } catch (e) { console.warn('IDB get failed:', e); return null; }
}
function saveSettings() {
    try { localStorage.setItem(SESSION_KEYS.settings, JSON.stringify(state.settings)); } catch (e) {}
}
function saveActiveJob(jobId) {
    try { localStorage.setItem(SESSION_KEYS.job, JSON.stringify({ jobId, ts: Date.now() })); } catch (e) {}
}
function clearActiveJob() {
    try { localStorage.removeItem(SESSION_KEYS.job); } catch (e) {}
}
function getActiveJob() {
    try { return JSON.parse(localStorage.getItem(SESSION_KEYS.job) || 'null'); } catch (e) { return null; }
}
function getSavedSettings() {
    try { return JSON.parse(localStorage.getItem(SESSION_KEYS.settings) || 'null'); } catch (e) { return null; }
}

// ============================================
// Video Loading
// ============================================
function loadVideoFile(file, autoNav = true) {
    if (!file.type.startsWith('video/')) {
        showToast('请选择视频文件', 'error');
        return;
    }

    state.videoFile = file;

    idbSet(IDB_VIDEO_KEY, file);   // 存入本地会话，刷新后可恢复

    const url = URL.createObjectURL(file);
    const video = $('preview-video');
    video.src = url;

    video.onloadedmetadata = () => {
        const info = $('video-info');
        const duration = video.duration;
        const width = video.videoWidth;
        const height = video.videoHeight;

        info.innerHTML = `
            <div class="video-info-item">
                <span class="label">分辨率</span>
                <span class="value">${width} × ${height}</span>
            </div>
            <div class="video-info-item">
                <span class="label">时长</span>
                <span class="value">${formatTime(duration)}</span>
            </div>
            <div class="video-info-item">
                <span class="label">大小</span>
                <span class="value">${formatBytes(file.size)}</span>
            </div>
            <div class="video-info-item">
                <span class="label">格式</span>
                <span class="value">${file.name.split('.').pop().toUpperCase()}</span>
            </div>
        `;

        // Warn for long videos
        if (duration > 60) {
            showToast('视频较长，处理可能需要较长时间，建议先裁剪到 1 分钟以内', 'warning', 6000);
        }
    };

    if (autoNav) {
        showSection('settings-section');
    }
}

// ============================================
// Depth Frame Processing
// ============================================

/**
 * Convert a depth estimation result to a grayscale ImageData.
 * Supports RawImage, Tensor, ImageData, HTMLCanvasElement, ImageBitmap.
 */
function depthResultToImageData(depthResult) {
    if (!depthResult) {
        throw new Error('深度估计结果为空');
    }

    // ImageData directly
    if (depthResult instanceof ImageData) {
        return depthResult;
    }

    // Helper: convert grayscale values to RGBA Uint8ClampedArray
    function grayToRGBA(values, width, height, isFloat = false) {
        const pixelCount = width * height;
        if (values.length !== pixelCount) {
            throw new Error(`深度数据长度不匹配: 期望 ${pixelCount} (w=${width}, h=${height}), 实际 ${values.length}`);
        }

        let min = Infinity;
        let max = -Infinity;
        for (let i = 0; i < values.length; i++) {
            const v = isFloat ? values[i] : values[i] / 255;
            if (v < min) min = v;
            if (v > max) max = v;
        }
        const range = max - min || 1;

        const pixels = new Uint8ClampedArray(pixelCount * 4);
        for (let i = 0; i < values.length; i++) {
            const v = isFloat ? values[i] : values[i] / 255;
            const normalized = (v - min) / range;
            const gray = Math.max(0, Math.min(255, Math.round(normalized * 255)));
            pixels[i * 4] = gray;
            pixels[i * 4 + 1] = gray;
            pixels[i * 4 + 2] = gray;
            pixels[i * 4 + 3] = 255;
        }
        return new ImageData(pixels, width, height);
    }

    // RawImage from Transformers.js { data, width, height, channels }
    if (depthResult.width && depthResult.height && depthResult.data) {
        const width = depthResult.width;
        const height = depthResult.height;
        const data = depthResult.data;
        const channels = depthResult.channels || (depthResult.data.length / (width * height));
        const expectedRGBA = width * height * 4;

        if (data.length === expectedRGBA) {
            // Already RGBA bytes
            const rgba = data instanceof Uint8ClampedArray ? data : new Uint8ClampedArray(data);
            return new ImageData(rgba, width, height);
        }

        if (channels === 1 || data.length === width * height) {
            // Single-channel grayscale (could be uint8 or float32)
            const isFloat = data instanceof Float32Array || depthResult.format?.includes('float');
            return grayToRGBA(data, width, height, isFloat);
        }

        // Unknown channels: try to interpret as single-channel if divisible
        if (data.length % (width * height) === 0) {
            const perPixel = data.length / (width * height);
            console.warn(`RawImage has unexpected channels=${perPixel}, interpreting first channel as grayscale`);
            const pixelCount = width * height;
            const firstChannel = new Array(pixelCount);
            for (let i = 0; i < pixelCount; i++) {
                firstChannel[i] = data[i * perPixel];
            }
            const isFloat = data instanceof Float32Array || depthResult.format?.includes('float');
            return grayToRGBA(firstChannel, width, height, isFloat);
        }

        throw new Error(`RawImage 数据长度不匹配: w=${width}, h=${height}, data.length=${data.length}`);
    }

    // Tensor { data: Float32Array/Uint8Array, dims: [...], type: 'float32' }
    if (depthResult.data && depthResult.dims) {
        const values = depthResult.data;
        const dims = depthResult.dims;

        // Find H and W from dims. Common shapes: [H,W], [1,H,W], [1,1,H,W], [B,C,H,W]
        let width, height;
        if (dims.length >= 2) {
            width = dims[dims.length - 1];
            height = dims[dims.length - 2];
        } else {
            throw new Error(`无法从 dims 解析尺寸: [${dims.join(', ')}]`);
        }

        const isFloat = depthResult.type?.startsWith('float') || values instanceof Float32Array || values instanceof Float64Array;
        return grayToRGBA(values, width, height, isFloat);
    }

    throw new Error('Unsupported depth result type: ' + Object.prototype.toString.call(depthResult));
}

/**
 * Draw depth result to canvas with effects
 */
function drawDepthToCanvas(ctx, depthResult, targetWidth, targetHeight, settings) {
    // Debug log the structure of the depth result
    if (depthResult) {
        const info = {
            type: Object.prototype.toString.call(depthResult),
            width: depthResult.width,
            height: depthResult.height,
            dims: depthResult.dims,
            dataType: depthResult.data ? Object.prototype.toString.call(depthResult.data) : 'none',
            dataLength: depthResult.data ? depthResult.data.length : 0,
            channels: depthResult.channels,
            format: depthResult.format,
            resultType: depthResult.type,
        };
        console.log('Depth result structure:', info);
    }

    // Create temp canvas from depth image
    const imageData = depthResultToImageData(depthResult);
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = imageData.width;
    tempCanvas.height = imageData.height;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.putImageData(imageData, 0, 0);

    // Draw scaled to target canvas
    ctx.drawImage(tempCanvas, 0, 0, targetWidth, targetHeight);

    // Apply post-processing effects
    if (settings.invert || settings.contrast !== 0 || settings.brightness !== 0) {
        const imageData = ctx.getImageData(0, 0, targetWidth, targetHeight);
        const data = imageData.data;

        const contrastFactor = (259 * (settings.contrast + 255)) / (255 * (259 - settings.contrast));
        const brightness = settings.brightness;

        for (let i = 0; i < data.length; i += 4) {
            let gray = data[i]; // Depth image is grayscale, R=G=B

            if (settings.invert) {
                gray = 255 - gray;
            }

            // Apply contrast
            gray = contrastFactor * (gray - 128) + 128;

            // Apply brightness
            gray += brightness;

            // Clamp
            gray = Math.max(0, Math.min(255, gray));

            data[i] = gray;
            data[i + 1] = gray;
            data[i + 2] = gray;
        }

        ctx.putImageData(imageData, 0, 0);
    }
}

/**
 * Extract a single frame from a video element into a canvas.
 * Transformers.js pipeline does not always accept HTMLVideoElement directly.
 */
function videoFrameToCanvas(video, width, height) {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, width, height);
    return canvas;
}

/**
 * Seek video to specific time
 */
function seekTo(video, time) {
    return new Promise((resolve) => {
        const onSeeked = () => {
            video.removeEventListener('seeked', onSeeked);
            resolve();
        };
        video.addEventListener('seeked', onSeeked);
        video.currentTime = Math.min(time, video.duration);
    });
}

/**
 * Get supported MIME type for MediaRecorder
 */
function getSupportedMimeType() {
    // Prefer MP4 (H.264) to match the WebCodecs path output format
    const types = [
        'video/mp4;codecs=avc1.42E01E',
        'video/mp4;codecs=avc1.4D401F',
        'video/mp4;codecs=avc1.640028',
        'video/mp4;codecs=avc1',
        'video/mp4',
    ];
    for (const type of types) {
        if (MediaRecorder.isTypeSupported(type)) {
            return type;
        }
    }
    // Last resort: WebM (only if no MP4 support at all)
    const fallback = [
        'video/webm;codecs=vp9',
        'video/webm;codecs=vp8',
        'video/webm',
    ];
    for (const type of fallback) {
        if (MediaRecorder.isTypeSupported(type)) {
            return type;
        }
    }
    return 'video/mp4';
}

// ============================================
// Audio Extraction & Encoding
// ============================================

/**
 * Extract audio from a video file using AudioContext.decodeAudioData.
 * Returns an AudioBuffer or null if the video has no audio track.
 */
async function extractAudioBuffer(videoFile) {
    try {
        const arrayBuffer = await videoFile.arrayBuffer();
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        const audioCtx = new AudioContextClass();
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
        audioCtx.close();
        return audioBuffer;
    } catch (err) {
        console.warn('Failed to extract audio from video:', err);
        return null;
    }
}

/**
 * Encode an AudioBuffer into AAC chunks and add them to the muxer.
 * Returns a promise that resolves when all audio has been encoded.
 */
async function encodeAudioToMuxer(muxer, audioBuffer, durationSec) {
    if (typeof AudioEncoder === 'undefined') {
        console.warn('AudioEncoder not supported, skipping audio');
        return false;
    }

    const sampleRate = audioBuffer.sampleRate;
    const numberOfChannels = audioBuffer.numberOfChannels;
    const totalSamples = audioBuffer.length;

    // Find supported AAC codec (mp4a.40.2 = AAC-LC)
    let codec = 'mp4a.40.2';
    const codecCandidates = ['mp4a.40.2', 'mp4a.40.5'];
    for (const c of codecCandidates) {
        try {
            const support = await AudioEncoder.isConfigSupported({
                codec: c,
                sampleRate,
                numberOfChannels,
                bitrate: 128_000,
            });
            if (support.supported) {
                codec = c;
                break;
            }
        } catch (e) { /* try next */ }
    }

    let audioEncoderError = null;
    const audioEncoder = new AudioEncoder({
        output: (chunk, meta) => muxer.addAudioChunk(chunk, meta),
        error: (e) => {
            audioEncoderError = e;
            console.error('Audio encoder error:', e);
        },
    });

    try {
        audioEncoder.configure({
            codec,
            sampleRate,
            numberOfChannels,
            bitrate: 128_000,
        });
    } catch (err) {
        console.error('Failed to configure audio encoder:', err);
        return false;
    }

    // Feed audio data in chunks of ~20ms
    const chunkSize = Math.floor(sampleRate * 0.02); // 20ms frames
    const channelData = [];
    for (let ch = 0; ch < numberOfChannels; ch++) {
        channelData.push(audioBuffer.getChannelData(ch));
    }

    for (let offset = 0; offset < totalSamples; offset += chunkSize) {
        if (audioEncoderError) {
            console.error('Aborting audio encoding due to error');
            return false;
        }

        const frames = Math.min(chunkSize, totalSamples - offset);
        const timestamp = Math.round((offset / sampleRate) * 1_000_000); // microseconds

        // Build interleaved or planar data for AudioData
        // AudioData supports 'f32-planar' format
        const planarData = new Float32Array(frames * numberOfChannels);
        for (let ch = 0; ch < numberOfChannels; ch++) {
            const src = channelData[ch];
            for (let i = 0; i < frames; i++) {
                planarData[ch * frames + i] = src[offset + i];
            }
        }

        const audioData = new AudioData({
            format: 'f32-planar',
            sampleRate,
            numberOfFrames: frames,
            numberOfChannels,
            timestamp,
            data: planarData,
        });

        audioEncoder.encode(audioData);
        audioData.close();

        // Control encode queue depth
        if (audioEncoder.encodeQueueSize > 10) {
            await new Promise(r => setTimeout(r, 1));
        }
    }

    await audioEncoder.flush();
    audioEncoder.close();

    if (audioEncoderError) {
        console.error('Audio encoding completed with errors');
        return false;
    }

    console.log('Audio encoding complete');
    return true;
}

// ============================================
// Video Processing - WebCodecs Path (preferred)
// ============================================
async function processWithWebCodecs(video, estimator, settings, callbacks) {
    // 加载 mp4-muxer（输出 MP4 容器，含 H.264 + AAC）
    // 优先使用全局变量（已通过 <script> 标签预加载的 IIFE 版本，兼容 file://）
    // 回退到 loadESModule（import() 或 fetch+blob）
    let Muxer, ArrayBufferTarget;
    if (window.Mp4Muxer && window.Mp4Muxer.Muxer) {
        Muxer = window.Mp4Muxer.Muxer;
        ArrayBufferTarget = window.Mp4Muxer.ArrayBufferTarget;
        log('mp4-muxer 已通过预加载就绪', 'info');
    } else {
        log('mp4-muxer 预加载未就绪，尝试动态加载...', 'info');
        const mod = await loadESModule([
            'https://cdn.jsdelivr.net/npm/mp4-muxer@5.1.3/+esm',
            'https://unpkg.com/mp4-muxer@5.1.3/build/mp4-muxer.mjs',
        ]);
        Muxer = mod.Muxer;
        ArrayBufferTarget = mod.ArrayBufferTarget;
    }

    const srcWidth = video.videoWidth;
    const srcHeight = video.videoHeight;
    const outWidth = Math.round(srcWidth * settings.resolution);
    const outHeight = Math.round(srcHeight * settings.resolution);
    const fps = settings.fps;
    const duration = video.duration;
    const totalFrames = Math.min(Math.floor(duration * fps), 720); // Cap at 720 frames

    callbacks.onStart(totalFrames);

    // Extract audio if requested
    let audioBuffer = null;
    if (settings.keepAudio) {
        log('正在提取原始音频...', 'info');
        audioBuffer = await extractAudioBuffer(state.videoFile);
        if (audioBuffer) {
            log(`音频提取成功: ${audioBuffer.sampleRate}Hz, ${audioBuffer.numberOfChannels}声道, ${audioBuffer.duration.toFixed(1)}秒`, 'success');
        } else {
            log('该视频没有音轨或音频提取失败，将输出无声视频', 'warning');
        }
    }

    // Setup canvases
    const inputCanvas = document.createElement('canvas');
    inputCanvas.width = srcWidth;
    inputCanvas.height = srcHeight;

    const outputCanvas = document.createElement('canvas');
    outputCanvas.width = outWidth;
    outputCanvas.height = outHeight;
    const outputCtx = outputCanvas.getContext('2d', { willReadFrequently: true });

    const originalCanvas = $('original-canvas');
    const depthCanvas = $('depth-canvas');

    // Setup muxer (with optional audio track) — MP4 container
    const muxerConfig = {
        target: new ArrayBufferTarget(),
        video: {
            codec: 'avc',
            width: outWidth,
            height: outHeight,
            frameRate: fps,
        },
        fastStart: 'in-memory',
    };
    if (audioBuffer) {
        muxerConfig.audio = {
            codec: 'mp4a.40.2',
            sampleRate: audioBuffer.sampleRate,
            numberOfChannels: audioBuffer.numberOfChannels,
        };
    }
    const muxer = new Muxer(muxerConfig);

    // Setup encoder
    let encoderError = null;
    const encoder = new VideoEncoder({
        output: (chunk, meta) => muxer.addVideoChunk(chunk, meta),
        error: (e) => {
            encoderError = e;
            console.error('Encoder error:', e);
        },
    });

    // Determine codec string — H.264 (AVC) for MP4 output
    let codecString = 'avc1.4D401F';
    // Try to find a supported H.264 codec (Main → High → Baseline profiles)
    const codecs = [
        'avc1.4D401F', // Main profile, level 3.1
        'avc1.4D0028', // Main profile, level 4.0
        'avc1.640028', // High profile, level 4.0
        'avc1.640033', // High profile, level 5.1
        'avc1.42E01F', // Baseline profile, level 3.1
    ];
    for (const cs of codecs) {
        try {
            const config = {
                codec: cs,
                width: outWidth,
                height: outHeight,
                bitrate: 8_000_000,
                framerate: fps,
            };
            const support = await VideoEncoder.isConfigSupported(config);
            if (support.supported) {
                codecString = cs;
                encoder.configure(config);
                break;
            }
        } catch (e) {
            // continue
        }
    }

    log(`编码器: WebCodecs (${codecString})`, 'info');
    log(`输出尺寸: ${outWidth}×${outHeight} @ ${fps}fps`, 'info');

    const frameDurationUs = Math.round(1_000_000 / fps);

    for (let i = 0; i < totalFrames; i++) {
        if (encoderError) throw encoderError;

        const time = (i / fps);
        await seekTo(video, time);

        // Transformers.js pipeline does not accept HTMLVideoElement directly.
        // Draw the current video frame to a canvas and pass the canvas instead.
        const inputCtx = inputCanvas.getContext('2d');
        inputCtx.drawImage(video, 0, 0, srcWidth, srcHeight);

        // Run depth estimation
        let result;
        try {
            result = await estimator(inputCanvas);
        } catch (err) {
            console.warn('Estimator failed on canvas input:', err);
            throw new Error(`深度估计失败: ${err.message}`);
        }

        if (!result) {
            throw new Error('深度估计返回空结果');
        }

        // Use result.depth (RawImage) if available; otherwise fall back to result.predicted_depth (Tensor)
        const depthData = result.depth || result.predicted_depth;
        if (!depthData) {
            console.error('Unexpected estimator result:', result);
            throw new Error('深度估计结果缺少 depth 或 predicted_depth 字段');
        }

        // Draw depth to output canvas
        drawDepthToCanvas(outputCtx, depthData, outWidth, outHeight, settings);

        // Update preview canvases
        const previewCtx = originalCanvas.getContext('2d');
        originalCanvas.width = srcWidth;
        originalCanvas.height = srcHeight;
        previewCtx.drawImage(video, 0, 0, srcWidth, srcHeight);

        depthCanvas.width = outWidth;
        depthCanvas.height = outHeight;
        depthCanvas.getContext('2d').drawImage(outputCanvas, 0, 0);

        // Create VideoFrame and encode
        const frame = new VideoFrame(outputCanvas, {
            timestamp: i * frameDurationUs,
            duration: frameDurationUs,
        });

        encoder.encode(frame, { keyFrame: i % (fps * 2) === 0 });
        frame.close();

        // Control encode queue
        if (encoder.encodeQueueSize > 8) {
            while (encoder.encodeQueueSize > 4) {
                await new Promise(r => setTimeout(r, 10));
            }
        }

        callbacks.onProgress(i + 1, totalFrames);
    }

    log('正在编码视频...', 'info');
    await encoder.flush();

    // Encode audio if available
    if (audioBuffer) {
        log('正在编码音频...', 'info');
        const audioOk = await encodeAudioToMuxer(muxer, audioBuffer, duration);
        if (audioOk) {
            log('音频编码完成', 'success');
        } else {
            log('音频编码失败，将输出无声视频', 'warning');
        }
    }

    muxer.finalize();

    const { buffer } = muxer.target;
    const blob = new Blob([buffer], { type: 'video/mp4' });

    log(`编码完成，文件大小: ${formatBytes(blob.size)}`, 'success');

    return blob;
}

// ============================================
// Video Processing - MediaRecorder Path (fallback)
// ============================================
async function processWithMediaRecorder(video, estimator, settings, callbacks) {
    const srcWidth = video.videoWidth;
    const srcHeight = video.videoHeight;
    const outWidth = Math.round(srcWidth * settings.resolution);
    const outHeight = Math.round(srcHeight * settings.resolution);
    const fps = settings.fps;
    const duration = video.duration;
    const totalFrames = Math.min(Math.floor(duration * fps), 720);

    callbacks.onStart(totalFrames);

    log('使用 MediaRecorder 编码（兼容模式）', 'info');
    log(`输出尺寸: ${outWidth}×${outHeight} @ ${fps}fps`, 'info');

    // Setup canvases
    const inputCanvas = document.createElement('canvas');
    inputCanvas.width = srcWidth;
    inputCanvas.height = srcHeight;

    const outputCanvas = document.createElement('canvas');
    outputCanvas.width = outWidth;
    outputCanvas.height = outHeight;
    const outputCtx = outputCanvas.getContext('2d', { willReadFrequently: true });

    // Setup preview canvases
    const originalCanvas = $('original-canvas');
    const depthCanvas = $('depth-canvas');

    // Setup MediaRecorder
    const stream = outputCanvas.captureStream(0);
    const track = stream.getVideoTracks()[0];

    // If keeping audio, extract audio from video file and add to stream
    let audioContext = null;
    let audioSourceNode = null;
    let mediaStreamDest = null;
    if (settings.keepAudio) {
        try {
            log('正在提取原始音频（兼容模式）...', 'info');
            const audioBuffer = await extractAudioBuffer(state.videoFile);
            if (audioBuffer) {
                log(`音频提取成功: ${audioBuffer.sampleRate}Hz, ${audioBuffer.numberOfChannels}声道`, 'success');
                const AudioContextClass = window.AudioContext || window.webkitAudioContext;
                audioContext = new AudioContextClass();
                mediaStreamDest = audioContext.createMediaStreamDestination();
                const source = audioContext.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(mediaStreamDest);
                source.start();
                // Add audio track to the output stream
                const audioTrack = mediaStreamDest.stream.getAudioTracks()[0];
                if (audioTrack) {
                    stream.addTrack(audioTrack);
                    log('已添加音频轨道到输出流', 'info');
                }
            } else {
                log('该视频没有音轨，将输出无声视频', 'warning');
            }
        } catch (err) {
            log(`音频提取失败: ${err.message}`, 'warning');
        }
    }

    const mimeType = getSupportedMimeType();
    const recorder = new MediaRecorder(stream, {
        mimeType: mimeType,
        videoBitsPerSecond: 8_000_000,
    });

    const chunks = [];
    recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
    };

    recorder.start(2000);
    // Wait a bit for recorder to initialize
    await new Promise(r => setTimeout(r, 100));

    for (let i = 0; i < totalFrames; i++) {
        const time = (i / fps);
        await seekTo(video, time);

        // Transformers.js pipeline does not accept HTMLVideoElement directly.
        // Draw the current video frame to a canvas and pass the canvas instead.
        const inputCtx = inputCanvas.getContext('2d');
        inputCtx.drawImage(video, 0, 0, srcWidth, srcHeight);

        // Run depth estimation
        let result;
        try {
            result = await estimator(inputCanvas);
        } catch (err) {
            console.warn('Estimator failed on canvas input:', err);
            throw new Error(`深度估计失败: ${err.message}`);
        }

        if (!result) {
            throw new Error('深度估计返回空结果');
        }

        // Use result.depth (RawImage) if available; otherwise fall back to result.predicted_depth (Tensor)
        const depthData = result.depth || result.predicted_depth;
        if (!depthData) {
            console.error('Unexpected estimator result:', result);
            throw new Error('深度估计结果缺少 depth 或 predicted_depth 字段');
        }

        // Draw depth to output canvas
        drawDepthToCanvas(outputCtx, depthData, outWidth, outHeight, settings);

        // Update preview
        const previewCtx = originalCanvas.getContext('2d');
        originalCanvas.width = srcWidth;
        originalCanvas.height = srcHeight;
        previewCtx.drawImage(video, 0, 0, srcWidth, srcHeight);

        depthCanvas.width = outWidth;
        depthCanvas.height = outHeight;
        depthCanvas.getContext('2d').drawImage(outputCanvas, 0, 0);

        // Capture frame
        if (track.requestFrame) {
            track.requestFrame();
        }

        // Small delay to ensure frame capture
        await new Promise(r => setTimeout(r, 5));

        callbacks.onProgress(i + 1, totalFrames);
    }

    log('正在编码视频...', 'info');

    // Stop recording
    await new Promise((resolve) => {
        recorder.onstop = resolve;
        recorder.stop();
    });

    // Cleanup audio context
    if (audioContext) {
        try { audioContext.close(); } catch (e) { /* ignore */ }
    }

    const blob = new Blob(chunks, { type: mimeType.split(';')[0] });
    log(`编码完成，文件大小: ${formatBytes(blob.size)}`, 'success');

    return blob;
}

// ============================================
// Main Processing Flow
// ============================================
// ============================================
// DA3 本地引擎模式（本地服务推理）
// ============================================
const DA3_SERVER = 'http://127.0.0.1:8765';

function isDa3Model(modelId) {
    return typeof modelId === 'string' && modelId.startsWith('da3:');
}

function da3ModelId(modelId) {
    return modelId.slice(4); // 去掉 "da3:" 前缀
}

function sleep(ms) {
    return new Promise(r => setTimeout(r, ms));
}

async function checkDa3Server(showStatus = false) {
    const bar = $('da3-status');
    const dot = $('da3-status-dot');
    const text = $('da3-status-text');
    if (!bar) return null;
    if (showStatus) bar.style.display = 'flex';
    dot.className = 'da3-dot';
    text.textContent = '正在检测本地引擎...';
    try {
        const ctrl = new AbortController();
        const t = setTimeout(() => ctrl.abort(), 3000);
        const resp = await fetch(`${DA3_SERVER}/api/health`, { signal: ctrl.signal });
        clearTimeout(t);
        const data = await resp.json();
        if (!data.ok) throw new Error('服务异常');
        dot.className = 'da3-dot ok';
        const ready = (data.models || []).filter(m => m.ready);
        const want = da3ModelId(state.settings.modelId);
        const found = ready.find(m => m.id === want);
        if (found) {
            // 引擎在线且模型就绪：不打扰用户，隐藏状态条
            bar.style.display = 'none';
            return data;
        }
        dot.className = 'da3-dot err';
        text.textContent = `本地模型 ${want} 未就绪，请检查 weights/ 目录`;
        bar.style.display = 'flex';
        return data;
    } catch (e) {
        dot.className = 'da3-dot err';
        text.textContent = '引擎未就绪，正在自动启动...';
        bar.style.display = 'flex';
        return null;
    }
}

async function startDa3Processing() {
    const startBtn = $('start-btn');
    showSection('processing-section');
    $('processing-log').innerHTML = '';
    $('error-detail').style.display = 'none';
    $('model-download-panel').style.display = 'none';
    $('progress-section').style.display = 'block';
    $('progress-fill').style.width = '0%';
    $('progress-percent').textContent = '0%';

    // 原始帧同步：本地视频 seek 到当前处理帧并绘制
    let lastSeekedFrame = -1;
    const srcVideo = $('preview-video');
    const drawSrcFrame = () => {
        const oc = $('original-canvas');
        if (!oc || !srcVideo || !srcVideo.videoWidth) return;
        oc.width = srcVideo.videoWidth;
        oc.height = srcVideo.videoHeight;
        oc.getContext('2d').drawImage(srcVideo, 0, 0);
    };
    if (srcVideo) {
        srcVideo.onseeked = drawSrcFrame;
        srcVideo.onloadeddata = drawSrcFrame;
    }

    try {
        $('processing-status').textContent = '正在准备...';
        let health = await checkDa3Server(false);
        if (!health || !health.ok) {
            // 引擎为系统常驻服务，若未就绪则等待其自动启动
            for (let i = 0; i < 15; i++) {
                await sleep(1000);
                health = await checkDa3Server(false);
                if (health && health.ok) break;
            }
        }
        if (!health || !health.ok) {
            throw new Error('引擎未就绪，请稍后重试');
        }
        const model = da3ModelId(state.settings.modelId);
        const ready = (health.models || []).find(m => m.id === model && m.ready);
        if (!ready) {
            throw new Error(`本地模型 ${model} 未就绪，请检查 weights/ 目录`);
        }

        $('processing-status').textContent = '正在准备...';
        log(`使用模型: ${ready.label}`, 'info');
        log('模式：逐帧处理深度估计', 'info');

        const form = new FormData();
        form.append('video', state.videoFile);
        form.append('model', model);
        form.append('invert', String(state.settings.invert));
        form.append('contrast', String(state.settings.contrast || 0));
        form.append('brightness', String(state.settings.brightness || 0));
        form.append('scale', String(state.settings.resolution || 1));
        form.append('fps', String(state.settings.fps));   // 0=自动（保持原视频帧率）
        form.append('frame_wise', 'true');     // 逐帧处理深度估计
        form.append('keep_audio', String(!!state.settings.keepAudio));
        form.append('per_frame', 'true');      // 每帧全范围拉伸
        form.append('pose_enabled', String(!!state.settings.poseEnabled));

        const resp = await fetch(`${DA3_SERVER}/api/convert`, { method: 'POST', body: form });
        const data = await resp.json();
        if (!data.job_id) {
            if (resp.status === 429) {
                log(data.error || '任务过多', 'error');
                showToast(data.error || '任务过多，请等待完成', 'error', 5000);
                $('processing-status').textContent = '任务过多';
                return;
            }
            throw new Error(data.error || '任务提交失败');
        }
        log('开始逐帧处理深度估计...', 'info');

        await showTaskDetailById(data.job_id);
    } catch (err) {
        console.error('DA3 processing error:', err);
        log(`处理出错: ${err.message}`, 'error');
        showToast(`处理失败: ${err.message}`, 'error', 6000);
        $('processing-status').textContent = '处理失败';
        $('progress-section').style.display = 'none';
        $('error-message').textContent = err.message || '未知错误';
        $('error-detail').style.display = 'block';
    }
}

// ============ 任务列表与轮询（刷新/提交共用，服务端任务持久化） ============
function escapeHtml(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, c => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
}

function fmtTaskTime(ts) {
    if (!ts) return '';
    const d = new Date(ts * 1000);
    const p = n => String(n).padStart(2, '0');
    return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

function taskCard(j) {
    const st = j.status || 'queued';
    const badgeMap = {
        queued: ['排队中', 'gray'],
        running: ['处理中', 'blue'],
        render: ['渲染中', 'blue'],
        done: ['已完成', 'green'],
        error: ['失败', 'red'],
        stopped: ['已停止', 'gray'],
    };
    const [badgeText, badgeCls] = badgeMap[st] || [st, 'gray'];
    const pct = Math.round((j.progress || 0) * 100);
    const preview = j.latest_frame
        ? `<div class="task-preview"><img src="${j.latest_frame}" alt="深度帧预览"/></div>` : '';
    const dl = j.can_download
        ? `<button class="btn btn-primary btn-sm" data-dl="${escapeHtml(j.id)}">下载结果</button>` : '';
    const stopBtn = ['queued', 'running', 'render'].includes(st)
        ? `<button class="btn btn-secondary btn-sm" data-stop="${escapeHtml(j.id)}">停止</button>` : '';
    const err = j.error ? `<div class="task-error">${escapeHtml(j.error)}</div>` : '';
    const running = ['queued', 'running', 'render'].includes(st);
    const check = state.taskManage
        ? `<input type="checkbox" class="task-check" data-check="${escapeHtml(j.id)}"${running ? ' disabled title="运行中不可删除"' : ''}>`
        : '';
    const hint = st === 'done' ? '点击查看结果'
        : (st === 'error' ? '点击查看详情' : '点击查看进度');
    const time = `<span class="task-time">${fmtTaskTime(j.created_at)}</span>`;
    const metaBits = [];
    if (j.model) metaBits.push('模型 ' + j.model);
    if (j.fps) metaBits.push(j.fps + ' fps');
    return `<div class="task-card" data-open="${escapeHtml(j.id)}" data-status="${st}">
        <div class="task-head">
            <div class="task-name">${check}${escapeHtml(j.input_name || j.id)}</div>
            <span class="task-status-badge ${badgeCls}">${badgeText}</span>
        </div>
        <div class="task-meta">${time}${time ? ' · ' : ''}${metaBits.join(' · ')} · ${hint}</div>
        <div class="task-progress-row">
            <div class="progress-bar"><div class="progress-fill" style="width:${pct}%"></div></div>
            <span class="progress-percent">${pct}%</span>
        </div>
        ${preview}${err}
        <div class="task-actions">${stopBtn}${dl}</div>
    </div>`;
}

function renderTaskList(jobs) {
    const wrap = $('task-list');
    if (!wrap) return;
    if (!jobs || !jobs.length) {
        wrap.innerHTML = '<div class="task-empty">暂无任务</div>';
        return;
    }
    wrap.innerHTML = jobs.map(taskCard).join('');
    wrap.querySelectorAll('.task-card').forEach(card => {
        const id = card.dataset.open;
        const st = card.dataset.status;
        const dlBtn = card.querySelector('[data-dl]');
        if (dlBtn) {
            dlBtn.addEventListener('click', e => {
                e.stopPropagation();
                downloadJob(id);
            });
        }
        const stopBtn = card.querySelector('[data-stop]');
        if (stopBtn) {
            stopBtn.addEventListener('click', e => {
                e.stopPropagation();
                stopJob(id);
            });
        }
        const cb = card.querySelector('[data-check]');
        if (cb) {
            cb.addEventListener('click', e => e.stopPropagation());
            cb.addEventListener('change', () => updateDeleteBtn());
        }
        card.addEventListener('click', e => {
            if (e.target.closest('[data-check]')) return;
            openTask(id, st);
        });
    });
    updateDeleteBtn();
}

// ---- 批量管理 ----
function setTaskManage(on) {
    state.taskManage = !!on;
    const bar = $('tasks-manage-bar');
    const btn = $('tasks-manage-btn');
    const selAll = $('tasks-select-all');
    if (bar) bar.style.display = state.taskManage ? 'flex' : 'none';
    if (btn) btn.textContent = state.taskManage ? '退出管理' : '批量管理';
    if (selAll) selAll.checked = false;
    refreshTaskList();
}

async function refreshTaskList() {
    try {
        const data = await (await fetch(`${DA3_SERVER}/api/jobs`)).json();
        renderTaskList(data.jobs || []);
        return data.jobs || [];
    } catch (e) {
        renderTaskList([]);
        return [];
    }
}

function selectedTaskIds() {
    return [...document.querySelectorAll('#task-list .task-check:checked')]
        .map(c => c.dataset.check);
}

function updateDeleteBtn() {
    const btn = $('tasks-delete-btn');
    if (btn) btn.disabled = selectedTaskIds().length === 0;
}

async function deleteSelectedTasks() {
    const ids = selectedTaskIds();
    if (!ids.length) return;
    if (!confirm(`确定删除选中的 ${ids.length} 个任务吗？\n将同时删除其输入视频、输出结果、日志等全部文件，且不可恢复。`)) return;
    try {
        const resp = await fetch(`${DA3_SERVER}/api/jobs/delete`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ids }),
        });
        const data = await resp.json();
        if (!resp.ok) {
            showToast(data.error || '删除失败', 'error', 5000);
            return;
        }
        showToast(`已删除 ${data.deleted} 个任务`, 'success');
        setTaskManage(false);
        const jobs = await refreshTaskList();
        if (jobs.some(j => ['queued', 'running', 'render'].includes(j.status))) {
            pollTasksList();
        }
    } catch (e) {
        showToast(`删除失败: ${e.message}`, 'error', 5000);
    }
}

async function openTask(jobId, status) {
    try {
        const data = await (await fetch(`${DA3_SERVER}/api/jobs`)).json();
        const job = (data.jobs || []).find(j => j.id === jobId);
        if (!job) { showToast('任务不存在', 'error'); return; }
        if (job.status === 'done') {
            showResultForJob(job);          // 已完成 → 结果页
        } else {
            showTaskDetail(job);            // 进行中/排队/失败 → 进度详情页
        }
    } catch (e) {
        showToast(`获取任务失败: ${e.message}`, 'error', 5000);
    }
}

async function downloadJob(jobId) {
    try {
        const blob = await (await fetch(`${DA3_SERVER}/api/download/${jobId}`)).blob();
        const jobs = (await (await fetch(`${DA3_SERVER}/api/jobs`)).json()).jobs || [];
        const j = jobs.find(x => x.id === jobId) || {};
        const stem = (j.input_name || 'depth').replace(/\.[^.]+$/, '');
        saveBlob(blob, `${stem}_depth.mp4`);
    } catch (e) {
        showToast(`下载失败: ${e.message}`, 'error', 5000);
    }
}

function saveBlob(blob, name) {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 3000);
}

// ============ 任务列表页 / 任务进度页 / 任务结果页 ============

// 打开任务列表页（顶栏入口 / 详情页返回共用）
async function openTasksPage() {
    showSection('tasks-section');
    try {
        const data = await (await fetch(`${DA3_SERVER}/api/jobs`)).json();
        const jobs = data.jobs || [];
        renderTaskList(jobs);
        if (jobs.some(j => ['queued', 'running', 'render'].includes(j.status))) {
            pollTasksList();          // 有活跃任务则持续刷新列表
        }
    } catch (e) {
        renderTaskList([]);
    }
}

// 任务列表页轮询：刷新卡片直到队列清空
async function pollTasksList() {
    let failCount = 0;
    while (true) {
        let jobs = [];
        try {
            const data = await (await fetch(`${DA3_SERVER}/api/jobs`)).json();
            jobs = data.jobs || [];
            failCount = 0;
        } catch (e) {
            failCount++;
            if (failCount >= 10) break;
            await sleep(2000);
            continue;
        }
        renderTaskList(jobs);
        if (!jobs.some(j => ['queued', 'running', 'render'].includes(j.status))) break;
        await sleep(2000);
    }
}

// 任务进度页：显示对照面板 + 进度 + 日志，轮询单个任务
async function showTaskDetail(job) {
    state.currentJobId = job.id;
    showSection('processing-section');
    $('task-detail-title').textContent = job.input_name || job.id;
    $('processing-status').textContent = '正在获取任务状态...';
    $('processing-log').innerHTML = '';
    $('error-detail').style.display = 'none';
    $('progress-section').style.display = 'block';
    const pv = $('processing-preview');
    if (pv) pv.style.display = '';
    const plog = $('processing-log');
    if (plog) plog.style.display = '';
    await pollTaskDetail(job.id);
}

async function showTaskDetailById(jobId) {
    try {
        const data = await (await fetch(`${DA3_SERVER}/api/jobs`)).json();
        const job = (data.jobs || []).find(j => j.id === jobId);
        if (job) await showTaskDetail(job);
        else showToast('任务创建失败', 'error', 5000);
    } catch (e) {
        showToast(`获取任务失败: ${e.message}`, 'error', 5000);
    }
}

async function pollTaskDetail(jobId) {
    const srcVideo = $('preview-video');
    const drawSrcFrame = () => {
        const oc = $('original-canvas');
        if (!oc || !srcVideo || !srcVideo.videoWidth) return;
        oc.width = srcVideo.videoWidth;
        oc.height = srcVideo.videoHeight;
        oc.getContext('2d').drawImage(srcVideo, 0, 0);
    };
    if (srcVideo) {
        srcVideo.onseeked = drawSrcFrame;
        srcVideo.onloadeddata = drawSrcFrame;
    }
    const startTime = Date.now();
    let lastFrameUrl = '';
    let lastPoseFrameUrl = '';
    let lastLogTail = '';
    let failCount = 0;

    while (true) {
        let s;
        try {
            const resp = await fetch(`${DA3_SERVER}/api/progress/${jobId}`);
            s = await resp.json();
            if (!resp.ok || s.error) throw new Error(s.error || '任务不存在');
            failCount = 0;
        } catch (e) {
            failCount++;
            $('processing-status').textContent = failCount > 3
                ? '无法连接本地引擎，请确认服务已启动' : '正在连接本地引擎...';
            if (failCount >= 10) {
                showToast('无法连接本地引擎，请重新打开服务后刷新页面', 'error', 6000);
                break;
            }
            await sleep(2000);
            continue;
        }
        const pct = Math.round((s.progress || 0) * 100);
        $('progress-fill').style.width = `${pct}%`;
        $('progress-percent').textContent = `${pct}%`;
        const elapsed = (Date.now() - startTime) / 1000;
        $('stat-elapsed').textContent = formatTime(elapsed);
        if (s.current != null && s.total) {
            $('stat-frames').textContent = `${s.current} / ${s.total}`;
            if (elapsed > 1 && s.current > 0) {
                const fps = s.current / elapsed;
                $('stat-speed').textContent = `${fps.toFixed(1)} fps`;
                $('stat-eta').textContent = formatTime((s.total - s.current) / fps);
            }
        }
        // 深度帧面板
        if (s.latest_frame && s.latest_frame !== lastFrameUrl) {
            lastFrameUrl = s.latest_frame;
            const img = new Image();
            img.onload = () => {
                const cv = $('depth-canvas');
                if (!cv) return;
                cv.width = img.naturalWidth;
                cv.height = img.naturalHeight;
                cv.getContext('2d').drawImage(img, 0, 0);
            };
            img.src = s.latest_frame;
        }
        // 骨架帧面板（pose_enabled 时显示三栏）
        if (s.pose_enabled) {
            const pv = $('processing-preview');
            if (pv && !pv.classList.contains('pose-mode')) pv.classList.add('pose-mode');
            if (s.latest_pose_frame && s.latest_pose_frame !== lastPoseFrameUrl) {
                lastPoseFrameUrl = s.latest_pose_frame;
                const pimg = new Image();
                pimg.onload = () => {
                    const pcv = $('pose-canvas');
                    if (!pcv) return;
                    pcv.width = pimg.naturalWidth;
                    pcv.height = pimg.naturalHeight;
                    pcv.getContext('2d').drawImage(pimg, 0, 0);
                };
                pimg.src = s.latest_pose_frame;
            }
        }
        // 原始帧同步
        if (s.current != null && s.total && srcVideo && srcVideo.duration) {
            const target = (s.current / s.total) * srcVideo.duration;
            if (Math.abs(srcVideo.currentTime - target) > 0.1) {
                srcVideo.currentTime = target;
            }
        }
        // 日志
        if (s.log_tail && s.log_tail !== lastLogTail) {
            lastLogTail = s.log_tail;
            const lines = s.log_tail.split('\n').slice(-2);
            for (const ln of lines) {
                if (ln.trim()) log(ln.replace(/\u001b\[[0-9;]*m/g, ''), 'info');
            }
        }
        $('processing-status').textContent = s.status === 'queued'
            ? '任务排队中（最多同时 2 个）...' : '正在逐帧处理深度估计...';
        // 停止按钮显隐
        const taskStopBtn = $('task-stop-btn');
        if (taskStopBtn) {
            taskStopBtn.style.display = ['queued', 'running', 'render'].includes(s.status) ? '' : 'none';
        }
        if (s.status === 'stopped') {
            $('processing-status').textContent = '任务已手动停止';
            break;
        }
        if (s.status === 'done') {
            $('processing-status').textContent = '任务完成，正在获取结果...';
            $('progress-fill').style.width = '100%';
            $('progress-percent').textContent = '100%';
            const data = await (await fetch(`${DA3_SERVER}/api/jobs`)).json();
            const job = (data.jobs || []).find(j => j.id === jobId);
            if (job) { showResultForJob(job); return; }
            break;
        }
        if (s.status === 'error') {
            $('processing-status').textContent = '处理失败';
            $('error-detail').style.display = 'block';
            $('error-message').textContent = s.error || '转换失败';
            break;
        }
        await sleep(2000);
    }
}

// 任务结果页：按任务 ID 加载结果
async function showResultForJob(job) {
    try {
        const blob = await (await fetch(`${DA3_SERVER}/api/download/${job.id}`)).blob();
        if (state.resultUrl) URL.revokeObjectURL(state.resultUrl);
        state.resultUrl = URL.createObjectURL(blob);
        state.resultBlob = blob;
        state.resultName = (job.input_name || 'depth').replace(/\.[^.]+$/, '');
        state.resultJobId = job.id;
        state.resultPoseEnabled = !!job.pose_enabled;
        state.processing.startTime = (job.created_at || Date.now() / 1000) * 1000;
        state.processing.processedFrames = (job.info && job.info.frames) || 0;
        showResult();
    } catch (e) {
        showToast(`加载结果失败: ${e.message}`, 'error', 6000);
    }
}

async function startProcessing() {
    if (state.isProcessing) return;
    // 本地 DA3 引擎模式：走本地服务
    if (isDa3Model(state.settings.modelId)) {
        state.isProcessing = true;
        const startBtn = $('start-btn');
        startBtn.disabled = true;
        startBtn.innerHTML = '<span class="spinner"></span> 正在准备...';
        try {
            await startDa3Processing();
        } finally {
            state.isProcessing = false;
            startBtn.disabled = false;
            startBtn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3" fill="currentColor"/></svg> 开始转换`;
        }
        return;
    }
    state.isProcessing = true;

    const startBtn = $('start-btn');
    startBtn.disabled = true;
    startBtn.innerHTML = '<span class="spinner"></span> 正在准备...';

    showSection('processing-section');
    $('processing-log').innerHTML = '';

    const video = $('preview-video');
    video.pause();

    // 重置错误显示区域
    $('error-detail').style.display = 'none';

    // 如果模型已加载，直接显示帧处理进度；否则显示模型下载面板
    if (state.modelLoaded && state.loadedModelId === state.settings.modelId) {
        $('model-download-panel').style.display = 'none';
        $('progress-section').style.display = 'block';
    } else {
        $('model-download-panel').style.display = 'none';
        $('progress-section').style.display = 'none';
    }

    try {
        // Load model
        $('processing-status').textContent = '正在加载 AI 模型...';
        $('progress-fill').style.width = '0%';
        $('progress-percent').textContent = '0%';

        const estimator = await loadModel(state.settings.modelId);

        // Start processing
        $('processing-status').textContent = '正在逐帧处理深度估计...';

        const useWebCodecs = checkWebCodecs();
        log(`浏览器支持 WebCodecs: ${useWebCodecs ? '是' : '否'}`, 'info');

        const callbacks = {
            onStart: (totalFrames) => {
                state.processing.startTime = Date.now();
                state.processing.totalFrames = totalFrames;
                state.processing.processedFrames = 0;
                state.processing.lastUpdate = Date.now();
                log(`开始处理 ${totalFrames} 帧`, 'info');
            },
            onProgress: (current, total) => {
                state.processing.processedFrames = current;

                const pct = Math.round((current / total) * 100);
                $('progress-fill').style.width = `${pct}%`;
                $('progress-percent').textContent = `${pct}%`;
                $('stat-frames').textContent = `${current} / ${total}`;

                const elapsed = (Date.now() - state.processing.startTime) / 1000;
                const speed = current / elapsed;
                $('stat-speed').textContent = `${speed.toFixed(1)} fps`;
                $('stat-elapsed').textContent = formatTime(elapsed);

                const remaining = (total - current) / speed;
                if (isFinite(remaining) && remaining > 0) {
                    $('stat-eta').textContent = formatTime(remaining);
                }

                if (current % 10 === 0 || current === total) {
                    log(`已处理 ${current}/${total} 帧 (${pct}%)`, 'info');
                }
            },
        };

        let blob;
        if (useWebCodecs) {
            try {
                blob = await processWithWebCodecs(video, estimator, state.settings, callbacks);
            } catch (e) {
                log(`WebCodecs 处理失败: ${e.message}，切换到兼容模式`, 'error');
                console.error(e);
                // Reset progress
                state.processing.startTime = Date.now();
                blob = await processWithMediaRecorder(video, estimator, state.settings, callbacks);
            }
        } else {
            blob = await processWithMediaRecorder(video, estimator, state.settings, callbacks);
        }

        state.resultBlob = blob;
        if (state.resultUrl) URL.revokeObjectURL(state.resultUrl);
        state.resultUrl = URL.createObjectURL(blob);

        // Show result
        showResult();

    } catch (err) {
        console.error('Processing error:', err);
        log(`处理出错: ${err.message}`, 'error');
        showToast(`处理失败: ${err.message}`, 'error', 6000);

        // 隐藏模型下载面板和进度条，显示错误面板
        $('model-download-panel').style.display = 'none';
        $('processing-status').textContent = '处理失败';
        $('progress-section').style.display = 'none';
        $('error-message').textContent = err.message || '未知错误';
        $('error-detail').style.display = 'block';
    } finally {
        state.isProcessing = false;
        startBtn.disabled = false;
        startBtn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3" fill="currentColor"/></svg> 开始转换`;
    }
}

// ============================================
// Result Display
// ============================================
function showResult() {
    const video = $('result-video');
    video.src = state.resultUrl;

    const elapsed = (Date.now() - state.processing.startTime) / 1000;
    const stats = $('result-stats');
    stats.innerHTML = `
        <div class="result-stat">
            <span class="label">总帧数</span>
            <span class="value">${state.processing.processedFrames}</span>
        </div>
        <div class="result-stat">
            <span class="label">处理耗时</span>
            <span class="value">${formatTime(elapsed)}</span>
        </div>
        <div class="result-stat">
            <span class="label">输出大小</span>
            <span class="value">${formatBytes(state.resultBlob.size)}</span>
        </div>
        <div class="result-stat">
            <span class="label">输出格式</span>
            <span class="value">${state.resultBlob.type.includes('mp4') ? 'MP4' : 'WebM'}</span>
        </div>
    `;

    showSection('result-section');
    // 骨架视频下载按钮显隐
    const poseBtn = $('download-pose-btn');
    if (poseBtn) {
        poseBtn.style.display = state.resultPoseEnabled ? '' : 'none';
    }
    showToast('深度视频转换完成！', 'success');
}

// ============================================
// Download
// ============================================
async function downloadResult() {
    if (!state.resultBlob) return;

    const originalName = state.resultName || state.videoFile?.name || 'video';
    const baseName = originalName.replace(/\.[^.]+$/, '');
    // Determine extension from actual blob type (MP4 primary, WebM last-resort fallback)
    const isMp4 = state.resultBlob.type.includes('mp4');
    const ext = isMp4 ? 'mp4' : 'webm';
    const mimeMain = isMp4 ? 'video/mp4' : 'video/webm';
    const typeLabel = isMp4 ? 'MP4 视频' : 'WebM 视频';
    const fileName = `${baseName}_depth.${ext}`;

    const btn = $('download-btn');
    const originalHtml = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> 准备下载...`;

    try {
        // 优先使用 File System Access API：直接弹出保存对话框，最稳定
        if (typeof window.showSaveFilePicker === 'function') {
            const handle = await window.showSaveFilePicker({
                suggestedName: fileName,
                types: [{
                    description: typeLabel,
                    accept: { [mimeMain]: ['.' + ext] },
                }],
            });
            const writable = await handle.createWritable();
            await writable.write(state.resultBlob);
            await writable.close();
            showToast('文件已保存', 'success');
            return;
        }

        // 备用方案 1：IE/Edge 旧版 msSaveOrOpenBlob
        if (typeof navigator.msSaveOrOpenBlob === 'function') {
            navigator.msSaveOrOpenBlob(state.resultBlob, fileName);
            showToast('下载已启动', 'success');
            return;
        }

        // 备用方案 2：创建临时 <a download> 触发下载
        // 为下载单独创建新的 Object URL，避免与预览视频共用 URL 被提前释放
        const downloadUrl = URL.createObjectURL(state.resultBlob);
        const a = document.createElement('a');
        a.href = downloadUrl;
        a.download = fileName;
        a.rel = 'noopener noreferrer';
        a.style.display = 'none';
        document.body.appendChild(a);

        // 使用 MouseEvent 触发，比 a.click() 更可靠
        const event = new MouseEvent('click', {
            bubbles: true,
            cancelable: true,
            view: window,
        });
        a.dispatchEvent(event);

        // 延迟清理 DOM 和 Object URL
        setTimeout(() => {
            if (a.parentNode) document.body.removeChild(a);
            URL.revokeObjectURL(downloadUrl);
        }, 200);

        showToast('下载已启动', 'success');
    } catch (err) {
        console.error('Download failed:', err);
        if (err.name === 'AbortError') {
            showToast('已取消保存', 'info');
        } else {
            showToast(`下载失败: ${err.message}`, 'error', 5000);
        }
    } finally {
        setTimeout(() => {
            btn.disabled = false;
            btn.innerHTML = originalHtml;
        }, 600);
    }
}

// ============================================
// Reset
// ============================================
function reset() {
    // Revoke URLs
    if (state.resultUrl) {
        URL.revokeObjectURL(state.resultUrl);
        state.resultUrl = null;
    }

    // Reset video
    const video = $('preview-video');
    if (video.src) {
        URL.revokeObjectURL(video.src);
        video.src = '';
    }

    // Reset state
    state.videoFile = null;
    state.resultBlob = null;
    state.isProcessing = false;

    // Reset progress display
    $('progress-fill').style.width = '0%';
    $('progress-percent').textContent = '0%';
    $('stat-frames').textContent = '0 / 0';
    $('stat-speed').textContent = '-- fps';
    $('stat-eta').textContent = '--';
    $('stat-elapsed').textContent = '0:00';
    $('processing-log').innerHTML = '';

    // Reset file input
    $('file-input').value = '';

    showSection('upload-section');
}

// ============================================
// Event Listeners
// ============================================
function bindEvents() {
    // File upload - click
    $('upload-area').addEventListener('click', () => {
        $('file-input').click();
    });

    // File upload - change
    $('file-input').addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) loadVideoFile(file);
    });

    // File upload - drag & drop
    const uploadArea = $('upload-area');
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    uploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
    });
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file) loadVideoFile(file);
    });

    // Back to upload
    $('back-to-upload-btn').addEventListener('click', () => {
        showSection('upload-section');
    });

    // Model select
    $('model-select').addEventListener('change', (e) => {
        state.settings.modelId = e.target.value;
        const option = e.target.selectedOptions[0];
        $('model-desc').textContent = option.dataset.desc || '';
        if (isDa3Model(e.target.value)) {
            checkDa3Server(false); // 静默检测，仅离线时给出提示
        } else {
            $('da3-status').style.display = 'none';
        }
        saveSettings();
    });

    // DA3 服务重新检测
    const da3Retry = $('da3-retry-btn');
    if (da3Retry) {
        da3Retry.addEventListener('click', () => checkDa3Server(true));
    }

    // Depth direction toggle
    $('depth-direction').addEventListener('click', (e) => {
        const btn = e.target.closest('.toggle-btn');
        if (!btn) return;
        $('depth-direction').querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.settings.invert = btn.dataset.invert === 'true';
        saveSettings();
    });

    // Contrast slider
    $('contrast-slider').addEventListener('input', (e) => {
        state.settings.contrast = parseInt(e.target.value);
        $('contrast-value').textContent = e.target.value;
        saveSettings();
    });

    // Brightness slider
    $('brightness-slider').addEventListener('input', (e) => {
        state.settings.brightness = parseInt(e.target.value);
        $('brightness-value').textContent = e.target.value;
        saveSettings();
    });

    // FPS toggle
    $('fps-group').addEventListener('click', (e) => {
        const btn = e.target.closest('.toggle-btn');
        if (!btn) return;
        $('fps-group').querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.settings.fps = parseInt(btn.dataset.fps) || 0;
        saveSettings();
    });

    // Resolution toggle（支持 min640 特殊档：短边不低于 640）
    $('resolution-group').addEventListener('click', (e) => {
        const btn = e.target.closest('.toggle-btn');
        if (!btn) return;
        $('resolution-group').querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.settings.resolution = btn.dataset.res;
        saveSettings();
    });

    // Audio toggle
    $('audio-group').addEventListener('click', (e) => {
        const btn = e.target.closest('.toggle-btn');
        if (!btn) return;
        $('audio-group').querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.settings.keepAudio = btn.dataset.audio === 'true';
        saveSettings();
    });

    // 同时输出骨架开关
    const poseGroup = $('pose-group');
    if (poseGroup) {
        poseGroup.addEventListener('click', (e) => {
            const btn = e.target.closest('.toggle-btn');
            if (!btn) return;
            poseGroup.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.settings.poseEnabled = btn.dataset.pose === 'true';
            saveSettings();
        });
    }

    // 首页入口：点击左上角 logo 返回首页
    const logoHome = $('logo-home');
    if (logoHome) {
        logoHome.addEventListener('click', () => {
            showSection('upload-section');
            if (state.isProcessing) return;   // 处理中不打断
        });
    }

    // 任务记录（顶栏入口：任务列表）
    const tasksNav = $('tasks-nav-btn');
    if (tasksNav) {
        tasksNav.addEventListener('click', () => openTasksPage());
    }

    // 任务进度页：返回任务列表
    const backToTasks = $('back-to-tasks-btn');
    if (backToTasks) {
        backToTasks.addEventListener('click', () => openTasksPage());
    }

    // 批量管理任务
    const manageBtn = $('tasks-manage-btn');
    if (manageBtn) {
        manageBtn.addEventListener('click', () => setTaskManage(!state.taskManage));
    }
    const manageCancel = $('tasks-manage-cancel-btn');
    if (manageCancel) {
        manageCancel.addEventListener('click', () => setTaskManage(false));
    }
    const selAll = $('tasks-select-all');
    if (selAll) {
        selAll.addEventListener('change', (e) => {
            const on = e.target.checked;
            document.querySelectorAll('#task-list .task-check').forEach(cb => {
                if (!cb.disabled) cb.checked = on;
            });
            updateDeleteBtn();
        });
    }
    const deleteBtn = $('tasks-delete-btn');
    if (deleteBtn) {
        deleteBtn.addEventListener('click', deleteSelectedTasks);
    }

    // 视频裁剪
    const clipNavBtn = $('clip-nav-btn');
    if (clipNavBtn) {
        clipNavBtn.addEventListener('click', () => showSection('clip-section'));
    }
    const clipUploadArea = $('clip-upload-area');
    const clipFileInput = $('clip-file-input');
    if (clipUploadArea && clipFileInput) {
        clipUploadArea.addEventListener('click', () => clipFileInput.click());
        clipFileInput.addEventListener('change', e => {
            if (e.target.files && e.target.files[0]) setClipFile(e.target.files[0]);
        });
        clipUploadArea.addEventListener('dragover', e => {
            e.preventDefault();
            clipUploadArea.classList.add('dragover');
        });
        clipUploadArea.addEventListener('dragleave', () => {
            clipUploadArea.classList.remove('dragover');
        });
        clipUploadArea.addEventListener('drop', e => {
            e.preventDefault();
            clipUploadArea.classList.remove('dragover');
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                setClipFile(e.dataTransfer.files[0]);
            }
        });
    }
    const clipStartBtn = $('clip-start-btn');
    if (clipStartBtn) clipStartBtn.addEventListener('click', startClip);
    const clipResetBtn = $('clip-reset-btn');
    if (clipResetBtn) clipResetBtn.addEventListener('click', resetClipForm);

    // 任务进度页停止按钮
    const taskStopBtn = $('task-stop-btn');
    if (taskStopBtn) {
        taskStopBtn.addEventListener('click', () => {
            if (state.currentJobId) stopJob(state.currentJobId);
        });
    }

    // Start processing
    $('start-btn').addEventListener('click', () => {
        console.log('[深度视频转换器] Start button clicked');
        startProcessing();
    });

    // Download
    $('download-btn').addEventListener('click', downloadResult);

    // 骨架视频下载
    const downloadPoseBtn = $('download-pose-btn');
    if (downloadPoseBtn) {
        downloadPoseBtn.addEventListener('click', () => {
            if (!state.resultJobId) return;
            const a = document.createElement('a');
            a.href = `${DA3_SERVER}/api/download/${state.resultJobId}?type=pose`;
            a.download = `${state.resultName || 'video'}_pose.mp4`;
            document.body.appendChild(a);
            a.click();
            setTimeout(() => a.remove(), 3000);
        });
    }

    // Reset
    $('reset-btn').addEventListener('click', reset);

    // Retry from error panel
    $('retry-btn')?.addEventListener('click', () => {
        $('error-detail').style.display = 'none';
        $('progress-section').style.display = 'block';
        startProcessing();
    });

    // Back to settings from error panel
    $('back-from-error-btn')?.addEventListener('click', () => {
        showSection('settings-section');
    });

    // 选择模型保存文件夹（将模型真正保存到磁盘，永久离线可用）
    $('choose-model-dir-btn')?.addEventListener('click', () => {
        chooseModelDir();
    });
}

// ============================================
// Init
// ============================================

// 全局错误捕获 — 确保任何未捕获的错误都能被看到
window.addEventListener('error', (e) => {
    console.error('Global error:', e.error || e.message);
    showToast(`脚本错误: ${e.message}`, 'error', 8000);
});

window.addEventListener('unhandledrejection', (e) => {
    console.error('Unhandled rejection:', e.reason);
    const msg = e.reason?.message || e.reason;
    if (msg && (String(msg).includes('Failed to fetch') || String(msg).includes('NetworkError'))) {
        showToast('网络请求失败，请检查网络连接或关闭广告拦截扩展后重试', 'error', 8000);
    } else {
        showToast(`异步错误: ${msg}`, 'error', 8000);
    }
});

async function init() {
    console.log('[深度视频转换器] Script loaded successfully');

    // 检查关键 DOM 元素是否存在
    const criticalIds = ['start-btn', 'upload-area', 'file-input', 'preview-video',
                         'model-select', 'contrast-slider', 'brightness-slider',
                         'progress-fill', 'progress-percent', 'processing-log',
                         'download-btn', 'reset-btn', 'error-detail', 'retry-btn',
                         'back-from-error-btn'];
    const missing = criticalIds.filter(id => !$(id));
    if (missing.length > 0) {
        console.error('[深度视频转换器] Missing DOM elements:', missing);
        showToast(`缺少页面元素: ${missing.join(', ')}`, 'error', 8000);
        return;
    }
    console.log('[深度视频转换器] All DOM elements verified');

    // 绑定事件监听器
    bindEvents();
    console.log('[深度视频转换器] Event listeners bound');

    // 检查浏览器能力
    const hasWebGPU = await checkWebGPU();
    const hasWebCodecs = checkWebCodecs();
    console.log('[深度视频转换器] WebGPU:', hasWebGPU, '| WebCodecs:', hasWebCodecs);

    if (IS_FILE_PROTOCOL) {
        console.log('[深度视频转换器] file:// 协议检测到，已启用兼容模式');
        showToast('双击打开模式已就绪，请上传视频开始', 'info', 4000);
    } else {
        showToast('深度视频转换器 已就绪，请上传视频开始', 'info', 3000);
    }

    restoreLastSection();   // 立即恢复刷新前所在页面（无闪烁）
    restoreSession();       // 异步恢复 参数/视频/任务（有活跃任务时覆盖为任务列表）
}

// 刷新后恢复：参数 / 视频 / 进行中的任务
async function restoreSession() {
    // 1) 恢复参数设置
    const saved = getSavedSettings();
    if (saved && typeof saved === 'object') {
        Object.assign(state.settings, saved);
        const ms = $('model-select');
        if (ms && [...ms.options].some(o => o.value === saved.modelId)) {
            ms.value = saved.modelId;
            const opt = ms.selectedOptions[0];
            $('model-desc').textContent = opt.dataset.desc || '';
        }
        const inv = !!saved.invert;
        $('depth-direction').querySelectorAll('.toggle-btn').forEach(b =>
            b.classList.toggle('active', b.dataset.invert === String(inv)));
        if (saved.contrast != null) { $('contrast-slider').value = saved.contrast; $('contrast-value').textContent = saved.contrast; }
        if (saved.brightness != null) { $('brightness-slider').value = saved.brightness; $('brightness-value').textContent = saved.brightness; }
        $('fps-group').querySelectorAll('.toggle-btn').forEach(b =>
            b.classList.toggle('active', parseInt(b.dataset.fps) === (saved.fps || 0)));
        $('resolution-group').querySelectorAll('.toggle-btn').forEach(b =>
            b.classList.toggle('active', b.dataset.res === String(saved.resolution)));
        $('audio-group').querySelectorAll('.toggle-btn').forEach(b =>
            b.classList.toggle('active', b.dataset.audio === String(!!saved.keepAudio)));
        const poseGroup = $('pose-group');
        if (poseGroup) {
            poseGroup.querySelectorAll('.toggle-btn').forEach(b =>
                b.classList.toggle('active', b.dataset.pose === String(!!saved.poseEnabled)));
        }
    }

    // 2) 恢复上次视频（IndexedDB 本地会话）
    const file = await idbGet(IDB_VIDEO_KEY);
    if (file && file.type && file.type.startsWith('video/')) {
        try { loadVideoFile(file, false); } catch (e) { console.warn('恢复视频失败:', e); }
    }

    // 3) 恢复任务（服务端持久化：刷新/重启后任务列表不丢）
    try {
        const resp = await fetch(`${DA3_SERVER}/api/jobs`);
        const data = await resp.json();
        const jobs = data.jobs || [];
        if (jobs.length) {
            renderTaskList(jobs);
        }
        if (jobs.some(j => ['queued', 'running', 'render'].includes(j.status))) {
            log('检测到未完成的任务，正在恢复...', 'info');
            showSection('tasks-section');
            renderTaskList(jobs);
            await pollTasksList();
        } else {
            restoreLastSection();   // 无活跃任务：回到刷新前所在页面
        }
    } catch (e) {
        console.warn('恢复任务失败:', e);
        restoreLastSection();
    }
}

async function stopJob(jobId) {
    if (!confirm('确定停止该任务吗？已处理的帧将保留进度记录，但不会生成最终视频。')) return;
    try {
        const resp = await fetch(`${DA3_SERVER}/api/jobs/${jobId}/stop`, { method: 'POST' });
        const data = await resp.json();
        if (!resp.ok) {
            showToast(data.error || '停止失败', 'error', 5000);
            return;
        }
        showToast('已发送停止指令，任务正在中断...', 'info');
        // 刷新列表
        const jobs = await refreshTaskList();
        if (jobs.some(j => ['queued', 'running', 'render'].includes(j.status))) {
            pollTasksList();
        }
    } catch (e) {
        showToast(`停止失败: ${e.message}`, 'error', 5000);
    }
}

// ============ 视频裁剪 ============
let clipFile = null;

function fmtClipSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
}

function setClipFile(file) {
    clipFile = file;
    const info = $('clip-file-info');
    const btn = $('clip-start-btn');
    if (file) {
        info.style.display = 'block';
        info.textContent = `已选择：${file.name}（${fmtClipSize(file.size)}）`;
        btn.disabled = false;
    } else {
        info.style.display = 'none';
        btn.disabled = true;
    }
}

function resetClipForm() {
    setClipFile(null);
    $('clip-file-input').value = '';
    $('clip-project-id').value = '';
    $('clip-seconds').value = '10';
    $('clip-result').style.display = 'none';
}

async function startClip() {
    if (!clipFile) { showToast('请先选择视频', 'error'); return; }
    const projectId = $('clip-project-id').value.trim();
    const seconds = $('clip-seconds').value;
    const btn = $('clip-start-btn');
    btn.disabled = true;
    btn.textContent = '裁剪中...';

    try {
        const form = new FormData();
        form.append('video', clipFile);
        form.append('project_id', projectId);
        form.append('segment_seconds', seconds);

        const resp = await fetch(`${DA3_SERVER}/api/clip`, { method: 'POST', body: form });
        const data = await resp.json();
        if (!resp.ok) {
            showToast(data.error || '裁剪失败', 'error', 5000);
            return;
        }
        $('clip-result-project').textContent = data.project_id;
        $('clip-result-dir').textContent = data.output_dir;
        $('clip-result-count').textContent = `${data.count} 段（每段约 ${data.segment_seconds} 秒）`;
        const list = $('clip-segment-list');
        list.innerHTML = data.segments.map(s => `
            <div class="clip-segment-item">
                <div>
                    <span class="clip-segment-name">${s.name}</span>
                    <span class="clip-segment-size">${fmtClipSize(s.size)}</span>
                </div>
                <a class="btn btn-primary btn-sm" href="${DA3_SERVER}${s.url}" download="${s.name}">下载</a>
            </div>
        `).join('');
        $('clip-result').style.display = 'block';
        showToast(`裁剪完成，共 ${data.count} 段`, 'success');
    } catch (e) {
        showToast(`裁剪失败: ${e.message}`, 'error', 5000);
    } finally {
        btn.disabled = false;
        btn.textContent = '开始裁剪';
    }
}

init();
