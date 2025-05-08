// 康复指导页面的JavaScript代码

// 全局变量
let sessionActive = false;
let userVideo = null;
let videoStream = null;
let canvasCtx = null;
let poseDetector = null;
let statusCheckInterval = null;
let lastKeypointsData = null; // 存储最近一次的关键点数据
let keypointsPollingInterval = null; // 轮询关键点的间隔
let skipFetching = false; // 控制是否跳过关键点获取
let cameraMonitorInterval = null; // 监控摄像头状态的间隔
let videoFailureCount = 0; // 摄像头失败计数
let pageVisible = true; // 页面可见性状态

// 练习状态跟踪
let currentExerciseStep = 1;
let lastGoodPostureTime = 0;
let holdTimeRequired = 3000; // 保持姿势的时间(毫秒)
let exerciseState = 'waiting'; // waiting, holding, completed

// 姿势评分图表
let scoreChart = null;

// 添加全局变量追踪初始化尝试次数
let cameraInitAttempts = 0;
const MAX_CAMERA_INIT_ATTEMPTS = 3;
let sessionStartTimer = null;

// 新增：后端连接状态跟踪
let backendSessionStarted = false;

// 语音提示
let lastSpokenFeedback = ""; // 上次播放的反馈内容
let lastSpeakTime = 0;       // 上次播放的时间
const MIN_SPEAK_INTERVAL = 3000; // 最小语音间隔（毫秒）
const NO_DETECTION_SPEAK_INTERVAL = 8000; // "未检测到人体"的语音间隔（毫秒）
let speechSynthesisActive = false; // 是否正在进行语音合成

/**
 * 清理语音合成队列并重置语音合成状态
 * 解决语音卡住的问题
 */
function resetSpeechSynthesis() {
    if ('speechSynthesis' in window) {
        // 取消所有排队的语音
        window.speechSynthesis.cancel();
        speechSynthesisActive = false;
        console.log("已重置语音合成系统");
    }
}

/**
 * 语音提示，有防重复机制
 * @param {string} text - 要播放的文本
 * @param {boolean} force - 是否强制播放，忽略重复检查
 */
function speakFeedback(text, force = false) {
    if (!text || text.trim() === "") return;
    
    // 防止语音卡住
    if (speechSynthesisActive) {
        const now = Date.now();
        // 如果上次语音播放时间超过10秒，可能是卡住了，强制重置
        if (now - lastSpeakTime > 10000) {
            resetSpeechSynthesis();
        }
    }
    
    const now = Date.now();
    
    // 如果不是强制播放，检查重复
    if (!force) {
        // 特殊处理"未检测到人体"消息，使用更长的间隔
        if (text.includes("未检测到人体")) {
            if (lastSpokenFeedback.includes("未检测到人体") && 
                now - lastSpeakTime < NO_DETECTION_SPEAK_INTERVAL) {
                console.log("跳过重复的'未检测到人体'提示");
                return;
            }
        } 
        // 一般性的重复语音检查
        else if (text === lastSpokenFeedback && now - lastSpeakTime < MIN_SPEAK_INTERVAL) {
            console.log(`跳过重复语音: ${text}`);
            return;
        }
    }
    
    // 如果语音合成API可用
    if ('speechSynthesis' in window) {
        // 先取消任何当前的语音
        window.speechSynthesis.cancel();
        
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = 'zh-CN';
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        
        // 设置语音结束或错误时的处理
        utterance.onend = function() {
            console.log(`语音播放完成: ${text}`);
            speechSynthesisActive = false;
            lastSpeakTime = Date.now(); // 更新为播放结束的时间
        };
        
        utterance.onerror = function(event) {
            console.error(`语音播放错误: ${event.error}`);
            speechSynthesisActive = false;
            // 如果出错了，重置系统
            resetSpeechSynthesis();
        };
        
        // 更新状态并播放
        lastSpokenFeedback = text;
        lastSpeakTime = now;
        speechSynthesisActive = true;
        
        // 开始播放
        window.speechSynthesis.speak(utterance);
        console.log(`正在播放语音: ${text}`);
    } else {
        console.warn("浏览器不支持语音合成");
    }
}

/**
 * 显示错误消息
 * @param {string} message - 错误消息内容
 * @param {string} title - 标题
 * @param {boolean} showRetry - 是否显示重试按钮
 * @param {Function} retryCallback - 重试回调函数
 */
function showErrorMessage(message, title = "提示", showRetry = true, retryCallback = null) {
    const container = document.getElementById('errorMessageContainer');
    const messageText = document.getElementById('errorMessageText');
    const retryBtn = document.getElementById('errorRetryBtn');
    const closeBtn = document.getElementById('errorCloseBtn');
    const headerDiv = container.querySelector('.error-header');
    
    if (!container || !messageText) {
        console.error('错误提示框元素不存在');
        alert(message);
        return;
    }
    
    // 设置标题和内容
    if (headerDiv) {
        headerDiv.textContent = title;
    }
    messageText.innerText = message;
    
    // 设置按钮显示
    retryBtn.style.display = showRetry ? 'inline-block' : 'none';
    container.style.display = 'block';
    
    // 清除之前的事件监听器
    const newRetryBtn = retryBtn.cloneNode(true);
    const newCloseBtn = closeBtn.cloneNode(true);
    retryBtn.parentNode.replaceChild(newRetryBtn, retryBtn);
    closeBtn.parentNode.replaceChild(newCloseBtn, closeBtn);
    
    // 添加新的事件监听器
    newCloseBtn.addEventListener('click', () => {
        container.style.display = 'none';
    });
    
    if (showRetry && retryCallback) {
        newRetryBtn.addEventListener('click', () => {
            container.style.display = 'none';
            retryCallback();
        });
    }
    
    // 5秒后自动关闭
    setTimeout(() => {
        if (container.style.display === 'block') {
            container.style.display = 'none';
        }
    }, 5000);
}

// 初始化页面
document.addEventListener('DOMContentLoaded', function() {
    // 初始化各种事件监听器
    document.getElementById('startSession').addEventListener('click', toggleSession);
    
    const templateSelector = document.getElementById('templateSelector');
    if (templateSelector) {
        templateSelector.addEventListener('change', changeTemplate);
    }
    
    // 初始化画布
    const canvas = document.getElementById('poseCanvas');
    if (canvas) {
        canvasCtx = canvas.getContext('2d');
    } else {
        console.error("找不到poseCanvas元素");
    }
    
    // 调整画布大小以匹配视频容器
    const resizeCanvas = () => {
        const container = document.querySelector('.camera-container');
        if (container && canvas) {
            canvas.width = container.offsetWidth;
            canvas.height = container.offsetHeight;
        }
    };
    
    // 初始调整和窗口大小变化时调整
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    
    // 初始化调试信息
    initDebugInfo();
    
    // 初始化姿势评分图表
    initScoreChart();
    
    // 初始化指导步骤
    const steps = document.querySelectorAll('.exercise-step');
    steps.forEach(step => {
        step.addEventListener('click', function() {
            const stepNumber = this.getAttribute('data-step');
            updateExerciseStep(stepNumber);
        });
    });
    
    // 添加全局错误监听
    window.addEventListener('error', function(event) {
        console.error('全局错误:', event.message, event.filename, event.lineno);
        updateDebugInfo('lastError', `${event.message} (${new Date().toLocaleTimeString()})`);
    });
    
    // 监听视频元素可能的错误
    const videoElement = document.getElementById('userVideo');
    if (videoElement) {
        videoElement.addEventListener('error', function(e) {
            console.error('视频元素错误:', e);
            updateDebugInfo('videoStatus', '错误');
        });
        
        // 监听视频播放状态
        videoElement.addEventListener('playing', function() {
            updateDebugInfo('videoStatus', '播放中');
            videoFailureCount = 0; // 重置失败计数
        });
        
        videoElement.addEventListener('pause', function() {
            updateDebugInfo('videoStatus', '已暂停');
        });
    }
    
    // 添加页面可见性监听
    document.addEventListener('visibilitychange', handleVisibilityChange);
    
    // 添加页面卸载监听
    window.addEventListener('beforeunload', cleanupResources);
    
    // 调试按钮 - 用于测试骨骼渲染
    const debugBtn = document.createElement('button');
    debugBtn.id = 'debugRenderBtn';
    debugBtn.textContent = '测试骨骼渲染';
    debugBtn.style.cssText = 'position: fixed; bottom: 50px; right: 10px; z-index: 9999; padding: 5px; background: blue; color: white;';
    debugBtn.addEventListener('click', testPoseRendering);
    document.body.appendChild(debugBtn);
    
    console.log('康复指导页面已初始化');
});

// 初始化评分图表
function initScoreChart() {
    try {
        const chartElement = document.getElementById('scoreChart');
        if (!chartElement) {
            console.error('找不到评分图表元素');
            return;
        }
        
        // 创建初始数据
        const initialData = {
            labels: [],
            datasets: [{
                label: '姿势评分',
                data: [],
                fill: false,
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1,
                borderWidth: 2
            }]
        };
        
        // 图表配置
        const config = {
            type: 'line',
            data: initialData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: 0 // 关闭动画以提高性能
                },
                plugins: {
                    legend: {
                        display: false // 隐藏图例
                    },
                    tooltip: {
                        enabled: false // 禁用提示以提高性能
                    }
                },
                scales: {
                    y: {
                        min: 0,
                        max: 100,
                        display: false // 隐藏y轴
                    },
                    x: {
                        display: false // 隐藏x轴
                    }
                }
            }
        };
        
        // 创建图表
        scoreChart = new Chart(chartElement, config);
        console.log('姿势评分图表已初始化');
    } catch (error) {
        console.error('初始化评分图表失败:', error);
    }
}

// 更新评分图表 - 防止未初始化的错误
function updateScoreChart(score, timestamp) {
    // 如果图表未初始化，先初始化
    if (!scoreChart) {
        initScoreChart();
        if (!scoreChart) {
            console.error('无法更新评分图表：图表未初始化');
            return;
        }
    }
    
    // 保持最多30个数据点
    if (scoreChart.data.labels.length > 30) {
        scoreChart.data.labels.shift();
        scoreChart.data.datasets[0].data.shift();
    }
    
    const time = new Date(timestamp).toLocaleTimeString();
    scoreChart.data.labels.push(time);
    scoreChart.data.datasets[0].data.push(score);
    
    try {
        scoreChart.update();
    } catch (e) {
        console.error('更新图表失败:', e);
    }
}

// 更新分数 - 防止未初始化的错误
function updateScore(score) {
    const scoreElement = document.getElementById('currentScore');
    if (scoreElement) {
        scoreElement.textContent = score;
    }
    
    // 使用当前时间戳更新图表
    updateScoreChart(score, Date.now());
}

// 页面可见性变化处理
function handleVisibilityChange() {
    pageVisible = document.visibilityState === 'visible';
    console.log(`页面可见性变化: ${pageVisible ? '可见' : '不可见'}`);
    
    if (!pageVisible) {
        // 页面不可见时，暂停非必要的处理以节省资源
        if (keypointsPollingInterval) {
            console.log('页面不可见，暂停关键点轮询');
            skipFetching = true;
        }
    } else {
        // 页面重新可见时，恢复处理
        if (sessionActive && keypointsPollingInterval) {
            console.log('页面重新可见，恢复关键点轮询');
            skipFetching = false;
        }
    }
    
    // 更新调试信息
    updateDebugInfo('pageVisibility', pageVisible ? '可见' : '不可见');
}

// 资源清理
function cleanupResources() {
    console.log('页面卸载，清理资源...');
    
    // 停止所有轮询和监控
    stopKeyPointsPolling();
    stopCameraMonitoring();
    
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
        statusCheckInterval = null;
    }
    
    // 停止摄像头
    stopCamera();
    
    // 如果会话活跃，通知服务器停止
    if (sessionActive) {
        const stopData = new FormData();
        stopData.append('action', 'stop_session');
        
        // 使用同步请求确保在页面卸载前发送
        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/api/rehab/stop', false); // 同步请求
        xhr.send(stopData);
    }
}

// 测试骨骼渲染
function testPoseRendering() {
    // 创建测试关键点数据
    const testWidth = userVideo ? userVideo.clientWidth : 640;
    const testHeight = userVideo ? userVideo.clientHeight : 480;
    const centerX = testWidth / 2;
    const centerY = testHeight / 2;
    
    const testKeypoints = {
        // 头部
        'nose': { x: centerX, y: centerY - 130, visibility: 0.9 },
        'left_eye': { x: centerX - 20, y: centerY - 140, visibility: 0.9 },
        'right_eye': { x: centerX + 20, y: centerY - 140, visibility: 0.9 },
        'left_ear': { x: centerX - 40, y: centerY - 130, visibility: 0.8 },
        'right_ear': { x: centerX + 40, y: centerY - 130, visibility: 0.8 },
        
        // 肩膀
        'left_shoulder': { x: centerX - 70, y: centerY - 70, visibility: 0.95 },
        'right_shoulder': { x: centerX + 70, y: centerY - 70, visibility: 0.95 },
        
        // 手肘
        'left_elbow': { x: centerX - 100, y: centerY, visibility: 0.9 },
        'right_elbow': { x: centerX + 100, y: centerY, visibility: 0.9 },
        
        // 手腕
        'left_wrist': { x: centerX - 130, y: centerY + 30, visibility: 0.85 },
        'right_wrist': { x: centerX + 130, y: centerY + 30, visibility: 0.85 },
        
        // 臀部
        'left_hip': { x: centerX - 50, y: centerY + 50, visibility: 0.9 },
        'right_hip': { x: centerX + 50, y: centerY + 50, visibility: 0.9 },
        
        // 膝盖
        'left_knee': { x: centerX - 55, y: centerY + 120, visibility: 0.85 },
        'right_knee': { x: centerX + 55, y: centerY + 120, visibility: 0.85 },
        
        // 脚踝
        'left_ankle': { x: centerX - 60, y: centerY + 190, visibility: 0.8 },
        'right_ankle': { x: centerX + 60, y: centerY + 190, visibility: 0.8 }
    };
    
    // 显示测试通知
    showNotification('测试骨骼渲染中...', 'info');
    
    // 绘制测试骨骼
    console.log('绘制测试骨骼:', testKeypoints);
    drawPose(testKeypoints);
    
    // 分析姿势
    analyzePose(testKeypoints);
}

// 使用后端骨骼检测功能，不再需要加载本地模型
async function loadPoseDetectionModel() {
    try {
        // 显示加载进度提示
        showNotification('正在初始化骨骼检测，请稍候...');
        document.getElementById('statusMessage').textContent = '正在初始化骨骼检测...';
        
        // 更新调试面板
        updateDebugInfo('modelStatus', '使用后端检测');
        
        if (window.updateCameraHint) {
            window.updateCameraHint('正在初始化骨骼检测，请稍候...');
        }
        
        // 检查后端服务是否可用
        console.log('检查后端服务状态...');
        const response = await fetch('/video/status');
        
        if (!response.ok) {
            throw new Error(`HTTP错误! 状态: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('后端服务状态:', data);
        
        if (data.status !== 'ok') {
            // 显示更详细的错误信息
            let errorMsg = data.message || '后端服务不可用';
            if (!data.initialized) {
                errorMsg = '视频处理服务未初始化';
            } else if (!data.camera_ready) {
                errorMsg = '摄像头未准备就绪或未授权';
            }
            
            throw new Error(errorMsg);
        }
        
        console.log('骨骼检测初始化成功');
        showNotification('骨骼检测已准备就绪');
        
        // 更新调试面板
        updateDebugInfo('modelStatus', '已连接到后端');
        
        if (window.updateCameraHint) {
            window.updateCameraHint('检测已准备就绪，请站在摄像头前方，确保全身可见');
        }
        return true;
    } catch (error) {
        console.error('初始化骨骼检测失败:', error);
        showNotification('骨骼检测初始化失败: ' + error.message);
        document.getElementById('statusMessage').textContent = '检测初始化失败: ' + error.message;
        
        // 更新调试面板显示错误
        updateDebugInfo('modelStatus', `初始化失败: ${error.message || '未知错误'}`);
        
        if (window.updateCameraHint) {
            window.updateCameraHint('检测初始化失败: ' + error.message);
        }
        return false;
    }
}

// 启动姿势检测
async function startPoseDetection() {
    try {
        // 开始轮询后端获取关键点数据
        startKeyPointsPolling();
        return true;
    } catch (error) {
        console.error('启动姿势检测失败:', error);
        return false;
    }
}

// 停止姿势检测
function stopPoseDetection() {
    // 停止轮询
    stopKeyPointsPolling();
    
    // 清空最后的关键点数据
    lastKeypointsData = null;
    
    // 清空画布
    clearCanvas();
}

/**
 * 开始轮询关键点数据
 */
function startKeyPointsPolling() {
    if (keypointsPollingInterval) {
        console.log('关键点轮询已经在运行中');
        return;
    }
    
    console.log('开始关键点轮询');
    // 每隔200ms查询一次关键点数据
    keypointsPollingInterval = setInterval(async () => {
        if (skipFetching || !sessionActive) return;
        
        // 获取关键点
        try {
            await fetchKeypoints();
        } catch (error) {
            console.error('获取关键点出错:', error);
            updateDebugInfo('lastError', `获取关键点出错: ${error.message}`);
        }
    }, 200);
}

/**
 * 停止轮询关键点数据
 */
function stopKeyPointsPolling() {
    if (keypointsPollingInterval) {
        console.log('停止关键点轮询');
        clearInterval(keypointsPollingInterval);
        keypointsPollingInterval = null;
    }
}

/**
 * 从后端获取关键点数据
 */
async function fetchKeypoints() {
    // 检查条件
    if (!sessionActive || !backendSessionStarted) return;
    
    try {
        // 调用后端API获取最新的康复会话状态
        const response = await fetch('/api/rehab/status');
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            console.error('获取康复状态失败:', response.status, errorData);
            return;
        }
        
        const data = await response.json();
        
        // 检查数据有效性
        if (data.status !== 'success') {
            console.log('获取康复状态返回非成功状态:', data.status);
            return;
        }
        
        // 检查是否有有效数据
        if (!data.data) {
            console.log('康复状态数据为空');
            return;
        }
        
        const result = data.data;
        
        // 更新摄像头提示
        const cameraHint = document.getElementById('cameraHint');
        if (cameraHint) {
            // 如果检测到人体，降低提示的不透明度并更新文字
            if (result.landmarks && Object.keys(result.landmarks).length > 0) {
                // 如果之前提示是未检测到人体，现在检测到了，更新提示
                if (cameraHint.textContent.includes('未检测到人体')) {
                    cameraHint.textContent = '已检测到人体，正在分析姿势...';
                    cameraHint.style.backgroundColor = 'rgba(40, 167, 69, 0.2)'; // 绿色背景
                    // 检测到人体后隐藏提示（逐渐变透明）
                    setTimeout(() => {
                        cameraHint.style.opacity = '0.3';
                    }, 2000);
                }
            } else {
                // 未检测到人体时提高提示的不透明度
                cameraHint.textContent = '未检测到人体姿势，请确保您完整地出现在画面中';
                cameraHint.style.backgroundColor = 'rgba(220, 53, 69, 0.2)'; // 红色背景
                cameraHint.style.opacity = '0.9';
                
                // 未检测到人体时进行语音提示（使用低频率，避免过于频繁）
                speakFeedback('未检测到人体姿势，请确保您完整地出现在画面中');
            }
        }
        
        // 更新评分
        if (result.score !== undefined) {
            updateScore(result.score);
            
            // 根据分数提供不同的视觉反馈
            const scoreElement = document.getElementById('currentScore');
            if (scoreElement) {
                // 根据分数设置不同颜色
                if (result.score >= 90) {
                    scoreElement.className = 'score-value excellent';
                } else if (result.score >= 75) {
                    scoreElement.className = 'score-value good';
                } else if (result.score >= 60) {
                    scoreElement.className = 'score-value average';
                } else {
                    scoreElement.className = 'score-value poor';
                }
            }
        }
        
        // 检查是否有反馈信息
        if (result.feedback && result.feedback.length > 0) {
            // 更新前端反馈显示
            updateFeedback(result.feedback);
            
            // 根据评分决定是否需要语音反馈
            if (result.score < 75) { // 只在评分较低时提供修正指导
                // 如果是多条反馈，只播放第一条最重要的
                if (result.feedback.length > 0) {
                    // 使用强调语气，帮助用户理解要点
                    const feedback = result.feedback[0];
                    
                    // 检查是否是重要的姿势校正提示
                    const isImportantCorrection = 
                        feedback.includes('脊柱') || 
                        feedback.includes('头部') || 
                        feedback.includes('肩膀') ||
                        feedback.includes('姿势');
                    
                    // 重要的校正提示更频繁播放
                    speakFeedback(feedback, isImportantCorrection);
                }
            } else if (result.score >= 90 && !lastSpokenFeedback.includes('姿势很好')) {
                // 高分时给予鼓励（但不要太频繁）
                speakFeedback('姿势很好，请保持');
            }
        }
        
        // 如果有偏差数据，更新偏差指示器
        if (result.deviations) {
            updateDeviationIndicators(result.deviations);
        }
        
        // 如果有关键点数据，更新康复指导界面
        if (result.landmarks) {
            // 保存关键点数据以便后续使用
            lastKeypointsData = result.landmarks;
            
            // 分析和绘制姿势
            analyzePose(result.landmarks);
            drawPose(result.landmarks);
            
            // 启用"下一步"按钮，如果完成了当前练习
            if (exerciseState === 'completed') {
                const nextButton = document.getElementById('nextExerciseButton');
                if (nextButton) {
                    nextButton.disabled = false;
                }
            }
        } else if (result.annotated_image) {
            // 如果后端直接返回了标注图像，直接显示
            drawPoseResult(result.annotated_image);
        } else {
            // 没有检测到关键点，清除画布
            clearCanvas();
        }
        
        // 更新模板信息
        if (data.template) {
            const templateSelector = document.getElementById('templateSelector');
            if (templateSelector && templateSelector.value !== data.template) {
                templateSelector.value = data.template;
                
                // 更新模板说明
                const templateDescription = document.getElementById('templateDescription');
                if (templateDescription) {
                    const selectedOption = templateSelector.options[templateSelector.selectedIndex];
                    templateDescription.textContent = selectedOption.getAttribute('data-description') || '请保持正确姿势';
                }
            }
        }
        
        // 更新FPS等调试信息
        if (data.fps) {
            updateDebugInfo('fpsMeter', `${data.fps.toFixed(1)} FPS`);
        }
        
    } catch (error) {
        console.error('获取关键点请求失败:', error);
        updateDebugInfo('lastError', `获取关键点失败: ${error.message}`);
        
        // 连续失败超过阈值时重新连接后端
        errorRecoveryCount = (errorRecoveryCount || 0) + 1;
        if (errorRecoveryCount > 5 && backendSessionStarted) {
            console.warn('多次获取关键点失败，尝试重新连接后端...');
            errorRecoveryCount = 0;
            
            // 重新初始化后端会话
            fetch('/api/rehab/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            }).then(response => {
                if (response.ok) {
                    console.log('已重新连接到后端');
                }
            }).catch(err => {
                console.error('重新连接后端失败:', err);
            });
        }
    }
}

// 清除画布
function clearCanvas() {
    if (canvasCtx) {
        canvasCtx.clearRect(0, 0, canvasCtx.canvas.width, canvasCtx.canvas.height);
    }
}

// 分析姿势并更新练习状态
function analyzePose(keypoints) {
    // 如果没有关键点数据，无法分析
    if (!keypoints || Object.keys(keypoints).length === 0) {
        return;
    }
    
    // 根据不同的练习步骤分析姿势
    let isCorrectPosture = false;
    
    // 获取必要的关键点
    const leftShoulder = keypoints.left_shoulder;
    const rightShoulder = keypoints.right_shoulder;
    const leftHip = keypoints.left_hip;
    const rightHip = keypoints.right_hip;
    
    // 确保我们有所有需要的关键点
    if (!leftShoulder || !rightShoulder || !leftHip || !rightHip) {
        exerciseState = 'waiting';
        lastGoodPostureTime = 0;
        updateExerciseProgress();
        return;
    }
    
    // 计算脊柱偏差
    let spineDeviation = 0;
    try {
        // 计算肩部中点
        const midShoulderX = (leftShoulder.x + rightShoulder.x) / 2;
        const midShoulderY = (leftShoulder.y + rightShoulder.y) / 2;
        
        // 计算臀部中点
        const midHipX = (leftHip.x + rightHip.x) / 2;
        const midHipY = (leftHip.y + rightHip.y) / 2;
        
        // 计算脊柱垂直偏差
        const dx = Math.abs(midShoulderX - midHipX);
        const dy = Math.abs(midShoulderY - midHipY) || 1;  // 避免除零错误
        spineDeviation = dx / dy;
    } catch (e) {
        console.warn('计算脊柱偏差失败:', e);
    }
    
    // 更新偏差指示器
    updateDeviationIndicatorsFromPose(keypoints);
    
    // 根据当前步骤判断是否是正确姿势
    switch (currentExerciseStep) {
        case 1:
            // 第一步：保持脊柱直立
            isCorrectPosture = spineDeviation < 0.1;
            break;
        case 2:
            // 第二步：左侧弯
            const leftLean = (rightShoulder.y - leftShoulder.y) / (rightShoulder.x - leftShoulder.x + 0.001);
            isCorrectPosture = leftLean > 0.2;
            break;
        case 3:
            // 第三步：右侧弯
            const rightLean = (leftShoulder.y - rightShoulder.y) / (leftShoulder.x - rightShoulder.x + 0.001);
            isCorrectPosture = rightLean > 0.2;
            break;
        default:
            isCorrectPosture = spineDeviation < 0.1;
    }
    
    // 根据姿势更新练习状态
    const currentTime = Date.now();
    
    if (isCorrectPosture) {
        if (exerciseState === 'waiting') {
            // 开始记录保持时间
            lastGoodPostureTime = currentTime;
            exerciseState = 'holding';
        } else if (exerciseState === 'holding') {
            // 检查是否已保持足够时间
            const holdTime = currentTime - lastGoodPostureTime;
            
            if (holdTime >= holdTimeRequired) {
                exerciseState = 'completed';
                speakFeedback(`步骤${currentExerciseStep}完成！`);
                
                // 自动进入下一步
                setTimeout(() => {
                    const nextStep = currentExerciseStep < 3 ? currentExerciseStep + 1 : 1;
                    updateExerciseStep(nextStep);
                }, 1000);
            }
        }
    } else {
        // 不是正确姿势，重置状态
        if (exerciseState !== 'waiting') {
            exerciseState = 'waiting';
            lastGoodPostureTime = 0;
        }
    }
    
    // 更新UI显示
    updateExerciseProgress();
}

// 更新练习进度显示
function updateExerciseProgress() {
    // 更新所有步骤的状态
    document.querySelectorAll('.exercise-step').forEach(step => {
        const stepNumber = parseInt(step.getAttribute('data-step'));
        
        if (stepNumber === currentExerciseStep) {
            step.classList.add('current');
            
            // 根据当前保持姿势的状态更新样式
            step.classList.remove('waiting', 'holding', 'completed');
            step.classList.add(exerciseState);
            
            // 更新保持时间显示
            if (exerciseState === 'holding') {
                const currentTime = Date.now();
                const holdTime = currentTime - lastGoodPostureTime;
                updateHoldTimeDisplay(holdTime, holdTimeRequired);
            } else if (exerciseState === 'waiting') {
                updateHoldTimeDisplay(0, holdTimeRequired);
            } else if (exerciseState === 'completed') {
                updateHoldTimeDisplay(holdTimeRequired, holdTimeRequired);
            }
        } else {
            step.classList.remove('current', 'waiting', 'holding', 'completed');
        }
    });
}

// 更新保持时间显示
function updateHoldTimeDisplay(currentTime, requiredTime) {
    const progressElement = document.querySelector('.hold-progress');
    if (progressElement) {
        const percent = Math.min(100, (currentTime / requiredTime) * 100);
        progressElement.style.width = `${percent}%`;
        
        if (percent > 66) progressElement.classList.add('bg-success');
        else progressElement.classList.remove('bg-success');
    }
}

// 绘制骨骼姿势
function drawPose(keypoints) {
    if (!canvasCtx) {
        console.error("Canvas 上下文未初始化");
        return;
    }
    
    const canvas = canvasCtx.canvas;
    if (!canvas) {
        console.error("找不到Canvas元素");
        return;
    }
    
    if (!keypoints || Object.keys(keypoints).length === 0) {
        console.warn("没有关键点数据可用于绘制");
        clearCanvas();
        return;
    }
    
    console.log(`开始绘制骨骼，Canvas尺寸: ${canvas.width}x${canvas.height}`);
    
    // 确保canvas尺寸与视频尺寸匹配
    const videoElement = document.getElementById('userVideo');
    if (videoElement) {
        // 强制设置Canvas尺寸与视频实际尺寸相同
        const rect = videoElement.getBoundingClientRect();
        if (canvas.width !== rect.width || canvas.height !== rect.height) {
            canvas.width = rect.width;
            canvas.height = rect.height;
            console.log(`调整canvas尺寸为 ${canvas.width}x${canvas.height}`);
        }
    }
    
    // 清除之前的绘制
    canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
    
    // 定义骨骼连接
    const connections = [
        ['nose', 'left_eye'], ['nose', 'right_eye'],
        ['left_eye', 'left_ear'], ['right_eye', 'right_ear'],
        ['left_shoulder', 'right_shoulder'],
        ['left_shoulder', 'left_elbow'], ['right_shoulder', 'right_elbow'],
        ['left_elbow', 'left_wrist'], ['right_elbow', 'right_wrist'],
        ['left_shoulder', 'left_hip'], ['right_shoulder', 'right_hip'],
        ['left_hip', 'right_hip'],
        ['left_hip', 'left_knee'], ['right_hip', 'right_knee'],
        ['left_knee', 'left_ankle'], ['right_knee', 'right_ankle']
    ];
    
    // 特殊分组 - 用于不同颜色绘制
    const spineConnections = [
        ['left_shoulder', 'left_hip'], ['right_shoulder', 'right_hip'],
        ['left_shoulder', 'right_shoulder'], ['left_hip', 'right_hip']
    ];
    
    // 获取关键点中心
    const centerPoint = getCenterPoint(keypoints);
    
    // 绘制半透明人体轮廓背景
    drawBodyOutline(keypoints, canvas.width, canvas.height);
    
    // 坐标映射函数 - 确保坐标正确映射到Canvas上
    const mapToCanvas = (point) => {
        // 检查坐标格式 - 处理不同格式的关键点数据
        if (typeof point.x === 'number' && point.x <= 1 && point.x >= 0) {
            // 如果是0-1之间的比例坐标，则直接乘以Canvas尺寸
            return {
                x: point.x * canvas.width,
                y: point.y * canvas.height,
                visibility: point.visibility || point.score || 1.0
            };
        } else {
            // 如果是绝对像素坐标，确保在Canvas范围内
            return {
                x: Math.min(canvas.width, Math.max(0, point.x)),
                y: Math.min(canvas.height, Math.max(0, point.y)),
                visibility: point.visibility || point.score || 1.0
            };
        }
    };
    
    // 绘制连接线 - 先绘制普通连接
    for (const [start, end] of connections) {
        // 跳过脊柱连接，稍后单独绘制
        if (spineConnections.some(conn => 
            (conn[0] === start && conn[1] === end) || 
            (conn[0] === end && conn[1] === start))) {
            continue;
        }
        
        if (keypoints[start] && keypoints[end]) {
            const startPoint = mapToCanvas(keypoints[start]);
            const endPoint = mapToCanvas(keypoints[end]);
            
            if (startPoint.visibility > 0.5 && endPoint.visibility > 0.5) {
                // 普通连接线 - 蓝色
                canvasCtx.beginPath();
                canvasCtx.moveTo(startPoint.x, startPoint.y);
                canvasCtx.lineTo(endPoint.x, endPoint.y);
                canvasCtx.lineWidth = 3;
                canvasCtx.strokeStyle = 'rgba(0, 119, 255, 0.8)';
                canvasCtx.stroke();
            }
        }
    }
    
    // 特别绘制脊柱连接 - 更粗更明显
    for (const [start, end] of spineConnections) {
        if (keypoints[start] && keypoints[end]) {
            const startPoint = mapToCanvas(keypoints[start]);
            const endPoint = mapToCanvas(keypoints[end]);
            
            if (startPoint.visibility > 0.5 && endPoint.visibility > 0.5) {
                // 脊柱连接线 - 绿色且更粗
                canvasCtx.beginPath();
                canvasCtx.moveTo(startPoint.x, startPoint.y);
                canvasCtx.lineTo(endPoint.x, endPoint.y);
                canvasCtx.lineWidth = 6;
                canvasCtx.strokeStyle = 'rgba(0, 255, 0, 0.9)';
                canvasCtx.stroke();
                
                // 如果是肩部与髋部之间的连接，添加辅助垂直参考线
                if ((start === 'left_shoulder' && end === 'left_hip') || 
                    (start === 'right_shoulder' && end === 'right_hip')) {
                    
                    // 绘制垂直参考线 - 淡蓝色虚线
                    canvasCtx.beginPath();
                    canvasCtx.moveTo(startPoint.x, startPoint.y);
                    canvasCtx.lineTo(startPoint.x, endPoint.y); // 保持X不变
                    canvasCtx.lineWidth = 2;
                    canvasCtx.setLineDash([5, 5]);
                    canvasCtx.strokeStyle = 'rgba(0, 200, 255, 0.5)';
                    canvasCtx.stroke();
                    canvasCtx.setLineDash([]); // 恢复实线
                    
                    // 计算与垂直线的偏差
                    const deviationX = Math.abs(startPoint.x - endPoint.x);
                    const deviationPercent = deviationX / canvas.width * 100;
                    
                    // 如果偏差较大，标记出来
                    if (deviationPercent > 2) {
                        const midX = (startPoint.x + endPoint.x) / 2;
                        const midY = (startPoint.y + endPoint.y) / 2;
                        
                        // 绘制偏差指示器
                        canvasCtx.beginPath();
                        canvasCtx.arc(midX, midY, 8, 0, 2 * Math.PI);
                        canvasCtx.fillStyle = deviationPercent > 5 ? 'rgba(255, 0, 0, 0.7)' : 'rgba(255, 165, 0, 0.7)';
                        canvasCtx.fill();
                    }
                }
            }
        }
    }
    
    // 绘制躯干中心线 - 从鼻子到髋部中心
    if (keypoints['nose'] && keypoints['left_hip'] && keypoints['right_hip'] &&
        keypoints['nose'].visibility > 0.5 && 
        keypoints['left_hip'].visibility > 0.5 && 
        keypoints['right_hip'].visibility > 0.5) {
        
        const noseX = keypoints['nose'].x * canvas.width;
        const noseY = keypoints['nose'].y * canvas.height;
        
        const hipCenterX = ((keypoints['left_hip'].x + keypoints['right_hip'].x) / 2) * canvas.width;
        const hipCenterY = ((keypoints['left_hip'].y + keypoints['right_hip'].y) / 2) * canvas.height;
        
        // 绘制躯干中心线 - 粗黄色
        canvasCtx.beginPath();
        canvasCtx.moveTo(noseX, noseY);
        canvasCtx.lineTo(hipCenterX, hipCenterY);
        canvasCtx.lineWidth = 4;
        canvasCtx.strokeStyle = 'rgba(255, 215, 0, 0.8)';
        canvasCtx.stroke();
    }
    
    // 绘制关键点
    for (const [part, point] of Object.entries(keypoints)) {
        if (point.visibility <= 0.5) continue;
        
        const x = point.x * canvas.width;
        const y = point.y * canvas.height;
        
        // 关键点直径根据重要性调整
        let pointSize = 4;
        let color = 'rgba(255, 255, 255, 0.8)';
        
        // 给不同部位设置不同颜色和大小
        if (['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'].includes(part)) {
            // 关键躯干点 - 绿色且较大
            pointSize = 8;
            color = 'rgba(0, 255, 0, 0.9)';
        } else if (part === 'nose') {
            // 鼻子 - 黄色
            pointSize = 6;
            color = 'rgba(255, 215, 0, 0.9)';
        } else if (['left_eye', 'right_eye', 'left_ear', 'right_ear'].includes(part)) {
            // 面部点 - 白色且更小
            pointSize = 3;
            color = 'rgba(255, 255, 255, 0.7)';
        } else if (['left_knee', 'right_knee', 'left_ankle', 'right_ankle'].includes(part)) {
            // 下肢点 - 蓝色
            pointSize = 6;
            color = 'rgba(0, 119, 255, 0.8)';
        } else if (['left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'].includes(part)) {
            // 上肢点 - 淡蓝色
            pointSize = 6;
            color = 'rgba(0, 200, 255, 0.8)';
        }
        
        // 绘制圆点
        canvasCtx.beginPath();
        canvasCtx.arc(x, y, pointSize, 0, 2 * Math.PI);
        canvasCtx.fillStyle = color;
        canvasCtx.fill();
        
        // 给重要关键点添加标签
        if (['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', 'nose'].includes(part)) {
            const labelMap = {
                'left_shoulder': '左肩', 
                'right_shoulder': '右肩',
                'left_hip': '左髋', 
                'right_hip': '右髋',
                'nose': '头部'
            };
            
            canvasCtx.font = '12px Arial';
            canvasCtx.fillStyle = 'white';
            canvasCtx.textAlign = 'center';
            canvasCtx.fillText(labelMap[part], x, y - 10);
        }
    }
    
    // 计算并显示关键身体角度
    displayBodyAngles(keypoints, canvas.width, canvas.height);
    
    // 添加可视化辅助信息
    addVisualGuides(keypoints, canvas.width, canvas.height);
}

// 绘制半透明人体轮廓背景
function drawBodyOutline(keypoints, canvasWidth, canvasHeight) {
    if (!canvasCtx) return;
    
    // 检查是否有足够的关键点来绘制轮廓
    const requiredPoints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', 
                           'left_knee', 'right_knee', 'left_ankle', 'right_ankle'];
    
    for (const point of requiredPoints) {
        if (!keypoints[point] || keypoints[point].visibility <= 0.5) {
            return; // 缺少关键点，不绘制轮廓
        }
    }
    
    // 坐标映射函数 - 与drawPose中相同的函数
    const mapToCanvas = (point) => {
        // 检查坐标格式 - 处理不同格式的关键点数据
        if (typeof point.x === 'number' && point.x <= 1 && point.x >= 0) {
            // 如果是0-1之间的比例坐标，则直接乘以Canvas尺寸
            return {
                x: point.x * canvasWidth,
                y: point.y * canvasHeight,
                visibility: point.visibility || point.score || 1.0
            };
        } else {
            // 如果是绝对像素坐标，确保在Canvas范围内
            return {
                x: Math.min(canvasWidth, Math.max(0, point.x)),
                y: Math.min(canvasHeight, Math.max(0, point.y)),
                visibility: point.visibility || point.score || 1.0
            };
        }
    };
    
    // 获取映射后的坐标点
    const rightShoulder = mapToCanvas(keypoints['right_shoulder']);
    const leftShoulder = mapToCanvas(keypoints['left_shoulder']);
    const leftHip = mapToCanvas(keypoints['left_hip']);
    const rightHip = mapToCanvas(keypoints['right_hip']);
    const leftKnee = mapToCanvas(keypoints['left_knee']);
    const rightKnee = mapToCanvas(keypoints['right_knee']);
    const leftAnkle = mapToCanvas(keypoints['left_ankle']);
    const rightAnkle = mapToCanvas(keypoints['right_ankle']);
    
    // 开始绘制轮廓
    canvasCtx.beginPath();
    
    // 移动到右肩
    canvasCtx.moveTo(rightShoulder.x, rightShoulder.y);
    
    // 绘制躯干轮廓 - 连接肩部和髋部
    canvasCtx.lineTo(leftShoulder.x, leftShoulder.y);
    canvasCtx.lineTo(leftHip.x, leftHip.y);
    canvasCtx.lineTo(rightHip.x, rightHip.y);
    canvasCtx.closePath();
    
    // 填充躯干区域
    canvasCtx.fillStyle = 'rgba(40, 120, 180, 0.2)';
    canvasCtx.fill();
    
    // 绘制腿部轮廓 - 左腿
    canvasCtx.beginPath();
    canvasCtx.moveTo(leftHip.x, leftHip.y);
    
    canvasCtx.lineTo(leftKnee.x, leftKnee.y);
    canvasCtx.lineTo(leftAnkle.x, leftAnkle.y);
    
    // 设置线条样式
    canvasCtx.lineWidth = 1;
    canvasCtx.strokeStyle = 'rgba(100, 160, 220, 0.4)';
    canvasCtx.stroke();
    
    // 绘制腿部轮廓 - 右腿
    canvasCtx.beginPath();
    canvasCtx.moveTo(rightHip.x, rightHip.y);
    
    canvasCtx.lineTo(rightKnee.x, rightKnee.y);
    canvasCtx.lineTo(rightAnkle.x, rightAnkle.y);
    canvasCtx.stroke();
}

// 计算关键点中心
function getCenterPoint(keypoints) {
    const validPoints = Object.values(keypoints).filter(point => point.visibility > 0.5);
    if (validPoints.length === 0) return { x: 0.5, y: 0.5 };
    
    const sumX = validPoints.reduce((sum, point) => sum + point.x, 0);
    const sumY = validPoints.reduce((sum, point) => sum + point.y, 0);
    
    return {
        x: sumX / validPoints.length,
        y: sumY / validPoints.length
    };
}

// 计算并显示关键身体角度
function displayBodyAngles(keypoints, canvasWidth, canvasHeight) {
    if (!canvasCtx) return;
    
    // 计算脊柱垂直度
    if (keypoints['nose'] && keypoints['left_hip'] && keypoints['right_hip'] &&
        keypoints['nose'].visibility > 0.5 && 
        keypoints['left_hip'].visibility > 0.5 && 
        keypoints['right_hip'].visibility > 0.5) {
        
        const noseX = keypoints['nose'].x * canvasWidth;
        const noseY = keypoints['nose'].y * canvasHeight;
        
        const hipCenterX = ((keypoints['left_hip'].x + keypoints['right_hip'].x) / 2) * canvasWidth;
        const hipCenterY = ((keypoints['left_hip'].y + keypoints['right_hip'].y) / 2) * canvasHeight;
        
        // 计算与垂直线的角度
        const dx = noseX - hipCenterX;
        const dy = noseY - hipCenterY;
        const angleRadians = Math.atan2(dx, -dy); // -dy 使0度表示垂直向上
        let angleDegrees = angleRadians * (180 / Math.PI);
        if (angleDegrees < 0) angleDegrees += 180;
        
        // 决定角度颜色 - 红色(不好)到绿色(好)
        let angleColor = 'green';
        if (Math.abs(angleDegrees) > 10) {
            angleColor = 'yellow';
        }
        if (Math.abs(angleDegrees) > 20) {
            angleColor = 'orange';
        }
        if (Math.abs(angleDegrees) > 30) {
            angleColor = 'red';
        }
        
        // 在画布顶部显示角度信息
        canvasCtx.font = '14px Arial';
        canvasCtx.fillStyle = angleColor;
        canvasCtx.textAlign = 'left';
        canvasCtx.fillText(`脊柱垂直度: ${Math.abs(angleDegrees).toFixed(1)}°`, 10, 20);
    }
}

// 添加可视化辅助信息
function addVisualGuides(keypoints, canvasWidth, canvasHeight) {
    if (!canvasCtx) return;
    
    // 检查肩膀水平度
    if (keypoints['left_shoulder'] && keypoints['right_shoulder'] &&
        keypoints['left_shoulder'].visibility > 0.5 && 
        keypoints['right_shoulder'].visibility > 0.5) {
        
        const leftShoulderX = keypoints['left_shoulder'].x * canvasWidth;
        const leftShoulderY = keypoints['left_shoulder'].y * canvasHeight;
        const rightShoulderX = keypoints['right_shoulder'].x * canvasWidth;
        const rightShoulderY = keypoints['right_shoulder'].y * canvasHeight;
        
        // 计算肩膀倾斜度
        const shoulderSlope = Math.abs(leftShoulderY - rightShoulderY);
        const shoulderSlopePercent = shoulderSlope / canvasHeight * 100;
        
        // 在肩膀位置绘制水平参考线
        canvasCtx.beginPath();
        canvasCtx.moveTo(leftShoulderX - 20, leftShoulderY);
        canvasCtx.lineTo(rightShoulderX + 20, leftShoulderY); // 使用左肩的Y值作为水平线
        canvasCtx.lineWidth = 1;
        canvasCtx.setLineDash([5, 5]);
        canvasCtx.strokeStyle = shoulderSlopePercent > 5 ? 'rgba(255,0,0,0.5)' : 'rgba(0,255,0,0.5)';
        canvasCtx.stroke();
        canvasCtx.setLineDash([]);
    }
    
    // 检查髋部水平度
    if (keypoints['left_hip'] && keypoints['right_hip'] &&
        keypoints['left_hip'].visibility > 0.5 && 
        keypoints['right_hip'].visibility > 0.5) {
        
        const leftHipX = keypoints['left_hip'].x * canvasWidth;
        const leftHipY = keypoints['left_hip'].y * canvasHeight;
        const rightHipX = keypoints['right_hip'].x * canvasWidth;
        const rightHipY = keypoints['right_hip'].y * canvasHeight;
        
        // 计算髋部倾斜度
        const hipSlope = Math.abs(leftHipY - rightHipY);
        const hipSlopePercent = hipSlope / canvasHeight * 100;
        
        // 在髋部位置绘制水平参考线
        canvasCtx.beginPath();
        canvasCtx.moveTo(leftHipX - 20, leftHipY);
        canvasCtx.lineTo(rightHipX + 20, leftHipY); // 使用左髋的Y值作为水平线
        canvasCtx.lineWidth = 1;
        canvasCtx.setLineDash([5, 5]);
        canvasCtx.strokeStyle = hipSlopePercent > 5 ? 'rgba(255,0,0,0.5)' : 'rgba(0,255,0,0.5)';
        canvasCtx.stroke();
        canvasCtx.setLineDash([]);
    }
}

// 初始化摄像头
async function initializeCamera() {
    try {
        console.log("初始化摄像头...");
        const userVideo = document.getElementById('userVideo'); 
        const cameraHint = document.getElementById('cameraHint');
        
        if (!userVideo) {
            console.error("找不到视频元素 'userVideo'");
            return false;
        }
        
        // 确保视频元素可见
        userVideo.style.display = 'block';
        
        // 停止任何已存在的流
        if (videoStream) {
            console.log("停止现有视频流");
            videoStream.getTracks().forEach(track => track.stop());
            videoStream = null;
        }
        
        // 清理视频元素
        userVideo.srcObject = null;
        userVideo.load();
        
        // 更新UI状态
        updateDebugInfo('cameraStatus', '正在初始化');
        
        if (cameraHint) {
            cameraHint.textContent = '正在启动摄像头...';
            cameraHint.style.opacity = '1';
        }
        
        // 检查浏览器支持
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('浏览器不支持摄像头访问，请使用Chrome、Firefox或Edge浏览器');
        }
        
        // 使用简单的摄像头约束
        const constraints = { 
            video: { 
                width: { ideal: 640 },
                height: { ideal: 480 }
            }
        };
        
        console.log("请求摄像头访问权限，约束:", constraints);
        
        // 直接尝试获取媒体流，使用Promise
        try {
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            
            // 检查流是否有视频轨道
            if (!stream || stream.getVideoTracks().length === 0) {
                throw new Error('获取的媒体流没有视频轨道');
            }
            
            // 成功获取流
            videoStream = stream;
            userVideo.srcObject = stream;
            
            // 设置视频元素属性
            userVideo.muted = true;
            userVideo.playsInline = true;
            
            // 记录摄像头信息
            const videoTracks = stream.getVideoTracks();
            if (videoTracks.length > 0) {
                console.log("已连接摄像头:", videoTracks[0].label);
                updateDebugInfo('videoSource', videoTracks[0].label || "未知设备");
            }
            
            // 播放视频
            try {
                await userVideo.play();
                console.log("视频成功开始播放");
            } catch (playError) {
                console.error("视频播放失败:", playError);
                
                // 添加点击事件来启动视频（解决自动播放限制）
                if (cameraHint) {
                    cameraHint.textContent = '请点击画面启动视频播放';
                    cameraHint.style.opacity = '1';
                }
                
                const videoContainer = document.querySelector('.camera-container');
                if (videoContainer) {
                    videoContainer.addEventListener('click', () => {
                        userVideo.play().catch(e => console.warn("用户交互播放失败:", e));
                    }, { once: true });
                }
            }
            
            // 设置canvas
            const canvas = document.getElementById('poseCanvas');
            if (canvas) {
                canvas.width = 640;
                canvas.height = 480;
                
                // 等待真实视频尺寸加载后调整
                userVideo.onloadedmetadata = () => {
                    if (userVideo.videoWidth && userVideo.videoHeight) {
                        canvas.width = userVideo.videoWidth;
                        canvas.height = userVideo.videoHeight;
                        console.log(`调整canvas尺寸为: ${canvas.width}x${canvas.height}`);
                    }
                };
            }
            
            updateDebugInfo('cameraStatus', '已连接');
            
            if (cameraHint) {
                cameraHint.textContent = '摄像头已连接，正在检测姿势...';
                setTimeout(() => {
                    cameraHint.style.opacity = '0.5';
                }, 2000);
            }
            
            return true;
            
        } catch (err) {
            console.error("获取媒体流失败:", err);
            
            let errorMessage = '摄像头访问失败';
            
            // 提供更友好的错误消息
            if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
                errorMessage = '摄像头访问被拒绝，请在浏览器设置中允许访问摄像头';
            } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
                errorMessage = '未找到摄像头设备，请确认摄像头已连接';
            } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
                errorMessage = '摄像头可能被其他应用程序占用，请关闭可能在使用摄像头的应用';
            }
            
            throw new Error(errorMessage + ': ' + err.message);
        }
        
    } catch (error) {
        console.error("摄像头初始化错误:", error);
        updateDebugInfo('cameraError', error.message);
        
        const cameraHint = document.getElementById('cameraHint');
        if (cameraHint) {
            cameraHint.textContent = `摄像头错误: ${error.message}`;
            cameraHint.style.opacity = '1';
        }
        
        // 增加尝试计数
        cameraInitAttempts++;
        
        // 显示错误消息
        showErrorMessage(`摄像头初始化失败: ${error.message}`, "摄像头错误", true, () => {
            // 重置尝试计数
            cameraInitAttempts = 0;
            // 重新尝试初始化
            initializeCamera();
        });
        
        return false;
    }
}

// 切换会话状态（开始/停止）
async function toggleSession() {
    const startButton = document.getElementById('startSession');
    const statusMessage = document.getElementById('statusMessage');
    const cameraHint = document.getElementById('cameraHint');
    
    if (!startButton || !statusMessage) {
        console.error('找不到必要的UI元素');
        return;
    }
    
    try {
        // 检查是否已经在处理中，防止重复点击
        if (startButton.disabled) {
            console.log('按钮已禁用，正在处理中，忽略点击');
            return;
        }
        
        if (!sessionActive) {
            // 开始会话
            console.log('尝试启动康复指导会话:', new Date().toLocaleTimeString());
            
            // 重置语音合成系统
            if (typeof resetSpeechSynthesis === 'function') {
                resetSpeechSynthesis();
            }
            
            // 更新按钮状态
            startButton.innerHTML = '<span class="loading-spinner"></span>正在启动...';
            startButton.disabled = true;
            startButton.classList.remove('btn-primary');
            startButton.classList.add('btn-secondary');
            
            cameraInitAttempts = 0;
            
            // 清除之前的会话计时器
            if (sessionStartTimer) {
                clearTimeout(sessionStartTimer);
            }
            
            // 设置15秒超时
            sessionStartTimer = setTimeout(() => {
                if (!sessionActive) {
                    startButton.innerHTML = '开始康复指导';
                    startButton.disabled = false;
                    startButton.classList.remove('btn-secondary');
                    startButton.classList.add('btn-primary');
                    updateStatusMessage('会话启动超时，请重试');
                    showErrorMessage('会话启动超时。请检查摄像头权限并重试。', '启动失败', true, toggleSession);
                    
                    // 清理资源但保留视频显示
                    cleanupSessionKeepVideo();
                }
            }, 15000);
            
            // 1. 先更新状态提示
            updateStatusMessage("正在初始化系统...");
            
            if (cameraHint) {
                cameraHint.textContent = '正在启动摄像头...';
                cameraHint.style.opacity = '1';
                cameraHint.style.backgroundColor = 'rgba(0, 123, 255, 0.2)'; // 蓝色背景
            }
            
            // 2. 初始化摄像头
            let cameraInitialized = false;
            try {
                // 显示提示帮助用户定位
                if (typeof speakFeedback === 'function') {
                    speakFeedback("正在启动摄像头，请稍候", true);
                }
                
                cameraInitialized = await initializeCamera();
                if (!cameraInitialized) {
                    throw new Error('摄像头初始化失败，请检查摄像头连接和浏览器权限');
                }
                
                // 确保视频元素可见
                const userVideoElement = document.getElementById('userVideo');
                if (userVideoElement) {
                    userVideoElement.style.display = 'block';
                }
                
                updateStatusMessage("摄像头已启动，正在连接康复指导系统...");
                if (cameraHint) {
                    cameraHint.textContent = '摄像头已启动，正在连接康复指导系统...';
                }
            } catch (error) {
                console.error('摄像头初始化失败:', error);
                throw new Error('摄像头初始化失败: ' + error.message);
            }
            
            // 3. 调用后端API启动康复会话
            try {
                console.log('调用后端API启动康复会话...');
                const response = await fetch('/api/rehab/start', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    }
                });
                
                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({}));
                    throw new Error(errorData.message || `服务器返回错误: ${response.status}`);
                }
                
                const data = await response.json();
                if (data.status !== 'success') {
                    throw new Error(data.message || '后端启动康复会话失败');
                }
                
                console.log('后端康复会话启动成功:', data);
                backendSessionStarted = true;
                
                // 启动关键点轮询
                if (typeof startKeyPointsPolling === 'function') {
                    startKeyPointsPolling();
                }
                
                // 会话成功启动，更新UI状态
                sessionActive = true;
                
                updateStatusMessage("康复指导已启动，请按照语音提示进行");
                if (cameraHint) {
                    cameraHint.textContent = '请站在摄像头前，保持身体完全可见';
                    cameraHint.style.backgroundColor = 'rgba(40, 167, 69, 0.2)'; // 绿色背景
                    setTimeout(() => {
                        cameraHint.style.opacity = '0.5';
                    }, 3000);
                }
                
                // 语音提示
                if (typeof speakFeedback === 'function') {
                    speakFeedback("康复指导已启动，请站在摄像头前，保持身体完全可见", true);
                }
                
                startButton.innerHTML = '结束康复指导';
                startButton.disabled = false;
                startButton.classList.remove('btn-secondary');
                startButton.classList.add('btn-danger');
                
                // 清除会话启动计时器
                if (sessionStartTimer) {
                    clearTimeout(sessionStartTimer);
                    sessionStartTimer = null;
                }
                
                // 开始监控摄像头状态
                if (typeof startCameraMonitoring === 'function') {
                    startCameraMonitoring();
                }
                
                // 显示成功通知
                if (typeof showNotification === 'function') {
                    showNotification("康复指导已成功启动", "success", 3000);
                }
                
            } catch (apiError) {
                console.error('调用后端API启动康复会话失败:', apiError);
                
                // 简化错误消息显示
                let errorMessage = apiError.message;
                if (errorMessage.includes('摄像头初始化失败') || errorMessage.includes('camera initialization')) {
                    errorMessage = '摄像头无法启动，请检查设备连接和浏览器权限';
                }
                
                if (typeof showNotification === 'function') {
                    showNotification(`启动失败: ${errorMessage}`, 'danger');
                }
                
                if (typeof speakFeedback === 'function') {
                    speakFeedback("康复指导启动失败，"+errorMessage, true);
                }
                
                // 清理资源但保持视频可见
                cleanupSessionKeepVideo();
                throw new Error(errorMessage);
            }
            
        } else {
            // 结束会话
            await stopActiveSession();
        }
    } catch (error) {
        console.error('会话切换失败:', error);
        
        // 重置会话状态
        sessionActive = false;
        backendSessionStarted = false;
        
        // 更新UI
        startButton.innerHTML = '开始康复指导';
        startButton.disabled = false;
        startButton.classList.remove('btn-secondary', 'btn-danger');
        startButton.classList.add('btn-primary');
        
        updateStatusMessage(`会话启动失败: ${error.message}`);
        
        if (cameraHint) {
            cameraHint.textContent = `会话启动失败: ${error.message}`;
            cameraHint.style.opacity = '1';
            cameraHint.style.backgroundColor = 'rgba(220, 53, 69, 0.2)'; // 红色背景
        }
        
        // 显示错误消息
        showErrorMessage(`会话启动失败: ${error.message}`, '启动失败', true, toggleSession);
        
        // 清除会话启动计时器
        if (sessionStartTimer) {
            clearTimeout(sessionStartTimer);
            sessionStartTimer = null;
        }
        
        // 确保清理任何摄像头资源但保持视频显示
        cleanupSessionKeepVideo();
    }
}

// 新增：清理会话资源但保持视频可见
function cleanupSessionKeepVideo() {
    // 停止姿态检测
    if (typeof stopPoseDetection === 'function') {
        stopPoseDetection();
    }
    
    // 停止摄像头监控
    if (typeof stopCameraMonitoring === 'function') {
        stopCameraMonitoring();
    }
    
    // 停止轮询关键点
    if (typeof stopKeyPointsPolling === 'function') {
        stopKeyPointsPolling();
    }
    
    // 重置语音系统
    if (typeof resetSpeechSynthesis === 'function') {
        resetSpeechSynthesis();
    }
    
    // 重置状态但保持视频流
    sessionActive = false;
    backendSessionStarted = false;
    
    // 确保视频元素可见
    const videoElement = document.getElementById('userVideo');
    if (videoElement) {
        videoElement.style.display = 'block';
    }
}

// 停止活动会话
async function stopActiveSession() {
    const startButton = document.getElementById('startSession');
    
    // 更新按钮状态
    if (startButton) {
        startButton.disabled = true;
        startButton.innerHTML = '<span class="loading-spinner"></span>正在关闭...';
        startButton.classList.remove('btn-danger');
        startButton.classList.add('btn-secondary');
    }
    
    // 更新状态提示
    updateStatusMessage("正在结束康复指导...");
    
    const cameraHint = document.getElementById('cameraHint');
    if (cameraHint) {
        cameraHint.textContent = '正在结束康复指导...';
        cameraHint.style.opacity = '1';
    }
    
    // 停止姿态检测相关功能
    stopPoseDetection();
    stopCameraMonitoring();
    stopKeyPointsPolling();
    
    // 调用后端API停止康复会话
    if (backendSessionStarted) {
        try {
            console.log('调用后端API停止康复会话...');
            await fetch('/api/rehab/stop', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            backendSessionStarted = false;
        } catch (apiError) {
            console.error('调用后端API停止康复会话失败:', apiError);
        }
    }
    
    // 停止摄像头
    stopCamera();
    
    // 重置语音系统
    resetSpeechSynthesis();
    
    // 更新状态
    sessionActive = false;
    updateStatusMessage("康复指导会话已结束");
    
    if (cameraHint) {
        cameraHint.textContent = '康复指导已结束，感谢您的参与';
        cameraHint.style.opacity = '1';
        cameraHint.style.backgroundColor = 'rgba(0, 123, 255, 0.2)'; // 蓝色背景
    }
    
    // 语音提示
    speakFeedback("康复指导会话已结束，感谢您的参与", true);
    
    // 更新按钮状态
    if (startButton) {
        startButton.innerHTML = '开始康复指导';
        startButton.disabled = false;
        startButton.classList.remove('btn-secondary');
        startButton.classList.add('btn-primary');
    }
    
    // 显示通知
    showNotification("康复指导已结束", "info", 3000);
    
    return true;
}

// 创建示例姿势数据用于测试
function createDummyPoseData() {
    // 创建测试关键点数据
    const testWidth = 640;
    const testHeight = 480;
    const centerX = testWidth / 2;
    const centerY = testHeight / 2;
    
    return {
        // 头部
        'nose': { x: centerX, y: centerY - 130, visibility: 0.9 },
        'left_eye': { x: centerX - 20, y: centerY - 140, visibility: 0.9 },
        'right_eye': { x: centerX + 20, y: centerY - 140, visibility: 0.9 },
        'left_ear': { x: centerX - 40, y: centerY - 130, visibility: 0.8 },
        'right_ear': { x: centerX + 40, y: centerY - 130, visibility: 0.8 },
        
        // 肩膀
        'left_shoulder': { x: centerX - 70, y: centerY - 70, visibility: 0.95 },
        'right_shoulder': { x: centerX + 70, y: centerY - 70, visibility: 0.95 },
        
        // 手肘
        'left_elbow': { x: centerX - 100, y: centerY, visibility: 0.9 },
        'right_elbow': { x: centerX + 100, y: centerY, visibility: 0.9 },
        
        // 手腕
        'left_wrist': { x: centerX - 130, y: centerY + 30, visibility: 0.85 },
        'right_wrist': { x: centerX + 130, y: centerY + 30, visibility: 0.85 },
        
        // 臀部
        'left_hip': { x: centerX - 50, y: centerY + 50, visibility: 0.9 },
        'right_hip': { x: centerX + 50, y: centerY + 50, visibility: 0.9 },
        
        // 膝盖
        'left_knee': { x: centerX - 55, y: centerY + 120, visibility: 0.85 },
        'right_knee': { x: centerX + 55, y: centerY + 120, visibility: 0.85 },
        
        // 脚踝
        'left_ankle': { x: centerX - 60, y: centerY + 190, visibility: 0.8 },
        'right_ankle': { x: centerX + 60, y: centerY + 190, visibility: 0.8 }
    };
}

// 停止摄像头
function stopCamera() {
    try {
        console.log("停止摄像头...");
        
        // 停止视频流
        if (videoStream) {
            const tracks = videoStream.getTracks();
            tracks.forEach(track => {
                console.log(`停止轨道: ${track.kind} - ${track.label || 'unknown'}`);
                track.stop();
            });
            videoStream = null;
        }
        
        // 清理视频元素
        const userVideo = document.getElementById('userVideo');
        if (userVideo) {
            if (userVideo.srcObject) {
                const oldTracks = userVideo.srcObject.getTracks();
                oldTracks.forEach(track => {
                    if (track.readyState === 'live') {
                        console.log(`停止旧轨道: ${track.kind} - ${track.label || 'unknown'}`);
                        track.stop();
                    }
                });
            }
            userVideo.srcObject = null;
            userVideo.pause();
            userVideo.load();  // 完全重置视频元素
            // 移除这一行: userVideo.style.display = 'none';
        }
        
        // 清除画布
        clearCanvas();
        
        updateDebugInfo('cameraStatus', '已停止');
        updateDebugInfo('videoStatus', '已停止');
        console.log("摄像头停止完成");
        
        return true;
    } catch (error) {
        console.error("停止摄像头错误:", error);
        updateDebugInfo('lastError', `停止摄像头错误: ${error.message}`);
        return false;
    }
}

// 清除画布
function clearCanvas() {
    try {
        const canvas = document.getElementById('poseCanvas');
        if (canvas) {
            const ctx = canvas.getContext('2d');
            if (ctx) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
            }
        }
    } catch (e) {
        console.error("清除画布错误:", e);
    }
}

// 检查会话状态并更新UI
function checkSessionStatus() {
    console.log('检查会话状态...');
    fetch('/video/status')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP错误! 状态: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('会话状态:', data);
            
            // 更新详细调试信息面板
            if (data.debug_info) {
                const debugInfoStr = Object.entries(data.debug_info)
                    .map(([key, value]) => `${key}: ${value}`)
                    .join(' | ');
                updateDebugInfo('detailedStatus', debugInfoStr);
            }
            
            // 更新摄像头状态
            let cameraStatusText = data.camera_ready ? '正常工作' : '未就绪';
            if (data.debug_info && data.debug_info.camera_error) {
                cameraStatusText += ` (错误: ${data.debug_info.camera_error})`;
            }
            updateDebugInfo('cameraStatus', cameraStatusText);
            
            // 更新后端连接状态
            let backendStatus = data.initialized ? '已连接' : '未连接';
            if (data.debug_info && data.debug_info.processor_error) {
                backendStatus += ` (错误: ${data.debug_info.processor_error})`;
            }
            updateDebugInfo('backendStatus', backendStatus);
            
            // 更新姿势检测状态
            updateDebugInfo('poseStatus', data.pose_detector_ready ? '已加载' : '未加载');
            
            // 显示FPS如果可用
            if (data.debug_info && data.debug_info.fps) {
                updateDebugInfo('fpsMeter', `${data.debug_info.fps.toFixed(1)} FPS`);
            }
            
            // 处理摄像头错误
            if (!data.camera_ready && sessionActive) {
                let errorMessage = data.message || '摄像头未就绪，请检查权限和连接';
                if (data.debug_info) {
                    // 添加更具体的错误信息
                    if (data.debug_info.camera_error) {
                        errorMessage += `\n摄像头错误: ${data.debug_info.camera_error}`;
                    }
                    if (data.debug_info.frame_error) {
                        errorMessage += `\n帧读取错误: ${data.debug_info.frame_error}`;
                    }
                }
                
                document.getElementById('cameraHint').textContent = errorMessage;
                
                // 如果摄像头已断开但会话仍然活跃，显示警告
                showNotification('摄像头连接异常，请检查设备', 'warning');
                
                // 如果连续多次检测到错误，尝试自动重连
                videoFailureCount++;
                if (videoFailureCount > 5) {
                    console.warn('检测到持续的摄像头错误，尝试重新连接...');
                    // 尝试重新初始化摄像头
                    initializeCamera().then(success => {
                        if (success) {
                            videoFailureCount = 0;
                            showNotification('摄像头已重新连接', 'success');
                        }
                    });
                }
            } else {
                videoFailureCount = 0;
            }
        })
        .catch(error => {
            console.error('获取状态失败:', error);
            updateDebugInfo('backendStatus', `错误: ${error.message}`);
            
            // 尝试恢复措施
            errorRecoveryCount++;
            if (errorRecoveryCount > 3) {
                errorRecoveryCount = 0;
                showNotification('连接服务器失败，尝试重新初始化...', 'danger');
                
                // 如果长时间无法连接服务器，可以尝试重新启动服务
                if (sessionActive) {
                    // 这里可以添加重启逻辑或通知用户
                }
            }
        });
}

// 更新偏差指标
function updateDeviationIndicators(deviations) {
    // 侧向偏差指示器
    const lateralElement = document.getElementById('lateralDeviation');
    lateralElement.style.width = `${Math.min(100, deviations.lateral_deviation * 5)}%`;
    
    if (deviations.lateral_deviation > 10) {
        lateralElement.className = 'progress-bar bg-danger';
    } else if (deviations.lateral_deviation > 5) {
        lateralElement.className = 'progress-bar bg-warning';
    } else {
        lateralElement.className = 'progress-bar bg-success';
    }
    
    // 前倾角度指示器
    const forwardElement = document.getElementById('forwardTilt');
    forwardElement.style.width = `${Math.min(100, deviations.forward_tilt * 5)}%`;
    
    if (deviations.forward_tilt > 10) {
        forwardElement.className = 'progress-bar bg-danger';
    } else if (deviations.forward_tilt > 5) {
        forwardElement.className = 'progress-bar bg-warning';
    } else {
        forwardElement.className = 'progress-bar bg-success';
    }
    
    // 肩部平衡指示器
    const shoulderElement = document.getElementById('shoulderBalance');
    shoulderElement.style.width = `${Math.min(100, deviations.shoulder_balance * 5)}%`;
    
    if (deviations.shoulder_balance > 8) {
        shoulderElement.className = 'progress-bar bg-danger';
    } else if (deviations.shoulder_balance > 4) {
        shoulderElement.className = 'progress-bar bg-warning';
    } else {
        shoulderElement.className = 'progress-bar bg-success';
    }
}

// 显示通知
function showNotification(message, type = 'info', duration = 3000) {
    const notification = document.getElementById('notification');
    const notificationText = document.getElementById('notificationText');
    
    if (!notification || !notificationText) {
        console.error('通知元素不存在');
        alert(message); // 回退到使用alert
        return;
    }
    
    // 设置消息内容和类型
    notificationText.textContent = message;
    
    // 设置样式类
    notification.className = `alert alert-${type} text-center mb-3`;
    
    // 显示通知
    notification.style.opacity = 0;
    notification.style.display = 'block';
    
    // 淡入效果
    setTimeout(() => {
        notification.style.opacity = 1;
    }, 10);
    
    // 设置自动关闭
    if (duration > 0) {
        setTimeout(() => {
            // 淡出效果
            notification.style.opacity = 0;
            setTimeout(() => {
                notification.style.display = 'none';
            }, 500); // 等待0.5秒完成淡出
        }, duration);
    }
}

// 在toggleSession函数中更新WebGL信息
function updateWebGLStatus() {
    // 获取并显示WebGL信息
    try {
        if (typeof tf !== 'undefined') {
            const backend = tf.getBackend();
            updateDebugInfo('tfBackend', backend || '未初始化');
            
            // 如果使用WebGL，获取更多信息
            if (backend === 'webgl') {
                const gl = document.createElement('canvas').getContext('webgl');
                if (gl) {
                    const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
                    if (debugInfo) {
                        const vendor = gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL);
                        const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
                        console.log('WebGL详细信息:', vendor, renderer);
                    }
                }
            }
        }
    } catch (e) {
        console.warn('获取WebGL信息失败:', e);
    }
}

// 显示错误指导
function showErrorGuidance(error) {
    const errorMsg = error.message || '未知错误';
    let guidance = '';
    
    if (errorMsg.includes('WebGL')) {
        guidance = `
            <div class="error-guidance">
                <p>您的浏览器不支持WebGL，无法加载AI模型。请尝试：</p>
                <ol>
                    <li>使用最新版Chrome或Edge浏览器</li>
                    <li>在浏览器设置中启用硬件加速</li>
                    <li>更新显卡驱动</li>
                </ol>
            </div>
        `;
    } else if (errorMsg.includes('模型加载')) {
        guidance = `
            <div class="error-guidance">
                <p>模型加载失败，可能是网络问题。请尝试：</p>
                <ol>
                    <li>检查网络连接</li>
                    <li>关闭VPN或代理</li>
                    <li>清除浏览器缓存并刷新</li>
                    <li>等待几分钟后再试</li>
                </ol>
            </div>
        `;
    } else if (errorMsg.includes('摄像头')) {
        guidance = `
            <div class="error-guidance">
                <p>摄像头访问失败。请尝试：</p>
                <ol>
                    <li>确保您已允许浏览器访问摄像头</li>
                    <li>检查摄像头是否被其他应用程序占用</li>
                    <li>重新连接摄像头设备</li>
                    <li>刷新页面重试</li>
                </ol>
            </div>
        `;
    }
    
    if (guidance) {
        // 查找或创建错误指导容器
        let guidanceContainer = document.getElementById('errorGuidance');
        if (!guidanceContainer) {
            guidanceContainer = document.createElement('div');
            guidanceContainer.id = 'errorGuidance';
            guidanceContainer.className = 'alert alert-danger mt-3';
            
            // 添加到页面
            const statusMessage = document.getElementById('statusMessage');
            if (statusMessage && statusMessage.parentNode) {
                statusMessage.parentNode.insertBefore(guidanceContainer, statusMessage.nextSibling);
            }
        }
        
        guidanceContainer.innerHTML = guidance;
        guidanceContainer.style.display = 'block';
    }
}

// 更新调试信息
function updateDebugInfo(key, value) {
    const element = document.getElementById(key);
    if (element) {
        element.textContent = value;
    }
    
    // 同时更新到控制台，便于调试
    if (key === 'detailedStatus' || key === 'lastError') {
        console.debug(`[DEBUG] ${key}: ${value}`);
    }
}

// 更新当前运动步骤
function updateExerciseStep(stepNumber) {
    // 更新全局状态
    currentExerciseStep = stepNumber;
    
    // 获取所有步骤元素
    const stepElements = document.querySelectorAll('.exercise-step');
    
    // 更新步骤高亮
    stepElements.forEach(step => {
        const stepIndex = parseInt(step.getAttribute('data-step'));
        if (stepIndex === currentExerciseStep) {
            step.classList.add('active');
        } else {
            step.classList.remove('active');
        }
    });
}

// 根据姿势数据更新偏差指标
function updateDeviationIndicatorsFromPose(pose) {
    const keypoints = pose.keypoints;
    if (!keypoints || keypoints.length < 17) return;
    
    // 获取关键点
    const leftShoulder = getKeypointByName(keypoints, 'left_shoulder');
    const rightShoulder = getKeypointByName(keypoints, 'right_shoulder');
    const leftHip = getKeypointByName(keypoints, 'left_hip');
    const rightHip = getKeypointByName(keypoints, 'right_hip');
    const nose = getKeypointByName(keypoints, 'nose');
    const leftEar = getKeypointByName(keypoints, 'left_ear');
    const rightEar = getKeypointByName(keypoints, 'right_ear');
    
    if (!leftShoulder || !rightShoulder || !leftHip || !rightHip) return;
    
    // 计算侧向偏差（脊柱侧弯）
    const midShoulder = {
        x: (leftShoulder.x + rightShoulder.x) / 2,
        y: (leftShoulder.y + rightShoulder.y) / 2
    };
    
    const midHip = {
        x: (leftHip.x + rightHip.x) / 2,
        y: (leftHip.y + rightHip.y) / 2
    };
    
    const lateralDeviation = Math.abs(midShoulder.x - midHip.x) / Math.abs(midShoulder.y - midHip.y) * 10;
    
    // 计算前倾角度
    let forwardTilt = 0;
    if (nose && leftEar && rightEar) {
        const midEar = {
            x: (leftEar.x + rightEar.x) / 2,
            y: (leftEar.y + rightEar.y) / 2
        };
        
        forwardTilt = Math.abs(nose.x - midEar.x) / Math.abs(leftShoulder.x - rightShoulder.x) * 20;
    }
    
    // 计算肩部平衡
    const shoulderBalance = Math.abs(leftShoulder.y - rightShoulder.y) / Math.abs(leftShoulder.x - rightShoulder.x) * 15;
    
    // 更新UI指标
    const deviations = {
        lateral_deviation: lateralDeviation,
        forward_tilt: forwardTilt,
        shoulder_balance: shoulderBalance
    };
    
    updateDeviationIndicators(deviations);
}

// 根据名称获取关键点
function getKeypointByName(keypoints, name) {
    const point = keypoints.find(kp => kp.name === name);
    if (point && point.score > 0.3) {  // 降低阈值以获取更多关键点
        return point;
    }
    return null;
}

// 更新进度显示
function updateProgress(percent, message) {
    const statusMessage = document.getElementById('statusMessage');
    if (statusMessage) {
        statusMessage.innerHTML = `${message} <span class="badge bg-info">${percent}%</span>`;
    }
    
    if (window.updateCameraHint) {
        window.updateCameraHint(message);
    }
}

/**
 * 更新反馈信息显示
 * @param {Array|String} feedback - 反馈信息，可以是字符串或字符串数组
 */
function updateFeedback(feedback) {
    const feedbackContainer = document.getElementById('feedbackContainer');
    
    if (!feedbackContainer) {
        console.error('找不到反馈容器元素');
        return;
    }
    
    // 清空现有内容
    feedbackContainer.innerHTML = '';
    
    // 确保feedback是数组
    const feedbackArray = Array.isArray(feedback) ? feedback : [feedback];
    
    // 添加新的反馈项
    feedbackArray.forEach(item => {
        if (!item) return; // 跳过空项
        
        const feedbackItem = document.createElement('div');
        feedbackItem.className = 'feedback-item';
        feedbackItem.textContent = item;
        
        // 添加动画效果
        feedbackItem.style.animation = 'feedback-fade-in 0.5s forwards';
        
        feedbackContainer.appendChild(feedbackItem);
    });
    
    // 设置一个计时器来淡出反馈
    setTimeout(() => {
        const items = feedbackContainer.querySelectorAll('.feedback-item');
        items.forEach(item => {
            item.style.animation = 'feedback-fade-out 1s forwards';
        });
    }, 5000); // 5秒后开始淡出
}

// 监控摄像头状态
function startCameraMonitoring() {
    stopCameraMonitoring(); // 先停止之前的监控
    
    cameraMonitorInterval = setInterval(() => {
        const video = document.getElementById('userVideo');
        
        if (!video || !videoStream) {
            return; // 没有初始化，无法监控
        }
        
        // 监测视频是否仍在播放
        if (video.paused) {
            console.warn('视频播放已暂停，尝试重新播放');
            videoFailureCount++;
            
            try {
                video.play().catch(e => {
                    console.error('无法重新播放视频:', e);
                });
            } catch (e) {
                console.error('尝试重新播放视频时出错:', e);
            }
        }
        
        // 检查视频轨道状态
        const videoTracks = videoStream.getVideoTracks();
        let trackStatus = '未知';
        
        if (videoTracks && videoTracks.length > 0) {
            const track = videoTracks[0];
            trackStatus = track.enabled ? (track.readyState === 'live' ? '活跃' : '就绪') : '已禁用';
            
            // 如果轨道不再活跃
            if (track.readyState !== 'live' || !track.enabled) {
                console.warn('视频轨道状态异常:', trackStatus);
                videoFailureCount++;
            } else {
                videoFailureCount = 0; // 重置失败计数
            }
        } else {
            console.warn('没有可用的视频轨道');
            videoFailureCount++;
        }
        
        updateDebugInfo('trackStatus', trackStatus);
        
        // 如果持续失败超过阈值，尝试重新初始化
        if (videoFailureCount >= 5 && sessionActive) {
            console.warn(`摄像头状态异常已达${videoFailureCount}次，尝试重新初始化`);
            videoFailureCount = 0;
            
            // 停止当前视频流
            stopCamera();
            
            // 短暂延迟后尝试重新初始化
            setTimeout(() => {
                if (sessionActive) {
                    document.getElementById('cameraHint').textContent = '检测到摄像头异常，正在尝试重新连接...';
                    initializeCamera().then(success => {
                        if (success) {
                            console.log('摄像头已成功重新连接');
                            document.getElementById('cameraHint').textContent = '摄像头已重新连接';
                        } else {
                            console.error('摄像头重新连接失败');
                            document.getElementById('cameraHint').textContent = '摄像头重新连接失败，请刷新页面';
                        }
                    });
                }
            }, 1000);
        }
    }, 3000); // 每3秒检查一次
}

// 停止摄像头监控
function stopCameraMonitoring() {
    if (cameraMonitorInterval) {
        clearInterval(cameraMonitorInterval);
        cameraMonitorInterval = null;
    }
    videoFailureCount = 0;
}

// 初始化调试信息面板
function initDebugInfo() {
    const debugContainer = document.getElementById('debugPanel');
    if (!debugContainer) {
        // 创建调试面板
        const panel = document.createElement('div');
        panel.id = 'debugPanel';
        panel.className = 'debug-panel';
        panel.style.cssText = 'position: fixed; bottom: 10px; right: 10px; background: rgba(0,0,0,0.7); color: white; padding: 10px; border-radius: 5px; font-size: 12px; max-width: 300px; z-index: 9999;';
        panel.innerHTML = `
            <div style="margin-bottom: 5px;">
                <strong>调试信息</strong>
                <button id="toggleDebug" style="float: right; font-size: 10px;">隐藏</button>
            </div>
            <div id="debugContent">
                <div>摄像头状态: <span id="cameraStatus">未初始化</span></div>
                <div>后端状态: <span id="backendStatus">未连接</span></div>
                <div>模型状态: <span id="modelStatus">未加载</span></div>
                <div>姿势检测: <span id="poseStatus">未加载</span></div>
                <div>页面状态: <span id="pageVisibility">可见</span></div>
                <div>FPS: <span id="fpsMeter">-</span></div>
                <div>视频来源: <span id="videoSource">-</span></div>
                <div>分辨率: <span id="videoResolution">-</span></div>
                <div>轨道状态: <span id="trackStatus">-</span></div>
                <div>详细状态: <span id="detailedStatus">-</span></div>
                <div>最近错误: <span id="lastError">无</span></div>
            </div>
        `;
        
        document.body.appendChild(panel);
        
        // 添加切换显示/隐藏的功能
        document.getElementById('toggleDebug').addEventListener('click', function() {
            const content = document.getElementById('debugContent');
            if (content.style.display === 'none') {
                content.style.display = 'block';
                this.textContent = '隐藏';
            } else {
                content.style.display = 'none';
                this.textContent = '显示';
            }
        });
    }
    
    // 添加错误恢复计数器
    window.errorRecoveryCount = 0;
}

// 切换姿势模板
function changeTemplate() {
    const templateSelector = document.getElementById('templateSelector');
    const selectedTemplate = templateSelector.value;
    
    // 显示状态提示
    showNotification(`正在切换到模板: ${selectedTemplate}...`, 'info');
    
    fetch('/api/rehab/template', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ template: selectedTemplate })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            // 模板切换成功
            showNotification(`已切换到模板: ${selectedTemplate}`, 'success');
            // 强制播放一次语音提示，忽略重复检查
            speakFeedback(`已切换到${selectedTemplate}模板`, true);
        } else {
            // 模板切换失败
            showNotification(`切换模板失败: ${data.message}`, 'danger');
        }
    })
    .catch(error => {
        console.error('切换模板失败:', error);
        showNotification(`切换模板失败: ${error.message}`, 'danger');
    });
}

// 清理会话资源的辅助函数
function cleanupSession() {
    // 停止姿态检测
    if (typeof stopPoseDetection === 'function') {
        stopPoseDetection();
    }
    
    // 停止摄像头监控
    if (typeof stopCameraMonitoring === 'function') {
        stopCameraMonitoring();
    }
    
    // 停止轮询关键点
    if (typeof stopKeyPointsPolling === 'function') {
        stopKeyPointsPolling();
    }
    
    // 停止摄像头
    if (typeof stopCamera === 'function') {
        stopCamera();
    }
    
    // 重置语音系统
    if (typeof resetSpeechSynthesis === 'function') {
        resetSpeechSynthesis();
    }
    
    // 重置状态
    sessionActive = false;
    backendSessionStarted = false;
}