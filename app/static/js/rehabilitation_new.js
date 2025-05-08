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

// 后端连接状态跟踪
let backendSessionStarted = false;

// 语音提示
let lastSpokenFeedback = ""; // 上次播放的反馈内容
let lastSpeakTime = 0;       // 上次播放的时间
const MIN_SPEAK_INTERVAL = 3000; // 最小语音间隔（毫秒）
const NO_DETECTION_SPEAK_INTERVAL = 8000; // "未检测到人体"的语音间隔（毫秒）
let speechSynthesisActive = false; // 是否正在进行语音合成

// 康复计划数据
let rehabPlanData = null;
let availableTemplates = null;

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

/**
 * 显示通知消息
 * @param {string} message - 消息内容
 * @param {string} type - 消息类型 (info, success, warning, danger)
 * @param {number} duration - 显示时长(毫秒)
 */
function showNotification(message, type = 'info', duration = 3000) {
    const notification = document.getElementById('notification');
    const notificationText = document.getElementById('notificationText');
    
    if (!notification || !notificationText) {
        console.error('通知元素不存在');
        return;
    }
    
    // 清除任何现有的类和超时
    notification.className = 'alert';
    clearTimeout(notification.timeout);
    
    // 添加适当的类
    switch(type) {
        case 'success':
            notification.classList.add('alert-success');
            break;
        case 'warning':
            notification.classList.add('alert-warning');
            break;
        case 'danger':
            notification.classList.add('alert-danger');
            break;
        default:
            notification.classList.add('alert-info');
    }
    
    // 设置消息并显示
    notificationText.textContent = message;
    notification.style.opacity = '1';
    
    // 设置超时以隐藏通知
    notification.timeout = setTimeout(() => {
        notification.style.opacity = '0';
    }, duration);
}

/**
 * 更新状态消息
 * @param {string} message - 状态消息
 */
function updateStatusMessage(message) {
    const statusMessage = document.getElementById('statusMessage');
    if (statusMessage) {
        statusMessage.textContent = message;
    }
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
    
    // 添加页面可见性监听
    document.addEventListener('visibilitychange', handleVisibilityChange);
    
    // 添加页面卸载监听
    window.addEventListener('beforeunload', cleanupResources);
    
    // 获取可用的康复模板
    fetchAvailableTemplates();
    
    // 加载康复计划(如果有)
    fetchRehabPlan();
    
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

// 更新评分图表
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

// 更新分数
function updateScore(score) {
    const scoreElement = document.getElementById('currentScore');
    if (scoreElement) {
        scoreElement.textContent = score;
        
        // 根据分数设置不同颜色
        if (score >= 90) {
            scoreElement.className = 'score-value excellent';
        } else if (score >= 75) {
            scoreElement.className = 'score-value good';
        } else if (score >= 60) {
            scoreElement.className = 'score-value average';
        } else {
            scoreElement.className = 'score-value poor';
        }
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

// 导出公共函数供页面直接调用
window.updateExerciseStep = updateExerciseStep;
window.changeTemplate = changeTemplate;
window.toggleSession = toggleSession; 