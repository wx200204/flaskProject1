/**
 * 康复指导界面交互功能
 * 负责界面元素更新和用户交互
 */

// 全局状态
let sessionActive = false;
let scoreChart = null;
let exerciseState = 'waiting'; // waiting, holding, completed
let currentExerciseStep = 1;
let lastGoodPostureTime = 0;
let holdTimeRequired = 3000; // 保持姿势的时间(毫秒)

// 语音状态
let lastSpokenFeedback = "";
let lastSpeakTime = 0;
let speechSynthesisActive = false;
const MIN_SPEAK_INTERVAL = 3000;

// 对外暴露的UI API
window.RehabUI = {
    initUI,
    updateScore,
    updateFeedback,
    updateExerciseStep,
    toggleRehabSession,
    showNotification,
    speakFeedback
};

/**
 * 初始化界面
 */
function initUI() {
    console.log('初始化康复指导界面');
    
    try {
        // 创建备用通知容器（如果页面中不存在）
        ensureNotificationContainer();
        
        // 显示加载状态通知
        showNotification('正在初始化康复界面...', 'info');
        
        // 初始化图表
        const chartSuccess = initScoreChart();
        
        // 即使图表初始化失败也继续
        if (!chartSuccess) {
            console.warn('图表初始化失败，但将继续初始化其他组件');
            showNotification('评分图表加载失败，但系统仍可使用', 'warning');
        }
        
        // 初始化模板选择
        initTemplateSelector();
        
        // 初始化步骤指示器
        initExerciseSteps();
        
        // 初始化开始/停止按钮
        const startButton = document.getElementById('startSession');
        if (startButton) {
            startButton.addEventListener('click', toggleRehabSession);
        } else {
            console.error('未找到开始会话按钮');
            showNotification('页面元素缺失，某些功能可能不可用', 'warning');
        }
        
        // 添加页面加载完成事件，确保组件正确显示
        window.addEventListener('load', function() {
            setTimeout(() => {
                try {
                    // 初始化康复计划数据
                    if (typeof loadRehabPlan === 'function') {
                        loadRehabPlan();
                    }
                    
                    // 确保模板选择器初始化
                    const templateSelector = document.getElementById('templateSelector');
                    if (templateSelector && templateSelector.options.length > 0) {
                        // 模拟触发一次change事件以更新描述
                        templateSelector.dispatchEvent(new Event('change'));
                    }
                    
                    // 显示康复指南提示
                    showNotification('康复指导系统已准备就绪，请选择一个模板并点击"开始康复指导"', 'info', 5000);
                } catch (loadError) {
                    console.error('页面加载事件处理错误:', loadError);
                    showNotification('初始化组件时出现问题，请刷新页面重试', 'error');
                }
            }, 1000);
        });
        
        // 注册网络状态监听器
        window.addEventListener('online', function() {
            showNotification('网络连接已恢复', 'success');
        });
        
        window.addEventListener('offline', function() {
            showNotification('网络连接已断开，康复功能可能受限', 'warning');
        });
        
        console.log('康复界面初始化完成');
    } catch (error) {
        console.error('初始化康复UI失败:', error);
        showError('康复界面初始化失败: ' + error.message);
    }
}

/**
 * 确保通知容器存在
 */
function ensureNotificationContainer() {
    if (!document.getElementById('notificationContainer')) {
        console.warn('未找到通知容器，将创建一个备用容器');
        
        const container = document.createElement('div');
        container.id = 'notificationContainer';
        container.className = 'notification';
        container.style.position = 'fixed';
        container.style.top = '20px';
        container.style.right = '20px';
        container.style.zIndex = '9999';
        container.style.backgroundColor = '#fff';
        container.style.padding = '15px 20px';
        container.style.borderRadius = '10px';
        container.style.boxShadow = '0 5px 15px rgba(0,0,0,0.2)';
        container.style.display = 'none';
        
        const content = document.createElement('div');
        content.className = 'd-flex align-items-center';
        
        const iconDiv = document.createElement('div');
        iconDiv.className = 'me-3';
        const icon = document.createElement('i');
        icon.id = 'notificationIcon';
        icon.className = 'fas fa-info-circle';
        iconDiv.appendChild(icon);
        
        const messageDiv = document.createElement('div');
        messageDiv.className = 'flex-grow-1';
        const message = document.createElement('div');
        message.id = 'notificationMessage';
        messageDiv.appendChild(message);
        
        const closeDiv = document.createElement('div');
        const closeBtn = document.createElement('button');
        closeBtn.type = 'button';
        closeBtn.className = 'btn-close';
        closeBtn.id = 'closeNotificationBtn';
        closeBtn.addEventListener('click', function() {
            container.style.display = 'none';
        });
        closeDiv.appendChild(closeBtn);
        
        content.appendChild(iconDiv);
        content.appendChild(messageDiv);
        content.appendChild(closeDiv);
        
        container.appendChild(content);
        document.body.appendChild(container);
    }
}

/**
 * 初始化评分图表
 * @returns {boolean} 是否成功初始化
 */
function initScoreChart() {
    try {
        const chartElement = document.getElementById('scoreChart');
        if (!chartElement) {
            console.error('找不到评分图表元素');
            return false;
        }
        
        // 检查Chart.js是否可用
        if (typeof Chart === 'undefined') {
            console.error('Chart.js库未加载');
            return false;
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
                animation: { duration: 0 },
                plugins: {
                    legend: { display: false },
                    tooltip: { enabled: false }
                },
                scales: {
                    y: { min: 0, max: 100, display: false },
                    x: { display: false }
                }
            }
        };
        
        // 创建图表
        scoreChart = new Chart(chartElement, config);
        return true;
    } catch (error) {
        console.error('初始化评分图表失败:', error);
        return false;
    }
}

/**
 * 更新评分图表
 * @param {number} score - 分数
 */
function updateScoreChart(score) {
    if (!scoreChart) return;
    
    // 保持最多30个数据点
    if (scoreChart.data.labels.length > 30) {
        scoreChart.data.labels.shift();
        scoreChart.data.datasets[0].data.shift();
    }
    
    const time = new Date().toLocaleTimeString();
    scoreChart.data.labels.push(time);
    scoreChart.data.datasets[0].data.push(score);
    
    try {
        scoreChart.update();
    } catch (e) {
        console.error('更新图表失败:', e);
    }
}

/**
 * 更新分数显示
 * @param {number} score - 分数
 */
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
    
    updateScoreChart(score);
}

/**
 * 更新反馈内容显示
 * @param {Array} feedback - 反馈消息数组
 */
function updateFeedback(feedback) {
    if (!feedback || !Array.isArray(feedback)) return;
    
    const feedbackBox = document.getElementById('postureFeedback');
    const feedbackContainer = document.getElementById('feedbackContainer');
    
    if (!feedbackBox || !feedbackContainer) return;
    
    // 主反馈框显示
    if (feedback.length > 0) {
        feedbackBox.textContent = feedback[0];
    } else {
        feedbackBox.textContent = '姿势评估中...';
    }
    
    // 动态反馈条目
    feedbackContainer.innerHTML = '';
    
    // 最多显示3条不同的反馈
    const uniqueFeedback = [...new Set(feedback)].slice(0, 3);
    
    uniqueFeedback.forEach(message => {
        const item = document.createElement('div');
        item.className = 'feedback-item';
        item.style.animation = 'feedback-fade-in 0.3s ease-in-out';
        item.textContent = message;
        feedbackContainer.appendChild(item);
    });
}

/**
 * 更新偏差指示器
 * @param {Object} deviations - 偏差数据
 */
function updateDeviationIndicators(deviations) {
    if (!deviations) return;
    
    // 获取各个偏差指示器元素
    const lateralElement = document.getElementById('lateralDeviation');
    const forwardElement = document.getElementById('forwardTilt');
    const shoulderElement = document.getElementById('shoulderBalance');
    
    if (lateralElement && deviations.lateral !== undefined) {
        const lateralPercent = Math.min(100, deviations.lateral * 100);
        lateralElement.style.width = `${lateralPercent}%`;
        updateProgressBarColor(lateralElement, lateralPercent);
    }
    
    if (forwardElement && deviations.forward !== undefined) {
        const forwardPercent = Math.min(100, deviations.forward * 100);
        forwardElement.style.width = `${forwardPercent}%`;
        updateProgressBarColor(forwardElement, forwardPercent);
    }
    
    if (shoulderElement && deviations.shoulder !== undefined) {
        const shoulderPercent = Math.min(100, deviations.shoulder * 100);
        shoulderElement.style.width = `${shoulderPercent}%`;
        updateProgressBarColor(shoulderElement, shoulderPercent);
    }
}

/**
 * 根据百分比更新进度条颜色
 * @param {HTMLElement} element - 进度条元素
 * @param {number} percent - 百分比值
 */
function updateProgressBarColor(element, percent) {
    // 移除现有的颜色类
    element.classList.remove('bg-success', 'bg-warning', 'bg-danger');
    
    // 根据百分比设置不同颜色
    if (percent < 30) {
        element.classList.add('bg-success');
    } else if (percent < 70) {
        element.classList.add('bg-warning');
    } else {
        element.classList.add('bg-danger');
    }
}

/**
 * 初始化模板选择器
 */
function initTemplateSelector() {
    const templateSelector = document.getElementById('templateSelector');
    const templateDescription = document.getElementById('templateDescription');
    
    if (!templateSelector) return;
    
    // 模板选择变化事件
    templateSelector.addEventListener('change', function() {
        const selectedOption = templateSelector.options[templateSelector.selectedIndex];
        
        // 更新模板描述
        if (templateDescription && selectedOption.dataset.description) {
            templateDescription.textContent = selectedOption.dataset.description;
        }
        
        // 如果会话已激活，发送模板变更请求
        if (sessionActive && window.RehabCore && window.RehabCore.changeRehabTemplate) {
            const templateName = selectedOption.value;
            
            // 显示加载中状态
            const statusMsg = document.getElementById('statusMessage');
            if (statusMsg) statusMsg.textContent = '正在切换模板...';
            
            window.RehabCore.changeRehabTemplate(templateName)
                .then(response => {
                    showNotification(`已切换到模板: ${templateName}`, 'success');
                    
                    // 更新状态消息
                    if (statusMsg) statusMsg.textContent = `当前模板: ${templateName}`;
                    
                    // 重置当前步骤
                    updateExerciseStep(1);
                })
                .catch(error => {
                    console.error('切换模板失败:', error);
                    showNotification('切换模板失败，请重试', 'error');
                });
        }
        
        // 选中对应的模板卡片
        const templateCards = document.querySelectorAll('.template-card');
        templateCards.forEach(function(card) {
            const cardTitle = card.querySelector('h6').textContent;
            if (cardTitle === selectedOption.text) {
                card.classList.add('selected');
            } else {
                card.classList.remove('selected');
            }
        });
    });
    
    // 加载可用模板
    if (window.RehabCore && window.RehabCore.fetchAvailableTemplates) {
        window.RehabCore.fetchAvailableTemplates().then(templates => {
            if (!templates || templates.length === 0) {
                console.warn('未获取到可用模板，使用默认模板');
                return;
            }
            
            try {
                // 清空现有选项
                templateSelector.innerHTML = '';
                
                // 添加新选项
                templates.forEach(template => {
                    const option = document.createElement('option');
                    option.value = template.id;
                    option.text = template.name;
                    option.dataset.description = template.description;
                    option.dataset.difficulty = template.difficulty || 'medium';
                    templateSelector.appendChild(option);
                });
                
                // 触发change事件更新描述
                templateSelector.dispatchEvent(new Event('change'));
                
                console.log('模板加载完成');
            } catch (error) {
                console.error('处理模板数据出错:', error);
            }
        }).catch(error => {
            console.error('获取模板列表失败:', error);
            showNotification('获取康复模板失败，将使用默认模板', 'warning');
        });
    }
}

/**
 * 初始化康复练习步骤
 */
function initExerciseSteps() {
    const steps = document.querySelectorAll('.exercise-step');
    if (!steps.length) return;
    
    steps.forEach(step => {
        step.addEventListener('click', function() {
            if (!sessionActive) return;
            
            const stepNumber = parseInt(this.getAttribute('data-step'));
            updateExerciseStep(stepNumber);
        });
    });
    
    // 下一步按钮
    const nextButton = document.getElementById('nextExerciseButton');
    if (nextButton) {
        nextButton.addEventListener('click', function() {
            if (!sessionActive) return;
            
            const nextStep = currentExerciseStep < 3 ? currentExerciseStep + 1 : 1;
            updateExerciseStep(nextStep);
            this.disabled = true;
        });
    }
}

/**
 * 更新练习步骤
 * @param {number} stepNumber - 步骤编号
 */
function updateExerciseStep(stepNumber) {
    stepNumber = parseInt(stepNumber);
    if (isNaN(stepNumber) || stepNumber < 1 || stepNumber > 3) return;
    
    currentExerciseStep = stepNumber;
    exerciseState = 'waiting';
    lastGoodPostureTime = 0;
    
    // 更新UI
    const steps = document.querySelectorAll('.exercise-step');
    steps.forEach(step => {
        const thisStep = parseInt(step.getAttribute('data-step'));
        
        if (thisStep === currentExerciseStep) {
            step.classList.add('current');
            step.classList.remove('waiting', 'holding', 'completed');
            step.classList.add(exerciseState);
        } else {
            step.classList.remove('current', 'waiting', 'holding', 'completed');
        }
    });
    
    // 更新下一步按钮状态
    const nextButton = document.getElementById('nextExerciseButton');
    if (nextButton) {
        nextButton.disabled = true;
    }
    
    // 语音提示
    speakFeedback(`开始第${stepNumber}步`, true);
    
    // 更新保持时间显示
    updateHoldTimeDisplay(0, holdTimeRequired);
}

/**
 * 更新保持时间显示
 * @param {number} currentTime - 当前保持时间
 * @param {number} requiredTime - 需要保持的时间
 */
function updateHoldTimeDisplay(currentTime, requiredTime) {
    const progressElements = document.querySelectorAll('.hold-progress');
    const activeElement = document.querySelector('.exercise-step.current .hold-progress');
    
    if (!progressElements.length) return;
    
    // 重置所有进度条
    progressElements.forEach(el => {
        el.style.width = '0%';
        el.classList.remove('bg-success');
    });
    
    // 更新活动步骤的进度条
    if (activeElement) {
        const percent = Math.min(100, (currentTime / requiredTime) * 100);
        activeElement.style.width = `${percent}%`;
        
        if (percent > 66) {
            activeElement.classList.add('bg-success');
        }
    }
}

/**
 * 语音提示功能
 * @param {string} text - 要播放的文本
 * @param {boolean} force - 是否强制播放
 */
function speakFeedback(text, force = false) {
    if (!text || text.trim() === "") return;
    
    // 防止语音卡住
    if (speechSynthesisActive) {
        const now = Date.now();
        // 如果上次语音播放时间超过10秒，可能是卡住了，强制重置
        if (now - lastSpeakTime > 10000) {
            if ('speechSynthesis' in window) {
                window.speechSynthesis.cancel();
                speechSynthesisActive = false;
            }
        }
    }
    
    const now = Date.now();
    
    // 检查重复 (如果不是强制播放)
    if (!force && text === lastSpokenFeedback && now - lastSpeakTime < MIN_SPEAK_INTERVAL) {
        return;
    }
    
    // 如果语音合成API可用
    if ('speechSynthesis' in window) {
        // 先取消任何当前的语音
        window.speechSynthesis.cancel();
        
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = 'zh-CN';
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        
        // 更新状态并播放
        lastSpokenFeedback = text;
        lastSpeakTime = now;
        speechSynthesisActive = true;
        
        // 设置语音结束或错误时的处理
        utterance.onend = function() {
            speechSynthesisActive = false;
        };
        
        utterance.onerror = function() {
            speechSynthesisActive = false;
        };
        
        // 开始播放
        window.speechSynthesis.speak(utterance);
    }
}

/**
 * 显示通知消息 - 增强的容错版本
 * @param {string} message - 消息内容
 * @param {string} type - 消息类型（info, success, warning, error）
 * @param {number} duration - 显示时长（毫秒）
 */
function showNotification(message, type = 'info', duration = 3000) {
    console.log(`${type.toUpperCase()} 通知: ${message}`);
    
    // 确保通知容器存在
    ensureNotificationContainer();
    
    const notificationContainer = document.getElementById('notificationContainer');
    const notificationMessage = document.getElementById('notificationMessage');
    const notificationIcon = document.getElementById('notificationIcon');
    
    if (!notificationContainer || !notificationMessage) {
        console.warn('通知容器不可用，使用console.log代替');
        return;
    }
    
    try {
        // 设置消息内容
        notificationMessage.textContent = message;
        
        // 清除所有类型类名
        notificationContainer.classList.remove('bg-info', 'bg-success', 'bg-warning', 'bg-danger', 'text-white');
        
        // 设置图标和背景颜色
        if (notificationIcon) {
            notificationIcon.className = 'fas';
            
            switch (type) {
                case 'success':
                    notificationContainer.classList.add('bg-success', 'text-white');
                    notificationIcon.classList.add('fa-check-circle');
                    break;
                case 'warning':
                    notificationContainer.classList.add('bg-warning');
                    notificationIcon.classList.add('fa-exclamation-triangle');
                    break;
                case 'error':
                    notificationContainer.classList.add('bg-danger', 'text-white');
                    notificationIcon.classList.add('fa-times-circle');
                    break;
                case 'info':
                default:
                    notificationContainer.classList.add('bg-info', 'text-white');
                    notificationIcon.classList.add('fa-info-circle');
                    break;
            }
        }
        
        // 显示通知
        notificationContainer.style.display = 'block';
        
        // 添加动画
        if ('animation' in notificationContainer.style) {
            notificationContainer.style.animation = 'none';
            // 触发重绘
            void notificationContainer.offsetWidth;
            notificationContainer.style.animation = 'slideIn 0.3s ease';
        }
        
        // 自动关闭
        if (duration > 0) {
            // 清除旧计时器
            if (notificationContainer.hideTimeout) {
                clearTimeout(notificationContainer.hideTimeout);
            }
            
            // 设置新计时器
            notificationContainer.hideTimeout = setTimeout(() => {
                if ('animation' in notificationContainer.style) {
                    notificationContainer.style.animation = 'slideIn 0.3s ease reverse';
                    
                    // 动画结束后隐藏
                    setTimeout(() => {
                        notificationContainer.style.display = 'none';
                    }, 300);
                } else {
                    notificationContainer.style.display = 'none';
                }
            }, duration);
        }
        
        // 添加关闭按钮事件（如果之前没有添加）
        const closeBtn = document.getElementById('closeNotificationBtn');
        if (closeBtn && !closeBtn._hasEventListener) {
            closeBtn.addEventListener('click', () => {
                notificationContainer.style.display = 'none';
            });
            closeBtn._hasEventListener = true;
        }
    } catch (error) {
        console.error('显示通知时出错:', error);
    }
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

/**
 * 处理姿势数据
 * @param {Object} poseData - 姿势数据
 */
function handlePoseData(poseData) {
    if (!poseData) return;
    
    // 更新评分
    if (typeof poseData.score === 'number') {
        updateScore(poseData.score);
    }
    
    // 更新姿势反馈
    if (poseData.feedback) {
        updateFeedback(poseData.feedback);
    }
    
    // 更新摄像头提示
    updateCameraHint(poseData.landmarks);
    
    // 分析当前练习状态
    if (sessionActive && poseData.landmarks && poseData.landmarks.length > 0) {
        analyzePoseForExercise(poseData);
    }
    
    // 如果姿势状态不是SUCCESS，显示相应状态
    if (poseData.status && poseData.status !== 'SUCCESS') {
        const statusMessage = document.getElementById('statusMessage');
        if (statusMessage) {
            if (poseData.status === 'NOT_DETECTED') {
                statusMessage.textContent = '未检测到人体，请确保您完全在摄像头范围内';
                statusMessage.className = 'text-warning';
            } else {
                statusMessage.textContent = poseData.feedback || '姿势检测异常';
                statusMessage.className = 'text-danger';
            }
        }
    }
}

/**
 * 更新摄像头提示
 * @param {Array} landmarks - 关键点数据
 */
function updateCameraHint(landmarks) {
    const cameraHint = document.getElementById('cameraHint');
    if (!cameraHint) return;
    
    if (!landmarks || landmarks.length === 0) {
        cameraHint.textContent = '未检测到人体，请站在摄像头前';
        cameraHint.style.opacity = '1';
        cameraHint.className = 'camera-hint text-warning';
        return;
    }
    
    // 检查关键点的可见性
    const visibleKeypoints = landmarks.filter(point => point && point.visibility > 0.5).length;
    const totalKeypoints = 33; // Mediapipe Pose的关键点数量
    const visibility = visibleKeypoints / totalKeypoints;
    
    if (visibility < 0.6) {
        // 关键点可见性低，给出适当提示
        cameraHint.textContent = '请调整位置，确保全身在摄像头范围内';
        cameraHint.style.opacity = '1';
        cameraHint.className = 'camera-hint text-warning';
    } else if (visibility < 0.8) {
        // 关键点可见性一般
        cameraHint.textContent = '姿势已检测，但部分身体区域不可见';
        cameraHint.style.opacity = '0.8';
        cameraHint.className = 'camera-hint text-info';
    } else {
        // 关键点可见性良好
        cameraHint.textContent = '姿势检测良好';
        cameraHint.style.opacity = '0.5';
        cameraHint.className = 'camera-hint text-success';
        
        // 几秒后淡出
        setTimeout(() => {
            if (cameraHint.textContent === '姿势检测良好') {
                cameraHint.style.opacity = '0';
            }
        }, 3000);
    }
}

/**
 * 分析当前姿势用于训练指导
 * @param {Object} poseData - 姿势数据
 */
function analyzePoseForExercise(poseData) {
    // 检查当前处于哪个步骤
    const currentStepElement = document.querySelector('.exercise-step.current');
    if (!currentStepElement) return;
    
    const stepNumber = parseInt(currentStepElement.getAttribute('data-step') || '1');
    const stepName = currentStepElement.querySelector('.step-name')?.textContent || '';
    
    // 获取评分
    const score = poseData.score || 0;
    
    // 获取当前状态
    const isWaiting = currentStepElement.classList.contains('waiting');
    const isHolding = currentStepElement.classList.contains('holding');
    const isCompleted = currentStepElement.classList.contains('completed');
    
    // 更新状态消息
    const statusMessage = document.getElementById('statusMessage');
    
    if (isCompleted) {
        // 已完成状态，不需要处理
        return;
    }
    
    // 根据评分判断动作是否正确
    if (score >= 70) { // 良好姿势阈值
        // 姿势良好
        if (isWaiting) {
            // 从等待状态切换到保持状态
            currentStepElement.classList.remove('waiting');
            currentStepElement.classList.add('holding');
            
            if (statusMessage) {
                statusMessage.textContent = `很好！请保持${stepName}姿势`;
                statusMessage.className = 'text-success';
            }
            
            // 播放语音反馈
            speakFeedback(`很好！请保持${stepName}姿势`);
            
            // 记录良好姿势的开始时间
            lastGoodPostureTime = Date.now();
        } else if (isHolding) {
            // 计算已经保持的时间
            const holdTime = Date.now() - lastGoodPostureTime;
            const progress = (holdTime / holdTimeRequired) * 100;
            
            // 更新保持时间条
            const holdTimerBar = currentStepElement.querySelector('.progress-bar');
            if (holdTimerBar) {
                holdTimerBar.style.width = `${Math.min(progress, 100)}%`;
                updateProgressBarColor(holdTimerBar, progress);
            }
            
            if (statusMessage) {
                const remainingSeconds = Math.ceil((holdTimeRequired - holdTime) / 1000);
                statusMessage.textContent = `保持${stepName}姿势，还需${remainingSeconds}秒`;
                statusMessage.className = 'text-success';
            }
            
            // 检查是否已经保持足够时间
            if (holdTime >= holdTimeRequired) {
                // 完成当前步骤
                currentStepElement.classList.remove('holding');
                currentStepElement.classList.add('completed');
                
                // 播放完成语音
                speakFeedback(`${stepName}姿势完成！做得非常好`);
                
                if (statusMessage) {
                    statusMessage.textContent = `${stepName}姿势完成！`;
                    statusMessage.className = 'text-success';
                }
                
                // 如果不是最后一步，启用下一步按钮
                const nextButton = document.getElementById('nextExerciseButton');
                if (nextButton && stepNumber < 3) {
                    nextButton.disabled = false;
                }
            }
        }
    } else if (score < 50) { // 姿势不正确阈值
        // 姿势不正确
        if (isHolding) {
            // 从保持状态回到等待状态
            currentStepElement.classList.remove('holding');
            currentStepElement.classList.add('waiting');
            
            if (statusMessage) {
                statusMessage.textContent = `姿势不正确，请按照指导调整`;
                statusMessage.className = 'text-warning';
            }
            
            // 播放语音反馈
            speakFeedback(poseData.feedback || `姿势不正确，请调整${stepName}姿势`);
            
            // 重置保持时间条
            const holdTimerBar = currentStepElement.querySelector('.progress-bar');
            if (holdTimerBar) {
                holdTimerBar.style.width = '0%';
            }
        } else if (isWaiting) {
            // 还在等待，根据分数给予建议
            if (statusMessage) {
                statusMessage.textContent = poseData.feedback || `请按指导完成${stepName}姿势`;
                statusMessage.className = 'text-warning';
            }
        }
    }
    
    // 更新训练步骤状态
    updateExerciseStepState();
}

/**
 * 更新练习步骤状态显示
 */
function updateExerciseStepState() {
    document.querySelectorAll('.exercise-step').forEach(step => {
        const stepNumber = parseInt(step.getAttribute('data-step'));
        
        if (stepNumber === currentExerciseStep) {
            step.classList.add('current');
            
            // 根据当前保持姿势的状态更新样式
            step.classList.remove('waiting', 'holding', 'completed');
            step.classList.add(exerciseState);
        }
    });
}

/**
 * 切换康复会话状态
 */
async function toggleRehabSession() {
    const startButton = document.getElementById('startSession');
    const statusMessage = document.getElementById('statusMessage');
    
    if (!startButton) return;
    
    try {
        if (!sessionActive) {
            // 启动会话
            startButton.disabled = true;
            startButton.textContent = '正在连接摄像头...';
            
            if (statusMessage) {
                statusMessage.textContent = '正在启动摄像头和姿势识别...';
                statusMessage.className = 'text-info';
            }
            
            // 初始化摄像头
            showNotification('正在启动摄像头...', 'info');
            await window.RehabCore.initializeCamera();
            
            // 确保当前步骤状态正确
            initExerciseSteps();
            
            // 启动后端会话
            showNotification('正在连接康复指导系统...', 'info');
            try {
                await window.RehabCore.startBackendSession();
            } catch (error) {
                console.error('启动后端会话失败:', error);
                showNotification('连接康复指导系统失败: ' + error.message, 'error');
                throw error;
            }
            
            // 启动关键点轮询
            window.RehabCore.startKeyPointsPolling(handlePoseData);
            
            // 更新UI状态
            sessionActive = true;
            startButton.textContent = '停止康复指导';
            startButton.classList.replace('btn-primary', 'btn-danger');
            startButton.disabled = false;
            
            // 显示指导提示
            if (statusMessage) {
                statusMessage.textContent = '请按照指导完成姿势';
                statusMessage.className = 'text-primary';
            }
            
            // 使用语音欢迎并指导
            speakFeedback('康复指导系统已启动，请站在摄像头前，按照步骤指导完成动作', true);
            
            // 显示成功消息
            showNotification('康复指导已开始，请按照步骤完成动作', 'success');
            
        } else {
            // 停止会话
            startButton.disabled = true;
            startButton.textContent = '正在停止...';
            
            if (statusMessage) {
                statusMessage.textContent = '正在停止康复指导...';
                statusMessage.className = 'text-info';
            }
            
            // 停止关键点轮询
            window.RehabCore.stopKeyPointsPolling();
            
            // 停止后端会话
            try {
                await window.RehabCore.stopBackendSession();
            } catch (error) {
                console.warn('停止后端会话出错:', error);
            }
            
            // 停止摄像头
            window.RehabCore.stopCamera();
            
            // 重置UI状态
            sessionActive = false;
            exerciseState = 'waiting';
            currentExerciseStep = 1;
            lastGoodPostureTime = 0;
            
            // 清除保持时间条
            const progressBars = document.querySelectorAll('.exercise-step .progress-bar');
            progressBars.forEach(bar => {
                bar.style.width = '0%';
                bar.className = 'progress-bar';
            });
            
            // 恢复第一步状态
            const steps = document.querySelectorAll('.exercise-step');
            steps.forEach((step, index) => {
                step.classList.remove('current', 'completed', 'holding');
                if (index === 0) {
                    step.classList.add('current', 'waiting');
                }
            });
            
            // 禁用下一步按钮
            const nextButton = document.getElementById('nextExerciseButton');
            if (nextButton) {
                nextButton.disabled = true;
            }
            
            // 更新UI
            startButton.textContent = '开始康复指导';
            startButton.classList.replace('btn-danger', 'btn-primary');
            startButton.disabled = false;
            
            if (statusMessage) {
                statusMessage.textContent = '请点击"开始康复指导"按钮';
                statusMessage.className = 'text-muted';
            }
            
            // 显示停止消息
            showNotification('康复指导已停止', 'info');
        }
    } catch (error) {
        console.error('切换康复会话状态出错:', error);
        
        // 恢复按钮状态
        startButton.textContent = sessionActive ? '停止康复指导' : '开始康复指导';
        startButton.className = `btn ${sessionActive ? 'btn-danger' : 'btn-primary'} btn-lg`;
        startButton.disabled = false;
        
        if (statusMessage) {
            statusMessage.textContent = `出错: ${error.message}`;
            statusMessage.className = 'text-danger';
        }
        
        // 显示错误消息
        showNotification(`康复指导启动失败: ${error.message}`, 'error');
    }
} 