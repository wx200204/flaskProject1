/**
 * 脊柱侧弯视频康复训练
 * 实现视频指导下的姿势检测和匹配功能
 */

// 全局变量
let userStream = null;         // 用户摄像头流
let isTrainingActive = false;  // 训练活动状态
let detectionInterval = null;  // 检测间隔定时器
let videoSequence = [];        // 视频动作序列
let currentPoseIndex = 0;      // 当前动作索引
let poseHoldTime = 0;          // 姿势保持时间
let requiredHoldTime = 2;      // 所需保持时间(秒)
let lowQualityMode = false;    // 低质量模式
let matchConfidence = 0;       // 匹配置信度
let lastDetectionTime = 0;     // 上次检测时间
let detectionRate = 300;       // 检测频率(毫秒)
let videoSyncTimer = null;     // 视频同步定时器
let useVideoSync = true;       // 是否使用视频同步
let exerciseType = '';         // 当前训练类型
let completedExercises = 0;    // 已完成动作数
let totalScore = 0;            // 总得分
let startTime = null;          // 训练开始时间
let cameraFacing = 'user';     // 摄像头朝向，默认前置
let cameraCheckInterval = null; // 摄像头检查定时器
let debugMode = false;            // 调试模式
let debugUpdateInterval = null;   // 调试信息更新间隔
let lastFrameTimestamp = 0;       // 上一帧时间戳
let frameReceived = false;        // 是否接收到有效帧

// 页面加载时初始化
document.addEventListener('DOMContentLoaded', function() {
    // 检查是否支持WebRTC
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showErrorMessage('浏览器不支持', '您的浏览器不支持WebRTC，无法使用摄像头功能。请使用最新版本的Chrome、Firefox或Edge浏览器。');
        document.getElementById('cameraErrorMessage').style.display = 'flex';
        return;
    }
    
    // 获取URL参数中的训练类型
    const urlParams = new URLSearchParams(window.location.search);
    exerciseType = urlParams.get('type') || 'spine_stretch';
    
    // 根据设备性能调整设置
    // 如果是移动设备，自动启用低质量模式
    if (/Mobi|Android/i.test(navigator.userAgent)) {
        lowQualityMode = true;
        detectionRate = 500; // 降低检测频率
    }
    
    // 绑定按钮事件
    document.getElementById('startBtn').addEventListener('click', startTraining);
    document.getElementById('pauseBtn').addEventListener('click', pauseTraining);
    document.getElementById('skipBtn').addEventListener('click', skipCurrentPose);
    document.getElementById('nextBtn').addEventListener('click', moveToNextPose);
    
    // 加载训练数据
    loadTrainingData()
        .then(() => {
            // 预加载资源
            preloadResources();
            
            // 创建序列指示器
            createSequenceIndicators(videoSequence);
            
            // 设置摄像头
            setupCamera()
                .then(() => {
                    hideLoadingMessage();
                    document.getElementById('startBtn').disabled = false;
                    
                    // 启动摄像头状态检查
                    startCameraCheck();
                })
                .catch(error => {
                    console.error('摄像头初始化失败:', error);
                    showCameraError(error.message);
                });
        })
        .catch(error => {
            showErrorMessage('加载错误', `无法加载训练数据: ${error.message}`);
        });
    
    // 监听页面离开事件，释放资源
    window.addEventListener('beforeunload', function() {
        // 释放摄像头资源
        if (userStream) {
            userStream.getTracks().forEach(track => {
                track.stop();
            });
        }
        
        // 清除检测定时器
        if (detectionInterval) {
            clearInterval(detectionInterval);
        }
        
        // 清除视频同步定时器
        if (videoSyncTimer) {
            clearInterval(videoSyncTimer);
        }
    });
    
    // 添加调试按钮
    addDebugButton();
    
    // 监听URL参数，开启调试模式
    if (urlParams.has('debug')) {
        toggleDebugMode(true);
    }
    
    // 在页面加载1.5秒后检查摄像头是否工作
    setTimeout(function() {
        const video = document.getElementById('userVideo');
        if (!video || video.paused || video.videoWidth === 0) {
            console.warn('页面加载后摄像头检查失败');
            // 显示兼容性提示
            document.getElementById('compatibilityNotice').style.display = 'block';
        }
    }, 1500);
});

/**
 * 加载训练序列数据
 */
async function loadTrainingData() {
    showLoadingMessage('加载训练数据...');
    
    try {
        // 从API获取训练序列
        const response = await fetch(`/api/rehab/video_sequence?type=${exerciseType}`);
        
        if (!response.ok) {
            throw new Error('无法加载训练序列数据');
        }
        
        const data = await response.json();
        
        if (!data || !data.sequence || data.sequence.length === 0) {
            throw new Error('训练序列数据为空');
        }
        
        // 保存视频序列
        videoSequence = data.sequence;
        
        // 更新训练信息
        updateExerciseInfo(data.title, data.description);
        
        // 加载第一个姿势
        loadPose(0);
        
        return data;
    } catch (error) {
        console.error('加载训练数据出错:', error);
        throw error;
    }
}

/**
 * 预加载资源
 */
function preloadResources() {
    // 预加载视频
    videoSequence.forEach(pose => {
        const videoPreload = document.createElement('link');
        videoPreload.rel = 'preload';
        videoPreload.href = pose.video_url;
        videoPreload.as = 'video';
        document.head.appendChild(videoPreload);
    });
    
    // 预加载姿势图片
    videoSequence.forEach(pose => {
        const img = new Image();
        img.src = `/static/img/rehab/${pose.type}_guide.svg`;
    });
}

/**
 * 创建序列指示器
 */
function createSequenceIndicators(sequence) {
    const container = document.getElementById('sequenceContainer');
    container.innerHTML = '';
    
    sequence.forEach((pose, index) => {
        const item = document.createElement('div');
        item.className = 'sequence-item';
        
        const img = document.createElement('img');
        img.className = index === 0 ? 'sequence-image active' : 'sequence-image';
        img.src = `/static/img/rehab/${pose.type}_thumbnail.svg`;
        img.alt = pose.name;
        img.onerror = function() {
            this.src = '/static/img/rehab/placeholder.svg';
        };
        
        const label = document.createElement('div');
        label.className = 'sequence-label';
        label.textContent = pose.name;
        
        item.appendChild(img);
        item.appendChild(label);
        
        // 点击序列项切换到对应动作
        item.addEventListener('click', function() {
            if (isTrainingActive && index > currentPoseIndex) {
                alert('请先完成当前动作');
                return;
            }
            
            currentPoseIndex = index;
            loadPose(index);
            updateSequenceIndicators();
        });
        
        container.appendChild(item);
    });
}

/**
 * 更新训练信息
 */
function updateExerciseInfo(title, description) {
    document.getElementById('exerciseTitle').textContent = title;
    document.getElementById('exerciseDescription').textContent = description;
}

/**
 * 加载指定索引的姿势
 */
function loadPose(index) {
    if (index < 0 || index >= videoSequence.length) {
        console.error('无效的姿势索引:', index);
        return;
    }
    
    const pose = videoSequence[index];
    currentPoseIndex = index;
    
    // 更新UI显示
    document.getElementById('exerciseTitle').textContent = pose.name;
    
    // 加载视频
    loadVideo(pose.video_url);
    
    // 加载姿势指导图
    loadPoseGuide(pose.type);
    
    // 更新序列指示器
    updateSequenceIndicators();
    
    // 更新下一个姿势预览
    updateNextPosePreview();
    
    // 重置姿势匹配状态
    resetPoseMatching();
    
    // 禁用下一步按钮，直到完成当前姿势
    document.getElementById('nextBtn').disabled = true;
    
    // 更新进度
    updateProgress();
    
    // 更新状态消息
    if (isTrainingActive) {
        updateStatusMessage('准备开始新动作...', 'waiting');
    } else {
        updateStatusMessage('等待开始训练...', 'waiting');
    }
}

/**
 * 加载视频
 */
function loadVideo(videoUrl) {
    const video = document.getElementById('guideVideo');
    const loadingOverlay = document.getElementById('videoLoadingOverlay');
    
    // 显示加载中
    loadingOverlay.style.display = 'flex';
    
    // 设置视频源
    video.src = videoUrl;
    
    // 视频加载完成后隐藏加载中
    video.onloadeddata = function() {
        loadingOverlay.style.display = 'none';
        
        // 如果训练正在进行，自动播放视频
        if (isTrainingActive) {
            video.play();
            
            // 添加关键帧检测
            if (useVideoSync && currentPoseIndex < videoSequence.length) {
                const pose = videoSequence[currentPoseIndex];
                
                if (pose.key_frames && pose.key_frames.length > 0) {
                    if (videoSyncTimer) {
                        clearInterval(videoSyncTimer);
                    }
                    
                    // 开始视频同步
                    startVideoSync();
                }
            }
        }
    };
    
    // 视频出错处理
    video.onerror = function() {
        loadingOverlay.style.display = 'none';
        console.error('视频加载失败:', videoUrl);
        
        // 使用占位视频
        video.src = '/static/videos/rehab/placeholder.mp4';
    };
    
    // 视频播放结束后
    video.onended = function() {
        // 如果训练进行中，并且当前姿势的视频播放结束，提示用户完成
        if (isTrainingActive) {
            updateStatusMessage('视频播放完成，请继续保持正确姿势', 'waiting');
            
            // 如果没有匹配到姿势，允许用户手动进入下一个姿势
            if (matchConfidence < 70) {
                document.getElementById('nextBtn').disabled = false;
            }
        }
    };
}

/**
 * 加载姿势指导图
 */
async function loadPoseGuide(poseType) {
    const guideImage = document.getElementById('poseGuideImage');
    
    try {
        // 尝试从API获取参考姿势图
        const response = await fetch(`/api/rehab/reference_pose/${poseType}`);
        
        if (response.ok) {
            const data = await response.json();
            
            if (data && data.image_base64) {
                guideImage.src = data.image_base64;
                return;
            }
        }
        
        // 如果API失败，使用静态图片
        guideImage.src = `/static/img/rehab/${poseType}_guide.svg`;
        
        // 图片加载错误时使用占位图
        guideImage.onerror = function() {
            this.src = '/static/img/rehab/placeholder.svg';
        };
    } catch (error) {
        console.error('加载姿势指导图出错:', error);
        guideImage.src = '/static/img/rehab/placeholder.svg';
    }
}

/**
 * 更新下一个姿势预览
 */
function updateNextPosePreview() {
    const nextIndex = currentPoseIndex + 1;
    const nextPoseImage = document.getElementById('nextPoseImage');
    const nextPoseDescription = document.getElementById('nextPoseDescription');
    
    if (nextIndex < videoSequence.length) {
        const nextPose = videoSequence[nextIndex];
        nextPoseImage.src = `/static/img/rehab/${nextPose.type}_thumbnail.svg`;
        nextPoseDescription.textContent = `${nextPose.name}`;
        
        // 图片加载错误时使用占位图
        nextPoseImage.onerror = function() {
            this.src = '/static/img/rehab/placeholder.svg';
        };
    } else {
        // 无下一个姿势
        nextPoseImage.src = '/static/img/rehab/complete.svg';
        nextPoseDescription.textContent = '完成所有训练';
        
        // 图片加载错误时使用占位图
        nextPoseImage.onerror = function() {
            this.src = '/static/img/rehab/placeholder.svg';
        };
    }
}

/**
 * 更新序列指示器
 */
function updateSequenceIndicators() {
    const sequenceItems = document.querySelectorAll('.sequence-image');
    
    sequenceItems.forEach((item, index) => {
        // 移除所有类
        item.classList.remove('active', 'completed');
        
        // 添加适当的类
        if (index === currentPoseIndex) {
            item.classList.add('active');
        } else if (index < currentPoseIndex) {
            item.classList.add('completed');
        }
    });
}

/**
 * 更新进度
 */
function updateProgress() {
    const totalPoses = videoSequence.length;
    const completedPoses = Math.min(currentPoseIndex, totalPoses);
    const progressPercent = (completedPoses / totalPoses) * 100;
    
    document.getElementById('progressFill').style.width = `${progressPercent}%`;
    document.getElementById('progressText').textContent = `${Math.round(progressPercent)}%`;
}

/**
 * 重置姿势匹配状态
 */
function resetPoseMatching() {
    poseHoldTime = 0;
    matchConfidence = 0;
    
    // 更新UI
    updateConfidence(0);
    updateStatusMessage('准备好检测姿势...', 'waiting');
    
    // 禁用下一步按钮
    document.getElementById('nextBtn').disabled = true;
}

/**
 * 开始训练
 */
async function startTraining() {
    if (isTrainingActive) return;
    
    // 检查摄像头是否准备好
    if (!userStream) {
        try {
            await setupCamera();
        } catch (error) {
            showErrorMessage('摄像头错误', `无法访问摄像头: ${error.message}`);
            return;
        }
    }
    
    // 记录开始时间
    startTime = new Date();
    isTrainingActive = true;
    
    // 更新按钮状态
    document.getElementById('startBtn').disabled = true;
    document.getElementById('pauseBtn').disabled = false;
    document.getElementById('skipBtn').disabled = false;
    
    // 播放当前视频
    const video = document.getElementById('guideVideo');
    video.play();
    
    // 开始检测循环
    startDetectionLoop();
    
    // 如果有关键帧，开始视频同步
    if (useVideoSync && currentPoseIndex < videoSequence.length) {
        const pose = videoSequence[currentPoseIndex];
        
        if (pose.key_frames && pose.key_frames.length > 0) {
            startVideoSync();
        }
    }
    
    // 更新状态
    updateStatusMessage('训练开始，正在分析姿势...', 'waiting');
}

/**
 * 暂停训练
 */
function pauseTraining() {
    if (!isTrainingActive) return;
    
    isTrainingActive = false;
    
    // 暂停视频
    const video = document.getElementById('guideVideo');
    video.pause();
    
    // 停止检测循环
    stopDetectionLoop();
    
    // 停止视频同步
    if (videoSyncTimer) {
        clearInterval(videoSyncTimer);
        videoSyncTimer = null;
    }
    
    // 更新按钮状态
    document.getElementById('startBtn').disabled = false;
    document.getElementById('startBtn').textContent = '继续训练';
    document.getElementById('pauseBtn').disabled = true;
    
    // 更新状态
    updateStatusMessage('训练已暂停', 'waiting');
}

/**
 * 设置摄像头
 */
async function setupCamera() {
    const userVideo = document.getElementById('userVideo');
    const loadingOverlay = document.getElementById('cameraLoadingOverlay');
    const maxRetries = 3; // 最大重试次数
    let retryCount = 0;
    
    // 显示加载中
    loadingOverlay.style.display = 'flex';
    
    // 尝试激活摄像头按钮
    document.getElementById('cameraControls').style.display = 'flex';
    
    // 定义摄像头设置函数
    async function trySetupCamera() {
        try {
            // 如果已经有流，先停止
            if (userStream) {
                userStream.getTracks().forEach(track => track.stop());
                userStream = null;
            }
            
            // 更新加载信息
            document.querySelector('#cameraLoadingOverlay div:nth-child(2)').textContent = 
                `正在启动摄像头...(尝试 ${retryCount+1}/${maxRetries})`;
            
            console.log('开始请求摄像头权限...');
            // 显式请求摄像头权限，即使可能失败
            try {
                // 先尝试仅请求权限
                await navigator.mediaDevices.getUserMedia({video: true, audio: false});
                console.log('摄像头权限获取成功');
            } catch (permErr) {
                console.warn('权限请求失败:', permErr);
                // 继续尝试，下面的代码会尝试更多选项
            }
            
            // 设置多种可能的约束条件
            const constraints = {
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    frameRate: { ideal: 30 },
                    facingMode: cameraFacing
                },
                audio: false
            };
            
            // 移动设备优化
            if (/Mobi|Android/i.test(navigator.userAgent)) {
                constraints.video.width = { ideal: 320 };
                constraints.video.height = { ideal: 240 };
                constraints.video.frameRate = { ideal: 15 };
            }
            
            console.log('尝试获取摄像头流，配置:', constraints);
            
            // 准备多种约束条件方案
            const constraintOptions = [
                constraints, // 默认约束
                {video: {facingMode: cameraFacing}, audio: false}, // 简化约束
                {video: true, audio: false}, // 最简单约束
                {video: {width: 320, height: 240}, audio: false} // 低分辨率
            ];
            
            // 尝试枚举设备
            let cameraDevices = [];
            try {
                if (navigator.mediaDevices.enumerateDevices) {
                    const devices = await navigator.mediaDevices.enumerateDevices();
                    cameraDevices = devices.filter(device => device.kind === 'videoinput');
                    console.log('发现摄像头设备:', cameraDevices.length, cameraDevices);
                }
            } catch (enumErr) {
                console.warn('枚举设备失败:', enumErr);
            }
            
            // 错误对象，用于收集所有尝试的错误
            let lastError = null;
            
            // 1. 首先尝试使用特定设备ID (如果有)
            if (cameraDevices.length > 0) {
                for (const device of cameraDevices) {
                    try {
                        console.log('尝试使用设备:', device.label || device.deviceId);
                        userStream = await navigator.mediaDevices.getUserMedia({
                            video: {deviceId: {exact: device.deviceId}},
                            audio: false
                        });
                        console.log('设备使用成功:', device.label || device.deviceId);
                        
                        // 如果获取成功，跳出循环
                        break;
                    } catch (err) {
                        console.warn(`设备 ${device.label || device.deviceId} 使用失败:`, err);
                        lastError = err;
                        // 继续尝试下一个设备
                    }
                }
            }
            
            // 2. 如果特定设备尝试都失败，尝试不同的约束条件
            if (!userStream) {
                for (const option of constraintOptions) {
                    try {
                        console.log('尝试约束条件:', JSON.stringify(option));
                        userStream = await navigator.mediaDevices.getUserMedia(option);
                        console.log('约束条件成功:', JSON.stringify(option));
                        break;
                    } catch (err) {
                        console.warn('约束失败:', err);
                        lastError = err;
                    }
                }
            }
            
            // 如果所有尝试都失败，抛出最后一个错误
            if (!userStream) {
                throw lastError || new Error('无法获取摄像头流，所有尝试均失败');
            }
            
            // 检查是否真的获得了视频轨道
            const videoTracks = userStream.getVideoTracks();
            if (videoTracks.length === 0) {
                throw new Error('未能获取视频轨道');
            }
            
            console.log('获取到视频轨道:', videoTracks[0].label);
            
            // 将流设置到视频元素
            try {
                userVideo.srcObject = userStream;
                console.log('视频源对象设置成功');
            } catch (e) {
                console.warn('设置srcObject失败，尝试备选方法:', e);
                try {
                    // 旧浏览器的备选方法
                    userVideo.src = window.URL.createObjectURL(userStream);
                    console.log('通过URL创建成功');
                } catch (urlErr) {
                    console.error('URL创建失败:', urlErr);
                    // 继续尝试，就算这里失败了
                }
            }
            
            // 确保视频属性正确设置
            userVideo.muted = true;
            userVideo.playsInline = true;
            userVideo.autoplay = true;
            
            // 尝试播放视频
            try {
                console.log('尝试播放视频...');
                await userVideo.play();
                console.log('视频播放成功启动');
            } catch (playErr) {
                console.error('视频自动播放失败，可能需要用户交互:', playErr);
                
                // 创建一个播放按钮，提示用户点击
                const playButton = document.createElement('button');
                playButton.innerHTML = '<i class="fas fa-play"></i> 点击启动摄像头';
                playButton.style.position = 'absolute';
                playButton.style.top = '50%';
                playButton.style.left = '50%';
                playButton.style.transform = 'translate(-50%, -50%)';
                playButton.style.zIndex = '999';
                playButton.style.padding = '10px 20px';
                playButton.style.background = '#3498db';
                playButton.style.color = 'white';
                playButton.style.border = 'none';
                playButton.style.borderRadius = '5px';
                playButton.style.cursor = 'pointer';
                
                // 点击后尝试播放
                playButton.onclick = async function() {
                    try {
                        await userVideo.play();
                        this.parentNode.removeChild(this);
                    } catch (e) {
                        console.error('用户交互后播放仍失败:', e);
                        this.innerHTML = '播放失败，请点击选项按钮';
                    }
                };
                
                document.getElementById('cameraContainer').appendChild(playButton);
            }
            
            // 等待视频元数据加载
            await new Promise((resolve) => {
                if (userVideo.readyState >= 2) { // HAVE_CURRENT_DATA 或更高
                    resolve();
                } else {
                    userVideo.addEventListener('loadeddata', resolve);
                    
                    // 添加超时
                    setTimeout(resolve, 5000); // 5秒后无论如何都继续
                }
            });
            
            // 等待一小段时间确保视频真正开始
            await new Promise(resolve => setTimeout(resolve, 300));
            
            // 检查视频尺寸
            const videoWidth = userVideo.videoWidth;
            const videoHeight = userVideo.videoHeight;
            
            console.log('视频元素尺寸:', userVideo.offsetWidth, userVideo.offsetHeight);
            console.log('实际视频尺寸:', videoWidth, videoHeight);
            
            // 如果视频尺寸为0，尝试刷新视图
            if (videoWidth === 0 || videoHeight === 0) {
                console.warn('视频尺寸为0，尝试刷新视图');
                
                // 尝试刷新视频元素样式
                userVideo.style.display = 'none';
                // 强制浏览器重新计算布局
                void userVideo.offsetHeight;
                userVideo.style.display = 'block';
                
                // 再次检查尺寸
                await new Promise(resolve => setTimeout(resolve, 500));
                
                if (userVideo.videoWidth === 0 || userVideo.videoHeight === 0) {
                    console.warn('刷新后视频尺寸仍为0，尝试备选方案');
                    // 检查视频是否正在播放
                    if (userVideo.paused) {
                        try {
                            await userVideo.play();
                        } catch (e) {
                            console.error('刷新后视频播放失败:', e);
                        }
                    }
                }
            }
            
            // 配置画布
            const canvas = document.getElementById('poseCanvas');
            canvas.width = videoWidth || 640;
            canvas.height = videoHeight || 480;
            
            // 如果视频流正常，隐藏加载中提示
            loadingOverlay.style.display = 'none';
            
            // 检查视频是否真的显示了内容 (通过分析帧像素)
            checkVideoHasContent();
            
            return userStream;
        } catch (error) {
            console.error('摄像头设置错误:', error);
            
            // 如果还有重试次数，再试一次
            if (retryCount < maxRetries - 1) {
                retryCount++;
                document.querySelector('#cameraLoadingOverlay div:nth-child(2)').textContent = 
                    `摄像头失败，正在重试... (${retryCount}/${maxRetries})`;
                
                // 短暂延迟后重试
                await new Promise(resolve => setTimeout(resolve, 1000));
                return trySetupCamera();
            }
            
            // 所有重试都失败，显示错误和备选方案
            loadingOverlay.style.display = 'none';
            
            // 显示错误和备选选项
            showCameraError(error.message || '摄像头初始化失败');
            document.getElementById('emergencyOptions').style.display = 'block';
            
            // 尝试其他备选方案
            setTimeout(() => {
                // 先尝试服务器视频流
                tryConnectToVideoStream();
            }, 500);
            
            throw error;
        }
    }
    
    return trySetupCamera();
}

/**
 * 检查视频是否真的显示了内容
 */
function checkVideoHasContent() {
    const userVideo = document.getElementById('userVideo');
    
    if (!userVideo || userVideo.videoWidth === 0) {
        console.warn('视频元素未就绪，不能检测内容');
        return false;
    }
    
    try {
        // 创建一个临时画布来分析视频帧
        const tempCanvas = document.createElement('canvas');
        const ctx = tempCanvas.getContext('2d');
        const sampleSize = 10; // 分析的点数量
        
        tempCanvas.width = userVideo.videoWidth;
        tempCanvas.height = userVideo.videoHeight;
        
        // 从视频中获取一帧
        ctx.drawImage(userVideo, 0, 0, tempCanvas.width, tempCanvas.height);
        
        // 获取多个随机点的像素数据
        let totalBrightness = 0;
        const points = [];
        
        for (let i = 0; i < sampleSize; i++) {
            const x = Math.floor(Math.random() * tempCanvas.width);
            const y = Math.floor(Math.random() * tempCanvas.height);
            points.push({x, y});
        }
        
        // 分析像素
        for (const point of points) {
            const pixel = ctx.getImageData(point.x, point.y, 1, 1).data;
            const brightness = (pixel[0] + pixel[1] + pixel[2]) / 3;
            totalBrightness += brightness;
        }
        
        const avgBrightness = totalBrightness / sampleSize;
        console.log('视频内容分析 - 平均亮度:', avgBrightness);
        
        // 如果亮度过低（基本全黑），视频可能未正确显示
        if (avgBrightness < 5) {
            console.warn('视频显示检测 - 画面全黑!');
            
            // 尝试调整视频样式和位置
            userVideo.style.display = 'none';
            setTimeout(() => {
                userVideo.style.display = 'block';
                
                // 如果仍然全黑，尝试备选方案
                setTimeout(checkVideoAgain, 1000);
            }, 100);
            
            return false;
        }
        
        console.log('视频显示检测 - 成功');
        return true;
    } catch (e) {
        console.error('视频内容检测失败:', e);
        return false;
    }
}

/**
 * 再次检查视频显示
 */
function checkVideoAgain() {
    const userVideo = document.getElementById('userVideo');
    
    try {
        const tempCanvas = document.createElement('canvas');
        const ctx = tempCanvas.getContext('2d');
        
        tempCanvas.width = userVideo.videoWidth || 160;
        tempCanvas.height = userVideo.videoHeight || 120;
        
        ctx.drawImage(userVideo, 0, 0, tempCanvas.width, tempCanvas.height);
        const imageData = ctx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
        const data = imageData.data;
        
        // 简单计算平均亮度
        let sum = 0;
        for (let i = 0; i < data.length; i += 4) {
            sum += data[i] + data[i+1] + data[i+2];
        }
        const avg = sum / (data.length / 4) / 3;
        
        console.log('第二次视频检查 - 平均亮度:', avg);
        
        if (avg < 5) {
            console.warn('第二次检查仍然显示全黑，尝试备选方案');
            
            // 显示备选选项
            document.getElementById('emergencyOptions').style.display = 'block';
            
            // 自动尝试服务器视频流
            setTimeout(() => {
                tryConnectToVideoStream();
            }, 300);
        }
    } catch (e) {
        console.error('第二次视频检查失败:', e);
        // 显示备选选项
        document.getElementById('emergencyOptions').style.display = 'block';
    }
}

/**
 * 尝试连接到服务器视频流
 */
function tryConnectToVideoStream() {
    console.log('尝试连接服务器视频流...');
    document.getElementById('emergencyOptions').style.display = 'none';
    
    // 显示加载提示
    const loadingOverlay = document.getElementById('cameraLoadingOverlay');
    loadingOverlay.style.display = 'flex';
    loadingOverlay.querySelector('div:nth-child(2)').textContent = '正在连接服务器视频流...';
    
    // 创建或获取图像元素
    let streamImg = document.getElementById('streamImage');
    
    if (!streamImg) {
        streamImg = document.createElement('img');
        streamImg.id = 'streamImage';
        streamImg.style.position = 'absolute';
        streamImg.style.top = '0';
        streamImg.style.left = '0';
        streamImg.style.width = '100%';
        streamImg.style.height = '100%';
        streamImg.style.objectFit = 'cover';
        streamImg.style.zIndex = '8'; // 确保正确的层级顺序
        
        // 添加到容器
        document.getElementById('cameraContainer').appendChild(streamImg);
    }
    
    // 添加时间戳避免缓存问题
    const streamUrl = `/video/feed?t=${Date.now()}`;
    
    // 设置加载回调
    streamImg.onload = function() {
        console.log('服务器视频流连接成功');
        loadingOverlay.style.display = 'none';
        
        // 激活开始按钮
        document.getElementById('startBtn').disabled = false;
        
        // 高亮显示提示用户正在使用服务器流
        const notice = document.createElement('div');
        notice.innerHTML = '使用服务器视频流 <button onclick="retryCamera()" style="background: #3498db; color: white; border: none; padding: 3px 8px; border-radius: 3px; margin-left: 5px;">切换回摄像头</button>';
        notice.style.position = 'absolute';
        notice.style.bottom = '45px';
        notice.style.left = '0';
        notice.style.width = '100%';
        notice.style.textAlign = 'center';
        notice.style.background = 'rgba(0,0,0,0.7)';
        notice.style.color = 'white';
        notice.style.padding = '8px 0';
        notice.style.zIndex = '10';
        notice.id = 'streamNotice';
        
        // 移除可能已存在的通知
        const existingNotice = document.getElementById('streamNotice');
        if (existingNotice) {
            existingNotice.parentNode.removeChild(existingNotice);
        }
        
        document.getElementById('cameraContainer').appendChild(notice);
    };
    
    streamImg.onerror = function(e) {
        console.error('服务器视频流连接失败:', e);
        loadingOverlay.style.display = 'none';
        
        // 显示错误信息
        const errorMsg = document.getElementById('cameraErrorMessage');
        errorMsg.style.display = 'flex';
        errorMsg.querySelector('.error-title').textContent = '视频流错误';
        errorMsg.querySelector('.error-details').textContent = '无法连接到服务器视频流，请尝试使用示例视频';
        
        // 重新显示选项
        document.getElementById('emergencyOptions').style.display = 'block';
    };
    
    // 设置图像源
    streamImg.src = streamUrl;
}

/**
 * 使用示例视频作为备选
 */
function useExampleVideo() {
    console.log('使用示例视频作为备选...');
    
    // 隐藏错误信息和选项
    document.getElementById('cameraErrorMessage').style.display = 'none';
    document.getElementById('emergencyOptions').style.display = 'none';
    
    // 显示加载提示
    const loadingOverlay = document.getElementById('cameraLoadingOverlay');
    loadingOverlay.style.display = 'flex';
    loadingOverlay.querySelector('div:nth-child(2)').textContent = '正在加载示例视频...';
    
    // 停止当前摄像头流
    if (userStream) {
        userStream.getTracks().forEach(track => track.stop());
        userStream = null;
    }
    
    // 获取或创建视频元素
    const userVideo = document.getElementById('userVideo');
    
    // 停止显示可能的服务器流图像
    const streamImg = document.getElementById('streamImage');
    if (streamImg) {
        streamImg.style.display = 'none';
    }
    
    // 确保视频显示
    userVideo.style.display = 'block';
    
    // 清除当前视频源
    userVideo.srcObject = null;
    
    // 设置示例视频作为源
    // 这里我们提供多个视频备选，如果一个加载失败可以尝试下一个
    const exampleVideos = [
        '/static/video/rehab/example_pose.mp4',
        '/static/video/example.mp4',
        '/static/example.mp4'
    ];
    
    let videoIndex = 0;
    
    // 尝试加载并播放视频
    function tryLoadVideo() {
        if (videoIndex >= exampleVideos.length) {
            // 所有视频都尝试失败
            loadingOverlay.style.display = 'none';
            showCameraError('无法加载任何示例视频');
            document.getElementById('emergencyOptions').style.display = 'block';
            return;
        }
        
        const videoSrc = exampleVideos[videoIndex];
        userVideo.src = videoSrc;
        
        // 设置视频属性
        userVideo.loop = true;
        userVideo.muted = true;
        userVideo.controls = false;
        
        // 视频加载成功处理
        userVideo.onloadeddata = function() {
            userVideo.play()
                .then(() => {
                    console.log('示例视频播放成功');
                    loadingOverlay.style.display = 'none';
                    
                    // 启用开始按钮
                    document.getElementById('startBtn').disabled = false;
                    
                    // 显示示例视频提示
                    const notice = document.createElement('div');
                    notice.innerHTML = '使用示例视频 <button onclick="retryCamera()" style="background: #3498db; color: white; border: none; padding: 3px 8px; border-radius: 3px; margin-left: 5px;">重试摄像头</button>';
                    notice.style.position = 'absolute';
                    notice.style.bottom = '45px';
                    notice.style.left = '0';
                    notice.style.width = '100%';
                    notice.style.textAlign = 'center';
                    notice.style.background = 'rgba(0,0,0,0.7)';
                    notice.style.color = 'white';
                    notice.style.padding = '8px 0';
                    notice.style.zIndex = '10';
                    notice.id = 'exampleNotice';
                    
                    // 移除可能已存在的通知
                    const existingNotice = document.getElementById('exampleNotice');
                    if (existingNotice) {
                        existingNotice.parentNode.removeChild(existingNotice);
                    }
                    
                    document.getElementById('cameraContainer').appendChild(notice);
                })
                .catch(err => {
                    console.error('示例视频播放失败:', err);
                    videoIndex++;
                    tryLoadVideo(); // 尝试下一个视频
                });
        };
        
        // 视频加载失败处理
        userVideo.onerror = function() {
            console.error('示例视频加载失败:', videoSrc);
            videoIndex++;
            tryLoadVideo(); // 尝试下一个视频
        };
    }
    
    // 开始尝试加载视频
    tryLoadVideo();
}

/**
 * 重试启动摄像头
 */
function retryCamera() {
    console.log('重试启动摄像头...');
    
    // 隐藏错误信息和选项
    document.getElementById('cameraErrorMessage').style.display = 'none';
    document.getElementById('emergencyOptions').style.display = 'none';
    
    // 移除通知提示
    const notices = ['streamNotice', 'exampleNotice'];
    notices.forEach(id => {
        const notice = document.getElementById(id);
        if (notice) notice.parentNode.removeChild(notice);
    });
    
    // 移除流图像
    const streamImg = document.getElementById('streamImage');
    if (streamImg) {
        streamImg.style.display = 'none';
    }
    
    // 显示加载提示
    const loadingOverlay = document.getElementById('cameraLoadingOverlay');
    loadingOverlay.style.display = 'flex';
    loadingOverlay.innerHTML = `
        <div class="spinner"></div>
        <div>重新启动摄像头...</div>
    `;
    
    // 重置视频元素
    const userVideo = document.getElementById('userVideo');
    userVideo.style.display = 'block';
    userVideo.srcObject = null;
    userVideo.src = '';
    userVideo.muted = true;
    
    // 停止当前流
    if (userStream) {
        userStream.getTracks().forEach(track => track.stop());
        userStream = null;
    }
    
    // 重新初始化摄像头
    setupCamera()
        .then(() => {
            // 成功启动
            loadingOverlay.style.display = 'none';
            document.getElementById('startBtn').disabled = false;
        })
        .catch(error => {
            console.error('摄像头重试失败:', error);
            // 错误处理由setupCamera函数内部完成
        });
}

/**
 * 开始检测循环
 */
function startDetectionLoop() {
    // 清除之前的循环
    if (detectionInterval) {
        clearInterval(detectionInterval);
    }
    
    // 设置新循环前检查视频显示
    checkVideoHasContent();
    
    // 设置新循环
    lastDetectionTime = Date.now();
    detectionInterval = setInterval(detectPose, detectionRate);
}

/**
 * 停止检测循环
 */
function stopDetectionLoop() {
    if (detectionInterval) {
        clearInterval(detectionInterval);
        detectionInterval = null;
    }
}

/**
 * 开始视频同步
 */
function startVideoSync() {
    if (videoSyncTimer) {
        clearInterval(videoSyncTimer);
    }
    
    const video = document.getElementById('guideVideo');
    const pose = videoSequence[currentPoseIndex];
    
    // 检查是否有关键帧
    if (!pose.key_frames || pose.key_frames.length === 0) {
        return;
    }
    
    videoSyncTimer = setInterval(() => {
        // 检查视频是否在播放
        if (video.paused || !isTrainingActive) {
            return;
        }
        
        // 检查当前视频时间是否匹配任何关键帧
        const currentTime = video.currentTime;
        
        // 查找最接近的关键帧
        for (const keyFrame of pose.key_frames) {
            // 时间差在0.5秒内视为匹配
            if (Math.abs(currentTime - keyFrame.time) < 0.5) {
                // 在UI上显示提示
                updateStatusMessage(`提示: ${keyFrame.hint}`, 'waiting');
                
                // 触发姿势检测
                detectPose(keyFrame.pose_type, keyFrame.hint);
                break;
            }
        }
    }, 200); // 每200毫秒检查一次
}

/**
 * 检测姿势
 */
async function detectPose(specificPoseType, frameHint) {
    // 如果训练未激活，不检测
    if (!isTrainingActive) return;
    
    // 计算距离上次检测的时间
    const now = Date.now();
    const elapsed = now - lastDetectionTime;
    
    // 如果时间太短，跳过这次检测
    if (elapsed < detectionRate * 0.8 && !specificPoseType) {
        return;
    }
    
    lastDetectionTime = now;
    
    try {
        // 获取当前姿势信息
        const currentPose = videoSequence[currentPoseIndex];
        
        // 使用特定姿势类型或当前姿势类型
        const poseType = specificPoseType || currentPose.type;
        
        // 获取摄像头画面
        const video = document.getElementById('userVideo');
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        
        // 设置画布尺寸匹配视频
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        // 绘制摄像头画面到画布
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // 获取图像数据
        const imageDataUrl = canvas.toDataURL('image/jpeg', lowQualityMode ? 0.5 : 0.7);
        
        // 转换为Blob
        const blob = await (await fetch(imageDataUrl)).blob();
        
        // 创建FormData
        const formData = new FormData();
        formData.append('image', blob, 'camera.jpg');
        formData.append('pose_type', poseType);
        
        // 如果有帧提示，添加到请求中
        if (frameHint) {
            formData.append('frame_hints', frameHint);
        }
        
        // 添加低质量模式标志
        formData.append('low_quality', lowQualityMode.toString());
        
        // 发送到API
        const response = await fetch('/api/rehab/detect_video_pose', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('检测姿势API请求失败');
        }
        
        // 处理响应
        const result = await response.json();
        processDetectionResult(result);
        
    } catch (error) {
        console.error('检测姿势出错:', error);
        updateStatusMessage('姿势检测失败，请重试', 'incorrect');
    }
}

/**
 * 处理检测结果
 */
function processDetectionResult(result) {
    // 如果训练未激活，不处理
    if (!isTrainingActive) return;
    
    // 检查结果状态
    if (result.status === 'ERROR') {
        updateStatusMessage('姿势检测错误: ' + result.message, 'incorrect');
        return;
    }
    
    // 如果未检测到姿势
    if (result.status === 'NOT_DETECTED') {
        updateStatusMessage(result.feedback || '未检测到姿势，请确保完全在画面中', 'incorrect');
        matchConfidence = 0;
        updateConfidence(0);
        return;
    }
    
    // 绘制检测结果
    if (result.annotated_image_base64) {
        drawPoseResult(result.annotated_image_base64);
    }
    
    // 更新匹配置信度
    matchConfidence = result.score;
    updateConfidence(matchConfidence);
    
    // 更新状态消息
    let statusClass = 'waiting';
    
    switch (result.status) {
        case 'CORRECT':
            statusClass = 'correct';
            break;
        case 'ALMOST':
            statusClass = 'almost';
            break;
        case 'INCORRECT':
            statusClass = 'incorrect';
            break;
    }
    
    updateStatusMessage(result.feedback, statusClass);
    
    // 处理匹配结果
    if (result.matches) {
        // 增加保持时间
        poseHoldTime += detectionRate / 1000;
        
        // 如果保持足够时间，完成当前姿势
        if (poseHoldTime >= requiredHoldTime) {
            completePose();
        }
    } else {
        // 重置保持时间
        poseHoldTime = 0;
    }
}

/**
 * 绘制姿势检测结果
 */
function drawPoseResult(base64Data) {
    const canvas = document.getElementById('poseCanvas');
    const ctx = canvas.getContext('2d');
    
    // 清除画布
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // 加载图像
    const img = new Image();
    img.onload = function() {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    };
    
    img.src = base64Data;
}

/**
 * 更新置信度显示
 */
function updateConfidence(confidence) {
    // 确保在0-100范围内
    confidence = Math.max(0, Math.min(100, confidence));
    
    // 更新UI
    document.getElementById('confidenceFill').style.width = `${confidence}%`;
    document.getElementById('confidenceText').textContent = `匹配度: ${Math.round(confidence)}%`;
    
    // 根据置信度更改颜色
    const fill = document.getElementById('confidenceFill');
    
    if (confidence < 50) {
        fill.style.backgroundColor = '#e74c3c'; // 红色
    } else if (confidence < 75) {
        fill.style.backgroundColor = '#f39c12'; // 橙色
    } else {
        fill.style.backgroundColor = '#2ecc71'; // 绿色
    }
}

/**
 * 更新状态消息
 */
function updateStatusMessage(message, status) {
    const statusMessage = document.getElementById('statusMessage');
    
    // 更新消息
    statusMessage.textContent = message;
    
    // 更新样式
    statusMessage.className = 'status-message';
    
    if (status) {
        statusMessage.classList.add(`status-${status}`);
    }
}

/**
 * 完成当前姿势
 */
function completePose() {
    // 只处理一次
    if (poseHoldTime < requiredHoldTime) return;
    
    // 记录分数
    totalScore += matchConfidence;
    completedExercises++;
    
    // 标记当前完成
    poseHoldTime = requiredHoldTime + 1;
    
    // 显示完成消息
    showCompletionMessage();
    
    // 启用下一步按钮
    document.getElementById('nextBtn').disabled = false;
    
    // 更新状态
    updateStatusMessage('姿势正确！可以进入下一个动作', 'correct');
    
    // 如果是最后一个姿势，显示完成按钮
    if (currentPoseIndex >= videoSequence.length - 1) {
        document.getElementById('nextBtn').textContent = '完成训练';
    }
}

/**
 * 显示完成消息
 */
function showCompletionMessage() {
    const message = document.getElementById('completionMessage');
    message.classList.add('show');
    
    // 3秒后隐藏
    setTimeout(() => {
        message.classList.remove('show');
    }, 3000);
}

/**
 * 跳过当前姿势
 */
function skipCurrentPose() {
    if (!isTrainingActive) return;
    
    moveToNextPose();
}

/**
 * 移动到下一个姿势
 */
function moveToNextPose() {
    // 移动到下一个
    const nextIndex = currentPoseIndex + 1;
    
    // 检查是否还有下一个姿势
    if (nextIndex < videoSequence.length) {
        // 加载下一个姿势
        loadPose(nextIndex);
        
        // 如果训练进行中，自动播放视频
        if (isTrainingActive) {
            const video = document.getElementById('guideVideo');
            video.play();
            
            // 如果有关键帧，开始视频同步
            if (useVideoSync) {
                const pose = videoSequence[nextIndex];
                
                if (pose.key_frames && pose.key_frames.length > 0) {
                    startVideoSync();
                }
            }
        }
    } else {
        // 完成所有训练
        completeTraining();
    }
}

/**
 * 完成训练
 */
function completeTraining() {
    // 停止训练
    isTrainingActive = false;
    
    // 停止检测循环
    stopDetectionLoop();
    
    // 停止视频同步
    if (videoSyncTimer) {
        clearInterval(videoSyncTimer);
    }
    
    // 计算训练时间
    const endTime = new Date();
    const trainingDuration = Math.floor((endTime - startTime) / 1000); // 秒
    
    // 计算平均分数
    const avgScore = completedExercises > 0 ? Math.round(totalScore / completedExercises) : 0;
    
    // 跳转到完成页面
    window.location.href = `/rehab/video_training/complete?exercises=${completedExercises}&duration=${trainingDuration}&score=${avgScore}`;
}

/**
 * 显示加载消息
 */
function showLoadingMessage(message) {
    // 视频加载中
    const videoLoading = document.getElementById('videoLoadingOverlay');
    videoLoading.querySelector('div:last-child').textContent = message;
    videoLoading.style.display = 'flex';
    
    // 摄像头加载中
    const cameraLoading = document.getElementById('cameraLoadingOverlay');
    cameraLoading.querySelector('div:last-child').textContent = message;
    cameraLoading.style.display = 'flex';
}

/**
 * 隐藏加载消息
 */
function hideLoadingMessage() {
    document.getElementById('videoLoadingOverlay').style.display = 'none';
    document.getElementById('cameraLoadingOverlay').style.display = 'none';
}

/**
 * 显示错误消息
 */
function showErrorMessage(title, message) {
    alert(`${title}: ${message}`);
    console.error(title, message);
    
    // 在状态区显示错误
    updateStatusMessage(`错误: ${message}`, 'incorrect');
}

/**
 * 开始摄像头状态检查
 */
function startCameraCheck() {
    if (cameraCheckInterval) {
        clearInterval(cameraCheckInterval);
    }
    
    // 定期检查摄像头状态
    cameraCheckInterval = setInterval(async function() {
        try {
            // 通过API检查摄像头状态
            const response = await fetch('/video/status');
            if (response.ok) {
                const data = await response.json();
                if (data.status === 'error') {
                    console.warn('摄像头状态异常:', data.error_message);
                    showCameraError(data.error_message);
                }
            }
            
            // 检查视频元素是否正在显示画面
            const video = document.getElementById('userVideo');
            if (video && (video.readyState === 0 || video.videoWidth === 0)) {
                // 视频元素没有接收到有效画面
                console.warn('摄像头画面无效');
                // 如果训练已经开始，尝试恢复摄像头
                if (isTrainingActive && !document.getElementById('cameraErrorMessage').style.display === 'flex') {
                    showCameraError('摄像头画面丢失，请尝试重新连接');
                }
            }
        } catch (error) {
            console.error('摄像头状态检查错误:', error);
        }
    }, 5000);  // 每5秒检查一次
}

/**
 * 切换前后摄像头
 */
function flipCamera() {
    console.log('切换摄像头方向...');
    
    // 切换方向
    cameraFacing = cameraFacing === 'user' ? 'environment' : 'user';
    
    // 显示加载提示
    const loadingOverlay = document.getElementById('cameraLoadingOverlay');
    loadingOverlay.style.display = 'flex';
    loadingOverlay.querySelector('div:nth-child(2)').textContent = '正在切换摄像头...';
    
    // 停止当前流
    if (userStream) {
        userStream.getTracks().forEach(track => track.stop());
    }
    
    // 确保视频元素显示
    const userVideo = document.getElementById('userVideo');
    userVideo.style.display = 'block';
    
    // 隐藏可能的流图像
    const streamImg = document.getElementById('streamImage');
    if (streamImg) {
        streamImg.style.display = 'none';
    }
    
    // 重新初始化摄像头
    setupCamera()
        .then(() => {
            console.log('摄像头切换成功');
            // 成功处理由setupCamera完成
        })
        .catch(error => {
            console.error('摄像头切换失败:', error);
            // 错误处理由setupCamera完成
        });
}

/**
 * 放大摄像头视图
 */
function expandCameraView() {
    const cameraContainer = document.querySelector('.camera-container');
    cameraContainer.classList.toggle('expanded');
    
    const expandBtn = document.getElementById('expandCameraBtn');
    if (cameraContainer.classList.contains('expanded')) {
        expandBtn.innerHTML = '<i class="fas fa-compress"></i>';
        expandBtn.title = '缩小视图';
    } else {
        expandBtn.innerHTML = '<i class="fas fa-expand"></i>';
        expandBtn.title = '放大视图';
    }
}

/**
 * 添加调试按钮
 */
function addDebugButton() {
    // 检查是否已存在
    if (document.getElementById('debugBtn')) {
        return;
    }
    
    // 创建调试按钮
    const debugBtn = document.createElement('button');
    debugBtn.id = 'debugBtn';
    debugBtn.className = 'debug-button';
    debugBtn.innerHTML = '<i class="fas fa-bug"></i>';
    debugBtn.title = '调试模式';
    debugBtn.onclick = function() {
        toggleDebugMode(!debugMode);
    };
    
    // 添加到页面
    document.body.appendChild(debugBtn);
    
    // 创建调试信息面板
    const debugInfo = document.createElement('div');
    debugInfo.id = 'debugInfo';
    debugInfo.className = 'debug-info';
    debugInfo.style.display = 'none';
    debugInfo.innerHTML = '调试信息加载中...';
    
    // 添加到页面
    document.body.appendChild(debugInfo);
}

/**
 * 切换调试模式
 */
function toggleDebugMode(enable) {
    debugMode = enable;
    const debugInfo = document.getElementById('debugInfo');
    
    if (debugMode) {
        debugInfo.style.display = 'block';
        // 启动调试信息更新
        if (!debugUpdateInterval) {
            debugUpdateInterval = setInterval(updateDebugInfo, 1000);
            updateDebugInfo(); // 立即更新一次
        }
    } else {
        debugInfo.style.display = 'none';
        // 停止调试信息更新
        if (debugUpdateInterval) {
            clearInterval(debugUpdateInterval);
            debugUpdateInterval = null;
        }
    }
}

/**
 * 更新调试信息
 */
function updateDebugInfo() {
    const debugInfo = document.getElementById('debugInfo');
    if (!debugInfo) return;
    
    const video = document.getElementById('userVideo');
    const streamImg = document.getElementById('streamImage');
    
    // 收集调试信息
    const info = {
        '时间': new Date().toLocaleTimeString(),
        '浏览器': navigator.userAgent.match(/chrome|firefox|safari|edge|opera/i) || ['未知'],
        '设备': /Mobile|Android|iPhone/i.test(navigator.userAgent) ? '移动设备' : '桌面设备',
        '训练状态': isTrainingActive ? '进行中' : '未开始',
        '视频状态': video ? (video.paused ? '已暂停' : '播放中') : '未加载',
        '视频尺寸': video ? `${video.videoWidth}x${video.videoHeight}` : '无',
        '流媒体': streamImg ? (streamImg.complete ? '已加载' : '加载中') : '未使用',
        'WebRTC': navigator.mediaDevices ? '支持' : '不支持',
        '摄像头': userStream ? `${userStream.getVideoTracks().length}个轨道` : '未初始化',
        '朝向': cameraFacing,
        'FPS': calculateFPS(),
        '检测频率': `${detectionRate}ms`,
        '低质量模式': lowQualityMode ? '开启' : '关闭',
        '匹配置信度': `${Math.round(matchConfidence)}%`
    };
    
    // 格式化为HTML
    let html = '<strong>调试信息</strong><br>';
    for (const [key, value] of Object.entries(info)) {
        html += `<span style="color:#aaa">${key}:</span> ${value}<br>`;
    }
    
    // 添加快捷操作按钮
    html += '<div style="margin-top:10px">';
    html += '<button onclick="flipCamera()" style="margin-right:5px;padding:2px 5px;font-size:10px;">切换摄像头</button>';
    html += '<button onclick="tryConnectToVideoStream()" style="margin-right:5px;padding:2px 5px;font-size:10px;">使用流媒体</button>';
    html += '<button onclick="useExampleVideo()" style="padding:2px 5px;font-size:10px;">使用示例</button>';
    html += '</div>';
    
    // 更新内容
    debugInfo.innerHTML = html;
}

/**
 * 计算当前FPS
 */
function calculateFPS() {
    const now = performance.now();
    const video = document.getElementById('userVideo');
    
    // 检测是否接收到新帧
    if (video && !video.paused && video.videoWidth > 0) {
        // 在调试模式下，绘制一个帧来检测
        if (debugMode) {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = 2;
            canvas.height = 2;
            
            try {
                ctx.drawImage(video, 0, 0, 2, 2);
                const pixels = ctx.getImageData(0, 0, 2, 2).data;
                const sum = pixels.reduce((a, b) => a + b, 0);
                
                // 如果上一帧和这一帧相同，可能视频静止
                if (lastFrameTimestamp > 0 && now - lastFrameTimestamp > 500 && !frameReceived) {
                    return '0 (已停止)';
                }
                
                frameReceived = true;
                lastFrameTimestamp = now;
            } catch (e) {
                console.error('FPS计算出错:', e);
                return '错误';
            }
        }
        
        const elapsed = now - lastFrameTimestamp;
        if (elapsed > 0 && elapsed < 1000) {
            return Math.round(1000 / elapsed);
        }
    }
    
    return '未知';
}

/**
 * 显示摄像头错误信息
 */
function showCameraError(message) {
    console.warn('显示摄像头错误:', message);
    
    // 隐藏加载中提示
    document.getElementById('cameraLoadingOverlay').style.display = 'none';
    
    // 禁用开始按钮
    document.getElementById('startBtn').disabled = true;
    
    // 显示错误信息
    const errorMsg = document.getElementById('cameraErrorMessage');
    if (!errorMsg) {
        console.error('找不到错误信息元素');
        return;
    }
    
    // 设置错误内容
    const errorDetails = errorMsg.querySelector('.error-details');
    if (errorDetails) {
        errorDetails.textContent = message || '无法访问摄像头，请检查设备连接和浏览器权限';
    }
    
    // 显示错误信息
    errorMsg.style.display = 'flex';
    
    // 确保元素可见
    errorMsg.style.zIndex = '999';
    
    // 30秒后自动显示备选方案
    setTimeout(function() {
        if (errorMsg.style.display === 'flex') {
            document.getElementById('emergencyOptions').style.display = 'block';
        }
    }, 30000);
} 