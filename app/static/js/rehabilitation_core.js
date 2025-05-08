/**
 * 康复指导系统核心功能
 * 负责摄像头处理、后端通信和姿态数据管理
 */

// 全局变量
let videoStream = null;
let backendSessionStarted = false;
let keypointsPollingInterval = null;
let lastKeypointsUpdate = null;
let availableTemplates = [];

// 康复核心API
window.RehabCore = {
    initializeCamera,
    stopCamera,
    startBackendSession,
    stopBackendSession,
    startKeyPointsPolling,
    stopKeyPointsPolling,
    fetchAvailableTemplates,
    changeRehabTemplate,
    get backendActive() {
        return backendSessionStarted;
    },
    get lastPoseData() {
        return lastKeypointsUpdate;
    }
};

/**
 * 初始化摄像头
 * @param {number} cameraId - 可选的摄像头ID
 * @returns {Promise<boolean>} 初始化结果
 */
async function initializeCamera(cameraId) {
    console.log('正在初始化摄像头...');
    
    if (videoStream) {
        console.log('摄像头已经初始化，先停止现有摄像头');
        stopCamera();
    }
    
    const userVideo = document.getElementById('userVideo');
    const cameraHint = document.getElementById('cameraHint');
    
    if (!userVideo) {
        throw new Error('找不到视频元素');
    }
    
    try {
        // 检查是否有可用摄像头
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');
        
        if (videoDevices.length === 0) {
            throw new Error('未找到可用摄像头设备');
        }
        
        console.log(`找到${videoDevices.length}个摄像头设备`);
        
        // 确定使用哪个摄像头
        let deviceId = undefined;
        if (cameraId !== undefined && videoDevices[cameraId]) {
            deviceId = videoDevices[cameraId].deviceId;
            console.log(`选择指定摄像头 ID: ${cameraId}, deviceId: ${deviceId}`);
        }
        
        // 尝试不同的摄像头约束条件
        const constraints = [
            { 
                video: { 
                    width: { ideal: 640 }, 
                    height: { ideal: 480 },
                    facingMode: 'user',
                    deviceId: deviceId ? { exact: deviceId } : undefined
                } 
            },
            { 
                video: { 
                    width: { ideal: 1280 }, 
                    height: { ideal: 720 },
                    facingMode: 'user',
                    deviceId: deviceId ? { exact: deviceId } : undefined
                } 
            },
            { video: deviceId ? { deviceId: { exact: deviceId } } : true } // 最基础的约束
        ];
        
        let lastError = null;
        
        // 逐个尝试不同的约束，直到一个成功
        for (const constraint of constraints) {
            try {
                console.log(`尝试使用约束: ${JSON.stringify(constraint)}`);
                videoStream = await navigator.mediaDevices.getUserMedia(constraint);
                
                if (videoStream) {
                    console.log('摄像头初始化成功');
                    userVideo.srcObject = videoStream;
                    
                    // 确保视频可以自动播放
                    userVideo.play().catch(e => {
                        console.warn('自动播放失败，需要用户交互:', e);
                        
                        // 在用户第一次点击页面时尝试播放
                        document.addEventListener('click', () => {
                            userVideo.play().catch(e => console.warn("用户交互播放失败:", e));
                        }, { once: true });
                    });
                    
                    // 设置canvas
                    const canvas = document.getElementById('poseCanvas');
                    if (canvas) {
                        canvas.width = userVideo.videoWidth || 640;
                        canvas.height = userVideo.videoHeight || 480;
                        
                        // 等待真实视频尺寸加载后调整
                        userVideo.onloadedmetadata = () => {
                            if (userVideo.videoWidth && userVideo.videoHeight) {
                                canvas.width = userVideo.videoWidth;
                                canvas.height = userVideo.videoHeight;
                                console.log(`调整canvas尺寸为: ${canvas.width}x${canvas.height}`);
                            }
                        };
                    }
                    
                    if (cameraHint) {
                        cameraHint.textContent = '摄像头已连接，正在检测姿势...';
                        setTimeout(() => {
                            cameraHint.style.opacity = '0.5';
                        }, 2000);
                    }
                    
                    return true;
                }
            } catch (err) {
                console.warn(`使用约束 ${JSON.stringify(constraint)} 失败:`, err);
                lastError = err;
                continue;
            }
        }
        
        // 所有约束都失败，抛出最后一个错误
        throw lastError || new Error('无法初始化摄像头');
        
    } catch (error) {
        console.error("摄像头初始化错误:", error);
        
        if (cameraHint) {
            cameraHint.textContent = `摄像头错误: ${error.message}`;
            cameraHint.style.opacity = '1';
        }
        
        // 根据错误类型提供更具体的错误信息
        if (error.name === 'NotAllowedError') {
            throw new Error('摄像头访问被拒绝，请在浏览器设置中允许访问摄像头');
        } else if (error.name === 'NotFoundError') {
            throw new Error('未找到可用摄像头设备，请确保摄像头已连接');
        } else if (error.name === 'NotReadableError') {
            throw new Error('摄像头可能被其他程序占用，请关闭其他使用摄像头的应用后重试');
        }
        
        throw error;
    }
}

/**
 * 停止摄像头
 */
function stopCamera() {
    if (videoStream) {
        console.log('停止摄像头流');
        videoStream.getTracks().forEach(track => {
            track.stop();
        });
        videoStream = null;
        
        // 清理视频元素
        const userVideo = document.getElementById('userVideo');
        if (userVideo) {
            userVideo.srcObject = null;
            userVideo.load();
        }
    }
}

/**
 * 启动后端康复会话
 * @returns {Promise<object>} 会话响应数据
 */
async function startBackendSession() {
    const apiPath = '/api/rehab/start';
    console.log(`正在调用API启动康复会话: ${apiPath}`);
    
    try {
        // 先检查API是否可用
        const checkResponse = await fetch('/api/rehab/templates', { method: 'GET' });
        if (!checkResponse.ok) {
            console.error(`API检查失败: ${checkResponse.status}`);
            throw new Error(`康复API不可用，请确保系统正常运行`);
        }
        
        const response = await fetch(apiPath, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            console.error(`API返回错误: ${response.status}`, errorData);
            throw new Error(errorData.message || `服务器返回错误: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('API响应数据:', data);
        
        if (data.status !== 'success') {
            throw new Error(data.message || '后端启动康复会话失败');
        }
        
        console.log('后端康复会话启动成功:', data);
        backendSessionStarted = true;
        
        return data;
    } catch (error) {
        console.error('启动康复会话出错:', error);
        throw error;
    }
}

/**
 * 停止后端康复会话
 * @returns {Promise<object>} 响应数据
 */
async function stopBackendSession() {
    if (!backendSessionStarted) {
        return { status: 'not_started' };
    }
    
    const apiPath = '/api/rehab/stop';
    console.log(`正在调用API停止康复会话: ${apiPath}`);
    
    try {
        const response = await fetch(apiPath, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            console.error(`API返回错误: ${response.status}`, errorData);
            throw new Error(errorData.message || `服务器返回错误: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('API响应数据:', data);
        
        backendSessionStarted = false;
        
        return data;
    } catch (error) {
        console.error('停止康复会话出错:', error);
        backendSessionStarted = false;
        return { status: 'error', message: error.message };
    }
}

/**
 * 开始轮询关键点数据
 * @param {Function} onKeypointsUpdate - 关键点更新回调
 */
function startKeyPointsPolling(onKeypointsUpdate) {
    if (keypointsPollingInterval) {
        console.log('关键点轮询已经在运行中');
        return;
    }
    
    // 添加到window.RehabCore对象
    window.RehabCore.onKeypointsUpdateCallback = onKeypointsUpdate;
    
    // 创建临时Canvas用于处理视频帧
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    
    // 获取视频元素
    const userVideo = document.getElementById('userVideo');
    
    if (!userVideo) {
        console.error('未找到视频元素，无法开始关键点轮询');
        return;
    }
    
    // 轮询流程
    keypointsPollingInterval = setInterval(async () => {
        if (!backendSessionStarted || !userVideo.srcObject) {
            return;
        }
        
        try {
            // 视频就绪，可以截取图像
            if (userVideo.readyState >= 2) {
                // 调整临时画布大小以匹配视频尺寸
                if (tempCanvas.width !== userVideo.videoWidth || tempCanvas.height !== userVideo.videoHeight) {
                    tempCanvas.width = userVideo.videoWidth;
                    tempCanvas.height = userVideo.videoHeight;
                }
                
                // 截取当前视频帧到临时画布
                tempCtx.drawImage(userVideo, 0, 0, tempCanvas.width, tempCanvas.height);
                
                // 转换为Blob数据准备发送
                tempCanvas.toBlob(async (blob) => {
                    if (!blob) {
                        console.error('无法创建图像Blob');
                        return;
                    }
                    
                    // 构建表单数据
                    const formData = new FormData();
                    formData.append('image', blob, 'frame.jpg');
                    
                    // 获取当前选择的姿势模板
                    const templateSelector = document.getElementById('templateSelector');
                    const poseType = templateSelector ? templateSelector.value : 'standard';
                    formData.append('pose_type', poseType);
                    
                    // 低质量模式，减少数据量
                    formData.append('low_quality', 'true');
                    
                    try {
                        // 发送图像到后端进行处理
                        const response = await fetch('/api/rehab/detect_video_pose', {
                            method: 'POST',
                            body: formData
                        });
                        
                        if (!response.ok) {
                            const errorText = await response.text();
                            console.error(`API返回错误: ${response.status}`, errorText);
                            return;
                        }
                        
                        // 解析关键点数据
                        const data = await response.json();
                        
                        // 转换后端数据格式为前端格式
                        const poseData = convertBackendData(data);
                        
                        // 更新最后一次关键点数据
                        lastKeypointsUpdate = poseData;
                        
                        // 调用回调通知UI更新
                        if (window.RehabCore.onKeypointsUpdateCallback) {
                            window.RehabCore.onKeypointsUpdateCallback(poseData);
                        }
                        
                        // 更新姿势可视化
                        updatePoseVisualization(poseData);
                        
                    } catch (error) {
                        console.error('发送图像进行处理时出错:', error);
                    }
                }, 'image/jpeg', 0.7); // 使用较低质量的JPEG以减少数据量
            }
        } catch (error) {
            console.error('关键点轮询错误:', error);
        }
    }, 200); // 每200毫秒发送一帧，约每秒5帧
    
    console.log('关键点轮询已启动');
}

/**
 * 停止关键点轮询
 */
function stopKeyPointsPolling() {
    if (keypointsPollingInterval) {
        clearInterval(keypointsPollingInterval);
        keypointsPollingInterval = null;
        window.RehabCore.onKeypointsUpdateCallback = null;
        console.log('关键点轮询已停止');
    }
}

/**
 * 获取可用的康复模板
 * @returns {Promise<Array>} 模板列表
 */
async function fetchAvailableTemplates() {
    const apiPath = '/api/rehab/templates';
    console.log(`正在获取康复模板: ${apiPath}`);
    
    try {
        const response = await fetch(apiPath);
        
        if (!response.ok) {
            console.error(`获取模板API返回错误: ${response.status}`);
            throw new Error(`获取模板失败: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('获取模板响应:', data);
        
        if (data.status === 'success' && Array.isArray(data.templates)) {
            availableTemplates = data.templates;
            return data.templates;
        } else {
            console.warn('获取模板响应格式不符合预期:', data);
            // 返回默认模板数据，以防API不可用
            return [
                {
                    id: "标准直立姿势",
                    name: "标准直立姿势",
                    description: "保持脊柱自然垂直，肩膀放松平衡，目视前方",
                    difficulty: "easy"
                },
                {
                    id: "侧弯矫正基础姿势",
                    name: "侧弯矫正基础姿势",
                    description: "针对轻度脊柱侧弯，通过调整肩部和髋部位置来改善侧弯",
                    difficulty: "medium"
                },
                {
                    id: "胸椎后凸矫正姿势",
                    name: "胸椎后凸矫正姿势",
                    description: "针对含胸驼背，注重拉伸胸肌，增强背部肌肉力量",
                    difficulty: "medium"
                },
                {
                    id: "腰椎前凸矫正姿势",
                    name: "腰椎前凸矫正姿势",
                    description: "针对腰椎过度前凸，着重收紧腹肌，调整骨盆位置",
                    difficulty: "hard"
                }
            ];
        }
    } catch (error) {
        console.error('获取康复模板错误:', error);
        // 返回默认模板数据，以防API不可用
        return [
            {
                id: "标准直立姿势",
                name: "标准直立姿势",
                description: "保持脊柱自然垂直，肩膀放松平衡，目视前方",
                difficulty: "easy"
            },
            {
                id: "侧弯矫正基础姿势",
                name: "侧弯矫正基础姿势",
                description: "针对轻度脊柱侧弯，通过调整肩部和髋部位置来改善侧弯",
                difficulty: "medium"
            }
        ];
    }
}

/**
 * 切换康复模板
 * @param {string} templateName - 模板名称
 * @returns {Promise<object>} 响应数据
 */
async function changeRehabTemplate(templateName) {
    if (!templateName) {
        throw new Error('模板名称不能为空');
    }
    
    console.log(`切换康复模板: ${templateName}`);
    
    try {
        const response = await fetch('/api/rehab/template', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                template: templateName
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.message || `服务器返回错误: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.status !== 'success') {
            throw new Error(data.message || '切换模板失败');
        }
        
        return data;
    } catch (error) {
        console.error('切换康复模板错误:', error);
        throw error;
    }
}

/**
 * 转换后端数据为前端格式
 * @param {Object} backendData - 从后端接收的数据
 * @returns {Object} 转换后的数据
 */
function convertBackendData(backendData) {
    // 检查数据状态
    if (backendData.status !== 'SUCCESS') {
        return {
            status: backendData.status,
            feedback: backendData.feedback || backendData.message || '姿势检测失败',
            timestamp: backendData.timestamp,
            score: 0,
            landmarks: []
        };
    }
    
    // 提取关键点
    const keypoints = backendData.poses && backendData.poses.length > 0
        ? backendData.poses[0].keypoints
        : [];
    
    // 转换关键点格式
    const landmarks = keypoints.map(kp => ({
        x: kp.x,
        y: kp.y,
        z: kp.z || 0,
        visibility: kp.score
    }));
    
    // 处理匹配数据
    const matchData = backendData.match_data || {};
    
    return {
        status: backendData.status,
        feedback: matchData.feedback || '',
        score: Math.round((matchData.confidence || 0) * 100),
        landmarks: landmarks,
        angles: matchData.angles || {},
        timestamp: backendData.timestamp
    };
}

/**
 * 更新姿势可视化
 * @param {Object} poseData - 姿势数据
 */
function updatePoseVisualization(poseData) {
    const canvas = document.getElementById('poseCanvas');
    if (!canvas || !poseData || !poseData.landmarks || poseData.landmarks.length === 0) {
        return;
    }
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // 清除画布
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // 绘制关键点连接线
    drawPoseConnections(ctx, poseData.landmarks);
}

/**
 * 绘制姿势连接线
 * @param {CanvasRenderingContext2D} ctx - Canvas上下文
 * @param {Array} landmarks - 关键点数据
 */
function drawPoseConnections(ctx, landmarks) {
    if (!landmarks || landmarks.length < 33) return;
    
    // 设置线条样式
    ctx.lineWidth = 3;
    ctx.strokeStyle = 'rgba(0, 255, 0, 0.8)';
    
    // 定义需要连接的点对（使用Mediapipe Pose的索引）
    const connections = [
        // 脸部连接
        [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8],
        // 上半身连接
        [9, 10], // 左右肩
        [11, 13], [13, 15], // 左臂
        [12, 14], [14, 16], // 右臂
        [11, 12], // 肩部
        [11, 23], [12, 24], // 躯干
        // 下半身连接
        [23, 24], // 髋部
        [23, 25], [25, 27], [27, 29], [29, 31], // 左腿
        [24, 26], [26, 28], [28, 30], [30, 32], // 右腿
    ];
    
    // 绘制连接线
    for (const [p1, p2] of connections) {
        const point1 = landmarks[p1];
        const point2 = landmarks[p2];
        
        // 检查点的可见性
        if (point1 && point2 && point1.visibility > 0.3 && point2.visibility > 0.3) {
            // 根据点的可见性设置线条透明度
            const alpha = Math.min(point1.visibility, point2.visibility);
            ctx.strokeStyle = `rgba(0, 255, 0, ${alpha})`;
            
            // 绘制线条
            ctx.beginPath();
            ctx.moveTo(point1.x * canvas.width, point1.y * canvas.height);
            ctx.lineTo(point2.x * canvas.width, point2.y * canvas.height);
            ctx.stroke();
        }
    }
    
    // 绘制关键点
    ctx.fillStyle = 'rgba(255, 0, 0, 0.7)';
    for (const point of landmarks) {
        if (point && point.visibility > 0.3) {
            ctx.beginPath();
            ctx.arc(point.x * canvas.width, point.y * canvas.height, 5, 0, 2 * Math.PI);
            ctx.fill();
        }
    }
} 