/**
 * 姿势检测器类
 * 用于检测和分析用户姿势，与参考姿势进行比对
 */
class PoseDetector {
    constructor(options = {}) {
        // API端点
        this.apiEndpoint = options.apiEndpoint || '/api/rehab/detect_video_pose';
        
        // DOM元素
        this.canvasElement = options.canvasElement || document.createElement('canvas');
        this.ctx = this.canvasElement.getContext('2d');
        
        // 回调函数
        this.onStatusChange = options.onStatusChange || function() {};
        this.onDetectionResult = options.onDetectionResult || function() {};
        
        // 配置
        this.confidenceThreshold = options.confidenceThreshold || 0.5;
        this.detectionInterval = options.detectionInterval || 300;
        this.maxConcurrentRequests = options.maxConcurrentRequests || 1;
        
        // 状态
        this.active = false;
        this.currentPoseType = '';
        this.lastDetectionTime = 0;
        this.lastDetectionResult = null;
        this.detectingInProgress = false;
        this.pendingRequests = 0;
        this.detectionTimer = null;
        this.consecutiveFailures = 0;
        this.matchThresholds = {
            excellent: 0.85,
            good: 0.75,
            fair: 0.6,
            poor: 0.0
        };
    }
    
    /**
     * 开始姿势检测
     */
    startDetection(poseType = '') {
        if (this.active) return;
        
        // 设置当前姿势类型
        this.currentPoseType = poseType;
        
        // 重置状态
        this.lastDetectionResult = null;
        this.consecutiveFailures = 0;
        
        // 设置为活跃状态
        this.active = true;
        
        // 通知状态变更
        this._notifyStatusChange('started', {
            poseType: this.currentPoseType
        });
        
        // 立即执行一次检测
        this.detectPose();
        
        // 启动定时检测
        this.detectionTimer = setInterval(() => {
            this.detectPose();
        }, this.detectionInterval);
        
        console.log('姿势检测已启动，类型:', poseType || '默认');
    }
    
    /**
     * 停止姿势检测
     */
    stopDetection() {
        if (!this.active) return;
        
        // 停止定时器
        if (this.detectionTimer) {
            clearInterval(this.detectionTimer);
            this.detectionTimer = null;
        }
        
        // 设置为非活跃状态
        this.active = false;
        
        // 通知状态变更
        this._notifyStatusChange('stopped');
        
        console.log('姿势检测已停止');
    }
    
    /**
     * 暂停姿势检测
     */
    pauseDetection() {
        if (!this.active) return;
        
        // 停止定时器
        if (this.detectionTimer) {
            clearInterval(this.detectionTimer);
            this.detectionTimer = null;
        }
        
        // 设置为非活跃状态但不完全停止
        this.active = false;
        
        // 通知状态变更
        this._notifyStatusChange('paused');
        
        console.log('姿势检测已暂停');
    }
    
    /**
     * 恢复姿势检测
     */
    resumeDetection() {
        if (this.active) return;
        
        // 设置为活跃状态
        this.active = true;
        
        // 通知状态变更
        this._notifyStatusChange('resumed');
        
        // 立即执行一次检测
        this.detectPose();
        
        // 启动定时检测
        this.detectionTimer = setInterval(() => {
            this.detectPose();
        }, this.detectionInterval);
        
        console.log('姿势检测已恢复');
    }
    
    /**
     * 执行姿势检测
     */
    async detectPose() {
        // 如果不活跃或已有检测进行中，跳过
        if (!this.active || this.detectingInProgress || this.pendingRequests >= this.maxConcurrentRequests) {
            return;
        }
        
        // 检查时间间隔
        const now = Date.now();
        if (now - this.lastDetectionTime < this.detectionInterval / 2) {
            return;
        }
        
        try {
            // 设置标志位
            this.detectingInProgress = true;
            this.pendingRequests++;
            
            // 更新最后检测时间
            this.lastDetectionTime = now;
            
            // 捕获当前帧
            const frameBase64 = window.captureVideoFrame ? window.captureVideoFrame() : null;
            
            if (!frameBase64) {
                console.warn('无法捕获视频帧');
                this.consecutiveFailures++;
                
                if (this.consecutiveFailures > 5) {
                    this._notifyStatusChange('error', {
                        message: '无法捕获视频帧'
                    });
                }
                
                return;
            }
            
            // 重置失败计数
            this.consecutiveFailures = 0;
            
            // 准备请求数据
            const formData = new FormData();
            
            // 将Base64转换为Blob
            const byteString = atob(frameBase64.split(',')[1]);
            const mimeString = frameBase64.split(',')[0].split(':')[1].split(';')[0];
            const ab = new ArrayBuffer(byteString.length);
            const ia = new Uint8Array(ab);
            
            for (let i = 0; i < byteString.length; i++) {
                ia[i] = byteString.charCodeAt(i);
            }
            
            const blob = new Blob([ab], { type: mimeString });
            formData.append('image', blob, 'frame.jpg');
            
            // 添加姿势类型和质量模式
            formData.append('pose_type', this.currentPoseType);
            formData.append('low_quality', window.lowQualityMode ? 'true' : 'false');
            
            // 发送请求
            const response = await fetch(this.apiEndpoint, {
                method: 'POST',
                body: formData
            });
            
            // 检查响应
            if (!response.ok) {
                throw new Error(`API响应错误: ${response.status} ${response.statusText}`);
            }
            
            // 解析响应
            const result = await response.json();
            
            // 处理检测结果
            this._processDetectionResult(result);
        } catch (error) {
            console.error('姿势检测错误:', error);
            this.consecutiveFailures++;
            
            if (this.consecutiveFailures > 5) {
                this._notifyStatusChange('error', {
                    message: '姿势检测服务出错'
                });
            }
        } finally {
            // 清除标志位
            this.detectingInProgress = false;
            this.pendingRequests--;
        }
    }
    
    /**
     * 处理检测结果
     */
    _processDetectionResult(result) {
        // 保存结果
        this.lastDetectionResult = result;
        
        // 如果结果有效，绘制到画布
        if (result && result.status === 'SUCCESS' && result.poses) {
            // 绘制骨架
            this._drawPose(result.poses);
            
            // 评估姿势匹配度
            const matchResult = this._evaluateMatch(result);
            
            // 调用回调
            this.onDetectionResult(result, matchResult);
            
            // 如果匹配度达到标准，通知完成
            if (matchResult.state === 'excellent' && matchResult.duration >= 2.0) {
                this._notifyStatusChange('pose_completed', {
                    poseType: this.currentPoseType,
                    confidence: matchResult.confidence,
                    duration: matchResult.duration
                });
            }
        } else if (result && result.status === 'NOT_DETECTED') {
            // 清除画布
            this._clearCanvas();
            
            // 调用回调，没有检测到姿势
            this.onDetectionResult(null, {
                state: 'none',
                confidence: 0,
                feedback: result.feedback || '未检测到人体，请确保完整出现在画面中'
            });
        } else {
            // 清除画布
            this._clearCanvas();
            
            // 调用回调，出错
            this.onDetectionResult(null, {
                state: 'none',
                confidence: 0,
                feedback: '姿势识别服务暂时不可用'
            });
        }
    }
    
    /**
     * 评估姿势匹配度
     */
    _evaluateMatch(result) {
        if (!result || !result.match_data) {
            return {
                state: 'none',
                confidence: 0,
                duration: 0,
                feedback: '无匹配数据'
            };
        }
        
        const matchData = result.match_data;
        const confidence = matchData.confidence || 0;
        const duration = matchData.duration || 0;
        let feedback = matchData.feedback || [];
        
        // 确定匹配状态
        let state = 'none';
        if (confidence >= this.matchThresholds.excellent) {
            state = 'excellent';
        } else if (confidence >= this.matchThresholds.good) {
            state = 'good';
        } else if (confidence >= this.matchThresholds.fair) {
            state = 'fair';
        } else if (confidence >= this.matchThresholds.poor) {
            state = 'poor';
        }
        
        // 如果没有反馈，生成通用反馈
        if (!feedback || feedback.length === 0) {
            if (state === 'poor') {
                feedback = ['请调整姿势，关注脊柱位置'];
            } else if (state === 'fair') {
                feedback = ['姿势接近标准，请继续保持'];
            } else if (state === 'good') {
                feedback = ['姿势良好，继续保持'];
            } else if (state === 'excellent') {
                feedback = ['姿势完美，做得非常好!'];
            }
        }
        
        return {
            state,
            confidence,
            duration,
            feedback: Array.isArray(feedback) ? feedback.join('；') : feedback
        };
    }
    
    /**
     * 绘制姿势到画布
     */
    _drawPose(poses) {
        // 清除画布
        this._clearCanvas();
        
        if (!poses || poses.length === 0) return;
        
        const pose = poses[0]; // 取第一个姿势
        
        // 设置画布尺寸与视频元素相同
        const canvasWidth = this.canvasElement.width;
        const canvasHeight = this.canvasElement.height;
        
        // 计算线条粗细 - 根据画布尺寸自适应
        const lineWidth = Math.max(3, Math.floor(canvasWidth / 150));
        const jointRadius = Math.max(4, Math.floor(canvasWidth / 200));
        
        // 绘制背景轮廓 - 增加人体形状的可视化
        this._drawBodyOutline(pose, canvasWidth, canvasHeight);
        
        // 定义连接线
        const connections = [
            // 面部
            [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8],
            // 躯干
            [9, 10], [11, 12], [11, 23], [12, 24], [23, 24],
            // 左臂
            [11, 13], [13, 15], [15, 17], [15, 19], [15, 21],
            // 右臂
            [12, 14], [14, 16], [16, 18], [16, 20], [16, 22],
            // 左腿
            [23, 25], [25, 27], [27, 29], [27, 31],
            // 右腿
            [24, 26], [26, 28], [28, 30], [28, 32]
        ];
        
        // 定义脊柱相关的连接 - 用于特殊绘制
        const spineConnections = [
            [11, 23], // 左肩到左髋
            [12, 24], // 右肩到右髋
            [11, 12], // 肩部连接线
            [23, 24]  // 髋部连接线
        ];
        
        // 绘制连接线
        this.ctx.lineWidth = lineWidth;
        
        // 先绘制非脊柱连接
        for (const [start, end] of connections) {
            // 跳过脊柱连接，稍后单独处理
            if (spineConnections.some(conn => 
                (conn[0] === start && conn[1] === end) || 
                (conn[0] === end && conn[1] === start))) {
                continue;
            }
            
            const startPoint = pose.keypoints[start];
            const endPoint = pose.keypoints[end];
            
            if (!startPoint || !endPoint || 
                startPoint.score < this.confidenceThreshold || 
                endPoint.score < this.confidenceThreshold) {
                continue;
            }
            
            // 计算坐标
            const startX = startPoint.x * canvasWidth;
            const startY = startPoint.y * canvasHeight;
            const endX = endPoint.x * canvasWidth;
            const endY = endPoint.y * canvasHeight;
            
            // 使用渐变色线条增强可视性
            const gradient = this.ctx.createLinearGradient(startX, startY, endX, endY);
            gradient.addColorStop(0, 'rgba(0, 119, 255, 0.8)');
            gradient.addColorStop(1, 'rgba(0, 200, 255, 0.7)');
            this.ctx.strokeStyle = gradient;
            
            // 绘制线条
            this.ctx.beginPath();
            this.ctx.moveTo(startX, startY);
            this.ctx.lineTo(endX, endY);
            this.ctx.stroke();
        }
        
        // 单独绘制脊柱连接 - 更粗更明显
        this.ctx.lineWidth = lineWidth + 2;
        for (const [start, end] of spineConnections) {
            const startPoint = pose.keypoints[start];
            const endPoint = pose.keypoints[end];
            
            if (!startPoint || !endPoint || 
                startPoint.score < this.confidenceThreshold || 
                endPoint.score < this.confidenceThreshold) {
                continue;
            }
            
            // 计算坐标
            const startX = startPoint.x * canvasWidth;
            const startY = startPoint.y * canvasHeight;
            const endX = endPoint.x * canvasWidth;
            const endY = endPoint.y * canvasHeight;
            
            // 脊柱连接使用更明显的绿色渐变
            const gradient = this.ctx.createLinearGradient(startX, startY, endX, endY);
            gradient.addColorStop(0, 'rgba(0, 255, 100, 0.9)');
            gradient.addColorStop(1, 'rgba(0, 255, 0, 0.9)');
            this.ctx.strokeStyle = gradient;
            
            // 绘制线条
            this.ctx.beginPath();
            this.ctx.moveTo(startX, startY);
            this.ctx.lineTo(endX, endY);
            this.ctx.stroke();
            
            // 为肩部到髋部的连接绘制参考垂直线
            if ((start === 11 && end === 23) || (start === 12 && end === 24)) {
                // 绘制垂直参考线 - 淡蓝色虚线
                this.ctx.beginPath();
                this.ctx.setLineDash([5, 5]);
                this.ctx.strokeStyle = 'rgba(0, 200, 255, 0.6)';
                this.ctx.lineWidth = lineWidth - 1;
                this.ctx.moveTo(startX, startY);
                this.ctx.lineTo(startX, endY); // 保持X不变
                this.ctx.stroke();
                this.ctx.setLineDash([]); // 恢复实线
                
                // 计算与垂直线的偏差
                const deviationX = Math.abs(startX - endX);
                const deviationPercent = (deviationX / canvasWidth) * 100;
                
                // 如果偏差较大，标记出来
                if (deviationPercent > 2) {
                    const midX = (startX + endX) / 2;
                    const midY = (startY + endY) / 2;
                    
                    // 绘制偏差指示器
                    this.ctx.beginPath();
                    this.ctx.arc(midX, midY, jointRadius + 3, 0, 2 * Math.PI);
                    this.ctx.fillStyle = deviationPercent > 5 ? 
                        'rgba(255, 0, 0, 0.8)' : 'rgba(255, 165, 0, 0.8)';
                    this.ctx.fill();
                    
                    // 显示偏差值
                    if (deviationPercent > 10) {
                        this.ctx.fillStyle = 'white';
                        this.ctx.font = '12px Arial';
                        this.ctx.fillText(`${Math.round(deviationPercent)}%`, midX + 8, midY - 8);
                    }
                }
            }
        }
        
        // 绘制关键点
        for (const point of pose.keypoints) {
            if (point.score < this.confidenceThreshold) continue;
            
            const x = point.x * canvasWidth;
            const y = point.y * canvasHeight;
            
            // 设置关键点大小和颜色 - 脊柱相关点使用特殊颜色和更大尺寸
            let radius = jointRadius;
            let fillColor = 'rgba(255, 255, 255, 0.8)';
            
            if ([11, 12, 23, 24].includes(point.id)) {
                // 脊柱相关点 - 肩部和髋部
                radius = jointRadius + 2;
                fillColor = 'rgba(0, 255, 0, 0.9)';
            } else if ([0, 9, 10].includes(point.id)) {
                // 头部和颈部关键点
                radius = jointRadius + 1;
                fillColor = 'rgba(255, 200, 0, 0.8)';
            }
            
            // 绘制圆点
            this.ctx.beginPath();
            this.ctx.arc(x, y, radius, 0, 2 * Math.PI);
            this.ctx.fillStyle = fillColor;
            this.ctx.fill();
            
            // 为关键点添加边框以增强可见性
            this.ctx.lineWidth = 1;
            this.ctx.strokeStyle = 'rgba(0, 0, 0, 0.5)';
            this.ctx.stroke();
        }
    }
    
    /**
     * 绘制人体轮廓背景
     */
    _drawBodyOutline(pose, canvasWidth, canvasHeight) {
        // 创建人体主要部位的轮廓
        if (!pose || !pose.keypoints) return;
        
        const keypoints = pose.keypoints;
        
        // 获取主要身体部位的关键点
        const getPoint = (id) => {
            const point = keypoints[id];
            if (!point || point.score < this.confidenceThreshold) return null;
            return {
                x: point.x * canvasWidth,
                y: point.y * canvasHeight
            };
        };
        
        // 躯干关键点
        const leftShoulder = getPoint(11);
        const rightShoulder = getPoint(12);
        const leftHip = getPoint(23);
        const rightHip = getPoint(24);
        
        // 如果没有足够的关键点，不绘制轮廓
        if (!leftShoulder || !rightShoulder || !leftHip || !rightHip) return;
        
        // 绘制躯干轮廓
        this.ctx.beginPath();
        this.ctx.moveTo(leftShoulder.x, leftShoulder.y);
        this.ctx.lineTo(rightShoulder.x, rightShoulder.y);
        this.ctx.lineTo(rightHip.x, rightHip.y);
        this.ctx.lineTo(leftHip.x, leftHip.y);
        this.ctx.closePath();
        
        // 填充半透明轮廓
        this.ctx.fillStyle = 'rgba(40, 120, 180, 0.2)';
        this.ctx.fill();
        
        // 尝试绘制腿部轮廓
        const leftKnee = getPoint(25);
        const rightKnee = getPoint(26);
        
        if (leftKnee && leftHip) {
            this.ctx.beginPath();
            this.ctx.moveTo(leftHip.x, leftHip.y);
            this.ctx.lineTo(leftKnee.x, leftKnee.y);
            this.ctx.strokeStyle = 'rgba(100, 160, 220, 0.4)';
            this.ctx.lineWidth = 4;
            this.ctx.stroke();
        }
        
        if (rightKnee && rightHip) {
            this.ctx.beginPath();
            this.ctx.moveTo(rightHip.x, rightHip.y);
            this.ctx.lineTo(rightKnee.x, rightKnee.y);
            this.ctx.strokeStyle = 'rgba(100, 160, 220, 0.4)';
            this.ctx.lineWidth = 4;
            this.ctx.stroke();
        }
    }
    
    /**
     * 清除画布
     */
    _clearCanvas() {
        this.ctx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);
    }
    
    /**
     * 通知状态变更
     */
    _notifyStatusChange(type, data = {}) {
        this.onStatusChange({
            type,
            timestamp: Date.now(),
            poseType: this.currentPoseType,
            ...data
        });
    }
    
    /**
     * 设置新的匹配阈值
     */
    setMatchThresholds(thresholds) {
        this.matchThresholds = {
            ...this.matchThresholds,
            ...thresholds
        };
    }
}

// 导出类
window.PoseDetector = PoseDetector;