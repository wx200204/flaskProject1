/**
 * 视频媒体管理器类
 * 用于管理摄像头视频流和图像捕获
 */
class VideoMediaManager {
    constructor(options = {}) {
        // 视频元素
        this.videoElement = options.videoElement || document.createElement('video');
        this.streamImage = options.streamImageElement || null;
        
        // 回调函数
        this.onStatusChange = options.onStatusChange || function() {};
        this.onError = options.onError || function() {};
        
        // 配置选项
        this.maxRetries = options.maxRetries || 3;
        this.aspectRatio = options.aspectRatio || 4/3;
        this.facing = options.facing || 'user';
        
        // 状态
        this.state = 'disconnected';
        this.stream = null;
        this.retryCount = 0;
        this.retryTimer = null;
        this.lastFrameTime = 0;
        this.fallbackMode = false;
    }
    
    /**
     * 开始初始化媒体管理器
     */
    async start() {
        try {
            this._setState('initializing');
            
            // 停止任何现有流
            this.stopStream();
            
            // 尝试启动摄像头
            await this._startCamera();
            
            return true;
        } catch (error) {
            console.error('媒体管理器启动失败:', error);
            
            // 如果有重试次数，尝试重试
            if (this.retryCount < this.maxRetries) {
                this.retryCount++;
                console.log(`尝试重试 (${this.retryCount}/${this.maxRetries})...`);
                
                // 延迟重试
                return new Promise((resolve, reject) => {
                    this.retryTimer = setTimeout(async () => {
                        try {
                            await this.start();
                            resolve(true);
                        } catch (e) {
                            reject(e);
                        }
                    }, 1000);
                });
            } else {
                // 超过重试次数，报告错误
                this._setState('error', error.message || '无法访问摄像头');
                this.onError({
                    type: 'camera_access',
                    message: error.message || '无法访问摄像头',
                    error: error
                });
                
                throw error;
            }
        }
    }
    
    /**
     * 重新启动媒体管理器
     */
    async restart() {
        // 重置重试计数器
        this.retryCount = 0;
        
        // 停止任何定时器
        if (this.retryTimer) {
            clearTimeout(this.retryTimer);
            this.retryTimer = null;
        }
        
        // 重新启动
        return this.start();
    }
    
    /**
     * 停止视频流
     */
    stopStream() {
        if (this.stream) {
            // 停止所有轨道
            this.stream.getTracks().forEach(track => {
                track.stop();
            });
            this.stream = null;
        }
        
        // 清除视频元素源
        if (this.videoElement) {
            this.videoElement.srcObject = null;
            this.videoElement.src = '';
        }
        
        // 设置状态为断开
        this._setState('disconnected');
    }
    
    /**
     * 启动摄像头
     */
    async _startCamera() {
        try {
            // 设置摄像头约束
            const constraints = {
                video: {
                    facingMode: this.facing,
                    aspectRatio: this.aspectRatio,
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                },
                audio: false
            };
            
            // 请求摄像头访问
            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            
            // 设置视频元素源
            this.videoElement.srcObject = this.stream;
            this.videoElement.muted = true;
            this.videoElement.playsInline = true;
            
            // 播放视频
            await this.videoElement.play();
            
            // 确保视频实际上在播放
            if (this.videoElement.readyState < 2) {
                await new Promise((resolve, reject) => {
                    const timeoutId = setTimeout(() => {
                        reject(new Error('视频播放超时'));
                    }, 5000);
                    
                    this.videoElement.onloadeddata = () => {
                        clearTimeout(timeoutId);
                        resolve();
                    };
                });
            }
            
            // 检查视频尺寸
            if (this.videoElement.videoWidth === 0 || this.videoElement.videoHeight === 0) {
                throw new Error('视频尺寸无效，可能未成功获取摄像头画面');
            }
            
            // 重置重试计数器
            this.retryCount = 0;
            this.fallbackMode = false;
            
            // 设置状态为活跃
            this._setState('active');
            
            console.log('摄像头启动成功，尺寸:', 
                        this.videoElement.videoWidth, 'x', this.videoElement.videoHeight);
            
            return true;
        } catch (error) {
            console.error('启动摄像头失败:', error);
            throw error;
        }
    }
    
    /**
     * 使用备选视频文件
     */
    async useFallbackVideo(videoUrl) {
        try {
            // 停止现有流
            this.stopStream();
            
            // 设置视频元素源为文件
            this.videoElement.srcObject = null;
            this.videoElement.src = videoUrl;
            this.videoElement.loop = true;
            this.videoElement.muted = true;
            this.videoElement.playsInline = true;
            
            // 播放视频
            await this.videoElement.play();
            
            // 确保视频实际上在播放
            if (this.videoElement.readyState < 2) {
                await new Promise((resolve, reject) => {
                    const timeoutId = setTimeout(() => {
                        reject(new Error('视频播放超时'));
                    }, 5000);
                    
                    this.videoElement.onloadeddata = () => {
                        clearTimeout(timeoutId);
                        resolve();
                    };
                });
            }
            
            // 设置备选模式
            this.fallbackMode = true;
            
            // 设置状态为备选
            this._setState('fallback');
            
            console.log('已切换到备选视频模式');
            
            return true;
        } catch (error) {
            console.error('切换到备选视频失败:', error);
            
            // 设置状态为错误
            this._setState('error', error.message || '无法加载备选视频');
            this.onError({
                type: 'fallback_video',
                message: error.message || '无法加载备选视频',
                error: error
            });
            
            throw error;
        }
    }
    
    /**
     * 切换摄像头朝向
     */
    async switchFacing(facing) {
        this.facing = facing || (this.facing === 'user' ? 'environment' : 'user');
        
        // 如果在备选模式，什么都不做
        if (this.fallbackMode) {
            return false;
        }
        
        // 重置状态并重新启动
        this.retryCount = 0;
        return this.start();
    }
    
    /**
     * 切换到指定的摄像头
     */
    async switchCamera(deviceId) {
        try {
            this._setState('initializing');
            
            // 停止现有流
            this.stopStream();
            
            // 设置摄像头约束
            const constraints = {
                video: {
                    deviceId: { exact: deviceId },
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                },
                audio: false
            };
            
            // 请求摄像头访问
            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            
            // 设置视频元素源
            this.videoElement.srcObject = this.stream;
            this.videoElement.muted = true;
            this.videoElement.playsInline = true;
            
            // 播放视频
            await this.videoElement.play();
            
            // 确保视频实际上在播放
            if (this.videoElement.readyState < 2) {
                await new Promise((resolve, reject) => {
                    const timeoutId = setTimeout(() => {
                        reject(new Error('视频播放超时'));
                    }, 5000);
                    
                    this.videoElement.onloadeddata = () => {
                        clearTimeout(timeoutId);
                        resolve();
                    };
                });
            }
            
            // 重置状态
            this.retryCount = 0;
            this.fallbackMode = false;
            
            // 设置状态为活跃
            this._setState('active');
            
            console.log('摄像头切换成功');
            
            return true;
        } catch (error) {
            console.error('切换摄像头失败:', error);
            
            // 设置状态为错误
            this._setState('error', error.message || '无法切换摄像头');
            this.onError({
                type: 'camera_switch',
                message: error.message || '无法切换摄像头',
                error: error
            });
            
            throw error;
        }
    }
    
    /**
     * 刷新视频流
     */
    async refreshStream() {
        // 如果在备选模式，什么都不做
        if (this.fallbackMode) {
            return false;
        }
        
        // 重置状态并重新启动
        this.retryCount = 0;
        return this.start();
    }
    
    /**
     * 捕获当前视频帧
     */
    captureFrame() {
        try {
            // 如果视频未就绪，返回null
            if (!this.videoElement || 
                this.videoElement.readyState < 2 || 
                !this.videoElement.videoWidth || 
                !this.videoElement.videoHeight) {
                return null;
            }
            
            // 创建canvas元素
            const canvas = document.createElement('canvas');
            canvas.width = this.videoElement.videoWidth;
            canvas.height = this.videoElement.videoHeight;
            
            // 绘制视频帧
            const ctx = canvas.getContext('2d');
            ctx.drawImage(this.videoElement, 0, 0, canvas.width, canvas.height);
            
            // 更新最后帧时间
            this.lastFrameTime = Date.now();
            
            // 返回数据URL
            return canvas.toDataURL('image/jpeg', 0.85);
        } catch (error) {
            console.error('捕获视频帧失败:', error);
            return null;
        }
    }
    
    /**
     * 设置状态
     */
    _setState(state, message = '') {
        if (this.state !== state) {
            const prevState = this.state;
            this.state = state;
            
            // 调用状态变更回调
            this.onStatusChange({
                prev: prevState,
                state: state,
                message: message,
                timestamp: Date.now()
            });
        }
    }
}

// 导出类
window.VideoMediaManager = VideoMediaManager; 