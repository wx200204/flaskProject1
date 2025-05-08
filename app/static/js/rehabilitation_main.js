/**
 * 康复指导系统主模块
 * 负责系统初始化和事件绑定
 */

// 在页面加载完成后初始化康复指导系统
document.addEventListener('DOMContentLoaded', function() {
    console.log("康复指导页面已加载，初始化系统");
    
    // 确保核心模块加载状态
    if (!window.RehabCore) {
        console.error('康复核心模块未正确加载');
        showError('康复系统初始化失败，请确保加载了rehabilitation_core.js');
        return;
    }
    
    if (!window.RehabUI) {
        console.error('康复UI模块未正确加载');
        showError('康复系统初始化失败，请确保加载了rehabilitation_ui.js');
        return;
    }
    
    try {
        // 初始化UI
        window.RehabUI.initUI();
        
        // 模板卡片交互
        initTemplateInteraction();
        
        // 帮助按钮功能
        initHelpDialog();
        
        // 初始化下一步按钮功能
        initNextStepButton();
        
        console.log('康复指导系统初始化完成');
    } catch (error) {
        console.error('初始化过程中发生错误:', error);
        showError('系统初始化失败: ' + error.message);
    }
});

/**
 * 初始化模板卡片交互
 */
function initTemplateInteraction() {
    const templateCards = document.querySelectorAll('.template-card');
    const templateSelector = document.getElementById('templateSelector');
    
    if (!templateCards.length || !templateSelector) {
        console.warn('未找到模板卡片或选择器，跳过初始化');
        return;
    }
    
    console.log(`找到${templateCards.length}个模板卡片，初始化交互`);
    
    templateCards.forEach(function(card) {
        card.addEventListener('click', function() {
            // 移除其他卡片的选中状态
            templateCards.forEach(function(c) {
                c.classList.remove('selected');
            });
            
            // 添加当前卡片的选中状态
            this.classList.add('selected');
            
            // 获取模板名称并选择下拉框中对应的选项
            const templateName = this.querySelector('h6').textContent;
            
            for (let i = 0; i < templateSelector.options.length; i++) {
                if (templateSelector.options[i].text === templateName) {
                    templateSelector.selectedIndex = i;
                    
                    // 触发change事件
                    const event = new Event('change');
                    templateSelector.dispatchEvent(event);
                    break;
                }
            }
        });
    });
}

/**
 * 初始化下一步按钮功能
 */
function initNextStepButton() {
    const nextButton = document.getElementById('nextExerciseButton');
    if (!nextButton) return;
    
    nextButton.addEventListener('click', function() {
        // 默认禁用状态
        nextButton.disabled = true;
        
        // 获取所有步骤元素
        const steps = document.querySelectorAll('.exercise-step');
        if (!steps.length) return;
        
        // 找到当前步骤
        let currentStep = 0;
        steps.forEach((step, index) => {
            if (step.classList.contains('current')) {
                currentStep = index;
            }
        });
        
        // 切换到下一步
        if (currentStep < steps.length - 1) {
            // 移除当前步骤的current类
            steps[currentStep].classList.remove('current', 'completed', 'holding', 'waiting');
            steps[currentStep].classList.add('completed');
            
            // 添加下一步的current类
            steps[currentStep + 1].classList.add('current');
            steps[currentStep + 1].classList.remove('completed', 'holding');
            
            // 如果UI模块存在，通知步骤变更
            if (window.RehabUI && window.RehabUI.updateExerciseStep) {
                window.RehabUI.updateExerciseStep(currentStep + 2); // +2因为步骤从1开始计数
            }
            
            // 播放语音指导
            if (window.RehabUI && window.RehabUI.speakFeedback) {
                const stepName = steps[currentStep + 1].querySelector('.step-name').textContent;
                window.RehabUI.speakFeedback(`现在进入${stepName}步骤，请按照指导完成动作`, true);
            }
            
            // 如果不是最后一步，启用按钮
            if (currentStep + 1 < steps.length - 1) {
                nextButton.disabled = false;
            }
        }
    });
    
    // 默认禁用下一步按钮，等待会话启动后启用
    nextButton.disabled = true;
}

/**
 * 初始化帮助对话框
 */
function initHelpDialog() {
    const helpBtn = document.getElementById('helpBtn');
    const helpDialog = document.getElementById('helpDialog');
    const closeHelpBtn = document.getElementById('closeHelpBtn');
    
    if (!helpBtn || !helpDialog || !closeHelpBtn) {
        console.warn('未找到帮助对话框组件，跳过初始化');
        return;
    }
    
    helpBtn.addEventListener('click', function() {
        helpDialog.style.display = 'flex';
    });
    
    closeHelpBtn.addEventListener('click', function() {
        helpDialog.style.display = 'none';
    });
    
    // 点击背景关闭
    helpDialog.addEventListener('click', function(e) {
        if (e.target === helpDialog) {
            helpDialog.style.display = 'none';
        }
    });
    
    // ESC键关闭
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && helpDialog.style.display === 'flex') {
            helpDialog.style.display = 'none';
        }
    });
}

/**
 * 显示错误消息
 * @param {string} message - 错误消息
 */
function showError(message) {
    console.error('错误:', message);
    
    const errorContainer = document.getElementById('errorMessageContainer');
    const errorText = document.getElementById('errorMessageText');
    
    if (errorContainer && errorText) {
        errorText.textContent = message;
        errorContainer.style.display = 'block';
        
        // 添加关闭按钮功能
        const closeBtn = document.getElementById('errorCloseBtn');
        if (closeBtn) {
            closeBtn.addEventListener('click', function() {
                errorContainer.style.display = 'none';
            });
        }
        
        // 添加重试按钮功能
        const retryBtn = document.getElementById('errorRetryBtn');
        if (retryBtn) {
            retryBtn.addEventListener('click', function() {
                location.reload();
            });
        }
    } else {
        // 如果错误容器不存在，创建一个简单的警告
        alert('错误: ' + message);
    }
}

/**
 * 加载康复计划数据
 * 这个函数在有康复计划数据时使用
 */
function loadRehabPlan() {
    const rehabPlanContainer = document.getElementById('rehabPlanContainer');
    const exerciseList = document.getElementById('exerciseList');
    
    if (!rehabPlanContainer || !exerciseList || !window.RehabCore) {
        console.warn('未找到康复计划容器或康复核心模块，跳过加载计划');
        return;
    }
    
    // 示例康复计划项目
    const defaultPlan = [
        {
            name: "标准直立姿势训练",
            description: "保持脊柱自然垂直，肩膀放松平衡",
            recommended: true
        },
        {
            name: "侧弯矫正基础训练",
            description: "针对轻度脊柱侧弯，调整肩部和髋部位置",
            recommended: false
        },
        {
            name: "胸椎后凸矫正训练",
            description: "针对含胸驼背，注重拉伸胸肌，增强背部肌肉力量",
            recommended: false
        }
    ];
    
    // 清空现有列表
    exerciseList.innerHTML = '';
    
    // 添加计划项目
    defaultPlan.forEach(function(exercise) {
        const item = document.createElement('li');
        item.className = 'exercise-item d-flex justify-content-between align-items-center';
        
        const contentDiv = document.createElement('div');
        const title = document.createElement('strong');
        title.textContent = exercise.name;
        
        const description = document.createElement('p');
        description.className = 'mb-0 small text-muted';
        description.textContent = exercise.description;
        
        contentDiv.appendChild(title);
        contentDiv.appendChild(description);
        
        item.appendChild(contentDiv);
        
        if (exercise.recommended) {
            const badge = document.createElement('span');
            badge.className = 'badge bg-primary';
            badge.textContent = '推荐';
            item.appendChild(badge);
        } else {
            const button = document.createElement('button');
            button.className = 'btn btn-sm btn-outline-primary';
            button.textContent = '开始';
            button.addEventListener('click', function() {
                // 模拟选择对应的模板并开始训练
                const templateSelector = document.getElementById('templateSelector');
                if (templateSelector) {
                    for (let i = 0; i < templateSelector.options.length; i++) {
                        if (templateSelector.options[i].text.includes(exercise.name.split(' ')[0])) {
                            templateSelector.selectedIndex = i;
                            
                            // 触发change事件
                            templateSelector.dispatchEvent(new Event('change'));
                            
                            // 如果会话未开始，自动开始会话
                            const startButton = document.getElementById('startSession');
                            if (startButton && startButton.textContent.includes('开始')) {
                                window.RehabUI.toggleRehabSession();
                            }
                            
                            break;
                        }
                    }
                }
            });
            item.appendChild(button);
        }
        
        exerciseList.appendChild(item);
    });
    
    console.log('康复计划数据加载完成');
} 